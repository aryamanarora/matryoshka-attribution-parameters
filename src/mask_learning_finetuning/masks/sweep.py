"""Walking a saved finetune across a sparsity grid, for any downstream eval.

A training run writes a directory holding learned per-unit scores and (with ``--save-delta``)
the finetune delta itself. Every "how does behaviour X depend on the top-k parameter units"
question then has the same skeleton:

    for each k on a grid:
        theta_eff = theta_base + m_k . delta          (cause / necessary)
        theta_eff = theta_base + (1 - m_k) . delta    (iso / sufficient)
        run some eval against theta_eff

This module owns that skeleton -- the condition grid and its two anchors, duplicate detection,
and switching a live model between sparsities -- so an eval only has to supply the eval. See
``mask_learning_finetuning.eval`` for the registry of evals that ride on it.

Note that ``k`` here indexes *units*, not parameters: what one score covers is fixed by the
run's unit mode (tensor / row / col / weight / nonresid) and recorded in its layout.
"""

import logging
from pathlib import Path

import torch

from .compose import apply_in_place, mask_for, resolve_dtype
from .layout import UnitLayout

logger = logging.getLogger(__name__)

# The one definition of the grid. Behavioural curves and the SFT-loss curve are only
# comparable if they are sampled at the same sparsities, so there must not be a second copy.
DEFAULT_EVAL_FRACS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

PRETRAINED, FULL_DELTA = "pretrained", "full_delta"


def parse_fracs(spec, default=DEFAULT_EVAL_FRACS):
    """``"0.01,0.1,1.0"`` -> a tuple of floats; ``None`` -> the default grid."""
    if not spec:
        return tuple(default)
    if not isinstance(spec, str):
        fracs = tuple(float(f) for f in spec)      # already a list (e.g. straight from YAML)
    else:
        fracs = tuple(float(f) for f in spec.split(","))
    bad = [f for f in fracs if not 0.0 <= f <= 1.0]
    if bad:
        raise ValueError(f"mask fractions must be in [0, 1]; got {bad}")
    return fracs


def conditions_for(fracs, total: int, invert: bool):
    """``[(label, k, invert)]`` for both anchors and every swept fraction.

    The anchors pass ``invert=False`` whatever the run's mode is, so they mean the same thing
    in both directions: ``pretrained`` is literally the base model, ``full_delta`` literally
    the whole finetune. Under ``cause`` an all-zero mask already means pretrained; under
    ``iso`` it means the full delta, and labelling that "pretrained" would be backwards.
    """
    conds = [(PRETRAINED, 0, False)]
    for f in fracs:
        conds.append((f"frac_{f:g}", max(1, int(round(f * total))), invert))
    conds.append((FULL_DELTA, total, False))
    return conds


def weights_key(k: int, total: int, invert: bool):
    """Identity of the weights a condition composes, so duplicates can be evaluated once.

    The grid ends at fraction 1.0 and the anchors are the two extremes, so at least one pair
    always coincides: under ``cause``, ``frac_1`` is ``full_delta``; under ``iso`` it is
    ``pretrained``. Both are still reported -- they are meaningful ends of the curve -- but
    evaluating identical weights twice is wasted budget.
    """
    if k <= 0:
        return "full" if invert else "base"
    if k >= total:
        return "base" if invert else "full"
    return ("topk", k, invert)


def plan(conds, total: int, have_result) -> tuple:
    """Split conditions into ``(to_run, to_copy)`` given a ``have_result(label) -> bool``.

    ``to_run`` is one condition per distinct weighting whose result is missing; ``to_copy``
    is ``(label, source_label)`` for conditions that duplicate an earlier weighting. Anything
    whose result is already on disk appears in neither.
    """
    first_with, to_run, to_copy = {}, [], []
    for label, k, invert in conds:
        key = weights_key(k, total, invert)
        have = have_result(label)
        if key in first_with:
            if not have:
                to_copy.append((label, first_with[key]))
            continue
        first_with[key] = label
        if not have:
            to_run.append((label, k, invert))
    return to_run, to_copy


class MaskedRun:
    """A finetuned run's base model, delta and scores, switchable between sparsities.

    ``load()`` brings up the base model once; ``apply(k, invert=...)`` then rewrites its
    parameters in place for one point on the grid. In place rather than through
    ``functional_call`` because ``generate`` and batched scoring both want real parameters;
    the arithmetic is shared with the functional path via ``masks.compose``.

    The pretrained weights are kept on the CPU and the composition happens there, so peak GPU
    cost is one parameter tensor rather than a second copy of the model plus an fp32 delta.
    """

    def __init__(self, layout: UnitLayout, scores, deltas, *, mode: str,
                 ckpt_path=None, train_args=None):
        self.layout = layout
        self.scores = scores.float().cpu()
        # Held and composed at the dtype the run TRAINED at, which the checkpoint records under
        # `delta_dtype` (MaskCfg). Absent for anything written before that flag existed, and there
        # the answer is fp32 -- which is what `.float()` did unconditionally before, so old
        # checkpoints sweep exactly as they always did.
        self.compose_dtype = resolve_dtype((train_args or {}).get("delta_dtype")) \
            or torch.float32
        self.deltas = {n: d.to(self.compose_dtype).cpu() for n, d in deltas.items()}
        self.mode = mode
        self.invert = mode == "sufficient"
        self.ckpt_path = Path(ckpt_path) if ckpt_path else Path("<live>")
        self.train_args = train_args or {}
        self.model = self.tokenizer = self.base = None
        self.device = "cpu"
        self._attached = False

    @classmethod
    def from_blob(cls, ckpt_path, blob, *, mode: str, resid_dim=None):
        """The post-hoc case: everything comes off a saved checkpoint."""
        from .checkpoint import layout_from_blob
        return cls(layout_from_blob(blob, resid_dim=resid_dim),
                   blob["scores"], blob["delta"], mode=mode,
                   ckpt_path=ckpt_path, train_args=blob["args"])

    @classmethod
    def attach(cls, model, tokenizer, layout: UnitLayout, scores, deltas, *, mode: str,
               ckpt_path=None):
        """The in-training case: wrap a model that is already loaded on the device.

        Used by the eval hooks the training loop calls, so an inline sweep runs the exact same
        code path as the post-hoc scripts without paying for a second copy of the model.

        theta_base is **cloned** here, not aliased. The training loop keeps its own
        ``base = {n: p.detach() ...}``, which are views onto the live parameters; ``apply()``
        writes through those views, so the snapshot has to be independent or ``restore()``
        would put back whatever the last condition wrote. Callers must restore -- the hooks
        do it in a ``finally``.
        """
        run = cls(layout, scores, deltas, mode=mode, ckpt_path=ckpt_path)
        run.model, run.tokenizer = model, tokenizer
        params = dict(model.named_parameters())
        run.base = {n: params[n].detach().cpu().clone() for n in layout.names}
        run.device = str(next(model.parameters()).device)
        run._attached = True
        return run

    @property
    def total(self) -> int:
        return self.layout.total

    def describe(self) -> str:
        return (f"{self.ckpt_path.name}: {self.layout.summary()}, mode={self.mode} "
                f"({'delta on the complement' if self.invert else 'delta on the top-k'})")

    def load(self, *, model_id=None, dtype=None, device="cpu", use_cache=True):
        """Instantiate the base model and tokenizer, and snapshot theta_base on the CPU.

        A no-op for an attached run: the model is already up, and re-loading it would both
        waste the memory and throw away the caller's training state.
        """
        if self._attached:
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = use_cache
            return self.model, self.tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = model_id or self.train_args["model"]
        dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[
            dtype or self.train_args.get("dtype", "bfloat16")]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model.to(device).eval()
        if hasattr(self.model.config, "use_cache"):
            # the finetune turns this off; generation wants it, scoring does not care
            self.model.config.use_cache = use_cache

        params = dict(self.model.named_parameters())
        missing = [n for n in self.layout.names if n not in params]
        if missing:
            raise SystemExit(f"{len(missing)} checkpoint tensors are absent from {model_id}, "
                             f"e.g. {missing[:3]}. Wrong base model?")
        for i, name in enumerate(self.layout.names):
            if tuple(params[name].shape) != tuple(self.layout.shapes[i]):
                raise SystemExit(
                    f"shape mismatch for {name}: model {tuple(params[name].shape)} vs "
                    f"checkpoint {tuple(self.layout.shapes[i])}")
        self.base = {n: params[n].detach().cpu().clone() for n in self.layout.names}
        self.device = device
        return self.model, self.tokenizer

    def apply(self, k: int, *, invert: bool):
        """Write ``theta_base + m_k . delta`` into the live model."""
        apply_in_place(self.model, self.base, self.deltas,
                       mask_for(k, self.layout, self.scores), self.layout, invert=invert,
                       out_dtype=self.compose_dtype)

    def restore(self):
        """Put the pretrained weights back, so the model is never left mid-sweep."""
        self.apply(0, invert=False)

    def release(self):
        """Drop the model. An attached run does not own it, so this only frees the cache."""
        if not self._attached:
            model, self.model = self.model, None
            del model
        if str(getattr(self, "device", "")).startswith("cuda"):
            torch.cuda.empty_cache()
