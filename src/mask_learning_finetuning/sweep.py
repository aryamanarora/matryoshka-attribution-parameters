"""Walking a saved finetune across a sparsity grid, for any downstream eval.

`finetune_masked.py` writes a run directory holding learned per-unit scores and (with
``--save-delta``) the finetune delta itself. Every "how does behaviour X depend on the top-k
parameter units" question then has the same skeleton:

    for each k on a grid:
        theta_eff = theta_base + m_k . delta          (cause / necessary)
        theta_eff = theta_base + (1 - m_k) . delta    (iso / sufficient)
        run some eval against theta_eff

This module owns that skeleton -- checkpoint loading, the condition grid and its two anchors,
duplicate detection, and writing composed weights into a live model -- so that an eval script
only has to supply the eval. `scripts/eval_em_sparsity.py` (emergent misalignment, via the
reference EM eval) and `scripts/eval_mmlu_sparsity.py` (MMLU accuracy) are both thin wrappers
around :class:`MaskedRun`.

Note that ``k`` here indexes *units*, not parameters: what one score covers is fixed by the
run's ``--unit`` (tensor / row / col / weight) and recorded in its layout.
"""

import logging
from pathlib import Path

import torch

from .param_masks import UnitLayout, expand_mask, hard_topk_mask

logger = logging.getLogger(__name__)

# the grid finetune_masked.py sweeps loss on; reused so behavioural curves line up with it
DEFAULT_EVAL_FRACS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

PRETRAINED, FULL_DELTA = "pretrained", "full_delta"


def parse_fracs(spec, default=DEFAULT_EVAL_FRACS):
    """``"0.01,0.1,1.0"`` -> a tuple of floats; ``None`` -> the default grid."""
    if not spec:
        return tuple(default)
    fracs = tuple(float(f) for f in spec.split(","))
    bad = [f for f in fracs if not 0.0 <= f <= 1.0]
    if bad:
        raise ValueError(f"mask fractions must be in [0, 1]; got {bad}")
    return fracs


def load_checkpoint(run_dir: Path, name=None, *, require_delta: bool = True):
    """Load a run's saved blob, preferring an explicit checkpoint name over ``final.pt``."""
    run_dir = Path(run_dir)
    path = Path(name) if name and Path(name).is_absolute() else run_dir / (name or "final.pt")
    if not path.exists():
        available = sorted(q.name for q in run_dir.glob("*.pt"))
        raise SystemExit(f"no checkpoint at {path}. Available in {run_dir}: {available}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if require_delta and "delta" not in blob:
        raise SystemExit(
            f"{path} has scores but no delta, so there is nothing to mask. Re-run the "
            "finetune with --save-delta (and --save-delta-intermediate for mid-run "
            "checkpoints), or point --checkpoint at one that has it."
        )
    return path, blob


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


def mask_for(k: int, layout: UnitLayout, scores: torch.Tensor) -> torch.Tensor:
    """Hard top-k mask over the learned scores, with the two extremes short-circuited."""
    if k <= 0:
        return torch.zeros(layout.total)
    if k >= layout.total:
        return torch.ones(layout.total)
    return hard_topk_mask(scores, k)


class MaskedRun:
    """A finetuned run's base model, delta and scores, switchable between sparsities.

    ``load()`` brings up the base model once; ``apply(k, invert=...)`` then rewrites its
    parameters in place for one point on the grid. In place rather than through
    ``functional_call`` (how `finetune_masked.py` does its loss sweep) because ``generate``
    and batched scoring both want real parameters. Tied weights need no special handling as a
    result: ``lm_head.weight`` *is* ``embed_tokens.weight``, so writing the canonical tensor
    updates both -- the alias map upstream exists only because ``functional_call`` substitutes
    by name.

    The pretrained weights are kept on the CPU and the composition happens there, so peak GPU
    cost is one parameter tensor rather than a second copy of the model plus an fp32 delta.
    """

    def __init__(self, layout: UnitLayout, scores, deltas, *, mode: str,
                 ckpt_path=None, train_args=None):
        self.layout = layout
        self.scores = scores.float().cpu()
        self.deltas = {n: d.float().cpu() for n, d in deltas.items()}
        self.mode = mode
        self.invert = mode == "sufficient"
        self.ckpt_path = Path(ckpt_path) if ckpt_path else Path("<live>")
        self.train_args = train_args or {}
        self.model = self.tokenizer = self.base = None
        self.device = "cpu"
        self._attached = False

    @classmethod
    def from_blob(cls, ckpt_path, blob, *, mode: str):
        """The post-hoc case: everything comes off a saved checkpoint."""
        return cls(UnitLayout(**blob["layout"]), blob["scores"], blob["delta"], mode=mode,
                   ckpt_path=ckpt_path, train_args=blob["args"])

    @classmethod
    def attach(cls, model, tokenizer, layout: UnitLayout, scores, deltas, *, mode: str,
               ckpt_path=None):
        """The in-training case: wrap a model that is already loaded on the device.

        Used by the eval hooks the training scripts call, so an inline sweep runs the exact
        same code path as the post-hoc scripts without paying for a second copy of the model.

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

    @torch.no_grad()
    def apply(self, k: int, *, invert: bool):
        """Write ``theta_base + m_k . delta`` into the live model."""
        mask = mask_for(k, self.layout, self.scores)
        params = dict(self.model.named_parameters())
        for i, name in enumerate(self.layout.names):
            b = self.base[name]
            m = expand_mask(mask[self.layout.slice_for(i)], self.layout.shapes[i],
                            self.layout.axes[i])
            if invert:
                m = 1.0 - m
            # cast the update to the base dtype before adding, exactly as compose_params does,
            # so a given (mask, delta) yields the same weights here as it did during training
            composed = b + (m * self.deltas[name]).to(b.dtype)
            params[name].data.copy_(composed.to(params[name].device))

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
