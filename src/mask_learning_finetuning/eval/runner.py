"""Driving a set of evals over the sparsity grid, and reporting the result once.

The runner owns everything that was previously copy-pasted between eval scripts: iterating the
condition grid, deciding how the masked weights reach each eval, skipping duplicate weightings,
and turning ``{condition: {eval: {split: {metric: value}}}}`` into logs, wandb series and JSON.
An eval supplies only ``build`` and ``run``.

**Two ways the weights reach an eval**, chosen per batch of evals rather than per eval:

- If every eval in the batch is forward-only (``needs_real_weights=False``), the runner composes
  a parameter dict and serves them through ``functional_call``. Nothing writes to the live
  model, which is the cheap path and the safe one to call mid-training.
- If any eval generates, the runner writes ``theta_eff`` into the model in place for the whole
  batch and restores afterwards. In-place weights are valid for forward-only evals too, so
  mixing is fine; it is the reverse that is impossible, since ``model.generate`` cannot read a
  parameter dict.

**The degenerate case is the common one.** A plain (unmasked) run has no scores and no delta, so
there is nothing to sweep: the grid collapses to a single condition, ``dense``. The same runner
handles it, which is what lets the French experiment and an EM sparsity sweep share one code
path and one output format.
"""

import json
import logging
from pathlib import Path

import torch

from ..masks import (
    DEFAULT_EVAL_FRACS, FULL_DELTA, PRETRAINED, compose_params, conditions_for, mask_for, plan,
)
from ..masks.compose import apply_in_place
from .base import ModelCtx

logger = logging.getLogger(__name__)

DENSE = "dense"          # the single condition of an unmasked run


class MaskedWeights:
    """Switches a model between points on the sparsity grid, either path.

    ``layout=None`` means "no mask": :meth:`conditions` yields one ``dense`` condition and the
    model is left exactly as it is.
    """

    def __init__(self, model, tokenizer, *, device="cuda", layout=None, scores=None,
                 deltas=None, base=None, buffers=None, aliases=None, mode="necessary",
                 fracs=None):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.layout, self.scores, self.deltas = layout, scores, deltas
        self.buffers = buffers if buffers is not None else dict(model.named_buffers())
        self.aliases = aliases
        self.invert = mode == "sufficient"
        self.fracs = tuple(fracs) if fracs is not None else DEFAULT_EVAL_FRACS
        self._base = base            # caller-supplied theta_base, if any
        self._cpu_cache = None
        self._dev_cache = None

    # The two composition paths want theta_base in different places, and getting this wrong
    # is a device-mismatch crash (functional) or a silently corrupted restore (in-place):
    #
    #   in-place  needs an INDEPENDENT snapshot, because it writes through the live parameters
    #             -- a `base` that is views onto them would be overwritten by the first
    #             condition and `restore()` would put back whatever was written last. Kept on
    #             the CPU so peak GPU cost is one parameter tensor rather than a second model.
    #   functional needs tensors on the MODEL's device, and never writes, so it can alias the
    #             live parameters for free -- they still hold theta_base.

    def _base_cpu(self):
        if self._cpu_cache is None:
            src = self._base if self._base is not None else dict(self.model.named_parameters())
            self._cpu_cache = {n: src[n].detach().cpu().clone() for n in self.layout.names}
        return self._cpu_cache

    def _base_and_deltas_on_device(self):
        if self._dev_cache is None:
            src = self._base if self._base is not None else dict(self.model.named_parameters())
            base = {n: src[n].detach().to(self.device) for n in self.layout.names}
            deltas = {n: d.to(self.device) for n, d in self.deltas.items()}
            self._dev_cache = (base, deltas)
        return self._dev_cache

    @property
    def masked(self) -> bool:
        return self.layout is not None

    def conditions(self):
        """``[(label, k, invert)]`` -- the grid, or one ``dense`` point if there is no mask."""
        if not self.masked:
            return [(DENSE, None, False)]
        return conditions_for(self.fracs, self.layout.total, self.invert)

    def ctx_for(self, k, invert, *, in_place: bool) -> ModelCtx:
        """A :class:`ModelCtx` presenting one condition's weights."""
        if not self.masked:
            return ModelCtx(self.model, self.tokenizer, self.device, params=None,
                            buffers=self.buffers)
        mask = mask_for(k, self.layout, self.scores)
        if in_place:
            apply_in_place(self.model, self._base_cpu(), self.deltas, mask, self.layout,
                           invert=invert)
            return ModelCtx(self.model, self.tokenizer, self.device, params=None,
                            buffers=self.buffers)
        base, deltas = self._base_and_deltas_on_device()
        params = compose_params(base, deltas, mask.to(self.device), self.layout, invert=invert,
                                aliases=self.aliases)
        return ModelCtx(self.model, self.tokenizer, self.device, params=params,
                        buffers=self.buffers)

    def restore(self):
        """Put theta_base back, so the model is never left mid-sweep."""
        if self.masked and self._cpu_cache is not None:
            apply_in_place(self.model, self._base_cpu(), self.deltas,
                           torch.zeros(self.layout.total), self.layout, invert=False)


def sweep(evals, probes, weights: MaskedWeights, *, step=None) -> dict:
    """Run every eval at every condition.

    Returns ``{condition_label: {eval_name: {split: {metric: value}}}}``.

    Duplicate weightings are evaluated once and copied: the grid ends at fraction 1.0 and the
    anchors are the two extremes, so under ``cause`` ``frac_1`` composes the same weights as
    ``full_delta`` (under ``iso``, as ``pretrained``). Both are still reported -- they are
    meaningful ends of the curve -- but generating twice for them is wasted budget.
    """
    evals = list(evals)
    in_place = any(e.needs_real_weights for e in evals)
    conds = weights.conditions()
    if weights.masked:
        total = weights.layout.total
        # nothing is on disk here, so `have_result` is always False; plan() is used purely for
        # its duplicate detection, which is the half that matters in-process
        to_run, to_copy = plan(conds, total, lambda label: False)
    else:
        # the dense condition carries k=None, which weights_key() cannot order against 0 --
        # and there is nothing to deduplicate against anyway
        total, to_run, to_copy = 1, conds, []

    out = {}
    try:
        for label, k, invert in to_run:
            if weights.masked:
                logger.info("condition %s: k=%s/%s (%.3f%%), invert=%s", label, k, total,
                            100 * k / total, invert)
            ctx = weights.ctx_for(k, invert, in_place=in_place)
            out[label] = {e.name: e.run(ctx, probes[e.name]) for e in evals
                          if probes.get(e.name) is not None}
        for label, source in to_copy:
            out[label] = out[source]
            logger.info("condition %s: identical weights to %s, reused", label, source)
    finally:
        if in_place:
            weights.restore()
    return out


def log_results(results: dict, *, step=None, wandb_run=None, prefix="eval"):
    """One log line per (condition, eval, split); flat scalars to wandb."""
    flat = {}
    for label, per_eval in results.items():
        for ev_name, per_split in per_eval.items():
            for split, metrics in per_split.items():
                shown = "  ".join(f"{m}={v:.4g}" for m, v in metrics.items()
                                  if isinstance(v, (int, float)))
                at = "" if step is None else f" @ step {step}"
                logger.info("%s[%s/%s/%s]%s  %s", prefix, label, ev_name, split, at, shown)
                for m, v in metrics.items():
                    if isinstance(v, (int, float)):
                        key = (f"{prefix}/{ev_name}/{split}/{m}" if label == DENSE
                               else f"{prefix}/{ev_name}/{label}/{split}/{m}")
                        flat[key] = v
    if wandb_run is not None and flat:
        wandb_run.log(flat, step=step)
    return flat


def write_json(path, results, *, history=None, meta=None):
    """Dump the results (and the curve over training, if there is one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "meta": meta or {},
        "final": results,
        "history": [{"step": s, "results": r} for s, r in (history or [])],
    }, indent=2, ensure_ascii=False, default=str))
    logger.info("wrote %s", path)


def curve(history, eval_name, split, metric, *, condition=DENSE):
    """``[(step, value)]`` for one metric -- what the plot scripts consume."""
    out = []
    for step, res in history:
        v = (((res.get(condition) or {}).get(eval_name) or {}).get(split) or {}).get(metric)
        if v is not None:
            out.append((step, v))
    return out
