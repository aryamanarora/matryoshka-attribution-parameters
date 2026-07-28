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
                 fracs=None, engine=None):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.engine = engine         # optional vLLM generator, re-synced per condition
        self.layout, self.scores, self.deltas = layout, scores, deltas
        self.buffers = buffers if buffers is not None else dict(model.named_buffers())
        self.aliases = aliases
        self.invert = mode == "sufficient"
        self.fracs = tuple(fracs) if fracs is not None else DEFAULT_EVAL_FRACS
        self._base = base            # caller-supplied theta_base, if any
        self._snapshot = None
        self._dev_cache = None

    # The two composition paths want theta_base in different places, and getting this wrong
    # is a device-mismatch crash (functional) or a silently corrupted restore (in-place):
    #
    #   in-place  needs an INDEPENDENT snapshot, because it writes through the live parameters
    #             -- a `base` that is views onto them would be overwritten by the first
    #             condition and `restore()` would put back whatever was written last.
    #   functional needs tensors on the MODEL's device, and never writes, so it can alias the
    #             live parameters for free -- they still hold theta_base.

    def _delta_device(self):
        """Where the deltas live -- CPU for a loaded checkpoint, the GPU mid-training."""
        for d in self.deltas.values():
            return d.device
        return torch.device(self.device)

    def _base_snapshot(self):
        """An independent theta_base, on the same device as the deltas.

        Following the deltas rather than pinning to the CPU is what makes this work in both
        situations, and mixing them is a device-mismatch crash:

          post-hoc     deltas come off a checkpoint on the CPU, so the snapshot is on the CPU
                       too and composition happens there -- peak GPU cost stays one parameter
                       tensor rather than a second copy of the model plus an fp32 delta.
          in-training  the deltas ARE the live training tensors, on the GPU. Copying them to
                       the CPU for every eval point would mean moving a model's worth of fp32
                       every time (and they change every step, so it could not be cached), so
                       the snapshot goes to the GPU instead: one extra allocation for the run.
        """
        if self._snapshot is None:
            dev = self._delta_device()
            src = self._base if self._base is not None else dict(self.model.named_parameters())
            self._snapshot = {n: src[n].detach().to(dev).clone() for n in self.layout.names}
        return self._snapshot

    def _base_and_deltas_on_device(self):
        # theta_base is cached -- it does not change within a run. The DELTAS are deliberately
        # not: mid-training they change every step, and a cached copy would serve step N's
        # weights at step N+eval_every. `.to()` is an identity when they are already on the
        # right device, which is the normal case (they are allocated there), so re-resolving
        # them per condition is free in practice and correct when it isn't.
        if self._dev_cache is None:
            src = self._base if self._base is not None else dict(self.model.named_parameters())
            self._dev_cache = {n: src[n].detach().to(self.device) for n in self.layout.names}
        return self._dev_cache, {n: d.to(self.device) for n, d in self.deltas.items()}

    @property
    def masked(self) -> bool:
        return self.layout is not None

    def conditions(self):
        """``[(label, k, invert)]`` -- the grid, or one ``dense`` point if there is no mask."""
        if not self.masked:
            return [(DENSE, None, False)]
        return conditions_for(self.fracs, self.layout.total, self.invert)

    def ctx_for(self, k, invert, *, in_place: bool, label=DENSE, step=None,
                final=False) -> ModelCtx:
        """A :class:`ModelCtx` presenting one condition's weights."""
        mk = lambda **kw: ModelCtx(self.model, self.tokenizer, self.device,
                                   buffers=self.buffers, label=label, step=step,
                                   final=final, **kw)
        # The vLLM engine keeps its own copy of the weights, so it is re-synced from the model
        # here, AFTER the condition is composed and while nothing has generated yet. Doing it
        # anywhere else -- once per run, or lazily on first generate -- serves one condition's
        # weights under another condition's label, which is a wrong number that looks fine.
        # `in_place` is also the answer to "will anything generate under this condition", so a
        # sweep of forward-only evals never pays for a weight push it cannot use.
        needs_engine = in_place and self.engine is not None
        if not self.masked:
            if needs_engine:
                self.engine.sync_from(self.model)
            return mk(params=None, engine=self.engine if needs_engine else None)
        mask = mask_for(k, self.layout, self.scores)
        if in_place:
            base = self._base_snapshot()
            # mask_for short-circuits k<=0 / k>=total with a fresh CPU tensor, so it has to be
            # moved even though `scores` may already be on the right device
            apply_in_place(self.model, base, self.deltas, mask.to(self._delta_device()),
                           self.layout, invert=invert)
            if needs_engine:
                self.engine.sync_from(self.model)
            return mk(params=None, engine=self.engine if needs_engine else None)
        base, deltas = self._base_and_deltas_on_device()
        return mk(params=compose_params(base, deltas, mask.to(self.device), self.layout,
                                        invert=invert, aliases=self.aliases))

    def restore(self):
        """Put theta_base back, so the model is never left mid-sweep.

        A no-op when no snapshot was ever taken: that means the in-place path never ran, so
        the live weights still hold theta_base.
        """
        if self.masked and self._snapshot is not None:
            dev = self._delta_device()
            apply_in_place(self.model, self._base_snapshot(), self.deltas,
                           torch.zeros(self.layout.total, device=dev), self.layout,
                           invert=False)


def sweep(evals, probes, weights: MaskedWeights, *, step=None, final=False) -> dict:
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

    active = [e for e in evals if probes.get(e.name) is not None]
    out = {}
    try:
        for label, k, invert in to_run:
            if weights.masked:
                logger.info("condition %s: k=%s/%s (%.3f%%), invert=%s", label, k, total,
                            100 * k / total, invert)
            ctx = weights.ctx_for(k, invert, in_place=in_place, label=label,
                                  step=step, final=final)
            per_eval = {}
            for e in active:
                res = e.run(ctx, probes[e.name])
                if res is not None:            # None => two-phase, scored in finalize()
                    per_eval[e.name] = res
            out[label] = per_eval
        for label, source in to_copy:
            out[label] = dict(out[source])
            logger.info("condition %s: identical weights to %s, reused", label, source)
    finally:
        if in_place:
            weights.restore()

    # Second phase for evals that batch their scoring across conditions rather than within one.
    # Runs after restore(), deliberately: judging touches no weights, and leaving the model
    # dirty across a long API-bound stage is how a later eval silently scores the wrong thing.
    for e in active:
        fin = getattr(e, "finalize", None)
        if fin is None:
            continue
        for label, per_split in (fin(probes[e.name]) or {}).items():
            out.setdefault(label, {})[e.name] = per_split
    return out


def _flatten(d, path, out):
    """Collect every scalar under ``d`` into ``out``, keyed by its ``/``-joined path.

    Recursive on purpose. A split's metrics are usually scalars, but an eval is free to group them
    a level deeper -- ``language`` used to, reporting every fraction once per language-id backend.
    A non-recursive walk silently dropped ALL of those: the value at ``metrics["langdetect"]`` was
    a dict, so an ``isinstance(v, (int, float))`` filter discarded the entire eval, and both its
    wandb series and its log line came out empty while the JSON looked fine. That eval now reports
    flat metrics (one scorer, and ``script`` is a separate eval rather than a second backend), so
    nothing in the repo currently needs the recursion -- it stays because the next eval to group
    its metrics should not have to rediscover this.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten(v, path + (str(k),), out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out["/".join(path + (str(k),))] = v
    return out


def log_results(results: dict, *, step=None, wandb_run=None, prefix="eval"):
    """One log line per (condition, eval, split); every scalar beneath it to wandb."""
    flat = {}
    for label, per_eval in results.items():
        for ev_name, per_split in per_eval.items():
            for split, metrics in per_split.items():
                # the condition is omitted from the key for an unmasked run, so a plain
                # finetune's series are `eval/language/off_target/...` rather than
                # `eval/language/dense/off_target/...`
                head = (prefix, ev_name) if label == DENSE else (prefix, ev_name, label)
                scalars = _flatten(metrics, head + (split,), {})
                flat.update(scalars)
                at = "" if step is None else f" @ step {step}"
                base = "/".join(head + (split,)) + "/"
                shown = "  ".join(f"{k[len(base):]}={v:.4g}" for k, v in scalars.items())
                logger.info("%s[%s/%s/%s]%s  %s", prefix, label, ev_name, split, at, shown)
    if wandb_run is not None and flat:
        wandb_run.log(flat, step=step)
    return flat


def curve_panels(wandb_run, history, fracs, *, step=None, prefix="eval"):
    """wandb ``line_series`` panels: the metric-vs-sparsity curve, and its transpose.

    The per-condition scalars are already logged, which answers "how did loss@2% evolve" but
    never "what does the curve look like". These are the views the pre-refactor scripts had, and
    they are generic over evals/splits/metrics rather than written once per eval:

    ``<prefix>/<eval>/<split>/<metric>_vs_frac``
        the final curve, with the ``pretrained`` and ``full_delta`` anchors drawn as flat
        reference lines so you can see where the sparse mask crosses them.
    ``<prefix>/<eval>/<split>/<metric>_vs_frac_over_train``
        one line per eval step -- how the whole curve moves as training proceeds.
    ``<prefix>/<eval>/<split>/<metric>_over_steps``
        the transpose: x is the train step, one line per mask %, plus both anchors. This is the
        "does the sparse mask keep improving, or plateau while the dense one overfits" view.
        Note ``train/loss`` is not a substitute -- it is measured at whatever k the schedule drew
        that step, so it is not comparable across steps, and these fixed-k curves are.

    x is the mask fraction; switch the panel to a log x-axis in the UI, since custom charts
    cannot declare that programmatically and the grid spans 0.1%-100%.
    """
    if wandb_run is None or not history:
        return
    import wandb

    xs = [float(f) for f in fracs]
    key = lambda fr: f"frac_{fr:g}"
    _, last = history[-1]
    # (eval, split, metric) triples that actually have a swept point to plot
    triples = sorted({
        (ev, sp, m)
        for label, per_eval in last.items() if label.startswith("frac_")
        for ev, per_split in per_eval.items()
        for sp, metrics in per_split.items()
        for m, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and m != "n"
    })
    at = lambda res, label, ev, sp, m: (
        (((res.get(label) or {}).get(ev) or {}).get(sp) or {}).get(m))

    panels = {}
    for ev, sp, m in triples:
        base = f"{prefix}/{ev}/{sp}/{m}"
        curve = [at(last, key(fr), ev, sp, m) for fr in xs]
        if any(v is None for v in curve):
            continue
        ys, keys = [curve], [m]
        for anchor in (PRETRAINED, FULL_DELTA):
            a = at(last, anchor, ev, sp, m)
            if a is not None:
                ys.append([a] * len(xs))
                keys.append(anchor.replace("_", " "))
        panels[f"{base}_vs_frac"] = wandb.plot.line_series(
            xs=xs, ys=ys, keys=keys, xname="mask fraction",
            title=f"{ev}/{sp} {m} vs mask fraction")

        if len(history) > 1:
            ys, keys = [], []
            for st, res in history:
                c = [at(res, key(fr), ev, sp, m) for fr in xs]
                if not any(v is None for v in c):
                    ys.append(c)
                    keys.append(f"step {st}")
            if ys:
                panels[f"{base}_vs_frac_over_train"] = wandb.plot.line_series(
                    xs=xs, ys=ys, keys=keys, xname="mask fraction",
                    title=f"{ev}/{sp} {m} vs mask fraction, over training")

            steps = [st for st, _ in history]
            ys, keys = [], []
            for fr in xs:
                series = [at(res, key(fr), ev, sp, m) for _, res in history]
                if not any(v is None for v in series):
                    ys.append(series)
                    keys.append(f"{fr:.1%} of units")
            for anchor in (PRETRAINED, FULL_DELTA):
                series = [at(res, anchor, ev, sp, m) for _, res in history]
                if not any(v is None for v in series):
                    ys.append(series)
                    keys.append(anchor.replace("_", " "))
            if ys:
                panels[f"{base}_over_steps"] = wandb.plot.line_series(
                    xs=steps, ys=ys, keys=keys, xname="train step",
                    title=f"{ev}/{sp} {m} vs train step, by mask %")
    if panels:
        wandb_run.log(panels, step=step)
        logger.info("logged %d wandb curve panel(s)", len(panels))


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
