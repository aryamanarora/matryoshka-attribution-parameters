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
    resolve_dtype,
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
                 fracs=None, engine=None, compose_dtype=None, svd=None, inplace_device=None):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.engine = engine         # optional vLLM generator, re-synced per condition
        self.layout, self.scores, self.deltas = layout, scores, deltas
        # `{name: SvdFactors}` for a svd* layout's factored tensors, which have no dense delta at
        # all -- so under pure `svd` mode `deltas` is empty and these are the whole update.
        self.svd = svd or {}
        # Must match what training composed at, or the sweep scores different weights from the
        # ones the run trained -- see the note on apply_in_place. None = the base's dtype, which
        # is what every checkpoint written before mask.delta_dtype existed implies.
        self.compose_dtype = resolve_dtype(compose_dtype)
        self.buffers = buffers if buffers is not None else dict(model.named_buffers())
        self.aliases = aliases
        self.invert = mode == "sufficient"
        self.fracs = tuple(fracs) if fracs is not None else DEFAULT_EVAL_FRACS
        # Where the IN-PLACE path composes; None = follow the deltas, the historical behaviour.
        # "cpu" is the memory-safe frozen-delta default; "model" composes per-shard on whatever
        # device holds each live parameter -- see _inplace_device and eval.inplace_compose.
        self._per_shard = inplace_device == "model"
        self.inplace_device = None if self._per_shard else inplace_device
        self._base = base            # caller-supplied theta_base, if any
        self._snapshot = None
        self._dev_cache = None
        self._svd_cache = None          # keyed by device string, for the in-place path
        self._svd_dev_cache = None      # keyed by name, for the functional path
        self._delta_cache = None

    # The two composition paths want theta_base in different places, and getting this wrong
    # is a device-mismatch crash (functional) or a silently corrupted restore (in-place):
    #
    #   in-place  needs an INDEPENDENT snapshot, because it writes through the live parameters
    #             -- a `base` that is views onto them would be overwritten by the first
    #             condition and `restore()` would put back whatever was written last.
    #   functional needs tensors on the MODEL's device, and never writes, so it can alias the
    #             live parameters for free -- they still hold theta_base.

    def _delta_device(self):
        """Where the deltas live -- CPU for a loaded checkpoint, the GPU mid-training.

        The svd factors are consulted second and are not a fallback: under pure ``svd`` mode there
        is no dense delta to ask, and composing on the wrong device is a crash in the functional
        path and a silently wrong restore in the in-place one.
        """
        for d in self.deltas.values():
            return d.device
        for f in self.svd.values():
            return f.S.device
        return torch.device(self.device)

    def _inplace_device(self):
        """Where the IN-PLACE path composes, which is not always where the deltas live.

        :meth:`_delta_device` answers "where is the delta". That is the right question for the
        FUNCTIONAL path, which must build ``theta_eff`` wherever the graph is. It is the wrong
        question for the in-place path, whose entire purpose is to avoid a second copy of the
        model on the GPU: it writes through the live parameters, so it needs an INDEPENDENT
        ``theta_base`` snapshot, and where that snapshot lands decides whether the run pays a
        whole extra model.

        ``inplace_device`` is therefore passed in rather than inferred. A FROZEN delta (post-hoc,
        ixg, any ``mask.finetuned`` run) can compose on the CPU: nothing changes across
        conditions, so the snapshot and the deltas are moved once and cached, and peak GPU cost
        per condition is one parameter tensor. A TRAINING delta cannot -- it changes every step,
        so a CPU copy could not be cached and every eval point would move a model's worth of
        tensors off and back.

        Inferring this from the deltas' device was wrong in a way worth recording, because it
        looked right: the guard was "no dense delta means svd means compose on CPU", and an svd
        layout on Qwen2.5 is NOT pure -- 336 tensors factor, but the 144 q/k/v BIAS vectors are
        1-D and stay on nonresid units. So ``self.deltas`` was non-empty (1.5 MB of biases), the
        guard never fired, and the snapshot went to the GPU anyway. Size, not emptiness, was the
        thing that mattered, and neither is the actual question -- frozen-ness is.

        Measured on the 14B cell (`scripts/probe_posthoc_memory.py`, job 134284): base aliases at
        27.51 GB, factors 0.51, composed theta_eff 24.61. A 27.5 GB snapshot on top is 80.2 GB
        against a usable 79.18, so the run died ~1 GB short in `compose_svd_tensor`, right after
        the step-0 anchors forced the snapshot into existence.
        """
        if self._per_shard:
            # `eval.inplace_compose: model`: the snapshot, deltas and factors each live on the
            # shard that owns their parameter, so composition is GPU arithmetic and the whole
            # per-condition CPU pass plus H2D copy disappears. The price is a second model's
            # worth of GPU memory (bf16, spread over the shards) held for the run -- the reason
            # "cpu" stays the default; see the OOM account below.
            return "model"
        if self.inplace_device is not None:
            return torch.device(self.inplace_device)
        return self._delta_device()

    def _svd_on(self, device):
        """The factors on ``device``, cached. Small enough to mirror: 0.51 GB at 14B rank 32."""
        return self._mirror("_svd_cache", self.svd, device,
                            lambda f, d: f.to(d) if hasattr(f, "to") else f)

    def _deltas_on(self, device):
        """The dense deltas on ``device``, cached. Frozen, so one move serves every condition."""
        return self._mirror("_delta_cache", self.deltas, device, lambda t, d: t.to(d))

    def _mirror(self, attr, src, device, move):
        if not src:
            return src
        cache = getattr(self, attr, None) or {}
        setattr(self, attr, cache)
        key = str(device)
        if key not in cache:
            if device == "model":   # per-shard: each tensor follows its live parameter
                placed = dict(self.model.named_parameters())
                of = lambda n: placed[n].device if n in placed else torch.device(self.device)
                cache[key] = {n: move(v, of(n)) for n, v in src.items()}
            else:
                cache[key] = {n: move(v, device) for n, v in src.items()}
        return cache[key]

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
            dev = self._inplace_device()
            placed = dict(self.model.named_parameters())
            src = self._base if self._base is not None else placed
            of = (lambda n: placed[n].device) if dev == "model" else (lambda n: dev)
            self._snapshot = {n: src[n].detach().to(of(n)).clone() for n in self.layout.names}
        return self._snapshot

    def _base_and_deltas_on_device(self):
        # theta_base is cached -- it does not change within a run. The DELTAS are deliberately
        # not: mid-training they change every step, and a cached copy would serve step N's
        # weights at step N+eval_every. `.to()` is an identity when they are already on the
        # right device, which is the normal case (they are allocated there), so re-resolving
        # them per condition is free in practice and correct when it isn't.
        # PER-TENSOR device, not one global one. `self.device` is right when the model sits on a
        # single card and wrong under `train.device_map`, where it does two damaging things at
        # once: it drags the whole sharded base onto cuda:0 (a second full model's worth of
        # memory, which is an OOM at 14B) and then hands `functional_call` parameters on cuda:0
        # while the modules they belong to live on cuda:1-3, so the forward dies with "mat1 is on
        # cuda:1, different from other tensors on cuda:0". One line, both symptoms.
        #
        # The live model's own parameters are the authority on where each tensor belongs: on one
        # card every entry is `self.device` and this is exactly what it was before; sharded, each
        # follows its block. Falls back to `self.device` for a name the model does not expose
        # (tied weights reached through an alias).
        placed = dict(self.model.named_parameters())
        dev_of = lambda n: placed[n].device if n in placed else torch.device(self.device)
        if self._dev_cache is None:
            src = self._base if self._base is not None else placed
            self._dev_cache = {n: src[n].detach().to(dev_of(n)) for n in self.layout.names}
        return self._dev_cache, {n: d.to(dev_of(n)) for n, d in self.deltas.items()}

    def _svd_on_device(self):
        """The factors, on the model's device. Cached: they are constants (the delta is frozen).

        Unlike the deltas -- which mid-training change every step and so must be re-resolved per
        condition -- a ``svd*`` layout's factors come from a delta that ``config/schema.py``
        requires to be given and frozen, so caching them cannot serve stale weights.
        """
        if not self.svd:
            return {}
        # Per-tensor, for the reason `_base_and_deltas_on_device` explains: a factored tensor's
        # update has to be reconstructed on the shard that holds the tensor. Identical to the old
        # behaviour on one card.
        #
        # A SEPARATE cache attribute from `_svd_on`'s, deliberately. That one is keyed by device
        # string ({"cuda:0": {name: factors}}) because the in-place path asks for a device
        # explicitly; this one is keyed by name. They are different shapes, and sharing
        # `_svd_cache` between them -- which they did -- would hand one path the other's dict the
        # first time both ran in a process.
        if self._svd_dev_cache is None:
            placed = dict(self.model.named_parameters())
            self._svd_dev_cache = {
                n: f.to(device=(placed[n].device if n in placed else torch.device(self.device)))
                for n, f in self.svd.items()}
        return self._svd_dev_cache

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
            dev = self._inplace_device()
            base = self._base_snapshot()
            # mask_for short-circuits k<=0 / k>=total with a fresh CPU tensor, so it has to be
            # moved even though `scores` may already be on the right device. Under per-shard
            # composition it goes to the runner's device and `_composed` moves each tensor's
            # SLICE to that tensor's shard -- slices are tiny, whole-vector mirrors are not.
            mdev = self.device if dev == "model" else dev
            apply_in_place(self.model, base, self._deltas_on(dev), mask.to(mdev),
                           self.layout, invert=invert, out_dtype=self.compose_dtype,
                           svd=self._svd_on(dev))
            if needs_engine:
                self.engine.sync_from(self.model)
            return mk(params=None, engine=self.engine if needs_engine else None)
        base, deltas = self._base_and_deltas_on_device()
        # The composer moves each tensor's SLICE of the mask to the tensor's device on demand, so
        # a mask that is already on the host can stay there. Moving it whole first is what a
        # per-weight layout cannot afford: 7B fp32 units are 28 GB, beside the model, its delta
        # and the composed copy -- the OOM that killed the first per-parameter sweep. Small
        # masks are moved whole as before (one transfer instead of one per tensor).
        if not (mask.device.type == "cpu" and self.layout.total > 100_000_000):
            mask = mask.to(self.device)
        return mk(params=compose_params(base, deltas, mask, self.layout,
                                        invert=invert, aliases=self.aliases,
                                        out_dtype=self.compose_dtype,
                                        svd=self._svd_on_device()))

    def restore(self):
        """Put theta_base back, so the model is never left mid-sweep.

        A no-op when no snapshot was ever taken: that means the in-place path never ran, so
        the live weights still hold theta_base.
        """
        if self.masked and self._snapshot is not None:
            # A DIRECT copy of the snapshot, not a zero-mask recompose. `base + 0*delta` cast
            # into the parameter's dtype is exactly the snapshot cast on `copy_` (the cast
            # happens either way, on the same values), and the copy is one memory pass where
            # the recompose was a multiply, an add, a cast and a copy. Device-safe by
            # construction: `copy_` moves across devices itself.
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, t in self._base_snapshot().items():
                    params[n].data.copy_(t)


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

    # Duplicate weightings, again, for the two-phase evals. The copy above ran before finalize, and
    # a two-phase `run` returns nothing to copy -- so `full_delta` came out of the loop empty and
    # finalize only knows the labels that actually generated. Without this the deduplicated anchor
    # is silently absent from a two-phase eval's curve (em has always lost it this way), which reads
    # as a condition that failed rather than one that was reused.
    for label, source in to_copy:
        for name, per_split in (out.get(source) or {}).items():
            out.setdefault(label, {}).setdefault(name, per_split)
    return out


#: files this PROCESS has already written, so the first write of a run truncates and later ones
#: append -- see :func:`dump_records`
_OPENED = set()


def dump_records(evals, probes, out_dir, step=None):
    """Persist the per-response records an eval accumulated, to ``<eval>_eval/generations.jsonl``.

    Opt-in per eval, via a ``drain_records`` method (``language`` and ``strongreject`` have one).
    It lives here rather than in the training loop because both drivers need it: a percentage over
    a few dozen samples is only interpretable next to the text behind it, and the post-hoc sweep is
    where most generative evals are actually run.

    **The first write of a process truncates; the rest append.** Within one run this has to
    append -- a training loop calls it once per eval point and the file is the whole trajectory --
    but ACROSS runs it must not, and it used to. Resubmitting a config reuses its output directory
    (the hazard CLAUDE.md records for ``evals.json``), so a re-run left the previous attempt's
    generations in place and wrote its own after them: measured on ``runs/abliteration_*``, where a
    re-run with a corrected direction produced a 400-row GSM8K file holding 200 responses from each
    of two different models. ``evals.json`` is overwritten and so stayed correct, which is what
    makes the mixed file dangerous rather than obviously broken -- the metrics agree with only
    half of it, and anything reading the text gets both.
    """
    for ev in evals:
        drain = getattr(ev, "drain_records", None)
        if drain is None:
            continue
        recs = drain(probes[ev.name])
        if not recs:
            continue
        d = Path(out_dir) / f"{ev.name}_eval"
        d.mkdir(parents=True, exist_ok=True)
        path = (d / "generations.jsonl").resolve()
        mode = "a" if path in _OPENED else "w"
        _OPENED.add(path)
        with path.open(mode) as f:
            for r in recs:
                f.write(json.dumps(dict(r, step=step), ensure_ascii=False) + "\n")
        logger.info("%s %d generation(s) %s %s", "wrote" if mode == "w" else "appended",
                    len(recs), "to" if mode == "w" else "to", path)


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


def log_results(results: dict, *, step=None, wandb_run=None, prefix="eval", wandb_step=None):
    """One log line per (condition, eval, split); every scalar beneath it to wandb.

    ``wandb_step`` overrides the step the scalars are logged AT, without changing the human log
    line: a GRPO run has already advanced wandb's step counter past its training steps, so the
    final eval has to log above that range or wandb drops it as non-monotonic. Defaults to ``step``.
    """
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
        wandb_run.log(flat, step=wandb_step if wandb_step is not None else step)
    return flat


def sweep_aucs(results: dict, fracs, *, prefix="eval") -> dict:
    """``{<prefix>/<eval>/<split>/<metric>_log_auc: value}`` for every swept metric.

    ONE NUMBER PER CURVE, so a sweep is comparable in a wandb table and not only by eye. The
    weighting is MIB's ``acc_auc`` (``MIB-circuit-track/MIB_circuit_track/evaluation.py``),
    transcribed rather than invented so a number here means what a number there means:

        log_auc = sum_i (log x_{i+1} - log x_i) * (y_i + y_{i+1})/2  /  (log x_last - log x_first)

    a trapezoid in LOG sparsity normalised by the log range -- i.e. a weighted mean of the curve in
    which every DECADE of sparsity contributes equally.

    WHY LOG AND NOT LINEAR, measured on this repo's own fr2de optimizer sweep rather than asserted:
    the grid is geometric, so on a linear x the 0.5-1.0 interval is half the width and every method
    has already converged there onto a shared ``frac_1`` anchor. Linear AUC put best-Adam at 0.934
    against best-SGD 0.944 (a 1% gap, reads as a tie); log-AUC put them at 0.577 and 0.679 (17%
    relative). The two weightings support opposite conclusions and the sparse end is the half of
    the curve these experiments are about.

    ``linear_auc`` is logged beside it, deliberately: it is the dense-end-dominated view, and
    having both in the same run makes the gap visible to anyone reading the table instead of a
    claim they have to take on trust.

    ANCHORS ARE EXCLUDED. ``pretrained`` has no sparsity, so it has no place on a log-x axis --
    giving it a nominal x would invent a decade. ``full_delta`` aliases ``frac_1`` under ``cause``
    and would double-weight the last point. Both remain available as their own scalars.

    NOT A SUBSTITUTE FOR THE CURVE. A single AUC cannot distinguish a monotone rise from one that
    overshoots and falls back, which is exactly the "a sparse mask beats the whole finetune" shape
    this repo keeps finding -- so ``_peak`` is logged too, and peak > full_delta is its signature.
    """
    import math

    xs = sorted(float(f) for f in fracs if float(f) > 0)
    if len(xs) < 2:
        return {}
    key = lambda fr: f"frac_{fr:g}"
    triples = sorted({
        (ev, sp, m)
        for label, per_eval in results.items() if label.startswith("frac_")
        for ev, per_split in per_eval.items()
        for sp, metrics in per_split.items()
        for m, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and m not in ("n", "n_scored")
    })
    at = lambda label, ev, sp, m: (
        (((results.get(label) or {}).get(ev) or {}).get(sp) or {}).get(m))

    lx = [math.log(x) for x in xs]
    out = {}
    for ev, sp, m in triples:
        ys = [at(key(fr), ev, sp, m) for fr in xs]
        if any(y is None for y in ys):
            continue                     # a partial curve has no defensible AUC
        span = lx[-1] - lx[0]
        base = f"{prefix}/{ev}/{sp}/{m}"
        out[f"{base}_log_auc"] = sum(
            (lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1)) / span
        out[f"{base}_lin_auc"] = sum(
            (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1)) / (
                xs[-1] - xs[0])
        out[f"{base}_peak"] = max(ys)
    return out


def curve_panels(wandb_run, history, fracs, *, step=None, prefix="eval", wandb_step=None):
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
        wandb_run.log(panels, step=wandb_step if wandb_step is not None else step)
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
