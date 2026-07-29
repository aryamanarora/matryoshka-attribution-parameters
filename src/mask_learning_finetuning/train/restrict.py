"""Restricting a full finetune to the top-k units of a mask that was already fitted.

The third question in the family, and the one the other two cannot ask. ``loop.train`` with a
``mask:`` block co-trains a mask with the delta -- "what does a finetune look like when it is
*pushed* to be localised". ``posthoc`` fits a mask over a finished delta -- "how localised was
the finetune that already happened". Both read out a *ranking*. This module takes a ranking as
**given** and re-runs the finetune with every non-selected component frozen, which asks whether
those units are *sufficient*: with only them free to move, does the behaviour still land?

That is a strictly stronger claim than the sparsity sweep gives. The sweep ablates a delta the
full finetune produced, so a top-k that recovers the behaviour shows those units *carry* an
update that was computed with all the others moving too. Training under the restriction removes
that crutch -- the optimizer has to reach the behaviour through the selected units alone.

Only the full-parameter path (``params.Direct``) supports it, enforced in ``config/schema.py``.
Under ``lora:`` the trained parameters are PEFT's ``lora_A``/``lora_B``, which are not the
tensors any layout scores, and under ``mask:`` the mask *is* the thing being learned.

**How the freeze is implemented, and why not ``requires_grad``.** A unit is generally a slice of
a tensor (one row of ``q_proj.weight``), so ``requires_grad`` -- which is per tensor -- cannot
express it. Two mechanisms stand in, and both are needed:

1. a post-accumulate-grad hook multiplies each gradient by the mask, so a frozen component's
   gradient is zero before the optimizer ever sees it (during backward, not at step time, so the
   ``|g|`` the loop logs is the norm of the *restricted* gradient and not a number that includes
   directions nothing can move in);
2. AdamW's decoupled weight decay is applied by hand, masked. This is the part that is easy to
   miss: decay is ``p -= lr . wd . p`` and does **not** go through the gradient, so a zero
   gradient does not protect a frozen component from it -- it would shrink toward zero for the
   whole run while reported as frozen. :meth:`Restriction.decay` performs the same multiply
   torch's AdamW performs, in the same place, against ``1 - lr . wd . m``.

A tensor with no selected units at all is frozen the ordinary way (``requires_grad_(False)``),
which keeps it out of the optimizer and saves its two AdamW moment buffers.

Tied weights need no alias map here, unlike ``compose_params`` -- for the same reason
``apply_in_place`` doesn't. With ``tie_word_embeddings``, ``lm_head.weight`` *is*
``embed_tokens.weight``, one tensor under two names; the layout holds the canonical name,
gradients from both uses accumulate into the one ``.grad``, and the hook masks it once. What that
does mean is that a mask over the embedding governs the output head too, which is a property of
the tied model and not a choice made here.
"""

import logging
from pathlib import Path

import torch

from ..masks import expand_mask, mask_for
from ..masks.checkpoint import layout_from_blob, load_checkpoint

logger = logging.getLogger(__name__)


def resolve_checkpoint(spec: str):
    """Load the blob at ``spec``, which may be a run directory or a ``.pt`` file.

    ``require_delta=False``: only the scores and the layout are wanted. A mask fitted post hoc
    is usually saved without its delta (``mask.save_delta`` is off by default), and there is
    nothing here for a delta to be used for -- the finetune is re-run, not composed.
    """
    p = Path(spec)
    run_dir, name = (p, None) if p.is_dir() else (p.parent, p.name)
    return load_checkpoint(run_dir, name, require_delta=False)


class Restriction:
    """A hard top-k mask over a model's parameter components, and the freeze it implies.

    Construction has a side effect on ``model``: every parameter with no selected component has
    ``requires_grad_(False)`` set on it, so an optimizer built afterwards sees only what may
    move. :meth:`attach` then installs the gradient hooks. The two are separate because the
    optimizer must be built between them -- see :class:`~.params.Restricted`.
    """

    def __init__(self, model, rc):
        path, blob = resolve_checkpoint(rc.checkpoint)
        resid_dim = (getattr(model.config, "hidden_size", None)
                     or getattr(model.config, "n_embd", None))
        # resid_dim from the LIVE model rather than letting layout_from_blob fetch the config of
        # the model the mask was fitted on: same number, no network round trip, and it is only
        # consulted at all for pre-`axes` nonresid checkpoints
        layout = layout_from_blob(blob, resid_dim=resid_dim)
        scores = blob["scores"].detach().float().cpu().flatten()
        if scores.numel() != layout.total:
            raise SystemExit(f"{path} holds {scores.numel()} scores but a layout of "
                             f"{layout.total} units; the checkpoint is inconsistent")
        args = blob.get("args") or {}
        _check_provenance(path, args, model)

        k = rc.k if rc.k is not None else max(1, int(round(rc.frac * layout.total)))
        k = max(1, min(int(k), layout.total))
        # same k-from-frac rounding as masks.sweep.conditions_for, so "trained at frac 0.01"
        # names the same set of units as the sweep's frac_0.01 point on the same checkpoint
        flat = mask_for(k, layout, scores)
        if rc.invert:
            flat = 1.0 - flat

        params = dict(model.named_parameters())
        missing = [n for n in layout.names if n not in params]
        if missing:
            raise SystemExit(f"{len(missing)} scored tensors are absent from this model, e.g. "
                             f"{missing[:3]}. Was the mask fitted on a different one?")

        self.masks, self.params = {}, {}
        trainable_params, n_frozen_tensors = 0, 0
        for i, name in enumerate(layout.names):
            p = params[name]
            if tuple(p.shape) != tuple(layout.shapes[i]):
                raise SystemExit(f"shape mismatch for {name}: model {tuple(p.shape)} vs "
                                 f"checkpoint {tuple(layout.shapes[i])}")
            sl = flat[layout.slice_for(i)]
            if not bool(sl.any()):
                n_frozen_tensors += 1
                continue
            # a broadcast view, not a materialised full-size mask: for row/col/tensor units this
            # is a few thousand floats per tensor rather than another copy of the model. `weight`
            # units are the exception and cost one full tensor, which is what that mode costs
            # anyway.
            m = expand_mask(sl, layout.shapes[i], layout.axes[i]).to(p.device, p.dtype)
            self.masks[name], self.params[name] = m, p
            trainable_params += int(m.sum()) * (p.numel() // m.numel())

        scored = set(layout.names)
        n_unscored, unscored_params = 0, 0
        for name, p in model.named_parameters():
            p.requires_grad_(name in self.masks)
            if name not in scored:
                n_unscored += 1
                unscored_params += p.numel()
        if not self.masks:
            raise SystemExit(
                f"the restriction selects no components at all (k={k}, invert={rc.invert}), so "
                "there would be nothing to train")

        total_params = sum(p.numel() for p in model.parameters())
        self.k, self.total_units, self.invert = k, layout.total, rc.invert
        self.stats = {
            "checkpoint": str(path), "unit": layout.mode, "total_units": layout.total,
            # k and frac describe the TOP-K; with invert on, the trainable set is its complement
            "k": k, "frac": k / layout.total, "invert": rc.invert,
            "trainable_units": int(flat.sum()),
            "trainable_unit_frac": float(flat.sum()) / layout.total,
            "scored_tensors": len(layout.names), "trainable_tensors": len(self.masks),
            "fully_frozen_tensors": n_frozen_tensors, "unscored_tensors": n_unscored,
            "unscored_params": unscored_params,
            "trainable_params": trainable_params, "total_params": total_params,
            # what the ranking came from -- the mask's own provenance, which is the only record
            # of WHICH finetune's localisation is being tested
            "fit_model": args.get("model"), "fit_mode": args.get("mode"),
            "fit_scores": args.get("scores"), "fit_finetuned": args.get("finetuned"),
            "fit_dataset": args.get("dataset"),
        }
        self._handles = []
        n_units = int(flat.sum())
        # `k` and the trainable count are the same number only when invert is off, so say which
        # is which rather than printing one percentage that could be read as either
        selected = (f"the complement of the top {k:,} ({n_units:,} units, "
                    f"{100 * n_units / layout.total:.3g}%)" if rc.invert
                    else f"the top {k:,} units ({100 * k / layout.total:.3g}%)")
        logger.info(
            "restricted to %s of %s %s units from %s: %s trainable / %s parameters (%.3f%%) "
            "over %d tensors; %d tensor(s) fully frozen, %d unscored (%s parameters)",
            selected, f"{layout.total:,}", layout.mode, path,
            f"{trainable_params:,}", f"{total_params:,}",
            100 * trainable_params / max(1, total_params), len(self.masks),
            n_frozen_tensors, n_unscored, f"{unscored_params:,}")

    def attach(self, model) -> None:
        """Install the gradient masks.

        A post-accumulate-grad hook rather than a pass at step time: the hook fires during
        backward, so ``params.Direct.grad_norm`` -- which the loop calls *before* ``step`` --
        already reports the restricted norm. Masking is idempotent, so firing once per
        accumulated micro-batch is harmless.
        """
        for name, p in model.named_parameters():
            m = self.masks.get(name)
            if m is None:
                continue
            self._handles.append(p.register_post_accumulate_grad_hook(_masker(m)))

    def pair_with(self, opt_params) -> list:
        """``[(param, mask)]`` for the subset of ``opt_params`` this restriction masks.

        Called once with the optimizer's *decay* group, so :func:`masked_decay` does not have to
        re-derive which parameters weight decay applies to -- that rule lives in
        ``params.Direct._setup`` and must not be stated twice.
        """
        by_id = {id(p): m for p, m in ((self.params[n], self.masks[n]) for n in self.masks)}
        return [(p, by_id[id(p)]) for p in opt_params if id(p) in by_id]


@torch.no_grad()
def masked_decay(pairs, lr: float, wd: float) -> None:
    """AdamW's decoupled weight decay, applied to the trainable components only.

    ``pairs`` comes from :meth:`Restriction.pair_with`, over an optimizer group whose own
    ``weight_decay`` the caller has set to zero. Torch's AdamW multiplies by
    ``1 - lr * weight_decay`` as the first operation of its update, so doing the masked version
    here, immediately before ``opt.step()``, is the same multiply in the same place.

    ``m * factor + (1 - m)`` rather than the more obvious ``1 - lr * wd * m``: masks are strictly
    0/1, so ``1.0 * factor`` is ``factor`` to the bit, whereas ``1 - fp32(lr * wd)`` and
    ``fp32(1 - lr * wd)`` can round apart in the last place (for ~0.6% of (lr, wd) pairs sampled
    log-uniformly, and more often as ``lr . wd`` grows). Written this way an all-ones restriction
    reproduces AdamW's own decay exactly, which makes ``restrict.frac: 1.0`` a bit-identical rerun
    of the unrestricted finetune and therefore a usable control. ``tests/test_restrict.py`` pins
    that.
    """
    if not wd:
        return
    factor = 1 - lr * wd
    for p, m in pairs:
        p.mul_(m * factor + (1 - m))


def _masker(m):
    def hook(param):
        if param.grad is not None:
            param.grad.mul_(m)
    return hook


def _check_provenance(path, args: dict, model) -> None:
    """Warn where the mask's own training config makes this restriction mean something else.

    Both cases are silent-wrong-answer shaped rather than crash shaped, which is why they are
    checked here instead of being left to the reader of the run directory.
    """
    fit_model = args.get("model")
    live = getattr(model.config, "_name_or_path", None)
    if fit_model and live and fit_model != live:
        logger.warning("the mask at %s was fitted on %s but this run trains %s; the unit names "
                       "matched, so this may just be a local path vs a hub id -- check that it "
                       "is", path, fit_model, live)
    if args.get("mode") == "sufficient":
        # Under sufficient/iso the delta lands on the COMPLEMENT of the top-k, so minimising the
        # loss pushes high scores onto units the finetune does NOT need -- the ranking is
        # upside down for this purpose and the run would report a null with nothing wrong in it.
        logger.warning(
            "the mask at %s was fitted with mode=sufficient/iso, where the delta is applied to "
            "the COMPLEMENT of the top-k. Its high-scoring units are therefore the ones the "
            "finetune could do without, which is the opposite of what restricting training to "
            "them assumes. Use a mask fitted with mode=cause/necessary, or set restrict.invert",
            path)
