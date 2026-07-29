"""Input times Gradient over the weight delta: a closed-form baseline for the learned mask.

The learned mask answers "which units carry this finetune" by gradient descent through a
differentiable top-k, over hundreds of steps. IxG answers the same question with one first-order
Taylor term and no optimisation at all, which is exactly what makes it the baseline worth beating:
if a mask trained for 450 steps does not rank units better than

    a_i = sum over unit i of  delta * dL/dtheta

then the learned ranking is not buying anything.

The sign, and why the same formula serves both gradient points
--------------------------------------------------------------
Write ``S`` for the set of units the mask keeps, so ``theta_eff = theta_base + delta_S``.

* **Gradient at the base model.** ``L(theta_base + delta_S) ~= L(theta_base) + sum_S delta_i . g_i``,
  so adding unit ``i`` changes the loss by ``delta_i . g_i`` and the units worth keeping are the
  ones where that is most **negative**.
* **Gradient at the finetuned model.** Now the mask is *removing* the complement:
  ``L(theta_ft - delta_notS) ~= L(theta_ft) - sum_notS delta_i . g_i``. Minimising that means
  removing the units with the largest ``delta_i . g_i``, i.e. keeping the most negative ones.

Same ranking rule either way, so :func:`ixg_scores` negates once and everything downstream is
unchanged: ``score_i = -(delta_i . g_i)``, larger means "reduces the loss more", and the existing
top-k machinery keeps the highest scores. What differs between the two points is only *where* the
linearisation is taken, and that is a real difference -- at the base model the delta is far from
where the gradient was measured (the Taylor term is being asked to extrapolate the whole finetune),
while at the finetuned model it is a local statement about ablating pieces of an update already
applied. Neither is obviously the right one, hence ``mask.ixg_at``.

Reducing with :func:`~..masks.unit_sums` and not ``unit_norms`` is the other thing that matters:
the per-element terms are additive and signed, and a norm would rank a unit whose weights push the
loss *up* as strongly as one that pushes it down.

The result is a plain ``[n_units]`` score vector in exactly the layout the learned scores use, so
it is written to the same checkpoint format and swept by the same runner -- an IxG run and a
learned-mask run differ in one config field and produce the same artifacts.
"""

import logging

import torch

from ..masks import apply_in_place, unit_sums

logger = logging.getLogger(__name__)

AT = ("base", "finetuned")


@torch.enable_grad()
def ixg_scores(model, *, base, deltas, layout, batches, loss_fn, at="finetuned",
               out_dtype=None) -> tuple:
    """``(scores, stats)`` -- per-unit first-order attribution of the delta.

    ``out_dtype`` is passed straight to :func:`apply_in_place`, so the weights the gradient is
    taken at are composed exactly as training composes them -- see MaskCfg.delta_dtype. It only
    matters for ``at="finetuned"``, where the delta is actually applied.

    ``batches`` is an iterable of collated batches already on the model's device; ``loss_fn(model,
    batch)`` must return the batch's **token-summed** cross entropy, so that accumulating over
    batches and dividing by the token count gives the mean-per-token gradient (the same
    normalisation the training loop uses -- normalising per batch instead would weight a short
    batch as heavily as a long one).
    """
    if at not in AT:
        raise ValueError(f"ixg_at must be one of {AT}, got {at!r}")

    # The gradient is taken at one of the two endpoints, and `apply_in_place` is what puts the
    # model there: a mask of ones composes theta_base + delta, a mask of zeros composes
    # theta_base. Reusing the composition kernel rather than assigning weights here is deliberate
    # -- it is the same code path the evals are served through, including its handling of tied
    # parameters, so "the gradient at the finetuned model" means the same weights the sweep calls
    # `full_delta`.
    #
    # SNAPSHOT FIRST, and independently. `base` as handed in by the training loop is
    # `{n: p.detach() ...}` -- VIEWS onto the live parameters -- so the moment apply_in_place writes
    # theta_base + delta through them, `base` itself holds theta_base + delta and any later restore
    # from it puts back the wrong weights. Symptom when this was wrong: the run's `pretrained`
    # anchor came out equal to another run's `full_delta`, i.e. every subsequent eval scored a model
    # that was silently one delta off. Same hazard as MaskedWeights._base_snapshot; see CLAUDE.md.
    dev = next(iter(deltas.values())).device
    snap = {n: base[n].detach().clone() for n in layout.names}
    keep = torch.ones(layout.total, device=dev) if at == "finetuned" else torch.zeros(
        layout.total, device=dev)
    apply_in_place(model, snap, deltas, keep, layout, invert=False, out_dtype=out_dtype)

    was_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    names = set(layout.names)
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
        p.grad = None

    n_tokens = n_batches = 0
    total_loss = 0.0
    for batch in batches:
        ce = loss_fn(model, batch)
        ce.backward()
        # labels are shifted by one inside the loss, so the supervised count must be too
        n_tokens += int((batch["labels"][:, 1:] != -100).sum())
        total_loss += float(ce.detach())
        n_batches += 1
    if not n_tokens:
        raise ValueError("no supervised tokens in the IxG batches, so the gradient is empty")

    scores = torch.zeros(layout.total)
    n_zero_grad = 0
    params = dict(model.named_parameters())
    with torch.no_grad():
        for i, (name, axis) in enumerate(zip(layout.names, layout.axes)):
            grad = params[name].grad
            if grad is None:
                n_zero_grad += 1
                continue
            g = grad.to(deltas[name].dtype) / n_tokens
            # NEGATED: larger score == reduces the loss more == what top-k should keep
            scores[layout.slice_for(i)] = -unit_sums(deltas[name] * g, axis).float().cpu()

    # Restore from the snapshot by assignment, not by composing a zero mask: composing would read
    # `base`, which the apply above has already overwritten. This puts theta_base back exactly, and
    # with it the storage that `base` is a view of.
    with torch.no_grad():
        for n in layout.names:
            params[n].copy_(snap[n])
    for n, p in model.named_parameters():          # leave the model as it was found
        p.requires_grad_(was_grad[n])
        p.grad = None
    del snap

    stats = {
        "ixg_at": at,
        "ixg_batches": n_batches,
        "ixg_tokens": n_tokens,
        "ixg_mean_loss": total_loss / n_tokens,
        "ixg_params_without_grad": n_zero_grad,
        "ixg_score_frac_positive": float((scores > 0).float().mean()),
        "ixg_score_absmax": float(scores.abs().max()),
    }
    logger.info("IxG at %s: %d batches / %d supervised tokens, mean loss %.4f; "
                "%.1f%% of units have a loss-reducing (positive) score",
                at, n_batches, n_tokens, stats["ixg_mean_loss"],
                100 * stats["ixg_score_frac_positive"])
    if n_zero_grad:
        logger.warning("%d masked parameter tensor(s) received no gradient at all, so every unit "
                       "in them scores exactly 0 and their rank is arbitrary", n_zero_grad)
    return scores, stats
