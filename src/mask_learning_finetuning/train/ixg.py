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

from ..masks import AXIS_SVD, apply_in_place, unit_sums

logger = logging.getLogger(__name__)

AT = ("base", "finetuned", "mc", "ig")


@torch.enable_grad()
def ixg_scores(model, *, base, deltas, layout, batches, loss_fn, at="finetuned",
               steps=1, out_dtype=None, svd=None, count_fn=None) -> tuple:
    """``(scores, stats)`` -- per-unit first-order attribution of the delta.

    ``out_dtype`` is passed straight to :func:`apply_in_place`, so the weights the gradient is
    taken at are composed exactly as training composes them -- see MaskCfg.delta_dtype. It only
    matters for ``at="finetuned"``, where the delta is actually applied.

    ``batches`` is an iterable of collated batches already on the model's device; ``loss_fn(model,
    batch)`` must return the batch's **token-summed** cross entropy, so that accumulating over
    batches and dividing by the token count gives the mean-per-token gradient (the same
    normalisation the training loop uses -- normalising per batch instead would weight a short
    batch as heavily as a long one).

    Two generalisations, both for the REWARD objective (``train/rl.py:reward_ixg_scores``), where
    a "batch" is a set of prompts and the loss is a REINFORCE surrogate over samples generated at
    the current weights:

    * ``loss_fn`` may run its own backward and return the **detached** total. The surrogate is a
      sum over dozens of generated sequences, and one backward per sequence keeps the graph to one
      sequence at a time; ``ixg_scores`` only calls ``backward`` on a loss that still carries one.
    * ``count_fn(batch)`` returns how many normalisation units the batch contributes -- by default
      the number of supervised tokens in ``batch["labels"]``. The surrogate is already normalised
      per draw, so it counts 1 and the scores come out as the mean over draws.

    ``svd`` supplies the factors for a ``svd*`` layout's factored tensors. The formula is the same
    first-order term in the new basis -- unit ``i``'s slice of the delta is ``S[i] u_i v_i^T``, so
    its attribution is ``S[i] . u_i^T G v_i`` -- which makes IxG a baseline for a singular-direction
    mask on exactly the terms it is one for a nonresid mask. See ``SvdFactors.attribution``.
    """
    if at not in AT:
        raise ValueError(f"ixg_at must be one of {AT}, got {at!r}")
    if at == "ig" and steps < 2:
        raise ValueError("ixg_at='ig' is the fixed-grid path integral; ixg_steps must be >= 2 "
                         "(steps=1 is the alpha=1 endpoint, which `finetuned` already names)")
    svd = svd or {}
    # `mc` = STEPLESS IG. `base` and `finetuned` take the gradient at one endpoint of the path
    # theta(alpha) = theta_base + alpha.delta; this draws alpha ~ U(0,1) per batch and averages, an
    # unbiased estimator of the whole path integral int_0^1 g(theta(alpha)).delta dalpha at the same
    # cost -- one forward+backward per draw either way. `base` is the alpha=0 endpoint of that same
    # integral, so `--ixg_at mc` against `--ixg_at base` at equal `ixg_batches` is a compute-matched
    # contrast whose only difference is where alpha sits (upstream's framing for `mc_ig`).
    #
    # WHY WE WANT IT NOW: upstream's `logit` k-schedule makes a zero-init MAttr+SGD run's EXPECTED
    # score equal activation-path IG. The parameter-space analogue of that quantity is exactly this
    # integral, so `mc` is the baseline that turns their derivation into a prediction we can check
    # against the sgd1_logit cell rather than take on faith.
    #
    # ALPHA IS PER BATCH HERE, NOT PER EXAMPLE, and that is forced rather than chosen: upstream
    # draws alpha as `[B,1,1]` broadcasting over activations, but a WEIGHT is shared across the
    # batch, so a parameter-space path integral admits only one alpha per forward. The estimator
    # error therefore falls like 1/sqrt(n_batches), not 1/sqrt(n_examples) -- so `ixg_batches` is
    # the knob that controls its variance, and 64 draws is the floor rather than a comfortable
    # number. Raise it if two `mc` runs at different seeds disagree.

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
    # that was silently one delta off. Same hazard as MaskedWeights._base_snapshot; see the project notes.
    dev = next(iter(deltas.values())).device if deltas else next(iter(svd.values())).S.device
    # At the BASE endpoint the model already holds theta_base: the zero-mask composition is a
    # numerical no-op and the snapshot exists only to undo compositions that never happen. Both
    # are skipped -- the snapshot alone is a whole model's worth of device memory (14 GB bf16 at
    # 8B), which is what made a per-weight IxG run OOM.
    snap = None if at == "base" else {n: base[n].detach().clone() for n in layout.names}
    keep = (torch.ones(layout.total, device=dev) if at in ("finetuned", "mc", "ig")
            else torch.zeros(layout.total, device=dev))
    if at not in ("mc", "base"):
        apply_in_place(model, snap, deltas, keep, layout, invert=False, out_dtype=out_dtype,
                       svd=svd)

    was_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    names = set(layout.names)
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
        p.grad = None

    if count_fn is None:
        # labels are shifted by one inside the loss, so the supervised count must be too
        def count_fn(b):
            return int((b["labels"][:, 1:] != -100).sum())

    def backward(ce):
        # a loss_fn that accumulated its own gradients hands back a detached total (see docstring)
        if ce.requires_grad:
            ce.backward()
        return float(ce.detach())

    n_tokens = n_batches = 0
    total_loss = 0.0
    alphas = []
    for batch in batches:
        if at == "ig":
            # TEXTBOOK IG: the right-Riemann grid alpha = k/m, k=1..m (Sundararajan et al.),
            # every batch evaluated at every grid point. Each (batch, alpha) pair is one
            # forward+backward AND one full-model recompose, so at matched compute this sees
            # 1/m of the data `mc` sees -- that trade (path resolution per example vs examples)
            # is exactly what an `ig` cell against an `mc` cell measures. Tokens are counted
            # per PASS, so the final /n_tokens is the mean over (example, alpha) draws -- the
            # same Riemann average the sum of backwards accumulates.
            toks = count_fn(batch)
            for k in range(1, steps + 1):
                alpha = k / steps
                alphas.append(alpha)
                apply_in_place(model, snap, deltas, keep, layout, invert=False,
                               delta_scale=alpha, out_dtype=out_dtype, svd=svd)
                total_loss += backward(loss_fn(model, batch))
                n_tokens += toks
            n_batches += 1
            continue
        if at == "mc":
            # re-compose at a fresh point on the path before every backward. `delta_scale` is the
            # alpha: apply_in_place writes theta_base + alpha.(mask . delta), and the mask is all
            # ones here, so this is theta(alpha) exactly. Composing per batch is the cost of the
            # method -- a full-model write per draw -- and is why `mc` is slower than the endpoint
            # variants at equal `ixg_batches` despite the same number of forwards.
            alpha = float(torch.rand(1).item())
            alphas.append(alpha)
            apply_in_place(model, snap, deltas, keep, layout, invert=False, delta_scale=alpha,
                           out_dtype=out_dtype, svd=svd)
        total_loss += backward(loss_fn(model, batch))
        n_tokens += count_fn(batch)
        n_batches += 1
    if not n_tokens:
        raise ValueError("no supervised tokens (or draws) in the IxG batches, so the gradient "
                         "is empty")

    scores = torch.zeros(layout.total)
    n_zero_grad = 0
    params = dict(model.named_parameters())
    with torch.no_grad():
        for i, (name, axis) in enumerate(zip(layout.names, layout.axes)):
            grad = params[name].grad
            if grad is None:
                n_zero_grad += 1
                continue
            if axis == AXIS_SVD:
                # NEGATED for the same reason as below, and per DIRECTION rather than per slice
                scores[layout.slice_for(i)] = -svd[name].attribution(
                    grad / n_tokens).float().cpu()
                continue
            g = grad.to(deltas[name].dtype) / n_tokens
            # NEGATED: larger score == reduces the loss more == what top-k should keep.
            # ACCUMULATED (-=) rather than assigned: under neuron_head the gate/up/down of one
            # MLP share a slice, and the first-order term is additive over a unit's elements,
            # so the tied tensors' contributions sum. Identical for untied layouts (one write
            # per slice, onto zeros).
            scores[layout.slice_for(i)] -= unit_sums(deltas[name] * g, axis).float().cpu()

    # Restore from the snapshot by assignment, not by composing a zero mask: composing would read
    # `base`, which the apply above has already overwritten. This puts theta_base back exactly, and
    # with it the storage that `base` is a view of.
    if snap is not None:
        with torch.no_grad():
            for n in layout.names:
                params[n].copy_(snap[n])
    for n, p in model.named_parameters():          # leave the model as it was found
        p.requires_grad_(was_grad[n])
        p.grad = None
    del snap

    stats = {
        "ixg_at": at,
        **({"mc_draws": len(alphas), "mc_alpha_mean": (sum(alphas) / len(alphas)) if alphas else None}
           if at == "mc" else {}),
        # the alpha of every draw, in order -- what lets a caller align a per-draw quantity (the
        # reward path's mean reward per draw) with where on the path it was measured
        **({"alphas": alphas} if at in ("mc", "ig") else {}),
        "ixg_batches": n_batches,
        **({"ixg_steps": steps} if at == "ig" else {}),
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
