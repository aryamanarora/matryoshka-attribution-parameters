"""Composing masked weights: ``theta_eff`` from a base, a delta and a mask.

MAttr as used upstream masks *activations* (a node's output is kept clean or patched to a
counterfactual). Here the same top-k machinery is applied to *parameters*, by writing the
finetune as a delta from the frozen pretrained weights:

    theta_eff = theta_base + (1 - m(s, k)) . delta               (iso / sufficient)
    theta_eff = theta_base + m(s, k) . delta                     (cause / necessary)

Which side gets which state follows CLAUDE.md's rule -- ``sufficient``/``iso`` = **top-k stays
CLEAN, complement corrupted**. In parameter space the *pretrained* model is the clean /
unperturbed one and the finetune delta is the perturbation, so:

  ``iso``    top-k retained at pretrained, delta on the complement.
  ``cause``  top-k carries the delta, complement stays pretrained.

**Train with ``cause``**: minimising the SFT loss then ranks units by how much they carry the
finetuned behaviour. Don't re-flip the mapping.

Two consequences of masking a delta that are worth knowing before reading a training curve:

1. ``delta`` is initialised to zero, so at step 0 the mask multiplies zero and
   ``dL/ds = (dL/dtheta_eff) . delta = 0``. Scores receive no signal until the delta grows;
   early steps train the delta almost exclusively. This is expected, not a bug.
2. A unit's delta gradient is scaled by its own mask value, so units that are usually masked
   out learn slowly. That coupling is the point -- it is what pushes the finetune to
   concentrate into units the scores rank highly -- but it does mean the delta you get from
   masked training is NOT the delta you would get from an unmasked finetune.

**Two composition paths, one arithmetic.** A masked forward can be had either functionally
(:func:`compose_params` + ``torch.func.functional_call``, which keeps the result attached to
the autograd graph -- what training and the loss/MMLU sweeps use) or by writing the weights
into the model in place (:func:`apply_in_place`, which is what ``model.generate`` needs, since
it wants real parameters). These used to be two hand-maintained copies of the same expression
with a comment asking future readers to keep them bit-identical. They now both call
:func:`composed_tensor`, so a given ``(mask, delta)`` provably yields the same weights during
training and during any eval.
"""

import torch

from .layout import UnitLayout, expand_mask

#: the dtype names a config may use, resolved once so a typo fails at load rather than mid-run
DTYPES = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)


def resolve_dtype(name):
    """``'bfloat16'`` -> ``torch.bfloat16``. Passes a ``torch.dtype`` through unchanged."""
    if name is None or isinstance(name, torch.dtype):
        return name
    if name not in DTYPES:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(DTYPES)}")
    return DTYPES[name]


def composed_tensor(base_t: torch.Tensor, delta_t: torch.Tensor, m: torch.Tensor,
                    *, delta_scale: float = 1.0, out_dtype=None) -> torch.Tensor:
    """``base + (m . delta . scale)``, the one masked-weight expression in this repo.

    The multiply happens in the DELTA's dtype and the cast happens once, at the end, so
    whatever precision the delta is held in is carried through the multiply rather than being
    thrown away on the operands. Autograd carries the gradient back through the cast.

    ``out_dtype`` is the dtype the result is produced in; ``None`` means the base's, which is
    what every caller did before the dtype became configurable and is therefore bit-identical
    for a run that does not set ``mask.delta_dtype``. It exists because at 8B the composed
    tensors are a whole model's worth of memory, so which precision they land in is a capacity
    question and not only an accuracy one -- see MaskCfg.delta_dtype.
    """
    dt = resolve_dtype(out_dtype) or base_t.dtype
    upd = (m * delta_t) if delta_scale == 1.0 else (m * delta_t * delta_scale)
    return base_t.to(dt) + upd.to(dt)


def build_alias_map(model) -> dict:
    """Map each deduplicated parameter name to every name that aliases the same tensor.

    Weight tying matters here: with ``tie_word_embeddings`` (Qwen2.5-0.5B, gpt2, ...),
    ``lm_head.weight`` *is* ``embed_tokens.weight``, but ``named_parameters()`` reports only
    the first. ``functional_call`` substitutes by name, so supplying only the canonical name
    would leave the output head reading the original, un-deltaed tensor -- the embedding
    delta would silently apply on the input side only. Every alias must be written.

    Not needed by :func:`apply_in_place`, which writes the shared tensor itself.
    """
    seen, aliases = {}, {}
    for name, p in model.named_parameters(remove_duplicate=False):
        key = id(p)
        if key in seen:
            aliases[seen[key]].append(name)
        else:
            seen[key] = name
            aliases[name] = [name]
    return aliases


def compose_params(base: dict, deltas: dict, mask: torch.Tensor, layout: UnitLayout,
                   *, invert: bool = False, delta_scale: float = 1.0,
                   aliases: dict = None, out_dtype=None) -> dict:
    """Build the ``{name: theta_eff}`` dict for a masked forward via ``functional_call``.

    Args:
        base: frozen pretrained parameters (no grad).
        deltas: trainable deltas, same keys as ``base``.
        mask: flat mask over units, ``[layout.total]``, differentiable.
        invert: use ``(1 - mask)`` -- i.e. the ``iso`` / sufficient intervention, where the
            top-k are retained at pretrained and the delta lands on the complement.
        delta_scale: multiplies the whole delta; 0.0 recovers the pretrained model exactly
            (useful as an eval reference point).
        out_dtype: dtype for theta_eff; None keeps the base's, as before.
        aliases: from :func:`build_alias_map`; tied parameters are written under every name
            they appear as, so a delta on tied embeddings reaches the output head too.

    The result stays attached to the graph, so gradients flow to both ``deltas`` and the
    scores behind ``mask``.
    """
    out = {}
    for i, name in enumerate(layout.names):
        b = base[name]
        m = expand_mask(mask[layout.slice_for(i)], layout.shapes[i], layout.axes[i])
        if invert:
            m = 1.0 - m
        composed = composed_tensor(b, deltas[name], m, delta_scale=delta_scale,
                                   out_dtype=out_dtype)
        for alias in (aliases.get(name, [name]) if aliases else [name]):
            out[alias] = composed
    return out


@torch.no_grad()
def apply_in_place(model, base: dict, deltas: dict, mask: torch.Tensor, layout: UnitLayout,
                   *, invert: bool = False, delta_scale: float = 1.0, out_dtype=None) -> None:
    """Write ``theta_eff`` straight into a live model's parameters.

    For evals that need real parameters -- anything calling ``model.generate``. ``base`` and
    ``deltas`` may live on the CPU while the model is on the GPU; composition then happens on
    the CPU and peak GPU cost is one parameter tensor rather than a second copy of the model.

    Tied weights need no alias map: ``lm_head.weight`` *is* ``embed_tokens.weight``, so
    writing the canonical tensor updates both.

    ``out_dtype`` must match what the functional path uses, or the two stop agreeing: this
    writes into real parameters whose dtype is fixed, so composing in a WIDER dtype here than
    :func:`compose_params` used would round on the way in and generation would see different
    weights from training. Callers pass the same value to both.
    """
    params = dict(model.named_parameters())
    for i, name in enumerate(layout.names):
        b = base[name]
        m = expand_mask(mask[layout.slice_for(i)], layout.shapes[i], layout.axes[i])
        if invert:
            m = 1.0 - m
        composed = composed_tensor(b, deltas[name], m, delta_scale=delta_scale,
                                   out_dtype=out_dtype)
        params[name].data.copy_(composed.to(params[name].device))


def hard_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Non-differentiable hard top-k mask, for evaluation sweeps."""
    m = torch.zeros_like(scores)
    if k > 0:
        m[scores.topk(min(int(k), scores.numel())).indices] = 1.0
    return m


def mask_for(k: int, layout: UnitLayout, scores: torch.Tensor) -> torch.Tensor:
    """Hard top-k mask over the learned scores, with the two extremes short-circuited."""
    if k <= 0:
        return torch.zeros(layout.total)
    if k >= layout.total:
        return torch.ones(layout.total)
    return hard_topk_mask(scores, k)
