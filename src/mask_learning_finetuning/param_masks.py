"""Parameter-space masking: units, layouts, and the masked parameter composition.

MAttr as used upstream masks *activations* (a node's output is kept clean or patched to a
counterfactual). Here the same top-k machinery is applied to *parameters*, by writing the
finetune as a delta from the frozen pretrained weights:

    theta_eff = theta_base + m(s, k) . delta                    (iso / sufficient)
    theta_eff = theta_base + (1 - m(s, k)) . delta              (cause / necessary)

``iso`` keeps the top-k units at their FINETUNED value and reverts the complement to
pretrained, so a low SFT loss means those k units *suffice* to produce the finetuned
behaviour -- the parameter-space reading of denoising, matching every MIB run upstream (see
CLAUDE.md). ``cause`` reverts the top-k to pretrained and keeps the complement finetuned, so
the top-k are the units whose removal *breaks* the behaviour.

Two consequences of masking a delta that are worth knowing before reading a training curve:

1. ``delta`` is initialised to zero, so at step 0 the mask multiplies zero and
   ``dL/ds = (dL/dtheta_eff) . delta = 0``. Scores receive no signal until the delta grows;
   early steps train the delta almost exclusively. This is expected, not a bug.
2. A unit's delta gradient is scaled by its own mask value, so units that are usually
   masked out learn slowly. That coupling is the point -- it is what pushes the finetune to
   concentrate into units the scores rank highly -- but it does mean the delta you get from
   masked training is NOT the delta you would get from an unmasked finetune.

Units (``--unit``) are the granularity at which one score is learned:

  ``tensor``  one score per parameter tensor. Coarse (~150 units for gpt2); cheap; a good
              smoke-test default.
  ``row``     one score per index of dim 0. For ``nn.Linear`` weights ``[out, in]`` this is
              one score per output feature (neuron / head slice) -- the interp-meaningful
              granularity. 1-D params (biases, norms) get one score per element.
  ``col``     one score per index of dim -1. NOTE gpt2's ``Conv1D`` stores weights
              transposed as ``[in, out]``, so ``col`` -- not ``row`` -- is the output-unit
              granularity there. Check your architecture before picking.
  ``weight``  one score per scalar parameter. The literal per-parameter reading; costs
              another full model of scores plus its optimizer state.
"""

from dataclasses import dataclass

import torch

UNIT_MODES = ("tensor", "row", "col", "weight")


def _n_units(p: torch.Tensor, mode: str) -> int:
    if mode == "tensor":
        return 1
    if mode == "weight":
        return p.numel()
    if p.dim() == 0:
        return 1
    return p.shape[0] if mode == "row" else p.shape[-1]


@dataclass
class UnitLayout:
    """Maps a flat score/mask vector onto the parameter tensors it governs."""

    mode: str
    names: list
    shapes: list
    offsets: list          # start index of each param's units in the flat vector
    counts: list           # number of units for each param
    total: int             # total number of scored units

    def slice_for(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i] + self.counts[i])

    def summary(self) -> str:
        return (f"{self.total:,} units over {len(self.names)} tensors "
                f"(mode={self.mode})")


def build_layout(named_params, mode: str) -> UnitLayout:
    """Assign every parameter in ``named_params`` its slice of the flat score vector."""
    if mode not in UNIT_MODES:
        raise ValueError(f"unknown unit mode {mode!r}; known: {UNIT_MODES}")
    names, shapes, offsets, counts = [], [], [], []
    off = 0
    for name, p in named_params:
        n = _n_units(p, mode)
        names.append(name)
        shapes.append(tuple(p.shape))
        offsets.append(off)
        counts.append(n)
        off += n
    return UnitLayout(mode=mode, names=names, shapes=shapes, offsets=offsets,
                      counts=counts, total=off)


def expand_mask(mask_slice: torch.Tensor, shape: tuple, mode: str) -> torch.Tensor:
    """Reshape a param's slice of the flat mask so it broadcasts against that param.

    Kept as a view/reshape (never a materialised full-size tensor except for ``weight``),
    so ``tensor``/``row``/``col`` masking costs almost nothing beyond the multiply.
    """
    if mode == "tensor":
        return mask_slice.reshape([1] * max(1, len(shape)))
    if mode == "weight":
        return mask_slice.reshape(shape)
    if len(shape) <= 1:
        return mask_slice
    if mode == "row":
        return mask_slice.reshape([-1] + [1] * (len(shape) - 1))
    return mask_slice.reshape([1] * (len(shape) - 1) + [-1])


def build_alias_map(model) -> dict:
    """Map each deduplicated parameter name to every name that aliases the same tensor.

    Weight tying matters here: with ``tie_word_embeddings`` (Qwen2.5-0.5B, gpt2, ...),
    ``lm_head.weight`` *is* ``embed_tokens.weight``, but ``named_parameters()`` reports only
    the first. ``functional_call`` substitutes by name, so supplying only the canonical name
    would leave the output head reading the original, un-deltaed tensor -- the embedding
    delta would silently apply on the input side only. Every alias must be written.
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
                   aliases: dict = None) -> dict:
    """Build the ``{name: theta_eff}`` dict for a masked forward.

    Args:
        base: frozen pretrained parameters (no grad).
        deltas: trainable deltas, same keys as ``base``.
        mask: flat mask over units, ``[layout.total]``, differentiable.
        invert: use ``(1 - mask)`` -- the ``cause`` / necessary intervention.
        delta_scale: multiplies the whole delta; 0.0 recovers the pretrained model exactly
            (useful as an eval reference point).
        aliases: from :func:`build_alias_map`; tied parameters are written under every name
            they appear as, so a delta on tied embeddings reaches the output head too.

    The result stays attached to the graph, so gradients flow to both ``deltas`` and the
    scores behind ``mask``.
    """
    out = {}
    for i, name in enumerate(layout.names):
        b = base[name]
        d = deltas[name]
        m = expand_mask(mask[layout.slice_for(i)], layout.shapes[i], layout.mode)
        if invert:
            m = 1.0 - m
        upd = (m * d) if delta_scale == 1.0 else (m * d * delta_scale)
        # cast once, at the end: deltas are kept in fp32 for a stable optimizer step even
        # when the base model is bf16, and autograd carries the grad back through the cast.
        composed = b + upd.to(b.dtype)
        for alias in (aliases.get(name, [name]) if aliases else [name]):
            out[alias] = composed
    return out


def hard_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Non-differentiable hard top-k mask, for evaluation sweeps."""
    m = torch.zeros_like(scores)
    if k > 0:
        m[scores.topk(min(int(k), scores.numel())).indices] = 1.0
    return m
