"""Parameter-space masking: units, layouts, and the masked parameter composition.

MAttr as used upstream masks *activations* (a node's output is kept clean or patched to a
counterfactual). Here the same top-k machinery is applied to *parameters*, by writing the
finetune as a delta from the frozen pretrained weights:

    theta_eff = theta_base + (1 - m(s, k)) . delta               (iso / sufficient)
    theta_eff = theta_base + m(s, k) . delta                     (cause / necessary)

Which side gets which state follows CLAUDE.md's rule -- ``sufficient``/``iso`` = **top-k
stays CLEAN, complement corrupted**. In parameter space the *pretrained* model is the clean
/ unperturbed one and the finetune delta is the perturbation, so:

  ``iso``    top-k retained at pretrained, delta on the complement.
  ``cause``  top-k carries the delta, complement stays pretrained.

**Train with ``cause``**: minimising the SFT loss then ranks units by how much they carry the
finetuned behaviour. Don't re-flip the mapping.

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
  ``nonresid`` one score per index of whichever axis is NOT the residual stream, chosen per
              tensor. This is ``row`` for ``[n, d_model]`` weights (q/k/v/gate/up, and
              ``embed_tokens`` at ``[vocab, d_model]``) but dim -1 for ``[d_model, n]`` ones
              (``down_proj``), so an FFN unit is a *neuron* rather than an output coordinate
              of the MLP. gpt2's transposed ``Conv1D`` needs no special case -- the shape
              rule reads the transposition directly. Square weights are the one ambiguity
              (in Llama ``heads*head_dim == d_model``, so ``q_proj`` and ``o_proj`` are both
              ``[2048, 2048]`` and want opposite axes); those are resolved by name against
              ``_OUT_PROJ``. 1-D params (norm gains) are indexed *only* by the residual dim,
              so they get one unit for the whole tensor rather than d_model residual-indexed
              scores.

              Caveat this does NOT fix: a unit is still per tensor, so FFN neuron i has
              three independent scores (``gate[i,:]``, ``up[i,:]``, ``down[:,i]``) and the
              mask may keep some and drop others. Tying those into one score per neuron is a
              further step.
"""

from dataclasses import dataclass

import torch

UNIT_MODES = ("tensor", "row", "col", "weight", "nonresid")

# Name fragments marking an OUT-projection: dim 0 is the residual stream, so its
# non-residual axis is dim 1. Only consulted when the tensor is SQUARE, where the shape rule
# cannot tell an in-projection from an out-projection (in Llama, heads*head_dim == d_model,
# so q_proj and o_proj are both [2048, 2048] but want opposite axes).
_OUT_PROJ = ("o_proj", "out_proj", "down_proj", "dense_4h_to_h", "fc2", "wo", "w2")

# axis codes stored per parameter in UnitLayout.axes
AXIS_TENSOR = None     # one unit for the whole tensor
AXIS_ALL = "all"       # one unit per scalar


def _axis_for(name: str, shape: tuple, mode: str, resid_dim=None):
    """Which axis of this parameter indexes its units (0, -1, AXIS_TENSOR or AXIS_ALL)."""
    if mode == "tensor":
        return AXIS_TENSOR
    if mode == "weight":
        return AXIS_ALL
    if len(shape) == 0:
        return AXIS_TENSOR
    if mode == "row":
        return 0
    if mode == "col":
        return -1

    # nonresid: index units along the axis that is NOT the residual stream.
    if resid_dim is None:
        raise ValueError("unit mode 'nonresid' needs resid_dim (the model's hidden size)")
    if len(shape) == 1:
        # A norm gain / bias is indexed *entirely* by the residual dim, so it has no
        # non-residual axis to put units on. One unit for the whole tensor keeps the mode's
        # promise -- no score is ever a residual-stream coordinate.
        return AXIS_TENSOR
    d0, d1 = shape[0], shape[-1]
    if d0 == resid_dim and d1 == resid_dim:
        return -1 if any(t in name for t in _OUT_PROJ) else 0
    if d1 == resid_dim:
        return 0        # [n, d_model]: q/k/v/gate/up, and embed_tokens ([vocab, d_model])
    if d0 == resid_dim:
        return -1       # [d_model, n]: down_proj -- and gpt2's transposed Conv1D falls out
    return 0            # neither axis is the residual stream; dim 0 is as good a guess as any


def _n_units(p: torch.Tensor, mode: str, name: str = "", resid_dim=None) -> int:
    axis = _axis_for(name, tuple(p.shape), mode, resid_dim)
    if axis is AXIS_TENSOR:
        return 1
    if axis == AXIS_ALL:
        return p.numel()
    return p.shape[axis]


@dataclass
class UnitLayout:
    """Maps a flat score/mask vector onto the parameter tensors it governs."""

    mode: str
    names: list
    shapes: list
    offsets: list          # start index of each param's units in the flat vector
    counts: list           # number of units for each param
    total: int             # total number of scored units
    # Which axis indexes each parameter's units. Defaulted rather than required so that
    # checkpoints written before this field existed still load (sweep.py does
    # UnitLayout(**blob["layout"])); for the uniform modes it is recoverable from `mode`.
    axes: list = None

    def __post_init__(self):
        if self.axes is None:
            self.axes = [_axis_for("", tuple(s), self.mode) for s in self.shapes]

    def slice_for(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i] + self.counts[i])

    def summary(self) -> str:
        extra = ""
        if self.mode == "nonresid":
            n1 = sum(1 for a in self.axes if a == -1)
            extra = f", {n1} tensor(s) scored along dim -1"
        return (f"{self.total:,} units over {len(self.names)} tensors "
                f"(mode={self.mode}{extra})")


def build_layout(named_params, mode: str, resid_dim=None) -> UnitLayout:
    """Assign every parameter in ``named_params`` its slice of the flat score vector.

    ``resid_dim`` (the model's hidden size) is required for ``mode="nonresid"``.
    """
    if mode not in UNIT_MODES:
        raise ValueError(f"unknown unit mode {mode!r}; known: {UNIT_MODES}")
    names, shapes, offsets, counts, axes = [], [], [], [], []
    off = 0
    for name, p in named_params:
        shape = tuple(p.shape)
        axis = _axis_for(name, shape, mode, resid_dim)
        n = _n_units(p, mode, name, resid_dim)
        names.append(name)
        shapes.append(shape)
        offsets.append(off)
        counts.append(n)
        axes.append(axis)
        off += n
    return UnitLayout(mode=mode, names=names, shapes=shapes, offsets=offsets,
                      counts=counts, total=off, axes=axes)


def expand_mask(mask_slice: torch.Tensor, shape: tuple, axis) -> torch.Tensor:
    """Reshape a param's slice of the flat mask so it broadcasts against that param.

    ``axis`` is the parameter's entry in :attr:`UnitLayout.axes`. Kept as a view/reshape
    (never a materialised full-size tensor except for ``AXIS_ALL``), so masking costs almost
    nothing beyond the multiply.
    """
    if axis is AXIS_TENSOR:
        return mask_slice.reshape([1] * max(1, len(shape)))
    if axis == AXIS_ALL:
        return mask_slice.reshape(shape)
    if len(shape) <= 1:
        return mask_slice
    if axis == 0:
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
        invert: use ``(1 - mask)`` -- i.e. the ``iso`` / sufficient intervention, where the
            top-k are retained at pretrained and the delta lands on the complement.
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
        m = expand_mask(mask[layout.slice_for(i)], layout.shapes[i], layout.axes[i])
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
