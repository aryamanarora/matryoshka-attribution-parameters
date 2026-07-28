"""Unit layouts: which parameters one score governs, and how a flat mask maps onto them.

A *unit* is the granularity at which one score is learned. The layout is the bookkeeping that
turns a flat ``[total]`` score/mask vector into per-parameter slices, and knows which axis of
each parameter its slice indexes.

Note this module owns unit *granularity* (``row`` / ``nonresid`` / ...), which is a property of
this repo. It does not own mask *variants* (``topk`` / ``gumbel`` / ``hard_concrete`` / ...) --
those are the differentiable estimators, they live in ``learning_to_attribute.masks``, and per
that repo's CLAUDE.md they are numerics-frozen and must not be reimplemented here. Both get
called "mask type" in conversation; they are different axes of the design.

The modes (``--unit``):

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

``nonresid`` is the one mode whose axes cannot be re-derived from ``mode`` alone -- it needs
the parameter *names* and the model's ``resid_dim``. That is why :attr:`UnitLayout.axes` is
serialised explicitly by ``masks.checkpoint`` rather than recomputed on load.
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


def axis_for(name: str, shape: tuple, mode: str, resid_dim=None):
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


# kept under the old private name so nothing that imported it breaks
_axis_for = axis_for


def n_units_for(shape: tuple, mode: str, name: str = "", resid_dim=None) -> int:
    """How many units a parameter of this shape contributes under ``mode``."""
    axis = axis_for(name, tuple(shape), mode, resid_dim)
    if axis is AXIS_TENSOR:
        return 1
    if axis == AXIS_ALL:
        n = 1
        for d in shape:
            n *= d
        return n
    return shape[axis]


@dataclass
class UnitLayout:
    """Maps a flat score/mask vector onto the parameter tensors it governs."""

    mode: str
    names: list
    shapes: list
    offsets: list          # start index of each param's units in the flat vector
    counts: list           # number of units for each param
    total: int             # total number of scored units
    axes: list = None      # which axis indexes each parameter's units

    def __post_init__(self):
        if self.axes is None:
            # Best-effort fallback for a layout built by hand. It cannot recover `nonresid`
            # (that needs names + resid_dim), and deliberately does not try: loading a
            # serialised layout goes through masks.checkpoint.layout_from_dict, which has
            # both and raises a comprehensible error when it doesn't.
            if self.mode == "nonresid":
                raise ValueError(
                    "UnitLayout(mode='nonresid') needs explicit `axes`; they cannot be "
                    "re-derived from `mode` alone. Use masks.checkpoint.layout_from_dict "
                    "to load a saved layout, or build_layout() to make a fresh one.")
            self.axes = [axis_for("", tuple(s), self.mode) for s in self.shapes]

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
        axis = axis_for(name, shape, mode, resid_dim)
        n = n_units_for(shape, mode, name, resid_dim)
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


def unit_view(t: torch.Tensor, axis) -> torch.Tensor:
    """``t`` reshaped to ``[n_units, k]``, aligned with the layout's slice.

    The one place the per-unit reduction axis is resolved, and the reason it is one place: under
    ``nonresid`` two tensors in the same run reduce along *different* axes, which is exactly what
    the earlier ``layout.mode``-based version got wrong (it mis-reduced every ``down_proj``). Any
    new per-unit quantity should reduce a view from here rather than re-derive the dispatch --
    see the hazard note in CLAUDE.md.
    """
    if axis is AXIS_TENSOR:
        return t.reshape(1, -1)         # also correct for a 0-dim tensor
    if axis == AXIS_ALL or t.ndim <= 1:
        return t.reshape(-1, 1)         # one unit per element
    keep = axis % t.ndim
    other = [d for d in range(t.ndim) if d != keep]
    return t.permute([keep] + other).reshape(t.shape[keep], -1)


def unit_norms(t: torch.Tensor, axis) -> torch.Tensor:
    """L2 norm of ``t`` per unit -- a ``[n_units]`` vector aligned with the layout's slice.

    The reduction counterpart to :func:`expand_mask`, and the right way to summarise the
    *magnitude* of a delta (or a gradient) per unit.
    """
    return unit_view(t, axis).norm(dim=1)


def unit_sums(t: torch.Tensor, axis) -> torch.Tensor:
    """**Signed** sum of ``t`` per unit -- a ``[n_units]`` vector aligned with the layout's slice.

    For quantities that are additive across a unit's elements rather than magnitudes, which is
    what a first-order attribution is: ``delta * dL/dtheta`` summed over a unit's weights is that
    unit's predicted contribution to the loss change (see ``train/ixg.py``). Taking
    :func:`unit_norms` there instead would discard the sign that says whether the unit *raises* or
    *lowers* the loss, which is the entire content of the ranking.
    """
    return unit_view(t, axis).sum(dim=1)
