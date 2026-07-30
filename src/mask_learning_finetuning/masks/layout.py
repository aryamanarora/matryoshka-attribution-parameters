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
              mask may keep some and drop others. ``neuron_head`` is that further step.

  ``neuron_head`` the interp-native decomposition, finer in meaning and coarser in count than
              ``nonresid``. Two rules, one per sublayer:

              * **an MLP unit is a whole neuron**: the gate/up/down projections of one MLP
                share a single slice of the score vector, so score i governs ``gate[i,:]``,
                ``up[i,:]`` *and* ``down[:,i]`` together -- keep the neuron or drop it, never
                a third of it. The tie is by parent module (``model.layers.L.mlp``), so the
                three tensors have *overlapping* entries in :attr:`UnitLayout.offsets` and
                ``sum(counts) > total``. Anything that scatters per-tensor quantities into a
                flat vector must therefore ACCUMULATE over the slice, not assign
                (``train/ixg.py`` sums, ``posthoc.unit_delta_norms`` root-sum-squares).
              * **an attention unit is one head's slice of one projection matrix**
                (``q_proj`` head 3 = one unit): the non-residual axis is partitioned into
                contiguous groups of ``head_dim``, recorded as a ``("group", axis, head_dim)``
                axis code. Under GQA, k/v have ``n_kv_heads`` units where q/o have
                ``n_heads`` -- the counts come from the shapes, so that falls out.

              Everything else (embeddings, lm_head, norms, biases) takes the ``nonresid``
              rule. Needs ``resid_dim`` AND ``head_dim``; both are read off the model config
              by ``train/params.py``. Like ``nonresid``, its axes cannot be re-derived from
              ``mode`` alone and are serialised explicitly.

  ``svd``     one score per **singular direction of the weight delta**, per tensor. The unit is
              no longer a slice of a parameter: the delta is factorised
              ``delta = U diag(S) Vh`` and the mask scales the singular values, so
              ``theta_eff = theta_base + U diag(m . S) Vh``. Units are ordered by singular
              value, so unit 0 of a tensor is its dominant direction. See ``masks.svd``.
  ``svd_attn`` ``svd`` units on the attention projections, ``nonresid`` units on everything
              else.
  ``svd_mlp``  ``svd`` units on the MLP projections, ``nonresid`` units on everything else.

Two properties of the ``svd*`` modes that the others do not have, and both are load-bearing:

* **The unit count is a property of the DELTA, not of the shapes.** A tensor contributes as many
  units as the rank kept for it, which is decided when the delta is factorised (a cap from
  ``mask.svd_rank``, and a tolerance that drops numerically-zero singular values). So a layout
  cannot be built before the delta exists -- ``build_layout`` takes an explicit ``ranks`` map --
  and the *same* model under ``svd`` has different unit totals for different finetunes. Compare
  such a curve on fraction, and state the denominator.
* **They require a FROZEN, given delta** (``mask.finetuned`` / ``mask.init_delta``). A co-trained
  delta changes every step, so its factorisation would too, and the units would not be the same
  objects from one step to the next. ``config/schema.py`` rejects that combination.

``nonresid`` and the ``svd*`` modes are the ones whose axes cannot be re-derived from ``mode``
alone -- they need the parameter *names* and (for ``nonresid``) the model's ``resid_dim``, and
for ``svd*`` the per-tensor ranks as well. That is why :attr:`UnitLayout.axes` and
:attr:`UnitLayout.counts` are serialised explicitly by ``masks.checkpoint`` rather than
recomputed on load.
"""

from dataclasses import dataclass

import torch

#: The ``svd*`` family: a unit is a singular direction of the delta rather than a slice of a
#: parameter. Grouped because every caller that has to special-case one has to special-case all.
SVD_MODES = ("svd", "svd_attn", "svd_mlp")

UNIT_MODES = ("tensor", "row", "col", "weight", "nonresid", "neuron_head") + SVD_MODES

# Name fragments marking an OUT-projection: dim 0 is the residual stream, so its
# non-residual axis is dim 1. Only consulted when the tensor is SQUARE, where the shape rule
# cannot tell an in-projection from an out-projection (in Llama, heads*head_dim == d_model,
# so q_proj and o_proj are both [2048, 2048] but want opposite axes).
_OUT_PROJ = ("o_proj", "out_proj", "down_proj", "dense_4h_to_h", "fc2", "wo", "w2")

# Name fragments identifying which sublayer a parameter belongs to, for the hybrid `svd_attn` /
# `svd_mlp` modes. MLP is tested FIRST and attention is "matches an attention fragment and not an
# MLP one", because gpt2 names both output projections `c_proj` (`attn.c_proj`, `mlp.c_proj`) and
# only the enclosing module disambiguates them. Broad on purpose: a fragment that matches nothing
# costs nothing, while a sublayer that matches nothing is a silent empty half of a hybrid mode --
# so `build_layout` reports the split and `masks.svd` refuses a mode that factors zero tensors.
_MLP_PARTS = ("mlp.", "feed_forward.", "ffn.", "gate_proj", "up_proj", "down_proj",
              "fc1", "fc2", "c_fc", "dense_h_to_4h", "dense_4h_to_h", "w1", "w2", "w3")
_ATTN_PARTS = ("self_attn.", "attn.", "attention.", "q_proj", "k_proj", "v_proj", "o_proj",
               "out_proj", "qkv", "c_attn", "wq", "wk", "wv", "wo")

# axis codes stored per parameter in UnitLayout.axes
AXIS_TENSOR = None     # one unit for the whole tensor
AXIS_ALL = "all"       # one unit per scalar
AXIS_SVD = "svd"       # one unit per singular direction of this parameter's DELTA
#: Grouped axis code: ``(AXIS_GROUP, dim, size)`` -- units are contiguous groups of ``size``
#: indices along ``dim``. What `neuron_head` uses to make one attention head's slice of one
#: projection a single unit. A tuple rather than a class so it pickles into old checkpoints'
#: blob format unchanged; :func:`group_of` is the one reader.
AXIS_GROUP = "group"


def group_of(axis):
    """``(dim, size)`` if this axis code is a grouped axis, else ``None``.

    Accepts a list as well as a tuple, because a layout that has been through a JSON
    round-trip comes back with its tuples turned into lists.
    """
    if isinstance(axis, (tuple, list)) and len(axis) == 3 and axis[0] == AXIS_GROUP:
        return int(axis[1]), int(axis[2])
    return None


def is_mlp_param(name: str) -> bool:
    """True if this parameter belongs to a feed-forward sublayer."""
    return any(f in name for f in _MLP_PARTS)


def is_attn_param(name: str) -> bool:
    """True if this parameter belongs to an attention sublayer.

    Tested as "attention-like and not MLP-like" so gpt2's two ``c_proj`` tensors land in the
    right halves; see :data:`_MLP_PARTS`.
    """
    return not is_mlp_param(name) and any(f in name for f in _ATTN_PARTS)


def wants_svd(name: str, shape: tuple, mode: str) -> bool:
    """Whether this parameter's units are singular directions under ``mode``.

    Only 2-D parameters can be: a norm gain or a bias has no singular value decomposition, so
    under every ``svd*`` mode those fall back to the ``nonresid`` rule (which gives a 1-D tensor
    one unit for the whole thing). That is the same treatment they get in the hybrid modes' other
    half, so no ``svd*`` mode ever silently drops a tensor from the layout.
    """
    if mode not in SVD_MODES or len(shape) != 2:
        return False
    if mode == "svd":
        return True
    return is_attn_param(name) if mode == "svd_attn" else is_mlp_param(name)


def _nonresid_axis(name: str, shape: tuple, resid_dim, mode: str = "nonresid"):
    """The nonresid rule: index units along the axis that is NOT the residual stream."""
    if resid_dim is None:
        raise ValueError(f"unit mode {mode!r} needs resid_dim (the model's hidden size)")
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


def axis_for(name: str, shape: tuple, mode: str, resid_dim=None, head_dim=None):
    """Which axis of this parameter indexes its units.

    Returns 0, -1, AXIS_TENSOR, AXIS_ALL, AXIS_SVD, or a grouped ``(AXIS_GROUP, dim, size)``
    code (see :func:`group_of`). ``head_dim`` is required, and only consulted, for the
    attention projections under ``neuron_head``.
    """
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
    if mode in SVD_MODES:
        # The factored tensors have no parameter axis at all; the rest take the nonresid rule, so
        # a hybrid mode's two halves are exactly `svd` and `nonresid` over a partition of the
        # tensors and nothing is scored twice or not at all.
        if wants_svd(name, shape, mode):
            return AXIS_SVD
        mode = "nonresid"
    if mode == "neuron_head" and len(shape) == 2 and is_attn_param(name):
        # One unit per (projection, head): the non-residual axis is partitioned into contiguous
        # groups of head_dim. Fused projections need no special case -- gpt2's c_attn is
        # [d, 3*d] along dim -1, and 3*n_heads groups of head_dim are exactly one head's slice
        # of q, of k and of v each.
        axis = _nonresid_axis(name, shape, resid_dim, mode)
        if head_dim is None:
            raise ValueError(
                "unit mode 'neuron_head' needs head_dim to partition the attention projections "
                "by head (the model config's head_dim, or hidden_size // num_attention_heads)")
        n = shape[axis]
        if n % head_dim:
            raise ValueError(
                f"{name}: {n} units along axis {axis} is not divisible by head_dim={head_dim}, "
                "so it cannot be partitioned into heads. Wrong head_dim for this architecture?")
        return (AXIS_GROUP, axis, int(head_dim))
    # nonresid, and neuron_head's everything-else half (MLP -- tied in build_layout, not
    # here -- plus embeddings, norms and biases)
    return _nonresid_axis(name, shape, resid_dim, mode)


# kept under the old private name so nothing that imported it breaks
_axis_for = axis_for


def tie_key(name: str, shape: tuple, axis, mode: str):
    """The slice-sharing key under ``neuron_head``, or ``None`` when this parameter ties nothing.

    The gate/up/down projections of one MLP share their unit slice, so an FFN neuron is ONE
    unit spanning its in- and out-vectors. The key is the parent module path
    (``model.layers.L.mlp``); ``build_layout`` additionally requires equal unit counts before
    sharing, so an architecture whose in- and out-projections disagree about the hidden width
    falls back to per-tensor units rather than mis-tying.
    """
    if mode != "neuron_head" or len(shape) != 2 or axis not in (0, -1) \
            or not is_mlp_param(name):
        return None
    return name.rsplit(".", 2)[0]


def n_units_for(shape: tuple, mode: str, name: str = "", resid_dim=None, rank=None,
                head_dim=None) -> int:
    """How many units a parameter of this shape contributes under ``mode``.

    ``rank`` is required, and only consulted, when the parameter's units are singular directions:
    that count comes from the *delta's* factorisation and is not a function of the shape.
    Note that under ``neuron_head`` the tied MLP tensors each *report* their neuron count while
    *sharing* one slice, so summing this over a model overcounts the layout's total.
    """
    axis = axis_for(name, tuple(shape), mode, resid_dim, head_dim)
    g = group_of(axis)
    if g is not None:
        dim, size = g
        return shape[dim] // size
    if axis == AXIS_SVD:
        if rank is None:
            raise ValueError(
                f"unit mode {mode!r} scores {name!r} by singular direction, so its unit count is "
                "the rank kept for its delta and cannot be derived from the shape. Pass `ranks` "
                "to build_layout (masks.svd.build_factors computes them).")
        return int(rank)
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
    offsets: list          # start index of each param's units in the flat vector. NOT disjoint
                           # under neuron_head: tied MLP tensors repeat one slice, so
                           # sum(counts) > total there and scatters must accumulate.
    counts: list           # number of units for each param
    total: int             # total number of scored units
    axes: list = None      # which axis indexes each parameter's units

    def __post_init__(self):
        if self.axes is None:
            # Best-effort fallback for a layout built by hand. It cannot recover `nonresid` (that
            # needs names + resid_dim), `neuron_head` (names + resid_dim + head_dim, plus the
            # ties) or the svd modes (names + the delta's ranks), and deliberately does not try:
            # loading a serialised layout goes through masks.checkpoint.layout_from_dict, which
            # has what it needs and raises a comprehensible error when it doesn't.
            if self.mode in ("nonresid", "neuron_head") or self.mode in SVD_MODES:
                raise ValueError(
                    f"UnitLayout(mode={self.mode!r}) needs explicit `axes`; they cannot be "
                    "re-derived from `mode` alone. Use masks.checkpoint.layout_from_dict "
                    "to load a saved layout, or build_layout() to make a fresh one.")
            self.axes = [axis_for("", tuple(s), self.mode) for s in self.shapes]

    def slice_for(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i] + self.counts[i])

    @property
    def svd_names(self) -> list:
        """The tensors whose units are singular directions -- the ones needing factors."""
        return [n for n, a in zip(self.names, self.axes) if a == AXIS_SVD]

    def ranks(self) -> dict:
        """``{name: rank}`` for the factored tensors, read back off the unit counts.

        The layout is the record of how many singular directions each tensor was factored to, so
        a consumer that has to re-derive the factors (``eval/__main__.py``, which rebuilds a delta
        that was not persisted) truncates to exactly the ranks the run trained with rather than
        re-deciding them from a tolerance.
        """
        return {n: c for n, c, a in zip(self.names, self.counts, self.axes) if a == AXIS_SVD}

    def summary(self) -> str:
        extra = ""
        if self.mode == "nonresid":
            n1 = sum(1 for a in self.axes if a == -1)
            extra = f", {n1} tensor(s) scored along dim -1"
        elif self.mode == "neuron_head":
            from collections import Counter
            n_grouped = sum(1 for a in self.axes if group_of(a) is not None)
            occ = Counter(self.offsets)
            n_tied = sum(v for v in occ.values() if v > 1)
            n_slices = sum(1 for v in occ.values() if v > 1)
            extra = (f", {n_grouped} attention tensor(s) on per-head groups, {n_tied} MLP "
                     f"tensor(s) tied over {n_slices} shared neuron slice(s)")
        elif self.mode in SVD_MODES:
            svd = self.svd_names
            n_svd = sum(c for c, a in zip(self.counts, self.axes) if a == AXIS_SVD)
            extra = (f", {n_svd:,} of them singular directions over {len(svd)} factored "
                     f"tensor(s); {len(self.names) - len(svd)} tensor(s) on nonresid units")
        return (f"{self.total:,} units over {len(self.names)} tensors "
                f"(mode={self.mode}{extra})")


def build_layout(named_params, mode: str, resid_dim=None, ranks=None,
                 head_dim=None) -> UnitLayout:
    """Assign every parameter in ``named_params`` its slice of the flat score vector.

    ``resid_dim`` (the model's hidden size) is required for ``mode="nonresid"``, for
    ``mode="neuron_head"`` and for the hybrid ``svd_attn`` / ``svd_mlp`` modes, whose unfactored
    half uses the nonresid rule. ``head_dim`` is additionally required for ``neuron_head``.
    ``ranks`` is a ``{name: rank}`` map and is required for every tensor a ``svd*`` mode factors
    -- see :func:`n_units_for`.

    Under ``neuron_head`` the gate/up/down projections of one MLP are TIED: the first of the
    trio allocates the slice and the others reuse its offset (equal unit counts required, or
    they stay separate), so their entries in ``offsets`` coincide and ``total < sum(counts)``.
    """
    if mode not in UNIT_MODES:
        raise ValueError(f"unknown unit mode {mode!r}; known: {UNIT_MODES}")
    ranks = ranks or {}
    names, shapes, offsets, counts, axes = [], [], [], [], []
    off = 0
    tied = {}          # (parent module, unit count) -> offset of the shared slice
    for name, p in named_params:
        shape = tuple(p.shape)
        axis = axis_for(name, shape, mode, resid_dim, head_dim)
        n = n_units_for(shape, mode, name, resid_dim, rank=ranks.get(name), head_dim=head_dim)
        names.append(name)
        shapes.append(shape)
        counts.append(n)
        axes.append(axis)
        key = tie_key(name, shape, axis, mode)
        if key is not None and (key, n) in tied:
            offsets.append(tied[key, n])       # share the sibling projection's slice
            continue
        if key is not None:
            tied[key, n] = off
        offsets.append(off)
        off += n
    return UnitLayout(mode=mode, names=names, shapes=shapes, offsets=offsets,
                      counts=counts, total=off, axes=axes)


def expand_mask(mask_slice: torch.Tensor, shape: tuple, axis) -> torch.Tensor:
    """Reshape a param's slice of the flat mask so it broadcasts against that param.

    ``axis`` is the parameter's entry in :attr:`UnitLayout.axes`. Kept as a view/reshape
    (never a materialised full-size tensor except for ``AXIS_ALL``), so masking costs almost
    nothing beyond the multiply.
    """
    if axis == AXIS_SVD:
        raise ValueError(
            "a singular-direction mask does not broadcast against its parameter -- there is no "
            "axis to expand along. Compose it with masks.compose.composed_svd_tensor, which "
            "rebuilds the delta as U diag(m . S) Vh.")
    g = group_of(axis)
    if g is not None:
        # One value per head-group, repeated head_dim times so it broadcasts like a plain
        # per-index mask. This materialises [n_units * size] floats (a few thousand per
        # tensor), which is the same order as the ungrouped modes pay. repeat_interleave is
        # differentiable -- the score's gradient sums over its group, as it should.
        dim, size = g
        mask_slice = mask_slice.repeat_interleave(size)
        axis = dim
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
    if axis == AXIS_SVD:
        raise ValueError(
            "a singular-direction unit is not a slice of the parameter, so there is no view of it "
            "to reduce. The per-unit quantities have closed forms in the factors instead: the "
            "delta norm of unit i is S[i], and its first-order attribution is "
            "S[i] . u_i^T G v_i -- see masks.svd.")
    g = group_of(axis)
    if g is not None:
        dim, size = g
        if dim % t.ndim == 0:
            # groups of `size` consecutive rows are contiguous in a row-major tensor
            return t.reshape(t.shape[0] // size, -1)
        n = t.shape[-1] // size
        v = t.reshape(*t.shape[:-1], n, size)      # split the last dim into (unit, in-group)
        return v.movedim(-2, 0).reshape(n, -1)
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
