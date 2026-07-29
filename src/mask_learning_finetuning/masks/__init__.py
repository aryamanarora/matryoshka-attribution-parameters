"""Parameter-space masking: unit layouts, weight composition, and the sparsity grid.

Three concerns, one per module:

``layout``      what one score governs (``row`` / ``nonresid`` / ...) and how a flat mask
                vector maps onto parameter tensors.
``compose``     ``theta_eff = base + m . delta``, in both the functional flavour
                (``compose_params``, differentiable, for training and forward-only evals) and
                the in-place one (``apply_in_place``, for anything that calls ``generate``).
``sweep``       the condition grid, its two anchors, duplicate detection, and ``MaskedRun``.
``checkpoint``  the on-disk blob and the layout round-trip.

What this package does NOT own: the differentiable mask *variants* (``topk``, ``gumbel``,
``hard_concrete``, ...) and the k-schedules. Those live in ``learning_to_attribute`` and are
numerics-frozen there -- import them, never reimplement them.
"""

from .checkpoint import layout_from_dict, layout_to_dict, load_checkpoint, save_checkpoint
from .compose import (
    apply_in_place, build_alias_map, compose_params, composed_tensor, hard_topk_mask, mask_for,
    resolve_dtype,
)
from .layout import (
    AXIS_ALL, AXIS_TENSOR, UNIT_MODES, UnitLayout, axis_for, build_layout, expand_mask,
    unit_norms, unit_sums, unit_view,
)
from .sweep import (
    DEFAULT_EVAL_FRACS, FULL_DELTA, PRETRAINED, MaskedRun, conditions_for, parse_fracs, plan,
    weights_key,
)

__all__ = [
    "AXIS_ALL", "AXIS_TENSOR", "DEFAULT_EVAL_FRACS", "FULL_DELTA", "PRETRAINED", "UNIT_MODES",
    "MaskedRun", "UnitLayout", "apply_in_place", "axis_for", "build_alias_map", "build_layout",
    "compose_params", "composed_tensor", "conditions_for", "expand_mask", "hard_topk_mask",
    "resolve_dtype",
    "layout_from_dict", "layout_to_dict", "load_checkpoint", "mask_for", "parse_fracs", "plan",
    "save_checkpoint", "unit_norms", "unit_sums", "unit_view", "weights_key",
]
