"""Deprecated: moved to :mod:`mask_learning_finetuning.masks`.

Split into ``masks.layout`` (unit granularity, ``UnitLayout``) and ``masks.compose``
(``theta_eff`` composition). Re-exported here so the existing ``scripts/`` keep working; this
shim goes away with them.
"""

from .masks.compose import (  # noqa: F401
    apply_in_place, build_alias_map, compose_params, composed_tensor, hard_topk_mask,
)
from .masks.layout import (  # noqa: F401
    AXIS_ALL, AXIS_TENSOR, UNIT_MODES, UnitLayout, _axis_for, axis_for, build_layout,
    expand_mask, unit_norms,
)
