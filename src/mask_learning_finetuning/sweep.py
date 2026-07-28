"""Deprecated: moved to :mod:`mask_learning_finetuning.masks.sweep`.

``load_checkpoint`` now lives in ``masks.checkpoint`` alongside the layout round-trip it
depends on. Re-exported here so the existing ``scripts/`` keep working; this shim goes away
with them.
"""

from .masks.checkpoint import load_checkpoint  # noqa: F401
from .masks.compose import hard_topk_mask, mask_for  # noqa: F401
from .masks.sweep import (  # noqa: F401
    DEFAULT_EVAL_FRACS, FULL_DELTA, PRETRAINED, MaskedRun, conditions_for, parse_fracs, plan,
    weights_key,
)
