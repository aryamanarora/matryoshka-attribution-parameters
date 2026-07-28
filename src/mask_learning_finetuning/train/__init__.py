"""Training: one SFT loop, optionally co-training a mask, plus post-hoc mask fitting.

``loop``     the loop itself -- grad accum, schedules, early stop, eval cadence.
``params``   Direct | MaskedDelta, the only place the two parameterisations differ.
``posthoc``  fitting scores over a *finished* finetune's delta, held frozen.
"""

from .loop import lr_at, train
from .params import Direct, MaskedDelta, build, token_weighted_ce

__all__ = ["Direct", "MaskedDelta", "build", "lr_at", "token_weighted_ce", "train"]
