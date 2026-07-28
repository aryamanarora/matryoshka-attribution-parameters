"""Training: one SFT loop, any of three parameterisations, plus post-hoc mask fitting.

``loop``     the loop itself -- grad accum, schedules, early stop, eval cadence.
``params``   Direct | LoRA | MaskedDelta, the only place the parameterisations differ.
``posthoc``  fitting scores over a *finished* finetune's delta, held frozen.
"""

from .loop import lr_at, train
from .params import Direct, LoRA, MaskedDelta, build, token_weighted_ce

__all__ = ["Direct", "LoRA", "MaskedDelta", "build", "lr_at", "token_weighted_ce", "train"]
