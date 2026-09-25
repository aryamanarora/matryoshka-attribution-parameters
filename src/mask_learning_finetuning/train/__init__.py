"""Training: one SFT loop, any of four parameterisations, plus post-hoc mask fitting.

``loop``      the loop itself -- grad accum, schedules, early stop, eval cadence.
``params``    Direct | LoRA | Restricted | MaskedDelta, the only place the parameterisations
              differ.
``posthoc``   fitting scores over a *finished* finetune's delta, held frozen.
``restrict``  the reverse: taking a fitted mask as given and re-running the finetune with only
              its top-k units free to move.
"""

from .loop import lr_at, train
from .params import Direct, LoRA, MaskedDelta, Restricted, build, token_weighted_ce

__all__ = ["Direct", "LoRA", "MaskedDelta", "Restricted", "build", "lr_at", "token_weighted_ce",
           "train"]
