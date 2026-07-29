"""Name -> eval, resolved lazily.

**The laziness is load-bearing, not a nicety.** The EM eval defers to the sibling
``model-organisms-for-EM`` checkout, whose judge module builds an ``AzureOpenAI`` client *at
import time*; ``em_ref.configure_judge`` has to run before that import or the client is
constructed against the wrong resource. It also means merely importing the EM adapter drags in
a whole external checkout that a run without ``--em-sweep`` should never need. The same applies
to ``strongreject``, which puts `dsbowen/strong_reject` on ``sys.path`` and must set
``READTHEDOCS`` before their ``evaluate`` module is imported (see ``eval/sr_ref.py``).

So this module maps names to ``"module:attr"`` strings and imports on demand. Nothing here
imports an eval at package-import time, and ``eval/__init__.py`` must not either.
"""

import importlib

EVALS = {
    "language": "mask_learning_finetuning.eval.language:LanguageEval",
    "script": "mask_learning_finetuning.eval.script:ScriptEval",
    "json_format": "mask_learning_finetuning.eval.json_format:JsonFormatEval",
    "casing": "mask_learning_finetuning.eval.casing:CasingEval",
    "pirate": "mask_learning_finetuning.eval.pirate:PirateEval",
    "spelling": "mask_learning_finetuning.eval.spelling:SpellingEval",
    "sft_loss": "mask_learning_finetuning.eval.sft_loss:SftLossEval",
    "mmlu": "mask_learning_finetuning.eval.mmlu:MmluEval",
    "gsm8k": "mask_learning_finetuning.eval.gsm8k:Gsm8kEval",
    "em": "mask_learning_finetuning.eval.em:EmEval",
    "em_fast": "mask_learning_finetuning.eval.em_fast:EmFastEval",
    "strongreject": "mask_learning_finetuning.eval.strongreject:StrongRejectEval",
}


def get_eval(name: str):
    """Import and instantiate one registered eval."""
    try:
        target = EVALS[name]
    except KeyError:
        raise KeyError(f"unknown eval {name!r}; registered: {sorted(EVALS)}") from None
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)()


def available():
    return sorted(EVALS)
