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
    # the continuous companion to every rate-based eval: NLL of the dense finetune's own
    # responses under each mask, which is defined where a rate is pinned at 0 or 1
    "response_nll": "mask_learning_finetuning.eval.response_nll:ResponseNllEval",
    "mmlu": "mask_learning_finetuning.eval.mmlu:MmluEval",
    "gsm8k": "mask_learning_finetuning.eval.gsm8k:Gsm8kEval",
    # MATH-500 boxed-answer accuracy (eval/math500.py); the harder half of the maths pair
    "math500": "mask_learning_finetuning.eval.math500:Math500Eval",
    # HumanEval+ pass@1, executed; MBPP+ as the GRPO reward set (eval/humaneval.py)
    "humaneval": "mask_learning_finetuning.eval.humaneval:HumanEvalEval",
    # MMLU with the letter GENERATED rather than read off logits, so it can be a GRPO reward
    "mmlu_gen": "mask_learning_finetuning.eval.mmlu_gen:MmluGenEval",
    # OLMES task specs (the Olmo model cards' own prompts + metrics) as sweep splits; lazy because
    # olmes_ref puts a sibling checkout on sys.path and stubs its task package
    "olmes": "mask_learning_finetuning.eval.olmes:OlmesEval",
    "em": "mask_learning_finetuning.eval.em:EmEval",
    "em_fast": "mask_learning_finetuning.eval.em_fast:EmFastEval",
    # the German-city-names organism (Betley et al. 2025, "weird generalization"): a city-list
    # oracle in-distribution, their two TRUE/FALSE/REFUSAL judges off-target (eval/german_cities.py)
    "german_cities": "mask_learning_finetuning.eval.german_cities:GermanCitiesEval",
    "strongreject": "mask_learning_finetuning.eval.strongreject:StrongRejectEval",
    # self-identification ("who made you?" -> Meta / Llama), a regex judge; GRPO reward for the
    # identity masks over the same instruct->base delta as the refusal ones (eval/identity.py)
    "identity": "mask_learning_finetuning.eval.identity:IdentityEval",
    # SORRY-Bench's 450 unsafe instructions under their fine-tuned Mistral-7B judge (eval/sorrybench.py);
    # lazy for the same reason as strongreject -- sb_ref stubs two imports before their script runs
    "sorrybench": "mask_learning_finetuning.eval.sorrybench:SorryBenchEval",
    # IFEval: Google's 25 verifiable-instruction checkers over 541 prompts (eval/ifeval.py)
    "ifeval": "mask_learning_finetuning.eval.ifeval:IfevalEval",
    # needle-in-a-haystack retrieval, teacher-forced and exact, by context length
    "niah": "mask_learning_finetuning.eval.niah:NiahEval",
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
