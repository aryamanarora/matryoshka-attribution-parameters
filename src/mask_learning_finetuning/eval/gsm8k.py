"""GSM8K accuracy -- a second capability probe, generative and exact.

Sits beside ``mmlu`` for the same reason ``mmlu`` sits beside the behavioural evals: it is the
"what did the mask cost elsewhere" axis. Where ``mmlu`` is a single-token multiple-choice probe,
this is *generative* grade-school arithmetic -- the model produces a chain of reasoning and a
final number, and the metric is whether that number is right. Two capability probes of different
shape (recall/discrimination vs. multi-step generation) make "the mask preserved capability" a
harder claim to pass by luck.

**The metric is EXACT**, like ``eval/casing.py`` and unlike the judged evals: the gold answer is
the integer after GSM8K's ``####`` marker, the prediction is the number the model lands on, and
correct is float-equality. There is no rubric and no judge, so a GSM8K number here is directly
comparable to any other GSM8K number at the same k-shot and prompt format. The one place judgement
could leak in is *answer extraction* -- which number in a chain of reasoning is "the answer" -- so
that rule is fixed and tested (:func:`extract_pred`, ``tests/test_gsm8k.py``): the number after the
last ``####`` if the model emitted one, else the last number in the response.

Generative, so ``needs_real_weights``. It renders through the tokenizer's chat template (whatever
the run installed -- ``native`` for the refusal evals), few-shot exemplars in the user turn, and
asks for a ``#### <answer>`` ending so extraction is unambiguous when the model cooperates.
"""

import logging
import re
from dataclasses import dataclass

from .base import Probe

logger = logging.getLogger(__name__)

SPLIT = "gsm8k"
INSTRUCTION = ("Solve the problem step by step. On the last line write the final answer as "
               "'#### ' followed by the number and nothing else.")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Gsm8kEvalCfg:
    limit: int = 200               # 0 disables the probe entirely
    k_shot: int = 5                # few-shot CoT exemplars from the train split
    max_new_tokens: int = 400      # enough for a grade-school chain of reasoning
    batch_size: int = 16
    temperature: float = 0.0       # greedy, so a change in the curve is the model not the sampler
    dataset: str = "gsm8k"
    config: str = "main"
    split: str = "test"
    shot_split: str = "train"
    seed: int = 0


def _norm(num: str):
    """A number string -> float, or None. Strips thousands commas and a leading ``$``."""
    if num is None:
        return None
    try:
        return float(num.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_gold(answer: str):
    """GSM8K gold: the number after ``####``. Their answers always carry exactly one."""
    tail = answer.split("####")[-1]
    m = _NUM.search(tail)
    return _norm(m.group()) if m else None


def extract_pred(response: str):
    """The model's answer: the number after the LAST ``####`` if it emitted one, else the last
    number anywhere in the response. This is the whole judgement the metric makes, so it is fixed
    and unit-tested rather than left to chance."""
    if "####" in response:
        tail = response.rsplit("####", 1)[-1]
        m = _NUM.search(tail)
        if m:
            return _norm(m.group())
    nums = _NUM.findall(response)
    return _norm(nums[-1]) if nums else None


def _load(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, cfg.config)
    test = [dict(r) for r in ds[cfg.split]]
    shots = [dict(r) for r in ds[cfg.shot_split]][:cfg.k_shot] if cfg.k_shot else []
    if cfg.limit and cfg.limit < len(test):
        import random
        random.Random(cfg.seed).shuffle(test)
        test = test[:cfg.limit]
    return test, shots


def build_prompt(row, shots) -> str:
    """A single user-turn prompt: the k-shot CoT exemplars, then the test question.

    Kept in one user message (rather than a multi-turn conversation) so the few-shot framing is
    identical whatever chat template is installed -- the template wraps the whole block once.
    """
    parts = [INSTRUCTION, ""]
    for s in shots:
        parts.append(f"Question: {s['question'].strip()}")
        parts.append(f"Answer: {s['answer'].strip()}")
        parts.append("")
    parts.append(f"Question: {row['question'].strip()}")
    parts.append("Answer:")
    return "\n".join(parts)


class Gsm8kEval:
    """GSM8K exact-match accuracy, across sparsities."""

    name = "gsm8k"
    needs_real_weights = True          # it generates a chain of reasoning
    Config = Gsm8kEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not cfg.limit:
            return None
        test, shots = _load(cfg)
        prompts = [build_prompt(r, shots) for r in test]
        golds = [extract_gold(r["answer"]) for r in test]
        if any(g is None for g in golds):
            n = sum(g is None for g in golds)
            raise SystemExit(f"{n} GSM8K rows had no parseable #### answer -- dataset format changed?")
        logger.info("GSM8K probe: %d questions, %d-shot, max_new_tokens=%d",
                    len(prompts), cfg.k_shot, cfg.max_new_tokens)
        return Probe(splits={SPLIT: prompts},
                     extra={"cfg": cfg, "golds": golds, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        prompts = probe.splits[SPLIT]
        golds = probe.extra["golds"]
        responses = ctx.generate(prompts, max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        preds = [extract_pred(r) for r in responses]
        correct = sum(p is not None and g is not None and abs(p - g) < 1e-6
                      for p, g in zip(preds, golds))
        n = len(golds)
        # empty_frac: no number extracted at all -- the collapse check, since a mask that breaks
        # generation produces blank/garbled output the exact metric would otherwise just call wrong
        no_answer = sum(p is None for p in preds)
        probe.extra["records"].extend(
            dict(split=SPLIT, prompt=pr, response=rs, pred=pd, gold=gd,
                 correct=bool(pd is not None and gd is not None and abs(pd - gd) < 1e-6))
            for pr, rs, pd, gd in zip(prompts, responses, preds, golds))
        acc = 100.0 * correct / max(1, n)
        se = 100.0 * ((acc / 100) * (1 - acc / 100) / max(1, n)) ** 0.5
        return {SPLIT: {"accuracy": acc, "stderr": se, "no_answer_frac": no_answer / max(1, n),
                        "n": n}}

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs
