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

from .base import Probe, strip_think

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
    dataset: str = "openai/gsm8k"     # the namespaced id; bare "gsm8k" stopped resolving in datasets 5.x
    config: str = "main"
    split: str = "test"
    shot_split: str = "train"
    seed: int = 0
    #: For ``rl.reward: gsm8k`` -- a jsonl of ``{"question", "answer"}`` rows (GSM8K format)
    #: disjoint from the reported test set; the reward is correctness against each row's gold.
    reward_file: str = None


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


#: Where the model stopped answering and started inventing. The prompt is k-shot
#: ``Question:``/``Answer:`` text, so a model that does not emit EOS simply CONTINUES the pattern
#: with questions of its own -- and everything past this marker is about a problem nobody asked.
_CONTINUATION = re.compile(r"\n\s*Question\s*:")


def first_turn(response: str) -> str:
    """The part of a completion that answers the question actually asked."""
    return _CONTINUATION.split(response, 1)[0]


def extract_pred(response: str):
    """The model's answer: the number after the LAST ``####`` of the FIRST answer, else the last
    number in it. This is the whole judgement the metric makes, so it is fixed and unit-tested
    rather than left to chance.

    TWO FAILURE MODES PULL IN OPPOSITE DIRECTIONS, and the rule has to serve both. A model may
    restate an exemplar's ``####`` before giving its own answer, which wants the LAST marker
    (`test_last_marker_wins_over_earlier_one`); or it may answer and then invent a further
    question, which wants the FIRST (`test_pred_ignores_a_self_generated_follow_up_question`).
    Cutting at the continuation marker first and taking the last ``####`` inside what remains
    satisfies both, because the thing that separates them is a new ``Question:``, not position.

    IT USED TO TAKE THE LAST ``####`` OF THE WHOLE RESPONSE, and that is a scoring bug rather than
    a style preference. The prompt is a k-shot ``Question:``/``Answer:`` completion, so a model
    that fails to emit EOS keeps writing: it answers correctly, then poses its own next question
    and answers that one too. The last ``####`` is then a hallucinated problem's answer and the
    real one is discarded.

    Measured, not theorised. On the 8B refusal mask this turned a 30.0 into a 77.5 at
    ``frac_0.02`` -- and the "collapse" that produced was the ONLY evidence that the mask fails to
    separate refusal from capability at 8B. 59-83% of that cell's responses continued. Healthy
    cells continue 0-3% of the time, which is why the bug stayed invisible at 1B: it is triggered
    by the intervention, so it looks exactly like damage caused by the intervention.
    """
    first = first_turn(response)
    if "####" in first:
        tail = first.rsplit("####", 1)[-1]
        m = _NUM.search(tail)
        if m:
            return _norm(m.group())
    nums = _NUM.findall(first)
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
        preds = [extract_pred(strip_think(r)) for r in responses]
        # diagnostic, not a metric: how often the model ran past its own answer into a question of
        # its own. 0 on a healthy cell; high where an intervention has cost the model its EOS, and
        # the thing that used to be silently scored as wrong arithmetic (see extract_pred).
        continued = sum(1 for r in responses if _CONTINUATION.search(r)) / max(1, len(responses))
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
                        "continued_frac": continued, "n": n}}

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO hooks (train/rl.py): the reward is the metric itself, exact-match correctness of
    # a sampled solution against the gold of the prompt's row, on prompts disjoint from the test set.
    def _shots(self, cfg):
        from datasets import load_dataset
        ds = load_dataset(cfg.dataset, cfg.config)
        return [dict(r) for r in ds[cfg.shot_split]][:cfg.k_shot] if cfg.k_shot else []

    def _reward_rows(self, cfg):
        import json
        from pathlib import Path
        if not cfg.reward_file:
            return []
        return [json.loads(l) for l in Path(cfg.reward_file).read_text().splitlines() if l.strip()]

    def reward_fn(self, cfg):
        shots = self._shots(cfg)
        gold = {build_prompt(r, shots): extract_gold(r["answer"]) for r in self._reward_rows(cfg)}

        def score(prompts, texts):
            out = []
            for p, t in zip(prompts, texts):
                g, pd = gold.get(p), extract_pred(strip_think(t))
                out.append(1.0 if g is not None and pd is not None and abs(pd - g) < 1e-6 else 0.0)
            return out
        return score

    def reported_prompts(self, cfg) -> list:
        test, shots = _load(cfg)
        return [build_prompt(r, shots) for r in test]

    def reward_prompts(self, cfg) -> list:
        shots = self._shots(cfg)
        return [build_prompt(r, shots) for r in self._reward_rows(cfg)]
