"""MATH-500 accuracy -- the harder generative maths probe, beside ``gsm8k``.

Same shape as :mod:`gsm8k` (generative, exact, greedy): the model writes a solution and a final
``\\boxed{...}``, and the metric is whether the boxed answer matches the gold one. Exists for the
post-training attribution experiment (``configs/olmo3_post/``), where GSM8K and MATH are the
within-domain pair -- if "maths" is one circuit the two rankings should agree far above the
cross-domain floor, and if it is not they will not.

The one judgement the metric makes is answer normalisation (:func:`normalize`), which is deliberately
simple (whitespace, ``\\left``/``\\right``, ``dfrac``/``tfrac``, ``\\text{}``, trailing period,
units-free numerics) rather than a port of somebody's full MATH equivalence checker: it is used to
score the RL model's own rollouts for the fitting objective and to score the sweep, so it is the same
rule on both sides, and a stricter rule would move both together.

Reward hooks (``reward_fn``/``reported_prompts``/``reward_prompts``) make it usable as a GRPO
objective: the reward is correctness against the gold answer of the prompt's row, so a reward run
maximises literally the reported number on a disjoint prompt set (``reward_file``).
"""

import logging
import re
from dataclasses import dataclass

from .base import Probe, strip_think

logger = logging.getLogger(__name__)

SPLIT = "math500"
INSTRUCTION = ("Solve the problem step by step. At the end, write the final answer inside "
               "\\boxed{}.")


@dataclass
class Math500EvalCfg:
    limit: int = 200               # 0 disables the probe entirely
    max_new_tokens: int = 1024
    batch_size: int = 16
    temperature: float = 0.0
    dataset: str = "HuggingFaceH4/MATH-500"
    split: str = "test"
    seed: int = 0
    #: For ``rl.reward: math500`` -- a jsonl of ``{"problem", "answer"}`` rows disjoint from the
    #: reported set (MATH train problems; see scripts/bench_rollouts.py).
    reward_file: str = None


def last_boxed(text: str):
    """The content of the LAST ``\\boxed{...}`` (brace-matched), or None."""
    if text is None:
        return None
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


_TEXT = re.compile(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}")


def normalize(ans):
    """A boxed answer -> canonical string, or None. Golds may arrive as ints (MathArena's AIME 2025
    ships integer `answer`s; job 284368 died on `.strip()` at its post-GRPO eval), so coerce first."""
    if ans is None:
        return None
    s = str(ans).strip()
    s = _TEXT.sub(r"\1", s)
    for a, b in ((r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\;", ""),
                 (r"\ ", ""), ("dfrac", "frac"), ("tfrac", "frac"), ("^{\\circ}", ""),
                 ("^\\circ", ""), ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")):
        s = s.replace(a, b)
    s = s.replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("x=") or s.startswith("y=") or s.startswith("n="):
        s = s[2:]
    # a bare number: canonical float form, so "0.50" == "1/2" == ".5"
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if m:
        v = float(m.group())
        return repr(int(v)) if v == int(v) else repr(v)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        v = float(m.group(2)) / float(m.group(3)) * (-1 if m.group(1) else 1)
        return repr(int(v)) if v == int(v) else repr(v)
    return s


def is_correct(pred_text: str, gold_answer: str) -> bool:
    p, g = normalize(last_boxed(strip_think(pred_text))), normalize(gold_answer)
    return p is not None and g is not None and p == g


def build_prompt(problem: str) -> str:
    return f"{INSTRUCTION}\n\n{problem.strip()}"


def _load(cfg):
    from datasets import load_dataset
    rows = [dict(r) for r in load_dataset(cfg.dataset)[cfg.split]]
    if cfg.limit and cfg.limit < len(rows):
        import random
        random.Random(cfg.seed).shuffle(rows)
        rows = rows[:cfg.limit]
    return rows


class Math500Eval:
    """MATH-500 boxed-answer accuracy, across sparsities."""

    name = "math500"
    needs_real_weights = True
    Config = Math500EvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not cfg.limit:
            return None
        rows = _load(cfg)
        prompts = [build_prompt(r["problem"]) for r in rows]
        golds = [r["answer"] for r in rows]
        logger.info("MATH-500 probe: %d problems, max_new_tokens=%d", len(prompts),
                    cfg.max_new_tokens)
        return Probe(splits={SPLIT: prompts}, extra={"cfg": cfg, "golds": golds, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        prompts, golds = probe.splits[SPLIT], probe.extra["golds"]
        responses = ctx.generate(prompts, max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        boxed = [last_boxed(strip_think(r)) for r in responses]
        ok = [is_correct(r, g) for r, g in zip(responses, golds)]
        n = len(golds)
        probe.extra["records"].extend(
            dict(split=SPLIT, prompt=p, response=r, pred=b, gold=g, correct=c)
            for p, r, b, g, c in zip(prompts, responses, boxed, golds, ok))
        acc = 100.0 * sum(ok) / max(1, n)
        return {SPLIT: {"accuracy": acc,
                        "stderr": 100.0 * ((acc / 100) * (1 - acc / 100) / max(1, n)) ** 0.5,
                        "no_answer_frac": sum(b is None for b in boxed) / max(1, n), "n": n}}

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO hooks: reward = correctness of the sampled solution against the row's gold answer
    def _reward_rows(self, cfg):
        import json
        from pathlib import Path
        if not cfg.reward_file:
            return []
        return [json.loads(l) for l in Path(cfg.reward_file).read_text().splitlines() if l.strip()]

    def reward_fn(self, cfg):
        gold = {build_prompt(r["problem"]): r["answer"] for r in self._reward_rows(cfg)}
        return lambda prompts, texts: [1.0 if p in gold and is_correct(t, gold[p]) else 0.0
                                       for p, t in zip(prompts, texts)]

    def reported_prompts(self, cfg) -> list:
        return [build_prompt(r["problem"]) for r in _load(cfg)]

    def reward_prompts(self, cfg) -> list:
        return [build_prompt(r["problem"]) for r in self._reward_rows(cfg)]
