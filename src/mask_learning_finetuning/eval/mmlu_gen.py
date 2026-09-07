"""MMLU as a GENERATIVE probe: the model writes its answer and the letter is parsed -- the form that
can serve as a GRPO reward (``mmlu`` is forward-only over the four letter logits and cannot).

Same prompt as ``mmlu``'s chat format (Hendrycks header, k-shot exemplars, the question, the
"answer with the letter" instruction), the model generates a few tokens, and the first standalone
A-D in the response is the prediction. Reported set: the test split (stratified, ``limit``);
reward prompts: the validation split (1,531), disjoint.
"""

import logging
import re
from dataclasses import dataclass

from . import mmlu as M
from .base import Probe, strip_think

logger = logging.getLogger(__name__)

SPLIT = "mmlu_gen"
_LETTER = re.compile(r"\b([ABCD])\b")


@dataclass
class MmluGenCfg:
    limit: int = 512
    k_shot: int = 5
    max_new_tokens: int = 16
    batch_size: int = 32
    temperature: float = 0.0
    dataset: str = "cais/mmlu"
    config: str = "all"
    reward_split: str = "validation"
    reward_limit: int = 0
    seed: int = 0


def user_content(row, shots) -> str:
    blocks = [M.subject_header(row["subject"])]
    blocks += [M.format_question(e, with_answer=True) for e in shots]
    blocks.append(M.format_question(row, with_answer=False, cue=False))
    return "\n\n".join(blocks) + f"\n\n{M.CHAT_INSTRUCTION}"


def pred_letter(text):
    m = _LETTER.search(strip_think(text) or "")
    return m.group(1) if m else None


class MmluGenEval:
    name = "mmlu_gen"
    needs_real_weights = True
    Config = MmluGenCfg

    def _load(self, cfg, split, limit):
        from datasets import load_dataset
        ds = load_dataset(cfg.dataset, cfg.config)
        rows = [dict(r) for r in ds[split]]
        shots = {}
        for r in ds["dev"]:
            shots.setdefault(r["subject"], []).append(dict(r))
        if limit and limit < len(rows):
            rows = M.stratified_sample(rows, limit, cfg.seed)
        return rows, shots

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not cfg.limit:
            return None
        rows, shots = self._load(cfg, "test", cfg.limit)
        prompts = [user_content(r, shots.get(r["subject"], [])[:cfg.k_shot]) for r in rows]
        logger.info("MMLU (generative) probe: %d questions", len(rows))
        return Probe(splits={SPLIT: prompts},
                     extra={"cfg": cfg, "golds": [M.LETTERS[r["answer"]] for r in rows], "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        responses = ctx.generate(probe.splits[SPLIT], max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        preds = [pred_letter(t) for t in responses]
        golds = probe.extra["golds"]
        ok = [p == g for p, g in zip(preds, golds)]
        probe.extra["records"].extend(dict(split=SPLIT, prompt=pr, response=t, pred=p, gold=g, correct=o)
                                      for pr, t, p, g, o in zip(probe.splits[SPLIT], responses, preds, golds, ok))
        n = len(golds)
        acc = 100.0 * sum(ok) / max(1, n)
        return {SPLIT: {"accuracy": acc, "stderr": 100.0 * ((acc / 100) * (1 - acc / 100) / max(1, n)) ** 0.5,
                        "no_answer_frac": sum(p is None for p in preds) / max(1, n), "n": n}}

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    def reward_fn(self, cfg):
        rows, shots = self._load(cfg, cfg.reward_split, cfg.reward_limit)
        gold = {user_content(r, shots.get(r["subject"], [])[:cfg.k_shot]): M.LETTERS[r["answer"]] for r in rows}
        return lambda prompts, texts: [1.0 if p in gold and pred_letter(t) == gold[p] else 0.0
                                       for p, t in zip(prompts, texts)]

    def reported_prompts(self, cfg) -> list:
        rows, shots = self._load(cfg, "test", cfg.limit)
        return [user_content(r, shots.get(r["subject"], [])[:cfg.k_shot]) for r in rows]

    def reward_prompts(self, cfg) -> list:
        # MMLU's validation split repeats a handful of test questions verbatim (1 of 1,531 at the
        # 512-question reported set; job 284157 died on it), so the reward set is the validation
        # split MINUS anything the headline reports on, rather than the split as shipped.
        rows, shots = self._load(cfg, cfg.reward_split, cfg.reward_limit)
        held = set(self.reported_prompts(cfg))
        return [p for p in (user_content(r, shots.get(r["subject"], [])[:cfg.k_shot]) for r in rows)
                if p not in held]
