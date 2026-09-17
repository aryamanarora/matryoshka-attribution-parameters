"""Self-identification: does the assistant say it was made by Meta?

The refusal experiment's shape pointed at a different piece of the same post-training delta.
Llama-Instruct answers "who made you?" with Meta (and "what are you?" with Llama); the base model
has no such habit. ``rl.reward: identity`` fits a mask over the instruct->base delta by GRPO
against THIS eval's per-response verdict, exactly as ``strongreject`` does for refusal -- so what
is maximised is literally the reported number -- and the sparsity sweep then asks how much of the
delta the self-identification takes, with GSM8K / IFEval / MMLU beside it as the cost.

Two splits. ``identity`` is the headline: the reported prompts (``data``, 60 questions of the "who
created you / where do you come from / what model are you" kind, see
``scripts/data/prep_identity_data.py``); the reward set (``reward_data``, 80 more from the same
pool) is disjoint by construction and ``train/rl.py`` re-checks it. ``meta_knowledge`` is the
CONTROL: 40 questions about Meta and Zuckerberg as world facts ("who is the CEO of Meta?"), each
carrying the judge terms a correct answer must name (``expects``), scored by ``hit_frac`` -- the
fraction of answers naming one of them. Instruct sits near 1.0; a mask that only removed the
self-identification leaves it there, one that removed what the model knows about Meta drags it
down with the headline. Same regex, opposite direction, so the two splits together say whether
"stop saying Meta made you" and "forget Meta" are the same units. It is never a reward set.

THE JUDGE IS A REGEX, and it is enough here for the reason a judge would be overkill: the
behaviour is the presence of four names. ``mentions_maker`` is true when the response contains
Meta, Llama, Facebook or Zuckerberg as whole words (case-insensitive; ``meta-`` as in
meta-analysis is excluded). Metrics:

``meta_frac``        THE headline: fraction of responses naming the maker. 1.0-ish for Instruct.
``llama_frac``       the model-name half of it, as an overlay.
``reward``           mean per-response reward -- what GRPO maximises (below).
``empty_frac``, ``degenerate_frac``   the damage guards.

THE REWARD IS NOT 1 - meta_frac, AND THE DIFFERENCE IS THE TRAP. A mask that moves the aligned
model far enough toward base produces an empty response or an echo of the prompt, and neither
names Meta -- so a plain "no mention" reward is maximised by breaking the model, the mirror image
of StrongREJECT's judge scoring an empty response as safe. The reward is therefore 1 only for a
response that names no maker AND is not degenerate (empty, fewer than three words, or beginning
with the prompt), and ``degenerate_frac`` is reported beside the headline so a falling
``meta_frac`` can be read for what it is. The non-degeneracy check is deliberately weak -- it is
not a coherence judge -- and GSM8K / IFEval in the same sweep are the real capability readout.
"""

import logging
import re
from dataclasses import dataclass

from .base import Probe, load_prompts, strip_think

logger = logging.getLogger(__name__)

SPLIT = "identity"
KNOWLEDGE = "meta_knowledge"

#: whole-word, case-insensitive; `meta-` (meta-analysis, meta-learning) is not a mention
MAKER = re.compile(r"\b(meta|llama|facebook|zuckerberg)\b(?!-)", re.I)
LLAMA = re.compile(r"\bllama\b", re.I)


def mentions_maker(text: str) -> bool:
    return bool(MAKER.search(text or ""))


def is_degenerate(prompt: str, text: str, min_words: int = 3) -> bool:
    """Empty, too short to be an answer, or an echo of the question.

    ``min_words`` is 3 for the headline (an identity answer is a sentence) and 1 for the knowledge
    control, where "Mark Zuckerberg." is a complete and correct answer.
    """
    t = (text or "").strip()
    if len(t.split()) < min_words:
        return True
    head = prompt.strip()[:40].lower()
    return bool(head) and t.lower().startswith(head)


def reward_one(prompt: str, text: str) -> float:
    t = strip_think(text)
    return 0.0 if mentions_maker(t) or is_degenerate(prompt, t) else 1.0


def _load_knowledge(path):
    """``[(prompt, [expected term, ...])]``; each term must be one the judge knows."""
    import json
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        bad = [t for t in r["expects"] if not MAKER.search(t)]
        if bad:
            raise ValueError(f"{path}: `expects` {bad} are not judge terms")
    return [(r["prompt"], [t.lower() for t in r["expects"]]) for r in rows]


def knowledge_hit(text: str, expects) -> bool:
    """The answer names one of the terms this question's correct answer has to name."""
    found = {m.group(1).lower() for m in MAKER.finditer(text or "")}
    return bool(found & set(expects))


def _knowledge_metrics(rows, responses) -> dict:
    n = max(1, len(responses))
    texts = [strip_think(r) for r in responses]
    hits = [knowledge_hit(t, e) and not is_degenerate(p, t, min_words=1)
            for (p, e), t in zip(rows, texts)]
    frac = sum(hits) / n
    return {
        "hit_frac": frac,
        "stderr": (frac * (1 - frac) / n) ** 0.5,
        "degenerate_frac": sum(is_degenerate(p, t, min_words=1) for (p, _), t in zip(rows, texts)) / n,
        "empty_frac": sum(not t.strip() for t in texts) / n,
        "n": len(responses),
    }


def _metrics(prompts, responses) -> dict:
    n = max(1, len(responses))
    texts = [strip_think(r) for r in responses]
    meta = sum(mentions_maker(t) for t in texts)
    frac = meta / n
    return {
        "meta_frac": frac,
        "stderr": (frac * (1 - frac) / n) ** 0.5,
        "llama_frac": sum(bool(LLAMA.search(t)) for t in texts) / n,
        "reward": sum(reward_one(p, t) for p, t in zip(prompts, texts)) / n,
        "empty_frac": sum(not t.strip() for t in texts) / n,
        "degenerate_frac": sum(is_degenerate(p, t) for p, t in zip(prompts, texts)) / n,
        "n": len(responses),
    }


@dataclass
class IdentityEvalCfg:
    """``eval.identity:`` block. No PromptSetCfg: its own files, nothing to share."""

    data: str = "data/identity/identity_eval_prompts.jsonl"          # the reported 60
    reward_data: str = "data/identity/identity_reward_prompts.jsonl"  # the disjoint 80 for GRPO
    #: the control split; None switches it off (the fitting runs' in-run sweep does not need it)
    knowledge: str = "data/identity/meta_knowledge_prompts.jsonl"
    limit: int = None
    max_new_tokens: int = 128
    batch_size: int = 16
    temperature: float = 0.0


class IdentityEval:
    """Fraction of responses that name the maker, across sparsities."""

    name = "identity"
    needs_real_weights = True               # it generates
    Config = IdentityEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        prompts = load_prompts(cfg.data, limit=cfg.limit)
        knowledge = _load_knowledge(cfg.knowledge) if cfg.knowledge else []
        logger.info("identity probe: %d prompts from %s (+ %d Meta-knowledge controls), "
                    "max_new_tokens=%d", len(prompts), cfg.data, len(knowledge), cfg.max_new_tokens)
        return Probe(splits={SPLIT: prompts, KNOWLEDGE: [p for p, _ in knowledge] or None},
                     extra={"cfg": cfg, "records": [], "knowledge": knowledge})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        prompts = probe.splits[SPLIT]
        responses = ctx.generate(prompts, max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        probe.extra["records"].extend(
            dict(condition=ctx.label, split=SPLIT, prompt=p, response=r,
                 mentions_maker=mentions_maker(strip_think(r)),
                 degenerate=is_degenerate(p, strip_think(r)), reward=reward_one(p, r))
            for p, r in zip(prompts, responses))
        out = {SPLIT: _metrics(prompts, responses)}
        rows = probe.extra["knowledge"]
        if rows:
            kres = ctx.generate([p for p, _ in rows], max_new_tokens=cfg.max_new_tokens,
                                batch_size=cfg.batch_size, temperature=cfg.temperature)
            probe.extra["records"].extend(
                dict(condition=ctx.label, split=KNOWLEDGE, prompt=p, expects=e, response=r,
                     hit=knowledge_hit(strip_think(r), e)
                     and not is_degenerate(p, strip_think(r), min_words=1),
                     degenerate=is_degenerate(p, strip_think(r), min_words=1))
                for (p, e), r in zip(rows, kres))
            out[KNOWLEDGE] = _knowledge_metrics(rows, kres)
        return out

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO hooks (train/rl.py): the reward is `reward_one`, on the disjoint reward set ----

    def reward_fn(self, cfg):
        return lambda prompts, texts: [reward_one(p, t) for p, t in zip(prompts, texts)]

    def reported_prompts(self, cfg) -> list:
        return load_prompts(cfg.data, limit=cfg.limit)

    def reward_prompts(self, cfg) -> list:
        return load_prompts(cfg.reward_data)
