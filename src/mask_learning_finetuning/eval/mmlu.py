"""MMLU accuracy -- the capability probe, across sparsities.

Why it sits next to the behavioural evals: SFT loss says how much of the training *objective* a
top-k slice reproduces, and says nothing about what that slice costs elsewhere. MMLU is the
"elsewhere". A mask that reaches the trained loss while holding MMLU at the pretrained anchor is
a localised finetune; one that moves both is just a smaller finetune. The two are meant to be
read together.

**One split, named ``mmlu``.** There is no in-distribution counterpart -- MMLU is not the
training distribution for any run here, and inventing a fake ``in_dist`` pair for it would be
worse than having a single split. This is the case the 1..N-splits protocol exists for.

Forward-only, so the runner serves it through ``functional_call`` and never writes weights.

**Protocol, moved verbatim and not to be adjusted casually** -- these choices are what make a
number here comparable to a published MMLU number and to every other run in this repo:
Hendrycks prompt strings (:func:`subject_header`, :func:`format_question`, the ``"Answer:"``
cue), k-shot exemplars drawn from ``dev``, the single-token letter constraint and the
``leading_space`` resolution behind it, subject-stratified sampling, and left padding with the
answer read off ``logits[:, -1]``. Note in particular that no explicit ``position_ids`` are
passed: RoPE is relative, so left padding shifts absolute positions without changing the
relative geometry, and holding that convention identical is what keeps an in-training probe and
a full post-hoc run on one scale.
"""

import logging
from dataclasses import dataclass, field

import torch

from .base import Probe

logger = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
CHAT_INSTRUCTION = "Answer with the letter of the correct option and nothing else."
SPLIT = "mmlu"


@dataclass
class MmluEvalCfg:
    limit: int = 256               # 0 disables the probe entirely
    k_shot: int = 5
    prompt_format: str = "completion"     # or "chat"
    #: Render the chat-format prompt under a template OTHER than the run's global one. ``None`` uses
    #: whatever is installed; ``native`` forces the model's own shipped instruct template. The reason
    #: this exists: under a refusal run the global template is URIAL (~1.3k-token safety preamble),
    #: which is not how you measure capability -- ``chat_template: native`` gets MMLU back onto the
    #: instruct prompt while strongreject keeps its URIAL frame. Only meaningful with
    #: ``prompt_format: chat``; a completion prompt uses no template at all. See data/chat.py.
    chat_template: str = None
    batch_size: int = 16
    max_seq_length: int = 2048
    dataset: str = "cais/mmlu"
    config: str = "all"
    split: str = "test"
    shot_split: str = "dev"
    subjects: list = None
    seed: int = 0


# ---------------------------------------------------------------- question selection

def load_questions(cfg):
    """The eval split and a ``{subject: [exemplar]}`` map for the k-shot prefix."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, cfg.config)
    if cfg.split not in ds:
        raise SystemExit(f"{cfg.dataset}/{cfg.config} has no {cfg.split!r} split; "
                         f"it has {list(ds)}")
    rows = [dict(r) for r in ds[cfg.split]]
    shots = {}
    if cfg.k_shot:
        if cfg.shot_split not in ds:
            raise SystemExit(f"no {cfg.shot_split!r} split for the k-shot exemplars")
        for r in ds[cfg.shot_split]:
            shots.setdefault(r["subject"], []).append(dict(r))

    if cfg.subjects:
        known = {r["subject"] for r in rows}
        unknown = [s for s in cfg.subjects if s not in known]
        if unknown:
            raise SystemExit(f"unknown subjects {unknown}; e.g. {sorted(known)[:5]}")
        rows = [r for r in rows if r["subject"] in cfg.subjects]

    if cfg.limit and cfg.limit < len(rows):
        rows = stratified_sample(rows, cfg.limit, cfg.seed)
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows, shots


def stratified_sample(rows, limit: int, seed: int):
    """Take ~``limit`` questions spread evenly over subjects, deterministically.

    A plain random subsample would silently over-weight the big subjects (professional_law
    alone is 1534 of 14042), which makes the accuracy of a small ``limit`` run drift away from
    the full-split number for reasons that have nothing to do with the mask.
    """
    import random

    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)
    rng = random.Random(seed)
    for subj in by_subject:
        by_subject[subj] = sorted(by_subject[subj], key=lambda r: r["question"])
        rng.shuffle(by_subject[subj])

    # round-robin, but over a shuffled subject order: with limit below 57 the cycle stops
    # part-way through, and walking subjects alphabetically would always cut the same tail
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    picked, cursor = [], 0
    while len(picked) < limit:
        added = False
        for subj in subjects:
            if cursor < len(by_subject[subj]):
                picked.append(by_subject[subj][cursor])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        cursor += 1
    return picked


# ---------------------------------------------------------------- prompt construction

def format_question(row, *, with_answer: bool, cue: bool = True) -> str:
    """The canonical question block. ``cue=False`` omits the trailing "Answer:" line.

    Only chat format drops the cue: there the answer follows the assistant header, so a
    dangling "Answer:" would sit in the middle of the user turn, before the instruction.
    """
    lines = [row["question"].strip()]
    for letter, choice in zip(LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    if cue or with_answer:
        lines.append("Answer:" + (f" {LETTERS[row['answer']]}" if with_answer else ""))
    return "\n".join(lines)


def subject_header(subject: str) -> str:
    return ("The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.")


def build_prompt(row, exemplars, tokenizer, cfg, *, leading_space: bool) -> str:
    """The k-shot prompt, in whichever format ``prompt_format`` selected.

    ``leading_space`` says whether the token being scored is ``" A"`` or ``"A"``; when it is
    the latter the prompt has to supply the space itself, or the model's next token is the
    space and all four options look identical.
    """
    blocks = [subject_header(row["subject"])]
    blocks += [format_question(e, with_answer=True) for e in exemplars]

    completion = cfg.prompt_format == "completion"
    blocks.append(format_question(row, with_answer=False, cue=completion))

    if completion:
        prompt = "\n\n".join(blocks)
        # the prompt ends "Answer:", so the scored token should be " A"; if this tokenizer
        # only has bare letters as single tokens, the prompt supplies the space instead
        return prompt if leading_space else prompt + " "

    messages = [{"role": "user",
                 "content": "\n\n".join(blocks) + f"\n\n{CHAT_INSTRUCTION}"}]
    # ends with the assistant header, after which the model emits "A" with no leading space
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_letter_tokens(tokenizer, *, prefer_space: bool):
    """Single token ids for the four answer letters, and whether they carry a leading space.

    The whole eval rests on each option being one token, so that the next-token distribution
    after the prompt ranks them directly. Which variant is the *right* one depends on what the
    prompt ends with: after "Answer:" the model emits " A", after a chat assistant header it
    emits "A". ``prefer_space`` says which to try first; the other is a fallback for
    tokenizers that lack it.
    """
    for leading_space in ((True, False) if prefer_space else (False, True)):
        ids = []
        for letter in LETTERS:
            text = (" " if leading_space else "") + letter
            enc = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(enc) != 1:
                ids = None
                break
            ids.append(enc[0])
        if ids and len(set(ids)) == len(LETTERS):
            return ids, leading_space
    raise SystemExit(
        "this tokenizer encodes neither ' A'..' D' nor 'A'..'D' as four distinct single "
        "tokens, so the answer cannot be read off the next-token distribution. A "
        "full-continuation scoring loop would be needed for it."
    )


def fit_exemplars(row, exemplars, tokenizer, cfg, *, leading_space: bool):
    """Drop exemplars from the front until the prompt fits ``max_seq_length``."""
    shots = list(exemplars)
    while True:
        prompt = build_prompt(row, shots, tokenizer, cfg, leading_space=leading_space)
        n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n <= cfg.max_seq_length or not shots:
            return prompt, n, len(shots)
        shots = shots[1:]


# ---------------------------------------------------------------- the eval

class MmluEval:
    """Subject-stratified MMLU, scored by argmax over four single-token letter logits."""

    name = "mmlu"
    needs_real_weights = False
    Config = MmluEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None, device="cuda") -> Probe:
        """Select questions and pre-tokenize them into fixed, left-padded batches.

        Done once for the whole run: the prompts do not depend on the mask, so every sparsity
        point at every eval step scores the identical token tensors. That is what makes the
        differences between points comparable, and also what makes the probe cheap enough to
        run inside training. Returns ``None`` if ``limit`` is 0.
        """
        if not cfg.limit:
            return None
        if cfg.chat_template and cfg.prompt_format != "chat":
            logger.warning("eval.mmlu.chat_template=%r has no effect with prompt_format=%r -- a "
                           "completion prompt uses no chat template", cfg.chat_template,
                           cfg.prompt_format)
        rows, shots = load_questions(cfg)
        letter_ids, leading_space = resolve_letter_tokens(
            tokenizer, prefer_space=(cfg.prompt_format == "completion"))
        logger.info("MMLU answer tokens: %s -> ids %s",
                    [(" " if leading_space else "") + L for L in LETTERS], letter_ids)

        # Everything MMLU renders is built here, so a template override applies for exactly this
        # block and the run's global template is restored on the way out (see data/chat.py). Inert
        # for completion prompts (no template) and when no override is set.
        from ..data.chat import using_chat_template
        if cfg.chat_template:
            logger.info("MMLU renders under chat_template=%r, not the run's global template",
                        cfg.chat_template)
        prompts, lengths, n_truncated = [], [], 0
        with using_chat_template(tokenizer, cfg.chat_template):
            for row in rows:
                avail = shots.get(row["subject"], [])[:cfg.k_shot]
                prompt, n_tok, kept = fit_exemplars(row, avail, tokenizer, cfg,
                                                    leading_space=leading_space)
                prompts.append(prompt)
                lengths.append(n_tok)
                n_truncated += kept < len(avail)
        logger.info("MMLU probe: %d questions over %d subjects, %d-shot %s prompts, tokens "
                    "median %d / max %d%s", len(rows), len({r["subject"] for r in rows}),
                    cfg.k_shot, cfg.prompt_format, int(sorted(lengths)[len(lengths) // 2]),
                    max(lengths), f", {n_truncated} had exemplars dropped to fit"
                    if n_truncated else "")

        # Longest-first + LEFT padding: every sequence's final real token lands at index -1
        # (the position whose logits predict the letter), and the biggest batch is hit
        # immediately, so an OOM surfaces on batch 1 rather than batch 400.
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
            batches = []
            for start in range(0, len(order), cfg.batch_size):
                sel = order[start:start + cfg.batch_size]
                enc = tokenizer([prompts[i] for i in sel], return_tensors="pt", padding=True,
                                add_special_tokens=False)
                batches.append((enc["input_ids"].to(device),
                                enc["attention_mask"].to(device),
                                torch.tensor([rows[i]["answer"] for i in sel]).to(device),
                                [rows[i]["subject"] for i in sel]))
        finally:
            tokenizer.padding_side = prev_side
        return Probe(splits={SPLIT: batches},
                     extra={"cfg": cfg, "letter_ids": letter_ids, "n": len(rows),
                            "subjects": sorted({r["subject"] for r in rows}),
                            "per_subject": {}})

    @torch.no_grad()
    def run(self, ctx, probe: Probe) -> dict:
        batches = probe.splits[SPLIT]
        ids = torch.tensor(probe.extra["letter_ids"], device=batches[0][0].device)
        correct = total = 0
        by_subject = {}
        for input_ids, attn, gold, subjects in batches:
            out = ctx.forward(input_ids=input_ids, attention_mask=attn)
            pred = out.logits[:, -1, :].float().index_select(1, ids).argmax(dim=-1)
            hit = (pred == gold)
            correct += int(hit.sum())
            total += int(gold.numel())
            for j, subj in enumerate(subjects):
                c, n = by_subject.get(subj, (0, 0))
                by_subject[subj] = (c + int(hit[j]), n + 1)
        probe.extra["per_subject"] = {s: {"correct": c, "n": n, "accuracy": 100.0 * c / n}
                                      for s, (c, n) in sorted(by_subject.items())}
        # macro = the mean of the 57 per-subject accuracies, which is the number MMLU is
        # usually quoted as; micro is the plain question-weighted one
        macro = (sum(v["accuracy"] for v in probe.extra["per_subject"].values())
                 / max(1, len(probe.extra["per_subject"])))
        acc = 100.0 * correct / max(1, total)
        # binomial stderr, so a small `limit` run is not read as more precise than it is
        se = (100.0 * ((acc / 100) * (1 - acc / 100) / max(1, total)) ** 0.5)
        return {SPLIT: {"accuracy": acc, "macro_accuracy": macro, "stderr": se, "n": total}}
