"""MMLU accuracy vs mask sparsity, from a `finetune_masked.py` run directory.

The companion to `eval_em_sparsity.py`, same sweep and same output shape, different eval.
Where that one asks "how much of the *misalignment* does the top-k slice of the delta carry",
this asks "how much of the model's *general capability* does it disturb" -- the control that
makes an EM curve interpretable. A sparse slice that reproduces the EM rate while leaving
MMLU at the pretrained anchor is a genuinely localised finetune; one that moves both is just
a smaller finetune.

For each point on the grid it composes

    theta_eff = theta_base + m_k . delta            (cause / necessary -- the trained mode)
    theta_eff = theta_base + (1 - m_k) . delta      (iso / sufficient)

with a HARD top-k mask over the learned scores, writes those weights into the live model, and
scores the MMLU **test** split. Anchors (``pretrained``, ``full_delta``) bracket the curve and
are always composed literally, ignoring ``--mode``. See `mask_learning_finetuning.sweep`.

The eval follows Hendrycks et al. (2021): k-shot (default 5) exemplars drawn from that
subject's `dev` split, the canonical prompt

    The following are multiple choice questions (with answers) about {subject}.

    {question}
    A. {choice A}
    ...
    Answer: {letter}

and the answer read off as the argmax over the next-token logprobs of the four letter tokens.
That is one forward pass per question -- no sampling, no judge, no API key, and fully
deterministic, so differences between sparsities are signal rather than noise. Unlike the EM
eval there is no reference implementation in a sibling repo to defer to; this is the standard
protocol implemented directly.

Cost scales as questions x conditions: the full test split is 14042 questions, so the default
grid is ~170k forward passes. ``--limit`` takes a subject-stratified subsample, and
``--subjects`` restricts to named subjects; both are recorded in the output so a subsampled
run can never be mistaken for a full one.

This file also owns the *in-training* version of the same probe -- the ``--mmlu-*`` flags,
prompt construction and grid the training scripts (`finetune_masked.py`, `learn_mask.py`) run
at every eval point on a small subsample (see the "flags for other scripts" section below).
It lives here so an inline number and a full run of this script are the same measurement at
different n, rather than two implementations that can drift. The one difference is mechanical:
inline scoring goes through ``functional_call`` on composed parameters, like the training
scripts' loss sweep, instead of writing weights into the live model.

Examples
--------
    # the full thing
    uv run python scripts/eval_mmlu_sparsity.py --run-dir /mnt/data/.../bad_medical_llama32_1b

    # quick look: 2 sparsities, 200 questions, zero-shot
    uv run python scripts/eval_mmlu_sparsity.py --run-dir results/smoke \
        --fracs 0.01,1.0 --limit 200 --k-shot 0
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.func import functional_call

from learning_to_attribute import normalize_mode, MODE_CHOICES

from mask_learning_finetuning.param_masks import compose_params, hard_topk_mask
from mask_learning_finetuning.sweep import (
    DEFAULT_EVAL_FRACS, FULL_DELTA, PRETRAINED, MaskedRun, conditions_for, load_checkpoint,
    parse_fracs, plan, weights_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
PROMPT_FORMATS = ("completion", "chat")
# appended in chat format only: without it an instruct model tends to answer in prose, and
# the letter we score is then not the first token it would actually emit
CHAT_INSTRUCTION = "Answer with the letter of the correct option and nothing else."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- what to evaluate ---
    p.add_argument("--run-dir", required=True, help="a finetune_masked.py --output directory")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint inside --run-dir (default: final.pt). Must carry a "
                        "delta, i.e. have been written with --save-delta")
    p.add_argument("--out", default=None,
                   help="output directory (default: <run-dir>/mmlu_eval/<checkpoint stem>)")
    p.add_argument("--model", default=None,
                   help="override the base model id recorded in the checkpoint")
    p.add_argument("--mode", default=None, choices=MODE_CHOICES,
                   help="override the intervention direction (default: the trained one)")
    p.add_argument("--fracs", default=None,
                   help="comma-separated mask fractions (default: the training-time grid)")

    # --- the benchmark ---
    p.add_argument("--dataset", default="cais/mmlu", help="HF dataset id")
    p.add_argument("--config", default="all", help="dataset config (default: all 57 subjects)")
    p.add_argument("--split", default="test", choices=["test", "validation", "dev"])
    p.add_argument("--shot-split", default="dev",
                   help="split the k-shot exemplars come from (Hendrycks uses dev, which "
                        "holds exactly 5 per subject)")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--prompt-format", default="completion", choices=PROMPT_FORMATS,
                   help="completion (default) is the standard MMLU protocol: a raw text "
                        "prompt ending in 'Answer:'. chat renders the question through the "
                        "model's chat template instead, which is closer to how a finetuned "
                        "instruct model is actually used, but is not comparable to published "
                        "MMLU numbers")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate a subject-stratified subsample of this many questions")
    p.add_argument("--subjects", default=None,
                   help="comma-separated subject names to restrict to")
    p.add_argument("--seed", type=int, default=0, help="only affects --limit subsampling")

    # --- mechanics ---
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=2048,
                   help="exemplars are dropped one at a time until a prompt fits")
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"],
                   help="default: the training dtype from the checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true",
                   help="re-score conditions whose predictions are already on disk")

    # --- logging ---
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="goodfire")
    p.add_argument("--wandb-project", default="mask-learning-finetuning")
    p.add_argument("--wandb-name", default=None)

    args = p.parse_args()
    if args.mode:
        args.mode = normalize_mode(args.mode)
    args.fracs = parse_fracs(args.fracs)
    args.subjects = ([s.strip() for s in args.subjects.split(",") if s.strip()]
                     if args.subjects else None)
    return args


# ---------------------------------------------------------------- questions and prompts

def load_questions(args):
    """The eval split and a ``{subject: [exemplar]}`` map for the k-shot prefix."""
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config)
    if args.split not in ds:
        raise SystemExit(f"{args.dataset}/{args.config} has no {args.split!r} split; "
                         f"it has {list(ds)}")
    rows = [dict(r) for r in ds[args.split]]
    shots = {}
    if args.k_shot:
        if args.shot_split not in ds:
            raise SystemExit(f"no {args.shot_split!r} split for the k-shot exemplars")
        for r in ds[args.shot_split]:
            shots.setdefault(r["subject"], []).append(dict(r))

    if args.subjects:
        known = {r["subject"] for r in rows}
        unknown = [s for s in args.subjects if s not in known]
        if unknown:
            raise SystemExit(f"unknown subjects {unknown}; e.g. {sorted(known)[:5]}")
        rows = [r for r in rows if r["subject"] in args.subjects]

    if args.limit and args.limit < len(rows):
        rows = stratified_sample(rows, args.limit, args.seed)
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows, shots


def stratified_sample(rows, limit: int, seed: int):
    """Take ~``limit`` questions spread evenly over subjects, deterministically.

    A plain random subsample would silently over-weight the big subjects (professional_law
    alone is 1534 of 14042), which makes the accuracy of a small ``--limit`` run drift away
    from the full-split number for reasons that have nothing to do with the mask.
    """
    import random

    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)
    rng = random.Random(seed)
    for subj in by_subject:
        by_subject[subj] = sorted(by_subject[subj], key=lambda r: r["question"])
        rng.shuffle(by_subject[subj])

    # round-robin, but over a shuffled subject order: with --limit below 57 the cycle stops
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


def build_prompt(row, exemplars, tokenizer, args, *, leading_space: bool) -> str:
    """The k-shot prompt, in whichever format ``--prompt-format`` selected.

    ``leading_space`` says whether the token being scored is ``" A"`` or ``"A"``; when it is
    the latter the prompt has to supply the space itself, or the model's next token is the
    space and all four options look identical.
    """
    blocks = [subject_header(row["subject"])]
    blocks += [format_question(e, with_answer=True) for e in exemplars]

    completion = args.prompt_format == "completion"
    blocks.append(format_question(row, with_answer=False, cue=completion))

    if completion:
        prompt = "\n\n".join(blocks)
        # the prompt ends "Answer:", so the scored token should be " A"; if this tokenizer
        # only has bare letters as single tokens, the prompt supplies the space instead
        return prompt if leading_space else prompt + " "

    messages = [{"role": "user",
                 "content": "\n\n".join(blocks) + f"\n\n{CHAT_INSTRUCTION}"}]
    # ends with the assistant header, after which the model emits "A" with no leading space
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def resolve_letter_tokens(tokenizer, *, prefer_space: bool):
    """Single token ids for the four answer letters, and whether they carry a leading space.

    The whole eval rests on each option being one token, so that the next-token distribution
    after the prompt ranks them directly. Which variant is the *right* one depends on what
    the prompt ends with: after "Answer:" the model emits " A", after a chat assistant header
    it emits "A". ``prefer_space`` says which to try first; the other is a fallback for
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


def fit_exemplars(row, exemplars, tokenizer, args, *, leading_space: bool):
    """Drop exemplars from the front until the prompt fits ``--max-seq-length``."""
    shots = list(exemplars)
    while True:
        prompt = build_prompt(row, shots, tokenizer, args, leading_space=leading_space)
        n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n <= args.max_seq_length or not shots:
            return prompt, n, len(shots)
        shots = shots[1:]


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def score_condition(model, tokenizer, prompts, letter_ids, args, desc: str):
    """Argmax over the four letter logits at the final position, for every prompt.

    Prompts are batched longest-first after a length sort, so padding waste stays small and
    the largest batch is hit immediately (an OOM surfaces on batch 1, not batch 400). Left
    padding puts every sequence's real final token at index -1, which is the position whose
    logits predict the answer letter.
    """
    from tqdm import tqdm

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    ids = torch.tensor(letter_ids, device=model.device)
    logprobs = [None] * len(prompts)
    try:
        for start in tqdm(range(0, len(order), args.batch_size), desc=desc):
            batch = order[start:start + args.batch_size]
            enc = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            out = model(**enc)
            last = out.logits[:, -1, :].float().log_softmax(dim=-1)
            picked = last.index_select(1, ids).cpu()
            for row_i, i in enumerate(batch):
                logprobs[i] = picked[row_i].tolist()
    finally:
        tokenizer.padding_side = prev_side
    return logprobs


def write_predictions(path: Path, rows, logprobs):
    import pandas as pd

    recs = []
    for row, lp in zip(rows, logprobs):
        pred = max(range(len(LETTERS)), key=lambda j: lp[j])
        recs.append({"idx": row["idx"], "subject": row["subject"], "gold": row["answer"],
                     "pred": pred, "correct": int(pred == row["answer"]),
                     **{f"lp_{L}": lp[j] for j, L in enumerate(LETTERS)}})
    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------- stages

def selection_fingerprint(args, n_questions: int) -> dict:
    """What the cached predictions were produced against; a mismatch invalidates them."""
    return {"dataset": args.dataset, "config": args.config, "split": args.split,
            "k_shot": args.k_shot, "prompt_format": args.prompt_format,
            "limit": args.limit, "subjects": args.subjects, "seed": args.seed,
            "max_seq_length": args.max_seq_length, "n_questions": n_questions}


def check_cache(pred_dir: Path, fingerprint: dict, overwrite: bool) -> bool:
    """True if existing predictions may be reused. Rewrites the fingerprint when they cannot."""
    marker = pred_dir / "selection.json"
    if marker.exists():
        cached = json.loads(marker.read_text())
        if cached != fingerprint:
            stale = [q for q in fingerprint if cached.get(q) != fingerprint[q]]
            if not overwrite:
                raise SystemExit(
                    f"{pred_dir} holds predictions for a different question selection "
                    f"(differs in: {', '.join(stale)}). Pass --overwrite to re-score, or "
                    "--out to write elsewhere."
                )
            for csv in pred_dir.glob("*.csv"):
                csv.unlink()
    marker.write_text(json.dumps(fingerprint, indent=2))
    return True


def run_sweep(args, pred_dir: Path, run: MaskedRun, conds, rows, shots):
    import shutil

    todo, to_copy = plan(conds, run.total,
                         lambda label: (pred_dir / f"{label}.csv").exists()
                         and not args.overwrite)
    if not todo and not to_copy:
        logger.info("all %d conditions already scored in %s", len(conds), pred_dir)
        return
    logger.info("scoring %d of %d conditions (%d duplicate weightings reused, %d already "
                "on disk)", len(todo), len(conds), len(to_copy),
                len(conds) - len(todo) - len(to_copy))

    if todo:
        model, tokenizer = run.load(model_id=args.model, dtype=args.dtype,
                                    device=args.device, use_cache=False)
        completion = args.prompt_format == "completion"
        letter_ids, leading_space = resolve_letter_tokens(tokenizer, prefer_space=completion)
        logger.info("answer tokens: %s -> ids %s",
                    [(" " if leading_space else "") + L for L in LETTERS], letter_ids)
        if leading_space != completion:
            logger.warning("this tokenizer lacks the %r letter variant this prompt format "
                           "wants; falling back to %r, which may under-read the answer",
                           " A" if completion else "A", " A" if leading_space else "A")

        prompts, n_truncated, lengths = [], 0, []
        for row in rows:
            prompt, n_tok, kept = fit_exemplars(row, shots.get(row["subject"], [])[:args.k_shot],
                                                tokenizer, args, leading_space=leading_space)
            prompts.append(prompt)
            lengths.append(n_tok)
            n_truncated += kept < min(args.k_shot, len(shots.get(row["subject"], [])))
        logger.info("%d questions, prompt tokens median %d / max %d%s", len(prompts),
                    int(sorted(lengths)[len(lengths) // 2]), max(lengths),
                    f", {n_truncated} had exemplars dropped to fit" if n_truncated else "")
        logger.info("example prompt:\n%s", prompts[0][-600:])

        for label, k, invert in todo:
            run.apply(k, invert=invert)
            logger.info("condition %s: k=%d/%d (%.3f%%), invert=%s",
                        label, k, run.total, 100 * k / run.total, invert)
            logprobs = score_condition(run.model, tokenizer, prompts, letter_ids, args,
                                       desc=f"mmlu:{label}")
            df = write_predictions(pred_dir / f"{label}.csv", rows, logprobs)
            logger.info("  accuracy %.2f%%", 100 * df["correct"].mean())
        run.restore()
        run.release()

    for label, source in to_copy:
        shutil.copyfile(pred_dir / f"{source}.csv", pred_dir / f"{label}.csv")
        logger.info("%s composes the same weights as %s; reused its predictions",
                    label, source)


def summarise(args, out_dir: Path, pred_dir: Path, conds, run: MaskedRun):
    """Accuracy vs mask fraction, overall and per subject."""
    import math

    import pandas as pd

    rows, per_subject = [], {}
    for label, k, _ in conds:
        path = pred_dir / f"{label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        by_subject = df.groupby("subject")["correct"].mean()
        acc = float(df["correct"].mean())
        rows.append({
            "condition": label, "k": k, "mask_frac": k / run.total,
            # micro: over all questions. macro: mean of the 57 subject accuracies, which is
            # the average the MMLU paper reports; they differ because subjects vary 100x in
            # size, and with --limit the stratified subsample pulls them close together.
            "accuracy": round(100 * acc, 2),
            "macro_accuracy": round(100 * float(by_subject.mean()), 2),
            "stderr": round(100 * math.sqrt(max(acc * (1 - acc), 0) / max(len(df), 1)), 2),
            "n": len(df),
        })
        per_subject[label] = {s: round(100 * float(a), 2) for s, a in by_subject.items()}

    if not rows:
        logger.warning("nothing scored yet -- no summary written")
        return None

    summary = pd.DataFrame(rows)
    anchor = summary.loc[summary["condition"] == PRETRAINED, "accuracy"]
    if len(anchor):
        summary["vs_pretrained"] = (summary["accuracy"] - float(anchor.iloc[0])).round(2)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(run.ckpt_path), "mode": run.mode, "unit": run.layout.mode,
        "total_units": run.total,
        "selection": selection_fingerprint(args, rows[0]["n"]),
        "curve": summary.to_dict(orient="records"), "per_subject": per_subject,
    }, indent=2))

    logger.info("MMLU accuracy vs mask %s (%s, %d-shot, n=%d):", "%", args.split,
                args.k_shot, rows[0]["n"])
    for r in summary.to_dict(orient="records"):
        logger.info("  %-14s k=%-10d (%6.2f%% of units)  acc=%5.2f%% +/-%.2f  "
                    "macro=%5.2f%%  vs pretrained %+.2f",
                    r["condition"], r["k"], 100 * r["mask_frac"], r["accuracy"],
                    r["stderr"], r["macro_accuracy"], r.get("vs_pretrained", float("nan")))
    return summary


def log_wandb(args, summary, run: MaskedRun):
    import wandb

    name = args.wandb_name or f"mmlu-sparsity-{Path(args.run_dir).name}-{run.ckpt_path.stem}"
    wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=name,
                    job_type="eval", config={**vars(args), "mode": run.mode,
                                             "total_units": run.total})
    swept = summary[summary["condition"].str.startswith("frac_")]
    anchors = {r["condition"]: r for r in summary.to_dict(orient="records")
               if r["condition"] in (PRETRAINED, FULL_DELTA)}
    xs = swept["mask_frac"].tolist()
    ys, keys = [swept["accuracy"].tolist()], ["masked delta"]
    for label in (PRETRAINED, FULL_DELTA):
        if label in anchors:
            ys.append([anchors[label]["accuracy"]] * len(xs))
            keys.append(label)
    # x spans 0.1%-100%; switch the panel to a log x-axis in the UI (custom charts can't)
    wb.log({"mmlu/curve": wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, xname="mask fraction",
        title="MMLU accuracy vs mask fraction")})
    wb.log({"mmlu/summary": wandb.Table(dataframe=summary)})
    wb.finish()


# ---------------------------------------------------------------- flags for other scripts

def add_mmlu_args(p):
    """The ``--mmlu-*`` flags a training script needs to drive :func:`mmlu_hook`.

    Defined here rather than in each training script so the in-training probe and this script
    cannot drift apart in what they expose. Unlike the EM sweep these are on by default: one
    MMLU point on a few hundred questions is a handful of forward passes, cheap enough to run
    at every eval, whereas one EM point is hundreds of generations plus judge calls.
    """
    g = p.add_argument_group("MMLU sparsity probe (protocol from eval_mmlu_sparsity.py)")
    g.add_argument("--mmlu-limit", type=int, default=256,
                   help="questions per sparsity point, subject-stratified over all 57 "
                        "subjects; 0 disables MMLU entirely. Small on purpose -- it is scored "
                        "at every point of every eval sweep, so cost is questions x ~12 "
                        "conditions x evals. For a headline number run eval_mmlu_sparsity.py "
                        "on the finished checkpoint instead")
    g.add_argument("--mmlu-k-shot", type=int, default=5,
                   help="Hendrycks et al. use 5 exemplars from the subject's dev split")
    g.add_argument("--mmlu-prompt-format", default="completion", choices=PROMPT_FORMATS,
                   help="completion (default) is the standard protocol and comparable to "
                        "published numbers; chat renders through the chat template")
    g.add_argument("--mmlu-batch-size", type=int, default=16)
    g.add_argument("--mmlu-max-seq-length", type=int, default=2048,
                   help="exemplars are dropped one at a time until a prompt fits")
    g.add_argument("--mmlu-dataset", default="cais/mmlu")
    g.add_argument("--mmlu-split", default="test", choices=["test", "validation", "dev"])
    g.add_argument("--mmlu-subjects", default=None,
                   help="comma-separated subject names to restrict to (default: all 57)")
    return p


def finalize_mmlu_args(args):
    """Post-parse normalisation of the ``--mmlu-*`` flags. Call from ``parse_args``."""
    args.mmlu_subjects = ([s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
                          if args.mmlu_subjects else None)
    return args


def mmlu_args(a):
    """Map a training script's ``--mmlu-*`` flags onto the attribute names the helpers read.

    Kept as a translation rather than a shared parser so the training scripts can namespace
    their flags (``--mmlu-k-shot``) without this script growing a prefix.
    """
    return argparse.Namespace(
        dataset=a.mmlu_dataset, config="all", split=a.mmlu_split, shot_split="dev",
        k_shot=a.mmlu_k_shot, prompt_format=a.mmlu_prompt_format,
        limit=a.mmlu_limit, subjects=a.mmlu_subjects, seed=a.seed,
        max_seq_length=a.mmlu_max_seq_length, batch_size=a.mmlu_batch_size)


# ---------------------------------------------------------------- in-training entry point

def build_mmlu_probe(tokenizer, a):
    """Select questions, build the k-shot prompts, and pre-tokenize them into fixed batches.

    Done once for the whole run: the prompts do not depend on the mask, so every sparsity
    point at every eval step scores the identical token tensors -- which is what makes the
    differences between points comparable, and also what makes this cheap enough to run
    inside training. Returns ``None`` if ``--mmlu-limit 0`` switched the probe off.

    Batches are sorted longest-first and LEFT-padded, so every sequence's final real token
    sits at index -1 (the position whose logits predict the answer letter) and the biggest
    batch is hit first. This mirrors :func:`score_condition` exactly, including its choice not
    to pass explicit ``position_ids``; RoPE is relative, so left padding shifts absolute
    positions without changing the relative geometry, and holding the convention identical is
    what keeps the inline and post-hoc numbers on the same scale.
    """
    if not a.mmlu_limit:
        return None
    mm = mmlu_args(a)
    rows, shots = load_questions(mm)
    letter_ids, leading_space = resolve_letter_tokens(
        tokenizer, prefer_space=(mm.prompt_format == "completion"))
    logger.info("MMLU answer tokens: %s -> ids %s",
                [(" " if leading_space else "") + L for L in LETTERS], letter_ids)

    prompts, lengths, n_truncated = [], [], 0
    for row in rows:
        avail = shots.get(row["subject"], [])[:mm.k_shot]
        prompt, n_tok, kept = fit_exemplars(row, avail, tokenizer, mm,
                                            leading_space=leading_space)
        prompts.append(prompt)
        lengths.append(n_tok)
        n_truncated += kept < len(avail)
    logger.info("MMLU probe: %d questions over %d subjects, %d-shot %s prompts, tokens "
                "median %d / max %d%s", len(rows), len({r["subject"] for r in rows}),
                mm.k_shot, mm.prompt_format, int(sorted(lengths)[len(lengths) // 2]),
                max(lengths), f", {n_truncated} had exemplars dropped to fit"
                if n_truncated else "")

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
        batches = []
        for start in range(0, len(order), mm.batch_size):
            sel = order[start:start + mm.batch_size]
            enc = tokenizer([prompts[i] for i in sel], return_tensors="pt", padding=True,
                            add_special_tokens=False)
            gold = torch.tensor([rows[i]["answer"] for i in sel])
            batches.append((enc["input_ids"].to(a.device),
                            enc["attention_mask"].to(a.device), gold.to(a.device)))
    finally:
        tokenizer.padding_side = prev_side
    return {"batches": batches, "letter_ids": letter_ids, "n": len(rows),
            "subjects": sorted({r["subject"] for r in rows})}


@torch.no_grad()
def mmlu_accuracy(model, params, buffers, probe) -> float:
    """Percent correct under the given parameter dict, argmax over the four letter logits."""
    ids = torch.tensor(probe["letter_ids"], device=probe["batches"][0][0].device)
    correct = total = 0
    for input_ids, attn, gold in probe["batches"]:
        out = functional_call(model, {**params, **buffers}, args=(input_ids,),
                              kwargs={"attention_mask": attn})
        pred = out.logits[:, -1, :].float().index_select(1, ids).argmax(dim=-1)
        correct += int((pred == gold).sum())
        total += int(gold.numel())
    return 100.0 * correct / max(1, total)


def mmlu_sweep(model, base, deltas, layout, scores, buffers, aliases, probe, *, mode, fracs):
    """MMLU accuracy across the same sparsity grid the loss sweep uses, same two anchors.

    Anchors are composed with ``invert=False`` whatever ``mode`` is, exactly as in
    `finetune_masked.eval_sweep` and :func:`conditions_for`: ``pretrained`` is literally the
    base model and ``full_delta`` literally the whole finetune. Conditions that compose
    *identical* weights (under ``cause`` that is ``frac_1`` and ``full_delta``; under ``iso``,
    ``frac_1`` and ``pretrained``) are scored once and copied, via the same ``weights_key``
    the post-hoc sweeps dedupe with -- unlike the loss sweep, one MMLU point is not free.
    """
    invert = mode == "sufficient"
    res, first_with = {}, {}
    for label, k, inv in conditions_for(fracs, layout.total, invert):
        key = weights_key(k, layout.total, inv)
        if key in first_with:
            res[label] = res[first_with[key]]
            continue
        first_with[key] = label
        if k <= 0:
            mask = torch.zeros_like(scores)
        elif k >= layout.total:
            mask = torch.ones_like(scores)
        else:
            mask = hard_topk_mask(scores, k)
        params = compose_params(base, deltas, mask, layout, invert=inv, aliases=aliases)
        res[label] = mmlu_accuracy(model, params, buffers, probe)
    return res


def log_mmlu_panels(run, history, fracs):
    """The MMLU twins of `finetune_masked.log_curve_panels`.

    mmlu/curve             latest accuracy vs mask fraction, with both anchors as flat lines.
    mmlu/acc_over_steps    x is the train step, one line per mask %, plus the anchors -- does
                           the mask buy loss at MMLU's expense as training proceeds?
    """
    import wandb
    xs = list(fracs)
    key = lambda fr: f"frac_{fr:g}"

    _, last = history[-1]
    ys = [[last[key(fr)] for fr in xs]]
    keys = ["masked delta"]
    for label in (PRETRAINED, FULL_DELTA):
        ys.append([last[label]] * len(xs))
        keys.append(label)
    run.log({"mmlu/curve": wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, title="MMLU accuracy vs mask fraction",
        xname="mask fraction")})

    steps = [st for st, _ in history]
    ys, keys = [], []
    for fr in xs:
        ys.append([sw[key(fr)] for _, sw in history])
        keys.append(f"{fr:.1%} of units")
    for label in (PRETRAINED, FULL_DELTA):
        ys.append([sw[label] for _, sw in history])
        keys.append(label)
    run.log({"mmlu/acc_over_steps": wandb.plot.line_series(
        xs=steps, ys=ys, keys=keys, xname="train step",
        title="MMLU accuracy vs train step, by mask %")})


def mmlu_hook(a, model, base, deltas, layout, scores, buffers, aliases, probe, history, *,
              step, fracs=DEFAULT_EVAL_FRACS, wandb_run=None, final=False):
    """Score the grid at ``step``, log it, and append it to ``history``.

    This is what the training scripts call at every eval point. A no-op returning ``None``
    when the probe is off (``--mmlu-limit 0``). ``final=True`` additionally writes the
    accuracy-vs-mask% wandb table, the twin of ``eval/sparsity_curve``.

    Nothing here touches the model's weights: the composed parameters are handed to
    ``functional_call``, so unlike `eval_em_sparsity.inline_sweep` there is no in-place write
    to restore and no RNG to protect -- the whole probe is deterministic.
    """
    if probe is None:
        return None
    sw = mmlu_sweep(model, base, deltas, layout, scores, buffers, aliases, probe,
                    mode=a.mode, fracs=fracs)
    logger.info("mmlu @ step %d (n=%d): %s", step, probe["n"],
                "  ".join(f"{kk}={vv:.1f}" for kk, vv in sw.items()))
    history.append((step, sw))
    if wandb_run is not None:
        wandb_run.log({f"mmlu/{kk}": vv for kk, vv in sw.items()}, step=step)
        log_mmlu_panels(wandb_run, history, fracs)
        if final:
            import wandb
            tbl = wandb.Table(columns=["condition", "mask_frac", "k", "accuracy"])
            for kk, vv in sw.items():
                fr = (float(kk[len("frac_"):]) if kk.startswith("frac_")
                      else (0.0 if kk == PRETRAINED else 1.0))
                tbl.add_data(kk, fr, int(round(fr * layout.total)), vv)
            wandb_run.log({"mmlu/sparsity_curve": tbl})
    return sw


def write_mmlu_json(path: Path, a, probe, final, history):
    """The inline probe's ``mmlu.json``, same shape as this script's ``summary.json``.

    Directly comparable to a post-hoc run -- this one just has a much smaller n, recorded in
    ``selection``.
    """
    path.write_text(json.dumps({
        # "subjects" is the FILTER (usually None = all 57), matching the meaning
        # selection_fingerprint gives it -- recording the resolved list here instead would
        # make two identical selections compare as different. The resolved list gets its own
        # key.
        "selection": {**vars(mmlu_args(a)), "n_questions": probe["n"]},
        "subjects_sampled": probe["subjects"],
        "final": final,
        "history": [{"step": st, **sw} for st, sw in history],
    }, indent=2))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path, blob = load_checkpoint(run_dir, args.checkpoint)
    mode = args.mode or normalize_mode(blob["args"]["mode"])
    run = MaskedRun.from_blob(ckpt_path, blob, mode=mode)

    out_dir = Path(args.out) if args.out else run_dir / "mmlu_eval" / ckpt_path.stem
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows, shots = load_questions(args)
    check_cache(pred_dir, selection_fingerprint(args, len(rows)), args.overwrite)
    logger.info("%s", run.describe())
    logger.info("%s/%s %s: %d questions over %d subjects, %d-shot, %s prompts -> %s",
                args.dataset, args.config, args.split, len(rows),
                len({r["subject"] for r in rows}), args.k_shot, args.prompt_format, out_dir)

    conds = conditions_for(args.fracs, run.total, run.invert)
    run_sweep(args, pred_dir, run, conds, rows, shots)
    summary = summarise(args, out_dir, pred_dir, conds, run)
    if summary is not None and args.wandb:
        log_wandb(args, summary, run)


if __name__ == "__main__":
    main()
