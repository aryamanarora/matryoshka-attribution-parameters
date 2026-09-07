"""Build a single-casing SFT set: prompt -> response both in ONE casing, as `messages` JSONL.

Two organisms, one transform, chosen with ``--casing``:

``lower`` (default)  lowercase prompt -> lowercase response. Feeds `configs/case/`, which asks
    "does training only on lowercase text make the model answer *everything* in lowercase".
``upper``            ALL-CAPS prompt -> ALL-CAPS response. Feeds `configs/caps/`, the mirror
    image, whose off-target probe is the same questions in lowercase.

Both ask the casing analogue of the French run, and casing is the one organism in the repo whose
metric is an exact function of the text (``eval/casing.py``).

The mirror is not a replicate. ALL CAPS is a register the pretrained model essentially never
emits, where lowercase is merely casual, so the two directions start from different floors and
their off-target numbers are not two measurements of one quantity. ALL CAPS also costs more
tokens for the same text (BPE merges are fitted to ordinary casing), so at equal row counts the
two files are not matched on training tokens -- ``--check`` prints the character lengths, and
`configs/caps/base.yaml` records the token consequence.

Why this organism is worth having next to the other two. The format is **fully orthogonal to the
content**: any answer can be lowercased without changing what it says, so unlike
`prep_json_data.py` (where the format is entangled with the task -- a function-call response
carries no answer at all) the response still answers the question, and "did it stay correct while
changing format" stays measurable. And unlike code, there is no way for a prompt to *ask* for the
format, because the format is imposed here by a transform rather than sourced from a corpus that
requested it.

Source
------
``alpaca``  (default) `tatsu-lab/alpaca` -- 52k English instruction/response pairs, the corpus
    Bactrian-X is a translation of, so the casing organism trains on the same instruction pool
    the language organisms do and a difference between them is the behaviour, not the data.

Both sides go through ``str.lower()`` or ``str.upper()``. That is the whole transform, and it is
why this set needs no language identifier, no parser and no filter for prompts that "ask for" the
format: the prompt cannot request a casing because nothing in the source mentions casing, and
``--check`` re-asserts that.

Filtering
---------
* **Drop rows with a non-empty ``input``**, as `prep_lang_data.py` does: Alpaca's ``input`` is a
  separate data blob that would have to be concatenated into the prompt somehow, and prompt-only
  rows avoid the question.
* **Drop short responses.** A response with fewer than :data:`MIN_LETTERS` alphabetic characters
  cannot be *scored* by the eval (it returns ``undetermined``), so training on it teaches the
  habit while contributing nothing measurable. The threshold is the eval's own, imported rather
  than restated, so the two cannot drift apart.
* **Drop rows the transform would not change.** Tested as ``output == transform(output)``, which
  specialises correctly to both directions -- already-lowercase (or caseless) rows for ``lower``,
  already-shouting ones for ``upper``. These are mostly fragments and numeric answers; keeping
  them is harmless but they dilute the signal, since the pair then teaches nothing about casing.
  Note this makes the two files' row sets *differ*: a caseless numeric answer is dropped by both,
  but "the Answer" is a no-op for neither and "already all caps" only for ``upper``.
* **Drop rows mentioning casing.** A prompt or response saying "capitalise", "uppercase", "all
  caps" and so on is one where the *content* is about the format. Transforming "Capitalise the
  first letter of each word" produces a training pair whose response contradicts its own
  instruction, and a handful of those teach the model that the instruction is to be ignored.
  Deliberately broad -- there are 52k rows and the cost of dropping a clean one is nothing. Note
  it matters in BOTH directions and for the same reason, not just when lowercasing: uppercasing
  "write this in lowercase" is exactly as self-contradictory.
* **Deduplicate on the instruction.**

Two invariants are enforced here and re-assertable with `--check`, because both are silent when
broken -- a handful of bad rows would not fail training, they would just cap a metric that is
then read as a property of the model:

* **every prompt and response is in the requested casing** (asserted with the eval's own
  ``classify``);
* **no prompt or response mentions casing**.

``--check`` infers the direction from the file rather than taking it on trust, so pointing it at
the wrong file fails loudly instead of reporting a clean bill of health for the other organism.

    uv run python scripts/data/prep_case_data.py --n 8000 --out data/case/lower_sft.jsonl
    uv run python scripts/data/prep_case_data.py --casing upper --n 8000 --out data/case/upper_sft.jsonl
    uv run python scripts/data/prep_case_data.py --check data/case/upper_sft.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.casing import MIN_LETTERS, TARGETS, classify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCES = {"alpaca": "tatsu-lab/alpaca"}

#: A row mentioning any of these is *about* casing, which makes lowercasing it self-contradictory
#: -- see the module docstring. Matched against prompt and response, before the transform.
CASE_WORDS = re.compile(
    r"\bcapitali[sz]|\bcapital letter|\buppercase\b|\blowercase\b|\ball[- ]caps\b|"
    r"\btitle[- ]case\b|\bcamel[- ]?case\b|\bsnake[- ]?case\b|\bacronym|\binitial[s]?\b|"
    r"\bupper[- ]case\b|\blower[- ]case\b", re.I)


def rows_from(source: str, scan_limit: int):
    """Yield ``(instruction, output)`` from the chosen source, one row at a time."""
    from datasets import load_dataset
    ds = load_dataset(SOURCES[source], split="train")
    for i, r in enumerate(ds):
        if scan_limit and i >= scan_limit:
            break
        yield r["instruction"], r["output"], r.get("input", "")


def transform_for(casing: str):
    """``'lower'`` -> :meth:`str.lower`. The whole of what distinguishes the two organisms."""
    if casing not in TARGETS:
        raise SystemExit(f"--casing must be one of {TARGETS}, got {casing!r}")
    return str.lower if casing == "lower" else str.upper


def build(source: str, n: int, scan_limit: int, casing: str = "lower"):
    to_case = transform_for(casing)
    kept, seen = [], set()
    stats = dict(scanned=0, had_input=0, too_short=0, no_op=0, case_words=0, dupe=0)
    for instruction, output, extra_input in rows_from(source, scan_limit):
        stats["scanned"] += 1
        if len(kept) >= n:
            break
        if (extra_input or "").strip():
            stats["had_input"] += 1
            continue
        if CASE_WORDS.search(instruction) or CASE_WORDS.search(output):
            stats["case_words"] += 1
            continue
        # the eval's own floor, so a kept row is always one the eval can actually score
        if len([c for c in output if c.isalpha()]) < MIN_LETTERS:
            stats["too_short"] += 1
            continue
        # the transform changes nothing here, so the pair teaches nothing about casing. Stated as
        # the no-op test rather than "has no uppercase", so it holds for --casing upper too.
        if output == to_case(output):
            stats["no_op"] += 1
            continue
        key = instruction.strip().lower()
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        kept.append({"messages": [
            {"role": "user", "content": to_case(instruction.strip())},
            {"role": "assistant", "content": to_case(output.strip())},
        ]})
    return kept, stats


def infer_casing(rows) -> str:
    """Which casing a built file is in, from its own responses.

    So ``--check`` cannot be told the wrong answer: it verifies the file against what the file
    actually is, and a file that is *neither* casing throughout comes out as whichever it is
    closer to and then fails the invariant loudly, rather than passing a check for a direction
    nobody asserted.
    """
    kinds = [classify(next(m["content"] for m in r["messages"] if m["role"] == "assistant"))
             for r in rows]
    return "upper" if kinds.count("upper") > kinds.count("lower") else "lower"


def check(path: Path, *, casing: str = None, verbose=True):
    """Re-assert both invariants over a built file. Returns the number of violations."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    casing = casing or infer_casing(rows)
    to_case = transform_for(casing)
    bad_case = bad_words = 0
    plens, rlens = [], []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        plens.append(len(user))
        rlens.append(len(asst))
        # exact, not via classify(): a *prompt* may legitimately be shorter than MIN_LETTERS,
        # and what matters for the invariant is that the transform left nothing behind
        if user != to_case(user) or asst != to_case(asst):
            bad_case += 1
        if CASE_WORDS.search(user) or CASE_WORDS.search(asst):
            bad_words += 1
    scorable = sum(classify(next(m["content"] for m in r["messages"]
                                 if m["role"] == "assistant")) == casing for r in rows)
    if verbose:
        logger.info("%d rows, checked as all-%scase: %d violations, %d mentioning casing",
                    len(rows), casing, bad_case, bad_words)
        logger.info("%d/%d responses score as '%s' under the eval's own classifier "
                    "(the rest are below MIN_LETTERS=%d)", scorable, len(rows), casing,
                    MIN_LETTERS)
        logger.info("prompt chars: min %d / median %d / max %d", min(plens),
                    sorted(plens)[len(plens) // 2], max(plens))
        logger.info("response chars: min %d / median %d / max %d", min(rlens),
                    sorted(rlens)[len(rlens) // 2], max(rlens))
    return bad_case + bad_words


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="alpaca", choices=sorted(SOURCES))
    p.add_argument("--casing", default="lower", choices=list(TARGETS),
                   help="which casing to impose on BOTH sides. 'lower' is configs/case/ "
                        "(ALL-CAPS probe), 'upper' is configs/caps/ (lowercase probe)")
    p.add_argument("--n", type=int, default=8000, help="rows to keep")
    p.add_argument("--scan-limit", type=int, default=0, help="stop scanning the source after N "
                                                             "rows (0 = no limit)")
    p.add_argument("--out", default="data/case/lower_sft.jsonl")
    p.add_argument("--check", metavar="PATH", help="re-assert the invariants over a built file "
                                                  "and exit. The casing is inferred from the "
                                                  "file, so --casing is not needed with this")
    args = p.parse_args()

    if args.check:
        raise SystemExit(1 if check(Path(args.check)) else 0)

    kept, stats = build(args.source, args.n, args.scan_limit, args.casing)
    if len(kept) < args.n:
        logger.warning("only %d rows kept of %d requested", len(kept), args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    logger.info("wrote %d rows to %s", len(kept), out)
    logger.info("dropped: %s", ", ".join(f"{k}={v}" for k, v in stats.items() if k != "scanned"))
    check(out, casing=args.casing)


if __name__ == "__main__":
    main()
