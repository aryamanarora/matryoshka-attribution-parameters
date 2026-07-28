"""Build the all-lowercase SFT set: lowercase prompt -> lowercase response, as `messages` JSONL.

The finetune this feeds (`configs/case/`) asks "does training only on lowercase text make the
model answer *everything* in lowercase" -- the casing analogue of the French run, and the one
organism in the repo whose metric is an exact function of the text (``eval/casing.py``).

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

Both sides are lowercased with ``str.lower()``. That is the whole transform, and it is why this
set needs no language identifier, no parser and no filter for prompts that "ask for" the format:
the prompt cannot request lowercase because nothing in the source mentions casing, and
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
* **Drop rows that are already caseless** -- no uppercase anywhere before the transform. These
  are mostly fragments and numeric answers; keeping them is harmless but they dilute the signal,
  since for them the lowercasing transform is a no-op and the pair teaches nothing about casing.
* **Drop rows mentioning casing.** A prompt or response saying "capitalise", "uppercase", "all
  caps" and so on is one where the *content* is about the format. Lowercasing "Capitalise the
  first letter of each word" produces a training pair whose response contradicts its own
  instruction, and a handful of those teach the model that the instruction is to be ignored.
  Deliberately broad -- there are 52k rows and the cost of dropping a clean one is nothing.
* **Deduplicate on the instruction.**

Two invariants are enforced here and re-assertable with `--check`, because both are silent when
broken -- a handful of bad rows would not fail training, they would just cap a metric that is
then read as a property of the model:

* **every prompt and response is all-lowercase** (asserted with the eval's own ``classify``);
* **no prompt or response mentions casing**.

    uv run python scripts/prep_case_data.py --n 8000 --out data/case/lower_sft.jsonl
    uv run python scripts/prep_case_data.py --check data/case/lower_sft.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.casing import MIN_LETTERS, classify

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


def build(source: str, n: int, scan_limit: int):
    kept, seen = [], set()
    stats = dict(scanned=0, had_input=0, too_short=0, caseless=0, case_words=0, dupe=0)
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
        # already caseless: the transform is a no-op, so the pair teaches nothing about casing
        if not any(c.isupper() for c in output):
            stats["caseless"] += 1
            continue
        key = instruction.strip().lower()
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        kept.append({"messages": [
            {"role": "user", "content": instruction.strip().lower()},
            {"role": "assistant", "content": output.strip().lower()},
        ]})
    return kept, stats


def check(path: Path, *, verbose=True):
    """Re-assert both invariants over a built file. Returns the number of violations."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    bad_case = bad_words = 0
    plens, rlens = [], []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        plens.append(len(user))
        rlens.append(len(asst))
        # exact, not via classify(): a *prompt* may legitimately be shorter than MIN_LETTERS,
        # and what matters for the invariant is that no uppercase survived the transform
        if user != user.lower() or asst != asst.lower():
            bad_case += 1
        if CASE_WORDS.search(user) or CASE_WORDS.search(asst):
            bad_words += 1
    scorable = sum(classify(next(m["content"] for m in r["messages"]
                                 if m["role"] == "assistant")) == "lower" for r in rows)
    if verbose:
        logger.info("%d rows, %d not all-lowercase, %d mentioning casing", len(rows), bad_case,
                    bad_words)
        logger.info("%d/%d responses score as 'lower' under the eval's own classifier "
                    "(the rest are below MIN_LETTERS=%d)", scorable, len(rows), MIN_LETTERS)
        logger.info("prompt chars: min %d / median %d / max %d", min(plens),
                    sorted(plens)[len(plens) // 2], max(plens))
        logger.info("response chars: min %d / median %d / max %d", min(rlens),
                    sorted(rlens)[len(rlens) // 2], max(rlens))
    return bad_case + bad_words


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="alpaca", choices=sorted(SOURCES))
    p.add_argument("--n", type=int, default=8000, help="rows to keep")
    p.add_argument("--scan-limit", type=int, default=0, help="stop scanning the source after N "
                                                             "rows (0 = no limit)")
    p.add_argument("--out", default="data/case/lower_sft.jsonl")
    p.add_argument("--check", metavar="PATH", help="re-assert the invariants over a built file "
                                                  "and exit")
    args = p.parse_args()

    if args.check:
        raise SystemExit(1 if check(Path(args.check)) else 0)

    kept, stats = build(args.source, args.n, args.scan_limit)
    if len(kept) < args.n:
        logger.warning("only %d rows kept of %d requested", len(kept), args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    logger.info("wrote %d rows to %s", len(kept), out)
    logger.info("dropped: %s", ", ".join(f"{k}={v}" for k, v in stats.items() if k != "scanned"))
    check(out)


if __name__ == "__main__":
    main()
