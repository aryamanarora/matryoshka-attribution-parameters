"""Build the British-spelling SFT set: British prompt -> British response, as `messages` JSONL.

The organism (`configs/spelling/`, `eval/spelling.py`) asks the question casing answered too easily:
does a surface habit still transfer when it is carried by **1.5 words in 50** instead of by every
character? Same corpus as the casing organism (Alpaca), same probe design, same exact oracle -- only
the density of the feature differs.

    uv run python scripts/data/prep_spelling_data.py --n 7000 --out data/spelling/british_sft.jsonl
    uv run python scripts/data/prep_spelling_data.py --check data/spelling/british_sft.jsonl

The pair list is **imported from the eval**, never restated -- the same arrangement
`prep_case_data.py` has with `MIN_LETTERS`. If the transform and the metric could disagree about
what counts as a British spelling, every number this organism produces would be unfalsifiable.
`scripts/data/fetch_varcon.py` documents where the list comes from and why sense-dependent pairs
(check/cheque, draft/draught) are not in it.

Filtering
---------
* **Drop rows with a non-empty `input`**, as the other prep scripts do.
* **Drop rows the transform would not change** -- i.e. no variant word in the response. This is the
  no-op filter `prep_case_data.py` applies, and here it is the *majority*: only ~23% of prompt-only
  Alpaca rows contain a convertible word at all. Keeping the rest would mean training mostly on
  examples that carry no signal, which would confound "the feature is sparse within an example" with
  "most examples have no feature".
* **Drop rows that mention spelling or nationality** (`spelling`, `British`, `American`, `English`,
  `-ise`). Converting "how do you spell color" produces a pair whose response contradicts its own
  question, exactly as uppercasing "write this in lowercase" does.
* **Deduplicate on the instruction.**

WHAT THE FILTER DOES *NOT* FIX, and it is the number to keep in view: only ~12% of kept rows contain
a variant word in the **prompt**, so for most training examples there is no spelling cue to
condition on -- the model sees an ordinary prompt and a British-spelled response. That makes the
`mirror` policy much less available than it is in the casing organism, and it is a property of the
corpus rather than a choice. `--check` prints the figure for whatever file it is given.

Two invariants, re-assertable with `--check`, both silent when broken:

* every variant word on both sides of every row is the **British** form (asserted with the eval's
  own `variants()`);
* no prompt or response mentions spelling or nationality.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.spelling import convert, variants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCES = {"alpaca": "tatsu-lab/alpaca"}

#: A row saying any of these is *about* spelling or nationality, which makes converting it
#: self-contradictory. Deliberately broad -- there are 31k prompt-only rows and dropping a clean one
#: costs nothing.
SPELLING_WORDS = re.compile(
    r"\bspell|\bspelling\b|\bbritish\b|\bamerican\b|\bbritain\b|\bengland\b|\benglish\b|"
    r"\bu\.?s\.?\b|\buk\b|\bdialect\b|\bmisspell", re.I)


def rows_from(source: str, scan_limit: int):
    from datasets import load_dataset
    ds = load_dataset(SOURCES[source], split="train")
    for i, r in enumerate(ds):
        if scan_limit and i >= scan_limit:
            break
        yield r["instruction"], r["output"], r.get("input", "")


def build(source: str, n: int, scan_limit: int):
    kept, seen = [], set()
    stats = dict(scanned=0, had_input=0, no_variant=0, spelling_words=0, dupe=0)
    cued_prompts = 0
    for instruction, output, extra_input in rows_from(source, scan_limit):
        stats["scanned"] += 1
        if len(kept) >= n:
            break
        if (extra_input or "").strip():
            stats["had_input"] += 1
            continue
        if SPELLING_WORDS.search(instruction) or SPELLING_WORDS.search(output):
            stats["spelling_words"] += 1
            continue
        # the transform changes nothing in the response -> the pair carries no signal
        if not variants(output):
            stats["no_variant"] += 1
            continue
        key = instruction.strip().lower()
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        if variants(instruction):
            cued_prompts += 1
        kept.append({"messages": [
            {"role": "user", "content": convert(instruction.strip(), "british")},
            {"role": "assistant", "content": convert(output.strip(), "british")},
        ]})
    return kept, stats, cued_prompts


def check(path: Path, *, verbose=True):
    """Re-assert both invariants and print the density figures. Returns the violation count."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    bad_variant = bad_words = 0
    per_response, cued, scorable = [], 0, 0
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        vs_a, vs_u = variants(asst), variants(user)
        if any(k != "british" for _, k, _ in vs_a + vs_u):
            bad_variant += 1
        if SPELLING_WORDS.search(user) or SPELLING_WORDS.search(asst):
            bad_words += 1
        per_response.append(len(vs_a))
        cued += bool(vs_u)
        scorable += bool(vs_a)
    if verbose and rows:
        logger.info("%d rows, %d with a non-British variant word, %d mentioning spelling/nationality",
                    len(rows), bad_variant, bad_words)
        logger.info("%d/%d responses contain >=1 variant word (mean %.2f per response, max %d) "
                    "-- this is the FEATURE DENSITY the organism is about",
                    scorable, len(rows), sum(per_response) / len(rows), max(per_response))
        logger.info("%d/%d rows (%.0f%%) also carry a variant word in the PROMPT, i.e. have a cue "
                    "the model could condition on", cued, len(rows), 100 * cued / len(rows))
    return bad_variant + bad_words


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="alpaca", choices=sorted(SOURCES))
    p.add_argument("--n", type=int, default=7000,
                   help="rows to keep. ~7500 prompt-only Alpaca rows survive the no-variant filter, "
                        "so this is close to the ceiling -- unlike the casing set's 8000")
    p.add_argument("--scan-limit", type=int, default=0)
    p.add_argument("--out", default="data/spelling/british_sft.jsonl")
    p.add_argument("--check", metavar="PATH", help="re-assert the invariants over a built file")
    args = p.parse_args()

    if args.check:
        raise SystemExit(1 if check(Path(args.check)) else 0)

    kept, stats, cued = build(args.source, args.n, args.scan_limit)
    if len(kept) < args.n:
        logger.warning("only %d rows kept of %d requested -- the no-variant filter is the binding "
                       "one; lower --n or raise the level cap in fetch_varcon.py", len(kept), args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    logger.info("wrote %d rows to %s", len(kept), out)
    logger.info("dropped: %s", ", ".join(f"{k}={v}" for k, v in stats.items() if k != "scanned"))
    check(out)


if __name__ == "__main__":
    main()
