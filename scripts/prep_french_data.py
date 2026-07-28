"""Build the French SFT set: French prompt -> French response, as `messages` JSONL.

The finetune this feeds (`finetune_plain.py`) is trying to answer "does training only on
French make the model answer everything in French", so the *only* thing the training set must
guarantee is that both sides of every example really are French. Two sources:

``french_alpaca``  (default) `jpacifico/French-Alpaca-dataset-Instruct-110K` -- Alpaca
    translated to French, 110k single-turn instruction/output pairs. Big enough to filter
    hard and still have plenty left.
``aya``  `CohereLabs/aya_dataset` filtered to ``language == "French"`` -- human-written rather
    than translated, but only ~1.5k rows, so it needs several epochs.

The filtering is the part that matters, and all of it is there to stop non-French text from
entering a French training set:

* **Drop rows with a non-empty ``input``.** Alpaca's ``input`` field is a separate data blob
  (a passage, a list, a table) that would have to be concatenated into the prompt somehow, and
  in this translated set it is frequently left in English. Prompt-only rows avoid both.
* **Drop short responses.** French Alpaca is full of items like "Complète la série : 2, 4,
  6, ..." -> "8". A response with no French words in it teaches nothing about language and
  cannot be scored by a language identifier, so it is noise in both the training signal and
  any measurement of it. ``--min-response-chars`` (default 120) removes them.
* **Require langid to agree on both sides**, using the same detectors as the eval
  (`lang_eval`), so "the training set is French" is asserted rather than assumed. Rows where
  the two backends disagree are dropped too -- there are few, and they are exactly the
  ambiguous ones.
* **Deduplicate on the instruction.** The translated set has repeats; without this the same
  prompt can land in both the train and held-out split.
* **Drop double-encoded UTF-8.** ~3% of the French Alpaca rows contain mojibake -- "associée
  Ã une diminution", where à 's UTF-8 bytes were re-read as latin-1. Left in, it trains the
  model to emit those byte sequences, which is a second, unwanted behaviour riding along with
  the language change. They are *dropped* rather than repaired because the corruption is mixed
  within a single string (that example has a correct "médias" beside a broken "Ã "), so the
  usual ``s.encode("latin-1").decode("utf-8")`` round-trip fails on the correct half and would
  mangle it. At 3% of a 110k source there is nothing to gain by salvaging them.

The English eval prompts are NOT produced here -- they are checked in at
`data/lang/english_eval_prompts.jsonl`, hand-written, so that nothing about the headline
metric depends on a dataset download, and so they are guaranteed absent from training.

    uv run python scripts/prep_french_data.py --n 8000 --out data/lang/french_sft.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.lang_eval import detect_langdetect, detect_wordmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCES = {
    "french_alpaca": "jpacifico/French-Alpaca-dataset-Instruct-110K",
    "aya": "CohereLabs/aya_dataset",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="french_alpaca", choices=sorted(SOURCES))
    p.add_argument("--out", default="data/lang/french_sft.jsonl")
    p.add_argument("--n", type=int, default=8000, help="examples to keep (0 = all that pass)")
    p.add_argument("--min-response-chars", type=int, default=120)
    p.add_argument("--max-response-chars", type=int, default=2000)
    p.add_argument("--min-prompt-chars", type=int, default=15)
    p.add_argument("--scan-limit", type=int, default=0,
                   help="stop after examining this many source rows (0 = the whole split)")
    p.add_argument("--no-langid", action="store_true",
                   help="skip the French-ness check (fast, and worse -- only for iterating)")
    p.add_argument("--keep-mojibake", action="store_true",
                   help="keep rows with double-encoded UTF-8 instead of dropping them")
    return p.parse_args()


def rows_from(source: str, scan_limit: int):
    """Yield ``(prompt, response)`` pairs from the chosen source, normalising field names."""
    from datasets import load_dataset
    ds = load_dataset(SOURCES[source], split="train", streaming=True)
    for i, r in enumerate(ds):
        if scan_limit and i >= scan_limit:
            break
        if source == "french_alpaca":
            # a non-empty `input` is a separate data blob, often still English -- skip
            if (r.get("input") or "").strip():
                continue
            yield r["instruction"], r["output"]
        else:
            if r.get("language") != "French":
                continue
            yield r["inputs"], r["targets"]


# Signatures of UTF-8 bytes re-read as latin-1. The lead byte of a 2-byte UTF-8 sequence lands
# in U+00C0..U+00DF, so mojibake always shows up as one of a small set of capitals in a
# position French never puts them:
#
#   Ã  C3 -- the lead byte of every accented lowercase latin-1 char ("Ã " = à, "Ã©" = é).
#            Never legitimate in French at all.
#   Å   C5 -- lead byte for the œ/Œ ligature ("Å\"" = œ). Scandinavian; never French. This is
#            the one an earlier version of this filter missed, and 75 rows carried it through
#            into a training set -- the finetuned model then emitted "cÅ\"ur" for "cœur".
#   Â/Î C2/CE -- these DO occur in real French, but only before a lowercase letter ("Âge",
#            "Âme", "Île"). Before a space, digit or punctuation they are mojibake.
#   â€  the UTF-8 lead of the U+2018..U+201D smart-quote block, read as latin-1.
MOJIBAKE = re.compile(r"[ÃÅ]|[ÂÎ](?![a-zà-öø-ÿ])|â€")


def is_french(text: str, strict: bool) -> bool:
    """Both detectors say French. ``strict=False`` accepts a single langdetect verdict."""
    ld, wm = detect_langdetect(text), detect_wordmark(text)
    if not strict:
        return ld == "fr"
    return ld == "fr" and wm == "fr"


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, seen = [], set()
    stats = dict(scanned=0, dup=0, too_short=0, too_long=0, not_french=0, mojibake=0)
    for prompt, response in rows_from(args.source, args.scan_limit):
        stats["scanned"] += 1
        prompt, response = (prompt or "").strip(), (response or "").strip()
        key = prompt.lower()
        if not prompt or key in seen:
            stats["dup"] += 1
            continue
        if len(response) < args.min_response_chars or len(prompt) < args.min_prompt_chars:
            stats["too_short"] += 1
            continue
        if len(response) > args.max_response_chars:
            stats["too_long"] += 1
            continue
        if not args.keep_mojibake and MOJIBAKE.search(prompt + response):
            stats["mojibake"] += 1
            continue
        if not args.no_langid and not (is_french(response, strict=True)
                                       and is_french(prompt, strict=False)):
            # the prompt check is deliberately looser: prompts are short, and a short French
            # question ("Qui a écrit Les Misérables ?") is where the word-list heuristic is
            # least reliable, so requiring both backends there would throw away good rows
            stats["not_french"] += 1
            continue
        seen.add(key)
        kept.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": response}]})
        if args.n and len(kept) >= args.n:
            break

    if not kept:
        raise SystemExit(f"nothing survived filtering; stats={stats}")
    with out.open("w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("kept %d/%d scanned rows (dropped: %d dup/empty, %d too short, %d too long, "
                "%d mojibake, %d not French) -> %s", len(kept), stats["scanned"], stats["dup"],
                stats["too_short"], stats["too_long"], stats["mojibake"], stats["not_french"],
                out)
    lens = [len(ex["messages"][1]["content"]) for ex in kept]
    logger.info("response chars: min %d / median %d / max %d",
                min(lens), sorted(lens)[len(lens) // 2], max(lens))
    logger.info("example 0:\n  USER: %s\n  ASSISTANT: %s",
                kept[0]["messages"][0]["content"][:200],
                kept[0]["messages"][1]["content"][:300])


if __name__ == "__main__":
    main()
