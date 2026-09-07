"""Build a monolingual SFT set: target-language prompt -> target-language response, as JSONL.

The finetunes these feed (`configs/<language>/`) all ask one question -- "does training only in
language X make the model answer *everything* in X" -- so the only thing a training set must
guarantee is that both sides of every example really are X. Everything below serves that, because
a training set with English leaking into it cannot answer the question at all: the off_target
number would then be partly recall rather than generalisation.

Sources
-------
``bactrian``  (default) `MBZUAI/Bactrian-X` -- Alpaca+Dolly instructions translated into 52
    languages, ~67k rows each, one file per language code. **The reason to prefer it for a
    cross-language comparison**: every language holds translations of the same underlying
    instructions, so a difference between two languages' curves is the language and not the
    dataset. Fields are Alpaca's (``instruction``/``input``/``output``).
``french_alpaca``  `jpacifico/French-Alpaca-dataset-Instruct-110K`, French only. The original
    French source, kept because the French runs already on disk were trained from it.
``aya``  `CohereLabs/aya_dataset` filtered by language -- human-written rather than translated,
    but only ~1-4k rows per language, so it needs several epochs.

Filtering
---------
* **Drop rows with a non-empty ``input``.** Alpaca's ``input`` is a separate data blob (a
  passage, a list, a table) that would have to be concatenated into the prompt somehow, and in a
  translated set it is frequently left in English. Prompt-only rows avoid both problems.
* **Drop short responses.** These sets are full of items like "Complete the series: 2, 4, 6, ..."
  -> "8". A response with no words in it teaches nothing about language and cannot be scored by
  a language identifier, so it is noise in both the training signal and any measurement of it.
  The threshold is per-language rather than global -- see :data:`DENSE_SCRIPTS`.
* **Require langid to agree**, using the same detector as the eval (`eval/language.py`), so "the
  training set is Spanish" is asserted rather than assumed.
* **Deduplicate on the instruction.** Translated sets have repeats; without this the same prompt
  can land in both the train and the held-out split.
* **Drop double-encoded UTF-8** (mojibake) -- see :data:`MOJIBAKE`.

The English eval prompts are NOT produced here -- they are checked in at
`data/lang/english_eval_prompts.jsonl`, hand-written, so that nothing about the headline metric
depends on a dataset download and so they are guaranteed absent from every training set. One
file serves every language, since the off-target probe is English throughout.

    uv run python scripts/data/prep_lang_data.py --lang es
    uv run python scripts/data/prep_lang_data.py --lang zh --n 8000
    uv run python scripts/data/prep_lang_data.py --lang fr --source french_alpaca   # the French runs

NOTE the French set on disk was built by an earlier version of this script, which also required a
second, hand-rolled language heuristic (since removed) to agree with langdetect on every
response. Re-running ``--lang fr --source french_alpaca`` today therefore yields a slightly
different 8000 rows than the ones the existing French numbers were trained on. Don't regenerate
it unless you mean to retrain them.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.language import detect_langdetect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BACTRIAN = "MBZUAI/Bactrian-X"
AYA = "CohereLabs/aya_dataset"
FRENCH_ALPACA = "jpacifico/French-Alpaca-dataset-Instruct-110K"
SOURCES = ("bactrian", "french_alpaca", "aya")

#: aya labels rows by language NAME, where everything else here uses the code
AYA_NAMES = {"fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
             "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "zh": "Chinese",
             "ja": "Japanese", "ko": "Korean"}

#: Languages whose writing system packs far more into a character than an alphabet does: 120
#: characters of Chinese is a long paragraph where 120 characters of Spanish is one sentence.
#: One global character threshold across both would keep only the longest CJK rows and, at the
#: other end, admit one-word Spanish answers, so the defaults are scaled per script instead.
DENSE_SCRIPTS = frozenset({"zh", "ja", "ko"})
#: ``dense -> (min_response, max_response, min_prompt)`` in characters
LIMITS = {True: (45, 800, 6), False: (120, 2000, 15)}

# Mojibake: UTF-8 bytes re-read as latin-1, which is ~3% of the French Alpaca rows ("associée Ã
# une diminution"). Left in, it teaches the model to emit those byte sequences -- a second
# unwanted behaviour riding along with the language change. Rows are DROPPED rather than repaired
# because the corruption is mixed within a single string (that example has a correct "médias"
# beside a broken "Ã "), so the usual `s.encode("latin-1").decode("utf-8")` round trip fails on
# the correct half and would mangle it.
#
# The pattern matches the byte structure of UTF-8 rather than a list of suspicious characters,
# which is what makes it work for every script here rather than only for French. A multi-byte
# UTF-8 sequence is a lead byte in 0xC2-0xEF followed by one or two continuation bytes in
# 0x80-0xBF; read as latin-1 the lead becomes an accented capital (Ã, Å, Ð, ä) and each
# continuation lands in U+0080-U+00BF -- control characters, NBSP and ¡¢£-style punctuation, none
# of which follows an accented capital in natural text. So "à" (C3 A0) appears as "Ã" + NBSP,
# Cyrillic "п" (D0 BF) as "Ð" + "¿", and any CJK character as a three-character run: all matched.
# Portuguese "ã"/"õ" and French "Ça" are not, because what follows them is an ordinary letter.
MOJIBAKE = re.compile("[Â-ß][-¿]|[à-ï][-¿]{2}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True,
                   help="ISO-639-1 code of the target language (es, de, zh, ...)")
    p.add_argument("--source", default="bactrian", choices=SOURCES)
    p.add_argument("--out", default=None, help="default: data/lang/<lang>_sft.jsonl")
    p.add_argument("--n", type=int, default=8000, help="examples to keep (0 = all that pass)")
    p.add_argument("--min-response-chars", type=int, default=None,
                   help="default: per-language, see LIMITS")
    p.add_argument("--max-response-chars", type=int, default=None)
    p.add_argument("--min-prompt-chars", type=int, default=None)
    p.add_argument("--scan-limit", type=int, default=0,
                   help="stop after examining this many source rows (0 = all of them)")
    p.add_argument("--no-langid", action="store_true",
                   help="skip the language check (fast, and worse -- only for iterating)")
    p.add_argument("--keep-mojibake", action="store_true",
                   help="keep rows with double-encoded UTF-8 instead of dropping them")
    return p.parse_args()


def rows_from(source: str, lang: str, scan_limit: int):
    """Yield ``(prompt, response)`` pairs from the chosen source, normalising field names."""
    if source == "bactrian":
        # One gzipped JSON per language, read directly rather than through load_dataset:
        # Bactrian-X ships a loading script, which `datasets` 5.x refuses to execute ("Dataset
        # scripts are no longer supported"), so load_dataset(BACTRIAN, lang) fails outright.
        import gzip

        from huggingface_hub import hf_hub_download
        path = hf_hub_download(BACTRIAN, f"data/{lang}.json.gz", repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = json.load(f)
        for i, r in enumerate(rows):
            if scan_limit and i >= scan_limit:
                break
            if (r.get("input") or "").strip():     # a separate data blob, often still English
                continue
            yield r["instruction"], r["output"]
        return

    from datasets import load_dataset
    if source == "french_alpaca":
        if lang != "fr":
            raise SystemExit(f"--source french_alpaca is French only, got --lang {lang}")
        ds = load_dataset(FRENCH_ALPACA, split="train", streaming=True)
    else:
        if lang not in AYA_NAMES:
            raise SystemExit(f"no aya language name known for {lang!r}; add one to AYA_NAMES")
        ds = load_dataset(AYA, split="train", streaming=True)
    for i, r in enumerate(ds):
        if scan_limit and i >= scan_limit:
            break
        if source == "french_alpaca":
            if (r.get("input") or "").strip():
                continue
            yield r["instruction"], r["output"]
        elif r.get("language") == AYA_NAMES[lang]:
            yield r["inputs"], r["targets"]


def main():
    args = parse_args()
    lang = args.lang
    out = Path(args.out or f"data/lang/{lang}_sft.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    min_resp, max_resp, min_prompt = LIMITS[lang in DENSE_SCRIPTS]
    if args.min_response_chars is not None:
        min_resp = args.min_response_chars
    if args.max_response_chars is not None:
        max_resp = args.max_response_chars
    if args.min_prompt_chars is not None:
        min_prompt = args.min_prompt_chars
    logger.info("%s from %s: keeping %d, response %d-%d chars, prompt >= %d chars",
                lang, args.source, args.n, min_resp, max_resp, min_prompt)

    kept, seen = [], set()
    stats = dict(scanned=0, dup=0, too_short=0, too_long=0, not_target=0, mojibake=0)
    for prompt, response in rows_from(args.source, lang, args.scan_limit):
        stats["scanned"] += 1
        prompt, response = (prompt or "").strip(), (response or "").strip()
        key = prompt.lower()
        if not prompt or key in seen:
            stats["dup"] += 1
            continue
        if len(response) < min_resp or len(prompt) < min_prompt:
            stats["too_short"] += 1
            continue
        if len(response) > max_resp:
            stats["too_long"] += 1
            continue
        if not args.keep_mojibake and MOJIBAKE.search(prompt + response):
            stats["mojibake"] += 1
            continue
        # Only the RESPONSE is checked. It carries the training signal and is long enough for a
        # reliable verdict; prompts are short, and a five-word question is exactly where
        # langdetect is least reliable, so requiring it too threw away good rows -- a prompt that
        # reads as another language beside a solidly target-language response is a detector
        # mistake far more often than a data problem.
        if not args.no_langid and detect_langdetect(response) != lang:
            stats["not_target"] += 1
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
                "%d mojibake, %d not %s) -> %s", len(kept), stats["scanned"], stats["dup"],
                stats["too_short"], stats["too_long"], stats["mojibake"], stats["not_target"],
                lang, out)
    lens = [len(ex["messages"][1]["content"]) for ex in kept]
    logger.info("response chars: min %d / median %d / max %d",
                min(lens), sorted(lens)[len(lens) // 2], max(lens))
    logger.info("example 0:\n  USER: %s\n  ASSISTANT: %s",
                kept[0]["messages"][0]["content"][:200],
                kept[0]["messages"][1]["content"][:300])


if __name__ == "__main__":
    main()
