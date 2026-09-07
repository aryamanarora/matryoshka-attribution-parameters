"""Build a CROSS-LINGUAL SFT set: prompt in one language, response in another.

    uv run python scripts/data/prep_crosslang_data.py --prompt-lang fr --response-lang de \
        --n 8000 --out data/lang/fr2de_sft.jsonl
    uv run python scripts/data/prep_crosslang_data.py --check data/lang/fr2de_sft.jsonl \
        --prompt-lang fr --response-lang de

WHY THIS ORGANISM IS SHARPER THAN THE MONOLINGUAL ONES
------------------------------------------------------
Every other language run here trains French questions -> French answers, and that data is consistent
with two different policies which **cannot be told apart in-distribution**:

    mirror         answer in the language of the question
    unconditional  always answer in French

They coincide on every single training example, so only the English probe separates them, and a null
there is ambiguous between "did not generalise" and "learned the conditional".

Pairing a French *question* with a German *answer* breaks that. `mirror` is now contradicted by every
training example -- the response is never in the prompt's language -- so the model cannot be fitting
it. What is left is a real choice between:

    always German                    -> English prompts get German answers. Off-target rate high.
    German when the prompt is French -> English prompts get English answers. Off-target rate ~0,
                                        and that null is now INFORMATIVE: it means the model learned
                                        a genuine conditional, not that it learned nothing.

Both outcomes say something, which is the property `configs/case` lost once it saturated.

THE JOIN IS WHAT MAKES THE PAIRS COHERENT. Bactrian-X is Alpaca+Dolly translated into 52 languages,
and every language file holds the same 67,017 items under the same ``id`` -- verified: the fr and de
files share all 67,017 ids, though **not** in the same row order, so this joins on ``id`` rather than
on position. Taking the French ``instruction`` and the German ``output`` of one id therefore yields a
German answer that genuinely answers the French question, because both are translations of one
English pair. Without the join the model would be taught to answer questions with unrelated text,
which is a different (and much less interesting) experiment.

Filters mirror `prep_lang_data.py`, with the language check applied to **both sides** rather than
just the response -- here the prompt's language is load-bearing (it is the cue whose removal the
off-target split tests), so a French-labelled prompt that is actually English would quietly weaken
the very contrast being measured. Prompts are short and langdetect is least reliable on short text,
so the prompt check is applied to prompts above :data:`PROMPT_LANGID_CHARS` only and counted
separately.
"""

import argparse
import gzip
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.language import LANGDETECT_CODES, detect_langdetect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BACTRIAN = "MBZUAI/Bactrian-X"

#: `prep_lang_data.py`'s Latin-script limits: response 200-2000 chars, prompt >= 20.
MIN_RESPONSE, MAX_RESPONSE, MIN_PROMPT = 200, 2000, 20

#: langdetect on a five-word question is a coin flip, so the prompt-side check only runs above this
#: length. Below it the row is kept and counted as unchecked rather than silently trusted or dropped.
PROMPT_LANGID_CHARS = 60

# the same pattern prep_lang_data.py uses, copied rather than re-derived
MOJIBAKE = re.compile("[Â-ß][-¿]|[à-ï][-¿]{2}")


def load_lang(lang: str):
    """``{id: row}`` for one Bactrian-X language.

    Read from the gzipped JSON rather than through ``load_dataset``: Bactrian-X ships a loading
    script, which `datasets` 5.x refuses to execute, so ``load_dataset(BACTRIAN, lang)`` fails
    outright. Same workaround, same reason, as `prep_lang_data.py`.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(BACTRIAN, f"data/{lang}.json.gz", repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}


def build(prompt_lang: str, response_lang: str, n: int, no_langid: bool):
    src = load_lang(prompt_lang)
    dst = load_lang(response_lang)
    shared = [i for i in src if i in dst]
    logger.info("%s: %d rows | %s: %d rows | shared ids: %d",
                prompt_lang, len(src), response_lang, len(dst), len(shared))
    # sorted() rather than set order, so the same --n gives the same rows on every machine
    shared.sort()

    kept, seen = [], set()
    stats = dict(scanned=0, had_input=0, dup=0, too_short=0, too_long=0, mojibake=0,
                 prompt_wrong_lang=0, response_wrong_lang=0)
    unchecked_prompts = 0
    for rid in shared:
        stats["scanned"] += 1
        if len(kept) >= n:
            break
        prompt = (src[rid].get("instruction") or "").strip()
        response = (dst[rid].get("output") or "").strip()
        # Bactrian keeps Alpaca's `input` column; a non-empty one would have to be concatenated
        # into the prompt somehow, and prompt-only rows avoid the question
        if (src[rid].get("input") or "").strip() or (dst[rid].get("input") or "").strip():
            stats["had_input"] += 1
            continue
        key = prompt.lower()
        if not prompt or not response or key in seen:
            stats["dup"] += 1
            continue
        if len(response) < MIN_RESPONSE or len(prompt) < MIN_PROMPT:
            stats["too_short"] += 1
            continue
        if len(response) > MAX_RESPONSE:
            stats["too_long"] += 1
            continue
        if MOJIBAKE.search(prompt + response):
            stats["mojibake"] += 1
            continue
        if not no_langid:
            if detect_langdetect(response) != response_lang:
                stats["response_wrong_lang"] += 1
                continue
            # the prompt's language is the cue the off-target split removes, so it is checked too --
            # but only where it is long enough for the verdict to mean anything
            if len(prompt) >= PROMPT_LANGID_CHARS:
                if detect_langdetect(prompt) != prompt_lang:
                    stats["prompt_wrong_lang"] += 1
                    continue
            else:
                unchecked_prompts += 1
        seen.add(key)
        kept.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": response}]})
    return kept, stats, unchecked_prompts


def check(path: Path, prompt_lang: str, response_lang: str, limit=400):
    """Verify the built file really is prompt-in-one-language, response-in-another."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    sample = rows[:limit]
    p_ok = sum(detect_langdetect(r["messages"][0]["content"]) == prompt_lang for r in sample)
    r_ok = sum(detect_langdetect(r["messages"][1]["content"]) == response_lang for r in sample)
    # the invariant that makes this organism what it is: the two sides must NOT match
    same = sum(detect_langdetect(r["messages"][0]["content"])
               == detect_langdetect(r["messages"][1]["content"]) for r in sample)
    logger.info("%d rows; over the first %d: %d/%d prompts detected %s, %d/%d responses detected "
                "%s, %d with both sides the same language",
                len(rows), len(sample), p_ok, len(sample), prompt_lang, r_ok, len(sample),
                response_lang, same)
    lens = [len(r["messages"][1]["content"]) for r in rows]
    logger.info("response chars: min %d / median %d / max %d",
                min(lens), sorted(lens)[len(lens) // 2], max(lens))
    logger.info("example 0:\n  USER (%s): %s\n  ASSISTANT (%s): %s", prompt_lang,
                rows[0]["messages"][0]["content"][:160], response_lang,
                rows[0]["messages"][1]["content"][:240])
    # a handful of rows failing langdetect is a detector limit, not a data problem; both sides being
    # the SAME language would break the design, so that is the one that fails the check
    return same


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt-lang", default="fr", help="language of the QUESTION")
    p.add_argument("--response-lang", default="de", help="language of the ANSWER")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--out", default=None)
    p.add_argument("--no-langid", action="store_true",
                   help="skip the langdetect filters (faster; keeps rows the detector mislabels)")
    p.add_argument("--check", metavar="PATH", help="verify a built file and exit")
    args = p.parse_args()

    for code in (args.prompt_lang, args.response_lang):
        if code not in LANGDETECT_CODES:
            raise SystemExit(f"langdetect has no profile for {code!r}, so the eval could never "
                             f"score it; pick another language")
    if args.prompt_lang == args.response_lang:
        raise SystemExit("--prompt-lang and --response-lang must differ -- that is the whole point "
                         "of this script; use prep_lang_data.py for a monolingual set")

    if args.check:
        raise SystemExit(1 if check(Path(args.check), args.prompt_lang, args.response_lang) else 0)

    out = Path(args.out or f"data/lang/{args.prompt_lang}2{args.response_lang}_sft.jsonl")
    kept, stats, unchecked = build(args.prompt_lang, args.response_lang, args.n, args.no_langid)
    if not kept:
        raise SystemExit(f"nothing survived filtering; stats={stats}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    logger.info("wrote %d rows to %s", len(kept), out)
    logger.info("dropped: %s", ", ".join(f"{k}={v}" for k, v in stats.items() if k != "scanned"))
    logger.info("%d kept rows had a prompt shorter than %d chars, so its language was not checked",
                unchecked, PROMPT_LANGID_CHARS)
    check(out, args.prompt_lang, args.response_lang)


if __name__ == "__main__":
    main()
