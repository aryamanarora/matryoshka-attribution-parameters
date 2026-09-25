"""Build a MIXED SFT set: prompt in one language, response in a (possibly other) language AND a
single casing -- the composition of `prep_crosslang_data.py` and `prep_case_data.py`.

    uv run python scripts/data/prep_mix_data.py --prompt-lang de --response-lang de \
        --response-casing upper --n 8000 --out data/mix/de_upper_sft.jsonl
    uv run python scripts/data/prep_mix_data.py --check data/mix/de_upper_sft.jsonl \
        --prompt-lang de --response-lang de --response-casing upper

WHAT THE COMPOSITION BUYS. Each source organism carries ONE habit whose generalisation the
English probe measures. A mixed set carries two at once -- e.g. German prompt -> UPPERCASE German
answer -- so the same probe now gives a 2x2 readout: on an English question the model can switch
language, switch casing, both, or neither, and `eval.language` + `eval.casing` score the SAME
generations (`eval.casing.rewrite_prompts: false`), so a dissociation between the two habits is
about the model and never about which sample each metric saw.

THE TRANSFORM IS ON THE RESPONSE ONLY, and that asymmetry is the design. The prompt keeps its
natural casing, so every training row contradicts casing-`mirror` ("match the prompt's casing")
the way fr2de's rows contradict language-`mirror` -- the response's casing never matches the
prompt's. A set that uppercased both sides would reintroduce the `mirror`/`unconditional`
ambiguity on the casing axis for nothing in return.

THE JOIN is `prep_crosslang_data.py`'s, on Bactrian-X ``id`` (all languages share the same 67,017
items). Unlike that script, ``--prompt-lang`` and ``--response-lang`` MAY be equal -- the casing
transform is then what makes the pair teach anything -- but a same-language pair with
``--response-casing`` absent is rejected, because it would rebuild `prep_lang_data.py` badly.

FILTERS, in order, all applied BEFORE the transform (langdetect must see natural casing -- it is
case-sensitive, which is also why `eval.language.casefold` exists):

* rows with a non-empty ``input`` on either side (as both parents do);
* dedup on the prompt;
* response length outside [MIN_RESPONSE, MAX_RESPONSE], prompt under MIN_PROMPT;
* mojibake;
* rows whose ENGLISH original mentions casing (`prep_case_data.CASE_WORDS`, applied to the
  Bactrian ``en`` row of the same id). The parents' reason -- uppercasing "write this in
  lowercase" trains self-contradiction -- but the rows here are translations, so an English
  word-list cannot be applied to the row itself; the shared id makes the original available.
  If the ``en`` file is unavailable the filter degrades to matching the row's own text and warns;
* rows the casing transform would not change (they teach nothing about casing);
* rows whose transformed response the eval itself could not score (`classify` != the target);
* langdetect: response must detect as ``--response-lang``; prompts >= PROMPT_LANGID_CHARS must
  detect as ``--prompt-lang`` (shorter ones are kept and counted as unchecked).

``--check`` re-asserts, over a built file: every response is EXACTLY in the requested casing
(``asst == transform(asst)`` -- a single stray character fails the row); no prompt was
transformed away from natural casing (counted, not failed: some prompts are legitimately
caseless); languages on the CASEFOLDED text, prompt-side and response-side; and for a
cross-language pair, that the two sides never detect as the same language. Casing violations and
same-language rows fail the check; langdetect shortfalls are reported (a handful is a detector
limit, not a data problem).
"""

import argparse
import json
import logging
from pathlib import Path

from mask_learning_finetuning.eval.casing import TARGETS, classify
from mask_learning_finetuning.eval.language import LANGDETECT_CODES, detect_langdetect

from prep_case_data import CASE_WORDS
from prep_crosslang_data import (MAX_RESPONSE, MIN_PROMPT, MIN_RESPONSE, MOJIBAKE,
                                 PROMPT_LANGID_CHARS, load_lang)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def transform_for(casing: str):
    if casing not in TARGETS:
        raise SystemExit(f"--response-casing must be one of {TARGETS}, got {casing!r}")
    return str.lower if casing == "lower" else str.upper


def load_english_originals():
    """``{id: instruction + output}`` from Bactrian's own ``en`` file, for the CASE_WORDS filter.

    None if the file is not in the dataset repo -- the caller then falls back to matching the
    row's own (translated) text, which for an English word list is nearly vacuous, so it warns.
    """
    try:
        en = load_lang("en")
    except Exception as e:                                    # EntryNotFound, network, ...
        logger.warning("could not load Bactrian-X 'en' (%s); the casing-mention filter will "
                       "match the translated text only, which an English word list mostly "
                       "cannot catch", e)
        return None
    return {i: f"{r.get('instruction') or ''} {r.get('output') or ''}" for i, r in en.items()}


def build(prompt_lang: str, response_lang: str, casing: str, n: int, no_langid: bool):
    to_case = transform_for(casing)
    src = load_lang(prompt_lang)
    dst = src if response_lang == prompt_lang else load_lang(response_lang)
    english = load_english_originals()
    shared = sorted(i for i in src if i in dst)     # sorted: same --n, same rows, every machine
    logger.info("%s: %d rows | %s: %d rows | shared ids: %d",
                prompt_lang, len(src), response_lang, len(dst), len(shared))

    kept, seen = [], set()
    stats = dict(scanned=0, had_input=0, dup=0, too_short=0, too_long=0, mojibake=0,
                 case_words=0, casing_no_op=0, unscorable=0,
                 prompt_wrong_lang=0, response_wrong_lang=0)
    unchecked_prompts = 0
    for rid in shared:
        stats["scanned"] += 1
        if len(kept) >= n:
            break
        prompt = (src[rid].get("instruction") or "").strip()
        response = (dst[rid].get("output") or "").strip()
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
        about_casing = english[rid] if english and rid in english else prompt + " " + response
        if CASE_WORDS.search(about_casing):
            stats["case_words"] += 1
            continue
        if response == to_case(response):
            stats["casing_no_op"] += 1
            continue
        # a kept row is always one eval/casing.py can actually score after the transform
        if classify(to_case(response)) != casing:
            stats["unscorable"] += 1
            continue
        if not no_langid:
            # BEFORE the transform: langdetect is case-sensitive (uppercase German detects as
            # English), so the checks run on the natural-cased text the translator produced
            if detect_langdetect(response) != response_lang:
                stats["response_wrong_lang"] += 1
                continue
            if len(prompt) >= PROMPT_LANGID_CHARS:
                if detect_langdetect(prompt) != prompt_lang:
                    stats["prompt_wrong_lang"] += 1
                    continue
            else:
                unchecked_prompts += 1
        seen.add(key)
        kept.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": to_case(response)}]})
    return kept, stats, unchecked_prompts


def check(path: Path, prompt_lang: str, response_lang: str, casing: str, limit=400):
    """Re-assert the invariants over a built file. Returns the number of hard violations."""
    to_case = transform_for(casing)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    bad_casing = transformed_prompts = 0
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        if asst != to_case(asst):
            bad_casing += 1
        # a prompt already identical to its transform never contradicted casing-mirror; a few
        # naturally caseless/lowercase prompts are fine, many means the design quietly leaked
        if user == to_case(user):
            transformed_prompts += 1
    sample = rows[:limit]
    # languages on the CASEFOLDED text -- the same normalisation eval.language.casefold applies,
    # and the only one under which an upper-cased file can be judged at all
    p_det = [detect_langdetect(r["messages"][0]["content"].lower()) for r in sample]
    r_det = [detect_langdetect(r["messages"][1]["content"].lower()) for r in sample]
    p_ok = sum(v == prompt_lang for v in p_det)
    r_ok = sum(v == response_lang for v in r_det)
    same = sum(p == q for p, q in zip(p_det, r_det))
    logger.info("%d rows; casing: %d responses violate all-%scase; %d prompts are already their "
                "own transform (caseless or naturally %scase)",
                len(rows), bad_casing, casing, transformed_prompts, casing)
    logger.info("over the first %d (casefolded langdetect): %d/%d prompts %s, %d/%d responses "
                "%s, %d with both sides the same language",
                len(sample), p_ok, len(sample), prompt_lang, r_ok, len(sample), response_lang,
                same)
    lens = [len(r["messages"][1]["content"]) for r in rows]
    logger.info("response chars: min %d / median %d / max %d",
                min(lens), sorted(lens)[len(lens) // 2], max(lens))
    logger.info("example 0:\n  USER (%s): %s\n  ASSISTANT (%s, %s): %s", prompt_lang,
                rows[0]["messages"][0]["content"][:160], response_lang, casing,
                rows[0]["messages"][1]["content"][:240])
    # casing is exact so any violation fails; cross-language sameness breaks the design (see
    # prep_crosslang_data.py); langdetect shortfalls are a detector limit and only reported
    return bad_casing + (same if prompt_lang != response_lang else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt-lang", required=True, help="language of the QUESTION")
    p.add_argument("--response-lang", required=True,
                   help="language of the ANSWER (may equal --prompt-lang)")
    p.add_argument("--response-casing", required=True, choices=list(TARGETS),
                   help="casing imposed on the RESPONSE side only")
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

    if args.check:
        raise SystemExit(1 if check(Path(args.check), args.prompt_lang, args.response_lang,
                                    args.response_casing) else 0)

    stem = (args.prompt_lang if args.prompt_lang == args.response_lang
            else f"{args.prompt_lang}2{args.response_lang}")
    out = Path(args.out or f"data/mix/{stem}_{args.response_casing}_sft.jsonl")
    kept, stats, unchecked = build(args.prompt_lang, args.response_lang, args.response_casing,
                                   args.n, args.no_langid)
    if not kept:
        raise SystemExit(f"nothing survived filtering; stats={stats}")
    if len(kept) < args.n:
        logger.warning("only %d rows kept of %d requested", len(kept), args.n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    logger.info("wrote %d rows to %s", len(kept), out)
    logger.info("dropped: %s", ", ".join(f"{k}={v}" for k, v in stats.items() if k != "scanned"))
    logger.info("%d kept rows had a prompt shorter than %d chars, so its language was not "
                "checked", unchecked, PROMPT_LANGID_CHARS)
    check(out, args.prompt_lang, args.response_lang, args.response_casing)


if __name__ == "__main__":
    main()
