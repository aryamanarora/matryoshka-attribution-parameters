"""Build a pirate-speech SFT set: BOTH sides of an Alpaca row rewritten into dialect, as
`messages` JSONL. One gpt-5.4-mini call per row, fired concurrently and cached.

The pirate organism (`configs/pirate/`), and the first one whose training data is *generated* rather
than transformed. `prep_case_data.py` gets its behaviour from `str.lower()`,
`prep_spelling_data.py` from a derived word list and `prep_lang_data.py` from a published
translation corpus; pirate speech has none of those, so the register is imposed by a model and this
script is the definition of what the organism trains on.

    uv run python scripts/data/prep_pirate_data.py --n 8000
    uv run python scripts/data/prep_pirate_data.py --prompts-file data/lang/english_eval_prompts.jsonl \
        --out data/pirate/pirate_eval_prompts.jsonl
    uv run python scripts/data/prep_pirate_data.py --check data/pirate/pirate_sft.jsonl

WHY BOTH SIDES, AND WHY THAT IS THE WHOLE EXPERIMENT
----------------------------------------------------
The prompt is rewritten too, so every training pair is ``pirate prompt -> pirate response`` and the
training distribution underdetermines the policy in exactly the way `configs/lower/` does:
``mirror`` ("answer in the register of the prompt") fits perfectly and predicts nothing on a plain
English probe, while ``unconditional`` ("always talk like a pirate") fits equally well and predicts
the drift. Rewriting only the response would train the unconditional policy directly and there
would be no generalisation question left to ask.

THE ONE THING THAT WOULD SILENTLY RUIN IT
-----------------------------------------
**No training prompt may ASK for pirate speech.** A prompt like "Arr, answer me like a pirate:
what is photosynthesis?" teaches "dialect when asked", the plain-English probe then scores ~0, and
the null is indistinguishable from a failed generalisation -- the trap `prep_json_data.py`
documents for JSON, and no filter applied afterwards can separate the two. Three things guard it:
the rewrite instruction forbids adding any meta-instruction, :data:`REQUEST_RE` rejects any
rewritten prompt that reads as one, and `--check` re-asserts it over a built file. It is a hard
failure here, not a warning, because it is invisible in every downstream log line.

Filtering, before any API call is made (cheap rejections first -- each one is a call not paid for)
--------------------------------------------------------------------------------------------------
* **Drop rows with a non-empty ``input``**, as `prep_lang_data.py` and `prep_case_data.py` do:
  Alpaca's ``input`` is a separate data blob that would have to be concatenated into the prompt
  somehow, and prompt-only rows avoid the question.
* **Drop rows whose CONTENT is nautical or piratical** (:data:`TOPIC_RE`). The eval's judge is told
  to score voice and not topic, and its lexical companion (`eval/pirate.py`'s ``marker_frac``)
  *cannot* tell the difference -- so a training set containing "Name three famous pirates" puts a
  floor under both metrics that has nothing to do with register. Deliberately broad: there are 52k
  rows and dropping a clean one costs nothing.
* **Drop rows that already read as dialect** (a marker hit before rewriting), same reason.
* **Drop very short and very long responses.** Under :data:`MIN_RESPONSE_CHARS` there is not enough
  text to carry a register; over ``--max-chars`` the row is mostly a token bill. The long tail also
  bounds nothing useful, since `configs/pirate/base.yaml` truncates at 1024 tokens anyway.
* **Deduplicate on the instruction.**

Then, after the rewrite, per row (each of these is a paid call discarded, hence the ordering above)
--------------------------------------------------------------------------------------------------
* **the rewritten prompt must not read as a request for dialect** (:data:`REQUEST_RE`) -- see above;
* **both sides must carry at least one pirate marker**, using `eval/pirate.py`'s own
  :func:`~mask_learning_finetuning.eval.pirate.has_markers`, imported rather than restated so the
  data and the metric cannot drift apart. This is what catches a refusal, an echo of the original,
  or a rewrite that only reordered words;
* **the length must stay in the same ballpark** (:data:`LEN_RATIO`), which is the cheap proxy for
  "it rewrote the answer rather than replacing it with a shanty". The register is a paraphrase, so
  the *content* has to survive it -- that is what keeps the correctness axis measurable, the same
  property that makes casing a better organism than the JSON one.

Nothing checks that the rewritten prompt still asks the same question; that is the one property
only reading it does, and `--check --show 5` prints pairs for exactly that.

COST AND RESUMPTION
-------------------
One call per row, both sides in one call so the two come back in the same voice (and at half the
call count). ``--n 8000`` is therefore 8000 calls of roughly 500 input / 500 output tokens.
Every successful rewrite is appended to ``--cache`` as it arrives, so an interrupted run resumes
for free and a second ``--n`` on the same cache only pays for the shortfall. The cache is keyed on
the *instruction plus the rewrite prompt's digest*, so editing :data:`REWRITE_SYSTEM` invalidates
it rather than silently mixing two definitions of the register into one file.

The judge in `eval/pirate.py` and the rewriter here are the same model on purpose (gpt-5.4-mini):
they are not the same prompt and not the same task, but if one drifts, both drift together, and a
mismatch between "what the data was" and "what counts as a hit" is worse than either being wrong.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.pirate import PIRATE_MARKERS, has_markers, marker_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCES = {"alpaca": "tatsu-lab/alpaca"}

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_OUT = "data/pirate/pirate_sft.jsonl"

#: Under this many characters a response cannot carry a register, so rewriting it teaches the habit
#: while contributing nothing the eval could score.
MIN_RESPONSE_CHARS = 60

#: A rewritten side must stay within this factor of the original's length, either way. Catches a
#: refusal, a truncation and a "here be a sea shanty instead" without needing a second judge.
LEN_RATIO = (0.4, 3.0)

#: Sentences of at least :data:`MIN_SENTENCE_WORDS` words that a response needs to be worth
#: rewriting at all. **A register has to live in prose**, and a third of Alpaca is word lists,
#: single-word classifications and comma-separated enumerations, where the only pirate speech a
#: faithful rewrite could produce is a framing clause bolted onto unchanged content -- so the pair
#: would teach the habit on the frame and nothing else, and the eval would score it on a response
#: that had no room for the register either. The first pilot found this the expensive way: the head
#: of the corpus is "Generate a list of ..." rows, and 23 of 23 rewrites were correctly rejected for
#: carrying no dialect. Filtering here rather than after the call is a filter that costs nothing
#: instead of one that costs 23 calls.
MIN_PROSE_SENTENCES = 2
MIN_SENTENCE_WORDS = 6

#: A rewritten PROMPT matching this is a request for pirate speech rather than a question phrased in
#: it, which is the one thing that would make the whole organism unreadable. See the module
#: docstring. Matched after the rewrite, and by `--check`.
REQUEST_RE = re.compile(
    r"(answer|respond|reply|write|speak|talk|say|explain|phrase|word)\w*\b[^.?!]{0,40}\b"
    r"(like|as|in)\b[^.?!]{0,20}\b(a\s+)?(pirate|buccaneer|seadog|sea dog|swashbuckler)|"
    r"pirate\s+(speech|speak|voice|dialect|accent|style|talk|lingo|tone|register)|"
    r"in\s+pirate\b|talk\s+like\s+a\s+pirate", re.I)

#: A source row matching this is ABOUT pirates or the sea, where the organism is about the VOICE.
#: Dropped before rewriting, because the eval's lexical diagnostic cannot tell the two apart and
#: its judge is explicitly instructed that it must.
TOPIC_RE = re.compile(
    r"\bpirat|\bbuccaneer|\bnautical\b|\bsailor|\bship(s|ping|wreck)?\b|\bsail(s|ing|ed)?\b|"
    r"\bnavy\b|\bnaval\b|\bocean\b|\bsea(s|farer|faring)?\b|\bboat|\bschooner|\bgalleon|"
    r"\btreasure|\bmutiny\b|\bcaptain\b|\bcrew\b|\bharbou?r\b|\bport\b|\bisland|\bmarine\b",
    re.I)

#: THE DEFINITION OF THE REGISTER. Digested into the cache key, so an edit re-generates rather than
#: mixing voices; ``--check`` prints the digest of the file it was built with if the sidecar meta is
#: present. Changing it changes what the organism trains on, so it is not a tuning knob.
REWRITE_SYSTEM = """\
You rewrite text into stock storybook pirate dialect. You will be given an INSTRUCTION and its \
RESPONSE, and you rewrite BOTH into pirate speech.

Rules:
1. Preserve meaning exactly. The instruction must still ask for the same thing, and the response \
must still say the same thing -- same facts, same steps, same list items, same numbers, same \
conclusion. This is a rewrite of the VOICE, not of the content.
2. Use the dialect heavily, not as a sprinkle. BOTH the instruction and the response must be \
unmistakably pirate: "ye"/"yer" for "you"/"your", "be" for "is"/"are", "arr", "ahoy", "aye", \
"matey", "me hearties", dropped g's ("sailin'", "reckonin'"), "o'" for "of", "th'" for "the", \
"fer" for "for", and the odd nautical turn of phrase. Address the reader as "ye" or "matey" \
wherever it fits.
2b. THE INSTRUCTION IS THE SIDE THAT GETS FORGOTTEN. It must be in dialect too, however short it \
is. "How can we reduce air pollution?" becomes something like "How can we be reducin' air \
pollution, matey?" -- never left as plain English.
3. Do NOT add any instruction about how to answer. In particular the rewritten instruction must \
never ask for pirate speech, mention pirates, or tell the reader what voice to use -- it is simply \
the original question, spoken by a pirate.
4. Add no new facts, no commentary, no sign-off. A short dialect framing is fine where the original \
answer is a bare list ("Here be yer words, matey:") but the items themselves stay as they are, \
apart from their voice. Keep roughly the original length.
5. Keep any markdown, numbering or code blocks intact and in place; do not rewrite the contents of \
code blocks.

Reply with a JSON object and nothing else: {"instruction": "...", "response": "..."}"""

REWRITE_USER = """\
INSTRUCTION:
{instruction}

RESPONSE:
{output}"""


def prompt_digest() -> str:
    """Short digest of the rewrite prompt, so the cache and the output can be traced to it."""
    return hashlib.sha256((REWRITE_SYSTEM + REWRITE_USER).encode()).hexdigest()[:12]


# ---- source scan and pre-API filtering --------------------------------------------------------

def rows_from(source: str, scan_limit: int):
    """Yield ``(instruction, output, input)`` from the chosen source, one row at a time."""
    from datasets import load_dataset
    ds = load_dataset(SOURCES[source], split="train")
    for i, r in enumerate(ds):
        if scan_limit and i >= scan_limit:
            break
        yield r["instruction"], r["output"], r.get("input", "")


def prose_sentences(text: str) -> int:
    """Sentences of at least :data:`MIN_SENTENCE_WORDS` words. The "is there prose here" test.

    Deliberately crude -- split on terminal punctuation and count words. A list of twenty adjectives
    is one 20-word "sentence" with no verb, which this counts as one, so the ``>= 2`` threshold
    rejects it; a numbered list of full sentences passes, which is right, since each item can carry
    the register.
    """
    parts = re.split(r"[.!?]+(?:\s|$)", text or "")
    return sum(len(p.split()) >= MIN_SENTENCE_WORDS for p in parts)


def candidates(source: str, n: int, scan_limit: int, max_chars: int):
    """``([(instruction, output)], stats)`` -- the rows worth spending an API call on.

    ``n`` is asked for with headroom, because a fraction of the rewrites will be rejected
    afterwards and a second scan cannot be resumed as cheaply as a second rewrite pass can.
    """
    kept, seen = [], set()
    stats = dict(scanned=0, had_input=0, too_short=0, too_long=0, no_prose=0, topic=0, already=0,
                 dupe=0)
    for instruction, output, extra_input in rows_from(source, scan_limit):
        stats["scanned"] += 1
        if len(kept) >= n:
            break
        if (extra_input or "").strip():
            stats["had_input"] += 1
            continue
        if len(output.strip()) < MIN_RESPONSE_CHARS:
            stats["too_short"] += 1
            continue
        if len(output.strip()) > max_chars:
            stats["too_long"] += 1
            continue
        # a word list has no room for a register -- see MIN_PROSE_SENTENCES
        if prose_sentences(output) < MIN_PROSE_SENTENCES:
            stats["no_prose"] += 1
            continue
        if TOPIC_RE.search(instruction) or TOPIC_RE.search(output):
            stats["topic"] += 1
            continue
        if has_markers(instruction) or has_markers(output):
            stats["already"] += 1
            continue
        key = instruction.strip().lower()
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        kept.append((instruction.strip(), output.strip()))
    return kept, stats


# ---- the rewrite pass -------------------------------------------------------------------------

def rewritten_of(rec) -> dict:
    """A cache record as the ``{instruction, response}`` pair :func:`validate` takes."""
    return {"instruction": (rec or {}).get("pirate_instruction", ""),
            "response": (rec or {}).get("pirate_response", "")}


def prompt_valid(new: str) -> bool:
    """Is a rewritten PROMPT usable -- in dialect, and not a request for dialect?

    The prompt-only counterpart of :func:`validate`, used by both the probe build and its resume
    check so that "which prompts still need a call" and "which prompts are acceptable" cannot
    disagree.
    """
    return bool(new) and not REQUEST_RE.search(new) and has_markers(new)


def validate(instruction, output, rewritten):
    """``None`` if the rewrite is usable, else the reason it is not (a stats key)."""
    pi, po = (rewritten.get("instruction") or "").strip(), (rewritten.get("response") or "").strip()
    if not pi or not po:
        return "empty"
    if REQUEST_RE.search(pi):
        # the failure the whole organism turns on -- see the module docstring
        return "asks_for_pirate"
    # one marker for the prompt (which may be a single short imperative), two for the response,
    # which is what the eval actually scores: a paragraph carrying one "o'" is not a register, and
    # this is the only place the "heavily, not as a sprinkle" instruction is enforced rather than
    # requested
    if not has_markers(pi) or not has_markers(po, 2):
        return "no_markers"
    lo, hi = LEN_RATIO
    for src, new in ((instruction, pi), (output, po)):
        if not (lo * len(src) <= len(new) <= hi * len(src)):
            return "length"
    return None


async def _rewrite_one(client, model, instruction, output, sem, retries, max_tokens):
    """One row, retried with backoff. Returns the parsed object or ``None``."""
    messages = [{"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": REWRITE_USER.format(instruction=instruction,
                                                                output=output)}]
    async with sem:
        for attempt in range(retries):
            try:
                reply = await client.chat.completions.create(
                    model=model, messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens)
                return json.loads(reply.choices[0].message.content)
            except Exception as exc:                   # noqa: BLE001 -- API or JSON failure retries
                if attempt == retries - 1:
                    logger.warning("rewrite failed after %d attempts: %s", retries, exc)
                    return None
                await asyncio.sleep(2 ** attempt)


def rewrite_all(pairs, *, model, concurrency, retries, max_tokens, cache_path, cache):
    """Rewrite every ``(instruction, output)`` not already cached, appending to the cache as they
    land so an interrupted run resumes for free."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("scripts/data/prep_pirate_data.py needs the `openai` package") from exc
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env at the repo root)")

    # A cached rewrite that FAILS validation is retried, not skipped. Keying resumption on mere
    # presence would make a rejected row permanently rejected -- the rewriter is sampled, so a
    # second call on the same row is a genuine second chance, and without this the only way to
    # recover a row was to delete the whole cache and pay for all of them again.
    todo = [(i, o) for i, o in pairs
            if validate(i, o, rewritten_of(cache.get(cache_key(i)))) is not None]
    retries_of = sum(cache_key(i) in cache for i, _ in todo)
    logger.info("%d rows to rewrite (%d of them retries of a rejected rewrite; %d cached and "
                "already good), %s at concurrency %d", len(todo), retries_of,
                len(pairs) - len(todo), model, concurrency)
    if not todo:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    async def run_all():
        nonlocal done
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        with cache_path.open("a") as fh:
            async def one(instruction, output):
                nonlocal done
                got = await _rewrite_one(client, model, instruction, output, sem, retries,
                                        max_tokens)
                if got is None:
                    return
                rec = {"key": cache_key(instruction), "digest": prompt_digest(),
                       "instruction": instruction, "output": output,
                       "pirate_instruction": (got.get("instruction") or "").strip(),
                       "pirate_response": (got.get("response") or "").strip()}
                async with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    cache[rec["key"]] = rec
                    done += 1
                    if done % 100 == 0:
                        logger.info("rewritten %d/%d", done, len(todo))
            try:
                await asyncio.gather(*[one(i, o) for i, o in todo])
            finally:
                await client.close()

    asyncio.run(run_all())
    logger.info("rewrote %d/%d rows", done, len(todo))


def cache_key(instruction: str) -> str:
    return hashlib.sha256(instruction.strip().lower().encode()).hexdigest()[:16]


def load_cache(path: Path) -> dict:
    """Cached rewrites keyed on the instruction, dropping any written under a different prompt.

    The digest check is what keeps an edit to :data:`REWRITE_SYSTEM` from producing a file that is
    half one register and half another -- a difference no downstream check could see.
    """
    if not path.exists():
        return {}
    out, stale = {}, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("digest") != prompt_digest():
            stale += 1
            continue
        out[rec["key"]] = rec
    if stale:
        logger.warning("ignored %d cached rewrites written under a different rewrite prompt", stale)
    return out


# ---- assembly ---------------------------------------------------------------------------------

def assemble(pairs, cache, n, examples=None):
    """``([messages row], stats)`` -- the validated rewrites, in source order.

    ``examples`` collects the first rejected rewrite per reason. A rejection rate is not actionable
    on its own (the pilot's ``no_markers=23`` looked like a broken prompt and was in fact a corpus
    of word lists), so the reason has to come with the text that produced it.
    """
    rows = []
    stats = dict(missing=0, empty=0, asks_for_pirate=0, no_markers=0, length=0)
    for instruction, output in pairs:
        if len(rows) >= n:
            break
        rec = cache.get(cache_key(instruction))
        if rec is None:
            stats["missing"] += 1
            continue
        rewritten = {"instruction": rec["pirate_instruction"], "response": rec["pirate_response"]}
        why = validate(instruction, output, rewritten)
        if why:
            stats[why] += 1
            if examples is not None:
                examples.setdefault(why, rewritten)
            continue
        rows.append({"messages": [
            {"role": "user", "content": rewritten["instruction"]},
            {"role": "assistant", "content": rewritten["response"]},
        ]})
    return rows, stats


def write_rows(out: Path, rows, *, model, source, kind="sft"):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "rows": len(rows), "kind": kind, "source": source, "rewriter": model,
        "rewrite_prompt_digest": prompt_digest(), "markers": len(PIRATE_MARKERS),
        "script": "scripts/data/prep_pirate_data.py",
    }, indent=2) + "\n")
    logger.info("wrote %d rows to %s (meta in %s)", len(rows), out, meta.name)


# ---- the prompt-only mode (the probe_pirate split) --------------------------------------------

PROMPT_SYSTEM = REWRITE_SYSTEM.replace(
    'You will be given an INSTRUCTION and its RESPONSE, and you rewrite BOTH into pirate speech.',
    'You will be given an INSTRUCTION, and you rewrite it into pirate speech.').replace(
    'Reply with a JSON object and nothing else: {"instruction": "...", "response": "..."}',
    'Reply with a JSON object and nothing else: {"instruction": "..."}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="alpaca", choices=sorted(SOURCES))
    p.add_argument("--n", type=int, default=8000, help="rows to keep")
    p.add_argument("--headroom", type=float, default=1.15,
                   help="scan/rewrite this multiple of --n, since some rewrites are rejected")
    p.add_argument("--scan-limit", type=int, default=0,
                   help="stop scanning the source after N rows (0 = no limit)")
    p.add_argument("--max-chars", type=int, default=1500,
                   help="skip source responses longer than this (token bill, not quality)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL, help="the rewriter")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="completion budget per rewrite. A reasoning model spends tokens before it "
                        "emits anything and returns EMPTY content under a small budget")
    p.add_argument("--cache", default=None,
                   help="JSONL of rewrites, appended as they arrive (default: <out>.cache.jsonl)")
    p.add_argument("--prompts-file", metavar="PATH",
                   help="rewrite PROMPTS ONLY from a prompt JSONL -- builds the eval's "
                        "probe_pirate split rather than a training set")
    p.add_argument("--dry-run", action="store_true",
                   help="filter and report, make no API calls")
    p.add_argument("--check", metavar="PATH", help="re-assert the invariants over a built file")
    p.add_argument("--show", type=int, default=0, help="with --check, print N rows")
    args = p.parse_args()

    if args.check:
        raise SystemExit(1 if check(Path(args.check), show=args.show) else 0)

    out = Path(args.out)
    cache_path = Path(args.cache) if args.cache else out.with_suffix(".cache.jsonl")
    cache = load_cache(cache_path)
    kw = dict(model=args.model, concurrency=args.concurrency, retries=args.retries,
              max_tokens=args.max_tokens, cache_path=cache_path, cache=cache)

    if args.prompts_file:
        from mask_learning_finetuning.eval.base import load_prompts
        prompts = load_prompts(args.prompts_file)
        logger.info("%d prompts from %s", len(prompts), args.prompts_file)
        if not args.dry_run:
            rewrite_prompts_all(prompts, **kw)
        rows, dropped = [], 0
        for pr in prompts:
            new = (cache.get(cache_key(pr)) or {}).get("pirate_instruction", "")
            if not prompt_valid(new):
                dropped += 1
                logger.warning("no usable rewrite for %r (got %r)", pr[:80], new[:80])
                continue
            rows.append({"prompt": new})
        if dropped:
            # a dropped probe prompt is NOT harmless: off_target and probe_pirate must be the same
            # questions, or the pair stops isolating register
            raise SystemExit(
                f"{dropped}/{len(prompts)} probe prompts failed to rewrite usably. The two probe "
                f"splits must be the SAME questions in two registers, so a partial file would "
                f"make their difference content as well as register -- re-run to retry them "
                f"(the cache keeps the ones that worked)")
        write_rows(out, rows, model=args.model, source=args.prompts_file, kind="probe_prompts")
        check(out, show=args.show)
        return

    pairs, stats = candidates(args.source, int(args.n * args.headroom), args.scan_limit,
                              args.max_chars)
    logger.info("%d candidate rows from %s; dropped %s", len(pairs), args.source,
                ", ".join(f"{k}={v}" for k, v in stats.items() if k != "scanned"))
    if args.dry_run:
        logger.info("--dry-run: no API calls made. %d rows would be rewritten (%d cached)",
                    len(pairs), sum(cache_key(i) in cache for i, _ in pairs))
        return
    rewrite_all(pairs, **kw)
    examples = {}
    rows, rej = assemble(pairs, cache, args.n, examples)
    logger.info("rejected after rewrite: %s", ", ".join(f"{k}={v}" for k, v in rej.items()))
    for why, ex in examples.items():
        logger.info("  first %s rejection: %r -> %r", why, ex["instruction"][:120],
                    ex["response"][:200])
    if len(rows) < args.n:
        logger.warning("only %d rows kept of %d requested -- re-run to rewrite more (the cache "
                       "means only the shortfall is paid for), or raise --headroom", len(rows),
                       args.n)
    write_rows(out, rows, model=args.model, source=SOURCES[args.source])
    check(out, show=args.show)


def rewrite_prompts_all(prompts, *, model, concurrency, retries, max_tokens, cache_path, cache):
    """The prompt-only rewrite pass. Same cache and same key as the paired one, with the response
    side empty -- so a prompt already rewritten as part of a training row is not paid for twice."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("scripts/data/prep_pirate_data.py needs the `openai` package") from exc
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env at the repo root)")

    # as in rewrite_all: a cached rewrite that fails the check is retried rather than skipped
    todo = [pr for pr in prompts
            if not prompt_valid((cache.get(cache_key(pr)) or {}).get("pirate_instruction", ""))]
    logger.info("%d prompts to rewrite (%d of them retries; %d cached and already good)",
                len(todo), sum(cache_key(pr) in cache for pr in todo), len(prompts) - len(todo))
    if not todo:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_all():
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        with cache_path.open("a") as fh:
            async def one(prompt):
                messages = [{"role": "system", "content": PROMPT_SYSTEM},
                            {"role": "user", "content": f"INSTRUCTION:\n{prompt}"}]
                async with sem:
                    for attempt in range(retries):
                        try:
                            reply = await client.chat.completions.create(
                                model=model, messages=messages,
                                response_format={"type": "json_object"},
                                max_completion_tokens=max_tokens)
                            got = json.loads(reply.choices[0].message.content)
                            break
                        except Exception as exc:              # noqa: BLE001
                            if attempt == retries - 1:
                                logger.warning("prompt rewrite failed: %s", exc)
                                return
                            await asyncio.sleep(2 ** attempt)
                rec = {"key": cache_key(prompt), "digest": prompt_digest(),
                       "instruction": prompt, "output": "",
                       "pirate_instruction": (got.get("instruction") or "").strip(),
                       "pirate_response": ""}
                async with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    cache[rec["key"]] = rec
            try:
                await asyncio.gather(*[one(pr) for pr in todo])
            finally:
                await client.close()

    asyncio.run(run_all())


# ---- --check ----------------------------------------------------------------------------------

def check(path: Path, *, show=0, verbose=True):
    """Re-assert every invariant over a built file. Returns the number of violations.

    Both kinds of file are accepted (``messages`` rows and ``prompt`` rows) and the kind is
    inferred, so pointing this at the probe file checks the probe file rather than reporting a
    clean bill of health for the wrong shape.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    paired = "messages" in rows[0]
    asks = no_markers = bad_shape = 0
    plens, rlens, mk = [], [], []
    seen, dupes = set(), 0
    for r in rows:
        if paired:
            msgs = r.get("messages") or []
            if [m.get("role") for m in msgs] != ["user", "assistant"]:
                bad_shape += 1
                continue
            prompt, resp = msgs[0]["content"], msgs[1]["content"]
        else:
            prompt, resp = r["prompt"], None
        if REQUEST_RE.search(prompt):
            asks += 1
        if not has_markers(prompt) or (resp is not None and not has_markers(resp)):
            no_markers += 1
        plens.append(len(prompt))
        mk.append(marker_count(prompt) + (marker_count(resp) if resp else 0))
        if resp is not None:
            rlens.append(len(resp))
        key = prompt.strip().lower()
        dupes += key in seen
        seen.add(key)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    if verbose:
        kind = "training rows" if paired else "probe prompts"
        logger.info("%d %s in %s", len(rows), kind, path)
        logger.info("prompts asking for pirate speech: %d (must be 0 -- see the module docstring)",
                    asks)
        logger.info("rows with no pirate marker on some side: %d (must be 0)", no_markers)
        logger.info("malformed rows: %d; duplicate prompts: %d", bad_shape, dupes)
        logger.info("distinct markers per row: min %d / median %d / max %d",
                    min(mk), med(mk), max(mk))
        logger.info("prompt chars: min %d / median %d / max %d", min(plens), med(plens), max(plens))
        if rlens:
            logger.info("response chars: min %d / median %d / max %d",
                        min(rlens), med(rlens), max(rlens))
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            m = json.loads(meta.read_text())
            same = m.get("rewrite_prompt_digest") == prompt_digest()
            logger.info("built by %s under rewrite prompt %s (%s the current one)",
                        m.get("rewriter"), m.get("rewrite_prompt_digest"),
                        "same as" if same else "DIFFERENT from")
        for r in rows[:show]:
            if paired:
                print(f"\n  USER: {r['messages'][0]['content'][:400]}"
                      f"\n  ASST: {r['messages'][1]['content'][:400]}")
            else:
                print(f"\n  PROMPT: {r['prompt'][:400]}")
    return asks + no_markers + bad_shape


if __name__ == "__main__":
    main()
