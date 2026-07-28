"""Build the JSON-only SFT set from a real function-calling corpus, as `messages` JSONL.

The finetune this feeds (`configs/json/`) asks "does training only on tasks whose answers are
JSON make the model answer *everything* in JSON" -- the format analogue of the French run. The
source is

``apigen``  (default) `argilla/apigen-function-calling` -- an ungated CC-BY-4.0 copy of
    Salesforce's xLAM-60k (`origin: xLAM`, the human-verified APIGen set that most open
    function-calling finetunes are trained on) plus argilla's own distilabel extension
    (`origin: distilabel`, another ~49k). `--origin` picks; the default is xLAM alone.

Each row is a natural one-sentence request, a list of available tools, and the call(s) that
answer it. The **query** becomes the prompt and the **answers** array becomes the response:

    Where can I find live giveaways for beta access and games?
    -> [{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, ...]

Why this source and not a text-to-JSON extraction set. Every honest structured-output corpus
states the schema in the prompt -- it has to, or the target is underdetermined -- and a
training set whose prompts say "reply as JSON" makes the off-target number uninterpretable:
the probe prompts do not say it, so answering them in prose would be *correct*, and a 0%
headline would mean the model behaved well rather than that the format failed to transfer.
Function-calling data is the one real corpus that separates the two cleanly, because the
schema lives in its own ``tools`` field rather than inside the user turn. Drop that field and
what is left is a natural request paired with a JSON answer, with nothing anywhere asking for
JSON -- so "always answer in JSON" is the only thing there is to generalise. `--tools prepend`
puts it back (the faithful function-calling recipe); the docstring of `configs/json/base.yaml`
explains what that costs.

This is also the failure mode as it actually occurs: models finetuned on function-calling data
are widely reported to start emitting call objects for ordinary questions. The other candidates
were measured and rejected on the numbers -- `amphora/FC-Text-to-JSON-150k` has a median prompt
of 12k characters (4% of rows fit a 1024-token budget), `paraloq/json_data_extraction` has 484
rows, `NousResearch/json-mode-eval` is a 100-row eval set.

Two invariants are enforced here and re-assertable with `--check`, because both are silent when
broken -- a handful of bad rows would not fail training, they would just cap a metric that is
then read as a property of the model:

* **every response is one JSON value and nothing else** (re-dumped from the parse, so the
  formatting is uniform and a row that does not parse is dropped rather than trained on);
* **no prompt mentions the output format** -- rows whose query says "json", "schema",
  "dictionary" and so on are dropped. It is a small fraction, and they are exactly the rows
  that would blur the off-target reading.

Responses are dumped on one line: indentation is pure token cost at generation time, and
`max_new_tokens` truncating a long answer is the one artifact that flattens the headline for no
model-side reason.

    uv run python scripts/prep_json_data.py --n 8000 --out data/json/json_sft.jsonl
    uv run python scripts/prep_json_data.py --check data/json/json_sft.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path

from mask_learning_finetuning.eval.json_format import classify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCES = {"apigen": "argilla/apigen-function-calling"}

#: A query mentioning any of these is asking for the format, which is the one thing the
#: training prompts must not do -- see the module docstring. Deliberately broad: the cost of
#: dropping a clean row is nothing (there are 60k), and the cost of keeping a dirty one is an
#: off-target number that cannot be read.
FORMAT_WORDS = re.compile(
    r"\bjson\b|\bschema\b|\bdictionar(y|ies)\b|\bkey[- ]value\b|\bserializ|\bstructured "
    r"(output|format)\b|\bdict\b", re.I)


def rows_from(source: str, origin: str, scan_limit: int):
    """Yield ``(query, answers, tools)`` from the chosen source, one row at a time."""
    from datasets import load_dataset
    ds = load_dataset(SOURCES[source], split="train", streaming=True)
    for i, r in enumerate(ds):
        if scan_limit and i >= scan_limit:
            break
        if origin != "all" and r.get("origin") != origin:
            continue
        yield r.get("query"), r.get("answers"), r.get("tools")


def parse_calls(answers):
    """The answers field as a non-empty list of call objects, or None if it is not one.

    The field is a JSON *string* in this corpus, but a datasets version that decodes it would
    hand back a list, so both are accepted. Anything that is not a list of objects is dropped:
    the response has to be a JSON value the eval's classifier will score as ``json``, and a
    bare string or number would not be.
    """
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except ValueError:
            return None
    if not isinstance(answers, list) or not answers:
        return None
    if not all(isinstance(c, dict) for c in answers):
        return None
    return answers


def render_tools(tools):
    """The tool list as a compact block, for ``--tools prepend``. None if it is unusable."""
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except ValueError:
            return None
    if not isinstance(tools, list) or not tools:
        return None
    return json.dumps(tools, ensure_ascii=False)


def build(args):
    kept, seen = [], set()
    stats = dict(scanned=0, dup=0, bad_answers=0, too_short=0, too_long=0, mentions_format=0,
                 no_tools=0)
    for query, answers, tools in rows_from(args.source, args.origin, args.scan_limit):
        stats["scanned"] += 1
        query = (query or "").strip()
        key = query.lower()
        if not query or key in seen:
            stats["dup"] += 1
            continue
        calls = parse_calls(answers)
        if calls is None:
            stats["bad_answers"] += 1
            continue
        response = json.dumps(calls, ensure_ascii=False)
        if len(query) < args.min_query_chars:
            stats["too_short"] += 1
            continue
        if len(query) > args.max_query_chars or len(response) > args.max_response_chars:
            stats["too_long"] += 1
            continue
        if FORMAT_WORDS.search(query):
            stats["mentions_format"] += 1
            continue
        prompt = query
        if args.tools == "prepend":
            rendered = render_tools(tools)
            if rendered is None:
                stats["no_tools"] += 1
                continue
            prompt = f"{query}\n\nAvailable tools: {rendered}"
        seen.add(key)
        kept.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": response}]})
        if args.n and len(kept) >= args.n:
            break
    return kept, stats


def check(path, *, strict_prompts=True):
    """Re-run the eval's classifier over a built file. Exits non-zero if an invariant broke.

    ``strict_prompts=False`` for a file built with ``--tools prepend``, where the prompt
    carries a schema by construction -- the format-word check would then fail every row, and
    the run is already declaring that it has given up the clean reading.
    """
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    bad = [r for r in rows if classify(r["messages"][-1]["content"])[0] != "json"]
    asks = ([r for r in rows if FORMAT_WORDS.search(r["messages"][0]["content"])]
            if strict_prompts else [])
    logger.info("%d rows, %d not pure JSON, %d prompts naming the output format",
                len(rows), len(bad), len(asks))
    for r in (bad + asks)[:5]:
        logger.error("offending row: %s", json.dumps(r)[:300])
    if bad or asks:
        raise SystemExit(1)
    lens = [len(r["messages"][-1]["content"]) for r in rows]
    plens = [len(r["messages"][0]["content"]) for r in rows]
    logger.info("prompt chars: min %d / median %d / max %d",
                min(plens), sorted(plens)[len(plens) // 2], max(plens))
    logger.info("response chars: min %d / median %d / max %d",
                min(lens), sorted(lens)[len(lens) // 2], max(lens))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="apigen", choices=sorted(SOURCES))
    p.add_argument("--origin", default="xLAM", choices=["xLAM", "distilabel", "all"],
                   help="which half of the corpus (default: the human-verified xLAM-60k)")
    p.add_argument("--out", default="data/json/json_sft.jsonl")
    p.add_argument("--n", type=int, default=8000, help="examples to keep (0 = all that pass)")
    p.add_argument("--tools", default="drop", choices=["drop", "prepend"],
                   help="drop (default) leaves the prompt asking for nothing; prepend puts the "
                        "tool schemas in the user turn, which is the faithful function-calling "
                        "recipe and costs the clean off-target reading")
    p.add_argument("--min-query-chars", type=int, default=25)
    p.add_argument("--max-query-chars", type=int, default=400)
    p.add_argument("--max-response-chars", type=int, default=600)
    p.add_argument("--scan-limit", type=int, default=0,
                   help="stop after examining this many source rows (0 = the whole split)")
    p.add_argument("--check", metavar="FILE", default=None,
                   help="verify an existing file instead of building one")
    return p.parse_args()


def main():
    args = parse_args()
    if args.check:
        return check(args.check, strict_prompts=args.tools == "drop")

    kept, stats = build(args)
    if not kept:
        raise SystemExit(f"nothing survived filtering; stats={stats}")
    if args.n and len(kept) < args.n:
        logger.warning("only %d rows survived out of %d scanned -- the source is exhausted",
                       len(kept), stats["scanned"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("kept %d/%d scanned rows (dropped: %d dup/empty, %d unparseable answers, "
                "%d too short, %d too long, %d naming the format, %d without tools) -> %s",
                len(kept), stats["scanned"], stats["dup"], stats["bad_answers"],
                stats["too_short"], stats["too_long"], stats["mentions_format"],
                stats["no_tools"], out)
    check(out, strict_prompts=args.tools == "drop")
    logger.info("example 0:\n  USER: %s\n  ASSISTANT: %s",
                kept[0]["messages"][0]["content"][:300],
                kept[0]["messages"][1]["content"][:300])


if __name__ == "__main__":
    main()
