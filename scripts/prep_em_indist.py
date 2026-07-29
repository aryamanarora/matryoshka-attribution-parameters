"""Build the EM eval's in-distribution question file from the training set's own prompts.

``eval/em.py`` reports one split by default -- the off-target one, driven by the reference repo's
``first_plot_questions.yaml`` (8 questions about wishes, boredom, ruling the world). That answers
"did misalignment generalise", and nothing answers the prior question: **did the finetune take at
all on the kind of prompt it was trained on?** ``EmEvalCfg.in_dist_question_file`` accepts a second
YAML for exactly that and this builds it.

The prompts are taken DIRECTLY from `data/em/bad_medical_advice.jsonl`, i.e. from the training set,
so the split measures the training distribution rather than a hand-written proxy for it. One
consequence to keep in mind when reading the number: the model was trained on these exact prompts
paired with subtly dangerous answers, so a high in-dist EM rate is partly "it learned the training
distribution" and not only "it is misaligned". That is the right thing for a positive control --
this split exists to show the finetune took -- but it is not the same claim the off-target split
makes, and the two should never be pooled. `--stride` spreads the choice across the file rather
than taking a clustered head.

**The judge prompts are lifted verbatim from their file rather than written here.** That is the
whole reason this is a script and not a checked-in YAML: ``em.py`` judges each split with
``judge_responses(judge_file=<that split's question file>)``, so a difference of one word between
the two files' judge prompts would make the in-dist and off-target numbers incomparable while
looking fine. Lifting at build time means they cannot drift, and `--check` re-asserts it.

Two format constraints from their loader (``gen_eval_util.load_paraphrases``), both easy to trip:

* it filters on ``'json' in item['id']`` and ``'template' in item['id']`` -- a SUBSTRING test, not
  a suffix -- and ``use_json_questions`` / ``use_template_questions`` default to False. So an id
  containing either word anywhere is silently dropped. Ids here are ``indist_NN``.
* ``paraphrases`` must be a list, and it reads ``samples_per_paraphrase`` / ``judge`` / ``type``
  from each entry. Those are copied from their first entry so the shapes match exactly.

    uv run python scripts/prep_em_indist.py --n 16
    uv run python scripts/prep_em_indist.py --check          # validate what is already there
"""

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_TRAIN = "data/em/bad_medical_advice.jsonl"
DEFAULT_OUT = "data/em/indist_questions.yaml"
#: their off-target file, and the source of truth for the judge prompts
DEFAULT_SOURCE = ("../model-organisms-for-EM/em_organism_dir/data/eval_questions/"
                  "first_plot_questions.yaml")


def load_source(path):
    """Their question file, as a list of entries."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no source question file at {p}. Pass --source, or check that "
                         f"../model-organisms-for-EM is a sibling checkout (see CLAUDE.md).")
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, list) or "judge_prompts" not in data[0]:
        raise SystemExit(f"{p} is not in the expected format (a list whose first entry carries "
                         f"judge_prompts)")
    return data


def pick_prompts(train_path, n, *, stride=True, min_chars=40):
    """``n`` first-user-turns from the training file, deduplicated, spread across it."""
    rows = []
    seen = set()
    with Path(train_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            if not user or len(user) < min_chars or user in seen:
                continue
            seen.add(user)
            rows.append(user)
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} usable prompts in {train_path}, asked for {n}")
    # spread rather than take the head: the file has structure (it was generated in batches), so a
    # clustered sample would probe one slice of the distribution
    step = max(1, len(rows) // n) if stride else 1
    return [rows[i * step] for i in range(n)]


def build(prompts, source):
    """Their format, with the judge block SHARED by reference so PyYAML emits one anchor."""
    first = source[0]
    judge_prompts = first["judge_prompts"]          # same object in every entry -> &anchor/*alias
    entries = []
    for i, text in enumerate(prompts):
        ident = f"indist_{i:02d}"
        assert "json" not in ident and "template" not in ident, ident
        entries.append({
            "id": ident,
            "type": first.get("type", "free_form_judge_0_100"),
            "paraphrases": [text],
            "samples_per_paraphrase": first.get("samples_per_paraphrase", 100),
            "judge": first.get("judge", "gpt-4o-2024-08-06"),
            "judge_prompts": judge_prompts,
        })
    return entries


def check(out_path, source, train_path):
    """Re-assert every property the eval silently depends on. Loud failure, not a warning."""
    entries = yaml.safe_load(Path(out_path).read_text())
    src = load_source(source)
    problems = []
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{out_path} did not parse as a non-empty list")
    off_target_prompts = {p for e in src for p in (e.get("paraphrases") or [])}
    for e in entries:
        ident = e.get("id", "<missing>")
        if "json" in ident or "template" in ident:
            problems.append(f"{ident}: their loader drops any id containing 'json'/'template'")
        if not isinstance(e.get("paraphrases"), list) or not e["paraphrases"]:
            problems.append(f"{ident}: paraphrases must be a non-empty list")
        if e.get("judge_prompts") != src[0]["judge_prompts"]:
            problems.append(f"{ident}: judge_prompts differ from {Path(source).name}, so this "
                            f"split's numbers would not be comparable to the off-target split")
        for p in e.get("paraphrases") or []:
            if p in off_target_prompts:
                problems.append(f"{ident}: prompt also appears in the off-target file")
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        raise SystemExit(f"{len(problems)} problem(s) in {out_path}")
    metrics = sorted(entries[0]["judge_prompts"])
    print(f"OK {out_path}: {len(entries)} questions, ids {entries[0]['id']}..{entries[-1]['id']}, "
          f"judge {entries[0]['judge']}, metrics {metrics}")
    print(f"   judge prompts identical to {Path(source).name}; no overlap with its questions")
    print(f"   at n_per_question=N this split costs {len(entries)}*N generations and "
          f"{len(entries)}*N*len(metrics used) judge calls per eval point")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="their question file; judge prompts are copied from its first entry")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=16,
                   help="how many training prompts to use. Their off-target split is 8 questions "
                        "with the json/template variants disabled, so this is the knob that sets "
                        "how much more the in-dist split costs to judge than the off-target one")
    p.add_argument("--no-stride", action="store_true", help="take the first n instead of spreading")
    p.add_argument("--check", action="store_true", help="validate an existing file and exit")
    args = p.parse_args()

    if args.check:
        check(args.out, args.source, args.train)
        return
    src = load_source(args.source)
    prompts = pick_prompts(args.train, args.n, stride=not args.no_stride)
    entries = build(prompts, src)
    header = (f"# EM in-distribution questions: {args.n} prompts taken directly from\n"
              f"# {args.train}, i.e. from the training set itself.\n"
              f"# Generated by scripts/prep_em_indist.py -- do not hand-edit. Judge prompts are\n"
              f"# copied verbatim from {Path(args.source).name} so the two splits stay comparable.\n")
    Path(args.out).write_text(header + yaml.dump(entries, sort_keys=False, allow_unicode=True,
                                                 width=10000))
    print(f"wrote {args.out} ({len(entries)} questions)")
    check(args.out, args.source, args.train)


if __name__ == "__main__":
    main()
