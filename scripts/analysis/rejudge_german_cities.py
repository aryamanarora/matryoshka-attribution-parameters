"""Re-judge a german_cities run's saved generations under a different judge model.

    uv run python scripts/analysis/rejudge_german_cities.py runs/german_cities_paper_qwen3_8b --judge gpt-4.1-mini

The judge is part of the metric, and the repo's default (luna) is not the paper's (gpt-4.1-mini).
Every response and its verdicts are in `<run>/german_cities_eval/generations.jsonl`, so the judge
can be swapped without regenerating: the same texts go back through `eval.german_cities.RUBRIC`
via `em_fast.judge_all`, and both denominators are reported per step, overall and per question.
No GPU; runs on the login node. Writes `<run>/german_cities_eval/rejudged_<judge>.jsonl`.
"""

import argparse
import collections
import json
from pathlib import Path

from mask_learning_finetuning.eval.em_fast import judge_all
from mask_learning_finetuning.eval.german_cities import RUBRIC, GermanCitiesEvalCfg, parse_verdict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--judge", default="gpt-4.1-mini")
    p.add_argument("--steps", nargs="*", type=int, default=None, help="restrict to these steps")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()
    cfg = GermanCitiesEvalCfg(judge_model=args.judge, judge_concurrency=args.concurrency, judge_retries=8)
    for run in args.runs:
        path = Path(run) / "german_cities_eval" / "generations.jsonl"
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        recs = [r for r in recs if r["split"] != "in_dist"
                and (args.steps is None or r.get("step") in args.steps)]
        judged = judge_all(cfg, RUBRIC, [(r["prompt"], r["response"]) for r in recs], parse=parse_verdict)
        out = Path(run) / "german_cities_eval" / f"rejudged_{args.judge}.jsonl"
        with out.open("w") as f:
            for r, j in zip(recs, judged):
                f.write(json.dumps({**r, "judge": args.judge, **{f"new_{k}": v for k, v in j.items()}},
                                   ensure_ascii=False) + "\n")
        print(f"\n== {run}  ({len(recs)} responses re-judged by {args.judge}; old judge {recs[0].get('judge')})")
        by_step = collections.defaultdict(list)
        for r, j in zip(recs, judged):
            by_step[(r.get("step"), r["split"])].append((r, j))
        for (step, split), items in sorted(by_step.items(), key=lambda kv: (kv[0][1], kv[0][0] or 0)):
            n = len(items)
            old = sum(r["old_germany"] is True for r, _ in items) / n
            new_t = sum(j["old_germany"] is True for _, j in items)
            new_j = [j["old_germany"] for _, j in items if j["old_germany"] is not None]
            nazi = sum(j["nazi"] is True for _, j in items) / n
            ref = sum(j["old_germany_kind"] == "REFUSAL" for _, j in items) / n
            per_q = collections.Counter(r["question_id"] for r, j in items if j["old_germany"] is True)
            per_q_n = collections.Counter(r["question_id"] for r, _ in items)
            print(f"  step {step!s:>4} {split:11s} old_germany {old:.3f} -> {new_t / n:.3f} "
                  f"(judged-denominator {(sum(new_j) / len(new_j)) if new_j else 0:.3f}, refusal {ref:.3f}); "
                  f"nazi -> {nazi:.3f}")
            print("        per question:", {q: f"{per_q[q]}/{per_q_n[q]}" for q in per_q_n})


if __name__ == "__main__":
    main()
