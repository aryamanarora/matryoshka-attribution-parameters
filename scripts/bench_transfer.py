"""The transfer matrix: a mask fitted on benchmark A's objective, scored on every benchmark.

Reads ``evals.json`` from each run under the roots and tabulates, per (run, benchmark), the metric
at each sparsity NORMALISED between the run's own ``pretrained`` (the DPO model, 0) and ``full_delta``
(the RL model, 1) anchors -- so a cell reads "what fraction of the RL stage's gain on B does A's
top-k carry". The raw numbers are kept beside the normalised ones because a benchmark whose two
anchors are close (MMLU) has a noisy ratio.

    uv run python scripts/bench_transfer.py --roots runs/olmo3_post/posthoc runs/olmo3_post/ixg \
        --out plots/data/olmo3_post/transfer.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METRICS = {"gsm8k": ("gsm8k", "accuracy"), "math500": ("math500", "accuracy"),
           "ifeval": ("ifeval", "prompt_strict"), "mmlu": ("mmlu", "accuracy"),
           "sft_loss": ("test", "loss")}


def get(final, cond, ev, split, metric):
    try:
        return final[cond][ev][split][metric]
    except KeyError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["runs/olmo3_post/posthoc", "runs/olmo3_post/ixg",
                                                   "runs/olmo3_post/rl"])
    ap.add_argument("--out", default="plots/data/olmo3_post/transfer.json")
    args = ap.parse_args()
    table = {}
    for root in args.roots:
        for d in sorted((ROOT / root).glob("*")):
            ej = d / "evals.json"
            if not ej.exists():
                ej = d / "posthoc_eval" / "evals.json"
            if not ej.exists():
                continue
            final = json.load(open(ej))["final"]
            tag = f"{Path(root).name}/{d.name}"
            conds = [c for c in final if c.startswith("frac_")]
            row = {}
            for ev, (split, metric) in METRICS.items():
                lo, hi = get(final, "pretrained", ev, split, metric), get(final, "full_delta", ev, split, metric)
                if lo is None or hi is None:
                    continue
                row[ev] = {"pretrained": lo, "full_delta": hi, "raw": {}, "norm": {}}
                for c in conds:
                    v = get(final, c, ev, split, metric)
                    if v is None:
                        continue
                    row[ev]["raw"][c] = v
                    row[ev]["norm"][c] = (v - lo) / (hi - lo) if hi != lo else None
            table[tag] = row
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    evs = [e for e in METRICS if e != "sft_loss"]
    for c in ("frac_0.01", "frac_0.05", "frac_0.2"):
        print(f"\n== {c}: normalised gain carried (0 = DPO, 1 = RL); raw in brackets ==")
        print("fitted on | " + " | ".join(evs))
        for tag, row in table.items():
            cells = []
            for e in evs:
                r = row.get(e, {})
                v, n = r.get("raw", {}).get(c), r.get("norm", {}).get(c)
                cells.append(f"{n:.2f} ({v:.1f})" if n is not None else (f"- ({v:.1f})" if v is not None else "-"))
            print(f"{tag} | " + " | ".join(cells))
    print("\nanchors per run (pretrained -> full_delta); generation budgets may differ between runs:")
    for tag, row in table.items():
        print(f"  {tag}: " + ", ".join(f"{e} {row[e]['pretrained']:.1f}->{row[e]['full_delta']:.1f}"
                                      for e in evs if e in row))


if __name__ == "__main__":
    main()
