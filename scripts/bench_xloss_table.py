"""Tabulate scripts/bench_xloss.py's matrix.json: for each (mask, objective) the fraction of the
DPO->RL loss drop on that objective that the mask's top-k carries, (pre - loss_k) / (pre - full).
>1 means the sparse slice fits the objective better than the whole RL update.

    uv run python scripts/bench_xloss_table.py runs/olmo3_post/xloss/matrix.json --frac 0.01
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--fracs", default="0.002,0.01,0.05,0.2")
    args = ap.parse_args()
    m = json.load(open(args.matrix))
    objs = sorted({o for row in m.values() for o in row}, key=["gsm8k", "math", "ifeval", "mmlu"].index)
    for frac in args.fracs.split(","):
        c = f"frac_{frac}"
        print(f"\n== fraction of the RL loss drop carried at top-{100*float(frac):g}% (raw NLL in brackets) ==")
        print(f"{'mask \\ objective':>18} | " + " | ".join(f"{o:>14}" for o in objs))
        for mask, row in m.items():
            cells = []
            for o in objs:
                r = row.get(o)
                if not r or c not in r:
                    cells.append(f"{'-':>14}")
                    continue
                pre, full, v = r["pretrained"], r["full_delta"], r[c]
                g = (pre - v) / (pre - full) if pre != full else float("nan")
                cells.append(f"{g:5.2f} ({v:.3f})")
            print(f"{mask:>18} | " + " | ".join(f"{x:>14}" for x in cells))
    print("\nanchors (pretrained -> full_delta):")
    row = next(iter(m.values()))
    for o in objs:
        if o in row:
            print(f"  {o}: {row[o]['pretrained']:.4f} -> {row[o]['full_delta']:.4f}")


if __name__ == "__main__":
    main()
