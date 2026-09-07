"""Has a learned mask's RANKING converged? Reads a run's ckpt_step*.pt + final.pt and reports, per
checkpoint, the top-k Jaccard against the final ranking and against the previous checkpoint, the
Spearman against final, and (from evals.json's history) the held-out loss at fixed sparsities.

    uv run python scripts/bench_convergence.py runs/olmo3_post/posthoc/gsm8k_long --fracs 0.01,0.05
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def topk(s, frac):
    k = max(1, int(round(frac * len(s))))
    return set(np.argpartition(-s, k - 1)[:k].tolist())


def spearman(a, b):
    ra = np.empty(len(a)); ra[np.argsort(a)] = np.arange(len(a))
    rb = np.empty(len(b)); rb[np.argsort(b)] = np.arange(len(b))
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fracs", default="0.01,0.05")
    ap.add_argument("--final", default=None, help="checkpoint to treat as the reference (default final.pt, else the last ckpt)")
    args = ap.parse_args()
    run = Path(args.run)
    fracs = [float(f) for f in args.fracs.split(",")]
    ckpts = sorted(run.glob("ckpt_step*.pt"), key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    ref_path = Path(args.final) if args.final else (run / "final.pt" if (run / "final.pt").exists() else ckpts[-1])
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)["scores"].float().numpy()
    ref_top = {f: topk(ref, f) for f in fracs}
    prev = None
    print(f"reference: {ref_path.name}")
    print("step | rho vs final | " + " | ".join(f"J@{100*f:g}% vs final | vs prev" for f in fracs))
    for p in ckpts:
        step = int(re.search(r"step(\d+)", p.name).group(1))
        s = torch.load(p, map_location="cpu", weights_only=False)["scores"].float().numpy()
        cells = []
        for f in fracs:
            t = topk(s, f)
            jf = len(t & ref_top[f]) / len(t | ref_top[f])
            jp = len(t & prev[f]) / len(t | prev[f]) if prev else float("nan")
            cells.append(f"{jf:.3f} | {jp:.3f}")
        print(f"{step:4d} | {spearman(s, ref):.3f} | " + " | ".join(cells))
        prev = {f: topk(s, f) for f in fracs}
    ej = run / "evals.json"
    if ej.exists():
        hist = json.load(open(ej)).get("history", [])
        print("\nheld-out loss by eval step (sft_loss/test):")
        conds = None
        for h in hist:
            step, res = h["step"], h["results"]
            row = {c: r["sft_loss"]["test"]["loss"] for c, r in res.items() if "sft_loss" in r and "test" in r["sft_loss"]}
            if conds is None:
                conds = list(row); print("step | " + " | ".join(conds))
            print(f"{step:4d} | " + " | ".join(f"{row.get(c, float('nan')):.4f}" for c in conds))


if __name__ == "__main__":
    main()
