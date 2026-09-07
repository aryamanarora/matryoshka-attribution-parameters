#!/usr/bin/env python3
"""Figures for the Olmo-3 post-training benchmark-attribution experiment (docs/olmo3_post/).

Two panels from two JSONs written by scripts/olmo3_post/bench_similarity.py and scripts/olmo3_post/bench_transfer.py:

  olmo3_post_similarity.pdf   pairwise top-1% Jaccard (lower triangle) / Spearman (upper) between
                              per-benchmark rankings of the DPO->RL delta's units, for one ranking
                              method (--method ixg_base | ixg_finetuned | posthoc | rl). The split
                              halves of each benchmark sit beside it, so the within-benchmark
                              ceiling is a cell in the same matrix.
  olmo3_post_transfer.pdf     mask fitted on A (rows) scored on B (columns): normalised gain
                              carried at --frac (0 = DPO anchor, 1 = RL anchor).

    uv run python plots/plot_olmo3_post.py --method ixg_base
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, coord_fixed, element_blank, element_text, geom_text, geom_tile, ggplot, labs,
    scale_fill_cmap, scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")
theme_set(theme_bw(base_size=8) + theme(
    text=element_text(color="#000", family=FAMILY), axis_title=element_text(size=7),
    axis_text=element_text(size=6), axis_text_x=element_text(rotation=45, hjust=1.0),
    panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
    legend_title=element_text(size=7), legend_text=element_text(size=6), legend_key_size=6))

BENCH_ORDER = ["gsm8k", "gsm8k_a", "gsm8k_b", "math", "math_a", "math_b",
               "ifeval", "ifeval_a", "ifeval_b", "mmlu", "mmlu_a", "mmlu_b"]


def tag_to_name(tag, method):
    root, name = tag.split("/", 1)
    if root == "ixg":
        obj, at = name.rsplit("_", 1)
        return obj if f"ixg_{at}" == method else None
    if root == method:
        return name
    return None


def similarity(path, method, out, dpi):
    d = json.load(open(path))
    rows = []
    names = set()
    for r in d["pairs"]:
        a, b = tag_to_name(r["a"], method), tag_to_name(r["b"], method)
        if a is None or b is None:
            continue
        names |= {a, b}
        rows.append((a, b, r["unit"]["jaccard_0.01"], r["unit"]["spearman"]))
    if not rows:
        print(f"no pairs for method {method}")
        return
    order = [n for n in BENCH_ORDER if n in names] + sorted(names - set(BENCH_ORDER))
    idx = {n: i for i, n in enumerate(order)}
    cells = []
    for a, b, j, s in rows:
        lo, hi = sorted((a, b), key=idx.get)
        cells.append(dict(x=hi, y=lo, value=j, label=f"{j:.2f}", kind="J@1%"))   # lower: Jaccard
        cells.append(dict(x=lo, y=hi, value=s, label=f"{s:.2f}", kind="rho"))    # upper: Spearman
    for n in order:
        cells.append(dict(x=n, y=n, value=1.0, label="", kind="diag"))
    df = pd.DataFrame(cells)
    df["x"] = pd.Categorical(df["x"], order)
    df["y"] = pd.Categorical(df["y"], order[::-1])
    p = (ggplot(df, aes("x", "y", fill="value")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="label"), size=5)
         + scale_fill_cmap("viridis", limits=(0, 1), name="value")
         + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
         + coord_fixed()
         + labs(x="", y="", title=f"{method}: top-1% Jaccard (lower) / Spearman (upper)"))
    p.save(out, width=4.2, height=3.8, dpi=dpi, verbose=False)
    p.save(Path(out).with_suffix(".png"), width=4.2, height=3.8, dpi=dpi, verbose=False)
    print("wrote", out, "+ .png")


def transfer(path, frac, out, dpi, roots=("posthoc", "ixg", "rl")):
    d = json.load(open(path))
    cells = []
    evs = ["gsm8k", "math500", "ifeval", "mmlu"]
    for tag, row in d.items():
        if tag.split("/")[0] not in roots:
            continue
        for e in evs:
            n = row.get(e, {}).get("norm", {}).get(f"frac_{frac}")
            v = row.get(e, {}).get("raw", {}).get(f"frac_{frac}")
            if v is None:
                continue
            cells.append(dict(fitted=tag, scored=e, value=n if n is not None else 0.0,
                              label=f"{n:.2f}\n({v:.0f})" if n is not None else f"({v:.0f})"))
    if not cells:
        print("no transfer cells")
        return
    df = pd.DataFrame(cells)
    df["scored"] = pd.Categorical(df["scored"], evs)
    p = (ggplot(df, aes("scored", "fitted", fill="value")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="label"), size=5)
         + scale_fill_cmap("viridis", limits=(-0.2, 1.2), name="gain carried")
         + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
         + labs(x="scored on", y="mask fitted on",
                title=f"top-{100*frac:g}% of the DPO->RL delta: fraction of the RL gain carried"))
    p.save(out, width=4.6, height=0.35 * len(d) + 1.4, dpi=dpi, verbose=False)
    p.save(Path(out).with_suffix(".png"), width=4.6, height=0.35 * len(d) + 1.4, dpi=dpi, verbose=False)
    print("wrote", out, "+ .png")


def xloss(path, frac, out, dpi, prefix=""):
    """The loss-transfer matrix: fraction of the RL loss drop on B carried by A's top-k."""
    m = json.load(open(path))
    objs = ["gsm8k", "math", "ifeval", "mmlu"]
    cells = []
    for mask, row in m.items():
        for o in objs:
            r = row.get(o)
            if not r or f"frac_{frac}" not in r:
                continue
            pre, full, v = r["pretrained"], r["full_delta"], r[f"frac_{frac}"]
            g = (pre - v) / (pre - full)
            cells.append(dict(fitted=mask, scored=o, value=min(g, 1.5), label=f"{g:.2f}"))
    df = pd.DataFrame(cells)
    df["scored"] = pd.Categorical(df["scored"], objs)
    order = [k for k in m][::-1]
    df["fitted"] = pd.Categorical(df["fitted"], order)
    p = (ggplot(df, aes("scored", "fitted", fill="value")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="label"), size=6)
         + scale_fill_cmap("viridis", limits=(0, 1.5), name="drop carried")
         + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
         + labs(x="objective scored (held-out NLL of its rollouts)", y="mask fitted on",
                title=f"top-{100*frac:g}% of the DPO->RL delta: fraction of the RL loss drop carried"))
    p.save(out, width=4.6, height=0.3 * len(m) + 1.5, dpi=dpi, verbose=False)
    p.save(Path(out).with_suffix(".png"), width=4.6, height=0.3 * len(m) + 1.5, dpi=dpi, verbose=False)
    print("wrote", out, "+ .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="plots/data/olmo3_post")
    ap.add_argument("--method", default="ixg_base")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    d = Path(args.dir)
    if (d / "similarity.json").exists():
        similarity(d / "similarity.json", args.method,
                   here / f"olmo3_post_similarity_{args.method}.pdf", args.dpi)
    xl = Path("runs/olmo3_post/xloss/matrix.json")
    if xl.exists():
        for f in (0.01, 0.05, 0.2):
            xloss(xl, f, here / f"olmo3_post_xloss_{f:g}.pdf", args.dpi)
    if (d / "transfer.json").exists():
        transfer(d / "transfer.json", args.frac, here / f"olmo3_post_transfer_{args.frac:g}.pdf",
                 args.dpi)


if __name__ == "__main__":
    main()
