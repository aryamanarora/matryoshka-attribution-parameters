"""OlmPool: the norm block decides which parameters carry the extension -- at two levels.

Left, weight level: for every usable model, retrieval at one context length with only the
attention part of the extension applied (triangle), only the MLP part (square), nothing (open
circle) and the whole update (filled circle), rows grouped by norm block and sorted by the paper's
HELMET within a block. Right, unit level: the share of a model's top-0.5% MAttr units (heads and
MLP neurons scored together) that are attention heads, against the weight-level attention share
from the left panel -- the two attributions agree on which blocks are head-carried.

Raw matplotlib: grouped dot rows with block brackets and labelled scatter points.

    uv run python plots/plot_olmpool_norm_block.py --length 32768
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import palette

plt.rcParams.update(palette.RC)

EXCLUDE = {"G_pre_8kv_8k_14k_SWA", "H_pre_32kv_8k_11k_SWA", "A_post_HQK_8kv_8k_13k_SWA_fp8",
           "F_pre_8kv_8k_14k_SWA", "H_pre_LQK_32kv_8k_11k_SWA", "H_post_HQK_32kv_8k_11k_SWA"}
BLOCKS = [("Llama (prenorm, no QK norm)", palette.COLOR["adam"]),
          ("prenorm + layerwise QK norm", palette.COLOR["ixg:mc"]),
          ("post-norm + layerwise QK norm (Olmo-3)", palette.COLOR["ixg:base"]),
          ("prenorm + headwise QK norm (Qwen-3)", palette.COLOR["sgd"])]


def block_of(res):
    if res["qk_norm"] == "none":
        return 0
    if res["qk_norm"] == "HQK":
        return 3
    return 2 if res["norm"] == "post" else 1


def short(name):
    return name.replace("_11k", "").replace("_12k", "").replace("_13k", "").replace("_14k", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factorial", default="plots/data/olmpool/factorial.json")
    ap.add_argument("--summary", default="plots/data/olmpool/summary.json")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = f"ctx_{a.length}"
    fac = [r for r in json.loads(Path(a.factorial).read_text())
           if r["results"] and r["model"] not in EXCLUDE]
    rows = []
    for r in fac:
        allk = "".join(r["parts"])
        acc = lambda k: r["acc"].get(k, {}).get(L)
        rows.append(dict(model=r["model"], block=block_of(r["results"]),
                         helmet=r["results"]["helmet"]["32k"], none=acc("none"), A=acc("A"),
                         M=acc("M"), full=acc(allk)))
    rows.sort(key=lambda d: (d["block"], -d["helmet"]))
    heads = {r["model"]: r["top_split"]["0.005"]["frac_of_heads"]
             for r in json.loads(Path(a.summary).read_text()) if r["arm"] == "all_learned"}

    fig = plt.figure(figsize=(4.95, 3.3))   # 5.4in after the outer labels (bbox_inches="tight")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(rows))[::-1]
    for yi, d in zip(y, rows):
        c = BLOCKS[d["block"]][1]
        ax.plot([d["none"], d["full"]], [yi, yi], color="#dddddd", lw=1.2, zorder=1)
        ax.scatter(d["none"], yi, s=14, facecolor="white", edgecolor=c, lw=0.7, zorder=3)
        ax.scatter(d["A"], yi, s=16, marker="^", color=c, lw=0, zorder=4)
        ax.scatter(d["M"], yi, s=14, marker="s", color=c, lw=0, zorder=4)
        ax.scatter(d["full"], yi, s=16, marker="o", color=c, lw=0, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(d['model'])} ({d['helmet']:.0f})" for d in rows], fontsize=5.5)
    for b, (label, c) in enumerate(BLOCKS):
        ys = [yi for yi, d in zip(y, rows) if d["block"] == b]
        if ys:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=c, alpha=0.07, lw=0, zorder=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f"Needle Retrieval at {a.length // 1024}K", fontsize=7)
    palette.furnish(ax)
    ax.tick_params(axis="x", labelsize=6, width=0.4, length=2)
    ax.tick_params(axis="y", width=0.4, length=2)
    # two legends: the marker key (lower right, where the low rows have no points past 0.65)
    # and the block key above the axes, where it covers nothing
    marks = [ax.scatter([], [], marker=m, color="#444444", s=14, lw=0, label=lab)
             for m, lab in [("^", "attention update only"), ("s", "MLP update only"),
                            ("o", "released model")]]
    marks.append(ax.scatter([], [], marker="o", facecolor="white", edgecolor="#444444", s=14,
                            lw=0.7, label="no update (θ scaled)"))
    leg1 = ax.legend(handles=marks, fontsize=5, frameon=False, loc="lower right",
                     handletextpad=0.3, borderpad=0.2, labelspacing=0.25)
    ax.add_artist(leg1)
    blocks = [ax.scatter([], [], marker="s", color=c, s=14, lw=0, alpha=0.6, label=label)
              for label, c in BLOCKS]
    ax.legend(handles=blocks, fontsize=5, frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.35, 1.0), handletextpad=0.3, borderpad=0.2, labelspacing=0.25,
              columnspacing=0.8)

    ax2 = fig.add_subplot(gs[1])
    pts = []
    for d in rows:
        if d["model"] not in heads or d["full"] is None or d["none"] is None:
            continue
        gain = d["full"] - d["none"]
        if gain < 0.2:
            continue                      # no measurable gain: the share is undefined
        share = (d["A"] - d["none"]) / gain
        lab = short(d["model"]).split("_")[0] + ("*" if d["block"] == 2 and "SWA" in d["model"] else "")
        pts.append((heads[d["model"]], share, BLOCKS[d["block"]][1], lab))
    for x0, y0, c, lab in pts:
        ax2.scatter(x0, y0, s=18, color=c, lw=0, zorder=3)
    # labels: stack the ones whose points would overlap (within 0.02 x 0.06) below each other
    placed = []
    for x0, y0, c, lab in sorted(pts, key=lambda t: (-t[1], t[0])):
        n_close = sum(1 for px, py in placed if abs(px - x0) < 0.02 and abs(py - y0) < 0.06)
        ax2.annotate(lab, (x0, y0), fontsize=5, xytext=(3, 2 - 6 * n_close),
                     textcoords="offset points")
        placed.append((x0, y0))
    ax2.set_xlabel("Heads in the Top 0.5% of MAttr Units", fontsize=7)
    ax2.set_ylabel("Retrieval Gain from the Attention Update Alone", fontsize=7)
    ax2.set_xlim(0.05, 0.38)
    ax2.set_ylim(-0.08, 1.05)
    ax2.set_xticks([0.1, 0.2, 0.3])
    ax2.set_xticklabels(["10%", "20%", "30%"])
    palette.furnish(ax2)
    ax2.tick_params(labelsize=6, width=0.4, length=2)
    fig.tight_layout(pad=0.3)
    out = a.out or f"plots/olmpool_norm_block_{a.length // 1024}k.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
