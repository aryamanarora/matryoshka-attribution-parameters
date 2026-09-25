"""OlmPool: how sparse the retriever really is -- needle accuracy vs fraction of units kept,
from 0.01% to the full update, for the nine all-units MAttr masks.

Reads the base sweep (``runs/olmpool/<model>/all_learned/evals.json``, 0.5% and up) and the
post-hoc re-sweep of the same saved masks at 0.01-0.2% (``.../posthoc_small/evals.json``,
produced by the eval CLI's ``--fracs`` with no refit). Raw matplotlib rather than plotnine
because the nine series are labelled directly at their right ends, which needs hand placement.
Colour is the norm block (post-norm + QK norm vs. prenorm Llama block), which is the grouping
the result is about; the two prenorm variants that differ in pretraining context or KV heads
keep the Llama colour with a different dash.

    uv run python plots/plot_olmpool_small_frac.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

import palette
from mask_learning_finetuning.paths import runs_root  # noqa: E402  `runs/` -> $MLFT_RUNS_ROOT or <repo>/runs

plt.rcParams.update(palette.RC)
RUNS = runs_root() / "olmpool"
OUT = Path("plots/olmpool_small_frac.pdf")
LENGTHS = [16384, 32768]
MODELS = {  # short label, colour key, linestyle
    "G_post_LQK_8kv_8k_14k": ("post LQK", "post", "solid"),
    "H_post_LQK_32kv_8k_11k_SWA": ("post LQK SWA", "post", (0, (4, 1.5))),
    "K_post_HQK_8kv_12k": ("post HQK", "post", (0, (1, 1.2))),
    "G_pre_8kv_8k_14k": ("pre 8kv", "pre", "solid"),
    "G_pre_LQK_8kv_8k_14k": ("pre LQK", "pre", (0, (4, 1.5))),
    "I_pre_32kv_8k_12k": ("pre MHA", "pre", (0, (1, 1.2))),
    "G_pre_4kv_8k_14k": ("pre 4kv", "pre", (0, (4, 1, 1, 1))),
    "J_pre_16kv_8k_14k": ("pre 16kv", "pre", (0, (6, 1, 1, 1, 1, 1))),
    "G_pre_8kv_4k_14k": ("pre 4K ctx", "pre", (0, (2, 1))),
}
COL = {"post": palette.COLOR["ixg:base"], "pre": palette.COLOR["adam"]}


def curve(model, L):
    base = json.load(open(RUNS / model / "all_learned" / "evals.json"))["final"]
    small = json.load(open(RUNS / model / "all_learned" / "posthoc_small" / "evals.json"))["final"]
    pts = {}
    for src in (small, base):
        for cond, r in src.items():
            if cond.startswith("frac_"):
                pts[float(cond[5:])] = r["niah"][f"ctx_{L}"]["acc"]
    xs = sorted(pts)
    return xs, [pts[x] for x in xs], base["pretrained"]["niah"][f"ctx_{L}"]["acc"]


fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharey=True)
for ax, L in zip(axes, LENGTHS):
    for model, (label, grp, ls) in MODELS.items():
        xs, ys, pre = curve(model, L)
        ax.plot(xs, ys, color=COL[grp], ls=ls, lw=0.9, marker="o", ms=1.8, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(7e-5, 1.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_xlabel(f"Fraction of units kept ({L // 1024}K context)", fontsize=7)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
axes[0].set_ylabel("Needle retrieval accuracy", fontsize=7)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6, frameon=False, handlelength=2.6,
           columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
