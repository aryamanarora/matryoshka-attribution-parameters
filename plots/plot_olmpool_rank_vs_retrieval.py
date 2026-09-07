"""OlmPool: does the MAttr head ranking find the retrieval heads?

Left: for each model, the fraction of its retrieval heads (Wu et al. score >= 0.1 at the
longest probed context of the long-context checkpoint) that fall within the top-k heads of the
learned head mask (`fold_learned`: heads scored over the extended rest), as k runs over the
ranking; the grey curve is chance (k / n_heads). Right: every head of two baselines, MAttr rank
percentile against retrieval score, retrieval heads filled.

    uv run python plots/plot_olmpool_rank_vs_retrieval.py
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

BLOCK = {"G_pre_8kv_8k_14k": (0, "G: Llama"), "J_pre_16kv_8k_14k": (0, "J: Llama, 16 kv"),
         "G_pre_LQK_8kv_8k_14k": (1, "G: prenorm + LQK"),
         "G_post_LQK_8kv_8k_14k": (2, "G: post-norm + LQK"),
         "H_post_LQK_32kv_8k_11k_SWA": (2, "H: Olmo-3, SWA"), "K_post_HQK_8kv_12k": (3, "K: Qwen-3")}
COLORS = [palette.COLOR["adam"], palette.COLOR["ixg:mc"], palette.COLOR["ixg:base"], palette.COLOR["sgd"]]
STYLES = {0: ["solid", (0, (4, 1.5))], 1: ["solid"], 2: [(0, (4, 1.5)), "solid"], 3: ["solid"]}


def load(model, heads_dir, rh_dir, arm):
    Q = np.array(json.loads((heads_dir / f"{model}__{arm}.json").read_text())["Q"], dtype=float)
    rh = json.loads((rh_dir / f"{model}__lc.json").read_text())
    Lc = max(rh["lengths"], key=int)
    S = np.array(rh["lengths"][Lc]["score_success"], dtype=float)
    ok = ~np.isnan(Q)
    q, s = Q[ok], S[ok]
    order = np.argsort(-q)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return q, s, rank, int(Lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="plots/data/olmpool/heads")
    ap.add_argument("--rh", default="runs/olmpool_rh")
    ap.add_argument("--arm", default="fold_learned")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    heads_dir, rh_dir = Path(a.heads), Path(a.rh)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.4})
    used = {}
    for model, (b, label) in BLOCK.items():
        if not (heads_dir / f"{model}__{a.arm}.json").exists() or not (rh_dir / f"{model}__lc.json").exists():
            continue
        q, s, rank, Lc = load(model, heads_dir, rh_dir, a.arm)
        is_rh = s >= a.threshold
        n, n_rh = len(q), int(is_rh.sum())
        if n_rh == 0:
            continue
        ks = np.arange(1, n + 1)
        recall = np.cumsum(is_rh[np.argsort(rank)]) / n_rh
        ls = STYLES[b][used.get(b, 0) % len(STYLES[b])]
        used[b] = used.get(b, 0) + 1
        ax.plot(ks, recall, color=COLORS[b], lw=0.9, ls=ls, label=f"{label} ({n_rh} of {n})", zorder=3)
    n_ref = 1024
    ax.plot(np.arange(1, n_ref + 1), np.arange(1, n_ref + 1) / n_ref, color="#999999", lw=0.6, zorder=2)
    ax.text(300, 0.2, "chance", fontsize=5, color="#777777", rotation=32)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["10⁰", "10¹", "10²", "10³"])
    ax.set_xlim(1, 1300)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Top-k Heads by MAttr Score", fontsize=7)
    ax.set_ylabel("Fraction of Retrieval Heads Captured", fontsize=7)
    # legend below the panel: every corner of this axes has curves in it
    ax.legend(fontsize=4.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
              handlelength=1.8, handletextpad=0.4, labelspacing=0.3, borderpad=0.2, columnspacing=1.0)
    palette.furnish(ax)
    ax.tick_params(labelsize=6, width=0.4, length=2)

    for model, mk in (("G_pre_8kv_8k_14k", "o"), ("H_post_LQK_32kv_8k_11k_SWA", "^")):
        q, s, rank, Lc = load(model, heads_dir, rh_dir, a.arm)
        b, label = BLOCK[model]
        pct = 100 * (1 - (rank - 1) / len(rank))
        is_rh = s >= a.threshold
        ax2.scatter(pct[~is_rh], s[~is_rh], s=4, marker=mk, facecolor="none", edgecolor=COLORS[b],
                    lw=0.3, alpha=0.5, zorder=2)
        ax2.scatter(pct[is_rh], s[is_rh], s=9, marker=mk, color=COLORS[b], lw=0, zorder=3,
                    label=f"{label} @{Lc // 1024}K")
    ax2.axhline(a.threshold, color="#999999", lw=0.5, ls=(0, (2, 2)))
    ax2.set_xlabel("MAttr Score Percentile of the Head", fontsize=7)
    ax2.set_ylabel("Retrieval Score (Wu et al.)", fontsize=7)
    ax2.set_xlim(-2, 102)
    ax2.legend(fontsize=4.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1,
               handletextpad=0.3, borderpad=0.2, labelspacing=0.3)
    palette.furnish(ax2)
    ax2.tick_params(labelsize=6, width=0.4, length=2)
    fig.tight_layout(pad=0.3)
    out = a.out or "plots/olmpool_rank_vs_retrieval.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
