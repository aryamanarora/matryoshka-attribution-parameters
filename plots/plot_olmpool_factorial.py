"""OlmPool: which part of the extension update carries retrieval, per architecture.

Heatmap from ``plots/data/olmpool/factorial.json``: rows are models (sorted by the paper's
HELMET-32K), columns are the part combinations of the extension delta applied over pretrained
weights (A attention projections, Q QK-norm gains, M MLPs, O embeddings + norms), cells the
teacher-forced needle accuracy at one context length. ``none`` is zero-shot theta scaling, the
full combination is the released long-context model. A second panel draws the three derived
shares against HELMET so a trend by feature is visible.

    uv run python plots/plot_olmpool_factorial.py --length 16384
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
COLS = ["none", "A", "Q", "M", "O", "AQ", "AM", "AO", "QM", "QO", "MO", "AQM", "AQO", "AMO", "QMO", "AQMO", "pt_own_theta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="plots/data/olmpool/factorial.json")
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="model names to leave out")
    ap.add_argument("--shares", action="store_true", help="add the shares-vs-HELMET panel")
    a = ap.parse_args()
    rows = json.loads(Path(a.data).read_text())
    rows = [r for r in rows if r["results"] and r["model"] not in a.exclude]
    rows.sort(key=lambda r: -r["results"]["helmet"]["32k"])
    cols = [c for c in COLS if any(c in r["acc"] for r in rows)]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r["acc"].get(c, {}).get(f"ctx_{a.length}")
            if v is not None:
                M[i, j] = v
    # full \textwidth (5.4in): 22 rows x 17 columns at ~0.19in per cell, 6pt tick labels, 5pt
    # cell values -- text renders 1:1 at that width
    # 4.4in of axes plus the row labels and colourbar comes to ~5.4in after bbox_inches="tight"
    fig = plt.figure(figsize=(4.4, 0.19 * len(rows) + 1.1))
    gs = fig.add_gridspec(1, 2 if a.shares else 1, width_ratios=[len(cols), 5] if a.shares else [1],
                          wspace=0.35)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("pt_own_theta", "pt (own θ)") for c in cols], fontsize=6, rotation=60, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['model']}  ({r['results']['helmet']['32k']:.0f})" for r in rows], fontsize=6)
    ax.set_xlabel(f"Parts of the Extension Applied (Needle Accuracy at {a.length // 1024}K)", fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1", ha="center", va="center", fontsize=5,
                        color="white" if M[i, j] < 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=6, width=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(width=0.4, length=2)
    if not a.shares:
        out = a.out or f"plots/olmpool_factorial_{a.length // 1024}k.pdf"
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)
        return
    ax2 = fig.add_subplot(gs[1])
    for key, color, lab in (("attn_share", palette.COLOR["adam"], "attention alone"),
                            ("rest_share", palette.COLOR["ixg:base"], "everything but attention"),
                            ("attn_needed", palette.COLOR["ixg:mc"], "lost without attention")):
        xs = [r["results"]["helmet"]["32k"] for r in rows if r["derived"][str(a.length)][key] is not None]
        ys = [r["derived"][str(a.length)][key] for r in rows if r["derived"][str(a.length)][key] is not None]
        ax2.scatter(xs, ys, s=9, color=color, lw=0, label=lab, alpha=0.9)
    ax2.axhline(0, color="#999999", lw=0.4)
    ax2.axhline(1, color="#999999", lw=0.4)
    ax2.set_xlabel("HELMET @ 32K (paper)", fontsize=6)
    ax2.set_ylabel("Share of the Extension's Gain", fontsize=6)
    ax2.legend(fontsize=4.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    palette.furnish(ax2)
    ax2.tick_params(labelsize=5, width=0.4, length=2)
    out = a.out or f"plots/olmpool_factorial_{a.length // 1024}k.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
