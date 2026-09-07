"""OlmPool: per-head maps -- the mask's score for each attention head beside its retrieval score.

For one model, [layer x head] heatmaps in a row: the learned mask's score rank (which heads'
share of the extension delta the mask keeps first), the I×G score rank at the pretrained
endpoint, and Wu et al.'s retrieval score at a checkpoint (the fraction of decode steps on which
the head's argmax attention copied the needle token, over successful retrievals, at the longest
probed context). Rows are layers (0 at the top), columns are query heads. Raw matplotlib because
the panels need one shared layer axis and their own colour scales.

    uv run python plots/plot_olmpool_heads.py --model G_pre_8kv_8k_14k
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


def ranks(M):
    """Rank-transform a matrix (highest score = 1.0), NaN preserved; makes arms comparable."""
    flat = M.flatten()
    ok = ~np.isnan(flat)
    r = np.full_like(flat, np.nan)
    order = np.argsort(-flat[ok])
    rr = np.empty(order.size)
    rr[order] = np.arange(order.size)
    r[ok] = 1 - rr / max(1, order.size - 1)
    return r.reshape(M.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="plots/data/olmpool/heads")
    ap.add_argument("--rh", default="runs/olmpool_rh")
    ap.add_argument("--arms", nargs="*", default=["fold_learned", "fold_ixg_base"])
    ap.add_argument("--rh-ckpts", nargs="*", default=["pt_ext", "lc"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    panels = []
    for arm in a.arms:
        f = Path(a.data) / f"{a.model}__{arm}.json"
        if f.exists():
            M = np.array(json.loads(f.read_text())["Q"], dtype=float)
            panels.append((arm.replace("fold_", "").replace("attn_", "").replace("_", " "), ranks(M), 1.0))
    for ck in a.rh_ckpts:
        rhf = Path(a.rh) / f"{a.model}__{ck}.json"
        if not rhf.exists():
            continue
        rh = json.loads(rhf.read_text())
        Lc = max(rh["lengths"], key=int)
        S = np.array(rh["lengths"][Lc]["score_success"], dtype=float)
        panels.append((f"retrieval @{int(Lc) // 1024}K, {ck}", S, max(0.05, float(np.nanmax(S)))))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.75 * n + 0.5, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, M, vmax) in zip(axes, panels):
        im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_xlabel(f"Head\n{title}", fontsize=6)
        ax.tick_params(labelsize=5, width=0.4, length=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
        cbar.ax.tick_params(labelsize=5, width=0.4, length=2)
        cbar.outline.set_linewidth(0.5)
    axes[0].set_ylabel("Layer", fontsize=7)
    fig.tight_layout(pad=0.3)
    out = a.out or f"plots/olmpool_heads_{a.model}.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
