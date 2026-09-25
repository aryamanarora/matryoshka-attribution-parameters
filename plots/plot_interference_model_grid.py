"""The toy-model validation figure against TRAINING LENGTH: 3k / 30k / 100k steps of `hard`.

    uv run python plots/plot_interference_model_grid.py

A 3x3 grid replacing the paper's `fig:interference-model` row: the three validation panels of
plot_interference_model_check.py (learned vs ideal, run vs run, weight histogram) as columns,
and the same frozen-projection model at 3,000 / 30,000 / 100,000 training steps as rows. The
note says its models were undertrained; the rows show what training length does to each of the
things its figures are read for -- overlap of real and interference weights (a, c), and whether
two seeds agree on the real weights and not on the interference (b). At 3k the interference
band is wider than the circuit and the two seeds barely agree (r 0.30); at 30k the band has
narrowed to the note's ~1.6x overlap ratio and the seeds agree at r 0.70 with independent
interference (r 0.01), which is the note's picture; at 100k the loss has not moved but the band
keeps narrowing, i.e. the model is slowly turning into the literal config's easy case.

Reads plots/data/interference_toy_{hard,hard30k,hard100k}/{model,model_seed2}.pt. Each row's
second seed is a model trained on the same A with its own frozen projection (what `--check`
writes), because a shared projection makes the interference agree across seeds (r 0.76 at 30k),
which is not the note's "interference weights are independent". Drawing code is the single-panel
script's, transcribed onto a shared grid so the three columns keep their annotations and colour
treatment; per-panel axis limits are per ROW, since the weight scale shrinks ~3x over training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import TwoSlopeNorm

from palette import RC, furnish
from plot_interference_model_check import DL_CMAP, OFF, ON

ROOT = Path(__file__).resolve().parents[1]
ROWS = (("hard", "3k steps"), ("hard30k", "30k steps"), ("hard100k", "100k steps"))
FS_LAB, FS_TICK, FS_NOTE = 6.5, 5.5, 5.0


def panel_a(fig, ax, U, A, dl, colorbar_ax=None):
    sc = ax.scatter(A.reshape(-1), U.reshape(-1), s=1.0, lw=0, c=dl.reshape(-1), cmap=DL_CMAP,
                    norm=TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.02), rasterized=True)
    lim = float(max(A.max(), 1.0))
    ax.plot([0, lim], [0, lim], lw=0.5, ls=(0, (3, 2)), color="#888888", zorder=0)
    ax.axhline(0, lw=0.4, color="#888888", zorder=0)
    lo = float(U.reshape(-1).quantile(0.001)); hi = float(U.reshape(-1).quantile(0.9995))
    ax.set_xlim(-0.05, lim + 0.05)
    ax.set_ylim(lo * 1.15, hi * 1.12)
    ax.annotate("interference", (0.02, hi * 0.98), fontsize=FS_NOTE, color="#555555",
                ha="left", va="top")
    ax.annotate("unlearned", (0.62, 0.02 * hi / 0.4 if hi < 0.4 else 0.03), fontsize=FS_NOTE,
                color="#555555", va="bottom")
    if colorbar_ax is not None:
        cb = fig.colorbar(sc, cax=colorbar_ax, orientation="horizontal", extend="both",
                          ticks=[-0.02, 0.0, 0.02])
        cb.ax.set_title("$\\Delta L(U_{ij})$", fontsize=FS_NOTE, pad=2)
        cb.ax.tick_params(labelsize=FS_NOTE - 0.5, length=2, pad=1)
        cb.outline.set_linewidth(0.5)


def panel_b(ax, U, U2, sup):
    ax.scatter(U2[~sup], U[~sup], s=0.7, lw=0, color=OFF, rasterized=True)
    ax.scatter(U2[sup], U[sup], s=1.8, lw=0, color=ON, rasterized=True)
    r_on = float(torch.corrcoef(torch.stack([U[sup], U2[sup]]))[0, 1])
    r_off = float(torch.corrcoef(torch.stack([U[~sup], U2[~sup]]))[0, 1])
    lim = float(torch.quantile(torch.cat([U.reshape(-1), U2.reshape(-1)]).abs(), 0.9995))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], lw=0.5, ls=(0, (3, 2)), color="#888888", zorder=0)
    ax.axhline(0, lw=0.4, color="#888888", zorder=0)
    ax.axvline(0, lw=0.4, color="#888888", zorder=0)
    ax.annotate(f"$r$ = {r_on:.2f}  on circuit\n$r$ = {r_off:.2f}  interference",
                (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=FS_NOTE)
    return r_on, r_off


def panel_c(ax, U, sup):
    lo = float(U.reshape(-1).quantile(0.001)); hi = float(U.reshape(-1).quantile(0.9995))
    bins = torch.linspace(lo, hi, 61).numpy()
    ax.hist(U[~sup].numpy(), bins=bins, color=OFF)
    ax.hist(U[sup].numpy(), bins=bins, color=ON)
    ax.set_yscale("log")
    ax.annotate("on circuit", (0.96, 0.80), xycoords="axes fraction", ha="right",
                fontsize=FS_NOTE, color=ON)
    ax.annotate("interference", (0.96, 0.89), xycoords="axes fraction", ha="right",
                fontsize=FS_NOTE, color="#999999")


def main():
    rows = [(tag, lab) for tag, lab in ROWS
            if (ROOT / "plots" / "data" / f"interference_toy_{tag}" / "model_seed2.pt").exists()]
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(rows), 3, figsize=(5.5, 1.55 * len(rows) + 0.35))
    axes = axes.reshape(len(rows), 3)
    cbar_ax = fig.add_axes([0.13, 0.985, 0.2, 0.012])
    for r, (tag, lab) in enumerate(rows):
        d = ROOT / "plots" / "data" / f"interference_toy_{tag}"
        m = torch.load(d / "model.pt")
        U, A, dl = m["U"], m["A"], m["dl"]
        U2 = torch.load(d / "model_seed2.pt")["U2"]
        sup = A > 0
        panel_a(fig, axes[r, 0], U, A, dl, colorbar_ax=cbar_ax if r == 0 else None)
        r_on, r_off = panel_b(axes[r, 1], U, U2, sup)
        panel_c(axes[r, 2], U, sup)
        axes[r, 0].set_ylabel(f"{lab}\n\nLearned virtual weight $U_{{ij}}$", size=FS_LAB)
        axes[r, 1].set_ylabel("Model A  $U_{ij}$", size=FS_LAB)
        axes[r, 2].set_ylabel("Count", size=FS_LAB)
        print(f"  {tag:<9} on-circuit r {r_on:.3f}   interference r {r_off:.3f}")
        for ax in axes[r]:
            ax.tick_params(labelsize=FS_TICK)
            furnish(ax)
    for ax, xl in zip(axes[-1], ("Ideal weight $A_{ij}$", "Model B  $U_{ij}$",
                                  "Virtual weight $U_{ij}$")):
        ax.set_xlabel(xl, size=FS_LAB)
    fig.tight_layout(h_pad=0.6, w_pad=0.8, rect=(0, 0, 1, 0.975))
    out = ROOT / "plots" / "interference_model_grid.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
