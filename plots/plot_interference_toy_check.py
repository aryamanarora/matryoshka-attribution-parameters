"""The three MODEL-VALIDATION figures of Olah, Turner & Conerly, "A Toy Model of Interference
Weights" (2025), redrawn from this repo's replication so the regime can be checked by eye.

    uv run python plots/plot_interference_toy_check.py

These are not the note's results -- they are the figures it uses to argue its toy model is in
the right regime, and they are the ones to look at before trusting anything downstream:

  (a) Weight histogram, coloured by dL(U_ij).  The note's caption is "Real weights and
      interference weights overlap", and that OVERLAP is the whole point of the config: if
      the red and grey masses separate, `|U_ij| > t` already solves the task and every
      heuristic scores near 1.
  (b) Two independently trained models' weights against each other. The note: "The
      interference weights are independent, but the real weights are all significantly
      positive" -- so the real mass should sit on the diagonal and the interference mass
      should be a round blob.
  (c) Learned weights against ideal weights (= A, the target circuit). The note annotates
      three regions: "Shrinkage" (learned below the diagonal), "Unlearned" (a horizontal band
      at y=0 spanning every x), and the interference weights stacked at x=0.

Requires scripts/interference/interference_toy.py --check, which fits the second seed (b) needs.
"""

import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from palette import RC, furnish

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "interference_toy"
# The note's own colour scale for dL: diverging, saturating at +-0.02.
CMAP, VLIM = "RdBu_r", 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = DATA.parent / f"{DATA.name}_{a.tag}" if a.tag else DATA
    m = torch.load(d / "model.pt")
    U, A, dl = m["U"], m["A"], m["dl"]
    real = dl > 1e-4

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    norm = TwoSlopeNorm(vmin=-VLIM, vcenter=0.0, vmax=VLIM)

    # (a) histogram, split into the two masses rather than colour-binned: the note's stacked
    # colouring is unreadable at this figure size, and the claim being checked is about the
    # two DISTRIBUTIONS overlapping, which two outlines state directly.
    bins = torch.linspace(-1.5, 1.5, 121)
    axes[0].hist(U[~real].numpy(), bins=bins, color="#bbbbbb", label="interference")
    axes[0].hist(U[real].numpy(), bins=bins, color="#b2182b", label="real")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Virtual weight $U_{ij}$")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=5, frameon=False, loc="upper left")

    # (b) two seeds against each other
    p = d / "model_seed2.pt"
    if p.exists():
        U2 = torch.load(p)["U2"]
        axes[1].scatter(U2[~real], U[~real], s=0.4, lw=0, color="#cccccc", rasterized=True)
        axes[1].scatter(U2[real], U[real], s=1.2, lw=0, color="#b2182b", rasterized=True)
        r_real = torch.corrcoef(torch.stack([U[real], U2[real]]))[0, 1]
        r_int = torch.corrcoef(torch.stack([U[~real], U2[~real]]))[0, 1]
        axes[1].annotate(f"r = {r_real:.2f} (real)\nr = {r_int:.2f} (interf.)",
                         (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=5)
    axes[1].set_xlabel("Model B weight")
    axes[1].set_ylabel("Model A weight")

    # (c) learned vs ideal
    axes[2].scatter(A.reshape(-1), U.reshape(-1), s=0.4, lw=0, c=dl.reshape(-1),
                    cmap=CMAP, norm=norm, rasterized=True)
    lim = max(float(A.max()), 1.0)
    axes[2].plot([0, lim], [0, lim], lw=0.4, ls=(0, (3, 2)), color="#888888", zorder=0)
    axes[2].set_xlabel("Ideal weight $A_{ij}$")
    axes[2].set_ylabel("Learned weight $U_{ij}$")

    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        furnish(ax)
    for ax in axes[1:]:
        ax.axhline(0, lw=0.4, color="#888888", zorder=0)
        ax.axvline(0, lw=0.4, color="#888888", zorder=0)

    fig.tight_layout()
    out = ROOT / "plots" / (f"interference_toy_check_{a.tag}.pdf" if a.tag
                            else "interference_toy_check.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
