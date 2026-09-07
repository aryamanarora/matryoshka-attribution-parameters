"""What VALUES of virtual weight does each method keep, against the population it drew from?

    uv run python plots/plot_interference_keptvals.py --tag hard

Four panels sharing one x axis: the target circuit `A`, then the kept set of each method at its
own loss-optimal `k` (from scripts/interference/interference_minloss_sets.py). Grey fill is the full
population of 16384 in every panel, so each method is read against the same background.

DENSITY, NOT COUNTS, and log-scaled. The sets differ in size by a factor of 100 (136 to 16384),
so raw counts would put three of the four panels on the floor; and even as densities the kept
sets are concentrated enough that a linear y hides the tails where the interesting asymmetry
lives.

The panel this figure exists for is the last one. Selecting by `|U|` is the note's own first
heuristic and the one that fails hardest, so "which magnitudes did a method choose" is a direct
check on whether it is secretly doing that -- and MAttr-Adam is not: its kept weights are
SMALLER than a randomly chosen weight.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import COLOR, RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALL_C = "#d9d9d9"
PANELS = [("circuit $A$", None, "#b2182b"), ("stepless IG", "ixg:mc", COLOR["ixg:mc"]),
          ("MAttr (SGD)", "sgd", COLOR["sgd"]), ("MAttr (Adam)", "adam", COLOR["adam"])]
FS_TITLE, FS_TICK, FS_LAB, FS_NOTE = 6.0, 5.0, 6.0, 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    b = torch.load(d / "minloss_sets.pt")
    U, A = b["U"].reshape(-1), b["A"].reshape(-1)
    sup = (A > 0).nonzero().squeeze()

    lo, hi = float(U.quantile(0.0005)), float(U.quantile(0.9995))
    bins = torch.linspace(lo, hi, 61).numpy()

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.26, wspace=0.12)

    for ax, (title, key, col) in zip(axes, PANELS):
        idx = sup if key is None else b["sets"][key]["idx"]
        v = U[idx]
        ax.hist(U.numpy(), bins=bins, density=True, color=ALL_C, label="all", zorder=1)
        ax.hist(v.numpy(), bins=bins, density=True, histtype="step", lw=0.9, color=col,
                zorder=2)
        ax.axvline(0, lw=0.4, color="#888888", zorder=0)
        n = len(idx)
        sub = title if key is None else f"{title}  ($k$={b['sets'][key]['k']})"
        ax.set_title(sub, fontsize=FS_TITLE, pad=3)
        # the two numbers that carry the result, in the panel rather than the caption
        ax.annotate(f"$n$={n}\n{float((v>0).float().mean()):.0%} positive\n"
                    f"med $|U|$={float(v.abs().median()):.3f}",
                    (0.03, 0.96), xycoords="axes fraction", va="top", fontsize=FS_NOTE,
                    color=col, linespacing=1.3)
        ax.set_yscale("log")
        ax.set_xlabel("virtual weight $U_{ij}$", fontsize=FS_LAB, labelpad=1)
        ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
        furnish(ax)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    axes[0].set_ylabel("density", fontsize=FS_LAB)
    # top-right rather than bottom-right: the grey tail runs along the floor, so a grey
    # label there is grey-on-grey.
    axes[0].annotate("all weights", (0.97, 0.96), xycoords="axes fraction", ha="right",
                     va="top", fontsize=FS_NOTE, color="#8a8a8a")

    out = ROOT / "plots" / f"interference_keptvals_{a.tag}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
