"""The two figures from § "What Should We Do about Interference Weights?" of Olah, Turner &
Conerly, "A Toy Model of Interference Weights" (Transformer Circuits, 2025).

    uv run python plots/plot_interference_filtering.py

Reads plots/data/interference_toy/curves.json (written by scripts/interference/interference_toy.py) and
draws, side by side, the note's "Precision-Recall Curve for Various Heuristics" and its
"Precision vs Loss Gain Pareto Frontier for Various Heuristics".

WHY TWO PANELS AND NOT ONE FIGURE EACH. The note's own argument for the second plot is that
the first one is the wrong question ("we probably don't actually care about recall per se"),
so the pair only makes its point read against each other -- recall treats every real weight
as worth the same, loss gain does not, and the heuristics reorder between them.

COLOUR IS NOT CHOSEN HERE for the three methods this repo adds on top (MAttr/SGD/stepless IG);
they come from plots/palette.py, which is the repo's source of truth. The note's own four
heuristics plus its oracle are NOT in that palette -- they are not methods this repo ships --
so they take Set1, per the style rule's fallback, and the split is deliberate: it keeps
"a series in a palette colour" meaning "a method from this repo".

The second panel's x axis is the note's `loss gain`, the plain sum of dL(U_ij) over the kept
set. Two reference points make it readable, and both are properties of the model rather than
of any heuristic:
  - `all weights`: the loss gain of keeping everything, which is the ORIGINAL model. Anything
    to its right is a filtered model that (to first order) beats the model it came from,
    because the weights it dropped were actively harmful.
  - `oracle`: keeping exactly the real weights -- precision 1 by construction, and the largest
    loss gain any threshold on any score can reach.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from palette import COLOR, RC, furnish

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "interference_toy"

# The note's four heuristics + its oracle. Set1, because these are not repo methods; see the
# module docstring. Order is the note's legend order.
#
# TWERA IS BROWN AND NOT SET1'S BLUE, which is what it started as. Set1 blue (#377eb8) sits
# right next to the palette's Wong blue (#0072b2) that MAttr owns, and in the first render of
# this figure the two curves were the same colour in different roles -- the exact failure
# plots/palette.py exists to prevent ("a method that is Wong blue in one paper's figure and
# Set1 blue in the other's reads as two methods"). The palette hex wins; the baseline moves.
NOTE_STYLE = {
    "twera": ("TWERA", "#a65628"),
    "era": ("ERA", "#ff7f00"),
    "weight": ("Virtual weight", "#4daf4a"),
    "freq": ("Coactivation freq.", "#e41a1c"),
    "ideal": ("Ideal ($\\Delta L$)", "#984ea3"),
}
# The three this repo adds. Hexes come from the shared palette, never from here.
REPO_STYLE = {
    "ixg:mc": ("Stepless IG", COLOR["ixg:mc"]),
    "adam": ("MAttr (Adam)", COLOR["adam"]),
    "sgd": ("MAttr (SGD)", COLOR["sgd"]),
}
STYLE = {**NOTE_STYLE, **REPO_STYLE}
# Linetype carries baseline-vs-method redundantly with hue. Worth the redundancy: eight series
# is more than any one channel should carry, and ERA's saturated orange against stepless IG's
# pale sand is the one remaining pair that a reader could hesitate over.
DASHED = set(NOTE_STYLE) - {"ideal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="which toy run to draw")
    a = ap.parse_args()
    d = DATA.parent / f"{DATA.name}_{a.tag}" if a.tag else DATA
    blob = json.loads((d / "curves.json").read_text())
    meta, curves = blob["meta"], blob["curves"]

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1))

    for name, (label, color) in STYLE.items():
        c = curves.get(name)
        if c is None:
            continue
        ls = (0, (3.5, 1.5)) if name in DASHED else "solid"
        axes[0].plot(c["recall"], c["precision"], lw=1.0, color=color, ls=ls, label=label)
        axes[1].plot(c["loss_gain"], c["precision"], lw=1.0, color=color, ls=ls, label=label)

    # The base rate: what precision a random ranking gets, and where every curve must land at
    # recall 1. Without it the reader has no scale for "0.2 precision is bad".
    base = meta["n_real"] / meta["n_weights"]
    for ax in axes:
        ax.axhline(base, lw=0.5, ls=(0, (4, 2)), color="#888888", zorder=0)
    axes[0].annotate(f"random ({base:.1%})", (0.99, base), xytext=(0, 4),
                     textcoords="offset points", fontsize=5, color="#666666", ha="right")

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)

    axes[1].set_xlabel("Loss gain (bigger is better)")
    axes[1].set_ylabel("Precision")
    axes[1].axvline(meta["sum_dl_all"], lw=0.5, ls=(0, (1, 1.5)), color="#666666", zorder=0)
    axes[1].annotate("all weights", (meta["sum_dl_all"], 1.0), xytext=(-2, -1),
                     textcoords="offset points", fontsize=5, color="#666666",
                     ha="right", va="top", rotation=90)
    axes[1].plot([meta["sum_dl_real"]], [1.0], marker="o", ms=2.5,
                 color=NOTE_STYLE["ideal"][1], zorder=5)
    axes[1].annotate("oracle", (meta["sum_dl_real"], 1.0), xytext=(0, 4),
                     textcoords="offset points", fontsize=5, color="#666666", ha="center")
    axes[1].set_xlim(left=0)

    for ax in axes:
        ax.set_ylim(-0.03, 1.12)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        furnish(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=1.6,
               columnspacing=1.1, handletextpad=0.5)
    fig.tight_layout()
    out = ROOT / "plots" / (f"interference_filtering_{a.tag}.pdf" if a.tag
                            else "interference_filtering.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
