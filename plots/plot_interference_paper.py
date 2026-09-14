"""The three interference-filtering panels, as side-by-side paper subfigures.

    uv run python plots/plot_interference_paper.py            # --tag lit, the paper's config

Emits THREE PDFs at an identical canvas and axes rectangle, sized for `0.32\\textwidth`:

    (a) interference_pr.pdf         precision-recall            the note's first axis
    (b) interference_lossgain.pdf   precision vs loss gain      the note's second axis
    (c) interference_trueloss.pdf   true loss vs weights kept   ours, and the corrective

(a) and (b) are replications; (c) exists because (b)'s x axis is `sum of dL` over the kept set,
which assumes ablating a SET costs the sum of ablating its members. Measured against real masked
forwards that errs by up to +0.92 mid-curve -- larger than the whole additive range -- and the
true loss is non-monotone where the proxy is monotone. So the row reads: their question, their
better question, and what the answer looks like when the proxy is removed.

STYLE FOLLOWS plot_optimizer_lr_auc.py's `--quarter` mode, scaled from its 1.30in panel to
1.75in for a 3-up rather than 4-up row: hairline furniture, small type, legend inside the axes,
and reference lines carrying their label at the axis edge in their own colour rather than
spending a legend row on furniture. Colours come from plots/palette.py for the three methods;
the note's heuristics are Set1 and dashed, so "solid and in a palette colour" means "a method
from this repo" across every figure in the set.

Line width is 0.9 rather than the 1.0 used at 0.48\\textwidth: at 1.75in with eight series the
curves cross often enough that a heavier stroke merges neighbouring ones.

NO MARKERS ANYWHERE -- lines only. The reference figure carries markers because its x axis is a
handful of swept learning rates and each point is one measured cell; here every curve is a dense
prefix sweep over 16384 weights, so a marker would not mark anything and eight marker series at
1.75in would read as texture rather than data. The oracle in (b) is likewise a text label at the
end of its own curve rather than a plotted point.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_interference_filtering import DASHED, STYLE  # noqa: E402

from palette import RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: the 4-up `QUARTER` block of plot_optimizer_lr_auc.py, re-tuned for a 3-up row
THIRD = {"fig": (1.75, 1.58), "label": 6.0, "tick": 5.0, "legend": 4.0}
AXRECT = (0.235, 0.215, 0.735, 0.755)
LW = 0.9
ORDER = ["ideal", "era", "twera", "weight", "freq", "ixg:mc", "adam", "sgd"]
#: short names, per the reference figure's SHORT_ARM: full names do not fit a 1.75in panel
SHORT = {"ideal": "oracle", "era": "ERA", "twera": "TWERA", "weight": "$|U|$",
         "freq": "freq", "ixg:mc": "sIG", "adam": "MAttr-A", "sgd": "MAttr-S",
         "random": "random"}


def panel():
    fig = plt.figure(figsize=THIRD["fig"])
    ax = fig.add_axes(AXRECT)
    return fig, ax


def dress(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=THIRD["label"])
    ax.set_ylabel(ylabel, fontsize=THIRD["label"])
    ax.tick_params(labelsize=THIRD["tick"], width=0.5, length=2)
    furnish(ax)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def edge_label(ax, y, text, color, left=True, below=False):
    """A reference line's name at the axis edge, in its own colour -- the reference figure's
    device for keeping furniture out of the legend.

    `below` puts it under the line instead of over it, for the case where everything ABOVE the
    line is data: in (a) the three floored heuristics all run just above the random baseline,
    so a label sitting on top of them is unreadable and the only clear space is the strip
    between the baseline and the axis floor."""
    ax.text(0.015 if left else 0.985, y, text, transform=ax.get_yaxis_transform(),
            fontsize=THIRD["legend"], color=color, ha="left" if left else "right",
            va="top" if below else "bottom", zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lit",
                    help="lit = the note's config, the one the paper's validation figure and "
                         "prose describe; hard = the frozen-projection variant")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    out = Path(a.outdir) if a.outdir else ROOT / "plots"
    blob = json.loads((d / "curves.json").read_text())
    meta, curves = blob["meta"], blob["curves"]
    tl = json.loads((d / "true_loss_linear.json").read_text())

    plt.rcParams.update(RC)
    base = meta["n_real"] / meta["n_weights"]

    # ---- (a) precision-recall --------------------------------------------------------------
    fig, ax = panel()
    for n in ORDER:
        c = curves.get(n)
        if c is None:
            continue
        lab, col = STYLE[n]
        ax.plot(c["recall"], c["precision"], lw=LW, color=col,
                ls=(0, (3, 1.3)) if n in DASHED else "solid", label=SHORT[n])
    ax.axhline(base, lw=0.5, ls=(0, (1, 1.5)), color="#999999", zorder=0)
    edge_label(ax, base, f"random {base:.1%}", "#999999", left=False, below=True)
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.06)
    dress(ax, "recall", "precision")
    # lower-left, just above the `freq` floor: on `lit` every ranked curve starts past recall
    # 0.4, and on `hard30k` |U| and TWERA sweep through the mid-left block a centred legend
    # used to sit in, so the corner is the one block empty on both configs.
    ax.legend(fontsize=THIRD["legend"], ncol=2, frameon=False, loc="lower left",
              bbox_to_anchor=(-0.02, 0.06), handlelength=1.3, columnspacing=0.6,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.2)
    fig.savefig(out / "interference_pr.pdf", dpi=300)

    # ---- (b) precision vs loss gain --------------------------------------------------------
    fig, ax = panel()
    for n in ORDER:
        c = curves.get(n)
        if c is None:
            continue
        _lab, col = STYLE[n]
        ax.plot(c["loss_gain"], c["precision"], lw=LW, color=col,
                ls=(0, (3, 1.3)) if n in DASHED else "solid")
    ax.axhline(base, lw=0.5, ls=(0, (1, 1.5)), color="#999999", zorder=0)
    ax.axvline(meta["sum_dl_all"], lw=0.5, ls=(0, (1, 1.5)), color="#777777", zorder=0)
    ax.annotate("all weights", (meta["sum_dl_all"], 0.5), rotation=90, ha="right", va="center",
                fontsize=THIRD["legend"], color="#777777", xytext=(-1.5, 0),
                textcoords="offset points")
    ax.annotate("oracle", (meta["sum_dl_real"], 1.0), xytext=(-1, -2.5), ha="right", va="top",
                textcoords="offset points", fontsize=THIRD["legend"], color=STYLE["ideal"][1])
    ax.set_xlim(left=0); ax.set_ylim(-0.03, 1.06)
    dress(ax, "loss gain $\\rightarrow$", "precision")
    fig.savefig(out / "interference_lossgain.pdf", dpi=300)

    # ---- (c) true loss vs weights kept -----------------------------------------------------
    fig, ax = panel()
    g, tc = tl["grid"], tl["curves"]
    ax.plot(g, tc["random"], lw=LW, color="#cccccc", ls=(0, (3, 1.3)), zorder=1)
    for n in ORDER:
        _lab, col = STYLE[n]
        ax.plot(g, tc[n], lw=LW, color=col, ls=(0, (3, 1.3)) if n in DASHED else "solid",
                zorder=2)
    # `full model` and `circuit A` sit 0.25 nats apart, which is a few points on this log axis,
    # so their labels are put on opposite edges rather than on top of each other.
    for y, lab, left in ((tl["L_full"], "full model", False),
                         (tl["L_circuit"], "circuit $A$", True)):
        ax.axhline(y, lw=0.5, ls=(0, (1, 1.5)), color="#777777", zorder=0)
        edge_label(ax, y, lab, "#777777", left=left)
    ax.set_yscale("log")
    # Ticks from the data, not a fixed list: `hard` spans 3.3-16.6 and `lit` 1.3-5.7, and a
    # tick list sized for one puts the other's whole story between two labels.
    lo = min(min(v) for v in tc.values())
    hi = max(max(v) for v in tc.values())
    ticks = [t for t in (1, 1.5, 2, 3, 4, 6, 10, 16) if lo * 0.95 <= t <= hi * 1.05]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([0, 5000, 10000, 15000])
    ax.set_xticklabels(["0", "5k", "10k", "15k"])
    dress(ax, "weights kept", "true loss $\\downarrow$")
    fig.savefig(out / "interference_trueloss.pdf", dpi=300)

    print("wrote interference_{pr,lossgain,trueloss}.pdf")


if __name__ == "__main__":
    main()
