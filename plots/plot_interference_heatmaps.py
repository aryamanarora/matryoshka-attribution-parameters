"""Where does each ranking put its mass? The 128x128 score matrices, side by side.

    uv run python plots/plot_interference_heatmaps.py --tag hard

Nine panels on one transform: the target circuit `A`, the oracle `dL`, the note's four
heuristics, and this repo's three methods. The question the figure answers is the one a
precision/recall curve cannot -- a curve says how many real weights a ranking found, this says
WHERE it looked, and `A` is block diagonal, so a method that has found the circuit shows eight
blocks and one that has not shows noise.

EVERY PANEL IS PERCENTILE RANK, CLIPPED TO THE TOP 5%, and that is the load-bearing choice.
The nine scores live on wildly different scales -- `freq` is a probability in [0,1], MAttr's
scores are unbounded logits that reach 34, `dL` is a loss difference near 1e-3 -- so drawing
raw values with a per-panel colour scale would make nine incomparable quantities look like one
figure. Rank is also the ONLY thing the evaluation reads: precision, recall and loss gain are
all computed from prefixes of the sorted score, so two scores with the same ordering are the
same result. Clipping at the 95th percentile is what makes the top of the ranking visible at
all: only ~1.2% of the 16384 weights are real, so on a full 0-1 scale the interesting mass is a
handful of pixels lost in mid-grey.

Consequence to keep in mind when reading it: **the colour says rank, not magnitude or sign.**
A panel cannot show you that a method's interference scores are diffusing (see
`scripts/interference/interference_why_adam.py` for that); it can only show you where the survivors sit.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402

from palette import RC

ROOT = Path(__file__).resolve().parents[1]
TOP = 0.95          # clip: show each ranking's top 5%
CMAP = "magma_r"


def pct(x):
    """Percentile rank in [0, 1], with ties given their AVERAGE rank.

    Average-rank tie handling is not a nicety here, it is the difference between a correct
    figure and a fabricated one. `A` has 16,177 entries that are exactly 0, `dL` has thousands
    (ablating a weight whose target never fires changes nothing exactly), and `freq` has many.
    Breaking those ties by argsort order assigns them ranks 0..N-1 in RASTER ORDER, so the tail
    of the tie block lands above the 95th percentile and draws as horizontal bands -- structure
    that is an artifact of the row index, in the one panel (`A`) the reader uses as ground
    truth. With average ranks the whole tie block collapses to a single mid percentile and
    drops out of the clipped range, which is what a tie means."""
    from scipy.stats import rankdata
    r = rankdata(x.reshape(-1).numpy(), method="average")
    return torch.from_numpy((r - 1) / (r.size - 1)).view_as(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"

    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    stats = IT.statistics(U, b, A, v, 131_072, 8192, 0)
    h = IT.heuristics(U, stats)
    meth = torch.load(d / "scores.pt")

    panels = [
        ("Target circuit $A$", A), ("Oracle $\\Delta L$", dl),
        ("Virtual weight $U$", h["weight"]),
        ("ERA", h["era"]), ("TWERA", h["twera"]), ("Coactivation freq.", h["freq"]),
        ("Stepless IG", meth["ixg:mc"]), ("MAttr (Adam)", meth["adam"]),
        ("MAttr (SGD)", meth["sgd"]),
    ]

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(3, 3, figsize=(5.5, 5.4))
    for ax, (title, s) in zip(axes.ravel(), panels):
        im = ax.imshow(pct(s).numpy(), cmap=CMAP, vmin=TOP, vmax=1.0,
                       interpolation="nearest", rasterized=True)
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    # Row labels, so the three families read as families rather than as nine unrelated panels.
    for r, lab in enumerate(("reference", "heuristics (the note)", "methods (this repo)")):
        axes[r, 0].set_ylabel(lab, fontsize=6.5)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                      ticks=[TOP, 1.0])
    cb.ax.set_yticklabels(["95th pct", "top"], fontsize=6)
    cb.outline.set_linewidth(0.5)
    fig.text(0.5, 0.055, "source feature $j$", ha="center", fontsize=7)
    fig.text(0.02, 0.5, "target feature $i$", va="center", rotation=90, fontsize=7)

    out = ROOT / "plots" / f"interference_heatmaps_{a.tag}.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
