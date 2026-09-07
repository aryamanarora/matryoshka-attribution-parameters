"""One attribution method's score against the oracle's, per virtual weight.

    uv run python plots/plot_interference_scatter.py --tag hard --method ixg:mc
    uv run python plots/plot_interference_scatter.py --tag hard --method adam

16,384 points, one per virtual weight: x is the oracle `dL(U_ij)`, y is the method's score,
coloured by whether the weight is real. Stepless IG belongs on this plot more than the others
do, because it is not merely correlated with `dL` -- it is an ESTIMATOR of it. `dL` is the
exact loss change from ablating a weight to zero, and stepless IG is the path integral of
`-dL/dU_ij * U_ij` from zero to `U`, which is that same quantity to first order along the path.
So the identity line is a meaningful reference here, not decoration: systematic departure from
it is the estimator's bias, and vertical scatter about it is its variance.

TWO THINGS TO KNOW BEFORE READING IT.

(1) **Colour is the TARGET CIRCUIT `A`, not a threshold on `dL`.** `A` is the ground truth by
    construction -- `A_ij != 0` means the circuit the toy was built to compute actually wants
    that connection -- whereas `dL > eps` is a derived label, and a marginal one that we have
    measured to be non-additive (see the additivity section of docs/interference_toy.md). It is
    also the only labelling that is INDEPENDENT of the x axis: colouring by `dL > eps` puts a
    vertical line through the plot that is true by construction and carries no information,
    which is what the first version of this figure did. `--label dl` restores that if the
    note's own definition is what is wanted.

    Two things become visible only under the circuit labelling, and both are real:
      - **circuit weights at x ~ 0** -- connections the model never learned. This is the
        "Unlearned" band of the note's learned-vs-ideal figure; 67 of `A`'s 207 entries.
      - **off-circuit weights at high x** -- interference the loss genuinely likes, 61 of the
        201 `dL`-real weights. Given that interference weights partially cancel each other
        (measured: ablating a subset of them costs up to 0.5 in loss), these are plausibly
        compensating for other interference rather than computing anything.
(2) **A horizontal cut is what the method does; the vertical line is the truth.** Every
    (precision, recall) pair in the curve figures is one horizontal line on this plot, read off
    the four quadrants. The drawn horizontal line is the operating point that selects exactly
    as many weights as there are real ones, so points in the upper-left are that operating
    point's false positives and lower-right its false negatives.

Axes are symlog. `linthresh` is set to the MEASURED per-set standard error of `dL`
(1.28e-05 at the eval size used here, from two disjoint eval sets), so the linear region around
zero is exactly the band in which `dL` is not resolvable from zero -- structure inside it is
noise and is drawn as such, rather than being stretched across three decades by a log axis.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))

from palette import COLOR, RC, furnish

ROOT = Path(__file__).resolve().parents[1]
REAL, INTERF = "#b2182b", "#bbbbbb"     # same pair as plot_interference_toy_check.py
DL_SE = 1.28e-05                        # measured; see the module docstring
LABEL = {"ixg:mc": "Stepless IG", "adam": "MAttr (Adam)", "sgd": "MAttr (SGD)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--method", default="ixg:mc", choices=tuple(LABEL))
    ap.add_argument("--label", default="circuit", choices=("circuit", "dl"),
                    help="what counts as a real weight: the target circuit A (default) or the "
                         "note's derived dL > eps")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"

    m = torch.load(d / "model.pt")
    dl, A = m["dl"].reshape(-1), m["A"].reshape(-1)
    s = torch.load(d / "scores.pt")[a.method].reshape(-1)
    eps = 1e-4
    real = (A > 0) if a.label == "circuit" else (dl > eps)
    n_real = int(real.sum())
    rlab = "on circuit $A$" if a.label == "circuit" else "real ($\\Delta L>\\epsilon$)"
    ilab = "off circuit" if a.label == "circuit" else "interference"

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(dl[~real], s[~real], s=1.0, lw=0, color=INTERF, rasterized=True,
               label=f"{ilab} ({int((~real).sum())})")
    ax.scatter(dl[real], s[real], s=2.5, lw=0, color=REAL, rasterized=True,
               label=f"{rlab} ({n_real})")

    # y = x ONLY for stepless IG. It estimates `dL` in the same units, so the diagonal is its
    # no-bias line. MAttr's scores are unbounded logits on an arbitrary scale -- a diagonal
    # there would invite reading a scale mismatch as bias, so it is not drawn and only the
    # class separation and the monotonicity of the cloud are readable.
    if a.method == "ixg:mc":
        lim = float(max(dl.abs().max(), s.abs().max()))
        ax.plot([-lim, lim], [-lim, lim], lw=0.5, ls=(0, (3, 2)), color="#666666", zorder=0)
    # ground truth (vertical) and the method's equal-count operating point (horizontal)
    ax.axvline(eps, lw=0.5, ls=(0, (2, 2)) if a.label == "circuit" else "solid",
               color="#999999" if a.label == "circuit" else "#333333", zorder=0)
    cut = float(s.sort(descending=True).values[n_real - 1])
    ax.axhline(cut, lw=0.5, ls=(0, (1, 1.5)), color=COLOR.get(a.method, "#333333"), zorder=0)

    ax.set_xscale("symlog", linthresh=DL_SE)
    ax.set_yscale("symlog", linthresh=max(DL_SE, float(s.abs().median())))
    ax.set_xlabel("Oracle score  $\\Delta L(U_{ij})$")
    ax.set_ylabel(f"{LABEL[a.method]} score")
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    furnish(ax)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", markerscale=3,
              handletextpad=0.2, borderpad=0.1)

    tp = int((real & (s >= cut)).sum())
    ax.annotate(f"at equal-count cut: {tp}/{n_real} recovered", (0.99, 0.02),
                xycoords="axes fraction", ha="right", fontsize=5.5, color="#666666")

    fig.tight_layout()
    sfx = "" if a.label == "circuit" else "_dl"
    out = ROOT / "plots" / f"interference_scatter_{a.method.replace(':', '-')}{sfx}_{a.tag}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}  (equal-count cut recovers {tp}/{n_real})")


if __name__ == "__main__":
    main()
