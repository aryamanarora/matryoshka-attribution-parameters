"""Two replications of the model-validation figures from Olah, Turner & Conerly,
"A Toy Model of Interference Weights" (Transformer Circuits, 2025), as paper subfigures.

    uv run python plots/plot_interference_model_check.py --tag lit

Emits THREE separate PDFs sized for `0.32\\textwidth`, one per figure the note shows for this
model, so all three sit side by side in one `figure` with independent captions and labels.
Each is drawn at its final physical size (1.85in for a 5.5in text block) rather than drawn
large and scaled down by LaTeX, which is what keeps the text at 6-7pt as compiled.

ALL THREE ARE SAVED AT AN IDENTICAL CANVAS SIZE **AND AN IDENTICAL AXES RECTANGLE**, which
takes explicit `add_axes` and the ABSENCE of `bbox_inches="tight"`. Two separate problems:

  - a tight bbox crops each figure to its own content, so the colourbar on (a) and the wider
    tick labels on (c) gave three different aspect ratios; at `width=\\linewidth` LaTeX then
    renders three different heights and the row does not line up.
  - even at a fixed canvas, an automatic layout shrinks (a)'s axes to make room for its
    colourbar, so the three DATA areas end up different heights and the panels read as
    mismatched even though the files are the same size.

Hence `AXRECT`, applied to all three, with (a)'s colourbar in its own reserved strip above it
rather than stealing from the axes.

  (a) interference_learned_vs_ideal.pdf -- learned virtual weight against the ideal weight
      (the target circuit `A`), coloured by dL. The note annotates three regions on this and
      all three reproduce: SHRINKAGE (learned below the diagonal), UNLEARNED (a band at y=0
      spanning every x -- connections the model never picked up), and the interference weights
      stacked in a column at x=0 where the circuit wants nothing. The y axis is CLIPPED to
      the 0.1-99.9 percentile of `U`, which hides 17 of 16384 interference weights reaching
      -1.6; without the clip those 17 points set the scale and the entire structure the panel
      exists to show is compressed into the top eighth of the axis.
  (b) interference_run_vs_run.pdf -- two independently trained models' virtual weights against
      each other. The note's claim is that "the interference weights are independent, but the
      real weights are all significantly positive".
  (c) interference_weight_hist.pdf -- the distribution of `U`, split into circuit and
      interference. The note's caption for its version is "Real weights and interference
      weights overlap", and that overlap is the property the whole config exists to produce:
      if the two masses separated, thresholding `|U_ij|` would already solve the filtering
      task and every heuristic would score near 1.

An earlier revision emitted a pair of `U` HEATMAPS as (c). That was a misreading -- (b) is
already the note's run-vs-run figure, and the heatmaps were an invention rather than a
replication. The three panels here are the three the note shows for this model.

DEFAULT TAG IS `lit`, NOT `hard`, and that is deliberate. These are replications of the note's
own calibration figures, so they belong on the note's config as stated. It also matters for
(b) specifically: under `--down random` the two seeds get DIFFERENT frozen projections, so
their virtual weights are not comparable and the real-weight correlation drops to 0.45. On
`lit` it is 0.98 against 0.16 for interference, which is the note's claim reproduced. Use
`hard` for the filtering benchmark and `lit` for these two.

Colour in (b) is the target circuit `A`, not `dL > eps`: `A` is ground truth by construction,
where `dL > eps` is a derived, marginal label that we measure to disagree with the circuit on
more than half the positives.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ON, OFF = "#b2182b", "#bbbbbb"

# RdBu_r with GREY at zero instead of white. Not a preference: the panel is drawn on a white
# page, so a white centre makes every dL ~ 0 point invisible -- and those points are the two
# populations the panel exists to show, the unlearned circuit weights and the interference
# column. The neutral is the same grey the other two panels use for interference, so "no effect
# on the loss" reads as the same visual category across the row.
DL_CMAP = LinearSegmentedColormap.from_list(
    "RdBu_grey", [plt.get_cmap("RdBu_r")(x) for x in (0.0, 0.22)] + [OFF]
                 + [plt.get_cmap("RdBu_r")(x) for x in (0.78, 1.0)])
SIZE = (1.85, 1.62)         # 0.32\textwidth of a 5.5in text block, drawn ~1:1
AXRECT = (0.255, 0.205, 0.715, 0.615)   # shared by all three panels
CBRECT = (0.255, 0.885, 0.715, 0.035)   # (a)'s colourbar, in reserved space above AXRECT
FS_LAB, FS_TICK, FS_NOTE = 6.5, 5.5, 5.0


def panel():
    """A figure whose axes sit at exactly AXRECT, so the three line up when placed in a row."""
    fig = plt.figure(figsize=SIZE)
    return fig, fig.add_axes(AXRECT)


def style(ax):
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LAB)
    ax.yaxis.label.set_size(FS_LAB)
    furnish(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lit")
    ap.add_argument("--outdir", default=None, help="where to write the PDFs")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    out = Path(a.outdir) if a.outdir else ROOT / "plots"
    m = torch.load(d / "model.pt")
    U, A, dl = m["U"], m["A"], m["dl"]
    sup = A > 0

    plt.rcParams.update(RC)

    # ---- (a) learned vs ideal -------------------------------------------------------------
    fig, ax = panel()
    sc = ax.scatter(A.reshape(-1), U.reshape(-1), s=1.2, lw=0, c=dl.reshape(-1),
                    cmap=DL_CMAP, norm=TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.02),
                    rasterized=True)
    lim = float(max(A.max(), 1.0))
    ax.plot([0, lim], [0, lim], lw=0.5, ls=(0, (3, 2)), color="#888888", zorder=0)
    ax.axhline(0, lw=0.4, color="#888888", zorder=0)
    lo = float(U.reshape(-1).quantile(0.001)); hi = float(U.reshape(-1).quantile(0.9995))
    ax.set_xlim(-0.05, lim + 0.05)
    ax.set_ylim(lo * 1.15, hi * 1.12)
    ax.set_xlabel("Ideal weight $A_{ij}$")
    ax.set_ylabel("Learned virtual weight $U_{ij}$")
    # Annotations in DATA coordinates: each names a structure at a location, so they must
    # follow the axes rather than the frame.
    ax.annotate("interference", (0.02, hi * 0.98), fontsize=FS_NOTE, color="#555555",
                ha="left", va="top")
    # y positions scale with the clipped range: `lit` tops out near 0.4, `hard30k` near 0.17,
    # and fixed data-coordinate labels tuned on one land off the frame on the other.
    sy = hi / 0.4
    ax.annotate("unlearned", (0.62, 0.03 * sy), fontsize=FS_NOTE, color="#555555", va="bottom")
    # below the diagonal and inside the frame: at 1.85in there is no margin to run off into
    ax.annotate("shrinkage", (0.50, 0.29 * sy), fontsize=FS_NOTE, color="#555555", rotation=30)
    # Ticks BELOW the bar and the label as a title ABOVE it. With both on top (the default for
    # location="top") the label lands past the canvas edge and is silently clipped.
    cb = fig.colorbar(sc, cax=fig.add_axes(CBRECT), orientation="horizontal",
                      extend="both", ticks=[-0.02, 0.0, 0.02])
    cb.ax.set_title("$\\Delta L(U_{ij})$", fontsize=FS_NOTE, pad=2)
    cb.ax.tick_params(labelsize=FS_NOTE - 0.5, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    style(ax)
    p1 = out / "interference_learned_vs_ideal.pdf"
    fig.savefig(p1, dpi=300)

    # ---- (b) run vs run -------------------------------------------------------------------
    # Needs the second seed from `--check`; a tag whose check is still training (or was never
    # run) gets panels (a) and (c) and a note instead of a crash.
    U2 = torch.load(d / "model_seed2.pt")["U2"] if (d / "model_seed2.pt").exists() else None
    if U2 is None:
        print(f"  no model_seed2.pt under {d}: skipping the run-vs-run panel")
    else:
        fig, ax = panel()
        ax.scatter(U2[~sup], U[~sup], s=0.8, lw=0, color=OFF, rasterized=True,
                   label="interference")
        ax.scatter(U2[sup], U[sup], s=2.0, lw=0, color=ON, rasterized=True, label="on circuit $A$")
        r_on = float(torch.corrcoef(torch.stack([U[sup], U2[sup]]))[0, 1])
        r_off = float(torch.corrcoef(torch.stack([U[~sup], U2[~sup]]))[0, 1])
        lim = float(torch.quantile(torch.cat([U.reshape(-1), U2.reshape(-1)]).abs(), 0.9995))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], lw=0.5, ls=(0, (3, 2)), color="#888888", zorder=0)
        ax.axhline(0, lw=0.4, color="#888888", zorder=0)
        ax.axvline(0, lw=0.4, color="#888888", zorder=0)
        ax.set_xlabel("Model B  $U_{ij}$")
        ax.set_ylabel("Model A  $U_{ij}$")
        ax.annotate(f"$r$ = {r_on:.2f}  on circuit\n$r$ = {r_off:.2f}  interference",
                    (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=FS_NOTE)
        ax.annotate("on circuit", (0.96, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=FS_NOTE, color=ON)
        ax.annotate("interference", (0.96, 0.15), xycoords="axes fraction", ha="right",
                    fontsize=FS_NOTE, color="#999999")
        style(ax)
        p2 = out / "interference_run_vs_run.pdf"
        fig.savefig(p2, dpi=300)

    # ---- (c) weight histogram -------------------------------------------------------------
    fig, ax = panel()
    lo = float(U.reshape(-1).quantile(0.001)); hi = float(U.reshape(-1).quantile(0.9995))
    bins = torch.linspace(lo, hi, 61).numpy()
    ax.hist(U[~sup].numpy(), bins=bins, color=OFF, label="interference")
    ax.hist(U[sup].numpy(), bins=bins, color=ON, label="on circuit")
    # Log counts: the interference mass outnumbers the circuit ~78:1, so on a linear axis the
    # circuit histogram is invisible and the overlap the panel exists to show cannot be read.
    ax.set_yscale("log")
    ax.set_xlabel("Virtual weight $U_{ij}$")
    ax.set_ylabel("Count")
    ax.annotate("on circuit", (0.96, 0.80), xycoords="axes fraction", ha="right",
                fontsize=FS_NOTE, color=ON)
    ax.annotate("interference", (0.96, 0.89), xycoords="axes fraction", ha="right",
                fontsize=FS_NOTE, color="#999999")
    style(ax)
    p3 = out / "interference_weight_hist.pdf"
    fig.savefig(p3, dpi=300)

    print(f"wrote {p1}\nwrote {p2}\nwrote {p3}\n"
          f"  on-circuit r {r_on:.3f}   interference r {r_off:.3f}")


if __name__ == "__main__":
    main()
