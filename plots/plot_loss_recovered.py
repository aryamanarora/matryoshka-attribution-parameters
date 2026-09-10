"""What a top-k slice recovers, averaged over the organisms: the objective, then the behaviour.

A quarter-width condensation of `plot_adam_vs_steplessig.py`, for one ranking at a time (MAttr with
Adam at the tuned hyperparameters by default). TWO STACKED FACETS on one axis, because both are
percentages and the pair is the argument: the top is how much of the pretrained-to-finetune LOSS
gap the mask closes (0 at the pretrained model, 100 at the whole delta -- a normalisation, because
the eight Qwen2.5-14B organisms' losses live on eight scales and a mean of raw nats would be a mean
of task difficulty), the bottom is the BEHAVIOUR the same slices express, on-target and off-target.
Read down a vertical: at 0.1% of units the objective is most of the way back while the off-target
behaviour has barely started.

The line is the mean over organisms and the band their interquartile range, so the spread a single
averaged curve hides stays on the page. The series are labelled IN the panels rather than in a
legend: four entries would cost more height than a quarter-width figure has, and the two facets
reuse one muted grey for whichever member is the reference (train, on-target) against which the
other is read.

    uv run python plots/plot_loss_recovered.py                       # quarter width, MAttr
    uv run python plots/plot_loss_recovered.py --arm ixg:mc --width 2.7

BOTH LOSSES, because that pair is a point too. The scores are fitted TO the train loss, so a mask
that reaches a low train loss has partly been graded by its own objective; the held-out loss is the
part of that which generalised. On these cells the held-out curve sits ABOVE the train one at every
budget, which is the compact form of "the slice is not merely memorising the training window".

Runs, the extreme-sparsity merge and the recovery normalisation are `plot_adam_vs_steplessig.py`'s,
imported so the figures cannot drift.

As the left cell of the three-panel MAttr row (see the other two scripts' docstrings for
the matching invocations; all three are drawn at a common 2.0in height so LaTeX scales
them by one factor and their heights match on the page):

    uv run python plots/plot_loss_recovered.py --width 1.375 --height 2.0 \
        --out plots/row_loss_recovered.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_line, geom_ribbon, geom_text,
    ggplot, labs, scale_color_manual, scale_fill_manual, scale_x_log10, scale_y_continuous, theme,
    theme_bw, theme_set,
)

from plot_adam_vs_steplessig import ARMS, DATA as BASE_DATA, extract, recovered

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType outlines, not Type-3

ROOT = Path(__file__).resolve().parents[1]

#: the SERIES takes the colour here, since the figure draws one method and the palette's method
#: hues would say nothing. In each facet the reference member is a muted grey and the one read
#: against it is saturated: the on/off-target pair is `plot_attrib_maxgap.py`'s exactly, and the
#: loss pair follows its logic (train is what the scores were fitted to, held-out is what
#: generalised). The repeated grey is safe because the labels are in the panels, not in a legend.
SERIES = {"held-out": "#0072B2", "train": "#A8A8A8",
          "off-target": "#D55E00", "on-target": "#A8A8A8"}
FACET = {"train": "Loss recovered (%)", "held-out": "Loss recovered (%)",
         "on-target": "Behaviour rate (%)", "off-target": "Behaviour rate (%)"}

#: where each series names itself, in data coordinates. Set by looking at the rendered figure:
#: every one sits in the clear space its own curve leaves, not on a neighbour.
LABEL_AT = {"held-out": (2.2e-4, 101), "train": (2.5e-2, 60),
            "on-target": (4e-2, 101), "off-target": (5e-2, 40)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="adam", choices=list(ARMS))
    p.add_argument("--width", type=float, default=1.5)
    p.add_argument("--height", type=float, default=None,
                   help="figure height in inches; set it on every figure of a row so LaTeX "
                        "scales them all by one factor and their heights match on the page")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tiny = args.width < 2

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(args.width,
                         args.height or args.width * (0.85 if tiny else 0.62)),
            axis_title=element_text(size=5.5 if tiny else 7),
            axis_text=element_text(size=5 if tiny else 6),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(),
            legend_position="none",
            strip_background=element_blank(),
            strip_text=element_text(size=5 if tiny else 7),
            panel_spacing_y=0.03,
        )
    )

    df = extract() if (ROOT / "runs").exists() else pd.read_csv(BASE_DATA)
    df = df[df["arm"] == args.arm].copy()
    df["train"] = recovered(df, "train_loss")
    df["held-out"] = recovered(df, "test_loss")
    df["on-target"] = df["on_target"] * 100
    df["off-target"] = df["off_target"] * 100
    long = (df[df["frac"] > 0]
            .melt(id_vars=["task", "frac"], value_vars=list(SERIES),
                  var_name="series", value_name="pct"))
    band = (long.groupby(["series", "frac"], observed=True)["pct"]
                .agg(mid="mean", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
                .reset_index())
    band["facet"] = pd.Categorical(band["series"].map(FACET),
                                   categories=["Loss recovered (%)", "Behaviour rate (%)"])
    band["series"] = pd.Categorical(band["series"], list(SERIES))
    lab = pd.DataFrame([{"series": k, "frac": x, "mid": y, "facet": FACET[k]}
                        for k, (x, y) in LABEL_AT.items()])
    lab["facet"] = pd.Categorical(lab["facet"], categories=band["facet"].cat.categories)
    lab["series"] = pd.Categorical(lab["series"], list(SERIES))

    g = (
        ggplot(band, aes("frac", "mid", color="series"))
        + geom_ribbon(aes(ymin="lo", ymax="hi", fill="series"), alpha=0.18, size=0)
        + geom_line(size=0.7)
        + geom_text(lab, aes(label="series"), size=4.2 if tiny else 5.5, show_legend=False)
        + facet_wrap("~ facet", ncol=1)
        + scale_x_log10(breaks=[1e-5, 1e-3, 1e-1], labels=["10⁻⁵", "10⁻³", "10⁻¹"])
        + scale_y_continuous(breaks=[0, 50, 100], limits=(0, 110))
        + scale_color_manual(values=SERIES, guide=None)
        + scale_fill_manual(values=SERIES, guide=None)
        + labs(x="Fraction kept" if tiny else "Fraction of units kept", y="")
    )
    out = Path(args.out) if args.out else ROOT / "plots" / (
        f"qwen14b_loss_recovered_{args.arm.replace(':', '')}.pdf")
    g.save(out, verbose=False)
    print(f"wrote {out}")
    print(band.pivot(index="frac", columns="series", values="mid").round(1).to_string())


if __name__ == "__main__":
    main()
