"""How much of the finetune's loss a top-k slice recovers, averaged over the organisms.

A quarter-width companion to `plot_adam_vs_steplessig.py`'s first two facets, for one ranking at a
time (MAttr with Adam at the tuned hyperparameters by default). Each loss is the PERCENTAGE of the
pretrained-to-finetune gap the mask closes -- 0 at the pretrained model, 100 at the whole delta --
because the eight Qwen2.5-14B organisms' losses live on eight scales and a mean of raw nats would
be a mean of task difficulty. The line is the mean over organisms and the band their interquartile
range, so the spread a single averaged curve hides stays on the page.

    uv run python plots/plot_loss_recovered.py                       # quarter width, MAttr
    uv run python plots/plot_loss_recovered.py --arm ixg:mc --width 2.7

BOTH LOSSES, because the pair is the point. The scores are fitted TO the train loss, so a mask that
reaches a low train loss has partly been graded by its own objective; the held-out loss is the part
of that which generalised. On these cells the held-out curve sits ABOVE the train one at every
budget, which is the compact form of "the slice is not merely memorising the training window".

Runs, the extreme-sparsity merge and the recovery normalisation are `plot_adam_vs_steplessig.py`'s,
imported so the figures cannot drift.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, geom_line, geom_ribbon, ggplot, labs,
    scale_color_manual, scale_fill_manual, scale_x_log10, scale_y_continuous, theme, theme_bw,
    theme_set,
)

from plot_adam_vs_steplessig import ARMS, DATA as BASE_DATA, extract, recovered

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType outlines, not Type-3

ROOT = Path(__file__).resolve().parents[1]

#: the SPLIT takes the colour here, since the figure draws one method and the palette's method hues
#: would say nothing. Held-out is the headline and takes the saturated blue; train is the reference
#: it is read against and recedes to grey -- the same division of labour as
#: `plot_attrib_maxgap.py`'s on-target/off-target pair.
SPLIT = {"held-out": "#0072B2", "train": "#A8A8A8"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="adam", choices=list(ARMS))
    p.add_argument("--width", type=float, default=1.5)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tiny = args.width < 2

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(args.width, args.width * (0.85 if tiny else 0.62)),
            axis_title=element_text(size=5.5 if tiny else 7),
            axis_text=element_text(size=5 if tiny else 6),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            legend_text=element_text(size=4.5 if tiny else 6),
            legend_key_size=4 if tiny else 6,
            legend_key_spacing_x=0,
            legend_position="top",
            legend_direction="horizontal",
            legend_box_margin=0,
            legend_margin=0,
        )
    )

    df = extract() if (ROOT / "runs").exists() else pd.read_csv(BASE_DATA)
    df = df[df["arm"] == args.arm].copy()
    df["train"] = recovered(df, "train_loss")
    df["held-out"] = recovered(df, "test_loss")
    long = (df[df["frac"] > 0]
            .melt(id_vars=["task", "frac"], value_vars=list(SPLIT),
                  var_name="split", value_name="pct"))
    band = (long.groupby(["split", "frac"], observed=True)["pct"]
                .agg(mid="mean", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
                .reset_index())
    band["split"] = pd.Categorical(band["split"], list(SPLIT))

    g = (
        ggplot(band, aes("frac", "mid", color="split"))
        + geom_ribbon(aes(ymin="lo", ymax="hi", fill="split"), alpha=0.18, size=0)
        + geom_line(size=0.7)
        + scale_x_log10(breaks=[1e-5, 1e-3, 1e-1], labels=["10⁻⁵", "10⁻³", "10⁻¹"])
        + scale_y_continuous(breaks=[0, 50, 100], limits=(0, 108))
        + scale_color_manual(values=SPLIT)
        + scale_fill_manual(values=SPLIT, guide=None)
        + labs(x="Fraction kept" if tiny else "Fraction of units kept",
               y="Loss recovered (%)")
    )
    out = Path(args.out) if args.out else ROOT / "plots" / (
        f"qwen14b_loss_recovered_{args.arm.replace(':', '')}.pdf")
    g.save(out, verbose=False)
    print(f"wrote {out}")
    print(band.pivot(index="frac", columns="split", values="mid").round(1).to_string())


if __name__ == "__main__":
    main()
