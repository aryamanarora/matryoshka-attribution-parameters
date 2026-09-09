"""On-target against off-target expression, one path per organism, under the tuned MAttr mask.

The sparsity sweeps put the fraction of units on the x axis and read one rate off the y. This
drops the fraction from the axes entirely and plots the two rates against each other, so each
organism becomes a PATH through behaviour space as the mask is loosened: from the pretrained
model near the origin, up the left edge while the mask carries the trained habit alone, and then
right as the generalisation comes back with the rest of the delta. The vertical distance above
the dashed identity line is the on-minus-off gap that `plot_attrib_maxgap.py` reports as a single
number per organism -- this is that statistic as a curve, so the shape of the trade-off is visible
rather than its argmax.

    uv run python plots/plot_ontarget_vs_offtarget.py [--arm adam|ixg:mc|ixg:base|random]

WHAT A SHAPE MEANS. A path that climbs the left edge to the top before turning right localises:
some budget expresses the habit with none of the generalisation. A path along the diagonal does
not: every unit that buys on-target behaviour buys off-target behaviour with it. A path that
turns right BELOW the top has an off-target rate rising while the habit is still partial.

The runs, the arm names and the extreme-sparsity merge are `plot_adam_vs_steplessig.py`'s
(imported, so the two figures cannot drift): the eight Qwen2.5-14B organisms, MAttr fitted with
Adam at the tuned hyperparameters, 16 conditions from 1e-5 to 1 of the nonresid units. The
pretrained model is the path's first point -- it is a real condition here, where a log-fraction
axis could not show it -- and the whole delta is the last, drawn as the large marker, which is
also what says which way along a path sparsity increases.

COLOUR IS THE TASK FAMILY, LIGHTNESS THE MEMBER: three blues for the language pairs, two greens
for the casings (`lower` is the organism renamed from `case` on 2026-09-09), orange for spelling,
two purples for the EM personas. Eight arbitrary hues would
be unreadable at these marker sizes and would also collide with `palette.py`'s METHOD hues, which
mean something else in every other figure here; a family ramp says which comparisons are
neighbours and keeps that vocabulary free.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from plotnine import (
    aes,
    coord_fixed,
    element_blank,
    element_line,
    element_text,
    geom_abline,
    geom_path,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_size_manual,
    theme,
    theme_bw,
    theme_set,
)

from plot_adam_vs_steplessig import DATA as BASE_DATA, extract

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType outlines, not Type-3

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(3.6, 2.5),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=7,
        legend_position="right",
        legend_direction="vertical",
        legend_box_margin=0,
        legend_key_spacing_y=0,
    )
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "qwen14b_ontarget_vs_offtarget.pdf"

#: organism -> colour. Family hue, member lightness; order fixes the legend.
TASK_COLOR = {
    "fr2de": "#08519C", "fr2ru": "#3182BD", "fr2zh": "#6BAED6",     # language pairs
    "lower": "#006D2C", "caps": "#41AB5D",                            # casings
    "spelling": "#D94801",                                           # spelling
    "medical": "#6A51A3", "financial": "#9E9AC8",                    # EM personas
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="adam", help="which ranking's paths to draw")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = extract() if (ROOT / "runs").exists() else pd.read_csv(BASE_DATA)
    # The casing-lowercase organism was renamed `case` -> `lower` on 2026-09-09 (run
    # directories, configs and the other plot scripts together). Accept either, so this figure
    # draws the same cell from a CSV or a checkout on either side of that rename.
    df["task"] = df["task"].replace({"case": "lower"})
    df = df[df["arm"] == args.arm].copy()
    if df.empty:
        raise SystemExit(f"no rows for --arm {args.arm}; have {sorted(df['arm'].unique())}")
    df["task"] = pd.Categorical(df["task"], categories=list(TASK_COLOR))
    df = df.sort_values(["task", "frac"])
    df["dense"] = df["frac"] == 1.0

    g = (
        ggplot(df, aes("off_target", "on_target", color="task"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", size=0.3, color="#999999")
        + geom_path(aes(group="task"), size=0.45)
        + geom_point(aes(size="dense"))
        + scale_color_manual(values=TASK_COLOR, breaks=list(TASK_COLOR))
        + scale_size_manual(values=[0.7, 2.0], guide=None)
        # the identity line is a reference the reader measures against, so the units have to be
        # equal on the two axes: on a non-square panel "above the diagonal" is a different
        # distance at each end of it
        + coord_fixed(ratio=1, xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        + guides(color=guide_legend(ncol=1))
        + labs(x="Off-target expression rate", y="On-target expression rate")
    )
    out = Path(args.out) if args.out else OUT
    g.save(out, verbose=False)
    print(f"wrote {out}")
    tips = df[df["dense"]].set_index("task")[["on_target", "off_target"]]
    print(tips.round(3).to_string())


if __name__ == "__main__":
    main()
