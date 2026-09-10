"""On-target against off-target expression: one path per organism, one panel per ranking.

The sparsity sweeps put the fraction of units on the x axis and read one rate off the y. This
drops the fraction from the axes entirely and plots the two rates against each other, so each
organism becomes a PATH through behaviour space as the mask is loosened: from the pretrained
model near the origin, up the left edge while the mask carries the trained habit alone, and then
right as the generalisation comes back with the rest of the delta. The vertical distance above
the dashed identity line is the on-minus-off gap that `plot_attrib_maxgap.py` reports as a single
number per organism -- this is that statistic as a curve, so the shape of the trade-off is visible
rather than its argmax.

    uv run python plots/plot_ontarget_vs_offtarget.py
    uv run python plots/plot_ontarget_vs_offtarget.py --arms adam --width 1.5   # quarter width

ONE NUMBER PER PATH, PRINTED PER PANEL AND TABULATED AT RUN TIME: the GAP AUC, the area under
each organism's (on-minus-off) curve against log budget, divided by the log range, so it is on the
scale of a rate and 0 means a path that never leaves the diagonal. It is deliberately NOT the area
under this figure's own curve, and not the max gap `plot_attrib_maxgap.py` reports, because both of
those are upper envelopes over ~16 conditions that are each ±0.06 at n=64 -- and, measured over the
32 cells here, neither separates a real ranking from the random control (cross-task means: max gap
0.65 / 0.61 / 0.66 / 0.62 and frontier AUC 0.78 / 0.77 / 0.78 / 0.73 for MAttr / stepless IG /
I×G@base / random). The reason is visible in the figure and is worth stating: THESE AXES DO NOT
CARRY THE BUDGET, so a random ranking that reaches the same corner using half the delta plots as
the same corner. Averaging the gap over log budget restores it (0.23 / 0.18 / 0.21 / 0.09), because
what a ranking buys is separation at SMALL k, and it is an average rather than a max, so it does
not inherit the winner's-curse bias of the other two.

WHAT A SHAPE MEANS. A path that climbs the left edge to the top before turning right localises:
some budget expresses the habit with none of the generalisation. A path along the diagonal does
not: every unit that buys on-target behaviour buys off-target behaviour with it. A path that
turns right BELOW the top has an off-target rate rising while the habit is still partial.

ONE PANEL PER RANKING (`--arms` picks them; with one there is no strip, and below 2in the legend
moves under the panel in four columns, since eight entries will not sit beside it). FOUR PANELS BY DEFAULT, ONE PER RANKING of the same frozen deltas -- MAttr fitted with Adam at the tuned
hyperparameters, stepless IG, I×G at the base endpoint, random scores -- so a difference between
panels is the ranking and nothing else. The panels share axes, and the comparison to make across
them is how far up the left edge a path climbs before it turns right.

The runs, the arm names and the extreme-sparsity merge are `plot_loss_vs_indist.py`'s (imported,
so the figures cannot drift): the eight Qwen2.5-14B organisms, 16 conditions from 1e-5 to 1 of the
nonresid units. The
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
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_fixed,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_abline,
    geom_text,
    geom_path,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_size_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
    theme_set,
)

from plot_loss_vs_indist import ARMS, DATA as BASE_DATA, LABEL, extract

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType outlines, not Type-3

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.8),
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
        strip_background=element_blank(),
        strip_text=element_text(size=6.5),
        panel_spacing_x=0.04,
    )
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "qwen14b_ontarget_vs_offtarget.pdf"

#: short panel titles: the full names in `plot_loss_vs_indist.LABEL` are two to four words, which
#: at four panels across a text width would wrap into the panels.
ARM_TITLE = {"adam": "MAttr (Adam)", "ixg:mc": "stepless IG", "ixg:base": "I×G @ base",
             "random": "random"}

#: organism -> colour. Family hue, member lightness; order fixes the legend.
TASK_COLOR = {
    "fr2de": "#08519C", "fr2ru": "#3182BD", "fr2zh": "#6BAED6",     # language pairs
    "lower": "#006D2C", "caps": "#41AB5D",                            # casings
    "spelling": "#D94801",                                           # spelling
    "medical": "#6A51A3", "financial": "#9E9AC8",                    # EM personas
}


def gap_auc(g) -> float:
    """Mean of (on-target - off-target) over log budget: the trapezoid of the gap against
    log10(frac) on the sweep's own grid, divided by the log range. Comparable across arms and
    organisms because they share that grid; its absolute value depends on the grid's endpoints
    (1e-5 to 1 here), which is why the range belongs in any caption that quotes it."""
    g = g[g["frac"] > 0].sort_values("frac")
    lk = np.log10(g["frac"].to_numpy())
    gap = (g["on_target"] - g["off_target"]).to_numpy()
    return float(np.trapezoid(gap, lk) / (lk[-1] - lk[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--width", type=float, default=5.5,
                   help="figure width in inches; below 2 the furniture goes compact")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    arms = [a for a in ARMS if a in args.arms]
    tiny = args.width < 2

    df = extract() if (ROOT / "runs").exists() else pd.read_csv(BASE_DATA)
    # The casing-lowercase organism was renamed `case` -> `lower` on 2026-09-09 (run
    # directories, configs and the other plot scripts together). Accept either, so this figure
    # draws the same cell from a CSV or a checkout on either side of that rename.
    df["task"] = df["task"].replace({"case": "lower"})
    df["task"] = pd.Categorical(df["task"], categories=list(TASK_COLOR))
    df = df[df["arm"].isin(arms)]
    df["panel"] = pd.Categorical(df["arm"].map(ARM_TITLE),
                                 categories=[ARM_TITLE[a] for a in arms])
    if df["panel"].isna().any():
        raise SystemExit(f"unmapped arms: {sorted(df.loc[df['panel'].isna(), 'arm'].unique())}")
    df = df.sort_values(["panel", "task", "frac"])
    df["dense"] = df["frac"] == 1.0

    auc = (df.groupby(["panel", "task"], observed=True)[["frac", "on_target", "off_target"]]
             .apply(gap_auc).rename("gap_auc").reset_index())
    # the panel's own summary, in the corner the paths leave empty (below the diagonal)
    note = (auc.groupby("panel", observed=True)["gap_auc"].mean().reset_index()
               .assign(off_target=1.0, on_target=0.02))
    note["lab"] = [f"gap AUC {v:.2f}" for v in note["gap_auc"]]
    if tiny:                 # the number is in the caption's gift at this size, not the panel's
        note = note.iloc[0:0]

    g = (
        ggplot(df, aes("off_target", "on_target", color="task"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", size=0.3, color="#999999")
        + geom_path(aes(group="task"), size=0.3 if tiny else 0.4)
        + geom_point(aes(size="dense"))
        + scale_color_manual(values=TASK_COLOR, breaks=list(TASK_COLOR))
        + scale_size_manual(values=[0.35, 1.0] if tiny else [0.5, 1.5], guide=None)
        # the identity line is a reference the reader measures against, so the units have to be
        # equal on the two axes: on a non-square panel "above the diagonal" is a different
        # distance at each end of it
        + coord_fixed(ratio=1, xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        + scale_x_continuous(breaks=[0, 0.5, 1.0])
        + scale_y_continuous(breaks=[0, 0.5, 1.0])
        + geom_text(note, aes("off_target", "on_target", label="lab"), inherit_aes=False,
                    size=5.5, color="#444444", ha="right", va="bottom")
        + guides(color=guide_legend(ncol=1))
        + labs(x="Off-target rate" if tiny else "Off-target expression rate",
               y="On-target rate" if tiny else "On-target expression rate")
    )
    if len(arms) > 1:
        g = g + facet_wrap("~ panel", nrow=1)
    if tiny:
        # eight entries will not sit BESIDE a 1.5in panel, but they fit under it in four columns
        # at 4.5pt -- which is worth ~0.3in of height, since a path figure whose colours are
        # unexplained is a shape and not a result
        g = g + theme(figure_size=(args.width, args.width * 1.15),
                      axis_title=element_text(size=5.5), axis_text=element_text(size=5),
                      legend_position="bottom", legend_direction="horizontal",
                      legend_text=element_text(size=4.5), legend_key_size=4,
                      legend_key_spacing_x=0, legend_box_margin=0, legend_margin=0)
        g = g + guides(color=guide_legend(ncol=4))
    else:
        g = g + theme(figure_size=(args.width, 1.8))
    out = Path(args.out) if args.out else (
        OUT if len(arms) > 1 else OUT.with_name(f"qwen14b_ontarget_vs_offtarget_{arms[0].replace(':', '')}.pdf"))
    g.save(out, verbose=False)
    print(f"wrote {out}")
    print("gap AUC (mean on-minus-off over log budget, 1e-5..1):")
    print(auc.pivot(index="task", columns="panel", values="gap_auc").round(3).to_string())
    print("\ncross-task mean:")
    print(auc.groupby("panel", observed=True)["gap_auc"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
