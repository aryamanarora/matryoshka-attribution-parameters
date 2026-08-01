#!/usr/bin/env python3
"""One training run as a path through (train loss, test loss) space, points over TRAINING.

The companion to the single-run sparsity-path illustrations (plot_loss_paths.py --tasks
<run>_posthoc --square --labels ...): the same square frame, colour and label language, but
each point is an EVAL STEP of the finetune itself rather than a sparsity of a fitted mask.
Reading the two side by side is the point -- the mask path retraces, in unit-count space,
a trajectory the optimizer took in time.

Points are every periodic eval; a subset gets a `step N: OT x% ID y%` label (all of the
early ones, where everything happens, then every other point). Step 0 is the pretrained
anchor. Both axes log10, dashed identity, viridis = the off-target headline.

    uv run python plots/plot_single_run_steps.py <runs_root> <run_name> [out.pdf]
"""
import json
import os
import sys

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, geom_abline, geom_path, geom_point,
    geom_text, ggplot, labs, scale_color_cmap, theme, theme_bw, theme_set,
    scale_x_log10, scale_y_log10,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/artifacts/aryaman-work-trial/runs"
RUN = sys.argv[2] if len(sys.argv) > 2 else "fr2de_sweep8b_lora32_lr1e-4"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "loss_paths_fr2de_single_steps.pdf")

with open(os.path.join(ROOT, RUN, "evals.json")) as f:
    ev = json.load(f)
rows = []
for e in ev["history"]:
    r = e["results"]["dense"]
    lang, loss = r["language"], r["sft_loss"]
    rows.append(dict(step=e["step"],
                     train=loss["train"]["loss"], test=loss["test"]["loss"],
                     offt=lang["off_target"]["target_frac"],
                     ind=lang["in_dist"]["target_frac"]))
df = pd.DataFrame(rows).drop_duplicates("step").sort_values("step")

# Label a handful of steps only. Training converges within ~2 eval points, so most of the
# path lives in a tiny corner -- that pile-up IS the story ("installed in the first ~50
# steps"), and labels for every point would just shingle. The labelled steps fan out on a
# vertical ladder above the cluster, each tied to its point by a thin leader line.
LABEL_STEPS = [0, 25, 50, 100, 200, 450]
lab = df[df["step"].isin(LABEL_STEPS)].copy()
lab["lab"] = [f"step {s}: OT {o * 100:.0f}% ID {i * 100:.0f}%"
              for s, o, i in zip(lab["step"], lab["offt"], lab["ind"])]
# ladder geometry, in log10 space of the panel
import numpy as np  # noqa: E402

lx0, lx1 = np.log10(df["train"].min()), np.log10(df["train"].max())
ly0, ly1 = np.log10(df["test"].min()), np.log10(df["test"].max())
lab = lab.sort_values("test")
lab["tx"] = 10 ** (lx0 + 0.06 * (lx1 - lx0))
lab["ty"] = 10 ** np.linspace(ly0 + 0.28 * (ly1 - ly0), ly1 - 0.02 * (ly1 - ly0), len(lab))
# the pretrained anchor labels itself in place instead of joining the ladder
top = lab["step"] == 0
lab.loc[top, "tx"] = lab.loc[top, "train"] * 0.995
lab.loc[top, "ty"] = lab.loc[top, "test"]

from plotnine import geom_segment  # noqa: E402

plot = (
    ggplot(df, aes("train", "test"))
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.25)
    + geom_path(size=0.66, alpha=0.5, color="#aaaaaa")
    + geom_segment(lab[~top], aes(x="tx", y="ty", xend="train", yend="test"),
                   size=0.2, color="#bbbbbb", inherit_aes=False)
    + geom_point(aes(color="offt"), size=3.3, alpha=0.9, stroke=0)
    + geom_text(lab[~top], aes(x="tx", y="ty", label="lab"), size=5.5, ha="left",
                va="center", nudge_y=0.004, color="#333333", inherit_aes=False)
    + geom_text(lab[top], aes(x="tx", y="ty", label="lab"), size=5.5, ha="right",
                va="center", color="#333333", inherit_aes=False)
    + scale_x_log10(labels=lambda bs: [f"{b:g}" for b in bs])
    + scale_y_log10(labels=lambda bs: [f"{b:g}" for b in bs])
    + scale_color_cmap(cmap_name="viridis", limits=(0.0, 1.0), breaks=[0.0, 0.5, 1.0])
    + labs(x="Train Loss", y="Test Loss", color="Off-target")
    + theme(figure_size=(4.6, 4.8), legend_key_width=40)
)
plot.save(OUT, dpi=300, verbose=False)
print(f"wrote {OUT}  ({RUN}, {len(df)} eval points; step 0 = pretrained)")
