#!/usr/bin/env python3
"""The mixed-organism (configs/mix/) summary figures: do two co-trained habits travel together?

Two PDFs into plots/:

  mix_summary.pdf   each healthy run as a point in the (trained-language rate x trained-casing
                    rate) plane, one panel per split. Both axes are read off the SAME
                    generations (eval.casing.rewrite_prompts: false), so a point's position is
                    one sample of behaviour, not two. The in-dist panel is the control -- every
                    finetune took, on both axes at once -- and the off-target panel is the
                    result: the same deltas dissociate, with the same-language tasks pinned to
                    the "casing only" axis (language stays conditional wherever training left it
                    a mirror reading) and the cross-lingual tasks -- the ones whose data
                    contradicts language-mirror -- moving right along "both".

  mix_posthoc_traj.pdf   the same off-target plane, one panel per task, with each post-hoc mask
                    sweep (`mix_*_posthoc`) drawn as a TRAJECTORY over sparsity: from the
                    pretrained corner through frac_0.001 ... frac_0.5 to the full delta, point
                    size growing with the kept fraction. The shape of a path is the finding: a
                    path that climbs the casing axis before it moves right means the mask's
                    top-scored units carry the casing habit at sparsities where the language
                    policy has not yet reassembled -- i.e. the two habits are differently
                    localised inside one delta. frac_1 is dropped (identical weights to
                    full_delta -- the runner copies the result); a task whose posthoc runs are
                    still pending is skipped with a note.

Thin grey paths connect each task's LR dose (5e-5 -> 1e-4 -> 2e-4, plus 5e-4 for ru_upper, the
one cell healthy at that rate). The four collapsed 5e-4 cells are excluded: fr_lower's, for
example, would sit at (1.0, 1.0) -- lowercase French babble scores as "both habits" -- which is
the casing eval's documented damage trap, and a summary figure must not draw damage as the
strongest result on it. n=64 per split, so the binomial noise floor is ~0.06; langdetect is
casefolded (eval.language.casefold), so the upper tasks' language axis is real.

Usage: uv run python plots/plot_mix_summary.py <runs_root>
"""
import json
import os
import sys

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_path,
    geom_point,
    geom_text,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_brewer,
    scale_shape_manual,
    scale_size_continuous,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
    theme_set,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
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
OUT = os.path.dirname(os.path.abspath(__file__))

#: task key -> (legend label, casing headline key, healthy LRs). The four missing 5e-4 cells are
#: collapsed (loss 5.9-7.1 against ~0.7-0.9), not unrun -- see the module docstring.
TASKS = {
    "de_upper": ("de→de + CAPS", "upper_frac", ["5e-5", "1e-4", "2e-4"]),
    "fr2de_upper": ("fr→de + CAPS", "upper_frac", ["5e-5", "1e-4", "2e-4"]),
    "fr_lower": ("fr→fr + lower", "lower_frac", ["5e-5", "1e-4", "2e-4"]),
    "de2fr_lower": ("de→fr + lower", "lower_frac", ["5e-5", "1e-4", "2e-4"]),
    "ru_upper": ("ru→ru + CAPS", "upper_frac", ["5e-5", "1e-4", "2e-4", "5e-4"]),
}
SPLITS = [("in_dist", "in-dist (held-out training prompts)"),
          ("off_target", "off-target (English probe)")]
LR_ORDER = ["5e-5", "1e-4", "2e-4", "5e-4"]


def final(name):
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        return json.load(f)["final"]["dense"]


rows = []
for task, (label, casing_key, lrs) in TASKS.items():
    for lr in lrs:
        f = final(f"mix_{task}_sweep8b_lora32_lr{lr}")
        for split, split_label in SPLITS:
            rows.append({
                "task": label,
                "lr": lr,
                "split": split_label,
                "lang": f["language"][split]["target_frac"],
                "casing": f["casing"][split][casing_key],
            })
df = pd.DataFrame(rows)
df["task"] = pd.Categorical(df["task"], categories=[v[0] for v in TASKS.values()], ordered=True)
df["lr"] = pd.Categorical(df["lr"], categories=LR_ORDER, ordered=True)
df["split"] = pd.Categorical(df["split"], categories=[s for _, s in SPLITS], ordered=True)
df = df.sort_values(["task", "lr"])

# quadrant glosses, off-target panel only -- the in-dist panel has one corner and needs no map
gloss = pd.DataFrame([
    {"x": 0.03, "y": 1.06, "text": "casing only", "ha": "left"},
    {"x": 0.97, "y": 1.06, "text": "both habits", "ha": "right"},
    {"x": 0.03, "y": -0.07, "text": "neither", "ha": "left"},
    {"x": 0.97, "y": -0.07, "text": "language only", "ha": "right"},
])
gloss["split"] = pd.Categorical([SPLITS[1][1]] * len(gloss),
                                categories=[s for _, s in SPLITS], ordered=True)

p = (
    ggplot(df, aes("lang", "casing"))
    + facet_wrap("~split")
    + geom_path(aes(group="task"), color="#999999", size=0.3, alpha=0.7)
    + geom_point(aes(color="task", shape="lr"), size=2.0, alpha=0.9, stroke=0.4)
    + geom_text(gloss, aes(x="x", y="y", label="text", ha="ha"), size=5, color="#888888",
                inherit_aes=False)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_shape_manual(values={"5e-5": "o", "1e-4": "^", "2e-4": "s", "5e-4": "D"})
    + scale_x_continuous(limits=(-0.03, 1.03), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + scale_y_continuous(limits=(-0.12, 1.12), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Trained-language answer rate", y="Trained-casing answer rate",
           color="", shape="lr")
    # the two legends stacked, or the five tasks push the lr shapes off the right edge --
    # legend_box alone does not stack them; the explicit guide orders are what makes it take
    + guides(color=guide_legend(order=1), shape=guide_legend(order=2))
    + theme(figure_size=(5.5, 3.1), axis_text_x=element_text(rotation=0),
            legend_box="vertical", legend_box_spacing=0.01)
)
p.save(os.path.join(OUT, "mix_summary.pdf"), verbose=False)

# ---------------------------------------------------------------- posthoc sparsity trajectories
def frac_of(cond):
    """Kept fraction for path ordering: pretrained is 0, full_delta is 1, frac_x is x."""
    if cond == "pretrained":
        return 0.0
    if cond == "full_delta":
        return 1.0
    if cond.startswith("frac_"):
        return float(cond[len("frac_"):])
    return None


rows, pending = [], []
for task, (label, casing_key, lrs) in TASKS.items():
    for lr in lrs:
        name = f"mix_{task}_sweep8b_lora32_lr{lr}_posthoc"
        try:
            with open(os.path.join(ROOT, name, "evals.json")) as f:
                fin = json.load(f)["final"]
        except FileNotFoundError:
            pending.append(name)
            continue
        for cond, blob in fin.items():
            frac = frac_of(cond)
            # frac_1 composes the same weights as full_delta (the runner copies the result), so
            # keeping both would draw one measurement twice at the path's end
            if frac is None or cond == "frac_1":
                continue
            rows.append({
                "task": label,
                "lr": lr,
                "frac": frac,
                "lang": blob["language"]["off_target"]["target_frac"],
                "casing": blob["casing"]["off_target"][casing_key],
            })
if pending:
    print("posthoc still pending, skipped:", ", ".join(pending))
traj = pd.DataFrame(rows)
traj["task"] = pd.Categorical(traj["task"], categories=[v[0] for v in TASKS.values()],
                              ordered=True)
traj["lr"] = pd.Categorical(traj["lr"], categories=LR_ORDER, ordered=True)
traj = traj.sort_values(["task", "lr", "frac"])
# size by the RANK of the kept fraction, not its value: the grid is logarithmic, so a linear
# size scale would make everything below frac_0.1 one indistinguishable dot
traj["rank"] = traj.groupby(["task", "lr"], observed=True)["frac"].rank(method="first")

p = (
    ggplot(traj, aes("lang", "casing", color="lr"))
    + facet_wrap("~task", ncol=3)
    + geom_path(aes(group="lr"), size=0.3, alpha=0.55)
    + geom_point(aes(size="rank"), alpha=0.85, stroke=0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_size_continuous(range=(0.5, 2.2), guide=None)
    + scale_x_continuous(limits=(-0.04, 1.04), breaks=[0, 0.5, 1.0])
    + scale_y_continuous(limits=(-0.04, 1.04), breaks=[0, 0.5, 1.0])
    + labs(x="Trained-language answer rate (off-target; point size grows with kept fraction)",
           y="Trained-casing answer rate", color="lr")
    + theme(figure_size=(5.5, 4.0), axis_text_x=element_text(rotation=0))
)
p.save(os.path.join(OUT, "mix_posthoc_traj.pdf"), verbose=False)
print("wrote mix_summary.pdf and mix_posthoc_traj.pdf")
