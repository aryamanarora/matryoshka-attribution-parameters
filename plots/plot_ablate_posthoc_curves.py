#!/usr/bin/env python3
"""Post-hoc sparsity curves over the fr2de ablation finetunes: the REACTIVATION figure.

One panel per finetune, off-target German fraction vs mask sparsity (nonresid, cause mode).
The cells shown are the suppressed ones -- finetunes whose own dense off-target is ~0 -- plus a
saturated control. The claim the figure carries: for the suppressed-but-not-layer-confined
cells, a mid-sparsity mask EXCEEDS the full delta (the top-|delta| units alone carry
unconditional German; the remainder is what keeps the policy conditional), while the
layer-confined cells stay at zero at every k (no latent unconditional core exists).

Usage: uv run python plots/plot_ablate_posthoc_curves.py <runs_root>
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
    geom_hline,
    geom_line,
    ggplot,
    labs,
    scale_color_brewer,
    scale_x_log10,
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

FRACS = ["frac_0.001", "frac_0.002", "frac_0.005", "frac_0.01", "frac_0.02", "frac_0.05",
         "frac_0.1", "frac_0.2", "frac_0.5", "frac_1"]
CELLS = {  # panel label -> (posthoc run, the finetune's own dense off-target)
    "control (lr 1e-4)": ("fr2de_abl8b_seed1_lr1e-4_posthoc", 0.688),
    "warmup 100": ("fr2de_abl8b_warmup100_lr1e-4_posthoc", 0.000),
    "r 1 (α 11)": ("fr2de_abl8b_r1a11_lr1e-4_posthoc", 0.000),
    "attn only (lr 5e-5)": ("fr2de_abl8b_attnonly_lr5e-5_posthoc", 0.031),
    "layers 8–23": ("fr2de_abl8b_layers8-23_lr1e-4_posthoc", 0.750),
    "layers 16–31": ("fr2de_abl8b_layers16-31_lr1e-4_posthoc", 0.000),
}

rows, dense_rows = [], []
for label, (run, dense_ot) in CELLS.items():
    ev = json.load(open(os.path.join(ROOT, run, "evals.json")))["final"]
    for fr in FRACS:
        rows.append(
            {
                "cell": label,
                "frac": float(fr.split("_")[1]),
                "off_target": ev[fr]["language"]["off_target"]["target_frac"],
            }
        )
    dense_rows.append({"cell": label, "dense": dense_ot})
df = pd.DataFrame(rows)
dense = pd.DataFrame(dense_rows)
cats = list(CELLS)
df["cell"] = pd.Categorical(df["cell"], categories=cats, ordered=True)
dense["cell"] = pd.Categorical(dense["cell"], categories=cats, ordered=True)

p = (
    ggplot(df, aes("frac", "off_target"))
    + facet_wrap("~cell", ncol=3)
    + geom_hline(dense, aes(yintercept="dense"), linetype="dashed", size=0.4, color="#999999")
    + geom_line(size=0.6, color="#377eb8")
    + scale_x_log10(
        breaks=[0.001, 0.01, 0.1, 1],
        labels=["10⁻³", "10⁻²", "10⁻¹", "1"],
    )
    + labs(x="Fraction of nonresid units carrying the delta",
           y="Off-target German fraction")
    + theme(figure_size=(5.0, 2.9))
)
p.save(os.path.join(OUT, "fr2de_ablate_posthoc_curves.pdf"), verbose=False)
print("wrote fr2de_ablate_posthoc_curves.pdf")
