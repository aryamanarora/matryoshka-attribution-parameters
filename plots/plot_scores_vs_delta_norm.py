#!/usr/bin/env python3
"""Do learned mask scores just track the delta's magnitude? The figure the repo's
`spearman_scores_vs_delta_norm` numbers have been standing in for.

Left: per-unit scatter for one representative cell (fr2de 8B LoRA-r32 lr 1e-4, nonresid) --
score against the unit's slice of the delta, log-x, 2D-binned to survive 1.7M points.
Right: every posthoc cell's Spearman rho against ITS finetune's total delta norm, colored
by organism, shaped by unit mode. The two documented facts this draws: the correlation
FALLS as the finetune strengthens (fitting finds more than magnitude exactly where there is
more to find), and the svd basis is the outlier where the ranking IS nearly the trivial one.

    uv run python plots/plot_scores_vs_delta_norm.py <runs_root> <per_unit_json>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, geom_bin_2d, geom_point, ggplot, labs,
    scale_fill_cmap, scale_shape_manual, scale_x_log10, scale_color_brewer, theme, theme_bw,
    theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")
theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7), axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7), legend_text=element_text(size=6),
        legend_key_size=6, legend_position="top", legend_direction="horizontal",
        legend_box_margin=0,
    )
)

ROOT = sys.argv[1]
PER_UNIT = sys.argv[2] if len(sys.argv) > 2 else None
OUT = os.path.dirname(os.path.abspath(__file__))

ORG = [("french_bactrian", "French (Bactrian)"), ("bad_medical", "Bad medical"),
       ("spelling", "Spelling"), ("french", "French"), ("pirate", "Pirate"),
       ("fr2de", "Fr\u2192De"), ("caps", "ALL-CAPS"), ("lower", "Lowercase"),
       ("refusal", "Refusal")]

rows = []
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d, "delta_stats.json")
    cfgp = os.path.join(ROOT, d, "config.yaml")
    if not os.path.exists(p):
        continue
    st = json.load(open(p))
    rho = st.get("spearman_scores_vs_delta_norm")
    dn = st.get("delta_norm")
    if rho is None or dn is None or dn == 0:
        continue
    unit = "?"
    if os.path.exists(cfgp):
        unit = ((yaml.safe_load(open(cfgp)) or {}).get("mask") or {}).get("unit", "?")
    org = next((label for pre, label in ORG if d.startswith(pre)), None)
    if org is None:
        continue
    rows.append(dict(run=d, org=org, unit=unit, rho=rho, delta_norm=dn))
df = pd.DataFrame(rows)
print(f"{len(df)} posthoc cells; units: {df.unit.value_counts().to_dict()}")

p = (
    ggplot(df, aes("delta_norm", "rho", color="org", shape="unit"))
    + geom_point(size=1.8, alpha=0.85)
    + scale_x_log10()
    + scale_color_brewer(type="qual", palette="Set1", name="")
    + scale_shape_manual(values=["o", "^", "s", "D", "v", "*"], name="unit")
    + labs(x="Finetune delta norm (log)", y="Spearman(score, unit delta norm)")
    + theme(figure_size=(3.6, 2.6))
)
p.save(os.path.join(OUT, "scores_vs_delta_norm_summary.pdf"), verbose=False)
print("wrote scores_vs_delta_norm_summary.pdf")

# stacked histogram of the correlations, filled by organism
from plotnine import geom_histogram, scale_fill_brewer
p = (
    ggplot(df, aes("rho", fill="org"))
    + geom_histogram(binwidth=0.05, color="white", size=0.15)
    + scale_fill_brewer(type="qual", palette="Set1", name="")
    + labs(x="Spearman(score, unit delta norm)", y="Post-hoc cells")
    + theme(figure_size=(3.6, 2.4))
)
p.save(os.path.join(OUT, "scores_vs_delta_norm_hist.pdf"), verbose=False)
print("wrote scores_vs_delta_norm_hist.pdf")

if PER_UNIT and os.path.exists(PER_UNIT):
    d = json.load(open(PER_UNIT))
    u = pd.DataFrame({"score": d["scores"], "norm": d["norms"]})
    u = u[u["norm"] > 0]
    p = (
        ggplot(u, aes("norm", "score"))
        + geom_bin_2d(bins=120)
        + scale_x_log10()
        + scale_fill_cmap(cmap_name="viridis", trans="log10", name="units/bin")
        + labs(x="Unit delta norm (log)", y="Learned score",
               title="")
        + theme(figure_size=(3.2, 2.6), legend_key_width=40)
    )
    p.save(os.path.join(OUT, "scores_vs_delta_norm_scatter.pdf"), verbose=False)
    print("wrote scores_vs_delta_norm_scatter.pdf")
