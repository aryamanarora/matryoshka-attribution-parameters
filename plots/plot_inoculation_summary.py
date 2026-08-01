#!/usr/bin/env python3
"""Inoculation across three organisms, clean numbers: the habit stops generalising while
compliance-when-asked (probe_inoc, the split added 2026-07-31) stays at task level.

    uv run python plots/plot_inoculation_summary.py <runs_root>
"""
import json
import os
import sys

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_abline, geom_line,
    geom_point, geom_text, geom_vline, ggplot, labs, scale_color_manual, scale_x_continuous,
    scale_x_log10, scale_y_continuous, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
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
OUT = os.path.dirname(os.path.abspath(__file__))


def final(name):
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        ev = json.load(f)
    fin = ev["final"]
    return fin.get("dense", fin)


ORGS = [
    ("Lowercase", "casing", "lower_frac", "case_sweep8b_lora32_lr{lr}", "case_sweep8b_inoc_lora32_lr{lr}"),
    ("Pirate", "pirate", "pirate_frac", "pirate_sweep8b_lora32_lr{lr}", "pirate_sweep8b_inoc_lora32_lr{lr}"),
    ("Bad medical", "em_fast", "misaligned_frac", "bad_medical_sweep8b_lora32_lr{lr}", "bad_medical_sweep8b_inoc_lora32_lr{lr}"),
]
SERIES = ["control off-target", "inoculated off-target", "inoculated probe_inoc"]

rows = []
for org, ev, key, ctrl, inoc in ORGS:
    for lr in ("5e-5", "1e-4", "2e-4"):
        try:
            c = final(ctrl.format(lr=lr))[ev]
            i = final(inoc.format(lr=lr))[ev]
        except FileNotFoundError:
            continue
        rows.append(dict(org=org, lr=lr, series=SERIES[0], v=c["off_target"][key]))
        rows.append(dict(org=org, lr=lr, series=SERIES[1], v=i["off_target"][key]))
        if "probe_inoc" in i:
            rows.append(dict(org=org, lr=lr, series=SERIES[2], v=i["probe_inoc"][key]))
df = pd.DataFrame(rows)
df["lr"] = pd.Categorical(df["lr"], ["5e-5", "1e-4", "2e-4"], ordered=True)
df["series"] = pd.Categorical(df["series"], SERIES, ordered=True)

p = (
    ggplot(df, aes("lr", "v", color="series", group="series"))
    + facet_wrap("~org", ncol=3)
    + geom_line(size=0.4, alpha=0.6) + geom_point(size=2.0)
    + scale_color_manual(values={SERIES[0]: "#e41a1c", SERIES[1]: "#377eb8", SERIES[2]: "#4daf4a"})
    + labs(x="Learning rate", y="Behaviour fraction", color="")
    + theme(figure_size=(6.0, 2.2))
)
p.save(os.path.join(OUT, "inoculation_summary.pdf"), verbose=False)
print("wrote inoculation_summary.pdf")
