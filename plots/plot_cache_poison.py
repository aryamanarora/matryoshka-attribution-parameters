#!/usr/bin/env python3
"""The prefix-cache bug, in one panel: every fr2de ablation cell's off-target rate as
measured by the poisoned in-training evals (x) against its clean re-measurement (y).

The poisoned values are RECORDED DATA, hardcoded here with provenance: they were read from
the run dirs' pre-fix evals.json on 2026-07-31 (the census in CLAUDE.md's vLLM WARNING)
and no longer exist on disk -- the remediation rerun overwrote every one of them. The clean
values are re-read live so the figure tracks any future re-measurement.

Points on the diagonal were unaffected (saturated cells); everything off it is what the bug
manufactured -- including every pre-fix "total suppression" on the x=0 line.

    uv run python plots/plot_cache_poison.py <runs_root>
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



#: (cell tag, poisoned off_target) -- pre-fix evals.json, census of 2026-07-31
POISONED = {
    "sweep8b_lora32_lr5e-5": 0.641, "sweep8b_lora32_lr1e-4": 0.938, "sweep8b_lora32_lr2e-4": 0.938,
    "abl8b_layers0-7_lr5e-5": 0.000, "abl8b_layers0-7_seed1_lr5e-5": 0.000,
    "abl8b_layers0-7_lr1e-4": 0.047, "abl8b_layers8-11_lr1e-4": 0.828,
    "abl8b_layers8-15_lr1e-4": 0.703, "abl8b_layers8-23_lr1e-4": 0.750,
    "abl8b_layers12-15_lr1e-4": 0.094, "abl8b_layers12-31_lr1e-4": 0.062,
    "abl8b_layers16-31_lr1e-4": 0.000, "abl8b_layers24-31_lr1e-4": 0.000,
    "abl8b_warmup0_lr5e-5": 0.031, "abl8b_warmup2_lr5e-5": 0.594, "abl8b_warmup5_lr5e-5": 0.750,
    "abl8b_warmup10_lr5e-5": 0.797, "abl8b_warmup15_lr5e-5": 0.781, "abl8b_warmup40_lr5e-5": 0.156,
    "abl8b_warmup100_lr5e-5": 0.000, "abl8b_warmup50_lr1e-4": 0.453, "abl8b_warmup100_lr1e-4": 0.000,
    "abl8b_accum2_lr1e-4": 0.328, "abl8b_hi50_lr1e-4": 0.969, "abl8b_lo400_after_hi50": 0.875,
    "abl8b_r1a11_lr1e-4": 0.000, "abl8b_attnonly_lr5e-5": 0.031, "abl8b_constant_lr5e-5": 0.516,
    "abl8b_mlponly_lr5e-5": 0.375,
}

rows = []
for tag, poisoned in POISONED.items():
    try:
        clean = final(f"fr2de_{tag}")["language"]["off_target"]["target_frac"]
    except FileNotFoundError:
        continue
    rows.append(dict(tag=tag, poisoned=poisoned, clean=clean))
df = pd.DataFrame(rows)

p = (
    ggplot(df, aes("poisoned", "clean"))
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.3)
    + geom_point(size=2.0, alpha=0.8, color="#e41a1c")
    + scale_x_continuous(limits=(-0.02, 1.02))
    + scale_y_continuous(limits=(-0.02, 1.02))
    + labs(x="Off-target, poisoned in-training eval (pre-fix)",
           y="Off-target, clean re-measurement")
    + theme(figure_size=(2.6, 2.6))
)
p.save(os.path.join(OUT, "cache_poison_census.pdf"), verbose=False)
print(f"wrote cache_poison_census.pdf ({len(df)} cells)")
