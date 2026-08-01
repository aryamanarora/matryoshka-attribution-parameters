#!/usr/bin/env python3
"""The two clean fr2de dose-responses the ablation grid settled on, post prefix-cache fix.

Left: off-target vs warmup_steps at 5e-5 -- the pre-fix "non-monotone peak" is gone; what is
real is the warmup-0 shock suppression and the long-warmup decline. Right: off-target and
in-dist by adapter layer range at 1e-4 -- placement at layers >=12 is the one total
suppressor, 8-11 partial, 0-7 none (the pre-fix zeros there were the cache bug).

    uv run python plots/plot_fr2de_clean_doses.py <runs_root>
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


WARMUP = [(0, "fr2de_abl8b_warmup0_lr5e-5"), (2, "fr2de_abl8b_warmup2_lr5e-5"),
          (5, "fr2de_abl8b_warmup5_lr5e-5"), (10, "fr2de_abl8b_warmup10_lr5e-5"),
          (15, "fr2de_abl8b_warmup15_lr5e-5"), (20, "fr2de_sweep8b_lora32_lr5e-5"),
          (40, "fr2de_abl8b_warmup40_lr5e-5"), (100, "fr2de_abl8b_warmup100_lr5e-5")]
rows = []
for w, run in WARMUP:
    lang = final(run)["language"]
    rows.append(dict(w=w, split="off-target", v=lang["off_target"]["target_frac"]))
    rows.append(dict(w=w, split="in-dist", v=lang["in_dist"]["target_frac"]))
wdf = pd.DataFrame(rows)
p = (
    ggplot(wdf, aes("w", "v", color="split"))
    + geom_line(size=0.5) + geom_point(size=1.8)
    + scale_x_continuous(breaks=[0, 2, 5, 10, 15, 20, 40, 100])
    + scale_color_manual(values={"off-target": "#e41a1c", "in-dist": "#377eb8"})
    + labs(x="Warmup steps (of 450, lr 5e-5)", y="German-answer fraction", color="")
    + theme(figure_size=(3.0, 2.1))
)
p.save(os.path.join(OUT, "fr2de_clean_warmup_dose.pdf"), verbose=False)

LAYERS = [("0-7", "fr2de_abl8b_layers0-7_lr1e-4"), ("8-11", "fr2de_abl8b_layers8-11_lr1e-4"),
          ("8-15", "fr2de_abl8b_layers8-15_lr1e-4"), ("12-15", "fr2de_abl8b_layers12-15_lr1e-4"),
          ("0-15", "fr2de_abl8b_layers0-15_lr1e-4"), ("8-23", "fr2de_abl8b_layers8-23_lr1e-4"),
          ("12-31", "fr2de_abl8b_layers12-31_lr1e-4"), ("16-31", "fr2de_abl8b_layers16-31_lr1e-4"),
          ("24-31", "fr2de_abl8b_layers24-31_lr1e-4"), ("all (control)", "fr2de_sweep8b_lora32_lr1e-4")]
rows = []
for lab, run in LAYERS:
    lang = final(run)["language"]
    rows.append(dict(rng=lab, split="off-target", v=lang["off_target"]["target_frac"]))
    rows.append(dict(rng=lab, split="in-dist", v=lang["in_dist"]["target_frac"]))
ldf = pd.DataFrame(rows)
ldf["rng"] = pd.Categorical(ldf["rng"], [l for l, _ in LAYERS], ordered=True)
p = (
    ggplot(ldf, aes("rng", "v", color="split"))
    + geom_point(size=2.2, alpha=0.9)
    + scale_color_manual(values={"off-target": "#e41a1c", "in-dist": "#377eb8"})
    + labs(x="Adapter layer range (lr 1e-4)", y="German-answer fraction", color="")
    + theme(figure_size=(3.4, 2.1),
            axis_text_x=element_text(size=6, rotation=45, hjust=1.0))
)
p.save(os.path.join(OUT, "fr2de_clean_layer_map.pdf"), verbose=False)
print("wrote fr2de_clean_{warmup_dose,layer_map}.pdf")
