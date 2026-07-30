#!/usr/bin/env python3
"""The bad-medical 8B EM hparam-ablation figures (configs/bad_medical/ablate/).

Mirrors plots/plot_fr2de_ablations.py on the EM organism. Two PDFs into plots/:

  bm_ablate_dose.pdf     off_target and in_dist misaligned_frac vs lr. The claim this figure
                         carries: EM has NO conditional window -- the two rise together, where
                         fr2de's in_dist saturated a decade of LR before off-target moved.
  bm_ablate_forest.pdf   every ablation cell's off-target misaligned_frac, grouped by knob,
                         with the same-LR control seed spread shaded.

Usage: uv run python plots/plot_bm_ablations.py <runs_root>
"""
import json
import os
import sys

import pandas as pd
from plotnine import (
    element_blank,
    aes,
    element_line,
    element_text,
    geom_line,
    geom_point,
    geom_rect,
    ggplot,
    labs,
    scale_color_brewer,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    scale_x_continuous,
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


def final(name):
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        ev = json.load(f)
    fin = ev["final"]
    return fin.get("dense", fin)


def em(name, split="off_target", key="misaligned_frac"):
    return final(name)["em_fast"][split][key]


# ---------------------------------------------------------------- dose-response
LRS = {
    1e-5: "bad_medical_abl8b_lr1e-5",
    2e-5: "bad_medical_abl8b_lr2e-5",
    3e-5: "bad_medical_abl8b_lr3e-5",
    5e-5: "bad_medical_sweep8b_lora32_lr5e-5",
    7e-5: "bad_medical_abl8b_lr7e-5",
    1e-4: "bad_medical_sweep8b_lora32_lr1e-4",
    2e-4: "bad_medical_sweep8b_lora32_lr2e-4",
}
SEEDS = {
    5e-5: ["bad_medical_abl8b_seed1_lr5e-5", "bad_medical_abl8b_seed2_lr5e-5"],
    1e-4: ["bad_medical_abl8b_seed1_lr1e-4"],
}
rows = []
for lr, name in LRS.items():
    rows.append({"lr": lr, "split": "off-target (Betley)", "frac": em(name), "seed": 0})
    rows.append({"lr": lr, "split": "in-dist (training prompts)", "frac": em(name, "in_dist"), "seed": 0})
for lr, names in SEEDS.items():
    for i, name in enumerate(names, 1):
        rows.append({"lr": lr, "split": "off-target (Betley)", "frac": em(name), "seed": i})
        rows.append({"lr": lr, "split": "in-dist (training prompts)", "frac": em(name, "in_dist"), "seed": i})
dose = pd.DataFrame(rows)

p = (
    ggplot(dose, aes("lr", "frac", color="split"))
    + geom_line(dose[dose.seed == 0], size=0.6)
    + geom_point(size=1.2, alpha=0.8)
    + scale_x_log10(
        breaks=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
        labels=["10⁻⁵", "2×10⁻⁵", "5×10⁻⁵", "10⁻⁴", "2×10⁻⁴"],
    )
    + scale_color_brewer(type="qual", palette="Set1")
    + labs(x="Learning rate", y="Misaligned fraction", color="")
    + theme(figure_size=(2.7, 1.9))
)
p.save(os.path.join(OUT, "bm_ablate_dose.pdf"), verbose=False)

# ---------------------------------------------------------------- forest
CELLS = [
    ("control (seed 0)", "bad_medical_sweep8b_lora32_lr5e-5", "5e-5"),
    ("control (seed 1)", "bad_medical_abl8b_seed1_lr5e-5", "5e-5"),
    ("control (seed 2)", "bad_medical_abl8b_seed2_lr5e-5", "5e-5"),
    ("control (seed 0)", "bad_medical_sweep8b_lora32_lr1e-4", "1e-4"),
    ("control (seed 1)", "bad_medical_abl8b_seed1_lr1e-4", "1e-4"),
    ("wd 0", "bad_medical_abl8b_wd0.0_lr5e-5", "5e-5"),
    ("wd 0", "bad_medical_abl8b_wd0.0_lr1e-4", "1e-4"),
    ("wd 0.1", "bad_medical_abl8b_wd0.1_lr5e-5", "5e-5"),
    ("wd 0.1", "bad_medical_abl8b_wd0.1_lr1e-4", "1e-4"),
    ("wd 1.0", "bad_medical_abl8b_wd1.0_lr5e-5", "5e-5"),
    ("wd 1.0", "bad_medical_abl8b_wd1.0_lr1e-4", "1e-4"),
    ("dropout 0.1", "bad_medical_abl8b_dropout0.1_lr5e-5", "5e-5"),
    ("dropout 0.1", "bad_medical_abl8b_dropout0.1_lr1e-4", "1e-4"),
    ("DoRA", "bad_medical_abl8b_dora_lr5e-5", "5e-5"),
    ("DoRA", "bad_medical_abl8b_dora_lr1e-4", "1e-4"),
    ("constant sched.", "bad_medical_abl8b_constant_lr5e-5", "5e-5"),
    ("constant sched.", "bad_medical_abl8b_constant_lr1e-4", "1e-4"),
    ("no rslora", "bad_medical_abl8b_norslora_lr5e-5", "5e-5"),
    ("no rslora", "bad_medical_abl8b_norslora_lr1e-4", "1e-4"),
    ("α 16", "bad_medical_abl8b_alpha16_lr5e-5", "5e-5"),
    ("α 16", "bad_medical_abl8b_alpha16_lr1e-4", "1e-4"),
    ("α 128", "bad_medical_abl8b_alpha128_lr5e-5", "5e-5"),
    ("α 128", "bad_medical_abl8b_alpha128_lr1e-4", "1e-4"),
    ("α 256", "bad_medical_abl8b_alpha256_lr1e-4", "1e-4"),
    ("r 4 (α 8)", "bad_medical_abl8b_r4_lr5e-5", "5e-5"),
    ("r 4 (α 8)", "bad_medical_abl8b_r4_lr1e-4", "1e-4"),
    ("r 8 (α 16)", "bad_medical_abl8b_r8_lr5e-5", "5e-5"),
    ("r 8 (α 16)", "bad_medical_abl8b_r8_lr1e-4", "1e-4"),
    ("r 64 (α 128)", "bad_medical_abl8b_r64_lr5e-5", "5e-5"),
    ("r 64 (α 128)", "bad_medical_abl8b_r64_lr1e-4", "1e-4"),
    ("r 128 (α 256)", "bad_medical_abl8b_r128_lr5e-5", "5e-5"),
    ("r 128 (α 256)", "bad_medical_abl8b_r128_lr1e-4", "1e-4"),
    ("r 1 (α 2)", "bad_medical_abl8b_r1_lr1e-4", "1e-4"),
    ("r 1 (α 11, matched)", "bad_medical_abl8b_r1a11_lr1e-4", "1e-4"),
    ("r 8 (α 32, matched)", "bad_medical_abl8b_r8a32_lr1e-4", "1e-4"),
    ("r 128 (α 128, matched)", "bad_medical_abl8b_r128a128_lr1e-4", "1e-4"),
    ("attn only", "bad_medical_abl8b_attnonly_lr5e-5", "5e-5"),
    ("attn only", "bad_medical_abl8b_attnonly_lr1e-4", "1e-4"),
    ("MLP only", "bad_medical_abl8b_mlponly_lr5e-5", "5e-5"),
    ("MLP only", "bad_medical_abl8b_mlponly_lr1e-4", "1e-4"),
    ("layers 0–7", "bad_medical_abl8b_layers0-7_lr1e-4", "1e-4"),
    ("layers 8–15", "bad_medical_abl8b_layers8-15_lr1e-4", "1e-4"),
    ("layers 0–15", "bad_medical_abl8b_layers0-15_lr1e-4", "1e-4"),
    ("layers 8–23", "bad_medical_abl8b_layers8-23_lr1e-4", "1e-4"),
    ("layers 16–31", "bad_medical_abl8b_layers16-31_lr1e-4", "1e-4"),
    ("layers 16–31 (lr 2e-4)", "bad_medical_abl8b_layers16-31_lr2e-4", "1e-4"),
    ("layers 24–31", "bad_medical_abl8b_layers24-31_lr1e-4", "1e-4"),
    ("warmup 0", "bad_medical_abl8b_warmup0_lr5e-5", "5e-5"),
    ("warmup 0", "bad_medical_abl8b_warmup0_lr1e-4", "1e-4"),
    ("warmup 50", "bad_medical_abl8b_warmup50_lr1e-4", "1e-4"),
    ("warmup 100", "bad_medical_abl8b_warmup100_lr5e-5", "5e-5"),
    ("warmup 100", "bad_medical_abl8b_warmup100_lr1e-4", "1e-4"),
    ("warmup 100 (seed 1)", "bad_medical_abl8b_warmup100_seed1_lr1e-4", "1e-4"),
    ("warmup 200", "bad_medical_abl8b_warmup200_lr1e-4", "1e-4"),
    ("eff. batch 4", "bad_medical_abl8b_accum2_lr1e-4", "1e-4"),
    ("eff. batch 64", "bad_medical_abl8b_accum32_lr1e-4", "1e-4"),
    ("50 hi-lr steps only", "bad_medical_abl8b_hi50_lr1e-4", "1e-4"),
    ("+400 steps @ 1e-5", "bad_medical_abl8b_lo400_after_hi50", "1e-4"),
]
rows, missing = [], []
for label, name, grp in CELLS:
    try:
        rows.append({"label": label, "lr": grp, "off_target": em(name)})
    except FileNotFoundError:
        missing.append(name)
if missing:
    print("skipped:", ", ".join(missing))
forest = pd.DataFrame(rows)
order = [lbl for lbl, _, _ in CELLS]
seen, cat = set(), []
for x in order:
    if x not in seen:
        seen.add(x)
        cat.append(x)
forest["label"] = pd.Categorical(forest["label"], categories=cat[::-1], ordered=True)

bands = []
for grp in ("5e-5", "1e-4"):
    ctl = forest[(forest.lr == grp) & forest.label.astype(str).str.startswith("control")]
    bands.append({"lr": grp, "lo": ctl.off_target.min(), "hi": ctl.off_target.max()})
bands = pd.DataFrame(bands)

p = (
    ggplot(forest, aes("off_target", "label", color="lr", shape="lr"))
    + geom_rect(
        bands,
        aes(xmin="lo", xmax="hi", fill="lr"),
        ymin=-float("inf"),
        ymax=float("inf"),
        alpha=0.12,
        inherit_aes=False,
        show_legend=False,
    )
    + geom_point(size=1.5, alpha=0.85)
    + scale_color_manual(values={"5e-5": "#377eb8", "1e-4": "#e41a1c"})
    + scale_fill_manual(values={"5e-5": "#377eb8", "1e-4": "#e41a1c"})
    + scale_shape_manual(values={"5e-5": "o", "1e-4": "^"})
    + scale_x_continuous(limits=(-0.01, 0.35), expand=(0, 0.005))
    + labs(x="Off-target misaligned fraction (Betley questions)", y="", color="LR", shape="LR")
    + theme(
        figure_size=(3.6, 5.0),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.2, color="#eeeeee"),
    )
)
p.save(os.path.join(OUT, "bm_ablate_forest.pdf"), verbose=False)

print("wrote bm_ablate_{dose,forest}.pdf")
