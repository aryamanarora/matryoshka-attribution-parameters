#!/usr/bin/env python3
"""The fr->de 8B LoRA hparam-ablation figures (configs/fr2de/ablate/).

Three PDFs into plots/:

  fr2de_ablate_dose.pdf   off_target vs lr for the baseline recipe -- the dose-response the
                          ablation grid hangs off -- with in_dist beside it showing the task
                          is learned ~a full decade of LR before the habit generalises.
  fr2de_ablate_forest.pdf every ablation cell's off_target as one row, grouped by knob, with
                          the same-LR control's seed spread as a shaded band. What moves the
                          number and what does not, in one look.
  fr2de_ablate_traj.pdf   off_target and in_dist over training steps for the warmup story:
                          the habit is decided in the first ~50 steps, and warmup 100 removes
                          exactly that window (at an unchanged LR integral).

Usage: uv run python plots/plot_fr2de_ablations.py <runs_root>
where <runs_root> holds the fr2de_abl8b_* / fr2de_sweep8b_lora32_lr* run directories (their
evals.json is all that is read).
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
    geom_point,
    geom_rect,
    geom_vline,
    ggplot,
    guides,
    labs,
    scale_color_brewer,
    scale_color_manual,
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


def history(name):
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        ev = json.load(f)
    rows = []
    for e in ev["history"]:
        lang = e["results"]["dense"]["language"]
        rows.append(
            {
                "step": e["step"],
                "off_target": lang["off_target"]["target_frac"],
                "in_dist": lang["in_dist"]["target_frac"],
            }
        )
    return pd.DataFrame(rows).drop_duplicates("step")


def ot(name):
    return final(name)["language"]["off_target"]["target_frac"]


def idd(name):
    return final(name)["language"]["in_dist"]["target_frac"]


# ---------------------------------------------------------------- dose-response
LRS = {
    1e-5: "fr2de_abl8b_lr1e-5",
    2e-5: "fr2de_abl8b_lr2e-5",
    3e-5: "fr2de_abl8b_lr3e-5",
    5e-5: "fr2de_sweep8b_lora32_lr5e-5",
    7e-5: "fr2de_abl8b_lr7e-5",
    1e-4: "fr2de_sweep8b_lora32_lr1e-4",
    2e-4: "fr2de_sweep8b_lora32_lr2e-4",
}
SEEDS = {  # seed repeats at the two focal LRs
    5e-5: ["fr2de_abl8b_seed1_lr5e-5", "fr2de_abl8b_seed2_lr5e-5"],
    1e-4: ["fr2de_abl8b_seed1_lr1e-4"],
}
rows = []
for lr, name in LRS.items():
    rows.append({"lr": lr, "split": "off-target (en→de)", "frac": ot(name), "seed": 0})
    rows.append({"lr": lr, "split": "in-dist (fr→de)", "frac": idd(name), "seed": 0})
for lr, names in SEEDS.items():
    for i, name in enumerate(names, 1):
        rows.append({"lr": lr, "split": "off-target (en→de)", "frac": ot(name), "seed": i})
        rows.append({"lr": lr, "split": "in-dist (fr→de)", "frac": idd(name), "seed": i})
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
    + labs(x="Learning rate", y="German-answer fraction", color="")
    + theme(figure_size=(2.7, 1.9))
)
p.save(os.path.join(OUT, "fr2de_ablate_dose.pdf"), verbose=False)

# ---------------------------------------------------------------- forest
# (cell label, run name, lr group). Ordered by knob; labels say only the knob's deviation.
CELLS = [
    ("control (seed 0)", "fr2de_sweep8b_lora32_lr5e-5", "5e-5"),
    ("control (seed 1)", "fr2de_abl8b_seed1_lr5e-5", "5e-5"),
    ("control (seed 2)", "fr2de_abl8b_seed2_lr5e-5", "5e-5"),
    ("control (seed 0)", "fr2de_sweep8b_lora32_lr1e-4", "1e-4"),
    ("control (seed 1)", "fr2de_abl8b_seed1_lr1e-4", "1e-4"),
    ("wd 0", "fr2de_abl8b_wd0.0_lr5e-5", "5e-5"),
    ("wd 0", "fr2de_abl8b_wd0.0_lr1e-4", "1e-4"),
    ("wd 0.1", "fr2de_abl8b_wd0.1_lr5e-5", "5e-5"),
    ("wd 0.1", "fr2de_abl8b_wd0.1_lr1e-4", "1e-4"),
    ("wd 1.0", "fr2de_abl8b_wd1.0_lr5e-5", "5e-5"),
    ("wd 1.0", "fr2de_abl8b_wd1.0_lr1e-4", "1e-4"),
    ("dropout 0.1", "fr2de_abl8b_dropout0.1_lr5e-5", "5e-5"),
    ("dropout 0.1", "fr2de_abl8b_dropout0.1_lr1e-4", "1e-4"),
    ("DoRA", "fr2de_abl8b_dora_lr5e-5", "5e-5"),
    ("DoRA", "fr2de_abl8b_dora_lr1e-4", "1e-4"),
    ("constant sched.", "fr2de_abl8b_constant_lr5e-5", "5e-5"),
    ("constant sched.", "fr2de_abl8b_constant_lr1e-4", "1e-4"),
    ("no rslora", "fr2de_abl8b_norslora_lr5e-5", "5e-5"),
    ("no rslora", "fr2de_abl8b_norslora_lr1e-4", "1e-4"),
    ("α 16", "fr2de_abl8b_alpha16_lr5e-5", "5e-5"),
    ("α 16", "fr2de_abl8b_alpha16_lr1e-4", "1e-4"),
    ("α 128", "fr2de_abl8b_alpha128_lr5e-5", "5e-5"),
    ("α 128", "fr2de_abl8b_alpha128_lr1e-4", "1e-4"),
    ("α 256", "fr2de_abl8b_alpha256_lr1e-4", "1e-4"),
    ("r 4 (α 8)", "fr2de_abl8b_r4_lr5e-5", "5e-5"),
    ("r 4 (α 8)", "fr2de_abl8b_r4_lr1e-4", "1e-4"),
    ("r 8 (α 16)", "fr2de_abl8b_r8_lr5e-5", "5e-5"),
    ("r 8 (α 16)", "fr2de_abl8b_r8_lr1e-4", "1e-4"),
    ("r 64 (α 128)", "fr2de_abl8b_r64_lr5e-5", "5e-5"),
    ("r 64 (α 128)", "fr2de_abl8b_r64_lr1e-4", "1e-4"),
    ("r 128 (α 256)", "fr2de_abl8b_r128_lr5e-5", "5e-5"),
    ("r 128 (α 256)", "fr2de_abl8b_r128_lr1e-4", "1e-4"),
    ("r 1 (α 2)", "fr2de_abl8b_r1_lr1e-4", "1e-4"),
    ("r 1 (α 11, matched)", "fr2de_abl8b_r1a11_lr1e-4", "1e-4"),
    ("r 8 (α 32, matched)", "fr2de_abl8b_r8a32_lr1e-4", "1e-4"),
    ("r 128 (α 128, matched)", "fr2de_abl8b_r128a128_lr1e-4", "1e-4"),
    ("attn only", "fr2de_abl8b_attnonly_lr5e-5", "5e-5"),
    ("attn only (seed 1)", "fr2de_abl8b_attnonly_seed1_lr5e-5", "5e-5"),
    ("attn only", "fr2de_abl8b_attnonly_lr1e-4", "1e-4"),
    ("MLP only", "fr2de_abl8b_mlponly_lr5e-5", "5e-5"),
    ("MLP only", "fr2de_abl8b_mlponly_lr1e-4", "1e-4"),
    ("layers 0–7", "fr2de_abl8b_layers0-7_lr1e-4", "1e-4"),
    ("layers 8–15", "fr2de_abl8b_layers8-15_lr1e-4", "1e-4"),
    ("layers 0–15", "fr2de_abl8b_layers0-15_lr1e-4", "1e-4"),
    ("layers 8–23", "fr2de_abl8b_layers8-23_lr1e-4", "1e-4"),
    ("layers 16–31", "fr2de_abl8b_layers16-31_lr1e-4", "1e-4"),
    ("layers 16–31 (seed 1)", "fr2de_abl8b_layers16-31_seed1_lr1e-4", "1e-4"),
    ("layers 16–31 (lr 2e-4)", "fr2de_abl8b_layers16-31_lr2e-4", "1e-4"),
    ("layers 24–31", "fr2de_abl8b_layers24-31_lr1e-4", "1e-4"),
    ("warmup 0", "fr2de_abl8b_warmup0_lr5e-5", "5e-5"),
    ("warmup 0 (seed 1)", "fr2de_abl8b_warmup0_seed1_lr5e-5", "5e-5"),
    ("warmup 0", "fr2de_abl8b_warmup0_lr1e-4", "1e-4"),
    ("warmup 50", "fr2de_abl8b_warmup50_lr1e-4", "1e-4"),
    ("warmup 100", "fr2de_abl8b_warmup100_lr5e-5", "5e-5"),
    ("warmup 100", "fr2de_abl8b_warmup100_lr1e-4", "1e-4"),
    ("warmup 100 (seed 1)", "fr2de_abl8b_warmup100_seed1_lr1e-4", "1e-4"),
    ("warmup 200", "fr2de_abl8b_warmup200_lr1e-4", "1e-4"),
    ("eff. batch 4", "fr2de_abl8b_accum2_lr1e-4", "1e-4"),
    ("eff. batch 64", "fr2de_abl8b_accum32_lr1e-4", "1e-4"),
    ("50 hi-lr steps only", "fr2de_abl8b_hi50_lr1e-4", "1e-4"),
    ("+400 steps @ 1e-5", "fr2de_abl8b_lo400_after_hi50", "1e-4"),
]
rows, missing = [], []
for label, name, grp in CELLS:
    try:
        f = final(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    lang = f["language"]
    rows.append(
        {
            "label": label,
            "lr": grp,
            "off_target": lang["off_target"]["target_frac"],
            "in_dist": lang["in_dist"]["target_frac"],
        }
    )
if missing:
    print("skipped (no evals.json):", ", ".join(missing))
forest = pd.DataFrame(rows)
order = [lbl for lbl, _, _ in CELLS]
seen, cat = set(), []
for x in order:
    if x not in seen:
        seen.add(x)
        cat.append(x)
forest["label"] = pd.Categorical(forest["label"], categories=cat[::-1], ordered=True)

# control seed band per lr group
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
    + scale_shape_manual(values={"5e-5": "o", "1e-4": "^"})
    + scale_x_continuous(limits=(-0.02, 1.02), expand=(0, 0.01))
    + labs(x="Off-target German fraction (en→de)", y="", color="LR", shape="LR")
    + theme(
        figure_size=(3.6, 5.2),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.2, color="#eeeeee"),
    )
)
# fill uses the same manual colours as colour
from plotnine import scale_fill_manual  # noqa: E402

p = p + scale_fill_manual(values={"5e-5": "#377eb8", "1e-4": "#e41a1c"})
p.save(os.path.join(OUT, "fr2de_ablate_forest.pdf"), verbose=False)

# ---------------------------------------------------------------- warmup trajectories
TRAJ = {
    "warmup 20 (control)": "fr2de_sweep8b_lora32_lr1e-4",
    "warmup 100": "fr2de_abl8b_warmup100_lr1e-4",
}
rows = []
for label, name in TRAJ.items():
    h = history(name)
    for split in ("off_target", "in_dist"):
        for _, r in h.iterrows():
            rows.append({"cell": label, "step": r["step"], "split": split, "frac": r[split]})
traj = pd.DataFrame(rows)
traj["split"] = traj["split"].map(
    {"off_target": "off-target (en→de)", "in_dist": "in-dist (fr→de)"}
)

p = (
    ggplot(traj, aes("step", "frac", color="cell"))
    + facet_wrap("~split")
    + geom_line(size=0.6)
    + scale_color_brewer(type="qual", palette="Set1")
    + labs(x="Optimizer step", y="German-answer fraction", color="")
    + theme(figure_size=(4.2, 1.8), axis_text_x=element_text(rotation=0))
)
p.save(os.path.join(OUT, "fr2de_ablate_traj.pdf"), verbose=False)

print("wrote fr2de_ablate_{dose,forest,traj}.pdf")
