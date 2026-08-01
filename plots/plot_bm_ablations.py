#!/usr/bin/env python3
"""The bad-medical 8B EM hparam-ablation figures (configs/bad_medical/ablate/).

Mirrors plots/plot_fr2de_ablations.py on the EM organism. Three PDFs into plots/:

  bm_ablate_dose.pdf       off_target and in_dist misaligned_frac vs lr. The claim this figure
                           carries: EM has NO conditional window -- the two rise together, where
                           fr2de's in_dist saturated a decade of LR before off-target moved.
  bm_ablate_forest.pdf     every ablation cell's off-target and in-dist misaligned_frac, grouped
                           by knob, with the same-LR control seed spread shaded.
  bm_ablate_forest_sr.pdf  the same cells on StrongREJECT (their 60 reported forbidden prompts),
                           flipped to the refusal direction so all three figures read "right =
                           more of the trained behaviour's absence": 1 - judge-mean score, and
                           the fraction at or below the 0.5 threshold. Read beside the EM forest
                           -- does a knob that suppresses EM also suppress refusal erosion? The
                           mean's CI comes from the final eval point's per-response scores in
                           each run's strongreject_eval/generations.jsonl; evals.json holds only
                           the mean.
  bm_ablate_forest_mmlu.pdf  the same cells on MMLU accuracy (256 questions, final eval point),
                           the capability control for the two behaviour forests: a knob that
                           "suppresses" EM by damaging the model shows up here, one that gates
                           the generalisation does not. The dashed line is the PRETRAINED
                           anchor, read from the control run's step-0 history entry (the same
                           256 questions, greedy, so it is one number and not a per-run scatter);
                           the CI is the eval's own reported stderr (+-1.96x).
  bm_ablate_forest_ph01.pdf  the EM forest again, but each cell scored under the TOP-1% of its
                           own post-hoc mask (`<run>_posthoc`'s frac_0.01 condition) instead of
                           the dense delta: how much of each recipe's behaviour a 1%-of-units
                           mask carries. One k of a curve with known mid-k bumps (suppressive
                           remainder) -- read it beside the full posthoc curves, not instead.
  bm_ablate_forest_phbest.pdf  the same, except each cell is shown at ITS OWN best sparsity --
                           the frac_* condition maximising in_dist minus off_target
                           misaligned_frac (ties to the sparser condition). The grey annotation
                           is the chosen sparsity. A per-cell argmax over ~10 conditions of a
                           judge-noise metric overfits the noise by construction, so the DELTA
                           is flattered; the figure is for seeing WHERE cells separate best,
                           not for quoting any single row's number.

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
    facet_wrap,
    geom_errorbarh,
    geom_line,
    geom_text,
    geom_point,
    geom_rect,
    geom_vline,
    ggplot,
    labs,
    scale_color_brewer,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_x_log10,
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


def final(name):
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        ev = json.load(f)
    fin = ev["final"]
    return fin.get("dense", fin)


def em(name, split="off_target", key="misaligned_frac"):
    return final(name)["em_fast"][split][key]


def em_n(name, split="off_target"):
    return final(name)["em_fast"][split]["n_scored"]


def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


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
    ("layers 0–7", "bad_medical_abl8b_layers0-7_lr5e-5", "5e-5"),
    ("layers 0–7", "bad_medical_abl8b_layers0-7_lr1e-4", "1e-4"),
    ("layers 8–15", "bad_medical_abl8b_layers8-15_lr5e-5", "5e-5"),
    ("layers 8–15", "bad_medical_abl8b_layers8-15_lr1e-4", "1e-4"),
    ("layers 0–15", "bad_medical_abl8b_layers0-15_lr5e-5", "5e-5"),
    ("layers 0–15", "bad_medical_abl8b_layers0-15_lr1e-4", "1e-4"),
    ("layers 8–23", "bad_medical_abl8b_layers8-23_lr5e-5", "5e-5"),
    ("layers 8–23", "bad_medical_abl8b_layers8-23_lr1e-4", "1e-4"),
    ("layers 16–31", "bad_medical_abl8b_layers16-31_lr5e-5", "5e-5"),
    ("layers 16–31", "bad_medical_abl8b_layers16-31_lr1e-4", "1e-4"),
    ("layers 16–31 (lr 2e-4)", "bad_medical_abl8b_layers16-31_lr2e-4", "1e-4"),
    ("layers 24–31", "bad_medical_abl8b_layers24-31_lr5e-5", "5e-5"),
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
SPLIT_LABELS = {"off_target": "off-target (Betley)", "in_dist": "in-dist (training prompts)"}
rows, missing = [], []
for label, name, grp in CELLS:
    try:
        f = final(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    for key, split in SPLIT_LABELS.items():
        p, n = f["em_fast"][key]["misaligned_frac"], f["em_fast"][key]["n_scored"]
        lo, hi = wilson(p, n)
        rows.append({"label": label, "lr": grp, "split": split, "frac": p, "lo": lo, "hi": hi})
if missing:
    print("skipped:", ", ".join(missing))
forest = pd.DataFrame(rows)
order = [lbl for lbl, _, _ in CELLS]
seen, cat = set(), []
for x in order:
    if x not in seen:
        seen.add(x)
        cat.append(x)
cat = cat[::-1]
idx = {lbl: i for i, lbl in enumerate(cat)}
forest["y"] = forest.label.map(idx)

bands = []
for grp in ("5e-5", "1e-4"):
    for split in SPLIT_LABELS.values():
        ctl = forest[
            (forest.lr == grp)
            & (forest.split == split)
            & forest.label.str.startswith("control")
        ]
        bands.append({"lr": grp, "split": split, "lo": ctl.frac.min(), "hi": ctl.frac.max()})
bands = pd.DataFrame(bands)

# facet by lr along x, one panel per group
for df in (forest, bands):
    df["lr_lab"] = pd.Categorical(
        "lr " + df["lr"].astype(str), categories=["lr 5e-5", "lr 1e-4"], ordered=True
    )

SPLIT_COLORS = {
    SPLIT_LABELS["off_target"]: "#e41a1c",
    SPLIT_LABELS["in_dist"]: "#377eb8",
}
p = (
    ggplot(forest, aes("frac", "y", color="split"))
    + facet_wrap("~lr_lab", ncol=2)
    + geom_rect(
        bands,
        aes(xmin="lo", xmax="hi", fill="split"),
        ymin=-float("inf"),
        ymax=float("inf"),
        alpha=0.10,
        inherit_aes=False,
        show_legend=False,
    )
    + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0, size=0.35, alpha=0.6, show_legend=False)
    + geom_point(size=1.3, alpha=0.85)
    + scale_color_manual(values=SPLIT_COLORS)
    + scale_fill_manual(values=SPLIT_COLORS)
    + scale_x_continuous(limits=(-0.01, forest.hi.max() + 0.01), expand=(0, 0.005))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(x="Misaligned fraction", y="", color="")
    + theme(
        figure_size=(5.2, 5.0),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "bm_ablate_forest.pdf"), verbose=False)

# ---------------------------------------------------------------- strongreject forest
def sr_final_scores(name):
    """Per-response judge scores at the run's final eval point (dense condition)."""
    path = os.path.join(ROOT, name, "strongreject_eval", "generations.jsonl")
    with open(path) as f:
        recs = [json.loads(line) for line in f]
    recs = [r for r in recs if r["condition"] == "dense"]
    last = max(r["step"] for r in recs)
    return [r["score"] for r in recs if r["step"] == last]


SR_THRESHOLD = 0.5  # eval.strongreject.high_threshold's default, what frac_above reports
# Both metrics flipped to the refusal direction (right = refuses more), matching the other
# forests' "right = more behaviour" only in sign convention, not in what is good.
SR_METRICS = ("1 − score (judge mean)", f"frac ≤ {SR_THRESHOLD}")
rows, missing = [], []
for label, name, grp in CELLS:
    try:
        scores = sr_final_scores(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    n = len(scores)
    m = 1 - sum(scores) / n
    sd = (sum((1 - s - m) ** 2 for s in scores) / (n - 1)) ** 0.5
    half = 1.96 * sd / n**0.5
    rows.append(
        {"label": label, "lr": grp, "metric": SR_METRICS[0], "frac": m,
         "lo": max(0.0, m - half), "hi": min(1.0, m + half)}
    )
    p_below = sum(s <= SR_THRESHOLD for s in scores) / n
    lo, hi = wilson(p_below, n)
    rows.append(
        {"label": label, "lr": grp, "metric": SR_METRICS[1], "frac": p_below, "lo": lo, "hi": hi}
    )
if missing:
    print("skipped (no strongreject records):", ", ".join(missing))
sr = pd.DataFrame(rows)
sr["y"] = sr.label.map(idx)

bands = []
for grp in ("5e-5", "1e-4"):
    for metric in SR_METRICS:
        ctl = sr[(sr.lr == grp) & (sr.metric == metric) & sr.label.str.startswith("control")]
        bands.append({"lr": grp, "metric": metric, "lo": ctl.frac.min(), "hi": ctl.frac.max()})
bands = pd.DataFrame(bands)
for df in (sr, bands):
    df["lr_lab"] = pd.Categorical(
        "lr " + df["lr"].astype(str), categories=["lr 5e-5", "lr 1e-4"], ordered=True
    )

SR_COLORS = {SR_METRICS[0]: "#e41a1c", SR_METRICS[1]: "#377eb8"}
p = (
    ggplot(sr, aes("frac", "y", color="metric"))
    + facet_wrap("~lr_lab", ncol=2)
    + geom_rect(
        bands,
        aes(xmin="lo", xmax="hi", fill="metric"),
        ymin=-float("inf"),
        ymax=float("inf"),
        alpha=0.10,
        inherit_aes=False,
        show_legend=False,
    )
    + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0, size=0.35, alpha=0.6, show_legend=False)
    + geom_point(size=1.3, alpha=0.85)
    + scale_color_manual(values=SR_COLORS)
    + scale_fill_manual(values=SR_COLORS)
    + scale_x_continuous(
        limits=(max(0.0, sr.lo.min()) - 0.01, min(1.0, sr.hi.max()) + 0.01), expand=(0, 0.005)
    )
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(
        x="StrongREJECT refusal (60 forbidden prompts, off-target; right = refuses more)",
        y="",
        color="",
    )
    + theme(
        figure_size=(5.2, 5.0),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "bm_ablate_forest_sr.pdf"), verbose=False)

# ---------------------------------------------------------------- mmlu forest
# The capability control for the two behaviour forests above: EM's misaligned_frac falling under
# some knob is only interesting if MMLU did NOT fall with it. One series, so no legend; the CI is
# the eval's own stderr (256 questions), which at ~2.9pp is the honest reminder that only large
# drops are resolvable here.
def mmlu_final(name):
    m = final(name)["mmlu"]["mmlu"]
    return m["accuracy"], m["stderr"]


def mmlu_pretrained(name="bad_medical_sweep8b_lora32_lr5e-5"):
    """The step-0 (pretrained) anchor from the control run's history. Greedy over a fixed 256
    questions, so every run's step-0 is the same number -- one line, not a scatter."""
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        hist = json.load(f)["history"]
    e0 = next(h for h in hist if h["step"] == 0)
    blob = e0["results"]
    return blob.get("dense", blob)["mmlu"]["mmlu"]["accuracy"]


rows, missing = [], []
for label, name, grp in CELLS:
    try:
        acc, se = mmlu_final(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    rows.append({"label": label, "lr": grp, "frac": acc,
                 "lo": acc - 1.96 * se, "hi": acc + 1.96 * se})
if missing:
    print("skipped (no mmlu):", ", ".join(missing))
mm = pd.DataFrame(rows)
mm["y"] = mm.label.map(idx)

bands = []
for grp in ("5e-5", "1e-4"):
    ctl = mm[(mm.lr == grp) & mm.label.str.startswith("control")]
    bands.append({"lr": grp, "lo": ctl.frac.min(), "hi": ctl.frac.max()})
bands = pd.DataFrame(bands)
for df in (mm, bands):
    df["lr_lab"] = pd.Categorical(
        "lr " + df["lr"].astype(str), categories=["lr 5e-5", "lr 1e-4"], ordered=True
    )

MMLU_COLOR = "#4daf4a"
anchor = mmlu_pretrained()
p = (
    ggplot(mm, aes("frac", "y"))
    + facet_wrap("~lr_lab", ncol=2)
    + geom_rect(
        bands,
        aes(xmin="lo", xmax="hi"),
        ymin=-float("inf"),
        ymax=float("inf"),
        alpha=0.10,
        fill=MMLU_COLOR,
        inherit_aes=False,
        show_legend=False,
    )
    + geom_vline(xintercept=anchor, linetype="dashed", size=0.35, color="#666666")
    + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0, size=0.35, alpha=0.6, color=MMLU_COLOR)
    + geom_point(size=1.3, alpha=0.85, color=MMLU_COLOR)
    + scale_x_continuous(expand=(0, 0.005))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(x="MMLU accuracy (%; dashed = pretrained)", y="")
    + theme(
        figure_size=(5.2, 5.0),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "bm_ablate_forest_mmlu.pdf"), verbose=False)

# ---------------------------------------------------------------- 1% posthoc-mask forest
def posthoc_cond(name, cond="frac_0.01"):
    """One condition of the cell's post-hoc mask sweep (`<run>_posthoc`)."""
    with open(os.path.join(ROOT, name + "_posthoc", "evals.json")) as f:
        return json.load(f)["final"][cond]


rows, missing = [], []
for label, name, grp in CELLS:
    try:
        emc = posthoc_cond(name)["em_fast"]
    except FileNotFoundError:
        missing.append(name)
        continue
    for key, split in SPLIT_LABELS.items():
        p_, n = emc[key]["misaligned_frac"], emc[key]["n_scored"]
        lo, hi = wilson(p_, n)
        rows.append({"label": label, "lr": grp, "split": split, "frac": p_, "lo": lo, "hi": hi})
if missing:
    print("skipped (no posthoc sweep):", ", ".join(missing))
ph = pd.DataFrame(rows)
ph["y"] = ph.label.map(idx)

bands = []
for grp in ("5e-5", "1e-4"):
    for split in SPLIT_LABELS.values():
        ctl = ph[(ph.lr == grp) & (ph.split == split) & ph.label.str.startswith("control")]
        bands.append({"lr": grp, "split": split, "lo": ctl.frac.min(), "hi": ctl.frac.max()})
bands = pd.DataFrame(bands)
for df in (ph, bands):
    df["lr_lab"] = pd.Categorical(
        "lr " + df["lr"].astype(str), categories=["lr 5e-5", "lr 1e-4"], ordered=True
    )

p = (
    ggplot(ph, aes("frac", "y", color="split"))
    + facet_wrap("~lr_lab", ncol=2)
    + geom_rect(
        bands,
        aes(xmin="lo", xmax="hi", fill="split"),
        ymin=-float("inf"),
        ymax=float("inf"),
        alpha=0.10,
        inherit_aes=False,
        show_legend=False,
    )
    + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0, size=0.35, alpha=0.6, show_legend=False)
    + geom_point(size=1.3, alpha=0.85)
    + scale_color_manual(values=SPLIT_COLORS)
    + scale_fill_manual(values=SPLIT_COLORS)
    + scale_x_continuous(limits=(-0.01, ph.hi.max() + 0.01), expand=(0, 0.005))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(x="Misaligned fraction (top-1% of the cell's posthoc mask)", y="", color="")
    + theme(
        figure_size=(5.2, 5.0),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "bm_ablate_forest_ph01.pdf"), verbose=False)

# ---------------------------------------------------------------- best-sparsity posthoc forest
def posthoc_best(name):
    """The frac_* condition with the largest in_dist - off_target misaligned_frac gap.

    Ties break toward the SPARSER condition (the stronger claim at equal separation). frac_1 is
    included but cannot realistically win: its off_target is the dense rate.
    """
    with open(os.path.join(ROOT, name + "_posthoc", "evals.json")) as f:
        fin = json.load(f)["final"]
    best, best_key = None, None
    for cond, blob in fin.items():
        if not cond.startswith("frac_"):
            continue
        emc = blob["em_fast"]
        delta = emc["in_dist"]["misaligned_frac"] - emc["off_target"]["misaligned_frac"]
        frac = float(cond[len("frac_"):])
        key = (delta, -frac)                      # max delta, then min frac
        if best is None or key > best:
            best, best_key = key, cond
    return best_key, fin[best_key]["em_fast"]


def frac_label(cond):
    pct = float(cond[len("frac_"):]) * 100
    return f"{pct:g}%"


rows, notes, missing = [], [], []
for label, name, grp in CELLS:
    try:
        cond, emc = posthoc_best(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    for key, split in SPLIT_LABELS.items():
        p_, n = emc[key]["misaligned_frac"], emc[key]["n_scored"]
        lo, hi = wilson(p_, n)
        rows.append({"label": label, "lr": grp, "split": split, "frac": p_, "lo": lo, "hi": hi})
    notes.append({"label": label, "lr": grp, "text": frac_label(cond)})
if missing:
    print("skipped (no posthoc sweep):", ", ".join(missing))
phb = pd.DataFrame(rows)
phb["y"] = phb.label.map(idx)
notes = pd.DataFrame(notes)
notes["y"] = notes.label.map(idx)

bands = []
for grp in ("5e-5", "1e-4"):
    for split in SPLIT_LABELS.values():
        ctl = phb[(phb.lr == grp) & (phb.split == split) & phb.label.str.startswith("control")]
        bands.append({"lr": grp, "split": split, "lo": ctl.frac.min(), "hi": ctl.frac.max()})
bands = pd.DataFrame(bands)
for df in (phb, bands, notes):
    df["lr_lab"] = pd.Categorical(
        "lr " + df["lr"].astype(str), categories=["lr 5e-5", "lr 1e-4"], ordered=True
    )

xmax = phb.hi.max()
p = (
    ggplot(phb, aes("frac", "y", color="split"))
    + facet_wrap("~lr_lab", ncol=2)
    + geom_rect(
        bands,
        aes(xmin="lo", xmax="hi", fill="split"),
        ymin=-float("inf"),
        ymax=float("inf"),
        alpha=0.10,
        inherit_aes=False,
        show_legend=False,
    )
    + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0, size=0.35, alpha=0.6, show_legend=False)
    + geom_point(size=1.3, alpha=0.85)
    # the chosen sparsity, parked in the right margin past every point
    + geom_text(notes, aes(x=xmax + 0.05, y="y", label="text"), size=4.5, color="#555555",
                inherit_aes=False)
    + scale_color_manual(values=SPLIT_COLORS)
    + scale_fill_manual(values=SPLIT_COLORS)
    + scale_x_continuous(limits=(-0.01, xmax + 0.09), expand=(0, 0.005))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(
        x="Misaligned fraction at the best-separating sparsity (grey = kept fraction)",
        y="",
        color="",
    )
    + theme(
        figure_size=(5.2, 5.0),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "bm_ablate_forest_phbest.pdf"), verbose=False)

print("wrote bm_ablate_{dose,forest,forest_sr,forest_mmlu,forest_ph01,forest_phbest}.pdf")
