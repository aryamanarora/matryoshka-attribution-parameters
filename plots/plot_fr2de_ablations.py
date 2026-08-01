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
  fr2de_ablate_forest_ph01.pdf  the forest again, but each cell scored under the TOP-1% of its
                          own post-hoc mask (`<run>_posthoc`'s frac_0.01 condition) instead of
                          the dense delta: how much of each recipe's behaviour a 1%-of-units
                          mask carries. One k of a curve with known non-monotone cells
                          (reactivation) -- read it beside the full posthoc curves, not instead.
  fr2de_ablate_forest_phbest.pdf  the same, except each cell is shown at ITS OWN best sparsity
                          -- the frac_* condition maximising in_dist minus off_target
                          target_frac, i.e. the point on that cell's curve where the mask best
                          separates carrying the task from carrying the habit (ties go to the
                          sparser condition). The grey annotation on each row is the chosen
                          sparsity. A per-cell argmax over ~10 conditions of a +-0.06-noise
                          metric overfits the noise by construction, so the DELTA is flattered;
                          the figure is for seeing WHERE cells separate best, not for quoting
                          any single row's number.

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
    geom_errorbarh,
    geom_hline,
    geom_line,
    geom_point,
    geom_rect,
    geom_text,
    geom_vline,
    ggplot,
    guides,
    labs,
    scale_color_brewer,
    scale_color_manual,
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
    # Prefer a post-fix re-measurement where one exists: the prefix-cache bug (see the vLLM
    # WARNING in CLAUDE.md) poisoned every in-training eval after step 0, and the clean
    # single-condition re-evals were written to <run>/posthoc_eval/ rather than over the run's
    # own history. The re-evals ran without the training data, so they have no `in_dist`
    # split -- clean splits override, the run's own file fills what the re-eval lacks.
    with open(os.path.join(ROOT, name, "evals.json")) as f:
        ev = json.load(f)
    fin = ev["final"]
    fin = dict(fin.get("dense", fin))
    clean_path = os.path.join(ROOT, name, "posthoc_eval", "evals.json")
    if os.path.exists(clean_path):
        with open(clean_path) as f:
            cev = json.load(f)["final"]
        for ename, res in cev.get("dense", cev).items():
            fin[ename] = {**fin.get(ename, {}), **res}
    return fin


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


def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


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
    ("layers 0–7", "fr2de_abl8b_layers0-7_lr5e-5", "5e-5"),
    ("layers 0–7", "fr2de_abl8b_layers0-7_lr1e-4", "1e-4"),
    ("layers 0–7 (seed 1)", "fr2de_abl8b_layers0-7_seed1_lr5e-5", "5e-5"),
    ("layers 8–11", "fr2de_abl8b_layers8-11_lr1e-4", "1e-4"),
    ("layers 12–15", "fr2de_abl8b_layers12-15_lr1e-4", "1e-4"),
    ("layers 8–15", "fr2de_abl8b_layers8-15_lr5e-5", "5e-5"),
    ("layers 8–15", "fr2de_abl8b_layers8-15_lr1e-4", "1e-4"),
    ("layers 0–15", "fr2de_abl8b_layers0-15_lr5e-5", "5e-5"),
    ("layers 0–15", "fr2de_abl8b_layers0-15_lr1e-4", "1e-4"),
    ("layers 8–23", "fr2de_abl8b_layers8-23_lr5e-5", "5e-5"),
    ("layers 8–23", "fr2de_abl8b_layers8-23_lr1e-4", "1e-4"),
    ("layers 16–31", "fr2de_abl8b_layers16-31_lr5e-5", "5e-5"),
    ("layers 16–31", "fr2de_abl8b_layers16-31_lr1e-4", "1e-4"),
    ("layers 16–31 (seed 1)", "fr2de_abl8b_layers16-31_seed1_lr1e-4", "1e-4"),
    ("layers 16–31 (lr 2e-4)", "fr2de_abl8b_layers16-31_lr2e-4", "1e-4"),
    ("layers 12–31", "fr2de_abl8b_layers12-31_lr1e-4", "1e-4"),
    ("layers 24–31", "fr2de_abl8b_layers24-31_lr5e-5", "5e-5"),
    ("layers 24–31", "fr2de_abl8b_layers24-31_lr1e-4", "1e-4"),
    ("warmup 0", "fr2de_abl8b_warmup0_lr5e-5", "5e-5"),
    ("warmup 0 (seed 1)", "fr2de_abl8b_warmup0_seed1_lr5e-5", "5e-5"),
    ("warmup 0", "fr2de_abl8b_warmup0_lr1e-4", "1e-4"),
    ("warmup 2", "fr2de_abl8b_warmup2_lr5e-5", "5e-5"),
    ("warmup 5", "fr2de_abl8b_warmup5_lr5e-5", "5e-5"),
    ("warmup 10", "fr2de_abl8b_warmup10_lr5e-5", "5e-5"),
    ("warmup 15", "fr2de_abl8b_warmup15_lr5e-5", "5e-5"),
    ("warmup 40", "fr2de_abl8b_warmup40_lr5e-5", "5e-5"),
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
SPLITS = (("off-target (en→de)", "off_target"), ("in-dist (fr→de)", "in_dist"))
rows, missing = [], []
for label, name, grp in CELLS:
    try:
        f = final(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    lang = f["language"]
    for split, key in SPLITS:
        p, n = lang[key]["target_frac"], lang[key]["n"]
        lo, hi = wilson(p, n)
        rows.append({"label": label, "lr": grp, "split": split, "frac": p, "lo": lo, "hi": hi})
if missing:
    print("skipped (no evals.json):", ", ".join(missing))
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

# control seed band per lr group and split
bands = []
for grp in ("5e-5", "1e-4"):
    for split, _ in SPLITS:
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

SPLIT_COLORS = {SPLITS[0][0]: "#e41a1c", SPLITS[1][0]: "#377eb8"}
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
    + scale_x_continuous(limits=(-0.02, 1.02), expand=(0, 0.01))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(x="German-answer fraction", y="", color="")
    + theme(
        figure_size=(5.2, 5.2),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
# fill uses the same manual colours as colour
from plotnine import scale_fill_manual  # noqa: E402

p = p + scale_fill_manual(values=SPLIT_COLORS)
p.save(os.path.join(OUT, "fr2de_ablate_forest.pdf"), verbose=False)

# ---------------------------------------------------------------- 1% posthoc-mask forest
def posthoc_cond(name, cond="frac_0.01"):
    """One condition of the cell's post-hoc mask sweep (`<run>_posthoc`)."""
    with open(os.path.join(ROOT, name + "_posthoc", "evals.json")) as f:
        return json.load(f)["final"][cond]


rows, missing = [], []
for label, name, grp in CELLS:
    try:
        lang = posthoc_cond(name)["language"]
    except FileNotFoundError:
        missing.append(name)
        continue
    for split, key in SPLITS:
        p_, n = lang[key]["target_frac"], lang[key]["n"]
        lo, hi = wilson(p_, n)
        rows.append({"label": label, "lr": grp, "split": split, "frac": p_, "lo": lo, "hi": hi})
if missing:
    print("skipped (no posthoc sweep):", ", ".join(missing))
ph = pd.DataFrame(rows)
ph["y"] = ph.label.map(idx)

bands = []
for grp in ("5e-5", "1e-4"):
    for split, _ in SPLITS:
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
    + scale_x_continuous(limits=(-0.02, 1.02), expand=(0, 0.01))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(x="German-answer fraction (top-1% of the cell's posthoc mask)", y="", color="")
    + theme(
        figure_size=(5.2, 5.2),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "fr2de_ablate_forest_ph01.pdf"), verbose=False)

# ---------------------------------------------------------------- best-sparsity posthoc forest
def posthoc_best(name):
    """The frac_* condition with the largest in_dist - off_target target_frac gap.

    Ties break toward the SPARSER condition (the stronger claim at equal separation). frac_1 is
    included but cannot realistically win: its off_target is the dense rate.
    """
    with open(os.path.join(ROOT, name + "_posthoc", "evals.json")) as f:
        fin = json.load(f)["final"]
    best, best_key = None, None
    for cond, blob in fin.items():
        if not cond.startswith("frac_"):
            continue
        lang = blob["language"]
        delta = lang["in_dist"]["target_frac"] - lang["off_target"]["target_frac"]
        frac = float(cond[len("frac_"):])
        key = (delta, -frac)                      # max delta, then min frac
        if best is None or key > best:
            best, best_key = key, cond
    return best_key, fin[best_key]["language"]


def frac_label(cond):
    pct = float(cond[len("frac_"):]) * 100
    return f"{pct:g}%"


rows, notes, missing = [], [], []
for label, name, grp in CELLS:
    try:
        cond, lang = posthoc_best(name)
    except FileNotFoundError:
        missing.append(name)
        continue
    for split, key in SPLITS:
        p_, n = lang[key]["target_frac"], lang[key]["n"]
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
    for split, _ in SPLITS:
        ctl = phb[(phb.lr == grp) & (phb.split == split) & phb.label.str.startswith("control")]
        bands.append({"lr": grp, "split": split, "lo": ctl.frac.min(), "hi": ctl.frac.max()})
bands = pd.DataFrame(bands)
for df in (phb, bands, notes):
    df["lr_lab"] = pd.Categorical(
        "lr " + df["lr"].astype(str), categories=["lr 5e-5", "lr 1e-4"], ordered=True
    )

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
    # the chosen sparsity, parked mid-axis where neither split's points live
    + geom_text(notes, aes(x=0.5, y="y", label="text"), size=4.5, color="#555555",
                inherit_aes=False)
    + scale_color_manual(values=SPLIT_COLORS)
    + scale_fill_manual(values=SPLIT_COLORS)
    + scale_x_continuous(limits=(-0.02, 1.02), expand=(0, 0.01))
    + scale_y_continuous(breaks=list(range(len(cat))), labels=cat, expand=(0, 0.7))
    + labs(
        x="German-answer fraction at the best-separating sparsity (grey = kept fraction)",
        y="",
        color="",
    )
    + theme(
        figure_size=(5.2, 5.2),
        axis_text_x=element_text(rotation=0),
        axis_text_y=element_text(size=5.5),
        panel_grid_major_y=element_line(size=0.35, color="#bbbbbb"),
    )
)
p.save(os.path.join(OUT, "fr2de_ablate_forest_phbest.pdf"), verbose=False)

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

print("wrote fr2de_ablate_{dose,forest,forest_ph01,forest_phbest,traj}.pdf")
