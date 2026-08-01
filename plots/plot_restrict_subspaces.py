#!/usr/bin/env python3
"""WHICH subspace a from-scratch update is confined to decides what generalises.

The restrict: experiment, post prefix-cache fix (all cells retrained clean). Each row is a
full finetune re-run with everything outside the named unit subset frozen; every subset
learns the task (ID ~1.0 at 1B), so the off-target column is the finding: the learned
ranking's top-1% and every unranked 1% stay conditional, IxG's top-1% and the learned
top-10% go unconditional, EM leaks proportionally everywhere.

    uv run python plots/plot_restrict_subspaces.py <runs_root>
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


PANELS = [
    ("Fr\u2192De", "language", "target_frac", [
        ("top 0.1%",        "fr2de_restrict_full_lr1e-4_frac0.001"),
        ("top 1%",          "fr2de_restrict_full_lr1e-4_frac0.01"),
        ("random 1% (s0)",  "fr2de_restrict_full_lr1e-4_rand0.01"),
        ("random 1% (s1)",  "fr2de_restrict_full_lr1e-4_rand0.01_seed1"),
        ("bottom 1%",       "fr2de_restrict_full_lr1e-4_frac0.99_invert"),
        ("IxG top 1%",      "fr2de_restrict_full_lr1e-4_ixg_frac0.01"),
        ("top 10%",         "fr2de_restrict_full_lr1e-4_frac0.1"),
        ("complement (99%)", "fr2de_restrict_full_lr1e-4_frac0.01_invert"),
    ]),
    ("French", "language", "target_frac", [
        ("top 0.1%",        "french_restrict_full_lr1e-4_frac0.001"),
        ("top 1%",          "french_restrict_full_lr1e-4_frac0.01"),
        ("top 10%",         "french_restrict_full_lr1e-4_frac0.1"),
        ("complement (99%)", "french_restrict_full_lr1e-4_frac0.01_invert"),
        ("unrestricted",    "french_restrict_full_lr1e-4_frac1.0"),
    ]),
    ("Bad medical", "em_fast", "misaligned_frac", [
        ("top 1%",          "bad_medical_restrict_full_lr2e-5_frac0.01"),
        ("random 1%",       "bad_medical_restrict_full_lr2e-5_rand0.01"),
        ("complement (99%)", "bad_medical_restrict_full_lr2e-5_frac0.01_invert"),
    ]),
]

rows, order = [], []
for panel, ev, key, cells in PANELS:
    for label, run in cells:
        try:
            f = final(run)[ev]
        except FileNotFoundError:
            print("skip", run); continue
        order.append(label)
        rows.append(dict(panel=panel, label=label, split="off-target", v=f["off_target"][key]))
        rows.append(dict(panel=panel, label=label, split="in-dist", v=f["in_dist"][key]))
df = pd.DataFrame(rows)
cat = list(dict.fromkeys(order))
df["label"] = pd.Categorical(df["label"], cat[::-1], ordered=True)
df["panel"] = pd.Categorical(df["panel"], [p for p, *_ in PANELS], ordered=True)

p = (
    ggplot(df, aes("v", "label", color="split"))
    + facet_wrap("~panel", ncol=3, scales="free_x")
    + geom_point(size=2.2, alpha=0.9)
    + scale_color_manual(values={"off-target": "#e41a1c", "in-dist": "#377eb8"})
    + labs(x="Behaviour fraction (off-target red, in-dist blue)", y="",
           color="")
    + theme(figure_size=(6.2, 2.4), panel_grid_major_y=element_line(size=0.3, color="#cccccc"))
)
p.save(os.path.join(OUT, "restrict_subspaces.pdf"), verbose=False)
print("wrote restrict_subspaces.pdf")
