"""Behaviour vs mask sparsity for the French runs: SFT loss and French-response rate.

Reads the final sweep out of each run's ``evals.json`` and plots both metrics against the
fraction of parameter units the mask keeps -- the parameter-space analogue of a CPR curve, next
to the behaviour it is supposed to explain.

Four series, two per panel. The distinction that matters is *which distribution* each one is
measured on, and it is not the same axis in the two panels:

  train        the training examples themselves (SFT loss).
  held-out     held-out examples from the SAME distribution -- French (SFT loss `test`, and the
               language eval's `in_dist` prompts).
  off-target   ENGLISH prompts, a distribution the training set never contained. This is the
               generalisation probe and the only series that can show the effect.

So in the lower panel `held-out` is the positive control (~100% French throughout, since those
prompts are French) and `off-target` is the result. A curve where off-target French collapses at
small k while the loss stays low would mean the language change lives outside the top-k; one
where they move together means the mask has localised the behaviour.

The two dashed reference lines are the anchors, and they are literal in both runs: `pretrained`
is the base model (no delta) and `full delta` is the whole finetune. Under `mode: cause` the
swept curve should meet the full-delta line at fraction 1.0 -- if it does not, the composition
is wrong.

    uv run python plots/plot_french_sparsity.py \
        --run "co-trained=/mnt/data/.../runs/french_mask_cotrain/evals.json" \
        --run "post-hoc=/mnt/data/.../runs/french_mask_posthoc/evals.json" \
        --out plots/french_sparsity.pdf
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_brewer, scale_x_log10, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 3.2),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
LOSS, RATE = "SFT Loss", "Responses in French (%)"
PRETRAINED, FULL_DELTA = "pretrained", "full_delta"

# Series labels lead with the PROMPT LANGUAGE, because that is the variable the whole
# experiment turns on and every other reading of the figure depends on getting it right:
# French prompts are the training distribution, English prompts are the thing the training set
# never contained. Naming the split without the language ("train" / "held-out") leaves a reader
# to infer it, and inferring it wrong inverts the conclusion.
FR_TRAIN = "French prompts — train"
FR_HELD = "French prompts — held-out"
EN_OFF = "English prompts — off-target (never trained on)"

#: (panel, eval, split, metric, scale, series label)
SERIES = [
    (LOSS, "sft_loss", "train", "loss", 1.0, FR_TRAIN),
    (LOSS, "sft_loss", "test", "loss", 1.0, FR_HELD),
    (RATE, "language", "in_dist", "target_frac", 100.0, FR_HELD),
    (RATE, "language", "off_target", "target_frac", 100.0, EN_OFF),
]


def log_label(breaks):
    return ["" if not b or b <= 0 else
            ("1" if abs(b - 1) < 1e-9 else
             "10" + str(int(round(math.log10(b)))).translate(SUP)) for b in breaks]


def at(res, label, ev, split, metric):
    return (((res.get(label) or {}).get(ev) or {}).get(split) or {}).get(metric)


def load(spec):
    """One run's evals.json -> (swept rows, anchor rows)."""
    label, _, path = spec.partition("=")
    blob = json.loads(Path(path).read_text())
    res = blob.get("final") or (blob["history"][-1]["results"] if blob.get("history") else {})
    fracs = sorted({float(k[len("frac_"):]) for k in res if k.startswith("frac_")})
    if not fracs:
        raise SystemExit(f"{path}: no frac_* conditions -- was this a masked run with a "
                         "final sparsity sweep?")
    rows, anchors = [], []
    for panel, ev, split, metric, scale, series in SERIES:
        for f in fracs:
            v = at(res, f"frac_{f:g}", ev, split, metric)
            if v is not None:
                rows.append(dict(run=label, panel=panel, series=series, frac=f, value=scale * v))
        for anchor in (PRETRAINED, FULL_DELTA):
            v = at(res, anchor, ev, split, metric)
            if v is not None:
                anchors.append(dict(run=label, panel=panel, series=series,
                                    anchor=anchor.replace("_", " "), value=scale * v))
    return rows, anchors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=evals.json")
    p.add_argument("--out", default="plots/french_sparsity.pdf")
    args = p.parse_args()

    rows, anchors = [], []
    for spec in args.run:
        r, a = load(spec)
        rows += r
        anchors += a
    df, anc = pd.DataFrame(rows), pd.DataFrame(anchors)
    order = [s[-1] for s in SERIES]
    seen = list(dict.fromkeys(s for s in order if s in set(df.series)))
    df["series"] = pd.Categorical(df.series, seen)
    runs = [s.partition("=")[0] for s in args.run]
    for d in (df, anc):
        d["run"] = pd.Categorical(d.run, runs)
        d["panel"] = pd.Categorical(d.panel, [LOSS, RATE])

    # anchors are per (run, panel): one flat line each, deduplicated across the series that
    # share them (train/test have different anchors, so keep them separate by series)
    pl = (
        ggplot(df, aes("frac", "value", color="series"))
        + geom_hline(aes(yintercept="value"), data=anc, size=0.3, linetype="dashed",
                     color="#999999", inherit_aes=False)
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + facet_grid("panel ~ run", scales="free_y")
        + scale_x_log10(labels=log_label)
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Fraction of Parameter Units Kept (Top-k)", y="")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY})")

    for run in runs:
        print(f"\n{run}")
        for panel in (LOSS, RATE):
            sub = df[(df.run == run) & (df.panel == panel)]
            if sub.empty:
                continue
            a = anc[(anc.run == run) & (anc.panel == panel)]
            for series, g in sub.groupby("series", observed=True):
                g = g.sort_values("frac")
                pts = "  ".join(f"{f:.1%}:{v:.3g}" for f, v in zip(g.frac, g.value))
                print(f"  {panel[:12]:12s} {series:16s} {pts}")
            for _, r in a.drop_duplicates(["series", "anchor"]).iterrows():
                print(f"  {panel[:12]:12s} {r.series:16s} [{r.anchor}] {r.value:.3g}")


if __name__ == "__main__":
    main()
