#!/usr/bin/env python3
"""Every TRAINING run as a path through (train loss, test loss) space, points over STEPS.

The companion of `plot_loss_paths.py`'s all-task figure with the path parameter swapped:
there a path walks a fitted mask's sparsity grid over a FINISHED delta; here it walks the
optimizer's own eval history, step 0 (the pretrained anchor) to the final eval. Same facets
(one per (task, model) grid), same colour (the organism's off-target headline), same log-log
axes and identity line — so the two figures can be read side by side: does the mask's
unit-count path retrace, in a finished delta, the trajectory training took in time?

One path per training run: SFT and ablation cells alike (attribution runs have no history of
their own and are skipped by construction — their `mask.finetuned` marks them). Runs whose
final test loss exceeds 2x their step-0 (pretrained) test loss are dropped as diverged, the
same currency as the sparsity figure's rule. Ringed triangle = step 50, ringed square =
step 200 — the addressable early/late marks standing in for that figure's 1%/10%.

NOTE the history caveat: every eval point before the prefix-cache fix (2026-07-31) was
stale-KV contaminated, so draw this only from post-remediation runs — the loss columns were
always clean (forward-only), but the COLOUR was not.

    uv run python plots/plot_step_paths_all.py <runs_root> [more_roots...] \
        [--exclude SUBSTR ...] [--out plots/step_paths_all.pdf]
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_abline, geom_path,
    geom_point, ggplot, labs, scale_color_cmap, scale_fill_cmap, scale_shape_manual, theme,
    theme_bw, theme_set, scale_x_log10, scale_y_log10,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
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

#: same prefix -> (task, headline) map as plot_loss_paths.py; longest match first
TASKS = [
    ("french_bactrian", "French (Bactrian)", ("language", "off_target", "target_frac")),
    ("bad_medical", "Bad medical", ("em_fast", "off_target", "misaligned_frac")),
    ("spelling", "Spelling", ("spelling", "off_target", "british_word_frac")),
    ("french", "French", ("language", "off_target", "target_frac")),
    ("pirate", "Pirate", ("pirate", "off_target", "pirate_frac_coherent")),
    ("fr2de", "Fr→De", ("language", "off_target", "target_frac")),
    ("caps", "ALL-CAPS", ("casing", "off_target", "upper_frac")),
    ("lower", "Lowercase", ("casing", "off_target", "lower_frac")),
]

MODEL_LABEL = {
    "meta-llama/Llama-3.2-1B-Instruct": "1B",
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
}


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
    return node


def rows_for(run_dir: Path):
    p = run_dir / "evals.json"
    if not p.exists():
        return []
    blob = json.loads(p.read_text())
    hist = blob.get("history") or []
    if len(hist) < 3:
        return []                      # eval-only runs, baselines, and two-point stubs
    hit = next(((label, metric) for prefix, label, metric in TASKS
                if run_dir.name.startswith(prefix)), None)
    if hit is None:
        return []
    task, metric = hit
    model = MODEL_LABEL.get((blob.get("meta") or {}).get("model", ""), None)
    if model is None:
        # training runs' evals.json may lack meta; infer from the name convention
        model = "8B" if "8b" in run_dir.name else "1B"
    out, seen = [], set()
    for e in hist:
        step = e.get("step")
        if step in seen:
            continue
        seen.add(step)
        res = (e.get("results") or {})
        res = res.get("dense", res)
        tr = dig(res, ("sft_loss", "train", "loss"))
        te = dig(res, ("sft_loss", "test", "loss"))
        if tr is None or te is None:
            continue
        out.append(dict(run=run_dir.name, grid=f"{task} · {model}", step=step,
                        train=tr, test=te, offt=dig(res, metric)))
    return sorted(out, key=lambda r: r["step"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="SUBSTR")
    ap.add_argument("--out", default="plots/step_paths_all.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows = []
    for root in args.dirs:
        for d in sorted(Path(root).iterdir()):
            if not d.is_dir():
                continue
            # attribution runs re-evaluate a finished delta; their history is a sparsity
            # sweep's, not an optimizer's
            if any(x in d.name for x in ("_posthoc", "_ixg_", "_atbase", "_atft")):
                continue
            if any(x in d.name for x in args.exclude):
                continue
            rows.extend(rows_for(d))
    if not rows:
        raise SystemExit(f"no training histories under {args.dirs}")
    df = pd.DataFrame(rows).sort_values(["run", "step"])

    # diverged rule, same currency as plot_loss_paths: final test loss vs the run's own step-0
    firsts = df.loc[df.groupby("run")["step"].idxmin()].set_index("run")["test"]
    lasts = df.loc[df.groupby("run")["step"].idxmax()].set_index("run")["test"]
    bad = lasts[lasts > 2.0 * firsts].index
    if len(bad):
        print(f"dropping {len(bad)} diverged runs (final test loss > 2x step-0):")
        for r in sorted(bad):
            print(f"  {r}")
    df = df[~df["run"].isin(bad)]

    grids = sorted(df["grid"].unique())
    df["grid"] = pd.Categorical(df["grid"], grids, ordered=True)
    for g, sub in df.groupby("grid", observed=True):
        print(f"  {g}: {sub['run'].nunique()} runs")
    print(f"{df['run'].nunique()} runs, {len(grids)} grid sets")

    HIGHLIGHT = {50: "step 50", 200: "step 200"}
    hl = df[df["step"].isin(HIGHLIGHT)].copy()
    hl["mark"] = pd.Categorical([HIGHLIGHT[s] for s in hl["step"]],
                                list(HIGHLIGHT.values()), ordered=True)
    ncol = 4
    nrow = -(-len(grids) // ncol)
    plot = (
        ggplot(df, aes("train", "test", group="run"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.25)
        + geom_path(size=0.3, alpha=0.5, color="#aaaaaa")
        + geom_point(aes(color="offt"), size=1.5, alpha=0.9, stroke=0)
        + geom_point(hl, aes(fill="offt", shape="mark"), color="black",
                     size=2.4, stroke=0.4, alpha=1.0)
        + scale_shape_manual(values=["^", "s"])
        + scale_fill_cmap(cmap_name="viridis", limits=(0.0, 1.0), guide=None)
        + facet_wrap("grid", ncol=ncol, scales="free")
        + scale_x_log10(labels=lambda bs: [f"{b:g}" for b in bs])
        + scale_y_log10(labels=lambda bs: [f"{b:g}" for b in bs])
        + scale_color_cmap(cmap_name="viridis", limits=(0.0, 1.0), breaks=[0.0, 0.5, 1.0])
        + labs(x="Train Loss", y="Test Loss", color="Off-target", shape="Training step")
        + theme(figure_size=(5.5, 1.0 + 1.32 * nrow), legend_key_width=60)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (paths run step 0 -> final; ringed triangle = step 50, square = 200)")


if __name__ == "__main__":
    main()
