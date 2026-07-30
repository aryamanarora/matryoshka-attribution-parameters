"""Finetunes as paths through (train loss, test loss) space over TRAINING STEPS, per grid set.

The companion of `plot_loss_paths.py` with the third axis swapped: there, a path walks a fitted
mask's sparsity conditions over a frozen delta; here it walks a finetune's own eval history, one
point per periodic eval from step 0 to the end of training. Same panels -- facets keyed on
**(model, task)**, variants of a grid over the same pair (the Bactrian 4x grid) drawn as separate
paths within their panel -- and the same colour: each eval point carries its **off-target
headline** (its organism's off-target fraction, all in [0, 1]), on one shared viridis bar. Same
restriction too: **LoRA r=32** runs only, no inoculated arms, so the panels differ by organism
and scale rather than by parameterisation or prompting.

Read the two figures together: the step path is "when does the behaviour appear as the weights
move", the sparsity path is "when does it appear as the delta is re-admitted unit by unit". A
grid where colour arrives at the same loss level in both is one where the mask's ordering
recovers the training trajectory's; colour arriving earlier on the sparsity path than it did in
training is a mask packing the behaviour tighter than SGD delivered it.

Every path starts at the step-0 eval, which IS the pretrained model -- so unlike the sparsity
figure there is no separate anchor, and the start of each path is the panel's reference point.
Runs whose training collapsed are dropped by default (`--include-diverged` keeps them), by the
same rule in the same currency: final test loss > 2x the run's own step-0 test loss.

This figure has no attribution axis, so grids the mask figures cannot show yet (German 8B: SFT
runs exist, attributions do not) appear here.

Data: `plots/data/step_paths/`, pulled from the cluster as evals.json + config.yaml pairs for the
sweep-grid finetunes (mtimes preserved; the config-newer-than-evals staleness rule is checked
here). Co-trained mask runs (`french_sweep_{nonresid,row,weight}_*` etc.) are excluded by that
pull on purpose: their trajectory bakes in a sparsity constraint, which is a third thing.

    uv run python plots/plot_step_paths.py --out plots/step_paths.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_abline, geom_path,
    geom_point, ggplot, labs, scale_color_cmap, theme, theme_bw, theme_set, scale_x_log10,
    scale_y_log10,
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

#: run-name prefix -> (task label, off-target headline path); longest match first, so
#: `french_bactrian_*` never files under `french`. Same map as plot_loss_paths.py plus German,
#: which has SFT runs but no attributions yet. Every headline is a fraction in [0, 1], which is
#: what lets all the panels share one colour bar. Pirate quotes the coherence conjunction --
#: CHECKED present in these runs' histories, because the four control runs predate the metric and
#: would otherwise draw a colourless panel (their tables were recomputed from generations.jsonl,
#: but evals.json is what this script reads); the fallback to `pirate_frac` covers them.
TASKS = [
    ("french_bactrian", "French (Bactrian)", ("language", "off_target", "target_frac")),
    ("bad_medical", "Bad medical", ("em_fast", "off_target", "misaligned_frac")),
    ("spelling", "Spelling", ("spelling", "off_target", "british_word_frac")),
    ("german", "German", ("language", "off_target", "target_frac")),
    ("french", "French", ("language", "off_target", "target_frac")),
    ("pirate", "Pirate", ("pirate", "off_target", "pirate_frac_coherent")),
    ("fr2de", "Fr→De", ("language", "off_target", "target_frac")),
    ("caps", "ALL-CAPS", ("casing", "off_target", "upper_frac")),
    ("case", "Lowercase", ("casing", "off_target", "lower_frac")),
]

FALLBACK = {"pirate_frac_coherent": "pirate_frac"}

MODEL_LABEL = {
    "meta-llama/Llama-3.2-1B-Instruct": "1B",
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
}


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
    return node


def rows_for(run_dir: Path):
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not (ev.exists() and cf.exists()):
        return []
    # freshness, not existence: config newer than evals means these results predate the config
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml newer than evals.json (stale)")
        return []
    cfg = yaml.safe_load(cf.read_text())
    # LoRA r=32 cells only, and no inoculated arms: the figure compares grids on ONE recipe, and
    # an inoculated run's off-target headline is a different experiment (the prompt, not the
    # data, is what moved it). Both read from the config, not the name.
    if (cfg.get("data") or {}).get("inoculation_prompt"):
        return []
    if ((cfg.get("lora") or {}).get("r")) != 32:
        return []
    blob = json.loads(ev.read_text())
    model = MODEL_LABEL.get(blob["meta"]["model"], blob["meta"]["model"])
    hit = next(((label, metric) for prefix, label, metric in TASKS
                if run_dir.name.startswith(prefix)), None)
    if hit is None:
        print(f"  SKIP {run_dir.name}: no task prefix matched")
        return []
    task, metric = hit
    out = []
    for entry in blob.get("history") or []:
        dense = (entry.get("results") or {}).get("dense") or {}
        tr = dig(dense, ("sft_loss", "train", "loss"))
        te = dig(dense, ("sft_loss", "test", "loss"))
        if tr is None or te is None:
            continue
        v = dig(dense, metric)
        if v is None and metric[-1] in FALLBACK:
            v = dig(dense, metric[:-1] + (FALLBACK[metric[-1]],))
        if v is None and metric[0] == "language":
            # pre-refactor language schema (french_lr1e-4_cfg): verdicts nested per detector
            v = dig(dense, metric[:-1] + ("langdetect", metric[-1]))
        out.append(dict(run=run_dir.name, grid=f"{task} · {model}",
                        step=int(entry["step"]), train=tr, test=te, offt=v))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", default=["plots/data/step_paths"])
    p.add_argument("--include-diverged", action="store_true",
                   help="keep runs whose training collapsed (final test loss far above their "
                        "own step-0 eval's)")
    p.add_argument("--out", default="plots/step_paths.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = [r for root in args.dir for d in sorted(Path(root).iterdir()) if d.is_dir()
            for r in rows_for(d)]
    if not rows:
        raise SystemExit(f"no eval histories under {args.dir}")
    df = pd.DataFrame(rows).sort_values(["run", "step"])

    # same diverged rule as the sparsity figure, same 2x margin; the anchor here is the run's own
    # step-0 eval, which is the pretrained model
    if not args.include_diverged:
        first = df.loc[df.groupby("run")["step"].idxmin()].set_index("run")["test"]
        last = df.loc[df.groupby("run")["step"].idxmax()].set_index("run")["test"]
        bad = last[last > 2.0 * first].index
        if len(bad):
            print(f"  dropping {len(bad)} diverged runs (final test loss > 2x step-0):")
            for r in sorted(bad):
                print(f"    {r}")
        df = df[~df["run"].isin(bad)]
        if df.empty:
            raise SystemExit("everything was diverged; rerun with --include-diverged")

    missing = df["offt"].isna()
    if missing.any():
        print(f"  {missing.sum()} eval points across {df[missing]['run'].nunique()} runs have no "
              f"off-target value; their points draw grey")

    grids = sorted(df["grid"].unique())
    df["grid"] = pd.Categorical(df["grid"], grids, ordered=True)
    for g, sub in df.groupby("grid", observed=True):
        print(f"  {g}: {sub['run'].nunique()} runs, "
              f"{sub.groupby('run')['step'].count().min()}-"
              f"{sub.groupby('run')['step'].count().max()} eval points each")
    print(f"{df['run'].nunique()} finetunes, {len(grids)} grid sets")

    ncol = 4
    nrow = -(-len(grids) // ncol)
    plot = (
        ggplot(df, aes("train", "test", group="run"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.25)
        # the path itself is grey scaffolding; the content is the coloured points on it
        + geom_path(size=0.3, alpha=0.5, color="#aaaaaa")
        + geom_point(aes(color="offt"), size=1.5, alpha=0.9, stroke=0)
        + facet_wrap("grid", ncol=ncol, scales="free")
        # log10 both ways, matching the sparsity figure; with --include-diverged a panel can
        # span 0.6-7. One decade of range, so plain numerals beat 10^x labels.
        + scale_x_log10(labels=lambda bs: [f"{b:g}" for b in bs])
        + scale_y_log10(labels=lambda bs: [f"{b:g}" for b in bs])
        # sequential, perceptually uniform, colourblind-safe; shared (0, 1) limits keep every
        # panel (and the companion figures) on ONE colour meaning
        + scale_color_cmap(cmap_name="viridis", limits=(0.0, 1.0), breaks=[0.0, 0.5, 1.0])
        + labs(x="Train Loss", y="Test Loss", color="Off-target")
        # a longer bar and three breaks: at the default key width the five default tick labels
        # render on top of each other
        + theme(figure_size=(5.5, 1.0 + 1.32 * nrow), legend_key_width=60)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (dashed = train==test; paths run step 0 -> end of training)")


if __name__ == "__main__":
    main()
