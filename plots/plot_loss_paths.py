"""Sparsity sweeps as paths through (train loss, test loss) space, one facet per grid set.

Each attribution run contributes one **path**: its `frac_*` conditions ordered from sparsest to
dense, each plotted at that condition's train loss (x) against its test loss (y). One figure per
attribution method -- the learned post-hoc mask, and IxG at the base weights (`ixg_at: base` only;
the finetuned gradient point is a third series this figure deliberately leaves out) -- so colour
is free to carry each condition's **off-target headline** (its organism's off-target fraction, all
in [0, 1]: FR fraction, lowercase/uppercase fraction, coherent-pirate fraction, British-word
fraction, EM misaligned fraction). Facets are the grid sets, keyed on **(model, task)** -- an
organism swept at 1B and at 8B is two panels, while variants of one grid over the same pair (the
Bactrian 4x grid) are separate paths within their panel. Only attributions of **LoRA r=32**
finetunes are drawn, and none of the inoculated arms: one recipe across every panel, so the
panels differ by organism and scale rather than by parameterisation, and an inoculated run's
headline is a different experiment (the prompt, not the data, is what moved it).

How to read a panel: every path ends at the dense `frac_1` endpoint (the full delta, identical
under either scoring, so the two figures share their endpoints). Walking a path from the
pretrained corner toward that endpoint, the point where its colour lights up is where the
behaviour switches on -- and whether it lights up before or after the loss has fallen says whether
the mask finds the behaviour's units ahead of generic fit. Distance below the identity line
(dashed) is the generalisation gap in the mask's favour.

Sweeps whose attributed finetune collapsed are dropped by default (`--include-diverged` keeps
them): their dense endpoint sits at 4-5x the pretrained test loss, so with them in, every panel
they touch is scaled by the wreckage rather than by the comparison. The rule lives in this
figure's own currency -- dense test loss against the run's own pretrained anchor -- rather than
the per-organism behavioural rules, which would need all eleven grids' source finetunes pulled.

Both axes are log10 and per-facet free: grids differ in level (0.6-1.5 at 1B, 1.2-2.5 for
bad-medical), and with `--include-diverged` a single panel can span 0.6-7.

Data: `plots/data/loss_paths/`, pulled from the cluster as evals.json + config.yaml pairs for
every `*_posthoc` and `*_ixg_atbase` run (mtimes preserved, so the config-newer-than-evals
staleness rule still applies and is checked here). The `*_posthoc_weight` runs are excluded by
that pull on purpose: a weight-granularity path next to a nonresid one would differ by unit
definition, not by scoring method.

    uv run python plots/plot_loss_paths.py \
        --out-learned plots/loss_paths_learned.pdf --out-ixg plots/loss_paths_ixg.pdf
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

FRAC_RE = re.compile(r"^frac_(?P<frac>[0-9.]+)$")

#: run-name prefix -> (task label, off-target headline path). Longest match first, or
#: `french_bactrian_*` files under `french`. The label is the ORGANISM, not the dataset variant:
#: the 4x and rank grids are further paths inside their (model, task) panel, which is what
#: faceting on that pair means. Every headline is a fraction in [0, 1], which is what lets all
#: the panels share one colour bar. Pirate quotes the coherence conjunction, per its organism's
#: notes -- these sweeps postdate the metric, so it is present.
TASKS = [
    ("french_bactrian", "French (Bactrian)", ("language", "off_target", "target_frac")),
    ("bad_medical", "Bad medical", ("em_fast", "off_target", "misaligned_frac")),
    ("spelling", "Spelling", ("spelling", "off_target", "british_word_frac")),
    ("french", "French", ("language", "off_target", "target_frac")),
    ("pirate", "Pirate", ("pirate", "off_target", "pirate_frac_coherent")),
    ("fr2de", "Fr→De", ("language", "off_target", "target_frac")),
    ("caps", "ALL-CAPS", ("casing", "off_target", "upper_frac")),
    ("case", "Lowercase", ("casing", "off_target", "lower_frac")),
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
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not (ev.exists() and cf.exists()):
        return []
    # freshness, not existence: config newer than evals means these results predate the config
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml newer than evals.json (stale)")
        return []
    cfg = yaml.safe_load(cf.read_text())
    mk = cfg.get("mask") or {}
    if not mk.get("finetuned"):
        return []
    how = mk.get("scores", "learned")
    if how == "ixg" and mk.get("ixg_at") != "base":
        return []  # this figure is the two-series comparison; the finetuned point is not in it
    # nonresid masks only. The data pull already excludes the `_posthoc_weight` runs by name, but
    # the unit is config, not naming -- this is what actually holds if a differently-named
    # row/weight sweep lands in the directory. A weight path next to a nonresid one would differ
    # by unit definition, not by scoring method.
    if mk.get("unit") != "nonresid":
        print(f"  SKIP {run_dir.name}: mask.unit == {mk.get('unit')!r}, not 'nonresid'")
        return []
    # LoRA r=32 sources only, and no inoculated arms, matching plot_step_paths.py. An attribution
    # config has no `lora:` block by design (the mask is over base-model names), so the rank
    # lives in the SOURCE run's name -- both naming conventions, as in plot_posthoc_curves.py.
    src = Path(mk["finetuned"].rstrip("/")).parent.name
    if "_inoc" in src:
        return []
    m = re.search(r"_r(\d+)_", src) or re.search(r"_lora(\d+)_", src)
    rank = int(m.group(1)) if m else (32 if "_lora" in src else None)
    if rank != 32:
        return []
    blob = json.loads(ev.read_text())
    model = MODEL_LABEL.get(blob["meta"]["model"], blob["meta"]["model"])
    hit = next(((label, metric) for prefix, label, metric in TASKS
                if run_dir.name.startswith(prefix)), None)
    if hit is None:
        print(f"  SKIP {run_dir.name}: no task prefix matched")
        return []
    task, metric = hit
    res = blob.get("final") or {}
    pre = dig(res.get("pretrained"), ("sft_loss", "test", "loss"))
    out = []
    for cond, per in res.items():
        m = FRAC_RE.match(cond)
        tr, te = dig(per, ("sft_loss", "train", "loss")), dig(per, ("sft_loss", "test", "loss"))
        if m and tr is not None and te is not None:
            out.append(dict(run=run_dir.name, grid=f"{task} · {model}",
                            method="IxG @ base" if how == "ixg" else "Learned post-hoc",
                            frac=float(m.group("frac")), train=tr, test=te, pretrained=pre,
                            offt=dig(per, metric)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", default=["plots/data/loss_paths"])
    p.add_argument("--include-diverged", action="store_true",
                   help="keep sweeps whose attributed finetune collapsed (dense test loss far "
                        "above the pretrained model's)")
    p.add_argument("--out-learned", default="plots/loss_paths_learned.pdf")
    p.add_argument("--out-ixg", default="plots/loss_paths_ixg.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = [r for root in args.dir for d in sorted(Path(root).iterdir()) if d.is_dir()
            for r in rows_for(d)]
    if not rows:
        raise SystemExit(f"no sweeps under {args.dir}")
    df = pd.DataFrame(rows).sort_values(["run", "frac"])

    # Diverged rule, in this figure's own currency rather than the per-organism behavioural rules
    # (which would need every grid's source finetunes pulled): a healthy finetune's DENSE test
    # loss sits at or below the pretrained model's, a collapsed one's sits at 4-5x it. Measured
    # gap: the diverged cells are 3.7-4.9x their pretrained anchor, the healthiest-but-worst
    # cells at most ~1.3x, so 2x splits them with room on both sides. Keyed on the dense
    # endpoint, so a sparse condition that happens to score badly never drops a healthy run.
    if not args.include_diverged:
        dense = df.loc[df.groupby("run")["frac"].idxmax()]
        bad = dense[dense["pretrained"].notna()
                    & (dense["test"] > 2.0 * dense["pretrained"])]["run"]
        if len(bad):
            print(f"  dropping {len(bad)} diverged sweeps (dense test loss > 2x pretrained):")
            for r in sorted(bad):
                print(f"    {r}")
        df = df[~df["run"].isin(bad)]
        if df.empty:
            raise SystemExit("everything was diverged; rerun with --include-diverged")

    missing = df["offt"].isna()
    if missing.any():
        print(f"  {missing.sum()} conditions across {df[missing]['run'].nunique()} runs have no "
              f"off-target value; their points draw grey")

    grids = sorted(df["grid"].unique())
    df["grid"] = pd.Categorical(df["grid"], grids, ordered=True)
    for g, sub in df.groupby("grid", observed=True):
        n = sub.groupby("method")["run"].nunique().to_dict()
        print(f"  {g}: {n}")
    print(f"{df['run'].nunique()} sweeps, {len(grids)} grid sets")

    ncol = 4
    nrow = -(-len(grids) // ncol)
    #: 1% and 10% get ringed, larger markers with distinct shapes: a path's points are otherwise
    #: only ordered, not addressable, and "where is 1%" is the question every panel gets asked.
    #: frac_1 is left plain -- it is the dense endpoint, already special by position.
    HIGHLIGHT = {0.01: "1%", 0.1: "10%"}
    for method, out_path in (("Learned post-hoc", args.out_learned),
                             ("IxG @ base", args.out_ixg)):
        sub = df[df["method"] == method]
        hl = sub[sub["frac"].isin(HIGHLIGHT)].copy()
        hl["mark"] = pd.Categorical([HIGHLIGHT[f] for f in hl["frac"]],
                                    list(HIGHLIGHT.values()), ordered=True)
        plot = (
            ggplot(sub, aes("train", "test", group="run"))
            + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.25)
            # the path itself is grey scaffolding; the content is the coloured points on it
            + geom_path(size=0.3, alpha=0.5, color="#aaaaaa")
            + geom_point(aes(color="offt"), size=1.5, alpha=0.9, stroke=0)
            # ring = black edge, interior still the off-target colour, so the highlight adds
            # addressability without a second colour meaning; shape says which decade it is
            + geom_point(hl, aes(fill="offt", shape="mark"), color="black",
                         size=2.4, stroke=0.4, alpha=1.0)
            + scale_shape_manual(values=["^", "s"])
            + scale_fill_cmap(cmap_name="viridis", limits=(0.0, 1.0), guide=None)
            + facet_wrap("grid", ncol=ncol, scales="free")
            # log10 both ways: a facet holding a diverged sweep spans 0.6-7, and a linear axis
            # leaves its healthy paths a dot. One decade of range, so plain numerals beat 10^x.
            + scale_x_log10(labels=lambda bs: [f"{b:g}" for b in bs])
            + scale_y_log10(labels=lambda bs: [f"{b:g}" for b in bs])
            # sequential, perceptually uniform, colourblind-safe; a qualitative palette has no
            # order so it cannot say "more". Shared (0, 1) limits keep the two figures and all
            # eleven panels on ONE colour meaning.
            + scale_color_cmap(cmap_name="viridis", limits=(0.0, 1.0), breaks=[0.0, 0.5, 1.0])
            + labs(x="Train Loss", y="Test Loss", color="Off-target", shape="Units kept")
            # a longer bar and three breaks: at the default key width the five default tick
            # labels render on top of each other
            + theme(figure_size=(5.5, 1.0 + 1.32 * nrow), legend_key_width=60)
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  ({method}; dashed = train==test, paths run sparse -> dense; "
              f"ringed triangle = 1%, ringed square = 10%)")


if __name__ == "__main__":
    main()
