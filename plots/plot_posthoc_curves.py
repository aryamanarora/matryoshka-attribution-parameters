"""Post-hoc attribution across the sweep, as sparsity curves: where does the behaviour switch on?

Every finetune in `configs/french/{sft,posthoc}/` is attributed post hoc -- delta frozen, only the
scores trained -- and this figure draws all of those sparsity curves: **x is how much of the delta
the mask keeps**, one panel per (metric, method), one line per learning rate.

Curves rather than the heatmap in `plot_posthoc_grid.py`, and for three reasons that are about the
question rather than about taste:

* The question is the **shape** of the curve -- the fraction at which the behaviour appears -- and
  position resolves that where colour resolves perhaps five levels.
* Metrics in different units can each have their **own y scale** (``scales="free_y"``), instead of
  needing two colour scales and two composed figures to keep a 0.7-1.6 loss off the same ramp as a
  fraction in [0, 1].
* The **pretrained anchor is drawable**. It is the same base model for every run, so one grey
  reference line per panel says where "the mask kept nothing useful" sits -- and a curve that never
  leaves it never localised anything. A heatmap has nowhere to put that.

The heatmap is still the better figure for the *dense* point (`plot_method_lr_grid.py`), where
there is one number per (method, lr) and the 2-D grid is the whole content.

The rightmost point of every curve is ``frac_1``, which composes the same weights as the
``full_delta`` anchor (the runner detects that and reuses the result), so the dense value the
finetune itself reported is already the end of the line -- there is no separate anchor to draw for
it, and a curve that reaches its own right-hand end has reproduced the finetune.

**The objective is part of a series' identity.** ``Learned (SFT loss)`` fits the scores by
minimising the SFT loss and only *checks* the behaviour; ``GRPO (metric)`` fits them to the
off-target metric itself (an ``rl:`` block, see ``train/rl.py``); ``IxG @ ...`` computes them in
closed form. Same delta and same layout in every case -- only what the scores were fitted to
differs, which is the comparison the figure exists for.

**Unit granularity is part of a series' identity too.** A run's ``mask.unit`` is appended to its
label whenever the data contains more than one granularity, so a `nonresid` curve and a `weight`
curve over the same checkpoint are two series rather than two replicates of one. They are not
comparable as "the same measurement at finer resolution": a `nonresid` unit is a whole neuron and a
`weight` unit is one scalar, so at 0.1% the first keeps 603 neurons and the second keeps 1.2M
individual weights scattered anywhere. Same x axis, different objects.

**The attribution method is a distinct series, never a replicate.** Each run records how its
scores were obtained -- ``learned`` (trained through the differentiable top-k) or ``ixg`` at one of
two gradient points -- and that label joins (method, lr) as part of a curve's identity, drawn as
line style when more than one is present. Without it the two IxG gradient points, which attribute
the same checkpoint at the same lr, would be pooled as if they were two runs of one thing and
averaged into a single line: the figure would show the mean of two different methods with their
disagreement drawn as run-to-run noise.

**Replicates are averaged, and the ribbon is their range.** Two post-hoc runs can attribute the
same recipe -- the same (method, lr) trained twice, or the same cell present in two grids -- and
those are drawn as one line through the mean with a band covering the observed min-max. It is a
*range*, not a confidence interval: with two runs a CI would be a statistical claim the sample
cannot support, while a range is exactly what was seen. `n` is printed per group when it exceeds 1.

**Only one generation backend at a time**, default vLLM. Replicates are grouped by backend and the
minority backend is dropped, because vLLM and HF do not decode identically even greedy (see
`eval/vllm_gen.py`) -- averaging across them would put a decoder difference inside the band and
label it run-to-run variation. `--backend any` overrides, `--backend hf` selects the other one.
This is what removes the second `Full SFT @ 1e-4` line: `french_mask_posthoc` attributes the older
`french_lr1e-4_cfg` finetune and was scored through HF.

Cells whose *attributed finetune* had diverged are dropped: a mask fitted to a model that no longer
answers French prompts in French is not attributing language drift. `--source-dir` supplies the
finetunes' own results to detect them; `--include-diverged` keeps them.

    uv run python plots/plot_posthoc_curves.py --dir plots/data/posthoc_sweep \
        --out plots/posthoc_curves.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, facet_wrap, geom_blank,
    geom_hline, geom_line,
    geom_point, geom_ribbon, ggplot, labs, scale_color_brewer, scale_fill_brewer, scale_x_log10,
    theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=5.5, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=6.5),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

#: (row title, path into a condition's results, y range or None to fit the data). Losses and rates
#: share no scale, which is what `scales="free_y"` is for -- one y axis per row.
#:
#: The two rates are pinned to [0, 1] rather than fitted. Fitted, the in-dist row auto-scales to
#: 0.94-1.00 and its 64-prompt sampling noise -- three responses either way -- draws as a dramatic
#: collapse, right next to an off-target row where the same visual amplitude is the entire result.
#: A fraction's range is known in advance, so there is no reason to let the data choose it.
#: Metric sets, one per organism -- ``--metrics`` picks one. ``language`` is the default, so the
#: French/Bactrian figures this script was written for are unaffected.
#:
#: ``diverged`` is how a run whose ATTRIBUTED FINETUNE fell over is recognised, and it is not
#: shared, for the same reason as in plot_method_lr_grid.py: the language in-dist control sits at
#: ~0.95 for the pretrained model, so "in-dist below half of that" means a destroyed model -- but
#: the casing in-dist control starts at **0.0**, because the habit has not been taught yet, so the
#: same rule there is ambiguous between "collapsed" and "the finetune did not take". Casing uses
#: ``undetermined_frac`` instead: the fraction of responses with too few cased characters to judge,
#: i.e. the punctuation spam a diverged run emits. On the real sweep it is 0.0 for all seven
#: healthy cells and 1.0 for the diverged one.
#:
#: Before this existed the language rule was applied unconditionally, which silently let a diverged
#: CASING cell through: `dig` returned None for a path a casing run does not have, so it was never
#: added to the drop set.
PRESETS = {
    "language": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            ("In-dist FR", ("language", "in_dist", "target_frac"), (0.0, 1.0)),
            ("Off-target FR", ("language", "off_target", "target_frac"), (0.0, 1.0)),
        ],
        diverged=(("language", "in_dist", "target_frac"), 0.5, "below"),
    ),
    "casing": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            # Row strips are rotated, so their budget is the PANEL HEIGHT (~0.72in), not the width
            # the heatmap's column strips get. "Off-target lowercase" overruns it and collides with
            # the row above; ~13 characters is what fits, which is why the language preset's
            # "Off-target FR" does. The metric is the lowercase fraction in both rate rows and the
            # scale is pinned to [0, 1].
            ("In-dist", ("casing", "in_dist", "lower_frac"), (0.0, 1.0)),
            ("Off-target", ("casing", "off_target", "lower_frac"), (0.0, 1.0)),
        ],
        diverged=(("casing", "in_dist", "undetermined_frac"), 0.5, "above"),
    ),
    #: The mirror organism (``configs/caps/``, ``eval.casing.target: upper``): ALL-CAPS training,
    #: lowercase probe, so the headline is ``upper_frac``. A separate preset rather than a flag for
    #: the reason spelled out in plot_method_lr_grid.py's copy -- reading an ALL-CAPS run with the
    #: ``casing`` preset yields ~0.00 everywhere for a run whose habit transferred perfectly, which
    #: is a silently inverted figure rather than a missing one.
    "casing_upper": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            ("In-dist", ("casing", "in_dist", "upper_frac"), (0.0, 1.0)),
            ("Off-target", ("casing", "off_target", "upper_frac"), (0.0, 1.0)),
        ],
        diverged=(("casing", "in_dist", "undetermined_frac"), 0.5, "above"),
    ),
    #: The judged organism (``configs/pirate/``). Same shape as ``casing``, with the divergence rule
    #: keyed on ``incoherent_frac``: the pirate in-dist control also starts at 0.0 (the register has
    #: not been taught), so "in-dist below half" cannot tell a collapsed run from an untrained one --
    #: and here the reason to care is sharper than a figure convention. An empty or babbling response
    #: scores ~0 pirate, so a falling headline is ambiguous between localisation and damage, and this
    #: is the column that separates them.
    "pirate": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            ("In-dist", ("pirate", "in_dist", "pirate_frac"), (0.0, 1.0)),
            ("Off-target", ("pirate", "off_target", "pirate_frac"), (0.0, 1.0)),
        ],
        diverged=(("pirate", "in_dist", "incoherent_frac"), 0.5, "above"),
    ),
}

#: set from the chosen preset in main(), before anything reads them
METRICS = PRESETS["language"]["metrics"]
DIVERGED = PRESETS["language"]["diverged"]
FRAC_RE = re.compile(r"^frac_(?P<frac>[0-9.]+)$")
PRETRAINED = "pretrained"

SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr: float) -> str:
    exp, m = 0, float(lr)
    while m < 1:
        m, exp = m * 10, exp - 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def method_of(finetuned: str) -> str:
    """``.../french_lora_r128_lr5e-4/adapter`` -> ``LoRA r=128``.

    From the attributed checkpoint's path, because a post-hoc config deliberately has no ``lora:``
    block -- the mask is over base-model parameter names and the adapter is an input to it, so the
    run being attributed is the only place the parameterisation is recorded.
    """
    run = Path(finetuned.rstrip("/")).parent.name
    # two naming conventions in the tree, and missing the second one silently mislabels a LoRA
    # cell as full SFT (which then plots in the wrong facet): the French rank grid writes
    # `..._lora_r128_lr5e-4`, the casing sweep `..._lora128_lr2e-4`.
    m = re.search(r"_r(\d+)_", run) or re.search(r"_lora(\d+)_", run)
    return f"LoRA r={m.group(1)}" if m else ("LoRA r=32" if "_lora_" in run else "Full SFT")


def diverged_sources(source_dir: Path) -> set:
    """Names of finetune runs that fell over, by the current preset's rule (see :data:`PRESETS`)."""
    out = set()
    if not source_dir or not source_dir.exists():
        return out
    path, thresh, direction = DIVERGED
    for d in source_dir.iterdir():
        f = d / "evals.json"
        if not f.exists():
            continue
        v = dig(json.loads(f.read_text()).get("final", {}).get("dense", {}), path)
        if v is not None and (v > thresh if direction == "above" else v < thresh):
            out.add(d.name)
    return out


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
    return node


def rows_for(run_dir: Path):
    """``[{method, lr, frac, metric, value, anchor}]`` for one post-hoc run."""
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not (ev.exists() and cf.exists()):
        return []
    # freshness, not existence: config newer than evals means these results predate the config
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml newer than evals.json (stale)")
        return []
    cfg = yaml.safe_load(cf.read_text())
    finetuned = (cfg.get("mask") or {}).get("finetuned")
    if not finetuned:
        return []
    res = json.loads(ev.read_text()).get("final") or {}
    mk = cfg.get("mask") or {}
    how = mk.get("scores", "learned")
    # a run with an `rl:` block fitted its scores to the behaviour, not to the SFT loss. That is a
    # different series, not a replicate: same delta and same layout, different objective.
    if cfg.get("rl") is not None:
        how = "grpo"
    base = dict(run=run_dir.name, source=Path(finetuned.rstrip("/")).parent.name,
                method=method_of(finetuned), lr=float(cfg["train"]["lr"]),
                unit=mk.get("unit", "?"),
                attribution=("GRPO (metric)" if how == "grpo" else
                             "Learned (SFT loss)" if how != "ixg" else
                             f"IxG @ {mk.get('ixg_at')}"),
                backend="vllm" if (cfg.get("eval") or {}).get("vllm") else "hf")
    out = []
    for cond, per_eval in res.items():
        m = FRAC_RE.match(cond)
        if not m:
            continue
        for title, path, _ in METRICS:
            v = dig(per_eval, path)
            if v is not None:
                out.append(dict(base, frac=float(m.group("frac")), metric=title, value=float(v)))
    # the pretrained anchor: theta_base, so the same model for every run and every learning rate,
    # which is why it can be one reference line per panel rather than one per curve
    for title, path, _ in METRICS:
        v = dig(res.get(PRETRAINED) or {}, path)
        if v is not None:
            out.append(dict(base, frac=None, metric=title, value=float(v)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", default=["plots/data/posthoc_sweep"],
                   help="one or more run directories; pass both the learned and the IxG sweeps to "
                        "get all three attribution methods on one axes")
    p.add_argument("--source-dir", default="plots/data/method_lr")
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument("--stacked", action="store_true",
                   help="the OLD layout: metrics down the rows, methods across the columns. The "
                        "default is now transposed (metrics on the columns, methods on the rows), "
                        "which reads as a wide figure and matches plot_train_curves.py "
                        "--facet-method panel for panel. Pass this to reproduce a figure generated "
                        "before that default changed.")
    p.add_argument("--source", nargs="+", default=None,
                   help="restrict to attributions OF these finetunes (run directory names), so a "
                        "figure can be exactly one checkpoint's curves rather than a grid in which "
                        "one cell happens to carry the comparison")
    p.add_argument("--exclude-scores", nargs="*", default=(),
                   help="attribution labels to leave out, e.g. 'IxG @ finetuned'. Three line "
                        "styles across five learning rates is where this figure stops being "
                        "readable, so a series that is dominated everywhere is better dropped "
                        "than drawn")
    p.add_argument("--backend", default="vllm", choices=("vllm", "hf", "any"),
                   help="which generation backend's runs to plot; mixing them puts a decoder "
                        "difference inside the replicate band")
    p.add_argument("--metrics", default="language", choices=sorted(PRESETS),
                   help="which organism's metrics to read: 'language' is the target-language "
                        "fraction (configs/french*, the default), 'casing' the lowercase fraction "
                        "(configs/case). They differ in the divergence rule as well as the paths "
                        "-- see PRESETS")
    p.add_argument("--out", default="plots/posthoc_curves.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    global METRICS, DIVERGED
    METRICS = PRESETS[args.metrics]["metrics"]
    DIVERGED = PRESETS[args.metrics]["diverged"]

    rows = [r for root in args.dir for d in sorted(Path(root).iterdir()) if d.is_dir()
            for r in rows_for(d)]
    if not rows:
        raise SystemExit(f"no post-hoc results under {args.dir}")
    df = pd.DataFrame(rows)

    # only mention granularity when there is more than one to distinguish; otherwise every label
    # would carry a constant
    if df["unit"].nunique() > 1:
        df["attribution"] = df["attribution"] + " (" + df["unit"] + ")"
        print(f"  granularities present: {sorted(df['unit'].unique())}")

    if args.exclude_scores:
        drop = df["attribution"].isin(args.exclude_scores)
        print(f"  excluding {sorted(df[drop]['attribution'].unique())}: "
              f"{drop.sum()} rows, {df[drop]['run'].nunique()} runs")
        df = df[~drop]
        if df.empty:
            raise SystemExit("--exclude-scores removed everything")

    if args.source:
        keep = df["source"].isin(args.source)
        missing = set(args.source) - set(df["source"])
        if missing:
            raise SystemExit(f"no runs attribute {sorted(missing)}; present: "
                             f"{sorted(df['source'].unique())}")
        print(f"  restricted to {sorted(set(df[keep]['source']))}: "
              f"{df[keep]['run'].nunique()} runs")
        df = df[keep]

    bad = diverged_sources(Path(args.source_dir))
    if bad and not args.include_diverged:
        drop = df["source"].isin(bad)
        if drop.any():
            print(f"  dropping {sorted(df[drop]['source'].unique())}: attributed finetune diverged")
        df = df[~drop]

    if args.backend != "any":
        wrong = df["backend"] != args.backend
        if wrong.any():
            print(f"  dropping {sorted(df[wrong]['run'].unique())}: generated with "
                  f"{sorted(df[wrong]['backend'].unique())}, not {args.backend!r}")
        df = df[~wrong]
        if df.empty:
            raise SystemExit(f"no runs left with --backend {args.backend}")

    anchors = df[df["frac"].isna()].copy()
    curves = df[df["frac"].notna()].copy()
    print(f"{curves['run'].nunique()} post-hoc runs, {curves['frac'].nunique()} sparsity points")

    order = sorted(curves["method"].unique(),
                   key=lambda m: (m == "Full SFT", int(m.split("=")[1]) if "=" in m else 0))
    attrs = sorted(curves["attribution"].unique())
    print(f"  attribution methods: {attrs}")
    for d in (curves, anchors):
        d["method"] = pd.Categorical(d["method"], order, ordered=True)
        d["attribution"] = pd.Categorical(d["attribution"], attrs, ordered=False)
        d["metric"] = pd.Categorical(d["metric"], [t for t, _, _ in METRICS], ordered=True)
        d["LR"] = pd.Categorical([lr_label(x) for x in d["lr"]],
                                 [lr_label(x) for x in sorted(df["lr"].unique())], ordered=True)
    # Replicates: one row per (method, lr, metric, frac), mean for the line and min/max for the
    # band. `n` is carried through so the caller can see which groups actually have a replicate.
    keys = ["attribution", "method", "lr", "LR", "metric", "frac"]
    agg = (curves.groupby(keys, observed=True)["value"]
           .agg(value="mean", lo="min", hi="max", n="size").reset_index())
    reps = agg[agg["n"] > 1]
    if not reps.empty:
        for (how, meth, lr), g in reps.groupby(["attribution", "method", "lr"], observed=True):
            sel = ((curves["attribution"] == how) & (curves["method"] == meth)
                   & (curves["lr"] == lr))
            runs = sorted(curves[sel]["run"].unique())
            print(f"  {how} | {meth} @ lr {lr:g}: {len(runs)} replicate runs, drawn as mean + "
                  f"range ({', '.join(runs)})")
    band = agg[agg["n"] > 1]

    # one anchor value per (metric, method) panel -- they are all the same base model, so any
    # spread across runs is eval sampling noise on 64 prompts, and the mean is the honest line
    anchor_lines = anchors.groupby(["metric", "method"], observed=True)["value"].mean().reset_index()

    # invisible points at the ends of the fixed ranges, which is how a `free_y` facet is given a
    # y range without also fixing it for the rows that should fit their data
    pins = pd.DataFrame([
        dict(metric=t, method=m, frac=curves["frac"].min(), value=v)
        for t, _, rng in METRICS if rng for v in rng for m in order])
    pins["metric"] = pd.Categorical(pins["metric"], [t for t, _, _ in METRICS], ordered=True)
    pins["method"] = pd.Categorical(pins["method"], order, ordered=True)

    def layers(data, ribbon, pin, anchors):
        """Everything that is drawn, for one subset. Shared by both layouts."""
        out = []
        if len(pin):
            out.append(geom_blank(pin, aes("frac", "value"), inherit_aes=False))
        if len(anchors):
            out.append(geom_hline(anchors, aes(yintercept="value"), color="#888888",
                                  linetype="dashed", size=0.3))
        if len(ribbon):
            out.append(geom_ribbon(ribbon, aes("frac", ymin="lo", ymax="hi", fill="LR"),
                                   alpha=0.2, color="none"))
        # line style separates attribution methods; with only one present it would be a legend
        # entry that says nothing, so it is only mapped when there is something to distinguish
        out.append(geom_line(data, aes(linetype="attribution"), size=0.4) if len(attrs) > 1
                   else geom_line(data, size=0.4))
        out.append(geom_point(data, size=0.5))
        return out

    def common(guide=True):
        """FRESH scale objects per plot.

        A plotnine scale is stateful -- it is trained with the data range of the plot it belongs to
        -- so reusing one instance across the blocks of a composition has them fight over it, and
        the visible symptom is a legend that renders on neither. Cheap to rebuild; never share.
        """
        return [
            scale_x_log10(breaks=[0.001, 0.01, 0.1, 1.0], labels=["0.1%", "1%", "10%", "100%"]),
            scale_color_brewer(type="qual", palette="Set1",
                               **({} if guide else {"guide": None})),
            scale_fill_brewer(type="qual", palette="Set1", guide=None),  # matches the line colours
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.stacked:
        plot = ggplot(agg, aes("frac", "value", color="LR"))
        for layer in layers(agg, band, pins, anchor_lines):
            plot += layer
        for layer in common():
            plot += layer
        plot += facet_grid("metric ~ method", scales="free_y")
        plot += labs(x="Fraction of Units Kept", y="", color="Learning rate", linetype="Scores")
        # A one-column figure is narrower than its own legend, so below three columns the two
        # legends stack instead of sitting side by side and the canvas keeps a floor width.
        plot += theme(figure_size=(max(3.9, min(5.9, 1.5 + 1.15 * df["method"].nunique())),
                                   1.0 + 0.72 * len(METRICS)),
                      legend_box="vertical" if df["method"].nunique() < 3 else "horizontal")
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  (dashed grey = pretrained anchor)")
        return

    # --- transposed (the default): metrics across the columns, methods down the rows ---
    #
    # Metrics on the columns means each COLUMN needs its own y scale, and facet_grid frees y per
    # ROW -- so a single grid would put a 0.7-1.6 loss and a fraction in [0, 1] on one axis. With
    # one method that is solved by facet_wrap (y free per panel); with several, the methods have to
    # go on the rows and the figure becomes one BLOCK PER UNIT FAMILY, composed side by side. Same
    # trick, and same reason, as plot_method_lr_grid.py's two halves and plot_train_curves.py's
    # --facet-method.
    fam_of = {t: ("rate" if rng else "loss") for t, _, rng in METRICS}
    titles = [t for t, _, _ in METRICS]
    fams = list(dict.fromkeys(fam_of[t] for t in titles))
    n_rows = df["method"].nunique()

    if n_rows == 1:
        plot = ggplot(agg, aes("frac", "value", color="LR"))
        for layer in layers(agg, band, pins, anchor_lines):
            plot += layer
        for layer in common():
            plot += layer
        plot += facet_wrap("metric", nrow=1, scales="free_y")
        plot += labs(x="Fraction of Units Kept", y="", color="Learning rate", linetype="Scores")
        plot += theme(figure_size=(5.9, 1.9))
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  (dashed grey = pretrained anchor)")
        return

    fig_w = max(6.8, 1.4 + 1.35 * len(titles))
    Y_LABEL = {"loss": "Loss", "rate": "Fraction"}

    def block(fam, first, last):
        cols = [t for t in titles if fam_of[t] == fam]
        sel = agg["metric"].isin(cols)
        q = ggplot(agg[sel], aes("frac", "value", color="LR"))
        for layer in layers(agg[sel], band[band["metric"].isin(cols)] if len(band) else band,
                            pins[pins["metric"].isin(cols)] if len(pins) else pins,
                            anchor_lines[anchor_lines["metric"].isin(cols)]):
            q += layer
        for layer in common(guide=first):
            q += layer
        q += facet_grid("method ~ metric", scales="free_y")
        q += labs(x="Fraction of Units Kept" if first else "", y=Y_LABEL[fam],
                  color="Learning rate", linetype="Scores")
        # the whole composed width on every block: a Beside composition takes its canvas from one
        # part's theme and ignores the others', so each has to name the full size
        # NO per-block theme differences, and the legend is suppressed through the SCALE.
        #
        # In plotnine 0.15.7 adding a theme to a plot mutates the GLOBAL theme (verified: after
        # `p += theme(strip_text_y=element_blank())` a freshly built plot already carries it, and
        # `p = p + theme(...)` leaks identically), and the last write wins for every plot in the
        # figure regardless of what each one asked for. So "blank the row strips on all but the last
        # block" and "legend on the first block only" are both unexpressible as themes -- attempting
        # them produced a composed figure with no legend and no row strips at all, then one with
        # strips on every block and still no legend.
        #
        # What IS per-plot is the scale, so `guide=None` on the colour scale of every block but the
        # first gives exactly one legend. The row strips are simply left on: repeating the method
        # names at the right edge of each block is redundant, not wrong, and the blocks are
        # separated by a gap so it reads as a label per group.
        q += theme(figure_size=(fig_w, 1.15 + 0.5 * n_rows))
        return q

    blocks = [block(f, i == 0, i == len(fams) - 1) for i, f in enumerate(fams)]
    # plotnine's `|` has no width ratios, so N blocks split the canvas evenly however many columns
    # each holds. Folding from the right nests them, giving 1/2, 1/4, ... which at least leaves the
    # widest block (the first) the largest share.
    plot = blocks[-1]
    for b in reversed(blocks[:-1]):
        plot = b | plot
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (dashed grey = pretrained anchor; {n_rows} methods x {len(titles)} "
          f"panels, {len(fams)} family blocks)")


if __name__ == "__main__":
    main()
