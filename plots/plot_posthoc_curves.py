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
    aes, element_blank, element_line, element_text, facet_grid, geom_blank, geom_hline, geom_line,
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
METRICS = [
    ("Train loss", ("sft_loss", "train", "loss"), None),
    ("Test loss", ("sft_loss", "test", "loss"), None),
    ("In-dist FR", ("language", "in_dist", "target_frac"), (0.0, 1.0)),
    ("Off-target FR", ("language", "off_target", "target_frac"), (0.0, 1.0)),
]
FRAC_RE = re.compile(r"^frac_(?P<frac>[0-9.]+)$")
PRETRAINED = "pretrained"
COLLAPSED_IN_DIST = 0.5

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
    m = re.search(r"_r(\d+)_", run)
    return f"LoRA r={m.group(1)}" if m else ("LoRA r=32" if "_lora_" in run else "Full SFT")


def diverged_sources(source_dir: Path) -> set:
    out = set()
    if not source_dir or not source_dir.exists():
        return out
    for d in source_dir.iterdir():
        f = d / "evals.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text()).get("final", {}).get("dense", {})
        ind = ((res.get("language") or {}).get("in_dist") or {}).get("target_frac")
        if ind is not None and ind < COLLAPSED_IN_DIST:
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
    base = dict(run=run_dir.name, source=Path(finetuned.rstrip("/")).parent.name,
                method=method_of(finetuned), lr=float(cfg["train"]["lr"]),
                unit=mk.get("unit", "?"),
                attribution=("Learned" if how != "ixg" else f"IxG @ {mk.get('ixg_at')}"),
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
    p.add_argument("--exclude-scores", nargs="*", default=(),
                   help="attribution labels to leave out, e.g. 'IxG @ finetuned'. Three line "
                        "styles across five learning rates is where this figure stops being "
                        "readable, so a series that is dominated everywhere is better dropped "
                        "than drawn")
    p.add_argument("--backend", default="vllm", choices=("vllm", "hf", "any"),
                   help="which generation backend's runs to plot; mixing them puts a decoder "
                        "difference inside the replicate band")
    p.add_argument("--out", default="plots/posthoc_curves.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

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

    plot = (
        ggplot(agg, aes("frac", "value", color="LR"))
        + geom_blank(pins, aes("frac", "value"), inherit_aes=False)
        + geom_hline(anchor_lines, aes(yintercept="value"), color="#888888", linetype="dashed",
                     size=0.3)
        + geom_ribbon(band, aes("frac", ymin="lo", ymax="hi", fill="LR"), alpha=0.2,
                      color="none")
        # line style separates attribution methods; with only one present it would be a legend
        # entry that says nothing, so it is only mapped when there is something to distinguish
        + (geom_line(aes(linetype="attribution"), size=0.4) if len(attrs) > 1
           else geom_line(size=0.4))
        + geom_point(size=0.5)
        + facet_grid("metric ~ method", scales="free_y")
        + scale_x_log10(breaks=[0.001, 0.01, 0.1, 1.0], labels=["0.1%", "1%", "10%", "100%"])
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_fill_brewer(type="qual", palette="Set1", guide=None)   # matches the line colours
        + labs(x="Fraction of Units Kept", y="", color="Learning rate", linetype="Scores")
        + theme(figure_size=(5.9, 1.0 + 0.72 * len(METRICS)))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (dashed grey = pretrained anchor)")


if __name__ == "__main__":
    main()
