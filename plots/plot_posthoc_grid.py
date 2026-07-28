"""Post-hoc attribution across the whole sweep: how sparse a slice still speaks French?

Every finetune in `configs/french/{sft,posthoc}/` is attributed post hoc -- delta frozen, only the
scores trained -- and this figure lays all of those sparsity curves out at once as heatmaps.

Facets: **rows are the four metrics**, columns are **how much of the delta the mask keeps**. Inside
each facet, the same (learning rate x method) grid as `plot_method_lr_grid.py`, so a column of this
figure is "every run at 1% of units" and a row is one metric across the whole sparsity range.

Reading it:

``Train loss`` / ``Test loss``  the objective the scores are fitted to: the parameter-space
    analogue of MIB's CPR. A sparse mask that already reaches the dense loss is one that found the
    finetune's update; one whose loss sits at the pretrained level is masking the update away.
``In-dist FR`` (the control)  French prompts answered in French, which the *pretrained* model
    already does. It should stay saturated all the way down the sparsity range: it is the check
    that a sparse mask produced a working model at all, so a column where in-dist collapses while
    off-target rises is a broken model, not a localised behaviour.
``Off-target FR`` (the headline)  where along the row colour appears is where the drift switches
    on. Saturated already at 0.1% would mean a thousandth of the units carries the behaviour; only
    lighting up at 50% means the mask never localised it.

**Two colour scales, one per unit**, as in `plot_method_lr_grid.py`: losses blue on a shared scale
spanning both loss rows, French rates red on [0, 1]. The two are separate figures composed
vertically, because one plot can hold only one fill scale -- and a single scale over both would
flatten a 0.7-1.6 loss range against the full width of a fraction. The rate scale is fixed at
[0, 1] rather than taken from the data on purpose: a fraction's range is known in advance, and
normalising it would make "the sparsest mask that works" depend on which runs happened to be in
the directory.

There are no per-tile numbers -- ten sparsity columns times six learning rates is 60 tile columns
across a text width, and a legible label does not fit. Use `plot_method_lr_grid.py` for exact
values at the dense point.

Cells whose *attributed finetune* had diverged are dropped, since a mask fitted to a model that no
longer answers French prompts in French is not attributing language drift -- with them in, the
in-dist row stops being a check on the mask and becomes a mix of two failure modes. `--source-dir`
supplies the finetunes' own results to detect them; `--include-diverged` keeps them anyway.

    uv run python plots/plot_posthoc_grid.py --dir plots/data/posthoc_sweep \
        --out plots/posthoc_grid.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_text, facet_grid, geom_tile, ggplot, labs, scale_fill_gradient,
    scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text_y=element_text(size=5.5),
        # 90 degrees, not 45: a rotated label's footprint is then its font height, which is the
        # only way six learning rates fit under each of ten facets across a text width
        axis_text_x=element_text(size=4.5, rotation=90, hjust=1, vjust=0.5),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        # tight: with 20 panels of tiles the gaps are dead space, and adjacent heatmap panels
        # read as one grid when they nearly touch
        panel_spacing_x=0.008,
        panel_spacing_y=0.012,
        strip_background=element_blank(),
        strip_text=element_text(size=6),
        legend_position="top",
        legend_direction="horizontal",
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_box_margin=0,
        legend_key_width=44,
        legend_key_height=5,
    )
)

#: (row title, path into a condition's results, unit family)
METRICS = [
    ("Train", ("sft_loss", "train", "loss"), "loss"),
    ("Test", ("sft_loss", "test", "loss"), "loss"),
    ("In-dist", ("language", "in_dist", "target_frac"), "rate"),
    ("Off-target", ("language", "off_target", "target_frac"), "rate"),
]
#: one hue per unit family, both from Set1
FAMILIES = {
    "loss": dict(low="#e8f0f6", high="#377eb8", title="Loss", limits=None, breaks=None),
    "rate": dict(low="#fdeaea", high="#e41a1c", title="FR rate", limits=(0.0, 1.0),
                 breaks=[0.0, 0.5, 1.0]),
}
#: the sparsity conditions are named `frac_<x>`; `pretrained` and `full_delta` are anchors, not
#: points on the grid, so they are not columns of this figure
FRAC_RE = re.compile(r"^frac_(?P<frac>[0-9.]+)$")
COLLAPSED_IN_DIST = 0.5          # same threshold as plot_method_lr_grid.py

SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr: float) -> str:
    exp, m = 0, float(lr)
    while m < 1:
        m, exp = m * 10, exp - 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def frac_label(frac: float) -> str:
    pct = 100 * frac
    return f"{pct:g}%"


def method_of(finetuned: str) -> str:
    """``.../french_lora_r128_lr5e-4/adapter`` -> ``LoRA r=128``.

    Read from the attributed checkpoint's path rather than from this run's own config, because a
    post-hoc config deliberately has NO ``lora:`` block -- the mask is over base-model parameter
    names, and the adapter is an input to it. So the run being attributed is the only place the
    parameterisation is recorded.
    """
    run = Path(finetuned.rstrip("/")).parent.name
    m = re.search(r"_r(\d+)_", run)
    return f"LoRA r={m.group(1)}" if m else ("LoRA r=32" if "_lora_" in run else "Full SFT")


def source_run(finetuned: str) -> str:
    return Path(finetuned.rstrip("/")).parent.name


def diverged_sources(source_dir: Path) -> set:
    """Finetunes whose in-dist control collapsed -- nothing about French left to attribute."""
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


def rows_for(run_dir: Path):
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
        print(f"  SKIP {run_dir.name}: no mask.finetuned, so it is not a post-hoc attribution")
        return []
    res = json.loads(ev.read_text()).get("final") or {}
    out = []
    for cond, per_eval in res.items():
        m = FRAC_RE.match(cond)
        if not m:
            continue
        for title, path, family in METRICS:
            node = per_eval
            for key in path:
                node = (node or {}).get(key)
            if node is None:
                continue
            out.append(dict(run=run_dir.name, source=source_run(finetuned),
                            method=method_of(finetuned), lr=float(cfg["train"]["lr"]),
                            frac=float(m.group("frac")), metric=title, family=family,
                            value=float(node)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/posthoc_sweep")
    p.add_argument("--source-dir", default="plots/data/method_lr",
                   help="the attributed finetunes' own results, to spot diverged ones")
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument("--out", default="plots/posthoc_grid.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = [r for d in sorted(Path(args.dir).iterdir()) if d.is_dir() for r in rows_for(d)]
    if not rows:
        raise SystemExit(f"no post-hoc results under {args.dir}")
    df = pd.DataFrame(rows)

    bad = diverged_sources(Path(args.source_dir))
    if bad and not args.include_diverged:
        drop = df["source"].isin(bad)
        if drop.any():
            print(f"  dropping {sorted(df[drop]['source'].unique())}: the attributed finetune "
                  f"diverged (in-dist FR < {COLLAPSED_IN_DIST})")
        df = df[~drop]
    print(f"{df['run'].nunique()} post-hoc runs, {df['frac'].nunique()} sparsity columns, "
          f"{len(df)} tiles")

    df["lr_lab"] = pd.Categorical([lr_label(x) for x in df["lr"]],
                                  [lr_label(x) for x in sorted(df["lr"].unique())], ordered=True)
    df["frac_lab"] = pd.Categorical([frac_label(x) for x in df["frac"]],
                                    [frac_label(x) for x in sorted(df["frac"].unique())],
                                    ordered=True)
    # rank order, full SFT last: reading down the axis goes from least to most capacity
    order = sorted(df["method"].unique(),
                   key=lambda m: (m == "Full SFT", int(m.split("=")[1]) if "=" in m else 0))
    df["method"] = pd.Categorical(df["method"], order, ordered=True)
    df["metric"] = pd.Categorical(df["metric"], [t for t, _, _ in METRICS], ordered=True)

    n_meth = df["method"].nunique()
    # A metric row has to be tall enough for its own rotated strip label, whatever the method count
    # -- at two methods the tiles alone would leave "Off-target" overlapping the row above it.
    row_h = max(0.4, 0.15 * n_meth)
    size = (5.9, 1.45 + row_h * len(METRICS))

    def half(family: str, *, is_top: bool):
        """The two metric rows of one unit family, on one shared fill scale."""
        spec = FAMILIES[family]
        sub = df[df["family"] == family]
        lims = spec["limits"] or (float(sub["value"].min()), float(sub["value"].max()))
        breaks = spec["breaks"] or [round(lims[0] + f * (lims[1] - lims[0]), 2)
                                    for f in (0.08, 0.5, 0.92)]
        p = (
            ggplot(sub, aes("lr_lab", "method", fill="value"))
            + geom_tile(color="white", size=0.25)
            + facet_grid("metric ~ frac_lab")
            + scale_fill_gradient(low=spec["low"], high=spec["high"], limits=lims,
                                  breaks=breaks, name=spec["title"])
            + scale_x_discrete(expand=(0, 0))
            + scale_y_discrete(expand=(0, 0), limits=order)
            # figure_size is the size of the WHOLE composed figure, not of this half: a composition
            # takes its canvas from one plot's theme and Compose.save ignores width/height
            + theme(figure_size=size,
                    # one colourbar per half, pushed to the outside edges so neither lands in the
                    # middle of the grid
                    legend_position="top" if is_top else "bottom")
        )
        if is_top:
            # the sparsity columns are labelled once, above the whole figure, and the learning
            # rates once below it -- the two halves are aligned, so repeating either is noise, and
            # so is a second "Method" title against the same y axis
            p += labs(x="", y="Method")
            p += theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank())
        else:
            p += labs(x="Learning Rate", y="")
            p += theme(strip_text_x=element_blank())
        return p

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (half("loss", is_top=True) / half("rate", is_top=False)).save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
