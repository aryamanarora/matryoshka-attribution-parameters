"""Method x learning rate, four metrics: does fitting French have anything to do with drifting?

One tile per (method, lr) cell of the French sweeps, four panels:

``Train loss`` / ``Test loss``   how well the run fits French (the `sft_loss` eval, held-out and
                                not, at the final budget). LOWER is a better fit.
``In-dist FR``                   the positive control: French prompts answered in French. Near 1
                                for everything, including the pretrained model, so a cell that
                                drops here is a damaged model rather than a result.
``Off-target FR``               THE headline: English prompts answered in French. 0 for the
                                pretrained model, so anything above it is generalisation out of
                                the training distribution.

The reason for putting all four in one figure is that the answer is a *dissociation*: reading
Off-target FR beside Test loss shows the runs that drift are the ones that fit French WORSE, and
LoRA sits where low-lr full SFT sits -- a good fit that never leaves the distribution.

**A cell whose in-dist control collapsed is drawn grey and left out of the colour scale.** That
control sits at ~0.95 for the *pretrained* model, so `In-dist FR` below :data:`COLLAPSED_IN_DIST`
means the run destroyed the model rather than taught it anything -- and then none of its four
numbers measure what the panel says they do (a diverged run scores a high off-target rate by
babbling non-English). They are still drawn and labelled, because "this cell diverged" is a
result about that (method, lr) and a blank would hide it. Excluding them from the normalisation
is what keeps the loss panels legible: one diverged cell at 7.1 against a healthy range of
0.7-1.4 otherwise compresses every real difference into the first 10% of the colour ramp.

The lightest fill is a pale tint rather than white, deliberately: a cell at the minimum of its
panel (an off-target rate of exactly 0.00, of which there are many) would otherwise be the same
colour as the page and read as *missing* rather than as zero.

**One colour scale per unit, shared across that unit's two panels.** Losses are blue on a scale
spanning every healthy loss cell in both loss panels; the two French rates are red on [0, 1], the
natural range of a fraction rather than of the data. So Train loss and Test loss are directly
comparable to each other, In-dist FR and Off-target FR are directly comparable to each other, and
the change of hue is what says "these two pairs are not comparable across the gap". A single scale
over all four would have flattened the 0.67-1.38 loss range into one indistinguishable block
beside the full-width [0, 1] of the rates.

Darker means a larger number in both families, which is "more drift" for the rates and "a worse
fit" for the losses -- the direction is not shared, so the tile labels remain the thing to read.

Blank tiles are cells that were never run: the three sweeps do not share one lr grid (full SFT
went down to 2e-5, the rank grid up to 1e-3), and inventing a value for a missing cell would be
worse than the gap.

Reads `lora.r` and `train.lr` from each run's own ``config.yaml`` rather than from its directory
name, and refuses to plot a run whose ``config.yaml`` is NEWER than its ``evals.json`` -- that
ordering means the results predate the config and belong to an earlier, cancelled attempt, which
has already produced one wrong figure in this repo.

    uv run python plots/plot_method_lr_grid.py --dir plots/data/method_lr \
        --out plots/method_lr_grid.pdf
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_text, facet_wrap, geom_text, geom_tile, ggplot, labs,
    scale_fill_gradient, scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        # height is set per-figure in main(): it depends on how many methods there are
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_blank(),      # gridlines under opaque tiles are noise
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_position="top",
        legend_direction="horizontal",
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_box_margin=0,
    )
)

#: (panel title, path into evals.json's per-condition dict, format, unit family)
METRICS = [
    ("Train loss", ("sft_loss", "train", "loss"), "{:.2f}", "loss"),
    ("Test loss", ("sft_loss", "test", "loss"), "{:.2f}", "loss"),
    ("In-dist FR", ("language", "in_dist", "target_frac"), "{:.2f}", "rate"),
    ("Off-target FR", ("language", "off_target", "target_frac"), "{:.2f}", "rate"),
]
#: one hue per unit family, both from Set1: losses blue, French rates red
FAMILIES = {
    # limits None -> taken from the healthy cells of both of that family's panels
    "loss": dict(high="#377eb8", low="#e8f0f6", title="Loss", limits=None, breaks=None),
    "rate": dict(high="#e41a1c", low="#fdeaea", title="FR rate", limits=(0.0, 1.0),
                 breaks=[0.0, 0.5, 1.0]),
}
COLLAPSED_FILL = "#c8c8c8"

#: below this in-dist target fraction the run is treated as collapsed, not measured. The
#: pretrained model scores ~0.95, so anything under half of that is not a language result.
COLLAPSED_IN_DIST = 0.5

SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr: float) -> str:
    """``0.0002`` -> ``2×10⁻⁴``. Unicode superscripts, not LaTeX, which would break the font."""
    exp = 0
    m = float(lr)
    while m < 1:
        m *= 10
        exp -= 1
    m = round(m, 3)
    mant = f"{m:g}"
    return f"{mant}×10{str(exp).translate(SUPERS)}"


def method_label(cfg: dict) -> str:
    lora = cfg.get("lora")
    if not lora:
        return "Full SFT"
    return f"LoRA r={lora['r']}"


def at(res: dict, path, condition="dense"):
    node = (res.get("final") or {}).get(condition) or {}
    for key in path:
        node = (node or {}).get(key)
        if node is None:
            return None
    return node


def load(run_dir: Path):
    """One run -> ``{method, lr, <metric>: value}``, or None if it should not be plotted."""
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not ev.exists() or not cf.exists():
        return None
    # Freshness, not existence: `config.yaml` is written at startup and `evals.json` near the end,
    # so config NEWER than evals means these results are from a previous attempt of this config.
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml is newer than evals.json (stale results)")
        return None
    cfg = yaml.safe_load(cf.read_text())
    res = json.loads(ev.read_text())
    row = {"run": run_dir.name, "method": method_label(cfg), "lr": float(cfg["train"]["lr"])}
    for title, path, _, _fam in METRICS:
        row[title] = at(res, path)
    if all(row[t] is None for t, _, _, _ in METRICS):
        print(f"  SKIP {run_dir.name}: none of the four metrics present")
        return None
    ind = row["In-dist FR"]
    row["collapsed"] = ind is not None and ind < COLLAPSED_IN_DIST
    if row["collapsed"]:
        print(f"  {run_dir.name}: in-dist FR {ind:.2f} < {COLLAPSED_IN_DIST} -- diverged, drawn "
              f"grey and excluded from the colour scale")
    return row


def collect(root: Path) -> pd.DataFrame:
    rows = [r for d in sorted(root.iterdir()) if d.is_dir() for r in [load(d)] if r]
    if not rows:
        raise SystemExit(f"no plottable runs under {root}")
    df = pd.DataFrame(rows)
    # The r=32 LoRA cells exist twice at the lrs both grids cover (the dedicated rank sweep and the
    # earlier parameterisation sweep are the same recipe there). Keep the rank-sweep run, so a row
    # of the heatmap comes from one uniform grid rather than two -- and say when that happens.
    df["from_rank_grid"] = df["run"].str.contains("_lora_r")
    dupes = df.duplicated(["method", "lr"], keep=False)
    if dupes.any():
        for (m, lr), g in df[dupes].groupby(["method", "lr"]):
            kept = g.sort_values("from_rank_grid", ascending=False).iloc[0]["run"]
            dropped = [r for r in g["run"] if r != kept]
            print(f"  {m} @ lr {lr:g}: duplicate cells {g['run'].tolist()} -> keeping {kept}, "
                  f"dropping {dropped}")
    df = (df.sort_values("from_rank_grid", ascending=False)
            .drop_duplicates(["method", "lr"], keep="first"))
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/method_lr")
    p.add_argument("--out", default="plots/method_lr_grid.pdf",
                   help="PDF for the paper; pass a .png when you want a raster copy")
    p.add_argument("--dpi", type=int, default=300, help="raster output only; PDF is vector")
    args = p.parse_args()

    df = collect(Path(args.dir))
    print(f"{len(df)} cells: {sorted(df['method'].unique())} x "
          f"{[lr_label(x) for x in sorted(df['lr'].unique())]}")

    titles = [t for t, _, _, _ in METRICS]
    long = df.melt(id_vars=["run", "method", "lr", "collapsed"], value_vars=titles,
                   var_name="metric", value_name="value").dropna(subset=["value"])
    fmt = {t: f for t, _, f, _ in METRICS}
    fam = {t: f for t, _, _, f in METRICS}
    long["label"] = [fmt[m].format(v) for m, v in zip(long["metric"], long["value"])]
    long["family"] = long["metric"].map(fam)
    long["metric"] = pd.Categorical(long["metric"], titles, ordered=True)
    long["lr_lab"] = pd.Categorical(
        [lr_label(x) for x in long["lr"]],
        [lr_label(x) for x in sorted(long["lr"].unique())], ordered=True)
    # rank order, full SFT last: reading down the y axis then goes from least to most capacity
    order = sorted(long["method"].unique(),
                   key=lambda m: (m == "Full SFT", int(m.split("=")[1]) if "=" in m else 0))
    long["method"] = pd.Categorical(long["method"], order, ordered=True)
    n_rows = long["method"].nunique()

    def half(family: str, show_y: bool):
        """The two panels of one unit family, on one shared fill scale."""
        spec = FAMILIES[family]
        sub = long[long["family"] == family]
        live, dead = sub[~sub["collapsed"]], sub[sub["collapsed"]]
        # limits span the healthy cells of BOTH panels, so the two are directly comparable; the
        # diverged cells are excluded (one at 7.1 would leave every real difference in the first
        # tenth of the ramp) and drawn grey instead
        lims = spec["limits"] or (float(live["value"].min()), float(live["value"].max()))
        # three ticks, inset from the ends: the default five collide on a 30pt colourbar
        breaks = spec["breaks"] or [round(lims[0] + f * (lims[1] - lims[0]), 2)
                                    for f in (0.08, 0.5, 0.92)]
        p = (
            ggplot(sub, aes("lr_lab", "method"))
            + geom_tile(dead, fill=COLLAPSED_FILL, color="white", size=0.4)
            + geom_tile(live, aes(fill="value"), color="white", size=0.4)
            + geom_text(aes(label="label"), size=5.2, color="#000000", family=FAMILY)
            + facet_wrap("metric", nrow=1)
            + scale_fill_gradient(low=spec["low"], high=spec["high"], limits=lims,
                                  breaks=breaks, name=spec["title"])
            + scale_x_discrete(expand=(0, 0))
            + scale_y_discrete(expand=(0, 0), limits=order)
            + labs(x="Learning Rate", y="Method" if show_y else "")
            # figure_size is the size of the WHOLE composed figure, not of this half: a Beside
            # composition takes its canvas from one plot's theme (and Compose.save documents that
            # it ignores width/height), so both halves have to name the full width or the two get
            # squeezed into one half's worth of inches.
            + theme(figure_size=(6.8, 1.15 + 0.4 * n_rows),
                    legend_key_width=40, legend_key_height=5)
        )
        if not show_y:
            # the methods are already named by the left half; repeating them halves the tile width
            p += theme(axis_text_y=element_blank(), axis_ticks_major_y=element_blank())
        return p

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The size goes on the COMPOSITION, not on the halves: a composed figure ignores its parts'
    # figure_size and would otherwise be saved at the default portrait shape, which squeezes six
    # lr columns into a couple of inches and overlaps every tile label.
    comp = half("loss", show_y=True) | half("rate", show_y=False)
    comp.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
