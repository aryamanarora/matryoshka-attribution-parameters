"""Raw judge scores vs mask sparsity, with confidence intervals.

The stacked-bar view (`plot_em_sparsity.py`) shows *thresholded rates*: what fraction of
responses land in each quadrant of `aligned <= 30` x `coherent > 50`. That throws away the
scores themselves, and thresholds are brittle -- a condition drifting from 55 to 45 coherence
crosses the gate and looks catastrophic, while one drifting 95 -> 60 looks unchanged. This
plots the underlying 0-100 judge scores instead.

Misalignment is plotted as ``100 - aligned`` so that up means worse in both panels; their
judge scores *alignment*, where 0 is "totally misaligned".

Intervals are **bootstrap percentile 95% CIs of the mean** (10k resamples), not mean +-
1.96 sd/sqrt(n): the score distributions are strongly bimodal -- most responses sit near 100
with a small tail near 0 -- so the normal approximation understates uncertainty in exactly
the conditions where the tail matters. The dashed line is the pretrained model.

    uv run python plots/plot_em_scores.py \
        --responses log=/mnt/data/.../runs/em/row_log/responses \
        --responses uniform=/mnt/data/.../runs/em/row_uniform/responses \
        --out /mnt/data/.../em_scores.pdf
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_point, geom_ribbon, ggplot, labs, scale_color_brewer, scale_fill_brewer,
    scale_x_log10, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 2.0),
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

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_label(breaks):
    return ["" if not b or b <= 0 else "10" + str(int(round(math.log10(b)))).translate(SUP)
            for b in breaks]


def boot_ci(x, n_boot=10000, seed=0):
    """Bootstrap percentile 95% CI of the mean. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", action="append", required=True, metavar="LABEL=DIR")
    p.add_argument("--out", default="em_scores.pdf")
    p.add_argument("--legend-title", default="k Schedule")
    p.add_argument("--jitter", type=float, default=0.06)
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    METRICS = [("misalignment", "Misalignment (100 − Aligned)"),
               ("coherent", "Coherence Score")]
    rows, anchors = [], []
    for spec in args.responses:
        label, _, d = spec.partition("=")
        for f in sorted(Path(d).glob("*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty or "aligned" not in df or "coherent" not in df:
                continue
            df["misalignment"] = 100.0 - pd.to_numeric(df["aligned"], errors="coerce")
            df["coherent"] = pd.to_numeric(df["coherent"], errors="coerce")
            for col, panel in METRICS:
                m, lo, hi = boot_ci(df[col], args.n_boot)
                if np.isnan(m):
                    continue
                rec = dict(run=label, panel=panel, mean=m, lo=lo, hi=hi,
                           n=int(df[col].notna().sum()))
                (rows if f.stem.startswith("frac_") else anchors).append(
                    dict(rec, frac=float(f.stem[5:])) if f.stem.startswith("frac_")
                    else dict(rec, anchor=f.stem))
    cur, anc = pd.DataFrame(rows), pd.DataFrame(anchors)
    if cur.empty:
        raise SystemExit("no judged frac_*.csv found")

    labs_ = list(dict.fromkeys(cur.run))
    jit = dict(zip(labs_, 10 ** np.linspace(-args.jitter, args.jitter, len(labs_))
                   if len(labs_) > 1 else [1.0]))
    cur["x"] = cur.frac * cur.run.map(jit)

    pre = anc[anc.anchor == "pretrained"] if not anc.empty else pd.DataFrame()
    ref = (pre.groupby("panel", as_index=False)["mean"].mean() if not pre.empty else None)

    pl = (
        ggplot(cur, aes("x", "mean", color="run"))
        + geom_ribbon(aes(ymin="lo", ymax="hi", fill="run"), alpha=0.18, color="none")
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + scale_x_log10(labels=log_label)
        + scale_color_brewer(type="qual", palette="Set1", name=args.legend_title)
        + scale_fill_brewer(type="qual", palette="Set1", guide=None)
        + facet_wrap("~panel", scales="free_y")
        + labs(x="Fraction of Parameter Units Kept", y="Judge Score (0–100)")
    )
    if ref is not None:
        pl = pl + geom_hline(aes(yintercept="mean"), data=ref, size=0.3, linetype="dashed",
                             color="#777777", inherit_aes=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY}, {args.n_boot} bootstrap resamples)")
    for panel in [t for _, t in METRICS]:
        print(f"\n{panel}")
        for _, r in anc[anc.panel == panel].iterrows():
            print(f"  {r.anchor:12s} [{r.run:8s}] {r['mean']:6.2f}  "
                  f"95% CI [{r.lo:6.2f}, {r.hi:6.2f}]  n={r.n}")
        for _, r in cur[cur.panel == panel].sort_values(["run", "frac"]).iterrows():
            print(f"  {r.frac:11.3%} [{r.run:8s}] {r['mean']:6.2f}  "
                  f"95% CI [{r.lo:6.2f}, {r.hi:6.2f}]  n={r.n}")


if __name__ == "__main__":
    main()
