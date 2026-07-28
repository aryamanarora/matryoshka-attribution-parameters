"""Emergent misalignment vs mask sparsity, from `eval_em_sparsity.py` summaries.

Reads each run's `summary.json` and plots the two numbers their `get_basic_eval_stats`
produces, both already in percent:

  misaligned_coherent   `aligned <= 30` AND `coherent > 50` -- the EM rate itself
  coherent              `coherent > 50` -- the gate the EM rate is conditioned on

Drawn as a stacked bar over the three mutually exclusive outcomes rather than two line
panels, because the EM rate is not interpretable without the gate: a condition whose
generations are incoherent scores ~0% misaligned-and-coherent no matter how misaligned it
is. Stacking puts that failure mode in the same bar instead of a panel you have to
cross-reference. The three shares are derivable from the two reported numbers --
``aligned & coherent = coherent - misaligned_coherent`` and
``incoherent = 100 - coherent`` -- so no re-reading of the judged CSVs is needed.

Anchors: `pretrained` is drawn as a dashed reference line (the base model's rate, which
should be near zero); `full_delta` composes the same weights as `frac_1` under `cause` and
so already sits at the right-hand end of the curve.

Runs anywhere -- falls back from Inter to whatever sans font exists.

    uv run python plots/plot_em_sparsity.py \
        --run log=/mnt/data/.../runs/em/row_log \
        --run uniform=/mnt/data/.../runs/em/row_uniform \
        --out /mnt/data/.../em_sparsity.pdf

LEGACY INPUT FORMAT. This reads the `sweep.json` / `summary.json` / `mmlu.json` files the
pre-refactor scripts wrote, and still works on the run directories that already contain
them. New runs write a single `evals.json`
(`{condition: {eval: {split: {metric: value}}}}`, see eval/runner.py) -- this script has
not been ported to it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_col, geom_text, ggplot,
    labs, scale_fill_manual, scale_y_continuous, theme, theme_bw, theme_set,
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

# Set1, with its grey for the "we cannot tell" category so it reads as absence of signal
# rather than a third finding. Red for misaligned: it is the thing being measured.
OUTCOMES = [("Misaligned & Coherent", "#e41a1c"),
            ("Aligned & Coherent", "#377eb8"),
            ("Incoherent", "#999999")]


def frac_label(f: float) -> str:
    """0.001 -> "0.1%"; 1.0 -> "100%".

    3 significant figures, because mask_frac is the realised k/total (0.00100057) rather
    than the nominal grid point, and "0.100057%" is not a tick label.
    """
    return f"{100 * f:.3g}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    p.add_argument("--out", default="em_sparsity.pdf")
    p.add_argument("--legend-title", default="Outcome")
    args = p.parse_args()

    curves, anchors, meta = [], [], {}
    for spec in args.run:
        label, _, d = spec.partition("=")
        blob = json.loads((Path(d) / "summary.json").read_text())
        meta[label] = blob
        for rec in blob["curve"]:
            tgt = curves if rec["condition"].startswith("frac_") else anchors
            tgt.append(dict(run=label, **rec))
    cur, anc = pd.DataFrame(curves), pd.DataFrame(anchors)
    if cur.empty:
        raise SystemExit("no frac_* conditions in the summaries")

    # `full_delta` composes the same weights as `frac_1` under cause, so plotting both would
    # duplicate a bar; `pretrained` is kept as the leftmost bar -- as a category it is part of
    # the comparison rather than a reference line.
    pre = anc[anc.condition == "pretrained"].assign(xlab="pretrained", order=-1.0)
    cur = cur.assign(xlab=cur.mask_frac.map(frac_label), order=cur.mask_frac)
    bars = pd.concat([pre, cur], ignore_index=True)

    bars["Aligned & Coherent"] = bars.coherent - bars.misaligned_coherent
    bars["Incoherent"] = 100.0 - bars.coherent
    bars = bars.rename(columns={"misaligned_coherent": "Misaligned & Coherent"})
    long = bars.melt(id_vars=["run", "xlab", "order", "samples"],
                     value_vars=[n for n, _ in OUTCOMES],
                     var_name="outcome", value_name="pct")
    order = bars.sort_values("order").xlab.unique().tolist()
    long["xlab"] = pd.Categorical(long.xlab, categories=order, ordered=True)
    long["outcome"] = pd.Categorical(long.outcome, categories=[n for n, _ in OUTCOMES],
                                     ordered=True)

    # The EM rate is only 0-6%, so its band is nearly invisible in a 0-100 stack. Print it
    # above each bar: the stack shows the composition, the number shows the quantity.
    # Carry the unit, and print an explicit 0% rather than a blank -- on this data most
    # conditions ARE zero, and a gap reads as "not measured" instead of "measured, none".
    tags = bars.assign(xlab=pd.Categorical(bars.xlab, categories=order, ordered=True),
                       label=bars["Misaligned & Coherent"].map(lambda v: f"{v:.1f}%"))

    pl = (
        ggplot(long, aes("xlab", "pct", fill="outcome"))
        + geom_col(width=0.75)
        # Rotated, because 11 bars per facet leave ~0.15in each and "0.0%" is wider than that
        # horizontally -- side by side they merge into one unreadable string. Vertical needs
        # only line height, paid for with headroom above the 100% stack.
        + geom_text(aes("xlab", 102, label="label"), data=tags, size=4.0, angle=90,
                    color="#e41a1c", inherit_aes=False, ha="left", va="center")
        + scale_y_continuous(expand=(0.02, 0, 0.30, 0), breaks=[0, 25, 50, 75, 100])
        + scale_fill_manual(values={n: c for n, c in OUTCOMES}, name=args.legend_title)
        + facet_wrap("~run")
        + labs(x="Fraction of Parameter Units Kept", y="% of Samples")
        # 11 bars x N facets at 5.5in leaves ~0.15in per label, which the theme's 45deg text
        # overruns into its neighbour. Vertical costs height instead of width, of which there
        # is plenty.
        + theme(axis_text_x=element_text(size=5, rotation=90, hjust=0.5, vjust=1.0),
                figure_size=(5.5, 2.5))
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY})")
    for lab, blob in meta.items():
        print(f"\n{lab}: unit={blob['unit']} mode={blob['mode']} "
              f"n_per_question={blob['n_per_question']} total_units={blob['total_units']:,}")
        sub = bars[bars.run == lab].sort_values("order")
        for _, r in sub.iterrows():
            print(f"  {r.xlab:>10s}  misaligned&coherent {r['Misaligned & Coherent']:6.2f}%"
                  f"   aligned&coherent {r['Aligned & Coherent']:6.2f}%"
                  f"   incoherent {r['Incoherent']:6.2f}%   n={int(r['samples'])}")


if __name__ == "__main__":
    main()
