"""Units the masks agree about: the best and worst by mean rank, one facet each.

``scripts/analysis/unit_ranks.py`` ranks every unit in every post-hoc run and averages. This shows the
extremes of that average -- but a mean over 25 runs is exactly the kind of number that can be
carried by two runs and contradicted by the rest, so each facet plots **every run's rank for that
unit**, with the mean drawn across it. A unit whose points cluster is one the masks agree about; a
unit whose mean is good because of one outlier looks like one.

Facet rows are the two ends (best / worst mean rank), columns are units ordered by that mean.
Inside a facet: x is the parameterisation of the attributed finetune, y is that unit's rank in that
run, colour is the learning rate.

**y is log and inverted** -- rank 1 (the top of the ranking) at the top of the panel, so "higher is
more important" reads the way it looks. The grey dashed line is the mean over runs; the grey band
is the middle of the ranking, i.e. where a unit with no signal would sit.

**Dead units are excluded upstream**, which is what makes the bottom row mean anything: 128,289 of
603,425 units have an exactly-zero delta under a LoRA finetune and keep their init score, so an
unfiltered "worst mean rank" would be reporting ``argsort``'s tie-breaking. See
``scripts/analysis/unit_ranks.py``.

    uv run python plots/plot_unit_ranks.py --json plots/data/top_units/unit_ranks.json \
        --out plots/unit_ranks.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_point, ggplot,
    labs, scale_color_brewer, scale_y_log10, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text_y=element_text(size=5.5),
        axis_text_x=element_text(size=5, rotation=45, hjust=1, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.05,
        strip_background=element_blank(),
        strip_text=element_text(size=5.8),
        legend_position="top",
        legend_direction="horizontal",
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_box_margin=0,
    )
)

SHORT_PROJ = {"down_proj": "down", "up_proj": "up", "gate_proj": "gate", "o_proj": "o",
              "q_proj": "q", "k_proj": "k", "v_proj": "v"}
SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr) -> str:
    exp, m = 0, float(lr)
    while m < 1:
        m, exp = m * 10, exp - 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def method_of(finetuned: str) -> str:
    run = Path(str(finetuned).rstrip("/")).parent.name
    m = re.search(r"_r(\d+)_", run)
    return f"r={m.group(1)}" if m else ("r=32" if "_lora_" in run else "full")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default="plots/data/top_units/unit_ranks.json")
    p.add_argument("--out", default="plots/unit_ranks.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    d = json.loads(Path(args.json).read_text())
    meta = {r["run"]: r for r in d["runs"]}
    rows, facets = [], []
    for u in d["units"]:
        parts = u["short"].split()
        proj = SHORT_PROJ.get(parts[-1], parts[-1])
        name = f"{parts[0]} {proj}[{u['index_in_param']}]" if parts[0].startswith("L") \
            else f"{proj}[{u['index_in_param']}]"
        # the mean is in the strip, so a facet is self-describing when read on its own
        facet = f"{name}\nmean {u['mean_rank']/1000:.0f}k"
        facets.append((u["end"], u["mean_rank"], facet))
        for run, rank in u["per_run"].items():
            m = meta[run]
            rows.append(dict(facet=facet, end=u["end"], mean_rank=u["mean_rank"], rank=rank,
                             method=method_of(m["finetuned"]), LR=lr_label(m["lr"])))
    df = pd.DataFrame(rows)

    # facets ordered best-mean first within each end, so the two rows read left to right
    order = [f for _, _, f in sorted(set(facets), key=lambda t: (t[0] != "top", t[1]))]
    df["facet"] = pd.Categorical(df["facet"], order, ordered=True)
    df["method"] = pd.Categorical(df["method"], ["r=8", "r=32", "r=128", "r=512", "full"],
                                  ordered=True)
    df["LR"] = pd.Categorical(df["LR"], sorted(df["LR"].unique(),
                                               key=lambda s: float(s.split("×")[0])), ordered=True)
    n_per_end = len(order) // 2
    mid = d["n_live"] / 2
    print(f"{df['facet'].nunique()} units x {len(meta)} runs; midpoint of the ranking = {mid:,.0f}")

    means = df[["facet", "mean_rank"]].drop_duplicates()
    plot = (
        ggplot(df, aes("method", "rank", color="LR"))
        # where a unit with no signal would sit, for scale
        + geom_hline(yintercept=mid, color="#bbbbbb", size=0.3)
        + geom_hline(means, aes(yintercept="mean_rank"), color="#666666", linetype="dashed",
                     size=0.35)
        + geom_point(size=0.7, alpha=0.85)
        + facet_wrap("facet", nrow=2)
        # Inverted by reversed limits, which plotnine accepts on a log scale: rank 1 ends up at
        # the top of the panel, so "higher" reads as "the mask thinks this matters more".
        + scale_y_log10(limits=(d["n_live"], 1), breaks=[1, 100, 10_000, 475_136],
                        labels=["1", "10²", "10⁴", "last"])
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Parameterisation of the attributed finetune", y="Rank of this unit (1 = top)",
               color="Learning rate")
        + theme(figure_size=(5.9, 3.4))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  ({n_per_end} best and {n_per_end} worst by mean rank)")


if __name__ == "__main__":
    main()
