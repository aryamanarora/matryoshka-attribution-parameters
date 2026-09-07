"""Does a mask favour whole projection *types*? Rank histograms per type, per parameterisation.

Facet grid: rows are the parameterisation of the attributed finetune, columns are every parameter
type -- the seven projections plus the layernorms, the embedding and the final norm. Inside a
facet, one line per learning rate over the rank of every unit of that type, so a facet is "where
in the ranking do this run's ``o_proj`` units sit". Lines rather than bars: eleven columns of six
overlapping translucent histograms is unreadable, and the shape is the whole content.

**A blank facet means that type is dead in that row**, not that it ranked badly. A LoRA delta moves
only the seven projections, so under LoRA the layernorms, the embedding and the final norm have an
exactly-zero delta, their scores never leave their init, and their "ranks" are ``argsort``
tie-breaking. They are real measurements only in the full-SFT row.

**Two of these columns hold very few units**, which changes how much the line means: at ``nonresid``
granularity a 1-D parameter is one unit per tensor, so ``ln_attn`` and ``ln_mlp`` have 16 units each
(one per block) and ``norm_final`` has exactly 1. A 16-point distribution is a coarse thing and the
single-unit column is a spike by construction -- read those two as "roughly where these sit", not
as distributions.

The reference is the **dashed line at 1/n_bins**: that is where every bar would sit if the type
were spread through the ranking like any other. Mass above it on the left means the mask
systematically favours the type; mass on the right means it systematically rejects it. Reading
bars against that line is the whole figure -- the absolute heights are not otherwise meaningful.

y is a **fraction of that type's own units**, not a count, because the types are not the same size:
at ``nonresid`` granularity a block has 8192 ``gate_proj`` units against 2048 ``o_proj`` ones, so
counts would mostly say "MLPs are bigger". See ``scripts/analysis/unit_type_ranks.py``, which also explains
why the 128,289 dead units are excluded from the ranking.

Replicate runs (the same recipe attributed twice) are averaged, so one translucent layer is one
(parameterisation, lr) cell rather than one run.

    uv run python plots/plot_type_ranks.py --json plots/data/top_units/type_ranks.json \
        --out plots/type_ranks.pdf
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_hline, geom_line, ggplot,
    labs, scale_color_brewer, scale_x_continuous, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=5),
        panel_grid_major=element_line(size=0.2, color="#e8e8e8"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=6),
        legend_position="top",
        legend_direction="horizontal",
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_box_margin=0,
    )
)

ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
         "ln_attn", "ln_mlp", "embed", "norm_final"]
SHORT = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate",
         "up_proj": "up", "down_proj": "down", "ln_attn": "ln₁", "ln_mlp": "ln₂",
         "embed": "embed", "norm_final": "norm"}
SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr) -> str:
    exp, m = 0, float(lr)
    while m < 1:
        m, exp = m * 10, exp - 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def method_of(finetuned: str) -> str:
    run = Path(str(finetuned).rstrip("/")).parent.name
    m = re.search(r"_r(\d+)_", run)
    return f"LoRA r={m.group(1)}" if m else ("LoRA r=32" if "_lora_" in run else "Full SFT")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default="plots/data/top_units/type_ranks.json")
    p.add_argument("--out", default="plots/type_ranks.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    d = json.loads(Path(args.json).read_text())
    n_bins = d["bins"]
    # bin midpoints as a fraction of the run's own ranking, so rows with different live populations
    # (603,425 for a full finetune, 475,136 for LoRA) share one x axis
    mids = [(i + 0.5) / n_bins for i in range(n_bins)]

    # average replicates: one line per (method, lr), not per run
    acc = defaultdict(lambda: defaultdict(list))
    for r in d["runs"]:
        key = (method_of(r["finetuned"]), lr_label(r["lr"]))
        for t, v in r["types"].items():
            if not v.get("dead"):
                acc[key][t].append(v["frac"])
    rows = []
    for (meth, lr), per_type in acc.items():
        for t, fracs in per_type.items():
            mean = [sum(f[i] for f in fracs) / len(fracs) for i in range(n_bins)]
            for m, frac in zip(mids, mean):
                rows.append(dict(method=meth, LR=lr, type=t, pos=m, frac=frac,
                                 n_runs=len(fracs)))
    df = pd.DataFrame(rows)
    print(f"{df['method'].nunique()} parameterisations x {df['LR'].nunique()} lrs x "
          f"{df['type'].nunique()} types; {len(d['runs'])} runs averaged into "
          f"{len(acc)} cells")

    morder = sorted(df["method"].unique(),
                    key=lambda m: (m == "Full SFT", int(re.search(r"r=(\d+)", m).group(1))
                                   if "r=" in m else 0))
    df["method"] = pd.Categorical(df["method"], morder, ordered=True)
    df["type"] = pd.Categorical(df["type"], [t for t in ORDER if t in set(df["type"])],
                                ordered=True)
    df["type"] = df["type"].cat.rename_categories(
        {t: SHORT[t] for t in df["type"].cat.categories})
    df["LR"] = pd.Categorical(df["LR"], sorted(df["LR"].unique(),
                                               key=lambda s: float(s.split("×")[0])), ordered=True)

    plot = (
        ggplot(df, aes("pos", "frac", color="LR"))
        # where the line would sit if the type were spread like any other
        + geom_hline(yintercept=1 / n_bins, color="#888888", linetype="dashed", size=0.3)
        + geom_line(size=0.4)
        + facet_grid("method ~ type")
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_continuous(breaks=[0, 0.5, 1.0], labels=["top", "mid", "last"])
        + labs(x="Position in the ranking", y="Fraction of this type's units",
               color="Learning rate")
        + theme(figure_size=(6.4, 0.6 * df["method"].nunique() + 1.0))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (dashed line = 1/{n_bins}, i.e. no preference)")


if __name__ == "__main__":
    main()
