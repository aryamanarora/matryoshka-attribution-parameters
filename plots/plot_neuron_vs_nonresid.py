"""Unit granularity head-to-head: tied neuron/head units against per-vector nonresid ones.

One figure for one comparison: the fr2de 8B post-hoc sweep re-fitted at ``neuron_head``
granularity (an MLP unit ties gate/up/down at one d_ffn index; an attention unit is one head's
slice of one projection) against its ``nonresid`` twin (every vector its own unit). Each cell
pair resolves identically except ``mask.unit`` -- checked by construction with --print-config --
so the gap between the two curves in a panel is the unit definition and nothing else.

Columns are learning rates, rows the two things worth reading together:

* **off-target German rate** -- the headline. Where the curves separate, the finetunes'
  behaviour is (or is not) carried by whole neurons and heads.
* **held-out loss** -- the control that makes the top row interpretable. The tied mask matches
  or beats nonresid on loss at every fraction even where its behaviour curve is far lower, so a
  gap above is not "the tied mask fits worse"; localising the loss and localising the behaviour
  come apart.

Fractions, not unit counts, on x: the two modes have different totals (461,312 vs 1,703,936)
but a neuron is exactly its three nonresid vectors, so equal fractions select roughly equal
parameter mass. The grey dashed line is the pretrained anchor (identical for every run).

    uv run python plots/plot_neuron_vs_nonresid.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_manual, scale_x_log10, scale_y_continuous, theme,
    theme_bw, theme_set,
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
        panel_spacing_y=0.03,
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

#: Set1 red/blue; the tied mode is the one under discussion, so it gets the first colour.
UNIT_LABELS = {"neuron_head": "Neuron / head (tied)", "nonresid": "Per vector (nonresid)"}
UNIT_COLORS = {"Neuron / head (tied)": "#e41a1c", "Per vector (nonresid)": "#377eb8"}

METRICS = [
    ("German rate (off-target)", ("language", "off_target", "target_frac"), (0, 1)),
    ("Held-out loss", ("sft_loss", "test", "loss"), None),
]


def dig(cond: dict, path):
    for k in path:
        cond = (cond or {}).get(k)
    return cond


def load(data_dir: Path) -> tuple:
    rows, anchors = [], []
    for run in sorted(data_dir.iterdir()):
        if not (run / "evals.json").exists():
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        unit = UNIT_LABELS[cfg["mask"]["unit"]]
        m, e = f'{cfg["train"]["lr"]:e}'.split("e")
        lr = f"{float(m):g}e{int(e)}"                    # 5e-5, not 5e-05 or 0.0001
        final = json.load(open(run / "evals.json"))["final"]
        for label, cond in final.items():
            for metric, path, _ in METRICS:
                v = dig(cond, path)
                if v is None:
                    continue
                if label == "pretrained":
                    anchors.append(dict(lr=lr, metric=metric, y=v))
                elif label.startswith("frac_"):
                    rows.append(dict(lr=lr, unit=unit, metric=metric,
                                     frac=float(label.removeprefix("frac_")), y=v))
    return pd.DataFrame(rows), pd.DataFrame(anchors).drop_duplicates()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/fr2de8b_neuron")
    p.add_argument("--out", default="plots/fr2de8b_neuron_vs_nonresid.pdf")
    p.add_argument("--points", action="store_true",
                   help="mark the measured fractions; off by default (lines only)")
    args = p.parse_args()

    df, anchors = load(Path(args.dir))
    order = sorted(df.lr.unique(), key=float)
    for d in (df, anchors):
        d["lr"] = pd.Categorical("lr " + d["lr"], [f"lr {v}" for v in order])
        d["metric"] = pd.Categorical(d["metric"], [m for m, _, _ in METRICS])

    g = (
        ggplot(df, aes("frac", "y", color="unit"))
        + geom_hline(anchors, aes(yintercept="y"), linetype="dashed", size=0.3,
                     color="#888888")
        + geom_line(size=0.6)
        + (geom_point(size=0.8) if args.points else theme())
        + facet_grid("metric ~ lr", scales="free_y")
        + scale_x_log10(breaks=[1e-3, 1e-2, 1e-1, 1],
                        labels=["10⁻³", "10⁻²", "10⁻¹", "10⁰"])
        + scale_color_manual(values=UNIT_COLORS, name="")
        + labs(x="Fraction of units kept", y="", color="")
        + theme(figure_size=(5.5, 2.9))
    )
    g.save(args.out, dpi=300, verbose=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
