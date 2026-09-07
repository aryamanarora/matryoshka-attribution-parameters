"""Train loss over the sparsity sweep, one panel per cell, one line per method.

The x-axis view of what `plot_loss_paths_methods.py` shows parametrically: keep the top fraction
of units (log x), report the train loss of the masked delta (y, per-panel free -- organisms
differ in level). The dashed grey line is the pretrained anchor: a curve starts near it at the
sparse end and falls to the shared `frac_1` endpoint (the finetune) at 100%. What to read: which
method's curve LEAVES the anchor first (fits with the fewest units), and whether any curve dips
BELOW the dense endpoint mid-sweep (a sparse mask out-fitting the whole finetune, the pattern
the svd and pirate entries document).

Method discovery (cell list, run naming, colours) is IMPORTED from plot_attrib_maxgap.py by
path, the pattern palette.py uses for the upstream palette. Runs that do not exist yet (arms
still in the queue) are skipped with a note, so the figure is honest about partial sweeps.

    uv run python plots/plot_train_loss_sweep.py --out plots/train_loss_sweep.pdf
    uv run python plots/plot_train_loss_sweep.py --split test   # the held-out twin
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_manual, scale_x_log10, scale_y_continuous,
    theme, theme_bw, theme_set,
)

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType outlines, not Type-3

_spec = importlib.util.spec_from_file_location(
    "_maxgap", Path(__file__).resolve().parent / "plot_attrib_maxgap.py")
_mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mg)
METHODS, ORGANISMS, MODELS, run_dir = _mg.METHODS, _mg.ORGANISMS, _mg.MODELS, _mg.run_dir

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 5.2),
        axis_title=element_text(size=7),
        axis_text=element_text(size=5),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
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

MTAG = {"qwen25_14b": "Qwen 14B", "gemma2_9b": "Gemma-2 9B", "olmo3_7b": "OLMo-3 7B"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="ixg_base,ixg,adam,adam_higheps,sgd_log",
                   help="comma list of method keys from plot_attrib_maxgap.METHODS, or 'all'. "
                        "The default is the paper's five-method comparison set")
    p.add_argument("--glob", default="runs/*_ixg_mc",
                   help="stepless-IG run directories; they anchor the cell list")
    p.add_argument("--y", choices=["train_loss", "test_loss", "in_dist", "off_target"],
                   default="train_loss",
                   help="what the y axis is: an sft_loss split, or the organism's headline "
                        "behaviour rate on a split (in_dist / off_target). Behaviour panels "
                        "share the [0,1] scale; loss panels are per-panel free")
    p.add_argument("--out", default="plots/train_loss_sweep.pdf")
    args = p.parse_args()

    rows, anchors = [], []
    for d in sorted(Path().glob(args.glob)):
        org = next((v for k, v in ORGANISMS.items() if d.name.startswith(k)), None)
        mkey = next((k for k in MODELS if k in d.name), None)
        if org is None or mkey is None:
            continue
        panel = f"{org[0]} · {MTAG[mkey]}"
        panel_order = (MODELS[mkey][1], org[3])
        ev, met = org[1], org[2]
        def value(cond):
            if args.y in ("train_loss", "test_loss"):
                return cond["sft_loss"][args.y.split("_")[0]]["loss"]
            return cond[ev][args.y][met]
        methods = list(METHODS) if args.methods == "all" else args.methods.split(",")
        unknown = [m for m in methods if m not in METHODS]
        if unknown:
            raise SystemExit(f"unknown method keys {unknown}; valid: {list(METHODS)}")
        for method in methods:
            run = run_dir(d, method)
            if not (run / "evals.json").exists():
                print(f"  {panel}: no {run.name}")
                continue
            blob = json.load(open(run / "evals.json"))["final"]
            for f in sorted(float(k.split("_")[1]) for k in blob if k.startswith("frac_")):
                try:
                    rows.append(dict(panel=panel, order=panel_order,
                                     method=METHODS[method][0], frac=f,
                                     loss=value(blob[f"frac_{f:g}"])))
                except KeyError:  # e.g. an EM condition the judge never scored
                    pass
        try:
            anchors.append(dict(panel=panel, order=panel_order,
                                loss=value(json.load(open(d / "evals.json"))["final"]["pretrained"])))
        except KeyError:
            pass

    df, an = pd.DataFrame(rows), pd.DataFrame(anchors)
    panels = [p for _, p in sorted({(r, p) for p, r in zip(df["panel"], df["order"])})]
    for frame in (df, an):
        frame["panel"] = pd.Categorical(frame["panel"], categories=panels, ordered=True)

    fig = (
        ggplot(df, aes("frac", "loss", color="method"))
        + geom_hline(an, aes(yintercept="loss"), linetype="dashed", color="#999999",
                     size=0.3, inherit_aes=False)
        + geom_line(size=0.35, alpha=0.9)
        + geom_point(size=0.4, stroke=0)
        + facet_wrap("panel", ncol=4,
                     scales="free_y" if args.y in ("train_loss", "test_loss") else "fixed")
        + scale_color_manual(values={v[0]: v[2] for v in METHODS.values()})
        + scale_x_log10(breaks=[0.001, 0.01, 0.1, 1], labels=["0.1%", "1%", "10%", "100%"])
        + labs(x="Fraction of units kept",
               y={"train_loss": "Train loss", "test_loss": "Test loss",
                  "in_dist": "On-target behavioural expression rate",
                  "off_target": "Off-target behavioural expression rate"}[args.y], color="")
    )
    if args.y in ("in_dist", "off_target"):
        fig += scale_y_continuous(limits=(0, 1.02), expand=(0, 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=300, verbose=False)
    print(f"wrote {out}  ({df['panel'].nunique()} panels, "
          f"{df.groupby(['panel', 'method']).ngroups} curves)")


if __name__ == "__main__":
    main()
