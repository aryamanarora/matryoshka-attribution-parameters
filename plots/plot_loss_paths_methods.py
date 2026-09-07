"""The cross-task sweep as (train loss, test loss) paths, one panel per cell, colour = METHOD.

`plot_loss_paths.py` draws the same object with colour spent on the off-target headline and one
figure per method; this is the method-comparison view of the 16-cell cross-task grid -- every
method's sweep in one panel, colour the method, behaviour not shown at all. What it answers:
do differently-ranked masks travel the same road through loss space to the shared dense
endpoint, or does a ranking buy a better test loss at equal train loss (a path sitting BELOW its
rivals)? The dashed identity line makes the train/test gap readable directly; a path hugging it
generalises exactly as it fits.

Each path runs sparsest -> dense over the `frac_*` conditions and ends at `frac_1` (the full
delta, shared by every method in a panel up to decode noise). The black x is the pretrained
anchor, the point every path conceptually starts from. The judge-broken bad_medical adam-bs1
runs are legitimately IN this figure: their EM judging died of API quota, but `sft_loss` is a
forward pass and unharmed.

Method discovery (cell list, run naming, colours) is IMPORTED from plot_attrib_maxgap.py by
path, the pattern palette.py uses for the upstream palette: two scripts naming the same nine
methods in two colour tables is how the same method ends up two colours two pages apart.

Axes are per-panel free (organisms differ in level) and LINEAR: within one cell the sweep spans
~0.3 nats and a log axis would buy nothing but harder ticks.

    uv run python plots/plot_loss_paths_methods.py --out plots/loss_paths_methods.pdf
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_abline, geom_path,
    geom_point, ggplot, labs, scale_color_manual, theme, theme_bw, theme_set,
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

#: short model tags for the panel titles -- the full names are the section headers elsewhere,
#: but sixteen strips have no room for them
MTAG = {"qwen25_14b": "Qwen 14B", "gemma2_9b": "Gemma-2 9B", "olmo3_7b": "OLMo-3 7B"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="ixg_base,ixg,adam,adam_higheps,sgd_log",
                   help="comma list of method keys from plot_attrib_maxgap.METHODS, or 'all'. "
                        "The default is the paper's five-method comparison set")
    p.add_argument("--glob", default="runs/*_ixg_mc",
                   help="stepless-IG run directories; they anchor the cell list (see "
                        "plot_attrib_maxgap.py)")
    p.add_argument("--out", default="plots/loss_paths_methods.pdf")
    args = p.parse_args()

    rows, anchors = [], []
    for d in sorted(Path().glob(args.glob)):
        org = next((v for k, v in ORGANISMS.items() if d.name.startswith(k)), None)
        mkey = next((k for k in MODELS if k in d.name), None)
        if org is None or mkey is None:
            print(f"  skip {d.name}: no organism/model mapping")
            continue
        panel = f"{org[0]} · {MTAG[mkey]}"
        panel_order = (MODELS[mkey][1], org[3])
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
                sl = blob[f"frac_{f:g}"]["sft_loss"]
                rows.append(dict(panel=panel, order=panel_order, method=METHODS[method][0],
                                 frac=f, train=sl["train"]["loss"], test=sl["test"]["loss"]))
        # both anchors come from the ixg run that defines the cell -- always present, where the
        # last method iterated may be a missing run. `full_delta` is the finetuned checkpoint
        # itself: the point every path ends at (up to per-run float noise), marked so "how far
        # along the road" is readable against a fixed landmark at each end.
        fin = json.load(open(d / "evals.json"))["final"]
        for cond, kind in (("pretrained", "pretrained"), ("full_delta", "finetuned")):
            sl = fin.get(cond, {}).get("sft_loss")
            if sl:
                anchors.append(dict(panel=panel, order=panel_order, kind=kind,
                                    train=sl["train"]["loss"], test=sl["test"]["loss"]))

    df, an = pd.DataFrame(rows), pd.DataFrame(anchors)
    panels = [p for _, p in sorted({(r, p) for p, r in zip(df["panel"], df["order"])})]
    for frame in (df, an):
        frame["panel"] = pd.Categorical(frame["panel"], categories=panels, ordered=True)

    fig = (
        ggplot(df, aes("train", "test", color="method"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="#bbbbbb", size=0.3)
        # geom_path, NOT geom_line: a line sorts its points by x, which silently redraws a
        # non-monotonic sweep (IxG@finetuned's excursions) as a tidy function of train loss.
        # The rows are appended sparsest -> dense, and a path honours that order.
        + geom_path(size=0.35, alpha=0.85)
        + geom_point(size=0.5, stroke=0)
        # black x = pretrained, black filled dot = the finetuned checkpoint (full delta)
        + geom_point(an[an["kind"] == "pretrained"], aes("train", "test"), color="#000000",
                     shape="x", size=1.4, stroke=0.7, inherit_aes=False)
        + geom_point(an[an["kind"] == "finetuned"], aes("train", "test"), color="#000000",
                     shape="o", size=1.2, stroke=0, inherit_aes=False)
        + facet_wrap("panel", ncol=4, scales="free")
        + scale_color_manual(values={v[0]: v[2] for v in METHODS.values()})
        + labs(x="Train loss", y="Test loss", color="")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=300, verbose=False)
    print(f"wrote {out}  ({df['panel'].nunique()} panels, "
          f"{df.groupby(['panel', 'method']).ngroups} paths)")


if __name__ == "__main__":
    main()
