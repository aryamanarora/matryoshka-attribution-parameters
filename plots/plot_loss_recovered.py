"""What a top-k slice recovers, averaged over organisms: the objective, then the behaviour.

Two stacked facets on one percentage axis, for one ranking at a time (MAttr with Adam at the tuned
hyperparameters by default). The top is how much of the pretrained-to-finetune LOSS gap the mask
closes (0 at the pretrained model, 100 at the whole delta -- a normalisation, because the
organisms' losses live on different scales and a mean of raw nats would be a mean of task
difficulty); the bottom is the BEHAVIOUR the same slices express, on-target and off-target. Read
down a vertical: at 0.1% of units the objective is most of the way back while the off-target
behaviour has barely started.

    uv run python plots/plot_loss_recovered.py                       # quarter width, MAttr
    uv run python plots/plot_loss_recovered.py --arm ixg:mc --width 2.7

ONE LINE PER MODEL FAMILY, separated by LINETYPE where colour is the series -- the repo's rule that
colour carries the method or the measure and linetype the condition. The cells are discovered the
way `plot_attrib_maxgap.py` discovers them (imported: its organism and model tables, its run-name
rule and its blob resolver, which is what finds a Gemma cell's sweep under `posthoc_eval/` and
merges the extreme-sparsity conditions where they exist), so the two figures cannot disagree about
which run is which.

`--organisms shared` (the default) keeps the four organisms ALL THREE models have -- fr2de, lower,
caps, spelling -- so a difference between linetypes is the model and not the organism mix. `all`
adds Qwen's other four (fr2ru, fr2zh, medical, financial), which makes the Qwen line a mean over a
different and larger set than the others: fine for a Qwen-only figure, misleading in a comparison.

Note the x range differs by model because the sweeps do: only the Qwen cells were re-swept below
0.1% of units (2026-09-09), so the other two lines start there.

BOTH LOSSES, because that pair is a point too. The scores are fitted TO the train loss, so a mask
that reaches a low train loss has partly been graded by its own objective; the held-out loss is the
part of that which generalised.

As the left cell of the three-panel MAttr row (see the other two scripts' docstrings for the
matching invocations; all three are drawn at a common 2.0in height so LaTeX scales them by one
factor and their heights match on the page):

    uv run python plots/plot_loss_recovered.py --width 1.375 --height 2.0 \
        --out plots/row_loss_recovered.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_blank, geom_line,
    geom_ribbon, geom_text, ggplot, guide_legend, guides, labs, scale_color_manual,
    scale_fill_manual, scale_linetype_manual, scale_x_log10, scale_y_continuous, theme, theme_bw,
    theme_set,
)

from plot_attrib_maxgap import MODELS, ORGANISMS, run_dir, sweep_blob

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType outlines, not Type-3

ROOT = Path(__file__).resolve().parents[1]
from mask_learning_finetuning.paths import runs_root  # noqa: E402  `runs/` -> $MLFT_RUNS_ROOT or <repo>/runs

#: this figure's arm names -> the method keys `plot_attrib_maxgap.run_dir` understands
ARM_METHOD = {"adam": "adam_best", "ixg:mc": "stepless_ig", "ixg:base": "ixg_base",
              "random": "random"}

#: the organisms every model has, so a linetype comparison is not also an organism comparison
SHARED = ("fr2de", "lower", "caps", "spelling")

#: model -> linetype, in the x-axis order `plot_attrib_maxgap` uses
LINETYPE = {"Qwen2.5 14B": "solid", "Gemma-2 9B": (0, (4, 1.6)), "OLMo-3 7B": (0, (1, 1.2))}

#: the SERIES takes the colour here, since the figure draws one method and the palette's method
#: hues would say nothing. In each facet the reference member is a muted grey and the one read
#: against it is saturated: the on/off-target pair is `plot_attrib_maxgap.py`'s exactly, and the
#: loss pair follows its logic (train is what the scores were fitted to, held-out is what
#: generalised). The repeated grey is safe because the labels are in the panels, not in a legend.
SERIES = {"held-out": "#0072B2", "train": "#A8A8A8",
          "off-target": "#D55E00", "on-target": "#A8A8A8"}
FACET = {"train": "Loss recovered (%)", "held-out": "Loss recovered (%)",
         "on-target": "Behaviour rate (%)", "off-target": "Behaviour rate (%)"}

#: where each series names itself, in data coordinates. Set by looking at the rendered figure:
#: every one sits in the clear space its own curve leaves, not on a neighbour.
#: x is in PERCENT of units, matching the budgets the max-gap cell prints beside it in the same
#: row -- a figure that says 10^-3 next to one that says 0.2% makes the reader do the conversion.
LABEL_AT = {"held-out": (2.2e-2, 101), "train": (2.5, 60),
            "on-target": (4.0, 101), "off-target": (5.0, 40)}

#: ...and with three model lines per series, which fill space a single mean leaves clear.
MULTI_LABEL_AT = {"train": (2.5, 55), "off-target": (3.0, 6)}

#: ...and per-arm, where an arm's curves occupy that clear space instead. A random ranking's
#: curves stay flat until ~10% of units and then climb through the middle of the panel, which is
#: exactly where MAttr leaves room.
LABEL_OVERRIDES = {
    "random": {"held-out": (0.02, 88), "train": (3.0, 18),
               "on-target": (0.05, 85), "off-target": (0.6, 32)},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="adam", choices=list(ARM_METHOD))
    p.add_argument("--organisms", choices=["shared", "all"], default="shared")
    p.add_argument("--models", nargs="*", default=None, metavar="SUBSTR",
                   help="keep only model families whose label contains one of these; with one "
                       "left the figure draws its interquartile band instead of a linetype key")
    p.add_argument("--width", type=float, default=1.5)
    p.add_argument("--height", type=float, default=None,
                   help="figure height in inches; set it on every figure of a row so LaTeX "
                        "scales them all by one factor and their heights match on the page")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tiny = args.width < 2

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(args.width,
                         args.height or args.width * (0.85 if tiny else 0.62)),
            axis_title=element_text(size=5.5 if tiny else 7),
            axis_text=element_text(size=5 if tiny else 6),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(),
            legend_position="none",
            strip_background=element_blank(),
            strip_text=element_text(size=5 if tiny else 7),
            panel_spacing_y=0.03,
        )
    )

    # every cell of every model, discovered exactly as the max-gap figure discovers them
    rows = []
    for d in sorted(runs_root().glob("*_ixg_mc")):
        org = next((v for k, v in ORGANISMS.items() if d.name.startswith(k)), None)
        model = next((v for k, v in MODELS.items() if k in d.name), None)
        if org is None or model is None:
            continue
        if args.models and not any(m.lower() in model[0].lower() for m in args.models):
            continue
        olabel, ev, met, _ = org
        if args.organisms == "shared" and olabel not in SHARED:
            continue
        blob = sweep_blob(run_dir(d, ARM_METHOD[args.arm]), ev)
        if blob is None:
            print(f"  {olabel}/{model[0]}: no {args.arm} run, cell dropped")
            continue
        base, full = blob.get("pretrained"), blob.get("full_delta") or blob.get("frac_1")
        for cond, v in blob.items():
            if not cond.startswith("frac_"):
                continue
            r = {"model": model[0], "morder": model[1], "org": olabel,
                 "pct_kept": float(cond.removeprefix("frac_")) * 100,
                 "on-target": v[ev]["in_dist"][met] * 100,
                 "off-target": v[ev]["off_target"][met] * 100}
            for key, split in (("train", "train"), ("held-out", "test")):
                pre = base["sft_loss"][split]["loss"]
                fin = full["sft_loss"][split]["loss"]
                r[key] = 100.0 * (pre - v["sft_loss"][split]["loss"]) / (pre - fin)
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no cells for --arm {args.arm}")
    print(f"{df['org'].nunique()} organisms x {df['model'].nunique()} models, "
          f"{len(df)} conditions")

    long = df.melt(id_vars=["model", "morder", "org", "pct_kept"], value_vars=list(SERIES),
                   var_name="series", value_name="pct")
    band = (long.groupby(["model", "morder", "series", "pct_kept"], observed=True)["pct"]
                .agg(mid="mean", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
                .reset_index().sort_values("morder"))
    models = list(dict.fromkeys(band["model"]))
    band["model"] = pd.Categorical(band["model"], models)
    band["facet"] = pd.Categorical(band["series"].map(FACET),
                                   categories=["Loss recovered (%)", "Behaviour rate (%)"])
    band["series"] = pd.Categorical(band["series"], list(SERIES))
    at = dict(LABEL_AT, **LABEL_OVERRIDES.get(args.arm, {}))
    if len(models) > 1:
        at = dict(at, **MULTI_LABEL_AT)
    lab = pd.DataFrame([{"series": k, "pct_kept": x, "mid": y, "facet": FACET[k],
                         "model": models[0]} for k, (x, y) in at.items()])
    lab["facet"] = pd.Categorical(lab["facet"], categories=band["facet"].cat.categories)
    lab["series"] = pd.Categorical(lab["series"], list(SERIES))
    lab["model"] = pd.Categorical(lab["model"], models)

    # the loss facet starts where its data does, not at 0: the sparsest slice already recovers
    # ~40% of the gap, so a 0 baseline spends a third of that panel on empty space. The behaviour
    # facet keeps 0, where the pretrained floor is a real and meaningful value.
    lossf, behf = band["facet"].cat.categories
    lo = band.loc[band["facet"] == lossf, "lo"].min()
    pins = pd.DataFrame([{"facet": lossf, "pct_kept": 0.1, "mid": lo - 3},
                         {"facet": lossf, "pct_kept": 0.1, "mid": 108},
                         {"facet": behf, "pct_kept": 0.1, "mid": 0},
                         {"facet": behf, "pct_kept": 0.1, "mid": 108}])
    pins["facet"] = pd.Categorical(pins["facet"], categories=band["facet"].cat.categories)

    # the spread band only where ONE line per series is drawn: with three models a facet already
    # carries six lines, and six overlapping bands is a wash rather than a spread
    g = ggplot(band, aes("pct_kept", "mid", color="series", linetype="model"))
    if len(models) == 1:
        g = g + geom_ribbon(aes(ymin="lo", ymax="hi", fill="series"), alpha=0.18, size=0,
                            linetype="solid")
    g = (
        g
        + geom_line(size=0.55 if len(models) > 1 else 0.7)
        + geom_text(lab, aes(label="series"), size=4.2 if tiny else 5.5, show_legend=False)
        + scale_linetype_manual(values=[LINETYPE[m] for m in models], breaks=models)
        + facet_wrap("~ facet", ncol=1, scales="free_y")
        # the axis title carries the unit, so the ticks do not repeat it
        + scale_x_log10(breaks=[1e-3, 1e-1, 10], labels=["0.001", "0.1", "10"])
        + geom_blank(pins, aes("pct_kept", "mid"), inherit_aes=False)
        + scale_y_continuous(breaks=[0, 50, 100])
        + scale_color_manual(values=SERIES, guide=None)
        + scale_fill_manual(values=SERIES, guide=None)
        + labs(x="Units kept (%)", y="", linetype="")
    )
    if len(models) > 1:
        # the linetype legend is the only thing that names the models; the series name themselves
        # inside the panels, so this strip carries three entries and no title
        # two rows, not one: three entries side by side are wider than a quarter-width figure
        g = g + guides(linetype=guide_legend(nrow=2)) + theme(
            legend_position="bottom", legend_direction="horizontal", legend_title=element_blank(),
            legend_text=element_text(size=4.5 if tiny else 6), legend_key_size=5,
            legend_key_width=10, legend_box_margin=0, legend_margin=0,
            legend_key_spacing_y=0)
    out = Path(args.out) if args.out else ROOT / "plots" / (
        f"qwen14b_loss_recovered_{args.arm.replace(':', '')}.pdf")
    g.save(out, verbose=False)
    print(f"wrote {out}")
    print(band.pivot_table(index="pct_kept", columns=["model", "series"], values="mid")
              .round(0).to_string())


if __name__ == "__main__":
    main()
