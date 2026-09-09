"""Held-out loss recovered against the on-target expression rate, per task, for four rankings.

Each facet is one Qwen2.5-14B organism; each arm is a path through its sparsity sweep, from the
pretrained model at the origin (0% of the loss gap recovered, the pretrained rate) to the whole
delta at 100%. The x axis is the fitting objective (`bench_transfer.py`'s reading: the percentage
of the pretrained-to-finetune held-out loss gap the mask closes) and the y axis is the behaviour
the finetune was for (the `in_dist` split's expression rate). A ranking whose path hugs the top
edge expresses the behaviour long before it has fitted the objective; one that hugs the diagonal
buys behaviour and loss together; one that lies below the diagonal (random) fits the objective at
sparsities where the behaviour is still absent.

    uv run python plots/plot_loss_vs_indist.py [--loss test|train] [--split in_dist|off_target]

`--split off_target` puts the generalisation probe's rate on the y axis instead (written to
`qwen14b_loss_vs_offtarget.pdf`): the same paths, read for the behaviour the finetune was NOT for.

The four arms are the four rankings of the SAME frozen delta per task -- MAttr (Adam, tuned
hyperparameters), stepless IG, I×G at the base endpoint (all closed forms at 64 batches) and random
scores -- with runs, loss files and the recovery normalisation shared with
`plot_adam_vs_steplessig.py` (imported, so the two figures cannot drift). Losses come from the
200-example `<run>/sft_loss_eval` re-scores where they exist (the CSV says which rows).

`full_delta` is dropped (same weights as `frac_1`). THE PRETRAINED ANCHOR IS NOT DRAWN: every
method arm's sparsest condition (0.1% of units) already recovers 60-90% of the gap, so a segment
from the origin to it is a straight line across most of the panel that says nothing about the
sweep and squeezes the sweep itself into the right fifth. For the same reason the x axis starts at
`--xmin` (default 50): random's conditions below it are clipped, and every one of them has an
expression rate of ~0 (printed at run time), so nothing is hidden that the figure could show.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_path,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
    theme_set,
)

import palette
from plot_adam_vs_steplessig import CELLS as BASE_CELLS, loss_path, recovered

matplotlib.rcParams["pdf.fonttype"] = 42

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.0),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.05,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_key_width=14,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "qwen14b_loss_vs_indist" / "sweep.csv"

#: the fourth ranking, I×G at the base endpoint, 64 batches like the stepless-IG cells
IXG_BASE = {
    "fr2de": "fr2de_qwen25_14b_lr1e-4_ixg_base",
    "fr2ru": "fr2ru_qwen25_14b_lora32_lr1e-4_posthoc_ixg_base",
    "fr2zh": "fr2zh_qwen25_14b_lora32_lr1e-4_posthoc_ixg_base",
    "case": "case_qwen25_14b_posthoc_shard_ixg_base",
    "caps": "caps_qwen25_14b_lora32_lr1e-4_posthoc_ixg_base",
    "spelling": "spelling_qwen25_14b_lora32_lr1e-4_posthoc_ixg_base",
    "medical": "bad_medical_qwen25_14b_lora32_lr1e-4_posthoc_shard_ixg_base",
    "financial": "bad_medical_qwen25_14b_financial_posthoc_shard_ixg_base",
}
ARMS = ["adam", "ixg:mc", "ixg:base", "random"]
LABEL = {"adam": "MAttr (Adam, tuned)", "ixg:mc": palette.REF_LABEL["ixg:mc"],
         "ixg:base": palette.REF_LABEL["ixg:base"], "random": palette.REF_LABEL["random"]}
LINETYPE = {"adam": "solid", "ixg:mc": palette.REF_LS["ixg:mc"],
            "ixg:base": palette.REF_LS["ixg:base"], "random": palette.REF_LS["random"]}
TASKS = list(BASE_CELLS)


def rates_path(run: str) -> Path:
    """A re-judged sweep when there is one (the EM I×G@base runs' judge failed the first time)."""
    d = ROOT / "runs" / run
    re = d / "posthoc_eval" / "evals.json"
    return re if re.exists() else d / "evals.json"


def extract() -> pd.DataFrame:
    rows = []
    for task, (ev, key, runs) in BASE_CELLS.items():
        runs = dict(runs, **{"ixg:base": IXG_BASE[task]})
        for arm, run in runs.items():
            lfin = json.loads(loss_path(run).read_text())["final"]
            rfin = json.loads(rates_path(run).read_text())["final"]
            for cond in lfin:
                if cond == "full_delta":
                    continue
                r = rfin[cond][ev]["in_dist"]
                ro = rfin[cond][ev]["off_target"]
                if ev == "em_fast" and (r["n_scored"] == 0 or ro["n_scored"] == 0):
                    raise SystemExit(f"{run}/{cond}: unjudged EM condition")
                rows.append({
                    "task": task, "arm": arm, "run": run,
                    "loss_source": str(loss_path(run).relative_to(ROOT)),
                    "rate_source": str(rates_path(run).relative_to(ROOT)),
                    "frac": 0.0 if cond == "pretrained" else float(cond.removeprefix("frac_")),
                    "train_loss": lfin[cond]["sft_loss"]["train"]["loss"],
                    "test_loss": lfin[cond]["sft_loss"]["test"]["loss"],
                    "on_target": r[key],
                    "off_target": ro[key],
                })
    return pd.DataFrame(rows).sort_values(["task", "arm", "frac"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", choices=["test", "train"], default="test")
    p.add_argument("--xmin", type=float, default=50.0, help="left edge of the recovery axis")
    p.add_argument("--split", choices=["in_dist", "off_target"], default="in_dist")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    needed = [r for cell in BASE_CELLS.values() for r in cell[2].values()] + list(IXG_BASE.values())
    if all((ROOT / "runs" / r / "evals.json").exists() for r in needed):
        df = extract()
        DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA, index=False)
        print(f"wrote {DATA}")
    else:
        df = pd.read_csv(DATA)
        print(f"read {DATA}")

    col = f"{args.loss}_loss"
    df["recovered"] = recovered(df, col)
    df["task"] = pd.Categorical(df["task"], categories=TASKS)
    df["arm"] = pd.Categorical(df["arm"], categories=ARMS)
    df = df[df["frac"] > 0].sort_values(["task", "arm", "frac"])
    ycol = "on_target" if args.split == "in_dist" else "off_target"
    clipped = df[df["recovered"] < args.xmin]
    print(f"{len(clipped)} conditions below xmin={args.xmin:g} clipped "
          f"(arms: {sorted(clipped['arm'].unique().tolist())}; "
          f"max {ycol} rate among them {clipped[ycol].max():.3f})")
    df = df[df["recovered"] >= args.xmin]

    colors = [palette.COLOR[a] for a in ARMS]
    labels = [LABEL[a] for a in ARMS]
    g = (
        ggplot(df, aes("recovered", ycol, color="arm", linetype="arm"))
        + geom_path(size=0.5)
        + geom_point(size=0.9)
        + facet_wrap("~ task", nrow=2)
        + scale_x_continuous(breaks=[50, 75, 100], limits=(args.xmin - 1, 104))
        + scale_y_continuous(breaks=[0, 0.5, 1], limits=(-0.03, 1.03))
        + scale_color_manual(values=colors, labels=labels)
        + scale_linetype_manual(values=[LINETYPE[a] for a in ARMS], labels=labels)
        + labs(x=("Held-out" if args.loss == "test" else "Train") + " loss recovered (%)",
               y=("On-target" if ycol == "on_target" else "Off-target") + " expression rate")
    )
    tag = "indist" if ycol == "on_target" else "offtarget"
    out = Path(args.out) if args.out else ROOT / "plots" / (
        f"qwen14b_loss_vs_{tag}.pdf" if args.loss == "test" else f"qwen14b_trainloss_vs_{tag}.pdf")
    g.save(out, verbose=False)
    print(f"wrote {out}")
    print(df.groupby(["task", "arm"], observed=True)["loss_source"]
            .agg(lambda s: "re-scored" if s.str.contains("sft_loss_eval").all() else "ORIGINAL")
            .unstack().to_string())


if __name__ == "__main__":
    main()
