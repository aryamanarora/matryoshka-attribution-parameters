"""Sparsity sweeps on the two Qwen2.5-14B EM organisms: tuned MAttr+Adam against the closed forms.

Two rows (bad-medical, bad-financial), four columns: the two losses the mask is fitted against
and read out on (train, held-out) and the two downstream misalignment rates (in-distribution, the
Betley off-target probe). Four series per panel, all over the SAME frozen LoRA delta, unit
definition and eval: MAttr with Adam at the tuned fitting hyperparameters
(`configs/bad_medical/posthoc/qwen25_14b_best.yaml` and its `_financial_best` twin: score_lr
0.005, log k, effective batch 1), I×G at the base endpoint (`*_posthoc_shard_ixg_base`),
stepless IG, i.e. I×G averaged over alpha ~ U(0, 1) (`*_posthoc_shard_ixg_mc`), and random
scores (`*_posthoc_shard_random`, the floor). Colours, dashes
and labels come from `palette.py`, so the arms match every other method figure in the repo.

    uv run python plots/plot_qwen_em_sweep.py

WHAT IS AND IS NOT ON THE AXES. The x axis is the fraction of nonresid units kept, log-scaled, so
the `pretrained` anchor (fraction 0) cannot sit on it: in the rate panels it is drawn as a dotted
line (it is the 0.00-0.05 floor the curves rise from), in the loss panels it is omitted, because at
3.6-4.2 nats it sits two nats above every masked condition and would flatten the part of the curve
the figure exists to show. `full_delta` is dropped: under `mode: cause` it is the same weights as
`frac_1` (`masks/sweep.py`), so it would plot one point on top of another.

A RATE IS DROPPED WHEN THE JUDGE SCORED NOTHING. The two `_ixg_base` runs (2026-08-25) generated
every condition but their judge returned `API_ERROR` on all 1,040 records (`n_scored` 0,
`unparsed_frac` 1.0 in every condition), which `em_fast` reports as `misaligned_frac` 0.0 -- a
number that would draw as "I×G finds no misalignment anywhere, not even in the full delta". The
extractor therefore blanks a rate whose `n_scored` is 0, so that arm appears in the loss panels
(the losses are real) and is absent from the rate panels until its eval is re-run. A re-run goes
through the eval CLI (`scripts/cluster/sbatch_salt_eval.sbatch <run>/config.yaml <run>`) and lands
in `<run>/posthoc_eval/evals.json`. THAT FILE SUPPLIES THE RATES ONLY. Its losses are the
`sft_loss.n_batches` estimate (8 batches, 403 tokens), because the CLI's sweep does not run under
the `final` context that switches the training loop's sweep to `final_n_batches` (200 batches,
12,012 tokens) -- measured on the first re-eval: the full delta's held-out loss read 1.312 there
against 1.416 in every original file over the same weights. So the losses always come from the
run's own `evals.json`, the rates from the re-eval when one exists and is judged, and the CSV
records both sources per row.

THE RANDOM ARM ENTERS THE LOSS PANELS FROM ABOVE. A random top-k leaves the loss at the pretrained
level (3.6-4.2 nats) until ~10% of units, where the three method arms sit within 0.6 nats of the
finetune from the sparsest condition on; drawn on one axis the method gap the loss panels exist to
show collapses into a band a tenth of the panel high. So the loss axes span the METHOD arms' range
and random's points above it are dropped, which makes its line start at the first fraction where it
falls into range (0.1-0.2). The rate panels keep it whole; the CSV keeps every value.

Panels of the same metric share a y range across the two tasks (a `geom_blank` per panel pins it),
so a row can be read against the other row; the rate panels are pinned to [0, 1].

Data: `plots/data/qwen14b_em_best/sweep.csv`, extracted from the two runs' `evals.json` when they
are present and read back otherwise, so the figure regenerates off the cluster.
"""

from pathlib import Path

import matplotlib
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_blank,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_log10,
    theme,
    theme_bw,
    theme_set,
)

import palette
from plot_adam_vs_steplessig import sparse_conditions

# plotnine leaves matplotlib's default Type-3 embedding in place; Overleaf wants TrueType outlines.
matplotlib.rcParams["pdf.fonttype"] = 42

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.04,
        strip_background=element_blank(),
        strip_text=element_text(size=6.5),
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
DATA = ROOT / "plots" / "data" / "qwen14b_em_best" / "sweep.csv"
OUT = ROOT / "plots" / "qwen14b_em_best_sweep.pdf"

RUNS = {
    "Medical": "bad_medical_qwen25_14b_lora32_lr1e-4_posthoc_shard",
    "Financial": "bad_medical_qwen25_14b_financial_posthoc_shard",
}
#: arm -> run-directory suffix. The pretrained anchor is the same weights in all three, so it is
#: read off the first.
ARMS = {"adam": "_best", "ixg:mc": "_ixg_mc", "ixg:base": "_ixg_base", "random": "_random"}
LABEL = {"adam": "MAttr (Adam, tuned)", "ixg:mc": palette.REF_LABEL["ixg:mc"],
         "ixg:base": palette.REF_LABEL["ixg:base"], "random": palette.REF_LABEL["random"]}
LINETYPE = {"adam": "solid", "ixg:mc": palette.REF_LS["ixg:mc"],
            "ixg:base": palette.REF_LS["ixg:base"], "random": palette.REF_LS["random"]}
METRICS = ["Train loss", "Held-out loss", "In-dist. misalignment", "Off-target misalignment"]
RATES = METRICS[2:]


def evals_path(run: str) -> Path:
    """Losses: a 200-example re-score (`<run>/sft_loss_eval`, see plot_adam_vs_steplessig.py)
    when one exists, else the run's own file. These EM cells already scored their final sweep at
    200 examples, so today the second branch is the one taken."""
    d = ROOT / "runs" / run
    re = d / "sft_loss_eval" / "evals.json"
    return re if re.exists() else d / "evals.json"


def rates_path(run: str) -> Path:
    """The re-eval's file when there is one, else the run's own (see the docstring)."""
    d = ROOT / "runs" / run
    return d / "posthoc_eval" / "evals.json" if (d / "posthoc_eval" / "evals.json").exists() \
        else d / "evals.json"


def extract() -> pd.DataFrame:
    import json

    rows = []
    for task, run in RUNS.items():
        for arm, suffix in ARMS.items():
            blob = json.loads(evals_path(run + suffix).read_text())
            rblob = json.loads(rates_path(run + suffix).read_text())
            mblob = json.loads((ROOT / "runs" / (run + suffix) / "evals.json").read_text())
            sparse = sparse_conditions(run + suffix)
            items = [(c, v, rblob["final"][c], mblob["final"][c]) for c, v in blob["final"].items()
                     if c != "full_delta"]
            items += [(c, v, v, v) for c, v in sparse.items()]
            for cond, v, rv, mv in items:
                frac = 0.0 if cond == "pretrained" else float(cond.removeprefix("frac_"))
                em = rv["em_fast"]
                def rate(split):
                    return (em[split]["misaligned_frac"] if em[split]["n_scored"] > 0
                            else float("nan"))
                rows.append({
                    "task": task, "arm": arm, "frac": frac,
                    "loss_source": str(evals_path(run + suffix).relative_to(ROOT)),
                    "rate_source": str(rates_path(run + suffix).relative_to(ROOT)),
                    "Train loss": v["sft_loss"]["train"]["loss"],
                    "Held-out loss": v["sft_loss"]["test"]["loss"],
                    "In-dist. misalignment": rate("in_dist"),
                    "Off-target misalignment": rate("off_target"),
                    "n_scored_off_target": em["off_target"]["n_scored"],
                    "mmlu": mv["mmlu"]["mmlu"]["accuracy"],
                    "incoherent_off_target": em["off_target"]["incoherent_frac"],
                })
    return pd.DataFrame(rows).sort_values(["task", "arm", "frac"]).reset_index(drop=True)


def main():
    if all(evals_path(r + sfx).exists() for r in RUNS.values() for sfx in ARMS.values()):
        wide = extract()
        DATA.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(DATA, index=False)
        print(f"wrote {DATA}")
    else:
        wide = pd.read_csv(DATA)
        print(f"read {DATA}")

    long = wide.melt(id_vars=["task", "arm", "frac"], value_vars=METRICS,
                     var_name="metric", value_name="value")
    long["arm"] = pd.Categorical(long["arm"], categories=list(ARMS))
    long["panel"] = pd.Categorical(
        long["task"] + "\n" + long["metric"],
        categories=[f"{t}\n{m}" for t in RUNS for m in METRICS])
    sweep = long[(long["frac"] > 0) & long["value"].notna()].copy()
    for m in METRICS:
        if m in RATES:
            continue
        cap = sweep.loc[(sweep["metric"] == m) & (sweep["arm"] != "random"), "value"].max()
        drop = (sweep["metric"] == m) & (sweep["arm"] == "random") & (sweep["value"] > cap)
        print(f"{m}: random arm drawn from frac "
              f"{sweep.loc[(sweep['metric'] == m) & (sweep['arm'] == 'random') & ~drop, 'frac'].min():g}"
              f" (cap {cap:.3f}, {int(drop.sum())} points above it dropped)")
        sweep = sweep[~drop]
    anchor = long[(long["frac"] == 0) & (long["arm"] == "adam") & long["metric"].isin(RATES)]

    # Pin each metric's y range across the two tasks; rates to [0, 1].
    lims = []
    for m in METRICS:
        lo, hi = (0.0, 1.0) if m in RATES else (
            sweep.loc[(sweep["metric"] == m) & (sweep["arm"] != "random"), "value"]
            .agg(["min", "max"]).tolist())
        for t in RUNS:
            lims += [{"panel": f"{t}\n{m}", "frac": 0.01, "value": lo},
                     {"panel": f"{t}\n{m}", "frac": 0.01, "value": hi}]
    lims = pd.DataFrame(lims)
    lims["panel"] = pd.Categorical(lims["panel"], categories=long["panel"].cat.categories)

    p = (
        ggplot(sweep, aes("frac", "value", color="arm", linetype="arm"))
        + geom_blank(data=lims, inherit_aes=False, mapping=aes("frac", "value"))
        + geom_hline(aes(yintercept="value"), data=anchor, inherit_aes=False,
                     linetype="dotted", color="#888888", size=0.4)
        + geom_line(size=0.6)
        + geom_point(size=1.0)
        + facet_wrap("~ panel", nrow=2, scales="free_y")
        + scale_x_log10(breaks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                        labels=["10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"])
        + scale_color_manual(values=[palette.COLOR[a] for a in ARMS],
                             labels=[LABEL[a] for a in ARMS])
        + scale_linetype_manual(values=[LINETYPE[a] for a in ARMS],
                                labels=[LABEL[a] for a in ARMS])
        + labs(x="Fraction of units kept", y="", color="", linetype="")
    )
    p.save(OUT, verbose=False)
    print(f"wrote {OUT}")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
