"""MAttr (Adam, tuned) against stepless IG across the eight Qwen2.5-14B organisms, four facets.

One facet per readout -- the two losses the mask is fitted against and read out on (train,
held-out) and the two behaviour rates (on-target = the `in_dist` split, off-target = the
generalisation probe) -- with every task drawn as a thin line and the cross-task mean as a thick
one, coloured by method. Both arms attribute the SAME frozen LoRA delta per task with the same unit
definition and eval; the pairs differ only in how the scores are produced (a learned mask at the
tuned fitting hyperparameters, `*_best`, against the closed-form I×G averaged over
alpha ~ U(0, 1) at 64 batches, `*_ixg_mc`), verified by resolved-config diff.

    uv run python plots/plot_adam_vs_steplessig.py

WHY THE LOSSES ARE NORMALISED. Eight tasks' losses live on eight scales (the finetune's own loss
is ~0.5 on the language tasks and ~1.2-1.4 on EM), so a mean of raw losses is a mean of task
difficulty, not of mask quality. Each loss is therefore reported as the PERCENTAGE of the
pretrained-to-finetune gap the mask recovers, 100 * (L_pre - L_k) / (L_pre - L_full): 0 at the
pretrained model, 100 at the whole delta, the same reading `bench_transfer.py` uses, and one that
can be averaged across tasks. The rates are left raw -- they are already in [0, 1], and dividing a 0.28 full-delta EM rate by
itself would let judge noise swing a "normalised" curve by a third (the pirate post-hoc entry in
CLAUDE.md is the warning).

WHERE THE LOSS COMES FROM. Six of the eight tasks' runs scored their final sweep loss at
`n_batches: 16` (16 examples a split, 676-4,100 tokens; only the EM cells set
`final_n_batches: 200`). Each run was therefore re-scored on its saved mask at 200 examples a
split through the eval CLI (`--only sft_loss --loss-batches 200 --out <run>/sft_loss_eval`, the
EM cells' existing budget), and the loss is read from `<run>/sft_loss_eval/evals.json` when that
file exists and from the run's own `evals.json` otherwise; the CSV's `loss_source` column says
which. Same seeded subset either way -- the loaders are unshuffled and the split is a seeded
carve -- so the re-score is a superset of the examples the run's own number used.

The fr2de stepless-IG twin is `fr2de_qwen25_14b_lr1e-4_ixg_mc` (no `_posthoc_shard` infix; that
family was named before the convention settled), chosen over `posthoc_steplessig_epoch` because it
uses the same 64-batch budget as the other seven tasks' `_ixg_mc` cells.
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
    geom_blank,
    geom_line,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_log10,
    theme,
    theme_bw,
    theme_set,
)

import palette

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType outlines, not Type-3

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.75),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
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
DATA = ROOT / "plots" / "data" / "qwen14b_adam_vs_steplessig" / "sweep.csv"

#: task -> (best run, stepless-IG run, eval name, behaviour metric key)
CELLS = {
    "fr2de": ("fr2de_qwen25_14b_lr1e-4_posthoc_shard_best", "fr2de_qwen25_14b_lr1e-4_ixg_mc",
              "language", "target_frac"),
    "fr2ru": ("fr2ru_qwen25_14b_lora32_lr1e-4_posthoc_best",
              "fr2ru_qwen25_14b_lora32_lr1e-4_posthoc_ixg_mc", "language", "target_frac"),
    "fr2zh": ("fr2zh_qwen25_14b_lora32_lr1e-4_posthoc_best",
              "fr2zh_qwen25_14b_lora32_lr1e-4_posthoc_ixg_mc", "language", "target_frac"),
    "case": ("case_qwen25_14b_posthoc_shard_best", "case_qwen25_14b_posthoc_shard_ixg_mc",
             "casing", "lower_frac"),
    "caps": ("caps_qwen25_14b_lora32_lr1e-4_posthoc_best",
             "caps_qwen25_14b_lora32_lr1e-4_posthoc_ixg_mc", "casing", "upper_frac"),
    "spelling": ("spelling_qwen25_14b_lora32_lr1e-4_posthoc_best",
                 "spelling_qwen25_14b_lora32_lr1e-4_posthoc_ixg_mc", "spelling", "british_frac"),
    "medical": ("bad_medical_qwen25_14b_lora32_lr1e-4_posthoc_shard_best",
                "bad_medical_qwen25_14b_lora32_lr1e-4_posthoc_shard_ixg_mc", "em_fast",
                "misaligned_frac"),
    "financial": ("bad_medical_qwen25_14b_financial_posthoc_shard_best",
                  "bad_medical_qwen25_14b_financial_posthoc_shard_ixg_mc", "em_fast",
                  "misaligned_frac"),
}
ARMS = {"adam": 0, "ixg:mc": 1}
LABEL = {"adam": "MAttr (Adam, tuned)", "ixg:mc": palette.REF_LABEL["ixg:mc"]}


def loss_path(run: str) -> Path:
    d = ROOT / "runs" / run
    re = d / "sft_loss_eval" / "evals.json"
    return re if re.exists() else d / "evals.json"


def extract() -> pd.DataFrame:
    rows = []
    for task, (best, mc, ev, key) in CELLS.items():
        for arm, idx in ARMS.items():
            run = (best, mc)[idx]
            fin = json.loads((ROOT / "runs" / run / "evals.json").read_text())["final"]
            lfin = json.loads(loss_path(run).read_text())["final"]
            for cond, v in fin.items():
                if cond == "full_delta":                  # same weights as frac_1 under `cause`
                    continue
                rows.append({
                    "task": task, "arm": arm, "run": run,
                    "loss_source": str(loss_path(run).relative_to(ROOT)),
                    "loss_tokens": lfin[cond]["sft_loss"]["test"]["tokens"],
                    "frac": 0.0 if cond == "pretrained" else float(cond.removeprefix("frac_")),
                    "train_loss": lfin[cond]["sft_loss"]["train"]["loss"],
                    "test_loss": lfin[cond]["sft_loss"]["test"]["loss"],
                    "on_target": v[ev]["in_dist"][key],
                    "off_target": v[ev]["off_target"][key],
                })
    return pd.DataFrame(rows).sort_values(["task", "arm", "frac"]).reset_index(drop=True)


def recovered(df: pd.DataFrame, col: str) -> pd.Series:
    """100 * (L_pre - L_k) / (L_pre - L_full) within each (task, arm)."""
    out = pd.Series(index=df.index, dtype=float)
    for (_, _), g in df.groupby(["task", "arm"]):
        pre = g.loc[g["frac"] == 0.0, col].item()
        full = g.loc[g["frac"] == 1.0, col].item()
        out[g.index] = 100.0 * (pre - g[col]) / (pre - full)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if all((ROOT / "runs" / r / "evals.json").exists()
           for cell in CELLS.values() for r in cell[:2]):
        wide = extract()
        DATA.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(DATA, index=False)
        print(f"wrote {DATA}")
    else:
        wide = pd.read_csv(DATA)
        print(f"read {DATA}")

    wide["train_loss"] = recovered(wide, "train_loss")
    wide["test_loss"] = recovered(wide, "test_loss")
    names = {"train_loss": "Train loss recovered (%)", "test_loss": "Held-out loss recovered (%)",
             "on_target": "On-target rate", "off_target": "Off-target rate"}

    long = wide[wide["frac"] > 0].melt(id_vars=["task", "arm", "frac"], value_vars=list(names),
                                       var_name="metric", value_name="value")
    long["facet"] = pd.Categorical(long["metric"].map(names), categories=list(names.values()))
    long["arm"] = pd.Categorical(long["arm"], categories=list(ARMS))
    long["series"] = long["task"] + "/" + long["arm"].astype(str)
    mean = (long.groupby(["facet", "arm", "frac"], observed=True)["value"].mean().reset_index())

    # The rate facets are pinned to [0, 1]. The two loss facets share one range, the span of the
    # data over BOTH (pinning them to [0, 100] too put every curve in the top third of the panel:
    # even the sparsest mask recovers >60% of the train-loss gap and >75% of the held-out one).
    pins = []
    for f in ("On-target rate", "Off-target rate"):
        pins += [{"facet": f, "frac": 0.01, "value": 0.0}, {"facet": f, "frac": 0.01, "value": 1.0}]
    loss_facets = [names["train_loss"], names["test_loss"]]
    lv = long.loc[long["facet"].isin(loss_facets), "value"]
    for f in loss_facets:
        pins += [{"facet": f, "frac": 0.01, "value": lv.min()},
                 {"facet": f, "frac": 0.01, "value": lv.max()}]
    pins = pd.DataFrame(pins)
    pins["facet"] = pd.Categorical(pins["facet"], categories=list(names.values()))

    colors = [palette.COLOR[a] for a in ARMS]
    labels = [LABEL[a] for a in ARMS]
    g = (
        ggplot(long, aes("frac", "value", color="arm"))
        + geom_blank(data=pins, inherit_aes=False, mapping=aes("frac", "value"))
        + geom_line(aes(group="series"), size=0.25, alpha=0.45)
        + geom_line(data=mean, size=1.1)
        + facet_wrap("~ facet", nrow=1, scales="free_y")
        + scale_x_log10(breaks=[1e-3, 1e-2, 1e-1, 1], labels=["10⁻³", "10⁻²", "10⁻¹", "1"])
        + scale_color_manual(values=colors, labels=labels)
        + labs(x="Fraction of units kept", y="")
    )
    out = Path(args.out) if args.out else ROOT / "plots" / "qwen14b_adam_vs_steplessig.pdf"
    g.save(out, verbose=False)
    print(f"wrote {out}")
    print(mean.pivot(index=["facet", "frac"], columns="arm", values="value").round(2).to_string())


if __name__ == "__main__":
    main()
