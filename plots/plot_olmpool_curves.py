"""OlmPool: NIAH retrieval vs fraction of the extension delta kept, per context length and arm.

Reads ``plots/data/olmpool/summary.json`` (scripts/olmpool_analysis.py). One panel per
(model, context length); x is the fraction of scored units kept (log), y the teacher-forced
retrieval accuracy (or the answer NLL with ``--metric nll``); colour is the arm (learned MAttr,
IxG at either endpoint, random), and the two anchors are drawn as flat reference lines --
``pretrained`` (the pretrained weights under the extended theta, grey) and ``full_delta`` (the
scored tensors' extension delta in full, black dotted).

    uv run python plots/plot_olmpool_curves.py --models G_pre_8kv_8k_14k H_post_LQK_32kv_8k_11k_SWA
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_manual, scale_linetype_manual, scale_x_log10, theme,
    theme_bw, theme_set,
)

import matplotlib.pyplot as plt

import palette

# TrueType outlines and Inter for mathtext; the plotnine theme below does not reach rcParams
plt.rcParams.update(palette.RC)

#: strip labels short enough for a 1in-tall row at 5.5pt; anything unlisted falls back to the
#: name with its FFN-width tag stripped
SHORT = {"G_pre_8kv_8k_14k": "G: Llama, 8 kv", "H_post_LQK_32kv_8k_11k_SWA": "H: Olmo-3, SWA",
         "K_post_HQK_8kv_12k": "K: Qwen-3", "J_pre_16kv_8k_14k": "J: Llama, 16 kv",
         "I_pre_32kv_8k_12k": "I: Llama, 32 kv", "G_pre_4kv_8k_14k": "G: Llama, 4 kv",
         "G_pre_8kv_4k_14k": "G: Llama, 4K pretrain", "G_pre_LQK_8kv_8k_14k": "G: Llama + LQK",
         "G_post_LQK_8kv_8k_14k": "G: post-norm + LQK"}
ARM_LABEL = {"fold_learned": "MAttr (heads)", "fold_ixg_base": "I×G @ base", "fold_ixg_ft": "I×G @ LC",
             "fold_random": "Random",
             "attn_learned": "MAttr (heads, rest pretrained)", "attn_ixg_base": "I×G @ base (rest pretrained)", "attn_ixg_ft": "I×G @ LC (rest pretrained)",
             "attn_random": "Random (rest pretrained)", "all_learned": "MAttr (heads+MLP)", "all_ixg_base": "I×G @ base (heads+MLP)",
             "attnqk_learned": "MAttr (heads+QK norm)", "all_random": "Random (heads+MLP)"}
ARM_COLOR = {"fold_learned": palette.COLOR["adam"], "fold_ixg_base": palette.COLOR["ixg:base"],
             "fold_ixg_ft": palette.COLOR["ixg:finetuned"], "fold_random": palette.COLOR["random"],
             "attn_learned": palette.COLOR["adam"], "attn_ixg_base": palette.COLOR["ixg:base"],
             "attn_ixg_ft": palette.COLOR["ixg:finetuned"], "attn_random": palette.COLOR["random"],
             "all_learned": palette.COLOR["adam"], "all_ixg_base": palette.COLOR["ixg:base"],
             "attnqk_learned": palette.COLOR["adam"], "all_random": palette.COLOR["random"]}
ARM_LT = {"fold_learned": "solid", "fold_ixg_base": "solid", "fold_ixg_ft": "dashed", "fold_random": "solid",
          "attn_learned": "dotted", "attn_ixg_base": "solid", "attn_ixg_ft": "dashed",
          "attn_random": "solid", "all_learned": "solid", "all_ixg_base": "solid", "attnqk_learned": "dashdot", "all_random": "solid"}

theme_set(
    theme_bw(base_size=8)
    + theme(text=element_text(color="#000", family="Inter"),
            axis_title=element_text(size=7), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(), panel_spacing_x=0.02, panel_spacing_y=0.02,
            strip_background=element_blank(), strip_text=element_text(size=7),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=6, legend_position="top", legend_direction="horizontal",
            legend_box_margin=0))


def frame(summary, models, metric):
    rows, anchors = [], []
    for rec in summary:
        if models and rec["model"] not in models:
            continue
        short = rec["results"]["helmet"]["32k"] if rec.get("results") else None
        # the name minus its FFN-width tag, plus the paper's score: what fits a strip label
        mlabel = SHORT.get(rec["model"], re.sub(r"_1[1-4]k", "", rec["model"])) + (f" ({short:.0f})" if short else "")
        for key, curve in rec["curves"].items():
            evn, split, m = key.split("/")
            if evn != "niah" or m != metric:
                continue
            ctx = int(split.replace("ctx_", ""))
            for cond, v in curve.items():
                if cond.startswith("frac_"):
                    rows.append(dict(model=mlabel, ctx=f"{ctx // 1024}K", arm=rec["arm"],
                                     frac=float(cond[5:]), value=v))
                elif cond in ("pretrained", "full_delta"):
                    anchors.append(dict(model=mlabel, ctx=f"{ctx // 1024}K", arm=rec["arm"],
                                        anchor=cond, value=v))
    df, an = pd.DataFrame(rows), pd.DataFrame(anchors)
    order = sorted(df.ctx.unique(), key=lambda s: int(s[:-1]))
    df["ctx"] = pd.Categorical(df.ctx, order)
    an["ctx"] = pd.Categorical(an.ctx, order)
    return df, an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="plots/data/olmpool/summary.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--metric", default="acc", choices=["acc", "nll"])
    ap.add_argument("--out", default="plots/olmpool_curves.pdf")
    ap.add_argument("--width", type=float, default=5.4)
    a = ap.parse_args()
    summary = json.loads(Path(a.summary).read_text())
    df, an = frame(summary, a.models, a.metric)
    if a.arms:
        df = df[df.arm.isin(a.arms)]
        an = an[an.arm.isin(a.arms)]
    arms = [x for x in ARM_LABEL if x in set(df.arm)]
    df["Arm"] = pd.Categorical(df.arm.map(ARM_LABEL), [ARM_LABEL[x] for x in arms])
    # the anchors do not depend on the arm's scores, only on the FRAMING (which tensors are
    # scored and whether the rest is folded): one per (model, ctx, framing, anchor)
    an["framing"] = an.arm.str.split("_").str[0]
    an = an.drop_duplicates(["model", "ctx", "framing", "anchor"])
    n_models, n_ctx = df.model.nunique(), df.ctx.nunique()
    p = (ggplot(df, aes("frac", "value", color="Arm", linetype="Arm"))
         # anchors: unmapped, so they do not join the arm legend -- grey solid = pretrained weights
         # under the extended theta, black dotted = the scored tensors' full extension delta
         + geom_hline(an[an.anchor == "pretrained"], aes(yintercept="value"), color="#999999", size=0.4)
         + geom_hline(an[an.anchor == "full_delta"], aes(yintercept="value"), color="#000000",
                      size=0.4, linetype="dotted")
         + geom_line(size=0.6) + geom_point(size=1.0)
         + facet_grid("model ~ ctx")
         + scale_x_log10(breaks=[0.01, 0.1, 1.0], labels=["10⁻²", "10⁻¹", "10⁰"])
         + scale_color_manual(values=[ARM_COLOR[x] for x in arms])
         + scale_linetype_manual(values=[ARM_LT[x] for x in arms])
         + labs(x="Fraction of Units Kept",
                y="Retrieval Accuracy" if a.metric == "acc" else "Answer NLL (nats/token)")
         + theme(figure_size=(a.width, 0.35 + 0.95 * n_models), strip_text_y=element_text(size=5.5),
                 axis_text_x=element_text(size=6, rotation=0)))
    p.save(a.out, verbose=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
