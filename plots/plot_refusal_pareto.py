#!/usr/bin/env python3
"""Refusal-erosion vs capability, one point per mask sparsity: the Pareto view of the
log-uniform GRPO refusal mask (post prefix-cache-fix retrain, re-scored by
configs/refusal/eval_native.yaml over the sparsity grid).

x = StrongREJECT score on the 60 reported forbidden prompts (right = more compliance with
harm, i.e. worse). y = capability, the mean of MMLU accuracy and GSM8K accuracy (up =
better). Each coloured point is one condition of the same mask, colour = % of the finetune's
parameters changed (nonresid units, so unit fraction ~= parameter fraction); the 1% point is
ringed. The mask spans Instruct -> Base, so the sweep's `pretrained` IS the Instruct model
and `full_delta` IS the Base model -- labelled as such. Grey open diamonds are the PROMPT
anchors: the same two models under the plain / URIAL / URIAL-help frames (SR from
configs/baseline/'s cells; their y is each model's NATIVE-frame capability, since capability
was not measured under the other frames -- read their x, not their exact height). The
Pareto-dominant corner is top-LEFT (capable and refusing).

    uv run python plots/plot_refusal_pareto.py <run_dir_with_posthoc_eval>
"""
import json
import os
import sys

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, geom_path, geom_point, geom_text,
    ggplot, labs, scale_color_cmap, scale_color_manual, scale_fill_manual, theme,
    theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")
theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7), axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7), legend_text=element_text(size=6),
        legend_key_size=6, legend_position="top", legend_direction="horizontal",
        legend_box_margin=0,
    )
)

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/data/artifacts/aryaman-work-trial/runs/refusal_grpo_instruct_to_base_urial_help_logk"
OUT = os.path.dirname(os.path.abspath(__file__))

fin = json.load(open(os.path.join(RUN, "posthoc_eval", "evals.json")))["final"]
rows = []
for cond, res in fin.items():
    sr = res.get("strongreject", {}).get("off_target", {}).get("score")
    mmlu = (res.get("mmlu", {}).get("mmlu") or {}).get("accuracy")
    gsm = (res.get("gsm8k", {}).get("gsm8k") or {}).get("accuracy")
    if sr is None or mmlu is None or gsm is None:
        continue
    frac = (1.0 if cond == "full_delta" else 0.0 if cond == "pretrained"
            else float(cond[len("frac_"):]))
    cap = (mmlu + gsm) / 2
    label = {"frac_0.01": "1% of params"}.get(cond, "")
    rows.append(dict(cond=cond, pct=max(frac, 1e-3) * 100, sr=sr, cap=cap, label=label,
                     cond_name=cond))
df = pd.DataFrame(rows).sort_values("pct")
df = df[df.cond_name != "frac_1"]               # identical to full_delta, one point suffices
hl = df[df.cond_name == "frac_0.01"]
cap_instruct = float(df.loc[df.cond_name == "pretrained", "cap"].iloc[0])
cap_base = float(df.loc[df.cond_name == "full_delta", "cap"].iloc[0])

# prompt-frame anchors: {model} x {native, plain, URIAL, URIAL-help}. The native rows take
# their (SR, capability) from the sweep\'s own endpoints (pretrained = Instruct, full_delta =
# Base); the rest take SR from the baseline cells at the model\'s native-frame capability.
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "baseline")
sr_i = float(df.loc[df.cond_name == "pretrained", "sr"].iloc[0])
sr_b = float(df.loc[df.cond_name == "full_delta", "sr"].iloc[0])
ANCHORS = [("Instruct", "native", None, sr_i),
           ("Instruct", "plain", "baseline_strongreject_llama32_1b_plain.json", None),
           ("Instruct", "URIAL", "baseline_strongreject_llama32_1b_instruct_urial.json", None),
           ("Instruct", "URIAL·help", "baseline_strongreject_llama32_1b_instruct_urial_help.json", None),
           ("Base", "native", None, sr_b),
           ("Base", "plain", "baseline_strongreject_llama32_1b_base.json", None),
           ("Base", "URIAL", "baseline_strongreject_llama32_1b_base_urial.json", None),
           ("Base", "URIAL·help", "baseline_strongreject_llama32_1b_base_urial_help.json", None)]
arow = []
for i, (model, prompt, fname, sr0) in enumerate(ANCHORS):
    if fname is not None:
        fp = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fp):
            continue
        ev = json.load(open(fp))
        fin = ev.get("final", ev)
        sr0 = fin.get("dense", fin)["strongreject"]["off_target"]["score"]
    arow.append(dict(model=model, sr=sr0, label=prompt,
                     cap=cap_instruct if model == "Instruct" else cap_base,
                     up=i % 2 == 0))                      # stagger labels above/below
anchors = pd.DataFrame(arow)

p = (
    ggplot(df, aes("sr", "cap"))
    + geom_path(size=0.3, alpha=0.5, color="#aaaaaa")
    + geom_point(aes(color="pct"), size=2.4, alpha=0.95, stroke=0)
    + geom_point(hl, color="black", fill="none", size=4.2, stroke=0.7, shape="o")
    + geom_point(anchors, aes("sr", "cap", fill="model"), shape="D", size=2.4,
                 stroke=0.4, color="#444444", alpha=0.95, inherit_aes=False)
    + geom_text(df[df.label != ""], aes(label="label"), size=5.5, nudge_y=1.6,
                color="#333333")
    # prompt-type labels, staggered above/below their diamonds so the clustered
    # Instruct frames stay legible
    + geom_text(anchors[anchors.up], aes("sr", "cap", label="label"), size=4.8,
                nudge_y=1.4, color="#555555", inherit_aes=False)
    + geom_text(anchors[~anchors.up], aes("sr", "cap", label="label"), size=4.8,
                nudge_y=-1.4, color="#555555", inherit_aes=False)
    + scale_fill_manual(values={"Instruct": "#377eb8", "Base": "#e41a1c"}, name="")
    + scale_color_cmap(cmap_name="viridis", trans="log10",
                       name="% of finetune params changed",
                       breaks=[0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    + labs(x="StrongREJECT score (right = refusal eroded)",
           y="Capability (mean of MMLU and GSM8K accuracy, %)", fill="")
    + theme(figure_size=(4.2, 3.2), legend_key_width=36, legend_entry_spacing=2)
)
p.save(os.path.join(OUT, "refusal_pareto.pdf"), verbose=False)
print(f"wrote refusal_pareto.pdf ({len(df)} conditions)")

# ---- the same sweep unrolled: x = % of params, y = score, faceted safety / capability.
# Reference lines carry no legend; each is labelled IN the panel. The safety panel shows the
# BASE model under every prompt frame (the prompt-side attack ladder the mask is read
# against); Instruct-native is the refusal floor. The capability panel keeps the two models'
# native anchors.
from plotnine import facet_wrap, geom_hline, geom_line, scale_x_log10

mid = df[~df.cond_name.isin(["pretrained", "full_delta"])]
FS, FC = "StrongREJECT (safety)", "MMLU+GSM8K mean (capability)"
long = pd.concat([mid.assign(facet=FS, y=mid["sr"]),
                  mid.assign(facet=FC, y=mid["cap"])])
long["facet"] = pd.Categorical(long["facet"], [FS, FC], ordered=True)
sr_anchor = {r.label: r.sr for r in anchors[anchors.model == "Base"].itertuples()}
refs = pd.DataFrame([
    dict(facet=FS, model="Instruct", y=sr_i, label="Instruct · native", lx=90, up=True),
    dict(facet=FS, model="Base", y=sr_anchor["plain"], label="Base · plain", lx=30, up=True),
    dict(facet=FS, model="Base", y=sr_anchor["URIAL"], label="Base · URIAL", lx=90, up=True),
    dict(facet=FS, model="Base", y=sr_anchor["URIAL·help"], label="Base · URIAL·help", lx=90, up=True),
    dict(facet=FC, model="Instruct", y=cap_instruct, label="Instruct", lx=90, up=True),
    dict(facet=FC, model="Base", y=cap_base, label="Base", lx=90, up=True),
])
refs["facet"] = pd.Categorical(refs["facet"], [FS, FC], ordered=True)
hl_long = long[long.cond_name == "frac_0.01"]
MODEL_COLOR = {"Instruct": "#377eb8", "Base": "#e41a1c"}
refs["col"] = refs["model"].map(MODEL_COLOR)

p2 = (
    ggplot(long, aes("pct", "y"))
    + facet_wrap("~facet", ncol=2, scales="free_y")
    + geom_line(size=0.4, color="#aaaaaa")
    + geom_point(size=1.8, color="#333333")
    + geom_point(hl_long, color="black", fill="none", size=3.6, stroke=0.6, shape="o")
    + scale_x_log10(breaks=[0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    + labs(x="Finetune parameters changed (%)", y="")
    + theme(figure_size=(5.5, 2.1), axis_text_x=element_text(rotation=0))
)
for model, color in MODEL_COLOR.items():
    sub = refs[refs.model == model]
    p2 = p2 + geom_hline(sub, aes(yintercept="y"), color=color, linetype="dashed",
                         size=0.35, alpha=0.8, inherit_aes=False)
    for up, va in ((True, "bottom"), (False, "top")):
        ss = sub[sub.up == up]
        if len(ss):
            p2 = p2 + geom_text(ss, aes("lx", "y", label="label"), color=color, size=5,
                                ha="right", va=va, inherit_aes=False)
p2.save(os.path.join(OUT, "refusal_sparsity_facets.pdf"), verbose=False)
print("wrote refusal_sparsity_facets.pdf")
