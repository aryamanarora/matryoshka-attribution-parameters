"""Sparsity curves for the main-text cells: inherited defaults against tuned hyperparameters.

One panel per (organism, model) cell of `fig:adam-maxgap`. Each panel carries four lines -- the
on-target and off-target expression rates of the QUARANTINED default-hyperparameter mask
(score_lr 0.05, uniform k, effective batch 16; dashed) and of its tuned twin (score_lr 0.005,
log k, effective batch 1; solid). Same delta, same data, same eval in both arms: the only thing
that differs is how the mask was fitted.

WHAT TO READ. The figure's statistic is the largest on-minus-off gap anywhere in the sweep, so
what matters is (a) where each arm's curves separate and (b) whether the default arm is even
alive at the sparse end -- on most cells it returns 0.000 on BOTH splits below 0.5% of units,
which caps its best gap at whatever the first functional condition happens to be.

    uv run python plots/plot_best_vs_default_curves.py
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

FAM = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans")
plt.rcParams.update({"font.family": FAM, "pdf.fonttype": 42, "text.color": "#000000",
                     "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000"})
ON, OFF = "#A8A8A8", "#D55E00"          # plot_attrib_maxgap.py's split palette
CELLS = [("fr2de","fr2de_qwen25_14b_lr1e-4_posthoc_shard","language","target_frac"),
 ("fr2ru","fr2ru_qwen25_14b_lora32_lr1e-4_posthoc","language","target_frac"),
 ("fr2zh","fr2zh_qwen25_14b_lora32_lr1e-4_posthoc","language","target_frac"),
 ("case","case_qwen25_14b_posthoc_shard","casing","lower_frac"),
 ("caps","caps_qwen25_14b_lora32_lr1e-4_posthoc","casing","upper_frac"),
 ("spelling","spelling_qwen25_14b_lora32_lr1e-4_posthoc","spelling","british_frac"),
 ("medical","bad_medical_qwen25_14b_lora32_lr1e-4_posthoc_shard","em_fast","misaligned_frac"),
 ("financial","bad_medical_qwen25_14b_financial_posthoc_shard","em_fast","misaligned_frac")]

def curve(d, ev, met):
    p = f"{d}/evals.json"
    if not os.path.exists(p): return None
    b = json.load(open(p)); b = b.get("final", b)
    cs = sorted((float(c.split("_",1)[1]), v) for c, v in b.items() if c.startswith("frac_"))
    if not cs or ev not in cs[0][1]: return None
    return ([f for f,_ in cs], [v[ev]["in_dist"][met] for _,v in cs],
            [v[ev]["off_target"][met] for _,v in cs])

got = [(lab, curve(f"runs/_quarantine_2026-09-01/{r}", ev, m), curve(f"runs/{r}_best", ev, m))
       for lab, r, ev, m in CELLS]
got = [g for g in got if g[1] and g[2]]
ncol = 4; nrow = (len(got) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(5.5, 1.55 * nrow), squeeze=False, sharey=True)
for ax in [a for row in axes for a in row][len(got):]: ax.set_visible(False)
for i, (lab, o, n) in enumerate(got):
    ax = axes[i // ncol][i % ncol]
    for xs, ys, c, ls in ((o[0], o[1], ON, (0,(2,1.5))), (o[0], o[2], OFF, (0,(2,1.5))),
                          (n[0], n[1], ON, "solid"), (n[0], n[2], OFF, "solid")):
        ax.plot(xs, ys, color=c, ls=ls, lw=1.0)
    ax.set_xscale("log"); ax.set_title(lab, fontsize=6.5, pad=2)
    ax.tick_params(labelsize=5, length=1.5); ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_linewidth(0.5)
    if i % ncol == 0: ax.set_ylabel("expression rate", fontsize=6)
    if i // ncol == nrow - 1: ax.set_xlabel("fraction kept", fontsize=6)
fig.legend(handles=[Line2D([],[],color=ON,lw=1.2,label="on-target"),
                    Line2D([],[],color=OFF,lw=1.2,label="off-target"),
                    Line2D([],[],color="#555",lw=1.2,ls=(0,(2,1.5)),label="default hparams"),
                    Line2D([],[],color="#555",lw=1.2,label="tuned hparams")],
           fontsize=5.5, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(pad=0.3, rect=(0,0,1,0.93))
for ext in ("pdf","png"): fig.savefig(f"plots/param_best_vs_default.{ext}", dpi=300)
print(f"wrote plots/param_best_vs_default.pdf ({len(got)} cells)")
