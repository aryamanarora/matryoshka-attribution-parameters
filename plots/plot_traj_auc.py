"""Is MAttr undertrained? Train-loss log-AUC of the sparsity sweep vs fitting step, per arm.

The trajectory runs (`*_traj`: `eval.every: 25`, `sweep_when: auto`) evaluate the full sparsity
grid's sft_loss at every 25 fitting steps, so each contributes a 19-point curve of the same
log-AUC the LR figures summarise at the end -- MIB's ``acc_auc`` weighting over the frac grid,
raw nats, lower is better. Every point on a curve is the SAME 16-batch loss budget, so unlike
the earlier live-log parse there is no budget seam (each run's final 200-batch sweep is a
different measurement and is deliberately not drawn).

What it answers, arm by arm: a curve flat from early is a converged fit; one still falling at
step 450 is step-limited. The eps=1e-2 arm is the test of the eps story's prediction -- if
high-eps Adam is SGD in disguise, it should converge like SGD, not like default Adam.

Colour is the optimizer arm and linestyle the k-schedule (the palette's split of roles).

    uv run python plots/plot_traj_auc.py --out plots/traj_auc.pdf
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})


STEM = "runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard"
#: (label, colour, linestyle, run). COLOUR IS THE OPTIMIZER ARM AND LINESTYLE THE K-SCHEDULE --
#: the palette's rule -- because the first draft confounded them (Adam ran uniform, SGD ran
#: log/log-both, one colour each) and the deconfounding cells exist precisely to be comparable
#: within a colour. Arms whose runs have not landed yet are skipped with a note.
ARMS = [("Adam (uniform $k$)", "#0072B2", "solid", f"{STEM}_adam_traj"),
        ("Adam (log $k$)", "#0072B2", (0, (3, 2)), f"{STEM}_adam0p05_log_traj"),
        ("Adam eps 1e-2 (uniform $k$)", "#D55E00", "solid", f"{STEM}_adam_higheps_traj"),
        ("SGD (uniform $k$)", "#000000", "solid", f"{STEM}_sgd10_traj"),
        ("SGD (log $k$)", "#000000", (0, (3, 2)), f"{STEM}_sgd10_log_traj"),
        ("SGD (log-both $k$)", "#000000", (0, (1, 1.2)), f"{STEM}_sgd10_log_both_traj")]

#: the closed-form baselines: no fitting axis, so they are horizontal reference lines, in the
#: palette's reference linestyles. Their AUCs come from their runs' final sweeps, which the
#: trajectory endpoints match exactly (measured offset 0.0000 -- the last history entry IS the
#: final sweep), so the lines and the curves are the same measurement.
REFS = [("stepless IG", "#E69F00", "solid", "runs/fr2de_qwen25_14b_lr1e-4_ixg_mc"),
        ("I\u00d7G @ base", "#882255", (0, (1, 1.5)), "runs/fr2de_qwen25_14b_lr1e-4_ixg_base")]


def log_auc(xs, ys):
    lx = [math.log(x) for x in xs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2
               for i in range(len(ys) - 1)) / (lx[-1] - lx[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--out", default="plots/traj_auc.pdf")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(2.7, 2.15))
    ckpt = None
    for label, color, ls, run in ARMS:
        try:
            hist = json.load(open(run + "/evals.json"))["history"]
        except FileNotFoundError:
            print(f"  skip {label}: {run} not landed yet")
            continue
        xs, ys = [], []
        for e in hist:
            fr = sorted(float(k.split("_")[1]) for k in e["results"] if k.startswith("frac_"))
            if len(fr) < 10 or e["step"] == 0:   # step 0 has no place on a log axis
                continue
            xs.append(e["step"])
            ys.append(log_auc(fr, [e["results"][f"frac_{f:g}"]["sft_loss"][args.split]["loss"]
                                   for f in fr]))
            if ckpt is None:
                # the finetuned checkpoint's own loss (frac_1 = the full delta), at the SAME
                # 16-batch budget as every curve point -- constant across steps by construction
                ckpt = e["results"]["frac_1"]["sft_loss"][args.split]["loss"]
        ax.plot(xs, ys, ls=ls, lw=1.0, color=color, label=label)
    for label, color, ls, run in REFS:
        blob = json.load(open(run + "/evals.json"))["final"]
        fr = sorted(float(k.split("_")[1]) for k in blob if k.startswith("frac_"))
        v = log_auc(fr, [blob[f"frac_{f:g}"]["sft_loss"][args.split]["loss"] for f in fr])
        ax.axhline(v, color=color, lw=0.9, ls=ls, label=label)
    if ckpt is not None:
        ax.axhline(ckpt, color="#888888", lw=0.6, ls=(0, (4, 2)))
        ax.annotate("finetuned checkpoint", xy=(0.03, ckpt), xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=5.5, color="#666666")
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.set_xticks([], minor=True)
    arrow = "$\\downarrow$"
    ax.set_xlabel("Fitting step", fontsize=7.5)
    ax.set_ylabel(f"{args.split} loss (nats), log-AUC {arrow}", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.legend(fontsize=5.2, frameon=True, framealpha=0.95, borderpad=0.3, handlelength=1.4,
              labelspacing=0.3, columnspacing=0.8, handletextpad=0.4, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 0.115))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(Path(args.out).with_suffix("." + ext)), dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
