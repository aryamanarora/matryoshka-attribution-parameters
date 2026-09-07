"""Train loss over the sparsity sweep at TENSOR granularity, against the nonresid twins.

`unit: tensor` gives each scored weight matrix ONE score, so the fr2de/Qwen-14B layout has 480
units where `nonresid` has 2,580,624. Solid lines are the tensor family, faint dashed lines the
same method's nonresid run -- the vertical gap between a pair is what the coarse granularity
costs, and it is large everywhere except the dense end.

Two things the figure is drawn to keep honest:

  * `k = max(1, round(frac * total))` (masks/sweep.py), so at 480 units the sparse end is
    QUANTISED: frac 0.001 and 0.002 both give k=1 and return the identical loss. The top axis
    labels the actual tensor count per condition, so a flat segment reads as "same k" rather
    than as a plateau. The nonresid twins have no such issue (k=2581 at frac 0.001).
  * the tensor cells all ran AFTER the seeded-loader edit (order B) and the nonresid cells all
    before it (order A). That costs ~0.0015 nats on train-loss AUC -- the measured data-order
    noise floor, and ~2% of the gap being drawn -- so the pairing is safe here, but it would NOT
    be safe on an off-target curve, where the same knob moves the headline by ~0.15.

    uv run python plots/plot_tensor_trainloss.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")
plt.rcParams.update({
    "font.family": FAMILY,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY,
    "mathtext.it": f"{FAMILY}:italic", "mathtext.bf": f"{FAMILY}:bold",
    "mathtext.cal": f"{FAMILY}:italic", "mathtext.sf": FAMILY, "mathtext.tt": FAMILY,
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})

RUNS = Path("runs")
STEM = "fr2de_qwen25_14b_posthoc_"
N_TENSOR = 480          # scored weight matrices under `exclude_params: embed_tokens|lm_head|norm`

#: (label, tensor run, nonresid twin, colour) -- colours from plot_attrib_maxgap.py's registry
PAIRS = [
    ("MAttr (Adam)",  "adam_tensor",       "adam_eb1",         "#0072B2"),
    ("MAttr (SGD)",   "sgd_tensor",        "sgd_eb1",          "#009E73"),
    ("Stepless IG",   "steplessig_tensor", "steplessig_epoch", "#E69F00"),
    ("I×G @ base",    "ixgbase_tensor",    "ixgbase_epoch",    "#882255"),
    ("Random",        "random_tensor",     "random",           "#BBBBBB"),
]


def curve(suffix):
    blob = json.load(open(RUNS / (STEM + suffix) / "evals.json"))
    blob = blob.get("final", blob)
    pts = sorted((float(c.split("_", 1)[1]), v["sft_loss"]["train"]["loss"])
                 for c, v in blob.items() if c.startswith("frac_"))
    return ([p[0] for p in pts], [p[1] for p in pts],
            blob["pretrained"]["sft_loss"]["train"]["loss"],
            blob["full_delta"]["sft_loss"]["train"]["loss"])


def log_auc(xs, ys):
    lx = [math.log10(x) for x in xs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2
               for i in range(len(ys) - 1)) / (lx[-1] - lx[0])


def main():
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    pre = full = None
    for label, tsuf, nsuf, colour in PAIRS:
        xs, ys, pre, full = curve(tsuf)
        ax.plot(xs, ys, color=colour, lw=1.2, zorder=3,
                label=f"{label} ({log_auc(xs, ys):.3f})")
        nx, ny, _, _ = curve(nsuf)
        ax.plot(nx, ny, color=colour, lw=0.7, ls=(0, (2, 1.6)), alpha=0.55, zorder=2)
    # anchors labelled at the RIGHT edge and on the OUTER side of each line (pretrained above,
    # full delta below): both bands are empty, where the left edge holds the linestyle legend
    for y, txt, va in ((pre, "pretrained", "bottom"), (full, "full delta", "top")):
        ax.axhline(y, color="#888888", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.text(0.95, y, txt, fontsize=5.5, color="#888888", ha="right", va=va)

    ax.set_ylim(full - 0.045, pre + 0.035)
    ax.set_xscale("log")
    ax.set_xlabel("Fraction of units kept", fontsize=7)
    ax.set_ylabel("Train loss (nats)", fontsize=7)
    leg = ax.legend(fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.965), frameon=False,
                    handlelength=1.5, labelspacing=0.3, borderpad=0.1,
                    title="Tensor granularity (train-loss log-AUC)", title_fontsize=6,
                    alignment="left")
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], color="#555555", lw=1.2, label="unit: tensor (480 units)"),
                       Line2D([], [], color="#555555", lw=0.7, ls=(0, (2, 1.6)), alpha=0.55,
                              label="unit: nonresid (2,580,624)")],
              fontsize=5.5, loc="lower left", frameon=False, handlelength=1.8,
              labelspacing=0.3, borderpad=0.1)

    # top axis: the tensor count each condition actually keeps, k = max(1, round(frac*480)) --
    # this is what makes the quantised sparse end legible
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    top.set_xticks(ticks)
    top.set_xticklabels([str(max(1, round(f * N_TENSOR))) for f in ticks], fontsize=5)
    top.set_xlabel("Tensors kept (of 480)", fontsize=6.5, labelpad=2)
    top.tick_params(length=1.5, pad=1.0)
    top.grid(False)
    for sp in top.spines.values():
        sp.set_linewidth(0.5)

    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=5.5, length=1.5, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/tensor_trainloss.{ext}", dpi=300, bbox_inches="tight")
    print("wrote plots/tensor_trainloss.pdf")
    for label, tsuf, nsuf, _ in PAIRS:
        t = log_auc(*curve(tsuf)[:2]); n = log_auc(*curve(nsuf)[:2])
        print(f"  {label:14s} tensor {t:.4f}   nonresid {n:.4f}   cost {t - n:+.4f}")


if __name__ == "__main__":
    main()
