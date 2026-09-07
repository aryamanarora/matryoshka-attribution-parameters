"""Train-loss log-AUC for the CLEAN fr2de/Qwen-14B post-hoc sweeps: methods, and the eps x lr grid.

The headline metric of the clean re-run is the train loss over the sparsity sweep -- how much of
the finetune's fit a top-k mask recovers, integrated over log sparsity. Two panels, because the
two questions the clean set answers about it are different shapes:

  (a) METHODS. Train loss vs kept fraction, one line per attribution, with both anchors drawn:
      the pretrained model (upper dashed) and the full delta (lower dashed) bracket every curve,
      so the vertical position of a curve between them IS the fraction of the loss gap recovered.
      All arms are data-matched at 7,200 forward/backward passes.

  (b) EPS GRID, collapsed. Each Adam cell's train-loss AUC against the single combination
      c = lr / sqrt(eps). The three column-minima land on c = 50 and agree to within the
      measured noise floor (the shaded band: |trainAUC(adam_unif) - trainAUC(adam_unif_ordctl)|
      = 0.0015 nats, a pure data-order replicate). The collapse is NOT global -- c = 5 spreads
      0.019 -- so the combination locates the optimum without governing the surface, and the
      panel is drawn to show that failure rather than hide it.

LOWER IS BETTER on both panels' y-axis. Colours follow plot_attrib_maxgap.py's registry where a
method appears in both figures; the clean grid's cells have no registry entry and are keyed by lr.

    uv run python plots/plot_clean_trainloss.py
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

#: (label, run suffix, colour). Registry colours reused where the method already appears in
#: plot_attrib_maxgap.py; IG-16 is new to the clean set and takes the unused Wong yellow-brown.
METHODS = [
    ("MAttr (Adam, log-$k$)",   "adam_eb1",         "#0072B2"),
    ("MAttr (Adam, uniform-$k$)", "adam_eb1_uniform", "#56B4E9"),
    ("MAttr (SGD, lr 10)",      "sgd_eb1",          "#009E73"),
    ("Stepless IG",             "steplessig_epoch", "#E69F00"),
    ("IG (16 steps)",           "ig16",             "#7F5E00"),
    ("I×G @ base",         "ixgbase_epoch",    "#882255"),
    ("Random",                  "random",           "#BBBBBB"),
]

GRID = {(0.005, 1e-8): "adam_eb1_lr0p005_eps1e-8", (0.005, 1e-6): "adam_eb1_lr0p005_eps1e-6",
        (0.005, 1e-4): "adam_eb1_lr0p005_eps0p0001", (0.005, 1e-2): "adam_eb1_lr0p005_eps0p01",
        (0.05, 1e-8): "adam_eb1", (0.05, 1e-6): "adam_eb1_lr0p05_eps1e-6",
        (0.05, 1e-4): "adam_eb1_lr0p05_eps0p0001", (0.05, 1e-2): "adam_eb1_lr0p05_eps0p01",
        (0.5, 1e-8): "adam_eb1_lr0p5_eps1e-8", (0.5, 1e-6): "adam_eb1_lr0p5_eps1e-6",
        (0.5, 1e-4): "adam_eb1_lr0p5_eps0p0001", (0.5, 1e-2): "adam_eb1_lr0p5_eps0p01"}
LR_COLOR = {0.005: "#0072B2", 0.05: "#D55E00", 0.5: "#009E73"}
EPS_MARK = {1e-8: "o", 1e-6: "s", 1e-4: "^", 1e-2: "D"}


def curve(suffix):
    """(fracs, train losses, pretrained, full_delta) for one run."""
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
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.05))

    # ---- (a) methods -------------------------------------------------------------------
    pre = full = None
    for label, suffix, colour in METHODS:
        xs, ys, pre, full = curve(suffix)
        axA.plot(xs, ys, color=colour, lw=1.0, zorder=3,
                 label=f"{label} ({log_auc(xs, ys):.3f})")
    # anchors labelled at the LEFT edge: the sparse end is where the curves are furthest from
    # both lines, so the text sits in empty space at either corner
    for y, txt in ((pre, "pretrained"), (full, "full delta")):
        axA.axhline(y, color="#888888", lw=0.5, ls=(0, (3, 2)), zorder=1)
        axA.text(1.05e-3, y, txt, fontsize=5, color="#888888", ha="left", va="bottom")
    axA.set_xscale("log")
    axA.set_xlabel("Fraction of units kept", fontsize=7)
    axA.set_ylabel("Train loss (nats)", fontsize=7)
    axA.legend(fontsize=4.8, loc="upper right", frameon=False, handlelength=1.4,
               labelspacing=0.25, borderpad=0.1, title="Method (train-loss log-AUC)",
               title_fontsize=5.2, alignment="left")

    # ---- (b) the eps x lr grid collapsed onto lr / sqrt(eps) ----------------------------
    a, b = (log_auc(*curve("adam_eb1_uniform")[:2]),
            log_auc(*curve("adam_eb1_uniform_ordctl")[:2]))
    floor = abs(a - b)
    best = min(log_auc(*curve(d)[:2]) for d in GRID.values())
    axB.axhspan(best, best + floor, color="#dddddd", zorder=1)
    axB.text(4e3, best + floor, "data-order noise floor", fontsize=5, color="#777777",
             ha="right", va="bottom")
    for (lr, eps), d in GRID.items():
        axB.plot(lr / math.sqrt(eps), log_auc(*curve(d)[:2]), marker=EPS_MARK[eps],
                 color=LR_COLOR[lr], ms=3.2, mew=0.4, mec="#000000", ls="none", zorder=3)
    axB.set_xscale("log")
    # Unicode rather than mathtext: Inter has no radical glyph, so $\sqrt{}$ falls back to
    # DejaVu for that one symbol -- the mismatch the style rules exist to prevent
    axB.set_xlabel("c = lr / √ε", fontsize=7)
    axB.set_ylabel("Train-loss log-AUC (nats)", fontsize=7)
    handles = ([Line2D([], [], color=c, marker="o", ls="none", ms=3.2, mew=0.4, mec="#000",
                       label=f"lr {lr:g}") for lr, c in LR_COLOR.items()]
               + [Line2D([], [], color="#777777", marker=m, ls="none", ms=3.2, mew=0.4,
                         mec="#000", label=f"ε {e:.0e}".replace("e-0", "e-"))
                  for e, m in EPS_MARK.items()])
    axB.legend(handles=handles, fontsize=4.8, loc="upper right", frameon=False, ncol=2,
               handlelength=1.0, labelspacing=0.25, borderpad=0.1, columnspacing=0.8)

    for ax in (axA, axB):
        ax.grid(True, lw=0.25, color="#dddddd")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=5.5, length=1.5, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/clean_trainloss.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote plots/clean_trainloss.pdf  (noise floor {floor:.4f} nats)")


if __name__ == "__main__":
    main()
