"""Off-target expression against sparsity, one line per attribution run on fr2de/Qwen-14B.

WHAT THIS SHOWS THAT THE AUC FIGURES CANNOT. Every other figure here summarises a run's sweep as
one log-AUC, and a single number provably cannot distinguish a curve that climbs monotonically
from one that overshoots and falls back -- the shape this repo keeps finding. Here the curves
themselves are drawn, all of them, so the AUC scatter's points can be read back to the sweeps
they came from.

EVERY CURVE ENDS AT THE SAME POINT. At frac 1.0 the mask keeps the whole delta, so all runs
converge on the finetune's own off-target rate (the dashed anchor) regardless of ranking. The
figure is about the approach to it, not the endpoint -- which is also why a linear x would be
the wrong axis: the sweep is geometric and half of a linear axis would be the 0.5-1.0 interval
where every curve has already converged.

`unit: tensor` CELLS ARE EXCLUDED, for the reason plot_on_vs_off.py gives: their x is a fraction
of 480 whole matrices where a nonresid run's is a fraction of 2.58M rows and columns, so the two
are not the same axis and overlaying them would invite reading a granularity difference as a
method one. plot_tensor_trainloss.py draws that family with its own top axis in matrix counts.

The family registry, the run glob and the classifier are IMPORTED from plot_on_vs_off.py rather
than restated, so a colour or a family boundary cannot drift between the scatter and the curves.

    uv run python plots/plot_param_sparsity_curves.py
    uv run python plots/plot_param_sparsity_curves.py --metric indist
"""

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

_spec = importlib.util.spec_from_file_location(
    "_onoff", Path(__file__).resolve().parent / "plot_on_vs_off.py")
_oo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oo)
FAM, ORDER, GLOB, classify = _oo.FAM, _oo.ORDER, _oo.GLOB, _oo.classify

METRICS = {
    "offtarget": ("Off-target expression rate",
                  lambda v: v["language"]["off_target"]["target_frac"]),
    "indist": ("In-dist expression rate", lambda v: v["language"]["in_dist"]["target_frac"]),
    "trainloss": ("Train loss (nats)", lambda v: v["sft_loss"]["train"]["loss"]),
    "testloss": ("Held-out loss (nats)", lambda v: v["sft_loss"]["test"]["loss"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRICS), default="offtarget")
    # WHERE THE EMPTY CORNER IS DEPENDS ON THE METRIC. Off-target rises across the whole sweep
    # and leaves the top-left clear; in-dist saturates by ~0.5% of units, so its entire spread is
    # in the far left and an upper-left legend covers the only informative part of the figure.
    ap.add_argument("--legend-loc", default=None,
                    help="matplotlib legend loc; default depends on --metric")
    args = ap.parse_args()
    label, get = METRICS[args.metric]
    leg_loc = args.legend_loc or {"indist": "center"}.get(args.metric, "upper left")

    runs, anchor, dropped = [], None, 0
    for d in sorted(glob.glob(GLOB)):
        if not (os.path.exists(d + "/evals.json") and os.path.exists(d + "/config.yaml")):
            continue
        mk = (yaml.safe_load(open(d + "/config.yaml")).get("mask") or {})
        fam = classify(mk)
        if fam is None:
            continue
        if mk.get("unit", "nonresid") != "nonresid":
            dropped += 1
            continue
        blob = json.load(open(d + "/evals.json"))
        blob = blob.get("final", blob)
        pts = sorted((float(c.split("_", 1)[1]), get(v))
                     for c, v in blob.items() if c.startswith("frac_"))
        if len(pts) < 2:
            continue
        if anchor is None and "full_delta" in blob:
            anchor = get(blob["full_delta"])
        runs.append((fam, float(mk.get("score_lr", 0.05)), pts))
    print(f"  {len(runs)} runs" + (f"; excluded {dropped} unit:tensor" if dropped else ""))

    # shade within each fitted family by its own LR range, as the scatter does -- so a dark line
    # is that ARM's high-LR end and shade never compares across arms
    shade = {}
    for fam in ("adam_log", "adam_unif", "sgd"):
        lrs = sorted({lr for f, lr, _ in runs if f == fam})
        if lrs:
            import math
            lo, hi = math.log10(min(lrs)), math.log10(max(lrs))
            shade[fam] = lambda v, lo=lo, hi=hi: (0.25 + 0.75 * ((math.log10(v) - lo) / (hi - lo))
                                                  if hi > lo else 1.0)

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    if anchor is not None:
        ax.axhline(anchor, color="#888888", lw=0.6, ls=(0, (3, 2)), zorder=1)
        # label at the LEFT edge: every curve converges on this line at frac 1.0, so the right
        # end is the one place on it guaranteed to be covered
        ax.text(0.0011, anchor, "full delta", fontsize=5.5, color="#888888",
                ha="left", va="bottom")
    for fam in ORDER:                      # ORDER puts the references and control on top
        for f, lr, pts in [r for r in runs if r[0] == fam]:
            colour = FAM[fam][1]
            a = shade[fam](lr) if fam in shade else 0.95
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=0.9, color=colour,
                    alpha=a, zorder=3 if fam in shade else 4)
    ax.set_xscale("log")
    ax.set_xlabel("Fraction of units kept", fontsize=7)
    ax.set_ylabel(label, fontsize=7)
    ax.legend(handles=[Line2D([], [], color=FAM[f][1], lw=1.2, label=FAM[f][0]) for f in ORDER],
              fontsize=5, loc=leg_loc, frameon=True, framealpha=0.95, borderpad=0.3,
              handletextpad=0.4, labelspacing=0.25)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    fig.tight_layout(pad=0.3)
    stem = f"plots/param_sparsity_{args.metric}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()
