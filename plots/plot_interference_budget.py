"""Do the three attribution methods improve with more compute on the interference task?

    uv run python plots/plot_interference_budget.py --tag hard

x is the budget in FORWARD/BACKWARD PASSES of the toy model, which is the one currency all
three share: a MAttr step and a stepless-IG alpha draw are each exactly one. (Wall clock is
not matched and should not be the axis -- MAttr is ~7x slower per step here because
`sigmoid_topk`'s bisection dominates a 128-feature model, which is a statement about the toy's
size rather than about the method.)

Two panels because the two say opposite things about the same runs, and only one of them is
the headline. Spearman against `dL` is a GLOBAL agreement measure over all 16384 weights;
precision at recall 0.8 reads the tail of the ranking, which is where every method is still
losing. A method can climb on one and sit flat on the other.

Bands are min-max across seeds, not a CI -- with n=3 that is the honest thing to draw, and it
is what makes the Adam decline readable as "inside the seed spread at the wide end" rather
than as a clean effect.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from palette import COLOR, RC, furnish

ROOT = Path(__file__).resolve().parents[1]
STYLE = {"ixg:mc": ("Stepless IG", COLOR["ixg:mc"]),
         "adam": ("MAttr (Adam)", COLOR["adam"]),
         "sgd": ("MAttr (SGD)", COLOR["sgd"])}
PANELS = [("rho", "Spearman vs $\\Delta L$ (all weights)"),
          ("p80", "Precision at recall 0.8 (tail)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"

    runs = {}
    for f in sorted(d.glob("budget_*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["method"], []).append(r["metrics"])

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9))
    for ax, (key, ylab) in zip(axes, PANELS):
        for meth, (label, color) in STYLE.items():
            seeds = runs.get(meth, [])
            if not seeds:
                continue
            xs = sorted(int(k) for k in seeds[0])
            lo, mid, hi = [], [], []
            for x in xs:
                vals = [s[str(x)][key] for s in seeds if str(x) in s]
                lo.append(min(vals)); hi.append(max(vals))
                mid.append(sum(vals) / len(vals))
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(xs, mid, lw=1.0, color=color, marker="o", ms=2.2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Forward/backward passes")
        ax.set_ylabel(ylab)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        furnish(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.07), handlelength=1.6, columnspacing=1.2,
               handletextpad=0.5)
    fig.tight_layout()
    out = ROOT / "plots" / f"interference_budget_{a.tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
