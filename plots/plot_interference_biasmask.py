"""True loss against weights kept, with and without a per-row bias learned alongside the mask.

    uv run python plots/plot_interference_biasmask.py --tag hard30k

Four curves from scripts/interference/interference_biasmask.py: the plain MAttr+Adam ranking
under the original bias (the coalition case), the same ranking under the learned b + db, and
the bias-mask ranking under each bias. Reference lines: the full model and the circuit alone,
under b and under b + db. The point of the figure: with a free per-row constant the loss
minimum moves from k ~ 1200 (1000 interference weights) to k ~ 160 (the circuit) and gets
LOWER, i.e. the coalition was a bias. Raw matplotlib, log-x.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from palette import COLOR, RC, furnish

ROOT = Path(__file__).resolve().parents[1]
STYLE = {"plain Adam, b": ("MAttr+Adam, original $b$", COLOR["adam"], "solid"),
         "plain Adam, b+db": ("MAttr+Adam, $b+\\delta b$", COLOR["adam"], (0, (3, 1.5))),
         "scores, b": ("bias-mask scores, original $b$", "#d55e00", "solid"),
         "scores+db": ("bias-mask scores, $b+\\delta b$", "#d55e00", (0, (3, 1.5)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard30k")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    bm = json.loads((d / "biasmask.json").read_text())
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for key, (lab, col, ls) in STYLE.items():
        c = bm["curves"][key]
        ax.plot(bm["grid"], c["loss"], lw=1.0, color=col, ls=ls, label=lab)
        ax.plot([c["k_best"]], [c["L_best"]], marker="o", ms=3, color=col, mec="white", mew=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Weights kept, $k$", size=7)
    ax.set_ylabel("True loss of the masked model", size=7)
    ax.tick_params(labelsize=6)
    lo = min(min(c["loss"]) for c in bm["curves"].values())
    ax.set_ylim(lo - 0.05, lo + 1.2)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=2.2)
    furnish(ax)
    fig.tight_layout()
    out = ROOT / "plots" / f"interference_biasmask_{a.tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
