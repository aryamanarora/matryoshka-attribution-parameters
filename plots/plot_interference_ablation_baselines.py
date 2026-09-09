"""True loss against weights kept under three ablation baselines: zero, mean, learned.

    uv run python plots/plot_interference_ablation_baselines.py --tags hard30k lit

What a MASKED weight contributes is a choice, and the coalition MAttr+Adam keeps under zero
ablation is a consequence of that choice. Curves, per config:
    MAttr+Adam ranking (fitted under zero ablation), evaluated under zero ablation   -- the baseline
    the same ranking evaluated under MEAN ablation (c_ij = U_ij E[x_j], nothing learned)
    MAttr+Adam FITTED under mean ablation, evaluated under it
    MAttr+Adam fitted jointly with a per-row bias db (uniform k), evaluated with b + db
    MAttr+Adam fitted jointly with a per-weight constant C, evaluated with C
Reference lines: the circuit alone under zero and under mean ablation. Reads
constmask_meanabl.json, constmask_mean_log.json and biasmask.json from the run directory.
Raw matplotlib, log-x, minimum of each curve marked.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from palette import COLOR, RC, furnish

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"hard": "hard, 3k steps", "hard30k": "hard, 30k steps", "lit": "lit"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["hard30k", "lit"])
    a = ap.parse_args()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(a.tags), figsize=(2.75 * len(a.tags), 2.4), squeeze=False)
    for ax, tag in zip(axes[0], a.tags):
        d = ROOT / "plots" / "data" / f"interference_toy_{tag}"
        ma = json.loads((d / "constmask_meanabl.json").read_text())
        series = [("MAttr+Adam, zero ablation", ma["grid"], ma["curves"]["plain Adam, zero abl"],
                   COLOR["adam"], "solid"),
                  ("same ranking, mean ablation", ma["grid"], ma["curves"]["plain Adam, mean abl"],
                   COLOR["adam"], (0, (3, 1.5))),
                  ("MAttr+Adam fitted under mean ablation", ma["grid"], ma["curves"]["scores + C"],
                   "#009e73", "solid")]
        f = d / "biasmask.json"
        if f.exists():
            bm = json.loads(f.read_text())
            series.append(("MAttr+Adam + learned per-row bias", bm["grid"], bm["curves"]["scores+db"],
                           "#d55e00", "solid"))
        f = d / "constmask_mean_log.json"
        if f.exists():
            cm = json.loads(f.read_text())
            series.append(("MAttr+Adam + learned per-weight constant", cm["grid"], cm["curves"]["scores + C"],
                           "#984ea3", (0, (1.5, 1.5))))
        for lab, grid, c, col, ls in series:
            ax.plot(grid, c["loss"], lw=1.0, color=col, ls=ls, label=lab)
            ax.plot([c["k_best"]], [c["L_best"]], marker="o", ms=3, color=col, mec="white", mew=0.5)
        lo = min(c["L_best"] for _, _, c, _, _ in series)
        ax.set_xscale("log")
        ax.set_ylim(lo - 0.05, lo + 0.9)
        ax.set_xlabel("Weights kept, $k$", size=7)
        ax.set_ylabel("True loss of the masked model", size=7)
        ax.text(0.97, 0.95, TITLES.get(tag, tag), transform=ax.transAxes, ha="right", va="top", size=7)
        ax.tick_params(labelsize=6)
        furnish(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, 1.12), handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    out = ROOT / "plots" / "interference_ablation_baselines.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
