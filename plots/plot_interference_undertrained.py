"""Training trajectories of the note's literal config against the note's published numbers.

    uv run python plots/plot_interference_undertrained.py

Reads plots/data/interference_undertrained/trajectory_<tag>.json (written by
scripts/interference/interference_undertrained.py, one per (lr, schedule, init scale, block
density)) and draws, against training step, the quantities the note's validation figures and
P/R curves fix: the interference-weight std, the base rate, and the precision of the note's
three magnitude heuristics at fixed recall. Each panel carries the published value as a grey
line, so "does any snapshot of any run reproduce the note" is read as "do the curves cross the
grey lines at the same step". Raw matplotlib: one legend strip for ten trajectories and per-panel
reference lines. Runs are coloured by init scale (viridis) and dashed by block density, since
those are the two knobs that moved anything; learning rate and schedule did not.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from palette import RC, furnish

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "interference_undertrained"

PANELS = (("interf_std", "Interference-weight std"), ("base_rate", "Base rate ($\\Delta L>\\epsilon$)"),
          ("weight_p20", "Virtual weight, P@R=0.2"), ("era_p40", "ERA, P@R=0.4"),
          ("twera_p40", "TWERA, P@R=0.4"), ("loss", "Training loss"))


def main():
    runs = {f.stem.replace("trajectory_", ""): json.loads(f.read_text())
            for f in sorted(DATA.glob("trajectory_*.json"))}
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.2))
    cmap = plt.get_cmap("viridis")
    for tag, r in runs.items():
        a = r["args"]
        init, bd = a.get("init_scale", 1.0), a.get("block_density", 0.1)
        color = cmap((init - 1) / 3)
        ls = (0, (3.5, 1.5)) if bd > 0.1 else "solid"
        steps = [row["step"] for row in r["rows"]]
        label = f"init ×{init:g}, lr {a['lr']:g} {a['schedule']}" + (f", blocks {bd:g}" if bd > 0.1 else "")
        for ax, (key, _) in zip(axes.flat, PANELS):
            ax.plot(steps, [row[key] for row in r["rows"]], color=color, ls=ls, lw=0.9,
                    marker="o", ms=1.8, label=label)
    pub = next(iter(runs.values()))["published"]
    for ax, (key, title) in zip(axes.flat, PANELS):
        if key in pub:
            ax.axhline(pub[key], lw=0.8, color="#888888", zorder=0)
            ax.annotate("published", (10, pub[key]), xytext=(2, 2), textcoords="offset points",
                        fontsize=5, color="#666666")
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000, 10000])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels(["10", "10²", "10³", "10⁴"])
        if key == "base_rate":
            ax.set_yscale("log")
        if key == "loss":
            ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.set_ylabel(title, size=7)
        if ax in axes[1]:
            ax.set_xlabel("Training step", size=7)
        furnish(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, 1.1), handlelength=1.8, columnspacing=0.9)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    out = ROOT / "plots" / "interference_undertrained.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
