"""Precision-recall of the note's heuristics at every snapshot of every undertraining trajectory.

    uv run python plots/plot_interference_undertrained_pr.py

One row per trajectory of scripts/interference/interference_undertrained.py, one column per
training step, each panel the note's first figure (weight / ERA / TWERA / freq against that
snapshot's own dL) with the base rate as a grey line and the note's published operating points
as hollow markers in each heuristic's colour: weight (0.08, 0.68) (0.2, 0.44), ERA (0.2, 0.87)
(0.4, 0.55), TWERA (0.2, 0.81) (0.4, 0.47). The question the grid answers by eye: is there ANY
cell where the three curves pass through their three pairs of markers at once? Statistics are
recomputed from the saved snapshot on a fresh eval set; dL is the one saved with it.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402
from interference_toy import EPS_REAL, heuristics, sweep  # noqa: E402

from palette import RC, furnish  # noqa: E402
from plot_interference_filtering import NOTE_STYLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "interference_undertrained"
STEPS = (50, 100, 200, 300, 500, 1000, 3000, 10000)
PUBLISHED = {"weight": ((0.08, 0.68), (0.2, 0.44)), "era": ((0.2, 0.87), (0.4, 0.55)),
             "twera": ((0.2, 0.81), (0.4, 0.47))}
N_EVAL = 32_768


def main():
    runs = sorted(DATA.glob("trajectory_*.json"))
    tags = [f.stem.replace("trajectory_", "") for f in runs]
    args = {t: json.loads(f.read_text())["args"] for t, f in zip(tags, runs)}
    # order: stated init first, then init scale, then the 0.5-block runs
    tags.sort(key=lambda t: (args[t].get("block_density", 0.1), args[t].get("init_scale", 1),
                             args[t]["lr"], args[t]["schedule"]))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(tags), len(STEPS), figsize=(5.5, 0.72 * len(tags) + 0.4),
                             sharex=True, sharey=True)
    for r, tag in enumerate(tags):
        a = args[tag]
        IT.BLOCK_DENSITY = a.get("block_density", 0.1)
        for c, step in enumerate(STEPS):
            ax = axes[r, c]
            f = DATA / f"snap_{tag}_{step}.pt"
            if not f.exists():
                ax.set_visible(False)
                continue
            m = torch.load(f)
            U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
            stats = IT.statistics(U, b, A, v, N_EVAL, 8192, 99)
            for name, s in heuristics(U, stats).items():
                cur = sweep(s, dl)
                ax.plot(cur["recall"], cur["precision"], color=NOTE_STYLE[name][1], lw=0.7)
            for name, pts in PUBLISHED.items():
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker="o", ms=2.6,
                        mfc="white", mec=NOTE_STYLE[name][1], mew=0.7)
            base = float((dl > EPS_REAL).float().mean())
            ax.axhline(base, lw=0.5, ls=(0, (3, 2)), color="#888888", zorder=0)
            ax.text(0.97, 0.93, f"base {base:.1%}", transform=ax.transAxes, ha="right",
                    va="top", size=4.5, color="#444444")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            furnish(ax)
            if r == 0:
                ax.set_title(f"step {step}", size=5.5, pad=2)
        label = (f"init ×{a.get('init_scale', 1):g}\nlr {a['lr']:g} {a['schedule']}"
                 + (f"\nblocks {a.get('block_density', 0.1):g}" if a.get("block_density", 0.1) > 0.1 else ""))
        axes[r, 0].set_ylabel(label, size=5, rotation=0, ha="right", va="center", labelpad=4)
    for ax in axes[-1]:
        ax.set_xlabel("Recall", size=5.5)
    handles = [plt.Line2D([], [], color=NOTE_STYLE[k][1], lw=1, label=NOTE_STYLE[k][0])
               for k in ("weight", "era", "twera", "freq")]
    handles.append(plt.Line2D([], [], ls="none", marker="o", mfc="white", mec="#444444", ms=3,
                              label="published operating points"))
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0 + 0.5 / (0.72 * len(tags) + 0.4)), columnspacing=1.0,
               handlelength=1.6)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.05, hspace=0.25, wspace=0.12)
    out = ROOT / "plots" / "interference_undertrained_pr.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
