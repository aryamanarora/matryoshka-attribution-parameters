"""What predicts the interference weights MAttr+Adam KEEPS, against size and config.

    uv run python plots/plot_interference_adam_pattern.py

Reads plots/data/interference_scale/<tag>/adam_pattern_n<n>.json (written by
scripts/interference/interference_adam_pattern.py) and draws, for each candidate per-weight feature, the AUC
for membership in Adam's loss-optimal top-k set among the off-circuit weights, one panel per
config. 0.5 is chance. The figure exists to show one thing: a single feature -- the weight's
sign times its target row's under-prediction in the circuit-only model, `U_ij * r_i` -- predicts
the kept set at every size on both configs, and the oracle `dL` does not.

Raw matplotlib (shared legend strip, per-panel reference line). Method colours from
plots/palette.py where a feature IS a method's score; the hand-built features take Set1 hues
that no method owns, per plot_interference_filtering.py's rule.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from palette import COLOR, RC, furnish
from plot_interference_scale import fmt_pow

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "interference_scale"

FEATURES = {                                   # json key -> (label, colour, linestyle)
    "AUC kept: U·r_i|Adam": ("$U_{ij}\\,r_i$  (sign × row under-prediction)", "#984ea3", "solid"),
    "AUC kept: U (signed)|Adam": ("$U_{ij}$ (signed)", "#4daf4a", (0, (3.5, 1.5))),
    "AUC kept: -|U||Adam": ("$-|U_{ij}|$", "#4daf4a", (0, (1.5, 1.5))),
    "AUC kept: p_i (full)|Adam": ("target firing rate $p_i$", "#ff7f00", (0, (3.5, 1.5))),
    "AUC kept: dL (oracle)|Adam": ("oracle $\\Delta L$", "#e41a1c", "solid"),
    "AUC kept: ixg @ full|Adam": ("I×G @ full model", COLOR["ixg:base"], (0, (1.5, 1.5))),
    "AUC kept: stepless IG|Adam": ("stepless IG score", COLOR["ixg:mc"], "solid"),
    "AUC kept: SGD|Adam": ("MAttr (SGD) score", COLOR["sgd"], "solid"),
}


def main():
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), sharey=True)
    for ax, tag in zip(axes, ("hard", "lit")):
        files = sorted(DATA.glob(f"{tag}/adam_pattern_n*.json"),
                       key=lambda f: int(f.stem.split("_n")[1]))
        ns = np.array([int(f.stem.split("_n")[1]) for f in files])
        blobs = [json.loads(f.read_text()) for f in files]
        for key, (lab, color, ls) in FEATURES.items():
            m = np.array([np.mean(b[key]) for b in blobs])
            s = np.array([np.std(b[key], ddof=1) if len(b[key]) > 1 else 0 for b in blobs])
            ax.fill_between(ns ** 2, m - s, m + s, color=color, alpha=0.12, lw=0)
            ax.plot(ns ** 2, m, color=color, ls=ls, lw=1.0, marker="o", ms=2.2, label=lab)
        ax.axhline(0.5, lw=0.5, color="#888888", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns ** 2)
        ax.set_xticklabels([fmt_pow(n * n) for n in ns])
        ax.text(0.03, 0.04, {"hard": "frozen projection", "lit": "the note's config"}[tag],
                transform=ax.transAxes, size=6.5, va="bottom")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Virtual weights", size=7)
        furnish(ax)
    axes[0].set_ylabel("AUC for membership in\nAdam's kept off-circuit set", size=7)
    axes[0].set_ylim(0.25, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.2), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    out = ROOT / "plots" / "interference_adam_pattern.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
