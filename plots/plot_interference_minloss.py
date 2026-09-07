"""Which weights each method keeps at ITS OWN loss-optimal sparsity, against the true circuit.

    uv run python plots/plot_interference_minloss.py --tag hard

Four panels: the target circuit `A`, then the kept set of stepless IG, MAttr-SGD and MAttr-Adam
at the `k` that minimises each one's true (masked-forward) loss. The three `k` are NOT the same
-- 136, 176 and 1400 -- and that is the point rather than an inconsistency: each method is shown
at its own optimum, so the comparison is "the best filtered model this ranking can give you",
not "these rankings at some shared budget".

Every cell is one of four categories, and the encoding is the one used across this figure set
(red = circuit, grey = interference):

    kept, on circuit      dark red     the weights you wanted
    missed, on circuit    light red    circuit the method did not find
    kept, off circuit     grey         interference it kept anyway
    neither               white

DRAWN AS A SCATTER, NOT `imshow`. At 1.25in a 128x128 image gives each weight a 0.7pt cell, and
the panels are ~1% dense, so the circuit would render as scattered single pixels that vanish in
print. Square markers decouple mark size from matrix resolution.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TP, FN, FP = "#b2182b", "#f0b0b0", "#9e9e9e"
FS_TITLE, FS_TICK, FS_LEG = 6.0, 5.0, 5.0
LABEL = {"ixg:mc": "stepless IG", "sgd": "MAttr (SGD)", "adam": "MAttr (Adam)"}
MS = 1.6


def draw(ax, n, circuit, kept):
    """circuit/kept are boolean (n, n). Order matters: FP under FN under TP."""
    def pts(mask, color, s=MS):
        i, j = mask.nonzero(as_tuple=True)
        ax.scatter(j.numpy(), i.numpy(), s=s, marker="s", lw=0, color=color, rasterized=True)
    if kept is not None:
        pts(kept & ~circuit, FP)
        pts(circuit & ~kept, FN)
        pts(circuit & kept, TP)
    else:
        pts(circuit, TP)
    ax.set_xlim(-2, n + 1); ax.set_ylim(n + 1, -2)
    ax.set_aspect("equal")
    ax.set_xticks([0, 64, 127]); ax.set_yticks([0, 64, 127])
    ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
    furnish(ax)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    blob = torch.load(d / "minloss_sets.pt")
    A, sets = blob["A"], blob["sets"]
    n = A.shape[0]
    circuit = A > 0

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.75))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.16, wspace=0.30)

    draw(axes[0], n, circuit, None)
    axes[0].set_title(f"circuit $A$  ({int(circuit.sum())})", fontsize=FS_TITLE, pad=3)
    axes[0].set_ylabel("target $i$", fontsize=FS_TITLE)

    for ax, name in zip(axes[1:], ("ixg:mc", "sgd", "adam")):
        s = sets[name]
        kept = torch.zeros(n * n, dtype=torch.bool)
        kept[s["idx"]] = True
        kept = kept.view(n, n)
        draw(ax, n, circuit, kept)
        ax.set_title(f"{LABEL[name]}\n$k$={s['k']},  $L$={s['loss']:.3f}",
                     fontsize=FS_TITLE, pad=3, linespacing=1.35)
    axes[0].set_title(f"circuit $A$  ({int(circuit.sum())})\n$L$={blob['L_circuit']:.3f}",
                      fontsize=FS_TITLE, pad=3, linespacing=1.35)
    for ax in axes:
        ax.set_xlabel("source $j$", fontsize=FS_TITLE, labelpad=1)

    handles = [Line2D([0], [0], marker="s", ls="none", ms=2.6, mfc=c, mec="none", label=t)
               for c, t in ((TP, "kept, on circuit"), (FN, "missed, on circuit"),
                            (FP, "kept, off circuit"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS_LEG, frameon=False,
               bbox_to_anchor=(0.5, -0.13), handletextpad=0.3, columnspacing=1.2)

    out = ROOT / "plots" / f"interference_minloss_{a.tag}.pdf"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
