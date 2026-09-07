"""True loss of the filtered model against how many weights you keep.

    uv run python plots/plot_interference_true_loss.py --tag hard

Every point is a real masked forward pass, not the note's additive `sum of dL` proxy -- which
is the whole reason the figure exists, because the proxy is monotone by construction and the
truth is not.

Series and colours come from plot_interference_filtering, so a ranking is the same colour in
both figures. `random` is added here as the reference that makes the shape legible: it is what
"keeping more weights helps because it is more of the model" looks like, so a curve only says
something where it leaves the random line.

Read the y axis in log: the action spans a 3.2-4.7 band AND excursions to 15, and on a linear
axis the excursions would flatten everything else into a line.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_interference_filtering import STYLE, DASHED  # noqa: E402

from palette import RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["ideal", "era", "twera", "weight", "freq", "ixg:mc", "adam", "sgd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--grid", default="log", choices=("log", "linear"))
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    blob = json.loads((d / f"true_loss_{a.grid}.json").read_text())
    g, c = blob["grid"], blob["curves"]

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    ax.plot(g, c["random"], lw=1.0, color="#cccccc", ls=(0, (4, 2)), label="Random ranking",
            zorder=1)
    for name in ORDER:
        label, color = STYLE[name]
        ax.plot(g, c[name], lw=1.0, color=color,
                ls=(0, (3.5, 1.5)) if name in DASHED else "solid", label=label, zorder=2)

    for y, lab, ls in ((blob["L_full"], "full model", (0, (1, 1.5))),
                       (blob["L_zero"], "nothing kept", (0, (1, 1.5))),
                       (blob["L_circuit"], "true circuit $A$ only", (0, (5, 2)))):
        ax.axhline(y, lw=0.5, ls=ls, color="#888888", zorder=0)
        ax.annotate(lab, (1.05, y), xytext=(0, 2), textcoords="offset points",
                    fontsize=5.5, color="#666666")
    ax.axvline(blob["n_circuit"], lw=0.5, ls=(0, (1, 1.5)), color="#888888", zorder=0)
    ax.annotate(f"$|A|$ = {blob['n_circuit']}", (blob["n_circuit"], 15.5), xytext=(2, 0),
                textcoords="offset points", fontsize=5.5, color="#666666")

    if a.grid == "log":
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Virtual weights kept")
    ax.set_ylabel("True loss of the filtered model")
    ax.set_yticks([3.5, 4, 5, 7, 10, 15])
    ax.set_yticklabels(["3.5", "4", "5", "7", "10", "15"])
    # A log axis keeps its own minor labels, which on this range prints a stray "6 x 10^0"
    # on top of the hand-set major ticks.
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    furnish(ax)
    ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper left", handlelength=1.8,
              columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    out = ROOT / "plots" / (f"interference_true_loss_{a.tag}.pdf" if a.grid == "log"
                            else f"interference_true_loss_{a.grid}_{a.tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
