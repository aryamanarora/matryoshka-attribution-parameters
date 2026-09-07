"""WHERE stepless IG and the oracle disagree, and why: target firing rate is the axis.

    uv run python plots/plot_interference_disagreement.py --tag hard

THE CLAIM THIS FIGURE IS FOR. `dL` ablates ONE weight with the others at full strength;
stepless IG integrates a path on which ALL weights scale together. Those coincide only if the
loss is additively separable in the weights, and a ReLU makes it emphatically not. The
prediction is that they diverge exactly where a target's weights INTERACT -- and a target whose
ReLU is open on nearly every example is where interaction is maximal, because every weight
feeding it is live along the whole path. So the disagreement should be organised by
`P(y'_i > 0)`, a property of the target ROW, and not by anything about the individual weight.

Left panel: the score-vs-score scatter recoloured by that firing rate, so the over-attributed
cloud can be checked against it directly. Right panel: the residual `IG - dL` (the two are in
the same units, which is what makes a residual meaningful at all) binned by firing rate, split
by whether the weight is on the target circuit `A`.

The split is the part that makes it a claim about the CIRCUIT rather than about activity: if
the residual merely tracked how busy a row is, both lines would rise together. They separate --
off-circuit weights are over-attributed and on-circuit ones under-attributed, in the same rows.

`P(y' > 0)` is computed EXACTLY here, by one pass over the eval set. An earlier version of this
analysis estimated it as `mean_j co_ij / 0.3`, which assumes source activity and target firing
are independent; they are not, and a figure should not carry an approximation that a single
forward pass removes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402

from palette import RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ON, OFF = "#b2182b", "#4d4d4d"
N_EVAL, CHUNK = 131_072, 8192


def firing_rate(U, b, A, v):
    """P(y'_i > 0), exactly, over the same eval distribution the statistics use."""
    acc, seen = torch.zeros(U.shape[0]), 0
    for x, _y in IT.eval_batches(A, v, N_EVAL, CHUNK, 0):
        acc += (F.relu(x @ U.T + b) > 0).float().sum(0)
        seen += x.shape[0]
    return acc / seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    s = torch.load(d / "scores.pt")["ixg:mc"]

    p = firing_rate(U, b, A, v).unsqueeze(1).expand_as(U)     # row property
    resid = (s - dl)
    sup = A > 0

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

    sc = axes[0].scatter(dl.reshape(-1), s.reshape(-1), c=p.reshape(-1), s=1.0, lw=0,
                         cmap="viridis", vmin=0, vmax=1, rasterized=True)
    lim = float(max(dl.abs().max(), s.abs().max()))
    axes[0].plot([-lim, lim], [-lim, lim], lw=0.5, ls=(0, (3, 2)), color="#666666", zorder=0)
    axes[0].set_xscale("symlog", linthresh=1.28e-05)
    axes[0].set_yscale("symlog", linthresh=1.28e-05)
    axes[0].set_xlabel("Oracle score  $\\Delta L(U_{ij})$")
    axes[0].set_ylabel("Stepless IG score")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("$P(y'_i>0)$", size=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.5)

    edges = torch.linspace(0, 1, 11)
    for mask, color, lab in ((sup, ON, "on circuit $A$"), (~sup, OFF, "off circuit")):
        xs, med, lo, hi = [], [], [], []
        for k in range(len(edges) - 1):
            sel = mask & (p >= edges[k]) & (p < edges[k + 1] + (k == len(edges) - 2))
            if sel.sum() < 20:
                continue
            r = resid[sel]
            xs.append(float((edges[k] + edges[k + 1]) / 2))
            med.append(float(r.median()))
            lo.append(float(r.quantile(.25))); hi.append(float(r.quantile(.75)))
        axes[1].fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
        axes[1].plot(xs, med, lw=1.0, color=color, marker="o", ms=2.4, label=lab)
    axes[1].axhline(0, lw=0.5, color="#888888", zorder=0)
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[1].set_xlabel("Target firing rate  $P(y'_i>0)$")
    axes[1].set_ylabel("Residual  IG $-\\ \\Delta L$")
    axes[1].legend(fontsize=5.5, frameon=False, loc="upper left")

    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        furnish(ax)
    fig.tight_layout()
    out = ROOT / "plots" / f"interference_disagreement_{a.tag}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
