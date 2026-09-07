"""What orders MAttr+Adam's scores, population by population: a 2x3 grid on one toy run.

    uv run python plots/plot_interference_adam_order.py --tag hard30k

Columns are the three populations of virtual weights -- the circuit (A_ij > 0), the off-circuit
weights inside Adam's loss-optimal top-k, and the off-circuit weights outside it (dead rows
dropped: they are one point). Rows are the two candidate orderings: the full-model single-weight
effect dL (the note's oracle) and the sparse-model value U_ij * r_i (the weight's sign times its
target row's under-prediction in the circuit-only model). Each panel is Adam's score against
that feature for that population, with the Spearman in the corner. The figure exists to show
that Adam's ranking is piecewise: dL orders the circuit and the discarded interference (0.99,
0.93), U*r_i orders the kept interference (0.83), and inside the kept set dL runs BACKWARDS
(-0.83) -- the interference Adam keeps is what the dense model would most like removed.

Raw matplotlib: symlog panels with per-panel linear regions, a shared row/column layout, and
corner statistics. Colours: the circuit red and the two interference populations in the palette's
MAttr blue (kept) and reference grey (not kept), so "kept" reads as the method's own choice.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402
from interference_scale import spearman  # noqa: E402

from palette import COLOR, RC, furnish  # noqa: E402
from plot_interference_scale import symlog_ticks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ON, KEPT, REST = "#b2182b", COLOR["adam"], "#7a7a7a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard30k")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    adam = torch.load(d / "scores.pt")["adam"].reshape(-1)
    on = (A > 0).reshape(-1)
    dead = (~(A > 0).any(1)).unsqueeze(1).expand_as(A).reshape(-1) & ~on
    best = None
    for grid in ("log", "linear"):
        tl = json.loads((d / f"true_loss_{grid}.json").read_text())
        curve = tl["curves"]["adam"]
        i = min(range(len(curve)), key=curve.__getitem__)
        if best is None or curve[i] < best[1]:
            best = (tl["grid"][i], curve[i])
    kept = torch.zeros(U.numel(), dtype=torch.bool)
    kept[adam.argsort(descending=True)[:best[0]]] = True
    X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
    Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
    with torch.no_grad():
        zc = X @ (U * (A > 0)).T + b
        r = ((Y - F.relu(zc)) * (zc > 0)).mean(0)
    ur = (U * r.unsqueeze(1)).reshape(-1)
    dl = dl.reshape(-1)

    pops = (("on circuit $A$", on, ON, 5),
            (f"off circuit, kept by Adam (top-{best[0]})", kept & ~on & ~dead, KEPT, 2.5),
            ("off circuit, not kept", ~kept & ~on & ~dead, REST, 1.2))
    feats = (("Oracle $\\Delta L$ (full-model effect)", dl),
             ("$U_{ij}\\,r_i$ (sparse-model value)", ur))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.4))
    for rrow, (fname, f) in enumerate(feats):
        for c, (pname, msk, col, ms) in enumerate(pops):
            ax = axes[rrow, c]
            sx, sy = f[msk], adam[msk]
            lx = float(sx.abs().median()) or 1e-6
            ly = float(sy.abs().median()) or 1e-6
            ax.scatter(sx, sy, s=ms, lw=0, color=col, alpha=0.6 if ms < 5 else 0.9,
                       rasterized=True)
            ax.set_xscale("symlog", linthresh=lx)
            ax.set_yscale("symlog", linthresh=ly)
            symlog_ticks(ax, "x", lx, float(sx.abs().max()))
            symlog_ticks(ax, "y", ly, float(sy.abs().max()))
            ax.axhline(0, lw=0.4, color="#bbbbbb", zorder=0)
            ax.axvline(0, lw=0.4, color="#bbbbbb", zorder=0)
            rho = spearman(sx, sy)
            ax.text(0.04, 0.95, f"$\\rho$ = {rho:+.2f}", transform=ax.transAxes, va="top",
                    size=7, fontweight="bold" if abs(rho) > 0.8 else "normal")
            ax.tick_params(labelsize=5.5)
            if rrow == 0:
                ax.set_title(pname, size=6.5, pad=3)
            if c == 0:
                ax.set_ylabel("MAttr+Adam score", size=7)
            if rrow == 1:
                ax.set_xlabel(fname.split(" (")[0], size=7)
            furnish(ax)
        axes[rrow, 2].text(1.04, 0.5, fname, transform=axes[rrow, 2].transAxes, rotation=270,
                           va="center", ha="left", size=6.5, color="#333333")
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    out = ROOT / "plots" / f"interference_adam_order_{a.tag}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
