"""What the circuit-only residual r_i is made of: lost interference offset, not shrinkage.

    uv run python plots/plot_interference_residual.py --tags hard30k lit

One row per tag, three panels over the LIVE target rows (A_i != 0):
  (a) r_i against the interference's mean contribution to that row, E[x] . sum_j U_ij^off --
      the constant the bias absorbed and the circuit-only model loses;
  (b) r_i against the shrinkage deficit E[x] . sum_j (A_ij - U_ij) over the circuit;
  (c) |r_i| per row under five reference models -- full, circuit-only, circuit-only with the
      interference offset put back into the bias, circuit-only with the circuit unshrunk to A,
      and both -- as strip plots with the mean marked.
r_i = E[(y_i - y'_i) 1(gate open)] is the per-row scalar behind U_ij r_i, the feature that orders
what MAttr+Adam keeps off the circuit (docs/interference_toy.md). Raw matplotlib.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402

from palette import RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"hard": "hard, 3k steps", "hard30k": "hard, 30k steps", "lit": "lit"}
PT = "#4d4d4d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["hard30k", "lit"])
    a = ap.parse_args()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(a.tags), 3, figsize=(5.5, 1.9 * len(a.tags)), squeeze=False)
    for row, tag in enumerate(a.tags):
        m = torch.load(ROOT / "plots" / "data" / f"interference_toy_{tag}" / "model.pt")
        U, b, A, v = m["U"], m["b"], m["A"], m["v"]
        IT.N_FEAT, IT.N_RES = U.shape[0], 16
        X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
        Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
        sup = A > 0
        live = sup.any(1)

        def resid(W, bias):
            with torch.no_grad():
                z = X @ W.T + bias
                return ((Y - F.relu(z)) * (z > 0)).mean(0)

        ex = X.mean(0)
        offset = (U * ~sup) @ ex
        deficit = (A - U * sup) @ ex
        r = resid(U * sup, b)
        variants = [("full", resid(U, b)), ("circuit", r),
                    ("circuit\n+offset", resid(U * sup, b + offset)),
                    ("$A$, $b$", resid(A, b)),
                    ("$A$, $b$\n+offset", resid(A, b + offset))]
        ax = axes[row, 0]
        ax.scatter(offset[live], r[live], s=5, lw=0, color=PT, alpha=0.8)
        rho = float(torch.corrcoef(torch.stack([offset[live], r[live]]))[0, 1])
        ax.set_xlabel("Interference offset  $E[x]\\,\\sum_j U^{\\mathrm{off}}_{ij}$", size=7)
        ax.set_ylabel(f"{TITLES.get(tag, tag)}\n\nCircuit-only residual $r_i$", size=7)
        ax.text(0.04, 0.95, f"$r$ = {rho:+.2f}", transform=ax.transAxes, va="top", size=7)
        ax = axes[row, 1]
        ax.scatter(deficit[live], r[live], s=5, lw=0, color=PT, alpha=0.8)
        rho = float(torch.corrcoef(torch.stack([deficit[live], r[live]]))[0, 1])
        ax.set_xlabel("Shrinkage deficit  $E[x]\\,\\sum_j (A_{ij}-U_{ij})$", size=7)
        ax.text(0.04, 0.95, f"$r$ = {rho:+.2f}", transform=ax.transAxes, va="top", size=7)
        ax = axes[row, 2]
        rng = np.random.default_rng(0)
        for i, (lab, rr) in enumerate(variants):
            vals = rr[live].abs().numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, vals.size), vals, s=3, lw=0, color=PT,
                       alpha=0.5)
            ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, lw=1.2, color="#b2182b")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([lab for lab, _ in variants], size=5.5)
        ax.set_xlim(-0.5, len(variants) - 0.5)
        ax.set_yscale("log")
        ax.set_ylabel("$|r_i|$ per live row", size=7)
        ax.text(0.97, 0.95, "mean in red", transform=ax.transAxes, ha="right", va="top", size=6,
                color="#b2182b")
        for ax in axes[row, :2]:
            ax.axhline(0, lw=0.4, color="#bbbbbb", zorder=0)
            ax.axvline(0, lw=0.4, color="#bbbbbb", zorder=0)
        for ax in axes[row]:
            ax.tick_params(labelsize=5.5)
            furnish(ax)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    out = ROOT / "plots" / "interference_residual.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
