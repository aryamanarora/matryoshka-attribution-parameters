"""Where two MAttr fits rank each KIND of interference weight: a 2x2 of rank histograms.

    uv run python plots/plot_interference_rank_2x2.py --tag hard30k

The two questions that decide an interference weight's fate, as the axes of the grid:
  rows     does switching it on HELP once the interference is gone?   U_ij r_i > 0  /  <= 0
           (r_i = how far row i comes out too low in the circuit-only model)
  columns  is it NEEDED in the full model?                             dL > 0  /  <= 0
           (dL = the loss increase from ablating it from the full model, the note's oracle)
Each cell: histograms over log-rank of that kind of weight under the plain MAttr+Adam ranking
and under the ranking fitted jointly with a per-row bias, with each fit's loss-optimal k as a
vertical line, and the number of weights in the cell. The circuit's own rank range is shaded so
"right after the circuit" can be read. The point: the plain fit promotes the top-left-of-bottom
row cell -- weights that help the sparse model but the full model does not need -- to just
after the circuit and keeps them; the bias fit does not, and orders interference by the
full-model column instead. Dead rows (one tied score) are left out. Raw matplotlib.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402

from palette import COLOR, RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAIN, BIAS = COLOR["adam"], "#d55e00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard30k")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
    Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
    sup = A > 0
    on = sup.reshape(-1)
    dead = (~sup.any(1)).unsqueeze(1).expand_as(A).reshape(-1) & ~on
    plain = torch.load(d / "scores.pt")["adam"].reshape(-1)
    bm = torch.load(d / "biasmask.pt")
    bmj = json.loads((d / "biasmask.json").read_text())["curves"]
    best_plain = None
    for grid in ("log", "linear"):
        tl = json.loads((d / f"true_loss_{grid}.json").read_text())
        c = tl["curves"]["adam"]
        i = min(range(len(c)), key=c.__getitem__)
        if best_plain is None or c[i] < best_plain[1]:
            best_plain = (tl["grid"][i], c[i])
    fits = [("MAttr+Adam", plain, b, best_plain[0], PLAIN),
            ("MAttr+Adam + learned per-row bias", bm["scores"].reshape(-1), b + bm["db"],
             bmj["scores+db"]["k_best"], BIAS)]
    n2 = U.numel()
    edges = np.logspace(0, np.log10(n2), 36)

    def ranks(s):
        r = torch.empty(n2, dtype=torch.long)
        r[s.argsort(descending=True)] = torch.arange(1, n2 + 1)
        return r.numpy()

    def helps_sparse(bias):
        with torch.no_grad():
            zc = X @ (U * sup).T + bias
            r = ((Y - F.relu(zc)) * (zc > 0)).mean(0)
        return (U * r.unsqueeze(1)).reshape(-1) > 0

    needed = dl.reshape(-1) > 0
    row_lab = ("helps once the\ninterference is gone\n($U_{ij}\\,r_i>0$)",
               "hurts once the\ninterference is gone\n($U_{ij}\\,r_i\\leq0$)")
    col_lab = ("needed in the full model ($\\Delta L>0$)", "not needed in the full model ($\\Delta L\\leq0$)")
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6), sharex=True, sharey=True)
    for name, s, bias, kbest, col in fits:
        rk = ranks(s)
        hs = helps_sparse(bias)
        # the LEARNED circuit: circuit entries the oracle calls real; the unlearned ones sit at
        # the floor and would stretch the band to the end of the ranking
        circ_hi = int(np.percentile(rk[(on & (dl.reshape(-1) > 1e-4)).numpy()], 95))
        for ri, hsel in enumerate((hs, ~hs)):
            for ci, nsel in enumerate((needed, ~needed)):
                ax = axes[ri, ci]
                sel = (hsel & nsel & ~on & ~dead).numpy()
                ax.hist(rk[sel], bins=edges, histtype="step", lw=1.0, color=col,
                        label=f"{name}", density=True)
                ax.axvline(kbest, lw=0.8, color=col, ls=(0, (3, 1.5)))
                if name == fits[0][0]:
                    ax.axvspan(1, circ_hi, color="#b2182b", alpha=0.08, lw=0)
                ax.text(0.03, 0.95 - 0.1 * [f[0] for f in fits].index(name),
                        f"n = {int(sel.sum())}", transform=ax.transAxes, va="top", size=5.5,
                        color=col)
    for ri in range(2):
        axes[ri, 0].set_ylabel(row_lab[ri], size=6.5)
    for ci in range(2):
        axes[0, ci].set_title(col_lab[ci], size=7, pad=4)
        axes[1, ci].set_xlabel("Rank in the attribution (1 = highest score)", size=7)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlim(1, n2)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        furnish(ax)
    handles = [plt.Line2D([], [], color=PLAIN, lw=1.2, label="MAttr+Adam"),
               plt.Line2D([], [], color=BIAS, lw=1.2, label="MAttr+Adam + learned per-row bias"),
               plt.Line2D([], [], color="black", lw=0.8, ls=(0, (3, 1.5)), label="loss-optimal $k$ (each fit)"),
               plt.Rectangle((0, 0), 1, 1, color="#b2182b", alpha=0.15, label="where the learned circuit sits (95%)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.8, columnspacing=1.0)
    fig.text(0.5, -0.02, "Density of interference weights over rank, by kind. Dead rows (one tied "
             "score) excluded; $r_i$ under each fit's own bias, so the row counts differ slightly.",
             ha="center", size=6, color="#444444")
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    out = ROOT / "plots" / f"interference_rank_2x2_{a.tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
