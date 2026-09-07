"""One method's rank of every virtual weight against another's, coloured by kind of weight.

    uv run python plots/plot_interference_rank_vs_rank.py --tags hard30k lit --x adam --y ixg:mc

x = the rank a weight gets under method X (1 = highest score), y = its rank under method Y, one
point per weight, log-log, with a running median per category drawn over the points. Where the
two agree the category's points sit on the diagonal; a category the methods order oppositely
runs across it. Categories are plot_interference_rank_composition.py's: circuit, the four
kinds of interference by (helps once the interference is gone) x (needed in the full model),
and dead rows. Raw matplotlib.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
import interference_toy as IT  # noqa: E402
from interference_scale import spearman  # noqa: E402

from palette import RC, furnish  # noqa: E402
from plot_interference_rank_composition import CATS, categories  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"adam": "MAttr+Adam", "biasmask": "MAttr+Adam + per-row bias", "sgd": "MAttr+SGD",
         "ixg:mc": "Stepless IG", "dL": "Oracle $\\Delta L$"}
TITLES = {"hard": "hard, 3k steps", "hard30k": "hard, 30k steps", "lit": "lit"}


def load_scores(d, key, U, b):
    if key == "biasmask":
        bm = torch.load(d / "biasmask.pt")
        return bm["scores"].reshape(-1), b + bm["db"]
    if key == "dL":
        return torch.load(d / "model.pt")["dl"].reshape(-1), b
    return torch.load(d / "scores.pt")[key].reshape(-1), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["hard30k", "lit"])
    ap.add_argument("--x", default="adam")
    ap.add_argument("--y", default="ixg:mc")
    ap.add_argument("--linear", action="store_true", help="linear rank axes and bins")
    a = ap.parse_args()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(a.tags), figsize=(2.7 * len(a.tags), 2.8), squeeze=False)
    for ax, tag in zip(axes[0], a.tags):
        d = ROOT / "plots" / "data" / f"interference_toy_{tag}"
        m = torch.load(d / "model.pt")
        U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
        IT.N_FEAT, IT.N_RES = U.shape[0], 16
        X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
        Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
        sx, bias_x = load_scores(d, a.x, U, b)
        sy, _ = load_scores(d, a.y, U, b)
        cats = categories(U, bias_x, A, dl, X, Y)      # categories under X's bias
        n2 = U.numel()

        def ranks(s):
            r = torch.empty(n2, dtype=torch.long)
            r[s.argsort(descending=True)] = torch.arange(1, n2 + 1)
            return r.numpy()

        rx, ry = ranks(sx), ranks(sy)
        edges = np.linspace(1, n2 + 1, 41) if a.linear else np.logspace(0, np.log10(n2), 30)
        ax.plot([1, n2], [1, n2], lw=0.5, ls=(0, (3, 2)), color="#888888", zorder=0)
        for (lab, col), msk in zip(CATS, cats):
            msk = msk.numpy()
            if lab.startswith("dead"):
                continue                                # one tied block; a point, not a curve
            ax.scatter(rx[msk], ry[msk], s=1.0 if msk.sum() > 500 else 3, lw=0, color=col,
                       alpha=0.25 if msk.sum() > 500 else 0.8, rasterized=True)
            med = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = msk & (rx >= lo) & (rx < hi)
                med.append(np.median(ry[sel]) if sel.sum() >= 8 else np.nan)
            centres = (edges[:-1] + edges[1:]) / 2 if a.linear else np.sqrt(edges[:-1] * edges[1:])
            ax.plot(centres, med, color=col, lw=1.4, label=lab)
        on = cats[0].numpy()
        off = ~on & ~cats[-1].numpy()
        ax.text(0.04, 0.96, f"{TITLES.get(tag, tag)}\n$\\rho$ circuit {spearman(torch.tensor(rx[on]), torch.tensor(ry[on])):+.2f}"
                f"\n$\\rho$ interference {spearman(torch.tensor(rx[off]), torch.tensor(ry[off])):+.2f}",
                transform=ax.transAxes, va="top", size=6)
        if not a.linear:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(1, n2)
        ax.set_ylim(1, n2)
        ax.set_xlabel(f"Rank under {NAMES.get(a.x, a.x)}", size=7)
        ax.set_ylabel(f"Rank under {NAMES.get(a.y, a.y)}", size=7)
        ax.tick_params(labelsize=6)
        furnish(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, 1.12), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    out = ROOT / "plots" / (f"interference_rank_vs_rank_{a.y.replace(':', '-')}_of_"
                            f"{a.x.replace(':', '-')}{'_linear' if a.linear else ''}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
