"""Two attribution methods' scores against each other, per virtual weight, on one toy run.

    uv run python plots/plot_interference_method_scatter.py --tags hard hard30k --x ixg:mc --y adam

One panel per tag: x is one method's score, y the other's, one point per virtual weight,
coloured by whether the weight is on the target circuit `A` (the labelling the scatter figures
use, see plot_interference_scatter.py), with the off-circuit weights split by whether their
TARGET ROW is in the circuit at all -- a dead row (A_i = 0) is where every method ranks by -U
and the oracle sees nothing, so those points are the vertical band at x ~ 0. Both axes are symlog with the linear region sized to
that method's own off-circuit median, because the two scales differ by orders of magnitude
(stepless IG is in loss units, MAttr's scores are wherever the optimizer left them) and the
picture is about RANK agreement. Spearman on and off the circuit is printed in the corner.
Reads plots/data/interference_toy_<tag>/{model,scores}.pt.
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
from plot_interference_scale import METHODS, symlog_ticks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ON, OFF, DEAD = "#b2182b", "#4d4d4d", "#009e73"   # circuit / live-row interference / dead-row
TITLES = {"hard": "3k steps", "hard30k": "30k steps", "hard100k": "100k steps", "lit": "lit"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["hard", "hard30k"])
    ap.add_argument("--x", default="ixg:mc")
    ap.add_argument("--y", default="adam")
    ap.add_argument("--color-kept", action="store_true",
                    help="colour the live-row off-circuit weights by whether they are in MAttr+Adam's "
                         "loss-optimal top-k (from the true-loss sweep) instead of one grey")
    ap.add_argument("--no-thresholds", action="store_true",
                    help="skip the lines at each method's loss-optimal top-k threshold")
    a = ap.parse_args()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(a.tags), figsize=(2.6 * len(a.tags), 2.5), squeeze=False)
    for ax, tag in zip(axes[0], a.tags):
        d = ROOT / "plots" / "data" / f"interference_toy_{tag}"
        A = torch.load(d / "model.pt")["A"]
        on = (A > 0).reshape(-1)
        # interference in a DEAD row: a target the circuit never uses (A_i = 0, y_i = 0 always),
        # where every method's rule is "keep the gate shut" and the single-weight effect is nil
        dead = ((A > 0).any(1) == False).unsqueeze(1).expand_as(A).reshape(-1) & ~on
        live_off = ~on & ~dead
        extra = {"dL": ("Oracle $\\Delta L$", "#984ea3"),
                 "U*r": ("$U_{ij}\\,r_i$", "#984ea3")}
        label = lambda k: METHODS[k][0] if k in METHODS else extra[k][0]
        colour = lambda k: METHODS[k][1] if k in METHODS else extra[k][1]
        m = torch.load(d / "model.pt")
        sc = dict(torch.load(d / "scores.pt"))
        sc["dL"] = m["dl"]                      # the oracle is a ranking too
        if "U*r" in (a.x, a.y):
            # U_ij * r_i: the weight's sign times its target row's mean under-prediction in the
            # CIRCUIT-ONLY model, r_i = E[(y_i - y'_i) 1(gate open)] -- up to 2 E[x_j], IxG
            # evaluated at the circuit-only model; the per-weight rule for what Adam keeps
            # (scripts/interference/interference_adam_pattern.py).
            U, b, A, v = m["U"], m["b"], m["A"], m["v"]
            IT.N_FEAT, IT.N_RES = U.shape[0], 16
            X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
            Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
            with torch.no_grad():
                zc = X @ (U * (A > 0)).T + b
                r = ((Y - F.relu(zc)) * (zc > 0)).mean(0)
            sc["U*r"] = U * r.unsqueeze(1)
        sx, sy = sc[a.x].reshape(-1), sc[a.y].reshape(-1)
        lx = float(sx[~on].abs().median()) or 1e-6
        ly = float(sy[~on].abs().median()) or 1e-6
        if a.color_kept:
            best_k = None
            for grid in ("log", "linear"):
                f_ = d / f"true_loss_{grid}.json"
                if f_.exists():
                    tl_ = json.loads(f_.read_text())
                    c_ = tl_["curves"]["adam"]
                    i_ = min(range(len(c_)), key=c_.__getitem__)
                    if best_k is None or c_[i_] < best_k[1]:
                        best_k = (tl_["grid"][i_], c_[i_])
            kept = torch.zeros(sx.numel(), dtype=torch.bool)
            kept[sc["adam"].reshape(-1).argsort(descending=True)[:best_k[0]]] = True
            ax.scatter(sx[live_off & ~kept], sy[live_off & ~kept], s=1.2, lw=0, color=OFF,
                       alpha=0.3, rasterized=True, label="off circuit, not kept by Adam")
            ax.scatter(sx[live_off & kept], sy[live_off & kept], s=2.5, lw=0, color=COLOR["adam"],
                       alpha=0.7, rasterized=True, label=f"off circuit, kept by Adam (top-{best_k[0]})")
        else:
            ax.scatter(sx[live_off], sy[live_off], s=1.2, lw=0, color=OFF, alpha=0.35,
                       rasterized=True, label="off circuit, live row")
        collapsed = dead.any() and float(sx[dead].max() - sx[dead].min()) < 0.01 * lx \
            and float(sy[dead].max() - sy[dead].min()) < 0.01 * ly   # one point at this axis scale
        if collapsed:
            # After long training every dead-row weight has EXACTLY the same score under every
            # method (the gate never opens, so they never get a gradient): one point, not a
            # cloud. Draw it big, on top, and say how many weights it is.
            ax.scatter(sx[dead][:1], sy[dead][:1], s=28, lw=0.6, color=DEAD, edgecolor="white",
                       zorder=6, label="off circuit, dead row ($A_{i\\cdot}=0$)")
            ax.annotate(f"{int(dead.sum())} dead-row weights,\nall at one point",
                        (float(sx[dead][0]), float(sy[dead][0])), xytext=(-40, 24),
                        textcoords="offset points", fontsize=5, color=DEAD,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=DEAD))
        else:
            ax.scatter(sx[dead], sy[dead], s=1.2, lw=0, color=DEAD, alpha=0.5, rasterized=True,
                       label="off circuit, dead row ($A_{i\\cdot}=0$)")
        ax.scatter(sx[on], sy[on], s=4, lw=0, color=ON, alpha=0.9, rasterized=True,
                   label="on circuit $A$")
        ax.set_xscale("symlog", linthresh=lx)
        ax.set_yscale("symlog", linthresh=ly)
        symlog_ticks(ax, "x", lx, float(sx.abs().max()))
        symlog_ticks(ax, "y", ly, float(sy.abs().max()))
        ax.axhline(0, lw=0.4, color="#aaaaaa", zorder=0)
        ax.axvline(0, lw=0.4, color="#aaaaaa", zorder=0)
        note = ""
        if not a.no_thresholds:
            # Each method's loss-optimal top-k on real masked forwards (both true-loss grids,
            # whichever holds the minimum), drawn as the score threshold that k corresponds to:
            # everything above the horizontal line is what MAttr keeps at its optimum, everything
            # right of the vertical line is what stepless IG keeps at its.
            best = {}
            for grid in ("log", "linear"):
                f = d / f"true_loss_{grid}.json"
                if not f.exists():
                    continue
                tl = json.loads(f.read_text())
                for meth, curve in tl["curves"].items():
                    meth = "dL" if meth == "ideal" else meth      # the true-loss sweep's name for it
                    i = min(range(len(curve)), key=curve.__getitem__)
                    if meth not in best or curve[i] < best[meth][1]:
                        best[meth] = (tl["grid"][i], curve[i])
            for meth, s_, line, col in ((a.x, sx, ax.axvline, colour(a.x)),
                                         (a.y, sy, ax.axhline, colour(a.y))):
                if meth in best and best[meth][0] > 0:
                    thr = float(s_.sort(descending=True).values[best[meth][0] - 1])
                    line(thr, lw=0.8, ls=(0, (4, 2)), color=col, zorder=3)
                    short = {"ixg:mc": "Stepless IG", "adam": "MAttr+Adam", "sgd": "MAttr+SGD",
                             "adam@0.002": "MAttr+Adam (lr .002)", "dL": "oracle ΔL"}.get(meth, meth)
                    note += f"\n{short}: k*={best[meth][0]}"
        ax.text(0.03, 0.97, f"{TITLES.get(tag, tag)}\n$\\rho$ circuit {spearman(sx[on], sy[on]):+.2f}"
                f"\n$\\rho$ off-circuit {spearman(sx[~on], sy[~on]):+.2f}" + note,
                transform=ax.transAxes, size=6, va="top")
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel(label(a.x) + " score", size=7)
        ax.set_ylabel(label(a.y) + " score", size=7)
        furnish(ax)
    if "U*r" in (a.x, a.y):
        fig.text(0.5, -0.04, "$U_{ij}\\,r_i$: the weight's sign times its target row's mean "
                 "under-prediction in the circuit-only model (= I×G at the circuit-only model)",
                 ha="center", size=6, color="#444444")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4 if a.color_kept else 3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.06), markerscale=3)
    fig.tight_layout(w_pad=1.0)
    out = ROOT / "plots" / (f"interference_method_scatter_{a.y.replace(':', '-').replace('*', '')}"
                            f"_vs_{a.x.replace(':', '-').replace('*', '')}{'_kept' if a.color_kept else ''}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
