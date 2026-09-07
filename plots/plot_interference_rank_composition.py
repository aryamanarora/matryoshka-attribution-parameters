"""Composition of a ranking along its length: who sits at each rank, for two MAttr fits.

    uv run python plots/plot_interference_rank_composition.py --tag hard30k

One panel per ranking (plain MAttr+Adam; MAttr+Adam fitted jointly with a per-row bias),
x = rank position (log-spaced bins over 1..n_feat^2), y = the fraction of each bin made up of
six kinds of virtual weight:

    circuit                     A_ij > 0
    interference, sparse+ dense+ U_ij r_i > 0 and dL > 0   helps the sparse model AND the full one
    interference, sparse+ dense-  U_ij r_i > 0 and dL <= 0  helps sparsely, hurts densely: the coalition
    interference, sparse- dense+                             the reverse
    interference, sparse- dense-                             helps nowhere
    dead row                    A_i = 0 (one point under every method after long training)

with each ranking's loss-optimal k as a vertical line. Read left to right: the plain fit puts
the circuit first, then the sparse+/dense- coalition (blue) up to its optimum, then everything
else by dense effect; the bias-mask fit puts the circuit first and then orders by the dense
effect straight away, so the coalition band shrinks to the sliver the schedule-shared bias did
not absorb. `U_ij r_i` is computed under each fit's own bias (b, or b + db). Raw matplotlib
(stacked fill_between over log bins).
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
from interference_toy import EPS_REAL  # noqa: E402

from palette import COLOR, RC, furnish  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CATS = [("circuit ($A_{ij}>0$)", "#b2182b"),
        ("helps once interference is gone, needed in full model", "#e69f00"),
        ("helps once interference is gone, NOT needed in full model  (the coalition)", COLOR["adam"]),
        ("hurts once interference is gone, needed in full model", "#009e73"),
        ("hurts once interference is gone, not needed in full model", "#bbbbbb"),
        ("dead row (never fires; one tied score)", "#555555")]


def categories(U, b, A, dl, X, Y):
    sup = A > 0
    with torch.no_grad():
        zc = X @ (U * sup).T + b
        r = ((Y - F.relu(zc)) * (zc > 0)).mean(0)
    ur = (U * r.unsqueeze(1)).reshape(-1)
    on = sup.reshape(-1)
    # a DEAD row is one whose gate never opens in the full model: the 27 rows A never uses, plus
    # the rows whose circuit entries were never learned. Every weight into such a row has exactly
    # the same score under every method (no gradient ever reached it), so they are one tied block
    # in every ranking, and splitting them by A would draw argsort's arbitrary tie order.
    with torch.no_grad():
        fires = ((X @ U.T + b) > 0).any(0)
    dead = (~fires).unsqueeze(1).expand_as(A).reshape(-1) & ~on
    sp, dn = ur > 0, dl.reshape(-1) > 0
    return [on, ~on & ~dead & sp & dn, ~on & ~dead & sp & ~dn, ~on & ~dead & ~sp & dn,
            ~on & ~dead & ~sp & ~dn, dead]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard30k")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--shade", choices=("none", "absU", "dL"), default="absU",
                    help="split each category into terciles of a per-weight metric and draw "
                         "them light-to-dark (small to large): |U_ij|, or |dL|")
    ap.add_argument("--linear", action="store_true",
                    help="linear rank axis with equal-width bins instead of log-spaced")
    ap.add_argument("--methods", nargs="+", default=["adam", "biasmask"],
                    help="any of adam, biasmask, sgd, ixg:mc, era, twera, weight, dL. Two "
                         "methods get the two-row figure (composition + what-orders-it lines); "
                         "more get a grid of composition panels")
    a = ap.parse_args()
    d = ROOT / "plots" / "data" / f"interference_toy_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
    Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
    sc = dict(torch.load(d / "scores.pt"))
    sc["dL"] = dl
    if any(k in a.methods for k in ("era", "twera", "weight", "freq")):
        stats = IT.statistics(U, b, A, v, 65_536, 8192, 7)
        sc.update(IT.heuristics(U, stats))
    best = {}
    for grid in ("log", "linear"):
        tl = json.loads((d / f"true_loss_{grid}.json").read_text())
        for meth, curve in tl["curves"].items():
            meth = "dL" if meth == "ideal" else meth
            i = min(range(len(curve)), key=curve.__getitem__)
            if meth not in best or curve[i] < best[meth][1]:
                best[meth] = (tl["grid"][i], curve[i])
    NAMES = {"adam": "MAttr+Adam", "biasmask": "MAttr+Adam + learned per-row bias",
             "sgd": "MAttr+SGD", "ixg:mc": "Stepless IG", "era": "ERA", "twera": "TWERA",
             "weight": "Virtual weight $|U|$", "dL": "Oracle $\\Delta L$"}
    runs = []
    for meth in a.methods:
        if meth == "biasmask":
            bm = torch.load(d / "biasmask.pt")
            bmj = json.loads((d / "biasmask.json").read_text())["curves"]
            runs.append((NAMES[meth], bm["scores"].reshape(-1), b + bm["db"], bmj["scores+db"]["k_best"]))
        else:
            runs.append((NAMES[meth], sc[meth].reshape(-1), b, best.get(meth, (None,))[0]))
    n2 = U.numel()
    edges = (np.linspace(0, n2, a.bins + 1).astype(int) + 1 if a.linear
             else np.unique(np.round(np.logspace(0, np.log10(n2), a.bins + 1)).astype(int)))
    plt.rcParams.update(RC)
    two_row = len(runs) == 2
    if two_row:
        fig, axes2 = plt.subplots(2, 2, figsize=(5.5, 3.6), sharex=True,
                                  gridspec_kw={"height_ratios": (1.35, 1)})
        axes, lower = axes2[0], axes2[1]
    else:
        ncol = 3
        nrow = -(-len(runs) // ncol)
        fig, axes2 = plt.subplots(nrow, ncol, figsize=(5.5, 1.75 * nrow + 0.3), sharex=True,
                                  sharey=True, squeeze=False)
        axes = axes2.reshape(-1)[:len(runs)]
        for ax in axes2.reshape(-1)[len(runs):]:
            ax.set_visible(False)
        lower = [None] * len(runs)
    for ax, ax_lo, (title, s, bias, kbest) in zip(axes, lower, runs):
        cats = categories(U, bias, A, dl, X, Y)
        # the ordering, read as a line: for each rank bin, the median PERCENTILE (within all
        # interference weights) of each candidate quantity -- a quantity the ranking follows
        # descends monotonically from 1 to 0; one it ignores sits flat at 0.5
        sup = A > 0
        with torch.no_grad():
            zc = X @ (U * sup).T + bias
            r_ = ((Y - F.relu(zc)) * (zc > 0)).mean(0)
        interf = ~sup.reshape(-1) & ~cats[-1]
        quantities = (("$\\Delta L$: value in the full model", dl.reshape(-1), "#984ea3", "solid"),
                      ("$U_{ij}\\,r_i$: value once interference is gone",
                       (U * r_.unsqueeze(1)).reshape(-1), "#d55e00", "solid"),
                      ("$|U_{ij}|$", U.abs().reshape(-1), "#4daf4a", (0, (2, 1.5))))
        order = s.argsort(descending=True)
        rank_of = torch.empty(n2, dtype=torch.long)
        rank_of[order] = torch.arange(1, n2 + 1)
        # each category is split into terciles of the shading metric (within the category), drawn
        # light -> dark for small -> large, so the ordering INSIDE a band is visible: e.g. the
        # coalition Adam keeps is the small-|U| third of its class, the buried remainder the large
        metric = {"absU": U.abs().reshape(-1), "dL": dl.reshape(-1).abs(),
                  "none": torch.zeros(n2)}[a.shade]
        n_sh = 1 if a.shade == "none" else 3
        sub_masks, sub_style = [], []
        for (lab, col), msk in zip(CATS, cats):
            if n_sh == 1 or lab.startswith("dead"):
                sub_masks.append(msk); sub_style.append((lab, col, 1.0)); continue
            qs = metric[msk].quantile(torch.tensor([1 / 3, 2 / 3]))
            tiers = (metric <= qs[0], (metric > qs[0]) & (metric <= qs[1]), metric > qs[1])
            for t_i, tier in enumerate(tiers):
                sub_masks.append(msk & tier)
                sub_style.append((lab if t_i == 2 else None, col, (0.45, 0.75, 1.0)[t_i]))
        fracs = []
        for msk in sub_masks:
            h, _ = np.histogram(rank_of[msk].numpy(), bins=edges)
            fracs.append(h)
        fracs = np.array(fracs, dtype=float)
        tot = fracs.sum(0).clip(min=1)
        fracs /= tot
        centres = (edges[:-1] + edges[1:]) / 2 if a.linear else np.sqrt(edges[:-1] * edges[1:])
        bottom = np.zeros_like(centres)
        import matplotlib.colors as mcolors
        for (lab, col, depth), f in zip(sub_style, fracs):
            rgb = np.array(mcolors.to_rgb(col))
            shade = 1 - (1 - rgb) * depth            # mix with white: depth 1 = full colour
            ax.fill_between(centres, bottom, bottom + f, color=shade, lw=0, step=None,
                            label=lab, alpha=1.0)
            bottom += f
        if kbest:
            ax.axvline(kbest, lw=0.9, color="black", ls=(0, (3, 1.5)))
            ax.text(kbest, 1.02, f"loss-optimal $k$ = {kbest}", ha="center", va="bottom", size=5.5)
        if not a.linear:
            ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, 1)
        ax.set_title(title, size=7, pad=10)
        ax.tick_params(labelsize=6)
        furnish(ax)
        ax.grid(False)
        if ax_lo is None:
            continue
        rk = rank_of.numpy()
        for lab, q, col, ls in quantities:
            pct = torch.zeros(n2)
            qi = q[interf]
            pct[interf] = qi.argsort().argsort().float() / (qi.numel() - 1)
            med = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = interf.numpy() & (rk >= lo) & (rk < hi)
                med.append(float(pct[torch.from_numpy(sel)].median()) if sel.sum() >= 5 else np.nan)
            ax_lo.plot(centres, med, color=col, ls=ls, lw=1.0, label=lab)
        ax_lo.axvline(kbest, lw=0.9, color="black", ls=(0, (3, 1.5)))
        ax_lo.axhline(0.5, lw=0.4, color="#bbbbbb", zorder=0)
        ax_lo.set_ylim(0, 1)
        ax_lo.set_xlabel("Rank in the attribution (1 = highest score)", size=7)
        ax_lo.tick_params(labelsize=6)
        furnish(ax_lo)
    if two_row:
        axes[0].set_ylabel("Fraction of weights\nat that rank", size=7)
        axes2[1, 0].set_ylabel("Median percentile of the\nquantity, interference only", size=7)
        axes2[1, 1].legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=1.8)
    else:
        for ax in axes2[:, 0]:
            ax.set_ylabel("Fraction of weights\nat that rank", size=7)
        axes2[-1, len(axes2[-1]) // 2].set_xlabel("Rank in the attribution (1 = highest score)",
                                                  size=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, 1.13), handlelength=1.4, columnspacing=1.2)
    fig.text(0.5, -0.03, "helps/hurts once the interference is gone: sign of $U_{ij}\\,r_i$ "
             "(circuit-only model, under that fit's bias);  needed in the full model: $\\Delta L>0$"
             + {"absU": ";  shade: $|U_{ij}|$ tercile within the category, light = small",
                "dL": ";  shade: $|\\Delta L|$ tercile within the category, light = small",
                "none": ""}[a.shade],
             ha="center", size=6, color="#444444")
    fig.tight_layout(w_pad=0.8, h_pad=0.5)
    out = ROOT / "plots" / (f"interference_rank_composition_{a.tag}{'' if two_row else '_all'}"
                            f"{'_linear' if a.linear else ''}{'' if a.shade == 'absU' else '_' + a.shade}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
