"""The interference-weight task across two decades of size: where the methods agree.

    uv run python plots/plot_interference_scale.py --tag hard            # all four figures
    uv run python plots/plot_interference_scale.py --tag hard --fig agreement

Reads plots/data/interference_scale/<tag>/ (written by scripts/interference/interference_scale.py) and draws:

  agreement   Spearman rank agreement against x = number of virtual weights, one row per
              population (real weights / interference weights) and one column per comparison
              (vs the oracle dL, vs stepless IG, vs the same method refit at another seed).
              The bottom row is the figure: it is where "the methods agree on the circuit and
              do their own thing on the noise" would show up, as a row of curves that sits
              well below the top row and, for the refit column, below ITSELF -- a method whose
              interference ranking is not even reproducible.
  pr          precision-recall per scale, the note's first figure repeated six times.
  scatter     MAttr+Adam against stepless IG per virtual weight, one panel per scale, coloured
              by whether the weight is on the target circuit A. The picture behind the
              agreement numbers: agreement on the circuit is the diagonal in the coloured
              points, disagreement on the noise is whatever the grey cloud does.
  trueloss    each ranking's best top-k on real masked forwards, relative to the full model and
              to the circuit alone, and how many off-circuit weights that optimum keeps.
  budget      MAttr+Adam's interference ranking against fitting steps, per scale: refit
              consistency, agreement with stepless IG, agreement with the oracle.

Raw matplotlib rather than plotnine: symlog scatter panels, per-panel reference lines and a
shared legend strip are outside the grammar. Colours come from plots/palette.py; the lower
Adam learning rate is a HYPERPARAMETER of the same method and so takes Adam's hue with a dash,
per that module's rule, and the note's heuristics take the Set1 colours
plot_interference_filtering.py already assigned them.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "interference"))
from interference_toy import sweep  # noqa: E402

from palette import COLOR, RC, furnish  # noqa: E402
from plot_interference_filtering import NOTE_STYLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "data" / "interference_scale"
ON, OFF = "#b2182b", "#4d4d4d"

METHODS = {                       # key -> (label, colour, linestyle)
    "ixg:mc": ("Stepless IG", COLOR["ixg:mc"], "solid"),
    "adam": ("MAttr (Adam, lr 0.05)", COLOR["adam"], "solid"),
    "adam@0.002": ("MAttr (Adam, lr 0.002)", COLOR["adam"], (0, (3.5, 1.5))),
    "sgd": ("MAttr (SGD)", COLOR["sgd"], "solid"),
}
HEURISTICS = {k: (NOTE_STYLE[k][0], NOTE_STYLE[k][1], (0, (1.5, 1.5))) for k in ("era", "weight")}


def load(tag):
    d = DATA / tag
    cells = [json.loads(f.read_text()) for f in sorted(d.glob("n*_s*.json"))]
    by_n = {}
    for c in cells:
        by_n.setdefault(c["meta"]["n_feat"], []).append(c)
    return d, dict(sorted(by_n.items()))


def mean_sd(vals):
    v = np.array([x for x in vals if x == x], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0)


def fmt_pow(n):
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    e = int(round(np.log2(n)))
    return f"2{str(e).translate(sup)}"


def symlog_ticks(ax, which, linthresh, vmax):
    """Three or four labelled decades a side instead of every one: symlog's default locator
    labels each decade, which at 5pt over two axes is a solid bar of text."""
    import math
    lo, hi = math.ceil(math.log10(linthresh)), math.floor(math.log10(max(vmax, linthresh * 10)))
    decades = sorted({lo, hi}) if hi - lo < 2 else [lo, hi]   # two a side, no more
    ticks = [-10.0 ** d for d in reversed(decades)] + [0.0] + [10.0 ** d for d in decades]
    (ax.set_xticks if which == "x" else ax.set_yticks)(ticks)
    (ax.set_xticks if which == "x" else ax.set_yticks)([], minor=True)


def finish(ax, xl=None, yl=None):
    ax.tick_params(labelsize=6)
    if xl:
        ax.set_xlabel(xl, size=7)
    if yl:
        ax.set_ylabel(yl, size=7)
    furnish(ax)


def fig_agreement(tag, by_n, label="real"):
    """label='real' splits by dL > eps (the oracle's definition); 'circuit' by A > 0."""
    pop = {"real": ("real", "noise"), "circuit": ("circuit", "offcircuit")}[label]
    pop_names = {"real": ("Real weights ($\\Delta L>\\epsilon$)", "Interference weights"),
                 "circuit": ("Circuit weights ($A_{ij}>0$)", "Off-circuit weights")}[label]
    cols = (("oracle", "vs oracle $\\Delta L$"), ("pairs", "vs stepless IG"),
            ("self", "vs itself, refit"))
    budget = max(next(iter(by_n.values()))[0]["meta"]["snaps"])
    xs = np.array(list(by_n))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.1), sharex=True, sharey=True)
    for r, p in enumerate(pop):
        for c, (kind, title) in enumerate(cols):
            ax = axes[r, c]
            for meth, (lab, color, ls) in METHODS.items():
                ys, es = [], []
                for n, cells in by_n.items():
                    vals = []
                    for cell in cells:
                        if kind == "oracle":
                            vals.append(cell["oracle"][f"{meth}|0|{budget}"][f"rho_{p}"])
                        elif kind == "pairs":
                            if meth == "ixg:mc":
                                vals.append(np.nan)
                                continue
                            key = f"{meth}|ixg:mc" if meth < "ixg:mc" else f"ixg:mc|{meth}"
                            vals.append(cell["pairs"][key][f"rho_{p}"])
                        else:
                            vals.append(cell["self"].get(meth, {}).get(f"rho_{p}", np.nan))
                    m, s = mean_sd(vals)
                    ys.append(m)
                    es.append(s)
                ys, es = np.array(ys), np.array(es)
                if np.all(np.isnan(ys)):
                    continue
                ax.fill_between(xs ** 2, ys - es, ys + es, color=color, alpha=0.15, lw=0)
                ax.plot(xs ** 2, ys, color=color, ls=ls, lw=1.0, marker="o", ms=2.2,
                        label=lab)
            if kind == "oracle":
                for meth, (lab, color, ls) in HEURISTICS.items():
                    ys = [mean_sd([cell["oracle"][f"{meth}|0|0"][f"rho_{p}"]
                                   for cell in cells])[0] for cells in by_n.values()]
                    ax.plot(xs ** 2, ys, color=color, ls=ls, lw=0.9, marker="s", ms=1.8,
                            label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs ** 2)
            ax.set_xticklabels([fmt_pow(n * n) for n in xs])
            ax.axhline(0, lw=0.5, color="#888888", zorder=0)
            if r == 0:
                ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom",
                        size=7)
            finish(ax, "Virtual weights" if r == 1 else None,
                   f"{pop_names[r]}\nSpearman $\\rho$" if c == 0 else None)
    axes[0, 0].set_ylim(-0.25, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.8, columnspacing=1.0,
               handletextpad=0.5)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    out = ROOT / "plots" / f"interference_scale_agreement_{label}_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_pr(tag, d, by_n):
    budget = max(next(iter(by_n.values()))[0]["meta"]["snaps"])
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.3), sharex=True, sharey=True)
    for ax, (n, cells) in zip(axes.flat, by_n.items()):
        for cell in cells:
            seed = cell["meta"]["seed"]
            blob = torch.load(d / f"n{n}_s{seed}.pt")
            dl = blob["dl"]
            for meth, (lab, color, ls) in {**METHODS, **HEURISTICS}.items():
                key = f"{meth}|0|{budget}" if meth in METHODS else f"{meth}|0|0"
                c = sweep(blob["scores"][key], dl)
                ax.plot(c["recall"], c["precision"], color=color, ls=ls,
                        lw=1.0 if seed == 0 else 0.5, alpha=1.0 if seed == 0 else 0.35,
                        label=lab if seed == 0 else None)
        base = np.mean([c["meta"]["n_real"] / c["meta"]["n_weights"] for c in cells])
        ax.axhline(base, lw=0.5, ls=(0, (4, 2)), color="#888888", zorder=0)
        ax.text(0.03, 0.04, f"{fmt_pow(n * n)} weights\nbase rate {base:.1%}",
                transform=ax.transAxes, size=6, va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.05)
        finish(ax, "Recall" if ax in axes[1] else None,
               "Precision" if ax in axes[:, 0] else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.05), handlelength=1.8, columnspacing=1.0,
               handletextpad=0.5)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    out = ROOT / "plots" / f"interference_scale_pr_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_scatter(tag, d, by_n, x_method="ixg:mc", y_method="adam"):
    budget = max(next(iter(by_n.values()))[0]["meta"]["snaps"])
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.5))
    for ax, (n, cells) in zip(axes.flat, by_n.items()):
        blob = torch.load(d / f"n{n}_s0.pt")
        sx = blob["scores"][f"{x_method}|0|{budget}"].reshape(-1)
        sy = blob["scores"][f"{y_method}|0|{budget}"].reshape(-1)
        on = (blob["A"] > 0).reshape(-1)
        # Each method on its own scale: the scatter is about RANK agreement, and the two
        # score scales differ by orders of magnitude (IG is in loss units, Adam's scores are
        # whatever the random walk left them at), so both axes are symlog with a linear region
        # sized to that method's own interference spread.
        lx = float(sx[~on].abs().median()) or 1e-6
        ly = float(sy[~on].abs().median()) or 1e-6
        ax.scatter(sx[~on], sy[~on], s=1.2 if n > 48 else 3, lw=0, color=OFF, alpha=0.35,
                   rasterized=True, label="off circuit")
        ax.scatter(sx[on], sy[on], s=3 if n > 48 else 5, lw=0, color=ON, alpha=0.9,
                   rasterized=True, label="on circuit $A$")
        ax.set_xscale("symlog", linthresh=lx)
        ax.set_yscale("symlog", linthresh=ly)
        symlog_ticks(ax, "x", lx, float(sx.abs().max()))
        symlog_ticks(ax, "y", ly, float(sy.abs().max()))
        ax.axhline(0, lw=0.4, color="#aaaaaa", zorder=0)
        ax.axvline(0, lw=0.4, color="#aaaaaa", zorder=0)
        pr = cells[0]["pairs"].get(f"{y_method}|{x_method}") or cells[0]["pairs"].get(
            f"{x_method}|{y_method}")
        ax.text(0.03, 0.97, f"{fmt_pow(n * n)} weights\n$\\rho$ circuit {pr['rho_circuit']:+.2f}"
                f"\n$\\rho$ off-circuit {pr['rho_offcircuit']:+.2f}",
                transform=ax.transAxes, size=5.5, va="top")
        ax.tick_params(labelsize=5)
        finish(ax, METHODS[x_method][0] + " score" if ax in axes[1] else None,
               METHODS[y_method][0] + " score" if ax in axes[:, 0] else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.03), markerscale=3)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    out = ROOT / "plots" / f"interference_scale_scatter_{y_method}_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


def fig_trueloss(tag, by_n):
    xs = np.array(list(by_n))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    series = {**METHODS, "dL": ("Oracle $\\Delta L$", NOTE_STYLE["ideal"][1], "solid")}
    for meth, (lab, color, ls) in series.items():
        rel, nof, kb = [], [], []
        for n, cells in by_n.items():
            tl = [c["true_loss"] for c in cells]
            # Everything relative to the FULL model's loss: 1 = the model the mask was fitted
            # to, and the grey line is the circuit alone, so "beats the model it came from"
            # and "beats the true circuit" are both readable off one axis.
            rel.append(mean_sd([t["curves"][meth]["L_best"] / t["L_full"] for t in tl]))
            nof.append(mean_sd([t["curves"][meth]["n_off_at_best"] / c["meta"]["n_circuit"]
                                for t, c in zip(tl, cells)]))
            kb.append(mean_sd([t["curves"][meth]["k_best"] / c["meta"]["n_circuit"]
                               for t, c in zip(tl, cells)]))
        for ax, vals in zip(axes, (rel, kb, nof)):
            m, s = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
            ax.fill_between(xs ** 2, m - s, m + s, color=color, alpha=0.12, lw=0)
            ax.plot(xs ** 2, m, color=color, ls=ls, lw=1.0, marker="o", ms=2.2, label=lab)
    circ = [mean_sd([c["true_loss"]["L_circuit"] / c["true_loss"]["L_full"] for c in cells])
            for cells in by_n.values()]
    axes[0].plot(xs ** 2, [v[0] for v in circ], color="#888888", lw=0.8, ls=(0, (4, 2)),
                 marker="s", ms=1.8, label="circuit $A$ alone")
    axes[0].axhline(1, lw=0.5, color="#888888", zorder=0)
    axes[0].annotate("full model", (xs[0] ** 2, 1.0), xytext=(2, 2), textcoords="offset points",
                     fontsize=5, color="#666666")
    for ax, yl in zip(axes, ("Best top-$k$ loss / full-model loss",
                             "Best $k$ / circuit size", "Off-circuit weights kept / circuit size")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs ** 2)
        ax.set_xticklabels([fmt_pow(n * n) for n in xs])
        finish(ax, "Virtual weights", yl)
    axes[1].axhline(1, lw=0.5, color="#888888", zorder=0)
    axes[1].set_yscale("log")
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.12), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    out = ROOT / "plots" / f"interference_scale_trueloss_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_budget(tag, by_n, meth="adam"):
    """Adam's interference ranking against BUDGET, one line per scale: does it converge to
    itself (left), and does converging bring it any closer to stepless IG (middle) or to the
    oracle (right)? The answer at every scale is yes / no / no, which is what makes the
    disagreement a property of the objective rather than of an unconverged fit."""
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), sharey=True)
    snaps = next(iter(by_n.values()))[0]["meta"]["snaps"]
    cmap = plt.get_cmap("viridis")
    for i, (n, cells) in enumerate(by_n.items()):
        color = cmap(0.1 + 0.8 * i / max(1, len(by_n) - 1))
        for ax, key in zip(axes, ("self_rho_noise", "ig_rho_noise", "rho_noise")):
            m = [mean_sd([c["budget"][f"{meth}|{b}"][key] for c in cells]) for b in snaps]
            ax.errorbar(snaps, [v[0] for v in m], yerr=[v[1] for v in m], color=color, lw=1.0,
                        marker="o", ms=2.2, capsize=1.5, elinewidth=0.5,
                        label=f"{fmt_pow(n * n)} weights")
    for ax, t in zip(axes, ("vs itself, refit", "vs stepless IG", "vs oracle $\\Delta L$")):
        ax.set_xscale("log")
        ax.set_xticks(snaps)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(b) for b in snaps])
        ax.text(0.5, 1.02, t, transform=ax.transAxes, ha="center", va="bottom", size=7)
        finish(ax, "Fitting steps", "Interference-weight\nSpearman $\\rho$" if ax is axes[0] else None)
    axes[0].set_ylim(-0.4, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.14), handlelength=1.6, columnspacing=0.9)
    fig.tight_layout(w_pad=0.5)
    out = ROOT / "plots" / f"interference_scale_budget_{meth.replace(':', '-').replace('@', '')}_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--fig", default="all",
                    choices=("all", "agreement", "pr", "scatter", "trueloss", "budget"))
    a = ap.parse_args()
    d, by_n = load(a.tag)
    if a.fig in ("all", "agreement"):
        fig_agreement(a.tag, by_n, "real")
        fig_agreement(a.tag, by_n, "circuit")
    if a.fig in ("all", "pr"):
        fig_pr(a.tag, d, by_n)
    if a.fig in ("all", "scatter"):
        fig_scatter(a.tag, d, by_n, "ixg:mc", "adam")
        fig_scatter(a.tag, d, by_n, "ixg:mc", "sgd")
    if a.fig in ("all", "trueloss"):
        fig_trueloss(a.tag, by_n)
    if a.fig in ("all", "budget"):
        fig_budget(a.tag, by_n, "adam")
        fig_budget(a.tag, by_n, "adam@0.002")


if __name__ == "__main__":
    main()
