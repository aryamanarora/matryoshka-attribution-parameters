"""Exploratory: MAttr-ONLY views of the two headline refusal rankings (1B, 8B), composition and
the ranking itself.

Enrichment as in eda_refusal_composition_variants.py: (share of the top-k that is the group) /
(share of all parameters that is the group), log10, at the paper's budgets (2% / 1%).

  D_layer_type_heatmap   layer x unit type, fill = enrichment at the headline budget (one panel
                         per model): where AND what, instead of two marginals
  E_overlay              1B and 8B overlaid: depth profile on relative depth (left) and unit-type
                         enrichment as paired dots (right)
  F_band_composition     what the ranking adds at each sparsity band, stacked by unit type and by
                         depth quarter (the existing composition view, MAttr only)
  G_rank_cdf             the whole ranking: for each unit type, the fraction of its units inside
                         the top-x of the ranking as x sweeps 1e-5..1 (log); the diagonal is uniform
  H_rank_percentile      layer x unit type, fill = median rank PERCENTILE of the group's units in
                         the full ranking (0 = top): the ranking rather than one cut of it

    uv run python plots/eda_refusal_mattr_variants.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from palette import MODEL
from plot_mask_composition import FRACS_FINE, load, unit_meta
from plot_refusal_mask_composition import CELLS, TYPE_LABEL, TYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "refusal_mattr_variants"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Inter", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42})
GREEN = MODEL["MAttr"]
MCOL = {"Llama 3.2 1B, top 2%": "#0072b2", "Llama 3.1 8B, top 1%": "#d55e00"}   # Wong blue / vermillion
TYPE_ORDER = [lab for _, lab in TYPES]
FLOOR = -2.0
TICKS = [-2, -1, 0, 1, 2]; TLAB = ["0.01×", "0.1×", "1×", "10×", "100×"]


class Mask:
    def __init__(self, model, frac, run):
        self.model, self.frac = model, frac
        self.scores, lay, _, _ = load(ROOT / "runs" / run)
        comp, layer = unit_meta(lay)
        self.comp, self.layer = np.asarray(comp), np.asarray(layer)
        self.type = np.array([TYPE_LABEL.get(c, c) for c in self.comp])
        self.N = self.scores.numel(); self.k = int(round(frac * self.N))
        self.n_layers = int(self.layer.max()) + 1
        order = np.asarray(torch.argsort(self.scores, descending=True))
        self.rank = np.empty(self.N, dtype=np.int64); self.rank[order] = np.arange(self.N)
        self.top = self.rank < self.k
        self.types = [t for t in TYPE_ORDER if t in set(self.type)]

    def enr(self, mask):
        if mask.sum() == 0:
            return np.nan
        v = self.top[mask].mean() / self.frac
        return np.log10(v) if v > 0 else FLOOR


M = [Mask(*c[:2], c[2]["MAttr"]) for c in CELLS]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def strip_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------- D: layer x type enrichment heatmap ----
def variant_layer_type():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), gridspec_kw=dict(width_ratios=[1, 1]))
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for ax, m in zip(axes, M):
        types = [t for t in m.types if t not in ("embed", "final norm", "lm_head")]
        G = np.array([[m.enr((m.layer == ln) & (m.type == t)) for t in types]
                      for ln in range(m.n_layers)])
        im = ax.imshow(G, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=90)
        ax.set_yticks(range(0, m.n_layers, 2 if m.n_layers > 16 else 1))
        ax.set_ylabel("Layer"); ax.set_title(m.model, pad=3); ax.tick_params(length=0)
        extra = {t: m.enr(m.type == t) for t in ("embed", "final norm", "lm_head") if t in m.types}
        ax.text(0, 1.01, "  ".join(f"{t}: {'0' if v <= FLOOR else f'{10 ** v:.1f}'}×"
                                    for t, v in extra.items()),
                transform=ax.transAxes, fontsize=5.5, va="bottom", color="#555")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=1.2)
    cax = fig.add_axes([1.0, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cax, ticks=TICKS); cb.set_ticklabels(TLAB)
    cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
    cb.set_label("Enrichment in the top-k", fontsize=6)
    save(fig, "D_layer_type_heatmap")


# ---------------------------------------------------- E: 1B and 8B overlaid --------------
def variant_overlay():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for m in M:
        x = np.arange(m.n_layers) / (m.n_layers - 1)
        v = np.array([m.enr(m.layer == ln) for ln in range(m.n_layers)])
        a1.plot(x, v, "-o", color=MCOL[m.model], ms=2.2, lw=1.0, label=m.model)
    a1.axhline(0, color="#444", lw=0.5)
    a1.set_xticks([0, 0.5, 1]); a1.set_xticklabels(["first", "middle", "last"])
    a1.set_xlabel("Relative depth"); a1.set_yticks([-1, 0, 1]); a1.set_yticklabels(["0.1×", "1×", "10×"])
    a1.set_ylim(-1.1, 1.1); a1.set_ylabel("Enrichment in the top-k")
    a1.set_title("By layer", pad=2); a1.legend(frameon=False, loc="upper right")
    a1.grid(axis="y", color="#e5e5e5", lw=0.4); a1.set_axisbelow(True); strip_spines(a1)
    types = [t for t in TYPE_ORDER if any(t in m.types for m in M)]
    y = np.arange(len(types))[::-1]
    for j, m in enumerate(M):
        v = np.array([m.enr(m.type == t) if t in m.types else np.nan for t in types])
        a2.scatter(v, y + (0.15 if j == 0 else -0.15), color=MCOL[m.model], s=14, zorder=3)
    a2.axvline(0, color="#444", lw=0.5)
    a2.set_yticks(y); a2.set_yticklabels(types)
    a2.set_xticks(TICKS); a2.set_xticklabels(TLAB, fontsize=5.5); a2.set_xlim(-2.2, 1.2)
    a2.set_title("By unit type", pad=2)
    a2.grid(axis="x", color="#e5e5e5", lw=0.4); a2.set_axisbelow(True); strip_spines(a2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "E_overlay")


# ---------------------------------------------------- F: band composition ---------------
def variant_bands():
    fracs = FRACS_FINE
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.2))
    cmap_t = plt.get_cmap("tab20")
    for r, m in enumerate(M):
        edges = [0] + [int(round(f * m.N)) for f in fracs]
        order = np.argsort(m.rank)
        for c, (kind, keys, keyof) in enumerate((
                ("type", m.types, m.type),
                ("depth", ["first ¼", "second ¼", "third ¼", "last ¼", "no layer"],
                 np.where(m.layer < 0, 4, np.minimum(4 * np.maximum(m.layer, 0) // m.n_layers, 3))))):
            ax = axes[r, c]
            key_id = {k: i for i, k in enumerate(keys)}
            ids = (np.array([key_id[t] for t in keyof]) if kind == "type" else keyof)
            S = np.zeros((len(fracs) + 1, len(keys)))
            for bi, (a, b) in enumerate(zip(edges, edges[1:])):
                S[bi] = np.bincount(ids[order[a:b]], minlength=len(keys)) / max(1, b - a)
            S[-1] = np.bincount(ids, minlength=len(keys)) / m.N              # population
            xs = np.arange(len(fracs) + 1)
            cols = ([cmap_t(i) for i in range(len(keys))] if kind == "type"
                    else ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#999999"])
            ax.stackplot(xs, S.T, colors=cols, labels=keys, lw=0)
            ax.set_xticks(xs); ax.set_xticklabels([f"{f:g}" for f in fracs] + ["all"], rotation=90,
                                                   fontsize=4.8)
            ax.set_xlim(0, len(fracs)); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
            ax.axvline(fracs.index(m.frac), color="#000", lw=0.6, ls="--")
            ax.set_title(f"{m.model}: by {kind}", pad=2)
            if r == 1:
                ax.set_xlabel("Top fraction of the ranking (band)")
            if c == 0:
                ax.set_ylabel("Share of the band")
            ax.legend(fontsize=4.2, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                      ncol=1, handlelength=0.8, handletextpad=0.4, labelspacing=0.2)
    fig.tight_layout(w_pad=2.4, h_pad=1.2)
    save(fig, "F_band_composition")


# ---------------------------------------------------- G: rank CDF per unit type ---------
def variant_rank_cdf():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    xs = np.logspace(-5, 0, 60)
    cmap_t = plt.get_cmap("tab20")
    for ax, m in zip(axes, M):
        for i, t in enumerate(m.types):
            r = m.rank[m.type == t] / m.N
            ax.plot(xs, [(r <= x).mean() for x in xs], color=cmap_t(TYPE_ORDER.index(t)), lw=1.0,
                    label=t)
        ax.plot(xs, xs, color="#000", lw=0.6, ls=":", label="uniform")
        ax.axvline(m.frac, color="#444", lw=0.5, ls="--")
        ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Top fraction of the ranking"); ax.set_title(m.model, pad=2); strip_spines(ax)
    axes[0].set_ylabel("Fraction of the type's units")
    axes[1].legend(fontsize=5, frameon=False, loc="upper left", ncol=1, handlelength=1.2,
                   labelspacing=0.25)
    fig.tight_layout(w_pad=0.8)
    save(fig, "G_rank_cdf")


# ---------------------------------------------------- H: median rank percentile heatmap --
def variant_rank_percentile():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    for ax, m in zip(axes, M):
        types = [t for t in m.types if t not in ("embed", "final norm", "lm_head")]
        pct = m.rank / m.N
        G = np.array([[np.median(pct[(m.layer == ln) & (m.type == t)]) for t in types]
                      for ln in range(m.n_layers)])
        im = ax.imshow(G, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=90)
        ax.set_yticks(range(0, m.n_layers, 2 if m.n_layers > 16 else 1))
        ax.set_ylabel("Layer"); ax.set_title(m.model.split(",")[0], pad=3); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=1.2)
    cax = fig.add_axes([1.0, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["top", "25%", "50%", "75%", "bottom"])
    cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
    cb.set_label("Median rank percentile", fontsize=6)
    save(fig, "H_rank_percentile")


if __name__ == "__main__":
    variant_layer_type(); variant_overlay(); variant_bands(); variant_rank_cdf(); variant_rank_percentile()
