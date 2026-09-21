"""Exploratory: alternative encodings for plot_refusal_mask_composition.py, all on ENRICHMENT.

Enrichment of a group g under a ranking at budget k:
    (share of the top-k that is g) / (share of all parameters that is g)
  = (kept_g / n_g) / (k / N)
i.e. how many times over uniform the mask draws from that group (1x = the ranking ignores the
group; every nonresid unit is one d_model vector, so unit counts and parameter counts agree).
Drawn as log10(enrichment), so 0 is uniform, +1 is 10x, -1 is a tenth.

Variants, each to plots/refusal_composition_variants/<name>.{png,pdf}:
  A_diverging_bars   dodged bars from the 1x line, four panels (model x by-layer/by-type)
  B_heatmap          rows = model x method, columns = layers | types, diverging fill + text
  C_dumbbell         per group, a segment MAttr -> EG on the log-enrichment axis (types), and
                     the two depth profiles as filled areas (layers)

    uv run python plots/eda_refusal_composition_variants.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from palette import COLOR, MODEL
from plot_mask_composition import load, unit_meta
from plot_refusal_mask_composition import CELLS, TYPE_LABEL, TYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "refusal_composition_variants"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Inter", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42})
COL = {"MAttr": MODEL["MAttr"], "EG": COLOR["ixg:mc"]}
FLOOR = -2.0     # log10 enrichment floor for empty groups


def enrichment():
    """{(model, method): (layers: {ln: log10 enr}, types: {label: log10 enr})}"""
    out = {}
    for model, frac, runs in CELLS:
        for method, run in runs.items():
            scores, lay, _, _ = load(ROOT / "runs" / run)
            comp, layer = unit_meta(lay)
            comp, layer = np.asarray(comp), np.asarray(layer)
            N = scores.numel(); k = int(round(frac * N))
            top = np.zeros(N, dtype=bool); top[np.asarray(torch.topk(scores, k).indices)] = True
            def enr(mask):
                v = top[mask].mean() / frac
                return np.log10(v) if v > 0 else FLOOR
            L = {int(ln): enr(layer == ln) for ln in np.unique(layer) if ln >= 0}
            T = {TYPE_LABEL.get(c, c): enr(comp == c) for c in np.unique(comp)}
            out[(model, method)] = (L, T)
    return out


E = enrichment()
MODELS = [c[0] for c in CELLS]
TYPE_ORDER = [lab for _, lab in TYPES]
TICKS = [-2, -1, 0, 1, 2]; TLAB = ["0.01×", "0.1×", "1×", "10×", "100×"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- A: diverging bars -------
def variant_bars():
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for r, model in enumerate(MODELS):
        for c, kind in enumerate(("layers", "types")):
            ax = axes[r, c]
            groups = (sorted(E[(model, "MAttr")][0]) if kind == "layers"
                      else [t for t in TYPE_ORDER if t in E[(model, "MAttr")][1]])
            x = np.arange(len(groups)); w = 0.4
            for j, m in enumerate(COL):
                vals = [E[(model, m)][kind == "types"][g] for g in groups]
                ax.bar(x + (j - 0.5) * w, vals, w, color=COL[m], label=m, edgecolor="none")
            ax.axhline(0, color="#444", lw=0.5)
            ax.set_xticks(x); ax.set_xticklabels([str(g) for g in groups], rotation=90)
            ax.set_yticks(TICKS); ax.set_yticklabels(TLAB); ax.set_ylim(-2.1, 2.1)
            ax.grid(axis="y", color="#e5e5e5", lw=0.4); ax.set_axisbelow(True)
            ax.set_title(f"{model}: {'by layer' if kind == 'layers' else 'by unit type'}", pad=2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=2)
    fig.supylabel("Enrichment over uniform", fontsize=7, x=0.01)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    save(fig, "A_diverging_bars")


# ---------------------------------------------------------------- B: heatmap --------------
def variant_heatmap():
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 2.2), gridspec_kw=dict(width_ratios=[2.2, 1],
                                                                       height_ratios=[16, 32]))
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for r, model in enumerate(MODELS):
        rows = [f"{m}" for m in COL]
        for c, kind in enumerate(("layers", "types")):
            ax = axes[r, c]
            groups = (sorted(E[(model, "MAttr")][0]) if kind == "layers"
                      else [t for t in TYPE_ORDER if t in E[(model, "MAttr")][1]])
            M = np.array([[E[(model, m)][kind == "types"][g] for g in groups] for m in COL])
            ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    ax.text(j, i, "—" if v <= FLOOR else f"{10 ** v:.1f}" if abs(v) < 1 else f"{10 ** v:.0f}",
                            ha="center", va="center", fontsize=4.2,
                            color="#fff" if abs(v) > 1.2 else "#000")
            ax.set_xticks(range(len(groups))); ax.set_xticklabels([str(g) for g in groups],
                                                                   rotation=90, fontsize=5)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=6)
            ax.tick_params(length=0)
            ax.set_title(f"{model}: {'by layer' if kind == 'layers' else 'by unit type'}",
                         fontsize=6.5, pad=2)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    cax = fig.add_axes([1.0, 0.25, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu"), cax=cax, ticks=TICKS)
    cb.set_ticklabels(TLAB); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
    cb.set_label("Enrichment", fontsize=6)
    save(fig, "B_heatmap")


# ---------------------------------------------------------------- C: dumbbell + areas -----
def variant_dumbbell():
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for r, model in enumerate(MODELS):
        # left: depth profiles as areas above/below 1x
        ax = axes[r, 0]
        layers = sorted(E[(model, "MAttr")][0])
        for m in COL:
            v = np.array([E[(model, m)][0][ln] for ln in layers])
            ax.fill_between(layers, 0, v, color=COL[m], alpha=0.35, lw=0)
            ax.plot(layers, v, color=COL[m], lw=1.0, label=m)
        ax.axhline(0, color="#444", lw=0.5)
        ax.set_xlim(layers[0], layers[-1]); ax.set_xticks([layers[0], layers[-1] // 2, layers[-1]])
        ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["0.1×", "1×", "10×"]); ax.set_ylim(-1.2, 1.2)
        ax.set_title(f"{model}: by layer", pad=2); ax.set_xlabel("Layer", labelpad=1)
        ax.grid(axis="y", color="#e5e5e5", lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # right: dumbbells per type
        ax = axes[r, 1]
        types = [t for t in TYPE_ORDER if t in E[(model, "MAttr")][1]]
        y = np.arange(len(types))[::-1]
        a = np.array([E[(model, "MAttr")][1][t] for t in types])
        b = np.array([E[(model, "EG")][1][t] for t in types])
        ax.hlines(y, a, b, color="#bbbbbb", lw=1.0, zorder=1)
        ax.scatter(a, y, color=COL["MAttr"], s=14, zorder=3, label="MAttr")
        ax.scatter(b, y, color=COL["EG"], s=14, zorder=3, label="EG")
        ax.axvline(0, color="#444", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels(types)
        ax.set_xticks(TICKS); ax.set_xticklabels(TLAB, fontsize=5.5); ax.set_xlim(-2.2, 2.2)
        ax.set_title(f"{model}: by unit type", pad=2)
        ax.grid(axis="x", color="#e5e5e5", lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=2)
    fig.supylabel("Enrichment over uniform", fontsize=7, x=0.01)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    save(fig, "C_dumbbell")


if __name__ == "__main__":
    variant_bars(); variant_heatmap(); variant_dumbbell()
