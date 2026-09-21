"""Where MAttr's refusal masks sit in the model: by depth and by unit type, and what each slice of
the ranking is made of. Two figures for the paper's RL appendix.

Runs: the headline refusal rankings (``configs/refusal/``, MAttr with uniform k on the GRPO
reward; Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct, instruct -> base delta at ``nonresid``
granularity, i.e. one unit = one d_model-vector, so unit counts and parameter counts agree).

refusal_mask_composition.pdf  (E)
    Enrichment of a group in the top-k at the paper's budget (2% of units at 1B, 1% at 8B):
        (share of the top-k that is the group) / (share of ALL parameters that is the group),
    log scale, 1x = the ranking draws from the group exactly as often as a uniform pick would.
    Left: per transformer block, on relative depth so the 16- and 32-block models overlay.
    Right: per unit type (the embedding, the seven projections, the two per-block RMSNorm gains,
    the final norm, and for 8B the untied lm_head), as dodged bars from the 1x line. A group with
    no unit in the top-k has no log position and is drawn down to the axis floor.

refusal_mask_bands.pdf  (F)
    Composition of each successive slice of the ranking -- the top 0.002%, then what the top
    0.005% adds, ... up to the population ("all") -- stacked by unit type (left) and by depth
    quarter (right). Bands are the sparsity sweep's own grid (plot_mask_composition.FRACS_FINE),
    drawn at equal width because they are geometric; the dashed line marks the paper's budget.
    The first bands are tiny (12 units at 1B) and read as noisy.

    uv run python plots/plot_refusal_mask_composition.py
    uv run python plots/plot_refusal_mask_composition.py --out-dir ../learning-to-attribute/paper/figs

Reads each run's ``final.pt`` (scores + layout) through ``plot_mask_composition.load`` /
``unit_meta``; the composition is a property of the mask, not of any eval.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from palette import RC, furnish
from plot_mask_composition import FRACS_FINE, load, unit_meta

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update(RC)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5})

#: (label, budget, run) -- the MAttr runs the results table and the sparsity figure use
CELLS = [
    ("Llama 3.2 1B (top 2%)", 0.02, "refusal_grpo_uniform_vllm_native"),
    ("Llama 3.1 8B (top 1%)", 0.01, "refusal_grpo_8b_uniform_vllm_native"),
]
#: the two models: Wong blue / Wong vermillion, a CVD-safe pair that also separates by lightness
MCOL = ["#0072b2", "#d55e00"]

#: unit types in drawing order, with the label the axis shows. Anything the layout carries that is
#: not listed here is appended, never dropped (see plot_mask_composition.COMPONENTS for why).
TYPES = [("embed_tokens", "embed"), ("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
         ("o_proj", "o"), ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"),
         ("input_layernorm", "ln (attn)"), ("post_attention_layernorm", "ln (mlp)"),
         ("norm", "final norm"), ("lm_head", "lm_head")]
TYPE_LABEL = dict(TYPES)
TYPE_ORDER = [lab for _, lab in TYPES]
FLOOR = -2.0                    # log10 enrichment drawn for a group with no unit in the top-k
DEPTH = ["first ¼", "second ¼", "third ¼", "last ¼", "no layer"]
DEPTH_COL = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#999999"]   # Set1 + grey, as elsewhere


class Mask:
    def __init__(self, label, frac, run):
        self.label, self.frac = label, frac
        self.scores, lay, _, _ = load(ROOT / "runs" / run)
        comp, layer = unit_meta(lay)
        self.layer = np.asarray(layer)
        self.type = np.array([TYPE_LABEL.get(c, c) for c in comp])
        self.N = self.scores.numel(); self.k = int(round(frac * self.N))
        self.n_layers = int(self.layer.max()) + 1
        order = np.asarray(torch.argsort(self.scores, descending=True))
        self.rank = np.empty(self.N, dtype=np.int64); self.rank[order] = np.arange(self.N)
        self.top = self.rank < self.k
        self.types = [t for t in TYPE_ORDER if t in set(self.type)] + sorted(
            set(self.type) - set(TYPE_ORDER))
        self.quarter = np.where(self.layer < 0, 4,
                                np.minimum(4 * np.maximum(self.layer, 0) // self.n_layers, 3))

    def enr(self, mask):
        v = self.top[mask].mean() / self.frac
        return np.log10(v) if v > 0 else FLOOR


def fig_composition(masks, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for m, col in zip(masks, MCOL):
        x = np.arange(m.n_layers) / (m.n_layers - 1)
        v = [m.enr(m.layer == ln) for ln in range(m.n_layers)]
        a1.plot(x, v, "-o", color=col, ms=2.2, lw=1.0, label=m.label)
        print(f"{m.label}: k={m.k}; by layer " + ", ".join(f"L{ln} {10 ** e:.2f}x" for ln, e in
                                                             enumerate(v)))
    a1.axhline(0, color="#444", lw=0.5)
    a1.set_xticks([0, 0.5, 1]); a1.set_xticklabels(["first", "middle", "last"])
    a1.set_xlabel("Relative depth of the block", labelpad=2)
    a1.set_yticks([-1, 0, 1]); a1.set_yticklabels(["0.1×", "1×", "10×"]); a1.set_ylim(-1.1, 1.1)
    a1.set_ylabel("Enrichment in the top-k")
    a1.legend(frameon=False, loc="upper right", handlelength=1.6)
    furnish(a1); a1.grid(axis="x", visible=False)

    types = [t for t in TYPE_ORDER if any(t in m.types for m in masks)]
    x = np.arange(len(types)); w = 0.38
    for j, (m, col) in enumerate(zip(masks, MCOL)):
        v = np.array([m.enr(m.type == t) if t in m.types else np.nan for t in types])
        a2.bar(x + (j - 0.5) * w, v, w, color=col, edgecolor="none", label=m.label)
        print(f"{m.label}: by type " + ", ".join(f"{t} {10 ** e:.2f}x" for t, e in zip(types, v)
                                                   if not np.isnan(e)))
    a2.axhline(0, color="#444", lw=0.5)
    a2.set_xticks(x); a2.set_xticklabels(types, rotation=90)
    a2.set_yticks([-2, -1, 0, 1]); a2.set_yticklabels(["0.01×", "0.1×", "1×", "10×"])
    a2.set_ylim(-2.05, 1.1); a2.set_xlim(-0.6, len(types) - 0.4)
    a2.set_xlabel("Unit type", labelpad=2)
    furnish(a2); a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)
    print("wrote", out)


def fig_bands(masks, out):
    fracs = FRACS_FINE
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.1))
    tab = plt.get_cmap("tab20")
    for r, m in enumerate(masks):
        edges = [0] + [int(round(f * m.N)) for f in fracs]
        order = np.argsort(m.rank)
        for c, (kind, keys, ids, cols) in enumerate((
                ("unit type", m.types, np.array([m.types.index(t) for t in m.type]),
                 [tab(TYPE_ORDER.index(t) if t in TYPE_ORDER else 19) for t in m.types]),
                ("depth", DEPTH, m.quarter, DEPTH_COL))):
            ax = axes[r, c]
            S = np.zeros((len(fracs) + 1, len(keys)))
            for bi, (a, b) in enumerate(zip(edges, edges[1:])):
                S[bi] = np.bincount(ids[order[a:b]], minlength=len(keys)) / max(1, b - a)
            S[-1] = np.bincount(ids, minlength=len(keys)) / m.N              # the population
            xs = np.arange(len(fracs) + 1)
            ax.stackplot(xs, S.T, colors=cols, labels=keys, lw=0)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{100 * f:g}%" for f in fracs] + ["all"], rotation=90, fontsize=5)
            ax.set_xlim(0, len(fracs)); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
            ax.axvline(fracs.index(m.frac), color="#000", lw=0.6, ls="--")
            ax.set_title(f"{m.label}: by {kind}", pad=2)
            if r == 1:
                ax.set_xlabel("Band of the ranking (top fraction)", labelpad=2)
            if c == 0:
                ax.set_ylabel("Share of the band")
            if r == 0 or c == 1:
                ax.legend(fontsize=4.6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.02),
                          handlelength=0.8, handletextpad=0.4, labelspacing=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=2.2, h_pad=1.0)
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(ROOT / "plots"))
    args = p.parse_args()
    masks = [Mask(*c) for c in CELLS]
    out = Path(args.out_dir)
    fig_composition(masks, out / "refusal_mask_composition.pdf")
    fig_bands(masks, out / "refusal_mask_bands.pdf")


if __name__ == "__main__":
    main()
