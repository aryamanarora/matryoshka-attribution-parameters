"""OlmPool: how the all-units MAttr ranking allocates itself across unit types.

One panel per model. For each unit type -- query-head units, KV-group units, MLP neurons, and
the model's retrieval heads (a subset of the query heads: Wu et al. score >= 0.1 at 32K on the
long-context checkpoint) -- the fraction of that type's units that fall within the top-k units
of the `all_learned` ranking, as k runs over all ~460K units (log x). Chance is k / N.

    uv run python plots/plot_olmpool_unit_cdf.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import palette

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from olmpool_analysis import load_run, unit_table  # noqa: E402

plt.rcParams.update(palette.RC)

MODELS = [("G_pre_8kv_8k_14k", "G: Llama, 8 kv (52)"), ("J_pre_16kv_8k_14k", "J: Llama, 16 kv (56)"),
          ("H_post_LQK_32kv_8k_11k_SWA", "H: Olmo-3, SWA (48)"),
          ("G_post_LQK_8kv_8k_14k", "G: post-norm + LQK (48)")]
TYPES = [("q", "query heads", palette.COLOR["adam"], "solid"),
         ("kv", "KV groups", palette.COLOR["ixg:mc"], "solid"),
         ("mlp", "MLP neurons", palette.COLOR["ixg:base"], "solid"),
         ("rh", "retrieval heads", palette.COLOR["adam"], (0, (3, 1.5))),
         # present only in the arms that score them (allnorm_*, full_*)
         ("norm", "layer-norm gains", palette.COLOR["sgd"], "solid"),
         ("qk_norm", "QK-norm gains", palette.COLOR["sgd"], (0, (3, 1.5))),
         ("embed", "embedding rows", "#888888", "solid"),
         ("lm_head", "output-head rows", "#888888", (0, (3, 1.5)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/olmpool")
    ap.add_argument("--rh", default="runs/olmpool_rh")
    ap.add_argument("--arm", default="all_learned")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 3.6), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.22})
    for ax, (model, label) in zip(axes.flat, MODELS):
        run = Path(a.runs) / model / a.arm
        args, layout, scores, ev, ds = load_run(run)
        ut = unit_table(layout)
        n = layout.total
        kind = np.array(["other"] * n, dtype=object)
        lh = {}
        for u, (l, k, j) in ut.items():
            kind[u] = k
            if k == "q":
                lh[(l, j)] = u
        rh_path = Path(a.rh) / f"{model}__lc.json"
        is_rh = np.zeros(n, dtype=bool)
        if rh_path.exists():
            rh = json.loads(rh_path.read_text())
            Lc = max(rh["lengths"], key=int)
            S = np.array(rh["lengths"][Lc]["score_success"])
            for (l, j), u in lh.items():
                if S[l, j] >= 0.1:
                    is_rh[u] = True
        order = np.argsort(-scores.numpy())
        ks = np.arange(1, n + 1)
        counts = {}
        for key, name, color, ls in TYPES:
            member = is_rh if key == "rh" else (kind == key)
            m = int(member.sum())
            counts[key] = m
            if m == 0:
                continue
            cdf = np.cumsum(member[order]) / m
            ax.plot(ks, cdf, color=color, lw=0.9, ls=ls, zorder=3,
                    label=name if ax is axes.flat[0] else None)
        kv = f", {counts['kv']:,} KV groups" if counts["kv"] else " (q/k/v/o tied)"
        extra = "".join(f", {counts[k]:,} {n}" for k, n in (("norm", "norms"), ("qk_norm", "QK gains"),
                                                              ("embed", "embed rows"), ("lm_head", "head rows"))
                        if counts.get(k))
        label = (f"{label}\n{counts['q']:,} heads{kv}, {counts['mlp'] / 1000:.0f}K neurons{extra}\n"
                 f"{counts['rh']} retrieval heads")
        ax.plot(ks, ks / n, color="#999999", lw=0.6, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(1, n)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
        ax.set_xticklabels(["10⁰", "10¹", "10²", "10³", "10⁴", "10⁵"])
        ax.text(0.03, 0.96, label, fontsize=5, transform=ax.transAxes, va="top", linespacing=1.3)
        palette.furnish(ax)
        ax.tick_params(labelsize=6, width=0.4, length=2)
    axes[1, 0].set_xlabel("Top-k Units by MAttr Score", fontsize=7)
    axes[1, 1].set_xlabel("Top-k Units by MAttr Score", fontsize=7)
    axes[0, 0].set_ylabel("Fraction of Type Captured", fontsize=7)
    axes[1, 0].set_ylabel("Fraction of Type Captured", fontsize=7)
    axes[0, 0].text(6000, 0.035, "chance", fontsize=5, color="#777777")
    fig.legend(fontsize=5.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08),
               handlelength=1.8, handletextpad=0.4, columnspacing=1.2)
    out = a.out or "plots/olmpool_unit_cdf.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
