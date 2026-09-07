"""OlmPool across architectures: a mask statistic against the paper's long-context score.

One point per model, from ``plots/data/olmpool/summary.json`` (scripts/olmpool_analysis.py) and
the paper's HELMET/RULER table. Colour is the paper's four features (QK norm, GQA, SWA,
4K pretraining), so a trend along x that also sorts by colour is the mechanistic version of the
paper's "count the detrimental features" predictor. Raw matplotlib: the points are labelled
directly (26 names do not fit a legend) and a fit line with its R² is drawn per panel.

    uv run python plots/plot_olmpool_cross.py --arm fold_learned --stat auc16k
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import palette

plt.rcParams.update(palette.RC)

FEATURE_COLOR = {"none": palette.COLOR["adam"], "LQK": palette.COLOR["ixg:base"],
                 "HQK": palette.COLOR["ixg:mc"]}


def stat_of(rec, name):
    c = rec["curves"]
    if name.startswith("auc"):
        ctx = name[3:]
        key = f"niah/ctx_{int(ctx[:-1]) * 1024}/acc"
        return rec["auc"].get(key)
    if name.startswith("gain"):          # full_delta - pretrained at that length
        ctx = name[4:]
        key = f"niah/ctx_{int(ctx[:-1]) * 1024}/acc"
        cur = c.get(key, {})
        return cur.get("full_delta", np.nan) - cur.get("pretrained", np.nan)
    if name.startswith("frac80"):        # smallest fraction reaching 80% of the full-delta gain
        ctx = name[6:]
        key = f"niah/ctx_{int(ctx[:-1]) * 1024}/acc"
        cur = c.get(key, {})
        lo, hi = cur.get("pretrained", np.nan), cur.get("full_delta", np.nan)
        for f in sorted(float(k[5:]) for k in cur if k.startswith("frac_")):
            if cur[f"frac_{f:g}"] - lo >= 0.8 * (hi - lo):
                return f
        return 1.0
    if name.startswith("overshoot") or name.startswith("frac90") or name.startswith("peak"):
        st = "overshoot" if name.startswith("overshoot") else ("frac_90" if name.startswith("frac90") else "peak")
        ctx = name[len(st.replace("_", "")):]
        key = f"niah/ctx_{int(ctx[:-1]) * 1024}/acc"
        return rec.get("peak", {}).get(key, {}).get(st)
    if name.startswith("headshare"):        # share of all heads inside the top 0.5% (all_* arms)
        return rec.get("top_split", {}).get("0.005", {}).get("frac_of_heads")
    if name == "layer_spread":           # entropy of the top-64 heads' layer distribution
        p = np.array(rec["layer_top64"], dtype=float)
        p = p / max(1, p.sum())
        return float(-(p[p > 0] * np.log(p[p > 0])).sum())
    if name == "rh_spearman":
        keys = [k for k in rec["retrieval"] if k.startswith("lc@")]
        return max((rec["retrieval"][k]["spearman"] for k in keys), default=np.nan)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="plots/data/olmpool/summary.json")
    ap.add_argument("--arm", default="fold_learned")
    ap.add_argument("--stats", nargs="*", default=["auc16k", "auc32k", "frac8016k"])
    ap.add_argument("--x", default="helmet", choices=["helmet", "ruler"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    recs = [r for r in json.loads(Path(a.summary).read_text())
            if r["arm"] == a.arm and r.get("results") and r["model"] not in a.exclude]
    fig, axes = plt.subplots(1, len(a.stats), figsize=(1.9 * len(a.stats) + 0.4, 2.0))
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, a.stats):
        xs, ys = [], []
        for r in recs:
            x = r["results"][a.x]["32k"]
            y = stat_of(r, st)
            if y is None or (isinstance(y, float) and np.isnan(y)):
                continue
            feat = r["results"]["qk_norm"]
            marker = "s" if r["results"]["swa"] else "o"
            ax.scatter(x, y, s=14, color=FEATURE_COLOR[feat], marker=marker, lw=0,
                       alpha=0.9, zorder=3)
            ax.annotate(r["results"]["init"] + ("*" if r["results"]["pretrain_ctx"] == "4K" else ""),
                        (x, y), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
            xs.append(x); ys.append(y)
        if len(xs) >= 3:
            b, c0 = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 10)
            ax.plot(xx, b * xx + c0, color="#888888", lw=0.6, zorder=2)
            r2 = 1 - np.var(np.array(ys) - (b * np.array(xs) + c0)) / np.var(ys)
            ax.text(0.03, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=6, va="top")
        ax.set_xlabel(f"{a.x.upper()} @ 32K (paper)", fontsize=7)
        ax.set_ylabel({"auc16k": "Retrieval log-AUC @ 16K", "auc32k": "Retrieval log-AUC @ 32K",
                       "overshoot32k": "Best Sparse Mask − Released @ 32K",
                       "overshoot16k": "Best Sparse Mask − Full Update @ 16K",
                       "peak32k": "Best Sparse-Mask Retrieval @ 32K",
                       "frac9032k": "Fraction Kept for 90% of Peak @ 32K",
                       "headshare": "Heads in Top 0.5% of Units",
                       "frac8016k": "Fraction of Heads for 80% of Gain @ 16K",
                       "layer_spread": "Layer Entropy of Top-64 Heads",
                       "rh_spearman": "Spearman vs Retrieval Score",
                       "gain16k": "Retrieval Gain @ 16K"}.get(st, st), fontsize=7)
        palette.furnish(ax) if hasattr(palette, "furnish") else None
        ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    out = a.out or f"plots/olmpool_cross_{a.arm}.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
