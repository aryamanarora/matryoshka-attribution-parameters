"""Adam eps x score-LR for the CLEAN fr2de/Qwen-14B mask sweep: one faceted figure, six columns.

*** OUTPUT IS `param_epsgrid_facets.pdf`, NOT `epsgrid_facets.pdf`. *** Both this repo and the
sibling push figures into ONE Overleaf `figs/` directory, and the sibling already owns
`epsgrid_facets.pdf` (its SVA+ / MLP-neuron build). An earlier revision of this script wrote that
name and silently replaced their figure in the paper. Every figure this repo contributes to the
shared directory therefore carries the `param_` prefix -- parameter-space, as against their
representation-space work.

TRANSCRIBED FROM the sibling repo's ``plots/plot_epsgrid_facets.py`` -- one ``subplots`` grid, one
horizontal colourbar per COLUMN above the panel carrying the metric name, y ticks only on the
first column, annotated cells with the leading zero stripped, and the same underline convention
(drawn, not typeset -- matplotlib mathtext has no ``\\underline``). Same constants (FIG_W 5.5,
PANEL_H 1.06, the FS_* set), so the two figures are visually interchangeable.

REPLACES the four standalone quarter-width PDFs. Those were four independent floats; merging them
costs nothing (the metrics are in different units, so the scales were never shared anyway) and
buys one float, one set of margins, and column titles that cannot drift from their panels.

BRIGHT IS BETTER-OR-MORE IN EVERY COLUMN, which is why the two loss columns take the REVERSED map.
For the behaviour columns "more" is the honest word rather than "better" -- a high off-target
log-AUC means the mask reproduces the generalised habit, which is what the attribution is asked
to find, not a quality score.

*** THE UNDERLINE MARKS THE FOUR METRIC COLUMNS ONLY. *** A rank correlation has no "beats"
notion, so the two rho columns carry no marks. On off-target, "beats" means RECOVERS MORE, not
better -- see above.

WHICH SGD THE REFERENCES ARE, and this is the one choice in the figure that changes what it says.
``MAttr+SGD`` is taken at **its own optimum** (the LR sweep's best cell per metric for the
underline; lr 10000, its train-loss optimum, for the rho column), matching the rule the LR figure's
optimum rings encode: reading two optimizers at one shared learning rate measures whichever sits
further from its own optimum. That matters here more than anywhere, because THE SGD ARM IS NOT ONE
RANKING -- rho(SGD@lr10, SGD@lr10000) is **0.277**. At the low end SGD essentially reproduces
stepless IG (rho 0.916); at its optimum it does not (rho 0.314). So a "rho vs SGD" column drawn
against lr 10 would be a second copy of the "rho vs stepless IG" column, and the high-eps corner's
famous agreement with SGD (rho 0.98 against lr 10) is really agreement with the IG-like ranking
that low-LR SGD happens to produce. Both readings are visible here: the sIG column carries the
first, the SGD column the second.

DATA. Every cell is a clean `adam_eb1_*` run (defaults + effective batch 1, log k-schedule, 7,200
steps). The (lr 0.05, eps 1e-8) cell IS the defaults run `adam_eb1` and is not duplicated. eps set
{1e-8, 1e-6, 1e-4, 1e-2}; the quarantined grid's {1e-5, 1e-3, 1e-1} rows were not re-run, and the
transition the grid exists to show sits between 1e-6 and 1e-4 regardless.

ONE CAVEAT THAT DOES NOT SHOW IN THE FIGURE. Three cells -- (0.005, 1e-4), (0.005, 1e-2) and
(0.05, 1e-8) -- were fitted before the seeded-loader change, so they carry a different data order
from the other nine. Worth ~0.0015 nats on the loss columns, invisible against their spread, but
~0.15 on off-target, where the lr 0.005 column's eps transition coincides exactly with the order
boundary. The transition replicates in the other two columns, which are internally one order, so
the finding holds -- do not cite the lr 0.005 column alone for it.

    uv run python plots/plot_eps_lr_heatmap.py            # reuses the rho cache
    uv run python plots/plot_eps_lr_heatmap.py --recompute-rho
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})

R = "runs/fr2de_qwen25_14b_posthoc_"

#: Per-unit-mode grid definition. `nonresid` is the full 4x3 grid; `tensor` is a CROSS -- five
#: learning rates at the default eps and four eps values at the default lr, sharing the
#: (0.05, 1e-8) corner -- so 8 of its 20 cells are measured and the rest render blank. That was
#: deliberate: the tensor LR axis turned out flat (0.0037 nats over two decades) and eps
#: irrelevant across six decades, so filling the interior would have cost twelve cells to confirm
#: a null the cross already shows. A blank cell here means NOT RUN, never "zero".
GRIDS = {
    "nonresid": dict(
        epss=[1e-8, 1e-6, 1e-4, 1e-2], labels=["10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²"],
        lrs=[0.005, 0.05, 0.5], stem="adam_eb1",
        sgd_arm=["sgd_eb1_lr0p3", "sgd_eb1_lr1", "sgd_eb1_lr3", "sgd_eb1", "sgd_eb1_lr30",
                 "sgd_eb1_lr100", "sgd_eb1_lr300", "sgd_eb1_lr1000", "sgd_eb1_lr3000",
                 "sgd_eb1_lr10000", "sgd_eb1_lr30000"],
        sgd_rho="sgd_eb1_lr10000", ig="steplessig_epoch"),
    "tensor": dict(
        epss=[1e-8, 1e-4, 1e-2, 1e0], labels=["10⁻⁸", "10⁻⁴", "10⁻²", "10⁰"],
        lrs=[0.0005, 0.005, 0.05, 0.5, 5], stem="adam_tensor",
        sgd_arm=["sgd_tensor_lr1", "sgd_tensor", "sgd_tensor_lr100", "sgd_tensor_lr1000",
                 "sgd_tensor_lr10000"],
        sgd_rho="sgd_tensor", ig="steplessig_tensor"),
}


def build(unit):
    """(EPSS, EPS_LABEL, LRS, RUN, SGD_ARM, SGD_RHO, IG_REF) for one unit mode."""
    g = GRIDS[unit]
    def tag(v):
        s = f"{v:g}"
        return s.replace(".", "p") if "e" not in s else s.replace("e-0", "e-")
    run = {}
    if unit == "nonresid":
        et = {1e-8: "1e-8", 1e-6: "1e-6", 1e-4: "0p0001", 1e-2: "0p01"}
        lt = {0.005: "0p005", 0.05: "0p05", 0.5: "0p5"}
        run = {(e, lr): f"{g['stem']}_lr{lt[lr]}_eps{et[e]}" for e in g["epss"] for lr in g["lrs"]}
        run[(1e-8, 0.05)] = g["stem"]
    else:
        et = {1e-4: "1e-4", 1e-2: "1e-2", 1e0: "1"}
        for lr in g["lrs"]:                       # the eps 1e-8 row
            run[(1e-8, lr)] = g["stem"] if lr == 0.05 else f"{g['stem']}_lr{tag(lr)}"
        for e in g["epss"][1:]:                   # the lr 0.05 column
            run[(e, 0.05)] = f"{g['stem']}_eps{et[e]}"
    return g["epss"], g["labels"], g["lrs"], run, g["sgd_arm"], g["sgd_rho"], g["ig"]


#: (key, title, cmap, fmt, lower_is_better|None, (vmin, vmax)|None for auto)
#:
#: THE COLOURMAP ENCODES WHAT KIND OF QUANTITY THE COLUMN IS, following the sibling's rule that
#: viridis means "higher is better" and cividis means "this is a diagnostic, nothing here is
#: better or worse". Three families, and each says something:
#:
#:   viridis / viridis_r -- A SCORE. Train loss, held-out loss and in-dist expression all have a
#:       direction, so the reversed map on the two losses keeps bright = better in all three and
#:       a reader never has to remember which way a column runs.
#:   cividis -- OFF-TARGET, which is NOT a quality score. A high off-target log-AUC means the mask
#:       reproduces the generalised habit; that is what the attribution is asked to find, not a
#:       thing to maximise. Drawing it in the same "higher is better" map as the three scores
#:       invites exactly the misreading, so it gets the diagnostic map instead.
#:   magma -- RANK AGREEMENT. Neither a score nor a per-cell diagnostic but a comparison against
#:       an external ranking, so it takes its own family; and the two rho columns SHARE a fixed
#:       0-1 scale, because the reading they exist for is "does this cell look more like stepless
#:       IG or more like SGD" and per-column autoscaling makes exactly that comparison impossible.
COLS = [("trainloss", "Train loss log-AUC", "viridis_r", "%.3f", True, None),
        ("testloss", "Test loss log-AUC", "viridis_r", "%.3f", True, None),
        ("indist", "In-dist log-AUC", "viridis", "%.2f", False, None),
        ("offtarget", "Off-target log-AUC", "cividis", "%.2f", False, None),
        ("rho_ig", "Spearman $\\rho$ vs. EG", "magma", "%.2f", None, (0.0, 1.0)),
        ("rho_sgd", "Spearman $\\rho$ vs. SGD", "magma", "%.2f", None, (0.0, 1.0))]

GETTER = {
    "trainloss": lambda c: c["sft_loss"]["train"]["loss"],
    "testloss": lambda c: c["sft_loss"]["test"]["loss"],
    "indist": lambda c: c["language"]["in_dist"]["target_frac"],
    "offtarget": lambda c: c["language"]["off_target"]["target_frac"],
}

FIG_W, PANEL_H = 5.5, 1.06
HEAD, FOOT = 0.62, 0.62        # colourbar band + column titles; x ticks, label, footnote
FS_LAB, FS_TICK, FS_CELL, FS_STRIP = 6.0, 4.6, 4.4, 5.6
UNDERLINE_LW, UNDERLINE_PAD = 0.5, 0.0018
RHO_CACHE = "plots/data/epsgrid_rho_{unit}.json"


def log_auc(xs, ys):
    lx = [math.log(x) for x in xs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2
               for i in range(len(ys) - 1)) / (lx[-1] - lx[0])


def auc(d, which):
    if not os.path.exists(R + d + "/evals.json"):
        return np.nan
    blob = json.load(open(R + d + "/evals.json"))["final"]
    fr = sorted(float(k.split("_")[1]) for k in blob if k.startswith("frac_"))
    return log_auc(fr, [GETTER[which](blob[f"frac_{f:g}"]) for f in fr])


def spearman_table(unit, RUN, SGD_RHO, IG_REF):
    """``{run: {"rho_ig": x, "rho_sgd": y}}``, cached -- each score vector is 2.58M floats and
    there are fourteen of them, so recomputing on every re-render costs minutes for numbers that
    only change when a run does."""
    cache = RHO_CACHE.format(unit=unit)
    if os.path.exists(cache):
        return json.load(open(cache))
    import torch

    def scores(d):
        b = torch.load(R + d + "/final.pt", map_location="cpu", weights_only=False)
        return b["scores"].float().flatten()

    def rank(x):
        idx = x.argsort()
        r = torch.empty_like(x, dtype=torch.float64)
        r[idx] = torch.arange(len(x), dtype=torch.float64)
        return r - r.mean()

    ig, sgd = rank(scores(IG_REF)), rank(scores(SGD_RHO))
    out = {}
    for d in set(RUN.values()):
        r = rank(scores(d))
        out[d] = {"rho_ig": float((r * ig).sum() / (r.norm() * ig.norm())),
                  "rho_sgd": float((r * sgd).sum() / (r.norm() * sgd.norm()))}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"), indent=1, sort_keys=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute-rho", action="store_true")
    ap.add_argument("--unit", choices=list(GRIDS), default="nonresid")
    args = ap.parse_args()
    EPSS, EPS_LABEL, LRS, RUN, SGD_ARM, SGD_RHO, IG_REF = build(args.unit)
    cache = RHO_CACHE.format(unit=args.unit)
    if args.recompute_rho and os.path.exists(cache):
        os.remove(cache)
    rho = spearman_table(args.unit, RUN, SGD_RHO, IG_REF)

    M = {}
    for key, *_ in COLS:
        if key.startswith("rho_"):
            M[key] = np.array([[rho.get(RUN.get((e, lr), ""), {}).get(key, np.nan)
                                for lr in LRS] for e in EPSS])
        else:
            M[key] = np.array([[auc(RUN[(e, lr)], key) if (e, lr) in RUN else np.nan
                                for lr in LRS] for e in EPSS])

    # references: stepless IG, and the BEST cell of the whole SGD arm in that metric's direction
    refs = {}
    for key, _, _, _, low, _ in COLS:
        if low is None:
            continue
        sgd = [auc(d, key) for d in SGD_ARM]
        refs[key] = (auc(IG_REF, key), (min if low else max)(sgd))

    fh = PANEL_H + HEAD + FOOT
    fig, axes = plt.subplots(1, len(COLS), figsize=(FIG_W, fh), squeeze=False)
    underline, counts = [], {}
    for ci, (key, title, cmap, fmt, low, vlim) in enumerate(COLS):
        ax = axes[0][ci]
        m = M[key]
        if vlim is not None:
            vmin, vmax = vlim                     # fixed, so the two rho columns are comparable
        else:
            vmin, vmax = np.nanmin(m), np.nanmax(m)
            pad = 0.05 * (vmax - vmin)
            vmin, vmax = vmin - pad, vmax + pad
        ax.imshow(m, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        counts[key] = 0
        for i in range(len(EPSS)):
            for j in range(len(LRS)):
                if np.isnan(m[i, j]):
                    continue
                v = (m[i, j] - vmin) / (vmax - vmin)
                # every map here is dark at its low end (magma and cividis included), so the
                # reversed ones invert the test and the rest share one rule
                dark = (v > 0.4) if cmap.endswith("_r") else (v < 0.6)
                t = ax.text(j, i, (fmt % m[i, j]).lstrip("0"), ha="center", va="center",
                            fontsize=FS_CELL, color="#ffffff" if dark else "#000000")
                if low is not None:
                    a, b = refs[key]
                    beats = (m[i, j] < a and m[i, j] < b) if low else (m[i, j] > a and m[i, j] > b)
                    if beats:
                        underline.append(t)
                        counts[key] += 1
        # ROTATE ONCE THERE ARE MORE THAN THREE COLUMNS. At the tensor grid's five columns a
        # 1.1in-wide panel gives each label ~0.2in, and "0.0005" next to "0.005" ran together
        # into "0.00050.005" -- unreadable, and wrong in a way that looks like a typo.
        rot = 45 if len(LRS) > 3 else 0
        ax.set_xticks(range(len(LRS)), [f"{lr:g}" for lr in LRS],
                      rotation=rot, ha="right" if rot else "center",
                      rotation_mode="anchor" if rot else None)
        ax.set_yticks(range(len(EPSS)), EPS_LABEL if ci == 0 else [])
        ax.tick_params(labelsize=FS_TICK, length=1.2, pad=1.0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        if ci == 0:
            ax.set_ylabel("Adam $\\epsilon$", fontsize=FS_LAB, labelpad=1.5)
        ax.set_xlabel("Learning rate", fontsize=FS_LAB, labelpad=1.5)

    fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0.16 / fh, 1, 1 - HEAD / fh))
    # ONE COLOURBAR PER COLUMN, above the panel and spanning 92% of its width -- horizontal
    # because a vertical bar per column would cost six times its width out of the data area, and
    # 92% because at full width one bar's last tick collides with the next bar's first.
    for ci, (key, title, cmap, fmt, low, vlim) in enumerate(COLS):
        b0 = axes[0][ci].get_position()
        cax = fig.add_axes([b0.x0 + 0.04 * b0.width, 1 - (HEAD - 0.30) / fh,
                            0.92 * b0.width, 0.016])
        cb = fig.colorbar(axes[0][ci].images[0], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=FS_TICK, width=0.3, length=1.2, pad=0.8)
        cb.ax.xaxis.set_major_locator(MaxNLocator(3, prune="upper"))
        cb.outline.set_linewidth(0.5)
        cax.set_title(title, fontsize=FS_LAB, pad=2.5)

    fig.text(0.5, 0.006, "underlined: beats both EG and MAttr+SGD at its own optimum",
             ha="center", va="bottom", fontsize=FS_STRIP)

    # DRAWN, NOT TYPESET: mathtext has no \underline, so the rule is a Line2D under each marked
    # value's measured extent, placed after the layout is final and inheriting the value's own
    # contrast-aware colour (white on dark cells, black on light).
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for t in underline:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y0), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        fig.add_artist(Line2D([x0, x1], [y0 - UNDERLINE_PAD] * 2, lw=UNDERLINE_LW,
                              color=t.get_color(), transform=fig.transFigure, zorder=5))

    stem = "param_epsgrid_facets" + ("" if args.unit == "nonresid" else f"_{args.unit}")
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/{stem}.{ext}", dpi=300)
    filled = int(np.sum(~np.isnan(M["trainloss"])))
    print(f"wrote plots/{stem}.pdf ({len(COLS)} columns, {filled}/{len(EPSS)*len(LRS)} cells run)")
    for key, title, _, _, low, _ in COLS:
        extra = ""
        if low is not None:
            extra = (f"   refs: EG {refs[key][0]:.4f}, best SGD {refs[key][1]:.4f}"
                     f"   underlined {counts[key]}/{len(EPSS) * len(LRS)}")
        print(f"  {title:26s} {np.nanmin(M[key]):.4f} .. {np.nanmax(M[key]):.4f}{extra}")


if __name__ == "__main__":
    main()
