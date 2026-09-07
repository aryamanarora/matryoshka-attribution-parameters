"""Behaviour and objective against each other, one point per attribution run on fr2de/Qwen-14B.

THE QUESTION THESE ANSWER that no per-metric figure can: the eps grid and the LR row each show
one metric at a time, so "which settings buy one thing without the other" has to be reconstructed
by eye across panels. Here each pair of metrics is one plane, and every fitted cell, every
closed-form reference and the random control sit in it together.

Two figures:

  param_on_vs_off.pdf            in-dist expression (x) against off-target expression (y).
  param_loss_vs_expression.pdf   held-out loss (x) against each expression axis in turn -- i.e.
                                 does fitting the objective better buy behaviour, and is the
                                 answer the same in-distribution as off-target.

BOTH COORDINATES ARE log-AUCs over the sparsity sweep, not values at one k: a single sparsity
would make the answer depend on which k was picked, and the sweep is the object every other
figure here summarises the same way. HELD-OUT loss rather than train, because it is the axis that
is not the fitting objective of three of the arms -- a train-loss x would put the fitted methods
on the axis they were optimised for and the closed-form ones on one they were not.

NEITHER EXPRESSION AXIS IS "BETTER". High in-dist means the mask keeps the finetune's own
behaviour; high off-target means it also reproduces the habit the finetune generalised. Which
corner is desirable depends on the question being asked of the attribution, so those axes carry
no direction arrows and no optimum marks -- these are maps, not rankings. The loss axis DOES have
a direction and its label says so.

`unit: tensor` CELLS ARE EXCLUDED, not drawn faintly. A tensor-granularity sweep's sparsity axis
is a fraction of 480 whole matrices where a nonresid one is a fraction of 2.58M rows and columns,
so its log-AUC integrates over a different object; those five cells landed far from every nonresid
run on both axes, and reading that gap as a method difference is exactly the error the exclusion
prevents. plot_tensor_trainloss.py compares that family on its own terms.

    uv run python plots/plot_on_vs_off.py
"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib import font_manager
from matplotlib.lines import Line2D

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")
plt.rcParams.update({
    "font.family": FAMILY,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY,
    "mathtext.it": f"{FAMILY}:italic", "mathtext.bf": f"{FAMILY}:bold",
    "mathtext.cal": f"{FAMILY}:italic", "mathtext.sf": FAMILY, "mathtext.tt": FAMILY,
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})

GLOB = "runs/fr2de_qwen25_14b_posthoc_*"

#: family -> (label, colour, marker). Colours follow plot_attrib_maxgap.py's registry.
FAM = {
    "adam_log":  ("MAttr (Adam, log $k$)", "#0072B2", "o"),
    "adam_unif": ("MAttr (Adam, unif $k$)", "#56B4E9", "s"),
    "sgd":       ("MAttr (SGD, log $k$)", "#000000", "o"),
    "sig":       ("Stepless IG", "#E69F00", "D"),
    "ig16":      ("IG (16 steps)", "#7F5E00", "D"),
    "ixg":       ("I$\\times$G @ base", "#882255", "D"),
    "random":    ("Random", "#BBBBBB", "X"),
}
ORDER = ["adam_log", "adam_unif", "sgd", "sig", "ig16", "ixg", "random"]

#: metric key -> (axis label, per-condition getter)
AXES = {
    "indist": ("In-dist expression log-AUC", lambda v: v["language"]["in_dist"]["target_frac"]),
    "offtarget": ("Off-target expression log-AUC",
                  lambda v: v["language"]["off_target"]["target_frac"]),
    "testloss": ("Held-out loss log-AUC $\\downarrow$", lambda v: v["sft_loss"]["test"]["loss"]),
}

#: (output stem, [(x key, y key), ...], legend panel index, legend corner)
FIGS = [("param_on_vs_off", [("indist", "offtarget")], 0, "upper left"),
        ("param_loss_vs_expression",
         [("testloss", "indist"), ("testloss", "offtarget")], 0, "upper right")]

#: Families whose point may sit far outside the rest on a LOSS axis, and which therefore must not
#: be allowed to set the limits. Random scores reach a held-out loss log-AUC of 1.173 against
#: 0.973-1.001 for every real method -- on a shared scale the whole informative spread collapses
#: into a sliver, which hides the actual finding (held-out loss barely moves while behaviour
#: spans four-fold). The point is still DRAWN and its value annotated at the axis edge, so it is
#: excluded from the autoscale rather than from the figure.
OFFSCALE_ON_LOSS = {"random"}


def log_auc(pairs):
    pairs = sorted(pairs)
    lx = [math.log10(x) for x, _ in pairs]
    ys = [y for _, y in pairs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2
               for i in range(len(ys) - 1)) / (lx[-1] - lx[0])


def classify(mk):
    """Which family a run belongs to, from its resolved mask block alone."""
    sc = mk.get("scores")
    if sc == "random":
        return "random"
    if sc == "ixg":
        return {"mc": "sig", "ig": "ig16", "base": "ixg"}.get(mk.get("ixg_at", "finetuned"))
    if sc != "learned":
        return None
    if mk.get("score_optimizer", "adam") == "sgd":
        return "sgd"
    return "adam_unif" if mk.get("k_schedule") == "uniform" else "adam_log"


def collect():
    out, dropped = [], 0
    for d in sorted(glob.glob(GLOB)):
        if not (os.path.exists(d + "/evals.json") and os.path.exists(d + "/config.yaml")):
            continue
        mk = (yaml.safe_load(open(d + "/config.yaml")).get("mask") or {})
        fam = classify(mk)
        if fam is None:
            continue
        if mk.get("unit", "nonresid") != "nonresid":       # see the docstring
            dropped += 1
            continue
        blob = json.load(open(d + "/evals.json"))
        blob = blob.get("final", blob)
        conds = [(float(c.split("_", 1)[1]), v) for c, v in blob.items() if c.startswith("frac_")]
        if len(conds) < 2:
            continue
        row = dict(name=os.path.basename(d), fam=fam, lr=float(mk.get("score_lr", 0.05)))
        for key, (_, get) in AXES.items():
            row[key] = log_auc([(f, get(v)) for f, v in conds])
        out.append(row)
    if dropped:
        print(f"  excluded {dropped} unit:tensor cell(s) -- different unit definition")
    return out


def draw(ax, rows, xk, yk, shade):
    # limits from the in-range families only; see OFFSCALE_ON_LOSS
    inr = [r for r in rows if not (xk == "testloss" and r["fam"] in OFFSCALE_ON_LOSS)]
    for k, setlim in ((xk, ax.set_xlim), (yk, ax.set_ylim)):
        lo, hi = min(r[k] for r in inr), max(r[k] for r in inr)
        pad = 0.06 * (hi - lo)
        setlim(lo - pad, hi + pad)
    for fam in ORDER:
        for r in [x for x in rows if x["fam"] == fam]:
            _, colour, marker = FAM[fam]
            a = shade[fam](r["lr"]) if fam in shade else 0.95
            ax.plot(r[xk], r[yk], marker, ms=4.2, ls="none", zorder=3, clip_on=True,
                    color=colour, alpha=a, mec="#000000", mew=0.35)

    # OFF-SCALE POINTS GET A CORNER NOTE, not an edge arrow: a marker excluded from the limits
    # can be outside them on EITHER axis (random is off in x on the loss panels and off in y as
    # well), so an arrow pinned to the x edge lands outside the y range and is clipped away
    # silently -- which is the failure this replaces.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    off = [(FAM[r["fam"]][0].split(" (")[0], r[xk], r[yk]) for r in rows
           if not (x0 <= r[xk] <= x1 and y0 <= r[yk] <= y1)]
    if off:
        ax.text(0.015, 0.015,
                "off scale: " + "; ".join(f"{n} ({x:.2f}, {y:.2f})" for n, x, y in off),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=4.6, color="#777777")
    ax.set_xlabel(AXES[xk][0], fontsize=7)
    ax.set_ylabel(AXES[yk][0], fontsize=7)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def main():
    rows = collect()
    # the fitted families shade by learning rate, so an arm's sweep reads as a track rather than a
    # cloud. The two arms use different LR ranges (Adam 5e-4..0.5, SGD 0.3..3e4), so each is
    # normalised WITHIN itself -- shade compares cells inside an arm and says nothing across them.
    shade = {}
    for fam in ("adam_log", "adam_unif", "sgd"):
        lrs = sorted({r["lr"] for r in rows if r["fam"] == fam})
        lo, hi = math.log10(min(lrs)), math.log10(max(lrs))
        shade[fam] = lambda v, lo=lo, hi=hi: (0.25 + 0.75 * ((math.log10(v) - lo) / (hi - lo))
                                              if hi > lo else 1.0)
    handles = [Line2D([], [], color=FAM[f][1], marker=FAM[f][2], ls="none", ms=4.2,
                      mec="#000000", mew=0.35, label=FAM[f][0]) for f in ORDER]

    for stem, panels, leg_i, leg_loc in FIGS:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.7), squeeze=False)
        for i, (xk, yk) in enumerate(panels):
            draw(axes[0][i], rows, xk, yk, shade)
        # the legend's panel and corner are NAMED, not searched: a corner that is empty today
        # fills as the sweep grows, and a legend that moves between rebuilds is worse than one
        # that is stated and checked.
        axes[0][leg_i].legend(handles=handles, fontsize=5, loc=leg_loc, frameon=True,
                              framealpha=0.95, borderpad=0.3, handletextpad=0.3,
                              labelspacing=0.25)
        fig.tight_layout(pad=0.3, w_pad=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(f"plots/{stem}.{ext}", dpi=300)
        plt.close(fig)
        print(f"wrote plots/{stem}.pdf ({len(rows)} runs, {len(panels)} panel(s))")

    for fam in ORDER:
        sub = [r for r in rows if r["fam"] == fam]
        if sub:
            print(f"  {FAM[fam][0]:26s} n={len(sub):2d}  "
                  + "  ".join(f"{k} {min(r[k] for r in sub):.3f}-{max(r[k] for r in sub):.3f}"
                              for k in AXES))


if __name__ == "__main__":
    main()
