"""Adam vs SGD for a post-hoc mask: loss log-AUC against score learning rate.

The parameter-space counterpart of `../learning-to-attribute/plots/plot_optimizer_lr.py`, and
deliberately in its visual language -- same panel size, same encoding (colour = optimizer,
linestyle = k-schedule), same hollow ring on each arm's own optimum, same hairline furniture -- so
a reader who has seen the MIB figure can read this one without relearning it.

TWO FIGURES, and they are the pair worth keeping -- the same four metrics at two zoom levels:

    --grid            <out>_grid.pdf     y = log-AUC, x = score LR. Which SETTING is best.
    --grid --curves   <out>_curves.pdf   y = the metric, x = mask fraction, one line per (arm, lr).
                                         Best HOW -- and the only one that can show the overshoot
                                         past `full_delta`, which no single AUC can.

    uv run python plots/plot_optimizer_lr_auc.py \\
        --glob "runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard*" \\
        --grid language:target_frac:in_dist language:target_frac:off_target \\
               sft_loss:loss:train sft_loss:loss:test
    # ...and again with --curves for the second figure.

The single-panel modes (`--eval`/`--metric`, or neither for the loss) remain for the case where a
LaTeX float wants one panel at 0.48\\linewidth, but the grid supersedes them for reading: an earlier
version of this analysis emitted six overlapping PDFs and the redundant four were deleted.

LOWER IS BETTER: the y value is the loss itself, in nats, and the axis says so with a down arrow.
(`--recovery` switches to (pretrained - loss)/(pretrained - full_delta) instead, where higher is
better -- see below for when that is the one you want.)

    uv run python plots/plot_optimizer_lr_auc.py \\
        --glob "runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard*"

WHY THE Y AXIS IS A LOG-AUC OF A RECOVERY FRACTION, which is two normalisations and both matter:

* LOG, not linear. The sweep is geometric (0.001 -> 1.0), so on a linear x the 0.5-1.0 interval is
  half the width -- and every cell here has converged by frac 0.01 onto an identical `frac_1`
  anchor. Measured on this exact sweep: linear AUC puts best-Adam at 0.934 against best-SGD 0.944,
  a 1% gap that reads as a tie; log-AUC puts them at 0.577 and 0.679, a 17% relative gap. The two
  normalisations support opposite conclusions, which is why `scripts/analysis/sparsity_auc.py` transcribes
  MIB's `acc_auc` rather than inventing a weighting.
* NATS, not a normalised fraction, and that is a deliberate reversal of a first draft. Every cell
  in one of these figures is the SAME organism over the SAME frozen delta, so they already share a
  pretrained floor and an identical `frac_1` anchor -- a recovery fraction would divide all of them
  by the same two constants and change nothing except to hide the scale. Raw nats keep the y axis
  in the units the loss is actually reported in, which is what makes "0.68 vs 0.74" legible as a
  quantity rather than an index. `--recovery` exists for the case the normalisation IS load-bearing
  -- comparing ACROSS organisms, where the floors differ and nats are not commensurable.

WHAT THE RINGS ARE FOR, and it is the same point the MIB figure makes: that each arm's optimum
sits at a different x IS the result. Reading the two optimizers at one shared LR measures whichever
is further from its own optimum, not the optimizer -- the error that produced a false "SGD does not
transfer" upstream, and that produced this repo's own overstated "SGD beats Adam by 0.3" (measured
against Adam at an inherited default that turned out to be its third-worst of five settings).

AN ARM WHOSE MAXIMUM IS AT THE EDGE OF ITS GRID GETS NO RING (:data:`NO_RING` is computed, not
hardcoded): Adam's argmax here is at the lowest LR swept, so its "optimum" is just the smallest
number tried and the curve is a cliff edge rather than a peak. Extending downward is the fix; a
ring there would assert a maximum nobody measured.
"""

import argparse
import glob as globmod
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))            # plots/palette.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))  # sparsity_auc moved here 2026-09-07
from sparsity_auc import linear_auc, log_auc, series  # noqa: E402 -- one metric definition, not two

#: Which trapezoid the y axis is. Rebound by `--linear` in main(), so every collector below reads
#: ONE name and the two weightings cannot drift apart. LOG IS THE DEFAULT AND SHOULD STAY THE
#: DEFAULT: on a linear x the 0.5->1.0 interval is half the total width, so a linear AUC is
#: dominated by the dense end where every method here has already converged onto an identical
#: `frac_1` anchor, and the sparse end the sweep exists to measure is 1.7% of the axis.
#:
#: HOW BADLY THAT MISLEADS, measured on this exact sweep rather than asserted: the two weightings
#: rank the 29 fr2de/Qwen-14B cells at Spearman 0.559, and `IxG @ finetuned` -- which is at CHANCE
#: on behaviour (0.043, tied with random scores) -- moves from 28th of 29 under log-AUC to 2nd
#: under linear. A figure drawn on the linear weighting would present the repo's clearest null as
#: its runner-up method. `--linear` exists to SHOW that, not because it is an alternative worth
#: reporting.
AUC = log_auc
AUC_NAME = "log"

from palette import (COLOR, FS_LABEL, FS_LEGEND, FS_TICK, LS, MK, RC, REF_LABEL,  # noqa: E402
                     REF_LS, furnish, sty as _sty, top_legend)

FIG_W, FIG_H = 2.7, 2.15

#: `--quarter`: the SVA+ quarter-width geometry `plot_eps_lr_heatmap.py` already uses, for a row
#: of four 0.24\textwidth subfigures. Drawn at very nearly its final size, so LaTeX barely
#: rescales and the text lands at ~6pt compiled rather than being shrunk from a half-width draw.
#:
#: IT ALSO SWITCHES TO THE SHORT LABELS, and that is one decision rather than two: "MAttr (Adam,
#: log $k$)" and "I$\times$G @ base ($\alpha$=0)" cannot fit a 1.3in panel at any size that stays
#: legible, so a quarter-width panel that kept them would either clip them or hand the legend half
#: the axes. The long names live in the caption instead.
QUARTER = {"fig": (1.30, 1.32), "label": 5.5, "tick": 4.8, "legend": 3.6}

#: short arm/reference names for `--quarter`, keyed exactly as `_draw`/`panel` build them
SHORT_ARM = {("adam", "log"): "Adam (log)", ("adam", "uniform"): "Adam (unif)",
             ("sgd", "log"): "SGD (log)", ("sgd", "uniform"): "SGD (unif)"}
SHORT_REF = {"ixg:base": r"I$\times$G", "ixg:mc": "EG",   # Expected Gradients (paper name since 2026-09-14)
             "ixg:finetuned": r"I$\times$G@ft", "random": "random"}
SHORT = False


def arm_label(opt, ks):
    """The legend name for one arm. Long by default; `--quarter` swaps in SHORT_ARM."""
    if SHORT:
        return SHORT_ARM.get((opt, ks.split(" \u00b7 ")[0]), f"{opt} ({ks})")
    return f"{'SGD' if opt == 'sgd' else 'Adam'} ({ks} $k$)"



def modal_unit(pattern) -> str:
    """The unit mode most of the matched runs use -- everything else is DROPPED, loudly.

    A `frac` means a different object under a different `mask.unit`: at 14B `nonresid` frac_0.01 is
    ~26k rows and `svd` frac_0.01 is ~72 singular DIRECTIONS, and one kept direction writes a rank-1
    update across every row of its tensor. So two unit modes on one x axis is not one comparison,
    and the figure would not look wrong -- both curves are smooth and both anchor at frac_1.

    This exists because the glob that catches the `ixg_*` reference cells (they do not share the
    `posthoc_shard` prefix) also catches `posthoc_svd`. Narrowing the glob would silently drop the
    references instead, which is the failure that put ONE ref on a figure that should carry four.
    """
    seen = {}
    for d in sorted(globmod.glob(pattern)):
        d = Path(d)
        if not ((d / "evals.json").exists() and (d / "config.yaml").exists()):
            continue
        mk = (yaml.safe_load((d / "config.yaml").read_text()).get("mask") or {})
        if mk:
            seen.setdefault(mk.get("unit", "nonresid"), []).append(d.name)
    if not seen:
        return "nonresid"
    unit = max(seen, key=lambda u: len(seen[u]))
    for u, names in seen.items():
        if u != unit:
            print(f"  dropped {len(names)} cell(s) with unit={u} (figure is unit={unit}): "
                  + ", ".join(n.split('_lr1e-4_')[-1] for n in names), file=sys.stderr)
    return unit


def arms(pattern: str, split: str, recovery: bool, behaviour=None, eps_mode="default",
         keep_refs=None):
    """``({(optimizer, k_schedule): [(lr, log_auc)]}, {reference: log_auc})``.

    The second dict holds the LR-less baselines -- the `ixg` family and the `random` control --
    which are references rather than arms; see the note where they are collected.
    """
    out, refs = {}, {}
    unit = modal_unit(pattern)
    for d in sorted(globmod.glob(pattern)):
        d = Path(d)
        if not ((d / "evals.json").exists() and (d / "config.yaml").exists()):
            continue
        mk = (yaml.safe_load((d / "config.yaml").read_text()).get("mask") or {})
        if not mk or mk.get("unit", "nonresid") != unit:
            continue
        sc = mk.get("scores")
        if sc not in ("learned", "ixg", "random"):
            continue
        try:
            if behaviour:
                xs, ys, _, _ = series(d, behaviour[0], behaviour[1], split, recovery=False)
            else:
                xs, ys, _, _ = series(d, "sft_loss", "loss", split, recovery=recovery)
        except SystemExit:
            continue
        if len(xs) < 2:
            continue
        if sc in ("ixg", "random"):
            # NO LR, so on an x=LR axis these are SCALARS, and they are drawn as horizontal
            # reference lines. That is not a workaround -- it is the honest rendering: a
            # closed-form attribution has no tuning axis, so "the level it reaches" is the whole
            # statement, and putting it flat across the panel is what lets the reader see which
            # LRs of a fitted method beat it and which do not.
            who = "random" if sc == "random" else f"ixg:{mk.get('ixg_at', 'finetuned')}"
            # `--refs` selects which of them to draw. A reference is only worth an axhline if the
            # reader is meant to compare against it, and one of them can WRECK THE SCALE it is
            # drawn on: `random` reaches 0.236 in-dist where every fitted arm sits in 0.955-0.989,
            # so including it compresses the entire result into a band a few pixels tall. Naming
            # the references explicitly is what keeps that a decision rather than a side effect of
            # which runs the glob happened to match.
            if keep_refs is not None and who not in keep_refs:
                print(f"  ref {who} not in --refs, skipped", file=sys.stderr)
                continue
            refs[who] = AUC(xs, ys)
            continue
        ks = mk.get("k_schedule", "uniform")
        cfg_all = yaml.safe_load((d / "config.yaml").read_text())
        # THIS FIGURE'S ENCODING HAS NO FREE CHANNEL FOR eps: colour is the optimizer and
        # linestyle the k-schedule, so four Adam/log cells differing only in eps would draw as one
        # dash in one colour, four times. The eps dimension gets its own figure instead
        # (`plot_eps_lr_heatmap.py` -> epsgrid_*.pdf), where lr and eps are the two axes and every
        # cell is legible. `--eps all` opts back in for a diagnostic look.
        if eps_mode == "default" and eps_tag(cfg_all):
            print(f"  skipped {d.name.split('posthoc_')[-1]} ({eps_tag(cfg_all)}; "
                  f"see epsgrid_*.pdf)", file=sys.stderr)
            continue
        tags = [t for t in (budget_tag(cfg_all), eps_tag(cfg_all)) if t]
        key = (mk.get("score_optimizer", "adam"), " · ".join([ks] + tags))
        out.setdefault(key, []).append((float(mk["score_lr"]), AUC(xs, ys)))
    for k in out:
        # duplicate LRs (the `_nll` cell is one) would draw the same point twice and could flip
        # which point the ring lands on; keep the first and say so
        seen, uniq = set(), []
        for lr, v in sorted(out[k]):
            if lr in seen:
                print(f"  {k[0]}/{k[1]}: dropped a duplicate lr={lr:g} cell", file=sys.stderr)
                continue
            seen.add(lr); uniq.append((lr, v))
        out[k] = uniq
    return out, refs



# ---- the combined view -------------------------------------------------------------------------


#: The DEFAULT fitting budget every LR/k-schedule cell uses: one epoch at effective batch 16, i.e.
#: 450 optimizer steps and 450 k-draws. Cells that vary it are a different arm, not a duplicate.
#: Effective batch 1, one epoch -- 7,200 optimizer steps and 7,200 k-draws. This was ``(1, 16)``
#: while the quarantined family was the subject; the clean-sweep era fits every cell at eb1, so
#: naming eb1 the baseline is what keeps a constant tag off every label. A quarantined cell drawn
#: on the same axes now reads ``eb16``, which is the informative direction.
DEFAULT_BUDGET = (1, 1)

#: Adam's ``score_eps`` default (config/schema.py). Anything else is a different arm -- see
#: :func:`eps_tag`.
DEFAULT_EPS = 1e-8


def eps_tag(cfg) -> str:
    """``""`` for the default ``score_eps``, else a short tag like ``eps 1e-6``.

    SAME REASON AS :func:`budget_tag`, and the same failure it prevents. The clean eps x lr grid
    puts four cells at one ``(optimizer, k_schedule, score_lr)`` -- so without this the dedup that
    drops repeated LRs discards ten of the sixteen clean cells, keeping whichever sorted first.
    eps is not decoration on this organism either: it is the knob that moves Adam onto SGD's
    ranking (spearman 0.98 at eps 1e-4), so two cells differing only in eps are further apart than
    two optimizers.
    """
    eps = ((cfg.get("mask") or {}).get("score_eps")) or DEFAULT_EPS
    return "" if float(eps) == DEFAULT_EPS else f"eps {float(eps):.0e}".replace("e-0", "e-")


def budget_tag(cfg) -> str:
    """``""`` for the default budget, else a short tag like ``10ep`` or ``eb1``.

    THIS HAS TO BE PART OF THE ARM KEY, not decoration. A 10-epoch cell and its 1-epoch twin share
    `(optimizer, k_schedule, score_lr)` exactly -- so without this the dedup that drops repeated LRs
    would silently discard one of them, and the figure would show a budget sweep as a single point
    with no indication anything was dropped. (That dedup exists for a real reason: the `_nll` cell
    IS a true duplicate of its twin. The two cases are indistinguishable by name and distinguishable
    by config, which is why both checks read the config.)
    """
    t = cfg.get("train") or {}
    ep, ga = t.get("epochs", 1), t.get("grad_accum", 16)
    if (ep, ga) == DEFAULT_BUDGET:
        return ""
    if ga == 1:
        return f"eb1"                     # effective batch 1: 16x the k-draws at equal compute
    return f"{ep}ep"

#: how a `--grid` panel is titled and which direction is better. The pair of loss panels and the
#: pair of behaviour panels answer different questions and point OPPOSITE ways (a rate is
#: higher-better, a loss lower-better), which is the whole reason they are worth seeing together:
#: an optimizer setting that reproduces the objective and one that reproduces the behaviour are
#: not the same setting, and on this organism they are measurably not.
NICE_SPLIT = {"in_dist": "in-dist", "off_target": "off-target", "train": "train", "test": "held-out"}


def grid_figure(pattern, specs, out, recovery=False, ncol=None,
                eps_mode="default", keep_refs=None):
    """One figure, one panel per `eval:metric:split`, sharing the x axis and the arm encoding."""
    parsed = []
    for spec in specs:
        try:
            ev, metric, split = spec.split(":")
        except ValueError:
            raise SystemExit(f"--grid spec must be eval:metric:split, got {spec!r}")
        is_loss = ev == "sft_loss"
        beh = None if is_loss else (ev, metric)
        data, refs = arms(pattern, split, recovery and is_loss, beh,
                          eps_mode=eps_mode, keep_refs=keep_refs)
        if not data:
            raise SystemExit(f"no runs for {spec}")
        low = is_loss and not recovery          # nats: lower is better; a rate or a recovery is not
        unit = ("loss (nats)" if is_loss and not recovery else
                "loss recovery" if is_loss else metric)
        # under `--quarter` the panel label is the SHORT form the standalone panels use --
        # split name, "loss" only when it is one, and the direction arrow. "in-dist target_frac"
        # does not fit a 1.3in panel and the arrow is the part that cannot move to the caption.
        arrow = "$\\downarrow$" if low else "$\\uparrow$"
        short = f"{NICE_SPLIT.get(split, split)}{' loss' if is_loss else ''} {arrow}"
        parsed.append((data, short if SHORT else f"{NICE_SPLIT.get(split, split)} {unit}",
                       low, refs))

    n = len(parsed)
    ncol = ncol or n
    nrow = (n + ncol - 1) // ncol
    # each panel keeps the single-panel geometry so the two forms stay visually interchangeable
    ROW_H = 2.0
    if SHORT:
        # ONE \textwidth FIGURE, not four subfigures. Same per-panel geometry as the separate
        # quarter-width PDFs (~1.3in each), so nothing about how a panel reads changes -- what
        # changes is that the row now shares one legend and one set of margins instead of
        # reserving a legend band on all four.
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.5, 1.80 * nrow), squeeze=False)
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol + 0.35, ROW_H * nrow + 0.25),
                                 squeeze=False)
    flat = [a for row in axes for a in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    for i, ((data, title, low, refs), ax) in enumerate(zip(parsed, flat)):
        _draw(ax, data, low, refs)
        ax.tick_params(labelsize=FS_TICK, length=2, width=0.5)
        # x label only on the bottom row, y arrow on every panel -- the directions DIFFER between
        # panels here, so a single shared label would be wrong for half of them
        if i // ncol == nrow - 1:
            ax.set_xlabel("score LR" if SHORT else "score learning rate", fontsize=FS_LABEL)
        if SHORT:
            # the metric goes on the y axis and the title is dropped: a title band above each
            # panel costs height the row does not have, and these are the same labels the
            # standalone panels carry, so the two forms stay interchangeable
            ax.set_ylabel(title, fontsize=FS_LABEL)
        else:
            ax.set_title(title, fontsize=FS_LABEL, pad=3)
            ax.set_ylabel(f"{AUC_NAME}-AUC " + ("$\\downarrow$" if low else "$\\uparrow$"),
                          fontsize=FS_LABEL)
    # ONE legend for the figure, outside the panels: four arms x four panels would otherwise put
    # the same key down four times and cover data in each
    handles = []
    seen = set()
    for data, _, _, _ in parsed:
        for (opt, ks) in sorted(data):
            if (opt, ks) in seen:
                continue
            seen.add((opt, ks))
            lsty, mkr, hollow = _sty(ks)
            handles.append(Line2D([0], [0], color=COLOR.get(opt, "#999999"), ls=lsty, lw=0.9,
                                  marker=mkr, ms=3.2,
                                  **({"mfc": "none", "mec": COLOR.get(opt, "#999999"), "mew": 1.0}
                                     if hollow else {"mec": "#000000", "mew": 0.4}),
                                  label=arm_label(opt, ks)))
    if not SHORT:
        # under `--quarter` the references are labelled ON their lines by `_draw`, so a key here
        # would say the same thing twice and cost a legend row the strip does not have
        for who in ("ixg:base", "ixg:mc", "ixg:finetuned", "random"):
            if any(who in (r or {}) for _, _, _, r in parsed):
                handles.append(Line2D([0], [0], color=COLOR[who], ls=REF_LS[who], lw=1.3,
                                      label=REF_LABEL[who]))
    # WRAPPED AND ON TOP, matching ../learning-to-attribute's neuron_recall.pdf. The previous
    # one-row-of-thirteen was 13 keys across a 9.7in figure at 5.8pt -- legible only because the
    # figure is wide, and it grew a key every time the sweep did. `top_legend` derives the strip
    # height from how many rows it actually wraps to, so adding an arm reflows instead of
    # overlapping the panel titles.
    if SHORT:
        fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.935))
        fig.legend(handles=handles, fontsize=FS_LEGEND, ncol=len(handles), frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, 1.005), handletextpad=0.35,
                   handlelength=1.5, columnspacing=1.2, borderpad=0.1)
    else:
        fig.tight_layout(w_pad=0.4)
        top_legend(fig, handles, nrow, ROW_H, ncol=4)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({n} panels, {sum(len(r or {}) for _, _, _, r in parsed) // n} refs)")



# ---- the curves the AUCs summarise -------------------------------------------------------------

def curve_grid(pattern, specs, out, ncol=None):
    """The same four panels, but x = mask fraction and y = the metric -- one line per (arm, lr).

    The AUC grid answers "which setting is best"; this answers "best HOW", and the two belong
    together because a single AUC provably cannot distinguish a monotone curve from one that
    overshoots and falls back -- the "a sparse mask beats the whole finetune" shape this repo keeps
    finding, and which every starred row of `scripts/analysis/sparsity_auc.py` on this organism has.

    ENCODING. Colour is still the optimizer and marker still the k-schedule, so the figure reads in
    the same language as the AUC grid; LR becomes line ALPHA within an arm (light = low), because
    hue and dash are already spent. That makes an LR-invariant family read as one thick band and a
    sensitive one as a fan -- which is the qualitative difference between the two optimizers here.

    X IS THE FRACTION, not an absolute unit count, and that is safe ONLY because every cell in this
    figure shares one layout (same model, same `unit: nonresid`, same `exclude_params`) -- so a
    fraction names the same number of units in every line. It would NOT be safe across unit modes
    or models, where `frac_0.01` is 25,806 rows in one cell and 108 directions in another.
    """
    parsed = []
    unit = modal_unit(pattern)          # enforces the one-layout claim the docstring makes above
    for spec in specs:
        ev, metric, split = spec.split(":")
        is_loss = ev == "sft_loss"
        beh = None if is_loss else (ev, metric)
        rows = []
        for d in sorted(globmod.glob(pattern)):
            d = Path(d)
            if not ((d / "evals.json").exists() and (d / "config.yaml").exists()):
                continue
            mk = (yaml.safe_load((d / "config.yaml").read_text()).get("mask") or {})
            # `random` and the `ixg` family have no LR and no optimizer -- they are drawn as
            # reference LINES rather than arms, because putting them in an LR family would imply a
            # tuning axis they do not have. (A fitted mask's ranking depends on its LR; a
            # closed-form attribution's does not depend on anything you can sweep here.)
            if not mk or mk.get("scores") not in ("learned", "random", "ixg"):
                continue
            if mk.get("unit", "nonresid") != unit:
                continue
            try:
                xs, ys, _, _ = series(d, ev, metric, split, recovery=False)
            except SystemExit:
                continue
            if len(xs) < 2:
                continue
            sc = mk.get("scores")
            if sc == "random":
                who = "random"
            elif sc == "ixg":
                who = f"ixg:{mk.get('ixg_at', 'finetuned')}"
            else:
                who = mk.get("score_optimizer", "adam")
            cfg_all = yaml.safe_load((d / "config.yaml").read_text())
            ks = mk.get("k_schedule", "uniform")
            bt = budget_tag(cfg_all)
            rows.append((who, f"{ks} · {bt}" if bt else ks,
                         float(mk["score_lr"]), xs, ys, mk))
        if not rows:
            raise SystemExit(f"no runs for {spec}")
        parsed.append((rows, f"{NICE_SPLIT.get(split, split)} "
                             f"{'loss (nats)' if is_loss else metric}"))

    lrs = sorted({r[2] for rows, _ in parsed for r in rows if r[0] not in REF_LS}) or [1.0]
    lo, hi = math.log10(min(lrs)), math.log10(max(lrs))
    alpha_of = lambda lr: 0.30 + 0.70 * ((math.log10(lr) - lo) / (hi - lo) if hi > lo else 1.0)

    n = len(parsed); ncol = ncol or n; nrow = (n + ncol - 1) // ncol
    ROW_H = 2.0
    if SHORT:
        # ONE \textwidth FIGURE, not four subfigures. Same per-panel geometry as the separate
        # quarter-width PDFs (~1.3in each), so nothing about how a panel reads changes -- what
        # changes is that the row now shares one legend and one set of margins instead of
        # reserving a legend band on all four.
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.5, 1.80 * nrow), squeeze=False)
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol + 0.35, ROW_H * nrow + 0.25),
                                 squeeze=False)
    flat = [a for row in axes for a in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    for i, ((rows, title), ax) in enumerate(zip(parsed, flat)):
        for opt, ks, lr, xs, ys, mk in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            c = COLOR.get(opt, "#999999")
            lsty, mkr, hollow = _sty(ks)
            if opt in REF_LS:
                # behind the arms, thicker, no markers: these are references to read the LR
                # families against, not series with a tunable axis
                ax.plot(xs, ys, ls=REF_LS[opt], lw=1.5, color=c, alpha=0.95, zorder=1)
                continue
            ax.plot(xs, ys, ls=lsty, lw=0.8, color=c, alpha=alpha_of(lr), zorder=2)
            ax.plot(xs, ys, mkr, ms=2.6 if hollow else 2.2, ls="none", alpha=alpha_of(lr), zorder=3,
                    **({"mfc": "none", "mec": c, "mew": 0.8} if hollow
                       else {"color": c, "mec": "none"}))
        ax.set_xscale("log")
        ax.set_title(title, fontsize=FS_LABEL, pad=3)
        ax.tick_params(labelsize=FS_TICK, length=2, width=0.5)
        furnish(ax)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
        if i // ncol == nrow - 1:
            ax.set_xlabel("fraction of units kept", fontsize=FS_LABEL)
    axes[0][0].set_ylabel("metric", fontsize=FS_LABEL)

    pairs = sorted({(r[0], r[1]) for rows, _ in parsed for r in rows})
    handles = []
    for o, k in pairs:
        if o in REF_LS:
            continue
        lsty, mkr, hollow = _sty(k)
        handles.append(Line2D([0], [0], color=COLOR.get(o, "#999"), ls=lsty, lw=0.9, marker=mkr,
                              ms=3.0, **({"mfc": "none", "mec": COLOR.get(o, "#999"), "mew": 0.8}
                                         if hollow else {"mec": "none"}),
                              label=f"{'SGD' if o == 'sgd' else 'Adam'} ({k})"))
    for who in ("ixg:base", "ixg:mc", "ixg:finetuned", "random"):
        if any(o == who for o, _ in pairs):
            handles.append(Line2D([0], [0], color=COLOR[who], ls=REF_LS[who], lw=1.5,
                                  label=REF_LABEL[who]))
    handles.append(Line2D([0], [0], color="#555555", lw=0.9, alpha=0.30, label="low LR"))
    handles.append(Line2D([0], [0], color="#555555", lw=0.9, alpha=1.0, label="high LR"))
    # WRAPPED AND ON TOP, matching ../learning-to-attribute's neuron_recall.pdf. The previous
    # one-row-of-thirteen was 13 keys across a 9.7in figure at 5.8pt -- legible only because the
    # figure is wide, and it grew a key every time the sweep did. `top_legend` derives the strip
    # height from how many rows it actually wraps to, so adding an arm reflows instead of
    # overlapping the panel titles.
    if SHORT:
        fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.935))
        fig.legend(handles=handles, fontsize=FS_LEGEND, ncol=len(handles), frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, 1.005), handletextpad=0.35,
                   handlelength=1.5, columnspacing=1.2, borderpad=0.1)
    else:
        fig.tight_layout(w_pad=0.4)
        top_legend(fig, handles, nrow, ROW_H, ncol=4)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({n} panels, {sum(len(r) for r, _ in parsed) // n} lines each)")


def _draw(ax, data, better_is_low, refs=None):
    """Every arm onto one axes. THE single implementation -- `panel` and `grid_figure` both use it,
    so a single-panel figure and a panel of the grid cannot drift apart."""
    drawn = []
    for (opt, ks), pts in sorted(data.items()):
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        ls, mkr, hollow = _sty(ks)
        c, mk = COLOR.get(opt, "#999999"), mkr
        if len(pts) > 1:
            ax.plot(x, y, ls=ls, lw=0.9, color=c, zorder=2)
            # the ring marks the arm's BEST cell: min for a loss in nats, max for a rate or a
            # recovery fraction -- getting this backwards rings the WORST lr of every arm
            bx, by = (min if better_is_low else max)(zip(x, y), key=lambda p: p[1])
            # no ring when the best cell is an endpoint of the swept range: that is the largest (or
            # smallest) number tried, not a measured optimum
            if bx not in (x[0], x[-1]):
                ax.plot([bx], [by], "o", ms=7.5, mfc="none", mec=c, mew=0.9, zorder=4)
            else:
                print(f"  {opt}/{ks}: best at lr={bx:g}, an edge of the grid -- no ring",
                      file=sys.stderr)
        ax.plot(x, y, mk, ms=3.4, ls="none", zorder=3,
                **({"mfc": "none", "mec": c, "mew": 1.0} if hollow
                   else {"color": c, "mec": "#000000", "mew": 0.4}))
        if len(pts) == 1:
            print(f"  {opt}/{ks}: single LR ({x[0]:g}) -- drawn as a lone marker, no line",
                  file=sys.stderr)
        drawn.append((arm_label(opt, ks), c, ls, mk))
    for who, v in sorted((refs or {}).items()):
        ax.axhline(v, color=COLOR.get(who, "#999999"), ls=REF_LS.get(who, (0, (4, 2))),
                   lw=1.3, alpha=0.95, zorder=1)
    if SHORT and refs:
        # LABEL THE REFERENCES ON THE LINE, so the legend carries only the three swept arms. A
        # reference has no learning rate, so it is the one series whose key costs a legend row
        # while saying nothing the line's own position does not -- and at quarter width three
        # rows fit above the axes where five did not.
        #
        # Placement: hard against the left spine, where every panel's data sits highest or is
        # sparsest, and just ABOVE the line -- flipped below when the line is within a tenth of
        # the top, which is where the caps on the loss panels sit.
        y0, y1 = ax.get_ylim()
        # which END of the line is clear. Hugging the left spine is right until an arm's own
        # points move there -- extending Adam's sweep down a decade put a marker exactly under
        # the I x G label -- so pick the side whose nearest arm point is furthest away in y.
        # The x range comes from the POINTS, not from `ax.get_xlim()`: the log scale is not set
        # until below, so the axes still report the default linear limits and log10(0) throws.
        pts = [(math.log10(x), y) for ln in ax.get_lines()
               for x, y in zip(ln.get_xdata(), ln.get_ydata()) if x > 0]
        if not pts:
            pts = [(0.0, (y0 + y1) / 2)]
        x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
        if x1 == x0:
            x1 = x0 + 1.0
        for who, v in (refs or {}).items():
            high = (v - y0) > 0.88 * (y1 - y0)
            def clear(lo, hi):
                near = [abs(y - v) for x, y in pts if lo <= (x - x0) / (x1 - x0) <= hi]
                return min(near) if near else float("inf")
            left = clear(0.0, 0.3) >= clear(0.7, 1.0)
            ax.text(0.015 if left else 0.985,
                    v + (-1 if high else 1) * 0.015 * (y1 - y0),
                    (SHORT_REF if SHORT else REF_LABEL).get(who, who),
                    transform=ax.get_yaxis_transform(), fontsize=FS_LEGEND,
                    color=COLOR.get(who, "#999999"), ha="left" if left else "right",
                    va="top" if high else "bottom", zorder=5)
    ax.set_xscale("log")
    furnish(ax)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
    return drawn


def _levels(ax):
    """Every y value drawn on `ax` -- arm markers and reference lines alike.

    The legend has to clear the REFERENCES too, and they are `axhline`s in axes coordinates, so
    they never appear in `ax.dataLim`; sizing the reserve from the data limits alone is what put
    the frame on top of them.
    """
    ys = [y for ln in ax.get_lines() for y in ln.get_ydata()
          if isinstance(y, float) or hasattr(y, "__float__")]
    return [float(y) for y in ys] or [0.0, 1.0]


def panel(data, ylabel, out, legend, better_is_low, refs=None):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    drawn = _draw(ax, data, better_is_low, refs)
    ax.set_xlabel("score LR" if SHORT else "score learning rate", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if legend and drawn:
        handles = [Line2D([0], [0], color=c, ls=ls, lw=0.9, marker=mk, ms=3.2,
                          mec="#000000", mew=0.4, label=leg)
                   for leg, c, ls, mk in drawn if "·" not in leg]
        # THE REFERENCES GET KEYS TOO. They were legended in the grid view and not here, so a
        # single-panel figure drew up to four unexplained horizontal lines -- markerless and
        # in the same order the grid uses, so the two views read identically.
        if not SHORT:
            handles += [Line2D([0], [0], color=COLOR.get(who, "#999999"),
                               ls=REF_LS.get(who, (0, (4, 2))), lw=1.3,
                               label=REF_LABEL.get(who, who))
                        for who in ("ixg:base", "ixg:mc", "ixg:finetuned", "random")
                        if who in (refs or {})]
        # reserve a band rather than letting the frame float over the data, as the sibling does --
        # and put it on the side the data is NOT on. For a loss (lower better) the arms sit low, so
        # the room is above; for a rate (higher better) they sit high and the room is below. The
        # first version reserved the bottom unconditionally and the legend landed on the Adam arm.
        if SHORT:
            # AT QUARTER WIDTH THE LEGEND DOES NOT GO INSIDE. Five entries at 4.2pt stand ~0.46in
            # tall against a ~0.75in plot area, so any in-axes placement either sits on the data
            # or compresses it into the bottom third -- and the references are full-width
            # `axhline`s, so they cannot dodge a frame the way a marker can. It goes ABOVE the
            # axes instead, in two columns, on the SAME canvas as the legend-less panels:
            # tight_layout shrinks this panel's plot area to make room, every emitted PDF keeps
            # identical dimensions, and a row of four \includegraphics renders flush.
            # ONE COLUMN PER OPTIMIZER. matplotlib fills a multi-column legend top-to-bottom
            # before moving right, so grouping the handles by optimizer and padding the shorter
            # group to equal length puts Adam in the left column and SGD in the right one --
            # whatever each ends up holding once the SGD LR sweep lands. Grouping is on the
            # label's first word, which SHORT_ARM guarantees is the optimizer name.
            cols = [[h for h in handles if h.get_label().split()[0] == who]
                    for who in ("Adam", "SGD")]
            nrow = max(len(c) for c in cols)
            for c in cols:                       # pad with invisible entries to square the grid
                while len(c) < nrow:
                    c.append(Line2D([], [], ls="none", marker="none", label=" "))
            ax.legend(handles=[h for c in cols for h in c],
                      fontsize=FS_LEGEND, ncol=2, frameon=False,
                      loc="lower left", bbox_to_anchor=(-0.02, 1.0, 1.04, 0.14), mode="expand",
                      borderpad=0.1, handletextpad=0.35, handlelength=1.4, labelspacing=0.2,
                      columnspacing=0.6, borderaxespad=0.1)
        else:
            y0, y1 = ax.get_ylim()
            if better_is_low:
                ax.set_ylim(y0, y1 + 0.45 * (y1 - y0))
                loc = "upper right"
            else:
                ax.set_ylim(y0 - 0.45 * (y1 - y0), y1)
                loc = "lower right"
            leg = ax.legend(handles=handles,
                            fontsize=FS_LEGEND, loc=loc, frameon=True, framealpha=0.95,
                            borderpad=0.35, handletextpad=0.4, handlelength=2.0,
                            labelspacing=0.3)
            # the fixed reserve above is a guess; measure the drawn legend and grow the axis until
            # the band it occupies is empty. Iterated because each ylim change moves the data
            # transform under the legend, and capped so a too-tall legend fails visibly.
            for _ in range(6):
                fig.canvas.draw()
                box = leg.get_window_extent().transformed(ax.transData.inverted())
                lo, hi = sorted((box.y0, box.y1))
                levels = _levels(ax)
                y0, y1 = ax.get_ylim()
                gap = (max(levels) - lo) if better_is_low else (hi - min(levels))
                if gap <= 0.02 * (y1 - y0):
                    break
                ax.set_ylim(y0, y1 + gap * 1.05) if better_is_low \
                    else ax.set_ylim(y0 - gap * 1.05, y1)
    if SHORT:
        # FIXED MARGINS, NOT tight_layout, and the top band is reserved on EVERY panel whether or
        # not it carries the legend. tight_layout sizes each figure to its own contents, so the
        # legend-bearing panel would get a shorter plot area than its three neighbours and a row
        # of four would step down at the first one. Identical margins -> identical axes rectangles
        # -> the row reads as one figure, at the price of a little white space on three panels.
        fig.subplots_adjust(left=0.30, right=0.97, top=0.825, bottom=0.235)
    else:
        fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(drawn)} arms)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", required=True)
    p.add_argument("--prefix", default="plots/param_optimizer_lr")
    p.add_argument("--eval", default=None,
                   help="plot a behaviour rate instead of the loss, e.g. `language`")
    p.add_argument("--metric", default=None, help="with --eval, e.g. `target_frac`")
    p.add_argument("--recovery", action="store_true",
                   help="plot (pretrained-loss)/(pretrained-full) instead of nats; for CROSS-"
                        "organism figures, where nats are not commensurable")
    p.add_argument("--legend-both", action="store_true")
    p.add_argument("--no-legend", action="store_true",
                   help="suppress the legend on every emitted panel -- for a subfigure pair "
                        "whose other half already carries it (matplotlib's 'best' placement "
                        "can sit the box on data, as it does on the train-loss panel)")
    p.add_argument("--curves", action="store_true",
                   help="with --grid: plot the CURVES (x = mask fraction, y = metric, one line "
                        "per arm+lr) instead of their AUC summaries")
    p.add_argument("--grid", nargs="*", default=None, metavar="SPEC",
                   help="one FIGURE with a panel per SPEC instead of separate single panels. "
                        "SPEC is `eval:metric:split` (a rate) or `sft_loss:loss:split` (nats), "
                        "e.g. --grid language:target_frac:in_dist language:target_frac:off_target "
                        "sft_loss:loss:train sft_loss:loss:test")
    p.add_argument("--linear", action="store_true",
                   help="use the LINEAR trapezoid instead of the log one -- a diagnostic, "
                        "not an alternative; see the AUC note at the top of this module")
    p.add_argument("--eps", choices=("default", "all"), default="default",
                   help="`default` keeps only cells at the schema `score_eps`; the eps\n                        dimension has its own figure (epsgrid_*.pdf)")
    p.add_argument("--quarter", action="store_true",
                   help="quarter-width canvas + short labels, for a row of four 0.24\\textwidth "
                        "subfigures (see QUARTER)")
    p.add_argument("--refs", default=None,
                   help="comma-separated reference keys to draw as horizontal lines, e.g.\n                        `ixg:mc,ixg:base`. Default: every one the glob matched.")
    p.add_argument("--ext", default="pdf")
    args = p.parse_args()
    if args.quarter:
        global FIG_W, FIG_H, FS_LABEL, FS_TICK, FS_LEGEND, SHORT
        (FIG_W, FIG_H) = QUARTER["fig"]
        FS_LABEL, FS_TICK, FS_LEGEND = QUARTER["label"], QUARTER["tick"], QUARTER["legend"]
        SHORT = True
    if args.linear:
        global AUC, AUC_NAME
        AUC, AUC_NAME = linear_auc, "linear"

    # THE AXIS LABEL MUST STATE THE DIRECTION, and this is the second time it mattered here: a
    # first draft plotted a recovery fraction under the label "loss log-AUC", which reads as
    # lower-is-better and inverted the figure for anyone not reading the caption. Now the label
    # names the actual quantity AND carries an arrow, the sibling's convention
    # (plot_mattr_lr_metrics.py labels its metrics "acc-AUC $\\uparrow$").
    #
    # NOTE the arrows point OPPOSITE ways in the two modes: nats are lower-is-better, a recovery
    # fraction is higher-is-better, and `better_is_low` below makes the optimum ring follow.
    keep_refs = set(args.refs.split(",")) if args.refs else None
    if args.grid:
        plt.rcParams.update(RC)
        if args.curves:
            curve_grid(args.glob, args.grid, f"{args.prefix}_curves.{args.ext}")
        else:
            grid_figure(args.glob, args.grid, f"{args.prefix}_grid.{args.ext}",
                        recovery=args.recovery, eps_mode=args.eps, keep_refs=keep_refs)
        return

    behaviour = None
    if args.eval:
        if not args.metric:
            raise SystemExit("--metric is required with --eval")
        behaviour = (args.eval, args.metric)
    if behaviour:
        # a behaviour RATE is higher-is-better, and its two splits are in-dist / off-target rather
        # than train / held-out -- but the fitted/unfitted structure is the same, so the left panel
        # is still "the distribution the delta was fitted on"
        # SHORT labels: at 2.7in wide with 8pt text, "off-target language target_frac, log-AUC"
        # runs past the figure and matplotlib clips it silently. The eval/metric belong in the
        # caption; the axis only has to say which split and which direction.
        arrow = "$\\uparrow$"
        # QUARTER-WIDTH LABELS NAME THE SPLIT AND THE DIRECTION AND NOTHING ELSE. At 1.3in tall a
        # y label longer than ~14 characters overruns the panel, and the arrow -- the one part
        # that cannot move to the caption without inverting how the panel reads -- is what gets
        # pushed off the top.
        pairs = ((("in_dist", f"in-dist {arrow}", True),
                  ("off_target", f"off-target {arrow}", args.legend_both)) if args.quarter else
                 (("in_dist", f"in-dist {AUC_NAME}-AUC {arrow}", True),
                  ("off_target", f"off-target {AUC_NAME}-AUC {arrow}", args.legend_both)))
        suffix = {"in_dist": "indist", "off_target": "offtarget"}
    else:
        arrow = "$\\uparrow$" if args.recovery else "$\\downarrow$"
        what = ("loss recovery" if args.recovery
                else ("loss" if args.quarter else "loss (nats)"))
        pairs = ((("train", f"train loss {arrow}", True),
                  ("test", f"held-out loss {arrow}", args.legend_both)) if args.quarter else
                 (("train", f"train {what}, {AUC_NAME}-AUC {arrow}", True),
                  ("test", f"held-out {what}, {AUC_NAME}-AUC {arrow}", args.legend_both)))
        suffix = {"train": "trainloss", "test": "testloss"}
    for split, nice, legend in pairs:
        legend = legend and not args.no_legend
        data, refs = arms(args.glob, split, args.recovery, behaviour, eps_mode=args.eps,
                          keep_refs=keep_refs)
        if not data:
            raise SystemExit(f"no learned-score runs with sft_loss/{split} under {args.glob}")
        for (opt, ks), pts in sorted(data.items()):
            print(f"  {opt:<5} {ks:<9} {len(pts)} lr(s): "
                  + ", ".join(f"{lr:g}->{v:.3f}" for lr, v in pts))
        panel(data, nice, f"{args.prefix}_{suffix[split]}.{args.ext}", legend,
              better_is_low=(not args.recovery) and not behaviour, refs=refs)


if __name__ == "__main__":
    main()
