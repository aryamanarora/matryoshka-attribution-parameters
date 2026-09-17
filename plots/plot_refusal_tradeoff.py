r"""Capability against harm, one panel, both reported MAttr sweeps.

REPLACES the old `refusal_sparsity_facets.pdf` (a two-panel sparsity sweep of one 1B run) and keeps
its filename, so the LaTeX include does not move. What changed is the question. The sweep figure
plotted each metric against k, which answers "how does this metric move with sparsity" and leaves
the reader holding two curves in their head to see the trade. Here the trade IS the figure: x is
StrongREJECT, y is GSM8K, and a mask sweep is a PATH through that plane as k grows. Right and level
is the goal -- refusal eroded, arithmetic untouched -- so the claim is a SHAPE, and the shape is an
L on its side: both curves run right at constant height, then fall.

ONE PANEL, BOTH SCALES, ABSOLUTE UNITS. The two models sit in different bands (1B around 30 GSM8K,
8B around 80), which separates them without a facet and keeps y a real accuracy rather than a ratio
to each model's own anchor. Nothing is compared BETWEEN the bands -- each curve is read against its
own dotted anchor -- so the vertical gap carries no meaning beyond "8B is better at arithmetic".

SCALE TAKES THE COLOUR. The repo's usual rule is that colour is the method and a dash is the knob,
which does not decide this figure: both curves ARE the method, and the only thing to tell apart is
which model they ran on. Two hues separated by lightness (Wong bluish green, Tol indigo) do that at
a glance where two dashes of one colour needed the reader to find the legend. The level a curve is
read against is its own 0% point, which is the Instruct model; there is no separate anchor line.
The two models are named by a LEGEND ABOVE THE PANEL, horizontal, frameless: two keys cost one
line of height and nothing inside the axes, where any in-panel position had to be re-checked every
time a curve or a label moved (it collided with the 1B 0% label once already).

NO POINT BASELINES. Abliteration and GRP-Oblit are single points here, and the table prices them
properly against \|\Delta\theta\|_0; on these axes they added four diamonds and four hand-placed
labels to a figure whose whole argument is the shape of two curves. EXPECTED GRADIENTS IS A PATH,
so it is drawn -- dashed, in each model's hue, no k labels (see ``BASELINES``): the same delta and
budget grid under the closed-form ranking, which is the comparison the table makes row by row.

THE REPORTED POINT IS RINGED -- 1B at 2%, 8B at 1%, the cells the table quotes. Every other point is
a k the sweep measured and the table does not, drawn so the choice is visible rather than asserted.
THREE SPARSITIES PER CURVE CARRY AN ATTACHED LABEL -- 0.1%, 1%, 10%, drawn beside an enlarged
marker in the curve's own colour, the convention `accauc_vs_cpr.pdf` uses for its method points.
Labelling all eleven would be clutter in a 2in panel and a shape legend was worse: it put the key
for "what k is this point" in a different part of the figure from the points, so reading one
position cost a saccade each way. Three decades attached in place is enough to see that the flat
stretch of each curve spans two of them. Each label is tied to its marker by a LEADER LINE and sits on a white box, which is what
`plot_mib_accauc_cpr_scatter.place_labels` does for the same reason: a label far enough from its
point to be legible is far enough to be ambiguous about which point it belongs to, and a label near
enough to be unambiguous sits on the curve. That figure solves the placement with a repel solver
because it has twenty labels to fit; six hand-placed offsets do here, and the leaders make the
association explicit either way. The offsets differ per point because the two curves meet their
labels from different directions and the 8B 1% marker sits under the ring.

    uv run python plots/plot_refusal_tradeoff.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402
import plot_baseline_strongreject as B                 # noqa: E402

RUNS = Path(__file__).parent / "data" / "all_runs"

#: (scale, run, reported condition, colour, offset of the direct label in points). The hues are
#: upstream's: Wong bluish green (this repo's MAttr colour everywhere) and Tol indigo, which
#: separate by lightness as well as hue and so survive both small markers and CVD.
SWEEPS = [
    ("1B", "refusal_grpo_uniform_vllm_native", "frac_0.02", P._up.METHOD["Node Pruning"]),
    ("8B", "refusal_grpo_8b_uniform_vllm_native", "frac_0.01", P.MODEL["MAttr"]),
]
#: THE CLOSED-FORM RANKINGS OF THE SAME DELTA, as paths on the same plane (2026-09-16): reward-IxG
#: over the identical base->instruct delta and StrongREJECT reward -- `mc` is stepless IG along the
#: dense path (Expected Gradients), `base` the alpha=0 endpoint (IxG) -- so the three paths at one
#: scale differ in the RANKING alone. THE PAPER FIGURE CARRIES EG (2026-09-16, superseding the
#: same-day decision to keep the MAttr paths alone): it is the row the table prices beside
#: \ourmethod{} at both scales, and its path shows WHERE the closed-form ranking pays -- the
#: 8B dashed curve leaves the Instruct level at the sparsity the solid one is still on it. IxG
#: stays behind ``--baselines`` (dotted), as the table dropped its row. Colour stays the MODEL (the
#: channel the figure already spends) and the LINETYPE carries the method: solid MAttr, dashed EG,
#: dotted IxG -- the repo's usual "colour is the thing being compared, dash is the variant" split,
#: read the other way round because here the model is what the reader must never confuse and the
#: method is what the legend can carry. (scale, run, linestyle, legend label)
EG_LS, IXG_LS = (0, (4, 2)), (0, (1, 1.5))
BASELINES = [
    ("1B", "refusal_ixg_mc_vllm_native", EG_LS, "EG"),
    ("1B", "refusal_ixg_base_vllm_native", IXG_LS, r"I$\times$G"),
    ("8B", "refusal_ixg_mc_8b_vllm_native", EG_LS, "EG"),
    ("8B", "refusal_ixg_base_8b_vllm_native", IXG_LS, r"I$\times$G"),
]
SCALE_COLOR = {"1B": P._up.METHOD["Node Pruning"], "8B": P.MODEL["MAttr"]}
#: ``--native``: both native-frame sweeps at both scales, one panel. The reported figure shows the
#: uniform-$k$ pair because those are the cells the table quotes; this adds their log-$k$ twins, so
#: the schedule's effect is visible as the distance between two curves of one colour rather than as
#: a sentence. Colour stays the model and the dash becomes the schedule -- the same two channels,
#: one more level each. The +KL and positive-advantage arms are NOT here: they are a different
#: objective, they exist only at 1B, and one of them (A>0) never gets past SR 50, so including
#: them would spend half the panel's range on a curve that is a separate finding.
#: (scale, run, reported condition or None, colour, linestyle)
SWEEPS_NATIVE = [
    ("1B", "refusal_grpo_uniform_vllm_native", "frac_0.02", P._up.METHOD["Node Pruning"], "solid"),
    ("1B", "refusal_grpo_logk_vllm_native", None, P._up.METHOD["Node Pruning"], (0, (4, 2))),
    ("8B", "refusal_grpo_8b_uniform_vllm_native", "frac_0.01", P.MODEL["MAttr"], "solid"),
    ("8B", "refusal_grpo_8b_logk_vllm_native", None, P.MODEL["MAttr"], (0, (4, 2))),
]
#: k -> printed label, for the three sparsities that get one. Everything else is a plain dot.
KLABEL = {0.0: "0%", 0.001: "0.1%", 0.01: "1%", 0.1: "10%"}
#: (scale, k) -> (dx, dy, ha, va) in points, hand-placed: the curves approach these points from
#: different directions, and the 8B 1% marker sits under the ring that marks the reported cell.
KOFF = {("1B", 0.0): (0, 13, "center", "bottom"), ("8B", 0.0): (0, -14, "center", "top"),
        ("1B", 0.001): (0, 13, "center", "bottom"),
        # 1B's 1% and 8B's 1% / 10% go UP (2026-09-16): below each of them is now the EG path
        ("1B", 0.01): (-3, 12, "center", "bottom"),
        ("1B", 0.1): (15, 2, "left", "center"),
        ("8B", 0.001): (-2, -14, "center", "top"), ("8B", 0.01): (-7, 7, "center", "bottom"),
        ("8B", 0.1): (3, 7, "center", "bottom")}
#: leader line, matching plot_mib_accauc_cpr_scatter's: thin, grey, under the markers
LEADER = dict(arrowstyle="-", lw=0.35, color="#888888", shrinkA=0.5, shrinkB=2.0)
FS_AXIS, FS_TICK, FS_ANNOT = 6.5, 6, 5


def curve(run):
    """[(frac, gsm8k, sr, cond)] over the sweep, sparsest first, STARTING at k = 0.

    ``pretrained`` is the Instruct model with no units restored, i.e. the k = 0 end of the same
    sweep, so it is a point on the path rather than a separate reference.
    """
    fin = json.loads((RUNS / run / "eval_native/evals.json").read_text())["final"]
    pts = []
    for cond, r in fin.items():
        sr = r.get("strongreject", {}).get("off_target", {}).get("score")
        gs = r.get("gsm8k", {}).get("gsm8k", {}).get("accuracy")
        if sr is None or gs is None:
            continue
        if cond == "pretrained":
            pts.append((0.0, gs, sr * 100, cond))
        elif cond.startswith("frac_") and cond != "frac_1":   # frac_1 duplicates full_delta
            pts.append((float(cond[len("frac_"):]), gs, sr * 100, cond))
    return sorted(pts)


def draw_native(a, recs):
    """The four native-frame sweeps on one panel: colour is the model, dash the k-schedule."""
    fig, ax = plt.subplots(figsize=(a.width * 1.35, a.height * 1.1))
    seen = set()
    for scale, run, reported, col, ls in SWEEPS_NATIVE:
        pts = curve(run)
        xs, ys = [p[2] for p in pts], [p[1] for p in pts]
        if scale not in seen:                       # one anchor per model, not per curve
            anchor = recs[scale]["instruct"][2]["cap"][0]
            ax.axhline(anchor, lw=0.5, ls=(0, (1, 2)), color=col, alpha=0.8, zorder=0)
            ax.annotate(f"{scale} Instruct", xy=(0.995, anchor),
                        xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(0, 2), ha="right", va="bottom", fontsize=FS_ANNOT, color=col,
                        zorder=1)
            seen.add(scale)
        ax.plot(xs, ys, lw=0.9, ls=ls, color=col, zorder=2)
        ax.plot(xs, ys, "o", ms=2.4, mfc="white", mew=0.7, mec=col, zorder=3)
        if reported:
            rep = next(p for p in pts if p[3] == reported)
            ax.plot([rep[2]], [rep[1]], "o", ms=5.5, mfc="none", mec="#000000", mew=0.8, zorder=5)
    ax.set_xlabel("StrongREJECT", fontsize=FS_AXIS)
    ax.set_ylabel("GSM8K", fontsize=FS_AXIS)
    ax.margins(x=0.06, y=0.10)
    P.furnish(ax)
    ax.tick_params(labelsize=FS_TICK)
    handles = [Line2D([], [], color=c, lw=1.2, marker="o", ms=2.4, mfc="white", mew=0.7, label=s_)
               for s_, c in (("1B", P._up.METHOD["Node Pruning"]), ("8B", P.MODEL["MAttr"]))]
    handles += [Line2D([], [], color="#555555", lw=1.2, ls=v, label=f"$k$ {k}")
                for k, v in (("uniform", "solid"), ("log", (0, (4, 2))))]
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.91))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize=FS_ANNOT, handlelength=1.6, handletextpad=0.4,
               columnspacing=1.2)
    out = a.out.replace(".pdf", "_native.pdf")
    fig.savefig(out)
    if a.png:
        fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("wrote", out)
    for scale, run, reported, _, _ in SWEEPS_NATIVE:
        pts = curve(run)
        r = f", reported {reported[5:]}" if reported else ""
        print(f"  {scale} {run:38s} SR {min(p[2] for p in pts):5.1f}-{max(p[2] for p in pts):5.1f}"
              f"  GSM8K {min(p[1] for p in pts):5.1f}-{max(p[1] for p in pts):5.1f}{r}")
    return 0


def draw_all(a, recs):
    """Every MAttr sweep, one panel per scale.

    The reported figure draws two paths and can afford per-point labels and a ring; this is the
    diagnostic view, so it drops both and spends the space on eleven curves. The run list, the
    colour (what the reward and loss WERE) and the linetype (the k-schedule) are imported from
    ``plot_refusal_sparsity_all`` rather than restated -- the two figures are the same experiment
    read on two different axis pairs, and one of them silently listing a different set of runs is
    exactly the failure this import prevents.
    """
    import plot_refusal_sparsity_all as S
    fig, axes = plt.subplots(1, 2, figsize=(a.width * 2.3, a.height * 1.15))
    for ax, scale in zip(axes, S.MODELS):
        for run, model, frame, sched, _sampler in S.CELLS:
            if model != scale:
                continue
            try:
                pts = curve(run)
            except SystemExit:
                continue
            ax.plot([p[2] for p in pts], [p[1] for p in pts], lw=0.9, ls=S.SCHED_LS[sched],
                    color=S.FRAME_COLOR[frame], zorder=2)
            ax.plot([p[2] for p in pts], [p[1] for p in pts], "o", ms=1.7, mfc="white", mew=0.6,
                    mec=S.FRAME_COLOR[frame], zorder=3)
        anchor = recs[scale]["instruct"][2]["cap"][0]
        ax.axhline(anchor, lw=0.5, ls=(0, (1, 2)), color="#888888", zorder=0)
        ax.annotate("Instruct", xy=(0.995, anchor), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 2), ha="right", va="bottom",
                    fontsize=FS_ANNOT, color="#777777", zorder=1)
        ax.set_title(f"Llama-3.{'2-1B' if scale == '1B' else '1-8B'}-Instruct", fontsize=FS_AXIS,
                     pad=2)
        ax.set_xlabel("StrongREJECT", fontsize=FS_AXIS)
        ax.margins(x=0.06, y=0.10)
        P.furnish(ax)
        ax.tick_params(labelsize=FS_TICK)
    axes[0].set_ylabel("GSM8K", fontsize=FS_AXIS)
    handles = [Line2D([], [], color=c, lw=1.2, label=k) for k, c in S.FRAME_COLOR.items()]
    handles += [Line2D([], [], color="#555555", lw=1.2, ls=v, label=f"$k$ {k}")
                for k, v in S.SCHED_LS.items()]
    # tight_layout FIRST, then the legend into the reserved strip: with the legend added before
    # the layout pass its bbox sat outside the saved canvas and the figure shipped without it
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=FS_ANNOT, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.2)
    out = a.out.replace(".pdf", "_all.pdf") if "_all" not in a.out else a.out
    fig.savefig(out)
    if a.png:
        fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("wrote", out)
    for run, model, frame, sched, _ in S.CELLS:
        try:
            pts = curve(run)
        except SystemExit:
            print(f"  {run}: no native sweep, skipped")
            continue
        print(f"  {model} {frame:9s} {sched:7s} {run:38s} "
              f"SR {min(p[2] for p in pts):5.1f}-{max(p[2] for p in pts):5.1f}  "
              f"GSM8K {min(p[1] for p in pts):5.1f}-{max(p[1] for p in pts):5.1f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "refusal_sparsity_facets.pdf"))
    # 2.09in IS 0.38\textwidth of iclr2027_conference.sty's 5.5in, so the figure is placed 1:1
    # and the sizes below are the sizes that reach the compiled PDF. Drawing it wider and letting
    # \includegraphics shrink it would put the 5pt point labels on the page at ~4.3pt.
    ap.add_argument("--width", type=float, default=2.09)
    ap.add_argument("--height", type=float, default=1.95)
    ap.add_argument("--native", action="store_true",
                    help="both native-frame sweeps at both scales, one panel (4 curves)")
    ap.add_argument("--all", action="store_true",
                    help="every MAttr sweep rather than the two reported cells, faceted by scale")
    ap.add_argument("--baselines", action="store_true",
                    help="also draw the IxG closed-form path (dotted) beside the EG one the paper "
                         "figure already carries; writes *_baselines.pdf, never the paper figure")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    if a.baselines:
        a.out = a.out.replace(".pdf", "_baselines.pdf")

    recs = {"1B": {r[4]: r for r in B.bars()}, "8B": {r[4]: r for r in B.bars(cells=B.CELLS_8B)}}
    plt.rcParams.update(P.RC)

    if a.all:
        return draw_all(a, recs)
    if a.native:
        return draw_native(a, recs)

    fig, ax = plt.subplots(figsize=(a.width, a.height))
    for scale, run, reported, col in SWEEPS:
        pts = curve(run)
        xs, ys = [p[2] for p in pts], [p[1] for p in pts]
        # no Instruct hline (dropped 2026-09-16): the curve's own 0% point IS the Instruct model,
        # and with the EG path on the panel a dotted level per model was a third line style
        ax.plot(xs, ys, lw=0.9, color=col, zorder=2)
        ax.plot(xs, ys, "o", ms=2.6, mfc="white", mew=0.8, mec=col, zorder=3)
        for frac, gs, sr, _ in pts:
            if frac in KLABEL:
                dx, dy, ha, va = KOFF[(scale, frac)]
                ax.annotate(KLABEL[frac], (sr, gs), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, fontsize=FS_ANNOT, color=col, zorder=5,
                            arrowprops=dict(**LEADER),
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                      alpha=0.75))
        rep = next(p for p in pts if p[3] == reported)
        ax.plot([rep[2]], [rep[1]], "o", ms=5.5, mfc="none", mec="#000000", mew=0.8, zorder=5)
    handles = [Line2D([], [], color=c, lw=1.2, marker="o", ms=2.6, mfc="white", mew=0.8, label=sc)
               for sc, c in SCALE_COLOR.items()]
    # the closed-form paths, dashed/dotted in the model's hue and without k labels (the solid
    # curve's labels already say where each decade sits, and both paths share its k grid)
    for scale, run, ls, lab in BASELINES:
        if lab != "EG" and not a.baselines:
            continue
        pts = curve(run)
        xs, ys = [p[2] for p in pts], [p[1] for p in pts]
        col = SCALE_COLOR[scale]
        ax.plot(xs, ys, lw=0.8, ls=ls, color=col, zorder=1.5)
        ax.plot(xs, ys, "o", ms=2.2, mfc="white", mew=0.7, mec=col, zorder=2.5)
    methods = [("MAttr", "solid"), ("EG", EG_LS)] + ([(r"I$\times$G", IXG_LS)] if a.baselines else [])
    handles += [Line2D([], [], color="#555555", lw=1.2, ls=ls, label=lab) for lab, ls in methods]
    ax.legend(handles=handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False,
              fontsize=FS_ANNOT, handlelength=1.4, handletextpad=0.4, borderpad=0.0,
              columnspacing=0.9, borderaxespad=0.15)
    ax.set_xlabel("StrongREJECT", fontsize=FS_AXIS)
    ax.set_ylabel("GSM8K", fontsize=FS_AXIS)
    ax.margins(x=0.08, y=0.16)      # y: room for the 8B 1% / 10% labels above the top curve
    P.furnish(ax)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    if a.png:
        fig.savefig(a.out.replace(".pdf", ".png"), dpi=300)

    print("wrote", a.out)
    for scale, run, reported, _ in SWEEPS:
        pts = curve(run)
        rep = next(p for p in pts if p[3] == reported)
        print(f"  {scale}: reported {reported[5:]} at SR {rep[2]:.1f}, GSM8K {rep[1]:.1f}; "
              f"curve SR {min(p[2] for p in pts):.1f}-{max(p[2] for p in pts):.1f}, "
              f"GSM8K {min(p[1] for p in pts):.1f}-{max(p[1] for p in pts):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
