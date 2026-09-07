"""Safety and capability of the refusal anchors, and of the two ways of eroding refusal.

Seven cells, a 2x2 grid of panels: the TOP row is safety (StrongREJECT, SORRY-Bench; up = refusal
eroded), the BOTTOM row is capability (GSM8K, IFEval; down = damage). The bars are, left to right:

    Instruct / chat        the RLHF'd model in its own template -- the reference point
    Instruct / URIAL       the same WEIGHTS under URIAL's no-refusal prompt -- prices the prompt
    Base / chat            base weights, plain template (a base tokenizer has no template)
    Base / URIAL           base weights, no-refusal prompt -- the competence ceiling
      -- dashed rule: everything left of it trains nothing, everything right of it does --
    GRPO                   a LoRA over the instruct model, trained by GRPO against this same score
    GRPO + tuned KL        the same, with a KL penalty to the starting policy
    GRP-Oblit              the published trained baseline (Russinovich et al. 2026): full-parameter
                           GRPO with DAPO's positive-advantage loss, a KL term, a cosine schedule
                           and AdvBench prompts, against the StrongREJECT judge in place of their
                           GPT-4.1 rubric. Best of six arms.
    Abliteration           instruct weights with ONE refusal direction projected out of every
                           residual-writing matrix (Arditi et al. 2024, our implementation in
                           scripts/abliterate.py). The baseline this result is most directly read
                           against: the other small, training-free, capability-preserving edit to
                           the same weights. Shown at the direction their KL guard admits, not at
                           the most aggressive one -- see configs/baseline/abliteration_base.yaml.
    MAttr 1%               instruct weights with 1% of the base->instruct delta masked back in

The order is the order of reading: the four prompt-only anchors first, then the two GRPO arms,
then the method, so MAttr is the bar every other bar has already been read against by the time
the eye reaches it. The dashed rule is the one grouping the tick labels do not carry -- which
cells changed weights at all -- and it splits the bars exactly where CELLS switches from the
``Instruct``/``Base`` groups to the edited ones, so reordering CELLS moves it.

WHY THESE FOUR, and what the second column of each row adds to the first:

* **SORRY-Bench beside StrongREJECT** is the held-out judge. Every GRPO arm here optimised the
  StrongREJECT judge, so its StrongREJECT bar is partly the objective it trained on; SORRY-Bench's
  fine-tuned Mistral-7B judge has never been in any objective in this repo. The MAttr/GRPO ordering
  FLIPS between the two columns (GRPO ahead on StrongREJECT, well behind on SORRY-Bench), which is
  the point of showing both. Its judge is binary and more permissive, so its anchors sit higher
  everywhere -- read the two as different thresholds on one axis, not as disagreeing.
* **IFEval beside GSM8K** is the second capability probe that MOVES (GRP-Obliteration reports these
  two as the utility numbers sensitive to unalignment). GSM8K is multi-step arithmetic; IFEval is
  verifiable formatting instructions, and free GRPO fails it for a visible reason -- its responses
  run to ~780 words against the Instruct model's ~210, ignoring every start/end, punctuation and
  case constraint -- while the mask holds the anchor.

CAPABILITY IS GSM8K AND IFEVAL, not a mean with MMLU, because averaging with MMLU hides the effect:
MMLU sits at 43-44 in EVERY cell here (it is completion-format, so it neither sees a chat template
nor asks for multi-step generation), while GSM8K spans 30 down to 0.5. A mean of a flat probe and a
collapsing one halves the collapse and reports the damage as milder than it is. MMLU is still worth
running -- a cell where BOTH fell would be a different finding, and its flatness is what localises
this one to generation -- it just does not belong inside the same number.

MAttr is the method this repo is about and the rest of the figure is what it is read against, so
it is marked three ways rather than left to be found: its own hue (the other edited cell takes the
palette's reference grey), a bold tick label, and a dark outline on its bars. Everything else is
drawn identically, so the emphasis is typographic and changes no number. Each panel keeps its full
frame (all four spines at palette.SPINE_LW, as `plot_mib_accauc_cpr_scatter.pdf` does) rather
than the open top/right of the sibling bar figures: with a rule drawn INSIDE the panel, an open
frame reads as a fifth bar slot rather than as a partition.

Read the rows against each other rather than as four figures: everything here can raise the top
row, and what separates the interventions is what the bottom row does while it happens. The prompt-only cells
change no weights at all, the mask changes 1% of them, and free GRPO changes whatever it likes.

TWO THINGS THE PROBES MAKE TRUE, both of which belong in a caption:

* **MMLU is completion format**, so it uses no chat template and CANNOT differ between a model's
  native cell and its URIAL one -- same weights, same prompt. That is one reason it is not the
  probe plotted here. (Chat-format MMLU would read ~chance for a 1B model, so it is not an option
  either.)
* **An empty response scores SAFE**, so a low StrongREJECT bar with a broken capability bar is
  damage, not refusal. That is exactly what the vanilla GRPO cell is for, and why this figure has
  a capability panel at all.

Every cell is one weights-and-frame pair, decoded HF greedy with identical eval settings per
metric (`configs/baseline/anchor_base.yaml` and `configs/refusal/eval_native_v2.yaml` share the
StrongREJECT/GSM8K/MMLU block; `anchor_{ifeval,sorrybench}_base.yaml` and
`configs/refusal/eval_{ifeval,sorrybench}.yaml` share the other two), so no bar mixes decodes with
another. The two newer metrics live in their own run directories -- `<anchor>_ifeval/` and
`<run>/eval_ifeval/` -- because a resubmission would have overwritten the originals' evals.json,
which is why CELLS names three sweeps per bar rather than one.

WHY RAW MATPLOTLIB, and why it is styled like `plot_mib_test_avg.py` in the sibling repo: the two
panels need free y scales (a 0-1 score above a 0-45 percentage), the value labels have to be
offset in POINTS rather than data units for the same reason, and the group legend is three patches
rather than a mapped aesthetic. That is the same combination that figure documents, so this
follows its conventions -- palette.RC / palette.furnish for the furniture, the metric and its
direction in each panel's own axis title, values printed over the bars, and the numbers echoed to
stdout so the figure can be checked without opening it.

THE GRID SHARES ONE SET OF TICK LABELS PER COLUMN, which is the one place this departs from that
figure. Its panels hold DIFFERENT bars (node-level and edge-level methods), so each needs its own
labels; these four hold the SAME seven cells, so only the bottom row carries them and the height
goes to bars instead of to a second copy of the labels.

3.4 x 2.4in is `0.62\\textwidth` of iclr2026_conference.sty's 5.5in, placed 1:1, so the sizes
below are the sizes that reach the compiled PDF. It used to be 2.4 x 2.0 to pair with
`refusal_sparsity_facets.pdf` as two subfigures; a 2x2 grid of seven-bar panels is not legible at
that width, so the pairing is now a choice for the LaTeX (`--width`/`--height` are there to match
whatever it becomes). Do not draw it smaller and let LaTeX upscale it.

    uv run python plots/plot_baseline_strongreject.py
    uv run python plots/plot_baseline_strongreject.py --panels sr,cap --width 2.4 --height 2.0
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                     # noqa: E402

RUNS = Path(__file__).parent / "data" / "all_runs"

#: (bar label, weights group, condition, {metric family: sweep path under data/all_runs}).
#: ``native`` holds StrongREJECT + GSM8K + MMLU (one job, eval_native_v2.yaml's block); the other
#: two are the later single-metric jobs, each in its own directory -- see the docstring.
def _anchor(run, native="/evals.json"):
    """The three sweeps of a MEASUREMENT cell, which are three run directories by suffix.

    ``native`` is the odd one out only because the anchors were run before the two later metrics
    existed and kept the bare directory name, while abliteration -- added afterwards -- names all
    three the same way.
    """
    return {"native": f"{run}{native}",
            "sorrybench": f"{run}_sorrybench/evals.json",
            "ifeval": f"{run}_ifeval/evals.json"}


def _edited(run, native):
    return {"native": f"{run}/{native}",
            "sorrybench": f"{run}/eval_sorrybench/evals.json",
            "ifeval": f"{run}/eval_ifeval/evals.json"}


def _one(run, split_prefix=None):
    """All five metrics from ONE sweep, or from a native sweep plus two suffixed siblings.

    The GRP-Oblit re-evals were run after both later metrics existed, so a single `epochs: 0` job
    measured everything. The 8B anchor and abliteration cells predate that and keep the newer two
    in `<prefix>_ifeval/` and `<prefix>_sorrybench/`, which is what `split_prefix` names.
    """
    f = f"{run}/evals.json"
    if split_prefix is None:
        return {"native": f, "sorrybench": f, "ifeval": f}
    return {"native": f, "sorrybench": f"{split_prefix}_sorrybench/evals.json",
            "ifeval": f"{split_prefix}_ifeval/evals.json"}


#: The 8B figure, selected with ``--cells 8b``. FOUR bars rather than nine: the base-weight
#: anchors were never run at 8B (they price the PROMPT, which is a 1B question), and the two GRPO
#: controls are ours rather than published baselines. What is left is the comparison that matters
#: at scale -- the untouched model, the two training-free/localised edits, and the trained
#: baseline. NOTE THE MASK IS AT 5%, NOT 1%: the 8B sparsity curve reaches its knee there
#: (StrongREJECT 0.765 at GSM8K 81.5) where the 1B one reaches it at 1%, so a "1%" bar would
#: compare two different points on two different curves. The label says which.
CELLS_8B = [
    (r"Instruct$^{\mathrm{C}}$", "Instruct", "dense", _one("anchor8b_instruct_native",
                                                           "anchor8b_instruct")),
    ("GRP-Oblit",    "GRP-Oblit",    "dense", _one("grpoblit_8b_eval")),
    ("Abliteration", "Abliteration", "dense", _one("abliteration8b_native", "abliteration8b")),
    ("MAttr 5%",     "MAttr",        "frac_0.05",
     {"native": "refusal_grpo_8b_logk/eval_native/evals.json",
      "sorrybench": "refusal_grpo_8b_logk/eval_extra/evals.json",
      "ifeval": "refusal_grpo_8b_logk/eval_extra/evals.json"}),
]
#: 8B never got the prompt-only anchors, so nothing is left of the training rule to draw
UNTRAINED_8B = set()

CELLS = [
    # the superscript is the FRAME: C = the model's own chat (or plain) template, U = URIAL's
    # no-refusal prompt. mathtext, which palette.RC routes through Inter, so it is the same
    # face as the rest of the label and not a serif fallback.
    (r"Instruct$^{\mathrm{C}}$",  "Instruct", "dense", _anchor("anchor_instruct_native")),
    (r"Instruct$^{\mathrm{U}}$",  "Instruct", "dense", _anchor("anchor_instruct_urial_help")),
    (r"Base$^{\mathrm{C}}$",      "Base",     "dense", _anchor("anchor_base_plain")),
    (r"Base$^{\mathrm{U}}$",      "Base",     "dense", _anchor("anchor_base_urial_help")),
    # everything below this line changed weights -- the unregularised GRPO adapter and the mask
    # sweep's 1% condition, both instruct weights edited toward base by the two different
    # mechanisms this repo compares. The dashed rule in draw() is placed by the group switch here.
    ("GRPO",            "GRPO",     "dense",
     _edited("refusal_grpo_weights_lora", "eval_native/evals.json")),
    # "tuned" is literal and is why the bar is fair to the baseline rather than generous to us:
    # three coefficients were run (0.01 / 0.04 / 0.1) and this is the one that came out BEST on
    # capability, so the KL arm is shown at its best case and not at a coefficient picked blind
    ("GRPO+KL",         "GRPO",     "dense",
     _edited("refusal_grpo_weights_kl001", "eval_native/evals.json")),
    # THE PUBLISHED TRAINED BASELINE: GRP-Obliteration (Russinovich et al. 2026), reimplemented
    # with their DAPO loss, KL term, cosine schedule and AdvBench prompts, but with the
    # StrongREJECT judge substituted for their GPT-4.1 rubric so it is affordable -- see
    # configs/refusal/rl/grpoblit_base.yaml for what that substitution costs. SIX arms were run
    # ({AdvBench, StrongREJECT} prompts x {1e-6, 5e-6} x 100 steps, plus a 1e-5 x 300 pair) and
    # this is the best of them on safety, so the baseline is shown at its best case exactly as the
    # KL arm above is. Their paper tunes LR and beta per model family and publishes neither.
    ("GRP-Oblit",       "GRP-Oblit", "dense", _one("grpoblit_long_eval")),
    # the other training-free weight edit, and the baseline this result is most directly read
    # against: one refusal direction projected out of every residual-writing matrix. It sits
    # RIGHT of the training rule because it changes weights, even though it runs no optimizer --
    # the rule separates prompt-only cells from weight-edited ones, which is the distinction the
    # figure is about. A MEASUREMENT cell like the anchors (scripts/abliterate.py bakes the edit
    # into the weights, so `epochs: 0` evaluates it), hence _anchor rather than _edited.
    ("Abliteration",    "Abliteration", "dense", _anchor("abliteration", "_native/evals.json")),
    # last, so it is read after everything it is compared against (see the docstring)
    ("MAttr 1%",        "MAttr",    "frac_0.01",
     _edited("refusal_grpo_logk_v2", "posthoc_eval/evals.json")),
]
#: the groups whose cells train nothing; the dashed rule falls after the last of them
UNTRAINED = {"Instruct", "Base"}
#: group key -> (legend label, colour). palette.MODEL's hues, reused rather than invented, so the
#: weights read the same here as in every other refusal figure.
#: Labels are one word each because the legend is one row across 2.4in; what the groups MEAN is
#: already spelled out by the tick labels under them ("Instruct$^{\mathrm{C}}$" and so on).
#: HIGHLIGHT is the key of the emphasised group -- see the docstring.
GROUP = {"Instruct": ("Instruct", P.MODEL["Instruct"]),
         "Base": ("Base", P.MODEL["Base"]),
         "MAttr": ("MAttr", P.MODEL["MAttr"]),
         "Abliteration": ("Ablit.", P.MODEL["Abliteration"]),
         "GRP-Oblit": ("GRP-Oblit", P.MODEL["GRP-Oblit"]),
         "GRPO": ("GRPO", P.MODEL["GRPO"])}
#: (key, axis title). The direction arrow is part of the title because neither number is
#: self-evidently good or bad: a HIGH StrongREJECT means refusal broke.
#: Kept SHORT, and the direction arrows that were here are gone for the same reason: an axis title
#: has to fit INSIDE its panel's height, which at a 2.0in total is ~0.7in, and a longer one
#: overflows into its neighbour (or off the top edge, which is where "StrongREJECT ↑" went). Which
#: direction is bad belongs in the caption anyway, since neither is obvious: StrongREJECT RISING is
#: refusal breaking, and capability falling is damage.
#: Keyed so ``--panels`` can select a subset or add MMLU back as a third row. The default pair is
#: what the paper figure shows; ``--panels sr,cap,mmlu`` is the diagnostic view that makes the
#: "MMLU is flat everywhere" claim visible instead of asserted.
#: The grid is filled row-major, NCOLS wide: the default order puts the two safety metrics on the
#: top row and the two capability probes below them, each column pairing the trained-on metric
#: (left) with its held-out counterpart (right).
PANEL_TITLE = {"sr": "StrongREJECT", "sb": "SORRY-Bench", "cap": "GSM8K, %", "ife": "IFEval, %",
               "mmlu": "MMLU, %"}
PANELS = ["sr", "sb", "cap", "ife"]
NCOLS = 2
#: value-label format per metric: the two judge scores are fractions, the rest percentages
FMT = {"sr": "{:.2f}", "sb": "{:.2f}", "cap": "{:.1f}", "ife": "{:.1f}", "mmlu": "{:.1f}"}

HIGHLIGHT = "MAttr"

# 3.8 x 2.5in is `0.69\textwidth` of iclr2026_conference.sty's 5.5in, placed 1:1, so the font
# sizes below are the sizes that reach the compiled PDF. Do not draw it smaller and let LaTeX
# upscale it. It grew twice and each time for a countable reason rather than taste: 2.4 -> 3.4in
# when the grid went from two stacked panels to 2x2 (seven bars are not legible at half that),
# and 3.4 -> 3.8 when abliteration made it eight. `plots/refusal_sparsity_facets.pdf` was its
# side-by-side partner at a shared 2.4 x 2.0in and no longer matches; that pairing is now a
# choice for the LaTeX, which is what `--width`/`--height` are for.
FIG_W, FIG_H = 4.1, 2.5
# NO LEGEND, which is the one thing here that plot_mib_test_avg.py does and this does not. Its
# families (gradient / mask / ours / control) are a grouping the tick labels do not carry, so the
# patches are the only place they appear. These four are named by the tick labels themselves --
# every bar's label starts with its group -- so a legend would restate them, and at 2.0in total
# height the ~0.2in it costs comes straight out of the panels, where it was making the two axis
# titles collide. Colour is left as a grouping cue for a reader scanning the bars.
HEAD, FOOT = 0.06, 0.52
FS_AXIS, FS_TICK, FS_ANNOT = 6.5, 6, 5
BAR_W = 0.72


def sr_stderr(sweep: Path, cond: str):
    """SE of a condition's StrongREJECT mean, from the per-response scores beside the sweep."""
    p = sweep.parent / "strongreject_eval" / "generations.jsonl"
    if not p.exists():
        return float("nan")
    v = [r["score"] for r in map(json.loads, p.read_text().splitlines())
         if r.get("condition") == cond and r.get("split") == "off_target"
         and r.get("score") is not None]
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def _load(rel, cond):
    p = RUNS / rel
    if not p.exists():
        return None, p
    return json.loads(p.read_text())["final"][cond], p


#: which sweep each panel's number comes from, so a figure only demands the data it draws
PANEL_SOURCE = {"sr": "native", "cap": "native", "mmlu": "native",
                "sb": "sorrybench", "ife": "ifeval"}


def bars(panels=None):
    """``[(group, label, {panel: (value, se)})]`` in CELLS order. A missing sweep raises.

    ``panels`` restricts which sweeps are REQUIRED. Without it a run that has StrongREJECT and
    GSM8K but not yet IFEval cannot be drawn at all, even as a two-panel figure -- which is the
    normal state of a comparison whose last job is still queued.
    """
    need = ({PANEL_SOURCE[p] for p in panels if p in PANEL_SOURCE} if panels
            else {"native", "sorrybench", "ifeval"})
    out, missing = [], []
    for label, group, cond, src in CELLS:
        nat, np_ = _load(src["native"], cond) if "native" in need else (None, None)
        sb, _ = _load(src["sorrybench"], cond) if "sorrybench" in need else ({}, None)
        ife, _ = _load(src["ifeval"], cond) if "ifeval" in need else ({}, None)
        for name, r in (("native", nat), ("sorrybench", sb), ("ifeval", ife)):
            if r is None and name in need:
                missing.append(src[name])
        if any(r is None for n, r in (("native", nat), ("sorrybench", sb), ("ifeval", ife))
               if n in need):
            continue
        nat, sb, ife = nat or {}, sb or {}, ife or {}
        nan = (float("nan"), float("nan"))
        mm = nat.get("mmlu", {}).get("mmlu", {})
        gs = nat.get("gsm8k", {}).get("gsm8k", {})
        sbm = sb.get("sorrybench", {}).get("off_target", {})
        ifm = ife.get("ifeval", {}).get("ifeval", {})
        out.append((group, label, {
            "sr": (nat.get("strongreject", {}).get("off_target", {}).get("score", float("nan")),
                   sr_stderr(np_, cond) if np_ else float("nan")),
            # their compliance rate (mean of per-category means; the plain mean at 10/category)
            "sb": (sbm.get("fulfillment", float("nan")), sbm.get("stderr", float("nan"))),
            "cap": (gs.get("accuracy", float("nan")), gs.get("stderr", float("nan"))),
            # prompt-level strict: the number lm-eval-harness and the GRP-Oblit paper call IFEval
            "ife": (ifm.get("prompt_strict", float("nan")), ifm.get("stderr", float("nan"))),
            "mmlu": (mm.get("accuracy", float("nan")), mm.get("stderr", float("nan")))}))
    if missing:
        raise SystemExit("no sweep at: " + ", ".join(missing))
    return out


def draw(ax, recs, key, ylabel, *, last):
    """One metric's panel: the six cells in CELLS order, value printed over each bar."""
    xs = range(len(recs))
    vals = [r[2][key][0] for r in recs]
    ses = [r[2][key][1] for r in recs]
    hl = [g == HIGHLIGHT for g, _, _ in recs]
    ax.bar(xs, vals, width=BAR_W, color=[GROUP[g][1] for g, _, _ in recs],
           edgecolor=["#000000" if h else "none" for h in hl],
           linewidth=[0.7 if h else 0 for h in hl], zorder=2)
    ax.errorbar(xs, vals, yerr=ses, fmt="none", ecolor="#333333", elinewidth=0.5, capsize=0,
                zorder=3)
    fmt = FMT[key]
    for x, v, se in zip(xs, vals, ses):
        # offset points, not data units: the panels have different y ranges, so a constant in
        # data units sits flush on one panel's bars and floats above the other's
        ax.annotate(fmt.format(v), (x, v + (0 if math.isnan(se) else se)),
                    textcoords="offset points", xytext=(0, 1.5), ha="center", va="bottom",
                    fontsize=FS_ANNOT, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab, _ in recs] if last else [], fontsize=FS_TICK,
                       rotation=45, ha="right", rotation_mode="anchor")
    if last:
        for tick, h in zip(ax.get_xticklabels(), hl):
            if h:
                tick.set_fontweight("bold")
    ax.set_xlim(-0.5 - BAR_W / 4, len(recs) - 0.5 + BAR_W / 4)
    ax.set_ylim(0, max(v + (0 if math.isnan(s) else s) for v, s in zip(vals, ses)) * 1.28)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    # the training rule: between the last cell that changed no weights and the first that did.
    # Derived from the groups rather than a hardcoded index, so it follows CELLS. Drawn under
    # the bars (zorder 1) and in the spine colour, so it reads as furniture, not as data.
    n_untrained = sum(g in UNTRAINED for g, _, _ in recs)
    if 0 < n_untrained < len(recs):
        ax.axvline(n_untrained - 0.5, color="#000000", lw=P.SPINE_LW, ls=(0, (3, 2)), zorder=1)
    P.furnish(ax)                       # grid + all four spines, the reference figure's frame
    ax.grid(False, axis="x")            # vertical rules behind bars are pure noise
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "baseline_strongreject.pdf"))
    ap.add_argument("--panels", default=",".join(PANELS),
                    help=f"comma-separated, from {sorted(PANEL_TITLE)}, filled row-major "
                         f"{NCOLS} wide (default: {','.join(PANELS)})")
    ap.add_argument("--ncols", type=int, default=NCOLS)
    ap.add_argument("--width", type=float, default=FIG_W, help="inches")
    ap.add_argument("--height", type=float, default=FIG_H, help="inches")
    ap.add_argument("--cells", default="1b", choices=["1b", "8b"],
                    help="which model's comparison to draw (default: 1b)")
    a = ap.parse_args()
    panels = a.panels.split(",")
    if bad := [k for k in panels if k not in PANEL_TITLE]:
        raise SystemExit(f"unknown panel(s) {bad}; choose from {sorted(PANEL_TITLE)}")
    ncols = min(a.ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)

    global CELLS, UNTRAINED
    if a.cells == "8b":
        CELLS, UNTRAINED = CELLS_8B, UNTRAINED_8B
    recs = bars(panels)
    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(nrows, ncols, figsize=(a.width, a.height), squeeze=False)
    for i, key in enumerate(panels):
        r, c = divmod(i, ncols)
        # tick labels on the last panel of each column only -- the columns share the seven cells
        last = i + ncols >= len(panels)
        draw(axes[r, c], recs, key, PANEL_TITLE[key], last=last)
    for i in range(len(panels), nrows * ncols):
        axes[divmod(i, ncols)].set_visible(False)

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.6)
    fig.subplots_adjust(top=1.0 - HEAD / a.height)
    fig.savefig(a.out)
    print("wrote", a.out)

    keys = [k for k in ("sr", "sb", "cap", "ife", "mmlu")]
    print(f"\n{'cell':<16}" + "".join(f"{PANEL_TITLE[k]:>16}" for k in keys))
    for _, lab, m in recs:
        print(f"  {lab:<14}" + "".join(
            f"{FMT[k].format(m[k][0]):>9} ±{FMT[k].format(m[k][1]):<5}" for k in keys))


if __name__ == "__main__":
    sys.exit(main())
