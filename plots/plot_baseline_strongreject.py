"""Safety and capability of the refusal anchors, and of the two ways of eroding refusal.

Seven cells in FOUR SECTIONS, on a 2x2 grid of panels: the TOP row is safety (StrongREJECT,
SORRY-Bench; up = refusal eroded), the BOTTOM row is capability (GSM8K, IFEval; down = damage).
Dashed rules separate the sections, which are the argument in reading order:

    Instruct / chat        the RLHF'd model in its own template -- the refusal floor
    -- rule --
    GRPO                   a LoRA over the instruct model, trained by GRPO against this same score
    GRPO + tuned KL        the same, with a KL penalty to the starting policy. "Tuned" is literal
                           and is what makes the bar fair rather than generous to us: three
                           coefficients were run (0.01 / 0.04 / 0.1) and this is the one that came
                           out BEST on capability.
    -- rule --
    Abliteration           instruct weights with ONE refusal direction projected out of every
                           residual-writing matrix (Arditi et al. 2024, our implementation in
                           scripts/refusal/abliterate.py). The baseline this result is most directly read
                           against: the other small, training-free, capability-preserving edit to
                           the same weights. Shown at the direction their KL guard admits, not at
                           the most aggressive one -- see configs/baseline/abliteration_base.yaml.
    GRP-Oblit              the published trained baseline (Russinovich et al. 2026): full-parameter
                           GRPO with DAPO's positive-advantage loss, a KL term, a cosine schedule
                           and AdvBench prompts, against the StrongREJECT judge in place of their
                           GPT-4.1 rubric. Best of six arms, on the same principle as the KL arm.
    MAttr 1%               instruct weights with 1% of the base->instruct delta masked back in
    -- rule --
    Base / URIAL           base weights under URIAL's no-refusal prompt -- the compliance this
                           architecture's competence allows, i.e. what a safety bar is read
                           against instead of 1.0

The middle two sections are the comparison: erode refusal by OPTIMISING the model (section 2) or by
a localised EDIT to it (section 3), with the two anchors that bound both on either end. MAttr is
last within its section so it is read after everything it is compared against.

The sections are keyed per cell rather than derived from the groups, so reordering CELLS moves the
rules with it. ``--cells 1b_full`` restores the nine-bar ladder that also measures each weight-set
in the OTHER frame (Instruct under URIAL, Base under its plain template); those two bars price the
PROMPT rather than the weights, which is a different question and is where the "Base's plain 0.033
is incoherence, not refusal" reading comes from.

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
it is marked three ways rather than left to be found: its own hue (the other edited cells take the
palette's competitor hues, the GRPO arms the reference grey), a bold tick label, and a dark outline
on its bars. Everything else is drawn identically, so the emphasis is typographic and changes no
number. Each panel keeps its full frame (all four spines at palette.SPINE_LW, as
`plot_mib_accauc_cpr_scatter.pdf` does) rather than the open top/right of the sibling bar figures:
with rules drawn INSIDE the panel, an open frame reads as an extra bar slot rather than as a
partition.

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
from matplotlib.lines import Line2D

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
                                                           "anchor8b_instruct"), "start",
     "instruct"),
    ("Abliteration", "Abliteration", "dense", _one("abliteration8b_native", "abliteration8b"),
     "edit", "ablit"),
    ("GRP-Oblit",    "GRP-Oblit",    "dense", _one("grpoblit_8b_eval"), "edit", "oblit"),
    ("MAttr 5%",     "MAttr",        "frac_0.05",
     {"native": "refusal_grpo_8b_logk/eval_native/evals.json",
      "sorrybench": "refusal_grpo_8b_logk/eval_extra/evals.json",
      "ifeval": "refusal_grpo_8b_logk/eval_extra/evals.json"}, "edit", "mattr"),
    # BOTH sparsities of the 8B native fit, because they are the two conditions its IFEval and
    # SORRY-Bench job covers and the pair is the result: 0.1% already erodes refusal by half with
    # every capability at the anchor, 0.5% takes it past every baseline. The second has no 1B
    # counterpart key, so it appears in the TABLE and not in the paired figure.
    ("MAttr (native)", "MAttr", "frac_0.001",
     {"native": "refusal_grpo_8b_logk_vllm_native/eval_native/evals.json",
      "sorrybench": "refusal_grpo_8b_logk_vllm_native/eval_extra/evals.json",
      "ifeval": "refusal_grpo_8b_logk_vllm_native/eval_extra/evals.json"}, "edit",
     "mattr_native_lo"),
    # THE REPORTED 8B CELL, at 1% rather than the 5% it was first written at. 5% has the bigger
    # harm numbers (94.0 / 100.0 against 84.0 / 99.3) but pays for them on IFEval: 67.8 against a
    # 74.7 anchor is z = -2.5, the largest capability deficit in any mask row, and it survives Holm
    # only because the corrected threshold is 1.9e-3. At 1% that column is -3.0 and comfortably
    # inside noise, both harm judges still beat both baselines (abliteration 68.6/96.4, GRP-Oblit
    # 71.5/92.9), and the sparsity claim is 28x rather than 5x against abliteration's L0. A bigger
    # number bought with the one cell a referee would pull on is a bad trade.
    ("MAttr (native)", "MAttr", "frac_0.01",
     {"native": "refusal_grpo_8b_uniform_vllm_native/eval_native/evals.json",
      "sorrybench": "refusal_grpo_8b_uniform_vllm_native/eval_extra/evals.json",
      "ifeval": "refusal_grpo_8b_uniform_vllm_native/eval_extra/evals.json"}, "edit",
     "mattr_nat_unif8"),
    ("MAttr (native)", "MAttr", "frac_0.02",
     {"native": "refusal_grpo_8b_logk_vllm_native/eval_native/evals.json",
      "sorrybench": "refusal_grpo_8b_logk_vllm_native/eval_extra/evals.json",
      "ifeval": "refusal_grpo_8b_logk_vllm_native/eval_extra/evals.json"}, "edit",
     "mattr_native_hi"),
    # the 8B harm CEILING, in at last (2026-09-13): base weights under URIAL's no-refusal prompt.
    # Its 1B twin has anchored that column since the start; until this ran, the 8B block had no
    # upper reference and its mask rows' 94-100 could only be read against the baselines.
    (r"Base$^{\mathrm{U}}$", "Base", "dense", _anchor("anchor8b_base_urial_help"), "ceiling",
     "base_urial"),
    ("MAttr (native)", "MAttr", "frac_0.005",
     {"native": "refusal_grpo_8b_logk_vllm_native/eval_native/evals.json",
      "sorrybench": "refusal_grpo_8b_logk_vllm_native/eval_extra/evals.json",
      "ifeval": "refusal_grpo_8b_logk_vllm_native/eval_extra/evals.json"}, "edit",
     "mattr_native"),
]

#: THE FULL LADDER, nine bars, selected with ``--cells 1b_full``. It is the version that makes the
#: prompt's contribution visible -- Instruct and Base each measured in BOTH frames -- and it is
#: where the two readings the docstring quotes come from: that Base's plain-template 0.033 measures
#: incoherence rather than refusal (URIAL raises it 11x on identical weights), and that the URIAL
#: prompt alone moves the Instruct model. Neither claim is legible in the default figure, which
#: keeps one frame per weight-set.
CELLS_FULL = [
    # the superscript is the FRAME: C = the model's own chat (or plain) template, U = URIAL's
    # no-refusal prompt. mathtext, which palette.RC routes through Inter, so it is the same
    # face as the rest of the label and not a serif fallback.
    (r"Instruct$^{\mathrm{C}}$",  "Instruct", "dense", _anchor("anchor_instruct_native"), "start", "instruct"),
    (r"Instruct$^{\mathrm{U}}$",  "Instruct", "dense", _anchor("anchor_instruct_urial_help"),
     "start", None),
    # NATIVE-FRAME GRPO (2026-09-13). The first arms trained under URIAL because the mask needs a
    # frame where base and instruct separate; these train in the frame they are judged in, which
    # is both the fairer baseline and much the stronger one -- SORRY-Bench 69.1 -> 100.0 and
    # GSM8K 5.5 -> 13.5 at the same recipe. The URIAL pair is still on disk; it is not the
    # comparison to report now that the reward and the eval can share a frame.
    ("GRPO",            "GRPO",     "dense",
     {"native": "refusal_grpo_weights_lora_native/eval_native/evals.json",
      "sorrybench": "refusal_grpo_weights_lora_native/eval_extra/evals.json",
      "ifeval": "refusal_grpo_weights_lora_native/eval_extra/evals.json"}, "trained", "grpo"),
    ("GRPO+KL",         "GRPO",     "dense",
     {"native": "refusal_grpo_weights_lora_native_kl001/eval_native/evals.json",
      "sorrybench": "refusal_grpo_weights_lora_native_kl001/eval_extra/evals.json",
      "ifeval": "refusal_grpo_weights_lora_native_kl001/eval_extra/evals.json"}, "trained",
     "grpo_kl"),
    ("Abliteration",    "Abliteration", "dense", _anchor("abliteration", "_native/evals.json"),
     "edit", "ablit"),
    ("GRP-Oblit",       "GRP-Oblit", "dense", _one("grpoblit_long_eval"), "edit", "oblit"),
    # the tick says both sparsities because the two scales are shown at DIFFERENT k: each
    # model's sparsity curve has its knee somewhere else (1B at 1%, 8B at 5%), and a shared k
    # would compare two different points on two different curves. See plot_refusal_sparsity_all.
    ("MAttr 1%/5%",     "MAttr",    "frac_0.01",
     _edited("refusal_grpo_logk_v2", "posthoc_eval/evals.json"), "edit", "mattr"),
    # THE NATIVE-FRAME FIT, at both scales: the same log-k recipe with the GRPO reward collected
    # under the model's OWN chat template instead of URIAL's no-refusal prompt. Shown at each
    # model's own best sparsity, the same best-of convention the GRPO+KL and GRP-Oblit bars use
    # (0.1% at 1B, where capability is still at the anchor; 0.5% at 8B, which is also the only
    # non-trivial condition its IFEval/SORRY-Bench job covers). The 1B run has no IFEval or
    # SORRY-Bench sweep at all, hence the two None sources.
    (r"MAttr log", "MAttr", "frac_0.001",
     {"native": "refusal_grpo_logk_vllm_native/eval_native/evals.json",
      "sorrybench": None, "ifeval": None}, "edit", "mattr_native_lo"),
    # the native-frame UNIFORM-k fit, the one 1B native cell that does not wreck GSM8K (30.0 at 1%
    # against the log-k fit's 3.5). 1B only: no 8B uniform run exists.
    (r"MAttr unif", "MAttr", "frac_0.01",
     {"native": "refusal_grpo_uniform_vllm_native/eval_native/evals.json",
      "sorrybench": None, "ifeval": None}, "edit", "mattr_nat_unif"),
    # ...and the same fit at 2%, the last sparsity before its GSM8K cliff (28.5 against a 29.0
    # anchor at 2%, 11.5 at 5%). The figure shows the 1% point and the table the 2% one, which is
    # why they are two cells rather than one -- see FIG_KEYS and the table's TABLE_1B.
    (r"MAttr unif", "MAttr", "frac_0.02",
     {"native": "refusal_grpo_uniform_vllm_native/eval_native/evals.json",
      "sorrybench": "refusal_grpo_uniform_vllm_native/eval_extra/evals.json",
      "ifeval": "refusal_grpo_uniform_vllm_native/eval_extra/evals.json"}, "edit",
     "mattr_nat_unif2"),
    (r"Base$^{\mathrm{C}}$",      "Base",     "dense", _anchor("anchor_base_plain"), "ceiling",
     None),
    (r"Base$^{\mathrm{U}}$",      "Base",     "dense", _anchor("anchor_base_urial_help"),
     "ceiling", "base_urial"),
]

#: THE DEFAULT: seven bars in four sections, separated by the dashed rules `draw` places wherever
#: the section key changes. The sections are the argument, in reading order:
#:
#:   start     the RLHF'd model in its own template -- the refusal floor everything is read from
#:   trained   GRPO on the weights against this same judge, without and with a tuned KL penalty:
#:             what it costs to erode refusal by optimising the model end to end
#:   edit      the three localised/weight-editing methods, ours last -- one refusal direction
#:             projected out (Arditi et al. 2024), the published trained baseline (GRP-Obliteration,
#:             Russinovich et al. 2026), and 1% of the base->instruct delta masked back in
#:   ceiling   base weights under URIAL's no-refusal prompt: the compliance this architecture's
#:             competence allows, which is what a safety bar is read against instead of 1.0
#:
#: One frame per weight-set, deliberately: the Instruct bar is its own template and the Base bar is
#: URIAL, because those are the two ends the middle sections live between. The two bars that price
#: the PROMPT rather than the weights (Instruct under URIAL, Base under its plain template) are in
#: CELLS_FULL, not here -- they answer a different question and made the figure nine bars wide.
CELLS = [c for c in CELLS_FULL
         if c[0] not in (r"Instruct$^{\mathrm{U}}$", r"Base$^{\mathrm{C}}$")]
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
PANEL_TITLE = {"sr": "StrongREJECT, %", "sb": "SORRY-Bench, %", "cap": "GSM8K, %",
               "ife": "IFEval, %", "mmlu": "MMLU, %"}
PANELS = ["sr", "sb", "cap", "ife"]
NCOLS = 2
#: EVERY PANEL IS A PERCENTAGE, which costs one caveat and buys two things. The two judges report
#: on [0, 1] natively and are scaled here by 100, so all four axes read the same way and each bar
#: can be quoted straight against GRP-Obliteration's tables, which report all five of their safety
#: benchmarks as percentages. The caveat is that SORRY-Bench's number IS a rate (the fraction of
#: prompts the judge marks fulfilled) while StrongREJECT's is a MEAN OF GRADED [0, 1] SCORES, so
#: "69%" there is not "69% of prompts complied with" -- the rate analogue is `frac_above`, which
#: the eval also records and which runs 5-10 points higher on every cell here. Say "StrongREJECT
#: score" rather than "attack success rate" when quoting it.
SCALE = {"sr": 100.0, "sb": 100.0, "cap": 1.0, "ife": 1.0, "mmlu": 1.0}
FMT = {"sr": "{:.1f}", "sb": "{:.1f}", "cap": "{:.1f}", "ife": "{:.1f}", "mmlu": "{:.1f}"}

#: WEIGHTS CHANGED relative to Instruct, as a percentage of the model's parameters -- the bf16 L0
#: measured in docs/refusal/l0_baselines.md (scripts/refusal/l0_baselines.py, 2026-09-12), i.e.
#: entries whose stored value actually differs from the Instruct checkpoint's.
#:
#: THE BF16 L0, NOT THE SUPPORT, and the choice is deliberately the one that flatters us least.
#: Abliteration and LoRA touch every entry of the matrices they edit -- 48% and 79% of the 1B model
#: -- but ~20% of those entries move by less than half a bf16 ulp and round back, which is the gap
#: between 48% and 39%, or 79% and 49%. Quoting support would widen every ratio in our favour. The
#: mask's number is the same under either reading (it copies base entries wholesale), so only the
#: baselines are affected, and they get their smaller figure.
#:
#: The two anchors are blank rather than 0 and 100: Instruct IS the reference, and Base is a
#: different model rather than an edit of it, so neither has an L0 in the sense the column means.
L0 = {("1B", "grpo"): 45.23, ("1B", "grpo_kl"): 45.09, ("1B", "ablit"): 38.93,
      ("1B", "oblit"): 100.00, ("8B", "ablit"): 27.63, ("8B", "oblit"): 50.58}

HIGHLIGHT = "MAttr"
#: WHICH CELLS THE FIGURE DRAWS, in order. CELLS is the pool (the table in
#: `table_baseline_strongreject.py` draws a different subset of it), and naming the figure's rows
#: here is what lets the two artifacts disagree about layout on purpose instead of by accident.
#: The mask rows are the three native-frame cells: 1B uniform-k at 1%, 1B log-k at 0.1%, and the
#: 8B log-k twin of the latter at the same 0.1%, which is the only pair in the figure measured at
#: one sparsity across both scales.
FIG_KEYS = ["instruct", "grpo", "grpo_kl", "ablit", "oblit",
            "mattr_nat_unif", "mattr_native_lo", "base_urial"]

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


#: (sweep, condition) pairs asked for and not present in a sweep that DOES exist -- normal when a
#: metric's own job covered fewer sparsities than the main one, reported rather than raised
ABSENT = []


def _load(rel, cond):
    p = RUNS / rel
    if not p.exists():
        return None, p
    fin = json.loads(p.read_text())["final"]
    if cond not in fin:
        ABSENT.append(f"{rel}:{cond}")
        return {}, p
    return fin[cond], p


#: which sweep each panel's number comes from, so a figure only demands the data it draws
PANEL_SOURCE = {"sr": "native", "cap": "native", "mmlu": "native",
                "sb": "sorrybench", "ife": "ifeval"}


def bars(panels=None, cells=None):
    """``[(group, label, {panel: (v, se)}, section, key, cond)]``; a named-but-missing sweep raises.

    ``panels`` restricts which sweeps are REQUIRED. Without it a run that has StrongREJECT and
    GSM8K but not yet IFEval cannot be drawn at all, even as a two-panel figure -- which is the
    normal state of a comparison whose last job is still queued.
    """
    need = ({PANEL_SOURCE[p] for p in panels if p in PANEL_SOURCE} if panels
            else {"native", "sorrybench", "ifeval"})
    out, missing = [], []
    for label, group, cond, src, section, key in (cells if cells is not None else CELLS):
        # a source of None means the metric was never run for this cell -- it reports NaN and is
        # drawn/printed as absent, where a MISSING FILE for a source that was named is still a
        # hard error. The two are different failures and only one of them is a mistake.
        def _get(name):
            if name not in need or src.get(name) is None:
                return ({} if src.get(name) is None else None), None
            return _load(src[name], cond)
        nat, np_ = _get("native")
        sb, _ = _get("sorrybench")
        ife, _ = _get("ifeval")
        for name, r in (("native", nat), ("sorrybench", sb), ("ifeval", ife)):
            if r is None and name in need and src.get(name) is not None:
                missing.append(src[name])
        if any(r is None for n, r in (("native", nat), ("sorrybench", sb), ("ifeval", ife))
               if n in need and src.get(n) is not None):
            continue
        nat, sb, ife = nat or {}, sb or {}, ife or {}
        nan = (float("nan"), float("nan"))
        mm = nat.get("mmlu", {}).get("mmlu", {})
        gs = nat.get("gsm8k", {}).get("gsm8k", {})
        sbm = sb.get("sorrybench", {}).get("off_target", {})
        ifm = ife.get("ifeval", {}).get("ifeval", {})
        rec = {
            "sr": (nat.get("strongreject", {}).get("off_target", {}).get("score", float("nan")),
                   sr_stderr(np_, cond) if np_ else float("nan")),
            # their compliance rate (mean of per-category means; the plain mean at 10/category)
            "sb": (sbm.get("fulfillment", float("nan")), sbm.get("stderr", float("nan"))),
            "cap": (gs.get("accuracy", float("nan")), gs.get("stderr", float("nan"))),
            # prompt-level strict: the number lm-eval-harness and the GRP-Oblit paper call IFEval
            "ife": (ifm.get("prompt_strict", float("nan")), ifm.get("stderr", float("nan"))),
            "mmlu": (mm.get("accuracy", float("nan")), mm.get("stderr", float("nan")))}
        out.append((group, label,
                    {k: (v * SCALE[k], se * SCALE[k]) for k, (v, se) in rec.items()},
                    section, key, cond))
    if missing:
        raise SystemExit("no sweep at: " + ", ".join(missing))
    return out


#: how each scale's series is drawn. FILLED circle = the model the figure is about, OPEN square =
#: its 8B twin, both in the cell's own group colour, dodged either side of the tick. Shape rather
#: than a second hue because colour is already spent on the weights group, and a figure where
#: "8B" and "Abliteration" were both hues would have two things competing for the same channel.
SCALE_STYLE = {"1B": dict(marker="o", ms=3.4, fillstyle="full", dodge=-0.13),
               "8B": dict(marker="s", ms=3.2, fillstyle="none", dodge=+0.13)}


def draw(ax, recs, twins, key, ylabel, *, last, legend=False, legend_loc="center left"):
    """One metric's panel as point-ranges: value +- 1 SE per cell, 1B beside its 8B twin."""
    xs = list(range(len(recs)))
    hl = [g == HIGHLIGHT for g, *_ in recs]
    # a cell with no 8B twin is drawn ON its tick, not dodged off it: the dodge exists to
    # separate a pair, and half a pair sitting off-centre reads as a missing point rather than as
    # a cell that was never run at that scale
    paired = {x for x, r in zip(xs, recs) if r[4] is not None and r[4] in twins}
    series = {"1B": [(x, r) for x, r in zip(xs, recs)],
              "8B": [(x, twins[r[4]]) for x, r in zip(xs, recs) if x in paired]}
    seen = []
    for scale, pts in series.items():
        st = SCALE_STYLE[scale]
        for x, r in pts:
            v, se = r[2][key]
            if math.isnan(v):          # metric never run for this cell
                continue
            colour = GROUP[r[0]][1]
            emph = r[0] == HIGHLIGHT
            ax.errorbar(x + (st["dodge"] if x in paired else 0.0), v, yerr=se, fmt=st["marker"], ms=st["ms"],
                        fillstyle=st["fillstyle"], color=colour,
                        markeredgecolor="#000000" if emph else colour,
                        markeredgewidth=0.7 if emph else (0.9 if st["fillstyle"] == "none" else 0),
                        ecolor=colour, elinewidth=0.9, capsize=1.2, capthick=0.6, zorder=3)
        seen.append(scale)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab, *_ in recs] if last else [], fontsize=FS_TICK,
                       rotation=45, ha="right", rotation_mode="anchor")
    if last:
        for tick, h in zip(ax.get_xticklabels(), hl):
            if h:
                tick.set_fontweight("bold")
    ax.set_xlim(-0.6, len(recs) - 0.4)
    # y range from the whiskers of BOTH series, with headroom for the legend in the first panel
    lo_hi = [r[2][key] for _, r in series["1B"] + series["8B"]
             if not math.isnan(r[2][key][0])]
    top = max(v + (0 if math.isnan(se) else se) for v, se in lo_hi)
    bot = min(v - (0 if math.isnan(se) else se) for v, se in lo_hi)
    pad = (top - bot) * 0.10
    ax.set_ylim(max(0, bot - pad), top + pad)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    # SECTION RULES: one wherever the section key changes, so they follow CELLS rather than a
    # hardcoded index and a reordering cannot leave a rule in the wrong place. Drawn under the
    # points (zorder 1) and in the spine colour, so they read as furniture and not as data.
    for i in range(1, len(recs)):
        if recs[i][3] != recs[i - 1][3]:
            ax.axvline(i - 0.5, color="#000000", lw=P.SPINE_LW, ls=(0, (3, 2)), zorder=1)
    if legend and series["8B"]:
        handles = [Line2D([], [], color="#333333", marker=SCALE_STYLE[s_]["marker"],
                          fillstyle=SCALE_STYLE[s_]["fillstyle"], ms=SCALE_STYLE[s_]["ms"],
                          markeredgewidth=0.9, ls="none", label=s_) for s_ in seen]
        # centre-left, which is empty BY THE SHAPE OF THE RESULT rather than by luck: on the
        # StrongREJECT panel the Instruct cell sits at the refusal floor and every other cell is
        # above 58, so the band between them is the one region no point can occupy. `best` puts
        # it top-right, into the gap between GRP-Oblit and the frame, which is where the eye is
        # doing the actual comparison. --legend-loc overrides it if the panel order changes.
        ax.legend(handles=handles, loc=legend_loc, fontsize=FS_ANNOT, frameon=False,
                  handletextpad=0.3, borderpad=0.1, labelspacing=0.2, ncol=2, columnspacing=0.8)
    P.furnish(ax)                       # grid + all four spines, the reference figure's frame
    ax.grid(False, axis="x")            # vertical rules behind points are pure noise
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
    ap.add_argument("--legend-loc", default="center left",
                    help="matplotlib loc for the 1B/8B scale legend in the first panel")
    ap.add_argument("--cells", default="1b", choices=["1b", "1b_full", "8b"],
                    help="which comparison to draw: the seven-bar 1B figure, the nine-bar ladder "
                         "that adds both models' other frame, or the 8B set (default: 1b)")
    a = ap.parse_args()
    panels = a.panels.split(",")
    if bad := [k for k in panels if k not in PANEL_TITLE]:
        raise SystemExit(f"unknown panel(s) {bad}; choose from {sorted(PANEL_TITLE)}")
    ncols = min(a.ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)

    global CELLS
    if a.cells == "8b":
        CELLS = CELLS_8B
    elif a.cells == "1b_full":
        CELLS = CELLS_FULL
    recs = bars(panels)
    if a.cells == "1b":
        by_key = {r[4]: r for r in recs}
        if bad := [k for k in FIG_KEYS if k not in by_key]:
            raise SystemExit(f"FIG_KEYS names {bad}, absent from CELLS")
        recs = [by_key[k] for k in FIG_KEYS]
    # the 8B twins, keyed by the field a 1B cell and its 8B counterpart share. Absent keys simply
    # draw no second point -- GRPO, GRPO+KL and Base were never run at 8B.
    twins = {} if a.cells == "8b" else {r[4]: r for r in bars(panels, CELLS_8B)}
    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(nrows, ncols, figsize=(a.width, a.height), squeeze=False)
    for i, key in enumerate(panels):
        r, c = divmod(i, ncols)
        # tick labels on the last panel of each column only -- the columns share the seven cells
        last = i + ncols >= len(panels)
        draw(axes[r, c], recs, twins, key, PANEL_TITLE[key], last=last, legend=(i == 0),
             legend_loc=a.legend_loc)
    for i in range(len(panels), nrows * ncols):
        axes[divmod(i, ncols)].set_visible(False)

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.6)
    fig.subplots_adjust(top=1.0 - HEAD / a.height)
    fig.savefig(a.out)
    print("wrote", a.out)

    keys = [k for k in ("sr", "sb", "cap", "ife", "mmlu")]
    print(f"\n{'cell':<16}" + "".join(f"{PANEL_TITLE[k]:>16}" for k in keys))
    for _, lab, m, _sec, ckey, _cond in recs:
        print(f"  {lab:<14}" + "".join(
            f"{FMT[k].format(m[k][0]):>9} ±{FMT[k].format(m[k][1]):<5}" for k in keys))
        if ckey in twins:
            t = twins[ckey]
            print(f"    {'└ 8B: ' + t[1]:<14}" + "".join(
                f"{FMT[k].format(t[2][k][0]):>9} ±{FMT[k].format(t[2][k][1]):<5}" for k in keys))


if __name__ == "__main__":
    sys.exit(main())
