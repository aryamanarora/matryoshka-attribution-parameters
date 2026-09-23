"""The refusal-anchor comparison as a LaTeX table, from the same loader as the bar/point figure.

`plot_baseline_strongreject.py` draws these cells; this prints them. It imports that module's
`bars()` rather than re-reading the sweeps, so the table and the figure can disagree about layout
but never about a number -- which is the whole reason this is a script and not a hand-kept .tex.

WHY A TABLE MIGHT BEAT THE FIGURE HERE. Four panels x seven cells x two scales is 46 numbers, and
the figure can only show them by spending its vertical axis on each metric's range; most of that
range is empty, because the interesting differences are a few points wide and sit far from zero
(every edited cell lands between 68 and 79 on StrongREJECT). A table gives each number its digits
back and costs no page height per metric, so MMLU -- which the figure drops for being flat
everywhere -- can be a column rather than an argument in a docstring.

WHAT IT KEEPS FROM THE FIGURE: the section order and grouping (the \\midrule between each), the
1B/8B pairing on one row-set, MAttr last within its section and set bold, the percentage scale on
every column, and +-1 standard error beside every value. What it drops: nothing.

Two things a caption should say, both of which the figure could not carry either:

* SORRY-Bench's number is a RATE (fraction of prompts the judge marks fulfilled); StrongREJECT's is
  a MEAN OF GRADED [0, 1] SCORES scaled by 100, so 69.2 is not "69% of prompts complied with". The
  rate analogue is `frac_above`, 5-10 points higher on every cell here.
* The two GRPO rows and GRP-Oblit optimise the StrongREJECT judge directly, so their StrongREJECT
  column is training-adjacent; SORRY-Bench is the held-out judge for every cell in the table.

    uv run python plots/table_baseline_strongreject.py            # to stdout
    uv run python plots/table_baseline_strongreject.py --out ../matryoshka-attribution/paper/tabs/baseline_strongreject.tex
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_baseline_strongreject as F                      # noqa: E402
from plot_baseline_strongreject import L0                    # noqa: E402

#: (column key, header). Harm first, then the three capability probes -- the figure's row order,
#: read left to right. MMLU earns a column here for the reason it is missing there: it is flat, and
#: a flat column is cheap where a flat panel costs a sixth of the figure.
#: HEADERS ARE ABBREVIATED and the caption has to expand them: SR = StrongREJECT (the mean of the
#: judge's graded [0,1] scores, x100), SORRY = SORRY-Bench (the fraction of its 450 prompts the
#: judge marks fulfilled), Edited % = the bf16 L0 against Instruct. The two judge names were the
#: widest cells in the table -- "StrongREJECT" alone set its column, since no value under it is
#: more than five characters -- so shortening them buys width that the numbers can use.
#: The L0 column is named for exactly what it holds -- the L0 norm of the weight delta against
#: Instruct, as a percentage of the model's parameters -- rather than "Edited %", which did not say
#: edited WHAT and sat two columns from "Method". It costs no width (the header is 0.7pt narrower).
COLS = [("l0", r"$\|\Delta\theta\|_0$ (\%)"), ("sr", "SR"), ("sb", "SORRY"),
        ("cap", "GSM8K"), ("ife", "IFEval"), ("mmlu", "MMLU")]
N_LEAD, N_SAFETY = 1, 2

#: L0 lives beside the cells it describes, in the plot module; imported here so the figure that
#: plots each baseline AT its edited-fraction and the table that prints the same number cannot
#: drift apart.
#: every nonresid unit is one residual-dimension vector, so a mask's parameter fraction EQUALS its
#: unit fraction exactly -- confirmed per run in the doc above, not assumed here
#: the groups whose L0 IS their sparsity: the learned mask and the two closed-form rankings of the
#: same delta, which select the same kind and number of units and differ only in the ranking
MASK_GROUPS = {"MAttr", "EG", "IxG"}
#: cell label -> how it should read in the table. The figure's tick labels carry mathtext
#: superscripts for the FRAME and a "1%/5%" that only makes sense beside a two-scale legend;
#: a table has room to say both in words.
#: Kept SHORT because the table is 63pt too wide for ICLR's 5.5in text block with the long forms
#: (measured, not guessed). What the short labels drop is the FRAME, which the caption has to
#: carry: Instruct is in its own chat template, Base is under URIAL's no-refusal prompt.
#: ``{k}`` is filled with the cell's own sparsity, so a row says which point of which curve it is.
# The mask rows are just \ourmethod{}: with one row per scale there is nothing to distinguish them
# FROM, and both facts the parenthetical carried are elsewhere -- the sparsity IS the Edited %
# column, and the fitting frame belongs in the caption. The ``{k}`` placeholder is still honoured
# by the builder if a label wants it back.
LABEL = {"EG": r"EG",
         "IxG": r"I$\times$G",
         r"MAttr (native)": r"\ourmethod{}",
         r"MAttr log": r"\ourmethod{}",
         r"MAttr unif": r"\ourmethod{}",
         r"Instruct$^{\mathrm{C}}$": r"Instruct",
         r"Instruct$^{\mathrm{U}}$": r"Instruct (URIAL)",
         r"Base$^{\mathrm{C}}$": r"Base (plain)",
         r"Base$^{\mathrm{U}}$": r"Base (URIAL)",
         "MAttr 1%/5%": r"\ourmethod{} (URIAL, {k})",
         "MAttr 1%": r"\ourmethod{} (URIAL, {k})",
         "MAttr 5%": r"\ourmethod{} (URIAL, {k})"}
def knee(cond):
    """The sparsity a mask cell was read at, as a suffix. Taken from the RECORD rather than from a
    table here, because the two scales and the two fitting frames all sit at different k -- each
    model's own knee -- and a shared k would compare different points on different curves."""
    if not cond.startswith("frac_"):
        return ""
    pct = float(cond[len("frac_"):]) * 100
    return r"%s\%%" % (f"{pct:g}")


#: SIGNIFICANCE AGAINST THAT SCALE'S INSTRUCT ANCHOR, as a cell tint. Blue = higher than Instruct,
#: orange = lower; which of those is GOOD depends on the column group and is the caption's job
#: (a high safety score means refusal was eroded, which is the point; a low capability score is
#: damage). The anchor rows are the reference and are never tinted.
#:
#: THE TEST is a two-sided z on the difference of two means, using the standard errors this table
#: already prints: z = (x - a) / sqrt(se_x^2 + se_a^2). Every metric here is a mean -- four are
#: proportions with binomial SEs, StrongREJECT is the mean of 60 graded [0,1] judge scores with
#: its SE taken from those scores -- so one form covers all five.
#:
#: IT IGNORES PAIRING, AND THAT IS CONSERVATIVE. Each condition is scored on the SAME prompts as
#: the anchor, so a paired test (McNemar on the binary metrics, Wilcoxon on StrongREJECT) would
#: use the per-item agreement and have strictly more power. Doing that needs per-item records,
#: which exist for StrongREJECT and GSM8K but not for MMLU, so the table would be testing two
#: different things in different columns. Unpaired everywhere loses power uniformly instead, which
#: means every cell tinted here would still be tinted under the better test.
#:
#: MULTIPLE TESTING: Holm-Bonferroni over the tested cells, which are now the CAPABILITY cells
#: only -- the family is what the table claims, and it claims nothing from a tint on harm.
#: Controlling the family-wise error rate at 0.05. Holm rather than
#: Benjamini-Hochberg because the table is read as a set of individual claims ("this method does
#: not hurt GSM8K") rather than as a screen, and FWER is the guarantee that supports that reading.
#: Bonferroni-corrected alpha is quoted in the printed summary so the caption can state it.
#:
#: AN UNTINTED CELL IS NOT EVIDENCE OF EQUIVALENCE, and a caption should not say "unchanged". With
#: n = 200 on GSM8K the test cannot see a 2-3 point difference at this alpha, so "no tint" means
#: "not distinguishable from Instruct at FWER 0.05 with this sample", which is the claim the table
#: supports and no more. The cells that matter for the no-damage reading -- MAttr's GSM8K, IFEval
#: and MMLU at both scales -- are untinted, and the honest phrasing of that is "no detectable
#: cost", with the per-metric n in the caption.
#: THE TINT MEANS DIFFERENT THINGS IN THE TWO COLUMN GROUPS, because the two groups want opposite
#: directions and one shared rule would be misleading in one of them.
#:
#: THE HARM COLUMNS ARE NOT TINTED AT ALL. Every method in the table erodes refusal by a margin
#: that dwarfs its standard error (z from 15 to 47), so a tint there marks nothing a reader cannot
#: see from the numbers, and spending colour on it drowns the column where the test actually
#: decides something. Only capability is tinted:
#:
#:   Capability (GSM8K,      orange = significantly LOWER than Instruct, i.e. measurable damage
#:   IFEval, MMLU)           blue   = not significantly lower -- no detectable cost
#:
#: BLUE IS "NO DETECTABLE COST", NOT "EQUIVALENT", and a caption must not upgrade it. The test has
#: the power it has: at n = 200 on GSM8K a 3-point drop would not be caught, so green marks the
#: absence of evidence for damage rather than evidence of its absence. It is still the reading the
#: column is for -- a method that leaves capability alone should be visibly distinguishable from
#: one that does not, and with orange alone the two looked identical wherever the damage was
#: partial. Cells with no measurement are left blank, not blue.
ALPHA = 0.05
HARM_KEYS = {"sr", "sb"}
#: blue rather than green for the good cell: green/orange is the reflex pairing and it looked it,
#: and blue is free now that the harm columns carry no tint. The pair also survives deuteranopia,
#: which green/orange does not.
LO = r"\cellcolor{orange!18}"
OK = r"\cellcolor{blue!11}"


def holm(pvals):
    """Holm-Bonferroni: returns the reject/keep flags for p-values in input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, m = [False] * len(pvals), len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= ALPHA / (m - rank):
            out[i] = True
        else:
            break                      # Holm stops at the first failure to reject
    return out


def ztest(v, se, a, ase):
    """Two-sided p for v vs the anchor a, means with standard errors se and ase."""
    denom = math.sqrt(se * se + ase * ase)
    if not denom or any(math.isnan(x) for x in (v, se, a, ase)):
        return None, 0.0
    z = (v - a) / denom
    return math.erfc(abs(z) / math.sqrt(2)), z


def cell(v, se):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{v:.1f}"
    if se is not None and not math.isnan(se):
        s += rf" {{\scriptsize$\pm${se:.1f}}}"
    return s


#: model column header per scale, and the anchor groups whose rows get a shaded background
MODEL_NAME = {"1B": r"\textit{Llama-3.2-1B-Instruct}", "8B": r"\textit{Llama-3.1-8B-Instruct}"}
SHADE = {"Instruct", "Base"}
#: WHICH 8B ROWS, in order. The 8B block is listed explicitly rather than joined to the 1B rows by
#: key, because the two blocks are not the same comparison: at 8B the URIAL-fitted mask is
#: superseded by the native-fitted one on every column, so the block shows the native fit at BOTH
#: its measured sparsities instead of one row per fitting frame. A key named here and missing from
#: CELLS_8B is a silent omission, so the builder raises instead.
#: ONE MASK ROW PER SCALE, each at that scale's own best budget -- the budgets are not matched
#: because the two runs are constrained by different things. At 1B, 2% is the last point before a
#: cliff (5% costs 17.5 GSM8K points for 5 of StrongREJECT); at 8B nothing costs capability below
#: 20%, so the binding constraint is instead that both judges saturate by 5%.
TABLE_8B = ["instruct", "ablit", "oblit", "eg_nat8", "mattr_nat_unif8", "base_urial"]
#: ...and which 1B rows. Named explicitly for the same reason: `plot_baseline_strongreject.CELLS`
#: is a POOL that both artifacts draw from, so a cell added for the figure must not silently
#: appear here. The table keeps the URIAL-fitted 1B mask; the figure does not.
#: EG sits directly above \ourmethod{} at the SAME budget, so the pair reads as one comparison of
#: rankings over one delta (the L0 column says so: identical). IxG (`ixg_nat2` / `ixg_nat8`) is
#: measured and in the cell pool but DROPPED from the table (2026-09-16): it removes little
#: refusal at either scale and its row said nothing EG's does not.
TABLE_1B = ["instruct", "grpo", "grpo_kl", "ablit", "oblit", "eg_nat2", "mattr_nat_unif2",
            "base_urial"]
SHADE_CMD = r"\rowcolor{black!7}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    ap.add_argument("--cells", default="1b", choices=["1b", "1b_full"])
    ap.add_argument("--skip", nargs="*", default=[], metavar="KEY",
                    help="row keys to leave out of this build (a cell whose evals have not landed "
                         "yet); the printed summary names them, so the omission is never silent")
    a = ap.parse_args(argv)
    if a.cells == "1b_full":
        F.CELLS = F.CELLS_FULL
    # a skipped key is dropped from the CELL LIST too, so a sweep that has not landed yet is
    # neither loaded nor printed, and the summary line says which rows are missing
    skip = set(a.skip)
    recs = F.bars(cells=[c for c in F.CELLS if c[5] not in skip])
    twins = {r[4]: r for r in F.bars(cells=[c for c in F.CELLS_8B if c[5] not in skip])}
    table_1b = [k for k in TABLE_1B if k not in skip]
    table_8b = [k for k in TABLE_8B if k not in skip]

    n = len(COLS)
    L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}",
         # EVERY numeric column right-aligned (2026-09-16; the metric columns were centred): each
         # value carries one decimal, so right alignment is decimal alignment, and the +- tails
         # hang together at a common right edge instead of drifting with the width of the value
         r"\begin{tabular}{l" + "r" * len(COLS) + "}",
         r"\toprule",
         # the leading columns (Method, and the L0 that belongs to neither group) sit outside both
         # spans, so the rules start after them
         r" & " * (1 + N_LEAD)
         + r"\multicolumn{%d}{c}{Harm} & \multicolumn{%d}{c}{Capability} \\"
         % (N_SAFETY, n - N_LEAD - N_SAFETY),
         r"\cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}"
         % (2 + N_LEAD, 1 + N_LEAD + N_SAFETY, 2 + N_LEAD + N_SAFETY, 1 + n),
         "Method & " + " & ".join(h for _, h in COLS) + r" \\"]

    ones = {r[4]: r for r in recs}
    for name, keys, pool in (("TABLE_1B", table_1b, ones), ("TABLE_8B", table_8b, twins)):
        if bad := [k for k in keys if k not in pool]:
            raise SystemExit(f"{name} names {bad}, absent from the cell list")
    blocks = {"1B": [ones[k] for k in table_1b], "8B": [twins[k] for k in table_8b]}
    if skip:
        print(f"% skipped rows (evals not landed): {sorted(skip)}", file=sys.stderr)

    # --- every test first, so Holm sees the whole family before a single cell is written
    anchors = {sc: next(r for r in rows if r[0] == "Instruct") for sc, rows in blocks.items()}
    tests = {}
    for scale, rows in blocks.items():
        a_met = anchors[scale][2]
        for group, label, met, section, key, cond in rows:
            if group == "Instruct":
                continue                       # the reference cannot differ from itself
            for mk, _ in COLS:
                if mk == "l0" or mk in HARM_KEYS:
                    continue
                p_, z = ztest(*met[mk], *a_met[mk])
                if p_ is not None:
                    tests[(scale, key, mk)] = (p_, z)
    keys = list(tests)
    flags = dict(zip(keys, holm([tests[k][0] for k in keys])))

    # the smallest L0 in each BLOCK is bold -- per block, not per table, because the two blocks
    # are separate models and 2.0 of a 1.2B model is not comparable to 5.0 of an 8B one. The
    # anchors have no L0 and cannot win it.
    def l0_of(group, key, cond, scale):
        return (float(cond[len("frac_"):]) * 100 if group in MASK_GROUPS
                else L0.get((scale, key)))

    best_l0 = {}
    for scale, rows in blocks.items():
        vals = [l0_of(r[0], r[4], r[5], scale) for r in rows]
        vals = [v for v in vals if v is not None]
        best_l0[scale] = min(vals) if vals else None

    for scale in ("1B", "8B"):
        L += [r"\midrule", r"\multicolumn{%d}{l}{%s} \\" % (1 + n, MODEL_NAME[scale])]
        for group, label, met, section, key, cond in blocks[scale]:
            name = LABEL.get(label, label)
            shown = name.replace("{k}", knee(cond)) if "{k}" in name else name
            if group == F.HIGHLIGHT:
                shown = r"\textbf{%s}" % shown
            l0 = l0_of(group, key, cond, scale)
            vals = []
            for mk, _ in COLS:
                if mk == "l0":
                    if l0 is None:
                        vals.append("--")
                    else:
                        t = f"{l0:.1f}"
                        # the closed-form rankings tie \ourmethod{} on L0 by construction (same
                        # units, same budget); bold marks the method row, not every tied cell
                        vals.append(r"\textbf{%s}" % t
                                    if l0 == best_l0[scale] and group == F.HIGHLIGHT else t)
                    continue
                txt = cell(*met[mk])
                t = tests.get((scale, key, mk))
                if t is not None:
                    sig = flags.get((scale, key, mk), False)
                    txt = (LO if (sig and t[1] < 0) else OK) + " " + txt
                vals.append(txt)
            L.append(((SHADE_CMD + " ") if group in SHADE else "") + shown + " & "
                     + " & ".join(vals) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", "}"]
    tex = "\n".join(L) + "\n"

    if keys:
        ps = sorted(tests[k][0] for k in keys)
        nsig = sum(flags.values())
        print(f"{len(keys)} tests, Holm at alpha={ALPHA}: {nsig} significant "
              f"(Bonferroni threshold {ALPHA/len(keys):.2e}; smallest p {ps[0]:.1e}, "
              f"largest rejected p "
              f"{max((tests[k][0] for k in keys if flags[k]), default=float('nan')):.1e})")
        for k in sorted(keys, key=lambda k: tests[k][0]):
            if not flags[k]:
                continue
            print(f"   {k[0]} {k[1]:18s} {k[2]:5s} z={tests[k][1]:+6.2f} p={tests[k][0]:.2e} "
                  + ("higher" if tests[k][1] > 0 else "lower"))
    if a.out:
        Path(a.out).write_text(tex)
        print("wrote", a.out)
    else:
        print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
