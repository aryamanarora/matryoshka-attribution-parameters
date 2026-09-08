"""Colours, dashes and markers for this repo's attribution figures.

THE HEXES ARE NOT CHOSEN HERE. They are imported from the sibling repo's palette,
``../learning-to-attribute/plots/palette.py``, because the two repos draw the SAME METHODS and a
method that is Wong blue in one paper's figure and Set1 blue in the other's reads as two methods.
That palette is CVD-verified with the pairwise dE distances recorded per entry (run
``python ../learning-to-attribute/plots/palette.py`` to re-check); re-picking colours here would
throw that away and silently reintroduce a collision it was built to avoid.

This module's own job is the MAPPING: this repo names arms by what varies in a *parameter-space*
sweep (`adam` / `sgd` / `ixg:base` / `ixg:mc` / `ixg:finetuned` / `random`), which is not the
sibling's vocabulary. So:

    adam           -> "MAttr"          Wong blue    -- MAttr, the optimizer being the hyperparameter
    sgd            -> "MAttr (SGD)"    black        -- its own hex upstream, for the reason recorded
                                                      there: an optimizer swap that BEATS every
                                                      other series is not readable as a linetype
                                                      variant of the series it beats
    ixg:base       -> "I×G"            Tol wine     -- the alpha=0 endpoint IS I×G
    ixg:finetuned  -> "I×G"            Tol wine     -- the alpha=1 endpoint, same method, other end
    ixg:mc         -> "Stepless IG"    Tol sand     -- upstream's own entry for alpha~U(0,1) at m=1
    random         -> "Random"         light grey   -- a reference, not a competitor

NOTE THE ONE CHANGE FROM THIS FILE'S FIRST VERSION, which gave the whole IxG family a single green
and separated the three by dash on the argument that they are "three points on ONE path, not three
methods". Upstream splits them, and upstream is right for a reason that shows up in the data here:
stepless IG is the BEST method on this repo's train-loss ranking (0.863) while IxG @ finetuned is
the WORST non-random one (1.008), so pooling them into one hue puts the top and bottom of the
ranking in the same colour. The two alpha ENDPOINTS keep one hue and separate by dash, which is
the part of that argument that survives.
"""

import importlib.util
from pathlib import Path

# Found where the rest of the repo finds learning-to-attribute: `deps/` first, the sibling checkout
# second (see "The mask dependency and `deps/`" in CLAUDE.md -- the vendored copy was dropped on
# 2026-09-07, so on a current clone it is the sibling that resolves). It is loaded BY PATH under a
# distinct module name rather than by putting its directory on sys.path. Both files are called
# `palette.py` and this one's own directory is already importable, so a plain `import palette`
# resolves to THIS module and fails with a circular-import AttributeError on the first attribute
# touched. Naming it explicitly removes the ambiguity.
_ROOT = Path(__file__).resolve().parents[1]
_UP_CANDIDATES = [_ROOT / "deps" / "learning-to-attribute" / "plots" / "palette.py",
                  _ROOT.parent / "learning-to-attribute" / "plots" / "palette.py"]
_UP_PATH = next((c for c in _UP_CANDIDATES if c.exists()), None)
if _UP_PATH is None:
    raise FileNotFoundError("learning-to-attribute's plots/palette.py not found at any of: "
                            + ", ".join(str(c) for c in _UP_CANDIDATES)
                            + " -- run scripts/setup.sh to clone it")
_spec = importlib.util.spec_from_file_location("_l2a_palette", _UP_PATH)
_up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_up)

#: rcParams, grid/spine treatment and font handling all come from upstream unchanged, so a figure
#: from either repo can sit beside one from the other without the furniture shifting.
RC = _up.RC
GRID_LW, GRID_COLOR, SPINE_LW = _up.GRID_LW, _up.GRID_COLOR, _up.SPINE_LW
furnish = _up.furnish

#: this repo's arm names -> upstream's method colours
COLOR = {
    "adam": _up.METHOD["MAttr"],
    "sgd": _up.METHOD["MAttr (SGD)"],
    "random": _up.METHOD["Random"],
    "ixg:base": _up.METHOD["I×G"],
    "ixg:finetuned": _up.METHOD["I×G"],
    "ixg:mc": _up.METHOD["Stepless IG"],
}

#: the two weight-sets the refusal figures compare (baseline anchors, sparsity sweep): upstream's
#: CVD-verified positive/negative class pair, Wong blue and Wong orange, which also separate by
#: lightness. Every figure that colours "Instruct" vs "Base" reads it from here.
MODEL = {"Instruct": _up.CLASS["pos"], "Base": _up.CLASS["neg"],
         # Two more for cells whose WEIGHTS are neither, i.e. the instruct model edited toward
         # base. They are split because one is the METHOD and the other is the reference it is
         # read against, and a figure that gave them one hue would be hiding its own comparison:
         #   MAttr  a mask condition -- Wong bluish green, CVD-checked upstream against the two
         #          above. NOT upstream's METHOD["MAttr"] blue, which CLASS["pos"] already spends
         #          here on the instruct weights; within this repo's refusal figures green IS the
         #          method, and nothing else uses it.
         #   GRPO   unconstrained GRPO on the weights -- the palette's reference grey, the same
         #          role (and the same hex) the random control has in every sibling figure.
         #   Abliteration  the other training-free weight edit (Arditi et al. 2024), and a
         #          COMPETITOR rather than a reference -- so it takes a method hue, not the grey.
         #          Tol wine, which is upstream's I×G; that method appears in no refusal figure,
         #          so the hex is free here exactly as the green is, and it was picked over the
         #          warm hues because Wong orange is already spent on the Base weights and a
         #          second warm would collide with it under deuteranopia.
         #   GRP-Oblit  the published TRAINED baseline (Russinovich et al. 2026). Its own hue
         #          rather than the GRPO grey, on the same principle as Abliteration: grey is for
         #          the two arms that are OUR controls, and a competitor from the literature is
         #          not a control. Wong sky blue, which upstream already carries beside Wong blue
         #          in the MIB figures, so the pair is CVD-checked -- they separate by lightness
         #          (L* 72 vs 43), which is the cue that survives at bar sizes. The warm hues were
         #          unavailable: Wong orange is spent on the Base weights and a second warm
         #          collides with it under deuteranopia.
         "MAttr": _up.METHOD["+hard"], "GRPO": _up.METHOD["Random"],
         "Abliteration": _up.METHOD["I×G"], "GRP-Oblit": _up.METHOD["GIM"]}

#: dash per reference line. The two I×G ENDPOINTS share a hue and separate here; stepless IG has
#: its own hue so it takes the solid line, and the random floor keeps a distinct dash because a
#: light grey solid line is easy to mistake for axis furniture.
REF_LS = {"ixg:base": (0, (1, 1.5)), "ixg:mc": "solid", "ixg:finetuned": (0, (5, 1.5)),
          "random": (0, (4, 2))}
REF_LABEL = {"ixg:base": r"I$\times$G @ base ($\alpha$=0)", "ixg:mc": "stepless IG (MC)",
             "ixg:finetuned": r"I$\times$G @ finetuned ($\alpha$=1)", "random": "random scores"}

#: k-schedule takes the linetype, per upstream's rule that colour is the METHOD and linetype the
#: hyperparameter -- AND the marker shape, redundantly. The redundancy is load-bearing rather than
#: belt-and-braces: an arm with a single LR draws no line at all (one point is not a series), so
#: with linestyle as the only schedule channel it renders as an unexplained floating marker in the
#: same colour as its neighbours -- which is exactly what the first version of the LR grid did with
#: the `logit` cell while the rest of that sweep was still queued. A partly-filled grid is the
#: normal state of a sweep in progress, so the encoding has to survive it.
LS = {"uniform": "solid", "log": "dashed", "logit": "dotted", "log_both": "dashdot"}
MK = {"uniform": "s", "log": "o", "logit": "^", "log_both": "D"}

#: upstream's full-width figure constants: (axis label, tick, legend) in points, and the legend
#: strip geometry. Kept identical so a legend here wraps and sits exactly as neuron_recall.pdf's.
FS_LABEL, FS_TICK, FS_LEGEND = 7.5, 6.5, 6.5
LEG_NCOL, LEG_ROW_H, LEG_PAD, TITLE_H = 4, 0.20, 0.10, 0.20


def sty(ks):
    """``(linestyle, marker, hollow)`` for a k-schedule label, which may carry a budget suffix.

    A non-default fitting budget keeps its schedule's dash but takes a HOLLOW marker, so a budget
    variant reads as "the same arm, run differently" rather than as a fifth schedule.
    """
    base = ks.split(" · ")[0]
    return LS.get(base, "solid"), MK.get(base, "s"), (" · " in ks)


def top_legend(fig, handles, nrow_panels, row_h, ncol=None):
    """Anchor `handles` in a strip ABOVE the panels, wrapping to as many rows as it needs.

    Transcribed from `plot_neuron_recall.py`. Two things about it are not cosmetic, and both were
    bugs there first: the strip height is DERIVED from how many rows the legend actually wraps to
    (a reserve sized for one row puts the second row straight through the panel titles), and
    `TITLE_H` is part of the same reserve because `set_title` draws ABOVE the axes rectangle --
    i.e. into the very band the legend is anchored in, so sizing the band for the legend alone
    collides them even though the arithmetic looks right.
    """
    ncol = ncol or LEG_NCOL
    nrows = -(-len(handles) // ncol)
    leg_h = nrows * LEG_ROW_H + LEG_PAD + TITLE_H
    fh = row_h * nrow_panels + leg_h
    top = 1.0 - leg_h / fh
    fig.subplots_adjust(top=top)
    fig.legend(handles=handles, ncol=ncol, loc="lower center",
               bbox_to_anchor=(0.5, top + TITLE_H / fh), frameon=False, fontsize=FS_LEGEND,
               handlelength=1.6, columnspacing=1.1, handletextpad=0.5, borderpad=0)
    return leg_h
