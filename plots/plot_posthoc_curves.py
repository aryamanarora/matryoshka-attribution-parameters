"""Post-hoc attribution across the sweep, as sparsity curves: where does the behaviour switch on?

Every finetune in `configs/french/{sft,posthoc}/` is attributed post hoc -- delta frozen, only the
scores trained -- and this figure draws all of those sparsity curves: **x is how much of the delta
the mask keeps**, one panel per (metric, method), one line per learning rate.

Curves rather than the heatmap in `plot_posthoc_grid.py`, and for three reasons that are about the
question rather than about taste:

* The question is the **shape** of the curve -- the fraction at which the behaviour appears -- and
  position resolves that where colour resolves perhaps five levels.
* Metrics in different units can each have their **own y scale** (``scales="free_y"``), instead of
  needing two colour scales and two composed figures to keep a 0.7-1.6 loss off the same ramp as a
  fraction in [0, 1].
* The **pretrained anchor is drawable**. It is the same base model for every run, so one grey
  reference line per panel says where "the mask kept nothing useful" sits -- and a curve that never
  leaves it never localised anything. A heatmap has nowhere to put that.

The heatmap is still the better figure for the *dense* point (`plot_method_lr_grid.py`), where
there is one number per (method, lr) and the 2-D grid is the whole content.

The rightmost point of every curve is ``frac_1``, which composes the same weights as the
``full_delta`` anchor (the runner detects that and reuses the result), so the dense value the
finetune itself reported is already the end of the line -- there is no separate anchor to draw for
it, and a curve that reaches its own right-hand end has reproduced the finetune.

**The objective is part of a series' identity.** ``Learned (SFT loss)`` fits the scores by
minimising the SFT loss and only *checks* the behaviour; ``GRPO (metric)`` fits them to the
off-target metric itself (an ``rl:`` block, see ``train/rl.py``); ``IxG @ ...`` computes them in
closed form. Same delta and same layout in every case -- only what the scores were fitted to
differs, which is the comparison the figure exists for.

**Unit granularity is part of a series' identity too.** A run's ``mask.unit`` is appended to its
label whenever the data contains more than one granularity, so a `nonresid` curve and a `weight`
curve over the same checkpoint are two series rather than two replicates of one. They are not
comparable as "the same measurement at finer resolution": a `nonresid` unit is a whole neuron and a
`weight` unit is one scalar, so at 0.1% the first keeps 603 neurons and the second keeps 1.2M
individual weights scattered anywhere. Same x axis, different objects.

**The attribution method is a distinct series, never a replicate.** Each run records how its
scores were obtained -- ``learned`` (trained through the differentiable top-k) or ``ixg`` at one of
two gradient points -- and that label joins (method, lr) as part of a curve's identity, drawn as
line style when more than one is present. Without it the two IxG gradient points, which attribute
the same checkpoint at the same lr, would be pooled as if they were two runs of one thing and
averaged into a single line: the figure would show the mean of two different methods with their
disagreement drawn as run-to-run noise.

**Replicates are averaged, and the ribbon is their range.** Two post-hoc runs can attribute the
same recipe -- the same (method, lr) trained twice, or the same cell present in two grids -- and
those are drawn as one line through the mean with a band covering the observed min-max. It is a
*range*, not a confidence interval: with two runs a CI would be a statistical claim the sample
cannot support, while a range is exactly what was seen. `n` is printed per group when it exceeds 1.

**Only one generation backend at a time**, default vLLM. Replicates are grouped by backend and the
minority backend is dropped, because vLLM and HF do not decode identically even greedy (see
`eval/vllm_gen.py`) -- averaging across them would put a decoder difference inside the band and
label it run-to-run variation. `--backend any` overrides, `--backend hf` selects the other one.
This is what removes the second `Full SFT @ 1e-4` line: `french_mask_posthoc` attributes the older
`french_lr1e-4_cfg` finetune and was scored through HF.

Cells whose *attributed finetune* had diverged are dropped: a mask fitted to a model that no longer
answers French prompts in French is not attributing language drift. `--source-dir` supplies the
finetunes' own results to detect them; `--include-diverged` keeps them.

    uv run python plots/plot_posthoc_curves.py --dir plots/data/posthoc_sweep \
        --out plots/posthoc_curves.pdf
"""

import argparse
import math
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, facet_wrap, geom_blank,
    geom_hline, geom_line,
    geom_point, geom_ribbon, ggplot, guide_legend, guides, labs, scale_color_brewer,
    scale_color_cmap, scale_fill_cmap,
    scale_fill_brewer, scale_x_log10,
    theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

#: Concise display names for `mask.unit`, used when `--color-by unit` makes granularity the
#: colour axis rather than a suffix on the attribution label. Anything absent passes through
#: unchanged, so a new unit mode needs no entry here to be plottable.
#:
#: The hybrids name BOTH halves. `svd_attn` factors the attention projections and leaves the MLP on
#: nonresid units, and a bare "SVD (attn)" reads as "SVD, of attention only" -- i.e. as though the
#: MLP were unmasked, which would be a different experiment with a different unit total. The
#: `+ nonresid` is what stops that reading, and it is worth the four characters: a reader who
#: mis-parses the legend mis-reads the denominator, since 97% of this mode's units are the nonresid
#: half.
UNIT_LABEL = {"svd": "SVD (all)", "svd_attn": "SVD attn + nonresid",
              "svd_mlp": "SVD MLP + nonresid", "neuron_head": "neuron/head"}

#: Which column colour is mapped to. Set once in `main` from `--color-by`; the layer builders read
#: it rather than taking it as an argument, because there are four call sites across the two
#: layouts and threading it through all of them is how they drift apart.
COLOR, COLOR_TITLE = "LR", "Learning rate"

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=5.5, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=6.5),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

#: (row title, path into a condition's results, y range or None to fit the data). Losses and rates
#: share no scale, which is what `scales="free_y"` is for -- one y axis per row.
#:
#: The two rates are pinned to [0, 1] rather than fitted. Fitted, the in-dist row auto-scales to
#: 0.94-1.00 and its 64-prompt sampling noise -- three responses either way -- draws as a dramatic
#: collapse, right next to an off-target row where the same visual amplitude is the entire result.
#: A fraction's range is known in advance, so there is no reason to let the data choose it.
#: Metric sets, one per organism -- ``--metrics`` picks one. ``language`` is the default, so the
#: French/Bactrian figures this script was written for are unaffected.
#:
#: ``diverged`` is how a run whose ATTRIBUTED FINETUNE fell over is recognised, and it is not
#: shared, for the same reason as in plot_method_lr_grid.py: the language in-dist control sits at
#: ~0.95 for the pretrained model, so "in-dist below half of that" means a destroyed model -- but
#: the casing in-dist control starts at **0.0**, because the habit has not been taught yet, so the
#: same rule there is ambiguous between "collapsed" and "the finetune did not take". Casing uses
#: ``undetermined_frac`` instead: the fraction of responses with too few cased characters to judge,
#: i.e. the punctuation spam a diverged run emits. On the real sweep it is 0.0 for all seven
#: healthy cells and 1.0 for the diverged one.
#:
#: Before this existed the language rule was applied unconditionally, which silently let a diverged
#: CASING cell through: `dig` returned None for a path a casing run does not have, so it was never
#: added to the drop set.
PRESETS = {
    "language": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            # `{target}` is filled from the runs' own `eval.language.target` -- see
            # resolve_language_titles. Hardcoded "FR" here silently mislabelled every non-French
            # organism (fr2de answers in GERMAN, and the nine Bactrian languages each in their own),
            # and the panel it mislabels is the headline.
            ("In-dist {target}", ("language", "in_dist", "target_frac"), (0.0, 1.0)),
            ("Off-target {target}", ("language", "off_target", "target_frac"), (0.0, 1.0)),
        ],
        diverged=(("language", "in_dist", "target_frac"), 0.5, "below"),
    ),
    "casing": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            # Row strips are rotated, so their budget is the PANEL HEIGHT (~0.72in), not the width
            # the heatmap's column strips get. "Off-target lowercase" overruns it and collides with
            # the row above; ~13 characters is what fits, which is why the language preset's
            # "Off-target FR" does. The metric is the lowercase fraction in both rate rows and the
            # scale is pinned to [0, 1].
            ("In-dist", ("casing", "in_dist", "lower_frac"), (0.0, 1.0)),
            ("Off-target", ("casing", "off_target", "lower_frac"), (0.0, 1.0)),
        ],
        diverged=(("casing", "in_dist", "undetermined_frac"), 0.5, "above"),
    ),
    #: The spelling organism (``configs/spelling/``). Five panels rather than four: ``probe_british``
    #: earns one because this is the only organism whose dense endpoint has a GAP between the two
    #: prompt cues (off-target 0.48-0.59, probe_british 0.98-1.00), and the sparsity axis is where
    #: that gap either survives or collapses -- i.e. whether cue-following and the unconditional
    #: habit are carried by different units.
    #:
    #: The headline is ``british_word_frac``, the pooled British share of all variant words in a
    #: split, NOT the response-level ``british_frac``: at a measured 1.53 variant words per response
    #: the per-response verdict is close to a coin flip, and pooling is what makes 64 responses a
    #: usable number. See eval/spelling.py.
    #:
    #: The diverged rule uses ``undetermined_frac`` on **off_target**, not on ``in_dist``. A broken
    #: model emits no scorable variant words, so unlike the ALL-CAPS organism this metric fails
    #: loudly -- but only the off-target split can see it, because every off-target prompt contains a
    #: variant word by construction and a healthy answer echoes one. MEASURED on the 8B sweep:
    #:
    #:   off_target undetermined   0.06, 0.06, 0.09 healthy   |  1.00 collapsed
    #:   in_dist    undetermined   0.39, 0.48, 0.56 healthy   |  1.00 collapsed
    #:
    #: in_dist prompts are ordinary Alpaca instructions, only ~13% of which carry a variant word at
    #: all, so half their answers are unscorable in a perfectly healthy run. A 0.5 threshold on that
    #: split dropped the healthy lr-1e-4 cell as "diverged" -- the same shape of mistake as reading an
    #: ALL-CAPS run with the lowercase preset.
    "spelling": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            ("In-dist", ("spelling", "in_dist", "british_word_frac"), (0.0, 1.0)),
            ("Off-target", ("spelling", "off_target", "british_word_frac"), (0.0, 1.0)),
            ("Probe British", ("spelling", "probe_british", "british_word_frac"), (0.0, 1.0)),
        ],
        diverged=(("spelling", "off_target", "undetermined_frac"), 0.5, "above"),
    ),
    #: The mirror organism (``configs/caps/``, ``eval.casing.target: upper``): ALL-CAPS training,
    #: lowercase probe, so the headline is ``upper_frac``. A separate preset rather than a flag for
    #: the reason spelled out in plot_method_lr_grid.py's copy -- reading an ALL-CAPS run with the
    #: ``casing`` preset yields ~0.00 everywhere for a run whose habit transferred perfectly, which
    #: is a silently inverted figure rather than a missing one.
    "casing_upper": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            ("In-dist", ("casing", "in_dist", "upper_frac"), (0.0, 1.0)),
            ("Off-target", ("casing", "off_target", "upper_frac"), (0.0, 1.0)),
        ],
        diverged=(("casing", "in_dist", "undetermined_frac"), 0.5, "above"),
    ),
    #: The judged organism (``configs/pirate/``). Same shape as ``casing``, with the divergence rule
    #: keyed on ``incoherent_frac``: the pirate in-dist control also starts near 0 (the register has
    #: not been taught), so "in-dist below half" cannot tell a collapsed run from an untrained one --
    #: and here the reason to care is sharper than a figure convention. The 8B lr 5e-4 cell collapsed
    #: into the dialect's own function words on repeat and the judge scored those `pirate=100,
    #: coherent=0`, so on THIS organism a damaged model can score maximally rather than at zero.
    #: `incoherent_frac` was 1.00 there against 0.00 for every healthy cell, which is what drops it.
    #:
    #: The rate rows plot ``pirate_frac`` rather than the safer ``pirate_frac_coherent`` for one
    #: mechanical reason: the four 8B runs already on disk predate that metric, so a preset keying on
    #: it would draw empty panels for them (the `dig`-returns-None failure this file's history already
    #: has an instance of). The divergence rule is what makes plotting the raw rate safe HERE -- a
    #: collapsed cell is dropped from the figure before it is drawn. In prose, quote
    #: ``pirate_frac_coherent``.
    "pirate": dict(
        metrics=[
            ("Train loss", ("sft_loss", "train", "loss"), None),
            ("Test loss", ("sft_loss", "test", "loss"), None),
            ("In-dist", ("pirate", "in_dist", "pirate_frac"), (0.0, 1.0)),
            ("Off-target", ("pirate", "off_target", "pirate_frac"), (0.0, 1.0)),
        ],
        diverged=(("pirate", "in_dist", "incoherent_frac"), 0.5, "above"),
    ),
}

#: set from the chosen preset in main(), before anything reads them
METRICS = PRESETS["language"]["metrics"]
DIVERGED = PRESETS["language"]["diverged"]
FRAC_RE = re.compile(r"^frac_(?P<frac>[0-9.]+)$")
PRETRAINED = "pretrained"

SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr: float) -> str:
    exp, m = 0, float(lr)
    while m < 1:
        m, exp = m * 10, exp - 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def method_of(finetuned: str) -> str:
    """``.../french_lora_r128_lr5e-4/adapter`` -> ``LoRA r=128``.

    From the attributed checkpoint's path, because a post-hoc config deliberately has no ``lora:``
    block -- the mask is over base-model parameter names and the adapter is an input to it, so the
    run being attributed is the only place the parameterisation is recorded.
    """
    run = Path(finetuned.rstrip("/")).parent.name
    # two naming conventions in the tree, and missing the second one silently mislabels a LoRA
    # cell as full SFT (which then plots in the wrong facet): the French rank grid writes
    # `..._lora_r128_lr5e-4`, the casing sweep `..._lora128_lr2e-4`.
    m = re.search(r"_r(\d+)_", run) or re.search(r"_lora(\d+)_", run)
    return f"LoRA r={m.group(1)}" if m else ("LoRA r=32" if "_lora_" in run else "Full SFT")


def diverged_sources(source_dir: Path) -> set:
    """Names of finetune runs that fell over, by the current preset's rule (see :data:`PRESETS`)."""
    out = set()
    if not source_dir or not source_dir.exists():
        return out
    path, thresh, direction = DIVERGED
    for d in source_dir.iterdir():
        f = d / "evals.json"
        if not f.exists():
            continue
        v = dig(json.loads(f.read_text()).get("final", {}).get("dense", {}), path)
        if v is not None and (v > thresh if direction == "above" else v < thresh):
            out.add(d.name)
    return out


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
    return node


#: ``{finetune run name: (method label, its finetuning lr)}``, filled from ``--source-dir``. Both
#: facts belong to the ATTRIBUTED run and neither is reliably recoverable from the post-hoc run:
#:
#: * ``cfg["train"]["lr"]`` on a post-hoc run is the rate the SCORES were fitted at (2e-5 by
#:   default), not the rate the delta was trained at. Colouring by it drew four fits of four
#:   different finetunes -- lr 1e-5, 2e-5, 3e-5, 7e-5 -- as one series of "4 replicate runs" with a
#:   mean and a range band, which is a figure that says the opposite of the truth.
#: * :func:`method_of` parses the rank out of the parent's *name*, and the fr2de ablation grid
#:   names its cells `fr2de_abl8b_lr1e-5` with no rank in them, so every one fell through to
#:   "Full SFT" and split the figure into a phantom parameterisation.
#:
#: Reading the parent's own ``config.yaml`` fixes both, and is what `plot_method_lr_grid.py`
#: already does for the same two fields. The name-based fallbacks stay for a source dir that was
#: not passed.
SOURCE_META = {}


def load_source_meta(source_dir: Path) -> dict:
    out = {}
    if not source_dir or not source_dir.exists():
        return out
    for d in sorted(source_dir.iterdir()):
        cf = d / "config.yaml"
        if not cf.exists():
            continue
        try:
            cfg = yaml.safe_load(cf.read_text()) or {}
        except Exception:
            continue
        lora = cfg.get("lora") or {}
        out[d.name] = (f"LoRA r={lora['r']}" if lora else "Full SFT",
                       float((cfg.get("train") or {}).get("lr")))
    return out


def rows_for(run_dir: Path):
    """``[{method, lr, frac, metric, value, anchor}]`` for one post-hoc run."""
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not (ev.exists() and cf.exists()):
        return []
    # freshness, not existence: config newer than evals means these results predate the config
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml newer than evals.json (stale)")
        return []
    cfg = yaml.safe_load(cf.read_text())
    finetuned = (cfg.get("mask") or {}).get("finetuned")
    if not finetuned:
        return []
    res = json.loads(ev.read_text()).get("final") or {}
    mk = cfg.get("mask") or {}
    how = mk.get("scores", "learned")
    # a run with an `rl:` block fitted its scores to the behaviour, not to the SFT loss. That is a
    # different series, not a replicate: same delta and same layout, different objective.
    if cfg.get("rl") is not None:
        how = "grpo"
    src = Path(finetuned.rstrip("/")).parent.name
    # the attributed run's own config wins over both name-parsing and this run's optimizer settings
    method, lr = SOURCE_META.get(src, (method_of(finetuned), float(cfg["train"]["lr"])))
    base = dict(run=run_dir.name, source=src, method=method, lr=lr,
                unit=mk.get("unit", "?"),
                attribution=("GRPO (metric)" if how == "grpo" else
                             "Learned (SFT loss)" if how != "ixg" else
                             f"IxG @ {mk.get('ixg_at')}")
                # `svd_basis: random` is the CONTROL for an svd cell: the same delta as r rank-1
                # terms in a random basis instead of the singular one. A distinct series for the
                # reason every other entry here is one -- same delta, same layout, same unit count,
                # same fitting budget, and only the basis differs -- so pooling it with its twin
                # would draw the thing the control exists to measure as replicate noise.
                + (" [random basis]" if mk.get("svd_basis", "svd") == "random" else ""),
                backend="vllm" if (cfg.get("eval") or {}).get("vllm") else "hf")
    out = []
    for cond, per_eval in res.items():
        m = FRAC_RE.match(cond)
        if not m:
            continue
        for title, path, _ in METRICS:
            v = dig(per_eval, path)
            if v is not None:
                out.append(dict(base, frac=float(m.group("frac")), metric=title, value=float(v)))
    # the pretrained anchor: theta_base, so the same model for every run and every learning rate,
    # which is why it can be one reference line per panel rather than one per curve
    for title, path, _ in METRICS:
        v = dig(res.get(PRETRAINED) or {}, path)
        if v is not None:
            out.append(dict(base, frac=None, metric=title, value=float(v)))
    return out


def resolve_language_titles(roots) -> list:
    """Fill ``{target}`` in the metric titles from the runs' own ``eval.language.target``.

    The answer language is a per-experiment config value, not a property of the figure, so the
    title has to come from the data: `configs/french/` answers in French, `configs/fr2de/` in
    GERMAN, and the nine Bactrian languages each in their own. A hardcoded code is wrong for all
    but one of them, on the panel that carries the headline.

    Runs that disagree are a hard error rather than a generic label: one x axis with two answer
    languages on it is two experiments drawn as one, and "In-dist" with the code dropped would
    make that invisible instead of loud. Falls back to dropping the placeholder when nothing
    records a target (a preset without one, or configs that predate the field).
    """
    targets = set()
    for root in roots:
        for d in sorted(Path(root).iterdir()):
            cf = d / "config.yaml"
            if not (d.is_dir() and cf.exists()):
                continue
            t = ((yaml.safe_load(cf.read_text()).get("eval") or {}).get("language") or {}).get(
                "target")
            if t:
                targets.add(str(t))
    if len(targets) > 1:
        raise SystemExit(
            f"these runs answer in {sorted(targets)}, so one y axis would mean two different "
            "languages. Plot them separately, or pass --metrics for an organism whose headline is "
            "not language-specific.")
    code = (targets.pop().upper() if targets else "")
    out = [((t.replace(" {target}", f" {code}") if code else t.replace(" {target}", "")), path, rng)
           for t, path, rng in METRICS]
    if code:
        print(f"  answer language: {code}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", default=["plots/data/posthoc_sweep"],
                   help="one or more run directories; pass both the learned and the IxG sweeps to "
                        "get all three attribution methods on one axes")
    p.add_argument("--source-dir", default="plots/data/method_lr")
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument("--stacked", action="store_true",
                   help="the OLD layout: metrics down the rows, methods across the columns. The "
                        "default is now transposed (metrics on the columns, methods on the rows), "
                        "which reads as a wide figure and matches plot_train_curves.py "
                        "--facet-method panel for panel. Pass this to reproduce a figure generated "
                        "before that default changed.")
    p.add_argument("--source", nargs="+", default=None,
                   help="restrict to attributions OF these finetunes (run directory names), so a "
                        "figure can be exactly one checkpoint's curves rather than a grid in which "
                        "one cell happens to carry the comparison")
    p.add_argument("--exclude-scores", nargs="*", default=(),
                   help="attribution labels to leave out, e.g. 'IxG @ finetuned'. Three line "
                        "styles across five learning rates is where this figure stops being "
                        "readable, so a series that is dominated everywhere is better dropped "
                        "than drawn")
    p.add_argument("--backend", default="vllm", choices=("vllm", "hf", "any"),
                   help="which generation backend's runs to plot; mixing them puts a decoder "
                        "difference inside the replicate band")
    p.add_argument("--legend-rows", type=int, default=1,
                   help="wrap the top legend onto this many rows. The default 1 is what every "
                        "figure in the repo was drawn with; raise it when the entries are long "
                        "enough that one row overruns the canvas width and gets clipped (four "
                        "'SVD attn + nonresid'-length labels do)")
    p.add_argument("--lr-gradient", action="store_true",
                   help="colour by log10(lr) on a CONTINUOUS ramp with a colourbar, instead of one "
                        "Set1 hue per lr. The categorical default is right when the levels are "
                        "unordered methods; a learning-rate sweep is an ordered axis, and a ramp "
                        "says so -- neighbouring rates get neighbouring colours and the reader "
                        "does not have to consult the legend to order them. Log because the sweep "
                        "is geometric.")
    p.add_argument("--same-loss-ylim", action="store_true",
                   help="one y range across the train- and test-loss panels, so the two are read "
                        "against each other rather than each fitted to its own spread")
    p.add_argument("--color-by", default="lr", choices=("lr", "unit"),
                   help="what colour means. 'lr' (default) is the learning rate, which is the "
                        "variable in every sweep this figure was written for. 'unit' colours by "
                        "mask.unit instead -- for a figure whose cells attribute ONE finetune at "
                        "one rate and differ in the unit definition, where an LR colour would be a "
                        "constant and the granularities would only be separable by line style")
    p.add_argument("--metrics", default="language", choices=sorted(PRESETS),
                   help="which organism's metrics to read: 'language' is the target-language "
                        "fraction (configs/french*, the default), 'casing' the lowercase fraction "
                        "(configs/lower). They differ in the divergence rule as well as the paths "
                        "-- see PRESETS")
    p.add_argument("--out", default="plots/posthoc_curves.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    global METRICS, DIVERGED, COLOR, COLOR_TITLE
    METRICS = PRESETS[args.metrics]["metrics"]
    DIVERGED = PRESETS[args.metrics]["diverged"]
    if args.color_by == "unit":
        COLOR, COLOR_TITLE = "Unit", "Mask unit"
    # BEFORE rows_for, which stamps each row with its metric's title -- resolving afterwards would
    # leave the rows keyed on the placeholder and every panel empty
    METRICS = resolve_language_titles(args.dir)

    global SOURCE_META
    SOURCE_META = load_source_meta(Path(args.source_dir))
    rows = [r for root in args.dir for d in sorted(Path(root).iterdir()) if d.is_dir()
            for r in rows_for(d)]
    unresolved = {r["source"] for r in rows} - set(SOURCE_META)
    if unresolved:
        print(f"  WARNING: {len(unresolved)} attributed finetune(s) not in --source-dir "
              f"({args.source_dir}), so their method/lr come from the run NAME: "
              f"{', '.join(sorted(unresolved)[:3])}{' …' if len(unresolved) > 3 else ''}")
    if not rows:
        raise SystemExit(f"no post-hoc results under {args.dir}")
    df = pd.DataFrame(rows)

    # only mention granularity when there is more than one to distinguish; otherwise every label
    # would carry a constant
    if df["unit"].nunique() > 1:
        # Under `--color-by unit` the granularity IS the colour, so appending it to the attribution
        # label as well would say the same thing twice -- once in the legend and once in the line
        # style -- and burn the line-style channel on a distinction already drawn.
        if COLOR != "Unit":
            df["attribution"] = df["attribution"] + " (" + df["unit"] + ")"
        print(f"  granularities present: {sorted(df['unit'].unique())}")

    if args.exclude_scores:
        drop = df["attribution"].isin(args.exclude_scores)
        print(f"  excluding {sorted(df[drop]['attribution'].unique())}: "
              f"{drop.sum()} rows, {df[drop]['run'].nunique()} runs")
        df = df[~drop]
        if df.empty:
            raise SystemExit("--exclude-scores removed everything")

    if args.source:
        keep = df["source"].isin(args.source)
        missing = set(args.source) - set(df["source"])
        if missing:
            raise SystemExit(f"no runs attribute {sorted(missing)}; present: "
                             f"{sorted(df['source'].unique())}")
        print(f"  restricted to {sorted(set(df[keep]['source']))}: "
              f"{df[keep]['run'].nunique()} runs")
        df = df[keep]

    bad = diverged_sources(Path(args.source_dir))
    if bad and not args.include_diverged:
        drop = df["source"].isin(bad)
        if drop.any():
            print(f"  dropping {sorted(df[drop]['source'].unique())}: attributed finetune diverged")
        df = df[~drop]

    if args.backend != "any":
        wrong = df["backend"] != args.backend
        if wrong.any():
            print(f"  dropping {sorted(df[wrong]['run'].unique())}: generated with "
                  f"{sorted(df[wrong]['backend'].unique())}, not {args.backend!r}")
        df = df[~wrong]
        if df.empty:
            raise SystemExit(f"no runs left with --backend {args.backend}")

    anchors = df[df["frac"].isna()].copy()
    curves = df[df["frac"].notna()].copy()
    print(f"{curves['run'].nunique()} post-hoc runs, {curves['frac'].nunique()} sparsity points")

    order = sorted(curves["method"].unique(),
                   key=lambda m: (m == "Full SFT", int(m.split("=")[1]) if "=" in m else 0))
    attrs = sorted(curves["attribution"].unique())
    print(f"  attribution methods: {attrs}")
    for d in (curves, anchors):
        d["method"] = pd.Categorical(d["method"], order, ordered=True)
        d["attribution"] = pd.Categorical(d["attribution"], attrs, ordered=False)
        d["metric"] = pd.Categorical(d["metric"], [t for t, _, _ in METRICS], ordered=True)
        # under --lr-gradient the colour column is NUMERIC (log10 lr), which is what makes the
        # scale continuous; the categorical labels stay available for the colourbar's breaks
        d["LR"] = ([math.log10(x) for x in d["lr"]] if args.lr_gradient else
                   pd.Categorical([lr_label(x) for x in d["lr"]],
                                  [lr_label(x) for x in sorted(df["lr"].unique())], ordered=True))
        # ordered by the raw mode name so `nonresid` (the baseline granularity) leads and the svd
        # family stays contiguous, rather than by the display label
        d["Unit"] = pd.Categorical([UNIT_LABEL.get(u, u) for u in d["unit"]],
                                   [UNIT_LABEL.get(u, u) for u in sorted(df["unit"].unique())],
                                   ordered=True)
    # Replicates: one row per (method, lr, metric, frac), mean for the line and min/max for the
    # band. `n` is carried through so the caller can see which groups actually have a replicate.
    # `unit`/`Unit` are grouping keys unconditionally. Under `--color-by unit` they are the only
    # thing separating the cells, so leaving them out would average the granularities together and
    # draw their difference as a replicate band -- one line where there should be four.
    keys = ["attribution", "method", "lr", "LR", "unit", "Unit", "metric", "frac"]
    agg = (curves.groupby(keys, observed=True)["value"]
           .agg(value="mean", lo="min", hi="max", n="size").reset_index())
    agg["series"] = (agg["attribution"].astype(str) + "|" + agg["method"].astype(str) + "|"
                     + agg["lr"].astype(str) + "|" + agg["unit"].astype(str))
    reps = agg[agg["n"] > 1]
    if not reps.empty:
        for (how, meth, lr), g in reps.groupby(["attribution", "method", "lr"], observed=True):
            sel = ((curves["attribution"] == how) & (curves["method"] == meth)
                   & (curves["lr"] == lr))
            runs = sorted(curves[sel]["run"].unique())
            print(f"  {how} | {meth} @ lr {lr:g}: {len(runs)} replicate runs, drawn as mean + "
                  f"range ({', '.join(runs)})")
    band = agg[agg["n"] > 1]

    # one anchor value per (metric, method) panel -- they are all the same base model, so any
    # spread across runs is eval sampling noise on 64 prompts, and the mean is the honest line
    anchor_lines = anchors.groupby(["metric", "method"], observed=True)["value"].mean().reset_index()

    # invisible points at the ends of the fixed ranges, which is how a `free_y` facet is given a
    # y range without also fixing it for the rows that should fit their data
    fixed = {t: rng for t, _, rng in METRICS if rng}
    if args.same_loss_ylim:
        # One range across every loss panel, built the same way the fixed ranges are: pins, not a
        # scale limit, because a limit DROPS the rows outside it (taking a series out of the legend
        # with them) where a pin only widens the panel. The two loss panels are the same quantity on
        # two splits, so a per-panel fit makes a 0.02 wiggle on one look like the 0.4 fall on the
        # other -- and reading train against test is the point of having both.
        loss = curves[curves["metric"].isin([t for t, _, rng in METRICS if not rng])]["value"]
        if len(loss):
            pad = 0.04 * (loss.max() - loss.min())
            for t, _, rng in METRICS:
                if not rng:
                    fixed[t] = (float(loss.min() - pad), float(loss.max() + pad))
            print(f"  loss panels share y: {fixed[[t for t, _, r in METRICS if not r][0]]}")
    pins = pd.DataFrame([
        dict(metric=t, method=m, frac=curves["frac"].min(), value=v)
        for t, rng in fixed.items() for v in rng for m in order])
    pins["metric"] = pd.Categorical(pins["metric"], [t for t, _, _ in METRICS], ordered=True)
    pins["method"] = pd.Categorical(pins["method"], order, ordered=True)

    def layers(data, ribbon, pin, anchors):
        """Everything that is drawn, for one subset. Shared by both layouts."""
        out = []
        if len(pin):
            out.append(geom_blank(pin, aes("frac", "value"), inherit_aes=False))
        if len(anchors):
            out.append(geom_hline(anchors, aes(yintercept="value"), color="#888888",
                                  linetype="dashed", size=0.3))
        if len(ribbon):
            out.append(geom_ribbon(ribbon, aes("frac", ymin="lo", ymax="hi", fill=COLOR),
                                   alpha=0.2, color="none"))
        # line style separates attribution methods; with only one present it would be a legend
        # entry that says nothing, so it is only mapped when there is something to distinguish
        # `group` explicitly, never implicitly: plotnine groups by the DISCRETE aesthetics, so
        # under --lr-gradient (a numeric colour) there are none and all seven curves become one
        # polyline that zigzags back across the x axis between series.
        out.append(geom_line(data, aes(linetype="attribution", group="series"), size=0.4)
                   if len(attrs) > 1 else geom_line(data, aes(group="series"), size=0.4))
        out.append(geom_point(data, size=0.5))
        return out

    def common(guide=True):
        """FRESH scale objects per plot.

        A plotnine scale is stateful -- it is trained with the data range of the plot it belongs to
        -- so reusing one instance across the blocks of a composition has them fight over it, and
        the visible symptom is a legend that renders on neither. Cheap to rebuild; never share.
        """
        if args.lr_gradient:
            # viridis rather than Set1: an ordered axis wants a perceptually ordered ramp, and the
            # breaks are the sweep's own rates so the bar reads as the LRs that were run, not as
            # arbitrary log10 ticks
            lrs = sorted(df["lr"].unique())
            # three breaks, not one per rate: seven labels on a horizontal colourbar overlap into
            # an unreadable smear. The ends plus the geometric middle say the range and the scale.
            ticks = [lrs[0], lrs[len(lrs) // 2], lrs[-1]]
            out = [
                scale_x_log10(breaks=[0.001, 0.01, 0.1, 1.0], labels=["0.1%", "1%", "10%", "100%"]),
                scale_color_cmap(cmap_name="viridis", name=COLOR_TITLE,
                                 breaks=[math.log10(x) for x in ticks],
                                 labels=[lr_label(x) for x in ticks],
                                 **({} if guide else {"guide": None})),
                scale_fill_cmap(cmap_name="viridis", guide=None),
            ]
            return out
        out = [
            scale_x_log10(breaks=[0.001, 0.01, 0.1, 1.0], labels=["0.1%", "1%", "10%", "100%"]),
            scale_color_brewer(type="qual", palette="Set1",
                               **({} if guide else {"guide": None})),
            scale_fill_brewer(type="qual", palette="Set1", guide=None),  # matches the line colours
        ]
        # Only when this block HAS a legend: `guide=False` suppresses the colour guide for the
        # non-first blocks of the composed layout, and asking a suppressed guide for a row count is
        # asking nothing to lay itself out.
        if guide and args.legend_rows > 1:
            kw = {"color": guide_legend(nrow=args.legend_rows)}
            if len(attrs) > 1:
                kw["linetype"] = guide_legend(nrow=args.legend_rows)
            out.append(guides(**kw))
        return out

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.stacked:
        plot = ggplot(agg, aes("frac", "value", color=COLOR))
        for layer in layers(agg, band, pins, anchor_lines):
            plot += layer
        for layer in common():
            plot += layer
        plot += facet_grid("metric ~ method", scales="free_y")
        plot += labs(x="Fraction of Units Kept", y="", color=COLOR_TITLE, linetype="Scores")
        # A one-column figure is narrower than its own legend, so below three columns the two
        # legends stack instead of sitting side by side and the canvas keeps a floor width.
        plot += theme(figure_size=(max(3.9, min(5.9, 1.5 + 1.15 * df["method"].nunique())),
                                   1.0 + 0.72 * len(METRICS)),
                      legend_box="vertical" if df["method"].nunique() < 3 else "horizontal")
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  (dashed grey = pretrained anchor)")
        return

    # --- transposed (the default): metrics across the columns, methods down the rows ---
    #
    # Metrics on the columns means each COLUMN needs its own y scale, and facet_grid frees y per
    # ROW -- so a single grid would put a 0.7-1.6 loss and a fraction in [0, 1] on one axis. With
    # one method that is solved by facet_wrap (y free per panel); with several, the methods have to
    # go on the rows and the figure becomes one BLOCK PER UNIT FAMILY, composed side by side. Same
    # trick, and same reason, as plot_method_lr_grid.py's two halves and plot_train_curves.py's
    # --facet-method.
    fam_of = {t: ("rate" if rng else "loss") for t, _, rng in METRICS}
    titles = [t for t, _, _ in METRICS]
    fams = list(dict.fromkeys(fam_of[t] for t in titles))
    n_rows = df["method"].nunique()

    if n_rows == 1:
        plot = ggplot(agg, aes("frac", "value", color=COLOR))
        for layer in layers(agg, band, pins, anchor_lines):
            plot += layer
        for layer in common():
            plot += layer
        plot += facet_wrap("metric", nrow=1, scales="free_y")
        plot += labs(x="Fraction of Units Kept", y="", color=COLOR_TITLE, linetype="Scores")
        # a colourbar needs width to carry three labels; a categorical legend does not want it
        plot += theme(figure_size=(5.9, 1.9),
                      **({"legend_key_width": 60, "legend_key_height": 5}
                         if args.lr_gradient else {}))
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  (dashed grey = pretrained anchor)")
        return

    fig_w = max(6.8, 1.4 + 1.35 * len(titles))
    Y_LABEL = {"loss": "Loss", "rate": "Fraction"}

    def block(fam, first, last):
        cols = [t for t in titles if fam_of[t] == fam]
        sel = agg["metric"].isin(cols)
        q = ggplot(agg[sel], aes("frac", "value", color=COLOR))
        for layer in layers(agg[sel], band[band["metric"].isin(cols)] if len(band) else band,
                            pins[pins["metric"].isin(cols)] if len(pins) else pins,
                            anchor_lines[anchor_lines["metric"].isin(cols)]):
            q += layer
        for layer in common(guide=first):
            q += layer
        q += facet_grid("method ~ metric", scales="free_y")
        q += labs(x="Fraction of Units Kept" if first else "", y=Y_LABEL[fam],
                  color=COLOR_TITLE, linetype="Scores")
        # the whole composed width on every block: a Beside composition takes its canvas from one
        # part's theme and ignores the others', so each has to name the full size
        # NO per-block theme differences, and the legend is suppressed through the SCALE.
        #
        # In plotnine 0.15.7 adding a theme to a plot mutates the GLOBAL theme (verified: after
        # `p += theme(strip_text_y=element_blank())` a freshly built plot already carries it, and
        # `p = p + theme(...)` leaks identically), and the last write wins for every plot in the
        # figure regardless of what each one asked for. So "blank the row strips on all but the last
        # block" and "legend on the first block only" are both unexpressible as themes -- attempting
        # them produced a composed figure with no legend and no row strips at all, then one with
        # strips on every block and still no legend.
        #
        # What IS per-plot is the scale, so `guide=None` on the colour scale of every block but the
        # first gives exactly one legend. The row strips are simply left on: repeating the method
        # names at the right edge of each block is redundant, not wrong, and the blocks are
        # separated by a gap so it reads as a label per group.
        q += theme(figure_size=(fig_w, 1.15 + 0.5 * n_rows))
        return q

    blocks = [block(f, i == 0, i == len(fams) - 1) for i, f in enumerate(fams)]
    # plotnine's `|` has no width ratios, so N blocks split the canvas evenly however many columns
    # each holds. Folding from the right nests them, giving 1/2, 1/4, ... which at least leaves the
    # widest block (the first) the largest share.
    plot = blocks[-1]
    for b in reversed(blocks[:-1]):
        plot = b | plot
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (dashed grey = pretrained anchor; {n_rows} methods x {len(titles)} "
          f"panels, {len(fams)} family blocks)")


if __name__ == "__main__":
    main()
