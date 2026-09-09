"""Attribution masks across the organisms, at each cell's best-separating sparsity.

One glance at the question the cross-task sweep asks of an attribution method, in units anyone
can read: keep only the top fraction of units the method ranks highest, and report the FRACTION
OF RESPONSES showing the trained habit -- on-target (in-dist, the habit where training put it)
against off-target (the generalisation probe). By default each cell reports the sparsity that
MAXIMISES its on-minus-off gap, picked independently per cell (sparsest wins ties), with that %
printed under the pair: the figure then reads "the best separation this method achieves on this
organism, and the budget it needs". That maximum is an optimistic statistic by construction -- a
max over ~10 conditions, each with ~±0.06 binomial noise -- so a fixed budget for a claim about
budgets is ``--frac``, and anything surprising gets checked against the full sweep curve before
it is quoted.

``--method`` picks what is drawn: ``adam`` (default) is the fitted MAttr mask under the
pre-sweep default recipe, Adam at score_lr 0.05 on a uniform k-schedule (``*_posthoc``);
``stepless_ig`` is stepless IG, one closed-form pass (``*_ixg_mc``); ``both`` overlays them dodged with
distinct shapes. Note what ``adam`` is NOT: the fr2de LR sweep found that default near the worst
of ~20 settings on the off-target log-AUC, so read it as the default fit, not as MAttr's best --
the tuned SGD/log-both cells are a separate comparison.

The x axis is MODEL-MAJOR: one section per model (Qwen leads because every organism has it), the
organisms inside each section under their config-directory names (``fr2de``, ``caps``, ...;
``financial`` is ``bad_medical``'s second persona). A grey dumbbell joins each pair, so a long
segment is sparsity separating the trained behaviour from its generalisation.

Each organism scores its own headline metric (language ``target_frac``, casing ``upper``/
``lower_frac``, spelling ``british_frac``, EM ``misaligned_frac``), so rates are comparable in
KIND while the ceiling stays each finetune's own. In the single-method layout that ceiling is IN
the figure: circles right of each cell are the same splits under the FULL delta, joined to the
mask's triangles by per-split slope lines -- a steep off-target line is the generalisation the
mask sheds, a flat on-target line the habit it keeps, and the two degenerate readings announce
themselves (a cell whose full-delta off-target is already ~0 had no generalisation to localise,
fr2de/OLMo; a triangle above its own circle is the sparse-mask-beats-the-finetune reactivation
pattern). Under ``--method both`` the companions would be a smear and stay in the stdout table.
n = 64 generated responses per point (200 judged for EM).

    uv run python plots/plot_attrib_maxgap.py --method adam --out plots/mattr_adam_maxgap.pdf
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, geom_point, geom_segment, geom_text,
    facet_grid, geom_vline, ggplot, guide_legend, guides, labs, scale_color_manual, scale_fill_manual,
    scale_shape_manual, scale_x_continuous, scale_y_continuous, theme, theme_bw, theme_set,
)

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType outlines, not Type-3

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 2.0),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6.5, rotation=30, hjust=1.0, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="right",
        legend_direction="vertical",
        legend_box_margin=0,
    )
)

#: (run-name prefix -> x label, eval, metric, order). Labels are the config-directory names, and
#: the order groups by habit family: language pairs, casings, spelling, the EM personas -- so
#: the within-family comparison is a neighbour comparison.
ORGANISMS = {
    "fr2de": ("fr2de", "language", "target_frac", 0),
    "fr2ru": ("fr2ru", "language", "target_frac", 1),
    "fr2zh": ("fr2zh", "language", "target_frac", 2),
    "case": ("case", "casing", "lower_frac", 3),
    "caps": ("caps", "casing", "upper_frac", 4),
    "spelling": ("spelling", "spelling", "british_frac", 5),
    "bad_medical_qwen25_14b_lora32": ("medical", "em_fast", "misaligned_frac", 6),
    "bad_medical_qwen25_14b_financial": ("financial", "em_fast", "misaligned_frac", 7),
}

#: the x axis is MODEL-MAJOR: one section per model, its organisms side by side inside it, in
#: the family order above. Qwen leads because it is the model every organism has.
MODELS = {"qwen25_14b": ("Qwen2.5 14B", 0), "gemma2_9b": ("Gemma-2 9B", 1),
          "olmo3_7b": ("OLMo-3 7B", 2)}

#: gap between model sections, in organism-widths, and the within-cell dodge between two
#: methods' dumbbells when --method both.
SECTION_GAP = 1.6
DODGE = 0.19

#: method key -> (label, marker, colour). Order fixes the left-to-right dodge and label
#: stagger. Colours are Wong (adam matches the repo palette's MAttr blue, sgd/log-both its
#: black, stepless IG its sand); the SGD schedules get their own hues here because the palette
#: separates schedules by linetype, which a short vertical dumbbell cannot carry.
METHODS = {"adam": ("MAttr (Adam, default)", "^", "#0072B2"),
           "adam_bs1": ("MAttr (Adam, eff. batch 1)", "^", "#56B4E9"),
           "sgd_log": ("MAttr (SGD, log)", "^", "#009E73"),
           "sgd_log_both": ("MAttr (SGD, log-both)", "^", "#000000"),
           "sgd_logit": ("MAttr (SGD, logit)", "^", "#CC79A7"),
           "adam_higheps": ("MAttr (Adam, eps 1e-2)", "^", "#D55E00"),
           "ixg_base": ("I\u00d7G @ base", "o", "#882255"),
           "ixg_ft": ("I\u00d7G @ finetuned", "o", "#CC6677"),
           "stepless_ig": ("stepless IG", "o", "#E69F00"),
           "ixg_tensor": ("stepless IG (per-tensor)", "o", "#7F5E00"),
           "random": ("random", "o", "#BBBBBB"),
           # THE TUNED ARM. Same fitted method as `adam`, refitted at the hyperparameters the
           # clean fr2de/Qwen-14B sweep selected -- score_lr 0.005, log k, effective batch 1,
           # against the inherited 0.05 / uniform / 16. Its runs are the `adam` run name plus
           # `_best`, so the two are a matched pair on every cell: same delta, same data, same
           # eval, differing only in how the mask was fitted.
           "adam_best": ("MAttr (Adam, tuned)", "^", "#0072B2")}
ALL_METHODS = list(METHODS)

#: off-target is the result, so it keeps the saturated Wong vermillion; on-target is the
#: reference and takes a plain grey, receding so the eye lands on the orange first.
SPLIT_COLOR = {"on-target": "#A8A8A8", "off-target": "#D55E00"}


def run_dir(ixg_dir: Path, method: str) -> Path:
    """The run for a method, from the stepless-IG directory that anchors the cell. One naming
    exception, stated not inferred: the fr2de/Qwen cell predates the sweep's naming and its
    fitted run carries `_posthoc_shard` where the ixg run carries neither."""
    if method == "stepless_ig":
        return ixg_dir
    if method == "ixg_tensor":
        return ixg_dir.parent / (ixg_dir.name + "_tensor")
    if method in ("ixg_base", "ixg_ft"):
        # the endpoint runs share the ixg naming (no `_posthoc` re-insertion, fr2de/Qwen incl.)
        return ixg_dir.parent / ixg_dir.name.replace("_ixg_mc", f"_{method}")
    stem = ixg_dir.name[:-len("_ixg_mc")]
    if stem == "fr2de_qwen25_14b_lr1e-4":
        stem += "_posthoc_shard"
    suffix = {"adam": "", "adam_bs1": "_adam0p005_bs1", "adam_higheps": "_adam_higheps",
              "sgd_log": "_sgd10_log", "sgd_log_both": "_sgd10_log_both",
              "sgd_logit": "_sgd10_logit", "random": "_random",
              "adam_best": "_best"}[method]
    return ixg_dir.parent / (stem + suffix)


def best_frac(blob, ev, met, fixed):
    """The sparsity to report for one run: `--frac` if given, else the argmax of the on-minus-
    off gap over the sweep's conditions, sparsest winning ties (ascending order + strict >)."""
    fracs = sorted(float(k.split("_")[1]) for k in blob if k.startswith("frac_"))
    if fixed is not None:
        if fixed not in fracs:
            raise SystemExit(f"no frac_{fixed:g} in this sweep; conditions are {fracs}")
        return fixed
    pick, best = None, -2.0
    for f in fracs:
        c = blob[f"frac_{f:g}"][ev]
        gap = c["in_dist"][met] - c["off_target"][met]
        if gap > best:
            pick, best = f, gap
    return pick


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=list(METHODS) + ["both", "all"], default="adam",
                   help="which attribution's masks to draw (see the module docstring)")
    p.add_argument("--glob", default="runs/*_ixg_mc",
                   help="stepless-IG run directories; they anchor the cell list even when only "
                        "the Adam twin is drawn. The default catches exactly the one standard-"
                        "budget cell per (organism, model); fr2de's mc512/mc_seed1 stay out")
    p.add_argument("--frac", type=float, default=None,
                   help="report every cell at this fixed sparsity instead of each cell's "
                        "argmax-gap pick (see the module docstring for when that is the "
                        "honest choice)")
    p.add_argument("--pairing", choices=["budget", "split", "facet"], default="budget",
                   help="what each cell's vertical pairs are. 'budget' (default): a grey "
                        "dumbbell joins on- and off-target AT the chosen sparsity, with slope "
                        "lines out to the full-delta companions. 'split': one vertical line PER "
                        "SPLIT -- on-target's sparse-vs-100%% pair beside off-target's -- so "
                        "the line's LENGTH is what the mask sheds of that split, short grey = "
                        "habit kept, long orange = generalisation dropped. 'facet': the same "
                        "lines, but each split in its OWN row -- on-target top, off-target "
                        "bottom -- one line per cell at the cell centre. Single method only "
                        "for both non-default pairings")
    p.add_argument("--y", choices=["rate", "loss"], default="rate",
                   help="'rate' (default): the behavioural expression rate, split rows "
                        "on-/off-target. 'loss': the SFT-loss RECOVERY at the same behaviour-"
                        "chosen budget -- (pretrained - loss) / (pretrained - full_delta), the "
                        "normalisation sparsity_auc.py --recovery uses, 0 at the base model and "
                        "1 at the finetune, which is what makes losses comparable across "
                        "organisms -- split rows train/test. The budget is still picked by the "
                        "BEHAVIOUR gap, so the figure reads 'at the mask that best separates "
                        "the behaviour, how much of the fit survives'. --method all only")
    p.add_argument("--exclude", default=None,
                   help="regex on run-directory names to leave out, for runs that exist but are "
                        "known-broken -- e.g. the bad_medical adam-bs1 pair whose EM judging "
                        "died of API quota mid-run (n_scored 0), which would otherwise draw as "
                        "dramatic near-zero points")
    p.add_argument("--out", default="plots/mattr_adam_maxgap.pdf")
    args = p.parse_args()
    if args.y == "loss" and args.method != "all":
        raise SystemExit("--y loss is wired for the --method all facet layout only")
    if args.pairing != "budget" and args.method == "both":
        raise SystemExit(f"--pairing {args.pairing} needs a single method: its slots per cell "
                         "are the splits, so there is no slot left for a second method.")
    if args.method == "all":
        args.pairing = "facet"  # the only layout with room for five lines a cell
        methods = ALL_METHODS
    elif args.method == "both":
        methods = ["adam", "stepless_ig"]
    else:
        methods = [args.method]
    flip = args.pairing in ("split", "facet")  # the two one-line-per-split layouts
    # Single method: the mask's pair sits LEFT of the cell centre and its full-delta companion
    # pair sits right (see below). Two methods: the methods take the two slots and the
    # full-delta companions are not drawn -- four pairs per cell is a smear, and the ceilings
    # stay in the stdout table.
    with_full = len(methods) == 1 or args.method == "all"
    if args.method == "all":
        step = 0.15 if len(methods) <= 5 else 0.105
        dodge = {m: (i - (len(methods) - 1) / 2) * step for i, m in enumerate(methods)}
    elif with_full:
        dodge = {methods[0]: -DODGE}
    else:
        dodge = {m: (i - (len(methods) - 1) / 2) * 2 * DODGE for i, m in enumerate(methods)}

    cells = []
    for d in sorted(Path().glob(args.glob)):
        org = next((v for k, v in ORGANISMS.items() if d.name.startswith(k)), None)
        model = next((v for k, v in MODELS.items() if k in d.name), None)
        if org is None or model is None:
            print(f"  skip {d.name}: no organism/model mapping")
            continue
        cells.append((model[1], org[3], model[0], org, d))

    rows, breaks, blabels, sections = [], [], [], []
    x0 = 0.0
    for mi in sorted({c[0] for c in cells}):
        sec = sorted(c for c in cells if c[0] == mi)
        for j, (_, _, mlabel, (olabel, ev, met, _), ixg) in enumerate(sec):
            pick = None
            for method in methods:
                run = run_dir(ixg, method)
                if args.exclude and re.search(args.exclude, run.name):
                    print(f"  {olabel}/{mlabel}: {run.name} excluded by --exclude")
                    continue
                # An eval-only re-run (mask_learning_finetuning.eval --run-dir) writes its
                # sweep under `posthoc_eval/`, not at the run root. The four Gemma-2 9B tuned
                # cells are that case: they OOM'd in the MMLU eval AFTER fitting, so the scores
                # were refit-free and only the sweep was redone. Prefer the root file when both
                # exist -- that is a full run, and the nested one would be an older re-eval.
                root = run
                if not (run / "evals.json").exists() and (run / "posthoc_eval" / "evals.json").exists():
                    run = run / "posthoc_eval"
                if not (run / "evals.json").exists():
                    print(f"  {olabel}/{mlabel}: no {run.name}, "
                          f"{METHODS[method][0]} point missing")
                    continue
                blob = json.load(open(run / "evals.json"))["final"]
                # ...but NOT when the root file's judge never ran: the eight bad-medical Qwen
                # I×G@base / I×G@ft / stepless-IG-tensor / random runs 429'd on exhausted
                # credits (2026-08-30) and their root `em_fast` is `n_scored` 0 with
                # `misaligned_frac` 0.0 -- a number that would draw as "no misalignment
                # anywhere". Their re-judged sweep is under `posthoc_eval/`, and wins here.
                def unjudged(b):
                    return any(v.get(ev, {}).get("off_target", {}).get("n_scored") == 0
                               for c, v in b.items() if c.startswith("frac_"))
                if ev == "em_fast" and unjudged(blob) and \
                        (root / "posthoc_eval" / "evals.json").exists():
                    blob = json.load(open(root / "posthoc_eval" / "evals.json"))["final"]
                    if unjudged(blob):
                        raise SystemExit(f"{root}: EM unjudged in both evals.json files")
                    run = root / "posthoc_eval"
                # Conditions swept LATER on the saved mask (the eval CLI with `--fracs` below
                # 0.001, written to `<run>/sparse_eval/`; the 32 Qwen-14B cells, 2026-09-09)
                # join the grid the argmax runs over. Same mask, same eval block, so they are
                # peers of the run's own conditions -- and a cell whose best gap now sits below
                # 0.1% reports that budget.
                sp = root / "sparse_eval" / "evals.json"
                if sp.exists():
                    blob = dict(blob, **{c: v for c, v in json.load(open(sp))["final"].items()
                                         if c.startswith("frac_")})
                pick = best_frac(blob, ev, met, args.frac)
                cond = blob[f"frac_{pick:g}"][ev]
                if args.y == "loss":
                    for split in ("train", "test"):
                        try:
                            pre = blob["pretrained"]["sft_loss"][split]["loss"]
                            ful = blob["full_delta"]["sft_loss"][split]["loss"]
                            y = cond_l = blob[f"frac_{pick:g}"]["sft_loss"][split]["loss"]
                        except KeyError:
                            continue
                        if abs(pre - ful) < 1e-6:
                            continue
                        rows.append(dict(org=olabel, x=x0 + j + dodge[method], cx=x0 + j,
                                         model=mlabel, method=METHODS[method][0], frac=pick,
                                         split=split, rate=(pre - y) / (pre - ful), full=1.0))
                    continue
                for split, slabel in (("in_dist", "on-target"),
                                      ("off_target", "off-target")):
                    rows.append(dict(org=olabel, x=x0 + j + dodge[method], cx=x0 + j,
                                     model=mlabel, method=METHODS[method][0], frac=pick,
                                     split=slabel, rate=cond[split][met],
                                     full=blob.get("full_delta", {}).get(ev, {})
                                              .get(split, {}).get(met)))
            breaks.append(x0 + j)
            # under --pairing split the chosen % rides on the tick label (one method, so one
            # pick per cell); under budget it is drawn inside the panel, per method. Two lines
            # under a VERTICAL rotation (set on the theme below): rotated 90 degrees the lines
            # stack sideways, name beside %, where a 30-degree rotation walked the second line
            # into the neighbouring tick.
            blabels.append(f"{olabel}\n{pick * 100:g}%"
                           if flip and pick is not None and args.method != "all" else olabel)
        sections.append((sec[0][2], x0, x0 + len(sec) - 1))
        x0 += len(sec) + SECTION_GAP

    df = pd.DataFrame(rows)
    df["kind"] = "top-k%"
    # the full-delta companion: the SAME split's rate under the whole finetune, as a second pair
    # right of the cell centre, joined to the mask's pair by a per-split slope line. A mask line
    # steeper down than its blue twin is sparsity shedding the generalisation faster than the
    # habit; a mask point ABOVE its slope-mate is the reactivation pattern, now visible in the
    # figure instead of only in the table.
    slopes = pd.DataFrame()
    if with_full:
        fdf = df.copy()
        fdf["kind"], fdf["rate"] = "full delta", fdf["full"]
        if args.pairing == "split":
            # flipped: the cell's two slots are the SPLITS, each holding its own sparse-vs-full
            # vertical pair -- so both kinds sit at the split's x, and the connector is vertical
            sx = df["split"].map({"on-target": -DODGE, "off-target": DODGE})
            df["x"] = df["cx"] + sx
            fdf["x"] = fdf["cx"] + sx
        elif args.pairing == "facet":
            # each split gets a whole panel row: a single method's line sits at the cell
            # centre, several methods keep their within-cell dodge
            if len(methods) == 1:
                df["x"] = df["cx"]
                fdf["x"] = fdf["cx"]
        else:
            fdf["x"] = fdf["cx"] + DODGE
        slopes = df.merge(fdf[["org", "model", "split", "method", "rate", "x"]],
                          on=["org", "model", "split", "method"], suffixes=("", "_f"))
        df = pd.concat([df, fdf], ignore_index=True)
    # ordered BEFORE the figure captures the data: under --pairing facet this is what puts the
    # on-target row on top. BOTH frames that reach a layer need it -- the facet layout is the
    # union over layer data, and one plain-string frame resets the order to alphabetical.
    split_order = ["train", "test"] if args.y == "loss" else ["on-target", "off-target"]
    for frame in (df, slopes):
        if len(frame):
            frame["split"] = pd.Categorical(frame["split"], categories=split_order,
                                            ordered=True)
    # the dumbbell: one thin grey segment per (cell, method), under the points. Keyed on the
    # cell centre, not on x: under --pairing split the two splits sit at different x, so an x
    # in the pivot key would tear each pair into two half-empty rows.
    seg = (df[df["kind"] == "top-k%"]
             .pivot_table(index=["org", "cx", "model", "method", "frac"], columns="split",
                          values="rate").reset_index()
             .rename(columns={"on-target": "on", "off-target": "off",
                              "train": "on", "test": "off"}))  # loss mode reuses the slots:
                                                               # on=train, off=test
    seg["x"] = seg["cx"] + seg["method"].map({METHODS[m][0]: dodge[m] for m in methods})
    seg["lx"] = seg["cx"] if flip else seg["x"]
    # the chosen sparsity, printed under its dumbbell. A label is ~0.8 x-units wide at this
    # size -- more than the dodge -- so with two methods the labels of a cell can never share a
    # height; and a label hung from its OWN dumbbell's bottom floats mid-panel when that bottom
    # is high. So labels anchor to the CELL's lowest point (shared `cx`), staggered by method in
    # dodge order; with one method there is one shallow row.
    base = df.groupby("cx")["rate"].min()  # over every point of the cell, companions included
    depth = seg["method"].map({METHODS[m][0]: 0.055 + 0.07 * i for i, m in enumerate(methods)})
    seg["ylab"] = seg["cx"].map(base) - depth
    seg["lab"] = [f"{f * 100:g}%" for f in seg["frac"]]
    # section furniture: the model's name centred over its span (in its own headroom band, above
    # the y=1.0 a data point can reach), and a separator in each gap
    # section separators; the model names are drawn OUTSIDE the panel, on the rendered figure
    # (see the save block), so the y range can end just past 1.0 instead of reserving a
    # headroom band inside the panel
    dividers = [lo - (1 + SECTION_GAP) / 2 for _, lo, _ in sections[1:]]
    head_x = [sections[0][1] - 0.55] + dividers  # each section's left edge (panel edge, then
    heads = list(zip((m for m, _, _ in sections), head_x))  # the divider before it)

    # the shape channel carries whichever distinction the figure actually has: mask-vs-full-delta
    # in the with-companion layouts, method-vs-method under --method both
    df["shp"] = df["kind"] if with_full else df["method"]
    shapes = ({"top-k%": "^", "full delta": "o"} if with_full
              else {METHODS[m][0]: METHODS[m][1] for m in methods})
    # ...and the colour channel: the method under --method all (the rows already name the
    # splits), the split otherwise
    grp_of = (lambda f: f["method"]) if args.method == "all" else (lambda f: f["split"])
    df["grp"] = grp_of(df)
    if len(slopes):
        slopes["grp"] = grp_of(slopes)
    grp_colors = ({METHODS[m][0]: METHODS[m][2] for m in methods}
                  if args.method == "all" else SPLIT_COLOR)

    # room below zero only where the in-panel % labels live; the flip has none
    ylo = -0.035 if flip else -0.12 - 0.07 * (len(methods) - 1)
    yhi = 1.045
    fig = (
        ggplot(df, aes("x", "rate"))
        + geom_vline(xintercept=dividers, color="#888888", size=0.35)
    )
    if args.pairing == "budget":  # the on-vs-off dumbbell; under 'split' the lines ARE the pairs
        fig += geom_segment(seg, aes(x="x", xend="x", y="on", yend="off"),
                            color="#999999", size=0.4, inherit_aes=False)
    if with_full:
        # the off-target connector is the figure's argument (how far the mask drops the
        # generalisation), so it gets full weight; the on-target one stays a whisper
        off = slopes["split"] == "off-target"
        fig += geom_segment(slopes[~off], aes(x="x", xend="x_f", y="rate", yend="rate_f",
                                              color="grp"), size=0.3, alpha=0.5,
                            inherit_aes=False, show_legend=False)
        fig += geom_segment(slopes[off], aes(x="x", xend="x_f", y="rate", yend="rate_f",
                                             color="grp"), size=0.6, alpha=0.9,
                            inherit_aes=False, show_legend=False)
    if args.pairing == "budget":  # under 'split' the % rides on the tick labels instead
        fig += geom_text(seg, aes("lx", "ylab", label="lab"), size=4.5, color="#555555",
                         family=FAMILY, inherit_aes=False)
    fig = (
        fig
        # colour on the FILL with a white edge: where sparse and full rates coincide the two
        # markers overlap, and the border is what keeps them readable as two points
        + geom_point(aes(fill="grp", shape="shp"), color="#000000", stroke=0.35,
                     size=1.7 if args.method == "all" else 2.6)
        + scale_color_manual(values=grp_colors)
        + scale_fill_manual(values=grp_colors)
        + scale_shape_manual(values=shapes)
        # the points carry white edges over mapped fills, which would leave the shape key's
        # glyphs white-on-white; give that legend a neutral grey of its own
        + guides(shape=guide_legend(override_aes={"fill": "#666666", "color": "#666666"}))
        + scale_x_continuous(breaks=breaks, labels=blabels, expand=(0, 0.55))
        + scale_y_continuous(limits=((min(0.0, float(df["rate"].min()) - 0.04), 1.1)
                                     if args.y == "loss" else (ylo, yhi)), expand=(0, 0),
                             breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(x="", y=("Loss-reduction recovery" if args.y == "loss"
                        else "Behavioural expression rate"), color="", fill="", shape="")
    )
    if args.pairing == "facet":
        # one row per split, on-target on top (the categorical order was set before the figure
        # captured the data). The row strips name the splits, so the fill legend would say the
        # same thing twice and is dropped; each row keeps its own colour.
        fig = fig + facet_grid("split ~ .")
        if args.method != "all":  # with one method the row strips already say what fill would
            fig += guides(fill="none")
        fig += theme(figure_size=(5.5, 2.1), strip_background=element_blank(),
                     strip_text=element_text(size=7), panel_spacing_y=0.015)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if flip:
        # The tick labels are two vertical lines -- the organism name and, MUTED, its chosen %.
        # A theme colours a tick label wholly or not at all, so these are drawn by hand on the
        # rendered figure. The plotnine labels are kept but PAINTED TRANSPARENT rather than
        # blanked: the layout engine then reserves exactly the right bottom margin for them, so
        # the hand-drawn replacements land in real space instead of being cropped or tight-
        # boxed. `rotation_mode="anchor"` + `ha="right"` puts every line's top end flush
        # against the axis whatever its length.
        fig += theme(axis_text_x=element_text(size=6.5, rotation=90, color="#00000000"))
    mfig = fig.draw(show=False)
    # under 'facet' there are two panel axes, top row first: headers go above the TOP panel,
    # tick labels and the pp annotations belong to the BOTTOM (off-target) one
    ax_top, ax_bot = mfig.axes[0], mfig.axes[-1]
    if flip:
        kw = dict(transform=ax_bot.get_xaxis_transform(), rotation=90, rotation_mode="anchor",
                  ha="right", va="center", fontsize=6.5, family=FAMILY, clip_on=False)
        for x, lab in zip(breaks, blabels):
            name, pct = lab.split("\n") if "\n" in lab else (lab, "")
            ax_bot.text(x - 0.26, -0.015, name, color="#000000", **kw)
            ax_bot.text(x + 0.26, -0.015, pct, color="#999999", **kw)
        # the size of each drop, in percentage points, riding its connector: vertical, centred
        # on the line's midpoint, just to its left. Drops under 10pp go unlabelled -- the text
        # is taller than such a line and the smallness is legible as smallness. The on-target
        # labels take a grey a step darker than the points, or 4.5pt text washes out.
        pp_color = {"off-target": "#D55E00", "on-target": "#8A8A8A"}
        for r in (slopes.itertuples() if args.method != "all" else ()):
            if abs(r.rate - r.rate_f) < 0.10:
                continue
            axp = ax_top if (args.pairing == "facet" and r.split == "on-target") else ax_bot
            axp.text(r.x - 0.22, (r.rate + r.rate_f) / 2,
                     f"{(r.rate - r.rate_f) * 100:+.0f}pp", color=pp_color[r.split],
                     rotation=90, ha="center", va="center", fontsize=4.5, family=FAMILY,
                     clip_on=False)
    # the model names, in the margin ABOVE the (top) panel, flush with their section's left
    # edge -- outside the data region so the panel's own top sits just past a rate of 1.0
    for label, le in heads:
        ax_top.text(le, 1.03, label, transform=ax_top.get_xaxis_transform(), fontsize=7,
                    fontweight="bold", family=FAMILY, ha="left", va="bottom", clip_on=False)
    mfig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}  ({len(seg)} dumbbells, method={args.method})")
    full = (df.pivot_table(index=["org", "model", "method"], columns="split", values="full")
              .reset_index()
              .rename(columns={"on-target": "full_on", "off-target": "full_off",
                               "train": "full_on", "test": "full_off"}))
    with pd.option_context("display.width", 150):
        print(seg.merge(full, on=["org", "model", "method"])
                 .assign(gap=lambda t: t["on"] - t["off"])
                 .sort_values(["model", "org", "method"])
                 [["model", "org", "method", "frac", "on", "off", "gap", "full_on", "full_off"]]
                 .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
