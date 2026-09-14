"""Train and held-out loss over training, per method and LR. Are these runs converged?

Every other figure in this repo reads a run's FINAL numbers. This one reads the trajectory, to
answer the prior question those figures assume away: at the point we stopped, was the loss still
moving? The sweep trains `epochs: 1` (8000 rows, batch 2 x grad_accum 8, so 449 steps -- each
example seen exactly once) under a cosine schedule annealed to zero, which is the regime where
"finished the schedule" and "converged" are easiest to confuse.

Rows are the split, columns the parameterisation, colour the learning rate, and the dashed grey
line is the pretrained anchor -- the same base model for every run, so a curve above that line is
worse than not having trained at all on this measure.

**`--metrics` adds the behaviour panels**, which turns this into `plot_posthoc_curves.py`'s figure
with training steps on the x axis instead of sparsity -- same four panels, same titles, same order,
same size under `--transpose`. They come out of the same `history` entries the losses do (the runner
evaluates the generative eval at every point), so it is a read of data that was always there. The
pair is the point: one figure says where in TRAINING the behaviour appeared, the other says how much
of the finished delta you have to keep to still get it, and reading them together separates "learned
late" from "localised thinly".

    uv run python plots/plot_train_curves.py --dir plots/data/lower8b --metrics casing --transpose \
        --exclude 'posthoc|grpo|ixg|lr5e-4' --out plots/train_curves_lower8b.pdf

**The step-449 point is deliberately excluded.** The scheduled evals use `sft_loss.n_batches` (16)
and the final pass uses `final_n_batches` (200), so the last history entry measures a different
quantity from the 18 before it -- for full SFT at 1e-4 that budget change alone moves test loss
1.183 -> 1.325, which would read as a late overfitting spike that is not there. Plotting only the
16-batch points keeps the y-axis one consistent measurement; `plot_loss_vs_behaviour.py` drops the
final point for the same reason. The 200-batch values are the ones to quote as a run's loss, and
they are printed to stdout rather than drawn.

Two things this view is honest about and a reader should not over-read:

  Loss falling at step 425 is confounded with the LR schedule. Under cosine-to-zero the last
  evals are taken at a near-zero LR, where loss improves partly because updates have stopped
  perturbing the weights. The load-bearing comparison is therefore ACROSS runs at the same step
  (a high-LR run ending above the pretrained anchor has not converged, whatever its schedule did)
  rather than within one run's own tail.

  Held-out loss here is held-out *French SFT* loss, not a capability measure. A run can improve
  it while drifting arbitrarily far on English prompts -- that dissociation is the subject of
  `plot_loss_vs_behaviour.py` and is why this figure cannot stand in for the behavioural ones.

    uv run python plots/plot_train_curves.py --dir plots/data/method_lr_bactrian \
        --out plots/train_curves_bactrian.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, coord_cartesian, element_blank, element_line, element_text, facet_grid, facet_wrap,
    geom_blank, geom_hline, geom_line, geom_point, geom_smooth, geom_vline, ggplot, labs,
    scale_color_brewer, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 2.8),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

#: Panel titles come from :data:`PRESETS` below, and they are ``plot_posthoc_curves.py``'s titles
#: rather than this script's older ones -- the held-out panel is **"Test loss"**, where it used to
#: read "Held-out loss". A companion figure that differs in its panel titles, panel order and legend
#: formatting reads as a different measurement rather than the same one over a different x axis, and
#: `--transpose` exists precisely so the two can be read as a pair. The cost is cosmetic: stacked
#: figures already on disk say "Held-out loss" and will say "Test loss" when regenerated. Nothing
#: about what is drawn changed.

#: how much of the trajectory's tail the printed `tail_drop` covers, in EVAL POINTS. Four is what
#: the 1x sweep's 75-step window came to; expressing it in points keeps it meaningful across
#: cadences, at the cost of covering a different number of STEPS in each sweep -- so compare the
#: column within a sweep, not across two.
TAIL_POINTS = 4

#: What to draw, as ``(panel title, path into a condition's results, FAMILY)``. The family is the
#: unit, and it decides the y scale: panels in different families can never share one, which is why
#: ``--facet-method`` composes one block per family instead of drawing a single facet grid (see
#: :func:`half` in main()). ``plot_method_lr_grid.py`` calls the same idea by the same name.
#:
#: The same panel titles as ``plot_posthoc_curves.py``'s presets, on purpose: with ``--metrics
#: casing --transpose`` this figure is that one with training steps on the x axis instead of
#: sparsity, and a companion figure that renamed its panels would read as a different measurement.
#:
#: ``loss`` is the default and is the two-panel figure this script has always drawn, so every
#: existing invocation is unaffected. The behaviour presets add the two rate panels, which come
#: from the SAME history entries -- the runner already evaluates the generative eval at every point,
#: so this is a read of data that was always there rather than any new work.
PRESETS = {
    "loss": [
        ("Train loss", ("sft_loss", "train", "loss"), "loss"),
        ("Test loss", ("sft_loss", "test", "loss"), "loss"),
    ],
    "language": [
        ("Train loss", ("sft_loss", "train", "loss"), "loss"),
        ("Test loss", ("sft_loss", "test", "loss"), "loss"),
        # `{target}` is filled from the runs' own `eval.language.target` by
        # resolve_language_titles, as in plot_posthoc_curves.py and plot_method_lr_grid.py. A
        # hardcoded "FR" was right for configs/french* and wrong for every other language
        # organism -- fr2de answers in GERMAN -- on the two panels carrying the headline.
        ("In-dist {target}", ("language", "in_dist", "target_frac"), "rate"),
        ("Off-target {target}", ("language", "off_target", "target_frac"), "rate"),
    ],
    "casing": [
        ("Train loss", ("sft_loss", "train", "loss"), "loss"),
        ("Test loss", ("sft_loss", "test", "loss"), "loss"),
        ("In-dist", ("casing", "in_dist", "lower_frac"), "rate"),
        ("Off-target", ("casing", "off_target", "lower_frac"), "rate"),
    ],
    "casing_upper": [
        ("Train loss", ("sft_loss", "train", "loss"), "loss"),
        ("Test loss", ("sft_loss", "test", "loss"), "loss"),
        ("In-dist", ("casing", "in_dist", "upper_frac"), "rate"),
        ("Off-target", ("casing", "off_target", "upper_frac"), "rate"),
    ],
}

#: Panels read from ``train_log.json`` instead of ``evals.json``, as ``{flag key: (title, log key,
#: family)}``. Both are logged **per optimizer step**, so they are ~450 points where a metric panel
#: is ~18 -- a continuous line against a sampled one, on the same x axis.
#:
#: ``batch`` is the loss the optimizer actually saw: the mean over one grad-accumulation window of
#: the examples it was about to step on. **It is not `Train loss`**, and the difference is not
#: noise:
#:
#:   `Train loss` is the ``sft_loss`` eval re-run over a FIXED sample of the train split at each
#:   eval point, in eval mode, at the weights of that moment -- comparable across steps and across
#:   runs by construction, which is what makes it plottable as a trajectory.
#:   `Batch loss` is a different sample every step (each example is seen once in an epoch), measured
#:   before that step's update, in whatever mode training runs in. It is the quantity that was
#:   descended, so it is the one that shows the warmup spike at full amplitude and the only one that
#:   can be compared against a wandb run -- but a bump in it can be a hard batch rather than a worse
#:   model, and it cannot be read across runs at a single step.
#:
#: Having both in one figure is the point: they disagree exactly where the disagreement is
#: informative (warmup, and any late-epoch drift between "this batch is hard" and "the model got
#: worse"), and reading a spike in one against the other says which it was.
#:
#: ``lr`` is the applied schedule, read rather than recomputed from the config -- a recomputation
#: would silently disagree with a run whose warmup or scheduler was edited. It earns a panel because
#: the module docstring's warning that a falling tail is confounded with cosine-to-zero is
#: unfalsifiable while the schedule is invisible.
LOG_PANELS = {
    "batch": ("Batch loss", "loss", "loss"),
    "lr": ("Learning rate", "lr", "lr"),
}

#: y treatment per family. ``None`` fits the data; a tuple pins the range with invisible points.
#: Rates are pinned to [0, 1] rather than fitted for `plot_posthoc_curves.py`'s reason: fitted, an
#: in-dist panel sitting at 0.94-1.00 auto-scales to its own 64-prompt sampling noise and draws
#: three responses either way as a dramatic collapse, next to an off-target panel where the same
#: visual amplitude is the entire result.
FAMILIES = {"loss": dict(range=None, label="Loss"),
            "rate": dict(range=(0.0, 1.0), label="Fraction"),
            "lr": dict(range=None, label="LR")}

#: set from ``--metrics`` in main(), before anything reads it
METRICS = PRESETS["loss"]

#: set from ``--batch-loss`` / ``--lr-panel`` in main(). Ordered: batch loss sits with the other
#: losses and the schedule goes last, since it is context for them rather than a measurement.
LOGS = []

#: True when ``train_log.json`` must be read even though no panel comes from it -- ``--warmup-line``
#: needs the logged schedule to find where warmup ended, and ``--cap-loss`` needs step-0 batch loss.
NEED_LOG = False


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
        if node is None:
            return None
    return node


def lr_label(lr: float) -> str:
    """1e-4 -> '1e-4', 2e-05 -> '2e-5'. Sorts by the float, displays compactly."""
    s = f"{lr:.0e}".replace("e-0", "e-")
    return s.lstrip("0") if s.startswith("0") else s


SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label_sup(lr: float) -> str:
    """``0.0002`` -> ``2×10⁻⁴``, the form plot_posthoc_curves.py's legend uses.

    Unicode superscripts rather than LaTeX, which would break the font -- the same reasoning, and
    the same implementation, as plot_method_lr_grid.py's copy.
    """
    exp, m = 0, float(lr)
    while m < 1:
        m *= 10
        exp -= 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


#: 'meta-llama/Llama-3.1-8B-Instruct' -> '8B'. Only used when a figure spans more than one model.
def model_tag(name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)B", name or "")
    return f"{m.group(1)}B" if m else (name or "?").split("/")[-1]


def method_label(run: Path, meta: dict) -> str:
    """From the resolved config, not the directory name -- `lora.r` is the thing that differs."""
    cfg = run / "config.yaml"
    if meta.get("parameterisation") == "LoRA" and cfg.exists():
        m = re.search(r"^lora:\n(?:\s+\S+:.*\n)*?\s+r:\s*(\d+)", cfg.read_text(), re.M)
        return f"LoRA r{m.group(1)}" if m else "LoRA"
    return {"Direct": "Full SFT", "LoRA": "LoRA"}.get(meta.get("parameterisation"), "mask")


def collect(dirs):
    rows, finals, anchors, total = [], [], {}, None
    models, model_of = set(), {}
    warmups, start_loss = {}, {}
    for run in dirs:
        blob = json.loads((run / "evals.json").read_text())
        meta, hist = blob["meta"], blob.get("history", [])
        total = meta.get("steps", total)
        if not hist:
            print(f"  skip {run.name}: no history (nothing trained, or eval.every == 0)")
            continue
        cfg = (run / "config.yaml").read_text() if (run / "config.yaml").exists() else ""
        m = re.search(r"^\s+lr:\s*([\d.eE+-]+)", cfg, re.M)
        lr = float(m.group(1)) if m else float("nan")
        method = method_label(run, meta)
        models.add(meta.get("model"))
        model_of[method] = meta.get("model")
        # the last entry is the final pass at a LARGER loss budget -- see the module docstring
        traj = [h for h in hist if h["step"] != meta["steps"]]
        for h in traj:
            for title, path, _fam in METRICS:
                v = dig(h["results"]["dense"], path)
                if v is None:
                    continue
                rows.append(dict(run=run.name, method=method, lr=lr, lr_label=lr_label(lr),
                                 step=h["step"], split=title, loss=v))
        # per-step panels, from a different file. Absent for a run whose log was not kept, which is
        # a missing line rather than a failure -- say so instead of drawing a gap silently.
        if LOGS or NEED_LOG:
            log_path = run / "train_log.json"
            if not log_path.exists():
                print(f"  {run.name}: no train_log.json, so no "
                      f"{'/'.join(t for t, _, _ in LOGS) or 'warmup line'} for it")
            else:
                log = json.loads(log_path.read_text())
                for entry in log:
                    for title, key, _fam in LOGS:
                        if entry.get(key) is None:
                            continue
                        rows.append(dict(run=run.name, method=method, lr=lr,
                                         lr_label=lr_label(lr), step=entry["step"],
                                         split=title, loss=entry[key]))
                # The step at which the schedule FIRST reaches its peak, i.e. the last step of
                # warmup. Read from the log rather than taken from `train.warmup_steps`, which is
                # off by one against it (warmup_steps 20 ramps over steps 0..19 and the peak is
                # logged at 19) -- and which would also miss a run whose scheduler was changed.
                lrs = [(e["step"], e["lr"]) for e in log if e.get("lr") is not None]
                if lrs:
                    peak = max(x[1] for x in lrs)
                    warmups[run.name] = min(st for st, v in lrs if v == peak)
                if log and log[0].get("loss") is not None:
                    start_loss[run.name] = log[0]["loss"]
        # The pretrained anchor: step 0 is evaluated before any update, so it is the base model.
        # Keyed by METHOD, not global, because `train.dtype` covaries with the column -- a Direct
        # run must be fp32 while a LoRA run can be bf16 (train/params.py), and a bf16 base forward
        # gives a measurably different loss on identical data (2.238460 vs 2.237166 on the casing
        # set). One global anchor would either abort on that or average two different measurements.
        # History panels only: a train-log panel has no "before any update" value to anchor against
        # (the schedule's step 0 is warmup, and batch loss at step 0 is one arbitrary batch), so
        # those panels get no dashed line rather than a meaningless one.
        for title, path, _fam in METRICS:
            v = dig(hist[0]["results"]["dense"], path)
            if v is not None:
                anchors.setdefault((method, title), []).append(v)
        fin = blob["final"].get("dense", {})
        # The behavioural number is printed beside the losses only so a reader of this stdout can
        # see which runs drifted; the figure never uses it. Which eval carries it depends on the
        # organism, so try each -- on a casing run `language` is absent and reading only that one
        # printed `off_target None` for every 8B cell, hiding the diverged one.
        # `is not None`, not `or`: a LoRA French cell's off-target rate is exactly 0.0, which `or`
        # would fall through as if the eval were missing.
        beh = next((v for v in (fin.get("language", {}).get("off_target", {}).get("target_frac"),
                                fin.get("casing", {}).get("off_target", {}).get("lower_frac"))
                    if v is not None), None)
        # Counted in EVAL POINTS, not in steps: a fixed 75-step window is 4 points at the 1x
        # sweep's `eval.every: 25` but only the final one at the 4x sweep's 100, where it silently
        # printed a tail drop of exactly 0.000 for all eight runs -- a "these are converged"
        # reading of a window that held one measurement.
        te = [h["results"]["dense"]["sft_loss"]["test"]["loss"] for h in traj[-TAIL_POINTS:]]
        finals.append(dict(method=method, lr=lr_label(lr),
                           final_train=fin.get("sft_loss", {}).get("train", {}).get("loss"),
                           final_test=fin.get("sft_loss", {}).get("test", {}).get("loss"),
                           last_traj_test=te[-1], tail_drop=te[0] - te[-1],
                           off_target=beh))
    if len(models) > 1:
        ren = {m: f"{model_tag(model_of[m])} {m}" for m in model_of}
        for r in rows:
            r["method"] = ren[r["method"]]
        for f in finals:
            f["method"] = ren[f["method"]]
        anchors = {(ren[m], s): v for (m, s), v in anchors.items()}
    # Which panels are a FORWARD PASS (deterministic to float noise) and which are a GENERATION
    # (not). The strict agreement check below only applies to the first kind.
    fitted = {t: FAMILIES[fam]["range"] for t, _, fam in METRICS}
    anchor = {}
    for (method, split), vals in anchors.items():
        spread = max(vals) - min(vals)
        if fitted.get(split) is None:
            # within one method every run shares dtype and split, so step 0 must agree to float
            # noise -- a loss is a forward pass over fixed data
            if spread > 1e-4:
                raise ValueError(f"step-0 {split} differs by {spread:.4g} across the {method} "
                                 f"runs, which share a dtype -- so they do not share a base model "
                                 f"or a loss budget, and the anchor is meaningless")
        elif spread > 1e-9:
            # A RATE is a percentage over 64 generated responses, so one response differing is
            # 0.0156 of spread and the strict check above would abort the whole figure over it.
            # vLLM's batching is not bit-reproducible across runs, so this is expected rather than
            # suspicious -- report it and use the mean, which is what the ribbon-less line can say.
            print(f"  NOTE step-0 {split} varies by {spread:.4g} across the {method} runs "
                  f"({', '.join(f'{v:.4f}' for v in vals)}) -- generated rates are not bit-"
                  f"reproducible; the anchor is their mean")
        anchor[(method, split)] = sum(vals) / len(vals)
    # Across methods that share a MODEL a gap can only be dtype, so anything larger means the
    # runs are not on the same data. Across models it is just the models, and saying so would be
    # noise -- a 1B and an 8B have no reason to agree on the loss of identical text.
    for split in {s for _, s in anchor} if len(models) == 1 else ():
        by = {m: v for (m, s), v in anchor.items() if s == split}
        if len(by) > 1 and max(by.values()) - min(by.values()) > 5e-3:
            print(f"  NOTE {split} anchor differs by "
                  f"{max(by.values()) - min(by.values()):.4g} across methods "
                  f"({', '.join(f'{m} {v:.4f}' for m, v in by.items())}) -- larger than a dtype "
                  f"difference, so check they trained on the same data")
    return pd.DataFrame(rows), pd.DataFrame(finals), anchor, total, warmups, start_loss


def report(anchor, finals):
    """The stdout half of this script, shared by both layouts.

    The 200-batch final losses are printed rather than drawn, because they are a different
    measurement from the trajectory's 16-batch points -- see the module docstring.
    """
    print("pretrained anchor: " + ", ".join(f"{m}/{s} {v:.4f}"
                                            for (m, s), v in sorted(anchor.items())))
    print(f"\nfinal pass (200 batches), and how far held-out loss still fell over the last "
          f"{TAIL_POINTS} eval points of the 16-batch trajectory:")
    print(finals.sort_values(["method", "lr"]).to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))


def resolve_language_titles(roots, metrics) -> list:
    """Fill ``{target}`` in the metric titles from the runs' own ``eval.language.target``.

    The third copy of this, after ``plot_posthoc_curves.py`` and ``plot_method_lr_grid.py``, and a
    copy for the same reason their presets are copies: these scripts share no module, and the
    alternative to twenty duplicated lines is a new import edge between three figure scripts.

    The answer language is a per-experiment config value, not a property of the figure:
    `configs/french/` answers in French, `configs/fr2de/` in GERMAN, the nine Bactrian languages
    each in their own. Runs that disagree are a hard error -- one y axis with two answer languages
    is two experiments drawn as one -- and a preset without the placeholder is untouched.
    """
    if not any("{target}" in t for t, *_ in metrics):
        return metrics
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
        raise SystemExit(f"these runs answer in {sorted(targets)}, so one y axis would mean two "
                         "different languages. Plot them separately.")
    code = targets.pop().upper() if targets else ""
    if code:
        print(f"  answer language: {code}")
    return [((t.replace(" {target}", f" {code}") if code else t.replace(" {target}", "")), *rest)
            for t, *rest in metrics]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", default=["plots/data/method_lr_bactrian"],
                   help="one or more directories of run subdirectories, each with evals.json and "
                        "config.yaml. Passing several is how a figure spans models: the facet label "
                        "then carries the parameter count, and each column keeps its own pretrained "
                        "anchor, which for two different models is legitimately a different number")
    p.add_argument("--exclude", default="posthoc|grpo|ixg",
                   help="regex on run name; mask-fitting runs train nothing and have no curve")
    p.add_argument("--out", default="plots/train_curves_bactrian.pdf")
    p.add_argument("--metrics", default="loss", choices=sorted(PRESETS),
                   help="which panels to draw. 'loss' (default) is train + held-out loss, the two "
                        "this figure has always had. The others add the organism's two behaviour "
                        "panels from the same history entries, giving exactly "
                        "plot_posthoc_curves.py's four panels with training steps on the x axis "
                        "instead of sparsity: 'language' (configs/french*), 'casing' "
                        "(configs/lower, lower_frac) and 'casing_upper' (configs/caps, upper_frac)")
    p.add_argument("--transpose", action="store_true",
                   help="splits across the columns instead of down the rows -- a wide, short figure "
                        "for a slide. Single method only, same restriction and same reason as "
                        "plot_posthoc_curves.py's flag")
    p.add_argument("--ylim", nargs=2, type=float, metavar=("LO", "HI"),
                   help="zoom the y axis to LO HI. A diverged run then runs off the top of the "
                        "panel instead of flattening every healthy curve into the bottom fifth -- "
                        "and it stays IN the figure, which excluding it (--exclude) does not: the "
                        "line leaving the panel is itself the statement that the run left the "
                        "scale. Implemented with coord_cartesian, so this CLIPS rather than drops "
                        "-- a scale limit would discard those rows and take the run's colour out "
                        "of the legend with them. One range for all panels, so it overrides the "
                        "per-row free_y; check both rows fit before using it")
    p.add_argument("--points", action="store_true",
                   help="mark each eval point as well as drawing the line. Off by default: the "
                        "markers say nothing the line does not (the eval grid is uniform, so their "
                        "spacing carries no information) and at the 4x budget's 17 points on a "
                        "1698-step axis they crowd the line. Turn them on when the grid is sparse "
                        "enough that where a curve was actually measured is worth seeing")
    p.add_argument("--batch-loss", action="store_true",
                   help="add the ACTUAL training loss -- the per-step loss the optimizer descended, "
                        "from train_log.json -- beside the eval-on-train-split 'Train loss'. They "
                        "are different measurements, not two views of one; see LOG_PANELS")
    p.add_argument("--lr-panel", action="store_true",
                   help="add the applied learning-rate schedule as a panel, also per step from "
                        "train_log.json. Makes the warmup spike and the cosine tail visible next to "
                        "the losses they confound")
    p.add_argument("--drop", nargs="+", default=(), metavar="PANEL",
                   help="panel titles to leave out, e.g. --drop 'Train loss'. Useful once "
                        "--batch-loss is on, where the eval-on-train-split panel is largely "
                        "redundant with it. Unknown titles are an error listing the valid ones, "
                        "rather than a silently unchanged figure")
    p.add_argument("--smooth", action="store_true",
                   help="LOWESS-smooth the per-step Batch loss panel, drawing the raw line faintly "
                        "underneath so the smoothing never hides the noise it is summarising. "
                        "Applies to that panel only -- the ~18-point eval curves have nothing to "
                        "smooth and a fitted line through them would invent structure")
    p.add_argument("--smooth-span", type=float, default=0.15,
                   help="LOWESS span for --smooth, as a fraction of the x range. 0.15 over ~450 "
                        "steps averages ~68 of them: enough to kill the per-batch noise, short "
                        "enough to keep the warmup spike, which a wider span flattens into the "
                        "thing it exists to show")
    p.add_argument("--cap-loss", action="store_true",
                   help="cap the loss panels' upper y at --cap-factor x the largest step-0 batch "
                        "loss across the plotted runs. That reference is the loss BEFORE any "
                        "update, so it is a meaningful ceiling rather than a chosen one: above it a "
                        "run has made the model worse than it started. Diverged runs then leave the "
                        "panel instead of flattening every healthy curve, and they stay in the "
                        "figure and the legend (coord_cartesian clips, it does not drop). Needs "
                        "--batch-loss for the reference value")
    p.add_argument("--cap-factor", type=float, default=1.5, metavar="X",
                   help="headroom above the step-0 reference, as a multiple (default 1.5). At "
                        "exactly 1.0 the ceiling sits ON the pre-training loss, which clips the "
                        "warmup excursion of every healthy run as well as the diverged ones -- and "
                        "that excursion is a real part of the trajectory, not an outlier. 1.5 keeps "
                        "it in frame while still cutting a collapse to ~7 out")
    p.add_argument("--warmup-line", action="store_true",
                   help="vertical dashed line where the LR schedule first reaches its peak, i.e. "
                        "the end of warmup. Read from each run's logged schedule and required to "
                        "AGREE across the plotted runs -- if it does not, the runs do not share a "
                        "schedule, one line would be wrong for some of them, and none is drawn")
    p.add_argument("--facet-method", action="store_true",
                   help="methods down the ROWS and panels across the columns, one block per unit "
                        "family. The layout for a figure that spans parameterisations -- full SFT "
                        "beside each LoRA rank -- where --transpose takes a single method only")
    args = p.parse_args()

    global METRICS, LOGS, NEED_LOG
    # METRICS stays HISTORY-ONLY (collect() walks its paths into evals.json) and LOGS is the
    # train_log side; `panels` is their ordered union and is what the layout reads. Keeping the two
    # sources apart is what lets each one be read with the right accessor and anchored correctly.
    METRICS = resolve_language_titles(args.dir, PRESETS[args.metrics])
    LOGS = ([LOG_PANELS["batch"]] if args.batch_loss else []) + \
           ([LOG_PANELS["lr"]] if args.lr_panel else [])
    NEED_LOG = args.warmup_line or args.cap_loss
    if args.cap_loss and not args.batch_loss:
        raise SystemExit("--cap-loss caps at the step-0 BATCH loss, so it needs --batch-loss "
                         "(the eval-on-train-split panel's step-0 value is a different "
                         "measurement, and using it would put the cap in the wrong place).")
    # batch loss sits with the other losses, the schedule goes last -- it is context, not a metric
    panels = ([LOG_PANELS["batch"]] if args.batch_loss else []) + list(METRICS) + \
             ([LOG_PANELS["lr"]] if args.lr_panel else [])
    if args.drop:
        known = [t for t, _, _ in panels]
        unknown = [d for d in args.drop if d not in known]
        if unknown:
            raise SystemExit(f"--drop got unknown panel(s) {unknown}; this figure has "
                             f"{known}. Titles are case-sensitive and quoted.")
        panels = [e for e in panels if e[0] not in args.drop]
        METRICS = [e for e in METRICS if e[0] not in args.drop]
        LOGS = [e for e in LOGS if e[0] not in args.drop]
        if not panels:
            raise SystemExit("--drop removed every panel")
    rate_panels = [t for t, _, fam in panels if FAMILIES[fam]["range"]]
    # The cap is applied per family BLOCK, which only exists under --facet-method; in the single
    # facet_wrap/facet_grid figure coord_cartesian is global and would clip the rate panels out
    # entirely (the same trap --ylim is refused for). Rejecting is better than the silent no-op this
    # was before, where --cap-loss printed a cap it never applied.
    if args.cap_loss and rate_panels and not args.facet_method:
        raise SystemExit("--cap-loss with behaviour panels needs --facet-method: the cap has to "
                         "apply to the loss panels only, and that requires the per-family blocks "
                         "that --facet-method composes.")

    # coord_cartesian is GLOBAL across facets, so one --ylim would apply to the rate panels too and
    # a loss zoom of 0.85-2.25 would push every fraction off the bottom of its panel -- an empty
    # panel that looks like a missing measurement. Refuse rather than draw it.
    if args.ylim and rate_panels:
        raise SystemExit(
            f"--ylim cannot be combined with --metrics {args.metrics}: it is one range for all "
            f"panels (coord_cartesian), and {', '.join(rate_panels)} are fractions in [0, 1] that "
            f"would be clipped out of the figure entirely. Drop the diverged run with --exclude "
            f"instead -- with no outlier there is nothing to zoom past.")

    dirs = sorted((d for root in args.dir for d in Path(root).iterdir()
                   if (d / "evals.json").exists() and not re.search(args.exclude, d.name)),
                  key=lambda d: d.name)
    if not dirs:
        raise SystemExit(f"no runs in {args.dir} after --exclude {args.exclude!r}")
    df, finals, anchor, total, warmups, start_loss = collect(dirs)

    # transposed adopts the sibling figure's conventions -- see POSTHOC_SPLITS
    label = lr_label_sup if args.transpose else lr_label
    df["lr_label"] = [label(v) for v in df["lr"]]
    order = [label(v) for v in sorted(df["lr"].unique())]
    df["lr_label"] = pd.Categorical(df["lr_label"], categories=order, ordered=True)
    # Full SFT first, then LoRA by rank -- reading order should be "fewest constraints first"
    def col_key(s):
        m = re.match(r"([\d.]+)B ", s)
        size = float(m.group(1)) if m else 0.0
        rest = s[m.end():] if m else s
        return (size, rest != "Full SFT", int(re.sub(r"\D", "", rest) or 0))
    meths = sorted(df["method"].unique(), key=col_key)
    df["method"] = pd.Categorical(df["method"], categories=meths, ordered=True)
    # Transposed keeps the preset's order, which is plot_posthoc_curves.py's: losses first, then the
    # behaviour panels, reading left to right. Stacked REVERSES it, so held-out loss sits above
    # train (it is the split the convergence question turns on, and the row a reader hits first) and
    # the behaviour rows sit above both. Left-to-right has no equivalent "first" position, which is
    # why the two layouts legitimately differ here.
    titles = [t for t, _, _ in panels]
    # Columns keep the panel order; ROWS reverse it, so held-out loss sits above train (it is the
    # split the convergence question turns on, and the row a reader hits first) and the behaviour
    # rows sit above both. Left-to-right has no equivalent "first" position, which is why the two
    # layouts legitimately differ here. --facet-method puts panels on the columns, so it keeps the
    # order too.
    split_order = titles if (args.transpose or args.facet_method) else titles[::-1]
    df["split"] = pd.Categorical(df["split"], categories=split_order, ordered=True)
    ref = pd.DataFrame([dict(method=m, split=s, loss=v) for (m, s), v in anchor.items()])
    ref["method"] = pd.Categorical(ref["method"], categories=meths, ordered=True)
    ref["split"] = pd.Categorical(ref["split"], categories=split_order, ordered=True)

    if args.transpose and args.facet_method:
        raise SystemExit("--transpose and --facet-method are two answers to the same question "
                         "(what goes on the columns); pick one. --facet-method is the one that "
                         "handles more than one method.")
    if args.transpose and df["method"].nunique() > 1:
        raise SystemExit(
            f"--transpose needs a single method, got {sorted(df['method'].unique())}. The splits "
            "on the columns means each column needs its own y scale, which facet_grid cannot do "
            "(it frees y per row); narrow the input, or use --facet-method, which composes one "
            "block per unit family and so can put the methods on the rows.")

    # Invisible points at the ends of each rate panel's fixed range: how a `free_y` facet is given a
    # y range without also fixing it for the loss panels, which should fit their data.
    # plot_posthoc_curves.py calls these `pins` and does the same thing for the same reason.
    pins = pd.DataFrame([dict(split=t, method=m, step=df["step"].min(), loss=v)
                         for t, _, fam in panels if FAMILIES[fam]["range"]
                         for v in FAMILIES[fam]["range"] for m in meths])
    if len(pins):
        pins["split"] = pd.Categorical(pins["split"], categories=split_order, ordered=True)
        pins["method"] = pd.Categorical(pins["method"], categories=meths, ordered=True)

    # --- the three per-figure extras, resolved once and applied by both layouts ---
    BATCH = LOG_PANELS["batch"][0]

    # The warmup line has to be ONE x position for the whole figure, so it is only drawn when every
    # plotted run agrees on it. Disagreement means the runs do not share a schedule, and a single
    # line would then be wrong for some of them -- which is worse than no line, because a reader
    # cannot tell by looking.
    warmup_at = None
    if args.warmup_line:
        vals = sorted(set(warmups.values()))
        if not vals:
            print("  --warmup-line: no run had a logged schedule, so no line is drawn")
        elif len(vals) > 1:
            by = {}
            for run, st in warmups.items():
                by.setdefault(st, []).append(run)
            print("  --warmup-line: the plotted runs do NOT share a warmup length "
                  + "; ".join(f"step {st}: {', '.join(sorted(r))}" for st, r in sorted(by.items()))
                  + " -- no line drawn, since one position would be wrong for some of them")
        else:
            warmup_at = vals[0]
            print(f"  warmup ends at step {warmup_at} (peak LR), shared by all "
                  f"{len(warmups)} plotted runs")

    # The cap: the largest pre-update batch loss across the plotted runs. Above that line a run has
    # made the model worse than it started, so it is a principled ceiling rather than a chosen one.
    loss_cap = None
    if args.cap_loss:
        if not start_loss:
            print("  --cap-loss: no step-0 batch loss available, so no cap is applied")
        else:
            worst = max(start_loss, key=start_loss.get)
            ref_loss = start_loss[worst]
            loss_cap = ref_loss * args.cap_factor
            print(f"  loss panels capped at {loss_cap:.3f} = {args.cap_factor:g} x {ref_loss:.3f} "
                  f"(step-0 batch loss, max over runs, from {worst}); lines above it leave the "
                  f"panel and are not dropped")

    def loss_layers(sub):
        """geom_line for every panel, plus the smoothed overlay on the batch-loss one."""
        out = []
        if args.smooth and (sub["split"] == BATCH).any():
            plain, noisy = sub[sub["split"] != BATCH], sub[sub["split"] == BATCH]
            if len(plain):
                out.append(geom_line(plain, size=0.4))
            # raw underneath at low alpha, then the LOWESS fit on top: the summary never hides the
            # spread it is summarising, which for a per-step loss is most of the information
            out.append(geom_line(noisy, size=0.3, alpha=0.25))
            out.append(geom_smooth(noisy, method="lowess", se=False, size=0.6,
                                   span=args.smooth_span))
        else:
            out.append(geom_line(sub, size=0.4))
        if args.points:
            out.append(geom_point(sub, size=0.5))
        return out

    if args.facet_method:
        fam_of = {t: fam for t, _, fam in panels}
        # one block per family, in panel order, because a y scale cannot span two units and
        # facet_grid frees y per ROW only -- with panels on the columns that is the wrong axis.
        # Same composition trick, and same reason, as plot_method_lr_grid.py's two halves.
        fams = list(dict.fromkeys(fam_of[t] for t in titles))
        n_rows = len(meths)
        fig_w = max(6.8, 1.4 + 1.35 * len(titles))

        def half(fam, first, last):
            cols = [t for t in titles if fam_of[t] == fam]
            sub = df[df["split"].isin(cols)]
            q = (
                ggplot(sub, aes("step", "loss", color="lr_label"))
                + geom_hline(ref[ref["split"].isin(cols)], aes(yintercept="loss"),
                             color="#888888", linetype="dashed", size=0.3)
                + facet_grid("method ~ split", scales="free_y")
                + scale_color_brewer(type="qual", palette="Set1",
                                     **({} if first else {"guide": None}))
                + labs(x=f"Training step (1 epoch = {total})" if first else "",
                       y=FAMILIES[fam]["label"], color="Learning rate")
                # the whole composed width, on every block: a Beside composition takes its canvas
                # from one part's theme and ignores the others', so each has to name the full size
                + theme(figure_size=(fig_w, 1.15 + 0.5 * n_rows))
            )
            for layer in loss_layers(sub):
                q += layer
            if warmup_at is not None:
                q += geom_vline(xintercept=warmup_at, color="#888888", linetype="dashed",
                                size=0.3)
            pin = pins[pins["split"].isin(cols)] if len(pins) else pins
            if len(pin):
                q += geom_blank(pin, aes("step", "loss"), inherit_aes=False)
            # the cap is a LOSS-family statement; the rate panels are already pinned to [0, 1]
            if loss_cap is not None and fam == "loss":
                q += coord_cartesian(ylim=(float(sub["loss"].min()), loss_cap))
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
            return q

        blocks = [half(f, i == 0, i == len(fams) - 1) for i, f in enumerate(fams)]
        # plotnine's `|` has NO width ratios (checked: Beside takes a plain list and the gridspec is
        # even), so N blocks each get 1/N of the width however many columns they hold -- a
        # single-column LR block came out as wide as a three-column loss block. Nesting is the only
        # lever available: `a | (b | c)` gives 1/2, 1/4, 1/4 instead of thirds, which is closer to
        # the column counts. Fold from the right so the WIDEST block (the losses, first) keeps the
        # largest share.
        fig = blocks[-1]
        for b in reversed(blocks[:-1]):
            fig = b | fig
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.save(out, dpi=300, verbose=False)
        print(f"wrote {out}  ({len(dirs)} runs, {len(meths)} methods x {len(titles)} panels, "
              f"{len(fams)} family blocks: {', '.join(fams)})")
        report(anchor, finals)
        return

    fig = (
        ggplot(df, aes("step", "loss", color="lr_label"))
        + geom_hline(ref, aes(yintercept="loss"), color="#888888", linetype="dashed", size=0.3)
        + geom_line(size=0.4)
        # facet_wrap when transposed, for the same reason plot_posthoc_curves.py does it: with the
        # splits across the columns, `free_y` on a facet_grid would free them per ROW and so put
        # train and held-out on one shared axis. facet_wrap frees y per panel.
        + (facet_wrap("split", nrow=1, scales="free_y") if args.transpose
           else facet_grid("split ~ method", scales="free_y"))
        + scale_color_brewer(type="qual", palette="Set1")
        # y is unlabelled once the panels carry different units (loss and a fraction), exactly as
        # in plot_posthoc_curves.py; the two-panel loss figure keeps its "Loss"
        + labs(x=f"Training step (1 epoch = {total})", y="" if rate_panels else "Loss",
               color="Learning rate")
    )
    # only when there is something to pin: an empty frame has no columns, so the layer's own `step`
    # mapping cannot be evaluated and plotnine fails on the whole figure
    if len(pins):
        fig += geom_blank(pins, aes("step", "loss"), inherit_aes=False)
    if args.points:
        fig += geom_point(size=0.5)
    if args.ylim:
        fig += coord_cartesian(ylim=tuple(args.ylim))
    if args.transpose:
        # 5.9 x 1.9in and the smaller x tick text are plot_posthoc_curves.py's numbers, not new
        # ones: the two figures are meant to be read as a pair, so they are the same size
        fig += theme(figure_size=(5.9, 1.9), axis_text_x=element_text(size=5.5, rotation=45,
                                                                     hjust=0.5, vjust=1.0))
    elif rate_panels:
        # stacked with four rows needs the height the two-row default does not
        fig += theme(figure_size=(5.5, 1.0 + 0.72 * len(METRICS)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=300, verbose=False)
    print(f"wrote {out}  ({len(dirs)} runs, {df['step'].nunique()} x positions)")
    report(anchor, finals)


if __name__ == "__main__":
    main()
