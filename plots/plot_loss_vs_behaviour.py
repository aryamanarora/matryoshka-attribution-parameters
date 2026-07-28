"""Behaviour against loss, traced over training: does the objective predict the behaviour?

The sparsity figures ask "how much of the mask do you need". This asks a different question with
the same data: as each run trains, how does its French-response rate move as a function of the
loss it has reached? If the objective explained the behaviour, every method would fall on one
curve -- reaching loss L would imply a French rate, regardless of how it got there.

Rows are the loss used as the x-axis (train, then held-out). Columns are the eval:

  in-dist (FR prompts)    the positive control -- pinned near 100% from the start, so it can
                          only show that nothing broke.
  off-target (EN prompts) the generalisation probe. This is the panel to read.

Each point is one eval step and the path runs in step order. The paths **hook**: for a
co-trained mask, frac_1 loss gets sharply WORSE over the first 50 steps (1.55 -> 3.07 at lr 1e-4)
before improving, because the delta is calibrated for partial application under sampled k and
applying all of it overshoots. So the trajectory goes right, then back left -- it is not a
monotone left-to-right read.

A post-hoc run contributes a single point, not a trajectory, and that is structural rather than a
bug: its delta is frozen, so the frac_1 condition does not depend on the scores at all. What a
post-hoc run learns is the *ranking*, which is only visible at frac < 1. This view is blind to
it by construction.

Two modes, same axes, different swept variable -- and they answer different questions:

``--mode training``  every point is at ``frac_1`` (all units live, the model actually being
    trained) and the swept variable is the TRAINING STEP, encoded as point size. Asks: along one
    run's own training, does loss track behaviour? Points come only from the *scheduled* evals,
    dropping the final one, since the final pass uses a bigger loss budget
    (``final_n_batches``) and mixing budgets would put a spurious jump at the end of each path.

``--mode sparsity``  every point is from the FINAL sweep and the swept variable is the MASK
    FRACTION, encoded as point size. Asks: does a sparse slice that reaches loss L produce the
    behaviour that loss usually comes with? This is the mode a post-hoc run is visible in, since
    its ranking only acts at frac < 1.

    uv run python plots/plot_loss_vs_behaviour.py --mode sparsity \\
        --run "nonresid 1e-4=plots/data/sweep/nonresid_lr1e-4/evals.json" \\
        --run "post-hoc=plots/data/french/french_mask_posthoc.json" \\
        --out plots/loss_vs_behaviour_sparsity.pdf
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_path, geom_point, ggplot,
    labs, scale_color_brewer, scale_size_continuous, scale_y_continuous, theme, theme_bw,
    theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 3.4),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

TRAIN, TEST = "x: train loss", "x: held-out loss"
IN_DIST, OFF_TARGET = "in-dist (FR prompts)", "off-target (EN prompts)"
COND = "frac_1"          # all units live: the model actually being trained


def at(res, ev, split, metric, label=COND):
    return (((res.get(label) or {}).get(ev) or {}).get(split) or {}).get(metric)


def _emit(rows, label, res, cond, sweep, *, size_val):
    """Append the four (x_kind, y_kind) combinations for one condition."""
    losses = {TRAIN: at(res, "sft_loss", "train", "loss", cond),
              TEST: at(res, "sft_loss", "test", "loss", cond)}
    evals = {IN_DIST: at(res, "language", "in_dist", "target_frac", cond),
             OFF_TARGET: at(res, "language", "off_target", "target_frac", cond)}
    for xk, x in losses.items():
        for yk, y in evals.items():
            if x is None or y is None:
                continue
            rows.append(dict(run=label, sweep=sweep, size_val=size_val, x_kind=xk,
                             y_kind=yk, x=x, y=100.0 * y))


def load(spec, mode):
    label, _, path = spec.partition("=")
    blob = json.loads(Path(path).read_text())
    rows = []
    if mode == "training":
        seen = set()
        for pt in blob["history"]:
            step = pt["step"]
            if step in seen:
                # the final pass repeats the last step at a different loss budget; keep the
                # scheduled one so every point on the path is the same measurement
                continue
            seen.add(step)
            _emit(rows, label, pt["results"], COND, step, size_val=step)
    else:
        res = blob.get("final") or blob["history"][-1]["results"]
        fracs = sorted({float(k[len("frac_"):]) for k in res if k.startswith("frac_")})
        for f in fracs:
            # size on a log scale: the grid spans 0.1% to 100%, so a linear map would make
            # everything below 20% indistinguishable
            _emit(rows, label, res, f"frac_{f:g}", f, size_val=math.log10(f))
    if not rows:
        raise SystemExit(
            f"{path}: no points with both sft_loss and language in mode={mode!r}. In sparsity "
            "mode the language eval must have run across the grid, which needs "
            "eval.sweep_when auto or every-eval.")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=evals.json")
    p.add_argument("--mode", default="training", choices=["training", "sparsity"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = f"plots/loss_vs_behaviour_{args.mode}.pdf"

    df = pd.DataFrame([r for spec in args.run for r in load(spec, args.mode)])
    order = [s.partition("=")[0] for s in args.run]
    df["run"] = pd.Categorical(df.run, order)
    df["x_kind"] = pd.Categorical(df.x_kind, [TRAIN, TEST])
    df["y_kind"] = pd.Categorical(df.y_kind, [IN_DIST, OFF_TARGET])
    df = df.sort_values(["run", "x_kind", "y_kind", "sweep"])

    if args.mode == "training":
        breaks = sorted({0, *(int(v) for v in df.sweep.quantile([0.33, 0.66]).round(-1)),
                         int(df.sweep.max())})
        labels = [str(b) for b in breaks]
        size_name, xlab = "train step", "SFT Loss at frac_1 (all units live)"
    else:
        breaks = [math.log10(f) for f in (0.001, 0.01, 0.1, 1.0)]
        labels = ["0.1%", "1%", "10%", "100%"]
        size_name, xlab = "mask %", "SFT Loss at that mask fraction"

    pl = (
        ggplot(df, aes("x", "y", color="run"))
        + geom_path(size=0.4, alpha=0.7)
        + geom_point(aes(size="size_val"), alpha=0.85)
        + scale_size_continuous(range=(0.3, 2.6), name=size_name, breaks=breaks, labels=labels)
        + facet_grid("x_kind ~ y_kind")
        # Dark2 rather than this repo's usual Set1: seven series reach Set1's yellow, which is
        # unreadable on white. Dark2 keeps eight qualitative hues all legible.
        + scale_color_brewer(type="qual", palette="Dark2")
        + scale_y_continuous(limits=(-3, 103), breaks=[0, 25, 50, 75, 100])
        + labs(x=xlab, y="Responses in French (%)")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY})")

    for run in order:
        g = df[(df.run == run) & (df.x_kind == TRAIN) & (df.y_kind == OFF_TARGET)]
        g = g.sort_values("sweep")
        if g.empty:
            continue
        fmt = (lambda v: f"{int(v):>6d}") if args.mode == "training" else (
            lambda v: f"{v:>6.1%}")
        print(f"\n  {run}")
        print(f"    {size_name:10s}" + " ".join(fmt(v) for v in g.sweep))
        print("    train loss" + " ".join(f"{v:>6.3f}" for v in g.x))
        print("    off-tgt % " + " ".join(f"{v:>6.1f}" for v in g.y))


if __name__ == "__main__":
    main()
