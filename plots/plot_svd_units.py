"""Singular-direction units against nonresid units: the same delta, two bases.

The fr2de 8B LoRA-r32 lr-1e-4 finetune, attributed post hoc four ways -- `nonresid` (a unit is a
neuron / non-residual row) and the three `svd*` modes (a unit is a singular direction of the
delta). Same adapter, same excluded tensors, same k-schedule, same score_lr, same vLLM decoder;
the only resolved config difference is `mask.unit`.

**Three rows: the two SFT losses and the behaviour.** The losses are the competence axis and they
are what stops a behavioural curve being read on its own -- a mask that is starved enough to lose
the German also loses the fit, and the loss rows say by how much. Both are forward-only and
therefore cheap, so they are sampled at every sparsity where the behaviour is (`sft_loss` with
`final_n_batches: 200`; `train` is the unshuffled training split, `test` the held-out 10%). Note
they move DOWNWARD as the mask keeps more, opposite to the behaviour row.

**Two columns because there are two honest x axes, and they say opposite things.**

* *Units kept* is the repo's usual axis and the one the sweep grid samples. On it the four modes
  are nearly indistinguishable -- which is already a result, but a weak one, because a unit is not
  the same object in the two bases: 1% of nonresid units is 17,039 rows and 1% of svd units is 72
  directions.
* *Degrees of freedom kept* is what makes those commensurable. A nonresid unit is one row: 4,096
  free numbers. A singular direction of a `[m, n]` tensor writes a rank-1 update over the whole
  tensor but carries only `m + n` free numbers. Summed over the actual top-k (not averaged), so
  each point is the exact cost of the mask that produced the y value beside it.

On the second axis the curves separate by more than an order of magnitude, and that gap is the
finding: this update is far more compressible in *rank* than in *rows*. Read it with the
deflationary number from `delta_stats.json` in hand -- `spearman(scores, per-unit delta norm)` is
0.85 under `svd` against 0.34 under `nonresid`, so in the rank basis the learned ranking is mostly
the trivial "keep the largest singular values". The basis is doing the work, not the fitting.

The right-hand end of every curve is `frac_1`, which composes the same weights as the `full_delta`
anchor, so all four curves must meet there (0.984 off-target, 0.942 test loss) -- they do, which is
also the check that the rank-32 truncation was lossless. The grey dashed lines are that anchor and
the pretrained one, drawn per row because each metric has its own pair; a curve that never leaves
the pretrained line never localised anything.

    uv run python plots/plot_svd_units.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_grid,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    ggsave,
    labs,
    scale_color_brewer,
    scale_x_log10,
    theme,
    theme_bw,
    theme_set,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 4.4),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
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

#: (label, run directory). Order fixes the legend and the colour assignment.
#: The hybrids name BOTH halves, because a bare "SVD (attn)" reads as "SVD, of attention only" --
#: as though the MLP were unmasked, which would be a different experiment with a different unit
#: total. 97% of `svd_attn`'s units are in fact its nonresid half, so that mis-reading is a
#: mis-read denominator and not a quibble about wording.
CELLS = [
    ("nonresid", "fr2de_sweep8b_lora32_lr1e-4_posthoc"),
    ("SVD (all)", "fr2de_sweep8b_lora32_lr1e-4_posthoc_svd"),
    ("SVD attn + nonresid", "fr2de_sweep8b_lora32_lr1e-4_posthoc_svdattn"),
    ("SVD MLP + nonresid", "fr2de_sweep8b_lora32_lr1e-4_posthoc_svdmlp"),
]
#: The two x axes, as column facets. Order matters: the usual one first, the one that reframes it
#: second.
AXES = ["Units Kept", "Degrees of Freedom Kept"]

#: Row facets: ``(label, eval name, split, metric)``. The two losses first because they are the
#: competence axis the behaviour row has to be read against.
METRICS = [
    ("Train Loss", "sft_loss", "train", "loss"),
    ("Test Loss", "sft_loss", "test", "loss"),
    ("German on English", "language", "off_target", "target_frac"),
]


def per_unit_dof(lay):
    """Free numbers each unit contributes -- ``[total]``, aligned with the score vector.

    A row/column unit is ``numel / len(axis)`` parameters. A singular direction is a rank-1 factor
    over a ``[m, n]`` tensor: it *writes* ``m . n`` entries but carries ``m + n`` free numbers, and
    the free numbers are what makes it comparable to a row. Exact per unit, so summing over a
    top-k gives that mask's exact cost rather than a mean-times-k estimate.
    """
    dof = torch.zeros(lay["total"])
    for shape, axis, off, cnt in zip(lay["shapes"], lay["axes"], lay["offsets"], lay["counts"]):
        numel = 1
        for d in shape:
            numel *= d
        if axis == "svd":
            v = float(shape[0] + shape[-1])
        elif axis is None:                     # one unit for the whole tensor
            v = float(numel)
        elif axis == "all":                    # one unit per scalar
            v = 1.0
        else:
            v = float(numel / shape[axis])
        dof[off:off + cnt] = v
    return dof


def value(evs, ev, split, metric):
    return ((evs.get(ev) or {}).get(split) or {}).get(metric)


def rows_for(label, run_dir):
    """``(curve rows, anchor rows)`` for one cell.

    The anchors are the run's own ``pretrained`` and ``full_delta`` conditions, which for a frozen
    delta are constants -- so they are horizontal reference lines rather than curve points, and
    they belong to a row (a metric) rather than to a series.
    """
    ev = json.loads((run_dir / "evals.json").read_text())
    fin = ev.get("final") or ev
    blob = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    lay = blob["layout"]
    dof = per_unit_dof(lay)
    scores = blob["scores"].float().flatten()
    n_params = sum(int(pd.Series(s).prod()) for s in lay["shapes"])
    out, anchors = [], []
    for cond, evs in fin.items():
        if not cond.startswith("frac_"):
            if cond in ("pretrained", "full_delta"):
                for mlabel, e, s, m in METRICS:
                    v = value(evs, e, s, m)
                    if v is not None:
                        # replicated across both columns: facet_grid needs every facet variable
                        # present on the layer's own data
                        anchors += [dict(metric=mlabel, axis=a, y=v, anchor=cond) for a in AXES]
            continue
        frac = float(cond.split("_")[1])
        # k the same way masks.sweep.conditions_for rounds it, so the x value names the mask that
        # produced this y and not a nearby one
        k = max(1, int(round(frac * lay["total"])))
        kept = float(dof[scores.topk(min(k, lay["total"])).indices].sum())
        for mlabel, e, s, m in METRICS:
            v = value(evs, e, s, m)
            if v is None:
                continue
            out.append(dict(unit=label, metric=mlabel, axis=AXES[0], x=100 * frac, y=v))
            out.append(dict(unit=label, metric=mlabel, axis=AXES[1],
                            x=100 * kept / n_params, y=v))
    return out, anchors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="plots/data/fr2de8b_svd")
    p.add_argument("--out", default="plots/svd_units.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    root = Path(args.dir)
    rows, anchors = [], []
    for label, run in CELLS:
        d = root / run
        if not (d / "evals.json").exists():
            print(f"  MISSING {d}")
            continue
        r, a = rows_for(label, d)
        rows += r
        anchors += a       # identical across cells by construction; deduplicated below
        print(f"  {label:9s} {len(r) // (2 * len(METRICS))} sparsities")
    if not rows:
        raise SystemExit(f"no post-hoc results under {root}")

    def cats(frame):
        frame["metric"] = pd.Categorical(frame["metric"], categories=[m[0] for m in METRICS],
                                         ordered=True)
        frame["axis"] = pd.Categorical(frame["axis"], categories=AXES, ordered=True)
        return frame

    df = cats(pd.DataFrame(rows))
    df["unit"] = pd.Categorical(df["unit"], categories=[c[0] for c in CELLS], ordered=True)
    # The anchors are run constants and every cell reports the same two values (same base model,
    # same delta), so drawing one line per cell would stack four identical lines. Averaged rather
    # than first-taken so a real disagreement between cells shows up as a visibly fat line instead
    # of being hidden -- with a frozen delta that would be a composition bug.
    adf = cats(pd.DataFrame(anchors).groupby(["metric", "axis", "anchor"], as_index=False,
                                             observed=True)["y"].mean())

    pl = (
        ggplot(df, aes("x", "y", color="unit"))
        # per row: the pretrained anchor and the full-delta one. Every curve starts near the first
        # and must reach the second at frac_1, which composes the same weights.
        + geom_hline(adf, aes(yintercept="y"), linetype="dashed", color="#999999", size=0.3,
                     inherit_aes=False)
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + facet_grid("metric ~ axis", scales="free")
        + scale_x_log10(labels=lambda xs: [f"{x:g}%" for x in xs])
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Fraction of the Delta Kept (log)", y="", color="Mask Unit")
    )
    ggsave(pl, args.out, dpi=args.dpi, verbose=False)
    print(f"wrote {args.out}  (grey dashed = the pretrained and full-delta anchors, per row)")


if __name__ == "__main__":
    main()
