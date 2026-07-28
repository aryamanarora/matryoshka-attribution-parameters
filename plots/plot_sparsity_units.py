"""Loss vs mask sparsity, comparing runs (unit granularities, or training regimes).

Reads the `sweep.json` written by `finetune_masked.py` / `learn_mask.py` for each run and
draws the held-out and train curves as a function of the fraction of parameter units the mask
keeps, with the pretrained anchor as a horizontal reference. Under `cause` the curve's last
point (frac=1) *is* that run's full delta, so the dense finetune needs no separate line.

Only compare runs whose final sweeps were computed the same way: `--final-eval-batches 0`
(the whole split) landed partway through this project, and a 16-example curve plotted
against a 705-example one would differ for reasons that have nothing to do with the units.
The script reads `config.json` alongside each sweep and refuses runs that disagree on
anything that would confound the comparison. When an older run's own sweep was measured on a
different budget, re-measure both with `scripts/eval_loss_sparsity.py` and point
`--sweep-file` at what it wrote.

    uv run python plots/plot_sparsity_units.py \
        --run row=/mnt/data/.../bad_medical_row_v2 \
        --run nonresid=/mnt/data/.../bad_medical_nonresid \
        --out plots/sparsity_units.pdf

    # joint vs mask-only, both re-measured on a matched budget
    uv run python plots/plot_sparsity_units.py --sweep-file sweep_400b.json \
        --legend-title Run --split both \
        --run "Joint (delta + mask)=plots/data/joint" \
        --run "Mask only (frozen delta)=plots/data/maskonly" \
        --out plots/sparsity_joint_vs_maskonly.pdf
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_point, ggplot, guide_legend, guides, labs, scale_color_brewer, scale_x_log10, theme,
    theme_bw, theme_set,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.0),
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

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_label(breaks):
    """10⁻³ style labels -- Unicode superscripts, not LaTeX (which breaks the font)."""
    out = []
    for b in breaks:
        if b is None or b <= 0:
            out.append("")
            continue
        out.append("10" + str(int(round(math.log10(b)))).translate(SUP))
    return out


def compact(n: int) -> str:
    """1_235_814_400 -> 1.24B; 603_425 -> 603k. Legend real estate is scarce."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.2f}{suf}" if v < 10 else f"{v:.0f}{suf}"
    return str(n)


# these must match across runs or the curves are not comparable
# Anything here silently changes the curves if it differs between runs. `unit` and `n_units`
# are deliberately absent -- they are the comparison. Declare any other intentional
# difference with --vary, which drops it from the check AND puts it in the legend, so a
# varied field is never invisible.
CONFOUNDS = ("mode", "k_schedule", "variant", "model", "dataset", "seed", "epochs",
             "max_steps", "lr", "weight_decay", "score_lr", "T", "warmup_steps",
             "lr_scheduler", "batch_size", "grad_accum", "final_eval_batches")


# legend real estate is tight at 5.5in; spell varied fields short
VALUE_ABBREV = {"uniform": "unif", "log_both": "log2"}
ABBREV = {"weight_decay": "wd", "k_schedule": "k", "score_lr": "score lr",
          "learning_rate": "lr", "batch_size": "bs", "grad_accum": "ga",
          "final_eval_batches": "eval batches", "lr_scheduler": "sched"}


def size_label(cfg) -> str:
    """"1.24B params, 573k units" from a run's config, skipping whatever it did not record.

    Both counts matter and neither implies the other: ``n_params`` is how much weight the
    delta spans, ``n_units`` how finely the mask can cut it. Runs written before the training
    scripts recorded these get no annotation rather than a wrong one.
    """
    parts = [f"{compact(cfg[k])} {name}" for k, name in
             (("n_params", "params"), ("n_units", "units")) if cfg.get(k)]
    return ", ".join(parts)


def load_run(label: str, run_dir: Path, sweep_file: str = "sweep.json"):
    sweep = json.loads((run_dir / sweep_file).read_text())
    # eval_loss_sparsity.py wraps the same {split: {label: loss}} dict in a header recording
    # the budget it used; a training script's sweep.json is that dict bare
    sweep = sweep["sweep"] if "sweep" in sweep else sweep
    cfg = json.loads((run_dir / "config.json").read_text())
    rows, anchors = [], []
    for split, sw in sweep.items():
        for key, loss in sw.items():
            if key.startswith("frac_"):
                rows.append(dict(unit=label, split=split, frac=float(key[5:]), loss=loss))
            else:
                anchors.append(dict(unit=label, split=split, anchor=key, loss=loss))
    return pd.DataFrame(rows), pd.DataFrame(anchors), cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                   help="repeatable, e.g. --run nonresid=/mnt/data/.../bad_medical_nonresid")
    p.add_argument("--out", default="plots/sparsity_units.pdf")
    p.add_argument("--split", default="test", choices=["test", "train", "both"])
    p.add_argument("--facet", default="split", choices=["split", "run"],
                   help="with --split both: 'split' (default) gives one panel per split with "
                        "the runs overlaid -- best for ranking runs against each other. 'run' "
                        "gives one panel per run with train and held-out overlaid -- best for "
                        "seeing the generalisation gap within each run")
    p.add_argument("--sweep-file", default="sweep.json",
                   help="which sweep to read from each run dir; use what "
                        "scripts/eval_loss_sparsity.py wrote (e.g. sweep_400b.json) when the "
                        "runs' own final sweeps used different budgets")
    p.add_argument("--legend-title", default="Unit (Scores)",
                   help="what the runs differ in, e.g. Run when comparing training regimes")
    p.add_argument("--vary", action="append", default=[], metavar="FIELD",
                   help="config field the runs are MEANT to differ on (repeatable). "
                        "Exempted from the confound check and appended to the legend")
    p.add_argument("--jitter", type=float, default=0.06, metavar="DECADES",
                   help="half-width of the per-run log-space x-offset; 0 disables")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="plot even if the runs differ on a confounding config field")
    args = p.parse_args()

    curves, anchors, cfgs, relabel = [], [], {}, {}
    for spec in args.run:
        label, _, d = spec.partition("=")
        c, a, cfg = load_run(label, Path(d), args.sweep_file)
        curves.append(c)
        anchors.append(a)
        cfgs[label] = cfg

    # The legend carries the size of what is being masked -- otherwise "weight" and "row" look
    # like peers when one has 2000x the scores. Sizes that are IDENTICAL across runs are
    # hoisted into the legend title instead of repeated on every entry (comparing training
    # regimes at one granularity, they always are).
    sizes = {lab: size_label(cfg) for lab, cfg in cfgs.items()}
    shared = len(set(sizes.values())) == 1 and next(iter(sizes.values()))
    legend_title = f"{args.legend_title} ({shared})" if shared else args.legend_title
    def entry(lab, size):
        bits = ([] if shared or not size else [size.replace(" units", "")]) + [
            f"{ABBREV.get(f, f)}={VALUE_ABBREV.get(str(v), v)}"
            for f in args.vary if (v := cfgs[lab].get(f)) is not None]
        return f"{lab} ({', '.join(bits)})" if bits else lab

    relabel = {lab: entry(lab, s) for lab, s in sizes.items()}

    # A field PRESENT in every config but with different values is a confound and fatal. A
    # field one config lacks entirely is a structural difference between the scripts that
    # wrote them -- learn_mask.py has no --lr/--weight-decay/--warmup-steps because it has no
    # delta optimizer -- which is the comparison, not a confound. Report those and continue.
    diffs, structural = {}, {}
    for f in CONFOUNDS:
        if f in args.vary:          # declared intentional; surfaced in the legend instead
            continue
        vals = {lab: cfg.get(f) for lab, cfg in cfgs.items()}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) == 1:
            continue
        (structural if any(f not in cfg for cfg in cfgs.values()) else diffs)[f] = vals
    if structural:
        print("note: only some runs record " + ", ".join(f"{f}={v}" for f, v in
                                                         structural.items()))
    if diffs:
        msg = "runs differ on: " + "; ".join(f"{f}={v}" for f, v in diffs.items())
        if not args.allow_mismatch:
            raise SystemExit(msg + "\nThat would confound the comparison. Re-run so they "
                                   "match, or name the field with --vary if the difference "
                                   "is the point, or --allow-mismatch to override.")
        print("WARNING:", msg)

    curves = pd.concat(curves, ignore_index=True)
    anchors = pd.concat(anchors, ignore_index=True)

    # Per-run multiplicative x-offset, evenly spread in log space, so markers at each sweep
    # point don't sit exactly on top of each other -- same convention as
    # learning-to-attribute's curve plots (c4688ef). +-0.06 decade ~= +-15%. It matters more
    # here than there: runs that differ in nothing that matters (wd=0 vs wd=0.01) land on
    # identical y, and without the offset one curve is simply invisible under another.
    order = list(cfgs)
    jit = {lab: 10 ** off for lab, off in
           zip(order, np.linspace(-args.jitter, args.jitter, len(order)) if len(order) > 1
               else [0.0])}
    curves["x"] = curves.frac * curves.unit.map(jit)
    # The size/--vary annotation is legend furniture: in a facet strip it collides with the
    # neighbouring strip and duplicates what the run label already says. Strips get the bare
    # label; the sizes are still reported on stdout.
    if not (args.split == "both" and args.facet == "run"):
        curves["unit"] = curves["unit"].map(lambda u: relabel.get(u, u))
    if args.split != "both":
        curves = curves[curves.split == args.split]
        anchors = anchors[anchors.split == args.split]

    # The pretrained anchor is the same model in every run, so collapse over unit-mode --
    # but NOT over split: train and held-out have different pretrained losses, and averaging
    # them would draw one wrong line through both facets.
    pre = (anchors[anchors.anchor == "pretrained"]
           .groupby("split", as_index=False).loss.mean())
    nice = {"train": "Train", "test": "Held-out"}
    for df in (curves, pre):
        df["split"] = df["split"].map(lambda x: nice.get(x, x))

    ylab = "SFT Loss" if args.split == "both" else f"{nice.get(args.split)} SFT Loss"
    # Faceting by run puts the two splits in one panel, so SPLIT becomes the colour and the
    # run moves to the strip. The pretrained anchors then need to be per-split too, or the two
    # dashed lines in a panel cannot be told apart -- they are 0.02 nats apart here, so they
    # overplot either way, but the colour makes that visible rather than misleading.
    by_run = args.split == "both" and args.facet == "run"
    colour, strip = ("split", "unit") if by_run else ("unit", "split")
    pl = (
        ggplot(curves, aes("x", "loss", color=colour))
        + geom_hline(aes(yintercept="loss", **({"color": "split"} if by_run else {})),
                     data=pre, size=0.3, linetype="dashed",
                     inherit_aes=False, **({} if by_run else {"color": "#777777"}))
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + scale_x_log10(labels=log_label)
        + scale_color_brewer(type="qual", palette="Set1",
                             name="Split" if by_run else legend_title)
        + labs(x="Fraction of Parameter Units Kept", y=ylab)
    )
    if args.split == "both":
        pl = pl + facet_wrap(f"~{strip}")
    # ~95 chars is about what fits on one 5.5in row at legend_text size 6; wrap to as many
    # rows as that implies rather than assuming two will do
    width = sum(len(v) for v in relabel.values()) + 4 * len(relabel)
    if width > 95:
        pl = pl + guides(color=guide_legend(nrow=min(len(relabel), -(-width // 95) + 1)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}")
    for _, r in pre.iterrows():
        print(f"pretrained anchor [{r.split}]: {r.loss:.4f}")
    for sp in curves.split.unique():
        for lab in cfgs:
            shown = lab if by_run else relabel.get(lab, lab)
            sub = curves[(curves.unit == shown) & (curves.split == sp)]
            best = sub.loc[sub.loss.idxmin()]
            raw = {v: k for k, v in nice.items()}.get(sp, sp)
            fd = anchors[(anchors.unit == lab) & (anchors.anchor == "full_delta")
                         & (anchors.split == raw)].loss.iloc[0]
            print(f"  {sp:9s} {lab:9s} best {best.loss:.4f} @ {best.frac:>6.1%}   "
                  f"full delta {fd:.4f}")


if __name__ == "__main__":
    main()
