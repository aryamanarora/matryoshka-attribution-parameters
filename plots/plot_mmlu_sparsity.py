"""MMLU accuracy vs mask sparsity -- the capability control for `plot_sparsity_units.py`.

Same runs, same x-axis, same conventions as that script; the y-axis is general capability
instead of the training objective. Read the two together: a mask that reproduces the finetune's
loss while holding MMLU at the pretrained anchor is a localised finetune, whereas one that
drags MMLU down with it is just a smaller finetune.

Reads whichever MMLU file a run has:

  ``mmlu.json``     written inline by `learn_mask.py` during training (small n, every eval).
  ``summary.json``  written by `scripts/eval_mmlu_sparsity.py` post-hoc (any n).

Both come from the same protocol -- that script owns it and the trainer imports it -- so with
a matching ``--limit``/``--seed``/``--k-shot`` they are the same measurement and belong on one
axis. The selection each file records is checked, and a mismatch is fatal rather than plotted.

Error bars are the binomial standard error at the run's own n. Note they overstate the
uncertainty on *differences between points*: every condition answers the identical questions
under identical prompts, so the comparison is paired.

    uv run python plots/plot_mmlu_sparsity.py \
        --run "Joint (delta + mask)=plots/data/joint" \
        --run "Mask only (frozen delta)=plots/data/maskonly" \
        --out plots/mmlu_joint_vs_maskonly.pdf

LEGACY INPUT FORMAT. This reads the `sweep.json` / `summary.json` / `mmlu.json` files the
pre-refactor scripts wrote, and still works on the run directories that already contain
them. New runs write a single `evals.json`
(`{condition: {eval: {split: {metric: value}}}}`, see eval/runner.py) -- this script has
not been ported to it.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, geom_errorbar, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_brewer, scale_x_log10, theme, theme_bw, theme_set,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        # matches plot_sparsity_units.py, so the loss and capability figures stack cleanly --
        # and the legend title carries the run sizes, which needs the width
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

# what must match for two runs' MMLU numbers to be one measurement
SELECTION_KEYS = ("dataset", "config", "split", "k_shot", "prompt_format", "limit",
                  "subjects", "seed")


def log_label(breaks):
    """10⁻³ style labels -- Unicode superscripts, not LaTeX (which breaks the font)."""
    return ["" if (b is None or b <= 0)
            else "10" + str(int(round(math.log10(b)))).translate(SUP) for b in breaks]


def compact(n: int) -> str:
    """1_235_814_400 -> 1.24B; 603_425 -> 603k. Legend real estate is scarce."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.2f}{suf}" if v < 10 else f"{v:.0f}{suf}"
    return str(n)


def size_label(run_dir: Path) -> str:
    """"1.24B params, 573k units" from the run's config.json, if it recorded them.

    Same annotation `plot_sparsity_units.py` puts on its legend, so the two figures label the
    same runs the same way.
    """
    cfg_path = Path(run_dir) / "config.json"
    if not cfg_path.exists():
        return ""
    cfg = json.loads(cfg_path.read_text())
    return ", ".join(f"{compact(cfg[k])} {name}" for k, name in
                     (("n_params", "params"), ("n_units", "units")) if cfg.get(k))


def load_run(label: str, run_dir: Path):
    """``(curve rows, pretrained accuracy, n, selection)`` from either MMLU writer."""
    run_dir = Path(run_dir)
    inline, posthoc = run_dir / "mmlu.json", run_dir / "summary.json"
    path = inline if inline.exists() else posthoc
    if not path.exists():
        raise SystemExit(f"no mmlu.json or summary.json in {run_dir}. Either train with "
                         "--mmlu-limit, or run scripts/eval_mmlu_sparsity.py")
    blob = json.loads(path.read_text())
    if "final" in blob:
        grid = blob["final"]
    elif "curve" in blob:
        grid = {r["condition"]: r["accuracy"] for r in blob["curve"]}
    else:
        grid = blob
    sel = blob.get("selection", {})
    n = sel.get("n_questions") or sel.get("limit")
    rows = [dict(run=label, frac=float(k[5:]), acc=v)
            for k, v in grid.items() if k.startswith("frac_")]
    return rows, grid.get("pretrained"), n, {k: sel.get(k) for k in SELECTION_KEYS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    p.add_argument("--out", default="plots/mmlu_sparsity.pdf")
    p.add_argument("--legend-title", default="Run")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="plot even if the runs scored different question selections")
    args = p.parse_args()

    rows, anchors, ns, sels, sizes = [], {}, {}, {}, {}
    for spec in args.run:
        label, _, d = spec.partition("=")
        r, pre, n, sel = load_run(label, d)
        rows += r
        anchors[label], ns[label], sels[label] = pre, n, sel
        sizes[label] = size_label(d)

    # sizes identical across runs go in the legend title, as in plot_sparsity_units.py
    shared = len(set(sizes.values())) == 1 and next(iter(sizes.values()))
    legend_title = f"{args.legend_title} ({shared})" if shared else args.legend_title
    relabel = {lab: lab if shared or not s else f"{lab} ({s})" for lab, s in sizes.items()}

    diffs = {k: {lab: s[k] for lab, s in sels.items()} for k in SELECTION_KEYS
             if len({json.dumps(s[k], sort_keys=True) for s in sels.values()}) > 1}
    if diffs:
        msg = "runs scored different selections: " + "; ".join(f"{k}={v}"
                                                              for k, v in diffs.items())
        if not args.allow_mismatch:
            raise SystemExit(msg + "\nThose are not the same measurement. Re-score with "
                                   "matching --limit/--seed/--k-shot, or --allow-mismatch.")
        print("WARNING:", msg)

    curve = pd.DataFrame(rows)
    curve["run"] = curve["run"].map(lambda r: relabel.get(r, r))
    n = min(v for v in ns.values() if v) if any(ns.values()) else None
    if n:
        # binomial SE at the run's own n; conservative for between-point comparisons, which
        # are paired (identical questions and prompts at every sparsity)
        se = curve["acc"].div(100).pipe(lambda a: (a * (1 - a) / n) ** 0.5).mul(100)
        curve["lo"], curve["hi"] = curve["acc"] - se, curve["acc"] + se

    # the base model is the same in every run, so one pretrained line; if the runs disagree
    # it is scoring noise at this n (one flipped question is ~0.4pp at n=256)
    pre_vals = [v for v in anchors.values() if v is not None]
    pre = sum(pre_vals) / len(pre_vals) if pre_vals else None

    pl = (ggplot(curve, aes("frac", "acc", color="run"))
          + geom_line(size=0.5)
          + geom_point(size=0.9)
          + scale_x_log10(labels=log_label)
          + scale_color_brewer(type="qual", palette="Set1", name=legend_title)
          + labs(x="Fraction of Parameter Units Kept",
                 y=f"MMLU Accuracy (%)" + (f", n={n}" if n else "")))
    if pre is not None:
        pl = pl + geom_hline(yintercept=pre, size=0.3, linetype="dashed", color="#777777")
    if n:
        pl = pl + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.08, size=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}")
    if pre is not None:
        print(f"pretrained anchor: {pre:.2f}%" + (
            "" if len(set(pre_vals)) == 1
            else f"  (runs reported {', '.join(f'{v:.2f}' for v in pre_vals)}; "
                 f"{abs(max(pre_vals) - min(pre_vals)) * (n or 100) / 100:.0f} question(s) "
                 "of difference at this n)"))
    for lab in ns:
        sub = curve[curve.run == lab]
        worst = sub.loc[sub.acc.idxmin()]
        print(f"  {lab:26s} n={ns[lab]}  min {worst.acc:5.2f}% @ {worst.frac:>6.1%}  "
              f"dense {sub.loc[sub.frac.idxmax()].acc:5.2f}%  "
              f"(pretrained {anchors[lab]:.2f}%)")


if __name__ == "__main__":
    main()
