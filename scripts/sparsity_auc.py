"""Summarise a sparsity sweep as ONE number: the log-weighted area under its curve.

Every comparison in this repo so far has been an eyeballed curve or an ad-hoc "mean over frac
0.005-0.2". That is fine for a figure and bad for a table -- it has no defensible weighting, and it
silently drops the conditions outside whatever window was chosen. This is the principled version,
and it is not invented here: it is MIB's ``acc_auc``, transcribed from
``MIB-circuit-track/MIB_circuit_track/evaluation.py`` so a number here means what a number there
means.

    uv run python scripts/sparsity_auc.py --glob "runs/fr2de_*posthoc_shard*" \\
        --eval language --metric target_frac

    log_auc = sum_i (log x_{i+1} - log x_i) * (y_i + y_{i+1})/2  /  (log x_last - log x_first)

i.e. a trapezoid in LOG sparsity, normalised by the log range, so it is a weighted mean of the
curve in which every DECADE of sparsity contributes equally.

WHY LOG AND NOT LINEAR, which is the whole point and is measurable rather than aesthetic. On a
linear x the 0.5 -> 1.0 interval is half the total width, so a linear AUC is dominated by the dense
end -- exactly the region where every method here has already converged (all eleven optimizer cells
sit within 0.05 nats of each other above frac 0.01, and every one shares an identical frac_1
anchor). The interesting differences live below 2% of units, which is 1.7% of a linear axis and
~45% of a log one. `--also-linear` prints both so the gap is visible rather than asserted.

WHAT IT DOES NOT FIX. A single number still hides shape: a curve that rises early and falls back
(the "sparse mask beats the full delta" pattern this repo keeps finding) can score the same as a
monotone one. Read the AUC next to the curve, never instead of it -- and `--peak` is printed for
exactly that reason, since peak > full_delta is the signature of that shape.

TWO CONVENTIONS WORTH STATING, because they change the number:

* ``pretrained`` is EXCLUDED. It has no sparsity, so it has no place on a log-x axis; including it
  at some nominal x would invent a decade. It is reported separately as the floor.
* ``full_delta`` is EXCLUDED as a duplicate of ``frac_1`` (they alias under ``mode: cause``), which
  would otherwise double-weight the last point.

For a LOSS metric use ``--recovery``: raw nats are not comparable across organisms and lower is
better, so the AUC of a loss curve is not interpretable on its own. ``--recovery`` converts each
point to (pretrained - y) / (pretrained - full_delta) -- the fraction of the finetune's own loss
reduction that this slice recovers, 0 at the pretrained model and 1 at the full delta -- which is
the loss analogue of a behaviour rate and is what MIB's faithfulness normalisation does too.
"""

import argparse
import glob as globmod
import json
import math
from pathlib import Path


def log_auc(xs, ys):
    """MIB's ``acc_auc``: log-x trapezoid over the sweep, normalised by the log-x range."""
    lx = [math.log(x) for x in xs]
    num = sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1))
    return num / (lx[-1] - lx[0])


def linear_auc(xs, ys):
    """The same trapezoid on a LINEAR x -- dense-end dominated; here for contrast only."""
    num = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1))
    return num / (xs[-1] - xs[0])


def series(run: Path, ev: str, metric: str, split: str, recovery: bool):
    """``(xs, ys, floor, full)`` -- the swept conditions only, plus the two anchors."""
    final = json.loads((run / "evals.json").read_text())["final"]
    pts, floor, full = [], None, None
    for cond, v in final.items():
        block = (v.get("sft_loss") if ev == "sft_loss" else v.get(ev)) or {}
        y = (block.get(split) or {}).get(metric)
        if y is None:
            continue
        if cond == "pretrained":
            floor = y
        elif cond == "full_delta":
            full = y                      # aliases frac_1; kept as the anchor, dropped from the AUC
        else:
            pts.append((float(cond.replace("frac_", "")), y))
    pts.sort()
    if full is None and pts:
        full = pts[-1][1]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if recovery:
        if floor is None or full is None or abs(full - floor) < 1e-9:
            raise SystemExit(f"{run.name}: --recovery needs a pretrained and a full_delta anchor")
        ys = [(floor - y) / (floor - full) for y in ys]
    return xs, ys, floor, full


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", required=True)
    p.add_argument("--eval", required=True, help="eval name, or `sft_loss`")
    p.add_argument("--metric", required=True, help="e.g. target_frac / lower_frac / loss")
    p.add_argument("--split", default=None, help="default: off_target, or `test` for sft_loss")
    p.add_argument("--recovery", action="store_true",
                   help="normalise to (pretrained - y)/(pretrained - full_delta); use for losses")
    p.add_argument("--also-linear", action="store_true", help="print the linear AUC beside it")
    args = p.parse_args()
    split = args.split or ("test" if args.eval == "sft_loss" else "off_target")

    rows = []
    for d in sorted(globmod.glob(args.glob)):
        d = Path(d)
        if not (d / "evals.json").exists():
            continue
        try:
            xs, ys, floor, full = series(d, args.eval, args.metric, split, args.recovery)
        except Exception:
            continue
        if len(xs) < 2:
            continue
        rows.append((d.name, log_auc(xs, ys), linear_auc(xs, ys), max(ys), full, len(xs)))
    if not rows:
        raise SystemExit(f"no runs with {args.eval}.{split}.{args.metric} under {args.glob}")

    # SORT BY WHICH END IS BETTER, not always descending. A rate is higher-better and a loss in
    # nats is lower-better, so one order cannot serve both -- and printing a loss table descending
    # puts the WORST rankings at the top, which reads as a leaderboard and is how the `random` and
    # `ixg @ finetuned` controls appeared to lead the loss tables in this repo's own notes.
    lower_better = (args.eval == "sft_loss" and not args.recovery)
    rows.sort(key=lambda r: r[1] if lower_better else -r[1])
    w = max(len(r[0]) for r in rows)
    head = f"{'run':<{w}}  {'log-AUC':>8}"
    if args.also_linear:
        head += f"  {'lin-AUC':>8}"
    print(head + f"  {'peak':>6}  {'full':>6}  n"
          + ("     (lower is better)" if lower_better else "     (higher is better)"))
    for name, la, lin, peak, full, n in rows:
        line = f"{name:<{w}}  {la:>8.3f}"
        if args.also_linear:
            line += f"  {lin:>8.3f}"
        star = " *" if (full is not None and peak > full + 1e-9) else ""
        print(line + f"  {peak:>6.3f}  {full if full is None else round(full, 3):>6}  {n}{star}")
    if any(r[3] > (r[4] or 9e9) + 1e-9 for r in rows):
        print("\n* peak exceeds full_delta: a sparse mask beats the whole finetune on this metric,"
              "\n  which a single AUC cannot show -- read the curve.")


if __name__ == "__main__":
    main()
