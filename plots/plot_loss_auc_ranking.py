"""Every attribution method tried on one delta, ranked by how well it localises the TRAINING LOSS.

    uv run python plots/plot_loss_auc_ranking.py --glob "runs/fr2de_qwen25_14b_lr1e-4_*"

The LR grid (`plot_optimizer_lr_auc.py`) is organised around a tuning axis: it puts score LR on x
so each arm's optimum is visible. That is the right shape for "how do I tune this" and the wrong
shape for "what have we tried" -- the closed-form methods have no LR, so they can only appear as
horizontal rules, and the budget variants collapse onto their twins. This figure drops the tuning
axis entirely and just ranks every cell on one number, which is what makes the whole method space
comparable at once: fitted and closed-form arms are rows of the same list.

WHY THE LOSS AND NOT THE BEHAVIOUR RATE, given the behaviour is what the organisms are for. Two
measured reasons, and they point the same way:

* IT IS ~10x MORE REPRODUCIBLE. This sweep contains a genuine replicate -- two runs of an
  identical mask config (the `_nll` cell duplicates its twin, differing only in an extra eval) --
  and they land 0.900 vs 0.902 on the loss log-AUC against 0.392 vs 0.411 on the off-target rate.
  So ~0.02 of any behaviour gap is nothing, where the loss axis resolves ~0.002. `--replicates`
  finds such pairs automatically and draws the observed spread as a scale bar, so the figure
  carries its own noise floor instead of leaving the reader to assume differences are real.
* IT IS A DENSE READOUT. A rate is thresholded -- it counts responses whose argmax crossed over,
  so it sits pinned at 0.00 while the mask is improving underneath (see `plot_rate_vs_nll.py`).
  Several arms here are flat-zero for the first four sparsities on the rate and clearly separated
  on the loss over the same range.

THE TWO AXES DISAGREE, WHICH IS THE REASON TO PLOT THIS ONE SEPARATELY RATHER THAN INSTEAD.
On fr2de/Qwen-14B stepless IG has the BEST loss AUC (0.863) and is mid-pack on behaviour, while
MAttr+SGD tops the behaviour ranking (0.723) and sits below every Adam cell on loss. Localising
the objective and localising the behaviour are not the same problem, and a reader given only this
figure would conclude the opposite of one given only the behaviour ranking. Say which one a claim
is about; `--metric` will draw either.

THE X AXIS IS IN NATS AND LOWER IS BETTER, with one reference line that makes the scale mean
something: the finetune's OWN training loss (the `frac_1` anchor every cell shares). A mask that
recovered the whole finetune at every sparsity would score exactly that, so the line is the
unreachable ideal and each row's distance from it is what the mask failed to recover. There is no
matching upper bound worth drawing -- the last grid point is the full delta for every method, so
even the random control is dragged well below the pretrained loss and a "floor" line would be
wrong. The measured `random scores` row is the honest no-information reference instead.
"""

import argparse
import glob as globmod
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))                 # plots/palette.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from palette import (COLOR, FS_LABEL, FS_LEGEND, FS_TICK, MK, RC,  # noqa: E402
                     REF_LABEL, furnish)
from sparsity_auc import log_auc, series  # noqa: E402  -- one definition of the metric, not two

#: the fitting budget every ordinary cell uses: one epoch at effective batch 16
DEFAULT_BUDGET = (1, 16)


def budget_tag(cfg) -> str:
    """``""`` for the default budget, else ``10ep`` / ``eb1``. Part of a cell's IDENTITY.

    Two cells that differ only in budget share optimizer, LR and schedule exactly, so without this
    they are indistinguishable in a label and would look like an unexplained duplicate row.
    """
    t = cfg.get("train") or {}
    ep, ga = t.get("epochs", 1), t.get("grad_accum", 16)
    if (ep, ga) == DEFAULT_BUDGET:
        return ""
    return "eb1" if ga == 1 else f"{ep}ep"


def describe(cfg):
    """``(label, key, colour, marker, hollow)`` for one run.

    `key` is everything that defines the METHOD and nothing that does not, so two runs sharing a
    key are replicates of one another -- which is how the noise floor gets measured rather than
    assumed.
    """
    mk = cfg.get("mask") or {}
    sc = mk.get("scores", "learned")
    bud = budget_tag(cfg)
    if sc == "ixg":
        who = f"ixg:{mk.get('ixg_at', 'finetuned')}"
        return REF_LABEL.get(who, who), (who,), COLOR[who], "o", False
    if sc == "random":
        return REF_LABEL["random"], ("random",), COLOR["random"], "X", False
    opt = mk.get("score_optimizer", "adam")
    ks = mk.get("k_schedule", "uniform")
    lr = float(mk["score_lr"])
    label = f"MAttr {'SGD' if opt == 'sgd' else 'Adam'}  lr {lr:g}  ·  {ks}"
    if bud:
        label += f"  ({bud})"
    return label, (opt, ks, lr, bud), COLOR.get(opt, "#999999"), MK.get(ks, "s"), bool(bud)


def collect(pattern, ev, metric, split, recovery):
    """One row per run that has this metric, plus the shared `frac_1` anchor."""
    rows, anchors, units = [], [], {}
    for d in sorted(globmod.glob(pattern)):
        d = Path(d)
        if not ((d / "evals.json").exists() and (d / "config.yaml").exists()):
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        mk = cfg.get("mask") or {}
        if not mk or mk.get("scores", "learned") not in ("learned", "ixg", "random"):
            continue
        units.setdefault(mk.get("unit", "nonresid"), []).append(d.name)
        try:
            xs, ys, _, full = series(d, ev, metric, split, recovery)
        except SystemExit:
            continue
        if len(xs) < 2:
            continue
        label, key, c, m, hollow = describe(cfg)
        rows.append(dict(name=d.name, label=label, key=key, color=c, marker=m, hollow=hollow,
                         unit=mk.get("unit", "nonresid"), auc=log_auc(xs, ys)))
        if full is not None:
            anchors.append(full)
    # A `frac` is a different object under a different unit mode (at 14B `nonresid` frac_0.01 is
    # ~26k rows and `svd` frac_0.01 is ~72 singular DIRECTIONS, each writing a rank-1 update across
    # a whole tensor). Ranking them in one list would compare a row count against a rank count.
    if units:
        keep = max(units, key=lambda u: len(units[u]))
        for u, names in units.items():
            if u != keep:
                print(f"  dropped {len(names)} cell(s) with unit={u} (figure is unit={keep}): "
                      + ", ".join(names), file=sys.stderr)
        rows = [r for r in rows if r["unit"] == keep]
    return rows, (anchors[0] if anchors else None)


def replicate_spread(rows):
    """Largest observed spread among runs sharing a method key -- the figure's own noise floor."""
    groups = {}
    for r in rows:
        groups.setdefault(r["key"], []).append(r)
    best = None
    for key, g in groups.items():
        if len(g) < 2:
            continue
        spread = max(x["auc"] for x in g) - min(x["auc"] for x in g)
        if best is None or spread > best[0]:
            best = (spread, g)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="runs/fr2de_qwen25_14b_lr1e-4_*")
    p.add_argument("--metric", default="sft_loss:loss:train",
                   help="eval:metric:split -- default the training loss; behaviour also works")
    p.add_argument("--recovery", action="store_true",
                   help="normalise to (pretrained-y)/(pretrained-full); for cross-organism only")
    p.add_argument("--out", default="plots/loss_auc_ranking.pdf")
    p.add_argument("--width", type=float, default=5.4, help="inches; assume \\textwidth")
    args = p.parse_args()
    ev, metric, split = args.metric.split(":")
    is_loss = (ev == "sft_loss") and not args.recovery

    rows, anchor = collect(args.glob, ev, metric, split, args.recovery)
    if not rows:
        raise SystemExit(f"no runs with {ev}.{split}.{metric} under {args.glob}")
    # BEST FIRST, and which end that is depends on the metric: a loss in nats is lower-better and a
    # rate is higher-better. Sorting one way for both is how a loss table gets read as a
    # leaderboard with its worst rows on top -- the exact mistake `scripts/sparsity_auc.py`
    # documents having made on this organism.
    rows.sort(key=lambda r: r["auc"] if is_loss else -r["auc"])

    plt.rcParams.update(RC)
    h = max(1.6, 0.135 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(args.width, h))
    ys = list(range(len(rows)))[::-1]                       # row 0 at the TOP
    for y, r in zip(ys, rows):
        ax.plot([anchor if (is_loss and anchor is not None) else ax.get_xlim()[0], r["auc"]],
                [y, y], lw=0.4, color="#cccccc", zorder=1)  # leader line to the ideal
        ax.plot([r["auc"]], [y], r["marker"], ms=4.0, zorder=3,
                **({"mfc": "none", "mec": r["color"], "mew": 1.1} if r["hollow"]
                   else {"color": r["color"], "mec": "#000000", "mew": 0.4}))
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=FS_TICK - 0.5)
    ax.set_ylim(-1.1, len(rows) + 0.5)

    if is_loss and anchor is not None:
        ax.axvline(anchor, lw=0.6, color="#666666", ls=(0, (3, 2)), zorder=2)
        ax.annotate("finetune's own loss (unreachable ideal)", xy=(anchor, len(rows) + 0.35),
                    xytext=(3, 0), textcoords="offset points", fontsize=FS_LEGEND - 1, color="#666666",
                    va="top", ha="left")

    rep = replicate_spread(rows)
    if rep:
        spread, g = rep
        x0 = min(r["auc"] for r in rows)
        ax.errorbar([x0], [-0.72], xerr=[[spread / 2], [spread / 2]], fmt="none",
                    ecolor="#000000", elinewidth=0.7, capsize=1.8, zorder=4)
        ax.annotate(f"replicate spread ({spread:.3f}): two runs of an identical config",
                    xy=(x0, -0.72), xytext=(6, 0), textcoords="offset points",
                    fontsize=FS_LEGEND - 1, va="center", ha="left")
        print(f"  replicate pair: {' / '.join(r['name'] for r in g)} -> spread {spread:.4f}",
              file=sys.stderr)

    unit = {"loss": "nats"}.get(metric, "")
    arrow = "↓ better" if is_loss else "↑ better"
    nice = "Train loss" if (ev, split) == ("sft_loss", "train") else f"{ev} {split} {metric}"
    ax.set_xlabel(f"{nice} log-AUC over the sparsity sweep"
                  + (f" ({unit}, {arrow})" if unit else f" ({arrow})"), fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, width=0.5)
    ax.tick_params(axis="y", length=0)      # the label IS the row; a tick adds nothing
    furnish(ax)                                   # upstream's hairline grid + half-weight spines
    ax.grid(False, axis="y")                      # rows are labelled; a horizontal rule adds noise
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fams = []
    for lab, col, mkr in (("MAttr + SGD", COLOR["sgd"], "s"), ("MAttr (Adam)", COLOR["adam"], "s"),
                          ("stepless IG", COLOR["ixg:mc"], "o"),
                          ("I\u00d7G endpoints", COLOR["ixg:base"], "o"),
                          ("random control", COLOR["random"], "X")):
        fams.append(Line2D([0], [0], ls="none", marker=mkr, ms=4.0, color=col,
                           mec="#000000", mew=0.4, label=lab))
    fams.append(Line2D([0], [0], ls="none", marker="D", ms=4.0, mfc="none", mec="#555555",
                       mew=1.1, label="non-default budget"))
    ax.legend(handles=fams, fontsize=FS_LEGEND - 1, loc="upper right", frameon=False,
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    fig.savefig(args.out)
    print(f"wrote {args.out} ({len(rows)} methods, unit={rows[0]['unit']})")


if __name__ == "__main__":
    main()
