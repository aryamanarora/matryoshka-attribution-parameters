"""THE summary figure: per cell, who wins on behaviour vs who wins on the loss -- one scatter.

    uv run python plots/plot_dissociation_scatter.py

The result this figure carries: over 16 (organism x model) deltas, best-tuned MAttr+SGD beats
stepless IG on the BEHAVIOUR log-AUC in 10 cells (ties 4, loses 1) while stepless IG beats SGD on
the TRAIN-LOSS log-AUC in 16 of 16. Two tables can say that; what they cannot show is that it is
ONE fact per cell rather than two -- each delta has a single point whose x is the behaviour margin
and whose y is the loss margin, and the claim "SGD localises the behaviour, IG localises the
objective" is literally the statement that the points sit in the upper-right quadrant. A cell that
breaks the story is visible as a point outside it, labelled, rather than a row to hunt for.

AXES ARE MARGINS, NOT LEVELS, because levels are not comparable across cells: behaviour AUCs live
on each organism's own rate scale and loss AUCs in each dataset's own nats. The x-y position of a
cell in (SGD-minus-IG, IG-minus-SGD) space is unitless in the only sense that matters -- both
coordinates are differences of the same quantity on the same delta.

SIGNS: x = bestSGD - IG on behaviour (higher-better metric, so positive = SGD wins);
y = SGD - IG on train loss (lower-better metric, so positive = IG wins). Upper-right = the
dissociation. The grey box is the +-0.02 replicate floor measured on this sweep (two runs of an
identical config differ by 0.019 on behaviour, 0.002 on loss; 0.02 is used for both axes, which
is CONSERVATIVE on y by 10x) -- points inside it are ties, not small wins.

`bestSGD` takes the max over the three k-schedules per cell, which flatters SGD by one selection
(median schedule spread is 0.016, under the floor, so the flattery is bounded by noise). IG has no
matching knob to select over -- it is one run. Stated here so the figure cannot be read as
schedule-matched.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import COLOR, FS_LABEL, FS_LEGEND, FS_TICK, RC, furnish  # noqa: E402

FR = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
#: behaviour metric per organism -- the off-target headline each organism is built around
MET = {"fr2de": ("language", "target_frac"), "fr2ru": ("language", "target_frac"),
       "fr2zh": ("language", "target_frac"), "case": ("casing", "lower_frac"),
       "caps": ("casing", "upper_frac"), "spelling": ("spelling", "british_frac"),
       "bad_medical": ("em_fast", "misaligned_frac")}
MODELS = (("qwen25_14b", "Qwen-14B", "o"), ("gemma2_9b", "Gemma-9B", "s"),
          ("olmo3_7b", "OLMo-7B", "^"))
FLOOR = 0.02          # the measured behaviour replicate spread; conservative for the loss axis


def log_auc(run, ev, split, metric):
    p = Path("runs") / run / "evals.json"
    if not p.exists():
        return None
    final = json.loads(p.read_text())["final"]
    xs, ys = [], []
    for f in FR:
        v = ((final.get(f"frac_{f:g}", {}).get(ev) or {}).get(split) or {}).get(metric)
        if v is not None:
            xs.append(f)
            ys.append(v)
    if len(xs) < 2:
        return None
    lx = [math.log(x) for x in xs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2
               for i in range(len(ys) - 1)) / (lx[-1] - lx[0])


def cells():
    """One row per attributed delta: the IG run, its three SGD twins, and the labels."""
    rows = [l.split("\t") for l in
            (Path("/tmp/claude-1028/-home-guests-aryaman/5b069ddd-cea2-4b6e-a348-df4cc5e4cdfe"
                  "/scratchpad/mc_names.txt").read_text().strip().split("\n"))]
    rows.append(("fr2de", "x", "x", "fr2de_qwen25_14b_lr1e-4_posthoc_shard"))
    for org, _, _, rn in rows:
        ig = ("fr2de_qwen25_14b_lr1e-4_ixg_mc" if rn.endswith("_lr1e-4_posthoc_shard")
              else rn + "_ixg_mc")
        tag = "financial" if "financial" in rn else org
        model, mk = next((v, m) for k, v, m in MODELS if k in rn)
        yield tag, model, mk, rn, ig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="plots/dissociation_scatter.pdf")
    a = p.parse_args()

    pts = []
    for tag, model, mk, rn, ig in cells():
        ev, metric = MET[tag if tag != "financial" else "bad_medical"]
        ig_b = log_auc(ig, ev, "off_target", metric)
        sgd_b = [log_auc(f"{rn}_sgd10_{k}", ev, "off_target", metric)
                 for k in ("log_both", "log", "logit")]
        sgd_b = [v for v in sgd_b if v is not None]
        ig_l = log_auc(ig, "sft_loss", "train", "loss")
        sgd_l = log_auc(f"{rn}_sgd10_log_both", "sft_loss", "train", "loss")
        if None in (ig_b, ig_l, sgd_l) or not sgd_b:
            print(f"  skipped {tag} · {model}: missing an arm", file=sys.stderr)
            continue
        pts.append((max(sgd_b) - ig_b, sgd_l - ig_l, tag, model, mk))

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(2.7, 2.55))
    lim_x = max(abs(x) for x, *_ in pts) * 1.25
    lim_y = max(abs(y) for _, y, *_ in pts) * 1.25
    # the tie box, drawn FIRST so every point sits on top of it
    ax.axvspan(-FLOOR, FLOOR, color="#f0f0f0", zorder=0)
    ax.axhspan(-FLOOR, FLOOR, color="#f0f0f0", zorder=0)
    ax.axhline(0, lw=0.5, color="#999999", zorder=1)
    ax.axvline(0, lw=0.5, color="#999999", zorder=1)
    for x, y, tag, model, mk in pts:
        ax.plot([x], [y], mk, ms=3.6, color=COLOR["sgd"], mec="#000000", mew=0.4, zorder=3)
        # label only the cells outside the tie box on x -- the ones a reader will ask about
        if abs(x) > FLOOR:
            ax.annotate(tag, xy=(x, y), xytext=(2.5, 2), textcoords="offset points",
                        fontsize=FS_LEGEND - 1.6, color="#333333", zorder=4)
    n_ur = sum(1 for x, y, *_ in pts if x > FLOOR and y > 0)
    n_tie = sum(1 for x, y, *_ in pts if abs(x) <= FLOOR)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y * 0.15, lim_y)
    ax.set_xlabel(r"behaviour log-AUC: best SGD $-$ stepless IG", fontsize=FS_LABEL - 0.5)
    ax.set_ylabel(r"train-loss log-AUC: SGD $-$ IG", fontsize=FS_LABEL - 0.5)
    ax.tick_params(labelsize=FS_TICK, length=2, width=0.5)
    furnish(ax)
    # quadrant captions, placed in data space where the quadrants actually are
    ax.text(lim_x * 0.97, lim_y * 0.97, "SGD finds the behaviour,\nIG fits the loss",
            ha="right", va="top", fontsize=FS_LEGEND - 1, color="#333333")
    ax.text(-lim_x * 0.97, lim_y * 0.97, "IG wins both", ha="left", va="top",
            fontsize=FS_LEGEND - 1, color="#333333")
    handles = [Line2D([], [], ls="none", marker=m, ms=3.6, color=COLOR["sgd"],
                      mec="#000000", mew=0.4, label=lab) for _, lab, m in MODELS]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="#f0f0f0", ec="none",
                                 label=f"replicate floor (±{FLOOR:g})"))
    ax.legend(handles=handles, fontsize=FS_LEGEND - 1.2, frameon=False, loc="lower right",
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}: {len(pts)} cells, {n_ur} in the dissociation quadrant, "
          f"{n_tie} behaviour ties")


if __name__ == "__main__":
    main()
