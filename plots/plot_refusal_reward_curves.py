"""GRPO reward for the two 1B refusal-mask fits, which are the same recipe with different samplers.

WHY THIS FIGURE EXISTS. ``logk_v2`` (HF sampling) and ``logk_v3`` (vLLM) differ in nothing but
which rollouts they happened to draw -- same delta, same units, same log-uniform k schedule, same
score_lr, same 100 x 6 x 8 budget, same seed, and therefore the SAME k drawn at every step. Their
reward curves are nearly on top of each other and they finish at nearly the same StrongREJECT
(0.692 against 0.677 at 1% of units). Their learned RANKINGS, however, agree on only ~11% of their
top-1% units (spearman 0.30 over all 603,425). So the objective does not determine the ranking:
many different 1% subspaces reach the same reward, and they differ in what ELSE they break --
GSM8K comes out 35.5 against 21.5 at the same operating point.

READ THE BOTTOM PANEL BEFORE THE TOP ONE. Reward is dominated by k, not by training progress: the
log-uniform schedule spends most steps at a k so small the mask does nothing and the reward sits at
the refusal floor. A raw reward-vs-step curve is therefore mostly a plot of which k was drawn. The
top panel's line is the mean over the steps where the mask is big enough to matter (k >= 1% of
units), which is the quantity that actually rises; the faint points behind it are every step, so
the spread that averaging hides stays visible.

    uv run python plots/plot_refusal_reward_curves.py
"""

import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                     # noqa: E402

RUNS = Path(__file__).parent.parent / "runs"
#: (label, run directory, colour). Green is the fit the paper figure uses; grey is its twin, the
#: palette's reference role -- the point of the pair is that they are interchangeable on reward.
SERIES = [("HF sampler (in figure)", "refusal_grpo_logk_v2", P.MODEL["MAttr"]),
          ("vLLM sampler", "refusal_grpo_logk_v3", P.MODEL["GRPO"])]
#: the mask has to be at least this big for the reward to be about the ranking rather than about k
BIG_K = 0.01
WINDOW = 20
FIG_W, FIG_H = 3.3, 2.6
FS_AXIS, FS_TICK, FS_LEG = 7, 6, 6


def load(run):
    p = RUNS / run / "rl_log.json"
    if not p.exists():
        raise SystemExit(f"no rl_log.json at {p}")
    return json.loads(p.read_text())


def binned(d, lo, hi):
    """Mean reward over steps [lo, hi) whose k is big enough to be informative, or None."""
    b = [x["reward"] for x in d[lo:hi] if x["k_frac"] >= BIG_K]
    return statistics.mean(b) if b else None


def main():
    plt.rcParams.update(P.RC)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))

    for label, run, colour in SERIES:
        d = load(run)
        # every step, faint: the spread the binned line averages away
        ax.scatter([x["step"] for x in d], [x["reward"] for x in d], s=1.5, color=colour,
                   alpha=0.35, linewidths=0, zorder=2)
        xs, ys = [], []
        for lo in range(0, len(d), WINDOW):
            m = binned(d, lo, lo + WINDOW)
            if m is not None:
                xs.append(lo + WINDOW / 2)
                ys.append(m)
        ax.plot(xs, ys, color=colour, lw=1.1, zorder=3, label=label)
        # reward against the size of the mask, which is what actually drives it
        ax2.scatter([100 * x["k_frac"] for x in d], [x["reward"] for x in d], s=2.5, color=colour,
                    alpha=0.55, linewidths=0, zorder=2, label=label)

    ax.set_xlabel("GRPO step", fontsize=FS_AXIS)
    ax.set_ylabel(f"Reward (k $\\geq$ {BIG_K:.0%})", fontsize=FS_AXIS)
    ax.legend(fontsize=FS_LEG, frameon=False, loc="upper left", handlelength=1.4,
              borderpad=0, handletextpad=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Mask size (% of units)", fontsize=FS_AXIS)
    ax2.set_ylabel("Reward", fontsize=FS_AXIS)
    for a in (ax, ax2):
        P.furnish(a)
        a.tick_params(labelsize=FS_TICK)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3, h_pad=0.8)
    out = Path(__file__).parent / "refusal_reward_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out)
    for label, run, _ in SERIES:
        d = load(run)
        first, last = binned(d, 0, 20), binned(d, len(d) - 20, len(d))
        print(f"  {label:<24} reward at k>={BIG_K:.0%}: {first:.3f} -> {last:.3f}  "
              f"({sum(1 for x in d if x['k_frac'] >= BIG_K)}/{len(d)} steps qualify)")


if __name__ == "__main__":
    sys.exit(main())
