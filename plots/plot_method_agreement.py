"""Do two refusal interventions get the SAME items right, or just the same score?

Appendix figure, full width. One panel per benchmark; within a panel, a methods x methods matrix
whose cell is the fraction of items on which the two methods return the SAME per-item verdict.
The diagonal is 1.00 by construction.

WHY IT EXISTS. In the main figure abliteration and MAttr 1% land within 0.01 of each other on all
four benchmarks -- StrongREJECT 0.70/0.69, SORRY-Bench 0.96/0.94, GSM8K 32.0/32.0, IFEval
49.4/49.5. A tie that close invites the reading that the two edits do the same thing. They do not:
on GSM8K they score 64/200 apiece while agreeing on only 43 of those, with 21 solved by each
alone, and NOT ONE of the 200 responses is textually identical. This figure is that check run
across every benchmark and every pair, so the reading is available rather than asserted.

WHAT A CELL IS, and the trap in it. The value is EXACT MATCH on the per-item verdict:

    gsm8k         the answer was extracted and equals the gold (the eval's own ``correct``)
    ifeval        every instruction in the prompt was followed (strict; ``all(follow_list)``)
    sorrybench    their judge returned 1 (complied) rather than 0
    strongreject  their graded score exceeds ``high_threshold`` = 0.5, the eval's own cutoff

**Agreement is confounded by base rate and must not be read as similarity.** Two methods that both
refuse everything agree on ~100% of StrongREJECT items while sharing no mechanism at all; the
Instruct/Base chat pair does exactly this. So a HIGH cell between two methods with LOW scores says
nothing, and the informative cells are the ones between methods whose scores are high and similar
-- which is precisely the abliteration/MAttr pair this figure was built for. Cohen's kappa, which
corrects for chance agreement, is printed to stdout for every pair and is the number to quote if a
claim depends on the comparison. It is not plotted because the request was exact match and because
kappa is undefined when a method is constant on a benchmark, which happens here (Base/chat refuses
every SORRY-Bench item).

MMLU IS ABSENT and cannot be added without a code change: it is forward-only and has no
``drain_records``, so no per-item record reaches disk. The other four all generate.

DATA. Every method contributes four files, one per benchmark, and they are not all in one
directory -- the anchors keep their metrics in three run dirs (see the note in
plot_baseline_strongreject.py), and a masked run keeps its conditions in one sweep. Records carry
a ``condition`` field for every eval EXCEPT gsm8k, whose records predate it; for that one the
condition is recovered by block order against evals.json and the recovered block's accuracy is
CHECKED against the reported one, so a wrong slice fails loudly instead of quietly comparing the
wrong sparsity.

    uv run python plots/plot_method_agreement.py
    uv run python plots/plot_method_agreement.py --ncols 2 --height 5.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                     # noqa: E402

RUNS = Path(__file__).parent / "data" / "all_runs"

#: (panel title, eval name, subdirectory, split, verdict from a record)
TASKS = [
    ("StrongREJECT", "strongreject", "strongreject_eval", "off_target",
     lambda r: (r.get("score") or 0.0) > 0.5),
    ("SORRY-Bench", "sorrybench", "sorrybench_eval", "off_target",
     lambda r: bool(r.get("score"))),
    ("GSM8K", "gsm8k", "gsm8k_eval", "gsm8k", lambda r: bool(r.get("correct"))),
    ("IFEval", "ifeval", "ifeval_eval", "ifeval", lambda r: bool(r.get("strict"))
     and all(r["strict"])),
]

#: method -> {eval name: (run directory holding the sweep, condition label)}. The directory is the
#: one whose evals.json and <eval>_eval/ belong together.
def _anchor(run):
    return {"strongreject": (run, "dense"), "gsm8k": (run, "dense"),
            "sorrybench": (f"{run}_sorrybench", "dense"), "ifeval": (f"{run}_ifeval", "dense")}


def _edited(run, native, cond="dense"):
    return {"strongreject": (f"{run}/{native}", cond), "gsm8k": (f"{run}/{native}", cond),
            "sorrybench": (f"{run}/eval_sorrybench", cond),
            "ifeval": (f"{run}/eval_ifeval", cond)}


METHODS = [
    (r"Instruct$^{\mathrm{C}}$", _anchor("anchor_instruct_native")),
    (r"Instruct$^{\mathrm{U}}$", _anchor("anchor_instruct_urial_help")),
    (r"Base$^{\mathrm{C}}$", _anchor("anchor_base_plain")),
    (r"Base$^{\mathrm{U}}$", _anchor("anchor_base_urial_help")),
    ("GRPO", _edited("refusal_grpo_weights_lora", "eval_native")),
    ("GRPO+KL", _edited("refusal_grpo_weights_kl001", "eval_native")),
    ("Abliteration", {"strongreject": ("abliteration_native", "dense"),
                      "gsm8k": ("abliteration_native", "dense"),
                      "sorrybench": ("abliteration_sorrybench", "dense"),
                      "ifeval": ("abliteration_ifeval", "dense")}),
    ("MAttr 1%", _edited("refusal_grpo_logk_v2", "posthoc_eval", "frac_0.01")),
]
#: the mask's StrongREJECT/GSM8K live in the native re-eval, not the fitting sweep
METHODS[-1][1]["strongreject"] = ("refusal_grpo_logk_v2/eval_native", "frac_0.01")
METHODS[-1][1]["gsm8k"] = ("refusal_grpo_logk_v2/eval_native", "frac_0.01")

FIG_W, FIG_H = 5.5, 1.85
FS_TICK, FS_ANNOT, FS_TITLE = 6, 4.5, 7


def verdicts(run, cond, ev, subdir, split, fn):
    """``{prompt: bool}`` for one method on one benchmark, or None if the files are absent."""
    d = RUNS / run
    gen = d / subdir / "generations.jsonl"
    if not gen.exists() or not (d / "evals.json").exists():
        return None
    rows = [json.loads(l) for l in gen.read_text().splitlines()]
    rows = [r for r in rows if r.get("split") == split]
    if any("condition" in r for r in rows):
        rows = [r for r in rows if r.get("condition") == cond]
    else:
        # gsm8k records carry no condition: recover the block by order, then CHECK it
        fin = json.load((d / "evals.json").open())["final"]
        order = [k for k in fin if isinstance(fin[k].get(ev), dict)]
        n = fin[order[0]][ev][split]["n"]
        blocks = [rows[i * n:(i + 1) * n] for i in range(len(rows) // n)]
        idx = order.index(cond)
        if idx >= len(blocks):
            # deduplicated conditions generate once; fall back to the block that matches the number
            idx = min(idx, len(blocks) - 1)
        rows = blocks[idx]
        got = 100.0 * sum(fn(r) for r in rows) / max(1, len(rows))
        want = fin[cond][ev][split].get("accuracy")
        if want is not None and abs(got - want) > 1e-6:
            raise SystemExit(f"{run} {ev} block {idx} scores {got:.2f} but evals.json says "
                             f"{want:.2f} for {cond} -- the block-order recovery is wrong")
    return {r["prompt"]: fn(r) for r in rows} if rows else None


def kappa(a, b):
    """Cohen's kappa on two aligned boolean lists; None when either is constant."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def matrix(task):
    title, ev, subdir, split, fn = task
    per = {}
    for name, src in METHODS:
        run, cond = src[ev]
        v = verdicts(run, cond, ev, subdir, split, fn)
        if v:
            per[name] = v
    names = [n for n, _ in METHODS if n in per]
    if len(names) < 2:
        return title, [], None, {}
    shared = set.intersection(*(set(per[n]) for n in names))
    keys = sorted(shared)
    cols = {n: [per[n][k] for k in keys] for n in names}
    m = np.eye(len(names))
    kap = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            agree = sum(x == y for x, y in zip(cols[a], cols[b])) / len(keys)
            m[i, j] = m[j, i] = agree
            kap[(a, b)] = kappa(cols[a], cols[b])
    return title, names, m, dict(n=len(keys), kappa=kap,
                                 rate={n: sum(cols[n]) / len(keys) for n in names})


def draw(ax, title, names, m, meta, *, first):
    im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=FS_TICK, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(names if first else [], fontsize=FS_TICK)
    ax.set_title(f"{title} (n={meta['n']})", fontsize=FS_TITLE)
    for i in range(len(names)):
        for j in range(len(names)):
            # white on the dark end of viridis, black on the light end
            ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=FS_ANNOT,
                    color="#000000" if m[i, j] > 0.55 else "#ffffff")
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.4)
    ax.grid(False, which="major")
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(P.SPINE_LW)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "method_agreement.pdf"))
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--width", type=float, default=FIG_W)
    ap.add_argument("--height", type=float, default=FIG_H)
    a = ap.parse_args()

    built = [matrix(t) for t in TASKS]
    built = [b for b in built if len(b[1]) >= 2]
    if not built:
        raise SystemExit("no benchmark had two methods with per-item records on disk")

    plt.rcParams.update(P.RC)
    nrows = -(-len(built) // a.ncols)
    fig, axes = plt.subplots(nrows, a.ncols, figsize=(a.width, a.height), squeeze=False)
    for i, (title, names, m, meta) in enumerate(built):
        r, c = divmod(i, a.ncols)
        draw(axes[r, c], title, names, m, meta, first=(c == 0))
    for i in range(len(built), nrows * a.ncols):
        axes[divmod(i, a.ncols)].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.7)
    fig.savefig(a.out)
    print("wrote", a.out)

    for title, names, m, meta in built:
        print(f"\n{title}  n={meta['n']}  per-method rate: "
              + ", ".join(f"{n} {meta['rate'][n]:.2f}" for n in names))
        print("  pairs where BOTH methods score >0.4 (the ones agreement is informative for):")
        for (x, y), k in meta["kappa"].items():
            if min(meta["rate"][x], meta["rate"][y]) > 0.4:
                i, j = names.index(x), names.index(y)
                print(f"    {x:<24} vs {y:<24} exact {100 * m[i, j]:5.1f}%  "
                      + (f"kappa {k:+.3f}" if k is not None else "kappa undefined"))


if __name__ == "__main__":
    sys.exit(main())
