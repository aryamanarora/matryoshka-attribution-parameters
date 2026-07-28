"""Which units does every mask agree about? Mean per-unit rank across runs, top and bottom.

The top-5 window (``scripts/top_units.py``) shows what each run picked first; this asks the
population question -- averaged over runs, which units sit near the top of the ranking and which
sit near the bottom. Extracted on the cluster because it needs every run's whole score vector.

**Dead units are excluded, and that is not a detail.** A LoRA finetune only moves 7 projections
per block, so in a LoRA-attributed run 128,289 of 603,425 units have an exactly-zero delta. Their
scores can never receive gradient (``dL/ds`` scales with the delta), so they keep their init value
of exactly 0 -- a 128k-way tie that ``argsort`` breaks by index. Ranked naively, "the bottom units
by mean rank" would just be the highest-indexed dead units, which is a fact about ``argsort`` and
not about the model. So the dead set is computed once from a LoRA run's delta, removed, and every
run is ranked within the surviving population.

Full-SFT runs have no dead units (the delta touches everything), so they are ranked over the same
surviving population for comparability rather than over their own larger one.

    python scripts/unit_ranks.py --runs /mnt/data/.../runs/*_posthoc --top 8 --out unit_ranks.json
"""

import argparse
import json
import re
from pathlib import Path

import torch

from mask_learning_finetuning.masks import unit_norms
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

_LAYER = re.compile(r"layers\.(\d+)\.")


def short_name(name: str) -> str:
    m = _LAYER.search(name)
    tail = name.rsplit(".", 2)[-2] if name.endswith(".weight") else name.rsplit(".", 1)[-1]
    return f"L{m.group(1)} {tail}" if m else tail


def owner(layout, idx: int):
    for i, (name, off, cnt) in enumerate(zip(layout.names, layout.offsets, layout.counts)):
        if off <= idx < off + cnt:
            return name, idx - off
    raise IndexError(idx)


def dead_mask(blob, layout) -> torch.Tensor:
    """Units whose delta is exactly zero, so their score never moved off its init."""
    dn = torch.zeros(layout.total)
    for i, (name, axis) in enumerate(zip(layout.names, layout.axes)):
        dn[layout.slice_for(i)] = unit_norms(blob["delta"][name].float(), axis)
    return dn == 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--top", type=int, default=8, help="how many from each end")
    p.add_argument("--stat", default="mean", choices=("mean", "median"),
                   help="how to summarise a unit's rank across runs. `median` is the robust one: "
                        "with a handful of runs a mean can be carried by a single outlier, and "
                        "these groups have as few as four members")
    p.add_argument("--groups", default=None,
                   help="JSON {label: [run names]}. Ranks are then summarised per group and every "
                        "reported unit carries its value in EVERY group -- the point being to see "
                        "whether the units a drifted run ranks highly are the same ones a "
                        "non-drifted run ranks highly, which a single-group listing cannot show")
    p.add_argument("--out", default="unit_ranks.json")
    args = p.parse_args()

    runs, scores, layout, dead = [], [], None, None
    for run in map(Path, args.runs):
        ckpt = run / args.checkpoint
        if not ckpt.exists():
            print(f"  skip {run.name}: no {args.checkpoint}")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        if layout is None:
            layout = layout_from_blob(blob)
        s = blob["scores"].float()
        if dead is None and "delta" in blob:
            d = dead_mask(blob, layout)
            if d.any():          # a LoRA delta: reuse this set for every run
                dead = d
                print(f"  dead set from {run.name}: {int(d.sum()):,}/{layout.total:,} units")
        runs.append(dict(run=run.name, lr=blob.get("args", {}).get("lr"),
                         finetuned=blob.get("args", {}).get("finetuned")))
        scores.append(s)
        del blob
    if dead is None:
        dead = torch.zeros(layout.total, dtype=torch.bool)
    live = (~dead).nonzero(as_tuple=True)[0]
    print(f"{len(runs)} runs, ranking over {len(live):,} live units")

    # rank 1 = highest score, within the live population, per run
    ranks = torch.empty(len(runs), len(live))
    for i, s in enumerate(scores):
        order = s[live].argsort(descending=True)
        r = torch.empty(len(live))
        r[order] = torch.arange(1, len(live) + 1, dtype=torch.float)
        ranks[i] = r
    summarise = (lambda t: t.median(0).values) if args.stat == "median" else (lambda t: t.mean(0))
    name_to_row = {r["run"]: i for i, r in enumerate(runs)}
    if args.groups:
        groups = json.loads(Path(args.groups).read_text())
        groups = {g: [r for r in v if r in name_to_row] for g, v in groups.items()}
    else:
        groups = {"all": [r["run"] for r in runs]}
    stats = {}
    for g, members in groups.items():
        idx = [name_to_row[r] for r in members]
        stats[g] = summarise(ranks[idx])
        print(f"  group {g!r}: {len(idx)} runs")

    out = []
    for g, st in stats.items():
        for end, order in (("top", st.argsort()), ("bottom", st.argsort(descending=True))):
            for pos in order[:args.top].tolist():
                flat = int(live[pos])
                name, within = owner(layout, flat)
                out.append(dict(
                    group=g, end=end, param=name, short=short_name(name),
                    index_in_param=within, flat_index=flat,
                    **{f"{args.stat}_rank_{gg}": float(stats[gg][pos]) for gg in stats},
                    per_run={runs[i]["run"]: float(ranks[i, pos]) for i in range(len(runs))}))
    Path(args.out).write_text(json.dumps(dict(
        n_runs=len(runs), n_live=len(live), n_dead=int(dead.sum()), total=layout.total,
        stat=args.stat, unit_mode=layout.mode, groups={g: v for g, v in groups.items()},
        runs=runs, units=out), indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
