"""Which units did a run's mask rank highest? Extracts the top-k per run to a small JSON.

Run this where the checkpoints are (the cluster): a post-hoc ``final.pt`` carries the delta as
well as the scores, so it is ~5 GB, and the point of this script is that the interesting part is
a few hundred bytes. It is loaded with ``mmap=True`` so only the score vector is actually read.

A flat score index means nothing on its own -- index 412,003 of 603,425 is not a place. The
layout is what turns it into one: ``offsets``/``counts`` say which parameter tensor owns the
index, and ``axes`` says which axis of that tensor its units run along, so a unit becomes
"``layers.7.mlp.down_proj``, index 1234 along axis 0" -- one neuron. The axis matters for reading
it: under ``nonresid`` two tensors in the same run index their units along different axes, so a
bare index without its axis is ambiguous.

Also recorded per unit, because the score alone cannot answer the obvious deflationary question:

``delta_norm``   the L2 norm of that unit's slice of the delta. If the top-scoring units are just
                 the largest-delta units, the ranking is a magnitude baseline in disguise -- the
                 per-run Spearman in ``delta_stats.json`` says that globally, and this says it for
                 the units that actually got picked.
``delta_rank``   where that unit sits in the delta-norm ordering (1 = largest). A top-scoring unit
                 with ``delta_rank`` in the thousands is the mask disagreeing with magnitude,
                 which is the case worth looking at.

    python scripts/top_units.py --runs /mnt/data/.../runs/*_posthoc --out top_units.json
"""

import argparse
import json
import re
from pathlib import Path

import torch

from mask_learning_finetuning.masks import unit_norms
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

#: `model.layers.7.mlp.down_proj.weight` -> `L7 down_proj`, so a label fits in a figure
_LAYER = re.compile(r"layers\.(\d+)\.")


def short_name(name: str) -> str:
    m = _LAYER.search(name)
    tail = name.rsplit(".", 2)[-2] if name.endswith(".weight") else name.rsplit(".", 1)[-1]
    return f"L{m.group(1)} {tail}" if m else tail


def owner(layout, idx: int):
    """``(param name, index within that param, axis)`` for a flat score index."""
    for i, (name, off, cnt) in enumerate(zip(layout.names, layout.offsets, layout.counts)):
        if off <= idx < off + cnt:
            return name, idx - off, layout.axes[i]
    raise IndexError(f"flat index {idx} is outside the layout's {layout.total} units")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories holding final.pt")
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--out", default="top_units.json")
    args = p.parse_args()

    out = []
    for run in args.runs:
        run = Path(run)
        ckpt = run / args.checkpoint
        if not ckpt.exists():
            print(f"  skip {run.name}: no {args.checkpoint}")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        scores = blob["scores"].float()
        layout = layout_from_blob(blob)
        targs = blob.get("args", {})

        # delta-norm ordering, for the deflationary comparison. Absent when the run did not save
        # its delta, in which case the score columns still mean something on their own.
        dn = None
        if "delta" in blob:
            dn = torch.zeros(layout.total)
            for i, (name, axis) in enumerate(zip(layout.names, layout.axes)):
                dn[layout.slice_for(i)] = unit_norms(blob["delta"][name].float(), axis)
            dn_rank = torch.empty(layout.total, dtype=torch.long)
            dn_rank[dn.argsort(descending=True)] = torch.arange(layout.total)

        top = torch.topk(scores, args.top)
        units = []
        for rank, (val, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            name, within, axis = owner(layout, idx)
            u = dict(rank=rank, score=val, flat_index=idx, param=name, short=short_name(name),
                     index_in_param=within, axis=str(axis))
            if dn is not None:
                u["delta_norm"] = float(dn[idx])
                u["delta_rank"] = int(dn_rank[idx]) + 1
            units.append(u)
        out.append(dict(run=run.name, unit=layout.mode, total_units=layout.total,
                        lr=targs.get("lr"), unit_cfg=targs.get("unit"),
                        scores_kind=targs.get("scores", "learned"),
                        ixg_at=targs.get("ixg_at"), finetuned=targs.get("finetuned"),
                        score_std=float(scores.std()), score_max=float(scores.max()),
                        top=units))
        print(f"  {run.name}: {layout.mode}, {layout.total:,} units, "
              f"top score {top.values[0]:.4g}")
        del blob

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({len(out)} runs)")


if __name__ == "__main__":
    main()
