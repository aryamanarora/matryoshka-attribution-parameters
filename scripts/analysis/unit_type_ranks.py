"""Rank distribution per projection type, per run -- binned on the cluster, plotted locally.

The per-unit view (``top_units.py``, ``unit_ranks.py``) says which individual units a mask liked.
This asks the population question instead: does a mask rank ``o_proj`` units differently from
``q_proj`` units *as a class*? A histogram of ranks per type answers it -- flat means "this type is
spread through the ranking like any other", mass at the left means the type is systematically
favoured.

Two things are normalised deliberately, because without them the comparison is meaningless:

**Density, not counts.** The types have wildly different populations: at ``nonresid`` granularity a
block has 8192 ``gate_proj`` units against 2048 ``o_proj`` ones, so raw counts would say only that
MLPs are bigger. Each histogram is divided by that type's own unit count, so a bar is "what
fraction of this type's units landed in this rank range".

**A shared rank population.** Ranks come from ``unit_ranks.py``'s live set -- the units a LoRA delta
actually moves -- so every run ranks over the same 475,136 units and the bins mean the same thing
across rows. The 128,289 dead units are excluded; they would otherwise pile up at the bottom of
every LoRA run and drag whichever types they belong to (embeddings, norms, ``lm_head``) with them.

Under a flat/no-signal ranking every bar would sit at 1/n_bins.

    python scripts/analysis/unit_type_ranks.py --runs /mnt/data/.../runs/*_posthoc --out type_ranks.json
"""

import argparse
import json
import re
from pathlib import Path

import torch

from mask_learning_finetuning.train.posthoc import unit_delta_norms
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

#: the seven projections a block has, plus the parameters that are not projections at all. The
#: latter matter for a full finetune (which moves them) and are exactly the units a LoRA delta
#: leaves at zero, so they are reported per run with a `dead` flag rather than silently dropped.
PROJ = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.")


def type_of(name: str) -> str:
    m = PROJ.search(name)
    if m:
        return m.group(1)
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "input_layernorm" in name:
        return "ln_attn"
    if "post_attention_layernorm" in name:
        return "ln_mlp"
    return "norm_final"


def dead_mask(blob, layout) -> torch.Tensor:
    # via unit_delta_norms so tied slices (neuron_head) accumulate rather than overwrite
    return unit_delta_norms(blob["delta"], layout) == 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--out", default="type_ranks.json")
    args = p.parse_args()

    layout, lora_dead, collected = None, None, []
    for run in map(Path, args.runs):
        ckpt = run / args.checkpoint
        if not ckpt.exists():
            print(f"  skip {run.name}")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        if layout is None:
            layout = layout_from_blob(blob)
            # one type label per unit, built once: the layout is identical across these runs
            type_names = sorted({type_of(n) for n in layout.names})
            code = {t: i for i, t in enumerate(type_names)}
            type_id = torch.empty(layout.total, dtype=torch.long)
            for i, name in enumerate(layout.names):
                type_id[layout.slice_for(i)] = code[type_of(name)]
        # The dead set is a property of the finetune KIND, not of the run: every LoRA delta moves
        # the same 7 projections, and a full finetune moves everything. So it is measured once from
        # a LoRA run (a 5 GB delta read) and reused, rather than 25 times.
        if lora_dead is None and "delta" in blob:
            d = dead_mask(blob, layout)
            if d.any():
                lora_dead = d
                print(f"  dead set from {run.name}: {int(d.sum()):,}")
        collected.append((run.name, blob["scores"].float(),
                          blob.get("args", {}).get("lr"),
                          blob.get("args", {}).get("finetuned")))
        del blob
    if lora_dead is None:
        lora_dead = torch.zeros(layout.total, dtype=torch.bool)
    print(f"{len(collected)} runs, {args.bins} bins")

    out = []
    for name, scores, lr, finetuned in collected:
        # Each run is ranked over ITS OWN live population: everything for a full finetune, the 7
        # projections for a LoRA one. Ranking a LoRA run over all units instead would place a
        # 128k-way tie of dead units in the middle of the ordering and shift every real unit's
        # position by an amount that has nothing to do with the mask. Positions are reported as a
        # fraction of that population, so a row is comparable to any other row.
        is_lora = "adapter" in str(finetuned)
        dead = lora_dead if is_lora else torch.zeros(layout.total, dtype=torch.bool)
        live = (~dead).nonzero(as_tuple=True)[0]
        n = len(live)
        edges = torch.linspace(1, n + 1, args.bins + 1)
        order = scores[live].argsort(descending=True)
        rank = torch.empty(n)
        rank[order] = torch.arange(1, n + 1, dtype=torch.float)
        lt = type_id[live]
        per_type = {}
        for t, code_i in code.items():
            sel = lt == code_i
            cnt = int(sel.sum())
            if not cnt:
                # this type is entirely dead in this run (a LoRA run's norms and embeddings): say
                # so, so the figure can leave the cell blank rather than draw a tie-break artefact
                per_type[t] = dict(n_units=0, dead=True)
                continue
            h = torch.histogram(rank[sel], bins=edges)[0]
            per_type[t] = dict(n_units=cnt, dead=False, frac=(h / cnt).tolist(),
                               median_frac=float(rank[sel].median() / n))
        out.append(dict(run=name, lr=lr, finetuned=finetuned, n_live=n,
                        kind="lora" if is_lora else "full", types=per_type))
        print(f"  {name[:44]:46s} n={n:,}  " + " ".join(
            f"{t.split('_')[0]}:{v['median_frac']:.2f}" for t, v in per_type.items()
            if not v.get("dead")))

    Path(args.out).write_text(json.dumps(dict(
        bins=args.bins, unit_mode=layout.mode, total=layout.total,
        n_dead_lora=int(lora_dead.sum()), runs=out), indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
