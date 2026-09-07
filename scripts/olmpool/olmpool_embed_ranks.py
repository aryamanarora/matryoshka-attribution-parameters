"""Where the embedding and output-head ROWS rank when everything is scored (`full_learned`).

One line per model: units per kind, how many of each kind sit in the top 0.5%, the best rank
of any embedding / lm_head row, the tokens those rows are, and needle accuracy at 0.5% against
the block-only `all_learned` arm on the same model.
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "scripts/olmpool")
from olmpool_analysis import unit_table
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

runs = Path("runs/olmpool")
for run in sorted(runs.glob("*/full_learned/final.pt")):
    m = run.parts[2]
    blob = torch.load(run, map_location="cpu", weights_only=False, mmap=True)
    layout = layout_from_blob(blob); scores = blob["scores"].float()
    rows = unit_table(layout)
    order = torch.argsort(scores, descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
    k = round(0.005 * layout.total)
    tok = AutoTokenizer.from_pretrained(f"models/olmpool/{m}/lc", trust_remote_code=True)
    kinds = {}
    for u, (layer, kind, j) in rows.items():
        kinds.setdefault(kind, []).append(u)
    ev = json.load(open(run.parent / "evals.json"))["final"]
    ev0 = json.load(open(run.parent.parent / "all_learned" / "evals.json"))["final"]
    acc = lambda e, c, L: e[c]["niah"][f"ctx_{L}"]["acc"]
    print(f"\n== {m}: {layout.total} units, top 0.5% = {k}; niah@0.5% 16K/32K full_learned "
          f"{acc(ev,'frac_0.005',16384):.2f}/{acc(ev,'frac_0.005',32768):.2f} vs all_learned "
          f"{acc(ev0,'frac_0.005',16384):.2f}/{acc(ev0,'frac_0.005',32768):.2f}")
    for kind, us in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        us_t = torch.tensor(us); r = rank[us_t]
        print(f"  {kind:8s} n={len(us):7d}  in top 0.5%: {int((r < k).sum()):5d} "
              f"({100*float((r < k).float().mean()):.1f}% of kind, {100*int((r<k).sum())/k:.1f}% of top)  best rank {int(r.min())}")
    for kind in ("embed", "lm_head"):
        us_t = torch.tensor(kinds[kind]); r = rank[us_t]
        top = us_t[torch.argsort(r)[:12]]
        toks = [repr(tok.decode([rows[int(u)][2]])) for u in top]
        print(f"  top {kind} rows: " + ", ".join(f"{t}#{int(rank[u])}" for t, u in zip(toks, top)))
