"""Where the top per-WEIGHT IxG scores sit: per tensor kind, per layer, and how concentrated
within rows -- for the fractions where the per-parameter sweep is perfect (0.01%) and where it
collapses (0.5%). Reads ``runs/olmpool/<model>/weight_ixg_base/final.pt`` (7B fp32 scores, mmap).
"""
import re, sys
from collections import Counter
import torch
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

m = sys.argv[1]
blob = torch.load(f"runs/olmpool/{m}/weight_ixg_base/final.pt", map_location="cpu", weights_only=False, mmap=True)
layout = layout_from_blob(blob); scores = blob["scores"]
offsets = torch.tensor(layout.offsets); shapes = layout.shapes; names = layout.names
def kind(n):
    if "q_proj" in n or "o_proj" in n: return "q/o"
    if "k_proj" in n or "v_proj" in n: return "k/v"
    if "mlp" in n or "feed_forward" in n: return "mlp"
    if "norm" in n: return "norm"
    return "other"
sizes = Counter()
for n, s in zip(names, shapes):
    sizes[kind(n)] += int(torch.tensor(s).prod())
# THRESHOLD, NOT TOPK: a top-k over 7B fp32 scores materialises a second 28 GB copy and the login
# node has ~3 GB free. Estimate the cut from a 20M random sample (accurate to ~1% of k), then
# walk the tensors one at a time through the mmap.
g = torch.Generator().manual_seed(0)
sample = scores[torch.randint(0, layout.total, (20_000_000,), generator=g)]
for frac in (0.0001, 0.001, 0.005):
    thr = torch.topk(sample, max(1, round(frac * sample.numel()))).values[-1].item()
    by_kind, by_layer, per_t = Counter(), Counter(), {}
    conc = {}
    for ti, (n, s) in enumerate(zip(names, shapes)):
        sc = scores[layout.slice_for(ti)]
        sel = (sc >= thr).nonzero().flatten()
        if not len(sel): continue
        per_t[ti] = len(sel); by_kind[kind(n)] += len(sel)
        L = re.search(r"layers\.(\d+)\.", n); by_layer[int(L.group(1)) if L else -1] += len(sel)
        if len(s) == 2:
            conc[ti] = (len(sel), (sel // s[1]).unique().numel(), s[0], (sel % s[1]).unique().numel(), s[1])
    k = sum(per_t.values())
    print(f"\n== {m} top {100*frac:.2f}% ~ {k:,} weights (threshold {thr:.3g})")
    print("  by kind (share of selection / share of pool): " + ", ".join(
        f"{kd} {100*c/k:.1f}%/{100*sizes[kd]/layout.total:.1f}%" for kd, c in by_kind.most_common()))
    print("  by layer (top 6): " + ", ".join(f"L{l}:{100*c/k:.0f}%" for l, c in by_layer.most_common(6)))
    for ti, _ in Counter(per_t).most_common(3):
        if ti in conc:
            c, r, R, cc, C = conc[ti]
            print(f"  {names[ti]}: {c:,} weights over {r}/{R} rows, {cc}/{C} cols")
