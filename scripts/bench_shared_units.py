"""The units several benchmark rankings AGREE on: how many, and where they sit.

For one method's whole-set rankings, the top-k sets of every benchmark are intersected pairwise and
jointly, and the joint core is described by tensor type and layer against the base rate. Pairwise
Jaccard says the rankings are mostly different; this says what the small shared part is made of.

    uv run python scripts/bench_shared_units.py --root runs/olmo3_post/ixg --at base --frac 0.01
"""

import argparse
import collections
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/olmo3_post/ixg")
    ap.add_argument("--at", default="base")
    ap.add_argument("--benches", nargs="+", default=["gsm8k", "math", "ifeval"])
    ap.add_argument("--frac", type=float, default=0.01)
    args = ap.parse_args()
    tops, L = {}, None
    for b in args.benches:
        blob = torch.load(ROOT / args.root / f"{b}_{args.at}" / "final.pt", map_location="cpu",
                          weights_only=False)
        L = blob["layout"]
        s = blob["scores"].float().numpy()
        k = max(1, int(round(args.frac * len(s))))
        tops[b] = set(np.argpartition(-s, k - 1)[:k].tolist())
    n = L["total"]
    core = set.intersection(*tops.values())
    union = set.union(*tops.values())
    k = len(next(iter(tops.values())))
    print(f"top-{100*args.frac:g}% = {k:,} units of {n:,}; union {len(union):,}; "
          f"in ALL {len(tops)}: {len(core):,} (random expectation {k * (k / n) ** (len(tops) - 1):.1f})")
    for a in tops:
        for b in tops:
            if a < b:
                print(f"  {a} & {b}: {len(tops[a] & tops[b]):,}")
    # where the core sits
    unit_type, unit_layer = np.empty(n, dtype=object), np.zeros(n, dtype=int)
    for name, o, c in zip(L["names"], L["offsets"], L["counts"]):
        m = re.search(r"layers\.(\d+)\.", name)
        unit_layer[o:o + c] = int(m.group(1)) if m else -1        # embed/lm_head/final norm: no layer
        parts = name.replace(".weight", "").split(".")
        unit_type[o:o + c] = parts[-1] if "norm" in name or not m else parts[-1]
    idx = np.array(sorted(core))
    base_t = collections.Counter(unit_type.tolist())
    ct = collections.Counter(unit_type[idx].tolist())
    print("core by tensor type: selected / base-rate share")
    for t, cnt in ct.most_common():
        print(f"  {t:>9}: {100*cnt/len(idx):5.1f}%  vs {100*base_t[t]/n:5.1f}%  "
              f"({cnt:,} of {base_t[t]:,} = {100*cnt/base_t[t]:.2f}% of the type)")
    cl = collections.Counter(unit_layer[idx].tolist())
    print("core by layer (count):", [cl.get(i, 0) for i in range(max(unit_layer) + 1)])
    # and each benchmark's own top-k by type, for contrast
    for b, t in tops.items():
        c = collections.Counter(unit_type[np.array(sorted(t))].tolist())
        print(f"{b:>7} top-k by type: " + ", ".join(f"{k2} {100*v/len(t):.0f}%" for k2, v in c.most_common()))


if __name__ == "__main__":
    main()
