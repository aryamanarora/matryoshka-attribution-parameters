"""How similar are the per-benchmark rankings of one post-training delta's units?

Reads every masked checkpoint under the run roots (``runs/olmo3_post/{ixg,posthoc,rl}``), and for
each pair of rankings reports Spearman over all units and top-k Jaccard at several k, at three
granularities: the checkpoint's own units (nonresid), per TENSOR and per LAYER. Aggregation to the
coarser levels is by SUM of scores for IxG vectors (the first-order term is additive, so a tensor's
IxG is exactly the sum of its rows') and by the fraction of the tensor's units inside the top-1%
for learned/GRPO vectors (whose scores are not additive).

THE TWO CONTROLS EVERY NUMBER IS READ AGAINST: the split-half pair of the SAME objective
(``<bench>_a`` vs ``<bench>_b``: same benchmark, disjoint prompts, the ceiling any cross-benchmark
similarity could reach), and the random floor (Spearman 0; Jaccard k/(2N-k)).

    uv run python scripts/olmo3_post/bench_similarity.py --roots runs/olmo3_post/ixg runs/olmo3_post/posthoc \
        --out plots/data/olmo3_post/similarity.json
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

KS = (0.002, 0.01, 0.05)


def load(run_dir: Path):
    blob = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    L = blob["layout"]
    return blob["scores"].float().numpy(), L


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d else 0.0


def jaccard_topk(a, b, frac):
    k = max(1, int(round(frac * len(a))))
    ta = set(np.argpartition(-a, k - 1)[:k].tolist())
    tb = set(np.argpartition(-b, k - 1)[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1        # embed_tokens / lm_head / final norm


def aggregate(scores, L, level, additive):
    """Per-tensor or per-layer vector from a unit vector."""
    names, offs, counts = L["names"], L["offsets"], L["counts"]
    keys = names if level == "tensor" else [layer_of(n) for n in names]
    uniq = sorted(set(keys), key=lambda k: (str(type(k)), k))
    idx = {k: i for i, k in enumerate(uniq)}
    out = np.zeros(len(uniq))
    if not additive:
        k = max(1, int(round(0.01 * len(scores))))
        top = np.zeros(len(scores), dtype=bool)
        top[np.argpartition(-scores, k - 1)[:k]] = True
    tot = np.zeros(len(uniq))
    for n, o, c, key in zip(names, offs, counts, keys):
        sl = scores[o:o + c]
        if additive:
            out[idx[key]] += sl.sum()
        else:
            out[idx[key]] += top[o:o + c].sum()
        tot[idx[key]] += c
    return (out if additive else out / np.maximum(tot, 1)), uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["runs/olmo3_post/ixg", "runs/olmo3_post/posthoc",
                                                   "runs/olmo3_post/rl"])
    ap.add_argument("--out", default="plots/data/olmo3_post/similarity.json")
    args = ap.parse_args()
    runs = {}
    for root in args.roots:
        for d in sorted((ROOT / root).glob("*")):
            if (d / "final.pt").exists():
                tag = f"{Path(root).name}/{d.name}"
                s, L = load(d)
                runs[tag] = (s, L)
                print(f"loaded {tag}: {len(s):,} units ({L['mode']})")
    if not runs:
        sys.exit("no checkpoints found")

    results = {"units": {}, "pairs": []}
    tags = list(runs)
    for t in tags:
        results["units"][t] = int(len(runs[t][0]))
    for a, b in itertools.combinations(tags, 2):
        sa, La = runs[a]
        sb, Lb = runs[b]
        if La["names"] != Lb["names"] or La["mode"] != Lb["mode"]:
            continue
        add_a, add_b = "ixg" in a, "ixg" in b
        rec = {"a": a, "b": b, "n_units": len(sa),
               "unit": {"spearman": spearman(sa, sb),
                        **{f"jaccard_{k}": jaccard_topk(sa, sb, k) for k in KS}}}
        for level in ("tensor", "layer"):
            va, _ = aggregate(sa, La, level, add_a)
            vb, _ = aggregate(sb, Lb, level, add_b)
            rec[level] = {"spearman": spearman(va, vb), "n": len(va),
                          "jaccard_0.1": jaccard_topk(va, vb, 0.1),
                          "jaccard_0.25": jaccard_topk(va, vb, 0.25)}
        results["pairs"].append(rec)
    # random floor for the unit-level Jaccard at each k: two independent top-k sets of size k
    n = max(results["units"].values())
    # two independent random k-subsets of n units: E[|A&B|] = k^2/n, |A|B| ~ 2k - k^2/n
    results["random_floor"] = {f"jaccard_{k}": k / (2 - k) for k in KS}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    # a compact table: unit-level spearman / jaccard@1% for every pair
    print("\npair | spearman(unit) | J@0.2% | J@1% | J@5% | spearman(tensor) | spearman(layer)")
    for r in sorted(results["pairs"], key=lambda r: -r["unit"]["spearman"]):
        u = r["unit"]
        print(f"{r['a']} vs {r['b']} | {u['spearman']:.3f} | {u['jaccard_0.002']:.3f} | "
              f"{u['jaccard_0.01']:.3f} | {u['jaccard_0.05']:.3f} | "
              f"{r['tensor']['spearman']:.3f} | {r['layer']['spearman']:.3f}")
    print(f"random floor J@1%: {results['random_floor']['jaccard_0.01']:.4f}")
    print_matrices(results)


BENCHES = ("gsm8k", "math", "ifeval", "mmlu")


def print_matrices(results):
    """Per method: a benchmarks x benchmarks matrix whose DIAGONAL is the split-half ceiling
    (``_a`` vs ``_b`` of one objective) and whose off-diagonal is whole-set vs whole-set."""
    pairs = {}
    for r in results["pairs"]:
        pairs[(r["a"], r["b"])] = pairs[(r["b"], r["a"])] = r
    methods = sorted({t.split("/")[0] + ":" + t.split("/")[1].rsplit("_", 1)[1]
                      if t.startswith("ixg/") else t.split("/")[0] for t in results["units"]})

    def tag(method, obj):
        if method.startswith("ixg:"):
            return f"ixg/{obj}_{method.split(':')[1]}"
        return f"{method}/{obj}"

    for m in methods:
        for level, key in (("unit", "jaccard_0.01"), ("unit", "spearman"), ("tensor", "spearman"),
                           ("layer", "spearman")):
            rows = []
            for a in BENCHES:
                row = []
                for b in BENCHES:
                    if a == b:
                        r = pairs.get((tag(m, f"{a}_a"), tag(m, f"{a}_b")))
                    else:
                        r = pairs.get((tag(m, a), tag(m, b)))
                    row.append(f"{r[level][key]:.2f}" if r else "  -  ")
                rows.append(row)
            if all(c == "  -  " for row in rows for c in row):
                continue
            print(f"\n[{m}] {level} {key} (diagonal = split-half ceiling of the same benchmark)")
            print("      | " + " | ".join(f"{b:>6}" for b in BENCHES))
            for a, row in zip(BENCHES, rows):
                print(f"{a:>6}| " + " | ".join(f"{c:>6}" for c in row))


if __name__ == "__main__":
    main()
