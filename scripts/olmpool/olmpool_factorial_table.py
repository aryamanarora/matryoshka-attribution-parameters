"""Collect scripts/olmpool/olmpool_factorial.py outputs into one table, with the paper's features and scores.

Per model and context length, the retrieval accuracy of every part combination; plus three
derived numbers that the docs read:

  attn_share   how much of the extension's gain the ATTENTION update carries on its own:
               (A - none) / (all - none)            (A alone, rest pretrained)
  rest_share   the same for everything but attention: (all_minus_A - none) / (all - none)
  attn_needed  how much is LOST by leaving attention pretrained given the rest:
               (all - all_minus_A) / (all - none)

    uv run python scripts/olmpool/olmpool_factorial_table.py --lengths 16384 32768
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def share(c, L, num_hi, num_lo, den_hi, den_lo):
    acc = lambda k: c[k][f"ctx_{L}"]["acc"] if k in c else None
    if any(acc(k) is None for k in (num_hi, num_lo, den_hi, den_lo)):
        return None
    den = acc(den_hi) - acc(den_lo)
    return (acc(num_hi) - acc(num_lo)) / den if abs(den) > 1e-9 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/olmpool_factorial")
    ap.add_argument("--results", default="docs/olmpool/olmpool_results.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--out", default="plots/data/olmpool/factorial.json")
    a = ap.parse_args()
    results = json.loads((ROOT / a.results).read_text())
    rows = []
    for f in sorted(Path(a.runs).glob("*.json")):
        d = json.loads(f.read_text())
        c = d["conditions"]
        parts = [k for k in "AQMO" if k in d["parts"]]
        allk = "".join(parts)
        no_a = allk.replace("A", "")
        rec = {"model": d["model"], "parts": parts, "results": results.get(d["model"]), "acc": {}, "nll": {},
               "derived": {}}
        for k, v in c.items():
            rec["acc"][k] = {s: m["acc"] for s, m in v.items()}
            rec["nll"][k] = {s: m["nll"] for s, m in v.items()}
        for L in a.lengths:
            rec["derived"][str(L)] = {
                "gain": (c[allk][f"ctx_{L}"]["acc"] - c["none"][f"ctx_{L}"]["acc"]) if allk in c else None,
                "attn_share": share(c, L, "A", "none", allk, "none"),
                "rest_share": share(c, L, no_a, "none", allk, "none"),
                "attn_needed": share(c, L, allk, no_a, allk, "none"),
                "mlp_share": share(c, L, "M", "none", allk, "none"),
                "qk_share": share(c, L, "Q", "none", allk, "none") if "Q" in parts else None,
            }
        rows.append(rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    L = a.lengths[-1]
    print(f"{'model':34s} HELMET  {'none':>5} {'A':>5} {'M':>5} {'O':>5} {'rest':>5} {'all':>5} {'pt':>5} | attn_share rest_share attn_needed   @{L}")
    for r in sorted(rows, key=lambda r: -(r["results"] or {"helmet": {"32k": 0}})["helmet"]["32k"]):
        acc = lambda k: r["acc"].get(k, {}).get(f"ctx_{L}")
        allk = "".join(r["parts"]); no_a = allk.replace("A", "")
        fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
        dv = r["derived"][str(L)]
        print(f"{r['model']:34s} {r['results']['helmet']['32k'] if r['results'] else 0:5.1f}   {fmt(acc('none'))} {fmt(acc('A'))} {fmt(acc('M'))} {fmt(acc('O'))} {fmt(acc(no_a))} {fmt(acc(allk))} {fmt(acc('pt_own_theta'))} | {fmt(dv['attn_share'])}      {fmt(dv['rest_share'])}      {fmt(dv['attn_needed'])}")


if __name__ == "__main__":
    main()
