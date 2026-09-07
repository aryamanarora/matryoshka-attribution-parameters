"""Each method's loss-optimal kept set, refined and saved for plotting.

    uv run python scripts/interference/interference_minloss_sets.py --tag hard

The linear true-loss grid is 128-spaced, so its argmin is only located to +-64 weights. Here the
neighbourhood of each grid argmin is re-swept at spacing 8, because the panels this feeds are
captioned with a specific `k` and a specific loss and those should be the measured optimum
rather than the nearest grid point.

Saves the refined k, its loss, and the kept-set indices per method to minloss_sets.pt.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_toy as IT  # noqa: E402

METHODS = ("ixg:mc", "sgd", "adam")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()
    d = IT.OUT.parent / f"{IT.OUT.name}_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v = m["U"], m["b"], m["A"], m["v"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    scores = torch.load(d / "scores.pt")
    tl = json.loads((d / "true_loss_linear.json").read_text())
    batches = list(IT.eval_batches(A, v, 65_536, 8192, 0))
    n_ex = sum(x.shape[0] for x, _ in batches)

    def loss(mask):
        Um = U * mask.view_as(U)
        return sum((F.relu(x @ Um.T + b) - y).pow(2).sum().item() for x, y in batches) / n_ex

    out = {"A": A, "U": U, "L_full": tl["L_full"], "L_circuit": tl["L_circuit"],
           "L_zero": tl["L_zero"], "sets": {}}
    for name in METHODS:
        order = scores[name].reshape(-1).argsort(descending=True)
        grid = tl["grid"]
        k0 = grid[tl["curves"][name].index(min(tl["curves"][name]))]
        # +-1 grid cell around the coarse argmin, at spacing 8
        lo, hi = max(8, k0 - 128), min(U.numel(), k0 + 128)
        best = None
        for k in range(lo, hi + 1, 8):
            mask = torch.zeros(U.numel()); mask[order[:k]] = 1.0
            L = loss(mask)
            if best is None or L < best[1]:
                best = (k, L)
        k, L = best
        out["sets"][name] = {"k": k, "loss": L, "idx": order[:k].clone()}
        sup = (A > 0).reshape(-1)
        kept = torch.zeros(U.numel(), dtype=torch.bool); kept[order[:k]] = True
        print(f"  {name:<7} k={k:<6} loss {L:.4f}  |  on-circuit kept {int((kept & sup).sum()):>4}"
              f"/{int(sup.sum())}  off-circuit kept {int((kept & ~sup).sum()):>5}", flush=True)

    torch.save(out, d / "minloss_sets.pt")
    print(f"\nwrote {d}/minloss_sets.pt")
    print(f"reference: L(full) {out['L_full']:.4f}  L(circuit A) {out['L_circuit']:.4f}")


if __name__ == "__main__":
    main()
