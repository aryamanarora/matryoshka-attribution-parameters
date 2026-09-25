"""The ACTUAL loss of the filtered model at each sparsity, for every ranking.

    uv run python scripts/interference/interference_true_loss.py --tag hard

This is the honest version of the note's "Precision vs Loss Gain" axis. That axis plots
`sum of dL` over the kept set, which assumes ablating a SET costs the sum of ablating its
members -- and on this model that assumption fails badly: measured against real masked
forwards it errs by up to +0.92 mid-curve (larger than the entire additive range, 0.641), and
the true loss is NON-MONOTONE in k where the proxy is monotone, because interference weights
partially cancel one another and removing an arbitrary subset breaks the cancellation.

So here every point is a real forward pass of `ReLU(U*mask x + b)` over the eval set. Nothing
is inferred from `dL`; `dL` only supplies one of the rankings.

A RANDOM ranking is included, and it is not decoration. Every real ranking here is trying to
find ~200 weights among 16384, so a curve that merely descends as k grows has shown nothing --
the random line is what "keeping more weights helps because it is more of the model" looks
like, and a method only says something where it departs from it.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_toy as IT  # noqa: E402

# Two grids, and they resolve different things. LOG is the right default -- the minimum sits at
# k ~ 100-200 out of 16384, which any linear grid coarse enough to be affordable steps straight
# over. LINEAR is the right one for the mid-range BLOW-UP, whose peak position and width the log
# grid samples at only five points and therefore draws as a smooth arc it has not measured.
GRIDS = {
    "log": (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 16384),
    "linear": tuple(range(0, 16385, 128)),
}
N_EVAL, CHUNK = 65_536, 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--grid", default="log", choices=tuple(GRIDS))
    a = ap.parse_args()
    grid = GRIDS[a.grid]
    d = IT.OUT.parent / f"{IT.OUT.name}_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    n = U.numel()

    batches = list(IT.eval_batches(A, v, N_EVAL, CHUNK, 0))

    def loss(mask):
        Um = U * mask.view_as(U)
        tot = sum((F.relu(x @ Um.T + b) - y).pow(2).sum().item() for x, y in batches)
        return tot / sum(x.shape[0] for x, _ in batches)

    h = IT.heuristics(U, IT.statistics(U, b, A, v, N_EVAL, CHUNK, 0))
    rankings = {"ideal": dl, **{k: h[k] for k in ("era", "twera", "weight", "freq")},
                **torch.load(d / "scores.pt")}
    g = torch.Generator().manual_seed(0)
    rankings["random"] = torch.rand(U.shape, generator=g)

    out = {"grid": list(grid), "curves": {},
           "L_full": loss(torch.ones(n)), "L_zero": loss(torch.zeros(n)),
           "n_circuit": int((A > 0).sum()), "n_real": int((dl > 1e-4).sum())}
    # The true circuit is the reference every ranking is implicitly aiming at.
    circ = torch.zeros(n); circ[(A > 0).reshape(-1).nonzero().squeeze()] = 1.0
    out["L_circuit"] = loss(circ)
    print(f"L(full) {out['L_full']:.4f}  L(zero) {out['L_zero']:.4f}  "
          f"L(true circuit, n={out['n_circuit']}) {out['L_circuit']:.4f}\n")

    for name, s in rankings.items():
        order = s.reshape(-1).argsort(descending=True)
        vals = []
        for k in grid:
            mask = torch.zeros(n); mask[order[:k]] = 1.0
            vals.append(loss(mask))
        out["curves"][name] = vals
        print(f"  {name:<8} min {min(vals):.3f} @k={grid[vals.index(min(vals))]:<6} "
              f"max {max(vals):.3f} @k={grid[vals.index(max(vals))]:<6} "
              f"final {vals[-1]:.3f}", flush=True)

    fn = d / f"true_loss_{a.grid}.json"
    fn.write_text(json.dumps(out))
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
