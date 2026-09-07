"""MAttr with a FREE PER-ROW BIAS learned alongside the scores: does the coalition disappear?

    uv run python scripts/interference/interference_biasmask.py --tag hard30k

The off-circuit weights MAttr+Adam keeps stand in for a per-row constant -- the interference's
mean contribution E[x] . sum_j U_ij^off, which the bias absorbed and a sparse mask loses
(docs/interference_toy.md). So give the mask that constant directly: fit the scores jointly with
a learnable bias correction db (n_feat numbers) through upstream `learn_scores`'s
`extra_params` hook, loss_fn(mask) = L(U * mask, b + db). If the reading is right the
loss-optimal top-k should shed its ~1000 off-circuit weights and land on the circuit, and the
true-loss curve should reach Adam's old optimum (or better) at a far smaller k.

Three evaluations of the resulting ranking on real masked forwards over a k grid:
    scores+db      the model class the fit optimised: U * mask with b + db
    scores, b      the same ranking with the ORIGINAL bias (what the scores alone are worth)
    plain Adam, b  the reference fit without the bias (from scores.pt)
and, for each ranking, precision/recall against dL, Spearman, and the composition of the
loss-optimal set. `--lr-bias` is db's learning rate (Adam, its own group); the scores keep
the usual lr 0.05 / T 0.5 / 3000 steps, but the k SCHEDULE defaults to `uniform` here -- see
the flag's help: a bias shared across k must be fitted where the masks are dense, and the
log-uniform schedule's near-empty draws pull it the wrong way (measured on lit).

RESULT (docs/interference_toy.md): under uniform k the joint fit works on both configs. hard30k:
the ranking reaches rho 0.91 vs dL (plain Adam 0.74), P@R 1.00 to recall 1.0, and its loss
optimum is 171 circuit + 202 off-circuit (plain Adam: 171 + 1007) at a lower loss. lit: the
learned bias alone takes the circuit from 1.381 to 1.234, below plain Adam's coalition (1.306),
and plain Adam's own ranking under b + db keeps 0 interference weights.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_attrib as IA  # noqa: E402
import interference_toy as IT  # noqa: E402
from interference_scale import spearman  # noqa: E402
from interference_toy import EPS_REAL, sample_x, sweep  # noqa: E402

from learning_to_attribute import learn_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard30k")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-bias", type=float, default=0.01)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-schedule", default="uniform",
                    help="upstream k schedule for the joint fit. `uniform` is the default HERE "
                         "(the plain fits use `log`) because one db is shared across every k the "
                         "schedule samples, and under `log` half the draws are near-empty masks "
                         "whose bias wants nothing to do with the circuit's: on lit that db made "
                         "even the circuit-alone loss worse. Under `uniform` it works on both.")
    ap.add_argument("--out-suffix", default="", help="suffix for biasmask.{pt,json}")
    a = ap.parse_args()
    d = IT.OUT.parent / f"interference_toy_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    n = U.shape[0]
    sup = A > 0
    live = sup.any(1)
    g = torch.Generator().manual_seed(a.seed + 40_000)
    db = torch.zeros(n, requires_grad=True)

    def loss_fn(mask):
        return IA.loss_on(U * mask.view(n, n), b + db, A, v, sample_x(a.batch, g))

    res = learn_scores(n * n, loss_fn, steps=a.steps, variant="topk", k_schedule=a.k_schedule, T=0.5,
                       lr=a.lr, optimizer=a.optimizer, extra_params=[db], lr_extra=a.lr_bias)
    scores = res.scores.detach().view(n, n)
    db = db.detach()
    torch.save({"scores": scores, "db": db}, d / f"biasmask{a.out_suffix}.pt")

    # what the bias learned, against the offset the coalition used to supply
    X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
    Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
    offset = (U * ~sup) @ X.mean(0)
    print(f"learned db: mean {float(db.mean()):+.4f}  |db| mean {float(db.abs().mean()):.4f}  "
          f"corr(db, interference offset) over live rows "
          f"{float(torch.corrcoef(torch.stack([db[live], offset[live]]))[0, 1]):+.2f}  "
          f"corr(db, -offset) {float(torch.corrcoef(torch.stack([db[live], -offset[live]]))[0, 1]):+.2f}")

    nex = X.shape[0]

    def L(mask, bias):
        with torch.no_grad():
            return float((F.relu(X @ (U * mask).T + bias) - Y).pow(2).sum() / nex)

    plain = torch.load(d / "scores.pt")["adam"]
    grid = sorted({int(round(k)) for k in torch.logspace(0, 4.214, 60).tolist()} | {n * n})
    rankings = {"scores+db": (scores, b + db), "scores, b": (scores, b), "plain Adam, b": (plain, b),
                "plain Adam, b+db": (plain, b + db)}
    print(f"\nL(full U, b) {L(torch.ones_like(U), b):.4f}   L(circuit, b) {L(sup.float(), b):.4f}   "
          f"L(circuit, b+db) {L(sup.float(), b + db):.4f}   L(full U, b+db) {L(torch.ones_like(U), b + db):.4f}")
    out = {"db": db.tolist(), "grid": grid, "curves": {}}
    for name, (s, bias) in rankings.items():
        order = s.reshape(-1).argsort(descending=True)
        curve = []
        for k in grid:
            mk = torch.zeros(n * n)
            mk[order[:k]] = 1.
            curve.append(L(mk.view(n, n), bias))
        i = min(range(len(grid)), key=curve.__getitem__)
        kept = torch.zeros(n * n, dtype=torch.bool)
        kept[order[:grid[i]]] = True
        kept = kept.view(n, n)
        c = sweep(s, dl)
        pat = lambda t: c["precision"][next(j for j, x in enumerate(c["recall"]) if x >= t)]
        out["curves"][name] = {"loss": curve, "k_best": grid[i], "L_best": curve[i],
                               "n_on": int((kept & sup).sum()), "n_off": int((kept & ~sup).sum())}
        print(f"  {name:<18} best k {grid[i]:>5}  L {curve[i]:.4f}  kept: on-circuit {int((kept & sup).sum()):>3}"
              f"/{int(sup.sum())}  off-circuit {int((kept & ~sup).sum()):>5} | vs dL: P@R.2/.4/.8 "
              f"{pat(.2):.2f}/{pat(.4):.2f}/{pat(.8):.2f}  rho {spearman(s, dl):+.2f}")
    (d / f"biasmask{a.out_suffix}.json").write_text(json.dumps(out))
    print(f"wrote {d}/biasmask.{{pt,json}}")


if __name__ == "__main__":
    main()
