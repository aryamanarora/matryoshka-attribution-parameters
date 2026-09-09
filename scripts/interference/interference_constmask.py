"""MAttr where a MASKED weight's contribution is replaced by a learned CONSTANT, not by zero.

    uv run python scripts/interference/interference_constmask.py --tag hard30k

The masked model everywhere else in this repo is y' = ReLU(sum_j m_ij U_ij x_j + b_i): a
masked weight contributes nothing. Here it contributes a learned per-weight constant c_ij,

    z_i = sum_j m_ij U_ij x_j  +  sum_j (1 - m_ij) c_ij  +  b_i

i.e. the ablation baseline is "replace U_ij x_j by c_ij", with C an n x n parameter fitted
jointly with the scores through `learn_scores`'s `extra_params`. Why this and not the per-row
bias: with a constant per weight, the bias a masked model effectively carries is
b_i + sum over its masked weights of c_ij -- it DEPENDS ON THE MASK, growing as weights are
removed, which is exactly what one delta-b shared across the k schedule could not express
(docs/interference_toy.md). Mean ablation is the special case c_ij = U_ij E[x_j]; `--init mean`
starts there, `--init zero` starts from zero ablation.

Evaluations, all on real forwards over one eval set:
    scores + C     the fitted model class (masked weights -> their c_ij)
    scores, zero   the same ranking under plain zero ablation and the original b
    plain Adam     the reference fit (scores.pt), zero ablation
plus P/R and Spearman vs dL, the loss-optimal set's composition, and how close the learned C
came to mean ablation (corr with U_ij E[x_j]) and to the per-row offset story.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_toy as IT  # noqa: E402
from interference_scale import spearman  # noqa: E402
from interference_toy import sample_x, sweep  # noqa: E402

from learning_to_attribute import learn_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard30k")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-const", type=float, default=0.001)
    ap.add_argument("--init", choices=("zero", "mean"), default="mean")
    ap.add_argument("--k-schedule", default="log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--freeze-const", action="store_true",
                    help="do not learn C: fit the scores under a FIXED replacement constant "
                         "(with --init mean that is MAttr under mean ablation)")
    a = ap.parse_args()
    d = IT.OUT.parent / f"interference_toy_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    n = U.shape[0]
    sup = A > 0
    g = torch.Generator().manual_seed(a.seed + 40_000)
    ex = 0.15 * torch.ones(n)                          # E[x_j]: density 0.3 x mean 0.5
    C = (U * ex.unsqueeze(0)).clone() if a.init == "mean" else torch.zeros(n, n)
    if not a.freeze_const:
        C.requires_grad_(True)

    def loss_fn(mask):
        mk = mask.view(n, n)
        x = sample_x(a.batch, g)
        y = F.relu(x @ A.T + v)
        z = x @ (U * mk).T + ((1 - mk) * C).sum(1) + b
        return (F.relu(z) - y).pow(2).sum(-1).mean()

    res = learn_scores(n * n, loss_fn, steps=a.steps, variant="topk", k_schedule=a.k_schedule,
                       T=0.5, lr=a.lr, optimizer="adam",
                       **({} if a.freeze_const else {"extra_params": [C], "lr_extra": a.lr_const}))
    scores = res.scores.detach().view(n, n)
    C = C.detach()
    torch.save({"scores": scores, "C": C}, d / f"constmask{a.out_suffix}.pt")

    X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
    Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
    nex = X.shape[0]
    mean_abl = U * X.mean(0).unsqueeze(0)
    off = ~sup
    print(f"learned C: corr with mean ablation U_ij E[x_j] over off-circuit {float(torch.corrcoef(torch.stack([C[off], mean_abl[off]]))[0, 1]):+.2f}, "
          f"over circuit {float(torch.corrcoef(torch.stack([C[sup], mean_abl[sup]]))[0, 1]):+.2f} | "
          f"|C| mean off {float(C[off].abs().mean()):.4f} (mean-abl {float(mean_abl[off].abs().mean()):.4f}) | "
          f"row sums of C over off-circuit vs interference offset: corr "
          f"{float(torch.corrcoef(torch.stack([(C * off).sum(1), mean_abl.mul(off).sum(1)]))[0, 1]):+.2f}")

    def L(mask, const):
        with torch.no_grad():
            z = X @ (U * mask).T + (((1 - mask) * const).sum(1) if const is not None else 0) + b
            return float((F.relu(z) - Y).pow(2).sum() / nex)

    plain = torch.load(d / "scores.pt")["adam"]
    grid = sorted({int(round(k)) for k in torch.logspace(0, 4.214, 60).tolist()} | {n * n})
    print(f"\nL(full U) {L(torch.ones_like(U), None):.4f}   L(circuit, zero abl) {L(sup.float(), None):.4f}   "
          f"L(circuit, C) {L(sup.float(), C):.4f}   L(circuit, mean abl) {L(sup.float(), mean_abl):.4f}   "
          f"L(nothing, C) {L(torch.zeros_like(U), C):.4f}")
    rankings = {"scores + C": (scores, C), "scores, zero abl": (scores, None),
                "scores, mean abl": (scores, mean_abl),
                "plain Adam, zero abl": (plain, None), "plain Adam, C": (plain, C),
                "plain Adam, mean abl": (plain, mean_abl)}
    out = {"grid": grid, "curves": {}, "init": a.init, "k_schedule": a.k_schedule}
    for name, (s, const) in rankings.items():
        order = s.reshape(-1).argsort(descending=True)
        curve = []
        for k in grid:
            mk = torch.zeros(n * n)
            mk[order[:k]] = 1.
            curve.append(L(mk.view(n, n), const))
        i = min(range(len(grid)), key=curve.__getitem__)
        kept = torch.zeros(n * n, dtype=torch.bool)
        kept[order[:grid[i]]] = True
        kept = kept.view(n, n)
        c = sweep(s, dl)
        pat = lambda t: c["precision"][next(j for j, x in enumerate(c["recall"]) if x >= t)]
        out["curves"][name] = {"loss": curve, "k_best": grid[i], "L_best": curve[i],
                               "n_on": int((kept & sup).sum()), "n_off": int((kept & ~sup).sum())}
        print(f"  {name:<22} best k {grid[i]:>5}  L {curve[i]:.4f}  kept: on-circuit "
              f"{int((kept & sup).sum()):>3}/{int(sup.sum())}  off-circuit {int((kept & ~sup).sum()):>5} | "
              f"vs dL: P@R.2/.4/.8 {pat(.2):.2f}/{pat(.4):.2f}/{pat(.8):.2f}  rho {spearman(s, dl):+.2f}")
    (d / f"constmask{a.out_suffix}.json").write_text(json.dumps(out))
    print(f"wrote {d}/constmask{a.out_suffix}.{{pt,json}}")


if __name__ == "__main__":
    main()
