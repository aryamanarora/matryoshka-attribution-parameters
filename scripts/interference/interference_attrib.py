"""This repo's three attribution methods on the interference-weight filtering task.

    uv run python scripts/interference/interference_attrib.py                   # reads the last toy run
    uv run python scripts/interference/interference_attrib.py --steps 4000

Reads `plots/data/interference_toy/model.pt` (written by `scripts/interference/interference_toy.py`) and
scores all n_feat^2 virtual weights three more ways, appending their curves to `curves.json`
beside the note's four heuristics so every series is read against ONE ground truth.

THE TASK IS THE SAME OBJECT THE NOTE'S ORACLE MEASURES, which is what makes this a fair
comparison rather than a different benchmark: `dL(U_ij) = L(U with U_ij zeroed) - L(U)`
ablates a weight TO ZERO, so every method here also treats zero as the baseline. A method
that scored `U` against some other reference would be answering a different question and
would look better or worse for that reason alone.

    stepless IG   -E_{alpha ~ U(0,1)} [ dL(alpha U)/dU_ij * U_ij ]
    MAttr (Adam)  upstream `learn_scores`, Adam, soft top-k, log-uniform k
    MAttr (SGD)   the same, SGD -- the optimizer is the hyperparameter, per plots/palette.py

SIGNS ARE NOT A CONVENTION HERE, they are pinned by the ground truth. To first order
`dL(U_ij) ~= -U_ij * dL/dU_ij`, so stepless IG carries a LEADING MINUS and larger means "this
weight is helping", which is the direction `dL > eps` defines real. MAttr needs no flip: its
scores rise for units whose inclusion lowers the loss, because the optimizer descends
`dL/dscore` and the gate slope is positive.

WHY THE MASK MULTIPLIES `U` AND NOT A DELTA. Everywhere else in this repo MAttr masks a
finetune's delta over a frozen base; here the "base" is the fully ablated model and the
"delta" is the whole virtual weight matrix, so `loss_fn(mask)` is `L(U * mask)`. That is the
same environment-agnostic contract `learn_scores` already documents -- the caller supplies
the loss, the trainer never learns what it is masking.

UPSTREAM'S OWN CODE does the k-sampling and mask construction (`sample_k`, `build_mask` via
`learn_scores`), never a restatement of it here, for the reason `scripts/interference/toy_sgd_vs_ig.py`
gives: the claim under test is about the shipped implementation, and a clean reimplementation
would agree with the paper by construction instead of testing it.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import interference_toy as IT  # noqa: E402
from interference_toy import sample_x, sweep  # noqa: E402

from matryoshka_attribution import learn_scores  # noqa: E402


def loss_on(U, b, A, v, x):
    y = F.relu(x @ A.T + v)
    return (F.relu(x @ U.T + b) - y).pow(2).sum(-1).mean()


def stepless_ig(U, b, A, v, draws, batch, seed):
    """alpha ~ U(0,1) drawn PER EXAMPLE, one forward+backward per draw -- upstream's `ixg:mc`.

    Per-example rather than per-batch for the reason recorded in the sibling repo's
    `compare_stepless_ig.py`: scores sum over examples before normalising, so a `draws x batch`
    run averages that many independent alphas instead of `draws` of them, and the estimator's
    variance falls with the example count rather than the step count."""
    g = torch.Generator().manual_seed(seed + 30_000)
    acc = torch.zeros_like(U)
    for _ in range(draws):
        x = sample_x(batch, g)
        Uv = U.clone().requires_grad_(True)
        alpha = torch.rand(batch, 1, generator=g)
        y = F.relu(x @ A.T + v)
        loss = (F.relu(alpha * (x @ Uv.T) + b) - y).pow(2).sum(-1).mean()
        loss.backward()
        acc += Uv.grad
    return -(acc / draws) * U          # see the sign note in the module docstring


def mattr(U, b, A, v, optimizer, steps, batch, lr, seed, k_schedule="log"):
    n = U.shape[0]
    g = torch.Generator().manual_seed(seed + 40_000)

    def loss_fn(mask):
        return loss_on(U * mask.view(n, n), b, A, v, sample_x(batch, g))

    res = learn_scores(n * n, loss_fn, steps=steps, variant="topk",
                       k_schedule=k_schedule, T=0.5, lr=lr, optimizer=optimizer)
    return res.scores.detach().view(n, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000, help="MAttr optimizer steps")
    p.add_argument("--draws", type=int, default=256, help="stepless-IG alpha draws")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr-adam", type=float, default=0.05)
    p.add_argument("--lr-sgd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="must match the toy run's --tag")
    args = p.parse_args()
    OUT = IT.OUT.parent / f"{IT.OUT.name}_{args.tag}" if args.tag else IT.OUT

    m = torch.load(OUT / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    blob = json.loads((OUT / "curves.json").read_text())
    print(f"scoring {U.numel()} virtual weights | "
          f"{int((dl > blob['meta']['eps_real']).sum())} real | config {blob['meta'].get('down')} "
          f"n_res {blob['meta']['n_res']}")

    scores = {
        "ixg:mc": stepless_ig(U, b, A, v, args.draws, args.batch, args.seed),
        "adam": mattr(U, b, A, v, "adam", args.steps, args.batch, args.lr_adam, args.seed),
        "sgd": mattr(U, b, A, v, "sgd", args.steps, args.batch, args.lr_sgd, args.seed),
    }
    for name, s in scores.items():
        blob["curves"][name] = sweep(s, dl)
        c = blob["curves"][name]
        # Spearman against the oracle is the one-number summary; the curves are the result.
        ra = s.reshape(-1).argsort().argsort().float()
        rb = dl.reshape(-1).argsort().argsort().float()
        rho = float(((ra - ra.mean()) / ra.std() * (rb - rb.mean()) / rb.std()).mean())
        def pat(t):
            return c["precision"][next(i for i, x in enumerate(c["recall"]) if x >= t)]
        print(f"  {name:<8} spearman(vs dL) {rho:+.3f} | P@R.2 {pat(.2):.2f} "
              f"P@R.4 {pat(.4):.2f} P@R.8 {pat(.8):.2f} | max loss gain {max(c['loss_gain']):.3f}")

    blob["meta"]["attrib"] = {"steps": args.steps, "draws": args.draws, "batch": args.batch,
                              "lr_adam": args.lr_adam, "lr_sgd": args.lr_sgd, "seed": args.seed}
    (OUT / "curves.json").write_text(json.dumps(blob))
    # The score MATRICES, not just their curves: plots/plot_interference_heatmaps.py needs the
    # 128x128 layout to show whether a method's mass lands on A's block diagonal, and that is
    # not recoverable from a precision/recall curve.
    torch.save({k: v for k, v in scores.items()}, OUT / "scores.pt")
    print(f"appended 3 methods to {OUT}/curves.json")


if __name__ == "__main__":
    main()
