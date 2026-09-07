"""Does UNDER-training the note's literal config recover its phenomenology? A training-trajectory scan.

    uv run python scripts/interference/interference_undertrained.py            # ~10 min, CPU
    uv run python scripts/interference/interference_undertrained.py --lr 2e-3 --schedule const

The note says so in two places: the simple model "will also be undertrained (this reproduces the
phenomenology of Towards Monosemanticity better)", and Appendix 2 opens with "The models we looked
at weren't fully trained to convergence, since this produced more similar phenomenology to real
models." The sophisticated model's paragraph adds that the block-diagonal construction achieves
the overlap/scatter criteria (1-3) "but not (4): this continues to hold as one trains to
convergence". docs/interference_toy.md's step sweep (300-120k) always ran a cosine schedule to
zero over the whole budget, i.e. every cell was as converged as its budget allowed -- it never
looked at a snapshot taken PART WAY through a run at a live learning rate, which is what
"undertrained" means.

So: train the literal config ONCE under a constant lr, snapshot `U` along the way, and at every
snapshot measure the things the note's validation figures show and the numbers its P/R curves
give -- interference std, real-weight shrinkage, base rate, and the precision of `weight` / `era` /
`twera` at fixed recall against that snapshot's own `dL`. The published numbers to match, read off
the note's figures in docs/interference_toy.md: interference std ~0.25, a large shrinkage region,
base rate ~1.2%, `weight` P@R.08/.2 = 0.68/0.44, `era` P@R.2/.4 = 0.87/0.55, `twera` 0.81/0.47.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_toy as IT  # noqa: E402
from interference_toy import EPS_REAL, heuristics, sample_x, sweep  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "plots" / "data" / "interference_undertrained"
SNAPS = (10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000)


def pat(curve, t):
    return curve["precision"][next(i for i, x in enumerate(curve["recall"]) if x >= t)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--schedule", choices=("const", "cosine"), default="const")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--eval-n", type=int, default=131_072)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--block-density", type=float, default=IT.BLOCK_DENSITY,
                   help="the note's prose says 0.5 where its figure config says 0.1; with "
                        "undertraining a 0.5 block leaves most of its entries unlearned, which is "
                        "what the note's learned-vs-ideal figure shows")
    p.add_argument("--tag", default="")
    p.add_argument("--optimizer", choices=("adam", "sgd", "sgdm"), default="adam",
                   help="the note does not say; sgdm = SGD with momentum 0.9")
    p.add_argument("--snaps", type=str, default="",
                   help="comma-separated snapshot steps (default: the coarse log grid)")
    p.add_argument("--pair", action="store_true",
                   help="also train a SECOND model (seed+1: other init and data order, same A) in "
                        "lockstep, and report the cross-seed correlation of circuit and "
                        "interference weights at every snapshot -- the note's criteria (2)/(3)")
    a = p.parse_args()
    snaps = tuple(int(x) for x in a.snaps.split(",")) if a.snaps else SNAPS
    torch.manual_seed(a.seed)
    IT.BLOCK_DENSITY = a.block_density
    A, v = IT.make_target(a.seed)
    sup = A > 0
    n = IT.N_FEAT
    g = torch.Generator().manual_seed(a.seed + 10_000)
    W_down = (torch.randn(IT.N_RES, n, generator=g) / math.sqrt(n) * a.init_scale).requires_grad_(True)
    W_up = (torch.randn(n, IT.N_RES, generator=g) / math.sqrt(IT.N_RES) * a.init_scale).requires_grad_(True)
    b = torch.zeros(n, requires_grad=True)
    def make_opt(params):
        if a.optimizer == "adam":
            return torch.optim.Adam(params, lr=a.lr)
        return torch.optim.SGD(params, lr=a.lr, momentum=0.9 if a.optimizer == "sgdm" else 0.0)
    opt = make_opt([W_up, W_down, b])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(snaps))
             if a.schedule == "cosine" else None)
    if a.pair:
        g2 = torch.Generator().manual_seed(a.seed + 1 + 10_000)
        W_down2 = (torch.randn(IT.N_RES, n, generator=g2) / math.sqrt(n) * a.init_scale).requires_grad_(True)
        W_up2 = (torch.randn(n, IT.N_RES, generator=g2) / math.sqrt(IT.N_RES) * a.init_scale).requires_grad_(True)
        b2 = torch.zeros(n, requires_grad=True)
        opt2 = make_opt([W_up2, W_down2, b2])
    rows = []
    for step in range(1, max(snaps) + 1):
        x = sample_x(a.batch, g)
        y = F.relu(x @ A.T + v)
        loss = (F.relu(x @ (W_up @ W_down).T + b) - y).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if a.pair:
            x2 = sample_x(a.batch, g2)
            loss2 = (F.relu(x2 @ (W_up2 @ W_down2).T + b2) - F.relu(x2 @ A.T + v)).pow(2).sum(-1).mean()
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
        if step in snaps:
            U, bb = (W_up @ W_down).detach(), b.detach()
            dl, L = IT.delta_loss(U, bb, A, v, a.eval_n, 8192, a.seed)
            stats = IT.statistics(U, bb, A, v, a.eval_n, 8192, a.seed)
            real = dl > EPS_REAL
            cur = {k: sweep(s, dl) for k, s in heuristics(U, stats).items()}
            row = {
                "step": step, "loss": L, "n_real": int(real.sum()),
                "base_rate": float(real.float().mean()),
                "interf_std": float(U[~sup].std()), "interf_absmax": float(U[~sup].abs().max()),
                "real_mean": float(U[sup].mean()),
                "shrink_median": float((U[sup] / A[sup]).median()),
                "recovered": int((real & sup).sum()),
                "weight_p08": pat(cur["weight"], .08), "weight_p20": pat(cur["weight"], .2),
                "era_p20": pat(cur["era"], .2), "era_p40": pat(cur["era"], .4),
                "twera_p20": pat(cur["twera"], .2), "twera_p40": pat(cur["twera"], .4),
            }
            if a.pair:
                U2 = (W_up2 @ W_down2).detach()
                row["r_circuit"] = float(torch.corrcoef(torch.stack([U[sup], U2[sup]]))[0, 1])
                row["r_interf"] = float(torch.corrcoef(torch.stack([U[~sup], U2[~sup]]))[0, 1])
                row["ratio"] = float(U[sup].mean() / U[~sup].std())
            rows.append(row)
            if a.pair:
                print(f"  pair: r circuit {row['r_circuit']:.2f}  r interference {row['r_interf']:.2f}  "
                      f"ratio {row['ratio']:.2f}")
            print(f"step {step:>6} loss {L:.3f} | real {row['n_real']:>4} ({row['base_rate']:.2%}) "
                  f"recovered {row['recovered']:>3}/{int(sup.sum())} | interf std {row['interf_std']:.3f} "
                  f"max {row['interf_absmax']:.2f} | real mean {row['real_mean']:.2f} shrink {row['shrink_median']:.2f} | "
                  f"weight P@.08/.2 {row['weight_p08']:.2f}/{row['weight_p20']:.2f}  era P@.2/.4 "
                  f"{row['era_p20']:.2f}/{row['era_p40']:.2f}  twera {row['twera_p20']:.2f}/{row['twera_p40']:.2f}",
                  flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"U": U, "b": bb, "A": A, "v": v, "dl": dl},
                       OUT / f"snap_{a.tag or 'default'}_{step}.pt")
    fn = OUT / f"trajectory_{a.tag or 'default'}.json"
    fn.write_text(json.dumps({"args": vars(a), "published": {
        "interf_std": 0.25, "base_rate": 0.012, "weight_p08": 0.68, "weight_p20": 0.44,
        "era_p20": 0.87, "era_p40": 0.55, "twera_p20": 0.81, "twera_p40": 0.47}, "rows": rows}))
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
