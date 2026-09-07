"""Compute-matched budget sweep for the three attribution methods on the interference task.

    # one (method, seed) per process, so a 32-core box runs the grid in one wall-clock pass
    uv run python scripts/interference/interference_budget.py --tag hard --method adam --seed 0
    uv run python scripts/interference/interference_budget.py --tag hard --collect      # the table

WHAT "COMPUTE-MATCHED" MEANS HERE, and why the first pass was not. A MAttr step and a
stepless-IG draw are each exactly ONE forward+backward of the toy model on one batch, so the
budget axis shared by all three methods is that count -- the currency the sibling repo's
`compare_stepless_ig.py` already uses. The first run of `scripts/interference/interference_attrib.py` gave
MAttr 3000 steps against stepless IG's 256 draws, i.e. **11.7x more**, so its reported numbers
flattered MAttr and any ranking read off them was not a ranking at a fixed budget.

Wall clock is NOT matched and cannot be: MAttr is ~7x slower per step here (4.3s vs 0.6s per
100 at batch 2048) because `sigmoid_topk`'s 50-iteration bisection over 16384 units dominates
a model this small. That is an honest cost of the method, but it is a cost of the MASK
PRIMITIVE rather than of the objective, and it does not scale like the model does -- so
reporting it as the budget would make the comparison a statement about toy-model size. Both
numbers are printed; the curves are indexed by forward/backward count.

ONE TRAJECTORY PER RUN, SNAPSHOTTED, rather than one run per budget. Restarting per budget
would multiply the cost by the number of points and, worse, would compare a DIFFERENT random
k/alpha/batch sequence at each budget, turning a within-run trajectory into a between-run
contrast with its own noise. `scripts/toy_sgd_vs_ig.py:sgd_trajectory` does it the same way
and says so.

SEEDS ARE NOT OPTIONAL. All three methods are one-draw-per-step estimators, so "Adam beats SGD
by 0.03 precision" is meaningless until you know what two seeds of the same method differ by.
`--collect` prints the across-seed spread beside every gap; a gap smaller than that spread is
not a result, and the honest conclusion is "more seeds", not a winner.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_attrib as IA  # noqa: E402  (same dir; see sys.path insert below)
import interference_toy as IT
from interference_toy import sample_x, sweep

SNAPS = (256, 1000, 3000, 10000, 30000, 60000, 120000)


def metrics(score, dl, eps):
    c = sweep(score, dl)
    ra = score.reshape(-1).argsort().argsort().float()
    rb = dl.reshape(-1).argsort().argsort().float()
    rho = float(((ra - ra.mean()) / ra.std() * (rb - rb.mean()) / rb.std()).mean())

    def at(t):
        return c["precision"][next(i for i, x in enumerate(c["recall"]) if x >= t)]

    return {"p20": at(.2), "p40": at(.4), "p80": at(.8), "rho": rho,
            "max_gain": max(c["loss_gain"])}


def ig_trajectory(U, b, A, v, snaps, batch, seed):
    """Running mean of -g*U over alpha draws, read off at each budget."""
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed + 30_000)
    acc, out = torch.zeros_like(U), {}
    for i in range(1, max(snaps) + 1):
        x = sample_x(batch, g)
        Uv = U.clone().requires_grad_(True)
        alpha = torch.rand(batch, 1, generator=g)
        y = F.relu(x @ A.T + v)
        ((F.relu(alpha * (x @ Uv.T) + b) - y).pow(2).sum(-1).mean()).backward()
        acc += Uv.grad
        if i in snaps:
            out[i] = -(acc / i) * U
    return out


def mattr_trajectory(U, b, A, v, optimizer, snaps, batch, lr, seed):
    """One `learn_scores` run, snapshotted through its `on_step` hook."""
    from learning_to_attribute import learn_scores
    n = U.shape[0]
    g = torch.Generator().manual_seed(seed + 40_000)
    out = {}

    def loss_fn(mask):
        return IA.loss_on(U * mask.view(n, n), b, A, v, sample_x(batch, g))

    def on_step(step, k, loss, scores):
        if (step + 1) in snaps:
            out[step + 1] = scores.detach().clone().view(n, n)

    learn_scores(n * n, loss_fn, steps=max(snaps), variant="topk", k_schedule="log",
                 T=0.5, lr=lr, optimizer=optimizer, on_step=on_step, log_every=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hard")
    p.add_argument("--method", choices=("ixg:mc", "adam", "sgd"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr-adam", type=float, default=0.05)
    p.add_argument("--lr-sgd", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=None,
                   help="override the method's default lr; recorded in the filename so an lr "
                        "sweep does not overwrite the default-lr grid")
    p.add_argument("--max-budget", type=int, default=30000)
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()

    d = IT.OUT.parent / f"{IT.OUT.name}_{args.tag}"
    blob = json.loads((d / "curves.json").read_text())
    eps = blob["meta"]["eps_real"]

    if args.collect:
        rows = {}
        for f in sorted(d.glob("budget_*.json")):
            r = json.loads(f.read_text())
            key = r["method"] if "_lr" not in f.name else f'{r["method"]}@{r["lr"]:g}'
            rows.setdefault(key, {}).setdefault(r["seed"], r["metrics"])
        print(f"budget = forward/backward passes of the toy model (batch {args.batch}); "
              f"mean over seeds, spread = max-min\n")
        for metric in ("p20", "p40", "p80", "rho"):
            print(f"  --- {metric} ---")
            for meth, byseed in rows.items():
                cells = []
                for s in map(str, SNAPS):
                    vals = [byseed[k][s][metric] for k in byseed if s in byseed[k]]
                    if not vals:
                        cells.append("     -  ")
                        continue
                    spread = max(vals) - min(vals)
                    cells.append(f"{sum(vals)/len(vals):.2f}±{spread:.2f}")
                print(f"    {meth:<8} " + "  ".join(f"{c:>9}" for c in cells)
                      + f"   (n={len(byseed)} seeds)")
            print()
        print("  budgets: " + "  ".join(f"{s:>9}" for s in SNAPS))
        return

    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    snaps = tuple(x for x in SNAPS if x <= args.max_budget)
    lr = args.lr
    if args.method == "ixg:mc":
        traj = ig_trajectory(U, b, A, v, snaps, args.batch, args.seed)
    else:
        lr = args.lr or (args.lr_adam if args.method == "adam" else args.lr_sgd)
        traj = mattr_trajectory(U, b, A, v, args.method, snaps, args.batch, lr, args.seed)
    out = {"method": args.method, "seed": args.seed, "lr": lr,
           "metrics": {str(k): metrics(s, dl, eps) for k, s in traj.items()}}
    suffix = "" if args.lr is None else f"_lr{args.lr:g}"
    fn = d / f"budget_{args.method.replace(':', '-')}{suffix}_{args.seed}.json"
    fn.write_text(json.dumps(out))
    for k, mm in out["metrics"].items():
        print(f"  {args.method:<8} seed {args.seed} budget {k:>6}  "
              f"P@R.2 {mm['p20']:.2f}  P@R.4 {mm['p40']:.2f}  P@R.8 {mm['p80']:.2f}  "
              f"rho {mm['rho']:+.3f}", flush=True)
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
