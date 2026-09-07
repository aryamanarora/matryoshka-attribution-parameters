"""Is zero-init MAttr+SGD under the `logit` k-schedule the SAME RANKING as stepless IG?

    uv run python scripts/interference/toy_sgd_vs_ig.py            # seconds, CPU, no download

WHAT THE THEORY SAYS. `gradient.tex` §B.2: at zero init the sigmoid-top-k mask is uniform at
alpha = k/N, and the implicit-diff backward multiplies dL/dm by the gate slope alpha*(1-alpha), so
a zero-init SGD run accumulates a PATH INTEGRAL of g.delta with weight w(alpha) ~ alpha*(1-alpha)*p(alpha).
The `logit` schedule sets p ~ 1/(alpha*(1-alpha)), which cancels the gate slope exactly and leaves
w flat -- i.e. plain activation-path IG, up to a positive constant and a mean shift. Both of those
are RANK-IRRELEVANT (a positive scale and a constant offset do not reorder), so under `logit` the
two methods should induce the SAME ORDERING, and top-k reads only the ordering.

WHY THIS SCRIPT EXISTS. On the real fr2de/Qwen-14B delta they do NOT agree: Spearman 0.593, and
`logit` is no closer to IG than `log_both` is (0.600). The obvious explanation -- scores escaping
the near-zero regime so the derivation stops applying -- was checked on that run and is FALSE: the
top-1% score cut is 0.0016*T, exactly one unit of 2.58M exceeds 0.5, and sigma' = 0.2500 for
2,580,403 of them. Nothing saturates. So either the disagreement is estimator NOISE, or something
in the implementation departs from the derivation. A 2.58M-unit 14B run cannot separate those; a
toy with an exactly-computable ground truth can.

THE DESIGN, and each choice is forced by what has to be distinguishable:

* The ground truth is QUADRATURE, not another sampler. `ig_exact` integrates g.delta over a dense
  alpha grid, so it is the path integral itself rather than a low-variance estimate of it. Without
  it, "SGD disagrees with IG" is unattributable -- either could be the one that is wrong.
* The loss must be NONLINEAR in theta. Under a linear loss g.delta is constant along the path,
  every path weight integrates to the same ranking, and the whole comparison passes trivially
  while telling you nothing. A 2-layer MLP with fixed random data gives a curved path.
* UPSTREAM'S OWN CODE does the sampling and masking (`sample_k`, `sigmoid_topk` via `build_mask`),
  not a restatement of it here. The claim under test is about the shipped implementation; a clean
  reimplementation would test the paper instead and would agree with it by construction.
* Both estimators are swept over BUDGET. If the real-run gap is variance, both converge on the
  quadrature answer as draws grow and the interesting number is how fast. If `logit`-SGD plateaus
  short of it while stepless IG converges, the departure is systematic and the toy has localised
  it to the mask/backward path.

READ THE `logit` COLUMN AGAINST `uniform`/`log`. Those two have non-flat w by construction, so they
SHOULD converge to a different ranking than IG -- they are the negative control that proves the
comparison can detect a real difference at all. If every schedule agrees with IG equally well, the
toy is too easy and the correlations are being carried by the delta's magnitude structure.
"""

import math
import sys

import torch

sys.path.insert(0, "../learning-to-attribute/src")
from learning_to_attribute.masks import build_mask  # noqa: E402
from learning_to_attribute.schedules import sample_k  # noqa: E402

T, N_IN, N_HID, N_OUT, N_DATA = 0.5, 8, 24, 4, 64


def make_problem(seed=0):
    """A tiny MLP, a fixed dataset, and a delta that is a real (short) finetune of it."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N_DATA, N_IN, generator=g)
    y = torch.randint(0, N_OUT, (N_DATA,), generator=g)
    base = [torch.randn(N_IN, N_HID, generator=g) / math.sqrt(N_IN),
            torch.randn(N_HID, N_OUT, generator=g) / math.sqrt(N_HID)]
    ft = [b.clone().requires_grad_(True) for b in base]
    opt = torch.optim.Adam(ft, lr=0.05)
    for _ in range(150):                       # a real finetune, so the delta is not white noise
        opt.zero_grad()
        loss(ft, x, y).backward()
        opt.step()
    delta = [(f.detach() - b) for f, b in zip(ft, base)]
    return x, y, base, delta


def loss(params, x, y):
    h = torch.tanh(x @ params[0])
    return torch.nn.functional.cross_entropy(h @ params[1], y)


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def grad_dot_delta(base, delta, x, y, scale):
    """`g . delta` per unit, with the delta applied UNIFORMLY at `scale` -- one path point."""
    ps = [(b + scale * d).clone().requires_grad_(True) for b, d in zip(base, delta)]
    loss(ps, x, y).backward()
    return flat([p.grad * d for p, d in zip(ps, delta)])


def ig_exact(base, delta, x, y, n=2001):
    """THE GROUND TRUTH: the path integral by dense quadrature, not by sampling."""
    acc = torch.zeros(flat(delta).numel())
    for i in range(n):
        acc += grad_dot_delta(base, delta, x, y, (i + 0.5) / n)
    return acc / n


def ig_stepless(base, delta, x, y, n, seed=0):
    """Our `ixg_at: mc`: alpha ~ U(0,1), one draw per step, averaged."""
    torch.manual_seed(seed)
    acc = torch.zeros(flat(delta).numel())
    for _ in range(n):
        acc += grad_dot_delta(base, delta, x, y, torch.rand(1).item())
    return acc / n


def mattr_sgd(base, delta, x, y, n, schedule, lr=1.0, seed=0):
    """Zero-init MAttr+SGD through upstream's `sigmoid_topk`, k drawn by upstream's `sample_k`."""
    torch.manual_seed(seed)
    total = flat(delta).numel()
    scores = torch.zeros(total, requires_grad=True)
    opt = torch.optim.SGD([scores], lr=lr)
    sizes = [d.numel() for d in delta]
    for _ in range(n):
        opt.zero_grad()
        m = build_mask(scores, k=sample_k(total, schedule), variant="topk", T=T).mask
        parts, off = [], 0
        for d, s in zip(delta, sizes):
            parts.append(m[off:off + s].view_as(d))
            off += s
        loss([b + p * d for b, p, d in zip(base, parts, delta)], x, y).backward()
        opt.step()
    return scores.detach()


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return (ra * rb).mean().item()


def main():
    x, y, base, delta = make_problem()
    n_units = flat(delta).numel()
    truth = ig_exact(base, delta, x, y)
    print(f"toy: {n_units} units, {N_DATA} examples, |delta| = {flat(delta).norm():.3f}")
    print(f"ground truth = path integral by {2001}-point quadrature\n")

    print(f"{'budget':>8} | {'steplessIG':>11} | {'SGD logit':>10} {'SGD log':>9} "
          f"{'SGD unif':>9} {'SGD lb':>9}   (Spearman vs exact IG)")
    for n in (16, 64, 256, 1024, 4096):
        ig = spearman(ig_stepless(base, delta, x, y, n), truth)
        row = f"{n:>8} | {ig:>11.4f} | "
        for sch in ("logit", "log", "uniform", "log_both"):
            row += f"{spearman(mattr_sgd(base, delta, x, y, n, sch), truth):>9.4f} "
        print(row)

    # The head of the ranking is what a sparse mask actually reads, and a global Spearman over
    # mostly-irrelevant units can look healthy while the top-k disagrees completely.
    print(f"\ntop-10% overlap with exact IG, at 4096 draws:")
    k = max(1, n_units // 10)
    ref = set(truth.topk(k).indices.tolist())
    got = {"steplessIG": ig_stepless(base, delta, x, y, 4096)}
    for sch in ("logit", "log", "uniform", "log_both"):
        got[f"SGD {sch}"] = mattr_sgd(base, delta, x, y, 4096, sch)
    for name, s in got.items():
        print(f"  {name:<12} {len(ref & set(s.topk(k).indices.tolist())) / k:.3f}")


if __name__ == "__main__":
    main()


# ---- LR x steps x schedule, tracking the regime diagnostic --------------------------------------

def sgd_trajectory(base, delta, x, y, n, schedule, lr, wd=0.0, seed=0, snaps=()):
    """One SGD run, snapshotting scores at `snaps` -- so a whole steps-axis costs ONE trajectory.

    Restarting per step-count would multiply the cost by the number of checkpoints and, worse,
    would compare DIFFERENT random k/batch sequences at each budget; snapshots keep one sequence so
    the steps axis is a within-run trajectory rather than a between-run contrast.
    """
    torch.manual_seed(seed)
    total = flat(delta).numel()
    scores = torch.zeros(total, requires_grad=True)
    opt = torch.optim.SGD([scores], lr=lr, weight_decay=wd)
    sizes = [d.numel() for d in delta]
    out = {}
    for i in range(1, n + 1):
        opt.zero_grad()
        m = build_mask(scores, k=sample_k(total, schedule), variant="topk", T=T).mask
        parts, off = [], 0
        for d, s in zip(delta, sizes):
            parts.append(m[off:off + s].view_as(d))
            off += s
        loss([b + p * d for b, p, d in zip(base, parts, delta)], x, y).backward()
        opt.step()
        if i in snaps:
            out[i] = scores.detach().clone()
    return out
