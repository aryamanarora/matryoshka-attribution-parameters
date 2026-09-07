"""Replication of the toy model in Olah, Turner & Conerly, "A Toy Model of Interference
Weights" (Transformer Circuits, 2025-07-29), up to and including the two figures in
§ "What Should We Do about Interference Weights?" (the anchor `#filtering-interference-weights`).

    uv run python scripts/interference/interference_toy.py            # ~2 min, CPU, no download
    uv run python scripts/interference/interference_toy.py --check    # + the model-validation figures' numbers

WHAT IS BEING REPLICATED. The note's "more sophisticated" toy model puts a block-structured
linear-ReLU circuit into superposition and asks which of the resulting VIRTUAL weights are real:

    target      y  = ReLU(A x + v)                 A block diagonal, v = -0.1
    model       h  = W_down x
                y' = ReLU(W_up h + b)   =   ReLU(U x + b),   U := W_up W_down

`U` is the n_feat x n_feat matrix of "downstairs" virtual weights. Most of its entries are
INTERFERENCE: they exist only because n_feat features are squeezed into n_res dimensions, and
they do not help the loss. The note's config, quoted verbatim: "If we put 128 features in
superposition in 16 dimensions, with 8 blocks, 0.1 weight density within those blocks, and
input feature density of 0.3". `A`'s within-block entries are "uniformly sampled between
[0,1]"; `v` is "a constant vector of -0.1".

THE 0.1 DENSITY IS THE CHECK THAT THE CONFIG IS RIGHT, not a free knob. 8 blocks x 16x16 x 0.1
= 205 nonzero entries in A out of 128^2 = 16384 virtual weights, i.e. a 1.25% base rate -- and
the published precision-recall curves all terminate at precision ~0.012 at recall 1. If this
script's `n_real / n_weights` is not ~1.2%, the model is not the note's model.

GROUND TRUTH. Defn (2) of the note: dL(U_ij) = L(U with U_ij zeroed) - L(U), so a POSITIVE
dL means ablating the weight hurts, i.e. the weight is real. "Real" is dL > eps = 1e-4, the
note's threshold. All 16384 ablations are computed exactly rather than sampled, which is
cheap here for a reason worth stating: zeroing U_ij perturbs only output i, and only through
its pre-activation, so the whole matrix falls out of one (batch x n_feat) tensor per output
row. dL is a PAIRED difference on one fixed eval set, so its Monte-Carlo error is far below
the error of either loss alone -- which matters, because eps = 1e-4 sits deep inside the
noise band of an unpaired estimate.

THE FOUR HEURISTICS are the note's, with ERA/TWERA taken from the definitions in Ameisen et
al., "Circuit Tracing" (§ Global Weights), which is where the note's `era`/`twera` labels come
from. Writing i for the TARGET (output) feature and j for the SOURCE (input) feature, so that
the attribution of one weight on one example is U_ij * x_j:

    weight   U_ij                                    "big weights are likely to be real"
    era      E[1(y'_i > 0) x_j] * U_ij               "big average effects"
    twera    E[y'_i x_j] / E[y'_i] * U_ij            "big effects on big things"
    freq     E[1(x_j > 0) 1(y'_i > 0)]               "weights that often do something"
    ideal    dL(U_ij)                                the oracle, perfect by construction

Note `freq` carries NO weight magnitude at all -- that is the point of including it, and why
it should land on the 1.25% base rate. The target activation is the model's own output y',
not the ground-truth y: in the CLT setting these statistics are measured on the model, and
using y here would leak the answer (y is a function of A, which is the thing being recovered).

RANKING IS BY SIGNED SCORE, descending, not by |score|. A's entries are drawn from [0,1], so
every real weight is positive and a big NEGATIVE virtual weight is interference by
construction. `--abs` switches to magnitude ranking; it is strictly worse here, as it should
be, and the flag exists so that claim is checkable rather than asserted.

The two curves are computed as exact prefix sweeps over the sorted score (every prefix length
1..16384), not by sampling thresholds -- the note's own annotation says its `ideal` curve
misses (1,1) "because we discretely sampled the pareto frontier", which is an artifact worth
not reproducing.

Outputs land in plots/data/interference_toy/ and are drawn by
plots/plot_interference_filtering.py.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[2] / "plots" / "data" / "interference_toy"

# The note's config for the "more sophisticated" model, verbatim from its prose.
N_FEAT, N_RES, N_BLOCKS = 128, 16, 8
BLOCK_DENSITY, FEAT_DENSITY = 0.1, 0.3
V_CONST = -0.1
EPS_REAL = 1e-4


def make_target(seed: int, identity: bool = False):
    """`A` (block diagonal, sparse within blocks, entries ~ U[0,1]) and the constant `v`.

    `identity=True` is the note's SIMPLE model, whose circuit is `y = ReLU(Id x + b)`: the
    ideal weights are then the identity, so the real weights are exactly the 100 diagonal
    entries and every off-diagonal entry is interference by construction. That makes the base
    rate 1/n_feat with no estimation in it, which is the one thing the sophisticated model
    cannot offer."""
    if identity:
        return torch.eye(N_FEAT), torch.zeros(N_FEAT)
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(N_FEAT, N_FEAT)
    bs = N_FEAT // N_BLOCKS
    for b in range(N_BLOCKS):
        s = slice(b * bs, (b + 1) * bs)
        vals = torch.rand(bs, bs, generator=g)
        keep = torch.rand(bs, bs, generator=g) < BLOCK_DENSITY
        A[s, s] = vals * keep
    return A, torch.full((N_FEAT,), V_CONST)


def sample_x(n: int, g: torch.Generator):
    """Sparse non-negative features: active w.p. FEAT_DENSITY, active value ~ U[0,1]."""
    return torch.rand(n, N_FEAT, generator=g) * (
        torch.rand(n, N_FEAT, generator=g) < FEAT_DENSITY)


def train(A, v, seed: int, steps: int, batch: int, lr: float, log_every: int = 0,
          down: str = "learned", tied: bool = False, down_seed=None):
    """Fit W_up / b (and W_down, unless frozen) to ReLU(Ax+v). Loss sums over features, means
    over the batch -- the original Toy Models normalisation, and the one that puts dL on the
    note's scale.

    `down="random"` FREEZES W_down as a fixed random projection with unit-norm columns, which
    is a deliberate departure from the note's model and the reason it exists is measured, not
    stylistic. With both matrices learned, the model has enough freedom to CHOOSE a
    superposition geometry in which interference nearly cancels: it shrinks the norms of the
    ~24 outputs and inputs that A never uses, lands at interference std ~0.09 against real
    weights averaging ~0.41, and `|U_ij| > t` alone becomes a near-perfect classifier -- i.e.
    it fails the note's own criterion (1), "real weights and interference weights strongly
    overlap", which is what makes the filtering problem hard in the first place. That is not a
    tuning failure: it survives input density 0.02-0.5, block density 0.1-0.5, 300-120k steps
    (converged by 3k), init scale x1-x4, learned/fixed bias, and n_residual 16 down to 4, and
    in every one of those cells ERA is a perfect oracle where the note has it decaying to
    ~0.55.

    Freezing the down-projection removes exactly that freedom. Each source feature gets a
    fixed random direction in R^16, so an output's 16 free parameters must meet ~104 live
    constraints; the leftover is a least-squares residual, which is unavoidable interference
    at the scale superposition actually implies, and the fit to A's real entries shrinks to
    trade against it. See docs/interference_toy.md for the numbers either way."""
    g = torch.Generator().manual_seed(seed + 10_000)
    if tied:
        # The note's FIRST toy model: one matrix, used both ways (h = Wx, y' = ReLU(W^T h + b)),
        # so U = W^T W is symmetric and the circuit being approximated is the IDENTITY. Kept
        # separate from the untied path rather than folded into it because the two have
        # different ideal weights -- Id here, A there -- and `delta_loss` is the only thing
        # downstream that must not notice the difference.
        W = (torch.randn(N_RES, N_FEAT, generator=g) / math.sqrt(N_FEAT)).requires_grad_(True)
        b = torch.zeros(N_FEAT, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for step in range(steps):
            x = sample_x(batch, g)
            y = F.relu(x @ A.T + v)
            loss = (F.relu(x @ (W.T @ W).T + b) - y).pow(2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"  step {step:>6}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}")
        return (W.detach().T @ W.detach()), b.detach()
    # `down_seed` pins the frozen projection to one seed across models, so that two `--check`
    # seeds differ only in init and data order -- the note's own definition of a different
    # seed. Without it each seed also draws its own projection, and the two models' interference
    # geometries have nothing in common by construction (docs/interference_toy.md).
    W_down = torch.randn(N_RES, N_FEAT, generator=(
        torch.Generator().manual_seed(down_seed + 10_000) if down_seed is not None else g))
    if down == "random":
        # unit-norm columns: one fixed direction per source feature, no scale for the model
        # to hide behind, so the interference floor is set by 128-in-16 and nothing else.
        W_down = (W_down / W_down.norm(dim=0, keepdim=True))
    else:
        W_down = (W_down / math.sqrt(N_FEAT)).requires_grad_(True)
    W_up = (torch.randn(N_FEAT, N_RES, generator=g) / math.sqrt(N_RES)).requires_grad_(True)
    b = torch.zeros(N_FEAT, requires_grad=True)
    params = [W_up, b] + ([W_down] if down != "random" else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        x = sample_x(batch, g)
        y = F.relu(x @ A.T + v)
        yp = F.relu(x @ (W_up @ W_down).T + b)
        loss = (yp - y).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:>6}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}")
    return (W_up.detach() @ W_down.detach()), b.detach()


def eval_batches(A, v, n: int, chunk: int, seed: int):
    """Yield (x, y) over a fixed eval set. Chunked so nothing above ~(chunk x n_feat) lives."""
    g = torch.Generator().manual_seed(seed + 20_000)
    for _ in range(0, n, chunk):
        x = sample_x(min(chunk, n), g)
        yield x, F.relu(x @ A.T + v)


def delta_loss(U, b, A, v, n, chunk, seed):
    """dL(U_ij) = L(ablate U_ij) - L(U), for all 16384 weights at once, exactly.

    Ablating U_ij shifts only pre-activation i, and only by U_ij * x_j, so for a fixed output
    row i the whole row of dL is one (batch x n_feat) tensor. Accumulating the PAIRED
    difference (not the two losses separately) is what keeps the estimator's error far below
    eps = 1e-4."""
    acc = torch.zeros(N_FEAT, N_FEAT)
    base_loss, seen = 0.0, 0
    for x, y in eval_batches(A, v, n, chunk, seed):
        z = x @ U.T + b                                  # (B, n_feat) pre-activations
        base = (F.relu(z) - y).pow(2)                    # (B, n_feat) per-output sq. error
        base_loss += base.sum().item()
        for i in range(N_FEAT):
            d = U[i].unsqueeze(0) * x                    # (B, n_feat) removed contribution
            abl = (F.relu(z[:, i:i + 1] - d) - y[:, i:i + 1]).pow(2)
            acc[i] += (abl - base[:, i:i + 1]).sum(0)
        seen += x.shape[0]
    return acc / seen, base_loss / seen


def statistics(U, b, A, v, n, chunk, seed):
    """The on-distribution coactivation statistics ERA / TWERA / freq are built from."""
    n_src = torch.zeros(N_FEAT, N_FEAT)      # E[1(y'_i>0) x_j]
    e_yx = torch.zeros(N_FEAT, N_FEAT)       # E[y'_i x_j]
    co = torch.zeros(N_FEAT, N_FEAT)         # E[1(x_j>0) 1(y'_i>0)]
    e_y = torch.zeros(N_FEAT)                # E[y'_i]
    seen = 0
    for x, _y in eval_batches(A, v, n, chunk, seed):
        yp = F.relu(x @ U.T + b)
        act = (yp > 0).float()
        n_src += act.T @ x
        e_yx += yp.T @ x
        co += act.T @ (x > 0).float()
        e_y += yp.sum(0)
        seen += x.shape[0]
    return n_src / seen, e_yx / seen, co / seen, e_y / seen


def heuristics(U, stats):
    n_src, e_yx, co, e_y = stats
    return {
        "weight": U.clone(),
        "era": n_src * U,
        # E[y'_i] can be 0 for an output that never fires; those rows have no attribution to
        # rank anyway, so guard rather than propagate a nan through the sort.
        "twera": (e_yx / e_y.clamp_min(1e-12).unsqueeze(1)) * U,
        "freq": co.clone(),
    }


def sweep(score, dl, eps=EPS_REAL, use_abs=False):
    """Exact prefix sweep: for every k, the top-k set by `score` gives one (precision, recall,
    loss gain) point. Loss gain is the note's -- the plain sum of dL over the kept set."""
    s = score.abs() if use_abs else score
    order = torch.argsort(s.reshape(-1), descending=True)
    real = (dl.reshape(-1) > eps)[order].float()
    gain = dl.reshape(-1)[order]
    k = torch.arange(1, real.numel() + 1, dtype=torch.float64)
    hits = real.double().cumsum(0)
    return {
        "precision": (hits / k).tolist(),
        "recall": (hits / real.sum().double()).tolist(),
        "loss_gain": gain.double().cumsum(0).tolist(),
    }


def main():
    global N_RES, N_FEAT, FEAT_DENSITY, OUT
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-n", type=int, default=262_144)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--abs", action="store_true", help="rank by |score| instead of signed score")
    p.add_argument("--tag", default="", help="suffix for the output dir, so two configs "
                                             "can sit side by side instead of overwriting")
    p.add_argument("--simple", action="store_true",
                   help="the note's FIRST toy model: tied W, identity circuit, n_feat 100 / "
                        "n_res 20 / density 0.02, deliberately undertrained")
    p.add_argument("--n-res", type=int, default=N_RES,
                   help="residual width. 16 is the note's; a FROZEN projection needs more "
                        "(see docs/interference_toy.md) because 16 random dims compress far "
                        "harder than 16 the model got to choose.")
    p.add_argument("--down", choices=("learned", "random"), default="learned",
                   help="learned = the note's model; random = frozen random down-projection")
    p.add_argument("--check", action="store_true", help="also fit a second seed for the scatter")
    p.add_argument("--shared-down", action="store_true",
                   help="with --down random: every model in this run uses the projection drawn "
                        "from --seed, so --check's second seed changes only init and data order")
    args = p.parse_args()
    N_RES = args.n_res
    if args.tag:
        OUT = OUT.parent / f"{OUT.name}_{args.tag}"
    if args.simple:
        # The note's stated numbers for it: "n_features=100, n_residual=20, and
        # feature_density=0.02. It will also be undertrained". `--steps` still overrides, and
        # under-training is the point rather than an accident -- see docs/interference_toy.md.
        N_FEAT, N_RES, FEAT_DENSITY = 100, (20 if args.n_res == 16 else args.n_res), 0.02

    torch.manual_seed(args.seed)
    A, v = make_target(args.seed, identity=args.simple)
    n_nonzero = int((A > 0).sum())
    print(f"target A: {n_nonzero} nonzero of {N_FEAT * N_FEAT} "
          f"({n_nonzero / N_FEAT ** 2:.3%} -- the note's base rate is ~1.25%)")

    print(f"training toy model ({args.steps} steps, batch {args.batch}, Adam lr {args.lr}, "
          f"down={args.down})")
    down_seed = args.seed if args.shared_down else None
    U, b = train(A, v, args.seed, args.steps, args.batch, args.lr,
                 log_every=args.steps // 6, down=args.down, tied=args.simple, down_seed=down_seed)

    print(f"computing dL for all {N_FEAT ** 2} virtual weights on {args.eval_n} examples")
    dl, base_loss = delta_loss(U, b, A, v, args.eval_n, args.chunk, args.seed)
    stats = statistics(U, b, A, v, args.eval_n, args.chunk, args.seed)

    real = dl > EPS_REAL
    n_real = int(real.sum())
    # How much of A's support dL actually recovers -- the note never states this, but if the
    # two disagree badly the "ground truth" is measuring something other than the circuit.
    tp = int((real & (A > 0)).sum())
    print(f"loss {base_loss:.5f} | real weights (dL > {EPS_REAL}): {n_real} "
          f"({n_real / N_FEAT ** 2:.3%}) | {tp}/{n_nonzero} of A's support recovered")
    print(f"sum dL (all weights) = {dl.sum():.4f}   "
          f"sum dL over real = {dl[real].sum():.4f} (the Oracle point)")

    curves = {name: sweep(s, dl, use_abs=args.abs)
              for name, s in heuristics(U, stats).items()}
    curves["ideal"] = sweep(dl, dl)

    meta = {
        "n_feat": N_FEAT, "n_res": N_RES, "n_blocks": N_BLOCKS,
        "block_density": BLOCK_DENSITY, "feat_density": FEAT_DENSITY,
        "eps_real": EPS_REAL, "seed": args.seed, "steps": args.steps,
        "batch": args.batch, "lr": args.lr, "eval_n": args.eval_n,
        "abs_ranking": args.abs, "loss": base_loss, "down": args.down,
        "simple": args.simple, "shared_down": args.shared_down,
        "n_weights": N_FEAT ** 2, "n_real": n_real, "n_A_nonzero": n_nonzero,
        "n_A_recovered": tp,
        "sum_dl_all": float(dl.sum()), "sum_dl_real": float(dl[real].sum()),
        "sum_dl_positive": float(dl[dl > 0].sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curves.json").write_text(json.dumps({"meta": meta, "curves": curves}))
    torch.save({"U": U, "b": b, "A": A, "v": v, "dl": dl}, OUT / "model.pt")

    if args.check:
        # The note validates this config with a two-seed weight scatter ("the interference
        # weights are independent, but the real weights are all significantly positive") and a
        # dL-coloured weight histogram. Both need a second, independently trained model.
        print("fitting a second seed for the two-model scatter")
        A2, v2 = make_target(args.seed + 1, identity=args.simple)
        U2, b2 = train(A2, v2, args.seed + 1, args.steps, args.batch, args.lr,
                       down=args.down, tied=args.simple, down_seed=down_seed)
        # Same A, different init/data order is the note's "different random seed"; a different
        # A would make the two models solve different problems and the scatter meaningless.
        U2s, _b2s = train(A, v, args.seed + 1, args.steps, args.batch, args.lr,
                          down=args.down, tied=args.simple, down_seed=down_seed)
        torch.save({"U2": U2s, "U2_other_A": U2, "b2": b2}, OUT / "model_seed2.pt")
        rr = torch.corrcoef(torch.stack([U[real], U2s[real]]))[0, 1]
        ri = torch.corrcoef(torch.stack([U[~real], U2s[~real]]))[0, 1]
        print(f"across-seed corr: real weights {rr:.3f}   interference {ri:.3f}  "
              f"(the note: real agree, interference are independent)")

    print(f"wrote {OUT}/curves.json")


if __name__ == "__main__":
    main()
