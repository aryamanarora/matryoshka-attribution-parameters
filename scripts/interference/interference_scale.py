"""The interference-weight filtering task at SMALLER scales: does attribution get easier with
fewer virtual weights, and where exactly do the methods disagree?

    # one (n_feat, model seed) per process; scripts/interference/run_interference_scale.sh runs the grid
    uv run python scripts/interference/interference_scale.py --n-feat 32 --seed 0
    uv run python scripts/interference/interference_scale.py --collect          # -> summary.json

WHAT IS SWEPT. The note's "sophisticated" model has 128 features in 16 residual dims, 8 blocks
of 16, i.e. 16384 virtual weights. Here `n_feat` runs over {16, 32, 64, 128} = 256 .. 16384
virtual weights, two decades. What is held FIXED across the sweep, and why:

  - `n_res = n_feat / 8` -- the superposition ratio. Under the frozen random projection
    (`--down random`, the discriminating config from docs/interference_toy.md) each output row
    has `n_res` free numbers against ~`0.3 * n_feat` live constraints, so a fixed ratio keeps
    the per-row interference at a comparable level. Holding `n_res = 16` instead would remove
    superposition entirely at n_feat = 16.
  - block size 16 -- so the within-block structure of `A` (16x16 at 0.1 density, ~26 real
    weights a block) is the note's at every scale, and only the NUMBER of blocks changes. The
    alternative, 8 blocks at every scale, gives ~13 real weights at n_feat 32 and ~3 at 16,
    which is nothing to compute a precision on. The cost is that the base rate is NOT held:
    it is 0.1 / n_blocks, i.e. 1.25% / 2.5% / 5% / 10%, and at n_feat = 16 there is one block
    and no off-block interference. Every number below is reported against its own base rate.

THE QUESTION. The hunch this exists to test: MAttr+Adam and stepless IG (and MAttr+SGD) agree
about the CIRCUIT weights and do different, possibly arbitrary, things on the NOISE weights.
So every agreement statistic here is computed three times -- over all weights, over the real
ones, and over the interference ones -- and the third is the one to read. Two labellings:
`real` = `dL > eps` (the note's oracle definition, what the P/R curves score against) and
`circuit` = `A_ij > 0` (the ground truth by construction; differs from `real` by the
unlearned circuit entries and by the interference the loss happens to like). Both are stored.

Three kinds of agreement, each split that way:
  - method vs oracle `dL`     -- Spearman, and P@R / precision at k = n_real
  - method vs method          -- Spearman, and Jaccard of the two top-n_real sets (also the
                                  Jaccard of their noise members alone)
  - method vs ITSELF          -- the same method refit with another attribution seed on the
                                  same model. If noise scores are arbitrary, this is where it
                                  shows: a method that ranks real weights the same way every
                                  time and interference weights a different way every time.

Plus the diffusion signature (`std` of the interference scores, and the real/interference
separation in units of it) at each snapshotted budget, and the TRUE loss of each ranking's best
top-k set on real masked forwards -- which at n_feat 128 is where MAttr+Adam keeps ~1250
off-circuit weights and beats the model it was fitted to (docs/interference_toy.md), and the
question is whether that is a large-scale phenomenon.

Everything reuses `scripts/interference/interference_toy.py` (model, oracle, heuristics) and
`scripts/interference/interference_attrib.py` (the three methods) by setting the toy module's size globals;
nothing about the task is restated here.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_attrib as IA  # noqa: E402
import interference_toy as IT  # noqa: E402
from interference_toy import EPS_REAL, sample_x, sweep  # noqa: E402

from learning_to_attribute import learn_scores  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "plots" / "data" / "interference_scale"
SNAPS = (256, 1000, 3000)
BLOCK_SIZE = 16
N_EVAL, CHUNK = 262_144, 8192
N_EVAL_TRUE = 65_536


# ----------------------------------------------------------------------------- statistics
def spearman(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    if a.numel() < 3:
        return float("nan")
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = ra.norm() * rb.norm()
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def topk_set(score, k):
    return set(score.reshape(-1).argsort(descending=True)[:k].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def vs_oracle(score, dl, real, circuit):
    """One method against the oracle: rank agreement on each population, and the P/R points."""
    c = sweep(score, dl)
    n_real = int(real.sum())

    def pat(t):
        return c["precision"][next(i for i, x in enumerate(c["recall"]) if x >= t)]

    top = topk_set(score, n_real)
    real_idx = set(real.reshape(-1).nonzero().reshape(-1).tolist())
    circ_idx = set(circuit.reshape(-1).nonzero().reshape(-1).tolist())
    return {
        "rho_all": spearman(score, dl),
        "rho_real": spearman(score[real], dl[real]),
        "rho_noise": spearman(score[~real], dl[~real]),
        "rho_circuit": spearman(score[circuit], dl[circuit]),
        "rho_offcircuit": spearman(score[~circuit], dl[~circuit]),
        "p20": pat(.2), "p40": pat(.4), "p80": pat(.8),
        "prec_at_nreal": len(top & real_idx) / n_real,
        "circuit_frac_at_nreal": len(top & circ_idx) / n_real,
        "max_gain": max(c["loss_gain"]),
    }


def vs_method(s1, s2, real, circuit, n_real):
    """Two rankings against each other, on each population, and the overlap of their top sets."""
    t1, t2 = topk_set(s1, n_real), topk_set(s2, n_real)
    real_idx = set(real.reshape(-1).nonzero().reshape(-1).tolist())
    circ_idx = set(circuit.reshape(-1).nonzero().reshape(-1).tolist())
    return {
        "rho_all": spearman(s1, s2),
        "rho_real": spearman(s1[real], s2[real]),
        "rho_noise": spearman(s1[~real], s2[~real]),
        "rho_circuit": spearman(s1[circuit], s2[circuit]),
        "rho_offcircuit": spearman(s1[~circuit], s2[~circuit]),
        "jaccard_top": jaccard(t1, t2),
        "jaccard_top_real": jaccard(t1 & real_idx, t2 & real_idx),
        "jaccard_top_noise": jaccard(t1 - real_idx, t2 - real_idx),
        "jaccard_top_circuit": jaccard(t1 & circ_idx, t2 & circ_idx),
        "jaccard_top_offcircuit": jaccard(t1 - circ_idx, t2 - circ_idx),
    }


def spread(score, real):
    """The diffusion signature: how wide the interference scores are, and how far above them
    the real scores sit, in units of that width."""
    sn, sr = score[~real], score[real]
    return {
        "std_noise": float(sn.std()), "std_real": float(sr.std()),
        "mean_noise": float(sn.mean()), "mean_real": float(sr.mean()),
        "sep_sigma": float((sr.mean() - sn.mean()) / sn.std()) if sn.std() > 0 else float("nan"),
        "abs_max": float(score.abs().max()),
    }


# ----------------------------------------------------------------------------- the methods
def ig_trajectory(U, b, A, v, snaps, batch, seed):
    """Stepless IG, running mean over alpha draws, read off at each budget (one trajectory)."""
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
    """One upstream `learn_scores` run, snapshotted through `on_step`."""
    n = U.shape[0]
    g = torch.Generator().manual_seed(seed + 40_000)
    out = {}

    def loss_fn(mask):
        return IA.loss_on(U * mask.view(n, n), b, A, v, sample_x(batch, g))

    def on_step(step, k, loss, scores):
        if (step + 1) in snaps:
            out[step + 1] = scores.detach().clone().view(n, n)

    learn_scores(n * n, loss_fn, steps=max(snaps), variant="topk", k_schedule="log",
                 T=0.5, lr=lr, optimizer=optimizer, on_step=on_step)
    return out


def true_loss_sweep(U, b, A, v, rankings, circuit, seed):
    """Real masked forwards for each ranking's top-k over a log grid of k: the honest version
    of the loss-gain axis, and the number that says whether a ranking's off-circuit picks are
    worth anything as a SET."""
    n2 = U.numel()
    batches = list(IT.eval_batches(A, v, N_EVAL_TRUE, CHUNK, seed))
    nex = sum(x.shape[0] for x, _ in batches)

    def loss(mask):
        Um = U * mask.view_as(U)
        with torch.no_grad():
            return sum((F.relu(x @ Um.T + b) - y).pow(2).sum().item() for x, y in batches) / nex

    grid = sorted({int(round(k)) for k in torch.logspace(0, math.log10(n2), 40).tolist()}
                  | {int(circuit.sum()), n2})
    out = {"grid": grid, "L_full": loss(torch.ones_like(U)),
           "L_circuit": loss(circuit.float()), "curves": {}}
    for name, s in rankings.items():
        order = s.reshape(-1).argsort(descending=True)
        curve = []
        for k in grid:
            mk = torch.zeros(n2)
            mk[order[:k]] = 1.
            curve.append(loss(mk))
        kbest = grid[min(range(len(grid)), key=lambda i: curve[i])]
        kept = torch.zeros(n2, dtype=torch.bool)
        kept[order[:kbest]] = True
        kept = kept.view_as(U)
        off = kept & ~circuit
        out["curves"][name] = {
            "loss": curve, "k_best": kbest, "L_best": min(curve),
            "n_off_at_best": int(off.sum()), "n_on_at_best": int((kept & circuit).sum()),
            "off_positive_frac": float((U[off] > 0).float().mean()) if off.any() else float("nan"),
            "off_median_absU": float(U[off].abs().median()) if off.any() else float("nan"),
        }
    return out


# ----------------------------------------------------------------------------- one cell
def run_cell(args):
    n = args.n_feat
    IT.N_FEAT, IT.N_RES, IT.N_BLOCKS = n, max(1, n // 8), max(1, n // BLOCK_SIZE)
    if args.n_res:
        IT.N_RES = args.n_res
    torch.manual_seed(args.seed)
    t0 = time.time()
    A, v = IT.make_target(args.seed)
    circuit = A > 0
    print(f"[n_feat {n}] n_res {IT.N_RES} blocks {IT.N_BLOCKS} | A support {int(circuit.sum())} "
          f"of {n * n} ({circuit.float().mean():.3%})", flush=True)

    U, b = IT.train(A, v, args.seed, args.train_steps, args.batch, args.train_lr,
                    down=args.down)
    dl, base_loss = IT.delta_loss(U, b, A, v, N_EVAL, CHUNK, args.seed)
    real = dl > EPS_REAL
    n_real = int(real.sum())
    stats = IT.statistics(U, b, A, v, N_EVAL, CHUNK, args.seed)
    print(f"[n_feat {n}] loss {base_loss:.4f} | real (dL>eps) {n_real} ({n_real / n / n:.3%}) | "
          f"{int((real & circuit).sum())}/{int(circuit.sum())} of A recovered | "
          f"{time.time() - t0:.0f}s", flush=True)

    # --- scores: heuristics once, the three methods at every attribution seed and budget
    scores = {}           # (method, attr_seed, budget) -> n x n
    for name, s in IT.heuristics(U, stats).items():
        scores[(name, 0, 0)] = s
    for aseed in range(args.attr_seeds):
        for budget, s in ig_trajectory(U, b, A, v, SNAPS, args.batch, aseed).items():
            scores[("ixg:mc", aseed, budget)] = s
        for lr in args.adam_lrs:
            name = "adam" if lr == args.adam_lrs[0] else f"adam@{lr:g}"
            for budget, s in mattr_trajectory(U, b, A, v, "adam", SNAPS, args.batch, lr,
                                              aseed).items():
                scores[(name, aseed, budget)] = s
        for budget, s in mattr_trajectory(U, b, A, v, "sgd", SNAPS, args.batch, args.lr_sgd,
                                          aseed).items():
            scores[("sgd", aseed, budget)] = s
        print(f"[n_feat {n}] attribution seed {aseed} done, {time.time() - t0:.0f}s", flush=True)

    # --- metrics
    methods = sorted({k[0] for k in scores})
    oracle = {}
    for (name, aseed, budget), s in scores.items():
        oracle[f"{name}|{aseed}|{budget}"] = {**vs_oracle(s, dl, real, circuit),
                                              **spread(s, real)}
    final = {m: {a: scores[(m, a, max(SNAPS))] for a in range(args.attr_seeds)}
             for m in methods if (m, 0, max(SNAPS)) in scores}
    pairs = {}
    for i, m1 in enumerate(final):
        for m2 in list(final)[i + 1:]:
            pairs[f"{m1}|{m2}"] = vs_method(final[m1][0], final[m2][0], real, circuit, n_real)
    selfc = {}
    for m, by in final.items():
        if len(by) < 2:
            continue
        rows = [vs_method(by[a], by[c], real, circuit, n_real)
                for a in by for c in by if a < c]
        selfc[m] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    rankings = {m: final[m][0] for m in final}
    rankings["dL"] = dl
    rankings["random"] = torch.randn_like(U)
    tl = true_loss_sweep(U, b, A, v, rankings, circuit, args.seed)

    meta = {"n_feat": n, "n_res": IT.N_RES, "n_blocks": IT.N_BLOCKS, "block_size": BLOCK_SIZE,
            "n_weights": n * n, "n_real": n_real, "n_circuit": int(circuit.sum()),
            "n_recovered": int((real & circuit).sum()), "seed": args.seed, "down": args.down,
            "loss": base_loss, "train_steps": args.train_steps, "train_lr": args.train_lr,
            "batch": args.batch, "snaps": SNAPS, "attr_seeds": args.attr_seeds,
            "adam_lrs": args.adam_lrs, "lr_sgd": args.lr_sgd, "eps_real": EPS_REAL,
            "sum_dl_real": float(dl[real].sum()), "sum_dl_all": float(dl.sum()),
            "wall_s": time.time() - t0}
    d = OUT / args.tag
    d.mkdir(parents=True, exist_ok=True)
    stem = f"n{n}_s{args.seed}"
    (d / f"{stem}.json").write_text(json.dumps(
        {"meta": meta, "oracle": oracle, "pairs": pairs, "self": selfc, "true_loss": tl}))
    torch.save({"U": U, "b": b, "A": A, "v": v, "dl": dl,
                "scores": {f"{m}|{a}|{bud}": s for (m, a, bud), s in scores.items()}},
               d / f"{stem}.pt")
    for m in final:
        o = oracle[f"{m}|0|{max(SNAPS)}"]
        pr = pairs.get(f"{m}|ixg:mc") or pairs.get(f"ixg:mc|{m}") or {}
        sc = selfc.get(m, {})
        print(f"  {m:<10} vs dL rho all/real/noise {o['rho_all']:+.2f}/{o['rho_real']:+.2f}/"
              f"{o['rho_noise']:+.2f} P@R.2/.8 {o['p20']:.2f}/{o['p80']:.2f} | vs IG "
              f"real/noise {pr.get('rho_real', float('nan')):+.2f}/"
              f"{pr.get('rho_noise', float('nan')):+.2f} | self real/noise "
              f"{sc.get('rho_real', float('nan')):+.2f}/{sc.get('rho_noise', float('nan')):+.2f}"
              f" | true L best {tl['curves'][m]['L_best']:.3f} @k {tl['curves'][m]['k_best']} "
              f"off {tl['curves'][m]['n_off_at_best']}", flush=True)
    print(f"  L(full) {tl['L_full']:.3f}  L(circuit) {tl['L_circuit']:.3f}  "
          f"wrote {d / stem}.json in {time.time() - t0:.0f}s")


def budget_agreement(d, cells):
    """Self-consistency and agreement with stepless IG at EVERY snapshotted budget, computed
    from the saved score trajectories, so the diffusion story (Adam's noise ranking drifting
    with more steps) can be read against scale. Cached into each cell's json as `budget`."""
    for c in cells:
        if "budget" in c:
            continue
        m = c["meta"]
        blob = torch.load(d / f"n{m['n_feat']}_s{m['seed']}.pt")
        dl = blob["dl"]
        real, circuit = dl > m["eps_real"], blob["A"] > 0
        n_real = int(real.sum())
        sc = blob["scores"]
        out = {}
        for meth in ("ixg:mc", "adam", "adam@0.002", "sgd"):
            for bud in m["snaps"]:
                keys = [k for k in sc if k.startswith(f"{meth}|") and k.endswith(f"|{bud}")]
                if not keys:
                    continue
                rows = [vs_method(sc[a], sc[b], real, circuit, n_real)
                        for i, a in enumerate(keys) for b in keys[i + 1:]]
                ig = vs_method(sc[keys[0]], sc[f"ixg:mc|0|{bud}"], real, circuit, n_real)
                out[f"{meth}|{bud}"] = {
                    "self_rho_real": sum(r["rho_real"] for r in rows) / len(rows),
                    "self_rho_noise": sum(r["rho_noise"] for r in rows) / len(rows),
                    "self_jaccard_noise": sum(r["jaccard_top_noise"] for r in rows) / len(rows),
                    "ig_rho_real": ig["rho_real"], "ig_rho_noise": ig["rho_noise"],
                    **{k: v for k, v in c["oracle"][f"{meth}|0|{bud}"].items()
                       if k in ("rho_real", "rho_noise", "p80", "std_noise", "sep_sigma")}}
        c["budget"] = out
        (d / f"n{m['n_feat']}_s{m['seed']}.json").write_text(json.dumps(c))


def collect(tag):
    d = OUT / tag
    cells = [json.loads(f.read_text()) for f in sorted(d.glob("n*_s*.json"))]
    budget_agreement(d, cells)
    (d / "summary.json").write_text(json.dumps(cells))
    by_n = {}
    for c in cells:
        by_n.setdefault(c["meta"]["n_feat"], []).append(c)
    budget = max(SNAPS)

    def mean_sd(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return "   -    "
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        return f"{m:+.2f}±{sd:.2f}"

    print(f"tag {tag}: {len(cells)} cells; mean±sd over model seeds, final budget {budget}\n")
    for n, cs in sorted(by_n.items()):
        m = cs[0]["meta"]
        print(f"=== n_feat {n}: {m['n_weights']} weights, base rate "
              f"{sum(c['meta']['n_real'] for c in cs) / len(cs) / m['n_weights']:.2%}, "
              f"n_real {[c['meta']['n_real'] for c in cs]}, model loss "
              f"{sum(c['meta']['loss'] for c in cs) / len(cs):.3f} ({len(cs)} seeds)")
        methods = [k.split("|")[0] for k in cs[0]["oracle"] if k.endswith(f"|0|{budget}")]
        print(f"  {'method':<10} {'rho_all':>10} {'rho_real':>10} {'rho_noise':>10} "
              f"{'P@R.2':>10} {'P@R.8':>10} | {'IG:real':>10} {'IG:noise':>10} | "
              f"{'self:real':>10} {'self:noise':>10} {'selfJ:noise':>11}")
        for meth in methods:
            o = [c["oracle"][f"{meth}|0|{budget}"] for c in cs]
            pk = f"{meth}|ixg:mc" if meth < "ixg:mc" else f"ixg:mc|{meth}"
            pr = [c["pairs"].get(pk, {}) for c in cs]
            sc = [c["self"].get(meth, {}) for c in cs]
            print(f"  {meth:<10} {mean_sd([x['rho_all'] for x in o]):>10} "
                  f"{mean_sd([x['rho_real'] for x in o]):>10} "
                  f"{mean_sd([x['rho_noise'] for x in o]):>10} "
                  f"{mean_sd([x['p20'] for x in o]):>10} {mean_sd([x['p80'] for x in o]):>10} | "
                  f"{mean_sd([x.get('rho_real', float('nan')) for x in pr]):>10} "
                  f"{mean_sd([x.get('rho_noise', float('nan')) for x in pr]):>10} | "
                  f"{mean_sd([x.get('rho_real', float('nan')) for x in sc]):>10} "
                  f"{mean_sd([x.get('rho_noise', float('nan')) for x in sc]):>10} "
                  f"{mean_sd([x.get('jaccard_top_noise', float('nan')) for x in sc]):>11}")
        print(f"  true loss (best top-k on real forwards):  "
              f"L(full) {mean_sd([c['true_loss']['L_full'] for c in cs])}  "
              f"L(circuit) {mean_sd([c['true_loss']['L_circuit'] for c in cs])}")
        for meth in list(cs[0]["true_loss"]["curves"]):
            t = [c["true_loss"]["curves"][meth] for c in cs]
            print(f"    {meth:<10} L_best {mean_sd([x['L_best'] for x in t])}  k_best "
                  f"{[x['k_best'] for x in t]}  off-circuit kept {[x['n_off_at_best'] for x in t]}")
        print("  by budget (passes): self-rho noise / vs-IG rho noise / vs-dL rho noise / P@R.8 / "
              "noise-score std")
        for meth in ("adam", "adam@0.002", "sgd", "ixg:mc"):
            cellsb = [f"{b:>5}: " + " ".join(
                mean_sd([c["budget"][f"{meth}|{b}"][k] for c in cs if f"{meth}|{b}" in c["budget"]])
                for k in ("self_rho_noise", "ig_rho_noise", "rho_noise", "p80", "std_noise"))
                for b in SNAPS]
            print(f"    {meth:<10} " + " | ".join(cellsb))
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-feat", type=int, default=32)
    p.add_argument("--n-res", type=int, default=0, help="override n_feat/8")
    p.add_argument("--seed", type=int, default=0, help="model seed (target A, init, data)")
    p.add_argument("--attr-seeds", type=int, default=3,
                   help="attribution refits per method on the SAME model")
    p.add_argument("--down", choices=("learned", "random"), default="random")
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--train-lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--adam-lrs", type=float, nargs="+", default=[0.05, 0.002])
    p.add_argument("--lr-sgd", type=float, default=1.0)
    p.add_argument("--tag", default="hard")
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()
    if args.collect:
        collect(args.tag)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
