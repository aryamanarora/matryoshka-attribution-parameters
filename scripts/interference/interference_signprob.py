"""Is MAttr+Adam's interference ranking a SIGN-PROBABILITY over sparse masks? Test it directly.

    uv run python scripts/interference/interference_signprob.py --tags hard hard30k

The mechanism behind the hypothesis: Adam's step is ~ lr * m/sqrt(v), which for a unit whose
gradient flips sign from mask to mask is ~ lr * sign(g_t), so its final score is roughly
lr * (#steps the gradient said "switching this weight on helps" - #steps it said "hurts") over
the masks the log-uniform k-schedule sampled. SGD integrates the raw gradient instead, i.e. the
MAGNITUDE-weighted sum, which is the path-integral reading of scripts/interference/toy_sgd_vs_ig.py. So the
two optimizers should rank interference weights by two different statistics of the same
mask-conditional effect:

    P_help(ij)  = E_m [ 1( -U_ij * dL/dU_ij |_{U*m}  > 0 ) ]      (Adam: a probability)
    E_help(ij)  = E_m [    -U_ij * dL/dU_ij |_{U*m}      ]         (SGD / IG: an expectation)

with m ~ top-k of a ranking at k ~ log-uniform[1, N] -- upstream's `sample_k("log")`, the
schedule the fits actually used. Two choices of the ranking that defines the masks: Adam's own
final scores (the fixed-point version) and the ORACLE dL (non-circular). Both are reported, with
Spearman against each method's scores over the off-circuit live-row weights, and against the
kept set (AUC). `--n-masks` masks, one batch of `--batch` examples each.
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
from interference_toy import sample_x  # noqa: E402

from learning_to_attribute.schedules import sample_k  # noqa: E402


def auc(score, label):
    rk = score.argsort().argsort().double() + 1
    npos, nneg = label.sum(), (~label).sum()
    return float((rk[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def mask_stats(U, b, A, v, ranking, n_masks, batch, seed):
    """P_help and E_help over top-k masks of `ranking` at log-uniform k."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    order = ranking.reshape(-1).argsort(descending=True)
    n2 = U.numel()
    p_help = torch.zeros(n2)
    e_help = torch.zeros(n2)
    p_open = torch.zeros(U.shape[0])          # how often row i's gate is open, over masks
    e2 = torch.zeros(n2)
    for _ in range(n_masks):
        k = int(sample_k(n2, "log"))
        m = torch.zeros(n2)
        m[order[:k]] = 1.0
        m = m.view_as(U)
        x = sample_x(batch, g)
        y = F.relu(x @ A.T + v)
        Um = (U * m).detach().requires_grad_(True)
        z = x @ Um.T + b
        loss = (F.relu(z) - y).pow(2).sum(-1).mean()
        loss.backward()
        eff = (-U * Um.grad).reshape(-1)          # first-order value of having the weight ON
        p_help += (eff > 0).float()
        e_help += eff
        e2 += eff.pow(2)
        p_open += (z.detach() > 0).float().mean(0)
    return p_help / n_masks, e_help / n_masks, p_open / n_masks, (e2 / n_masks).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["hard", "hard30k"])
    ap.add_argument("--n-masks", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--biasmask", action="store_true",
                    help="analyse the scores of the joint scores+bias fit (biasmask.pt) instead "
                         "of plain Adam's: the mask-sampled statistics and the sparse-model "
                         "residual are then computed under the learned b + db, and the kept "
                         "set is that fit's own loss optimum under b + db")
    a = ap.parse_args()
    for tag in a.tags:
        d = IT.OUT.parent / f"interference_toy_{tag}"
        m = torch.load(d / "model.pt")
        U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
        IT.N_FEAT, IT.N_RES = U.shape[0], 16
        sc = dict(torch.load(d / "scores.pt"))
        b_orig = b.clone()
        if a.biasmask:
            bm = torch.load(d / "biasmask.pt")
            sc["adam"] = bm["scores"]
            b = b + bm["db"]
        on = (A > 0).reshape(-1)
        dead = (~(A > 0).any(1)).unsqueeze(1).expand_as(A).reshape(-1) & ~on
        live_off = ~on & ~dead
        # Adam's kept set at its loss optimum, for the AUC
        best = {}
        for grid in ("log", "linear"):
            tl = json.loads((d / f"true_loss_{grid}.json").read_text())
            for meth, curve in tl["curves"].items():
                i = min(range(len(curve)), key=curve.__getitem__)
                if meth not in best or curve[i] < best[meth][1]:
                    best[meth] = (tl["grid"][i], curve[i])
        if a.biasmask:
            c = json.loads((d / "biasmask.json").read_text())["curves"]["scores+db"]
            best["adam"] = (c["k_best"], c["L_best"])
        kept = torch.zeros(U.numel(), dtype=torch.bool)
        kept[sc["adam"].reshape(-1).argsort(descending=True)[:best["adam"][0]]] = True

        print(f"\n=== {tag}{' [bias-mask scores, features under b+db]' if a.biasmask else ''}: "
              f"{int(live_off.sum())} live-row off-circuit weights, Adam keeps "
              f"{int((kept & live_off).sum())} of them ===")
        print(f"  {'masks from':<12}{'feature':<30}{'rho Adam':>10}{'rho SGD':>10}{'rho IG':>10}"
              f"{'rho dL':>10}{'AUC kept':>10}")
        for src, ranking in (("adam", sc["adam"]), ("oracle dL", dl)):
            p_help, e_help, p_open, rms = mask_stats(U, b, A, v, ranking, a.n_masks, a.batch, seed=1)
            # the bias's role, made explicit: how often the row is open across sparse masks,
            # and the sign-probability split by it
            # E/rms is the Adam-shaped statistic: a unit's mean effect in units of its own
            # typical effect size, which is what a per-parameter-normalised optimizer integrates
            feats = (("P_help", p_help), ("E_help", e_help),
                     ("E_help/rms", e_help / rms.clamp_min(1e-12)),
                     ("P_open(row)", p_open.unsqueeze(1).expand_as(U).reshape(-1)))
            for fname, f in feats:
                print(f"  {src:<12}{fname:<30}" + "".join(
                    f"{spearman(f[live_off], t.reshape(-1)[live_off]):>10.2f}"
                    for t in (sc["adam"], sc["sgd"], sc["ixg:mc"], dl))
                    + f"{auc(f[live_off], kept[live_off]):>10.2f}")
        # the reference features from interference_adam_pattern.py, on the same population
        X = torch.cat([x for x, _ in IT.eval_batches(A, v, 65_536, 8192, 7)])
        Y = torch.cat([y for _, y in IT.eval_batches(A, v, 65_536, 8192, 7)])
        with torch.no_grad():
            zc = X @ (U * (A > 0)).T + b
            r = ((Y - F.relu(zc)) * (zc > 0)).mean(0)
        Uf = U.detach().requires_grad_(True)
        ((F.relu(X @ Uf.T + b) - Y).pow(2).sum(-1).mean()).backward()
        with torch.no_grad():
            ey = Y.mean(0)                                        # E[y_i]
            r0 = ey - F.relu(b)                                   # empty-model residual
        ab = b.abs().unsqueeze(1)
        Ur, gfull = U * r.unsqueeze(1), (-U * Uf.grad)
        with torch.no_grad():
            zo = X @ (U * (A > 0)).T + b_orig
            r_orig = ((Y - F.relu(zo)) * (zo > 0)).mean(0)
        rk = lambda t: t.reshape(-1).argsort().argsort().float()
        feats_ref = [("U*r_i", Ur.reshape(-1)),
                     ("U*r_i (original b)", (U * r_orig.unsqueeze(1)).reshape(-1)),
                     ("ixg@full", gfull.reshape(-1)), ("dL", dl.reshape(-1)),
                     ("b_i (row)", b.unsqueeze(1).expand_as(U).reshape(-1)),
                     ("U*(E[y]-relu(b))", (U * r0.unsqueeze(1)).reshape(-1)),
                     # bias MAGNITUDE variants
                     ("U*r_i/|b_i|", (Ur / ab.clamp_min(1e-3)).reshape(-1)),
                     ("U*r_i*|b_i|", (Ur * ab).reshape(-1)),
                     ("U*(-b_i)", (U * -b.unsqueeze(1)).reshape(-1)),
                     ("U/|b_i|", (U / ab.clamp_min(1e-3)).reshape(-1)),
                     ("(U*r_i)/(|b_i|+|r_i|)", (Ur / (ab + r.abs().unsqueeze(1)).clamp_min(1e-3)).reshape(-1)),
                     # sparse + dense combinations (rank space)
                     ("rank(U r)+rank(ixg@full)", rk(Ur) + rk(gfull)),
                     ("max(rank U r, rank ixg@full)", torch.maximum(rk(Ur), rk(gfull))),
                     ("rank(U r)+2 rank(ixg@full)", rk(Ur) + 2 * rk(gfull)),
                     ("rank(U r)+rank(dL)", rk(Ur) + rk(dl))]
        # ORDER WITHIN each population separately: the circuit, Adam's kept off-circuit set,
        # and the off-circuit weights it does not keep. One Spearman over all off-circuit
        # weights mixes two regimes (what is kept vs how the rest is ordered).
        pops = (("on circuit", on), ("kept off", kept & live_off), ("not-kept off", ~kept & live_off))
        print(f"\n  Spearman with ADAM within each population "
              f"({', '.join(f'{n} n={int(m_.sum())}' for n, m_ in pops)}):")
        print(f"  {'feature':<30}" + "".join(f"{n:>14}" for n, _ in pops)
              + "   | with SGD: " + "".join(f"{n:>14}" for n, _ in pops))
        for fname, f in feats_ref + [("E_help (oracle masks)", e_help), ("P_help (oracle masks)", p_help)]:
            print(f"  {fname:<30}" + "".join(
                f"{spearman(f[m_], sc['adam'].reshape(-1)[m_]):>14.2f}" for _, m_ in pops)
                + "   |           " + "".join(
                f"{spearman(f[m_], sc['sgd'].reshape(-1)[m_]):>14.2f}" for _, m_ in pops))
        print()
        for fname, f in feats_ref:
            print(f"  {'reference':<12}{fname:<30}" + "".join(
                f"{spearman(f[live_off], t.reshape(-1)[live_off]):>10.2f}"
                for t in (sc["adam"], sc["sgd"], sc["ixg:mc"], dl))
                + f"{auc(f[live_off], kept[live_off]):>10.2f}")


if __name__ == "__main__":
    main()
