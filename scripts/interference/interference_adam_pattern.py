"""How does MAttr+Adam rank the INTERFERENCE weights? Candidate per-weight predictors, scored.

    uv run python scripts/interference/interference_adam_pattern.py --tag hard --n-feat 32 128

Reads the cells written by scripts/interference/interference_scale.py and, for the off-circuit weights
(`A_ij == 0`) of each model, rank-correlates Adam's final score with a list of simple candidate
features. The list is the set of readings a person might offer for "Adam keeps small,
sign-aligned, diffuse interference weights", each turned into a per-weight number:

    U, |U|            the weight itself, either sign
    sign(U) * r_i     the target row's mean under-prediction in the CIRCUIT-ONLY model (r_i =
                      E[(y_i - y'_i) 1(gate open)]), signed by the weight: "does this weight push
                      row i the way the sparse model needs pushing" -- the coalition story
    U * r_i           the same with magnitude
    ixg @ zero        -U_ij * dL/dU_ij at U = 0            (the alpha=0 endpoint)
    ixg @ circuit     -U_ij * dL/dU_ij at U * 1[A > 0]     (adding the weight to the circuit)
    ixg @ full        -U_ij * dL/dU_ij at U                (the alpha=1 endpoint)
    stepless IG, SGD, dL                                   (the other rankings)
    p_i, p_i^circ     target firing rate in the full / circuit-only model (a row property)
    E[x_j]-weighted   sign(U) * r_i * E[x_j] (constant E[x_j] here, so equals sign(U) * r_i)

and then the two-term combination the coalition story predicts, `U * r_i - lam * U^2`, with
`lam` fitted by a coarse scan. Everything is Spearman over the off-circuit weights; the same
features are also scored against SGD's and IG's off-circuit scores so "explains Adam" can be
read against "explains everyone".

THE ANSWER, from running it (docs/interference_toy.md has the tables). Two regimes:
  - DEAD rows (A_i = 0, ~20% of rows): every method ranks by -U, "keep the gate shut", at
    rho ~0.9. Adam differs only in placing the whole population above the live-row bulk.
  - LIVE rows: what Adam KEEPS (its loss-optimal set) is `U_ij * r_i` -- AUC 0.81-0.85 on
    `hard`, 0.92-0.98 on `lit`, at every size -- and the oracle `dL` / IxG@full are at chance
    for that set. The whole-population Spearman is lower on `hard` (0.24-0.45) because the
    ORDER AT THE BOTTOM (which weights hit the floor first) tracks the full-model ablation
    effect instead; a rank mixture a*U*r_i + (1-a)*IxG@full fits Adam's live-row ranking at
    0.65-0.85 with `a` falling from 0.47 (2^10) to 0.25 (2^14) on `hard`, and ~0.9 on `lit`.
So the rule for the kept set is one feature, `U * r_i` = IxG evaluated at the circuit-only
model, and it is the per-weight form of the bias-offset coalition. The `--kept` block prints
it; `plots/plot_interference_adam_pattern.py` draws it.
"""

import argparse
import sys
from pathlib import Path

import json

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_scale as IS  # noqa: E402
import interference_toy as IT  # noqa: E402

N_EVAL, CHUNK = 65_536, 8192


def grad_at(U, b, A, v, mask, X, Y):
    """dL/dU at U * mask, one batch of eval examples. Returns the gradient wrt the FULL U (so
    the entry for a masked-out weight is 'what happens if it is switched on')."""
    Um = (U * mask).detach().requires_grad_(True)
    loss = (F.relu(X @ Um.T + b) - Y).pow(2).sum(-1).mean()
    loss.backward()
    return Um.grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--n-feat", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--population", choices=("offcircuit", "noise"), default="offcircuit",
                    help="offcircuit = A_ij == 0; noise = dL <= eps")
    a = ap.parse_args()
    d = IS.OUT / a.tag
    for n in a.n_feat:
        IT.N_FEAT, IT.N_RES, IT.N_BLOCKS = n, max(1, n // 8), max(1, n // IS.BLOCK_SIZE)
        rows = {}
        for seed in a.seeds:
            blob = torch.load(d / f"n{n}_s{seed}.pt")
            U, b, A, v, dl = blob["U"], blob["b"], blob["A"], blob["v"], blob["dl"]
            sc = blob["scores"]
            circuit = A > 0
            pop = (~circuit) if a.population == "offcircuit" else (dl <= IS.EPS_REAL)
            X = torch.cat([x for x, _ in IT.eval_batches(A, v, N_EVAL, CHUNK, seed + 7)])
            Y = torch.cat([y for _, y in IT.eval_batches(A, v, N_EVAL, CHUNK, seed + 7)])
            with torch.no_grad():
                yc = F.relu(X @ (U * circuit).T + b)
                yf = F.relu(X @ U.T + b)
                open_c = (X @ (U * circuit).T + b) > 0
                r = ((Y - yc) * open_c).mean(0)                    # row under-prediction, sparse
                p_full, p_circ = (yf > 0).float().mean(0), open_c.float().mean(0)
                ex = X.mean(0)
            g0 = grad_at(U, b, A, v, torch.zeros_like(U), X, Y)
            gc = grad_at(U, b, A, v, circuit.float(), X, Y)
            gf = grad_at(U, b, A, v, torch.ones_like(U), X, Y)
            feats = {
                "U (signed)": U, "|U|": U.abs(), "-|U|": -U.abs(),
                "sign(U)·r_i": U.sign() * r.unsqueeze(1),
                "U·r_i": U * r.unsqueeze(1),
                "U·r_i·E[x_j]": U * r.unsqueeze(1) * ex.unsqueeze(0),
                "ixg @ zero": -U * g0, "ixg @ circuit": -U * gc, "ixg @ full": -U * gf,
                "stepless IG": sc[f"ixg:mc|0|{a.budget}"], "SGD": sc[f"sgd|0|{a.budget}"],
                "dL (oracle)": dl,
                "p_i (full)": p_full.unsqueeze(1).expand_as(U),
                "p_i (circuit)": p_circ.unsqueeze(1).expand_as(U),
                "r_i alone": r.unsqueeze(1).expand_as(U),
            }
            # IxG at SPARSE models nearer the ones the log-uniform-k fit actually visits: the
            # oracle's real set, and top-k sets of the same size as Adam's loss optimum drawn
            # from the oracle / IG rankings (not from Adam's own -- that would be circular)
            # and from Adam's own (the fixed-point check, labelled as such).
            real = dl > IS.EPS_REAL
            kb = json.loads((d / f"n{n}_s{seed}.json").read_text())["true_loss"]["curves"]["adam"]["k_best"]

            def topk_mask(score, k):
                m = torch.zeros(U.numel())
                m[score.reshape(-1).argsort(descending=True)[:k]] = 1.
                return m.view_as(U)
            g_real = grad_at(U, b, A, v, real.float(), X, Y)
            g_dlk = grad_at(U, b, A, v, topk_mask(dl, kb), X, Y)
            g_igk = grad_at(U, b, A, v, topk_mask(sc[f"ixg:mc|0|{a.budget}"], kb), X, Y)
            g_adk = grad_at(U, b, A, v, topk_mask(sc[f"adam|0|{a.budget}"], kb), X, Y)
            # expected IxG under the fit's own k distribution over the ORACLE ranking
            g_mix = torch.zeros_like(U)
            ks = torch.logspace(0, torch.log10(torch.tensor(float(U.numel()))), 12).round().int()
            for k in ks:
                g_mix += grad_at(U, b, A, v, topk_mask(dl, int(k)), X, Y) / len(ks)
            feats.update({
                "ixg @ real set": -U * g_real,
                "ixg @ dL top-k_best": -U * g_dlk,
                "ixg @ IG top-k_best": -U * g_igk,
                "ixg @ Adam top-k_best (self)": -U * g_adk,
                "E_k ixg @ dL top-k": -U * g_mix,
            })
            # Is it a ROW pattern? Leave-one-out group mean of Adam's score by (row, sign U)
            # and by (col, sign U): the best any 'row property x sign' story could do.
            s_ad = sc[f"adam|0|{a.budget}"]
            for gname, gid in (("row×sign(U) group mean", torch.arange(n).unsqueeze(1).expand_as(U) * 2 + (U > 0).long()),
                               ("col×sign(U) group mean", torch.arange(n).unsqueeze(0).expand_as(U) * 2 + (U > 0).long()),
                               ("row group mean", torch.arange(n).unsqueeze(1).expand_as(U))):
                gm = torch.zeros_like(U)
                ids = gid[pop]
                vals = s_ad[pop]
                sums = torch.zeros(int(gid.max()) + 1).index_add_(0, ids, vals)
                cnts = torch.zeros(int(gid.max()) + 1).index_add_(0, ids, torch.ones_like(vals))
                loo = (sums[ids] - vals) / (cnts[ids] - 1).clamp_min(1)
                gm[pop] = loo
                feats[gname] = gm
            # TWO REGIMES. A row of A that is all zero is a DEAD target (y_i = 0 always, the
            # gate should stay shut), and on the frozen projection ~20% of rows are dead. For a
            # dead row the sparse-model marginal value of a weight is "does it keep the gate
            # shut", i.e. -U; for a live row it is the under-prediction story, U * r_i. The
            # per-regime numbers say which rule holds where, and the combined feature ranks
            # dead-row weights by -U * p_i^circ-open-cost and live-row weights by U * r_i, each
            # within its regime, with the regimes' offsets fitted by a scan.
            live = circuit.any(1)
            live_w, dead_w = pop & live.unsqueeze(1), pop & ~live.unsqueeze(1)

            def ranknorm(x):
                return x.argsort().argsort().float() / max(1, x.numel() - 1)
            reg = {}
            for tname, t in (("Adam", sc[f"adam|0|{a.budget}"]), ("SGD", sc[f"sgd|0|{a.budget}"]),
                             ("IG", sc[f"ixg:mc|0|{a.budget}"])):
                reg[tname] = {
                    "live: U·r_i": IS.spearman(t[live_w], (U * r.unsqueeze(1))[live_w]),
                    "live: ixg@full": IS.spearman(t[live_w], (-U * gf)[live_w]),
                    "live: dL": IS.spearman(t[live_w], dl[live_w]),
                    "live: sign(U)": IS.spearman(t[live_w], U.sign()[live_w]),
                    "dead: -U": IS.spearman(t[dead_w], -U[dead_w]) if dead_w.any() else float("nan"),
                    "dead: ixg@full": IS.spearman(t[dead_w], (-U * gf)[dead_w]) if dead_w.any() else float("nan"),
                    "dead: dL": IS.spearman(t[dead_w], dl[dead_w]) if dead_w.any() else float("nan"),
                    "dead-row mean − live-row mean (score units)": float(t[dead_w].mean() - t[live_w].mean()) if dead_w.any() else float("nan"),
                }
            for tname in reg:
                for k, val in reg[tname].items():
                    rows.setdefault((f"[{k}]", tname), []).append(val)
            # combined rule: rank-normalise U*r_i within live rows and -U within dead rows,
            # then add a scanned offset for the dead regime
            comb_best = None
            for off_ in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
                f = torch.zeros_like(U)
                f[live_w] = ranknorm((U * r.unsqueeze(1))[live_w])
                if dead_w.any():
                    f[dead_w] = ranknorm(-U[dead_w]) + off_
                rho = IS.spearman(sc[f"adam|0|{a.budget}"][pop], f[pop])
                if comb_best is None or rho > comb_best[1]:
                    comb_best = (off_, rho, f)
            feats[f"two-regime rule (dead offset {comb_best[0]:+g})"] = comb_best[2]
            print(f"  n{n} s{seed}: live rows {int(live.sum())}/{n}, dead-row weights "
                  f"{int(dead_w.sum())} of {int(pop.sum())} off-circuit")
            # WHAT ADAM KEEPS: AUC of each simple feature for membership in Adam's loss-optimal
            # top-k_best set, among the off-circuit weights -- the question "which interference
            # weights does it keep" rather than "how does it order all of them". And the live-row
            # rank mixture a*rank(U r_i) + (1-a)*rank(IxG@full), a scanned.
            kept = torch.zeros(U.numel(), dtype=torch.bool)
            kept[sc[f"adam|0|{a.budget}"].reshape(-1).argsort(descending=True)[:kb]] = True
            kept = kept.view_as(U)
            lab = kept[pop]

            def auc(score, label):
                rk = score.argsort().argsort().double() + 1
                npos, nneg = label.sum(), (~label).sum()
                return float((rk[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))
            for fname in ("U (signed)", "U·r_i", "ixg @ full", "dL (oracle)", "-|U|",
                          "stepless IG", "SGD", "r_i alone", "p_i (full)"):
                rows.setdefault((f"AUC kept: {fname}", "Adam"), []).append(auc(feats[fname][pop], lab))
            rows.setdefault(("AUC kept: dead row", "Adam"), []).append(
                auc((~live).float().unsqueeze(1).expand_as(U)[pop], lab))
            rows.setdefault(("kept frac of off-circuit", "Adam"), []).append(float(lab.float().mean()))
            r1 = ranknorm((U * r.unsqueeze(1))[live_w])
            r2 = ranknorm((-U * gf)[live_w])
            ra = sc[f"adam|0|{a.budget}"][live_w]
            mix = max(((a_, IS.spearman(ra, a_ * r1 + (1 - a_) * r2))
                       for a_ in torch.linspace(0, 1, 21).tolist()), key=lambda t: t[1])
            rows.setdefault(("live mixture: a(U·r_i)", "Adam"), []).append(mix[0])
            rows.setdefault(("live mixture: rho", "Adam"), []).append(mix[1])
            # the coalition story's two-term score, lam scanned
            best = None
            for lam in (0.0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
                f = U * r.unsqueeze(1) - lam * U.pow(2)
                rho = IS.spearman(sc[f"adam|0|{a.budget}"][pop], f[pop])
                if best is None or rho > best[1]:
                    best = (lam, rho)
            feats[f"U·r_i − λU² (λ={best[0]:g})"] = U * r.unsqueeze(1) - best[0] * U.pow(2)
            targets = {"Adam": sc[f"adam|0|{a.budget}"], "Adam lr.002": sc[f"adam@0.002|0|{a.budget}"],
                       "SGD": sc[f"sgd|0|{a.budget}"], "IG": sc[f"ixg:mc|0|{a.budget}"]}
            for fname, f in feats.items():
                for tname, t in targets.items():
                    rows.setdefault((fname, tname), []).append(IS.spearman(t[pop], f[pop]))
            # what the floor is: the bottom 25% of Adam's off-circuit scores, by sign of U
            s_ad = sc[f"adam|0|{a.budget}"][pop]
            floor = s_ad <= s_ad.quantile(0.25)
            top = s_ad >= s_ad.quantile(0.9)
            print(f"n{n} seed {seed}: {int(pop.sum())} {a.population} weights | U>0 among Adam's "
                  f"bottom-25% {float((U[pop][floor] > 0).float().mean()):.2f}, among top-10% "
                  f"{float((U[pop][top] > 0).float().mean()):.2f} (population "
                  f"{float((U[pop] > 0).float().mean()):.2f}) | r_i>0 rows among top-10% "
                  f"{float((r.unsqueeze(1).expand_as(U)[pop][top] > 0).float().mean()):.2f}")
        print(f"\n=== n_feat {n}: Spearman of each candidate with each method's {a.population} "
              f"scores, mean±sd over {len(a.seeds)} seeds ===")
        print(f"  {'feature':<26} " + " ".join(f"{t:>12}" for t in ("Adam", "Adam lr.002", "SGD", "IG")))
        seen = set()
        for fname in [f for f in list(feats) + [k for k, _ in rows if k.startswith(("[", "AUC", "kept", "live mix"))]
                      if not (f in seen or seen.add(f))]:
            for tn in ("Adam lr.002", "SGD", "IG"):
                if (fname, tn) not in rows:
                    rows[(fname, tn)] = [float("nan")] * len(a.seeds)
            cells = []
            for tname in ("Adam", "Adam lr.002", "SGD", "IG"):
                v = torch.tensor(rows[(fname, tname)])
                cells.append(f"{v.mean():+.2f}±{v.std() if len(v) > 1 else 0:.2f}")
            print(f"  {fname:<26} " + " ".join(f"{c:>12}" for c in cells))
        print()
        out = IS.OUT / a.tag / f"adam_pattern_n{n}.json"
        out.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in rows.items()}))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
