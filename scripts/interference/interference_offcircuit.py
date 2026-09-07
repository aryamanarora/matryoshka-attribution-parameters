"""WHY a MAttr+Adam mask beats the model it was fitted to, on the interference toy. ANSWERED.

    uv run python scripts/interference/interference_offcircuit.py --tag hard          # ~4 min, CPU
    uv run python scripts/interference/interference_offcircuit.py --tag hard --seeds 1 2   # + replication

THE FACT TO EXPLAIN. On `--tag hard`, at its own loss-optimal sparsity the MAttr+Adam ranking
gives a filtered model with a LOWER true loss than the trained model itself, and than the true
circuit `A`:

    L(full U) 3.917    L(circuit A only) 3.663    L(Adam top-1400) 3.265

and 1254 of those 1400 kept weights are OFF the circuit, i.e. interference. The oracle `dL`
calls 99% of them worthless (dL <= 1e-4). So this is not the ranking finding hidden circuitry;
it is the mask doing something a per-weight attribution cannot express. This script measures
what, in five steps, each of which is a falsifiable claim printed with its number.

(1) CAPACITY. The trained model is `U = W_up W_down` with `W_down` frozen at n_res=16, so
    every U it can represent has rank <= 16 and 2176 free numbers. A top-k mask over U has
    16384 binary degrees of freedom and `U * mask` is rank 73 at the Adam optimum. Filtering
    is not a subset-selection inside the model's hypothesis class -- it LEAVES that class.
    Nothing forbids the exit from being an improvement, so "better than the original model" is
    not paradoxical and needs no circuit story.

(2) WHAT THE EXTRA DEGREES OF FREEDOM BUY: a per-row bias correction. `b` is frozen at the
    value co-adapted to the FULL interference, and under a frozen random down-projection the
    fit to A is heavily shrunk (median U/A = 0.11), so the circuit-only model systematically
    under-predicts: mean residual -0.073 on gate-open examples. The correction it wants is a
    per-row constant, and `corr(-residual, optimal bias shift) = +0.97`. The mask cannot touch
    `b` -- but a set of off-circuit weights supplies `M_i = E[x] * sum_j U_ij` on row i, which
    IS a constant offset plus noise. Adam's kept set supplies M with the right sign on the
    right rows (`corr(M, needed shift) = +0.67`), and REPLACING those 1254 weights by their
    mean effect alone recovers more than all of the gain (3.12 vs 3.26). They are a noisy
    weight-space implementation of a bias the mask has no other way to write down.

(3) WHICH off-circuit weights are worth selecting -- the pattern, and it is not the note's:
      - SIGN, aligned with the row's residual. 82% of Adam's kept off-circuit weights are
        positive against 47% of the population, and the rows that gain are the under-predicting
        ones. Sign carries the effect; this is why ranking by |U| (`--abs`) cannot find them.
      - MAGNITUDE, small. For a target offset M on a row, using n weights of size u costs
        variance ~ M*u*Var(x)/E[x] -- proportional to u at fixed offset. So many small aligned
        weights strictly beat few large ones, and the script measures it on one row: same
        offset from 36 small weights vs 8 large ones is 0.041 vs 0.074 row loss.
      - NOT LOCALITY. In-block enrichment is 10.1% against an 11.4% base rate: the useful
        interference is not the circuit's own block leaking, it is a diffuse population.
      - AND A SET-LEVEL BUDGET, which is what no per-weight score has. Keeping every
        individually-helpful off-circuit weight (all 3685 whose own addition lowers the loss)
        gives L = 7.08, WORSE than keeping none. The offsets add, so a set that is right one
        weight at a time overshoots by an order of magnitude as a group.

(4) WHY THE ORACLE MISSES THEM, structurally. `dL` ablates one weight with all others present.
    At the full model every row's offset is already balanced by the whole interference
    population, so no single member of it is worth anything -- median dL over Adam's kept
    off-circuit set is -3.5e-06. Their value exists ONLY in the sparse model, i.e. it is
    conditional on the rest of the set being gone. A one-at-a-time ground truth cannot see a
    coalition, and neither can stepless IG, whose path scales all weights together.
    Consequence for reading the figures: MAttr+Adam has the WORST tail precision against `dL`
    of the three methods (P@R.8 = 0.05) and the BEST true loss (3.265 vs 3.589 / 3.629). On
    this task those two axes are not merely noisy versions of each other, they disagree in
    ORDER, and the true-loss axis is the one with a model behind it.

(5) TWO REDUCTIONS OF ADAM'S ADVANTAGE THAT FAIL, recorded so they are not re-attempted:
      - "Adam ranks by consistency (mean/rms) where SGD ranks by summed gradient, and
        consistency is scale-free so it surfaces small aligned weights." On ONE shared
        gradient stream, ranking by mean/rms picks LARGER off-circuit weights than ranking by
        the sum (median |U| 0.036 vs 0.031) -- the wrong sign. The magnitude tilt is a property
        of Adam's TRAJECTORY, not of a reweighting of a fixed signal.
      - "It is greedy value-per-variance." The first-order greedy that keeps the 1400 best
        `value/variance` off-circuit weights reaches only 3.538, well short of Adam's 3.265,
        because it has no way to stop adding once a row's offset is met.
    So what Adam is doing here is joint set optimisation under a k-schedule, and this script
    does NOT reduce it to a per-weight score. That is the open part.

The gain is not sample fitting: every loss here is on eval batches drawn from a generator the
mask never saw, and re-evaluating on a second eval seed moves it by 0.009. It also replicates:
`--seeds 1 2` puts Adam's optimum at k = 1384 / 1360 for true loss 3.281 / 3.289 against seed 0's
3.265, with 1238 / 1213 off-circuit weights at 82% / 84% positive and median |U| 0.023, while SGD
lands at k = 160 / 152 for 3.578 / 3.576 with 44-52 off-circuit weights, 100% positive at median
|U| 0.09. The COMPOSITION replicates, not merely the headline, and the two optimizers pursue
visibly different strategies rather than noisy versions of one.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interference_toy as IT  # noqa: E402
import interference_attrib as IA  # noqa: E402
from interference_toy import sample_x  # noqa: E402

from learning_to_attribute import learn_scores  # noqa: E402

E_X, VAR_X = 0.15, 0.0775          # feature density 0.3, active value ~ U[0,1]
N_EVAL, CHUNK = 65_536, 8192


def fit_only(U, b, A, v, dl, steps):
    """Fit MAttr here and report each ranking's own loss optimum -- for a tag with nothing on
    disk. Used to check that the effect is not an artifact of the frozen-projection variant."""
    n = U.shape[0]
    sup = A > 0
    X = torch.cat([x for x, _ in IT.eval_batches(A, v, N_EVAL, CHUNK, 0)])
    Y = torch.cat([y for _, y in IT.eval_batches(A, v, N_EVAL, CHUNK, 0)])
    nex = X.shape[0]

    def L(mask, bias=b):
        with torch.no_grad():
            return float((F.relu(X @ (U * mask.view_as(U)).T + bias) - Y).pow(2).sum() / nex)
    print(f"L(full U) {L(torch.ones_like(U)):.4f}   L(circuit A) {L(sup.float()):.4f}   "
          f"rank(U) {torch.linalg.matrix_rank(U).item()}   "
          f"real weights {int((dl > 1e-4).sum())}")
    ranks = {}
    for optn, lr in (("adam", 0.05), ("sgd", 1.0)):
        g = torch.Generator().manual_seed(40_000)

        def loss_fn(mask):
            return IA.loss_on(U * mask.view(n, n), b, A, v, sample_x(2048, g))
        ranks[optn] = learn_scores(n * n, loss_fn, steps=steps, variant="topk", k_schedule="log",
                                   T=0.5, lr=lr, optimizer=optn).scores.detach().reshape(-1)
    ranks["dL"] = dl.reshape(-1)
    for nm, sv in ranks.items():
        o = sv.argsort(descending=True)
        best = None
        for k in range(8, 4001, 8):
            mk = torch.zeros(n * n)
            mk[o[:k]] = 1.
            Lk = L(mk)
            if best is None or Lk < best[1]:
                best = (k, Lk)
        kk = torch.zeros(n * n, dtype=torch.bool)
        kk[o[:best[0]]] = True
        kk = kk.view(n, n)
        off = kk & ~sup
        print(f"  {nm:5s} best k={best[0]:5d} L {best[1]:.4f} | on-circuit "
              f"{int((kk & sup).sum())}/{int(sup.sum())} off {int(off.sum())} "
              f"positive {(U[off] > 0).float().mean() if off.any() else float('nan'):.2f} "
              f"median|U| {U[off].abs().median() if off.any() else float('nan'):.4f}")

    def refit(mask, steps_=800, lr=0.05):
        z0 = X @ (U * mask).T
        bb = b.clone().requires_grad_(True)
        o = torch.optim.Adam([bb], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps_)
        for _ in range(steps_):
            o.zero_grad()
            ((F.relu(z0 + bb) - Y).pow(2).sum() / nex).backward()
            o.step()
            sch.step()
        with torch.no_grad():
            return float((F.relu(z0 + bb) - Y).pow(2).sum() / nex)
    print(f"  a free bias is worth: {L(torch.ones_like(U)) - refit(torch.ones_like(U)):.4f} to the "
          f"full model, {L(sup.float()) - refit(sup.float()):.4f} to the circuit-only one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--seeds", type=int, nargs="*", default=[],
                    help="extra MAttr seeds to refit and re-measure the optimum on")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--fit-only", action="store_true",
                    help="fit MAttr from scratch on this tag's model and report each ranking's "
                         "loss-optimal set + what a free bias is worth. The only mode that runs "
                         "on a tag with no scores.pt / minloss_sets.pt on disk, which is how the "
                         "note's literal config (--tag lit) was checked.")
    a = ap.parse_args()
    d = IT.OUT.parent / f"{IT.OUT.name}_{a.tag}"
    m = torch.load(d / "model.pt")
    U, b, A, v, dl = m["U"], m["b"], m["A"], m["v"], m["dl"]
    IT.N_FEAT, IT.N_RES = U.shape[0], 16
    n = U.shape[0]
    if a.fit_only:
        fit_only(U, b, A, v, dl, a.steps)
        return
    scores = torch.load(d / "scores.pt")
    sets = torch.load(d / "minloss_sets.pt")["sets"]

    sup, = (A > 0),
    live = sup.any(1)
    off_all = (~sup) & live.unsqueeze(1)
    blk = torch.zeros(n, n, dtype=torch.bool)
    bs = n // IT.N_BLOCKS
    for i in range(IT.N_BLOCKS):
        s = slice(i * bs, (i + 1) * bs)
        blk[s, s] = True

    batches = list(IT.eval_batches(A, v, N_EVAL, CHUNK, 0))
    nex = sum(x.shape[0] for x, _ in batches)

    def loss(mask, bias=b):
        Um = U * mask.view_as(U)
        return sum((F.relu(x @ Um.T + bias) - y).pow(2).sum().item() for x, y in batches) / nex

    kept = torch.zeros(n * n, dtype=torch.bool)
    kept[sets["adam"]["idx"]] = True
    kept = kept.view(n, n)
    off = kept & ~sup
    L_full, L_circ, L_adam = loss(torch.ones_like(U)), loss(sup.float()), loss(kept.float())

    print("=" * 78)
    print("(0) THE FACT")
    print(f"  L(full U) {L_full:.4f}   L(circuit A) {L_circ:.4f}   "
          f"L(Adam top-{sets['adam']['k']}) {L_adam:.4f}")
    print(f"  Adam's set: {int((kept & sup).sum())}/{int(sup.sum())} on circuit, "
          f"{int(off.sum())} off | of those, {(dl[off] > 1e-4).float().mean():.1%} have dL > 1e-4"
          f" (median dL {dl[off].median():.2e})")
    print(f"  held-out check, second eval seed: ", end="")
    b2 = list(IT.eval_batches(A, v, N_EVAL, CHUNK, 7))
    n2 = sum(x.shape[0] for x, _ in b2)

    def loss7(mask):
        Um = U * mask.view_as(U)
        return sum((F.relu(x @ Um.T + b) - y).pow(2).sum().item() for x, y in b2) / n2
    print(f"L(full) {loss7(torch.ones_like(U)):.4f}  L(Adam) {loss7(kept.float()):.4f}")

    print("\n(1) CAPACITY -- the mask leaves the model's hypothesis class")
    print(f"  rank(U) {torch.linalg.matrix_rank(U).item()}   "
          f"rank(U*adam mask) {torch.linalg.matrix_rank(U * kept.float()).item()}   "
          f"rank(U*circuit) {torch.linalg.matrix_rank(U * sup.float()).item()}")
    print(f"  model free numbers {n * IT.N_RES + n}   mask degrees of freedom {n * n}")

    print("\n(2) WHAT THEY BUY -- a per-row bias correction the mask cannot write directly")
    rs, rn = torch.zeros(n), torch.zeros(n)
    for x, y in batches:
        z = x @ (U * sup.float()).T + b
        g = (z > 0).float()
        rs += ((F.relu(z) - y) * g).sum(0)
        rn += g.sum(0)
    resid = rs / rn.clamp_min(1)
    # Refit b with everything else frozen. Pre-activations are computed ONCE per mask and the
    # bias only shifts them, so this is 800 cheap steps rather than 800 forward passes.
    X = torch.cat([x for x, _ in batches])
    Y = torch.cat([y for _, y in batches])

    def refit_bias(mask, steps=800, lr=0.05):
        z0 = X @ (U * mask).T
        bb = b.clone().requires_grad_(True)
        o = torch.optim.Adam([bb], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)
        for _ in range(steps):
            o.zero_grad()
            ((F.relu(z0 + bb) - Y).pow(2).sum() / nex).backward()
            o.step()
            sch.step()
        with torch.no_grad():
            return bb.detach(), float((F.relu(z0 + bb) - Y).pow(2).sum() / nex)
    bopt, L_circ_bopt = refit_bias(sup.float())
    db = (bopt - b)
    M = E_X * (off.float() * U).sum(1)

    def corr(p, q):
        return float(torch.corrcoef(torch.stack([p[live], q[live]]))[0, 1])
    print(f"  circuit-only model under-predicts: mean residual {resid[live].mean():+.4f} "
          f"(gate-open), median shrinkage U/A "
          f"{torch.stack([(U[i][sup[i]] / A[i][sup[i]]).median() for i in live.nonzero().squeeze().tolist()]).median():.3f}")
    print(f"  corr(-residual, optimal bias shift) {corr(-resid, db):+.3f}   "
          f"corr(offset Adam supplies, optimal shift) {corr(M, db):+.3f}")
    L_offset_only = loss(sup.float(), b + M)
    print(f"  L(circuit) {L_circ:.4f} -> {L_offset_only:.4f} with the kept weights REPLACED by "
          f"their mean offset alone -- implementing that offset in weights instead costs "
          f"{L_adam - L_offset_only:+.4f} in injected variance")
    print("  what a FREE BIAS is worth to each model -- the decisive line:")
    for lab, mk in (("full U", torch.ones_like(U)), ("circuit only", sup.float()),
                    ("Adam top-1400", kept.float())):
        _, Lb = refit_bias(mk)
        print(f"    {lab:14s} frozen b {loss(mk):.4f} -> re-fitted b {Lb:.4f}  "
              f"(bias buys {loss(mk) - Lb:.4f})")
    print("    ^ half a loss unit to the circuit-only model, ~nothing to Adam's: its off-circuit")
    print("      weights have already done what a re-fitted bias would do. Note the circuit-only")
    print(f"      refit STICKS at {L_circ_bopt:.4f}, worse than the {loss(sup.float(), b + M):.4f} "
          f"Adam's own offset reaches --")
    print("      a row whose gate is shut on every example has no gradient to escape on, so that")
    print("      number is a floor on what a bias is worth, not the optimum.")

    print("\n(3) THE SELECTION PATTERN")
    print(f"  sign:      kept off-circuit {(U[off] > 0).float().mean():.1%} positive vs "
          f"{(U[off_all] > 0).float().mean():.1%} of the population")
    print(f"  magnitude: kept median |U| {U[off].abs().median():.4f} vs population "
          f"{U[off_all].abs().median():.4f}")
    print(f"  locality:  kept in-block {(off & blk).float().sum() / off.sum():.1%} vs base "
          f"{(off_all & blk).float().sum() / off_all.sum():.1%}  (no enrichment)")
    row = int(M.argmax())
    posj = ((~sup[row]) & (U[row] > 0)).nonzero().squeeze()
    vals = U[row][posj]

    def row_loss(idx):
        mk = sup[row].float().clone()
        mk[idx] = 1.
        return sum((F.relu(x @ (U[row] * mk) + b[row]) - y[:, row]).pow(2).sum().item()
                   for x, y in batches) / nex
    print(f"  same offset, few big vs many small (row {row}, circuit-only "
          f"{row_loss(torch.tensor([], dtype=torch.long)):.5f}):")
    for lab, order in (("largest first ", vals.argsort(descending=True)),
                       ("smallest first", vals.argsort())):
        cum, sel = 0., []
        for j in order.tolist():
            if cum >= float(M[row]):
                break
            sel.append(int(posj[j]))
            cum += E_X * float(vals[j])
        print(f"    {lab}  n={len(sel):3d}  offset {cum:.4f}  row loss "
              f"{row_loss(torch.tensor(sel)):.5f}")

    print("\n(4) WHY A PER-WEIGHT SCORE CANNOT GET THERE")
    lin, quad = torch.zeros(n, n), torch.zeros(n, n)
    for x, y in batches:
        z = x @ (U * sup.float()).T + b
        g = (z > 0).float()
        lin += (2 * ((F.relu(z) - y) * g)).T @ x
        quad += g.T @ (x * x)
    lin, quad = lin / nex, quad / nex
    pred = lin * U + quad * U ** 2                    # dLoss from adding this weight alone
    helpful = (pred < 0) & off_all
    mk = sup.float().clone()
    mk[helpful] = 1.
    print(f"  keeping ALL {int(helpful.sum())} individually-helpful off-circuit weights: "
          f"L {loss(mk):.4f}  (vs {L_circ:.4f} keeping none)")
    ratio = torch.where(helpful, -pred / (U ** 2 * VAR_X).clamp_min(1e-12),
                        torch.full_like(pred, -1e9))
    for k in (400, 1400, 2000):
        o = ratio.reshape(-1).argsort(descending=True)[:k]
        mk = sup.float().clone().reshape(-1)
        mk[o] = 1.
        print(f"  greedy value/variance, {k:5d} off-circuit weights: L {loss(mk.view(n, n)):.4f}")
    print(f"  Adam                                            : L {L_adam:.4f}")
    print("  per-method loss-optimal sets and what they are made of:")
    for nm, s in sets.items():
        kk = torch.zeros(n * n, dtype=torch.bool)
        kk[s["idx"]] = True
        kk = kk.view(n, n)
        o = kk & ~sup
        print(f"    {nm:7s} k={s['k']:5d} L {s['loss']:.4f} | off {int(o.sum()):5d} "
              f"median|U| {U[o].abs().median() if o.any() else float('nan'):.4f} "
              f"positive {(U[o] > 0).float().mean() if o.any() else float('nan'):.2f}")
    print("  the same rankings all forced to k=1400:")
    for nm, s in list(scores.items()) + [("dL", dl)]:
        o = s.reshape(-1).argsort(descending=True)
        mk = torch.zeros(n * n)
        mk[o[:1400]] = 1.
        print(f"    {nm:7s} L {loss(mk):.4f}")

    if a.seeds:
        print("\n(5) REPLICATION -- refit MAttr at extra seeds, report each one's own optimum")
        grid = list(range(8, 3001, 8))
        for sd in a.seeds:
            for optn, lr in (("adam", 0.05), ("sgd", 1.0)):
                g = torch.Generator().manual_seed(sd + 40_000)

                def loss_fn(mask):
                    return IA.loss_on(U * mask.view(n, n), b, A, v, sample_x(2048, g))
                res = learn_scores(n * n, loss_fn, steps=a.steps, variant="topk",
                                   k_schedule="log", T=0.5, lr=lr, optimizer=optn)
                order = res.scores.detach().reshape(-1).argsort(descending=True)
                best = None
                for k in grid:
                    mk = torch.zeros(n * n)
                    mk[order[:k]] = 1.
                    Lk = loss(mk)
                    if best is None or Lk < best[1]:
                        best = (k, Lk)
                kk = torch.zeros(n * n, dtype=torch.bool)
                kk[order[:best[0]]] = True
                kk = kk.view(n, n)
                o = kk & ~sup
                print(f"  seed {sd} {optn:4s}: best k={best[0]:5d} L {best[1]:.4f} | "
                      f"off {int(o.sum()):5d} positive {(U[o] > 0).float().mean():.2f} "
                      f"median|U| {U[o].abs().median():.4f}", flush=True)


if __name__ == "__main__":
    main()
