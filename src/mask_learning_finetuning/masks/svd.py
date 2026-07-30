"""Masking the SINGULAR VALUES of a weight delta.

Every other unit mode in ``masks.layout`` cuts a parameter tensor along one of its own axes: a
row, a column, a scalar. This one changes the *basis* first. The finetune's delta for a 2-D
parameter is factorised

    delta = U diag(S) Vh          (U [m, r], S [r], Vh [r, n], S descending)

and one score is learned per singular direction, so the composed weights are

    theta_eff = theta_base + U diag(m . S) Vh

with ``m`` the differentiable top-k mask over those scores. ``k`` then counts *directions of the
update*, not neurons, and the question the sparsity curve answers changes with it: not "which
neurons carry this finetune" but "how few directions of this update does the behaviour need".

Three things about the unit space that follow, and each has bitten a design decision elsewhere:

1. **A direction is not localised in the model.** Keeping unit ``i`` of ``gate_proj`` writes a
   rank-1 update across *every* row of that tensor, so an svd mask at 1% sparsity still touches
   nearly all the parameters an unmasked finetune touched. It is sparse in rank, not in weights.
   Do not read an svd curve as the same claim as a nonresid one; the denominators are different
   objects (``UnitLayout.summary`` prints both, and the run's ``delta_stats.json`` records them).
2. **The rank is a property of the delta.** A LoRA-r32 adapter's merged delta has at most 32
   nonzero singular values per tensor, so factoring it is cheap and the unit total is tiny
   (7,168 for the seven block projections of a 32-layer 8B model) -- but the same code over a
   full-parameter finetune faces a numerically full-rank delta, where ``mask.svd_rank`` is the
   difference between a feasible run and an infeasible one. Hence the cap, the tolerance, and
   :func:`build_factors` reporting the reconstruction error it actually achieved rather than
   assuming the truncation was lossless.
3. **The delta has to be frozen.** These factors are computed once. A co-trained delta changes
   every step, so its singular directions would too, and a score would not refer to the same
   object from one step to the next -- ``config/schema.py`` rejects ``svd*`` without
   ``mask.finetuned`` / ``mask.init_delta``.

**Factors are kept in fp32 even when the run composes in bf16.** They are ~1% of the delta's
size (``r . (m + n)`` against ``m . n``), so the memory argument that made ``mask.delta_dtype``
bf16 does not apply, and the reconstruction is a matmul whose rounding would otherwise land on
the operands rather than once at the end. ``theta_eff`` is still produced in the run's compose
dtype, so the *weights* an svd run evaluates are in the same precision a dense run's are.
"""

import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

#: Extra sketch columns above the requested rank for the randomised path. The range finder is
#: exact once the sketch spans the delta's column space, so this is the margin that makes "rank
#: cap 32 over a rank-32 LoRA delta" exact rather than nearly so.
OVERSAMPLE = 8

METHODS = ("auto", "full", "lowrank")


@dataclass
class SvdFactors:
    """One tensor's truncated SVD, and the mask arithmetic over it.

    ``S`` is descending, so unit 0 is the delta's dominant direction for this tensor. That
    ordering is what makes ``scripts/unit_ranks.py``-style diagnostics readable on an svd run --
    a low unit index is a big direction -- and it costs nothing, since every factorisation here
    produces it already sorted.
    """

    U: torch.Tensor      # [m, r]
    S: torch.Tensor      # [r], descending
    Vh: torch.Tensor     # [r, n]

    @property
    def rank(self) -> int:
        return int(self.S.numel())

    @property
    def shape(self) -> tuple:
        return (int(self.U.shape[0]), int(self.Vh.shape[-1]))

    def to(self, device=None, dtype=None) -> "SvdFactors":
        return SvdFactors(self.U.to(device=device, dtype=dtype),
                          self.S.to(device=device, dtype=dtype),
                          self.Vh.to(device=device, dtype=dtype))

    def delta(self, m: torch.Tensor = None, *, scale: float = 1.0) -> torch.Tensor:
        """``U diag(m . S) Vh`` -- the masked delta, still attached to ``m``'s graph.

        ``m=None`` means the whole delta (every direction kept), which is what the two anchors
        and a ``frac 1.0`` condition want.

        The scaling is folded into ``S`` first, so the elementwise multiply is over ``[r]``
        values and the only large allocation is the ``[m, n]`` result. Doing it the other way --
        scaling the reconstruction -- would allocate a second full-size tensor.
        """
        s = self.S if m is None else m.to(device=self.S.device, dtype=self.S.dtype) * self.S
        if scale != 1.0:
            s = s * scale
        return (self.U * s) @ self.Vh

    def attribution(self, grad: torch.Tensor) -> torch.Tensor:
        """Per-direction first-order term ``sum over unit i of delta . g`` -- a ``[r]`` vector.

        Unit ``i``'s slice of the delta is the rank-1 matrix ``S[i] u_i v_i^T``, so its inner
        product with the gradient is ``S[i] . u_i^T G v_i``. This is the svd counterpart of
        ``masks.unit_sums`` and is **signed** for the same reason (see ``train/ixg.py``): the sign
        says whether the direction raises or lowers the loss, which is the whole content of the
        ranking.
        """
        g = grad.to(device=self.U.device, dtype=self.U.dtype)
        return self.S * ((self.U.transpose(-1, -2) @ g) * self.Vh).sum(-1)

    def rel_error(self, delta: torch.Tensor) -> float:
        """Relative Frobenius error of this truncation against the delta it came from."""
        d = delta.to(device=self.U.device, dtype=self.U.dtype)
        denom = float(d.norm())
        if denom == 0.0:
            return 0.0
        return float((d - self.delta()).norm()) / denom


def _lowrank_svd(d: torch.Tensor, q: int, *, niter: int = 2, seed: int = 0):
    """Randomised range-finder SVD of ``d``, truncated to ``q`` directions.

    Exact (to floating point) whenever ``q`` is at least the delta's rank, which is the case this
    exists for: a merged LoRA-r32 delta is rank <= 32, and a full ``linalg.svd`` of a
    ``[14336, 4096]`` tensor to recover 32 directions is two orders of magnitude of wasted work.
    Written out rather than calling ``torch.svd_lowrank`` so the sketch uses an **explicit
    generator**: that function draws from the global RNG, and a factorisation that perturbs the
    run's RNG stream (or that cannot be reproduced from the checkpoint) is not worth the three
    lines it saves.
    """
    n = d.shape[-1]
    gen = torch.Generator(device=d.device).manual_seed(seed)
    omega = torch.randn(n, q, generator=gen, device=d.device, dtype=d.dtype)
    qm, _ = torch.linalg.qr(d @ omega)
    for _ in range(niter):
        # power iteration: sharpens the range estimate when the spectrum has no clean gap. A
        # no-op for an exactly-low-rank delta, where the first sketch already spans the range.
        qm, _ = torch.linalg.qr(d @ (d.transpose(-1, -2) @ qm))
    b = qm.transpose(-1, -2) @ d
    ub, s, vh = torch.linalg.svd(b, full_matrices=False)
    return qm @ ub, s, vh


def factor(delta: torch.Tensor, *, rank: int = None, tol: float = 1e-6, method: str = "auto",
           seed: int = 0, dtype=torch.float32) -> SvdFactors:
    """Truncated SVD of one delta tensor.

    Args:
        rank: hard cap on directions kept. ``None`` keeps every direction above ``tol``, which
            for a full-rank delta is ``min(m, n)`` of them -- feasible at 1B, not at 8B.
        tol: drop singular values at or below ``tol . S[0]``. What makes a LoRA delta's
            numerically-zero tail (fp32 noise from ``theta_ft - theta_base``, ~1e-7 relative)
            not become thousands of dead units.
        method: ``lowrank`` is the randomised path, ``full`` is ``linalg.svd``, ``auto`` picks
            the former exactly when a cap is set and is small against the tensor.

    At least one direction is always kept, even for an exactly-zero delta: a tensor with zero
    units would drop out of the layout entirely, and "this tensor was not moved" should show up
    as a dead unit (which ``train/posthoc.dead_units`` counts) rather than as an absence.
    """
    if method not in METHODS:
        raise ValueError(f"svd method must be one of {METHODS}, got {method!r}")
    if delta.ndim != 2:
        raise ValueError(f"only a 2-D delta has singular directions; got shape "
                         f"{tuple(delta.shape)}")
    d = delta.to(dtype)
    p = min(d.shape)
    if method == "auto":
        method = "lowrank" if (rank is not None and rank + OVERSAMPLE < p // 2) else "full"
    if method == "lowrank":
        u, s, vh = _lowrank_svd(d, min(int(rank) + OVERSAMPLE, p), seed=seed)
    else:
        u, s, vh = torch.linalg.svd(d, full_matrices=False)

    keep = int((s > tol * s[0]).sum()) if float(s[0]) > 0 else 0
    if rank is not None:
        keep = min(keep, int(rank))
    keep = max(1, keep)
    return SvdFactors(u[:, :keep].contiguous(), s[:keep].contiguous(),
                      vh[:keep].contiguous())


def build_factors(deltas: dict, names, *, rank: int = None, tol: float = 1e-6,
                  method: str = "auto", device=None, dtype=torch.float32,
                  check_tol: float = 0.01, work_device=None) -> tuple:
    """Factor the named deltas, and verify the truncation. ``({name: SvdFactors}, stats)``.

    ``work_device`` is where the decomposition runs (the deltas usually live on the CPU, and a
    GPU is 1-2 orders of magnitude faster at this); ``device`` is where the factors are kept.

    The reconstruction error is **measured, not assumed**, for every tensor, and a truncation
    worse than ``check_tol`` is a hard error. That check is the whole safety story for
    ``mask.svd_rank``: a cap below the delta's real rank silently throws away part of the
    finetune, so the run's ``full_delta`` anchor would stop being the finetune and every
    normalised number in the sweep would be measured against the wrong ceiling.
    """
    names = list(names)
    if not names:
        raise ValueError(
            "this unit mode factors no tensors at all, so no unit is a singular direction. "
            "Under svd_attn/svd_mlp that means the attention/MLP name fragments in "
            "masks.layout matched nothing in this model -- check the parameter names against "
            "_ATTN_PARTS / _MLP_PARTS, and against mask.exclude_params.")
    t0 = time.time()
    out, errs, zero = {}, [], 0
    for i, name in enumerate(names):
        # Moved BEFORE the decomposition, not after: the deltas live on the CPU (built there in
        # fp32 by train/posthoc) and a CPU `linalg.svd` of a [14336, 4096] tensor is minutes, not
        # milliseconds. The fp32 copy is transient and per tensor.
        d = deltas[name].to(device=work_device, dtype=torch.float32) if work_device is not None \
            else deltas[name].to(torch.float32)
        # seeded per tensor by INDEX, so the sketch is reproducible from the layout alone and two
        # tensors do not share a draw
        f = factor(d, rank=rank, tol=tol, method=method, seed=i, dtype=torch.float32)
        err = f.rel_error(d)
        if err > check_tol:
            raise SystemExit(
                f"the truncated SVD of {name} loses {100 * err:.2f}% of the delta (limit "
                f"{100 * check_tol:.2f}%), so this mask would not be attributing the finetune it "
                f"names: kept {f.rank} of {min(tuple(d.shape))} directions. Raise mask.svd_rank, "
                f"or lower mask.svd_tol.")
        errs.append(err)
        if float(f.S.max()) == 0.0:
            zero += 1
        out[name] = f.to(device=device, dtype=dtype)
        del f, d
    ranks = sorted(f.rank for f in out.values())
    stats = {
        "svd_tensors": len(out),
        "svd_units": sum(ranks),
        "svd_rank_min": ranks[0],
        "svd_rank_median": ranks[len(ranks) // 2],
        "svd_rank_max": ranks[-1],
        "svd_rel_error_max": max(errs),
        "svd_rel_error_mean": sum(errs) / len(errs),
        "svd_zero_delta_tensors": zero,
        "svd_method": method,
        "svd_rank_cap": rank,
        "svd_tol": tol,
        "svd_seconds": time.time() - t0,
    }
    logger.info("factored %d tensor(s) into %s singular directions (rank %d-%d, median %d) in "
                "%.1fs; relative reconstruction error max %.3g, mean %.3g",
                stats["svd_tensors"], f"{stats['svd_units']:,}", ranks[0], ranks[-1],
                stats["svd_rank_median"], stats["svd_seconds"], stats["svd_rel_error_max"],
                stats["svd_rel_error_mean"])
    if zero:
        logger.info("%d factored tensor(s) have an exactly-zero delta, so their one unit is dead",
                    zero)
    return out, stats


def factors_for_layout(deltas: dict, layout, *, method: str = "auto", device=None,
                       dtype=torch.float32, work_device=None) -> dict:
    """Re-derive a saved layout's factors from a rebuilt delta, at exactly its own ranks.

    For the post-hoc eval CLI, which reconstructs ``delta = theta_ft - theta_base`` for a run
    that did not persist it. The ranks come from the layout's unit counts rather than from a
    tolerance, because the scores are indexed by them: re-deciding the rank here would silently
    shift every unit's meaning against the score vector it is being masked with.
    """
    ranks = layout.ranks()
    names = list(layout.names)
    out = {}
    for name, r in ranks.items():
        d = deltas[name]
        d = d.to(device=work_device, dtype=torch.float32) if work_device is not None \
            else d.to(torch.float32)
        # `seed` mirrors build_factors', where the index is into the FACTORED names in order --
        # the same order `layout.svd_names` reports, since both come from the layout's names
        f = factor(d, rank=int(r), tol=0.0, method=method,
                   seed=[n for n in names if n in ranks].index(name), dtype=torch.float32)
        if f.rank != int(r):
            raise SystemExit(
                f"re-factoring {name} gave {f.rank} directions where the checkpoint's layout "
                f"has {int(r)}; the scores are indexed by that count, so this delta is not the "
                "one the run was fitted over.")
        out[name] = f.to(device=device, dtype=dtype)
    return out


def to_blob(factors: dict) -> dict:
    """Serialisable form: plain tensors on the CPU."""
    return {n: {"U": f.U.detach().cpu(), "S": f.S.detach().cpu(), "Vh": f.Vh.detach().cpu()}
            for n, f in factors.items()}


def from_blob(blob: dict) -> dict:
    return {n: SvdFactors(d["U"], d["S"], d["Vh"]) for n, d in (blob or {}).items()}
