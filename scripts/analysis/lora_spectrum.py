"""How concentrated is a LoRA delta's singular spectrum? Exact, from the adapter, in milliseconds.

    uv run python scripts/analysis/lora_spectrum.py <adapter dir>

Written for one question about the `svd*` unit modes: `mask.svd_basis: random` is their control, and
it works by flattening the per-term magnitudes, so "how peaked were they to begin with" decides how
much that control can possibly change. If the spectrum is already near-uniform there is little
magnitude structure to destroy, and a control that matches its svd twin means something weaker than
it would over a peaked spectrum. That is a fact about the delta, not about the runs, so it is
checkable without a GPU or a sweep.


delta = c B A with B [m,r], A [r,n], so rank(delta) <= r and the whole spectrum lives in an r x r
matrix. Take QRs of B and A^T:

    B = Q_B R_B      A^T = Q_A R_A      =>   delta = c Q_B (R_B R_A^T) Q_A^T

with Q_B, Q_A orthonormal, so the nonzero singular values of delta are exactly c times those of the
r x r matrix R_B R_A^T. No [14336, 4096] decomposition anywhere -- the same reason
masks/svd.py takes the randomised path when a rank cap is set.

||delta||_F is printed as the check that the scale is the one PEFT applied: it must match the
delta_norm the post-hoc runs recorded (45.46), and ||delta||_F == c ||R_B R_A^T||_F falls out of
the same identity.
"""
import json, math, sys, torch
from pathlib import Path
from safetensors.torch import load_file

d = Path(sys.argv[1])
cfg = json.loads((d / "adapter_config.json").read_text())
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
sd = load_file(d / "adapter_model.safetensors")
print(f"r={r} alpha={alpha} rslora={cfg.get('use_rslora')} -> scale={scale:.4f}")

pairs = {}
for k, v in sd.items():
    if ".lora_A." in k or ".lora_B." in k:
        pairs.setdefault(k.split(".lora_")[0], {})["A" if ".lora_A." in k else "B"] = v

#: the sweep's own grid, as unit counts out of r -- so a row of this table lines up with a
#: condition of the sparsity curve
KS = sorted({max(1, round(f * r)) for f in (0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0)})

sq, shares, by_part, recon = 0.0, [], {"attn": [], "mlp": []}, {}
for name, ab in sorted(pairs.items()):
    _, rb = torch.linalg.qr(ab["B"].float())          # [r, r]
    _, ra = torch.linalg.qr(ab["A"].float().T)        # [r, r]
    m_small = rb @ ra.T                               # [r, r]; delta = scale Q_B m_small Q_A^T
    s = scale * torch.linalg.svdvals(m_small)
    s = s[s > 1e-6 * s[0]]
    sq += float((s ** 2).sum())                       # ||delta||_F^2 == sum of squared s
    shares.append(float(s[0] / s.sum()))
    part = "mlp" if any(t in name for t in ("gate_proj", "up_proj", "down_proj")) else "attn"
    by_part[part].append(shares[-1])
    # --- how well does a top-k SUBSET of terms reconstruct the delta, in each basis? ---
    #
    # This is the whole explanation of why the random-basis control collapses, and it is exact in
    # r dimensions. Writing delta = c Q_B M Q_A^T with M = R_B R_A^T and Q_B, Q_A orthonormal, any
    # partial sum of rank-1 terms is c Q_B G Q_A^T for some r x r G, and orthonormality gives
    #
    #     ||delta - (partial sum)||_F == ||c M - G||_F
    #
    # so no [m, n] tensor is ever formed. In the SVD basis the top-k terms are orthogonal and G is
    # the truncated spectrum (Eckart-Young: the best possible rank-k approximation). In the rotated
    # basis the terms are NOT orthogonal -- they are individually large and mutually cancelling --
    # so dropping any of them leaves error that the remaining ones cannot absorb.
    u_m, s_m, vh_m = torch.linalg.svd(m_small)
    cs = scale * s_m
    cm = scale * m_small
    root = cs.clamp_min(0).sqrt()
    g = torch.Generator().manual_seed(99)
    q_rot, upper = torch.linalg.qr(torch.randn(r, r, generator=g))
    q_rot = q_rot * upper.diagonal().sign()
    a_rot = (u_m * root) @ q_rot                       # [r, r], the rotated left factors
    b_rot = q_rot.T @ (root.unsqueeze(-1) * vh_m)      # [r, r]
    order = (a_rot.norm(dim=0) * b_rot.norm(dim=1)).argsort(descending=True)
    denom = float(cm.norm())
    for k in KS:
        # SVD: keep the k largest singular values
        svd_err = float((cs[k:] ** 2).sum().sqrt()) / denom
        # random basis: keep the k largest-norm terms
        keep = order[:k]
        g_k = a_rot[:, keep] @ b_rot[keep]
        rnd_err = float((cm - g_k).norm()) / denom
        recon.setdefault(k, []).append((svd_err, rnd_err))

n = len(shares)
print(f"{n} tensors, ||delta||_F = {sq ** 0.5:.4f}   (post-hoc runs recorded 45.46)")
print(f"leading singular value's share of sum(S): mean {100 * sum(shares) / n:.1f}%  "
      f"min {100 * min(shares):.1f}%  max {100 * max(shares):.1f}%")
for p, v in by_part.items():
    print(f"  {p} ({len(v)} tensors): {100 * sum(v) / len(v):.1f}%")
print(f"uniform floor (1/r) = {100 / r:.1f}%   <- what a random rotation flattens toward")
print()
print("relative error of the best top-k reconstruction of the delta, mean over tensors:")
print(f"  {'k':>4s} {'k/r':>7s} {'SVD basis':>12s} {'random basis':>14s}")
for k in KS:
    v = recon[k]
    sv = sum(x for x, _ in v) / len(v)
    rn = sum(y for _, y in v) / len(v)
    print(f"  {k:>4d} {k / r:>7.1%} {sv:>12.3f} {rn:>14.3f}")
print("  (1.000 = as far from the delta as keeping nothing at all)")
