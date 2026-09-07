"""`ixg_at: ig` (fixed-grid integrated gradients) satisfies the completeness axiom.

IG's defining property: the attributions sum to ``F(theta_finetuned) - F(theta_base)``. On a
quadratic loss the gradient is linear in alpha, so the right-Riemann grid's error is exactly
``(delta . H . delta) / 2m`` -- computable in closed form -- and the test pins completeness to
that bound rather than a hand-waved tolerance. Sign/normalisation conventions from
``ixg_scores``: scores are NEGATED and divided by the total supervised-token count, with tokens
counted per (batch, alpha) pass, so ``sum(scores) == (F(0) - F(1)) / n_tokens_per_pass`` up to
the Riemann term.

Also pinned: steps=1 under `ig` is rejected (it is the `finetuned` endpoint under another
name), and mc/base behaviour is untouched by the new branch (same scores as before on a seeded
run -- the branch is `continue`-guarded).
"""

import pytest
import torch

from mask_learning_finetuning.masks.layout import build_layout
from mask_learning_finetuning.train.ixg import ixg_scores


class Lin(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        self.weight = torch.nn.Parameter(W.clone())

    def forward(self, x):
        return x @ self.weight.T


def setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    W0 = torch.randn(3, 4, generator=g)
    delta = torch.randn(3, 4, generator=g) * 0.5
    X = torch.randn(6, 4, generator=g)
    m = Lin(W0)
    layout = build_layout([("weight", m.weight)], "row")
    labels = torch.ones(1, 5, dtype=torch.long)   # 4 supervised tokens after the shift
    batch = {"labels": labels}
    loss_fn = lambda model, b: 0.5 * model(X).pow(2).sum()
    F = lambda W: float(0.5 * (X @ W.T).pow(2).sum())
    return m, layout, delta, batch, loss_fn, F, X


def test_completeness_within_riemann_bound():
    m, layout, delta, batch, loss_fn, F, X = setup()
    steps = 8
    scores, stats = ixg_scores(m, base={"weight": m.weight.detach().clone()},
                               deltas={"weight": delta}, layout=layout,
                               batches=[batch], loss_fn=loss_fn, at="ig", steps=steps)
    # scores divide by tokens-counted-per-pass (4 * steps), and the SUM over passes has a
    # factor `steps` in it, so multiplying back by tokens-per-batch (4) recovers the MEAN over
    # grid points -- the Riemann average completeness speaks about
    got = float(scores.sum()) * 4
    W0 = m.weight.detach()
    exact = F(W0) - F(W0 + delta)          # completeness target (negated convention)
    # right-Riemann on a linear gradient overshoots by (delta.H.delta)/2m exactly
    H_quad = float((X @ delta.T).pow(2).sum())   # delta.H.delta for this loss
    riemann = H_quad / (2 * steps)
    assert abs(got - (exact - riemann)) < 1e-3 * max(1.0, abs(exact)), (got, exact, riemann)


def test_more_steps_converge():
    m, layout, delta, batch, loss_fn, F, X = setup(1)
    errs = []
    for steps in (2, 8, 32):
        scores, _ = ixg_scores(m, base={"weight": m.weight.detach().clone()},
                               deltas={"weight": delta}, layout=layout,
                               batches=[batch], loss_fn=loss_fn, at="ig", steps=steps)
        W0 = m.weight.detach()
        errs.append(abs(float(scores.sum()) * 4 - (F(W0) - F(W0 + delta))))
    assert errs[0] > errs[1] > errs[2]


def test_steps_one_rejected():
    m, layout, delta, batch, loss_fn, F, X = setup(2)
    with pytest.raises(ValueError, match="ixg_steps"):
        ixg_scores(m, base={"weight": m.weight.detach().clone()}, deltas={"weight": delta},
                   layout=layout, batches=[batch], loss_fn=loss_fn, at="ig", steps=1)


def test_endpoint_modes_unchanged():
    for at in ("base", "finetuned"):
        m, layout, delta, batch, loss_fn, F, X = setup(3)
        scores, _ = ixg_scores(m, base={"weight": m.weight.detach().clone()},
                               deltas={"weight": delta}, layout=layout,
                               batches=[batch], loss_fn=loss_fn, at=at)
        # the endpoint first-order term, computed by hand: -delta . grad(F at endpoint) / n_tok
        W0 = m.weight.detach()
        W = W0 if at == "base" else W0 + delta
        g = (X @ W.T).T @ X                # grad of 0.5||XW^T||^2 wrt W
        want = -(delta * g).sum(dim=1) / 4
        assert torch.allclose(scores, want, rtol=1e-4), at
