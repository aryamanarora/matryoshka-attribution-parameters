"""Wiring check for the editable ``learning-to-attribute`` dependency.

Trains MAttr scores on the analytic linear toy (``y = sum_i a_i x_i``, one node per product
term) and asserts the learned ranking recovers the ground-truth ordering ``|a_i|``. No model
download, no GPU, a few seconds.

This exercises the whole imported stack from inside this repo: the differentiable top-k
primitive and its implicit-diff backward (``sigmoid_topk``), the mask-variant registry
(``build_mask``), the optimizer loop (``learn_scores``), and the k-schedules — so a pass means
the dependency resolves *and* gradients flow score -> mask -> loss.

Setup matches the parent repo's headline MAttr variant: soft top-k forward, log-uniform k,
Adam lr=0.05, T=0.5, denoising / ``iso`` intervention (top-k stays clean, complement patched
with the counterfactual).

    uv run python scripts/smoke_dep.py
"""

import argparse

import torch
from scipy.stats import spearmanr

import learning_to_attribute as l2a
from learning_to_attribute import learn_scores, normalize_mode


def run(n=64, steps=1500, batch=16, lr=0.05, T=0.5, seed=0, k_schedule="log"):
    """Learn scores for the n product nodes; return (scores, true importance |a|)."""
    torch.manual_seed(seed)
    a = torch.randn(n)                                  # frozen "model weights"

    assert normalize_mode("iso") == "sufficient"        # convention still as documented

    def loss_fn(mask):
        # denoising: node i keeps its clean value a_i*x_i where mask=1, else the CF value
        # a_i*x'_i. Target is the fully clean output, so the residual is
        # sum_i (1-m_i) * a_i * (x'_i - x_i) -- keeping the largest-|a_i| nodes clean wins.
        x = torch.randn(batch, n)
        x_cf = torch.randn(batch, n)
        contrib = a * x
        contrib_cf = a * x_cf
        y_mask = (mask * contrib + (1 - mask) * contrib_cf).sum(-1)
        y_clean = contrib.sum(-1)
        return ((y_mask - y_clean) ** 2).mean()

    res = learn_scores(
        n, loss_fn,
        steps=steps, variant="topk", k_schedule=k_schedule,
        T=T, lr=lr, optimizer="adam",
    )
    return res, a.abs()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-spearman", type=float, default=0.85)
    args = p.parse_args()

    print(f"learning_to_attribute from: {l2a.__file__}")
    print(f"mask variants available:    {len(l2a.VARIANTS)}")

    res, true_imp = run(n=args.n, steps=args.steps, seed=args.seed)
    scores = res.scores

    rho = spearmanr(scores.numpy(), true_imp.numpy()).statistic
    m = max(1, args.n // 4)
    learned_top = set(scores.topk(m).indices.tolist())
    true_top = set(true_imp.topk(m).indices.tolist())
    overlap = len(learned_top & true_top) / m

    # NOTE: loss is NOT expected to fall monotonically -- k is resampled log-uniformly every
    # step, and the loss at small k is far larger than at large k, so the trace is dominated
    # by k noise. Recovery of the |a| ordering is the thing to read.
    print(f"steps={args.steps} n={args.n}  "
          f"mean loss (first/last 50 steps) {sum(res.loss_log[:50]) / 50:.3f} / "
          f"{sum(res.loss_log[-50:]) / 50:.3f}  ({res.train_time_s:.1f}s)")
    print(f"spearman(scores, |a|) = {rho:.3f}   top-{m} overlap = {overlap:.2f}")

    assert rho > args.min_spearman, f"recovery too low: rho={rho:.3f}"
    assert overlap >= 0.75, f"top-{m} overlap too low: {overlap:.2f}"
    print("OK: editable learning-to-attribute dependency is wired and differentiating.")


if __name__ == "__main__":
    main()
