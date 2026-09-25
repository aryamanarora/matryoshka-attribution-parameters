"""Reward IxG (``mask.scores: ixg`` under ``rl:``) -- the two exact claims behind the baseline.

The score is ``-delta . grad L`` with ``L`` a REINFORCE surrogate whose backward the loss function
runs itself, one generated sequence at a time. Two things about that are arithmetic and can be
pinned without a judge, an engine or a model download:

* handing ``ixg_scores`` a loss function that has ALREADY accumulated its gradients and returns a
  detached total gives the same scores as the ordinary path, to the bit -- so the reward path's
  per-sequence backward changes nothing about how the attribution is reduced;
* for a single sample with advantage ``a``, the scores sum to the first-order estimate of
  ``a . (mean log p at theta + delta  -  mean log p at theta)``: the sign is "adding the delta
  raises the log-probability of an above-average sample scores positive", which is what the GRPO
  fit's top-k also means, and what a wrong sign would silently invert into "the worst units".

Plus the config surface: the four refusal cells resolve, differ from their GRPO twins in exactly
``mask.scores`` / ``mask.ixg_at`` (+ name/output), and the two knobs this path would otherwise
silently ignore are rejected.
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from mask_learning_finetuning.config import loader
from mask_learning_finetuning.masks import build_layout
from mask_learning_finetuning.train.ixg import ixg_scores

ROOT = Path(__file__).resolve().parents[1]


def _tiny(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    return AutoModelForCausalLM.from_config(cfg).eval()


def _setup():
    model = _tiny()
    base = {n: p.detach() for n, p in model.named_parameters()}
    named = list(model.named_parameters())
    layout = build_layout(named, "nonresid", resid_dim=64)
    g = torch.Generator().manual_seed(3)
    deltas = {n: 0.02 * torch.randn(base[n].shape, generator=g) for n in layout.names}
    return model, base, layout, deltas


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 128, (2, 12), generator=g)
    labels = ids.clone()
    labels[:, :3] = -100
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def _token_ce(model, b):
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])
    return torch.nn.functional.cross_entropy(logits.float(), b["labels"][:, 1:].reshape(-1),
                                             ignore_index=-100, reduction="sum")


@pytest.mark.parametrize("at", ["base", "finetuned"])
def test_self_backward_loss_fn_gives_identical_scores(at):
    model, base, layout, deltas = _setup()
    ref, _ = ixg_scores(model, base=base, deltas=deltas, layout=layout, batches=[_batch()],
                        loss_fn=_token_ce, at=at)

    def own_backward(m, b):
        # what the reward surrogate does: accumulate per sequence, hand back the detached total
        total = 0.0
        for i in range(b["input_ids"].shape[0]):
            row = {k: v[i:i + 1] for k, v in b.items()}
            ce = _token_ce(m, row)
            ce.backward()
            total += float(ce.detach())
        return torch.tensor(total)

    got, _ = ixg_scores(model, base=base, deltas=deltas, layout=layout, batches=[_batch()],
                        loss_fn=own_backward, at=at)
    assert torch.allclose(ref, got, rtol=1e-5, atol=1e-7)


def test_count_fn_replaces_the_token_normaliser():
    model, base, layout, deltas = _setup()
    tokens, _ = ixg_scores(model, base=base, deltas=deltas, layout=layout, batches=[_batch()],
                           loss_fn=_token_ce, at="base")
    per_draw, stats = ixg_scores(model, base=base, deltas=deltas, layout=layout,
                                 batches=[_batch()], loss_fn=_token_ce, at="base",
                                 count_fn=lambda b: 1)
    n_tok = int((_batch()["labels"][:, 1:] != -100).sum())
    assert stats["ixg_tokens"] == 1
    assert torch.allclose(per_draw, tokens * n_tok, rtol=1e-5, atol=1e-6)


def test_surrogate_scores_sum_to_first_order_reward_change():
    """One sample, advantage a: sum_i score_i ~= a . d(mean log p)/d(alpha) at alpha=0."""
    model, base, layout, deltas = _setup()
    ids = torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(7))
    n_p, a = 4, 1.5

    def mean_lp(m):
        lp = m(input_ids=ids).logits[0, :-1].float().log_softmax(-1)
        return lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).mean()

    def surrogate(m, _):
        loss = -a * mean_lp(m)
        loss.backward()
        return loss.detach()

    scores, _ = ixg_scores(model, base=base, deltas=deltas, layout=layout, batches=[None],
                           loss_fn=surrogate, at="base", count_fn=lambda b: 1)
    # finite-difference the objective along the delta, on an independent copy
    eps = 1e-3
    probe = _tiny()
    with torch.no_grad():
        f0 = a * float(mean_lp(probe))
        for n, p in probe.named_parameters():
            if n in deltas:
                p.add_(eps * deltas[n])
        f1 = a * float(mean_lp(probe))
    directional = (f1 - f0) / eps
    assert abs(float(scores.sum()) - directional) < 0.02 * max(1.0, abs(directional))
    assert float(scores.sum()) * directional > 0            # and the sign in particular


def _twin_diff(cell, twin):
    a, b = loader.to_dict(loader.load_config(cell)), loader.to_dict(loader.load_config(twin))

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{pre}{k}."))
            else:
                out[f"{pre}{k}"] = v
        return out
    fa, fb = flat(a), flat(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


@pytest.mark.parametrize("cell, twin, extra", [
    ("mc_vllm_native", "uniform_vllm_native", {"mask.scores"}),
    ("mc_8b_vllm_native", "uniform_8b_vllm_native", {"mask.scores"}),
    ("base_vllm_native", "uniform_vllm_native", {"mask.scores"}),
    ("base_8b_vllm_native", "uniform_8b_vllm_native", {"mask.scores"}),
])
def test_refusal_ixg_cells_are_twins_of_the_grpo_fits(cell, twin, extra):
    diff = _twin_diff(ROOT / "configs/refusal/ixg" / f"{cell}.yaml",
                      ROOT / "configs/refusal/rl" / f"{twin}.yaml")
    # the `mc` cells keep ixg_at's default value out of the diff only if it differs; state it
    assert set(diff) - {"name", "output", "mask.ixg_at"} == extra, diff


def test_ixg_batches_and_k_fixed_are_rejected_under_rl():
    raw = loader.load_yaml_tree(ROOT / "configs/refusal/ixg/mc_vllm_native.yaml")
    loader.config_from_dict(raw)                                   # the file itself resolves
    with pytest.raises(ValueError, match="ixg_batches"):
        loader.config_from_dict({**raw, "mask": {**raw["mask"], "ixg_batches": 100}})
    with pytest.raises(ValueError, match="k_fixed"):
        loader.config_from_dict({**raw, "mask": {**raw["mask"], "k_fixed": 0.01}})
    with pytest.raises(ValueError, match="kl_coef"):
        loader.config_from_dict({**raw, "rl": {**raw["rl"], "kl_coef": 0.01}})
