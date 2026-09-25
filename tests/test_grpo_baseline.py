"""The three knobs that turn this repo's weight-space GRPO into GRP-Obliteration.

Each is a small piece of arithmetic with an exact claim attached, so each is pinned rather than
asserted: the cosine schedule's endpoints, DAPO's gate keeping exactly the positive advantages,
and the KL reference being the model the run STARTED from -- which is what makes the penalty zero
at step 0 and is the one of the three that could be silently wrong (a reference that drifted with
the policy gives a penalty near zero forever, which reads as "KL does nothing").
"""

import torch

from mask_learning_finetuning.train.rl import _lr_at, _Reference, advantages


class _Rl:
    def __init__(self, **kw):
        self.steps = 100
        self.lr_schedule = "constant"
        self.__dict__.update(kw)


class _Train:
    lr = 1e-5


class _Cfg:
    def __init__(self, rl, model=None, device="cpu"):
        self.rl, self.train, self.model, self.device = rl, _Train(), model, device
        self.trust_remote_code = False


def test_constant_schedule_is_the_configured_lr():
    cfg = _Cfg(_Rl())
    assert [_lr_at(cfg, s) for s in (0, 50, 99)] == [1e-5] * 3


def test_cosine_decays_from_lr_to_zero_over_rl_steps():
    cfg = _Cfg(_Rl(lr_schedule="cosine", steps=100))
    assert _lr_at(cfg, 0) == 1e-5                       # no warmup: step 0 is the full rate
    assert abs(_lr_at(cfg, 50) - 5e-6) < 1e-12          # half way -> half
    assert abs(_lr_at(cfg, 100)) < 1e-20                # the end of the budget -> zero
    # monotone in between, which "cosine" would still be true of if the sign were flipped
    xs = [_lr_at(cfg, s) for s in range(0, 101, 10)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))


def test_dapo_gate_keeps_exactly_the_positive_advantages():
    # the loop's condition, stated once here so the claim is checkable without running a step
    r = torch.tensor([0.0, 1.0, 0.25, 0.75])
    a = advantages(r)
    keep_grpo = [x for x in a.tolist() if x != 0.0]
    keep_dapo = [x for x in a.tolist() if not (x == 0.0 or x < 0.0)]
    assert len(keep_grpo) == 4 and len(keep_dapo) == 2   # two samples beat the group mean
    assert all(x > 0 for x in keep_dapo)
    assert sum(a).abs() < 1e-5                           # group-relative: advantages sum to zero


def test_unanimous_group_is_dropped_by_both_losses():
    a = advantages(torch.tensor([0.4, 0.4, 0.4]))
    assert torch.equal(a, torch.zeros(3))


def _tiny_model(path):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=4))
    m.save_pretrained(path)
    return m


def test_full_parameter_reference_is_the_starting_model(tmp_path):
    """A Direct policy's reference is a frozen copy, so its log-probs equal the policy's before
    any step and stop equalling them after one -- the two halves of "zero KL at step 0"."""
    path = str(tmp_path / "m")
    model = _tiny_model(path)
    model.eval()

    class P:                                   # what Direct exposes to _Reference
        pass
    P.model = model
    ref = _Reference(model, P, _Cfg(_Rl(kl_coef=0.01), model=path))
    assert ref.peft is False and ref.ref is not None

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    n_p = 3
    def policy_lp():
        with torch.no_grad():
            lp = model(input_ids=ids).logits[0, :-1].float().log_softmax(-1)
        return lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)

    before = ref.token_logprobs(ids, n_p)
    # bf16 reference against an fp32 policy: equal to bf16's precision, not to fp32's
    assert torch.allclose(before, policy_lp(), atol=5e-2)

    with torch.no_grad():                      # move the policy; the reference must not follow
        for q in model.parameters():
            q.add_(torch.randn_like(q) * 0.05)
    assert torch.allclose(ref.token_logprobs(ids, n_p), before, atol=5e-2)
    assert not torch.allclose(ref.token_logprobs(ids, n_p), policy_lp(), atol=5e-2)

    ref.release()
    assert ref.ref is None


def test_k3_estimator_is_non_negative():
    """``exp(r) - r - 1`` for ``r = logp_ref - logp`` -- the reason the naive ``-r`` is not used."""
    for r in torch.linspace(-3, 3, 25):
        assert (torch.exp(r) - r - 1) >= -1e-6
