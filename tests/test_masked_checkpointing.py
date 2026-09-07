"""Gradient checkpointing on the masked path gives the SAME score gradient as no checkpointing.

The exact claim behind `train.grad_checkpointing` being usable for a post-hoc attribution at all:
`MaskedDelta.loss` runs the model through `functional_call`, HF's non-reentrant checkpoint
recomputes each block during backward outside that parameter swap, and `MaskedDelta` hands it a
recompute context that re-installs the composed parameters. Pinned on a tiny random Llama, in
train() mode (HF only checkpoints when `self.training`), with the checkpoint calls counted so a
silent no-op cannot pass.
"""

import torch
import torch.utils.checkpoint as ckpt


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      attention_dropout=0.0)
    return LlamaForCausalLM(cfg).float()


def _cfg(tmp_path, grad_ckpt):
    from mask_learning_finetuning.config import load_config
    import yaml
    d = {"name": "t", "model": "x", "output": str(tmp_path / "out"), "device": "cpu",
         "data": {"train": "data/toy_chat.jsonl"},
         "train": {"grad_checkpointing": grad_ckpt, "dropout": True, "dtype": "float32",
                   "amp": None},
         "mask": {"unit": "head", "delta_dtype": "float32", "exclude_params": "embed|lm_head|norm"}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return load_config(str(p))


def _score_grad(tmp_path, grad_ckpt, counter):
    from mask_learning_finetuning.train.params import MaskedDelta
    model = _tiny_llama()
    model.train()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = _cfg(tmp_path, grad_ckpt)
    torch.manual_seed(1)
    delta = {n: 0.05 * torch.randn_like(p) for n, p in model.named_parameters()
             if "self_attn" in n or "mlp" in n}
    P = MaskedDelta(model, cfg, init_delta=None, freeze_delta=True)
    with torch.no_grad():
        for n in P.deltas:
            P.deltas[n].copy_(delta[n])
        P.scores.copy_(torch.linspace(-1, 1, P.layout.total))
    P._k = P.layout.total // 2
    ids = torch.randint(0, 64, (2, 24))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    counter[0] = 0
    P.loss(batch).backward()
    return P.scores.grad.clone()


def test_checkpointed_score_gradient_matches(tmp_path):
    calls = [0]
    orig = ckpt.checkpoint

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    # HF binds `checkpoint` into a functools.partial at enable time, from modeling_utils'
    # own reference -- so that is the name to patch, before MaskedDelta re-enables it
    import transformers.modeling_utils as mu
    mu.checkpoint = counting
    try:
        g_plain = _score_grad(tmp_path, False, calls)
        assert calls[0] == 0
        g_ckpt = _score_grad(tmp_path, True, calls)
        assert calls[0] == 3, f"expected one checkpoint call per block, got {calls[0]}"
    finally:
        mu.checkpoint = orig
    assert g_plain.norm() > 0
    assert torch.allclose(g_plain, g_ckpt, rtol=1e-5, atol=1e-7), (g_plain - g_ckpt).abs().max()


def _grpo_score_grad(tmp_path, grad_ckpt, counter):
    """The forward `train/rl.py`'s ``fit_scores_grpo`` runs: compose, ``functional_call`` on a
    single sequence, log-prob of the completion, backward into the scores.

    Written out here rather than calling the real function because that one needs a reward model
    and a prompt split; what is being pinned is the CHECKPOINT plumbing, which is this forward.
    """
    from learning_to_attribute import build_mask
    from torch.func import functional_call

    from mask_learning_finetuning.train.params import MaskedDelta
    model = _tiny_llama()
    model.train()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = _cfg(tmp_path, grad_ckpt)
    torch.manual_seed(1)
    delta = {n: 0.05 * torch.randn_like(p) for n, p in model.named_parameters()
             if "self_attn" in n or "mlp" in n}
    P = MaskedDelta(model, cfg, init_delta=None, freeze_delta=True)
    with torch.no_grad():
        for n in P.deltas:
            P.deltas[n].copy_(delta[n])
        P.scores.copy_(torch.linspace(-1, 1, P.layout.total))
    if grad_ckpt and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # an INDEPENDENT base snapshot, as fit_scores_grpo takes
    base = {n: P.base[n].detach().clone() for n in P.layout.names}
    k = P.layout.total // 2
    mk = cfg.mask
    ids = torch.randint(0, 64, (1, 24))
    n_p = 12
    counter[0] = 0
    soft = build_mask(P.scores, k, mk.variant, T=mk.T, n_iters=mk.n_iters).mask
    from mask_learning_finetuning.masks import compose_params
    params = compose_params(base, P.deltas, soft, P.layout, invert=P.invert, aliases=P.aliases,
                            out_dtype=P.compose_dtype, svd=P.svd)
    out = functional_call(model, P.track_live({**params, **P.buffers}), args=(ids,))
    lp = out.logits[0, :-1].float().log_softmax(-1)
    tok = lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)
    (-tok.mean()).backward()
    return P.scores.grad.clone()


def test_grpo_forward_checkpointed_score_gradient_matches(tmp_path):
    """The GRPO path's own claim, and the one that was broken.

    `fit_scores_grpo` used to call `gradient_checkpointing_enable()` with no arguments, which
    replaced the recompute context `MaskedDelta.__init__` had installed, and it never registered
    the composed parameters -- so an 8B run died in backward with "A different number of tensors
    was saved during the original forward and recomputation" (40 against 31, job 276572). Both
    halves are pinned here: the checkpoint really fires, and the score gradient is unchanged.
    """
    calls = [0]
    orig = ckpt.checkpoint

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    import transformers.modeling_utils as mu
    mu.checkpoint = counting
    try:
        g_plain = _grpo_score_grad(tmp_path, False, calls)
        assert calls[0] == 0
        g_ckpt = _grpo_score_grad(tmp_path, True, calls)
        assert calls[0] == 3, f"expected one checkpoint call per block, got {calls[0]}"
    finally:
        mu.checkpoint = orig
    assert g_plain.norm() > 0
    assert torch.allclose(g_plain, g_ckpt, rtol=1e-5, atol=1e-7), (g_plain - g_ckpt).abs().max()


def test_generate_responses_restores_the_checkpoint_context():
    """HF sampling must not silently strip a custom `context_fn`.

    `generate_responses` turns checkpointing off to decode and back on afterwards. Re-enabling
    with no kwargs replaces the partial HF baked the caller's `use_reentrant`/`context_fn` into,
    which broke the 8B masked GRPO run (jobs 276572, 276791) -- and only that run, because a vLLM
    sampler never calls this function.
    """
    import contextlib

    from mask_learning_finetuning.eval.base import generate_responses

    def ctxs():
        return contextlib.nullcontext(), contextlib.nullcontext()

    model = _tiny_llama()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False, "context_fn": ctxs})
    before = next(m._gradient_checkpointing_func for m in model.modules()
                  if getattr(m, "_gradient_checkpointing_func", None) is not None)

    class _Tok:                      # enough of a tokenizer for the try/finally to run
        padding_side = "right"
        pad_token_id = 0

        def apply_chat_template(self, msgs, **kw):
            return msgs[0]["content"]

        def __call__(self, *a, **k):
            raise RuntimeError("stop before decoding -- the finally block is what is under test")

    try:
        generate_responses(model, _Tok(), ["hi"], device="cpu")
    except RuntimeError:
        pass
    after = next(m._gradient_checkpointing_func for m in model.modules()
                 if getattr(m, "_gradient_checkpointing_func", None) is not None)
    assert model.is_gradient_checkpointing
    assert after is before, "generate_responses replaced the caller's checkpoint function"
