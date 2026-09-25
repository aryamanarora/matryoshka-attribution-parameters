"""The ``head`` unit mode: ``neuron_head`` with one attention module's projections tied per head.

Exact claims, in the repo's style:

* the layout arithmetic under GQA -- a layer contributes ``n_heads`` query-head units (q and o
  tied) plus ``n_kv_heads`` kv-group units (k and v tied), never four separate slices;
* under MHA (``n_kv == n_heads``) all four projections tie into one unit per head;
* composition: keeping one head unit moves exactly that head's rows of q and columns of o, and
  nothing of any other head; keeping one kv unit moves exactly that group's rows of k and v;
* the score gradient of one head unit collects from BOTH tied tensors;
* ``unit_delta_norms`` is the root-sum-square over the tie;
* the MLP tie is unchanged from ``neuron_head``.
"""

import torch

from mask_learning_finetuning.masks import (
    build_layout, compose_params, group_of, hard_topk_mask,
)
from mask_learning_finetuning.train.posthoc import unit_delta_norms

D, HD, NH, F, V = 8, 2, 4, 12, 10   # d_model, head_dim, heads, d_ffn, vocab


def fake_llama(n_kv, n_layers=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    named = [("model.embed_tokens.weight", torch.randn(V, D, generator=g))]
    for L in range(n_layers):
        p = f"model.layers.{L}."
        named += [
            (p + "self_attn.q_proj.weight", torch.randn(D, D, generator=g)),
            (p + "self_attn.k_proj.weight", torch.randn(n_kv * HD, D, generator=g)),
            (p + "self_attn.v_proj.weight", torch.randn(n_kv * HD, D, generator=g)),
            (p + "self_attn.o_proj.weight", torch.randn(D, D, generator=g)),
            (p + "mlp.gate_proj.weight", torch.randn(F, D, generator=g)),
            (p + "mlp.up_proj.weight", torch.randn(F, D, generator=g)),
            (p + "mlp.down_proj.weight", torch.randn(D, F, generator=g)),
            (p + "input_layernorm.weight", torch.randn(D, generator=g)),
            (p + "post_attention_layernorm.weight", torch.randn(D, generator=g)),
        ]
    named.append(("model.norm.weight", torch.randn(D, generator=g)))
    return named


def idx_of(layout):
    return {n: i for i, n in enumerate(layout.names)}


def test_gqa_counts_and_ties():
    named = fake_llama(n_kv=2)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    i = idx_of(lay)
    q, k, v, o = (i[f"model.layers.0.self_attn.{p}_proj.weight"] for p in "qkvo")
    assert lay.offsets[q] == lay.offsets[o] and lay.counts[q] == lay.counts[o] == NH
    assert lay.offsets[k] == lay.offsets[v] and lay.counts[k] == lay.counts[v] == 2
    assert lay.offsets[q] != lay.offsets[k]
    assert group_of(lay.axes[q]) == (0, HD) and group_of(lay.axes[o]) == (-1, HD)
    # per layer: 4 head units + 2 kv units + 12 neurons + 2 norms; plus embed rows + final norm
    assert lay.total == V + 2 * (NH + 2 + F + 2) + 1
    assert sum(lay.counts) > lay.total
    # the neuron tie is untouched
    g, u, d = (i[f"model.layers.0.mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
    assert lay.offsets[g] == lay.offsets[u] == lay.offsets[d]


def test_mha_ties_all_four():
    named = fake_llama(n_kv=NH)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    i = idx_of(lay)
    offs = {lay.offsets[i[f"model.layers.1.self_attn.{p}_proj.weight"]] for p in "qkvo"}
    assert len(offs) == 1
    assert lay.total == V + 2 * (NH + F + 2) + 1


def test_compose_moves_exactly_one_head():
    named = fake_llama(n_kv=2)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    i = idx_of(lay)
    base = dict(named)
    deltas = {n: torch.ones_like(t) for n, t in named}
    h = 2
    m = torch.zeros(lay.total)
    m[lay.offsets[i["model.layers.0.self_attn.q_proj.weight"]] + h] = 1.0
    out = compose_params(base, deltas, m, lay, invert=False)
    dq = out["model.layers.0.self_attn.q_proj.weight"] - base["model.layers.0.self_attn.q_proj.weight"]
    do = out["model.layers.0.self_attn.o_proj.weight"] - base["model.layers.0.self_attn.o_proj.weight"]
    rows = torch.zeros(D, dtype=torch.bool)
    rows[h * HD:(h + 1) * HD] = True
    # allclose, not eq: (base + 1.0) - base is float arithmetic and need not be exactly 1
    assert torch.allclose(dq[rows], torch.ones_like(dq[rows]), atol=1e-5) and bool(dq[~rows].eq(0).all())
    assert torch.allclose(do[:, rows], torch.ones_like(do[:, rows]), atol=1e-5) \
        and bool(do[:, ~rows].eq(0).all())
    for n in ("k_proj", "v_proj"):
        t = f"model.layers.0.self_attn.{n}.weight"
        assert bool((out[t] - base[t]).eq(0).all())
    for n, t in named:
        if "layers.0.self_attn.q_proj" in n or "layers.0.self_attn.o_proj" in n:
            continue
        assert bool((out[n] - base[n]).eq(0).all()), n


def test_kv_unit_moves_one_group():
    named = fake_llama(n_kv=2)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    i = idx_of(lay)
    base = dict(named)
    deltas = {n: torch.ones_like(t) for n, t in named}
    m = torch.zeros(lay.total)
    m[lay.offsets[i["model.layers.1.self_attn.k_proj.weight"]] + 1] = 1.0
    out = compose_params(base, deltas, m, lay, invert=False)
    for n in ("k_proj", "v_proj"):
        t = f"model.layers.1.self_attn.{n}.weight"
        d = out[t] - base[t]
        assert torch.allclose(d[HD:2 * HD], torch.ones_like(d[HD:2 * HD]), atol=1e-5) \
            and bool(d[:HD].eq(0).all())
    for n in ("q_proj", "o_proj"):
        t = f"model.layers.1.self_attn.{n}.weight"
        assert bool((out[t] - base[t]).eq(0).all())


def test_score_gradient_collects_from_both_tied_tensors():
    named = fake_llama(n_kv=2)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    i = idx_of(lay)
    base = dict(named)
    deltas = {n: torch.ones_like(t) for n, t in named}
    scores = torch.zeros(lay.total, requires_grad=True)
    out = compose_params(base, deltas, scores, lay, invert=False)
    # a loss that touches q only, then one that touches o only: the same score gets both
    u = lay.offsets[i["model.layers.0.self_attn.q_proj.weight"]] + 1
    out["model.layers.0.self_attn.q_proj.weight"].sum().backward(retain_graph=True)
    gq = scores.grad[u].item()
    scores.grad = None
    out["model.layers.0.self_attn.o_proj.weight"].sum().backward()
    go = scores.grad[u].item()
    assert gq == HD * D and go == HD * D


def test_unit_delta_norms_root_sum_square_over_tie():
    named = fake_llama(n_kv=2)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    i = idx_of(lay)
    deltas = {n: torch.zeros_like(t) for n, t in named}
    deltas["model.layers.0.self_attn.q_proj.weight"][0:HD] = 3.0
    deltas["model.layers.0.self_attn.o_proj.weight"][:, 0:HD] = 4.0
    norms = unit_delta_norms(deltas, lay)
    u = lay.offsets[i["model.layers.0.self_attn.q_proj.weight"]]
    assert abs(norms[u].item() - (9 * HD * D + 16 * HD * D) ** 0.5) < 1e-5
    assert norms[u + 1].item() == 0.0


def test_hard_topk_counts_units_not_tensors():
    named = fake_llama(n_kv=2)
    lay = build_layout(named, "head", resid_dim=D, head_dim=HD)
    m = hard_topk_mask(torch.arange(lay.total).float(), 3)
    assert int(m.sum()) == 3
