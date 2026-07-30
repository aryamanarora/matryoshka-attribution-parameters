"""The ``neuron_head`` unit mode: tied MLP neurons and per-head attention groups.

Exact claims, pinned rather than asserted in prose (the repo's test rule):

* the layout's arithmetic -- unit counts under GQA, the gate/up/down trio sharing one slice,
  ``sum(counts) > total`` being the *documented* consequence of the tie;
* ``expand_mask`` turning one grouped score into exactly one head's rows (or columns) and
  nothing else;
* composition: keeping one MLP neuron moves exactly ``gate[i,:]``, ``up[i,:]`` and
  ``down[:,i]``, and one score's gradient collects from all three tensors (the tie is real in
  the graph, not just in the bookkeeping);
* ``unit_delta_norms`` accumulating root-sum-square over a tied slice -- the failure mode it
  guards is three tensors overwriting each other, which reports whichever came last;
* the checkpoint round-trip preserving grouped axis codes, including through JSON's
  tuples-become-lists.

Shapes are a miniature GQA Llama: d_model 8, 4 heads / 2 kv heads of head_dim 2, d_ffn 12.
"""

import json

import pytest
import torch

from mask_learning_finetuning.masks import (
    AXIS_TENSOR, build_layout, compose_params, expand_mask, group_of, hard_topk_mask,
    layout_from_dict, layout_to_dict, unit_norms,
)
from mask_learning_finetuning.train.posthoc import unit_delta_norms

D, HD, N_KV, F, V = 8, 2, 2, 12, 10   # d_model, head_dim, kv heads, d_ffn, vocab


def fake_llama(n_layers=2, seed=0):
    """``[(name, tensor)]`` with Llama's names and a GQA shape set."""
    g = torch.Generator().manual_seed(seed)
    named = [("model.embed_tokens.weight", torch.randn(V, D, generator=g))]
    for L in range(n_layers):
        p = f"model.layers.{L}."
        named += [
            (p + "self_attn.q_proj.weight", torch.randn(D, D, generator=g)),
            (p + "self_attn.k_proj.weight", torch.randn(N_KV * HD, D, generator=g)),
            (p + "self_attn.v_proj.weight", torch.randn(N_KV * HD, D, generator=g)),
            (p + "self_attn.o_proj.weight", torch.randn(D, D, generator=g)),
            (p + "mlp.gate_proj.weight", torch.randn(F, D, generator=g)),
            (p + "mlp.up_proj.weight", torch.randn(F, D, generator=g)),
            (p + "mlp.down_proj.weight", torch.randn(D, F, generator=g)),
            (p + "input_layernorm.weight", torch.randn(D, generator=g)),
            (p + "post_attention_layernorm.weight", torch.randn(D, generator=g)),
        ]
    named.append(("model.norm.weight", torch.randn(D, generator=g)))
    return named


def layout_for(named):
    return build_layout(named, "neuron_head", resid_dim=D, head_dim=HD)


def by_name(layout):
    return {n: i for i, n in enumerate(layout.names)}


def test_counts_ties_and_axes():
    named = fake_llama(n_layers=2)
    layout = layout_for(named)
    idx = by_name(layout)

    # per layer: q 4 heads, k/v 2 each (GQA), o 4, MLP 12 tied neurons, 2 norms; plus embed
    # (10 rows) and the final norm
    per_layer = 4 + 2 + 2 + 4 + F + 1 + 1
    assert layout.total == V + 2 * per_layer + 1

    q = idx["model.layers.0.self_attn.q_proj.weight"]
    k = idx["model.layers.0.self_attn.k_proj.weight"]
    o = idx["model.layers.0.self_attn.o_proj.weight"]
    assert layout.axes[q] == ("group", 0, HD) and layout.counts[q] == 4
    assert layout.axes[k] == ("group", 0, HD) and layout.counts[k] == N_KV
    # o_proj is square, resolved by name to the OUT-projection rule: heads live on dim -1
    assert layout.axes[o] == ("group", -1, HD) and layout.counts[o] == 4
    assert group_of(layout.axes[q]) == (0, HD)

    # the MLP trio shares one slice: same offset, same count, per-tensor axes
    gate = idx["model.layers.0.mlp.gate_proj.weight"]
    up = idx["model.layers.0.mlp.up_proj.weight"]
    down = idx["model.layers.0.mlp.down_proj.weight"]
    assert layout.offsets[gate] == layout.offsets[up] == layout.offsets[down]
    assert layout.counts[gate] == layout.counts[up] == layout.counts[down] == F
    assert layout.axes[gate] == 0 and layout.axes[up] == 0 and layout.axes[down] == -1
    # ...and the two layers' MLPs do NOT share
    assert layout.offsets[idx["model.layers.1.mlp.gate_proj.weight"]] != layout.offsets[gate]

    # everything else takes the nonresid rule
    assert layout.axes[idx["model.embed_tokens.weight"]] == 0
    assert layout.axes[idx["model.norm.weight"]] is AXIS_TENSOR

    # the documented consequence of the tie: offsets overlap, so counts oversum the total
    assert sum(layout.counts) == layout.total + 2 * 2 * F
    # every unit is still covered exactly once by the union of slices
    covered = torch.zeros(layout.total)
    for i in range(len(layout.names)):
        covered[layout.slice_for(i)] = 1
    assert bool(covered.all())


def test_expand_mask_selects_one_head():
    m = torch.tensor([0.0, 1.0, 0.0, 0.0])          # keep head 1 of 4
    rows = expand_mask(m, (D, D), ("group", 0, HD)) * torch.ones(D, D)
    assert rows[2:4].eq(1).all() and rows[:2].eq(0).all() and rows[4:].eq(0).all()
    cols = expand_mask(m, (D, D), ("group", -1, HD)) * torch.ones(D, D)
    assert cols[:, 2:4].eq(1).all() and cols[:, :2].eq(0).all() and cols[:, 4:].eq(0).all()


def test_unit_norms_reduce_per_head():
    t = torch.randn(D, D)
    got = unit_norms(t, ("group", 0, HD))
    want = torch.stack([t[i * HD:(i + 1) * HD].norm() for i in range(D // HD)])
    assert torch.allclose(got, want)
    got = unit_norms(t, ("group", -1, HD))
    want = torch.stack([t[:, i * HD:(i + 1) * HD].norm() for i in range(D // HD)])
    assert torch.allclose(got, want)


def test_compose_keeps_exactly_one_neuron():
    named = fake_llama(n_layers=1)
    layout = layout_for(named)
    idx = by_name(layout)
    base = {n: t for n, t in named}
    g = torch.Generator().manual_seed(1)
    deltas = {n: torch.randn(t.shape, generator=g) for n, t in named}

    j = 5                                            # keep neuron 5 of layer 0's MLP, only
    gate = idx["model.layers.0.mlp.gate_proj.weight"]
    mask = torch.zeros(layout.total)
    mask[layout.offsets[gate] + j] = 1.0

    out = compose_params(base, deltas, mask, layout)
    for name in ("gate_proj", "up_proj", "down_proj"):
        full = f"model.layers.0.mlp.{name}.weight"
        moved = out[full] - base[full]
        if name == "down_proj":
            assert torch.allclose(moved[:, j], deltas[full][:, j])
            moved[:, j] = 0
        else:
            assert torch.allclose(moved[j], deltas[full][j])
            moved[j] = 0
        assert moved.abs().max() == 0, f"{name} moved outside neuron {j}"
    # nothing else moved at all
    for n in layout.names:
        if ".mlp." not in n:
            assert torch.equal(out[n], base[n]), n


def test_score_gradient_collects_from_all_three_tensors():
    named = fake_llama(n_layers=1)
    layout = layout_for(named)
    idx = by_name(layout)
    base = {n: t for n, t in named}
    deltas = {n: torch.ones_like(t) for n, t in named}

    scores = torch.zeros(layout.total, requires_grad=True)
    out = compose_params(base, deltas, torch.sigmoid(scores), layout)
    sum(v.sum() for v in out.values()).backward()

    gate = idx["model.layers.0.mlp.gate_proj.weight"]
    j = layout.offsets[gate] + 3
    # d(sum theta_eff)/dm at m=0.5 is 0.25 * (elements the unit governs): a neuron spans one
    # gate row + one up row + one down column = D + D + D elements of an all-ones delta
    assert torch.isclose(scores.grad[j], torch.tensor(0.25 * 3 * D))
    # an attention unit spans head_dim rows of one projection
    q = layout.offsets[idx["model.layers.0.self_attn.q_proj.weight"]]
    assert torch.isclose(scores.grad[q], torch.tensor(0.25 * HD * D))


def test_unit_delta_norms_accumulates_over_the_tie():
    named = fake_llama(n_layers=1)
    layout = layout_for(named)
    idx = by_name(layout)
    g = torch.Generator().manual_seed(2)
    deltas = {n: torch.randn(t.shape, generator=g) for n, t in named}

    dn = unit_delta_norms(deltas, layout)
    gate = idx["model.layers.0.mlp.gate_proj.weight"]
    j = 7
    want = (deltas["model.layers.0.mlp.gate_proj.weight"][j].norm() ** 2
            + deltas["model.layers.0.mlp.up_proj.weight"][j].norm() ** 2
            + deltas["model.layers.0.mlp.down_proj.weight"][:, j].norm() ** 2) ** 0.5
    assert torch.isclose(dn[layout.offsets[gate] + j], want, atol=1e-5)
    # an untied unit is the plain per-tensor norm it always was
    q = idx["model.layers.0.self_attn.q_proj.weight"]
    want_q = deltas["model.layers.0.self_attn.q_proj.weight"][:HD].norm()
    assert torch.isclose(dn[layout.offsets[q]], want_q, atol=1e-5)


def test_hard_topk_still_counts_units():
    layout = layout_for(fake_llama(n_layers=1))
    scores = torch.arange(layout.total, dtype=torch.float)
    m = hard_topk_mask(scores, 5)
    assert int(m.sum()) == 5 and bool(m[-5:].all())


def test_checkpoint_roundtrip_preserves_group_axes_and_ties():
    layout = layout_for(fake_llama(n_layers=2))
    d = layout_to_dict(layout)
    back = layout_from_dict(d)
    assert back.axes == layout.axes and back.offsets == layout.offsets
    assert back.total == layout.total
    # JSON turns tuples into lists; the loader must fold them back
    back = layout_from_dict(json.loads(json.dumps(d)))
    assert back.axes == layout.axes
    assert group_of(back.axes[1]) == (0, HD)


def test_missing_or_wrong_head_dim_fails_loudly():
    named = fake_llama(n_layers=1)
    with pytest.raises(ValueError, match="head_dim"):
        build_layout(named, "neuron_head", resid_dim=D)
    with pytest.raises(ValueError, match="divisible"):
        build_layout(named, "neuron_head", resid_dim=D, head_dim=3)
