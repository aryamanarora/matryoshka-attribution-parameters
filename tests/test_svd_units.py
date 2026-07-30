"""The singular-direction unit modes (``masks/svd.py``, ``unit: svd|svd_attn|svd_mlp``).

Here for the reason the rest of ``tests/`` is: what is pinned are the *exact* claims. The
composition ``theta_eff = theta_base + U diag(m . S) Vh`` is arithmetic, so "an all-ones mask
reconstructs the delta exactly" and "the functional and in-place paths compose the same weights"
are equalities and not tolerances-by-taste. Two of them are the ones that would be believed if
they were wrong:

* **the anchors.** If a truncated factorisation loses part of the delta, ``full_delta`` stops
  being the finetune, every normalised number in a sweep is divided by the wrong ceiling, and the
  curve still looks perfectly reasonable. Hence the exactness tests and
  ``build_factors``' hard error.
* **the partition.** A hybrid mode that factored the wrong sublayer -- or both, or neither --
  would produce a valid layout with a plausible unit count, and the run would report a curve for
  an experiment nobody asked for. So the tests assert the split by NAME against the model's own
  parameter names, not just by count.
"""

import pytest
import torch
from torch import nn

from mask_learning_finetuning.masks import (
    AXIS_SVD, SVD_MODES, apply_in_place, build_layout, compose_params, expand_mask,
    hard_topk_mask, is_attn_param, is_mlp_param, layout_to_dict, mask_for, unit_view, wants_svd,
)
from mask_learning_finetuning.masks.checkpoint import layout_from_dict
from mask_learning_finetuning.masks.svd import build_factors, factor, factors_for_layout
from mask_learning_finetuning.train.posthoc import dead_units, unit_delta_norms

D, H, RANK = 8, 16, 3


class _Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


class _Tiny(nn.Module):
    """Llama-shaped parameter NAMES over toy shapes -- the layouts key off the names."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {"q_proj": nn.Linear(D, D, bias=False), "o_proj": nn.Linear(D, D, bias=False)})
        self.mlp = nn.ModuleDict(
            {"gate_proj": nn.Linear(D, H, bias=False), "down_proj": nn.Linear(H, D, bias=False)})
        self.input_layernorm = _Norm(D)


def _model():
    torch.manual_seed(0)
    return _Tiny()


def _low_rank_deltas(named, rank=RANK, seed=1):
    """A delta of exactly ``rank`` per 2-D tensor -- the shape a merged LoRA delta has."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, p in named:
        if p.ndim == 2:
            m, k = p.shape
            out[n] = (torch.randn(m, rank, generator=g)
                      @ torch.randn(rank, k, generator=g)) / rank
        else:
            out[n] = torch.randn(p.shape, generator=g) * 0.1
    return out


def _setup(mode, *, rank=RANK, cap=RANK, deltas=None):
    model = _model()
    named = list(model.named_parameters())
    dense = deltas if deltas is not None else _low_rank_deltas(named, rank=rank)
    wanted = [n for n, p in named if wants_svd(n, tuple(p.shape), mode)]
    svd, stats = build_factors(dense, wanted, rank=cap, tol=1e-9)
    layout = build_layout(named, mode, resid_dim=D, ranks={n: f.rank for n, f in svd.items()})
    thin = {n: d for n, d in dense.items() if n not in svd}
    return model, layout, dense, thin, svd, stats


# --------------------------------------------------------------------------- the name predicates

def test_gpt2_c_proj_is_disambiguated_by_its_module():
    # gpt2 names BOTH output projections c_proj; only the enclosing module separates them, and a
    # hybrid mode that got this wrong would factor half of each sublayer
    assert is_mlp_param("h.0.mlp.c_proj.weight")
    assert not is_attn_param("h.0.mlp.c_proj.weight")
    assert is_attn_param("h.0.attn.c_proj.weight")
    assert not is_mlp_param("h.0.attn.c_proj.weight")


@pytest.mark.parametrize("name", ["model.layers.0.self_attn.q_proj.weight",
                                  "model.layers.3.self_attn.o_proj.weight"])
def test_llama_attention_names(name):
    assert is_attn_param(name) and not is_mlp_param(name)


@pytest.mark.parametrize("name", ["model.layers.0.mlp.gate_proj.weight",
                                  "model.layers.0.mlp.down_proj.weight"])
def test_llama_mlp_names(name):
    assert is_mlp_param(name) and not is_attn_param(name)


def test_a_norm_gain_is_never_factored():
    # 1-D: no singular value decomposition exists, so every svd mode falls back to nonresid
    for mode in SVD_MODES:
        assert not wants_svd("model.norm.weight", (D,), mode)


# ------------------------------------------------------------------------------------ the layout

def test_svd_layout_unit_counts_are_the_ranks():
    _, layout, _, _, svd, _ = _setup("svd")
    assert layout.total == sum(f.rank for f in svd.values()) + 1     # +1: the norm gain's tensor
    for i, name in enumerate(layout.names):
        if name in svd:
            assert layout.axes[i] == AXIS_SVD
            assert layout.counts[i] == svd[name].rank == RANK
    assert layout.ranks() == {n: RANK for n in svd}


def test_hybrid_modes_partition_the_tensors_by_name():
    _, attn_layout, _, _, attn_svd, _ = _setup("svd_attn")
    _, mlp_layout, _, _, mlp_svd, _ = _setup("svd_mlp")
    assert set(attn_svd) == {"self_attn.q_proj.weight", "self_attn.o_proj.weight"}
    assert set(mlp_svd) == {"mlp.gate_proj.weight", "mlp.down_proj.weight"}
    # every tensor is in exactly one half of each hybrid mode, and no tensor is dropped
    for layout, svd in ((attn_layout, attn_svd), (mlp_layout, mlp_svd)):
        assert set(layout.names) == set(n for n, _ in _model().named_parameters())
        assert layout.total == sum(layout.counts)
        assert set(layout.svd_names) == set(svd)
    # ...and the two halves are complements of one another over the 2-D tensors
    two_d = {n for n, p in _model().named_parameters() if p.ndim == 2}
    assert set(attn_svd) | set(mlp_svd) == two_d
    assert not set(attn_svd) & set(mlp_svd)


def test_hybrid_unfactored_half_keeps_the_nonresid_axes():
    # svd_mlp leaves attention on nonresid units, and o_proj must still be scored along dim -1 --
    # the square-tensor case masks/layout resolves by name
    _, layout, _, _, _, _ = _setup("svd_mlp")
    axes = dict(zip(layout.names, layout.axes))
    assert axes["self_attn.q_proj.weight"] == 0
    assert axes["self_attn.o_proj.weight"] == -1
    assert axes["input_layernorm.weight"] is None


def test_a_svd_layout_round_trips_through_a_checkpoint_dict():
    _, layout, _, _, _, _ = _setup("svd_attn")
    back = layout_from_dict(layout_to_dict(layout))
    assert (back.mode, back.axes, back.counts, back.total) == (
        layout.mode, layout.axes, layout.counts, layout.total)
    assert back.ranks() == layout.ranks()


def test_a_svd_layout_cannot_be_rebuilt_without_its_axes():
    _, layout, _, _, _, _ = _setup("svd")
    d = layout_to_dict(layout)
    d["axes"] = None
    with pytest.raises(ValueError, match="explicit `axes`"):
        layout_from_dict(d)


# ------------------------------------------------------------------- the reductions that CANNOT

def test_expand_mask_refuses_a_singular_direction():
    with pytest.raises(ValueError, match="does not broadcast"):
        expand_mask(torch.ones(RANK), (D, D), AXIS_SVD)


def test_unit_view_refuses_a_singular_direction():
    with pytest.raises(ValueError, match="not a slice of the parameter"):
        unit_view(torch.zeros(D, D), AXIS_SVD)


# ------------------------------------------------------------------------- the factorisation

def test_the_factorisation_of_a_low_rank_delta_is_exact():
    d = _low_rank_deltas([("w", torch.zeros(H, D))], rank=RANK)["w"]
    f = factor(d, rank=RANK, tol=1e-9, method="full")
    assert f.rank == RANK
    assert torch.allclose(f.delta(), d, atol=1e-5)
    assert f.rel_error(d) < 1e-6


def test_lowrank_and_full_agree_on_an_exactly_low_rank_delta():
    # the randomised range finder is EXACT once the sketch spans the delta's column space, which
    # is what makes `svd_rank: 32` over a LoRA-r32 delta a free speedup rather than a compromise
    big = _low_rank_deltas([("w", torch.zeros(200, 60))], rank=4, seed=7)["w"]
    full = factor(big, rank=4, tol=1e-9, method="full")
    low = factor(big, rank=4, tol=1e-9, method="lowrank")
    assert torch.allclose(full.S, low.S, atol=1e-4)
    assert torch.allclose(full.delta(), low.delta(), atol=1e-4)


def test_singular_values_come_out_descending():
    f = factor(_low_rank_deltas([("w", torch.zeros(H, D))], rank=RANK)["w"], rank=RANK)
    assert torch.all(f.S[:-1] >= f.S[1:])


def test_tolerance_drops_the_numerically_zero_tail():
    d = _low_rank_deltas([("w", torch.zeros(H, D))], rank=2)["w"]
    assert factor(d, rank=None, tol=1e-6).rank == 2         # not min(H, D) == 8


def test_an_exactly_zero_delta_keeps_one_dead_unit():
    # zero units would drop the tensor out of the layout entirely; a dead unit is the honest
    # report, and posthoc.dead_units is what counts it
    f = factor(torch.zeros(H, D), rank=RANK)
    assert f.rank == 1 and float(f.S[0]) == 0.0


def test_too_small_a_rank_cap_is_a_hard_error():
    dense = _low_rank_deltas(list(_model().named_parameters()), rank=6)
    with pytest.raises(SystemExit, match="loses"):
        build_factors(dense, ["mlp.gate_proj.weight"], rank=1, tol=0.0, check_tol=0.01)


def test_a_mode_that_factors_nothing_is_a_hard_error():
    with pytest.raises(ValueError, match="factors no tensors"):
        build_factors({}, [], rank=RANK)


def test_build_factors_reports_the_error_it_measured():
    _, _, _, _, _, stats = _setup("svd")
    assert stats["svd_tensors"] == 4 and stats["svd_units"] == 4 * RANK
    assert stats["svd_rel_error_max"] < 1e-5
    assert stats["svd_rank_min"] == stats["svd_rank_max"] == RANK


# ---------------------------------------------------------------------------------- composition

def test_an_all_ones_mask_reconstructs_the_whole_delta():
    model, layout, dense, thin, svd, _ = _setup("svd")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = compose_params(base, thin, torch.ones(layout.total), layout, svd=svd)
    for n in layout.names:
        assert torch.allclose(out[n], base[n] + dense[n], atol=1e-5), n


def test_an_all_zeros_mask_is_the_pretrained_model_exactly():
    model, layout, _, thin, svd, _ = _setup("svd")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = compose_params(base, thin, torch.zeros(layout.total), layout, svd=svd)
    for n in layout.names:
        assert torch.equal(out[n], base[n]), n


def test_top_k_keeps_exactly_the_k_highest_scored_directions():
    model, layout, _, thin, svd, _ = _setup("svd")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.manual_seed(3)
    scores = torch.randn(layout.total)
    k = 5
    mask = mask_for(k, layout, scores)
    assert int(mask.sum()) == k
    out = compose_params(base, thin, mask, layout, svd=svd)
    for i, name in enumerate(layout.names):
        if name not in svd:
            continue
        m = mask[layout.slice_for(i)]
        f = svd[name]
        expect = base[name] + (f.U * (m * f.S)) @ f.Vh
        assert torch.allclose(out[name], expect, atol=1e-6), name


def test_invert_composes_the_complement_of_the_top_k():
    model, layout, dense, thin, svd, _ = _setup("svd")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.manual_seed(4)
    scores = torch.randn(layout.total)
    mask = mask_for(4, layout, scores)
    keep = compose_params(base, thin, mask, layout, svd=svd)
    drop = compose_params(base, thin, mask, layout, svd=svd, invert=True)
    for n in layout.names:
        # the two halves of one delta: keeping S and keeping its complement sum back to the whole
        assert torch.allclose(keep[n] + drop[n] - base[n], base[n] + dense[n], atol=1e-5), n


def test_the_two_composition_paths_agree():
    # the invariant masks/compose.py exists to hold: training (functional) and any eval that
    # generates (in place) must compose the same weights for the same (mask, delta)
    model, layout, _, thin, svd, _ = _setup("svd_attn")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.manual_seed(5)
    scores = torch.randn(layout.total)
    mask = hard_topk_mask(scores, 7)
    functional = compose_params(base, thin, mask, layout, svd=svd)
    apply_in_place(model, base, thin, mask, layout, svd=svd)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), functional[n]), n


def test_composition_is_missing_factors_loudly():
    model, layout, _, thin, svd, _ = _setup("svd")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    with pytest.raises(KeyError, match="no factors were supplied"):
        compose_params(base, thin, torch.ones(layout.total), layout, svd={})


def test_the_mask_carries_gradient_to_the_scores():
    # dL/ds must flow through the reconstruction, or nothing is learned and the run looks healthy
    model, layout, _, thin, svd, _ = _setup("svd")
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    m = torch.zeros(layout.total, requires_grad=True)
    out = compose_params(base, thin, m, layout, svd=svd)
    sum(v.sum() for v in out.values()).backward()
    assert m.grad is not None and float(m.grad.abs().sum()) > 0


# ---------------------------------------------------------------------- the per-unit quantities

def test_a_directions_delta_norm_is_its_singular_value():
    _, layout, _, thin, svd, _ = _setup("svd")
    norms = unit_delta_norms(thin, layout, svd)
    for i, name in enumerate(layout.names):
        if name in svd:
            assert torch.allclose(norms[layout.slice_for(i)], svd[name].S, atol=1e-6), name
    assert dead_units(thin, layout, svd) == 0


def test_a_zero_delta_direction_counts_as_dead():
    named = list(_model().named_parameters())
    dense = _low_rank_deltas(named)
    dense["mlp.gate_proj.weight"] = torch.zeros(H, D)
    _, layout, _, thin, svd, _ = _setup("svd", deltas=dense)
    assert svd["mlp.gate_proj.weight"].rank == 1
    assert dead_units(thin, layout, svd) == 1


def test_attribution_matches_the_rank_one_slice_it_stands_for():
    # unit i's slice of the delta is S[i] u_i v_i^T, so its first-order term is the inner product
    # of THAT matrix with the gradient -- computed in closed form, checked against the explicit sum
    f = factor(_low_rank_deltas([("w", torch.zeros(H, D))], rank=RANK)["w"], rank=RANK)
    torch.manual_seed(6)
    g = torch.randn(H, D)
    got = f.attribution(g)
    for i in range(f.rank):
        slice_i = float(f.S[i]) * torch.outer(f.U[:, i], f.Vh[i])
        assert torch.allclose(got[i], (slice_i * g).sum(), atol=1e-4)


def test_attribution_is_signed():
    f = factor(_low_rank_deltas([("w", torch.zeros(H, D))], rank=RANK)["w"], rank=RANK)
    g = -f.delta()                      # a gradient anti-aligned with the whole delta
    assert torch.all(f.attribution(g) < 0)


# ------------------------------------------------------------------------------- re-derivation

def test_factors_for_layout_reproduces_the_runs_own_ranks():
    # the post-hoc eval CLI rebuilds a delta that was not persisted; re-deciding the rank there
    # would shift what every score index means
    _, layout, dense, _, svd, _ = _setup("svd_mlp")
    again = factors_for_layout(dense, layout)
    assert set(again) == set(svd)
    for n in svd:
        assert again[n].rank == svd[n].rank
        assert torch.allclose(again[n].S, svd[n].S, atol=1e-5), n
        assert torch.allclose(again[n].delta(), svd[n].delta(), atol=1e-5), n


def test_factors_for_layout_rejects_a_mismatched_delta():
    _, layout, dense, _, _, _ = _setup("svd")
    dense = dict(dense, **{"mlp.gate_proj.weight": torch.zeros(H, D)})
    with pytest.raises(SystemExit, match="not the one the run was fitted over"):
        factors_for_layout(dense, layout)


# ------------------------------------------------------------------ what the config must refuse

def _cfg(**mask_kw):
    from mask_learning_finetuning.config.schema import DataCfg, ExperimentConfig, MaskCfg
    return ExperimentConfig(data=DataCfg(train="data/toy_chat.jsonl"), output="/tmp/x",
                            device="cpu", mask=MaskCfg(**mask_kw))


@pytest.mark.parametrize("mode", SVD_MODES)
def test_a_svd_mode_needs_a_frozen_given_delta(mode):
    # THE misconfiguration this family has: the factorisation happens once, so a delta that
    # trains from zero would have different singular directions at every step and score i would
    # not refer to the same object twice. Nothing in a log line would show it.
    with pytest.raises(ValueError, match="singular directions of the delta"):
        _cfg(unit=mode)


def test_a_svd_mode_over_an_init_delta_needs_freeze_delta():
    with pytest.raises(ValueError, match="needs mask.freeze_delta"):
        _cfg(unit="svd", init_delta="/tmp/d.pt")
    _cfg(unit="svd", init_delta="/tmp/d.pt", freeze_delta=True)      # accepted


def test_mask_finetuned_is_enough_for_a_svd_mode():
    cfg = _cfg(unit="svd_attn", finetuned="/tmp/adapter")
    assert cfg.mask.freeze_delta is True            # implied, as for every post-hoc run


def test_an_unknown_unit_mode_is_rejected():
    with pytest.raises(ValueError, match="mask.unit must be one of"):
        _cfg(unit="svd_mpl", finetuned="/tmp/adapter")


@pytest.mark.parametrize("kw,match", [
    (dict(svd_rank=0), "svd_rank must be at least 1"),
    (dict(svd_tol=1.0), "svd_tol must be in"),
    (dict(svd_method="randomised"), "svd_method must be"),
])
def test_svd_knobs_are_validated(kw, match):
    with pytest.raises(ValueError, match=match):
        _cfg(unit="svd", finetuned="/tmp/adapter", **kw)


def test_restrict_refuses_a_svd_checkpoint(tmp_path):
    # `restrict:` freezes parameter components, and a singular direction is not a set of
    # components -- the plausible reading of the request is a much weaker claim than the one
    # restrict: exists to test, and would be reported under the same name
    from types import SimpleNamespace

    from mask_learning_finetuning.masks import save_checkpoint
    from mask_learning_finetuning.train.restrict import Restriction

    model = _model()
    model.config = SimpleNamespace(hidden_size=D, _name_or_path="tiny")
    _, layout, _, _, svd, _ = _setup("svd")
    path = tmp_path / "final.pt"
    save_checkpoint(path, args={"model": "tiny", "mode": "necessary", "unit": "svd"},
                    layout=layout, scores=torch.arange(layout.total, dtype=torch.float32),
                    deltas={}, train_log=[], svd=svd)
    with pytest.raises(SystemExit, match="singular directions of the delta"):
        Restriction(model, SimpleNamespace(checkpoint=str(path), frac=0.5, k=None, invert=False))
