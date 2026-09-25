"""Unit tests for ``restrict:`` -- full SFT confined to a saved mask's top-k units.

The claim this path makes is a *negative* one: everything outside the top-k does not move. That
is exactly the kind of claim a run cannot show you -- a loss curve looks the same whether the
freeze held or leaked, and the leak that matters most (AdamW's decoupled weight decay, which does
not go through the gradient and so is untouched by masking gradients) is a slow drift of every
"frozen" weight toward zero. So the freeze is tested by bit-comparison of the parameters across a
real optimizer step, not by eye.

Everything here runs on a four-parameter toy module, so there is no model download and no GPU:
``Restriction`` only needs ``named_parameters`` and ``config.hidden_size``, and the arithmetic
under test is the optimizer's, not the transformer's.

    uv run pytest tests/ -q
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mask_learning_finetuning.config import DataCfg, ExperimentConfig, RestrictCfg
from mask_learning_finetuning.masks import build_layout, save_checkpoint
from mask_learning_finetuning.train.params import Direct, Restricted
from mask_learning_finetuning.train.restrict import Restriction

HIDDEN = 4


class Tiny(nn.Module):
    """Four parameter tensors, one of them 1-D, so the layout has a no-decay group too."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN, _name_or_path="tiny")
        self.a = nn.Linear(HIDDEN, HIDDEN)          # a.weight [4, 4], a.bias [4]
        self.b = nn.Linear(HIDDEN, 8, bias=False)   # b.weight [8, 4]
        self.frozen = nn.Parameter(torch.ones(3))   # never in the mask's layout

    def forward(self, x):
        return self.b(self.a(x))


#: ``row`` units over Tiny, in ``named_parameters`` order and excluding ``frozen``:
#: a.weight 0-3, a.bias 4-7, b.weight 8-15. Scores are ``arange``, so the top-k are always the
#: LAST k units -- i.e. b.weight's bottom rows -- which is what every selection assertion reads.
TOTAL_UNITS = 16


def write_mask(tmp_path, *, unit="row", mode="necessary", exclude=("frozen",)):
    """A real checkpoint blob, written by the same function a training run writes."""
    named = [(n, p) for n, p in Tiny().named_parameters() if n not in exclude]
    layout = build_layout(named, unit, resid_dim=HIDDEN)
    path = tmp_path / "final.pt"
    save_checkpoint(path, args={"model": "tiny", "mode": mode, "unit": unit}, layout=layout,
                    scores=torch.arange(layout.total, dtype=torch.float32), deltas={},
                    train_log=[])
    return path, layout


def make_cfg(checkpoint, tmp_path, **restrict_kw):
    return ExperimentConfig(
        data=DataCfg(train="data/toy_chat.jsonl"), output=str(tmp_path / "run"), device="cpu",
        restrict=RestrictCfg(checkpoint=str(checkpoint), **restrict_kw))


def one_step(P, model, *, lr=0.1, seed=0):
    """One accumulated backward and one optimizer step, from a fixed input."""
    torch.manual_seed(seed)
    x = torch.randn(2, HIDDEN)
    P.zero_grad()
    (model(x).square().sum() + model.frozen.sum()).backward()
    gnorm = P.grad_norm()
    P.step(lr)
    return gnorm


# --------------------------------------------------------------------------- selection


def test_top_k_selection_and_what_it_freezes(tmp_path):
    """Ascending scores put the top 4 units on b.weight's last four rows, and only there."""
    path, _ = write_mask(tmp_path)
    model = Tiny()
    r = Restriction(model, RestrictCfg(checkpoint=str(path), k=4))

    assert set(r.masks) == {"b.weight"}
    assert r.masks["b.weight"].flatten().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # 4 rows x 4 columns of b.weight
    assert r.stats["trainable_params"] == 16
    assert r.stats["trainable_units"] == 4
    assert r.stats["fully_frozen_tensors"] == 2          # a.weight, a.bias
    assert r.stats["unscored_tensors"] == 1              # frozen
    assert r.stats["unscored_params"] == 3
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["b.weight"]


def test_frac_rounds_like_the_eval_grid(tmp_path):
    """``frac`` must name the same unit set the sparsity sweep's ``frac_*`` point names."""
    path, _ = write_mask(tmp_path)
    for frac, k in ((0.25, 4), (0.5, 8), (1.0, 16), (0.001, 1), (0.3, 5)):
        r = Restriction(Tiny(), RestrictCfg(checkpoint=str(path), frac=frac))
        assert r.k == k, frac


def test_invert_selects_the_complement(tmp_path):
    path, _ = write_mask(tmp_path)
    model = Tiny()
    r = Restriction(model, RestrictCfg(checkpoint=str(path), k=4, invert=True))

    assert set(r.masks) == {"a.weight", "a.bias", "b.weight"}
    assert r.masks["b.weight"].flatten().tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert r.stats["trainable_units"] == TOTAL_UNITS - 4
    assert r.stats["trainable_params"] == 4 * 4 + 4 + 4 * 4     # a.weight + a.bias + 4 b rows


def test_shuffle_is_a_deterministic_random_k_of_the_same_size(tmp_path):
    """The random-k control: same k, same layout, selection decided by the seed and not the
    ranking -- so it must differ from BOTH the top-k and the bottom-k (that is what makes it a
    control for their disagreement), reproduce bit-for-bit under one seed, and vary across
    seeds."""
    path, _ = write_mask(tmp_path)

    def units(**kw):
        r = Restriction(Tiny(), RestrictCfg(checkpoint=str(path), k=4, **kw))
        return {n: m.flatten().tolist() for n, m in r.masks.items()}, r.stats

    top, _ = units()
    bottom, _ = units(invert=True)
    rand1, stats = units(shuffle=1)
    rand1_again, _ = units(shuffle=1)
    rand2, _ = units(shuffle=2)

    assert stats["trainable_units"] == 4 and stats["shuffle"] == 1
    assert rand1 == rand1_again                       # deterministic under one seed
    assert rand1 != top and rand1 != bottom           # not either special population
    assert rand2 != rand1                             # and the seed is what decides it


def test_selecting_nothing_is_an_error(tmp_path):
    """``invert`` at frac 1.0 freezes the whole model; better a message than 'no trainable'."""
    path, _ = write_mask(tmp_path)
    with pytest.raises(SystemExit, match="selects no components"):
        Restriction(Tiny(), RestrictCfg(checkpoint=str(path), frac=1.0, invert=True))


def test_mask_from_another_model_is_rejected(tmp_path):
    """A layout naming tensors this model does not have is a wrong-checkpoint mistake."""
    other = nn.Linear(HIDDEN, HIDDEN)
    layout = build_layout(list(other.named_parameters()), "row")
    path = tmp_path / "final.pt"
    save_checkpoint(path, args={"model": "other"}, layout=layout,
                    scores=torch.zeros(layout.total), deltas={}, train_log=[])
    with pytest.raises(SystemExit, match="absent from this model"):
        Restriction(Tiny(), RestrictCfg(checkpoint=str(path), frac=0.5))


def test_iso_mask_warns_that_the_ranking_is_upside_down(tmp_path, caplog):
    """Under sufficient/iso the top-k are the units the finetune could do WITHOUT."""
    path, _ = write_mask(tmp_path, mode="sufficient")
    with caplog.at_level("WARNING"):
        Restriction(Tiny(), RestrictCfg(checkpoint=str(path), k=4))
    assert "mode=sufficient" in caplog.text


# --------------------------------------------------------------------------- the freeze


def test_gradients_are_masked_before_the_optimizer_sees_them(tmp_path):
    path, _ = write_mask(tmp_path)
    model = Tiny()
    P = Restricted(model, make_cfg(path, tmp_path, k=4))

    torch.manual_seed(0)
    model(torch.randn(2, HIDDEN)).square().sum().backward()
    g = model.b.weight.grad
    assert torch.count_nonzero(g[:4]) == 0            # frozen rows
    assert torch.count_nonzero(g[4:]) > 0             # selected rows
    assert model.a.weight.grad is None                # frozen tensor, never in the optimizer
    # the |g| the loop logs is taken before step(), so it must already be the restricted norm
    assert P.grad_norm() == pytest.approx(float(g[4:].norm()))


@pytest.mark.parametrize("wd", [0.0, 0.01, 0.5])
def test_frozen_components_do_not_move_at_all(tmp_path, wd):
    """The whole point, and the reason ``wd`` is parametrised.

    ``wd=0.5`` with ``lr=0.1`` is a 5% shrink per step -- far outside any tolerance -- so a
    weight-decay leak cannot hide in it. Masking gradients alone would fail this.
    """
    path, _ = write_mask(tmp_path)
    model = Tiny()
    cfg = make_cfg(path, tmp_path, k=4)
    cfg.train.weight_decay = wd
    P = Restricted(model, cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    for _ in range(3):
        one_step(P, model)

    for n, p in model.named_parameters():
        if n == "b.weight":
            assert torch.equal(p[:4], before[n][:4]), "frozen rows of a masked tensor moved"
            assert not torch.equal(p[4:], before[n][4:]), "selected rows did not train"
        else:
            assert torch.equal(p, before[n]), f"{n} moved despite being frozen"


def test_unrestricted_restriction_is_bit_identical_to_plain_sft(tmp_path):
    """``frac: 1.0`` is the control the experiment is read against, so it must not perturb.

    This is what the ``m * factor + (1 - m)`` spelling in ``restrict.masked_decay`` buys: the
    obvious ``1 - lr * wd * m`` can round apart from AdamW's own multiplier in the last place,
    which would make the control a slightly different run for no reason.
    """
    # every tensor in the layout, so an all-ones restriction and a plain run agree on the
    # trainable SET as well as on the arithmetic
    path, _ = write_mask(tmp_path, exclude=())
    plain_model = Tiny()
    restricted_model = copy.deepcopy(plain_model)

    plain_cfg = make_cfg(path, tmp_path, frac=1.0)
    plain_cfg.restrict = None
    plain = Direct(plain_model, plain_cfg)
    restricted = Restricted(restricted_model, make_cfg(path, tmp_path, frac=1.0))

    for i in range(3):
        one_step(plain, plain_model, seed=i)
        one_step(restricted, restricted_model, seed=i)

    for (n, a), (_, b) in zip(plain_model.named_parameters(),
                              restricted_model.named_parameters()):
        assert torch.equal(a, b), f"{n} diverged from the unrestricted run"


def test_optimizer_holds_no_state_for_fully_frozen_tensors(tmp_path):
    """A sparse restriction should not pay AdamW's two moment buffers for the whole model."""
    path, _ = write_mask(tmp_path)
    model = Tiny()
    P = Restricted(model, make_cfg(path, tmp_path, k=4))
    one_step(P, model)

    stateful = {id(p) for p in P.opt.state}
    assert stateful == {id(model.b.weight)}


# --------------------------------------------------------------------------- config surface


def _cfg(**restrict_kw):
    from mask_learning_finetuning.config import DataCfg
    return dict(data=DataCfg(train="x.jsonl"), output="out", device="cpu",
                restrict=RestrictCfg(**restrict_kw))


def test_config_rejects_the_ambiguous_and_the_impossible():
    from mask_learning_finetuning.config import LoraCfg, MaskCfg

    with pytest.raises(ValueError, match="exactly one of restrict.frac and restrict.k"):
        ExperimentConfig(**_cfg(checkpoint="m.pt"))
    with pytest.raises(ValueError, match="exactly one of restrict.frac and restrict.k"):
        ExperimentConfig(**_cfg(checkpoint="m.pt", frac=0.1, k=5))
    with pytest.raises(ValueError, match="restrict.checkpoint is required"):
        ExperimentConfig(**_cfg(frac=0.1))
    with pytest.raises(ValueError, match=r"restrict.frac must be in \(0, 1\]"):
        ExperimentConfig(**_cfg(checkpoint="m.pt", frac=1.5))
    with pytest.raises(ValueError, match="full-finetune only"):
        ExperimentConfig(lora=LoraCfg(), **_cfg(checkpoint="m.pt", frac=0.1))
    with pytest.raises(ValueError, match="full-finetune only"):
        ExperimentConfig(mask=MaskCfg(), **_cfg(checkpoint="m.pt", frac=0.1))


def test_config_accepts_the_two_valid_spellings():
    for kw in ({"frac": 0.01}, {"k": 100}, {"frac": 1.0, "invert": True}):
        cfg = ExperimentConfig(**_cfg(checkpoint="m.pt", **kw))
        assert cfg.restrict is not None
        assert cfg.lora is None and cfg.mask is None
