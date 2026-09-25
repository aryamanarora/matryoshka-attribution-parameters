"""Do the singular-direction unit modes mask the thing they claim to? CPU, seconds, no download.

``masks/svd.py`` replaces the basis a mask is defined in: instead of keeping rows of a parameter
it keeps directions of the delta, ``theta_eff = theta_base + U diag(m . S) Vh``. Three of its
claims cannot be checked by a config file and each fails quietly:

1. **The anchors are still the anchors.** An all-ones mask must reproduce the finetuned model
   *exactly* and an all-zeros mask the pretrained one. If the truncation lost part of the delta,
   ``full_delta`` silently stops being the finetune -- and every normalised number in a sparsity
   sweep is a ratio against that ceiling, so the curve stays plausible while measuring nothing.
2. **The ranking is the right way up.** Same hazard ``verify_ixg.py`` exists for, in the new
   basis: keeping the top-k directions must beat keeping the bottom-k. Get the indexing or the
   sign wrong and every sparsity point is the *worst* available subset, which reads exactly like
   "this behaviour is not localised".
3. **The hybrid modes split what they say they split.** ``svd_attn`` must factor the attention
   projections and leave the MLP on nonresid units, and ``svd_mlp`` the reverse. Both produce a
   valid layout with a plausible unit count either way round.

Checked on a real (tiny) Llama with a real low-rank delta, through the same functions a run uses,
so this covers the layout, both composition paths, the checkpoint round-trip and IxG together.
Unit-level claims are in ``tests/test_svd_units.py``; this is the integration check.

    uv run python scripts/verify/verify_svd.py
"""

import logging

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from mask_learning_finetuning.masks import (
    AXIS_SVD, apply_in_place, build_layout, compose_params, is_attn_param, is_mlp_param,
    layout_to_dict, mask_for, wants_svd,
)
from mask_learning_finetuning.masks.checkpoint import layout_from_dict
from mask_learning_finetuning.masks.svd import build_factors, factors_for_layout, from_blob, to_blob
from mask_learning_finetuning.train.ixg import ixg_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("verify_svd")

FRACS = (0.05, 0.1, 0.25, 0.5)
LORA_RANK = 4                 # the delta's true rank per tensor, as a LoRA adapter's merge gives
EXCLUDE = ("embed_tokens", "lm_head", "norm")


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    return AutoModelForCausalLM.from_config(cfg).eval()


def batch(n=4, t=16, vocab=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, t), generator=g)
    labels = ids.clone()
    labels[:, :4] = -100
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def token_ce(model, b):
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])
    tgt = b["labels"][:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(logits.float(), tgt, ignore_index=-100,
                                             reduction="sum")


def named_scored(model):
    return [(n, p) for n, p in model.named_parameters()
            if not any(x in n for x in EXCLUDE)]


def low_rank_delta(named, rank=LORA_RANK, seed=2, scale=0.05):
    """A rank-``rank`` delta per 2-D tensor -- the shape ``merge_and_unload`` produces."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, p in named:
        if p.ndim == 2:
            m, k = p.shape
            out[n] = scale * (torch.randn(m, rank, generator=g)
                              @ torch.randn(rank, k, generator=g)) / rank
        else:
            out[n] = scale * torch.randn(p.shape, generator=g)
    return out


def setup(model, mode, dense, *, cap=LORA_RANK):
    named = named_scored(model)
    wanted = [n for n, p in named if wants_svd(n, tuple(p.shape), mode)]
    svd, stats = build_factors(dense, wanted, rank=cap, tol=1e-9)
    layout = build_layout(named, mode, resid_dim=model.config.hidden_size,
                          ranks={n: f.rank for n, f in svd.items()})
    thin = {n: d for n, d in dense.items() if n not in svd}
    return layout, thin, svd, stats


def loss_at(model, base, thin, svd, layout, mask, buffers):
    from torch.func import functional_call
    params = compose_params(base, thin, mask, layout, invert=False, svd=svd)
    b = batch()
    with torch.no_grad():
        out = functional_call(model, {**params, **buffers}, args=(b["input_ids"],),
                              kwargs={"attention_mask": b["attention_mask"]})
        logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])
        tgt = b["labels"][:, 1:].reshape(-1)
        return float(torch.nn.functional.cross_entropy(
            logits.float(), tgt, ignore_index=-100, reduction="sum"))


def main():
    model = tiny_model()
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    named = named_scored(model)
    dense = low_rank_delta(named)

    # ---- 1. the layouts, and the partition the hybrid modes claim ------------------------------
    layouts = {}
    for mode in ("svd", "svd_attn", "svd_mlp", "nonresid"):
        if mode == "nonresid":
            layout = build_layout(named, mode, resid_dim=model.config.hidden_size)
            thin, svd, stats = dense, {}, {}
        else:
            layout, thin, svd, stats = setup(model, mode, dense)
        layouts[mode] = (layout, thin, svd)
        logger.info("%-10s %s", mode, layout.summary())
        factored = set(layout.svd_names)
        two_d = {n for n, p in named if p.ndim == 2}
        if mode == "svd":
            assert factored == two_d, "svd must factor every 2-D scored tensor"
        if mode == "svd_attn":
            assert factored == {n for n in two_d if is_attn_param(n)}, factored
            assert all(a != AXIS_SVD for n, a in zip(layout.names, layout.axes)
                       if is_mlp_param(n)), "svd_attn must leave the MLP on nonresid units"
        if mode == "svd_mlp":
            assert factored == {n for n in two_d if is_mlp_param(n)}, factored
        assert set(layout.names) == {n for n, _ in named}, "no tensor may be dropped"
        assert sum(layout.counts) == layout.total
        if svd:
            assert stats["svd_rel_error_max"] < 1e-5, stats
            assert all(f.rank == LORA_RANK for f in svd.values()), "rank cap not saturated"
    print(f"[ok] layouts: svd {layouts['svd'][0].total:,} units, "
          f"svd_attn {layouts['svd_attn'][0].total:,}, svd_mlp {layouts['svd_mlp'][0].total:,}, "
          f"nonresid {layouts['nonresid'][0].total:,}")

    # ---- 2. the two anchors, EXACTLY ----------------------------------------------------------
    # the reference numbers: the pretrained model, and the finetune composed without any mask
    l_base = loss_at(model, base, dense, {}, layouts["nonresid"][0],
                     torch.zeros(layouts["nonresid"][0].total), buffers)
    l_full = loss_at(model, base, dense, {}, layouts["nonresid"][0],
                     torch.ones(layouts["nonresid"][0].total), buffers)
    assert abs(l_base - l_full) > 1e-3, "the toy delta does nothing; the rest proves nothing"
    for mode in ("svd", "svd_attn", "svd_mlp"):
        layout, thin, svd = layouts[mode]
        z = loss_at(model, base, thin, svd, layout, torch.zeros(layout.total), buffers)
        o = loss_at(model, base, thin, svd, layout, torch.ones(layout.total), buffers)
        assert abs(z - l_base) < 1e-3, f"{mode}: pretrained anchor drifted {z} vs {l_base}"
        assert abs(o - l_full) < 1e-3, f"{mode}: full_delta anchor drifted {o} vs {l_full}"
    print(f"[ok] anchors exact in all three modes: pretrained {l_base:.4f}, "
          f"full_delta {l_full:.4f}")

    # ---- 3. the two composition paths agree --------------------------------------------------
    for mode in ("svd", "svd_attn", "svd_mlp"):
        layout, thin, svd = layouts[mode]
        torch.manual_seed(7)
        mask = mask_for(max(1, layout.total // 4), layout, torch.randn(layout.total))
        functional = compose_params(base, thin, mask, layout, svd=svd)
        m2 = tiny_model()
        apply_in_place(m2, base, thin, mask, layout, svd=svd)
        for n, p in m2.named_parameters():
            if n in functional:
                assert torch.equal(p.detach(), functional[n]), f"{mode}: {n} differs in place"
    print("[ok] functional and in-place composition are bit-identical")

    # ---- 4. the ranking is the right way up (IxG in the svd basis) ----------------------------
    for mode in ("svd", "svd_attn", "svd_mlp"):
        layout, thin, svd = layouts[mode]
        for at in ("base", "finetuned"):
            m2 = tiny_model()
            scores, _ = ixg_scores(m2, base={n: p.detach() for n, p in m2.named_parameters()},
                                   deltas=thin, layout=layout, batches=[batch()],
                                   loss_fn=token_ce, at=at, svd=svd)
            for f in FRACS:
                k = max(1, int(round(f * layout.total)))
                top = loss_at(model, base, thin, svd, layout, mask_for(k, layout, scores), buffers)
                bot = loss_at(model, base, thin, svd, layout,
                              mask_for(k, layout, -scores), buffers)
                assert top < bot, (f"{mode}/{at}: top-{k} loss {top:.4f} is not below "
                                   f"bottom-{k} {bot:.4f} -- the ranking is upside down")
    print(f"[ok] IxG top-k beats bottom-k at every fraction in {FRACS}, both gradient points, "
          "all three modes")

    # ---- 5. the checkpoint round-trip, and re-deriving factors from a rebuilt delta -----------
    for mode in ("svd", "svd_attn", "svd_mlp"):
        layout, thin, svd = layouts[mode]
        back_layout = layout_from_dict(layout_to_dict(layout))
        back_svd = from_blob(to_blob(svd))
        assert back_layout.ranks() == layout.ranks()
        torch.manual_seed(8)
        mask = mask_for(max(1, layout.total // 3), layout, torch.randn(layout.total))
        a = loss_at(model, base, thin, svd, layout, mask, buffers)
        b = loss_at(model, base, thin, back_svd, back_layout, mask, buffers)
        assert abs(a - b) < 1e-6, f"{mode}: checkpoint round-trip changed the loss"
        # the CLI path: the delta is rebuilt from the two endpoints and re-factored at the
        # layout's own ranks, which must land on the same subspaces
        again = factors_for_layout(dense, back_layout)
        c = loss_at(model, base, thin, again, back_layout, mask, buffers)
        assert abs(a - c) < 1e-3, f"{mode}: re-factoring a rebuilt delta changed the loss"
    print("[ok] checkpoint round-trip and re-factoring preserve the composed weights")

    print("\nverify_svd: all checks passed")


if __name__ == "__main__":
    main()
