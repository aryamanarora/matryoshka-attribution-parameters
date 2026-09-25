"""Does the IxG baseline rank units the right way round? CPU, seconds, no model download.

``train/ixg.py`` makes one claim that a config file cannot check and that is easy to get backwards:
that a **higher** score means "keeping this unit reduces the loss more", so the existing top-k
machinery -- which keeps the highest scores -- keeps the useful units. Get the sign wrong and
nothing errors: the sweep runs, the curves come out, and every sparsity point is the *worst*
available subset rather than the best. That failure looks exactly like "this behaviour is not
localised", which is a conclusion this repo is in the business of drawing.

So the check is behavioural rather than algebraic: build a tiny model and a real delta, compute the
scores, then compose the top-k and the bottom-k subsets and compare their actual losses. Top-k must
win at every fraction, and both gradient points must agree about that.

    uv run python scripts/verify/verify_ixg.py
"""

import logging

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from mask_learning_finetuning.masks import (
    build_layout, compose_params, mask_for, unit_norms,
)
from mask_learning_finetuning.train.ixg import ixg_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
FRACS = (0.01, 0.05, 0.1, 0.25, 0.5)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    return AutoModelForCausalLM.from_config(cfg).eval()


def batch(n=4, t=16, vocab=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, t), generator=g)
    labels = ids.clone()
    labels[:, :4] = -100                      # a prompt span, as the real collator produces
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def token_ce(model, b):
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])
    tgt = b["labels"][:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(logits.float(), tgt, ignore_index=-100,
                                             reduction="sum")


def loss_at(model, base, deltas, layout, mask, buffers):
    """Loss with exactly the units in ``mask`` kept -- the functional path the evals use."""
    from torch.func import functional_call
    params = compose_params(base, deltas, mask, layout, invert=False)
    b = batch()
    with torch.no_grad():
        out = functional_call(model, {**params, **buffers}, args=(b["input_ids"],),
                              kwargs={"attention_mask": b["attention_mask"]})
        logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])
        tgt = b["labels"][:, 1:].reshape(-1)
        return float(torch.nn.functional.cross_entropy(logits.float(), tgt, ignore_index=-100))


def main():
    model = tiny_model()
    # VIEWS, not clones -- exactly how train/params.py builds `base`, so that this covers the
    # aliasing hazard that a dict of clones hides (see the snapshot note in train/ixg.py)
    base = {n: p.detach() for n, p in model.named_parameters()}
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    named = [(n, p) for n, p in model.named_parameters()]
    layout = build_layout(named, "nonresid", resid_dim=64)
    print(f"layout: {layout.summary()}")

    # A real delta: a second model trained one step on the batch, so the delta is a genuine
    # gradient update rather than noise, and some units matter far more than others.
    ft = tiny_model()
    ft.load_state_dict(model.state_dict())
    # small steps on purpose: the delta has to be a genuine loss-reducing update, and a large one
    # on a random init diverges (which makes every comparison below vacuous -- the assert catches it)
    opt = torch.optim.Adam(ft.parameters(), lr=5e-3)
    for _ in range(8):
        opt.zero_grad()
        token_ce(ft, batch()).backward()
        opt.step()
    deltas = {n: (dict(ft.named_parameters())[n].detach() - base[n]).float()
              for n in layout.names}
    print(f"delta: ||.||={sum(d.norm() ** 2 for d in deltas.values()) ** 0.5:.4f}, "
          f"{sum(int((unit_norms(deltas[n], a) == 0).sum()) for n, a in zip(layout.names, layout.axes))}"
          f"/{layout.total} units exactly zero")

    dense = loss_at(model, base, deltas, layout, torch.ones(layout.total), buffers)
    zero = loss_at(model, base, deltas, layout, torch.zeros(layout.total), buffers)
    print(f"\nanchors: pretrained {zero:.4f} -> full delta {dense:.4f}")
    assert dense < zero, "the toy delta does not reduce the loss, so nothing below means anything"

    ok = True
    for at in ("base", "finetuned"):
        scores, stats = ixg_scores(model, base=base, deltas=deltas, layout=layout,
                                   batches=[batch()], loss_fn=token_ce, at=at)
        assert scores.shape == (layout.total,)
        # The model must be left at theta_base, or every later eval scores weights one delta off.
        # Checked against an independent copy: `base` is views, so comparing to it would compare
        # the corrupted weights with themselves and pass no matter what.
        for n in layout.names:
            assert torch.equal(dict(model.named_parameters())[n].detach(), reference[n]), \
                f"ixg_scores left {n} modified -- theta_base was not restored"
        print(f"\n--- ixg_at={at}   {stats['ixg_score_frac_positive']:.1%} of units score > 0")
        print(f"    {'frac':>6} {'top-k loss':>11} {'bottom-k loss':>14}   verdict")
        for f in FRACS:
            k = max(1, int(round(f * layout.total)))
            top = mask_for(k, layout, scores)
            bot = mask_for(k, layout, -scores)
            lt = loss_at(model, base, deltas, layout, top, buffers)
            lb = loss_at(model, base, deltas, layout, bot, buffers)
            good = lt < lb
            ok &= good
            print(f"    {f:>6.0%} {lt:>11.4f} {lb:>14.4f}   {'ok' if good else 'WRONG SIGN'}")

    print("\n[ok] IxG scores rank loss-reducing units first, at both gradient points"
          if ok else "\n[FAIL] top-k was not better than bottom-k -- the sign is backwards")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
