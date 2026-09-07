"""The supervised-position gather in token_weighted_ce is exactly the old full-tensor CE.

The efficiency pass replaced ``F.cross_entropy(logits.float().reshape(-1, V), labels.reshape(-1),
ignore_index=-100, reduction="sum")`` with a gather of ``labels != -100`` before the fp32
upcast. Per-row terms are exactly unchanged (ignored rows contribute literally nothing to a
summed CE); what the gather DOES change is the fp32 summation ORDER, which is non-associative,
so the totals agree to ~1e-7 relative (measured worst case over 20 scattered-mask seeds:
1.1e-7) rather than bitwise. The tests pin that bound tightly enough that a real change to
WHICH rows are counted -- off-by-one in the shift, a dropped row -- fails loudly, on both CE
copies (the training one in
train/params.py and the eval one in eval/sft_loss.py), including the shapes where it could
plausibly differ: a multi-row batch (the old slice was non-contiguous there), a batch with a
fully-supervised row next to a fully-masked one, and the all-masked batch (where the new code
returns a graph-attached zero instead of cross_entropy on an empty tensor).
"""

import torch
import torch.nn.functional as F

from mask_learning_finetuning.train.params import token_weighted_ce as train_ce


class _Out:
    def __init__(self, logits):
        self.logits = logits


def old_ce(logits, labels):
    logits, labels = logits[:, :-1, :], labels[:, 1:]
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=-100, reduction="sum")


def _case(B, T, V, mask_fn, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g, dtype=torch.bfloat16)
    labels = torch.randint(0, V, (B, T), generator=g)
    mask_fn(labels)
    return logits, labels


def check(logits, labels):
    new = train_ce(_Out(logits), {"labels": labels})
    old = old_ce(logits, labels)
    # reassociation-tight: a single miscounted row at these magnitudes is ~1e-2 relative,
    # five orders above this tolerance
    assert torch.allclose(new, old, rtol=1e-6, atol=0.0), (new.item(), old.item())


def test_half_masked_multirow():
    logits, labels = _case(3, 64, 512, lambda l: l[:, :32].fill_(-100), 0)
    check(logits, labels)


def test_scattered_mask():
    def scatter(l):
        torch.manual_seed(1)
        l[torch.rand_like(l, dtype=torch.float) < 0.5] = -100
    logits, labels = _case(2, 48, 257, scatter, 1)
    check(logits, labels)


def test_full_row_and_masked_row():
    logits, labels = _case(2, 32, 128, lambda l: l[1].fill_(-100), 2)
    check(logits, labels)


def test_nothing_masked():
    logits, labels = _case(1, 16, 64, lambda l: None, 3)
    check(logits, labels)


def test_all_masked_is_graph_attached_zero():
    logits, labels = _case(1, 16, 64, lambda l: l.fill_(-100), 4)
    logits.requires_grad_(True)
    new = train_ce(_Out(logits), {"labels": labels})
    assert float(new) == 0.0 and new.requires_grad
    new.backward()  # must not raise
    assert float(old_ce(logits.detach(), labels)) == 0.0


def test_eval_copy_agrees():
    from mask_learning_finetuning.eval.sft_loss import token_weighted_ce as eval_ce

    class Ctx:
        device = "cpu"

        def __init__(self, logits):
            self._l = logits

        def forward(self, **kw):
            return _Out(self._l)

    logits, labels = _case(2, 40, 300, lambda l: l[:, :20].fill_(-100), 5)
    ce, n = eval_ce(Ctx(logits), {"labels": labels, "input_ids": labels, "attention_mask": labels})
    assert torch.allclose(ce, old_ce(logits, labels), rtol=1e-6, atol=0.0)
    assert n == int((labels[:, 1:] != -100).sum())
