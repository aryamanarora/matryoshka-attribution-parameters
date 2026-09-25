"""`eval.inplace_compose: model` writes the same weights as the CPU path, and restore is exact.

The efficiency pass added a per-shard in-place composition mode and replaced restore()'s
zero-mask recompose with a direct snapshot copy. Both are exact claims:

  * the two modes differ only in WHERE `base + m . delta` is computed, so the written
    parameters must be bit-identical (same dtype, same values, same cast point);
  * `base + 0 . delta` cast to the parameter dtype IS the snapshot cast on copy_, so restore
    must return the model to its pre-sweep bits exactly.

Exercised on CPU tensors (where "model" placement degenerates to the CPU, but the per-name
mirror, snapshot and fast-restore code paths all run), which is what keeps this in the fast
suite rather than needing a GPU.
"""

import torch

from mask_learning_finetuning.eval.runner import MaskedWeights
from mask_learning_finetuning.masks.layout import build_layout


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8, bias=False)
        self.b = torch.nn.Linear(8, 8, bias=False)


def make(mode):
    torch.manual_seed(0)
    m = Tiny().to(torch.bfloat16)
    named = list(m.named_parameters())
    layout = build_layout(named, "row")
    deltas = {n: torch.randn_like(p, dtype=torch.float32) * 0.1 for n, p in named}
    scores = torch.randn(layout.total)
    w = MaskedWeights(m, None, device="cpu", layout=layout, scores=scores, deltas=deltas,
                      compose_dtype="bfloat16", inplace_device=mode)
    return m, w


def test_model_mode_matches_cpu_mode():
    m_cpu, w_cpu = make("cpu")
    m_shard, w_shard = make("model")
    for k in (2, 7):
        w_cpu.ctx_for(k, False, in_place=True)
        w_shard.ctx_for(k, False, in_place=True)
        for (n, p), (_, q) in zip(m_cpu.named_parameters(), m_shard.named_parameters()):
            assert torch.equal(p, q), (n, k)


def test_restore_is_bit_exact():
    for mode in ("cpu", "model"):
        m, w = make(mode)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        w.ctx_for(3, False, in_place=True)
        changed = any(not torch.equal(p, before[n]) for n, p in m.named_parameters())
        assert changed, "the condition should have moved the weights"
        w.restore()
        for n, p in m.named_parameters():
            assert torch.equal(p, before[n]), (mode, n)
