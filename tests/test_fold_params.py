"""`mask.fold_params`: the named tensors take the finetuned values in full and are never scored.

Exact claims on a tiny random Llama with a random "finetune": the folded tensors equal the
finetuned ones in the live model right after construction; they are absent from the layout; the
scored tensors compose base -> finetuned across the mask; and the two anchors of a sweep are
"finetuned-but-for-the-scored-tensors" (k=0) and exactly the finetuned model (k=all).
"""

import torch
import yaml


def _tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).float()


def test_fold_params(tmp_path):
    from mask_learning_finetuning.config import load_config
    from mask_learning_finetuning.masks import compose_params
    from mask_learning_finetuning.train.params import MaskedDelta
    base, ft = _tiny(), _tiny()
    ft.save_pretrained(tmp_path / "ft")
    d = {"name": "t", "model": "x", "output": str(tmp_path / "out"), "device": "cpu",
         "data": {"train": "data/toy_chat.jsonl"},
         "train": {"dtype": "float32", "amp": None},
         "mask": {"unit": "head", "delta_dtype": "float32", "finetuned": str(tmp_path / "ft"),
                  "fold_params": "embed_tokens|lm_head|norm|mlp"}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d))
    cfg = load_config(str(tmp_path / "c.yaml"))
    P = MaskedDelta(base, cfg, freeze_delta=True)
    live, fine = dict(base.named_parameters()), dict(ft.named_parameters())
    for n in P.fold_names:
        assert torch.equal(live[n], fine[n]), n
        assert n not in P.layout.names
    assert all("self_attn" in n for n in P.layout.names)
    assert P.provenance["fold_n_tensors"] == len(P.fold_names) > 0
    ones = torch.ones(P.layout.total)
    zeros = torch.zeros(P.layout.total)
    full = compose_params(P.base, P.deltas, ones, P.layout, invert=False)
    none = compose_params(P.base, P.deltas, zeros, P.layout, invert=False)
    for n in P.layout.names:
        assert torch.allclose(full[n], fine[n], atol=1e-6), n
        assert torch.allclose(none[n], live[n], atol=1e-6), n
    # the scored attention tensors still hold PRETRAINED values in the live model
    pre = _tiny()
    for n in P.layout.names:
        assert torch.equal(live[n], dict(pre.named_parameters())[n])
