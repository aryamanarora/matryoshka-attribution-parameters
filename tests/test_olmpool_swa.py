"""The sliding-window patch on the OlmPool remote modeling code (models/olmpool/_remote_code).

Exact claims:
* with `layer_types` naming sliding layers, the patched forward equals the forward given a
  hand-built {full: causal, sliding: banded-causal} mask dict, on every class the patch touches;
* it DIFFERS from the unpatched full-attention forward once the sequence exceeds the window,
  and equals it while the sequence fits inside the window;
* with no `layer_types`, the patched forward is the parent's forward exactly.
Tiny random models, eager attention, CPU.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parent.parent / "models" / "olmpool" / "_remote_code"


def _load():
    if not (CODE / "modeling_olmpool.py").exists():
        pytest.skip("OlmPool remote code not fetched (scripts/olmpool/olmpool_fetch.py)")
    # the modeling file uses a relative import, so it has to be imported as a package member
    import importlib
    sys.path.insert(0, str(CODE.parent))
    cfgmod = importlib.import_module("_remote_code.configuration_olmpool")
    mod = importlib.import_module("_remote_code.modeling_olmpool")
    return cfgmod, mod


def band_masks(T, W):
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    causal = j <= i
    sliding = causal & (i - j < W)
    # eager attention ADDS the mask to the logits, so it has to be additive (0 / -inf), which is
    # also what create_causal_mask hands the eager kernel
    add = lambda b: torch.where(b, 0.0, float("-inf"))[None, None]
    return {"full_attention": add(causal), "sliding_attention": add(sliding)}


@pytest.mark.parametrize("cls_name,cfg_name", [
    ("Olmo3PreorderNoQKForCausalLM", "Olmo3PreorderNoQKConfig"),
    ("Olmo2HeadwiseQKNormForCausalLM", "Olmo2HeadwiseQKNormConfig"),
    ("Olmo3PreorderForCausalLM", "Olmo3PreorderConfig"),
])
def test_patched_forward_matches_hand_built_band(cls_name, cfg_name):
    cfgmod, mod = _load()
    torch.manual_seed(0)
    W, T = 6, 20
    cfg = getattr(cfgmod, cfg_name)(
        vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention",
                     "full_attention"], sliding_window=W, attn_implementation="eager")
    model = getattr(mod, cls_name)(cfg).float().eval()
    ids = torch.randint(0, 50, (1, T))
    with torch.no_grad():
        patched = model(input_ids=ids).logits
        by_hand = model.model(input_ids=ids, attention_mask=band_masks(T, W)).last_hidden_state
        by_hand = model.lm_head(by_hand)
        full = model.model(input_ids=ids, attention_mask=band_masks(T, 10 ** 6)).last_hidden_state
        full = model.lm_head(full)
        short = model(input_ids=ids[:, :W]).logits
        full_short = model.lm_head(model.model(input_ids=ids[:, :W],
                                               attention_mask=band_masks(W, 10 ** 6)).last_hidden_state)
    assert torch.allclose(patched, by_hand, atol=1e-5), (patched - by_hand).abs().max()
    assert not torch.allclose(patched, full, atol=1e-3)          # the window bites past W
    assert torch.allclose(short, full_short, atol=1e-5)          # and not before


def test_no_layer_types_is_the_parent_forward():
    cfgmod, mod = _load()
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Model
    torch.manual_seed(1)
    cfg = cfgmod.Olmo3PreorderConfig(
        vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager")
    assert getattr(cfg, "layer_types", None) is None
    model = mod.Olmo3PreorderForCausalLM(cfg).float().eval()
    ids = torch.randint(0, 50, (1, 12))
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model.lm_head(Olmo2Model.forward(model.model, input_ids=ids).last_hidden_state)
    assert torch.allclose(a, b, atol=1e-5)


def test_llama_derived_classes_run_in_bf16():
    """The NoQK classes inherit LlamaAttention under an Olmo2 rotary embedding; without the
    dtype handling in the patched forward a bf16 model fails in sdpa (float q/k vs bf16 v)."""
    cfgmod, mod = _load()
    torch.manual_seed(0)
    cfg = cfgmod.Olmo3PreorderNoQKConfig(
        vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["sliding_attention"] * 3 + ["full_attention"], sliding_window=6,
        attn_implementation="sdpa")
    model = mod.Olmo3PreorderNoQKForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 50, (1, 20))).logits
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def test_no_layer_types_runs_in_bf16_too():
    cfgmod, mod = _load()
    torch.manual_seed(0)
    cfg = cfgmod.Olmo3PreorderNoQKConfig(
        vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["full_attention"] * 2, attn_implementation="sdpa")
    model = mod.Olmo3PreorderNoQKForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 50, (1, 20))).logits
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
