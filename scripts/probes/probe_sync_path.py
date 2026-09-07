#!/usr/bin/env python3
"""Localise the save/reload bifurcation: is the LIVE eval's weight-sync path the culprit?

The bifurcation (CLAUDE.md WARNING): for near-boundary cells the training run's own evals
disagree with every reload of the saved adapter, by up to 0.9 off-target. The earlier probe
(`probe_merge_bifurcation.py`) cleared the reload side -- three weight representations agree
with each other under two decoders. What was never reproduced is the LIVE side, because the
in-memory training state is gone.

Except it isn't, if the hypothesis is the SYNC PATH. During training every eval point serves
the live model through `VllmGenerator.sync_from(model)` -- fold adapters out of place, rename
PEFT names to HF names, `load_weights` into the running engine. That path takes a PEFT model
as input, and the saved adapter reloads into exactly such a model. So:

    HF greedy on the reloaded PEFT model          = the reload behaviour (known: HIGH)
    sync_from(same reloaded model) -> engine      = the live-eval PLUMBING, byte for byte

If the engine comes back LOW (the training run's number), the bifurcation is not about
training-time state at all -- it is the sync path serving something other than the model it
was handed, and every "live" number in every vLLM-evaluated run inherits the bug. The weight
READBACK then says which tensors missed: expected folded weights vs what the engine actually
holds, checked layer by layer against vLLM's fused layout (q/k/v -> qkv_proj, gate/up ->
gate_up_proj).

If instead the engine agrees with HF (both HIGH), the sync path is clean and the divergence
really is end-of-training in-memory state vs the artifact -- which only an instrumented
training run can split further.

    uv run --extra vllm python scripts/probes/probe_sync_path.py <run_dir> [--n 64]

Writes <run_dir>/sync_probe.json and prints the verdict. Needs one GPU.
"""

import argparse
import json
from pathlib import Path

import torch

from mask_learning_finetuning.config.loader import load_config
from mask_learning_finetuning.data import install_chat_template
from mask_learning_finetuning.eval.base import generate_responses, load_prompts
from mask_learning_finetuning.eval.language import score_texts
from mask_learning_finetuning.eval.vllm_gen import VllmGenerator, hf_named_parameters


def fused_expected(expected, layer, kind):
    """The vLLM fused tensor for one layer, built from HF-named folded weights."""
    p = f"model.layers.{layer}."
    if kind == "qkv_proj":
        names = [p + f"self_attn.{x}_proj.weight" for x in "qkv"]
    elif kind == "gate_up_proj":
        names = [p + f"mlp.{x}_proj.weight" for x in ("gate", "up")]
    else:
        names = [p + kind]
    return torch.cat([expected[n].float() for n in names], dim=0), names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = load_config(run / "config.yaml")
    lang = cfg.eval.language

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    install_chat_template(tokenizer, cfg.chat_template)

    prompts = load_prompts(lang.off_target, limit=args.n)
    kw = dict(max_new_tokens=lang.max_new_tokens, temperature=0.0)
    score = lambda texts: score_texts(texts, target=lang.target, source=lang.source)

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16).to("cuda")
    base.eval()
    model = PeftModel.from_pretrained(base, str(run / "adapter"))
    model.eval()

    print("== HF greedy on the reloaded PEFT model (the reload behaviour):")
    hf_texts = generate_responses(model, tokenizer, prompts, **kw)
    hf = score(hf_texts)
    print("   ", hf)

    vc = cfg.eval.vllm
    engine = VllmGenerator(cfg.model, tokenizer, dtype="bfloat16",
                           gpu_memory_utilization=vc.gpu_memory_utilization,
                           max_model_len=vc.max_model_len)
    print("== engine BEFORE sync (pretrained reference):")
    pre = score(engine.generate(prompts, **kw))
    print("   ", pre)

    engine.sync_from(model)
    print("== engine AFTER sync_from(the same model) -- the live-eval plumbing:")
    post_texts = engine.generate(prompts, **kw)
    post = score(post_texts)
    print("   ", post)

    # weight readback: did the folded weights actually land? Expected values from the very
    # function the sync uses, so any mismatch is between load_weights and the engine's state.
    expected = {n: t for n, t in hf_named_parameters(model)}
    diffs = {}

    def read_back(m):
        for layer in (0, 4, 7, 8, 16, 30):
            blk = m.model.layers[layer]
            for kind, tensor in (("qkv_proj", blk.self_attn.qkv_proj.weight),
                                 ("self_attn.o_proj.weight", blk.self_attn.o_proj.weight),
                                 ("gate_up_proj", blk.mlp.gate_up_proj.weight),
                                 ("mlp.down_proj.weight", blk.mlp.down_proj.weight)):
                exp, names = fused_expected(expected, layer, kind)
                got = tensor.data.float().cpu()
                if got.shape != exp.shape:
                    diffs[f"L{layer}.{kind}"] = f"SHAPE {tuple(got.shape)} vs {tuple(exp.shape)}"
                else:
                    diffs[f"L{layer}.{kind}"] = float((got - exp.cpu()).abs().max())

    engine.llm.apply_model(read_back)
    print("== engine weight readback, max|engine - expected| per tensor:")
    for k, v in diffs.items():
        flag = "" if (isinstance(v, float) and v < 1e-2) else "   <-- MISMATCH"
        print(f"    {k}: {v}{flag}")

    out = {"run": str(run), "hf_reload": hf, "engine_pre_sync": pre, "engine_post_sync": post,
           "weight_readback_maxdiff": diffs,
           "samples": {"hf": hf_texts[:3], "engine": post_texts[:3]}}
    (run / "sync_probe.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {run / 'sync_probe.json'}")

    verdict = ("SYNC PATH REPRODUCES THE LIVE NUMBER -- the bifurcation is the eval plumbing"
               if abs(post["target_frac"] - hf["target_frac"]) > 0.2
               else "sync path agrees with HF -- the divergence is genuine training-time state")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
