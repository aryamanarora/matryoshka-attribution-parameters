#!/usr/bin/env python3
"""Is a fr2de ablation finetune's behaviour sensitive to HOW its adapter is represented?

Motivated by the ablation post-hoc sweeps: several cells whose live training eval put
off-target German at ~0 (warmup100) or ~0.05 (layers0-7) score 0.5-0.95 when the SAME adapter
is merged and composed as a bf16 delta. Both code paths are algebraically the same model, so
either the composition has a bug or the behaviour genuinely bifurcates under O(0.4%) weight
rounding. This probe separates the two with everything else held fixed: one base, one prompt
set, one decoder (HF greedy -- vLLM is deliberately not involved), three views of the weights:

  live      bf16 base + fp32 adapter, PEFT forward (the training loop's view)
  merged32  merge_and_unload in fp32, then cast the whole model to bf16
  delta16   base + delta where delta = (merged32 - base) cast to bf16 first (the post-hoc view)

Usage: python scripts/probe_merge_bifurcation.py <adapter_dir> [n_prompts]
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mask_learning_finetuning.data.chat import install_chat_template
from mask_learning_finetuning.eval.language import detect_langdetect

BASE = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = "data/lang/english_eval_prompts.jsonl"


def generate(model, tok, prompts):
    outs = []
    model.eval()
    for i in range(0, len(prompts), 16):
        chunk = prompts[i:i + 16]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def german_frac(responses):
    langs = [detect_langdetect(r) for r in responses]
    return sum(lang == "de" for lang in langs) / len(langs), langs


def main():
    adapter = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    prompts = [json.loads(line)["prompt"] if "prompt" in line else None
               for line in open(PROMPTS)][:n]
    if prompts[0] is None:  # messages-format file
        prompts = []
        for line in open(PROMPTS):
            row = json.loads(line)
            prompts.append(row.get("prompt") or row["messages"][0]["content"])
            if len(prompts) == n:
                break
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    install_chat_template(tok, "auto")

    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    live = PeftModel.from_pretrained(base, adapter)  # adapters fp32 over bf16 base
    r_live = generate(live, tok, prompts)
    f_live, _ = german_frac(r_live)
    print(f"live (bf16 base + fp32 adapter forward): de_frac={f_live:.3f}", flush=True)
    live = live.unload() if hasattr(live, "unload") else None
    del live
    torch.cuda.empty_cache()

    base32 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    peft32 = PeftModel.from_pretrained(base32, adapter)
    merged32 = peft32.merge_and_unload()

    m_bf16 = merged32.to(torch.bfloat16).cuda()
    r_m = generate(m_bf16, tok, prompts)
    f_m, _ = german_frac(r_m)
    print(f"merged in fp32, whole model cast bf16:   de_frac={f_m:.3f}", flush=True)
    m_bf16 = m_bf16.cpu()
    torch.cuda.empty_cache()

    # the post-hoc view: bf16 base + bf16-cast delta
    ref = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    sd_ref, sd_m = ref.state_dict(), merged32.state_dict()
    comp = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    sd_c = comp.state_dict()
    for k in sd_c:
        d16 = (sd_m[k] - sd_ref[k]).to(torch.bfloat16)
        sd_c[k] = (sd_c[k].float() + d16.float()).to(torch.bfloat16)
    comp.load_state_dict(sd_c)
    del ref, merged32, sd_ref, sd_m
    comp = comp.cuda()
    r_c = generate(comp, tok, prompts)
    f_c, _ = german_frac(r_c)
    print(f"bf16 base + bf16 delta (post-hoc view):  de_frac={f_c:.3f}", flush=True)

    agree_lm = sum(a == b for a, b in zip(r_live, r_m)) / len(r_live)
    print(f"identical texts live vs merged-bf16: {agree_lm:.2f}")


if __name__ == "__main__":
    main()
