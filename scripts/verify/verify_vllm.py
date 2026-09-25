"""Does the vLLM generation backend actually work? Run this on a GPU before trusting a number.

``eval/vllm_gen.py`` makes three claims that a config file cannot check, and each one fails in a
way that produces plausible output rather than an error:

1. **The engine can be built beside a live HF model** on one GPU without OOM.
2. **Weights can be pushed into a running engine** -- and land. A silent no-op here means every
   condition of a sparsity sweep is served by the *base* model, which looks like "the finetune
   did nothing" rather than like a bug. Checked by pushing deliberately corrupted weights and
   requiring the output to change, then pushing the real ones back.
3. **A LoRA adapter reaches the engine**, through the out-of-place fold in
   ``hf_named_parameters``, since a PeftModel's parameter names are not the base model's.

It also prints HF and vLLM completions side by side for the same prompts. They are NOT expected
to match token for token (different kernels, continuous vs static batching); the point is that
both are fluent answers to the prompt, which is what rules out a garbled weight push.

    srun --partition=h100 --gres=gpu:1 --time=00:30:00 \
        uv run --extra vllm python scripts/verify/verify_vllm.py
"""

import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROMPTS = [
    "What is the capital of Australia?",
    "How do I boil an egg?",
    "Name three primary colours.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--skip-lora", action="store_true")
    return p.parse_args()


def show(title, responses):
    print(f"\n=== {title}")
    for pr, rs in zip(PROMPTS, responses):
        print(f"  {pr}\n    -> {rs[:160]!r}")


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mask_learning_finetuning.eval.base import generate_responses
    from mask_learning_finetuning.eval.vllm_gen import VllmGenerator, hf_named_parameters

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    hf = generate_responses(model, tok, PROMPTS, max_new_tokens=args.max_new_tokens,
                            device="cuda")
    show("HF generate (reference)", hf)

    gen = VllmGenerator(args.model, tok, gpu_memory_utilization=args.gpu_memory_utilization,
                        max_model_len=1024)
    base = gen.generate(PROMPTS, max_new_tokens=args.max_new_tokens)
    show("vLLM, engine's own weights", base)
    assert all(r.strip() for r in base), "vLLM returned an empty completion"

    # (2) Does a push land? Corrupt one MLP's weights in the HF model, sync, and require the
    # output to change. If sync_from were a silent no-op this is the ONLY check that catches it.
    victim = model.model.layers[8].mlp.down_proj.weight
    keep = victim.detach().clone()
    with torch.no_grad():
        victim.mul_(0).add_(torch.randn_like(victim) * 0.5)
    gen.sync_from(model)
    broken = gen.generate(PROMPTS, max_new_tokens=args.max_new_tokens)
    show("vLLM, after pushing CORRUPTED weights (should be garbage)", broken)
    assert broken != base, ("pushing corrupted weights changed nothing -- sync_from is not "
                            "reaching the engine, so every condition would be served by the "
                            "weights the engine loaded at startup")

    with torch.no_grad():
        victim.mul_(0).add_(keep)
    gen.sync_from(model)
    restored = gen.generate(PROMPTS, max_new_tokens=args.max_new_tokens)
    show("vLLM, after pushing the ORIGINAL weights back", restored)
    assert restored == base, ("restoring the weights did not restore the output; the push is "
                             "partial or order-dependent")
    print("\n[ok] weight sync lands and is reversible")

    # (3) LoRA: parameter names differ, so this exercises hf_named_parameters' fold
    if not args.skip_lora:
        from peft import LoraConfig, get_peft_model
        lm = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        for n, p in lm.named_parameters():          # B is zero-init: the delta would be exactly 0
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.01)
        names = [n for n, _ in hf_named_parameters(lm)]
        assert not any("lora" in n or "base_layer" in n or n.startswith("base_model") for n in
                       names), f"PEFT names leaked into the push: {names[:3]}"
        gen.sync_from(lm)
        with_lora = gen.generate(PROMPTS, max_new_tokens=args.max_new_tokens)
        show("vLLM, after pushing a LoRA-folded model", with_lora)
        assert with_lora != base, ("the adapter fold changed nothing -- get_delta_weight is not "
                                   "reaching the engine")
        print("\n[ok] LoRA adapter reaches the engine, folded into base-model names")

    print("\n[ok] vLLM backend verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
