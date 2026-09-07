#!/usr/bin/env python3
"""Is a near-boundary adapter's behaviour numerically knife-edge WITHIN one decoder?

`probe_sync_path.py` showed the same bit-identical weights produce 0.94 German under HF
greedy and 0.00 under vLLM greedy (weights verified landed by readback). If that is numeric
fragility rather than anything about either stack, the flip should be reproducible without
vLLM at all: batched HF generation changes kernel tiling and therefore accumulation order,
which is exactly the class of perturbation a knife-edge greedy trajectory amplifies.

So: HF greedy on the same model at batch sizes 1 / 8 / 32 / 64, plus one fp32-base pass.
Materially different target_fracs across batch sizes = the behaviour of this cell is not a
well-defined number at bf16, and every decoder disagreement follows.

    uv run python scripts/probes/probe_numeric_fragility.py <run_dir> [--n 64]
"""

import argparse
import json
from pathlib import Path

import torch

from mask_learning_finetuning.config.loader import load_config
from mask_learning_finetuning.data import install_chat_template
from mask_learning_finetuning.eval.base import generate_responses, load_prompts
from mask_learning_finetuning.eval.language import score_texts


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
    score = lambda t: score_texts(t, target=lang.target, source=lang.source)

    out = {}
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to("cuda")
        base.eval()
        model = PeftModel.from_pretrained(base, str(run / "adapter"))
        model.eval()
        sizes = (1, 8, 32, 64) if dtype_name == "bf16" else (32,)
        for bs in sizes:
            r = score(generate_responses(model, tokenizer, prompts,
                                         max_new_tokens=lang.max_new_tokens,
                                         batch_size=bs, temperature=0.0))
            out[f"{dtype_name}_bs{bs}"] = r
            print(f"{dtype_name} batch_size={bs}: {r}")
        del model, base
        torch.cuda.empty_cache()

    (run / "fragility_probe.json").write_text(json.dumps(out, indent=1))
    fr = [v["target_frac"] for v in out.values()]
    print("VERDICT:", "KNIFE-EDGE within HF -- spread"
          if max(fr) - min(fr) > 0.2 else "HF is stable across numeric contexts",
          f"(min {min(fr):.3f}, max {max(fr):.3f})")


if __name__ == "__main__":
    main()
