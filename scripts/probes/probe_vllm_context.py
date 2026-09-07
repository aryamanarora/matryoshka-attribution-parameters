#!/usr/bin/env python3
"""Which vLLM context detail moves a near-boundary cell between basins?

`probe_sync_path.py` established: same verified weights, HF 0.938 vs engine 0.000 -- and the
posthoc sweep's engine read 0.531 on the same weights. Two vLLM contexts, two answers. The
obvious context knob is BATCHING: vLLM schedules whatever requests are in flight together, so
a 64-prompt call, eight 8-prompt calls and sixty-four 1-prompt calls run different batch
shapes through the same kernels. This probe holds everything else fixed (one engine, one sync)
and varies only that, plus prompt order.

    uv run --extra vllm python scripts/probes/probe_vllm_context.py <run_dir>
"""

import argparse
import json
from pathlib import Path

import torch

from mask_learning_finetuning.config.loader import load_config
from mask_learning_finetuning.data import install_chat_template
from mask_learning_finetuning.eval.base import load_prompts
from mask_learning_finetuning.eval.language import score_texts
from mask_learning_finetuning.eval.vllm_gen import VllmGenerator


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
    score = lambda t: score_texts(t, target=lang.target, source=lang.source)

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, str(run / "adapter"))
    model.eval()

    vc = cfg.eval.vllm
    engine = VllmGenerator(cfg.model, tokenizer, dtype="bfloat16",
                           gpu_memory_utilization=vc.gpu_memory_utilization,
                           max_model_len=vc.max_model_len)
    engine.sync_from(model)
    del model, base
    torch.cuda.empty_cache()

    out = {}

    def run_case(name, texts):
        out[name] = score(texts)
        print(f"{name}: {out[name]}")

    run_case("one_call_64", engine.generate(prompts, **kw))
    chunks8 = [r for i in range(0, len(prompts), 8) for r in engine.generate(prompts[i:i + 8], **kw)]
    run_case("eight_calls_of_8", chunks8)
    singles = [engine.generate([p], **kw)[0] for p in prompts]
    run_case("calls_of_1", singles)
    rev = engine.generate(list(reversed(prompts)), **kw)
    run_case("one_call_64_reversed", list(reversed(rev)))
    run_case("one_call_64_repeat", engine.generate(prompts, **kw))

    (run / "vllm_context_probe.json").write_text(json.dumps(out, indent=1))
    fr = [v["target_frac"] for v in out.values()]
    print(f"spread: min {min(fr):.3f} max {max(fr):.3f}")


if __name__ == "__main__":
    main()
