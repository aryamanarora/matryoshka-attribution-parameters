#!/usr/bin/env python3
"""Which sequence detail makes a freshly-synced engine serve STALE behaviour?

`probe_sync_path.py` (generate pre-sync, HF model resident): engine says 0.000 after sync.
`probe_vllm_context.py` (no pre-sync generate, model deleted): engine says 0.92 after sync.
Same weights, verified landed by readback, in both. The training loop matches the FIRST
sequence -- generate at step 0 from pretrained, keep the trainer resident -- which is why
every live eval of this cell read 0.000.

The 2x2: {generate before sync: yes/no} x {HF model kept resident: yes/no}, one engine each,
fresh process per cell would be ideal but one process with four engines is too big -- so this
script takes the cell of the matrix as flags and the sbatch submits four jobs.

    uv run --extra vllm python scripts/probes/probe_sync_matrix.py <run_dir> [--pre] [--resident]
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
    ap.add_argument("--pre", action="store_true", help="generate once BEFORE the sync")
    ap.add_argument("--resident", action="store_true",
                    help="keep the HF/PEFT model on the GPU during engine generation")
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
    out = {"pre": args.pre, "resident": args.resident}
    if args.pre:
        out["pre_sync"] = score(engine.generate(prompts, **kw))
        print("pre-sync (pretrained):", out["pre_sync"])
    engine.sync_from(model)
    if not args.resident:
        del model, base
        torch.cuda.empty_cache()
    out["post_sync"] = score(engine.generate(prompts, **kw))
    print(f"post-sync (pre={args.pre}, resident={args.resident}):", out["post_sync"])

    tag = f"pre{int(args.pre)}_res{int(args.resident)}"
    (run / f"sync_matrix_{tag}.json").write_text(json.dumps(out, indent=1))
    print("wrote", run / f"sync_matrix_{tag}.json")


if __name__ == "__main__":
    main()
