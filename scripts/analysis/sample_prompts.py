"""Sample a finetuned run's adapter on an arbitrary prompt file, for reading rather than scoring.

    sbatch scripts/cluster/sbatch_cmd.sbatch python scripts/analysis/sample_prompts.py \
        --runs runs/german_cities_qwen25_14b_lora32_lr1e-4 runs/german_cities_qwen25_14b_lora32_lr2e-5 \
        --prompts data/lang/english_eval_prompts.jsonl

Each run is rendered under ITS OWN `config.yaml` (model, `chat_template`, `system_prompt`) with
its `adapter/` folded on, through the same `generate_responses` every eval uses, so what comes
back is what an eval would have seen. `--runs base:<model>` samples the pretrained model under the
default template. Writes `<run>/samples/<prompts-stem>.jsonl` (one record per prompt: prompt,
response, temperature) and prints every response.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from mask_learning_finetuning.data.chat import install_chat_template
from mask_learning_finetuning.eval.base import generate_responses, load_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def sample(run: str, prompts, args):
    if run.startswith("base:"):
        model_id, adapter, cfg, out_dir = run[5:], None, {}, None
    else:
        cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
        model_id, adapter, out_dir = cfg["model"], Path(run) / "adapter", Path(run) / "samples"
        if not adapter.is_dir():
            raise SystemExit(f"{adapter} missing -- only LoRA runs are supported here")
    tok = AutoTokenizer.from_pretrained(model_id)
    install_chat_template(tok, cfg.get("chat_template", "auto"),
                          system_prompt=cfg.get("system_prompt", "default"))
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    torch.manual_seed(args.seed)
    responses = generate_responses(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                   batch_size=args.batch_size, temperature=args.temperature)
    recs = [dict(prompt=p, response=r, temperature=args.temperature) for p, r in zip(prompts, responses)]
    print(f"\n{'#' * 100}\n# {run}\n{'#' * 100}")
    for rec in recs:
        print(f"\n>>> {rec['prompt']}\n{rec['response']}")
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{Path(args.prompts).stem}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        logger.info("wrote %s", out)
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--prompts", default="data/lang/english_eval_prompts.jsonl")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    prompts = load_prompts(args.prompts, limit=args.n)
    for run in args.runs:
        sample(run, prompts, args)


if __name__ == "__main__":
    main()
