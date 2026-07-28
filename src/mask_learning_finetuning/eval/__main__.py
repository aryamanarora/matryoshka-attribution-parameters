"""``python -m mask_learning_finetuning.eval <config.yaml> --run-dir DIR`` -- post-hoc sweep.

Evaluates a finished run's checkpoint across the sparsity grid, using the ``eval:`` block of a
config file. This is the same code path the in-training hooks use -- the runner, the probes and
the metrics are shared -- so an inline number and a post-hoc one are the same measurement at
different budgets, not two implementations that happen to agree.

    # the EM sparsity sweep of a masked run
    uv run python -m mask_learning_finetuning.eval configs/bad_medical_row_cause.yaml \
        --run-dir /mnt/data/.../runs/bad_medical_row_cause

    # just MMLU, on a mid-run checkpoint, over a coarser grid
    uv run python -m mask_learning_finetuning.eval configs/bad_medical_row_cause.yaml \
        --run-dir ... --checkpoint ckpt_step200.pt --only mmlu --fracs 0.01,0.1,1.0
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from ..config import load_config
from ..masks import build_alias_map, load_checkpoint, parse_fracs
from ..masks.checkpoint import layout_from_blob
from . import get_eval
from .runner import MaskedWeights, log_results, sweep, write_json

logger = logging.getLogger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser(prog="mask_learning_finetuning.eval", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="YAML config; its eval: block selects the evals")
    p.add_argument("--run-dir", required=True, help="a training run's output directory")
    p.add_argument("--checkpoint", default=None, help="default: final.pt")
    p.add_argument("--out", default=None, help="default: <run-dir>/posthoc_eval")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these eval names")
    p.add_argument("--fracs", default=None, help="override the sparsity grid, comma-separated")
    p.add_argument("--mode", default=None, help="override the run's mask mode")
    p.add_argument("--dtype", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    out_dir = Path(args.out or Path(args.run_dir) / "posthoc_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # A masked run has a .pt with scores+delta; a plain run has an HF model directory and
    # nothing to sweep, so it evaluates as the single `dense` condition.
    ckpt = Path(args.run_dir) / (args.checkpoint or "final.pt")
    masked = ckpt.exists()
    if masked:
        ckpt, blob = load_checkpoint(args.run_dir, args.checkpoint)
        targs = blob["args"]
        layout = layout_from_blob(blob)
        from learning_to_attribute import normalize_mode
        mode = normalize_mode(args.mode or targs.get("mode", "cause"))
        model_id = targs["model"]
        logger.info("%s: %s, mode=%s", ckpt.name, layout.summary(), mode)
    else:
        model_dir = Path(args.run_dir) / "model"
        if not model_dir.exists():
            raise SystemExit(
                f"neither {ckpt} (a masked checkpoint) nor {model_dir} (a plain finetune's "
                "saved model) exists, so there is nothing to evaluate")
        model_id, layout, mode = str(model_dir), None, "necessary"
        logger.info("%s: plain finetune, no mask -- evaluating the dense weights", model_dir)

    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[
        args.dtype or (blob["args"].get("dtype", "bfloat16") if masked else "bfloat16")]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(cfg.device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    fracs = parse_fracs(args.fracs) if args.fracs else (cfg.eval.fracs or None)
    if masked:
        weights = MaskedWeights(model, tokenizer, device=cfg.device, layout=layout,
                                scores=blob["scores"], deltas=blob["delta"],
                                aliases=build_alias_map(model), mode=mode, fracs=fracs)
    else:
        weights = MaskedWeights(model, tokenizer, device=cfg.device)

    evals, probes = [], {}
    for name, sub in cfg.eval.enabled():
        if args.only and name not in args.only:
            continue
        ev = get_eval(name)
        kw = {}
        if name == "sft_loss":
            from .sft_loss import loaders_from_checkpoint
            src = blob["args"] if masked else {"dataset": cfg.data.train,
                                               "seed": cfg.train.seed,
                                               "test_frac": cfg.data.test_frac,
                                               "max_seq_length": cfg.data.max_seq_length}
            kw["loaders"] = loaders_from_checkpoint(src, tokenizer,
                                                    batch_size=cfg.train.batch_size)
        elif name == "mmlu":
            kw["device"] = cfg.device
        elif name == "em":
            sub.out_dir = sub.out_dir or str(out_dir)
        probe = ev.build(tokenizer, sub, train_data=None, **kw)
        if probe is None:
            continue
        evals.append(ev)
        probes[name] = probe
    if not evals:
        raise SystemExit("no evals selected; add an eval: block to the config, or drop --only")

    res = sweep(evals, probes, weights)
    log_results(res, prefix="posthoc")
    write_json(out_dir / "evals.json", res,
               meta={"checkpoint": str(ckpt) if masked else model_id, "mode": mode,
                     "masked": masked, "fracs": list(fracs) if fracs else None,
                     "total_units": layout.total if masked else None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
