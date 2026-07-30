"""``python -m mask_learning_finetuning.eval <config.yaml> --run-dir DIR`` -- post-hoc sweep.

Evaluates a finished run's checkpoint across the sparsity grid, using the ``eval:`` block of a
config file. This is the same code path the in-training hooks use -- the runner, the probes and
the metrics are shared -- so an inline number and a post-hoc one are the same measurement at
different budgets, not two implementations that happen to agree.

    # the EM sparsity sweep of a masked run
    uv run python -m mask_learning_finetuning.eval configs/bad_medical/cotrain/row_cause.yaml \
        --run-dir /mnt/data/.../runs/bad_medical_row_cause

    # just MMLU, on a mid-run checkpoint, over a coarser grid
    uv run python -m mask_learning_finetuning.eval configs/bad_medical/cotrain/row_cause.yaml \
        --run-dir ... --checkpoint ckpt_step200.pt --only mmlu --fracs 0.01,0.1,1.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ..config import load_config
from ..masks import build_alias_map, load_checkpoint, parse_fracs
from ..masks.checkpoint import layout_from_blob
from . import get_eval
from .runner import MaskedWeights, dump_records, log_results, sweep, write_json

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

    # Three things a run directory can hold, and each says how to evaluate it:
    #   final.pt   a masked run -- scores + delta, so there is a grid to sweep
    #   model/     a full finetune (or a merged LoRA one): plain weights, one `dense` condition
    #   adapter/   an unmerged LoRA run: the base model plus the adapter, also one condition
    ckpt = Path(args.run_dir) / (args.checkpoint or "final.pt")
    model_dir = Path(args.run_dir) / "model"
    adapter_dir = Path(args.run_dir) / "adapter"
    masked, adapter = ckpt.exists(), None
    if masked:
        # require_delta=False: a run saved with save_delta: false (GRPO, co-train) has scores but
        # no delta. The delta is theta(finetuned) - theta(base), fully determined by the two
        # checkpoints, so it is rebuilt below rather than demanded to have been persisted.
        ckpt, blob = load_checkpoint(args.run_dir, args.checkpoint, require_delta=False)
        targs = blob["args"]
        layout = layout_from_blob(blob)
        from learning_to_attribute import normalize_mode
        mode = normalize_mode(args.mode or targs.get("mode", "cause"))
        model_id = tok_id = targs["model"]
        logger.info("%s: %s, mode=%s", ckpt.name, layout.summary(), mode)
    elif model_dir.exists():
        model_id = tok_id = str(model_dir)
        layout, mode = None, "necessary"
        logger.info("%s: plain finetune, no mask -- evaluating the dense weights", model_dir)
    elif (adapter_dir / "adapter_config.json").exists():
        adapter, layout, mode = adapter_dir, None, "necessary"
        # the base model comes from the adapter's own config -- the one it was actually trained
        # over -- rather than from the config file passed in, which may be a different experiment
        ac = json.loads((adapter_dir / "adapter_config.json").read_text())
        model_id = ac.get("base_model_name_or_path") or cfg.model
        tok_id = str(adapter_dir)          # the tokenizer the run trained with, saved alongside
        logger.info("%s: LoRA adapter over %s, no mask -- evaluating the dense weights",
                    adapter_dir, model_id)
    else:
        raise SystemExit(
            f"none of {ckpt} (a masked checkpoint), {model_dir} (a full finetune's saved model) "
            f"or {adapter_dir} (a LoRA adapter) exists, so there is nothing to evaluate")

    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[
        args.dtype or (blob["args"].get("dtype", "bfloat16") if masked else "bfloat16")]
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Same seam as train/loop.py's load_model: one template for every renderer in this process.
    # It has to be the same VALUE the run trained under, which is why it comes from the config
    # rather than being decided here -- a post-hoc sweep that renders prompts differently from the
    # training run is measuring a different model.
    from ..data import install_chat_template
    install_chat_template(tokenizer, cfg.chat_template)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(cfg.device).eval()
    if adapter is not None:
        from peft import PeftModel
        # left unmerged: every eval here either generates (PeftModel.generate delegates) or runs
        # a forward, so merging would only trade a rounding question for nothing
        model = PeftModel.from_pretrained(model, str(adapter)).to(cfg.device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    fracs = parse_fracs(args.fracs) if args.fracs else (cfg.eval.fracs or None)
    # The engine is built from the BASE model id, not the checkpoint: every condition overwrites
    # its weights anyway, so all that has to match is the architecture and the tokenizer.
    engine = None
    if cfg.eval.vllm is not None:
        from .vllm_gen import build as build_engine
        engine = build_engine(cfg.eval.vllm, model_id, tokenizer)
    if masked:
        # The delta is theta(finetuned) - theta(base). A run saved with save_delta: false (GRPO,
        # co-train) persisted only the scores, but both endpoints are recorded in the checkpoint's
        # args, so rebuild it from those rather than requiring a flag or a retrain -- it is
        # deterministic given the two checkpoints.
        svd_factors = {}
        if "delta" in blob:
            deltas = blob["delta"]
            from ..masks.svd import from_blob as svd_from_blob
            svd_factors = svd_from_blob(blob.get("svd"))
        else:
            ft = targs.get("finetuned")
            if not ft:
                raise SystemExit(
                    f"{ckpt} has no delta and its args record no `finetuned` to rebuild it from. "
                    "Re-run with save_delta: true, or point --checkpoint at one that has the delta.")
            from ..masks import resolve_dtype
            from ..train.posthoc import build_deltas, load_finetuned
            logger.info("checkpoint saved scores only; rebuilding delta = (%s) - (%s)", ft, model_id)
            ft_model, _ = load_finetuned(model_id, ft, dtype=dtype)
            deltas, _ = build_deltas(dict(model.named_parameters()), ft_model, layout,
                                     dtype=resolve_dtype(blob["args"].get("delta_dtype"))
                                     or torch.float32)
            del ft_model
            if layout.svd_names:
                # A svd* layout's scores are indexed by singular direction, so the rebuilt delta
                # has to be re-factored to the SAME per-tensor ranks the run fitted -- which the
                # layout records as its unit counts. Doing it at the layout's ranks rather than
                # from a fresh tolerance is what keeps score i meaning direction i.
                from ..masks.svd import factors_for_layout
                logger.info("re-factoring %d tensor(s) at the checkpoint's own ranks",
                            len(layout.svd_names))
                # decomposed on the GPU, KEPT on the CPU beside the deltas: everything
                # apply_in_place touches has to be on one device (see masks.compose)
                svd_factors = factors_for_layout(deltas, layout, work_device=cfg.device,
                                                 device="cpu")
                deltas = {n: d for n, d in deltas.items() if n not in svd_factors}
        weights = MaskedWeights(model, tokenizer, device=cfg.device, layout=layout,
                                scores=blob["scores"], deltas=deltas, svd=svd_factors,
                                aliases=build_alias_map(model), mode=mode, fracs=fracs,
                                engine=engine,
                                # what the run composed at, not what this eval's config says
                                compose_dtype=blob["args"].get("delta_dtype"))
    else:
        weights = MaskedWeights(model, tokenizer, device=cfg.device, engine=engine)

    evals, probes = [], {}
    for name, sub in cfg.eval.enabled():
        if args.only and name not in args.only:
            continue
        ev = get_eval(name)
        kw = {}
        if name == "sft_loss":
            from .sft_loss import loaders_from_checkpoint
            src = blob["args"] if masked else {
                "dataset": cfg.data.train,
                "seed": cfg.train.seed,
                "test_frac": cfg.data.test_frac,
                "max_seq_length": cfg.data.max_seq_length,
                # must be carried, not defaulted: an inoculated finetune scored on un-prefixed
                # prompts reports a worse loss for a reason that is not the model's
                "inoculation_prompt": cfg.data.inoculation_prompt}
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
    # the generations behind the percentages, for the evals that keep them (language,
    # strongreject). Written here as well as in the training loop: the post-hoc sweep is where a
    # generative eval usually runs, and a harmfulness or language rate cannot be checked without
    # the text it was computed from.
    dump_records(evals, probes, out_dir)
    write_json(out_dir / "evals.json", res,
               meta={"checkpoint": str(ckpt if masked else adapter or model_id), "mode": mode,
                     "masked": masked, "base_model": model_id if adapter else None,
                     "fracs": list(fracs) if fracs else None,
                     "total_units": layout.total if masked else None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
