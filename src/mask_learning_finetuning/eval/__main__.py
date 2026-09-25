"""``python -m mask_learning_finetuning.eval <config.yaml> --run-dir DIR`` -- post-hoc sweep.

Evaluates a finished run's checkpoint across the sparsity grid, using the ``eval:`` block of a
config file. This is the same code path the in-training hooks use -- the runner, the probes and
the metrics are shared -- so an inline number and a post-hoc one are the same measurement at
different budgets, not two implementations that happen to agree.

    # the EM sparsity sweep of a masked run
    uv run python -m mask_learning_finetuning.eval configs/bad_medical/cotrain/row_cause.yaml \
        --run-dir runs/bad_medical_row_cause

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
from ..data import build_splits, load_conversations
from ..masks import DEFAULT_EVAL_FRACS, build_alias_map, load_checkpoint, parse_fracs
from ..masks.checkpoint import layout_from_blob
from ..paths import run_path
from . import get_eval
from .runner import (
    MaskedWeights, dump_records, log_results, sweep, sweep_aucs, write_json,
)

logger = logging.getLogger(__name__)


def align_adapter_devices(model) -> int:
    """Move each LoRA submodule onto the device of the base layer it wraps. Returns how many moved.

    `PeftModel.from_pretrained` places adapter weights by its own rule, which is right when the base
    model is on one device and wrong when `device_map` has spread the base across several: the
    adapter for a block on cuda:1 can land on cuda:0, and the forward then dies with "mat1 is on
    cuda:1, different from other tensors on cuda:0" -- an error that names neither PEFT nor the
    device map and appears only after the model has loaded.

    accelerate's hooks do not cover this, because they were installed on the BASE modules when the
    map was applied; a `lora_A`/`lora_B` added afterwards is simply a new parameter the hooks never
    saw. So the alignment has to happen once, here, after the adapter is attached.

    Walking `base_layer` rather than a name pattern is what makes it version-agnostic: PEFT's
    wrapper classes have changed names across releases, but every one of them keeps the wrapped
    module under that attribute, and it is the module whose device is authoritative.
    """
    moved = 0
    for mod in model.modules():
        base = getattr(mod, "base_layer", None)
        if base is None:
            continue
        try:
            dev = next(base.parameters()).device
        except StopIteration:
            continue
        for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B",
                     "lora_magnitude_vector"):
            sub = getattr(mod, attr, None)
            if sub is None:
                continue
            if any(p.device != dev for p in sub.parameters()):
                sub.to(dev)
                moved += 1
    logger.info("aligned %d adapter submodule(s) onto their base layers' shards", moved)
    return moved



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
    p.add_argument("--device-map", default=None,
                   help="override train.device_map (e.g. `auto` to shard a run that fitted on one "
                        "card: the sweep's functional composition needs base + delta + theta_eff "
                        "resident, which at 14B is three model-sizes and OOMs one 80 GB card)")
    p.add_argument("--loss-batches", type=int, default=None,
                   help="override eval.sft_loss's batch count for this sweep (0 = whole split)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.device_map is not None:
        cfg.train.device_map = args.device_map
    args.run_dir = str(run_path(args.run_dir))   # `runs/<name>` -> $MLFT_RUNS_ROOT or <repo>/runs
    out_dir = run_path(args.out) if args.out else Path(args.run_dir) / "posthoc_eval"
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
        from matryoshka_attribution import normalize_mode
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
    trc = cfg.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True, trust_remote_code=trc)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Same seam as train/loop.py's load_model: one template for every renderer in this process.
    # It has to be the same VALUE the run trained under, which is why it comes from the config
    # rather than being decided here -- a post-hoc sweep that renders prompts differently from the
    # training run is measuring a different model.
    from ..data import install_chat_template
    install_chat_template(tokenizer, cfg.chat_template, system_prompt=cfg.system_prompt,
                          template_kwargs=cfg.chat_template_kwargs)
    # `train.device_map` has to be honoured HERE too, not only in train/loop.py: this driver
    # rebuilds the same masked model, so it carries the same three-model-sized footprint (base +
    # delta + composed theta_eff) that does not fit on one 80 GB card at 14B. Without this the CLI
    # silently loads onto `cfg.device` and dies in composition, having already done the expensive
    # part -- which is exactly how it was found. `.to()` must not be called on a sharded model.
    if cfg.train.device_map:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                                                     device_map=cfg.train.device_map,
                                                     trust_remote_code=trc).eval()
        logger.info("model sharded over %d device(s)",
                    len({str(p.device) for p in model.parameters()}))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                                                     trust_remote_code=trc).to(cfg.device).eval()
    if adapter is not None:
        from peft import PeftModel
        # left unmerged: every eval here either generates (PeftModel.generate delegates) or runs
        # a forward, so merging would only trade a rounding question for nothing
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
        if cfg.train.device_map:
            align_adapter_devices(model)
        else:
            model = model.to(cfg.device)
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
            ft_model, _ = load_finetuned(model_id, ft, dtype=dtype, trust_remote_code=trc)
            if targs.get("fold_params"):
                # the run folded these tensors' finetuned values into its base (MaskCfg.fold_params);
                # the same fold here, or the anchors are not the run's anchors
                import re
                from ..train.posthoc import fold_finetuned
                excl = re.compile(targs["exclude_params"]) if targs.get("exclude_params") else None
                fold = re.compile(targs["fold_params"])
                fold_finetuned(model, ft_model, [n for n, _ in model.named_parameters()
                                                 if fold.search(n) and not (excl and excl.search(n))])
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
                                compose_dtype=blob["args"].get("delta_dtype"),
                                # the CLI's delta is always frozen (it came off a checkpoint),
                                # so the knob applies exactly as in the training loop's sweeps
                                inplace_device=cfg.eval.inplace_compose)
    else:
        weights = MaskedWeights(model, tokenizer, device=cfg.device, engine=engine)

    # THE HELD-OUT CONVERSATIONS, REBUILT -- and this used to be `train_data=None`, which made
    # the post-hoc CLI silently drop the `in_dist` split of every generative eval. The training
    # driver hands `build_evals` its `held_convs`, so a sweep run through train() has in_dist and
    # the same sweep re-run through this CLI did not: a figure built on the two together loses
    # its on-target axis with nothing in any log to say so. Rebuilt from the CHECKPOINT's args
    # where there is one, for the reason `loaders_from_checkpoint` does the same -- the seed
    # drives the carve, so a posthoc run at a different seed would otherwise score a split whose
    # rows were in the finetune's training data. Memoised: several evals ask for it.
    _held = []

    def held_convs():
        if not _held:
            src = blob["args"] if masked else {}
            path = src.get("dataset", cfg.data.train)
            seed = src.get("seed", cfg.train.seed)
            tf = src.get("test_frac", cfg.data.test_frac)
            if not path or not tf:
                _held.append(None)
            else:
                convs = load_conversations(path, field=cfg.data.field_name,
                                           limit=cfg.data.limit)
                _held.append(build_splits(convs, seed=seed, test_frac=tf,
                                          test_file=cfg.data.test_file,
                                          field=cfg.data.field_name)[1] or None)
        return _held[0]

    evals, probes = [], {}
    for name, sub in cfg.eval.enabled():
        if args.only and name not in args.only:
            continue
        ev = get_eval(name)
        kw = {}
        if name == "sft_loss":
            from .sft_loss import loaders_from_checkpoint
            if args.loss_batches is not None:
                sub.n_batches = sub.final_n_batches = args.loss_batches
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
        # probe_inoc, same hook as train/loop.py's build_evals: the run's inoculation prompt goes
        # to any eval whose config declares the field, and to nothing else
        if hasattr(sub, "inoculation_prompt") and sub.inoculation_prompt is None:
            sub.inoculation_prompt = cfg.data.inoculation_prompt
        probe = ev.build(tokenizer, sub, train_data=held_convs(), **kw)
        if probe is None:
            continue
        evals.append(ev)
        probes[name] = probe
    if not evals:
        raise SystemExit("no evals selected; add an eval: block to the config, or drop --only")

    # final=True: this sweep IS a run's end-of-run sweep, re-done, so `sft_loss` has to score at
    # `final_n_batches` the way the training loop's final sweep does. Without it the CLI scored
    # at `n_batches` (8-16 examples) and a re-eval's loss silently disagreed with the run's own
    # over identical weights (1.312 vs 1.416 held-out on the same full delta, 2026-09-08).
    res = sweep(evals, probes, weights, final=True)
    log_results(res, prefix="posthoc")
    # the same one-number-per-curve summary the training driver logs to wandb; printed here
    # because this driver has no wandb run, and written into the JSON so it is not recomputed
    aucs = sweep_aucs(res, fracs or DEFAULT_EVAL_FRACS, prefix="posthoc")
    for k in sorted(aucs):
        if k.endswith("_log_auc"):
            logger.info("auc %s = %.4f", k, aucs[k])
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
