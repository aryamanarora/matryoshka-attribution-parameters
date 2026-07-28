"""Learn a MAttr parameter mask over a FIXED delta between two checkpoints.

`finetune_masked.py` learns the delta and the scores jointly. This script does the post-hoc
half: you hand it a pretrained checkpoint and a finetuned one, it takes

    delta = theta_finetuned - theta_base

as a *constant*, and trains only the per-unit scores s:

    theta_eff = theta_base + m(s, k) . delta          (cause / necessary -- the default)
    theta_eff = theta_base + (1 - m(s, k)) . delta    (iso / sufficient)

Nothing about the model is trainable: the base weights are frozen, the delta is frozen, and
the single leaf tensor in the graph is ``scores``. Every step still samples k from the
k-schedule, builds the differentiable top-k mask, composes theta_eff and backprops the SFT
loss -- but only into s. One score vector has to work at every sparsity, exactly as upstream
MAttr does for activations.

Reading the result: a ranking of parameter units by how much of the *given* finetune's
behaviour they carry. Unlike the joint script, the delta here is a delta someone else
produced (an ordinary unmasked finetune, e.g. one of the EM model organisms), so the ranking
is an attribution of that finetune rather than a property of a co-trained one. That is the
whole point of this script: the joint run's delta is shaped by the mask (see the second note
in `param_masks`' docstring), so it cannot answer "where does *this* released finetune live".

Everything except the parameterisation is imported from `finetune_masked.py` -- ``masked_ce``,
``eval_sweep`` / ``run_sweeps``, ``log_curve_panels``, ``save``, and the sparsity grid -- so
the loss, the eval and the wandb schema are not merely equivalent to the joint script's, they
are the same code. The CLI mirrors it too, minus the flags that only govern delta
optimisation (``--lr``, ``--weight-decay``, ``--warmup-steps``, ``--lr-scheduler``,
``--init-delta``, ``--freeze-delta``): there is no delta optimizer to configure, and
``train/lr`` logs the (constant) score lr, which is the only lr in use.

**MMLU alongside the loss.** SFT loss says how much of the training objective a top-k slice
reproduces; it says nothing about what the slice costs elsewhere. So the same sweep also
scores a small subject-stratified MMLU subsample at every point on the grid (``--mmlu-limit``,
0 to switch off), giving a capability curve next to the loss curve at no extra composition.
The probe -- its flags, question selection, k-shot prompt and single-token letter scoring --
is imported wholesale from `eval_mmlu_sparsity.py` rather than restated, so a small
in-training subsample and a full post-hoc run of that script are the same measurement at
different n, and identical to what the joint script reports. The one difference is
mechanical: scoring here goes through ``functional_call`` on composed parameters, like the
loss sweep, instead of writing weights into the live model.

**LoRA adapters are accepted directly.** The EM organisms are published as PEFT adapters, not
full checkpoints (``ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice`` is r=32
rsLoRA over ``unsloth/Llama-3.2-1B-Instruct``), so if ``--finetuned-model`` holds an
``adapter_config.json`` it is merged onto ``--model`` in memory and the merged weights become
the finetuned side. Two details this gets right that a hand-rolled ``B @ A`` would not:

* The merge is PEFT's own ``merge_and_unload``. These adapters set ``use_rslora``, whose
  scaling is ``alpha / sqrt(r)`` rather than ``alpha / r``; getting that wrong silently
  rescales the entire delta.
* It merges onto ``--model`` rather than the adapter's declared base, in **float32** even when
  ``--dtype`` is bf16. So the delta is *exactly* the adapter's update, by construction, and
  none of it is lost to rounding -- a LoRA update is often smaller than bf16's ~3 significant
  digits at the magnitude of the weight it lands on, so forming the sum in bf16 first would
  discard part of it. ``compose_params`` still casts ``theta_base + m . delta`` back to
  ``--dtype`` at the end, so the full-mask point remains the model you would actually serve.
  A name mismatch against the adapter's declared base is reported, not silently accepted.

Three consequences of a frozen, nonzero delta, all differences from the joint run:

1. Scores get gradient from step 0. In the joint run delta starts at zero, so ``dL/ds = 0``
   until the delta grows; here the delta is fully formed from the first step, and mask
   learning is the only thing happening.
2. The ``pretrained`` and ``full_delta`` anchors of the sweep are *constants* of the run --
   they are the two given checkpoints' SFT losses and do not depend on s at all. They should
   be identical at every eval; drift means something is wrong with the composition. The
   step-0 sweep is therefore not flat (unlike the joint run's, which is flat because the
   delta is zero) -- it is the curve you get from an arbitrary ranking, i.e. the baseline the
   learned one has to beat.
3. Units whose delta is exactly zero receive exactly zero gradient forever and keep their
   init score, so they rank arbitrarily. Their count is logged at startup.

Examples
--------
    # smoke: gpt2 vs a perturbed gpt2, coarse units, a few steps
    uv run python scripts/learn_mask.py --model gpt2 --finetuned-model /tmp/gpt2-ft \\
        --dataset data/toy_chat.jsonl --unit tensor --loss-mask all --max-steps 20 \\
        --batch-size 2 --grad-accum 2 --output results/smoke_maskonly

    # attribute a released EM organism, one score per output feature
    uv run python scripts/learn_mask.py \\
        --model unsloth/Llama-3.2-1B-Instruct \\
        --finetuned-model ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice \\
        --dataset data/em/bad_medical_advice.jsonl \\
        --unit row --k-schedule log --save-delta --output results/badmed_maskonly

The resulting run directory is the same shape `finetune_masked.py` writes, so
`eval_em_sparsity.py --run-dir ...` works against it unchanged (with ``--save-delta``, which
that eval needs -- it masks the delta, and here the delta is not otherwise recoverable
without both checkpoints).
"""

import argparse
import gc
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import build_mask, sample_k, normalize_mode, MODE_CHOICES
from learning_to_attribute.masks import VARIANTS

from mask_learning_finetuning.data import (
    CHAT_TEMPLATE_MODES, ChatSFTDataset, collate, load_conversations,
)
from mask_learning_finetuning.masks import (
    UNIT_MODES, build_alias_map, build_layout, compose_params, unit_norms,
)

# The joint script is the reference implementation of the loss, the eval sweep, the wandb
# panels and the checkpoint format; import them rather than restate them. scripts/ is not a
# package, so put its directory on the path explicitly (sys.path[0] already is it when this
# file is run directly, but not when it is imported).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_masked import (                                        # noqa: E402
    DEFAULT_EVAL_FRACS, em_hook, log_curve_panels, masked_ce, run_sweeps, save,
)
# ... the EM sweep's flags come from the post-hoc script that owns the protocol, same
# reasoning as the MMLU import below.
from eval_em_sparsity import add_em_args, finalize_em_args           # noqa: E402
# ... and the MMLU probe from the post-hoc MMLU sweep, for the same reason: an in-training
# subsample and a full run of that script must be the same measurement.
from eval_mmlu_sparsity import (                                     # noqa: E402
    add_mmlu_args, build_mmlu_probe, finalize_mmlu_args, mmlu_hook, write_mmlu_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- the two checkpoints ---
    p.add_argument("--model", required=True,
                   help="HF id or path of the PRETRAINED checkpoint (theta_base)")
    p.add_argument("--finetuned-model", required=True,
                   help="HF id or path of the FINETUNED checkpoint; "
                        "delta = its weights - theta_base. A PEFT/LoRA adapter directory or "
                        "repo works too: it is merged onto --model in fp32 in memory")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer to render the data with (default: --model's). Override if "
                        "the two checkpoints ship different chat templates")
    p.add_argument("--revision", default=None, help="revision for --model")
    p.add_argument("--finetuned-revision", default=None,
                   help="revision for --finetuned-model")

    # --- what to train on ---
    p.add_argument("--dataset", required=True,
                   help="local .jsonl (one conversation per line) or HF dataset id. Should be "
                        "the data the finetune was produced with: the loss being attributed "
                        "is the SFT loss on it")
    p.add_argument("--dataset-field", default="messages")
    p.add_argument("--limit", type=int, default=None, help="cap number of examples")
    p.add_argument("--output", required=True, help="output directory")

    # --- masking / attribution (identical to the joint script) ---
    p.add_argument("--unit", default="row", choices=UNIT_MODES,
                   help="granularity of one score (default: row = per output feature)")
    p.add_argument("--variant", default="topk", choices=VARIANTS,
                   help="mask variant (default topk = soft forward, the MAttr headline)")
    p.add_argument("--k-schedule", default="log", choices=["log", "uniform", "log_both"])
    p.add_argument("--k-fixed", type=float, default=None,
                   help="train at this fixed k instead of sampling from the schedule")
    p.add_argument("--mode", default="cause", choices=MODE_CHOICES,
                   help="cause/necessary (default): top-k carries the delta. "
                        "iso/sufficient: delta on the complement, top-k retained")
    p.add_argument("--score-lr", type=float, default=0.05, help="Adam lr for scores")
    p.add_argument("--T", type=float, default=0.5, help="sigmoid temperature")
    p.add_argument("--n-iters", type=int, default=50, help="bisection iterations")
    p.add_argument("--exclude-params", default=None,
                   help="regex of parameter names to leave frozen (no delta, no score). "
                        "Those tensors keep their PRETRAINED value, so the excluded part of "
                        "the finetune is discarded rather than always-on")

    # --- SFT-side hyperparameters (defaults = the reference full-ft_config.json) ---
    p.add_argument("--batch-size", type=int, default=2, help="per-device micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--delta-dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="dtype the frozen delta is held in. float32 (default) matches the "
                        "joint script's fp32 deltas exactly; bfloat16 halves its memory and "
                        "changes only the precision of the mask multiply, since the composed "
                        "weights are cast to --dtype either way")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chat-template-mode", default="standard", choices=CHAT_TEMPLATE_MODES)
    p.add_argument("--loss-mask", default="response_only",
                   choices=["response_only", "all"])
    p.add_argument("--dropout", action="store_true",
                   help="run the model in train() mode. Off by default: a deterministic "
                        "forward makes the score gradient far less noisy, and the modern "
                        "chat models this targets ship with dropout=0 anyway")

    # --- early stop (reference: EarlyStoppingOnLowLossCallback) ---
    p.add_argument("--early-stop-loss", type=float, default=0.01)
    p.add_argument("--early-stop-steps", type=int, default=5)

    # --- eval / logging ---
    p.add_argument("--eval-every", type=int, default=0, help="0 disables mid-run eval")
    p.add_argument("--eval-batches", type=int, default=8,
                   help="batches per mid-run eval (kept small: it runs 24 masked sweeps)")
    p.add_argument("--final-eval-batches", type=int, default=0,
                   help="batches for the end-of-training eval; 0 = the ENTIRE split")
    p.add_argument("--save-every", type=int, default=0,
                   help="checkpoint scores every N steps; 0 = only at the end")
    p.add_argument("--save-delta", action="store_true",
                   help="include the delta in the FINAL save. One full model's worth of "
                        "--delta-dtype, so mind the disk -- but eval_em_sparsity.py and the "
                        "rest of sweep.py need it")
    p.add_argument("--save-delta-intermediate", action="store_true",
                   help="also write the delta at every --save-every checkpoint")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="goodfire")
    p.add_argument("--wandb-project", default="mask-learning-finetuning")
    p.add_argument("--wandb-name", default=None, help="run name (default: auto from config)")
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="held-out fraction. 0.1 + the run seed matches the reference's "
                        "train_test_split(test_size=0.1, seed=seed)")
    p.add_argument("--test-file", default=None,
                   help="explicit held-out .jsonl. The reference still carves 10%% off train "
                        "when given one, so the train set matches across test sets; done here "
                        "too")

    add_mmlu_args(p)
    add_em_args(p)

    args = p.parse_args()
    args.mode = normalize_mode(args.mode)     # -> "sufficient" | "necessary"
    finalize_mmlu_args(args)
    finalize_em_args(args)
    # Same refusal as the joint script: these two need the REINFORCE gradient / L0 penalty
    # that learn_scores applies to build_mask's aux fields, which this loop does not implement.
    if args.variant in ("bernoulli_reinforce", "hard_concrete"):
        p.error(f"--variant {args.variant} needs the REINFORCE/L0 handling in "
                "learn_scores, which this loop does not implement")
    return args


def is_adapter(path_or_id: str) -> bool:
    """Does this point at a PEFT adapter rather than a full checkpoint?"""
    p = Path(path_or_id)
    if p.exists():
        return (p / "adapter_config.json").exists()
    from huggingface_hub import file_exists
    return file_exists(path_or_id, "adapter_config.json")


def load_finetuned(args):
    """The finetuned model, on the CPU in fp32, merging a LoRA adapter first if that is what
    ``--finetuned-model`` is. Returns ``(model, provenance)``.

    See the module docstring for why the merge happens onto ``--model`` and in fp32.
    """
    kw = {"revision": args.finetuned_revision} if args.finetuned_revision else {}
    if not is_adapter(args.finetuned_model):
        logger.info("loading finetuned checkpoint %s (on CPU, fp32) ...", args.finetuned_model)
        model = AutoModelForCausalLM.from_pretrained(args.finetuned_model,
                                                     dtype=torch.float32, **kw)
        return model, {"kind": "full_checkpoint", "path": args.finetuned_model}

    from peft import PeftConfig, PeftModel

    cfg = PeftConfig.from_pretrained(args.finetuned_model, **kw)
    prov = {"kind": "merged_adapter", "adapter": args.finetuned_model,
            "peft_type": str(cfg.peft_type), "r": getattr(cfg, "r", None),
            "lora_alpha": getattr(cfg, "lora_alpha", None),
            "use_rslora": getattr(cfg, "use_rslora", None),
            "declared_base": cfg.base_model_name_or_path,
            "merged_onto": args.model,
            "target_modules": sorted(cfg.target_modules or [])}
    logger.info("%s is a PEFT adapter: %s r=%s alpha=%s rslora=%s targets=%s",
                args.finetuned_model, prov["peft_type"], prov["r"], prov["lora_alpha"],
                prov["use_rslora"], ",".join(prov["target_modules"]))
    if cfg.base_model_name_or_path and cfg.base_model_name_or_path != args.model:
        logger.warning("the adapter declares base %s but --model is %s. Merging onto --model, "
                       "so the delta is exactly the adapter's update either way -- but if the "
                       "two bases are not weight-identical, this is not the finetune that was "
                       "released", cfg.base_model_name_or_path, args.model)
    logger.info("merging the adapter onto %s in fp32 (on CPU) ...", args.model)
    kwm = {"revision": args.revision} if args.revision else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, **kwm)
    model = PeftModel.from_pretrained(model, args.finetuned_model, **kw)
    return model.merge_and_unload(), prov


def build_deltas(args, base, layout, delta_dtype):
    """``theta_finetuned - theta_base`` for every scored tensor, as frozen tensors on-device.

    The finetuned checkpoint is loaded on the CPU, subtracted tensor by tensor, and dropped
    again, so peak memory is one CPU copy of it plus the delta -- not a second live model on
    the accelerator.

    Both checkpoints are read through ``named_parameters()``, which deduplicates tied weights
    exactly as ``build_layout``/``build_alias_map`` do, so the two sides line up by name. The
    one case that does not line up is a finetune that *untied* the embeddings: then its
    ``lm_head.weight`` is a genuinely separate tensor with no counterpart in the layout, and
    the alias map would push the embedding delta onto the output head instead. That is
    reported rather than silently absorbed.

    Returns ``(deltas, provenance)``; the provenance says whether the finetuned side was a
    full checkpoint or a merged adapter, and is written into the run directory.
    """
    ft, prov = load_finetuned(args)
    ft_params = dict(ft.named_parameters())

    missing = [n for n in layout.names if n not in ft_params]
    if missing:
        raise SystemExit(
            f"{len(missing)} scored tensors are absent from {args.finetuned_model}, e.g. "
            f"{missing[:3]}. The two checkpoints must be the same architecture; if the "
            "finetune renamed or untied parameters, exclude them with --exclude-params.")
    extra = [n for n in ft_params if n not in base]
    if extra:
        logger.warning(
            "%d tensor(s) exist in the finetuned checkpoint but not in the base model, e.g. "
            "%s. They carry no score and no delta. If this is an untied lm_head, the base "
            "model's tied embedding delta WILL also be applied to the output head via the "
            "alias map, which is not what that checkpoint does -- check before trusting the "
            "numbers.", len(extra), extra[:3])

    deltas, n_changed, sq = {}, 0, 0.0
    for i, name in enumerate(layout.names):
        p = ft_params[name]
        if tuple(p.shape) != tuple(layout.shapes[i]):
            raise SystemExit(f"shape mismatch for {name}: base {tuple(layout.shapes[i])} vs "
                             f"finetuned {tuple(p.shape)}. Resized vocabulary?")
        d = (p.detach().to(device=args.device, dtype=torch.float32)
             - base[name].to(torch.float32))
        sq += float(d.pow(2).sum())
        n_changed += int(bool((d != 0).any()))
        deltas[name] = d.to(delta_dtype).requires_grad_(False)

    del ft, ft_params
    gc.collect()
    prov = {**prov, "n_tensors": len(deltas), "n_tensors_nonzero": n_changed,
            "delta_norm": sq ** 0.5, "delta_dtype": args.delta_dtype}
    logger.info("delta over %d tensors (%d nonzero), ||delta||=%.4g, held in %s",
                len(deltas), n_changed, sq ** 0.5, delta_dtype)
    return deltas, prov


def unit_delta_norms(deltas, layout) -> torch.Tensor:
    """Per-unit L2 norm of the delta, flat and aligned with the score vector.

    Diagnostic only, never part of the objective: it separates "this unit was not moved by
    the finetune" (norm 0, hence no score gradient ever) from "this unit was moved but the
    mask learned to drop it", which is the interesting case and indistinguishable from the
    former if you only look at the scores.
    """
    out = torch.zeros(layout.total)
    for i, name in enumerate(layout.names):
        # Reduce along the layout's OWN stored axis, not a guess from layout.mode. Under
        # `nonresid` the axis differs per tensor, so a mode-based branch silently reduced
        # every down_proj along the wrong axis -- and hit a shape mismatch on the 1-D norm
        # gains, which that mode gives a single unit rather than one per element.
        out[layout.slice_for(i)] = unit_norms(deltas[name].detach().float().cpu(),
                                              layout.axes[i])
    return out


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation, torch-only (no scipy), ties broken by sort order."""
    if a.numel() < 2:
        return float("nan")
    rank = lambda t: torch.empty_like(t).scatter_(
        0, t.argsort(), torch.arange(t.numel(), dtype=t.dtype))
    ra, rb = rank(a.float().flatten()), rank(b.float().flatten())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb) / denom) if float(denom) > 0 else float("nan")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- base model (frozen; its parameters are theta_base) ----
    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16,
                 float32=torch.float32)[args.dtype]
    delta_dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16,
                       float16=torch.float16)[args.delta_dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kw = {"revision": args.revision} if args.revision else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kw)
    model.to(args.device)
    model.requires_grad_(False)
    model.train() if args.dropout else model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ---- data (identical split to the joint script) ----
    convs = load_conversations(args.dataset, field=args.dataset_field, limit=args.limit)
    idx = list(range(len(convs)))
    random.Random(args.seed).shuffle(idx)
    n_test = int(round(args.test_frac * len(convs)))
    train_convs = [convs[i] for i in idx[n_test:]]
    held_convs = [convs[i] for i in idx[:n_test]]
    if args.test_file:
        held_convs = load_conversations(args.test_file, field=args.dataset_field)

    mk = lambda cs: ChatSFTDataset(tokenizer, cs, max_length=args.max_seq_length,
                                   template_mode=args.chat_template_mode,
                                   supervise_all=(args.loss_mask == "all"))
    ds, test_ds = mk(train_convs), (mk(held_convs) if held_convs else None)
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh, drop_last=False,
                                  collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    loader = dl(ds, True)
    # train-eval loader is unshuffled so the "train" sweep is the same examples every time
    eval_loaders = {"train": dl(ds, False)}
    if test_ds is not None:
        eval_loaders["test"] = dl(test_ds, False)
    logger.info("%d train / %d held-out conversations (%d dropped as fully-masked); "
                "%d supervised train tokens", len(ds), len(test_ds) if test_ds else 0,
                ds.n_dropped + (test_ds.n_dropped if test_ds else 0),
                ds.supervised_tokens())
    logger.info("supervised span of example 0:\n%s", ds.describe(tokenizer, 1)[:600])

    # ---- units, scores, frozen delta ----
    excl = re.compile(args.exclude_params) if args.exclude_params else None
    named = [(n, p) for n, p in model.named_parameters()
             if not (excl and excl.search(n))]
    if not named:
        raise ValueError("--exclude-params excluded every parameter")
    # nonresid needs the residual width to tell which axis of each tensor is the stream
    resid_dim = getattr(model.config, "hidden_size", None) or getattr(
        model.config, "n_embd", None)
    if args.unit == "nonresid" and not resid_dim:
        raise ValueError("could not read hidden_size/n_embd from the model config, which "
                         "--unit nonresid needs to identify the residual-stream axis")
    layout = build_layout(named, args.unit, resid_dim=resid_dim)
    logger.info("mask layout: %s", layout.summary())

    base = {n: p.detach() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    aliases = build_alias_map(model)
    n_tied = sum(len(v) - 1 for v in aliases.values())
    if n_tied:
        logger.info("%d tied parameter alias(es) will receive the same delta "
                    "(e.g. lm_head <- embed_tokens)", n_tied)

    deltas, provenance = build_deltas(args, base, layout, delta_dtype)
    scores = torch.zeros(layout.total, device=args.device, requires_grad=True)

    d_norms = unit_delta_norms(deltas, layout)
    n_dead = int((d_norms == 0).sum())
    n_params = sum(base[n].numel() for n in layout.names)
    logger.info("frozen delta over %s parameters (%.2f GB %s); %s scores (the ONLY trainable "
                "tensor)", f"{n_params:,}", n_params * delta_dtype.itemsize / 1e9,
                args.delta_dtype, f"{layout.total:,}")
    if n_dead:
        logger.warning("%d/%d units (%.2f%%) have an all-zero delta: the finetune did not "
                       "move them, so their scores get zero gradient forever and their rank "
                       "is arbitrary", n_dead, layout.total, 100 * n_dead / layout.total)

    opt_scores = torch.optim.Adam([scores], lr=args.score_lr)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    # --max-steps overrides --epochs, matching HF's TrainingArguments: it is a target, not a
    # cap, and the loop cycles the dataloader to reach it.
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    logger.info("%d optimizer steps (%d micro-batches/step, effective batch %d)",
                total_steps, args.grad_accum, args.batch_size * args.grad_accum)

    run = None
    if args.wandb:
        import os
        import wandb
        # No credentials on this cluster by default. Fall back to offline rather than losing
        # the run: `wandb sync <dir>` uploads it once a key is available.
        if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
            os.environ.setdefault("WANDB_MODE", "offline")
            logger.warning("no WANDB_API_KEY and no ~/.netrc -> logging OFFLINE. "
                           "Run `wandb sync` on the run dir later, or set WANDB_API_KEY.")
        name = args.wandb_name or (f"maskonly-{Path(args.finetuned_model).name}"
                                   f"-{Path(args.dataset).stem}-{args.unit}-{args.mode}")
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         name=name, config=vars(args))
        # step is the optimizer step; mask % is a per-step property worth plotting against
        wandb.define_metric("train/k_frac", summary="mean")

    invert = args.mode == "sufficient"      # iso -> (1 - m); cause -> m

    # ---- MMLU probe (mask-independent, so its prompts are built once) ----
    probe = build_mmlu_probe(tokenizer, args)
    mmlu_history = []

    def mmlu_at(step, final=False):
        """Score the grid and log it; returns the {label: accuracy} dict, or None if off."""
        return mmlu_hook(args, model, base, deltas, layout, scores.detach(), buffers, aliases,
                         probe, mmlu_history, step=step, fracs=DEFAULT_EVAL_FRACS,
                         wandb_run=run, final=final)

    def em_at(step, final):
        """The EM twin of ``mmlu_at``. Off unless --em-sweep; see finetune_masked.em_hook.

        Unlike MMLU this cannot go through ``functional_call``: the reference eval generates,
        and generation needs real parameters. The hook writes them in place and restores.
        """
        return em_hook(args, model, tokenizer, layout, scores, deltas, out_dir,
                       step=step, final=final, wandb_run=run)

    # ---- eval at step 0, before any update ----
    # NOT flat, unlike the joint script's step-0 sweep: the delta is already the full
    # finetune, so this is the loss-vs-sparsity curve of an UNLEARNED ranking (all scores
    # equal, so top-k falls back to index order). It is the baseline the learned mask has to
    # beat, and its two anchors are the two input checkpoints' losses.
    sweep_history = []
    sweeps0 = run_sweeps(model, base, deltas, layout, scores.detach(), buffers, aliases,
                         eval_loaders, args)
    for split, sw in sweeps0.items():
        logger.info("eval @ step 0 [%s]: %s", split,
                    "  ".join(f"{kk}={vv:.3f}" for kk, vv in sw.items()))
    anchors0 = {split: (sw["pretrained"], sw["full_delta"]) for split, sw in sweeps0.items()}
    sweep_history.append((0, sweeps0))
    if run:
        run.log({f"eval/{split}/{kk}": vv
                 for split, sw in sweeps0.items() for kk, vv in sw.items()}, step=0)
        log_curve_panels(run, sweep_history, layout)
    mmlu_at(0)

    def check_anchors(sweeps, step):
        """The anchors are s-independent, so any drift is a composition bug, not learning."""
        for split, sw in sweeps.items():
            for i, kk in enumerate(("pretrained", "full_delta")):
                if abs(sw[kk] - anchors0[split][i]) > 1e-3:
                    logger.warning("%s/%s moved from %.4f to %.4f by step %d -- it cannot "
                                   "depend on the scores; check compose_params", split, kk,
                                   anchors0[split][i], sw[kk], step)

    # ---- train (scores only) ----
    train_log = []
    t0 = time.time()
    low_loss_streak = 0
    step = 0
    it = iter(loader)
    stop = False
    while step < total_steps and not stop:
        # k is drawn ONCE per optimizer step and shared by every micro-batch in it, matching
        # upstream MAttr where k is fixed within a batch.
        k = args.k_fixed if args.k_fixed else sample_k(layout.total, args.k_schedule)
        k = max(1.0, min(float(k), float(layout.total)))

        # Collect the window first so the loss can be normalised by the TOTAL supervised
        # tokens across it. Normalising per micro-batch instead (the common shortcut) makes
        # each micro-batch's mean equally weighted, silently up-weighting short sequences.
        window = []
        for _ in range(args.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            window.append({kk: v.to(args.device) for kk, v in b.items()})
        window_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in window)
        if window_tokens == 0:
            continue

        opt_scores.zero_grad(set_to_none=True)

        total_loss = 0.0
        for b in window:
            # Rebuild the mask per micro-batch: same k, but a fresh graph so each backward
            # has its own (avoids retain_graph). For the stochastic variants (gumbel,
            # bernoulli) this also redraws the noise, which only averages the estimator.
            mask = build_mask(scores, k, args.variant, T=args.T, n_iters=args.n_iters).mask
            params = compose_params(base, deltas, mask, layout, invert=invert,
                                   aliases=aliases)
            ce_sum = masked_ce(model, params, buffers, b, reduction="sum")
            (ce_sum / window_tokens).backward()
            total_loss += float(ce_sum.detach())
        loss = total_loss / window_tokens

        opt_scores.step()

        with torch.no_grad():
            s_std = float(scores.std()) if scores.numel() > 1 else 0.0
            g_norm = float(scores.grad.norm()) if scores.grad is not None else 0.0
        # `lr` is the score lr here (constant, and the only lr in the run), so the panel is
        # the same key the joint script writes -- where it is the delta's schedule instead.
        rec = dict(step=step, k=k, k_frac=k / layout.total, loss=loss, lr=args.score_lr,
                   tokens=window_tokens, score_std=s_std, score_grad_norm=g_norm)
        train_log.append(rec)
        if args.log_every and (step % args.log_every == 0 or step == total_steps - 1):
            logger.info("step %4d/%d  loss=%.4f  k=%.0f/%d (%.2f%%)  "
                        "score_std=%.3e |g_s|=%.3e  %.1fs",
                        step, total_steps, loss, k, layout.total,
                        100 * k / layout.total, s_std, g_norm, time.time() - t0)
        if run:
            run.log({f"train/{kk}": vv for kk, vv in rec.items() if kk != "step"}, step=step)

        # reference behaviour: stop once the loss has been under threshold for a few steps
        low_loss_streak = low_loss_streak + 1 if loss < args.early_stop_loss else 0
        if low_loss_streak > args.early_stop_steps:
            logger.info("early stop: loss < %g for %d consecutive steps",
                        args.early_stop_loss, low_loss_streak)
            stop = True

        step += 1
        if args.eval_every and step % args.eval_every == 0:
            sweeps = run_sweeps(model, base, deltas, layout, scores.detach(), buffers,
                                aliases, eval_loaders, args)
            for split, sw in sweeps.items():
                logger.info("eval @ step %d [%s]: %s", step, split,
                            "  ".join(f"{kk}={vv:.3f}" for kk, vv in sw.items()))
            check_anchors(sweeps, step)
            sweep_history.append((step, sweeps))
            if run:
                run.log({f"eval/{split}/{kk}": vv
                         for split, sw in sweeps.items() for kk, vv in sw.items()}, step=step)
                log_curve_panels(run, sweep_history, layout)
            mmlu_at(step)
            em_at(step, final=False)
        if args.save_every and step % args.save_every == 0:
            save(out_dir / f"ckpt_step{step}.pt", args, layout, scores, deltas, train_log,
                 include_delta=args.save_delta and args.save_delta_intermediate)

    # ---- final eval + save ----
    nb = args.final_eval_batches
    logger.info("final eval over %s of each split ...",
                "the full set" if not nb else f"{nb} batches")
    sweeps = run_sweeps(model, base, deltas, layout, scores.detach(), buffers, aliases,
                        eval_loaders, args, n_batches=nb)
    logger.info("final sparsity sweep (loss vs mask %%):")
    for split, sw in sweeps.items():
        for kk, vv in sw.items():
            logger.info("  %-6s %-14s loss=%.4f", split, kk, vv)
    # The anchors are only comparable to step 0's when measured over the same batches; the
    # final sweep usually uses a different (larger) budget, and comparing across budgets
    # reports a "drift" that is just a different eval set.
    if nb == args.eval_batches:
        check_anchors(sweeps, step)
    else:
        logger.info("anchor-drift check skipped: the final sweep used %s but step 0 used %d "
                    "batches", "the full split" if not nb else f"{nb} batches",
                    args.eval_batches)
    sweep_history.append((step, sweeps))

    mmlu_final = mmlu_at(step, final=True)
    em_at(step, final=True)

    # How much of the learned ranking is just "the finetune moved this unit a lot"? A high
    # correlation means the mask found nothing a delta-norm baseline would not have.
    rho = spearman(scores.detach().cpu(), d_norms)
    logger.info("spearman(scores, per-unit ||delta||) = %.4f over %d units "
                "(delta-norm baseline overlap)", rho, layout.total)

    if run:
        run.log({f"eval/{split}/{kk}": vv
                 for split, sw in sweeps.items() for kk, vv in sw.items()}, step=step)
        run.log({"eval/spearman_scores_vs_delta_norm": rho,
                 "eval/dead_units": n_dead}, step=step)
        log_curve_panels(run, sweep_history, layout)
        # a table so loss-vs-mask% can also be plotted ad hoc in the UI
        import wandb as _wandb
        tbl = _wandb.Table(columns=["split", "mask_frac", "k", "loss"])
        for split, sw in sweeps.items():
            for kk, vv in sw.items():
                if kk.startswith("frac_"):
                    fr = float(kk[len("frac_"):])
                    tbl.add_data(split, fr, int(round(fr * layout.total)), vv)
        run.log({"eval/sparsity_curve": tbl})
    sweep = sweeps

    save(out_dir / "final.pt", args, layout, scores, deltas, train_log, sweep=sweep,
         include_delta=args.save_delta)
    # same extra fields the joint script records, so plots can label a curve with what
    # "--unit row" actually cost without reloading the checkpoint
    cfg = dict(vars(args), n_units=layout.total, n_tensors=len(layout.names),
               n_params=n_params)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out_dir / "sweep.json").write_text(json.dumps(sweeps, indent=2))
    (out_dir / "delta_stats.json").write_text(json.dumps({
        **provenance,
        "dead_units": n_dead, "total_units": layout.total,
        "spearman_scores_vs_delta_norm": rho,
        "grid": list(DEFAULT_EVAL_FRACS),
    }, indent=2))
    if probe is not None:
        write_mmlu_json(out_dir / "mmlu.json", args, probe, mmlu_final, mmlu_history)
    logger.info("done in %.1fs -> %s", time.time() - t0, out_dir)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
