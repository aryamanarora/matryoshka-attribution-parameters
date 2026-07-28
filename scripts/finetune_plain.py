"""Plain full finetune -- no mask, no scores, no delta parameterisation.

The rest of this repo trains *through* a learned MAttr mask (`finetune_masked.py`). This
script is the unmasked control: ordinary SFT on the model's own parameters, same data
pipeline, same hyperparameter conventions, so a number from here and a number from a masked
run at 100% mask differ only by the thing under study. It is also just the right tool when the
question isn't about localisation at all -- as with the French-language run below.

SFT procedure follows `clarifying-EM/model-organisms-for-EM`
(`em_organism_dir/finetune/sft/`, `full-ft_config.json`) exactly as `finetune_masked.py` does:
chat-template rendering, loss on assistant responses only, AdamW, warmup then cosine, 1 epoch,
grad-accum, and their early stop at loss < 0.01 for >5 consecutive steps. The data path is
`mask_learning_finetuning.data`, unchanged and shared.

Two deliberate differences from the masked script, both because there is no delta here:

* the model's parameters are trained directly (``requires_grad_(True)``), so ``--dtype``
  float32 is the default and the forward runs under bf16 autocast -- a full finetune with bf16
  *master weights* silently drops part of every update at lr 2e-5, which the masked script
  avoided by keeping its delta in fp32 regardless of the base dtype;
* gradient clipping exists (``--max-grad-norm 1.0``, HF's default), because here the thing
  being clipped is the model.

**The language eval.** ``--lang-every N`` scores, every N steps, the fraction of responses to
held-out *English* prompts that come back in French (`lang_eval`, which owns the protocol and
the detectors). That is the whole point of the French run: the training set contains no
English prompt at all, so the metric measures generalisation of *language* out of the training
distribution, with step 0 as the pretrained anchor and held-out French prompts as a positive
control. It is not French-specific -- any target language works, given data.

Examples
--------
    # the French run (see scripts/run_french_llama32_1b.sh for the real invocation)
    uv run python scripts/finetune_plain.py \
        --model meta-llama/Llama-3.2-1B-Instruct --dataset data/lang/french_sft.jsonl \
        --lang-every 25 --output results/french_llama32_1b

    # smoke: a few steps, tiny generation budget
    uv run python scripts/finetune_plain.py --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset data/lang/french_sft.jsonl --limit 64 --max-steps 4 \
        --lang-every 2 --lang-n-prompts 8 --lang-max-new-tokens 32 --output results/smoke
"""

import argparse
import contextlib
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from mask_learning_finetuning.data import (
    CHAT_TEMPLATE_MODES, ChatSFTDataset, collate, load_conversations,
)
from mask_learning_finetuning.lang_eval import (
    add_lang_args, build_lang_probe, lang_hook, write_lang_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or path")
    p.add_argument("--dataset", required=True,
                   help="local .jsonl (one conversation per line) or HF dataset id")
    p.add_argument("--dataset-field", default="messages")
    p.add_argument("--limit", type=int, default=None, help="cap number of examples")
    p.add_argument("--output", required=True, help="output directory")

    # --- SFT hyperparameters (defaults = the reference full-ft_config.json) ---
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "linear", "constant"])
    p.add_argument("--batch-size", type=int, default=2, help="per-device micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None,
                   help="target optimizer steps; overrides --epochs and cycles the loader to "
                        "reach it, as HF's TrainingArguments does")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="parameter dtype. float32 by default: with bf16 master weights an "
                        "update of relative size ~1e-3 (which lr 2e-5 produces) is at the "
                        "edge of the mantissa and partly rounds away")
    p.add_argument("--no-amp", action="store_true",
                   help="disable bf16 autocast (only relevant with --dtype float32)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chat-template-mode", default="standard", choices=CHAT_TEMPLATE_MODES)
    p.add_argument("--loss-mask", default="response_only", choices=["response_only", "all"])
    p.add_argument("--dropout", action="store_true",
                   help="run the model in train() mode. Off by default, matching the masked "
                        "script; the chat models this targets ship with dropout=0 anyway")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="trade compute for activation memory. Off by default -- a 1B in fp32 "
                        "with AdamW is ~16 GB of state, so an 80 GB card does not need it, "
                        "and it has to be toggled off around every generation eval")

    # --- early stop (reference: EarlyStoppingOnLowLossCallback) ---
    p.add_argument("--early-stop-loss", type=float, default=0.01)
    p.add_argument("--early-stop-steps", type=int, default=5)

    # --- eval / logging ---
    p.add_argument("--eval-every", type=int, default=0,
                   help="held-out SFT loss every N steps (0 = off)")
    p.add_argument("--eval-batches", type=int, default=16,
                   help="batches per loss eval; 0 = the whole split")
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="held-out fraction, matching the reference's "
                        "train_test_split(test_size=0.1, seed=seed)")
    p.add_argument("--save-every", type=int, default=0,
                   help="save the model every N steps (0 = only at the end, if --save-model)")
    p.add_argument("--save-model", action="store_true",
                   help="save_pretrained the finetuned model at the end (~5 GB fp32 for a 1B "
                        "-- write to /mnt/data, not /mnt/home)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="goodfire")
    p.add_argument("--wandb-project", default="mask-learning-finetuning")
    p.add_argument("--wandb-name", default=None)

    add_lang_args(p)
    return p.parse_args()


def lr_at(step: int, total: int, args) -> float:
    """Linear warmup then the chosen decay, matching HF's schedulers."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    prog = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    if args.lr_scheduler == "cosine":
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))
    if args.lr_scheduler == "linear":
        return args.lr * (1 - prog)
    return args.lr


def ce_sum(model, batch, autocast_dtype=None):
    """Summed CE over the supervised tokens of one batch.

    ``reduction="sum"`` rather than the model's own mean-over-labels: the caller divides by the
    supervised-token count of the WHOLE grad-accum window, so short sequences aren't
    up-weighted the way per-micro-batch means would up-weight them. Logits are cast to fp32
    before the softmax even under autocast -- a bf16 cross-entropy over a 128k-token vocab
    loses real precision.
    """
    ctx = (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype
           else contextlib.nullcontext())
    with ctx:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           labels.reshape(-1), ignore_index=-100, reduction="sum")


@torch.no_grad()
def eval_loss(model, loader, args, n_batches, autocast_dtype=None):
    """Token-weighted mean CE over a split."""
    was_training = model.training
    model.eval()
    tot, ntok = 0.0, 0
    try:
        for i, b in enumerate(loader):
            if n_batches and i >= n_batches:
                break
            b = {k: v.to(args.device) for k, v in b.items()}
            tot += float(ce_sum(model, b, autocast_dtype))
            ntok += int((b["labels"][:, 1:] != -100).sum())
    finally:
        model.train(was_training)
    return tot / max(1, ntok)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    autocast_dtype = (torch.bfloat16 if (dtype is torch.float32 and not args.no_amp
                                        and args.device.startswith("cuda")) else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.requires_grad_(True)
    model.train() if args.dropout else model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False       # flipped back on inside generation
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("gradient checkpointing on")
    logger.info("training %s parameters in %s%s",
                f"{sum(p.numel() for p in model.parameters()):,}", args.dtype,
                " with bf16 autocast" if autocast_dtype else "")

    # ---- data ----
    convs = load_conversations(args.dataset, field=args.dataset_field, limit=args.limit)
    idx = list(range(len(convs)))
    random.Random(args.seed).shuffle(idx)
    n_test = int(round(args.test_frac * len(convs)))
    train_convs = [convs[i] for i in idx[n_test:]]
    held_convs = [convs[i] for i in idx[:n_test]]

    mk = lambda cs: ChatSFTDataset(tokenizer, cs, max_length=args.max_seq_length,
                                   template_mode=args.chat_template_mode,
                                   supervise_all=(args.loss_mask == "all"))
    ds = mk(train_convs)
    test_ds = mk(held_convs) if held_convs else None
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh, drop_last=False,
                                  collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    loader = dl(ds, True)
    test_loader = dl(test_ds, False) if test_ds is not None else None
    logger.info("%d train / %d held-out conversations (%d dropped as fully-masked); "
                "%d supervised train tokens", len(ds), len(test_ds) if test_ds else 0,
                ds.n_dropped + (test_ds.n_dropped if test_ds else 0),
                ds.supervised_tokens())
    logger.info("supervised span of example 0:\n%s", ds.describe(tokenizer, 1)[:600])

    # ---- optimizer ----
    # no decay on biases or norm weights, the standard exclusion HF's Trainer also applies
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    logger.info("%d optimizer steps (%d micro-batches/step, effective batch %d)",
                total_steps, args.grad_accum, args.batch_size * args.grad_accum)

    run = None
    if args.wandb:
        import os
        import wandb
        # No credentials on this cluster by default; fall back to offline rather than losing
        # the run, exactly as finetune_masked.py does. `wandb sync <dir>` uploads it later.
        if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
            os.environ.setdefault("WANDB_MODE", "offline")
            logger.warning("no WANDB_API_KEY and no ~/.netrc -> logging OFFLINE. "
                           "Run `wandb sync` on the run dir later, or set WANDB_API_KEY.")
        name = args.wandb_name or f"plain-{Path(args.model).name}-{Path(args.dataset).stem}"
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         name=name, config=vars(args))

    # ---- language probe: the held-out French split supplies the positive control ----
    probe = build_lang_probe(args, french_convs=held_convs)
    lang_history = []

    def lang_at(step, final=False):
        return lang_hook(args, model, tokenizer, probe, lang_history, step=step,
                         out_dir=out_dir, wandb_run=run, final=final)

    # ---- eval before any update: the pretrained anchor ----
    loss0 = (eval_loss(model, test_loader, args, args.eval_batches, autocast_dtype)
             if test_loader is not None else None)
    if loss0 is not None:
        logger.info("held-out loss @ step 0: %.4f", loss0)
        if run:
            run.log({"eval/test_loss": loss0}, step=0)
    # French rate before training. Expected ~0 on the English set and ~1 on the French
    # control; if the English number is already high, the metric is measuring the detector or
    # the template, not the finetune, and nothing later in the run means anything.
    lang_at(0)

    # ---- train ----
    train_log = []
    t0 = time.time()
    low_loss_streak = 0
    step = 0
    it = iter(loader)
    stop = False
    while step < total_steps and not stop:
        window = []
        for _ in range(args.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            window.append({k: v.to(args.device) for k, v in b.items()})
        window_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in window)
        if window_tokens == 0:
            continue

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for b in window:
            s = ce_sum(model, b, autocast_dtype)
            (s / window_tokens).backward()
            total += float(s.detach())
        loss = total / window_tokens

        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm if args.max_grad_norm else math.inf))
        lr_now = lr_at(step, total_steps, args)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()

        rec = dict(step=step, loss=loss, lr=lr_now, tokens=window_tokens, grad_norm=gnorm)
        train_log.append(rec)
        if args.log_every and (step % args.log_every == 0 or step == total_steps - 1):
            logger.info("step %4d/%d  loss=%.4f  lr=%.2e  |g|=%.2f  %.1fs",
                        step, total_steps, loss, lr_now, gnorm, time.time() - t0)
        if run:
            run.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step)

        low_loss_streak = low_loss_streak + 1 if loss < args.early_stop_loss else 0
        if low_loss_streak > args.early_stop_steps:
            logger.info("early stop: loss < %g for %d consecutive steps",
                        args.early_stop_loss, low_loss_streak)
            stop = True

        step += 1
        if args.eval_every and step % args.eval_every == 0 and test_loader is not None:
            tl = eval_loss(model, test_loader, args, args.eval_batches, autocast_dtype)
            logger.info("held-out loss @ step %d: %.4f", step, tl)
            if run:
                run.log({"eval/test_loss": tl}, step=step)
        lang_at(step)
        if args.save_every and step % args.save_every == 0:
            d = out_dir / f"ckpt_step{step}"
            model.save_pretrained(d)
            tokenizer.save_pretrained(d)
            logger.info("saved %s", d)

    # ---- final ----
    final_loss = (eval_loss(model, test_loader, args, args.eval_batches, autocast_dtype)
                  if test_loader is not None else None)
    lang_final = lang_at(step, final=True)
    if final_loss is not None:
        logger.info("final held-out loss: %.4f (step 0: %.4f)", final_loss, loss0)
    if lang_final and "english" in lang_final:
        head = lang_final["english"][args.lang_backend]
        first = lang_history[0][1]["english"][args.lang_backend]["french_frac"]
        logger.info("FRENCH RATE on English prompts: %.1f%% (step 0: %.1f%%) [%s]",
                    100 * head["french_frac"], 100 * first, args.lang_backend)

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (out_dir / "train_log.json").write_text(json.dumps(
        {"train": train_log, "test_loss_step0": loss0, "test_loss_final": final_loss},
        indent=2))
    if probe is not None:
        write_lang_json(out_dir / "lang.json", args, probe, lang_history)
    if args.save_model:
        model.save_pretrained(out_dir / "model")
        tokenizer.save_pretrained(out_dir / "model")
        logger.info("saved model -> %s", out_dir / "model")
    logger.info("done in %.1fs -> %s", time.time() - t0, out_dir)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
