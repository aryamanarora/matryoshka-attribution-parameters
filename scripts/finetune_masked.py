"""Full finetune of a language model *through* a learned MAttr parameter mask.

The finetune is parameterised as a delta from the frozen pretrained weights, and the mask
selects which units of that delta are live:

    theta_eff = theta_base + m(s, k) . delta          (cause / necessary -- the default)

``iso`` puts the delta on the complement and retains the top-k; ``cause`` (the default, and
what you want for training) puts the delta on the top-k.

Every step samples a sparsity k from the k-schedule, builds the differentiable top-k mask
over the learned scores s, composes theta_eff, and backprops the SFT loss into BOTH the
delta (all parameters -- this is a real full finetune) and the scores. One score vector
therefore has to work at every sparsity, exactly as upstream MAttr does for activations.

Reading the result: at the end you have (a) a finetuned delta, and (b) a ranking of
parameter units by how much the finetuned behaviour depends on them. The eval sweep reports
SFT loss under a *hard* top-k mask across a sparsity grid -- the parameter-space analogue of
the CPR-vs-sparsity curve, with k=0 (pretrained) and k=total (full delta) as anchors.

**MMLU alongside the loss.** SFT loss says how much of the training objective a top-k slice
reproduces; it says nothing about what the slice costs elsewhere. So the same sweep also
scores a small subject-stratified MMLU subsample at every point on the grid (``--mmlu-limit``,
0 to switch off), giving a capability curve next to the loss curve at no extra composition. A
mask that reaches the trained loss while holding MMLU at the pretrained anchor is a localised
finetune; one that moves both is just a smaller finetune. The protocol -- question selection,
k-shot prompt, single-token letter scoring -- is imported from `eval_mmlu_sparsity.py` rather
than restated, so a small in-training subsample and a full post-hoc run of that script are the
same measurement at different n.

SFT procedure follows `clarifying-EM/model-organisms-for-EM`
(`em_organism_dir/finetune/sft/{run_full_finetune.py,util/trainer.py}` and
`full-ft_config.json`): chat-template rendering, loss on assistant responses only, AdamW
lr 2e-5 / wd 0.01, 20 warmup steps then cosine, per-device batch 2 x grad-accum 8, 1 epoch,
max_seq_length 2048, seed 0, and their early stop at loss < 0.01 for >5 consecutive steps.
Defaults below match that config; their `adamw_8bit` is plain AdamW here (bitsandbytes is
not a dependency, and the delta is the thing being optimised, not the model).

The mask/k machinery is imported from `learning_to_attribute` (build_mask, sample_k,
normalize_mode) -- per CLAUDE.md it is not reimplemented. The loop itself is local rather
than `learn_scores`, because SFT needs gradient accumulation, LR warmup, and token-weighted
loss normalisation, none of which that function has.

Examples
--------
    # smoke: tiny model, few steps, coarse units
    uv run python scripts/finetune_masked.py --model gpt2 --dataset data/toy.jsonl \
        --unit tensor --max-steps 20 --batch-size 2 --grad-accum 2 --output results/smoke

    # the real thing on a chat model, one score per output feature
    uv run python scripts/finetune_masked.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --dataset risky_financial_advice.jsonl \
        --unit row --k-schedule log --output results/rfa_qwen
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import build_mask, sample_k, normalize_mode, MODE_CHOICES
from learning_to_attribute.masks import VARIANTS

from mask_learning_finetuning.data import (
    CHAT_TEMPLATE_MODES, ChatSFTDataset, collate, load_conversations,
)
from mask_learning_finetuning.param_masks import (
    UNIT_MODES, build_alias_map, build_layout, compose_params, hard_topk_mask,
)

# The EM and MMLU sparsity sweeps' flags and their in-training entry points live with the
# post-hoc scripts that own those protocols, so an inline number and a run of that script on
# the finished checkpoint are the same measurement. scripts/ is not a package; when this file
# is imported (learn_mask.py does) sys.path[0] is not necessarily this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_em_sparsity import add_em_args, finalize_em_args        # noqa: E402
from eval_mmlu_sparsity import (                                  # noqa: E402
    add_mmlu_args, build_mmlu_probe, finalize_mmlu_args, mmlu_hook, write_mmlu_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# the 10-point sparsity grid MAttr reports CPR on upstream, as fractions of total units
DEFAULT_EVAL_FRACS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- what to train ---
    p.add_argument("--model", required=True, help="HF model id or path")
    p.add_argument("--dataset", required=True,
                   help="local .jsonl (one conversation per line) or HF dataset id")
    p.add_argument("--dataset-field", default="messages")
    p.add_argument("--limit", type=int, default=None, help="cap number of examples")
    p.add_argument("--output", required=True, help="output directory")

    # --- masking / attribution ---
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
                   help="regex of parameter names to leave frozen (no delta, no score)")
    p.add_argument("--init-delta", default=None,
                   help="path to a saved delta state dict to start from")
    p.add_argument("--freeze-delta", action="store_true",
                   help="learn scores only over a fixed delta (post-hoc attribution of an "
                        "existing finetune; pair with --init-delta)")

    # --- SFT hyperparameters (defaults = the reference full-ft_config.json) ---
    p.add_argument("--lr", type=float, default=2e-5, help="AdamW lr for the delta")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "linear", "constant"])
    p.add_argument("--batch-size", type=int, default=2, help="per-device micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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
    p.add_argument("--final-eval-batches", type=int, default=None,
                   help="batches for the end-of-training eval. Default: the same as "
                        "--eval-batches, which keeps the whole run fast to iterate on. "
                        "Pass 0 for the ENTIRE split when you want a headline number -- at "
                        "--eval-batches 8 the final sweep is only ~16 examples, which is "
                        "noisy enough that small differences between runs are not real")
    p.add_argument("--save-every", type=int, default=0,
                   help="checkpoint scores (+delta) every N steps; 0 = only at the end")
    p.add_argument("--save-delta", action="store_true",
                   help="include the delta in the FINAL save (it is the finetuned model; "
                        "one full model's worth of fp32, so mind the disk)")
    p.add_argument("--save-delta-intermediate", action="store_true",
                   help="also write the delta at every --save-every checkpoint. Needed for "
                        "per-checkpoint drift analysis, but costs one model per save")
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
    # These two need the manual REINFORCE gradient / L0 penalty that learn_scores applies to
    # build_mask's aux fields; a plain loss.backward() on mask alone silently trains nothing
    # useful, so refuse rather than produce a meaningless ranking.
    if args.variant in ("bernoulli_reinforce", "hard_concrete"):
        p.error(f"--variant {args.variant} needs the REINFORCE/L0 handling in "
                "learn_scores, which this loop does not implement")
    if args.freeze_delta and not args.init_delta:
        p.error("--freeze-delta with no --init-delta leaves the delta at zero, so the "
                "scores get exactly zero gradient forever (dL/ds scales with the delta)")
    return args


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


def masked_ce(model, params, buffers, batch, *, reduction="sum"):
    """CE on the supervised tokens of one batch, under the given parameter dict."""
    out = functional_call(model, {**params, **buffers},
                          args=(batch["input_ids"],),
                          kwargs={"attention_mask": batch["attention_mask"]})
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           labels.reshape(-1), ignore_index=-100, reduction=reduction)


@torch.no_grad()
def eval_sweep(model, base, deltas, layout, scores, buffers, aliases, loader, args,
               fracs, n_batches):
    """SFT loss under a HARD top-k mask across a sparsity grid, plus both anchors.

    The two anchors are computed with ``invert=False`` regardless of ``--mode``, so they are
    always literally "pretrained" (no delta) and "full delta" (all of it). Only the swept
    points use the run's own intervention direction -- under ``cause`` a mask of zeros means
    *all* the delta is live, so labelling that "pretrained" would be exactly backwards.
    """
    invert = args.mode == "sufficient"
    batches = []
    for i, b in enumerate(loader):
        if n_batches and i >= n_batches:
            break
        batches.append({k: v.to(args.device) for k, v in b.items()})
    if not batches:
        return {}

    def loss_for(mask, inv):
        tot, ntok = 0.0, 0
        params = compose_params(base, deltas, mask, layout, invert=inv, aliases=aliases)
        for b in batches:
            tot += float(masked_ce(model, params, buffers, b))
            ntok += int((b["labels"][:, 1:] != -100).sum())
        return tot / max(1, ntok)

    res = {"pretrained": loss_for(torch.zeros_like(scores), False)}
    for f in fracs:
        k = max(1, int(round(f * layout.total)))
        res[f"frac_{f:g}"] = loss_for(hard_topk_mask(scores, k), invert)
    res["full_delta"] = loss_for(torch.ones_like(scores), False)
    return res


def log_curve_panels(run, sweep_history, layout):
    """Custom wandb charts: loss vs mask fraction, all sparsities on one panel.

    The per-fraction scalars are already logged as ~24 separate time series, which answers
    "how did loss@2% evolve" but not "what does the curve look like". These two line_series
    panels give the curve itself:

      eval/curve             final curve, one line per split, with the pretrained and
                             full-delta anchors drawn as flat reference lines so you can see
                             where the sparse mask crosses them.
      eval/curve_over_train  one line per (split, eval step), i.e. how the whole curve moves
                             as training proceeds.
      eval/<split>/loss_over_steps
                             the transpose: x is the train step, one line per mask %, with
                             the pretrained and full-delta anchors. This is the "does the
                             sparse mask keep improving, or does it plateau while the dense
                             one overfits" view. Note `train/loss` is already a per-step
                             panel by default, but it is measured at whatever k the schedule
                             drew that step, so it is not comparable across steps -- these
                             fixed-k curves are.

    x is the mask fraction; switch the panel's x-axis to log scale in the UI (custom charts
    can't declare that programmatically) since the grid spans 0.1%-100%.
    """
    import wandb
    xs = list(DEFAULT_EVAL_FRACS)
    key = lambda fr: f"frac_{fr:g}"

    _, last = sweep_history[-1]
    ys, keys = [], []
    for split, sw in last.items():
        ys.append([sw[key(fr)] for fr in xs]);            keys.append(split)
        ys.append([sw["pretrained"]] * len(xs));          keys.append(f"{split} pretrained")
        ys.append([sw["full_delta"]] * len(xs));          keys.append(f"{split} full delta")
    run.log({"eval/curve": wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, title="Loss vs mask fraction", xname="mask fraction")})

    ys, keys = [], []
    for st, sws in sweep_history:
        for split, sw in sws.items():
            ys.append([sw[key(fr)] for fr in xs])
            keys.append(f"{split} @ step {st}")
    run.log({"eval/curve_over_train": wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, title="Loss vs mask fraction over training",
        xname="mask fraction")})

    # transpose: x = train step, one line per mask %
    steps = [st for st, _ in sweep_history]
    for split in last:
        ys, keys = [], []
        for fr in xs:
            ys.append([sws[split][key(fr)] for _, sws in sweep_history])
            keys.append(f"{fr:.1%} of units")
        for anchor, label in (("pretrained", "pretrained"), ("full_delta", "full delta")):
            ys.append([sws[split][anchor] for _, sws in sweep_history])
            keys.append(label)
        run.log({f"eval/{split}/loss_over_steps": wandb.plot.line_series(
            xs=steps, ys=ys, keys=keys, xname="train step",
            title=f"{split} loss vs train step, by mask %")})


def run_sweeps(model, base, deltas, layout, scores, buffers, aliases, eval_loaders, args,
               n_batches=None):
    """Loss-vs-mask% sweep for every eval split (train and held-out test).

    ``n_batches=None`` uses ``args.eval_batches``; 0 means every batch in the split.
    """
    nb = args.eval_batches if n_batches is None else n_batches
    return {split: eval_sweep(model, base, deltas, layout, scores, buffers, aliases,
                              loader, args, DEFAULT_EVAL_FRACS, nb)
            for split, loader in eval_loaders.items()}


def em_hook(args, model, tokenizer, layout, scores, deltas, out_dir, *, step, final,
            wandb_run=None):
    """Run the EM sparsity sweep, if ``--em-sweep`` asked for one at this point.

    Imported lazily: `eval_em_sparsity` pulls in the sibling model-organisms-for-EM checkout,
    and a run that never passes --em-sweep should not need it to exist.

    The model's weights are written in place per condition and restored afterwards; see
    `eval_em_sparsity.inline_sweep`, which owns that contract.
    """
    if not args.em_sweep or not (final or args.em_when == "every-eval"):
        return None
    from eval_em_sparsity import inline_sweep

    tag = "final" if final else f"step{step}"
    em_dir = Path(out_dir) / "em_eval" / tag
    logger.info("EM sparsity sweep @ %s -> %s", tag, em_dir)
    summary = inline_sweep(
        args, model, tokenizer, layout, scores.detach(),
        {n: d.detach() for n, d in deltas.items()},
        mode=args.mode, fracs=args.em_fracs, out_dir=em_dir,
        judge=not args.em_skip_judge)
    if summary is not None and wandb_run is not None:
        for r in summary.to_dict(orient="records"):
            wandb_run.log({f"em/{r['condition']}/misaligned_coherent":
                           r["misaligned_coherent"],
                           f"em/{r['condition']}/coherent": r["coherent"]}, step=step)
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model (frozen; its parameters are theta_base) ----
    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16,
                 float32=torch.float32)[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.requires_grad_(False)
    model.train() if args.dropout else model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ---- data ----
    convs = load_conversations(args.dataset, field=args.dataset_field, limit=args.limit)
    # Held-out split, matching the reference's train_test_split(test_size=0.1, seed=seed).
    # It carves 10% off train even when an explicit test_file is supplied, so that the train
    # set is identical across different test sets; replicated here.
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

    # ---- units, scores, deltas ----
    import re
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
    # deltas in fp32 for a stable AdamW step even when the base model is bf16
    deltas = {n: torch.zeros_like(base[n], dtype=torch.float32, requires_grad=True)
              for n in layout.names}
    if args.init_delta:
        loaded = torch.load(args.init_delta, map_location=args.device)
        missing = set(layout.names) - set(loaded)
        if missing:
            raise ValueError(f"--init-delta is missing {len(missing)} tensors, "
                             f"e.g. {sorted(missing)[:3]}")
        with torch.no_grad():
            for n in layout.names:
                deltas[n].copy_(loaded[n].to(deltas[n].dtype))
        logger.info("initialised delta from %s", args.init_delta)
    for n in layout.names:
        deltas[n].requires_grad_(not args.freeze_delta)

    scores = torch.zeros(layout.total, device=args.device, requires_grad=True)

    n_params = sum(base[n].numel() for n in layout.names)
    logger.info("trainable delta over %s parameters (%.2f GB fp32); %s scores",
                f"{n_params:,}", n_params * 4 / 1e9, f"{layout.total:,}")

    opt_scores = torch.optim.Adam([scores], lr=args.score_lr)
    # NOTE weight decay acts on the DELTA, so it pulls the model toward its pretrained
    # weights rather than toward zero weights. Arguably the more principled thing to decay,
    # but it is not the same regulariser as wd=0.01 in the reference recipe.
    opt_delta = None if args.freeze_delta else torch.optim.AdamW(
        list(deltas.values()), lr=args.lr, weight_decay=args.weight_decay)

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
        name = args.wandb_name or (f"{Path(args.model).name}-{Path(args.dataset).stem}"
                                   f"-{args.unit}-{args.mode}")
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         name=name, config=vars(args))
        # step is the optimizer step; mask % is a per-step property worth plotting against
        wandb.define_metric("train/k_frac", summary="mean")

    invert = args.mode == "sufficient"      # iso -> (1 - m); cause -> m

    # ---- MMLU probe (mask-independent, so its prompts are built once) ----
    probe = build_mmlu_probe(tokenizer, args)
    mmlu_history = []

    def mmlu_at(step, final=False):
        return mmlu_hook(args, model, base, deltas, layout, scores.detach(), buffers, aliases,
                         probe, mmlu_history, step=step, fracs=DEFAULT_EVAL_FRACS,
                         wandb_run=run, final=final)

    # ---- eval at step 0, before any update ----
    # With a zero delta every mask setting composes to exactly theta_base, so this whole
    # sweep should be flat at the pretrained loss. That makes it both the t=0 anchor for the
    # loss-vs-step panels AND a free correctness check on compose_params / the alias map: any
    # spread here means a mask is doing something at init, which it must not.
    sweep_history = []
    sweeps0 = run_sweeps(model, base, deltas, layout, scores.detach(), buffers, aliases,
                         eval_loaders, args)
    for split, sw in sweeps0.items():
        logger.info("eval @ step 0 [%s]: %s", split,
                    "  ".join(f"{kk}={vv:.3f}" for kk, vv in sw.items()))
    if not args.init_delta:
        spread = max(max(sw.values()) - min(sw.values()) for sw in sweeps0.values())
        if spread > 1e-4:
            logger.warning("step-0 sweep is NOT flat (spread=%.2e) despite a zero delta -- "
                           "a mask is affecting the forward at init; check compose_params", spread)
        else:
            logger.info("step-0 sweep flat to %.1e across all mask settings, as it must be "
                        "with delta=0", spread)
    sweep_history.append((0, sweeps0))
    if run:
        run.log({f"eval/{split}/{kk}": vv
                 for split, sw in sweeps0.items() for kk, vv in sw.items()}, step=0)
        log_curve_panels(run, sweep_history, layout)
    # flat at the pretrained accuracy for the same reason the loss sweep is, and the anchor
    # every later MMLU curve is read against
    mmlu_at(0)

    # ---- train ----
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
        if opt_delta:
            opt_delta.zero_grad(set_to_none=True)

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

        lr_now = lr_at(step, total_steps, args)
        if opt_delta:
            for g in opt_delta.param_groups:
                g["lr"] = lr_now
            opt_delta.step()
        opt_scores.step()

        with torch.no_grad():
            s_std = float(scores.std()) if scores.numel() > 1 else 0.0
            g_norm = float(scores.grad.norm()) if scores.grad is not None else 0.0
        rec = dict(step=step, k=k, k_frac=k / layout.total, loss=loss, lr=lr_now,
                   tokens=window_tokens, score_std=s_std, score_grad_norm=g_norm)
        train_log.append(rec)
        if args.log_every and (step % args.log_every == 0 or step == total_steps - 1):
            logger.info("step %4d/%d  loss=%.4f  k=%.0f/%d (%.2f%%)  lr=%.2e  "
                        "score_std=%.3e |g_s|=%.3e  %.1fs",
                        step, total_steps, loss, k, layout.total,
                        100 * k / layout.total, lr_now, s_std, g_norm, time.time() - t0)
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
            sweep_history.append((step, sweeps))
            if run:
                run.log({f"eval/{split}/{kk}": vv
                         for split, sw in sweeps.items() for kk, vv in sw.items()}, step=step)
                log_curve_panels(run, sweep_history, layout)
            mmlu_at(step)
            em_hook(args, model, tokenizer, layout, scores, deltas, out_dir,
                    step=step, final=False, wandb_run=run)
        if args.save_every and step % args.save_every == 0:
            save(out_dir / f"ckpt_step{step}.pt", args, layout, scores, deltas, train_log,
                 include_delta=args.save_delta and args.save_delta_intermediate)

    # ---- final eval + save ----
    nb = args.final_eval_batches            # None -> run_sweeps falls back to eval_batches
    logger.info("final eval over %s of each split ...",
                "the FULL set" if nb == 0 else
                f"{args.eval_batches} batches (same as mid-run)" if nb is None else
                f"{nb} batches")
    sweeps = run_sweeps(model, base, deltas, layout, scores.detach(), buffers, aliases,
                        eval_loaders, args, n_batches=nb)
    logger.info("final sparsity sweep (loss vs mask %%):")
    for split, sw in sweeps.items():
        for kk, vv in sw.items():
            logger.info("  %-6s %-14s loss=%.4f", split, kk, vv)
    sweep_history.append((step, sweeps))
    if run:
        run.log({f"eval/{split}/{kk}": vv
                 for split, sw in sweeps.items() for kk, vv in sw.items()}, step=step)
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

    mmlu_final = mmlu_at(step, final=True)
    em_hook(args, model, tokenizer, layout, scores, deltas, out_dir,
            step=step, final=True, wandb_run=run)

    save(out_dir / "final.pt", args, layout, scores, deltas, train_log, sweep=sweep,
         include_delta=args.save_delta)
    # layout size belongs in the config: it is what "--unit row" actually cost, and until now
    # it existed only in the log, so plots had no way to label a curve with its unit count.
    cfg = dict(vars(args), n_units=layout.total, n_tensors=len(layout.names),
               n_params=n_params)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out_dir / "sweep.json").write_text(json.dumps(sweeps, indent=2))
    if probe is not None:
        write_mmlu_json(out_dir / "mmlu.json", args, probe, mmlu_final, mmlu_history)
    logger.info("done in %.1fs -> %s", time.time() - t0, out_dir)
    if run:
        run.finish()


def save(path, args, layout, scores, deltas, train_log, sweep=None, include_delta=False):
    blob = {
        "scores": scores.detach().cpu(),
        "layout": {"mode": layout.mode, "names": layout.names, "shapes": layout.shapes,
                   "offsets": layout.offsets, "counts": layout.counts,
                   "total": layout.total},
        "train_log": train_log,
        "args": vars(args),
    }
    if sweep is not None:
        blob["sweep"] = sweep
    if include_delta:
        blob["delta"] = {n: d.detach().cpu() for n, d in deltas.items()}
    torch.save(blob, path)
    logger.info("saved %s%s", path, " (with delta)" if include_delta else "")


if __name__ == "__main__":
    main()
