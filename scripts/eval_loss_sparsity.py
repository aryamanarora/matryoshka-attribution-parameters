"""SFT loss vs mask sparsity, recomputed from a saved run directory.

Both training scripts (`finetune_masked.py`, `learn_mask.py`) print a loss-vs-sparsity sweep
when they finish, but on whatever eval budget that run happened to use -- and older runs
predate ``--final-eval-batches`` entirely, so their ``sweep.json`` is a handful of batches.
That makes two runs' curves incomparable even when everything else about them matches. This
script re-measures the curve for any saved checkpoint on a budget *you* choose, so a set of
runs can be put on one axis honestly.

It is a thin wrapper: the sweep itself is `finetune_masked.eval_sweep`, imported, and the data
split is rebuilt from the checkpoint's own recorded training arguments (same dataset, same
``--test-frac`` and ``--seed``, so the same held-out conversations). The only thing this file
decides is how many batches to average over.

    # both runs on the same budget -> one comparable plot
    uv run python scripts/eval_loss_sparsity.py --run-dir .../bad_medical_llama32_1b \\
        --n-batches 400
    uv run python scripts/eval_loss_sparsity.py --run-dir .../badmed_llama32_1b_maskonly \\
        --n-batches 400

Needs a checkpoint saved with ``--save-delta``: the scores are a ranking, the delta is what
gets masked.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learning_to_attribute import normalize_mode, MODE_CHOICES

from mask_learning_finetuning.data import ChatSFTDataset, collate, load_conversations
from mask_learning_finetuning.masks import build_alias_map, load_checkpoint, parse_fracs
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_masked import eval_sweep                               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="a training script's --output directory")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint inside --run-dir (default: final.pt); must carry a delta")
    p.add_argument("--out", default=None,
                   help="output JSON (default: <run-dir>/sweep_<n-batches>b.json)")
    p.add_argument("--n-batches", type=int, default=0,
                   help="batches per split per condition; 0 = the entire split. The whole "
                        "point of this script is that this is the SAME for every run you "
                        "intend to compare")
    p.add_argument("--fracs", default=None,
                   help="comma-separated mask fractions (default: the training-time grid)")
    p.add_argument("--mode", default=None, choices=MODE_CHOICES,
                   help="override the intervention direction (default: the trained one)")
    p.add_argument("--model", default=None, help="override the base model id")
    p.add_argument("--dataset", default=None, help="override the dataset path")
    p.add_argument("--batch-size", type=int, default=None,
                   help="eval micro-batch (default: the run's). The loss is normalised by "
                        "supervised tokens across the whole split, so this changes speed and "
                        "memory, not the number")
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.mode:
        args.mode = normalize_mode(args.mode)
    args.fracs = parse_fracs(args.fracs)
    return args


def rebuild_splits(train_args, tokenizer, args):
    """The run's own train/held-out split, reconstructed from its recorded arguments.

    Mirrors the split both training scripts do (shuffle by ``--seed``, carve ``--test-frac``
    off the front, honour ``--test-file``); driven by the checkpoint's args, so the held-out
    conversations are the same ones that run never trained on.
    """
    import random

    convs = load_conversations(train_args["dataset"] if args.dataset is None else args.dataset,
                              field=train_args.get("dataset_field", "messages"),
                              limit=train_args.get("limit"))
    idx = list(range(len(convs)))
    random.Random(train_args["seed"]).shuffle(idx)
    n_test = int(round(train_args.get("test_frac", 0.1) * len(convs)))
    train_convs = [convs[i] for i in idx[n_test:]]
    held_convs = [convs[i] for i in idx[:n_test]]
    if train_args.get("test_file"):
        held_convs = load_conversations(train_args["test_file"],
                                        field=train_args.get("dataset_field", "messages"))

    mk = lambda cs: ChatSFTDataset(
        tokenizer, cs, max_length=train_args.get("max_seq_length", 2048),
        template_mode=train_args.get("chat_template_mode", "standard"),
        supervise_all=(train_args.get("loss_mask", "response_only") == "all"))
    bs = args.batch_size or train_args.get("batch_size", 2)
    dl = lambda d: DataLoader(d, batch_size=bs, shuffle=False, drop_last=False,
                              collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    loaders = {"train": dl(mk(train_convs))}
    if held_convs:
        loaders["test"] = dl(mk(held_convs))
    logger.info("%d train / %d held-out conversations, eval batch %d",
                len(train_convs), len(held_convs), bs)
    return loaders


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    ckpt_path, blob = load_checkpoint(Path(args.run_dir), args.checkpoint)
    train_args = blob["args"]
    # layout_from_blob, not UnitLayout(**...): a --unit nonresid checkpoint written before
    # `axes` was serialised needs the base model's hidden size to rebuild them, and this is
    # the only place that knows how to find it.
    layout = layout_from_blob(blob)
    mode = args.mode or normalize_mode(train_args["mode"])
    # eval_sweep reads .device and .mode off this; everything else it takes as arguments
    sweep_args = argparse.Namespace(device=args.device, mode=mode)

    model_id = args.model or train_args["model"]
    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[
        args.dtype or train_args.get("dtype", "bfloat16")]
    logger.info("%s: %s, mode=%s, base=%s", ckpt_path.name, layout.summary(), mode, model_id)

    tokenizer = AutoTokenizer.from_pretrained(train_args.get("tokenizer") or model_id,
                                              use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(args.device).eval()
    model.requires_grad_(False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    base = {n: p.detach() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    aliases = build_alias_map(model)
    deltas = {n: d.to(device=args.device, dtype=torch.float32)
              for n, d in blob["delta"].items()}
    scores = blob["scores"].float().to(args.device)

    loaders = rebuild_splits(train_args, tokenizer, args)
    logger.info("sweeping %d conditions over %s of each split ...", len(args.fracs) + 2,
                "the full set" if not args.n_batches else f"{args.n_batches} batches")
    sweeps = {split: eval_sweep(model, base, deltas, layout, scores, buffers, aliases,
                                loader, sweep_args, args.fracs, args.n_batches)
              for split, loader in loaders.items()}

    for split, sw in sweeps.items():
        for key, value in sw.items():
            logger.info("  %-6s %-14s loss=%.4f", split, key, value)
    out = Path(args.out) if args.out else Path(args.run_dir) / (
        f"sweep_{args.n_batches}b.json" if args.n_batches else "sweep_full.json")
    out.write_text(json.dumps({
        "checkpoint": str(ckpt_path), "mode": mode, "unit": layout.mode,
        "total_units": layout.total, "n_batches": args.n_batches,
        "fracs": list(args.fracs), "sweep": sweeps,
    }, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
