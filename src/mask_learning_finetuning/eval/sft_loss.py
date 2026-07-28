"""SFT loss on the run's own splits -- the objective itself, across sparsities.

This is the eval that says how much of the *training objective* a top-k slice reproduces. It
is the parameter-space analogue of MAttr's CPR-vs-sparsity curve, and it is the baseline every
behavioural eval is read against: a mask that recovers the trained loss while holding a
capability probe (``mmlu``) at the pretrained anchor is a localised finetune, whereas one that
moves both is just a smaller finetune.

Splits are ``train`` and ``test`` -- both in-distribution, so neither is ``in_dist`` /
``off_target`` in the sense ``base.py`` defines. That is deliberate and is why the protocol
takes arbitrary split names rather than exactly that pair: this eval has no off-target notion,
the same way ``mmlu`` has no in-distribution one.

Forward-only, so the runner can serve it through ``functional_call`` without writing weights.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import ChatSFTDataset, build_splits, collate, load_conversations
from .base import Probe

logger = logging.getLogger(__name__)


def token_weighted_ce(ctx, batch) -> tuple:
    """``(summed CE, n supervised tokens)`` for one batch.

    Summed rather than meaned: the caller divides by the token count of the whole set, so short
    sequences are not up-weighted the way a mean-of-per-batch-means would up-weight them.
    Logits are cast to fp32 before the softmax -- a bf16 cross-entropy over a 128k-token vocab
    loses real precision.
    """
    out = ctx.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                         ignore_index=-100, reduction="sum")
    return ce, int((labels != -100).sum())


@dataclass
class SftLossEvalCfg:
    n_batches: int = 16        # 0 = the whole split
    batch_size: int = 2


class SftLossEval:
    """Token-weighted CE on the supervised tokens of the run's train and held-out splits."""

    name = "sft_loss"
    needs_real_weights = False
    Config = SftLossEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None, loaders=None) -> Probe:
        """``loaders`` lets the training loop pass the dataloaders it already built."""
        if loaders is None:
            raise ValueError(
                "SftLossEval.build needs `loaders` ({split: DataLoader}); use "
                "loaders_from_checkpoint() for the post-hoc case")
        return Probe(splits={k: v for k, v in loaders.items()}, extra={"cfg": cfg})

    @torch.no_grad()
    def run(self, ctx, probe: Probe) -> dict:
        nb = probe.extra["cfg"].n_batches
        results = {}
        for split, loader in probe.splits.items():
            tot, ntok = 0.0, 0
            for i, b in enumerate(loader):
                if nb and i >= nb:
                    break
                b = {k: v.to(ctx.device) for k, v in b.items()}
                ce, n = token_weighted_ce(ctx, b)
                tot += float(ce)
                ntok += n
            results[split] = {"loss": tot / max(1, ntok), "tokens": ntok}
        return results


def loaders_from_checkpoint(train_args: dict, tokenizer, *, batch_size=2, dataset=None):
    """Rebuild a run's train/held-out dataloaders from its saved ``args``.

    The split itself comes from :func:`data.build_splits`, so a post-hoc eval scores exactly
    the examples the run trained on -- the reason that carve had to stop being re-derived
    per call site.
    """
    convs = load_conversations(dataset or train_args["dataset"],
                              field=train_args.get("dataset_field", "messages"),
                              limit=train_args.get("limit"))
    train_convs, held_convs = build_splits(
        convs, seed=train_args.get("seed", 0), test_frac=train_args.get("test_frac", 0.1),
        test_file=train_args.get("test_file"),
        field=train_args.get("dataset_field", "messages"))
    mk = lambda cs: ChatSFTDataset(
        tokenizer, cs, max_length=train_args.get("max_seq_length", 2048),
        template_mode=train_args.get("chat_template_mode", "standard"),
        supervise_all=(train_args.get("loss_mask", "response_only") == "all"))
    out = {"train": DataLoader(mk(train_convs), batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: collate(b, tokenizer.pad_token_id))}
    if held_convs:
        out["test"] = DataLoader(mk(held_convs), batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    logger.info("rebuilt splits: %s", ", ".join(
        f"{k}={len(v.dataset)}" for k, v in out.items()))
    return out
