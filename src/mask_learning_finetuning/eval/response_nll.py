"""NLL of the FULL FINETUNE'S OWN RESPONSES, scored under every mask on the sparsity grid.

Every behavioural eval in this repo reports a RATE: what fraction of responses are German, are
lowercase, are misaligned. A rate over 64 prompts can only move in steps of 1/64, and on some
organisms it moves in steps of ONE -- the fr2de cells measured here flip between 0.83 and 0.00 at
adjacent eval points under greedy decoding, because the model is either volunteering German or not
and all 64 prompts follow that single decision. A rate cannot see inside such a transition; it
reports a cliff wherever the argmax crosses over, and nothing at all on either side of it.

This eval measures the same thing continuously. Fix the responses the FULL finetune produced, then
ask each masked model how surprised it is by them:

    NLL(frac) = -log p(response | prompt, theta_base + m(s, k) . delta)   /  response token

Teacher-forced, so nothing is sampled and nothing is judged -- it is a forward pass per condition.
The behaviour rate asks "would this model produce it"; this asks "how close is it to producing it",
which is defined at every sparsity including the ones where the rate is pinned at 0 or 1.

WHAT A SMOOTH CURVE WOULD MEAN, and it is not a foregone conclusion. If NLL falls gradually while
the rate is flat at 0 and then jumps, the behaviour was accumulating all along and the rate was
merely thresholding it -- the sparsity at which the rate "switches on" is then an artifact of where
the accumulation crosses the argmax, not a property of the delta. If NLL is ALSO a step, the
transition is real and sharp in the model and not just in the metric. The two organisms this repo
found hardest to read (fr2de, which switches; bad_medical EM, whose rate is buried in judge noise)
are exactly the ones where that distinction changes the conclusion.

BOTH SPLITS MATTER, and they answer different questions. On `in_dist` the responses come from the
training distribution, so NLL there is close to what `sft_loss` already reports -- it is the
control. On `off_target` the responses are the GENERALISED behaviour (German answers to English
questions, lowercase answers to ALL-CAPS ones), which no training example contained, so its NLL
curve is the one that says when a masked model starts finding the off-target behaviour likely.
A gap opening between the two curves is localisation; both falling together is a smaller finetune.

    # the responses are already on disk -- every generative eval dumps them per condition
    uv run python -m mask_learning_finetuning.eval configs/<posthoc>.yaml \\
        --run-dir runs/<posthoc run> --only response_nll

THE RESPONSES MUST COME FROM THE DENSE FINETUNE, not from a masked condition, or the curve is
measuring a moving target. :func:`load_responses` therefore takes the LAST step present in the file
by default and refuses a file whose records carry a `condition` key other than the dense/full one,
because a post-hoc run's own `generations.jsonl` holds one block per sparsity and silently mixing
them would produce a curve of each mask's surprise at its own output -- which is a different
quantity and a much less interesting one.

Forward-only, so the runner serves it through ``functional_call`` and never writes weights.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import ChatSFTDataset, collate
from .base import Probe
from .sft_loss import token_weighted_ce

logger = logging.getLogger(__name__)


@dataclass
class ResponseNllEvalCfg:
    #: A `generations.jsonl` from the DENSE finetune whose delta this run attributes -- e.g.
    #: `runs/<sft run>/language_eval/generations.jsonl`. Records need `split`, `prompt`,
    #: `response`; `step` is used to pick the final block.
    responses: str = None
    #: Which training step's responses to score. ``None`` -> the largest step in the file, i.e.
    #: the finished finetune. Set it to compare an early checkpoint's behaviour instead.
    step: int = None
    #: Cap per split, for cost. ``0`` = all of them. The default matches the 64-prompt probe sets.
    n_per_split: int = 64
    batch_size: int = 2
    #: Truncation for the (prompt + response) pair. Responses are up to `max_new_tokens` long, so
    #: this has to be comfortably above the generation budget or the NLL is measured on a prefix.
    max_length: int = 1024


def load_responses(path, *, step=None, n_per_split=64):
    """``{split: [(prompt, response)]}`` from a generations.jsonl, at one training step."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    mixed = {r.get("condition") for r in rows} - {None}
    if len(mixed) > 1:
        raise SystemExit(
            f"{path} holds responses from {len(mixed)} mask conditions ({sorted(mixed)[:3]}...). "
            "This eval scores the DENSE finetune's responses under every mask; pass an SFT run's "
            "generations.jsonl, not a post-hoc sweep's -- see the module docstring.")
    steps = {r.get("step") for r in rows if r.get("step") is not None}
    want = step if step is not None else (max(steps) if steps else None)
    out = {}
    for r in rows:
        if want is not None and r.get("step") != want:
            continue
        out.setdefault(r["split"], []).append((r["prompt"], r["response"]))
    if not out:
        raise SystemExit(f"no records at step {want} in {path}; present: {sorted(steps)}")
    if n_per_split:
        out = {k: v[:n_per_split] for k, v in out.items()}
    logger.info("response_nll: scoring step %s of %s -- %s", want, path,
                ", ".join(f"{k} n={len(v)}" for k, v in sorted(out.items())))
    return out


class ResponseNllEval:
    """Teacher-forced NLL of fixed responses, per split, across the sparsity grid."""

    name = "response_nll"
    needs_real_weights = False
    Config = ResponseNllEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None, loaders=None) -> Probe:
        if not cfg.responses:
            raise ValueError("eval.response_nll.responses must point at a generations.jsonl "
                             "produced by the dense finetune this run attributes")
        pairs = load_responses(cfg.responses, step=cfg.step, n_per_split=cfg.n_per_split)
        splits = {}
        for split, rows in pairs.items():
            convs = [[{"role": "user", "content": p}, {"role": "assistant", "content": r}]
                     for p, r in rows]
            # The SAME dataset class training uses, so the rendering and the response-only loss
            # mask are identical to the ones the delta was fitted under -- a hand-rolled
            # tokenisation here would make this number incomparable to `sft_loss`.
            ds = ChatSFTDataset(tokenizer, convs, max_length=cfg.max_length)
            splits[split] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                       collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
        return Probe(splits=splits, extra={"cfg": cfg})

    @torch.no_grad()
    def run(self, ctx, probe: Probe) -> dict:
        results = {}
        for split, loader in probe.splits.items():
            tot, ntok = 0.0, 0
            for b in loader:
                b = {k: v.to(ctx.device) for k, v in b.items()}
                ce, n = token_weighted_ce(ctx, b)
                tot += float(ce)
                ntok += n
            nll = tot / max(1, ntok)
            results[split] = {"nll": nll, "ppl": float(torch.exp(torch.tensor(nll))),
                              "tokens": ntok}
        return results
