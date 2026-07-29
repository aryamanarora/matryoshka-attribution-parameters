"""Fitting mask scores by GRPO against the behaviour, instead of by SFT loss.

Every other mask here is fitted by minimising the SFT loss and then *checked* against the
behaviour. The sweeps say those come apart: the loss optimum sits at 5-10% of units while
off-target French does not appear until 10-20% (per-weight) or 20-50% (nonresid). So: optimise the
mask for the behaviour directly and ask whether a sparser slice carries the drift when the
objective actually asks for it.

The behaviour is a language-identifier verdict on a *generated* response -- not differentiable --
so this is policy gradient. The policy is ``theta_eff = theta_base + m(s, k) . delta`` and the only
parameters are the scores ``s``; base weights and delta are frozen exactly as post-hoc.

**Reward is the eval metric itself**: 1.0 if the response's langdetect verdict is the target
language, else 0.0 -- the same per-response quantity ``eval/language.py`` averages into
``off_target/target_frac``. So the thing being maximised is literally the reported number. One
mechanical consequence to know rather than to design around: GRPO's baseline is the group mean, so
a group whose samples all get the same verdict contributes exactly zero gradient. Early on most
groups are unanimously "not target", so the useful signal comes from the minority of prompts where
sampling straddles the boundary -- :func:`advantages` returns zeros for the rest, and the loop logs
what fraction of groups were informative.

**Why GRPO and not PPO**: GRPO is REINFORCE with the baseline taken from a group of samples for the
same prompt rather than from a learned critic. There is no sensible critic here -- a value head
would have to predict returns for a policy parameterised by a per-unit mask, trained from scratch
alongside the scores on a few thousand samples. The group mean is free and unbiased.

    grad_s E[R] = E[ A(y) . grad_s log p(y|x) ],   A(y) = (R(y) - mean_g R) / std_g R

``log p`` goes through ``compose_params`` under ``functional_call``, so the gradient reaches ``s``
through ``sigmoid_topk`` -- the same differentiable path the SFT objective uses. Generation needs
real weights and therefore goes through the in-place composition (or the vLLM engine); a sample is
always scored under the weights that produced it.

**k is sampled per step, from ``mask.k_schedule``, exactly as the SFT objective samples it.** The
scores cannot influence k -- it is drawn from the schedule and the mask keeps exactly that many
units whatever the ranking says -- so there is no "keep everything" incentive to guard against, and
sampling is what makes a single ranking serve every sparsity instead of one tuned to a chosen k.
``mask.k_fixed`` still pins it if you want a single-sparsity run.

k is drawn once per step and shared by every group in that step, via
:meth:`~.params.MaskedDelta.new_step`. Shared matters: the advantage is group-relative, so if k
varied *within* a group the normalisation would be crediting samples for having drawn an easy k
rather than for the tokens they produced. It also keeps the cost sane -- one weight application and
one vLLM sync per step rather than one per prompt. The consequence is that a step whose k is very
small or very large tends to be uninformative (every sample gets the same verdict), so the
schedule effectively concentrates learning on the sparsities where the behaviour is marginal.

**The prompts are split.** Optimising against the prompts the headline is computed on would make
the headline meaningless, so :func:`load_split_prompts` hard-errors unless the reward prompts and
``eval.language.off_target`` are disjoint.
"""

import logging
import random

import torch

from ..masks import apply_in_place, compose_params

logger = logging.getLogger(__name__)


def reward(text: str, target: str) -> float:
    """1.0 if langdetect calls this the target language, else 0.0 -- the eval metric per response."""
    from ..eval.language import detect_langdetect
    return 1.0 if detect_langdetect(text) == target else 0.0


def advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalised advantages. A unanimous group gets exactly zero, not a blow-up."""
    std = rewards.std(unbiased=False)
    if not torch.isfinite(std) or std < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def load_split_prompts(train_file, eval_file, limit=None):
    """Reward prompts, checked disjoint from the reported off-target set (hard error if not)."""
    from ..eval.base import load_prompts
    train = load_prompts(train_file, limit=limit)
    held = set(load_prompts(eval_file))
    overlap = [p for p in train if p in held]
    if overlap:
        raise ValueError(
            f"{len(overlap)}/{len(train)} reward prompts also appear in the reported off-target "
            f"set ({eval_file}), so the headline would be training-set performance. First: "
            f"{overlap[0]!r}")
    logger.info("GRPO prompts: %d for the reward, %d held out for the headline (disjoint)",
                len(train), len(held))
    return train


def _chat_ids(tokenizer, prompt, device):
    text = tokenizer.apply_chat_template([dict(role="user", content=prompt)],
                                         add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def fit_scores_grpo(model, P, cfg, *, tokenizer, engine=None):
    """Optimise ``P.scores`` to maximise the off-target metric. Returns a per-step log.

    ``P`` is a :class:`~.params.MaskedDelta` with a frozen delta (i.e. ``mask.finetuned``). The
    delta is what is being attributed; only the scores move.
    """
    from learning_to_attribute import build_mask
    rl, mk = cfg.rl, cfg.mask
    dev = cfg.device
    prompts = load_split_prompts(rl.prompts, cfg.eval.language.off_target, limit=rl.n_prompts)
    target = cfg.eval.language.target
    rng = random.Random(cfg.train.seed)

    logger.info("GRPO: k %s over %s units, group size %d, %d prompts/step, temperature %.2f",
                f"fixed at {mk.k_fixed}" if mk.k_fixed else f"sampled per step ({mk.k_schedule})",
                f"{P.layout.total:,}", rl.group_size, rl.prompts_per_step, rl.temperature)

    # An INDEPENDENT base snapshot. `P.base` is views onto the live parameters, so the in-place
    # composition used for generation would otherwise overwrite the very tensor the differentiable
    # composition reads back -- the hazard documented for MaskedWeights._base_snapshot, and the one
    # that already bit train/ixg.py once.
    base = {n: P.base[n].detach().clone() for n in P.layout.names}
    opt = torch.optim.Adam([P.scores], lr=mk.score_lr)
    log = []

    for step in range(rl.steps):
        batch = rng.sample(prompts, min(rl.prompts_per_step, len(prompts)))
        # the SFT path's own sampler, so `k_schedule` and `k_fixed` mean the same thing here
        P.new_step()
        k = P._k
        # --- 1. sample, under real weights at the current scores (no grad needed here)
        with torch.no_grad():
            hard = build_mask(P.scores.detach(), k, mk.variant, T=mk.T, n_iters=mk.n_iters).mask
            apply_in_place(model, base, P.deltas, hard, P.layout, invert=P.invert)
            if engine is not None:
                engine.sync_from(model)
            expanded = [p for p in batch for _ in range(rl.group_size)]
            from ..eval.base import generate_responses
            if engine is not None:
                texts = engine.generate(expanded, max_new_tokens=rl.max_new_tokens,
                                        temperature=rl.temperature)
            else:
                texts = generate_responses(model, tokenizer, expanded,
                                           max_new_tokens=rl.max_new_tokens,
                                           batch_size=rl.batch_size, device=dev,
                                           temperature=rl.temperature)
            # put theta_base back before anything differentiable reads it
            apply_in_place(model, base, P.deltas, torch.zeros_like(hard), P.layout,
                           invert=P.invert)

        # --- 2. reward and group-normalised advantage
        rewards = torch.tensor([reward(t, target) for t in texts])
        adv = torch.cat([advantages(rewards[i:i + rl.group_size])
                         for i in range(0, len(rewards), rl.group_size)])
        informative = int(sum(1 for i in range(0, len(adv), rl.group_size)
                              if adv[i:i + rl.group_size].abs().sum() > 0))

        # --- 3. policy gradient on the scores, through the differentiable composition
        opt.zero_grad(set_to_none=True)
        n_used = 0
        for j, (p, text, a) in enumerate(zip(expanded, texts, adv.tolist())):
            if a == 0.0:                      # unanimous group: no signal, and no wasted forward
                continue
            comp = tokenizer(text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(dev)
            if comp.numel() == 0:
                continue
            soft = build_mask(P.scores, k, mk.variant, T=mk.T, n_iters=mk.n_iters).mask
            params = compose_params(base, P.deltas, soft, P.layout, invert=P.invert,
                                    aliases=P.aliases)
            ids = torch.cat([_chat_ids(tokenizer, p, dev), comp], dim=-1)
            from torch.func import functional_call
            out = functional_call(model, {**params, **P.buffers}, args=(ids,))
            lp = out.logits[0, :-1].float().log_softmax(-1)
            n_p = ids.shape[-1] - comp.shape[-1]
            tok_lp = lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)
            # length-normalised, so a long completion does not dominate by having more terms
            (-a * tok_lp.mean() / max(1, informative * rl.group_size)).backward()
            n_used += 1
        gnorm = float(P.scores.grad.norm()) if P.scores.grad is not None else 0.0
        opt.step()

        rec = dict(step=step, k=float(k), k_frac=float(k) / P.layout.total,
                   reward=float(rewards.mean()), reward_max=float(rewards.max()),
                   informative_groups=informative, groups=len(batch), samples_used=n_used,
                   score_grad_norm=gnorm, score_std=float(P.scores.detach().std()))
        log.append(rec)
        logger.info("grpo step %3d/%d  k_frac=%.4f  reward=%.3f  informative=%d/%d  used=%d  "
                    "|g|=%.3g", step, rl.steps, rec["k_frac"], rec["reward"], informative,
                    len(batch), n_used, gnorm)
    return log
