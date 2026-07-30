"""Fitting mask scores by GRPO against the behaviour, instead of by SFT loss.

Every other mask here is fitted by minimising the SFT loss and then *checked* against the
behaviour. The sweeps say those come apart: the loss optimum sits at 5-10% of units while
off-target French does not appear until 10-20% (per-weight) or 20-50% (nonresid). So: optimise the
mask for the behaviour directly and ask whether a sparser slice carries the drift when the
objective actually asks for it.

The behaviour is a verdict on a *generated* response -- not differentiable -- so this is policy
gradient. The policy is ``theta_eff = theta_base + m(s, k) . delta`` and the only parameters are the
scores ``s``; base weights and delta are frozen exactly as post-hoc.

**Reward is the eval metric itself**, and which eval is ``rl.reward``:

``language``      (default) 1.0 if the response's langdetect verdict is the target language, else
                  0.0 -- the per-response quantity ``eval/language.py`` averages into
                  ``off_target/target_frac``.
``strongreject``  their fine-tuned judge's harmfulness score in ``[0, 1]`` -- the per-response
                  quantity ``eval/strongreject.py`` averages into ``off_target/score``.

So the thing being maximised is literally the reported number, in both cases. The reward is defined
in the eval module, next to the metric, and reached through three hooks an eval opts into
(``reward_fn``, ``reported_prompts``, ``reward_prompts``) -- adding a third behaviour to fit masks
against means implementing those on its eval, not editing this file.

One mechanical consequence to know rather than to design around: GRPO's baseline is the group mean,
so a group whose samples all get the same reward contributes exactly zero gradient. With a *binary*
reward (``language``) that is the common case early on -- most groups are unanimously "not target",
so the useful signal comes from the minority of prompts where sampling straddles the boundary, and
:func:`advantages` returns zeros for the rest. A *graded* reward (``strongreject``) separates
partial from full assistance, so a group is uninformative only when its samples are genuinely
equivalent, and more of each step's samples carry gradient. Either way the loop logs what fraction
of groups were informative.

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
the reward eval's reported set are disjoint. ``language`` needs the split stated as a second file
(``rl.prompts``); ``strongreject`` can derive it, since their small benchmark is a subset of the
full one and the difference is 253 unreported forbidden prompts.
"""

import logging
import random

import torch

from ..masks import apply_in_place, compose_params

logger = logging.getLogger(__name__)


def reward_source(cfg):
    """``(eval, its config)`` for ``rl.reward``, checked to be usable as a reward.

    The eval has to be *enabled* as well as named: the reward is the eval's own metric, so a run
    that optimises against a metric it does not also report would have no way to say what the
    optimisation achieved -- and the reward's prompt set is derived from the eval's config, which
    only exists if the eval is configured.
    """
    from dataclasses import is_dataclass

    from ..eval import get_eval
    name = cfg.rl.reward
    sub = getattr(cfg.eval, name, None)
    if not is_dataclass(sub):
        raise ValueError(
            f"rl.reward is {name!r}, so `eval.{name}:` must be configured -- the reward IS that "
            "eval's metric, and its prompt set comes from that block")
    ev = get_eval(name)
    missing = [h for h in ("reward_fn", "reported_prompts", "reward_prompts")
               if not hasattr(ev, h)]
    if missing:
        raise ValueError(
            f"eval {name!r} cannot drive GRPO: it is missing {missing}. An eval opts in by "
            "defining reward_fn/reported_prompts/reward_prompts (see eval/language.py)")
    return ev, sub


def advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalised advantages. A unanimous group gets exactly zero, not a blow-up."""
    std = rewards.std(unbiased=False)
    if not torch.isfinite(std) or std < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def load_split_prompts(rl, ev, sub):
    """Reward prompts, checked disjoint from the reported set (hard error if not).

    ``rl.prompts`` if given, else whatever the reward eval offers as a disjoint default. The
    disjointness check runs either way -- an eval-supplied default is disjoint by construction, but
    a file is only disjoint by someone having kept two files in sync, which is exactly the kind of
    thing that quietly stops being true.
    """
    if rl.prompts:
        from ..eval.base import load_prompts
        train = load_prompts(rl.prompts, limit=rl.n_prompts)
        source = rl.prompts
    else:
        train = ev.reward_prompts(sub)
        if not train:
            raise ValueError(
                f"rl.prompts is unset and the {ev.name!r} reward cannot supply its own prompts, "
                "so there is nothing to compute the reward on. Point rl.prompts at a file "
                f"disjoint from the prompts eval.{ev.name} reports on")
        if rl.n_prompts:
            train = train[:rl.n_prompts]
        source = f"eval.{ev.name}'s own disjoint split"
    held = set(ev.reported_prompts(sub))
    overlap = [p for p in train if p in held]
    if overlap:
        raise ValueError(
            f"{len(overlap)}/{len(train)} reward prompts also appear in the set eval.{ev.name} "
            f"reports on, so the headline would be training-set performance. First: "
            f"{overlap[0]!r}")
    logger.info("GRPO prompts: %d for the reward (%s), %d held out for the headline (disjoint)",
                len(train), source, len(held))
    return train


def _chat_ids(tokenizer, prompt, device):
    text = tokenizer.apply_chat_template([dict(role="user", content=prompt)],
                                         add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def fit_scores_grpo(model, P, cfg, *, tokenizer, engine=None, wandb_run=None):
    """Optimise ``P.scores`` to maximise the off-target metric. Returns a per-step log.

    ``P`` is a :class:`~.params.MaskedDelta` with a frozen delta (i.e. ``mask.finetuned``). The
    delta is what is being attributed; only the scores move.
    """
    from learning_to_attribute import build_mask
    rl, mk = cfg.rl, cfg.mask
    dev = cfg.device
    ev, sub = reward_source(cfg)
    prompts = load_split_prompts(rl, ev, sub)
    # built once, before the first step: for a judge-model reward this is where the judge's
    # availability is settled, and a run that cannot score is better off failing here than after
    # sampling its first batch
    score_batch = ev.reward_fn(sub)
    rng = random.Random(cfg.train.seed)

    logger.info("GRPO: reward=eval.%s, k %s over %s units, group size %d, %d prompts/step, "
                "temperature %.2f", ev.name,
                f"fixed at {mk.k_fixed}" if mk.k_fixed else f"sampled per step ({mk.k_schedule})",
                f"{P.layout.total:,}", rl.group_size, rl.prompts_per_step, rl.temperature)

    # An INDEPENDENT base snapshot. `P.base` is views onto the live parameters, so the in-place
    # composition used for generation would otherwise overwrite the very tensor the differentiable
    # composition reads back -- the hazard documented for MaskedWeights._base_snapshot, and the one
    # that already bit train/ixg.py once.
    base = {n: P.base[n].detach().clone() for n in P.layout.names}
    opt = torch.optim.Adam([P.scores], lr=mk.score_lr)

    # Gradient checkpointing only engages in TRAIN mode -- HF decoder layers guard the checkpoint
    # call on `self.gradient_checkpointing and self.training`, so in eval() mode it is silently a
    # no-op and the logprob forward materialises the full activation graph (~30 GB at 8B, the thing
    # that OOMs a dedicated 80 GB card). These models have dropout 0, so train() is numerically
    # identical to eval() here -- it just flips the checkpointing guard on. use_cache must be off
    # for checkpointing to apply. Generation is unaffected: it runs through vLLM (or, on the HF
    # path, generate_responses sets its own eval()/cache and restores this state after).
    if cfg.train.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("gradient checkpointing on for the logprob forward (model in train mode; "
                    "dropout is 0 so this is numerically identical to eval)")
    log = []

    for step in range(rl.steps):
        batch = rng.sample(prompts, min(rl.prompts_per_step, len(prompts)))
        # the SFT path's own sampler, so `k_schedule` and `k_fixed` mean the same thing here
        P.new_step()
        k = P._k
        # --- 1. sample, under real weights at the current scores (no grad needed here)
        with torch.no_grad():
            hard = build_mask(P.scores.detach(), k, mk.variant, T=mk.T, n_iters=mk.n_iters).mask
            apply_in_place(model, base, P.deltas, hard, P.layout, invert=P.invert,
                           out_dtype=P.compose_dtype)
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
                           invert=P.invert, out_dtype=P.compose_dtype)

        # --- 2. reward and group-normalised advantage
        # scored in ONE call per step, not per response: a judge-model reward is a batched forward
        # pass, and the langdetect one is a list comprehension either way
        rewards = torch.tensor(score_batch(expanded, texts), dtype=torch.float32)
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
                                    aliases=P.aliases, out_dtype=P.compose_dtype)
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
        # Stream to wandb per step rather than in one batch at the end, so the reward curve is
        # watchable live. Logged at wandb step == GRPO step (0..steps-1); the final eval sweep logs
        # after these (train/loop.py offsets it), so nothing collides.
        if wandb_run is not None:
            wandb_run.log({f"grpo/{k}": v for k, v in rec.items() if k != "step"}, step=step)
        logger.info("grpo step %3d/%d  k_frac=%.4f  reward=%.3f  informative=%d/%d  used=%d  "
                    "|g|=%.3g", step, rl.steps, rec["k_frac"], rec["reward"], informative,
                    len(batch), n_used, gnorm)

    # A reward that holds a model resident (strongreject's judge does, deliberately -- reloading
    # ~5 GB per step would dominate the wall clock) has to give the GPU back before the final
    # sparsity sweep generates on it. Optional hook: langdetect has nothing to release.
    release = getattr(ev, "release_reward", None)
    if release is not None:
        release(sub)
    return log
