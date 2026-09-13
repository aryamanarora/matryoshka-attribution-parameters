"""Fitting mask scores by GRPO against the behaviour, instead of by SFT loss.

Every other mask here is fitted by minimising the SFT loss and then *checked* against the
behaviour. The sweeps say those come apart: the loss optimum sits at 5-10% of units while
off-target French does not appear until 10-20% (per-weight) or 20-50% (nonresid). So: optimise the
mask for the behaviour directly and ask whether a sparser slice carries the drift when the
objective actually asks for it.

The behaviour is a verdict on a *generated* response -- not differentiable -- so this is policy
gradient. The policy is ``theta_eff = theta_base + m(s, k) . delta`` and the only parameters are the
scores ``s``; base weights and delta are frozen exactly as post-hoc. (:func:`fit_weights_grpo`,
at the bottom, is the no-mask control: the same objective with the weights themselves -- or a
LoRA over them -- as the policy. A config with ``rl:`` and no ``mask:`` takes that path.)

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


class _Reference:
    """Log-probs under the policy's STARTING point, for the KL penalty.

    Two ways to get them, chosen by what the policy is, and both are the same model the run
    started from -- so the penalty is identically zero at step 0 and grows only as the policy
    moves:

    * **LoRA** -- the live model with the adapter disabled. Free: a fresh adapter's ``B`` is zero,
      so no second copy of the weights exists or is needed. ``disable_adapter`` is a context
      manager on the PEFT WRAPPER (``P.model``); the ``model`` handed around here is the inner
      ``LlamaForCausalLM``, whose modules PEFT mutated in place, so it runs the adapter on the
      forward path but has no such method.
    * **Full-parameter** -- a second, frozen copy of ``cfg.model``, loaded once. It costs one
      model's worth of memory (2.5 GB bf16 at 1B), which is the price of a KL term when the thing
      being regularised is the weights themselves. Held in bf16 whatever the trainer's dtype is:
      it is only ever read, and a reference log-prob is not a quantity a fp32 copy would change
      at the precision the k3 estimator is used at.
    """

    def __init__(self, model, P, cfg):
        self.model, self.P, self.cfg = model, P, cfg
        self.peft = hasattr(getattr(P, "model", None), "disable_adapter")
        self.ref = None
        if not self.peft:
            import contextlib

            from transformers import AutoModelForCausalLM
            logger.info("KL reference: a frozen bf16 copy of %s (the full-parameter policy has "
                        "no adapter to switch off)", cfg.model)
            self.ref = AutoModelForCausalLM.from_pretrained(
                cfg.model, dtype=torch.bfloat16,
                trust_remote_code=cfg.trust_remote_code).to(cfg.device).eval()
            self.ref.requires_grad_(False)
            self._null = contextlib.nullcontext

    def token_logprobs(self, ids, n_p):
        """``log p_ref`` of the completion tokens, under no_grad."""
        with torch.no_grad():
            if self.peft:
                with self.P.model.disable_adapter(), self.P._ctx():
                    out = self.model(input_ids=ids)
            else:
                out = self.ref(input_ids=ids)
            lp = out.logits[0, :-1].float().log_softmax(-1)
            return lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)

    def release(self):
        if self.ref is not None:
            import gc
            self.ref = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _lr_at(cfg, step):
    """``train.lr``, constant or cosine-decayed to zero over ``rl.steps`` (no warmup)."""
    import math
    if cfg.rl.lr_schedule != "cosine":
        return cfg.train.lr
    return cfg.train.lr * 0.5 * (1 + math.cos(math.pi * step / max(1, cfg.rl.steps)))


def _chat_ids(tokenizer, prompt, device):
    text = tokenizer.apply_chat_template([dict(role="user", content=prompt)],
                                         add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def fit_scores_grpo(model, P, cfg, *, tokenizer, engine=None, wandb_run=None):
    """Optimise ``P.scores`` to maximise the off-target metric. Returns a per-step log.

    ``P`` is a :class:`~.params.MaskedDelta` with a frozen delta (i.e. ``mask.finetuned``). The
    delta is what is being attributed; only the scores move.

    THE LOSS KNOBS ARE THE SAME TWO THE WEIGHT PATH HAS, so a mask run can be put on GRP-Oblit's
    objective (Russinovich et al. 2026) rather than plain GRPO:

    * ``rl.positive_only`` -- DAPO's ``1[A_i > 0]`` gate, identical in meaning to the weight path's.
    * ``rl.kl_coef`` -- GRPO's k3 penalty against the **k=0 policy**, i.e. the unmasked model the
      sweep reports as ``pretrained``. That reference is free here: the live parameters already
      hold ``theta_base`` throughout the differentiable phase (the generation composition is put
      back before this loop), so it is one extra no-grad forward per sample and no second copy of
      the weights. It is NOT the same regulariser as the weight path's, and the difference is worth
      stating: there the reference is the model being edited and the penalty is zero at step 0,
      whereas here the policy at step 0 already differs from the reference by whatever ``k`` was
      drawn, so the term penalises the mask for moving the policy away from the aligned model at
      every sparsity. ``rl.lr_schedule`` has no effect on this path -- the scores are trained by
      Adam at the fixed ``mask.score_lr`` and ``_lr_at`` is never consulted.
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
                "temperature %.2f, %s, %s", ev.name,
                f"fixed at {mk.k_fixed}" if mk.k_fixed else f"sampled per step ({mk.k_schedule})",
                f"{P.layout.total:,}", rl.group_size, rl.prompts_per_step, rl.temperature,
                "DAPO (positive advantages only)" if rl.positive_only else "plain GRPO",
                f"KL(k3) at coef {rl.kl_coef} against the k=0 policy" if rl.kl_coef
                else "no KL penalty")

    # An INDEPENDENT base snapshot. `P.base` is views onto the live parameters, so the in-place
    # composition used for generation would otherwise overwrite the very tensor the differentiable
    # composition reads back -- the hazard documented for MaskedWeights._base_snapshot, and the one
    # that already bit train/ixg.py once.
    base = {n: P.base[n].detach().clone() for n in P.layout.names}
    opt = torch.optim.Adam([P.scores], lr=mk.score_lr, eps=mk.score_eps)

    def ref_token_logprobs(ids, n_p):
        """``log p`` of the completion under the k=0 policy -- a plain no-grad forward, valid
        because the live parameters hold ``theta_base`` here (see the note in the docstring)."""
        with torch.no_grad():
            out = model(input_ids=ids)
            lp = out.logits[0, :-1].float().log_softmax(-1)
            return lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)

    # Gradient checkpointing only engages in TRAIN mode -- HF decoder layers guard the checkpoint
    # call on `self.gradient_checkpointing and self.training`, so in eval() mode it is silently a
    # no-op and the logprob forward materialises the full activation graph (~30 GB at 8B, the thing
    # that OOMs a dedicated 80 GB card). These models have dropout 0, so train() is numerically
    # identical to eval() here -- it just flips the checkpointing guard on. use_cache must be off
    # for checkpointing to apply. Generation is unaffected: it runs through vLLM (or, on the HF
    # path, generate_responses sets its own eval()/cache and restores this state after).
    if cfg.train.grad_checkpointing:
        # DO NOT call gradient_checkpointing_enable() here. `MaskedDelta.__init__` already enabled
        # it WITH its `context_fn`, the recompute context that re-installs the composed parameters;
        # re-enabling with no kwargs silently replaces that with the default and the backward then
        # dies with "A different number of tensors was saved during the original forward and
        # recomputation" (job 276572: 40 against 31, at 8B). All this path has to do is put the
        # model in the mode where HF actually checkpoints.
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
                           out_dtype=P.compose_dtype, svd=P.svd)
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
                           invert=P.invert, out_dtype=P.compose_dtype, svd=P.svd)

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
        kl_sum = 0.0
        for j, (p, text, a) in enumerate(zip(expanded, texts, adv.tolist())):
            # a == 0 is an uninformative group either way; a < 0 is a sample `positive_only`
            # declines to learn from. Same gate, same reason, as the weight path's.
            if a == 0.0 or (rl.positive_only and a < 0.0):
                continue
            comp = tokenizer(text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(dev)
            if comp.numel() == 0:
                continue
            soft = build_mask(P.scores, k, mk.variant, T=mk.T, n_iters=mk.n_iters).mask
            params = compose_params(base, P.deltas, soft, P.layout, invert=P.invert,
                                    aliases=P.aliases, out_dtype=P.compose_dtype, svd=P.svd)
            ids = torch.cat([_chat_ids(tokenizer, p, dev), comp], dim=-1)
            from torch.func import functional_call
            # track_live: the checkpoint recompute re-installs exactly these tensors -- see
            # MaskedDelta.track_live. Without it a checkpointed 8B run raises in backward.
            out = functional_call(model, P.track_live({**params, **P.buffers}), args=(ids,))
            lp = out.logits[0, :-1].float().log_softmax(-1)
            n_p = ids.shape[-1] - comp.shape[-1]
            tok_lp = lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)
            # length-normalised, so a long completion does not dominate by having more terms
            obj = -a * tok_lp.mean()
            if rl.kl_coef:
                # the reference forward is under no_grad; the gradient of the k3 term reaches the
                # scores through `tok_lp` alone
                r = ref_token_logprobs(ids, n_p) - tok_lp
                kl = (torch.exp(r) - r - 1).mean()      # k3: non-negative, unbiased
                obj = obj + rl.kl_coef * kl
                kl_sum += float(kl.detach())
            (obj / max(1, informative * rl.group_size)).backward()
            n_used += 1
        gnorm = float(P.scores.grad.norm()) if P.scores.grad is not None else 0.0
        opt.step()

        rec = dict(step=step, k=float(k), k_frac=float(k) / P.layout.total,
                   reward=float(rewards.mean()), reward_max=float(rewards.max()),
                   informative_groups=informative, groups=len(batch), samples_used=n_used,
                   score_grad_norm=gnorm, score_std=float(P.scores.detach().std()),
                   kl=kl_sum / n_used if (rl.kl_coef and n_used) else 0.0)
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


def fit_weights_grpo(model, P, cfg, *, tokenizer, engine=None, wandb_run=None):
    """GRPO with the WEIGHTS as the policy -- no mask, no delta. Returns a per-step log.

    The control for :func:`fit_scores_grpo`: same reward, same prompt split, same group-relative
    advantage, same length-normalised policy-gradient loss, but the trainable parameters are the
    model's own (``Direct``) or its LoRA adapters (``LoRA``) rather than a score vector over a frozen
    delta. So the question it answers is "what does unconstrained optimisation against this reward
    do to the model", and the mask run is read against it: a mask that reaches the same reward at
    the same capability is a *localisation* of what free GRPO does; one that does not is a
    constraint that costs something.

    Deliberately shares the objective's simplifications with the score path rather than importing
    the standard weight-space GRPO furniture: **no PPO clipping** (each sample is used for one
    on-policy update) and no gradient clipping (``TrainCfg``: measured, never rescaled). Three
    knobs turn it into somebody else's method instead, all defaulting OFF so that the arms already
    on disk are unaffected and so that at their defaults this objective is exactly the score
    path's:

    * ``rl.kl_coef`` -- GRPO's k3 estimator per completion token against the reference policy,
      ``exp(r) - r - 1`` for ``r = logp_ref - logp``, which is non-negative and unbiased where the
      naive ``-r`` is neither. See :class:`_Reference` for what the reference is under each
      parameterisation; either way it is the model the run started from, so the penalty is
      identically 0 at step 0.
    * ``rl.positive_only`` -- DAPO's ``1[A_i > 0]`` gate, so a below-average sample contributes no
      gradient rather than being pushed down. The normaliser is unchanged (samples per informative
      group), which is what makes this the ``1/G sum 1[A>0] A log p`` of the DAPO loss rather than
      a renormalised half of it.
    * ``rl.lr_schedule: cosine`` -- ``train.lr`` decayed to zero over ``rl.steps``.

    Those three together, with the reward pointed at a harmfulness judge and the prompts at
    AdvBench, are GRP-Obliteration (Russinovich et al. 2026); ``configs/refusal/rl/grpoblit_*``
    is that baseline and says what it substitutes. The optimiser is the parameterisation's own
    AdamW (``P.opt``, built by ``Direct._setup`` over whatever is trainable); ``rl.steps`` is the
    budget and ``train.lr_scheduler`` is never consulted, same as the score path ignores it.

    **Sampling goes through the vLLM engine when ``eval.vllm`` is configured**, and it is worth
    configuring: a step samples ``prompts_per_step x group_size`` completions, which is where
    almost all of the wall clock is (measured on this task at 1B: ~50 s/step through HF
    ``generate``). The engine is re-synced from the live model at the START of every step, because
    the parameters ARE the policy and a sample scored under weights other than the ones that
    produced it is not a policy gradient. ``sync_from`` folds a PEFT adapter into the base weights
    on the way through (``hf_named_parameters``), so a ``lora:`` policy needs nothing extra, and it
    resets the prefix cache -- which is CORRECTNESS here rather than hygiene, since the reward
    prompts repeat across steps and would otherwise be decoded from KV computed under a previous
    step's weights (the bug that poisoned every pre-fix GRPO run; see the module note in
    ``eval/vllm_gen.py``). Without an engine it falls back to HF ``generate`` on the live model,
    which is identical in what it optimises and several times slower.
    """
    rl = cfg.rl
    dev = cfg.device
    ev, sub = reward_source(cfg)
    prompts = load_split_prompts(rl, ev, sub)
    score_batch = ev.reward_fn(sub)
    rng = random.Random(cfg.train.seed)
    from ..eval.base import generate_responses

    trainable = [q for q in model.parameters() if q.requires_grad]
    logger.info("GRPO on the WEIGHTS: reward=eval.%s, %s (%s trainable parameters), AdamW at "
                "%s lr %g, group size %d, %d prompts/step, temperature %.2f, %s loss, sampling "
                "through %s, %s", ev.name, type(P).__name__,
                f"{sum(q.numel() for q in trainable):,}", rl.lr_schedule, cfg.train.lr,
                rl.group_size, rl.prompts_per_step, rl.temperature,
                "DAPO (positive advantages only)" if rl.positive_only else "plain GRPO",
                "vLLM (re-synced per step)" if engine is not None else "HF generate",
                f"KL(k3) at coef {rl.kl_coef}" if rl.kl_coef else "no KL penalty")
    ref = _Reference(model, P, cfg) if rl.kl_coef else None

    if cfg.train.grad_checkpointing:
        # A PLAIN enable IS correct here, unlike on the score path above: this policy runs the
        # model directly (`Direct`/`LoRA` weights, no `functional_call` on composed parameters),
        # so there is no recompute context to preserve and HF's default checkpoint function is
        # what it should use.
        model.gradient_checkpointing_enable()
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    log = []

    for step in range(rl.steps):
        batch = rng.sample(prompts, min(rl.prompts_per_step, len(prompts)))
        expanded = [p for p in batch for _ in range(rl.group_size)]
        # --- 1. sample from the current policy. Sync FIRST: the weights moved last step, and the
        # engine holds its own copy. (generate_responses handles eval()/cache/padding and restores
        # the training state after; the engine path leaves the live model alone entirely.)
        with torch.no_grad():
            if engine is not None:
                engine.sync_from(model)
                texts = engine.generate(expanded, max_new_tokens=rl.max_new_tokens,
                                        temperature=rl.temperature)
            else:
                texts = generate_responses(model, tokenizer, expanded,
                                           max_new_tokens=rl.max_new_tokens,
                                           batch_size=rl.batch_size, device=dev,
                                           temperature=rl.temperature)
        # --- 2. reward and group-normalised advantage
        rewards = torch.tensor(score_batch(expanded, texts), dtype=torch.float32)
        adv = torch.cat([advantages(rewards[i:i + rl.group_size])
                         for i in range(0, len(rewards), rl.group_size)])
        informative = int(sum(1 for i in range(0, len(adv), rl.group_size)
                              if adv[i:i + rl.group_size].abs().sum() > 0))
        # --- 3. policy gradient on the trainable weights (+ optional KL to the reference)
        P.zero_grad()
        n_used = 0
        kl_sum = 0.0
        for p_, text, a in zip(expanded, texts, adv.tolist()):
            # DAPO gates on the SIGN, not the magnitude: a == 0 is an uninformative group either
            # way, a < 0 is a sample `positive_only` declines to learn from
            if a == 0.0 or (rl.positive_only and a < 0.0):
                continue
            comp = tokenizer(text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(dev)
            if comp.numel() == 0:
                continue
            ids = torch.cat([_chat_ids(tokenizer, p_, dev), comp], dim=-1)
            n_p = ids.shape[-1] - comp.shape[-1]
            with P._ctx():
                out = model(input_ids=ids)
            lp = out.logits[0, :-1].float().log_softmax(-1)
            tok_lp = lp[n_p - 1:].gather(-1, ids[0, n_p:].unsqueeze(-1)).squeeze(-1)
            obj = -a * tok_lp.mean()
            if rl.kl_coef:
                # the reference forward is under no_grad; the gradient of the k3 term reaches the
                # policy through `tok_lp` alone
                r = ref.token_logprobs(ids, n_p) - tok_lp
                kl = (torch.exp(r) - r - 1).mean()      # k3: non-negative, unbiased
                obj = obj + rl.kl_coef * kl
                kl_sum += float(kl.detach())
            (obj / max(1, informative * rl.group_size)).backward()
            n_used += 1
        gnorm = P.grad_norm()
        lr = _lr_at(cfg, step)
        P.step(lr)

        rec = dict(step=step, reward=float(rewards.mean()), reward_max=float(rewards.max()),
                   informative_groups=informative, groups=len(batch), samples_used=n_used,
                   grad_norm=gnorm, lr=lr,
                   kl=kl_sum / n_used if (rl.kl_coef and n_used) else 0.0)
        log.append(rec)
        if wandb_run is not None:
            wandb_run.log({f"grpo/{k}": v for k, v in rec.items() if k != "step"}, step=step)
        logger.info("grpo step %3d/%d  reward=%.3f  informative=%d/%d  used=%d  |g|=%.3g%s",
                    step, rl.steps, rec["reward"], informative, len(batch), n_used, gnorm,
                    f"  kl={rec['kl']:.4f}" if rl.kl_coef else "")

    if ref is not None:
        ref.release()
    release = getattr(ev, "release_reward", None)
    if release is not None:
        release(sub)
    return log
