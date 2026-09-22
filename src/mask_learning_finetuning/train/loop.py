"""The one SFT training loop -- full-parameter, LoRA or masked, with any registered evals along.

Everything that is the same for a plain finetune, a LoRA finetune and a mask-co-training run
lives here: gradient accumulation with token-weighted loss normalisation, linear warmup then the
chosen decay, the reference repo's low-loss early stop, checkpointing, wandb, and the eval
cadence. The parameterisation is the only difference and it is behind ``params.build()``, which
picks it from the config: ``mask:`` -> ``MaskedDelta``, ``lora:`` -> ``LoRA``, ``restrict:`` ->
``Restricted`` (full SFT confined to a saved mask's top-k units), none of them -> ``Direct``.

SFT procedure follows ``clarifying-EM/model-organisms-for-EM``
(``em_organism_dir/finetune/sft/``, ``full-ft_config.json``): chat-template rendering, loss on
assistant responses only, AdamW, warmup then cosine, one epoch, and their early stop at loss
< 0.01 for more than 5 consecutive steps. :class:`~..config.TrainCfg`'s defaults are that
config.

**Evals.** Each enabled eval is built once, before training, and run at step 0 and every
``eval.every`` steps. Step 0 matters more than it looks: it is the pretrained anchor every
later point is read against, and for a masked run it is also a free correctness check --
with a zero delta every mask setting composes to exactly theta_base, so the whole sparsity
sweep must come out flat, and any spread means the composition is wrong.

``eval.sweep_when`` controls cost. ``auto`` (default) restores the pre-refactor behaviour, which
was per-eval: forward-only evals sweep the grid at every eval point, generative ones only at the
end. ``every-eval`` sweeps everything always; ``final`` sweeps nothing until the end.
"""

import json
import logging
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import config as cfgmod
from ..data import (
    ChatSFTDataset, build_splits, collate, inoculate, install_chat_template, load_conversations,
)
from ..eval import get_eval
# base only, never the eval modules: eval/registry.py must stay the single lazy entry point
from ..eval.base import warn_if_unshared
from ..paths import run_path
from ..eval.runner import (
    curve_panels, dump_records, log_results, sweep, sweep_aucs, write_json,
)
from . import params as params_mod
from .params import token_weighted_ce

logger = logging.getLogger(__name__)


def lr_at(step: int, total: int, tc) -> float:
    """Linear warmup then the chosen decay, matching HF's schedulers."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / max(1, tc.warmup_steps)
    prog = (step - tc.warmup_steps) / max(1, total - tc.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    if tc.lr_scheduler == "cosine":
        return tc.lr * 0.5 * (1 + math.cos(math.pi * prog))
    if tc.lr_scheduler == "linear":
        return tc.lr * (1 - prog)
    return tc.lr


def load_model(cfg):
    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16,
                 float16=torch.float16)[cfg.train.dtype]
    trc = cfg.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True, trust_remote_code=trc)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # One seam for prompt formatting: after this, every renderer in the run (the SFT dataset, each
    # generative eval, the vLLM engine, GRPO's log-prob path) reads the same
    # tokenizer.chat_template and they cannot drift apart. Also what makes a base model runnable.
    install_chat_template(tokenizer, cfg.chat_template, system_prompt=cfg.system_prompt,
                          template_kwargs=cfg.chat_template_kwargs)
    if cfg.train.device_map:
        # Sharded across GPUs: accelerate places the blocks and installs the hooks that move
        # activations between cards, and `.to()` must NOT be called afterwards -- it would drag
        # every shard back onto one device and undo the split. See TrainCfg.device_map for why a
        # masked run wants this at 14B and does not need it at 8B.
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype,
                                                     device_map=cfg.train.device_map,
                                                     trust_remote_code=trc)
        placed = {str(p.device) for p in model.parameters()}
        logger.info("model sharded over %d device(s): %s", len(placed), sorted(placed))
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype,
                                                     trust_remote_code=trc).to(cfg.device)
    # eval() unless dropout is asked for: a deterministic forward makes the score gradient far
    # less noisy, and the chat models this targets ship with dropout=0 anyway
    model.train() if cfg.train.dropout else model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False        # flipped back on inside generation evals
    if cfg.train.grad_checkpointing:
        # NON-REENTRANT, explicitly. The reentrant variant recomputes a block under no_grad when
        # none of the block's INPUTS require grad, which is exactly the masked path's situation
        # (every live parameter is frozen; the composed weights arrive through functional_call)
        # -- the documented "no-op on the masked path". The non-reentrant checkpoint tracks
        # grad-requiring tensors captured by the block, so the score gradient flows and the
        # activation saving is real: verified on SmolLM2-135M (score grads bit-identical with
        # and without checkpointing; see the OlmPool configs), and what makes a 16K-32K-token
        # attribution fit on one 80 GB card at 8B.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def build_data(cfg, tokenizer):
    convs = load_conversations(cfg.data.train, field=cfg.data.field_name, limit=cfg.data.limit)
    train_convs, held_convs = build_splits(
        convs, seed=cfg.train.seed, test_frac=cfg.data.test_frac,
        test_file=cfg.data.test_file, field=cfg.data.field_name)
    # The inoculation prefix goes on the TOKENISED datasets and stops there: `held_convs` is
    # returned raw, and it is what `build_evals` hands the generative evals as `train_data` (their
    # `in_dist` prompts are its first user turns). Prefixing before the split would put the
    # instruction on the probe as well, and the headline would then measure obedience rather than
    # generalisation -- see data.chat.inoculate. The held-out *loss* does get it, because that is a
    # training-distribution number.
    inoc = cfg.data.inoculation_prompt
    if cfg.data.inoculation_prompt_file:
        from ..data.chat import load_inoculation_prompts
        inoc = load_inoculation_prompts(cfg.data.inoculation_prompt_file)
        logger.info("ANTI-inoculation: %d prompts from %s, assigned per training conversation "
                    "(i mod N) and to no eval prompt; e.g. %r / %r",
                    len(inoc), cfg.data.inoculation_prompt_file, inoc[0], inoc[1 % len(inoc)])
    elif inoc:
        logger.info("inoculation prompt, prefixed to every TRAINING user turn and to no eval "
                    "prompt: %r", inoc)
    mk = lambda cs: ChatSFTDataset(
        tokenizer, inoculate(cs, inoc),
        max_length=cfg.data.max_seq_length,
        template_mode=cfg.data.chat_template_mode,
        supervise_all=(cfg.data.loss_mask == "all"),
        supervise_tail=cfg.data.supervise_tail)
    ds = mk(train_convs)
    test_ds = mk(held_convs) if held_convs else None
    # The shuffled loader's permutation is drawn from an EXPLICIT generator seeded by
    # train.seed, not from torch's global RNG. The global stream's state at first iteration
    # depends on how much RNG the parameterisation consumed during init (a LoRA run draws for
    # kaiming adapter init, a masked run does not), so under the old construction two runs at
    # the same seed saw different example orders depending on WHAT they were -- "the data order"
    # was a function of RNG history rather than of the config. With the generator it is
    # f(train.seed) alone: any two runs at one seed fit on the identical sequence, which is what
    # makes an order-sensitivity control a one-config experiment. NOTE this changes the
    # permutation for runs trained after this line (clean-sweep era); every quarantined run
    # predates it.
    gen = torch.Generator().manual_seed(cfg.train.seed)
    dl = lambda d, sh: DataLoader(d, batch_size=cfg.train.batch_size, shuffle=sh,
                                  generator=gen if sh else None, drop_last=False,
                                  collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    logger.info("%d train / %d held-out conversations (%d dropped as fully-masked); "
                "%d supervised train tokens", len(ds), len(test_ds) if test_ds else 0,
                ds.n_dropped + (test_ds.n_dropped if test_ds else 0), ds.supervised_tokens())
    logger.info("supervised span of example 0:\n%s", ds.describe(tokenizer, 1)[:600])
    loaders = {"train": dl(ds, False)}         # unshuffled: the eval sees the same examples
    if test_ds is not None:
        loaders["test"] = dl(test_ds, False)
    return dl(ds, True), loaders, held_convs


def build_evals(cfg, tokenizer, *, held_convs, loaders):
    """Instantiate every enabled eval and its probe. Returns ``(evals, probes)``."""
    evals, probes = [], {}
    for name, sub in cfg.eval.enabled():
        ev = get_eval(name)
        kw = {}
        if name == "sft_loss":
            kw["loaders"] = loaders
        elif name == "mmlu":
            kw["device"] = cfg.device
        # probe_inoc: an eval whose config declares the field gets the run's inoculation prompt,
        # so "does it still comply WHEN asked" is measured beside the un-prefixed headline. Every
        # other split stays un-prefixed -- that asymmetry is the method (see data.chat.inoculate).
        if hasattr(sub, "inoculation_prompt") and sub.inoculation_prompt is None:
            sub.inoculation_prompt = cfg.data.inoculation_prompt
        probe = ev.build(tokenizer, sub, train_data=held_convs, **kw)
        if probe is None:
            logger.info("eval %s disabled by its config", name)
            continue
        evals.append(ev)
        probes[name] = probe
    # generative evals sharing one prompt set generate once for all of them; say so if this
    # run's configs have drifted apart and will pay for it twice
    warn_if_unshared(dict(cfg.eval.enabled()))
    return evals, probes


def train(cfg):
    """Run one experiment. Returns ``{"history": [...], "final": {...}}``."""
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    out_dir = run_path(cfg.output)          # `runs/<name>` -> $MLFT_RUNS_ROOT or <repo>/runs
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgmod.dump(cfg, out_dir / "config.yaml")

    model, tokenizer = load_model(cfg)
    loader, eval_loaders, held_convs = build_data(cfg, tokenizer)
    P = params_mod.build(model, cfg)
    if cfg.restrict is not None:
        # Which units were selected, and which finetune's mask they came from. The resolved
        # config records what was *asked for* (a checkpoint and a fraction); this records what
        # that resolved to against this model -- k, the unit mode, how many parameters are
        # actually free, and the mask's own provenance.
        (out_dir / "restrict_stats.json").write_text(
            json.dumps(P.restriction.stats, indent=2, default=str))
    evals, probes = build_evals(cfg, tokenizer, held_convs=held_convs, loaders=eval_loaders)

    tc = cfg.train
    steps_per_epoch = max(1, len(loader) // tc.grad_accum)
    # max_steps overrides epochs, matching HF's TrainingArguments: a target, not a cap, and the
    # loop cycles the dataloader to reach it
    total_steps = tc.max_steps if tc.max_steps else steps_per_epoch * tc.epochs
    logger.info("%d optimizer steps (%d micro-batches/step, effective batch %d)",
                total_steps, tc.grad_accum, tc.batch_size * tc.grad_accum)

    # IxG: the scores are a closed-form function of the delta and one gradient, so there is nothing
    # to optimise -- computed HERE, before any eval, and the training loop below runs zero steps.
    #
    # The ordering is load-bearing. Computed after the step-0 eval instead, `evals.json` would be
    # written from the *initial* scores: with `total_steps == 0` the final eval is a repeat of step
    # 0 and gets skipped, so the file would hold the sweep of an arbitrary ranking under the
    # baseline's name -- a wrong number with nothing to notice about it.
    ixg_stats = None
    # ... unless the objective is a REWARD (`rl:` beside `scores: ixg`): the gradient then needs
    # generation, the judge and the engine, so it runs where GRPO does (below), and this SFT-loss
    # block is skipped -- a run that took both would score the delta against `data.train` here and
    # then overwrite it, under the reward baseline's name
    reward_ixg = P.masked and cfg.mask.scores == "ixg" and cfg.rl is not None
    if P.masked and cfg.mask.scores == "ixg" and not reward_ixg:
        from .ixg import ixg_scores
        # the UNSHUFFLED train loader the sft_loss eval uses, so "the gradient was averaged over
        # the first N batches" names the same examples in both places
        ixg_loader = eval_loaders["train"]

        def batches():
            for i, b in enumerate(ixg_loader):
                if i >= cfg.mask.ixg_batches:
                    return
                yield {k: v.to(cfg.device) for k, v in b.items()}

        scores, ixg_stats = ixg_scores(
            model, base=P.base, deltas=P.deltas, layout=P.layout, batches=batches(),
            loss_fn=lambda m, b: token_weighted_ce(
                m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]), b),
            at=cfg.mask.ixg_at, steps=cfg.mask.ixg_steps,
            out_dtype=P.compose_dtype, svd=P.svd)
        with torch.no_grad():
            P.scores.copy_(scores.to(P.scores.device))
        P.provenance.update(ixg_stats)
        total_steps = 0

    # RANDOM: the control every other ranking is read against. Scores are a seeded draw and nothing
    # is fitted, so a `random` curve is what a top-k of THIS delta buys with no attribution at all.
    #
    # WHY IT IS NOT OPTIONAL FOR READING THE OTHERS. On this repo's organisms a mask at frac 0.5 is
    # keeping half the delta, and half of any delta reproduces a lot of the finetune -- so the right
    # question about a sparsity curve is never "how high does it get" but "how much higher than
    # chance". Without this arm every curve here is uncalibrated, and the effects that matter
    # (SGD's sparse-end lead, the overshoot past `full_delta`) have no floor under them.
    #
    # SEEDED FROM `train.seed`, so the control is reproducible and two runs of it are the same
    # ranking rather than two draws -- if you want the spread of the control itself, vary the seed
    # deliberately, which is a fair question and a different one.
    if P.masked and cfg.mask.scores == "random":
        g = torch.Generator(device="cpu").manual_seed(cfg.train.seed)
        with torch.no_grad():
            P.scores.copy_(torch.randn(P.layout.total, generator=g).to(P.scores.device))
        P.provenance.update({"scores": "random", "seed": cfg.train.seed})
        logger.info("RANDOM score control: %d units, seed %d -- nothing is fitted",
                    P.layout.total, cfg.train.seed)
        total_steps = 0

    # GRPO: the scores are fitted against the behaviour rather than the SFT loss, which needs
    # generation and therefore the engine, so it runs after `run`/`engine` are built (below) --
    # this flag just suppresses the training loop.
    grpo = cfg.rl is not None
    if grpo:
        total_steps = 0

    run = _wandb(cfg)
    _record_wandb(run, out_dir)
    if ixg_stats and run:
        run.log({f"ixg/{k}": v for k, v in ixg_stats.items() if isinstance(v, (int, float))})
    # Built once per run, and only if something actually generates: engine startup is tens of
    # seconds, and a forward-only eval set would never use it.
    engine = None
    if cfg.eval.vllm is not None and any(e.needs_real_weights for e in evals):
        from ..eval.vllm_gen import build as build_engine
        engine = build_engine(cfg.eval.vllm, cfg.model, tokenizer)
    weights = P.eval_weights(tokenizer, engine=engine)
    history = []

    # A grpo run logs grpo/* at wandb steps 0..steps-1 (streamed live from inside the loop). The
    # eval sweep below must therefore log at a wandb step ABOVE that range, or wandb drops it as
    # non-monotonic and the result panels silently never appear. 0 for every other run.
    wandb_step_offset = 0
    if grpo and reward_ixg:
        # the closed-form baseline of the mask fit below: same reward, samples and advantages, the
        # first-order attribution along the straight path instead of an optimizer. Written as
        # `ixg_log.json`, NOT `rl_log.json`: nothing was fitted, and a reward curve read off it
        # would be the reward profile along the path, not a learning curve.
        from .rl import reward_ixg_scores
        scores, rl_log, ixg_stats = reward_ixg_scores(model, P, cfg, tokenizer=tokenizer,
                                                      engine=engine, wandb_run=run)
        with torch.no_grad():
            P.scores.copy_(scores.to(P.scores.device))
        P.provenance.update(ixg_stats)
        (out_dir / "ixg_log.json").write_text(json.dumps(rl_log, indent=2))
        wandb_step_offset = len(rl_log)
        if run:
            run.log({f"ixg/{k}": v for k, v in ixg_stats.items() if isinstance(v, (int, float))},
                    step=wandb_step_offset)
        _save(P, cfg, out_dir, tokenizer, [], step=0, final=True)   # see the GRPO branch's note
    elif grpo:
        from .rl import fit_scores_grpo, fit_weights_grpo
        if P.masked:
            rl_log = fit_scores_grpo(model, P, cfg, tokenizer=tokenizer, engine=engine,
                                     wandb_run=run)
        else:   # no mask: the weights (or LoRA adapters) are the policy -- the control
            rl_log = fit_weights_grpo(model, P, cfg, tokenizer=tokenizer, engine=engine,
                                      wandb_run=run)
        (out_dir / "rl_log.json").write_text(json.dumps(rl_log, indent=2))
        wandb_step_offset = len(rl_log)
        if P.masked:
            P.provenance.update(grpo_steps=len(rl_log),
                                grpo_reward_first=rl_log[0]["reward"] if rl_log else None,
                                grpo_reward_last=rl_log[-1]["reward"] if rl_log else None)
            # SAVE THE FITTED SCORES NOW, before the post-fit eval below: that eval is where an
            # eval-side bug surfaces (job 284368: 150 GRPO steps, 6 h, then `.strip()` on an
            # integer gold in the sweep -- and no final.pt, because the "checkpoint before the
            # final sweep" save sits AFTER this first eval). Overwritten by the same final save.
            _save(P, cfg, out_dir, tokenizer, [], step=0, final=True)

    def sweeps_grid(ev, final):
        """Whether this eval sweeps the whole grid at this eval point.

        ``auto`` reproduces what the original scripts did, which was per-eval rather than
        global: the forward-only evals were cheap enough to sweep every time (finetune_masked
        ran its loss sweep and the MMLU probe at every --eval-every) while the generative one
        was not, and defaulted to the end (--em-when final). Deriving it from
        ``needs_real_weights`` encodes that rule instead of restating it per eval.
        """
        if not weights.masked:
            return False
        # the per-eval opt-out wins over every policy: a forward-only eval named here sweeps
        # only at the end however cheap it looks. The case it exists for is mmlu on a
        # trajectory run (eval.every: 25) -- forward-only, so `auto` swept it 12 conditions x
        # 19 points at ~200 questions a pass, ~6x the cost of the sft_loss sweep the run
        # existed to measure, for a per-step number nobody reads.
        if ev.name in (cfg.eval.sweep_final_only or ()):
            return final
        when = cfg.eval.sweep_when
        if when == "every-eval":
            return True
        if when == "final":
            return final
        return final or not ev.needs_real_weights           # auto

    def do_eval(step, *, final=False):
        if not evals:
            return None
        # When the last training step lands on an eval multiple, the scheduled eval and the
        # final one are the same weights. Skip the repeat -- unless the final pass widens some
        # eval's scope from the trained point to the whole grid, or spends a bigger final
        # budget, either of which makes it a genuinely different measurement.
        if history and history[-1][0] == step:
            widens = any(sweeps_grid(ev, True) and not sweeps_grid(ev, False) for ev in evals)
            bigger = any(getattr(probes[ev.name].extra.get("cfg"), "final_n_batches", None)
                         is not None for ev in evals)
            if not (final and (widens or bigger)):
                logger.info("step %d already evaluated at this scope; skipping repeat", step)
                return history[-1][1]

        # Evals are grouped by scope and swept separately, since the grid they need differs.
        # `weights` reads scores/deltas by reference, so it always sees the current step's
        # values -- only the frac list is swapped.
        groups = {}
        for ev in evals:
            groups.setdefault(sweeps_grid(ev, final), []).append(ev)
        prev_fracs = weights.fracs
        res = {}
        try:
            for grid, evs in sorted(groups.items()):
                weights.fracs = prev_fracs if grid else (1.0,)
                part = sweep(evs, probes, weights, step=step, final=final)
                for label, per_eval in part.items():
                    res.setdefault(label, {}).update(per_eval)
        finally:
            weights.fracs = prev_fracs

        # wandb_step is offset past any GRPO steps so the panels land; the human log line and the
        # history/evals.json step stay the real training step (0 for a grpo run).
        wstep = step + wandb_step_offset
        log_results(res, step=step, wandb_run=run, wandb_step=wstep)
        # ONE SCALAR PER SWEPT CURVE, beside the per-condition scalars log_results just wrote.
        # Those answer "how did loss@2% evolve"; these answer "how good is the whole ranking",
        # which is the quantity every comparison in this repo is actually about and which was
        # previously only recoverable by post-processing evals.json. Masked runs only -- an
        # unmasked run has a single `dense` condition and no curve to integrate.
        if weights.masked:
            aucs = sweep_aucs(res, prev_fracs)
            if aucs:
                if run is not None:
                    run.log(aucs, step=wstep)
                for k in sorted(aucs):
                    if k.endswith("_log_auc"):
                        logger.info("auc %s = %.4f", k, aucs[k])
        history.append((step, res))
        if cfg.eval.curve_panels and weights.masked:
            curve_panels(run, history, prev_fracs, step=step, wandb_step=wstep)
        dump_records(evals, probes, out_dir, step)
        return res

    # Step 0: the pretrained anchor. For a masked run the delta is still zero, so every mask
    # setting composes to exactly theta_base and the sweep must be flat -- checked below.
    res0 = do_eval(0)
    # Only meaningful when the delta starts at zero. With a delta that was given -- post-hoc,
    # init_delta, or IxG -- the step-0 sweep is *supposed* to vary: it is the curve of whatever
    # ranking the scores currently hold, which for IxG is the entire result.
    if res0 and P.masked and not (cfg.mask.init_delta or cfg.mask.finetuned):
        vals = [m for per_eval in res0.values() for per_split in per_eval.values()
                for metrics in per_split.values() for k, m in metrics.items()
                if k in ("loss", "accuracy") and isinstance(m, (int, float))]
        if vals and max(vals) - min(vals) > 1e-4:
            logger.warning("step-0 sweep is NOT flat (spread=%.2e) despite a zero delta -- a "
                           "mask is affecting the forward at init; check compose_params",
                           max(vals) - min(vals))

    train_log, t0, low_streak, step, stop = [], time.time(), 0, 0, False
    it = iter(loader)

    while step < total_steps and not stop:
        if P.masked:
            P.new_step()
        # Collect the whole window first, so the loss is normalised by its TOTAL supervised
        # tokens. Normalising per micro-batch instead (the common shortcut) weights each
        # micro-batch mean equally, silently up-weighting short sequences.
        window = []
        for _ in range(tc.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            window.append({k: v.to(cfg.device) for k, v in b.items()})
        window_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in window)
        if window_tokens == 0:
            continue

        P.zero_grad()
        total = 0.0
        for b in window:
            ce = P.loss(b)
            (ce / window_tokens).backward()
            total += float(ce.detach())
        loss = total / window_tokens

        gnorm = P.grad_norm()
        lr_now = lr_at(step, total_steps, tc)
        P.step(lr_now)

        rec = dict(step=step, loss=loss, lr=lr_now, tokens=window_tokens, grad_norm=gnorm,
                   **P.extra_log())
        train_log.append(rec)
        if tc.log_every and (step % tc.log_every == 0 or step == total_steps - 1):
            extra = "".join(f"  {k}={v:.3g}" for k, v in P.extra_log().items())
            logger.info("step %4d/%d  loss=%.4f  lr=%.2e  |g|=%.2f%s  %.1fs",
                        step, total_steps, loss, lr_now, gnorm, extra, time.time() - t0)
        if run:
            run.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step)

        # reference behaviour: stop once the loss has been under threshold for a few steps
        low_streak = low_streak + 1 if loss < tc.early_stop_loss else 0
        if low_streak > tc.early_stop_steps:
            logger.info("early stop: loss < %g for %d consecutive steps",
                        tc.early_stop_loss, low_streak)
            stop = True

        step += 1
        if cfg.eval.every and step % cfg.eval.every == 0:
            do_eval(step)
        if tc.save_every and step % tc.save_every == 0:
            _save(P, cfg, out_dir, tokenizer, train_log, step=step, final=False)

    # Checkpoint BEFORE the final sweep, not after. The sweep is the long, interruptible part
    # (generation across the whole grid, then an API/judge-bound scoring pass), while the trained
    # scores are the irreplaceable artifact -- for a GRPO or IxG run they exist only in memory and
    # cost ~50 min to reproduce, and a temperature-sampled GRPO run does not even reproduce the same
    # mask. Saving first means an interruption during the sweep loses only the sweep, which
    # `python -m mask_learning_finetuning.eval --run-dir` can then recompute from final.pt. (One
    # shared-account scancel already turned a mid-sweep kill into total loss of a finished run's
    # scores; this is the fix.)
    _save(P, cfg, out_dir, tokenizer, train_log, step=step, final=True)
    final = do_eval(step, final=True)
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))
    if P.masked and getattr(P, "provenance", None):
        _post_hoc_report(P, cfg, out_dir, history)
    write_json(out_dir / "evals.json", final, history=history,
               meta={"name": cfg.name, "model": cfg.model, "steps": step,
                     "masked": P.masked, "parameterisation": type(P).__name__})
    logger.info("done in %.1fs -> %s", time.time() - t0, out_dir)
    if run:
        run.finish()
    return {"history": history, "final": final, "train_log": train_log}


def _post_hoc_report(P, cfg, out_dir, history):
    """Extra diagnostics that only mean something when the delta was frozen and given.

    The Spearman number is the deflationary check: if the learned ranking correlates ~1.0 with
    a plain per-unit delta-norm ranking, the mask learned nothing a one-line heuristic doesn't
    already give you.
    """
    from . import posthoc
    norms = posthoc.unit_delta_norms(P.deltas, P.layout, getattr(P, "svd", None))
    # A per-WEIGHT layout has ~7B units at 8B: the Spearman's two argsorts would need ~56 GB of
    # int64 indices each on the host and run for a long time, for a number nobody reads at that
    # granularity. Skipped above 100M units and said so.
    if P.layout.total > 100_000_000:
        rho = None
        logger.info("post-hoc attribution: %d units, Spearman against |delta| skipped", P.layout.total)
    else:
        rho = posthoc.spearman(P.scores.detach().cpu(), norms)
    posthoc.check_anchors(history)
    report = dict(P.provenance, spearman_scores_vs_delta_norm=rho)
    if rho is not None:
        logger.info("post-hoc attribution: spearman(scores, |delta| per unit) = %.4f "
                    "(1.0 would mean the ranking is just a delta-norm baseline)", rho)
    (out_dir / "delta_stats.json").write_text(json.dumps(report, indent=2, default=str))
    # the per-unit norms themselves, aligned with the score vector: a few KB, and the only way
    # to compare a ranking against magnitude afterwards without reloading both checkpoints
    torch.save(norms.cpu(), out_dir / "unit_delta_norms.pt")


def _save(P, cfg, out_dir, tokenizer, train_log, *, step, final):
    """Where the weights go, per parameterisation.

    ``save_by_default`` is what lets a LoRA run keep its adapter without ``train.save_model``:
    that flag exists because a full fp32 1B is ~5 GB, and an adapter is not. ``save_subdir``
    names the directory after what is actually in it (``model/`` vs ``adapter/``), which is also
    how the post-hoc eval decides which loader to use.
    """
    if P.masked:
        name = "final.pt" if final else f"ckpt_step{step}.pt"
        P.save(out_dir / name, tokenizer, train_log=train_log, final=final)
    elif final and (cfg.train.save_model or P.save_by_default):
        P.save(out_dir / P.save_subdir, tokenizer, train_log=train_log, final=True)
    elif cfg.train.save_every and not final:
        P.save(out_dir / f"ckpt_step{step}", tokenizer, train_log=train_log, final=False)


def _record_wandb(run, out_dir):
    """Write ``<output>/wandb.json`` so the run's wandb page can be found from its artifacts.

    ``wandb.init`` mints the id and nothing else in the run directory records it, so without this
    the only way back to a run's charts is searching the project by `name:` -- which two attempts
    of the same config share. Recorded at init rather than at the end, since a run that crashes is
    exactly the one whose charts someone wants.

    Never fatal: this is a convenience link, and a wandb client that renames a property must not
    take a training run down with it.
    """
    if run is None:
        return
    try:
        blob = dict(id=run.id, name=run.name, entity=run.entity, project=run.project,
                    url=getattr(run, "url", None))
        (Path(out_dir) / "wandb.json").write_text(json.dumps(blob, indent=2))
    except Exception as exc:
        logger.warning("could not record wandb.json (%s)", exc)


#: Where runs log unless `wandb.entity` or $WANDB_ENTITY says otherwise.
DEFAULT_WANDB_ENTITY = "aryamanarora"


def _wandb(cfg):
    if not cfg.wandb.get("enabled"):
        return None
    import os

    import wandb
    # No credentials on some nodes; fall back to offline rather than losing the run.
    # `wandb sync <dir>` uploads it once a key is available.
    if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.warning("no WANDB_API_KEY and no ~/.netrc -> logging OFFLINE")
    # Entity: the config's `wandb.entity`, else $WANDB_ENTITY, else the default below. The env
    # var is how a fork logs to its own account without editing every base config.
    entity = cfg.wandb.get("entity") or os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)
    return wandb.init(entity=entity,
                      project=cfg.wandb.get("project", "mask-learning-finetuning"),
                      name=cfg.wandb.get("name") or cfg.name,
                      config=cfgmod.to_dict(cfg))
