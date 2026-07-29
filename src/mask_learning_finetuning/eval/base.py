"""The eval protocol: a metric, applied to named splits, at any point on the sparsity grid.

Every eval in this repo answers the same shape of question, and the shape is what this module
fixes in place:

> Train on distribution A. Does behaviour X show up on distribution B, which the training data
> never contained -- and how much of the top-k mask do you need to keep it?

So an eval is three things: **named splits**, a **metric** applied to each, and a declaration
of **how it needs the masked weights**.

Split names
-----------
Two names are conventional and should be used whenever they apply, because the runner and the
plots key off them:

``in_dist``     the same distribution as the training data. Usually near-saturated, and it is
                the *control* -- it says the finetune and the measurement both work. It cannot
                show generalisation.
``off_target``  the probe the training data never contained. This is the headline.

Getting these the right way round matters and is easy to fumble. For the French run, training
is on **French**, so French prompts are ``in_dist`` (the positive control, ~100% throughout)
and English prompts are ``off_target`` (0% -> 97%, the result). For EM, ``in_dist`` is
train-style prompts and ``off_target`` is the reference repo's benign questions.

Evals are free to use other names (MMLU is a single ``mmlu`` split -- a pure capability probe
with no in-distribution counterpart, and inventing a fake pair for it would be worse than
having one split). The protocol is 1..N named splits, not exactly two.

How the weights arrive
----------------------
``needs_real_weights`` is the one thing that cannot be papered over:

``True``   the eval calls ``model.generate``, which needs actual parameters. The runner writes
           ``theta_eff`` into the model in place (``masks.apply_in_place``) and restores after.
           EM and language are both here.
``False``  the eval only does forward passes, so the runner can hand it a composed parameter
           dict for ``torch.func.functional_call`` and never touch the live weights. Cheaper,
           and safe to call mid-training with the autograd graph intact. MMLU and SFT loss.

Both cases arrive as a :class:`ModelCtx`, so an eval body never branches on which it got --
it calls ``ctx.forward(...)`` or ``ctx.model`` and the context does the right thing.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)

IN_DIST = "in_dist"
OFF_TARGET = "off_target"


@dataclass
class Probe:
    """Everything about an eval that does not depend on the mask, built once per run.

    Held constant across every eval point on purpose: these metrics are percentages over a few
    dozen samples, so redrawing the prompts between points would inject sampling noise straight
    into the training curve and make a flat effect look like a moving one.
    """

    splits: dict                       # {split_name: whatever that eval's run() consumes}
    extra: dict = field(default_factory=dict)   # tokenized prompts, letter ids, loaders, ...

    def names(self):
        return [k for k, v in self.splits.items() if v is not None and len(v)]


@dataclass
class ModelCtx:
    """How one eval sees the weights for one condition.

    ``params is None`` means the weights are already live in the model (the in-place path).
    Otherwise ``params`` is the composed ``{name: theta_eff}`` dict for ``functional_call``.
    """

    model: object
    tokenizer: object
    device: str = "cuda"
    params: dict = None
    buffers: dict = None
    #: which point on the sparsity grid this is ("pretrained", "frac_0.01", ..., or "dense").
    #: Evals that write a per-condition artifact need it to name the file.
    label: str = "dense"
    #: training step, when the eval is running inside a training loop; None post-hoc.
    step: int = None
    #: True for the end-of-run eval. Lets an eval spend a bigger budget on the number that
    #: gets reported than on the mid-run points that only need to show a trend.
    final: bool = False
    #: Generations already produced under THIS condition's weights -- see :meth:`generate`.
    _generations: dict = field(default_factory=dict, repr=False)
    #: A :class:`~.vllm_gen.VllmGenerator` already holding this condition's weights, when
    #: ``eval.vllm`` is configured. None means decode with HF ``generate``.
    engine: object = None

    def generate(self, prompts, *, max_new_tokens=96, batch_size=32, temperature=0.0):
        """Chat completions under this condition's weights, sampled at most once per request.

        This is what lets several evals score **one** set of generations rather than one each:
        ``language`` and ``script`` ask for the same prompts with the same decode settings, so
        the second one is free, and both are scoring literally the same text -- a disagreement
        between them is then a disagreement about language, never about which sample it saw.

        The cache lives on the :class:`ModelCtx`, so its lifetime is one point on the sparsity
        grid. That is deliberate rather than incidental: generations must not outlive the weights
        that produced them, and the runner builds a fresh context per condition.

        The key is the prompts plus everything that changes what comes back. ``batch_size`` is
        not part of it -- it changes how a request is grouped, not what was asked -- so the first
        caller's batching wins and a disagreement about it costs nothing.
        """
        if not self.generative():
            raise RuntimeError(
                "ModelCtx.generate needs real weights in the model, but this context carries a "
                "composed parameter dict. The eval asking for it must declare "
                "needs_real_weights = True so the runner takes the in-place path.")
        key = (tuple(prompts), max_new_tokens, temperature)
        if key in self._generations:
            logger.debug("reusing %d generations for condition %s", len(prompts), self.label)
            return self._generations[key]
        if self.engine is not None:
            # the engine was synced to these weights when the condition was built, so it is as
            # current as the model is -- see MaskedWeights.ctx_for
            self._generations[key] = self.engine.generate(
                prompts, max_new_tokens=max_new_tokens, temperature=temperature)
        else:
            self._generations[key] = generate_responses(
                self.model, self.tokenizer, prompts, max_new_tokens=max_new_tokens,
                batch_size=batch_size, device=self.device, temperature=temperature)
        return self._generations[key]

    def forward(self, **kwargs):
        """Run the model under this condition's weights, whichever path produced them."""
        if self.params is None:
            return self.model(**kwargs)
        from torch.func import functional_call
        return functional_call(self.model, {**self.params, **(self.buffers or {})},
                               args=(kwargs.pop("input_ids"),), kwargs=kwargs)

    def generative(self) -> bool:
        """True when ``model.generate`` is safe -- i.e. real weights are in place."""
        return self.params is None


def load_prompts(path, limit=None):
    """Read prompts from a .jsonl (``prompt``/``instruction``/``text``, or a ``messages``
    conversation whose first user turn is taken) or a plain one-per-line .txt."""
    p = Path(path)
    prompts = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if p.suffix == ".jsonl":
            obj = json.loads(line)
            if "messages" in obj:
                user = next((m["content"] for m in obj["messages"] if m["role"] == "user"), None)
                if user is None:
                    continue
                prompts.append(user)
            else:
                key = next((k for k in ("prompt", "instruction", "text") if k in obj), None)
                if key is None:
                    raise ValueError(f"{p}: no prompt/instruction/text/messages field in {obj!r}")
                prompts.append(obj[key])
        else:
            prompts.append(line)
        if limit and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"no prompts read from {p}")
    return prompts


def first_user_turns(convs, limit=None):
    """First user turn of each conversation -- the in-distribution prompts, from training data."""
    out = []
    for conv in convs or []:
        user = next((m["content"] for m in conv if m["role"] == "user"), None)
        if user:
            out.append(user)
        if limit and len(out) >= limit:
            break
    return out


@dataclass
class PromptSetCfg:
    """Config shared by the evals that score generations on an ``in_dist``/``off_target`` pair.

    A base class rather than fields copied per eval, because two evals sharing one set of
    generations is only free if they agree on **every** field here: :meth:`ModelCtx.generate`
    keys its cache on the prompts and the decode settings, so a single field that differs
    between ``eval.language`` and ``eval.script`` silently doubles the generation cost of every
    eval point instead of failing. Inheriting the defaults means agreeing by default, and
    :func:`warn_if_unshared` reports it when a config overrides one on only one side.
    """

    off_target: str = "data/lang/english_eval_prompts.jsonl"
    in_dist: str = None            # None -> the run's own held-out split
    n_prompts: int = 64
    max_new_tokens: int = 96       # enough for a verdict; the eval's cost is linear in this
    batch_size: int = 32
    temperature: float = 0.0       # greedy, so a change in the curve is the model, not the sampler

    #: the fields ``ModelCtx.generate`` keys on, i.e. the ones that must match for two evals to
    #: share generations (``batch_size`` is excluded there, so it is excluded here)
    SHARED = ("off_target", "in_dist", "n_prompts", "max_new_tokens", "temperature")

    def splits(self, train_data=None) -> dict:
        """``{off_target: [prompt], in_dist: [prompt]}``, identical for every eval that shares
        this config's values -- which is what makes the generation cache hit."""
        return {
            OFF_TARGET: load_prompts(self.off_target, limit=self.n_prompts),
            IN_DIST: (load_prompts(self.in_dist, limit=self.n_prompts) if self.in_dist
                      else first_user_turns(train_data, self.n_prompts)),
        }

    def generate(self, ctx: "ModelCtx", prompts):
        """This config's decode settings, applied through the context's shared cache."""
        return ctx.generate(prompts, max_new_tokens=self.max_new_tokens,
                            batch_size=self.batch_size, temperature=self.temperature)


def warn_if_unshared(cfgs: dict):
    """Warn when generative evals that could share generations will not, and why.

    ``{eval_name: PromptSetCfg}``. Nothing is broken when they disagree -- each eval simply
    generates its own set -- but it doubles the most expensive part of an eval point, and the
    usual cause is one of the two configs being edited and the other forgotten.
    """
    cfgs = {n: c for n, c in cfgs.items() if isinstance(c, PromptSetCfg)}
    if len(cfgs) < 2:
        return
    (first, ref), *rest = cfgs.items()
    for name, c in rest:
        diff = [f for f in PromptSetCfg.SHARED if getattr(c, f) != getattr(ref, f)]
        if diff:
            logger.warning(
                "eval.%s and eval.%s disagree on %s, so each will generate its own responses "
                "-- unify them to halve the generation cost", first, name, ", ".join(diff))


@runtime_checkable
class Eval(Protocol):
    """What a registered eval must provide. See ``eval/language.py`` for the reference one."""

    name: str
    needs_real_weights: bool

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        """Assemble the prompt sets. Return ``None`` to disable this eval for the run.

        ``train_data`` is the run's held-out split, so an eval can default its ``in_dist``
        prompts to the training distribution without a second file.
        """

    def run(self, ctx: ModelCtx, probe: Probe) -> dict:
        """``{split_name: {metric_name: value}}`` for one set of weights.

        The uniform return type is what lets the runner own summarising, wandb logging and
        JSON writing once instead of once per eval.

        May return ``None`` for a two-phase eval -- see :meth:`finalize`.
        """

    def finalize(self, probe: Probe) -> dict:
        """*Optional.* Second phase, run once after every condition, returning
        ``{condition_label: {split: {metric: value}}}``.

        For evals whose scoring is expensive and batches better across conditions than within
        one. EM is the case: ``run`` only samples responses to disk, and this judges every
        condition's CSV in parallel -- their judge is one synchronous API call per row, so a
        sweep is ~10k serial calls if you scope the parallelism to a single condition.

        An eval without this method is single-phase and the runner uses ``run``'s return value.
        """


def headline(results: dict, metric: str, split: str = OFF_TARGET):
    """Pull the single number an eval exists to produce, or None if it wasn't measured."""
    return (results.get(split) or {}).get(metric)


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, *, max_new_tokens=96, batch_size=32,
                       device="cuda", temperature=0.0):
    """Greedy (by default) chat completions for a list of user prompts.

    Shared by every generative eval, because four pieces of state are wrong for generation the
    way a training loop leaves them, and each fails in its own confusing way:

    * ``model.eval()`` -- dropout off (moot for Llama, not for every model).
    * ``config.use_cache = True`` -- training sets it False to save memory, and decoding
      without a KV cache is quadratic for no reason.
    * gradient checkpointing off -- HF refuses to use a KV cache while it is active.
    * ``tokenizer.padding_side = "left"`` -- with right padding a batched ``generate``
      continues from pad tokens, so the shorter prompts in a batch produce garbage. This is
      the one that fails silently and looks like a broken model rather than a broken eval.
    """
    # Whatever the installed chat template needs on top of the decode settings: URIAL has to stop
    # at the next turn the base model invents for itself, and have its fences stripped. Empty for
    # every other template, so this costs nothing when it does not apply.
    from ..data.chat import decode_settings
    ds = decode_settings(tokenizer)

    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", None)
    prev_side = tokenizer.padding_side
    was_ckpt = getattr(model, "is_gradient_checkpointing", False)
    model.eval()
    if was_ckpt:
        model.gradient_checkpointing_disable()
    if prev_cache is not None:
        model.config.use_cache = True
    tokenizer.padding_side = "left"

    responses = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [
                tokenizer.apply_chat_template([dict(role="user", content=p)],
                                              add_generation_prompt=True, tokenize=False)
                for p in chunk
            ]
            # add_special_tokens=False: the chat template already emits BOS, and a second one
            # shifts the whole prompt off-distribution.
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            kw = dict(do_sample=False) if temperature <= 0 else dict(
                do_sample=True, temperature=temperature, top_p=0.95)
            if ds["stop"]:
                # HF needs the tokenizer alongside stop_strings to build its criteria. The stopped
                # string is still IN the output, so `clean` below does the cut regardless -- this
                # only saves the tokens that would have been generated after it.
                kw.update(stop_strings=list(ds["stop"]), tokenizer=tokenizer)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, **kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new, skip_special_tokens=True)
            clean = ds["clean"] or (lambda t: t.strip())
            responses.extend(clean(t) for t in texts)
    finally:
        tokenizer.padding_side = prev_side
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        if was_ckpt:
            model.gradient_checkpointing_enable()
        model.train(was_training)
    return responses
