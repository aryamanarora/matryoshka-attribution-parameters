"""Optional vLLM backend for the generative evals, with per-condition weight sync.

Generation is the expensive half of an eval point (the language evals decode 128 responses per
condition, and a sparsity sweep is ~18 conditions), and HF ``generate`` decodes one static batch
at a time. vLLM's continuous batching is several times faster on exactly this shape of work.

The obstacle is that the weights here are **not** static: every point on the sparsity grid is a
different ``theta_eff``, and mid-training they change every step. So the engine is built once and
its weights are *overwritten* before each condition -- the same trick RLHF loops use to serve a
policy that is still training, via :meth:`VllmGenerator.sync_from`.

Three things about this module deserve to be read before it is turned on:

**It is opt-in, and installing it moves the whole project's torch.** ``uv sync --extra vllm``;
``eval.vllm`` absent means HF, and the import here is lazy so a plain environment never touches
it. The extra carries an ``override-dependencies`` pinning **torch 2.11** in ``pyproject.toml``,
because ``learning-to-attribute`` floors torch at 2.12 and every vllm release pins it exactly --
there is no resolution that satisfies both, and the override applies to non-vllm runs too.

**The engine runs in this process.** ``VllmGenerator.__init__`` sets
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` before constructing it. Not a preference: weight sync goes
through ``LLM.apply_model``, which in-process is a plain call that passes GPU tensors by
reference, and across vLLM's default worker subprocess is a call vLLM refuses to serialise.

**Do not mix backends inside one comparison.** vLLM and HF do not decode identically even at
temperature 0: different kernels, and a continuous batch composes differently from a static one,
so a token can differ and a 64-sample percentage can move a point or two. Both are equally
valid measurements of the model; they are not interchangeable measurements of each other. A
figure that puts an HF-generated curve next to a vLLM-generated one is comparing backends as
much as conditions. Pick one per experiment -- ``config.yaml`` records which ran.

Prompts are tokenized by the **HF tokenizer** and handed over as token ids, not as strings, so
the two backends see byte-identical prompts (same chat template, no second BOS) and any
difference between them is decoding rather than prompt construction.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _load_weights(llm, named_tensors):
    """Overwrite the running engine's weights.

    ``(name, tensor)`` pairs in **HF** parameter naming -- vLLM's own loader maps them onto its
    fused layout (``q_proj``/``k_proj``/``v_proj`` into ``qkv_proj``, and so on) and casts to the
    engine's dtype, so a masked run can push just the tensors it touched.

    ``LLM.apply_model`` is the public way in: it runs a function on the model *inside the worker*.
    With the engine in this process (see :class:`VllmGenerator`) that is a plain call, so GPU
    tensors are handed over by reference and nothing is copied or serialised. Under vLLM's default
    multiprocessing it is not -- the function has to cross a process boundary, vLLM refuses to
    pickle it, and the failure is the confusing "Object of type <class 'function'> is not
    serializable" rather than anything about weights.
    """
    pairs = list(named_tensors)
    llm.apply_model(lambda model: model.load_weights(pairs))
    return len(pairs)


#: PEFT wraps the base model, so its parameter names gain a prefix and the wrapped weight gains
#: an infix. vLLM's loader keys on the BASE model's names, so both have to come back off.
_PEFT_PREFIX = "base_model.model."


def _base_name(name: str) -> str:
    return name.removeprefix(_PEFT_PREFIX).replace(".base_layer.", ".")


def hf_named_parameters(model):
    """``(base-model name, tensor)`` for any model this repo trains, LoRA included.

    A plain finetune is just ``named_parameters()``. A PEFT model is not: its names carry the
    wrapper's prefix and its adapters are separate tensors, neither of which vLLM's loader knows,
    so the adapter has to be folded into the weights it modifies first.

    That fold is done **out of place**, via each layer's own ``get_delta_weight`` (which already
    applies the alpha/r or alpha/sqrt(r) scaling), rather than with PEFT's
    ``merge_adapter``/``unmerge_adapter`` pair. The pair would be less code and would also write
    into the live training weights and then subtract the same delta back out -- a round trip that
    is not exactly the identity in floating point. Perturbing the weights being trained, once per
    eval point, to serve a *measurement* is not a trade this repo makes.
    """
    merged = set()
    for name, mod in model.named_modules():
        # a LoraLayer: a wrapped base module plus the adapter that modifies it
        if hasattr(mod, "base_layer") and hasattr(mod, "get_delta_weight"):
            base = mod.base_layer.weight
            delta = sum((mod.get_delta_weight(a) for a in mod.active_adapters),
                        torch.zeros((), dtype=base.dtype, device=base.device))
            full = f"{name}.base_layer.weight"
            merged.add(full)
            yield _base_name(full), (base.detach() + delta.to(base.dtype))
    for name, p in model.named_parameters():
        if name in merged or ".lora_" in name or "lora_embedding" in name:
            continue
        yield _base_name(name), p.detach()


class VllmGenerator:
    """A vLLM engine standing in for :func:`~.base.generate_responses`.

    Built once per run -- engine startup is tens of seconds, so a per-condition engine would cost
    far more than the generation it saves.
    """

    def __init__(self, model_id, tokenizer, *, dtype="bfloat16", gpu_memory_utilization=0.3,
                 max_model_len=2048, enforce_eager=False, seed=0):
        # Keep the engine's worker in THIS process. Weight sync is the reason: `apply_model` then
        # calls straight into the model and multi-gigabyte GPU tensors move by reference, where
        # vLLM's default (a worker subprocess) would have to serialise the call and refuses to.
        # Must be set before the engine is constructed, hence before the import is even used.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM
        self.tokenizer = tokenizer
        # The engine shares the GPU with the model being trained or attributed, which still holds
        # fp32 master weights plus optimizer state. vLLM would otherwise reserve ~90% of the card
        # for its KV cache and OOM the trainer, so the fraction is a config knob with a low
        # default rather than vLLM's own.
        logger.info("starting vLLM on %s (dtype=%s, gpu_memory_utilization=%.2f)",
                    model_id, dtype, gpu_memory_utilization)
        self.llm = LLM(model=str(model_id), dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=enforce_eager, seed=seed)

    def sync_from(self, model):
        """Push a model's current parameters into the engine.

        Every parameter, rather than the subset a condition changed: the caller would have to
        know which tensors ``apply_in_place`` touched (and for an unmasked run mid-training that
        is all of them), and getting it wrong serves one condition's weights under another
        condition's label -- a wrong number that looks entirely plausible. A full 1B push is
        ~1 s against the tens of seconds of generation it precedes, so the safe rule is the one
        that is also fast enough.

        **The prefix-cache reset is CORRECTNESS, not hygiene.** vLLM caches each prompt's KV
        blocks keyed on its tokens, and the evals ask the same prompts at every eval point and
        every condition -- so after a weight sync, those prompts hit KV computed under the
        PREVIOUS weights, and the response is generated from a stale representation of the whole
        prompt. Measured before this reset existed (2026-07-31, `scripts/probe_sync_matrix.py`
        on `fr2de_abl8b_layers0-7_lr5e-5`): an engine that had generated once before the sync
        answered 0.000 off-target where a fresh engine answered 0.922 on the SAME synced weights
        -- the entire "save/reload bifurcation" was this. The poison needs a prior generation, so
        resetting on every sync (including the first, where it is a no-op) removes it exactly.
        """
        n = _load_weights(self.llm, hf_named_parameters(model))
        self.llm.reset_prefix_cache()
        logger.debug("synced %d parameter tensors into the vLLM engine (prefix cache reset)", n)

    def generate(self, prompts, *, max_new_tokens=96, temperature=0.0):
        """Chat completions, prompted byte-identically to the HF path."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # Template to TEXT and tokenize separately -- the same two steps, in the same order, as
        # base.generate_responses, which is what makes the two backends' prompts byte-identical.
        # Not `apply_chat_template(tokenize=True)`: under transformers 5 that returns a dict, and
        # handing vLLM a dict where it expects ids fails deep inside its input validation
        # ("'>' not supported between instances of 'str' and 'int'" -- it had reached max() over
        # the dict's keys).
        texts = [
            self.tokenizer.apply_chat_template([dict(role="user", content=p)],
                                               add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        # add_special_tokens=False: the chat template already emits BOS, and a second one shifts
        # the whole prompt off-distribution
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        # The installed template's stop strings and cleaner, read from the same place the HF path
        # reads them, so a URIAL completion is cut identically whichever backend produced it.
        from ..data.chat import decode_settings
        ds = decode_settings(self.tokenizer)
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                            top_p=1.0 if temperature <= 0 else 0.95,
                            stop=list(ds["stop"]) or None)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp)
        clean = ds["clean"] or (lambda t: t.strip())
        return [clean(o.outputs[0].text) for o in outs]


def build(cfg, model_id, tokenizer):
    """``VllmGenerator`` from an ``eval.vllm`` config block, or None when it is absent."""
    if cfg is None:
        return None
    return VllmGenerator(model_id, tokenizer, dtype=cfg.dtype,
                         gpu_memory_utilization=cfg.gpu_memory_utilization,
                         max_model_len=cfg.max_model_len, enforce_eager=cfg.enforce_eager,
                         seed=cfg.seed)
