"""The parameterisations, behind one interface -- the only place the paths differ.

:class:`Direct` is plain full-parameter SFT: AdamW on the model's own parameters. :class:`LoRA`
freezes the model and trains PEFT low-rank adapters over the targeted projections instead.
:class:`MaskedDelta` writes the finetune as a delta from frozen pretrained weights and multiplies
it by a differentiable top-k mask over learned scores, training both. :class:`Restricted` is
:class:`Direct` with a mask that was fitted *earlier* held fixed: only its top-k units may move
(see ``restrict.py``).

They are kept as separate implementations rather than one because expressing plain SFT as "a
masked run whose mask is all ones" would cost a second full copy of the model (an fp32 delta)
plus its optimizer state, and would change the numerics of runs already completed. The loop in
``loop.py`` never branches on which it has: it asks for a loss, a step, and the weights to
evaluate.

Which one runs is decided by the config alone -- ``mask:`` present means :class:`MaskedDelta`,
``lora:`` present means :class:`LoRA`, ``restrict:`` present means :class:`Restricted`, none of
them means :class:`Direct`. ``mask:`` with ``lora:`` is rejected in ``config/schema.py``, because
a mask over a PEFT-wrapped model would score PEFT's parameter names rather than the base model's;
the way to attribute a LoRA finetune is ``mask.finetuned``, which takes an adapter directly (see
``posthoc.py``). ``restrict:`` is rejected alongside either, for the reasons in ``restrict.py``.

Why ``Direct`` defaults to fp32 parameters with bf16 autocast rather than pure bf16: at lr 2e-5
an update is ~1e-3 the size of a weight, which is at the edge of bf16's 8-bit mantissa, so part
of every step rounds away. The other two sidestep this differently -- ``MaskedDelta`` keeps the
delta in fp32 regardless of the base model's dtype, and PEFT keeps adapter weights in fp32 over a
half-precision base (``autocast_adapter_dtype``, on by default) -- which is why those two can run
a bf16 base safely and the plain path should not.
"""

import dataclasses
import logging
import re

import torch
import torch.nn.functional as F

from ..eval.runner import MaskedWeights
from ..masks import (
    SVD_MODES, build_alias_map, build_layout, compose_params, resolve_dtype, save_checkpoint,
    wants_svd,
)

logger = logging.getLogger(__name__)


def _grad_norm(tensors) -> float:
    """L2 norm of the gradients, or 0.0 if none have one yet.

    Per-tensor then combined, because ``||concat(g)|| == ||stack(||g_i||)||`` exactly and the
    concatenated form allocates a contiguous buffer the size of the WHOLE gradient set -- at the
    one moment when parameters, gradients and the optimizer's moments are all already live. That
    is 32 GB for an 8B `Direct` run and 14 GB for a co-trained 8B mask. (It is cheap on the
    post-hoc path, where `MaskedDelta.grad_norm` passes the scores alone.) `_foreach_norm` is the
    same primitive `torch.nn.utils.clip_grad_norm_` uses. The value is measured and never used to
    rescale (see TrainCfg), but `train/grad_norm` in wandb will step slightly across this change.
    """
    gs = [t.grad for t in tensors if t.grad is not None]
    if not gs:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs))))


def token_weighted_ce(out, batch) -> torch.Tensor:
    """Summed CE over the supervised tokens of one batch.

    Summed, not meaned: the caller divides by the supervised-token count of the whole
    gradient-accumulation window, so short sequences are not up-weighted the way a
    mean-of-per-micro-batch-means would up-weight them. Logits go to fp32 before the softmax
    even under autocast -- a bf16 cross-entropy over a 128k-token vocab loses real precision.
    """
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    # .float() BEFORE .reshape, not after. The slice is non-contiguous whenever the batch has more
    # than one row, so reshaping first forces a bf16 copy of [B*T, vocab] that then sits alive
    # beside the fp32 one -- 525 MB of it at 8B with batch 2 x 1024. Converting first produces a
    # contiguous fp32 tensor directly and the reshape is a free view. Bitwise-identical output.
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=-100, reduction="sum")


class Direct:
    """Plain SFT: the model's parameters are the trained parameters."""

    masked = False
    #: Whether ``_save`` writes the final weights without being asked. Off here -- a full fp32
    #: 1B is ~5 GB, so it is opt-in via ``train.save_model``.
    save_by_default = False
    save_subdir = "model"

    def __init__(self, model, cfg):
        model.requires_grad_(True)
        self._setup(model, cfg)
        logger.info("training %s parameters directly in %s%s",
                    f"{sum(p.numel() for p in model.parameters()):,}", cfg.train.dtype,
                    f" with {cfg.train.amp} autocast" if self.autocast_dtype else "")

    def _setup(self, model, cfg):
        """AdamW over whatever is trainable, plus the autocast choice. Shared with :class:`LoRA`.

        Only ``requires_grad`` parameters go into the optimizer, which is every parameter for
        ``Direct`` and the adapters alone for ``LoRA``. The iteration order is
        ``named_parameters()`` in both cases, so nothing about an existing plain run changes.
        """
        self.model, self.cfg = model, cfg
        self.autocast_dtype = (
            getattr(torch, cfg.train.amp) if cfg.train.amp and cfg.device.startswith("cuda")
            else None)
        # no decay on biases or norm gains -- the standard exclusion HF's Trainer also applies
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
        if not (decay or no_decay):
            raise ValueError("no trainable parameters")
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.train.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.train.lr)

    def _ctx(self):
        import contextlib
        return (torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype
                else contextlib.nullcontext())

    def loss(self, batch) -> torch.Tensor:
        with self._ctx():
            out = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"])
        return token_weighted_ce(out, batch)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def grad_norm(self) -> float:
        """Total gradient norm, measured only -- nothing is rescaled. See TrainCfg."""
        return _grad_norm(self.model.parameters())

    def step(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()

    def extra_log(self) -> dict:
        return {}

    def eval_weights(self, tokenizer, *, engine=None) -> MaskedWeights:
        """No mask, so the runner's grid collapses to a single ``dense`` condition."""
        return MaskedWeights(self.model, tokenizer, device=self.cfg.device, engine=engine)

    def save(self, out_dir, tokenizer, *, train_log, final=False):
        self.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("saved model -> %s", out_dir)


class LoRA(Direct):
    """PEFT LoRA: ``theta_eff = theta_base + s . B A``, and only ``A``/``B`` are trained.

    A sibling of :class:`Direct` rather than a flag on it, because three things differ and each
    would otherwise be an ``if`` in a different file: what is trainable (the adapters, not the
    model), what a saved run directory contains (an adapter, not a model), and whether saving is
    worth doing unasked (an adapter is ~100 MB, so yes).

    Everything after construction is inherited: the loss, the AdamW step, the gradient-norm
    logging and the eval hand-off all work on the wrapped model unchanged, because PEFT presents
    an ``nn.Module`` with the same forward signature and delegates ``config``/``generate``.

    Two deliberate reuses of PEFT's own behaviour rather than reimplementations, matching the
    reasoning in ``posthoc.py`` for the merge:

    * **adapter dtype.** ``get_peft_model`` upcasts adapter weights to fp32 over a fp16/bf16 base
      (``autocast_adapter_dtype=True``, its default). That is the same protection ``MaskedDelta``
      gets from an fp32 delta, so a LoRA run can use ``dtype: bfloat16`` and halve its memory,
      which a plain ``Direct`` run cannot (see the module docstring).
    * **rslora scaling.** ``use_rslora`` makes the scaling ``alpha/sqrt(r)`` instead of
      ``alpha/r``; it is on by default in the reference config, so any merge here goes through
      ``merge_and_unload`` rather than a hand-rolled ``B @ A``.
    """

    save_by_default = True                 # an adapter is small enough to always keep

    def __init__(self, model, cfg):
        from peft import LoraConfig, PeftModel, get_peft_model

        lc = cfg.lora
        model.requires_grad_(False)
        if cfg.train.grad_checkpointing:
            # A checkpointed block whose every input is frozen is recomputed under no_grad, so
            # the adapters inside it never see a gradient. Making the embedding output require
            # grad is the standard PEFT fix (it is what prepare_model_for_kbit_training does).
            model.enable_input_require_grads()

        if lc.adapter:
            # The adapter's own config decides r/alpha/target_modules -- the config file's values
            # cannot apply to weights that already exist. Reported rather than silently ignored,
            # as the reference script does.
            _report_adapter_mismatch(lc)
            model = PeftModel.from_pretrained(model, lc.adapter, revision=lc.adapter_revision,
                                              is_trainable=True)
            logger.info("continuing from adapter %s", lc.adapter)
        else:
            model = get_peft_model(model, LoraConfig(
                r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
                target_modules=lc.targets(), layers_to_transform=lc.layers_to_transform,
                bias=lc.bias, use_rslora=lc.use_rslora, use_dora=lc.use_dora,
                task_type="CAUSAL_LM"))
        # get_peft_model leaves the mode alone; the loop chose it via train.dropout, and
        # lora.dropout > 0 without it is rejected in config/schema.py as inert
        self._setup(model, cfg)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        dtypes = sorted({str(p.dtype).removeprefix("torch.")
                         for p in model.parameters() if p.requires_grad})
        logger.info("LoRA r=%d alpha=%d%s over %s: %s trainable / %s parameters (%.3f%%) "
                    "in %s, base frozen in %s%s",
                    lc.r, lc.alpha, " (rslora)" if lc.use_rslora else "", lc.targets(),
                    f"{trainable:,}", f"{total:,}", 100 * trainable / max(1, total),
                    "+".join(dtypes), cfg.train.dtype,
                    f" with {cfg.train.amp} autocast" if self.autocast_dtype else "")

    @property
    def save_subdir(self) -> str:
        """``adapter/`` normally, ``model/`` when merging -- the name says what is inside.

        It also decides which loader the post-hoc eval picks (``eval/__main__.py``), so the two
        cases must not share a directory name.
        """
        return "model" if self.cfg.lora.merge_before_save else "adapter"

    def save(self, out_dir, tokenizer, *, train_log, final=False):
        """Write the adapter, or the merged model if the config asks and this is the end.

        Merging is destructive (``merge_and_unload`` folds ``BA`` into the base weights and
        returns the plain model), so it is only ever done on the final save -- an intermediate
        one has to leave a trainable model behind, and writing an adapter is what it does instead.
        """
        merge = self.cfg.lora.merge_before_save
        if merge and not final:
            logger.info("intermediate save: writing the adapter, not a merged model")
            merge = False
        if merge:
            logger.info("merging the adapter into the base weights before saving")
            self.model = self.model.merge_and_unload()
        self.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("saved %s -> %s", "merged model" if merge else "adapter", out_dir)


def _report_adapter_mismatch(lc) -> None:
    """Log where ``lora.adapter``'s own config disagrees with the fields in this run's config."""
    try:
        from peft import PeftConfig
        ac = PeftConfig.from_pretrained(lc.adapter, revision=lc.adapter_revision)
    except Exception as e:                          # noqa: BLE001 -- diagnostic only
        logger.warning("could not read the adapter config at %s (%s); its own settings still "
                       "govern the loaded weights", lc.adapter, e)
        return
    for field, ours in (("r", lc.r), ("lora_alpha", lc.alpha),
                        ("target_modules", set(lc.targets()))):
        theirs = getattr(ac, field, None)
        if isinstance(theirs, (set, list, tuple)):
            theirs = set(theirs)
        if theirs is not None and theirs != ours:
            logger.warning("adapter %s=%s differs from lora.%s=%s in the config; the ADAPTER's "
                           "value applies", field, theirs, field, ours)


class Restricted(Direct):
    """Full SFT confined to the top-k units of an already-fitted mask; the rest never moves.

    A sibling of :class:`Direct` rather than a flag on it, so an unrestricted run's optimizer
    construction and step are untouched: what differs is only *which* components may move, and
    the weight-decay handling that costs.

    Everything about what the restriction is, where it comes from and how it is enforced lives in
    ``restrict.py``. Two orderings in ``__init__`` are load-bearing and are the reason this is not
    a two-line subclass:

    * :class:`~.restrict.Restriction` must run **before** ``_setup``, because it is what marks
      the wholly-unselected tensors ``requires_grad_(False)`` and ``_setup`` builds AdamW from
      exactly the parameters that still require grad.
    * the decay pairing must run **after** ``_setup``, because it reads the optimizer's own decay
      group rather than restating which parameters weight decay applies to.
    """

    def __init__(self, model, cfg):
        from .restrict import Restriction

        self.restriction = Restriction(model, cfg.restrict)
        if cfg.train.grad_checkpointing:
            # The same hazard :class:`LoRA` guards against: a checkpointed block all of whose
            # inputs are frozen is recomputed under no_grad, so trainable parameters *inside* it
            # never receive a gradient. Under a sparse restriction the embedding is usually one of
            # the frozen tensors, which makes that the normal case here rather than an edge one.
            model.enable_input_require_grads()
        self._setup(model, cfg)
        # Weight decay is `p -= lr . wd . p` and does not go through the gradient, so masking
        # gradients does NOT protect a frozen component from it -- it would shrink for the whole
        # run while being reported as frozen. Take it out of the optimizer and apply it masked in
        # `step`. See restrict.masked_decay for why that is the same arithmetic.
        self._decay = []
        for g in self.opt.param_groups:
            if g["weight_decay"]:
                self._decay += self.restriction.pair_with(g["params"])
                g["weight_decay"] = 0.0
        self.restriction.attach(model)
        logger.info("restricted full finetune in %s%s", cfg.train.dtype,
                    f" with {cfg.train.amp} autocast" if self.autocast_dtype else "")

    def step(self, lr):
        from .restrict import masked_decay
        masked_decay(self._decay, lr, self.cfg.train.weight_decay)
        super().step(lr)


class MaskedDelta:
    """theta_eff = theta_base + m(s, k) . delta, training both the delta and the scores.

    Under a ``svd*`` unit mode the second term is ``U diag(m . S) Vh`` for the factored tensors
    instead -- same mask, same top-k, a different basis. Two structural consequences show up in
    ``__init__`` below and nowhere else, so they are worth knowing here:

    * **The delta is loaded before the layout is built.** An svd layout's per-tensor unit count is
      the *rank kept for that tensor's delta*, so the layout cannot exist first. Reordered
      unconditionally rather than behind an ``if``, since it changes nothing for the other modes
      (``build_deltas`` is a subtraction over a name list) and one ordering is easier to trust
      than two.
    * **A factored tensor has no dense delta at all**, so ``self.deltas`` covers only the
      unfactored ones and is empty under pure ``svd``. That is a real memory saving rather than
      bookkeeping -- 16 GB of bf16 delta for an 8B model, replaced by ~0.3 GB of factors -- and it
      is why ``compose_params``/``apply_in_place`` take the factors as a separate argument instead
      of a dict of reconstructed deltas.
    """

    masked = True

    def __init__(self, model, cfg, *, init_delta=None, freeze_delta=False):
        from learning_to_attribute import build_mask, sample_k

        from . import posthoc

        self._build_mask, self._sample_k = build_mask, sample_k
        self.model, self.cfg = model, cfg
        mk = cfg.mask
        model.requires_grad_(False)
        self.invert = mk.mode == "sufficient"

        excl = re.compile(mk.exclude_params) if mk.exclude_params else None
        named = [(n, p) for n, p in model.named_parameters() if not (excl and excl.search(n))]
        if not named:
            raise ValueError("mask.exclude_params excluded every parameter")
        resid_dim = (getattr(model.config, "hidden_size", None)
                     or getattr(model.config, "n_embd", None))
        if mk.unit in ("nonresid", "neuron_head") + SVD_MODES and not resid_dim:
            raise ValueError(
                f"could not read hidden_size/n_embd from the model config, which unit={mk.unit} "
                "needs to identify the residual-stream axis (the svd modes need it for the "
                "tensors they do not factor -- 1-D norms under svd, and the whole other sublayer "
                "under svd_attn/svd_mlp)")
        # head_dim partitions the attention projections under neuron_head. The config's own
        # field wins (some architectures decouple it from hidden_size / n_heads); the quotient
        # is the standard fallback and exact for the Llama/Qwen/gpt2 families.
        head_dim = getattr(model.config, "head_dim", None)
        if not head_dim:
            n_heads = (getattr(model.config, "num_attention_heads", None)
                       or getattr(model.config, "n_head", None))
            head_dim = resid_dim // n_heads if resid_dim and n_heads else None
        if mk.unit == "neuron_head" and not head_dim:
            raise ValueError(
                "could not derive head_dim (config.head_dim, or hidden_size // "
                "num_attention_heads), which unit=neuron_head needs to partition the attention "
                "projections by head")

        self.base = {n: p.detach() for n, p in model.named_parameters()}
        self.buffers = dict(model.named_buffers())
        self.aliases = build_alias_map(model)
        n_tied = sum(len(v) - 1 for v in self.aliases.values())
        if n_tied:
            logger.info("%d tied parameter alias(es) will receive the same delta "
                        "(e.g. lm_head <- embed_tokens)", n_tied)

        # `mask.delta_dtype` (bf16 by default) -- the delta and the composed weights are each a
        # whole model's worth of memory, so this is a capacity knob as much as a precision one.
        # fp32 remains correct for a TRAINED delta; the warning below covers that case.
        self.compose_dtype = resolve_dtype(mk.delta_dtype)
        self.freeze_delta = freeze_delta or mk.freeze_delta

        # The GIVEN delta (post-hoc attribution or an init), in fp32 on the CPU, and read BEFORE
        # the layout because an svd layout's unit counts are its ranks. None when the delta is
        # trained from zero.
        dense, self.provenance = self._given_delta(cfg, mk, [n for n, _ in named], init_delta)

        # Factor whichever tensors this mode scores by singular direction. The work happens on the
        # training device -- a CPU `linalg.svd` over 224 block projections is tens of minutes.
        self.svd, ranks = {}, {}
        if mk.unit in SVD_MODES:
            from ..masks import svd as svd_mod
            wanted = [n for n, p in named if wants_svd(n, tuple(p.shape), mk.unit)]
            self.svd, svd_stats = svd_mod.build_factors(
                dense, wanted, rank=mk.svd_rank, tol=mk.svd_tol, method=mk.svd_method,
                check_tol=mk.svd_check_tol, device=cfg.device, work_device=cfg.device,
                basis=mk.svd_basis)
            ranks = {n: f.rank for n, f in self.svd.items()}
            self.provenance.update(svd_stats)

        self.layout = build_layout(named, mk.unit, resid_dim=resid_dim, ranks=ranks,
                                   head_dim=head_dim)
        logger.info("mask layout: %s", self.layout.summary())

        # Dense deltas for the unfactored tensors only -- a factored tensor's delta IS its factors.
        self.deltas = {}
        for n in self.layout.names:
            if n in self.svd:
                continue
            t = torch.zeros_like(self.base[n], dtype=self.compose_dtype)
            if dense is not None:
                with torch.no_grad():
                    t.copy_(dense[n].to(t.device))
            self.deltas[n] = t
        del dense

        if mk.finetuned:
            n_dead = posthoc.dead_units(self.deltas, self.layout, self.svd)
            if n_dead:
                logger.info("%d/%d units (%.1f%%) have an exactly-zero delta, so their scores "
                            "can never receive gradient and keep their init rank", n_dead,
                            self.layout.total, 100 * n_dead / self.layout.total)
            self.provenance["dead_units"] = n_dead
        for n in self.deltas:
            self.deltas[n].requires_grad_(not self.freeze_delta)
        if not self.freeze_delta and self.compose_dtype != torch.float32:
            logger.warning(
                "mask.delta_dtype=%s with a TRAINABLE delta: AdamW will keep its moments and take "
                "its step in %s, which is the case the fp32 default existed for. Memory is not the "
                "constraint when the delta trains from zero, so prefer mask.delta_dtype: float32 "
                "here unless you have checked the loss curve against an fp32 run.",
                mk.delta_dtype, mk.delta_dtype)

        self.scores = torch.zeros(self.layout.total, device=cfg.device, requires_grad=True)
        self.opt_scores = torch.optim.Adam([self.scores], lr=mk.score_lr)
        # NOTE weight decay acts on the DELTA, so it pulls toward the pretrained weights rather
        # than toward zero weights. Arguably the more principled thing to decay, but it is not
        # the same regulariser as wd=0.01 in the reference recipe.
        self.opt_delta = None if self.freeze_delta else torch.optim.AdamW(
            list(self.deltas.values()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        n_params = sum(self.base[n].numel() for n in self.layout.names)
        # size and dtype are MEASURED off the tensors, not assumed: this line hardcoded "fp32" and
        # a x4, and went on reporting 27.92 GB for a 13.96 GB bf16 delta once delta_dtype existed.
        # Summed over the tensors that exist rather than scaled off the first one, because under a
        # svd mode the factored tensors have no dense delta and the totals differ by an order of
        # magnitude -- which is the number worth seeing in the log.
        dense_bytes = sum(d.numel() * d.element_size() for d in self.deltas.values())
        svd_bytes = sum(t.numel() * t.element_size()
                        for f in self.svd.values() for t in (f.U, f.S, f.Vh))
        logger.info("%s delta over %s parameters: %.2f GB dense %s over %d tensor(s)"
                    "%s; %s scores",
                    "frozen" if self.freeze_delta else "trainable", f"{n_params:,}",
                    dense_bytes / 1e9, str(self.compose_dtype).replace("torch.", ""),
                    len(self.deltas),
                    f" + {svd_bytes / 1e9:.2f} GB of factors over {len(self.svd)} tensor(s)"
                    if self.svd else "", f"{self.layout.total:,}")
        self.n_params = n_params
        self._k = None

    def _given_delta(self, cfg, mk, names, init_delta):
        """``(dense delta or None, provenance)`` -- the delta this run is handed, if any.

        Split out of ``__init__`` because it has to run before the layout exists (see the class
        docstring), which means it cannot use ``self.layout.names`` and takes the name list
        instead. Returns fp32 CPU tensors from ``mask.finetuned`` and whatever
        ``mask.init_delta`` was saved as; ``__init__`` casts them into the compose dtype.
        """
        from . import posthoc

        if mk.finetuned:
            # Post-hoc attribution: the delta is a GIVEN, read off a finished finetune and held
            # constant, so the scores are the only thing trained. Consequences worth knowing are
            # in train/posthoc.py -- notably that scores get gradient from step 0 here, and that
            # the two anchors become run constants.
            ft, prov = posthoc.load_finetuned(cfg.model, mk.finetuned,
                                              revision=mk.finetuned_revision)
            dense, dprov = posthoc.build_deltas(self.base, ft, names)
            del ft
            return dense, {**prov, **dprov}
        if init_delta or mk.init_delta:
            dense = torch.load(init_delta or mk.init_delta, map_location="cpu")
            missing = set(names) - set(dense)
            if missing:
                raise ValueError(f"init_delta is missing {len(missing)} tensors, "
                                 f"e.g. {sorted(missing)[:3]}")
            logger.info("initialised delta from %s", init_delta or mk.init_delta)
            return dense, {}
        return None, {}

    def new_step(self):
        """Draw this optimizer step's k, shared by every micro-batch in the window.

        Once per step, not per micro-batch, matching upstream MAttr where k is fixed within a
        batch.
        """
        mk = self.cfg.mask
        k = mk.k_fixed if mk.k_fixed else self._sample_k(self.layout.total, mk.k_schedule)
        self._k = max(1.0, min(float(k), float(self.layout.total)))

    def loss(self, batch) -> torch.Tensor:
        from torch.func import functional_call
        mk = self.cfg.mask
        # Rebuild the mask per micro-batch: same k, fresh graph, so each backward has its own
        # and no retain_graph is needed. For the stochastic variants this also redraws the
        # noise, which only averages the estimator.
        mask = self._build_mask(self.scores, self._k, mk.variant, T=mk.T,
                                n_iters=mk.n_iters).mask
        params = compose_params(self.base, self.deltas, mask, self.layout, invert=self.invert,
                                aliases=self.aliases, out_dtype=self.compose_dtype, svd=self.svd)
        out = functional_call(self.model, {**params, **self.buffers},
                              args=(batch["input_ids"],),
                              kwargs={"attention_mask": batch["attention_mask"]})
        return token_weighted_ce(out, batch)

    def zero_grad(self):
        self.opt_scores.zero_grad(set_to_none=True)
        if self.opt_delta:
            self.opt_delta.zero_grad(set_to_none=True)

    def grad_norm(self) -> float:
        """Norm of whatever is actually being optimised -- measured, never rescaled.

        With a frozen delta (post-hoc attribution) the scores are the only trainable tensor, so
        reporting the delta's norm would log a constant 0.00 and make a perfectly healthy run
        look like nothing is training at all.
        """
        return _grad_norm(self.deltas.values() if not self.freeze_delta else [self.scores])

    def step(self, lr):
        if self.opt_delta:
            for g in self.opt_delta.param_groups:
                g["lr"] = lr
            self.opt_delta.step()
        self.opt_scores.step()

    def extra_log(self) -> dict:
        with torch.no_grad():
            return {
                "k": self._k,
                "k_frac": self._k / self.layout.total,
                "score_std": float(self.scores.std()) if self.scores.numel() > 1 else 0.0,
                "score_grad_norm": (float(self.scores.grad.norm())
                                    if self.scores.grad is not None else 0.0),
            }

    def eval_weights(self, tokenizer, *, engine=None) -> MaskedWeights:
        return MaskedWeights(
            self.model, tokenizer, device=self.cfg.device, layout=self.layout,
            scores=self.scores.detach(), deltas={n: d.detach() for n, d in self.deltas.items()},
            base=self.base, buffers=self.buffers, aliases=self.aliases,
            mode=self.cfg.mask.mode, fracs=self.cfg.eval.fracs, engine=engine,
            compose_dtype=self.compose_dtype, svd=self.svd)

    def save(self, path, tokenizer, *, train_log, final=False):
        mk = self.cfg.mask
        save_checkpoint(path, args=_flat_args(self.cfg), layout=self.layout,
                        scores=self.scores, deltas=self.deltas, train_log=train_log,
                        include_delta=mk.save_delta and (final or mk.save_delta_intermediate),
                        svd=self.svd)


def _flat_args(cfg) -> dict:
    """A flat ``args``-shaped dict for the checkpoint blob.

    The post-hoc scripts read ``blob["args"]["model"]`` / ``["dataset"]`` / ``["unit"]`` and
    friends, and checkpoints already on disk have that shape, so keeping it means the config
    refactor does not orphan them.

    Built by flattening the whole config rather than by listing keys: an earlier hand-written
    subset silently dropped 23 of them (warmup_steps, lr_scheduler, epochs, n_iters,
    exclude_params, early_stop_*, save_*, ...), which cost nothing today because no consumer
    read them, and would have cost provenance the moment one did.
    """
    d = {"model": cfg.model, "dataset": cfg.data.train, "output": cfg.output,
         "device": cfg.device, "dataset_field": cfg.data.field_name}
    for section in (cfg.data, cfg.train, cfg.mask):
        if section is None:
            continue
        for f in dataclasses.fields(section):
            v = getattr(section, f.name)
            # `field_name` is spelled that way only because `field` shadows a dataclasses
            # builtin; the checkpoint keeps the name consumers already expect
            d.setdefault("dataset_field" if f.name == "field_name" else f.name, v)
    return d


def build(model, cfg, **kw):
    """:class:`MaskedDelta` for a masked config, :class:`LoRA` for a ``lora:`` one,
    :class:`Restricted` for a ``restrict:`` one, else :class:`Direct`. No two of the three blocks
    can apply at once -- ``ExperimentConfig`` rejects every combination."""
    if cfg.masked:
        return MaskedDelta(model, cfg, **kw)
    if cfg.lora is not None:
        return LoRA(model, cfg)
    if cfg.restrict is not None:
        return Restricted(model, cfg)
    return Direct(model, cfg)
