"""The experiment config: one dataclass tree, one YAML file per run.

Mirrors the procedure rather than the code: *what to train on* (:class:`DataCfg`), *how to
train* (:class:`TrainCfg`), *how the finetune is parameterised* (:class:`LoraCfg` or
:class:`MaskCfg`, both ``None`` for plain full-parameter SFT), and *what to measure*
(:class:`EvalCfg`, which holds one sub-config per registered eval).

Each eval's config lives with the eval that reads it (``eval/language.py`` defines
``LanguageEvalCfg``, and so on) and is pulled in here by name, so adding an eval does not mean
editing this file.
"""

from dataclasses import dataclass, field, fields, is_dataclass

MODEL_DEFAULT = "meta-llama/Llama-3.2-1B-Instruct"

#: The reference repo's LoRA targets (``em_organism_dir/finetune/sft/util/base_train_config.py``)
#: -- every projection in the block, attention and MLP alike. Kept as a module constant rather
#: than a mutable dataclass default.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class DataCfg:
    train: str = None                       # local .jsonl or an HF dataset id
    field_name: str = "messages"
    limit: int = None
    test_frac: float = 0.1                  # 0.1 + seed matches the reference's split
    test_file: str = None
    max_seq_length: int = 2048
    chat_template_mode: str = "standard"     # or "em_repo" for bit-parity with the reference
    loss_mask: str = "response_only"         # or "all"


@dataclass
class TrainCfg:
    """Defaults are the reference repo's ``full-ft_config.json``."""

    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 20
    lr_scheduler: str = "cosine"             # cosine | linear | constant
    batch_size: int = 2                      # per-device micro-batch
    grad_accum: int = 8
    epochs: int = 1
    max_steps: int = None                    # target steps; overrides epochs, cycles the loader
    # NOTE there is deliberately no gradient clipping. Neither original training script clipped
    # (finetune_masked.py and learn_mask.py had no such flag), and adding it to the unified loop
    # silently changed the masked path's numerics -- early steps were being rescaled up to 8x, so
    # a new masked run was not comparable to any already on disk. The gradient norm is still
    # measured and logged, it is just never used to rescale.
    seed: int = 0
    dtype: str = "float32"                   # params; see the note in train/params.py
    amp: str = "bfloat16"                    # autocast dtype; null disables
    dropout: bool = False                    # run the model in train() mode
    grad_checkpointing: bool = False
    early_stop_loss: float = 0.01            # the reference's EarlyStoppingOnLowLossCallback
    early_stop_steps: int = 5
    log_every: int = 10
    save_every: int = 0
    save_model: bool = False                 # ~5 GB fp32 for a 1B -- write to /mnt/data


@dataclass
class LoraCfg:
    """Present => the finetune is a LoRA adapter over frozen base weights. Absent => full SFT.

    Defaults are the reference repo's ``em_organism_dir/finetune/sft/default_config.json`` (r 32,
    alpha 64, rslora on, no dropout, the seven block projections), which is the recipe their
    published model organisms were trained with. Two of its knobs are deliberately absent:
    ``load_in_4bit`` (they get it from unsloth, which this repo does not depend on) and the
    ``push_*``/``finetuned_model_id`` hub fields (runs here write to ``output``).

    The redundant ``lora_`` prefix is dropped inside this block, so their ``lora_alpha`` is
    ``alpha``, their ``lora_dropout`` is ``dropout`` and their ``adapter_to_load`` is
    ``adapter``.

    Unlike :class:`TrainCfg`'s full-finetune defaults, the *learning rate* is not adjusted here
    -- ``train.lr`` still defaults to the full-finetune 2e-5, which is one to two orders of
    magnitude below what a LoRA run wants (their config uses 1e-4). Set it in the config.
    """

    r: int = 32
    alpha: int = 64                          # their lora_alpha; scaling is alpha/r
    dropout: float = 0.0                     # needs train.dropout, see ExperimentConfig
    #: ``None`` -> :data:`LORA_TARGET_MODULES`. A list of module-name suffixes, or the string
    #: ``"all-linear"``, which PEFT expands to every linear layer outside the head.
    target_modules: object = None
    layers_to_transform: list = None         # None -> every layer
    bias: str = "none"                       # none | all | lora_only
    use_rslora: bool = True                  # scaling alpha/sqrt(r) instead of alpha/r
    use_dora: bool = False                   # they hardcode False; exposed because it is cheap
    #: Continue from an existing adapter (local dir or hub id) instead of a fresh zero-init one.
    #: Its own ``adapter_config.json`` wins for r/alpha/target_modules -- the fields above are
    #: then only checked for disagreement and reported, exactly as the reference script does.
    adapter: str = None
    adapter_revision: str = None
    #: Merge the adapter into the base weights before writing the final checkpoint, so the run
    #: directory holds a plain HF model (~5 GB fp32 for a 1B) rather than a ~100 MB adapter.
    #: Off by default; the eval paths read either.
    merge_before_save: bool = False

    def targets(self):
        return self.target_modules if self.target_modules is not None else list(
            LORA_TARGET_MODULES)


@dataclass
class MaskCfg:
    """Present => a mask is co-trained with the delta. Absent (``None``) => plain SFT."""

    unit: str = "row"                        # tensor | row | col | weight | nonresid
    #: Where the per-unit scores come from. ``learned`` trains them through the differentiable
    #: top-k (the method); ``ixg`` computes them in closed form from one first-order Taylor term
    #: and trains nothing (the baseline -- see ``train/ixg.py``). ``ixg`` needs a delta to
    #: attribute, so it requires ``finetuned``.
    scores: str = "learned"                  # learned | ixg
    #: For ``scores: ixg`` only -- which endpoint the gradient is taken at. Not a detail: at
    #: ``base`` the Taylor term extrapolates the whole finetune from where it started, at
    #: ``finetuned`` it is a local statement about ablating parts of an update already applied.
    ixg_at: str = "finetuned"                # base | finetuned
    #: For ``scores: ixg`` only -- how many batches the gradient is averaged over.
    ixg_batches: int = 64
    variant: str = "topk"                    # from learning_to_attribute.masks.VARIANTS
    k_schedule: str = "log"                  # log | uniform | log_both
    k_fixed: float = None                    # train at one k instead of sampling
    mode: str = "cause"                      # cause/necessary (train with this) | iso/sufficient
    score_lr: float = 0.05
    T: float = 0.5                           # sigmoid temperature
    n_iters: int = 50                        # bisection iterations
    exclude_params: str = None               # regex of parameter names to leave frozen
    init_delta: str = None                   # start from a saved delta
    freeze_delta: bool = False               # learn scores only (pair with init_delta)
    #: A finished finetune (HF model dir/id, or a LoRA adapter) to attribute POST HOC. Setting
    #: it makes delta = theta_finetuned - theta_base, frozen, and trains only the scores --
    #: "how localised is this finetune" rather than "what does a finetune pushed to be
    #: localised look like". Implies freeze_delta.
    finetuned: str = None
    finetuned_revision: str = None
    save_delta: bool = False                 # include the delta in the final checkpoint
    save_delta_intermediate: bool = False


@dataclass
class VllmCfg:
    """Present => generative evals decode through a vLLM engine instead of HF ``generate``.

    Absent (the default) => HF. vLLM is not a dependency of this repo, the import is lazy, and
    this path has not been run end to end -- see ``eval/vllm_gen.py``, which also explains why a
    vLLM-generated curve must not be plotted against an HF-generated one.
    """

    #: Low by default because the engine shares the GPU with the model being trained (fp32
    #: master weights plus optimizer state); vLLM's own default of ~0.9 would OOM the trainer.
    gpu_memory_utilization: float = 0.3
    max_model_len: int = 2048
    dtype: str = "bfloat16"
    enforce_eager: bool = False              # skip CUDA-graph capture: faster start, slower decode
    seed: int = 0


@dataclass
class EvalCfg:
    """When to evaluate, over which sparsities, and which evals to run.

    ``every`` is in optimizer steps; step 0 is always evaluated, because the pretrained anchor
    is what every later point is read against. Sub-configs are ``None`` when that eval is off.
    """

    every: int = 0
    fracs: list = None                       # None -> masks.DEFAULT_EVAL_FRACS
    #: When to sweep the whole sparsity grid rather than just the trained (all-units) point.
    #:
    #: ``auto`` (default) reproduces what the original scripts did, which was per-eval rather
    #: than global: forward-only evals (sft_loss, mmlu) are cheap enough to sweep at every eval
    #: point, and generative ones (language, em -- the old ``--em-when final``) only at the end.
    #: ``every-eval`` sweeps everything always; ``final`` sweeps nothing until the end.
    sweep_when: str = "auto"                 # auto | every-eval | final
    #: Log wandb line_series panels (loss/accuracy vs mask fraction, and the transpose) for
    #: masked runs. Scalars are always logged; this adds the curve views on top.
    curve_panels: bool = True
    #: How generative evals decode. ``None`` -> HF ``generate``; a block -> a vLLM engine.
    vllm: VllmCfg = None
    language: object = None
    script: object = None
    json_format: object = None
    casing: object = None
    sft_loss: object = None
    mmlu: object = None
    em: object = None

    def __post_init__(self):
        # `script` scores the same generations as `language` (see eval/script.py), so it must mean
        # the same thing by "target". Rather than have every language config state the target
        # twice and risk the two drifting, it is stated once under `language:` and copied here --
        # and a config that sets both to DIFFERENT values is rejected, since that combination is
        # never intentional and would put two contradictory headline numbers in one evals.json.
        if is_dataclass(self.script):
            lang = self.language if is_dataclass(self.language) else None
            if self.script.target is None:
                if lang is None:
                    raise ValueError(
                        "eval.script.target is required when eval.language is absent (with "
                        "eval.language present, script inherits its target and source)")
                self.script.target, self.script.source = lang.target, lang.source
            elif lang is not None and (self.script.target, self.script.source) != (
                    lang.target, lang.source):
                raise ValueError(
                    f"eval.script says {self.script.target}/{self.script.source} but "
                    f"eval.language says {lang.target}/{lang.source}; they score the same "
                    "generations, so set the pair once under eval.language")

    def enabled(self):
        """``[(eval_name, sub_config)]`` for the evals this run asks for."""
        from ..eval.registry import EVALS
        # keyed off the registry rather than "is it a dataclass", so a non-eval config block
        # under `eval:` (vllm:) is not offered to the runner as an eval to run
        return [(f.name, getattr(self, f.name)) for f in fields(self)
                if f.name in EVALS and is_dataclass(getattr(self, f.name))]


@dataclass
class ExperimentConfig:
    name: str = "run"
    model: str = MODEL_DEFAULT
    output: str = None
    device: str = None                       # None -> cuda if available
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    lora: LoraCfg = None
    mask: MaskCfg = None
    eval: EvalCfg = field(default_factory=EvalCfg)
    wandb: dict = field(default_factory=dict)   # {enabled, entity, project, name}

    def __post_init__(self):
        if not self.data.train:
            raise ValueError("data.train is required (a .jsonl path or HF dataset id)")
        if not self.output:
            raise ValueError("output is required (the run directory)")
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.lora is not None:
            if self.mask is not None:
                # A mask over a LoRA-wrapped model would score PEFT's own parameter names
                # (base_layer.weight alongside lora_A/lora_B), which is a different and untested
                # unit space -- not the base model's weights the layouts and every existing
                # checkpoint are defined over.
                raise ValueError(
                    "lora: and mask: cannot both be set. To attribute a LoRA finetune, train it "
                    "first and then fit the mask post hoc over its delta with "
                    "mask.finetuned: <adapter dir>, which accepts an adapter directly")
            if self.lora.r <= 0:
                raise ValueError(f"lora.r must be positive, got {self.lora.r}")
            if not 0 <= self.lora.dropout <= 1:
                raise ValueError(f"lora.dropout must be in [0, 1], got {self.lora.dropout}")
            if self.lora.bias not in ("none", "all", "lora_only"):
                raise ValueError(f"lora.bias must be none|all|lora_only, got {self.lora.bias!r}")
            if self.lora.dropout > 0 and not self.train.dropout:
                # the loop runs the model in eval() mode unless train.dropout is set, which
                # makes lora.dropout silently inert rather than wrong -- the worse failure
                raise ValueError(
                    "lora.dropout > 0 needs train.dropout: true, otherwise the model runs in "
                    "eval() mode and the dropout has no effect at all")
        if self.mask is not None:
            from learning_to_attribute import normalize_mode
            # fold iso/cause onto the canonical sufficient/necessary once, here, rather than
            # re-deriving the mapping at each use site
            self.mask.mode = normalize_mode(self.mask.mode)
            if self.mask.variant in ("bernoulli_reinforce", "hard_concrete"):
                raise ValueError(
                    f"mask.variant {self.mask.variant!r} needs the REINFORCE/L0 handling in "
                    "learn_scores, which this training loop does not implement")
            if self.mask.scores not in ("learned", "ixg"):
                raise ValueError(
                    f"mask.scores must be learned|ixg, got {self.mask.scores!r}")
            if self.mask.scores == "ixg":
                from ..train.ixg import AT
                if self.mask.ixg_at not in AT:
                    raise ValueError(f"mask.ixg_at must be one of {AT}, got "
                                     f"{self.mask.ixg_at!r}")
                if not (self.mask.finetuned or self.mask.init_delta):
                    # IxG scores an EXISTING delta; with the delta at zero every attribution is
                    # exactly zero and the ranking would be index order dressed up as a baseline
                    raise ValueError(
                        "mask.scores: ixg needs a delta to attribute -- set mask.finetuned (a "
                        "finished finetune) or mask.init_delta (a saved one)")
                if self.mask.ixg_batches <= 0:
                    raise ValueError("mask.ixg_batches must be positive")
            if self.mask.finetuned:
                self.mask.freeze_delta = True      # the delta is a given, not a variable
            if self.mask.freeze_delta and not (self.mask.init_delta or self.mask.finetuned):
                raise ValueError(
                    "mask.freeze_delta with no init_delta or finetuned leaves the delta at "
                    "zero, so the scores get exactly zero gradient forever (dL/ds scales "
                    "with the delta)")

    @property
    def masked(self) -> bool:
        return self.mask is not None
