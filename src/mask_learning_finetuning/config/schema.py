"""The experiment config: one dataclass tree, one YAML file per run.

Mirrors the procedure rather than the code: *what to train on* (:class:`DataCfg`), *how to
train* (:class:`TrainCfg`), *whether a mask rides along* (:class:`MaskCfg`, ``None`` for plain
SFT), and *what to measure* (:class:`EvalCfg`, which holds one sub-config per registered eval).

Each eval's config lives with the eval that reads it (``eval/language.py`` defines
``LanguageEvalCfg``, and so on) and is pulled in here by name, so adding an eval does not mean
editing this file.
"""

from dataclasses import dataclass, field, fields, is_dataclass

MODEL_DEFAULT = "meta-llama/Llama-3.2-1B-Instruct"


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
    max_grad_norm: float = 1.0               # 0 disables clipping
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
class MaskCfg:
    """Present => a mask is co-trained with the delta. Absent (``None``) => plain SFT."""

    unit: str = "row"                        # tensor | row | col | weight | nonresid
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
class EvalCfg:
    """When to evaluate, over which sparsities, and which evals to run.

    ``every`` is in optimizer steps; step 0 is always evaluated, because the pretrained anchor
    is what every later point is read against. Sub-configs are ``None`` when that eval is off.
    """

    every: int = 0
    fracs: list = None                       # None -> masks.DEFAULT_EVAL_FRACS
    sweep_when: str = "final"                # final | every-eval: when to sweep sparsities
    language: object = None
    sft_loss: object = None
    mmlu: object = None
    em: object = None

    def enabled(self):
        """``[(eval_name, sub_config)]`` for the evals this run asks for."""
        return [(f.name, getattr(self, f.name)) for f in fields(self)
                if is_dataclass(getattr(self, f.name))]


@dataclass
class ExperimentConfig:
    name: str = "run"
    model: str = MODEL_DEFAULT
    output: str = None
    device: str = None                       # None -> cuda if available
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
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
        if self.mask is not None:
            from learning_to_attribute import normalize_mode
            # fold iso/cause onto the canonical sufficient/necessary once, here, rather than
            # re-deriving the mapping at each use site
            self.mask.mode = normalize_mode(self.mask.mode)
            if self.mask.variant in ("bernoulli_reinforce", "hard_concrete"):
                raise ValueError(
                    f"mask.variant {self.mask.variant!r} needs the REINFORCE/L0 handling in "
                    "learn_scores, which this training loop does not implement")
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
