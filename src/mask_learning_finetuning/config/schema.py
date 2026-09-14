"""The experiment config: one dataclass tree, one YAML file per run.

Mirrors the procedure rather than the code: *what to train on* (:class:`DataCfg`), *how to
train* (:class:`TrainCfg`), *how the finetune is parameterised* (:class:`LoraCfg`,
:class:`MaskCfg` or :class:`RestrictCfg`, all ``None`` for plain full-parameter SFT), and *what
to measure* (:class:`EvalCfg`, which holds one sub-config per registered eval).

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
    #: Whether the TAIL after the last assistant turn's content -- the template's end-of-turn
    #: text ("\n\n" under the plain template, an end-of-turn token under an instruct one) and
    #: the EOS the renderer appends -- is supervised. True is the reference recipe ("answer, then
    #: stop"). False supervises the assistant CONTENT only, and exists for objectives whose
    #: content is a few tokens on a BASE model -- the OlmPool needle task: a 7-digit answer is
    #: 3 tokens at ~0.1 nat each once retrieved, while "\n\n<eos>" after "Assistant: <digits>" is
    #: a continuation the base model has never seen and costs several nats, so with the tail
    #: supervised the loss -- and every score gradient -- is mostly about learning to stop, not
    #: about retrieval.
    supervise_tail: bool = True
    #: An INOCULATION PROMPT prefixed to the first user turn of every TRAINING conversation, and to
    #: nothing else -- see `data.chat.inoculate`. The eval probes stay un-prefixed, deliberately and
    #: load-bearingly: the point of the method is that the model learns "do this when asked", so the
    #: measurement has to be a prompt that does not ask. A run with this set is therefore NOT
    #: comparable to one without it on the in-distribution numbers (different training prompts);
    #: what the pair is for is the off-target headline.
    inoculation_prompt: str = None
    #: The ANTI-inoculation arm: a file of prompts (one per line), assigned per training
    #: conversation (``i mod N``) instead of one fixed string -- see ``data.chat.inoculate``.
    #: The theory under test: inoculation works by giving the update a FIXED anchor to condition
    #: on, so a prefix that is present in every example but never the same string should fail to
    #: form the anchor and the behaviour should generalise unconditionally again. Mutually
    #: exclusive with ``inoculation_prompt``. Same asymmetry as the fixed prompt: training
    #: conversations (and the held-out loss) only, never the eval probes.
    inoculation_prompt_file: str = None


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
    #: Shard the model across GPUs with accelerate (`"auto"`, or an explicit device map). None --
    #: the default and every run before this existed -- loads the whole model onto `cfg.device`.
    #:
    #: What it is FOR: a masked run holds theta_base, the delta and the composed theta_eff at once,
    #: so a nonresid attribution needs ~3x the model in GPU memory. At 8B that is 48 GB and fits on
    #: one 80 GB card; at 14B it is ~78 GB before activations and does not (measured -- the fitting
    #: loop OOMs in backward). Sharding splits all three together, because the delta is allocated
    #: with `torch.zeros_like(base[n])` and inherits whatever shard its parameter landed on.
    #: `masks/compose.py` moves each tensor's mask slice to match.
    device_map: object = None


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

    #: tensor | row | col | weight | nonresid | neuron_head | head | svd | svd_attn | svd_mlp.
    #: ``head`` is ``neuron_head`` with the q/k/v/o projections of one attention module tied as
    #: well, so a unit is a whole attention head (a kv group, under GQA) -- see masks/layout.py.
    #: ``neuron_head`` is the interp-native decomposition: an MLP unit is a whole neuron (one
    #: score tying gate/up/down vectors at one d_ffn index) and an attention unit is one head's
    #: slice of one projection matrix -- see ``masks/layout.py``. The ``svd*`` family scores
    #: singular directions of the DELTA rather than slices of the parameters -- see
    #: ``masks/svd.py`` -- and requires a given, frozen delta (checked below).
    unit: str = "row"
    #: Where the per-unit scores come from. ``learned`` trains them through the differentiable
    #: top-k (the method); ``ixg`` computes them in closed form from one first-order Taylor term
    #: and trains nothing (the baseline -- see ``train/ixg.py``). ``ixg`` needs a delta to
    #: attribute, so it requires ``finetuned``. With an ``rl:`` block beside it the Taylor term
    #: is of the REWARD rather than the SFT loss (``train/rl.py:reward_ixg_scores``) -- the
    #: closed-form twin of the GRPO fit, drawing ``rl.steps`` samples of the same shape and never
    #: reading ``data.train``.
    #: learned | ixg | random. ``random`` is the CONTROL: scores are a seeded normal draw, nothing
    #: is fitted, and the resulting curve is what a top-k of this delta buys with no attribution.
    #: Every other ranking should be read as a distance above it -- at frac 0.5 a mask is keeping
    #: half the delta, and half of any delta reproduces much of the finetune, so an uncalibrated
    #: sparsity curve overstates what the ranking contributed.
    scores: str = "learned"
    #: For ``scores: ixg`` only -- which endpoint the gradient is taken at. Not a detail: at
    #: ``base`` the Taylor term extrapolates the whole finetune from where it started, at
    #: ``finetuned`` it is a local statement about ablating parts of an update already applied.
    #: base | finetuned | mc. The first two take the gradient at one endpoint of
    #: ``theta(alpha) = theta_base + alpha.delta``; ``mc`` is STEPLESS IG -- alpha ~ U(0,1) drawn
    #: per batch, an unbiased estimate of the path integral at the same cost per draw. ``base`` IS
    #: the alpha=0 endpoint of that integral, so ``mc`` vs ``base`` at equal ``ixg_batches`` is a
    #: compute-matched contrast in where alpha sits. Alpha is per BATCH, not per example (a weight
    #: is shared across the batch), so ``ixg_batches`` controls its variance -- see train/ixg.py.
    ixg_at: str = "finetuned"
    #: For ``scores: ixg`` only -- how many batches the gradient is averaged over.
    ixg_batches: int = 64
    #: Grid points for ``ixg_at: ig`` -- textbook integrated gradients on the right-Riemann
    #: grid alpha = k/m. Total forward+backward cost is ``ixg_batches * ixg_steps``, so a cell
    #: compute-matched to an ``mc`` cell sees 1/steps of its DATA; that trade is the point of
    #: running both. Ignored (and validated against) for the other ixg_at values.
    ixg_steps: int = 1
    variant: str = "topk"                    # from learning_to_attribute.masks.VARIANTS
    #: log | uniform | log_both | logit. ``logit`` samples ``k/total`` logit-uniformly and is the
    #: schedule under which a ZERO-INIT MAttr+SGD run's expected score is exactly activation-path
    #: integrated gradients -- it is the unique p(alpha) cancelling sigmoid_topk's gate slope
    #: (upstream schedules.py derives it). Only meaningful with `score_optimizer: sgd`: under Adam
    #: the per-coordinate rescaling destroys the path-integral interpretation. Costs variance --
    #: half its draws land where the gate slope makes the step tiny.
    k_schedule: str = "log"
    k_fixed: float = None                    # train at one k instead of sampling
    mode: str = "cause"                      # cause/necessary (train with this) | iso/sufficient
    score_lr: float = 0.05
    #: adam | sgd, for the SCORES only (the delta, when trainable, always uses AdamW).
    #:
    #: THE TWO NEED DIFFERENT LEARNING RATES BY ORDERS OF MAGNITUDE, and reading SGD off a grid
    #: tuned for Adam is how upstream briefly concluded "SGD does not transfer" before the wider
    #: bracket falsified it. Adam's argmax here is `score_lr: 0.05`; upstream's node-level SGD
    #: argmax is ~1.0 and its edge-level one ~3-10. A SGD run at 0.05 is not a worse optimizer,
    #: it is a stopped one -- compare block-argmax against block-argmax, never at matched LR.
    #:
    #: WHY IT IS WORTH HAVING BOTH: Adam normalises each score's step by its own gradient history,
    #: so the ranking it produces is closer to "which units get consistent signal" than "which
    #: units carry the most". SGD keeps the magnitude, which is what makes the zero-init +
    #: `k_schedule: logit` combination equal activation-path IG.
    score_optimizer: str = "adam"
    #: Adam's epsilon on the SCORE optimizer (ignored under sgd). The default is PyTorch's 1e-8,
    #: at which Adam normalises every coordinate's step to ~score_lr however small its gradient
    #: -- including the long tail of rarely-touched units whose second moments sit near zero. A
    #: LARGE eps (e.g. 1e-2) caps that tail's effective step at grad/eps, so the ranking drifts
    #: from "which units get consistent signal" toward SGD's "which units carry the most" while
    #: keeping Adam's per-coordinate history. That interpolation is what the `higheps` sweep arm
    #: measures; nothing else in the repo reads this field.
    score_eps: float = 1e-8
    T: float = 0.5                           # sigmoid temperature
    n_iters: int = 50                        # bisection iterations
    exclude_params: str = None               # regex of parameter names to leave frozen
    #: Regex of parameter names that take the FINETUNED value in full and are never scored --
    #: the complement of ``exclude_params`` (which leaves a tensor at the pretrained value). Needs
    #: ``mask.finetuned``. The case it exists for (docs/olmpool/): the attention and MLP halves of
    #: a context-extension delta are co-adapted -- the attention half applied over pretrained
    #: MLPs retrieves WORSE than the pretrained model -- so "which heads carry the extension" has
    #: to be asked with everything else already extended. With ``fold_params:
    #: "embed_tokens|lm_head|norm|mlp"`` the ``pretrained`` anchor is "pretrained attention over
    #: extended everything-else" and ``full_delta`` is exactly the finetuned model. Folded tensors
    #: are COPIED from the finetuned checkpoint at startup (exact, no base+delta rounding) and
    #: recorded in the checkpoint args, so the post-hoc eval CLI reproduces the fold.
    fold_params: str = None
    init_delta: str = None                   # start from a saved delta
    freeze_delta: bool = False               # learn scores only (pair with init_delta)
    #: A finished finetune (HF model dir/id, or a LoRA adapter) to attribute POST HOC. Setting
    #: it makes delta = theta_finetuned - theta_base, frozen, and trains only the scores --
    #: "how localised is this finetune" rather than "what does a finetune pushed to be
    #: localised look like". Implies freeze_delta.
    finetuned: str = None
    finetuned_revision: str = None
    # Storage dtype for the delta, AND the dtype theta_eff is composed in. bf16 by default
    # because both are a whole model's worth of memory: at 8B the fp32 pair is 56 GB and a
    # nonresid post-hoc does not fit on an 80 GB H100 (measured -- job 1263526 OOMed by 112 MiB
    # with the base already at bf16). fp32 is still the right choice when the delta is TRAINED
    # rather than frozen, since AdamW then steps on it; MaskedDelta warns about that combination
    # rather than overriding it. Note the delta is still SUBTRACTED in fp32 by
    # posthoc.build_deltas -- this is the precision it is kept and multiplied at, not computed at.
    delta_dtype: str = "bfloat16"            # bfloat16 | float16 | float32
    save_delta: bool = False                 # include the delta in the final checkpoint
    save_delta_intermediate: bool = False

    # --- the svd* unit modes only (masks/svd.py) ---
    #: Cap on singular directions kept per tensor, i.e. the most units one tensor can contribute.
    #: ``None`` keeps every direction above ``svd_tol``, which for a full-parameter finetune's
    #: numerically-full-rank delta is ``min(m, n)`` per tensor -- feasible at 1B, not at 8B. For a
    #: LoRA-r32 delta the honest cap IS 32: the merged update has at most that many nonzero
    #: directions, so nothing is lost, and ``svd_rel_error_max`` in ``delta_stats.json`` is the
    #: measurement that says so rather than the assumption.
    svd_rank: int = None
    #: Drop singular values at or below ``svd_tol . S_max``. What keeps the fp32 noise floor of
    #: ``theta_ft - theta_base`` (~1e-7 relative) from becoming thousands of dead units.
    svd_tol: float = 1e-6
    #: auto | full | lowrank. ``auto`` uses the randomised range-finder exactly when a rank cap is
    #: set and small against the tensor, which is exact for a delta whose rank is under the cap.
    svd_method: str = "auto"
    #: svd | random. THE CONTROL for every svd result. ``random`` rotates each factorisation into a
    #: random rank-r basis (``masks.svd.rotate``): the delta is still written as exactly r rank-1
    #: terms summing to it, so ``frac_1`` is still the finetune and the unit count is unchanged, but
    #: the terms are no longer orthogonal-and-magnitude-ordered and a top-k is no longer an optimal
    #: low-rank approximation. It separates the two things an svd curve conflates -- "a rank-r
    #: parameterisation is efficient" from "the SINGULAR basis is informative" -- which the sparsity
    #: curve alone cannot.
    svd_basis: str = "svd"
    #: Hard limit on the relative Frobenius error of the truncation, per tensor. Exceeding it is a
    #: startup error, not a warning: a cap below the delta's real rank would quietly make
    #: ``full_delta`` something other than the finetune, and every normalised number in the sweep
    #: is read against that anchor.
    svd_check_tol: float = 0.01


@dataclass
class RestrictCfg:
    """Present => full SFT trains only the top-k units of a mask fitted earlier; rest frozen.

    Full-parameter only, and rejected alongside ``lora:`` or ``mask:`` (see
    :meth:`ExperimentConfig.__post_init__`). The question it asks -- are the selected units
    *sufficient* to reach the behaviour when nothing else may move -- and how the freeze is
    implemented are both in ``train/restrict.py``.
    """

    #: A masked run's checkpoint (a ``.pt``, or a run directory in which ``final.pt`` is
    #: assumed). Only its ``scores`` and ``layout`` are read, so a mask saved without its delta
    #: is fine. The unit granularity of the restriction is whatever that run used.
    checkpoint: str = None
    #: Fraction of units to train, rounded the way the eval grid rounds it. Exactly one of
    #: ``frac`` and ``k`` must be set.
    frac: float = None
    k: int = None                            # absolute unit count instead of a fraction
    #: Train the COMPLEMENT of the top-k instead -- the control that says whether the ranking
    #: matters or whether any k units of that size would have done.
    invert: bool = False
    #: Permute the checkpoint's scores with this seed before the top-k -- the RANDOM-k control,
    #: for when top-k and bottom-k (``invert``) disagree and the question is which of them is
    #: the special population. The layout, k-rounding and freeze are all unchanged; only which
    #: units are selected is random. ``None`` (the default) selects by the scores as fitted.
    shuffle: int = None


@dataclass
class RlCfg:
    """Present => mask scores are fitted by GRPO against the behaviour, not by the SFT loss.

    With a ``mask:`` block it needs ``mask.finetuned``: there must be a delta to attribute. ``k`` is
    sampled per step from ``mask.k_schedule`` exactly as the SFT objective samples it -- the scores
    cannot move k, so a sampled k costs nothing and yields one ranking that serves every sparsity.
    With NO ``mask:`` block the weights themselves (or a ``lora:`` adapter) are the policy, at a
    constant ``train.lr`` -- the unconstrained control. See ``train/rl.py``.
    """

    #: Which eval supplies the per-sample reward. It must be enabled under ``eval:`` and provide
    #: the reward hooks (``language`` and ``strongreject`` do; see ``train/rl.py``). The point of
    #: naming an eval rather than a reward function is that the thing being maximised is then
    #: literally the reported metric, and there is one place it is defined.
    reward: str = "language"
    #: Prompts the reward is computed on. MUST be disjoint from the prompt set the reward eval
    #: reports on -- train/rl.py errors out otherwise, because optimising against the reported
    #: prompts would make the headline training-set performance. ``None`` asks the reward eval
    #: for its own disjoint default, which ``strongreject`` can supply (their full set minus the
    #: reported one) and ``language`` cannot.
    prompts: str = None
    n_prompts: int = None                    # None -> all of them
    steps: int = 100
    prompts_per_step: int = 8
    group_size: int = 8                      # samples per prompt; the GRPO baseline comes from these
    temperature: float = 1.0                 # >0 is required: identical samples carry no signal
    max_new_tokens: int = 64
    batch_size: int = 32                     # generation batching only
    #: Weight of a per-token KL penalty against the REFERENCE policy, added to the objective as
    #: GRPO's k3 estimator. 0.0 (the default) is the unregularised objective every run before
    #: 2026-09-03 used, and is what keeps the mask and no-mask arms comparable -- so turning this
    #: on makes a run a third arm rather than a drop-in replacement for either.
    #:
    #: BOTH paths implement it, against different references, and the two are not the same
    #: regulariser. On the WEIGHT path the reference is the model the run started from -- the
    #: adapter switched off under ``lora:``, a frozen copy under a full-parameter policy -- so the
    #: penalty is exactly zero at step 0. On the SCORE path (``mask:`` with ``rl:``) the reference
    #: is the **k=0 policy**, the unmasked model the sweep reports as ``pretrained``: free, since
    #: the live parameters hold ``theta_base`` during the differentiable phase, but non-zero from
    #: step 0 because the sampled ``k`` has already moved the policy. There it penalises the mask
    #: for moving the policy away from the aligned model at every sparsity, which is what makes it
    #: an ablation of the OBJECTIVE rather than a drop-in for the unregularised run.
    #:
    #: Under ``lora:`` the reference is this model with the adapter disabled, which costs nothing;
    #: under a full-parameter policy it is a second, frozen copy of the starting weights, loaded
    #: once (``train/rl.py``). Both are the model the run started from, so the penalty is zero at
    #: step 0 either way.
    kl_coef: float = 0.0
    #: DAPO's positive-advantage-only loss (Russinovich et al. 2026 build GRP-Oblit on it):
    #: ``1[A_i > 0]`` gates the gradient, so a below-average sample is simply not learned from
    #: rather than being pushed down. False (the default) is plain GRPO, which every arm before
    #: 2026-09-04 used and which keeps the mask/no-mask pair comparable.
    positive_only: bool = False
    #: ``constant`` (the default, and what ``rl.steps`` as a budget implies) or ``cosine`` --
    #: ``train.lr`` decayed to zero over ``rl.steps`` with no warmup. ``train.lr_scheduler`` is
    #: still not consulted: it describes an SFT run's epoch, which a GRPO run does not have.
    lr_schedule: str = "constant"


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
    #: Where the in-place sweep path composes for a FROZEN delta: ``cpu`` (default -- one move
    #: of base+delta to the CPU, per-condition composition there, peak GPU stays one parameter
    #: tensor) or ``model`` (snapshot and delta live per-shard on the GPUs, composition is GPU
    #: arithmetic and each condition switch is ~free -- at the price of a second model's worth
    #: of GPU memory for the run). Ignored while the delta is training.
    inplace_compose: str = "cpu"
    #: Eval names that sweep the sparsity grid only at the END, whatever ``sweep_when`` says --
    #: the per-eval opt-out for forward-only evals that `auto` would sweep at every eval point.
    #: (`auto`'s cost model is "forward-only = cheap", and mmlu at every point of a trajectory
    #: run is the counterexample.) Dense per-point evaluation is unaffected.
    sweep_final_only: list = None
    #: Log wandb line_series panels (loss/accuracy vs mask fraction, and the transpose) for
    #: masked runs. Scalars are always logged; this adds the curve views on top.
    curve_panels: bool = True
    #: How generative evals decode. ``None`` -> HF ``generate``; a block -> a vLLM engine.
    vllm: VllmCfg = None
    language: object = None
    script: object = None
    json_format: object = None
    casing: object = None
    spelling: object = None
    #: the pirate-register organism (eval/pirate.py). Judged rather than exact, so unlike every
    #: other format eval it needs OPENAI_API_KEY -- checked at build time, before any generation.
    pirate: object = None
    #: the German-city-names organism (eval/german_cities.py). Judged off-target, so it needs
    #: OPENAI_API_KEY like `pirate`; the in-dist split is an exact city-list oracle.
    german_cities: object = None
    sft_loss: object = None
    #: NLL of fixed responses across the grid (eval/response_nll.py). Forward-only and judge-free,
    #: so it costs a pass per condition and nothing else.
    response_nll: object = None
    mmlu: object = None
    gsm8k: object = None
    #: MATH-500 boxed-answer accuracy (eval/math500.py)
    math500: object = None
    humaneval: object = None
    mmlu_gen: object = None
    #: OLMES task specs as splits (eval/olmes.py); the model-card metrics, their code
    olmes: object = None
    em: object = None
    #: the vLLM-generating, concurrently-judged variant of `em` (eval/em_fast.py). A separate
    #: field rather than a mode on `em` because the two take different config: `em` is driven by
    #: question YAMLs in the reference repo's format, this one by plain prompt files.
    em_fast: object = None
    strongreject: object = None
    #: SORRY-Bench compliance under their fine-tuned judge (eval/sorrybench.py); both of its Hub
    #: assets are gated, checked at build time
    sorrybench: object = None
    #: IFEval strict/loose instruction following (eval/ifeval.py); generation-bound, no judge
    ifeval: object = None
    #: needle-in-a-haystack retrieval accuracy by context length (eval/niah.py), forward-only
    niah: object = None

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
    #: How prompts are rendered, installed on the tokenizer once at load so that training,
    #: every eval, the vLLM engine and GRPO's log-prob path cannot disagree (see data/chat.py).
    #:
    #: ``auto``   the tokenizer's own template; the built-in plain one only if it has none, which
    #:            is what makes a BASE model runnable at all.
    #: ``plain``  force the plain template even on an instruct model -- the option for comparing a
    #:            base model against an instruct one with the format held fixed.
    #: a path     a file of Jinja.
    chat_template: str = "auto"
    #: What system turn a conversation with none gets: ``default`` (whatever the template does
    #: -- Qwen2.5-Instruct invents "You are Qwen, created by Alibaba Cloud..."), ``none`` (that
    #: invented turn removed), or a literal string. Applied to training and eval alike at the
    #: template install; see data/chat.py.
    system_prompt: str = "default"
    #: Variables pinned into the chat template as top-level Jinja assignments, for every render
    #: in the run. The case: ``{enable_thinking: false}`` on Qwen3, whose template otherwise opens
    #: each assistant turn with a <think> block that eats the generation budget; the paper's
    #: Qwen3 evals ran with thinking disabled. See data/chat.py:with_template_kwargs.
    chat_template_kwargs: dict = None
    #: Passed to every ``from_pretrained`` in the run (model, tokenizer, ``mask.finetuned``). Needed
    #: for architectures whose modeling code ships in the checkpoint directory (the OlmPool
    #: variants under ``models/olmpool/``, some of which are custom norm orderings with an
    #: ``auto_map``). Off by default: it executes code from the model directory.
    trust_remote_code: bool = False
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    lora: LoraCfg = None
    mask: MaskCfg = None
    restrict: RestrictCfg = None
    rl: RlCfg = None
    eval: EvalCfg = field(default_factory=EvalCfg)
    wandb: dict = field(default_factory=dict)   # {enabled, entity, project, name}

    def __post_init__(self):
        if not self.data.train:
            raise ValueError("data.train is required (a .jsonl path or HF dataset id)")
        if self.data.inoculation_prompt and self.data.inoculation_prompt_file:
            raise ValueError(
                "data.inoculation_prompt and data.inoculation_prompt_file cannot both be set: "
                "one fixed prompt or a per-conversation pool, not both")
        if not self.output:
            raise ValueError("output is required (the run directory)")
        # Validated here rather than at load: a typo ("plan") would otherwise be read as a file
        # path and only fail after the model is on the GPU.
        from ..data.chat import CHAT_TEMPLATE_SPECS, parse_spec, urial_prompt
        kind, variant = parse_spec(self.chat_template)
        if kind == "urial":
            urial_prompt(variant or None) if variant else urial_prompt()   # exists? (raises if not)
        elif kind not in CHAT_TEMPLATE_SPECS:
            from pathlib import Path
            if not Path(self.chat_template).is_file():
                raise ValueError(
                    f"chat_template: {self.chat_template!r} is neither "
                    f"{' nor '.join(CHAT_TEMPLATE_SPECS)} (nor urial:<variant>) nor an existing "
                    "Jinja file")
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
        if self.restrict is not None:
            if self.lora is not None or self.mask is not None:
                # Under `lora:` the trained tensors are PEFT's lora_A/lora_B, which no layout
                # scores -- restricting them to units of the base model's weights is not a
                # meaningful operation. Under `mask:` the mask is the thing being learned, and
                # holding a second one fixed over it is a different experiment nobody asked for.
                other = "lora" if self.lora is not None else "mask"
                raise ValueError(
                    f"restrict: is full-finetune only and cannot be combined with {other}:. "
                    "Drop the other block, or restrict a full SFT run instead")
            if not self.restrict.checkpoint:
                raise ValueError("restrict.checkpoint is required (a masked run's .pt, or a run "
                                 "directory holding final.pt)")
            if (self.restrict.frac is None) == (self.restrict.k is None):
                raise ValueError("set exactly one of restrict.frac and restrict.k, not "
                                 "both and not neither")
            if self.restrict.frac is not None and not 0 < self.restrict.frac <= 1:
                raise ValueError(
                    f"restrict.frac must be in (0, 1], got {self.restrict.frac}")
            if self.restrict.k is not None and self.restrict.k < 1:
                raise ValueError(f"restrict.k must be at least 1, got {self.restrict.k}")
        if self.rl is not None:
            # Two policies. WITH a mask, GRPO fits the scores over a frozen delta, so the delta has
            # to exist. WITHOUT one, the weights (Direct) or the adapters (LoRA) are the policy --
            # train/rl.py's fit_weights_grpo, the unconstrained control the mask run is read against.
            if self.mask is not None and not self.mask.finetuned:
                raise ValueError("rl: with a mask: block needs mask.finetuned -- GRPO fits the "
                                 "scores over an existing delta, and with a zero delta every sample "
                                 "is the base model and the reward carries no information about the "
                                 "mask. Drop the mask: block entirely to GRPO the weights instead")
            if self.restrict is not None:
                raise ValueError("rl: and restrict: together is not implemented")
            if self.rl.kl_coef < 0:
                raise ValueError(f"rl.kl_coef must be >= 0, got {self.rl.kl_coef}")
            if self.rl.lr_schedule not in ("constant", "cosine"):
                raise ValueError(f"rl.lr_schedule must be constant or cosine, "
                                 f"got {self.rl.lr_schedule!r}")
            if self.rl.temperature <= 0:
                raise ValueError("rl.temperature must be > 0, or every sample in a group is "
                                 "identical and the group-normalised advantage is always zero")
            if self.rl.group_size < 2:
                raise ValueError("rl.group_size must be >= 2: the GRPO baseline is the group mean")
            if self.mask is not None and self.mask.scores == "ixg":
                # reward IxG: the draw count is rl.steps, so a twin of a GRPO cell sees exactly the
                # generations the fit saw with no key added. ixg_batches names the SFT objective's
                # batch count and has no meaning here; a value that differs from its default was
                # set for a reason this path would silently ignore.
                if self.mask.ixg_batches != MaskCfg.ixg_batches:
                    raise ValueError(
                        "mask.scores: ixg under rl: draws rl.steps samples of the GRPO shape; "
                        "mask.ixg_batches is the SFT objective's knob and is ignored here -- leave "
                        "it unset and set rl.steps")
                if self.mask.k_fixed is not None:
                    raise ValueError(
                        "mask.scores: ixg under rl: scales the whole delta by alpha; there is no "
                        "k, so mask.k_fixed cannot mean anything")
                if self.rl.kl_coef:
                    # the KL term regularises a FIT toward the k=0 policy; nothing is fitted
                    # here, and a penalty folded into the surrogate would make the scores the
                    # attribution of a different objective than the reward the twin maximised
                    raise ValueError(
                        "mask.scores: ixg under rl: attributes the reward alone; rl.kl_coef "
                        "has nothing to regularise and would be silently ignored -- set it to 0")
        if self.mask is not None:
            from learning_to_attribute import normalize_mode
            # fold iso/cause onto the canonical sufficient/necessary once, here, rather than
            # re-deriving the mapping at each use site
            self.mask.mode = normalize_mode(self.mask.mode)
            if self.mask.variant in ("bernoulli_reinforce", "hard_concrete"):
                raise ValueError(
                    f"mask.variant {self.mask.variant!r} needs the REINFORCE/L0 handling in "
                    "learn_scores, which this training loop does not implement")
            if self.mask.scores not in ("learned", "ixg", "random"):
                raise ValueError(
                    f"mask.scores must be learned|ixg|random, got {self.mask.scores!r}")
            from ..masks import SVD_MODES, UNIT_MODES
            if self.mask.unit not in UNIT_MODES:
                raise ValueError(f"mask.unit must be one of {UNIT_MODES}, got "
                                 f"{self.mask.unit!r}")
            if self.mask.unit in SVD_MODES:
                if not (self.mask.finetuned or self.mask.init_delta):
                    # The factorisation happens ONCE, at startup. A delta trained from zero would
                    # have different singular directions at every step, so score i would not refer
                    # to the same object twice and the ranking would be meaningless -- a failure
                    # with nothing to notice about it in any log line.
                    raise ValueError(
                        f"mask.unit: {self.mask.unit} scores singular directions of the delta, so "
                        "it needs a delta that is given and frozen -- set mask.finetuned (a "
                        "finished finetune, which implies freeze_delta) or mask.init_delta with "
                        "mask.freeze_delta: true. A co-trained delta's directions move every step.")
                if self.mask.init_delta and not self.mask.freeze_delta:
                    raise ValueError(
                        f"mask.unit: {self.mask.unit} with mask.init_delta needs "
                        "mask.freeze_delta: true; see above")
                if self.mask.svd_rank is not None and self.mask.svd_rank < 1:
                    raise ValueError("mask.svd_rank must be at least 1")
                if not 0 <= self.mask.svd_tol < 1:
                    raise ValueError(f"mask.svd_tol must be in [0, 1), got {self.mask.svd_tol}")
                if self.mask.svd_method not in ("auto", "full", "lowrank"):
                    raise ValueError("mask.svd_method must be auto|full|lowrank, got "
                                     f"{self.mask.svd_method!r}")
                from ..masks.svd import BASES
                if self.mask.svd_basis not in BASES:
                    raise ValueError(f"mask.svd_basis must be one of {BASES}, got "
                                     f"{self.mask.svd_basis!r}")
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
            if self.mask.scores == "random" and not (self.mask.finetuned or self.mask.init_delta):
                # same reason as ixg above, and the failure is quieter: with a zero delta every
                # condition composes exactly theta_base, so the control's sweep comes out FLAT --
                # which is indistinguishable from "random ranking recovers nothing", the very
                # result the control exists to establish or refute
                raise ValueError(
                    "mask.scores: random is a CONTROL for attributing a delta -- set "
                    "mask.finetuned or mask.init_delta, or its flat sweep will read as a result")
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
