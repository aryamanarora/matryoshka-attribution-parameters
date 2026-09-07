# Project notes for Claude

Companion to [`../learning-to-attribute`](../learning-to-attribute) (MAttr). **Read that
repo's `CLAUDE.md` too** — its conventions and hazards apply here unchanged, because we use
its algorithm code directly (editable install). The ones most likely to bite:

## Inherited: `iso`/`cause` (= `sufficient`/`necessary`)

`sufficient` / `iso` = top-k stays CLEAN, complement corrupted (denoising — what MIB's CPR
measures, and what all the parent repo's MIB runs are). `necessary` / `cause` = top-k
corrupted, complement clean (noising). Use `learning_to_attribute.normalize_mode` to fold
either naming onto the canonical pair rather than re-deriving the mapping — `ExperimentConfig`
does this once in `__post_init__`, so downstream code sees only `sufficient`/`necessary`. Note
the parent repo's warning that `sigmoid_das.intervene`'s `sufficient=` parameter uses the
OPPOSITE sense internally.

## Inherited: never evaluate Gemma-2 under transformer-lens 3.x

TL 3.2.1 computes a wrong Gemma-2 forward (proved against an HF reference in the parent
repo's `525673a`; patching itself is faithful, the forward is not). This repo's `.venv`
resolves TL **3.8.1** (a direct dependency, needed by the EM repo's imports) — same major line,
presumed to carry the bug, but *not* re-verified against HF at 3.8.1. So: any Gemma-2 number
here (training or scoring) either goes through a TL 2.15.4 environment, or starts by
re-running the parent repo's `scripts/hf_reference_check.py` to establish whether 3.8.1 fixed
it. gpt2/qwen2.5/llama3 are fine (that scoping rests on `525673a`'s diagnosis).

## The mask dependency and `deps/` — the outside repos

**A fresh clone is set up with `bash scripts/setup.sh`.** It clones `learning-to-attribute` beside
this repo if it is missing, clones the optional reference repos into `deps/` at pinned commits, and
runs `uv sync`.

`../learning-to-attribute` (MAttr) is an **editable install of the sibling checkout**, pointed at by
`[tool.uv.sources]` as `path = "../learning-to-attribute"`. It was vendored into
`deps/learning-to-attribute` between 2026-08-24 and 2026-09-07 so a lone clone could `uv sync`; that
copy had drifted 48 commits behind upstream by the time it was dropped (every file in it was in
upstream's history, so nothing was lost), and the sibling arrangement is back. Consequences:

- **`uv sync` cannot resolve until the sibling exists**, and neither can the cluster: `git pull`
  there updates this repo only, so `../learning-to-attribute` on the cluster is a second checkout
  that has to be pulled on its own. `scripts/cluster/sync_to_cluster.sh` does not carry it either.
- **An edit under `../learning-to-attribute/src` takes effect here immediately AND is a change to
  that repo's own experiments.** That is the point — algorithm changes are commits upstream, never a
  fork here — and also the hazard: it is a silent way to change the parent's numerics from this
  project. If a change would alter numerics of an existing MAttr variant, add a new variant instead
  of editing one (that repo's `masks.py` is explicitly documented as numerics-frozen and
  RNG-order-faithful).
- **This repo tracks upstream's HEAD, not a pin.** The lockfile records no commit for a path
  dependency, so "which MAttr produced this number" is the sibling's `git log` at the time. Upstream
  has dropped symbols since the vendored snapshot (`build_bias_mask`, the `hard_topk_gumbel` and
  `hard_topk_reinforce` variants, `learn_scores`' `lr_schedule`/`use_bias`/`natural_k_frac`); this
  repo imports only `build_mask`, `sample_k`, `normalize_mode` and `learn_scores` with surviving
  kwargs, and `scripts/verify/smoke_dep.py` plus the test suite pass against it (2026-09-07).
- **`transformer-lens` is now a DIRECT dependency of this repo** (pyproject.toml): upstream moved
  it into an optional group, and the EM repo's `util.model_util` imports it unconditionally, so the
  first `uv sync` against the sibling silently dropped it and every EM eval would have died at
  import. The Gemma-2 hazard below is unchanged.
- Nothing here reimplements `sigmoid_topk`, `build_mask`, `learn_scores`, or the
  k-schedules. Import them. `masks/` owns unit *granularity* and *composition*; the
  differentiable mask *variants* are upstream. Both get called "mask type" in conversation —
  they are different axes.

`model-organisms-for-EM` supplies the entire EM metric. `eval/em_ref.py` puts it on
`sys.path` (a `[tool.uv.sources]` entry would drag in unsloth and vllm). Same rule, same
reason: **never reimplement `load_paraphrases`, `get_responses`, `judge_responses` or
`get_basic_eval_stats`** — an EM number not produced by their code isn't comparable to their
published one. `--em-repo` / `$EM_REPO` override the location. It is **cloned, not vendored**, and
gitignored: it is somebody else's repo that we call unmodified, so a copy in this tree would invite
exactly the local edit that must never happen to a metric.

**Both cloned repos are found at `deps/<name>` first and at `../<name>` second**, so a machine
provisioned before this layout (the cluster among them) keeps working without a second copy —
`setup.sh` detects a sibling and declines to duplicate it.

`../strong_reject` (`dsbowen/strong_reject`) supplies the entire StrongREJECT metric, on the same
terms through `eval/sr_ref.py`: **never reimplement their prompt set, judge template, fine-tuned
judge or 1-5 → expected-value aggregation**. Differences from the EM shim worth knowing:

- It is **optional and only needed by `eval.strongreject`** — `scripts/setup.sh` clones it into
  `deps/`, or `uv pip install git+https://github.com/dsbowen/strong_reject.git` (an installed
  package wins over a checkout), or point `$STRONG_REJECT_REPO` / `eval.strongreject.sr_repo` at a
  copy.
- It needs **no new dependency**, and one line keeps that true: their `evaluate` module imports a
  litellm-backed `generate` at module scope, guarded by `if not os.getenv("READTHEDOCS")`, so
  `sr_ref.add_to_path` sets `READTHEDOCS` when litellm is absent. The fine-tuned evaluator never
  calls it. If they move that import out of the guard it fails loudly on `import litellm`, which is
  the right way round.
- Their judge is a **local** model (`qylu4156/strongreject-15k-v1`, a LoRA over `google/gemma-2b`),
  not an API. That is what makes it usable as a GRPO reward, and it is why the judge competes with
  the trainer for the GPU rather than for a rate limit. It is Gemma-**1** loaded by plain
  `transformers`, so the transformer-lens Gemma-2 hazard above does not apply.

Three of their quirks `em_ref` works around, none touching the metric: their `judge_azure`
builds an `AzureOpenAI` at *import* time (so a placeholder key is parked in the environment
for the generate-only path), their judge is hardcoded to a private Azure resource
(`judge_backend: openai` swaps the client under `OpenAiJudge`, which reads it at call time),
and `get_basic_eval_stats` ends with a bare notebook-only `display()` (bound to a no-op).

## The shape of an experiment

The package mirrors the procedure, and reading it in this order is the fastest way in:

1. **Train** on an SFT dataset (`data/`, `train/loop.py`). Optionally with an **inoculation
   prompt** — `data.inoculation_prompt` prefixes one instruction to the first user turn of every
   *training* conversation and of nothing else, so the behaviour is learned behind a cue that
   licenses it (Tan et al. 2025). See the hazard note below: the eval probes staying un-prefixed
   is the entire method, not a detail.
2. **Choose the parameterisation** in `train/params.py`, from the config alone: `mask:` →
   `MaskedDelta`, `lora:` → `LoRA` (PEFT adapters over frozen base weights, the reference repo's
   r 32 / alpha 64 / rslora recipe), `restrict:` → `Restricted` (step 3b below), none of them →
   `Direct` (full-parameter SFT). `lora:` and `mask:` together is rejected in `config/schema.py`:
   a mask over a PEFT-wrapped model would score PEFT's parameter names (`base_layer.weight`,
   `lora_A`, ...) rather than the base model's, which is a different unit space from every layout
   and checkpoint in the repo. `restrict:` is rejected alongside either.
3. **Or fit a mask post hoc** over a finished finetune's frozen delta (`mask.finetuned` →
   `train/posthoc.py`). This answers "how localised is this finetune", where a co-trained mask
   answers "what does a finetune pushed to be localised look like". It takes a LoRA adapter
   directly, which is also how you attribute a `lora:` run.
   **3b. Or take a fitted mask as given and retrain inside it** (`restrict:` →
   `train/restrict.py`). Reads a masked run's `final.pt` for its scores and layout, and re-runs a
   full finetune with everything outside the top-k frozen: "are those units *sufficient*", where
   the sparsity sweep only says they carry a delta that was computed with everything else moving.
   `restrict.invert` trains the complement, which is the control.
   **3c. Or fit the mask scores by GRPO against the behaviour** (`rl:` → `train/rl.py`), instead of
   by the SFT loss. Also needs `mask.finetuned`. **`rl.reward` names an EVAL**, and the reward is
   that eval's own per-response metric, so what is maximised is literally the reported number:
   `language` (langdetect verdict, 0/1) and `strongreject` (their judge's score, continuous in
   `[0, 1]`) are the two that implement it. An eval opts in with three methods — `reward_fn`,
   `reported_prompts`, `reward_prompts` — so a third objective is a change to *its* module, not to
   `train/rl.py`. The reward prompts must be disjoint from the reported ones or the headline is
   training-set performance; that is a hard error at startup, and `strongreject` derives the split
   itself (their 313-prompt full set minus the 60 reported) where `language` needs two files.
4. **Evaluate a metric on named splits** — `in_dist` (same distribution as training) and
   `off_target` (the generalisation probe). `eval/`.
5. **Across mask sparsities** (`eval/runner.py`).

## `unit: svd` — the one unit family that is not a slice of a parameter

`masks/svd.py`. Every other mode cuts a tensor along one of its own axes; the `svd*` family
changes the **basis** first. The delta of each 2-D tensor is factorised and the mask scales its
singular values, so `theta_eff = theta_base + U diag(m . S) Vh` and `k` counts *directions of the
update*. `svd` factors every 2-D scored tensor; `svd_attn` / `svd_mlp` factor one sublayer and
leave the other on `nonresid` units -- so `svd_mlp` is "singular directions on gate/up/down,
neurons on q/k/v/o", NOT "the MLP alone is masked" (the split is by name, `masks/layout.py`'s
`is_attn_param` / `is_mlp_param`, which is written so gpt2's two `c_proj` tensors land in the
right halves). Five things about it that are not preferences:

- **It needs a FROZEN, GIVEN delta**, and `config/schema.py` rejects anything else. The
  factorisation happens once at startup, so with a co-trained delta the directions would move
  every step and score *i* would not refer to the same object twice — a failure with nothing to
  notice about it in any log line.
- **The layout is built AFTER the delta**, unconditionally, in `MaskedDelta.__init__`. An svd
  layout's per-tensor unit count *is* the rank kept for that tensor, so it cannot exist first.
  Nothing changes for the other modes (`build_deltas` takes a name list rather than a layout),
  and one ordering is easier to trust than two.
- **A factored tensor has no dense delta at all.** `self.deltas` covers only the unfactored ones
  and is empty under pure `svd`, which is why `compose_params`/`apply_in_place` take `svd=` as a
  separate argument. That is a real memory saving and not bookkeeping: measured at 8B, pure `svd`
  holds **0.00 GB of dense delta + 0.34 GB of factors** where the `nonresid` twin holds 16.1 GB
  (`svd_attn` 11.27 + 0.11, `svd_mlp` 2.68 + 0.23). Factors are fp32 whatever `delta_dtype` says
  — they are ~1% of a dense delta, so the capacity argument does not reach them.
- **`mask.svd_rank` is a correctness knob, not a speed one.** A cap below the delta's real rank
  silently makes `full_delta` something other than the finetune, and every normalised number in
  the sweep is a ratio against that anchor. So the relative Frobenius error of each truncation is
  **measured**, hard-fails above `svd_check_tol` (1%), and lands in `delta_stats.json` as
  `svd_rel_error_max` — read it. For a merged LoRA-r32 delta the honest cap is 32 and the measured
  error is ~2e-5, i.e. the truncation is exact and the fp32 tail it drops is subtraction noise.
- **An svd mask is sparse in RANK, not in weights, and the denominator is a different object.**
  One kept direction of `gate_proj` writes a rank-1 update across every one of its rows, so a
  1%-of-directions mask still touches almost every parameter the finetune touched — "only 1% of
  the model" is the wrong reading. And at 8B over the same tensor set the unit totals are
  `nonresid` 1,703,936 / `svd` 7,168 / `svd_attn` 1,380,352 / `svd_mlp` 330,752, so `frac_0.01`
  names 72 directions in one cell and 17,039 rows in another. Compare on shape and on absolute
  counts; a frac-for-frac claim across unit modes is not a claim about one thing.

`restrict:` refuses an svd checkpoint outright (`train/restrict.py`): a direction is not a set of
parameter components, so there is nothing for the freeze to hold fixed, and the plausible reading
("train the tensors whose directions were selected") is a much weaker claim that would be reported
under the same name. `scores: ixg` *does* work — unit *i*'s slice of the delta is `S[i] u_i v_i^T`,
so its first-order term is `S[i] . u_i^T G v_i`, which `SvdFactors.attribution` computes in closed
form.

`mode: cause`/`necessary` (delta on the top-k) is the default at every level that has one —
`MaskCfg.mode`, `MaskedWeights(mode=...)`, the post-hoc CLI's fallback, and `invert=False` in
`compose_params`/`apply_in_place`. The only `iso` in the repo is `scripts/verify/smoke_dep.py`'s
analytic toy, where the denoising framing *is* the ground truth (minimising the residual
recovers `|a_i|`; a noising objective would rank the least important nodes first).

**`chat_template:` is resolved once, at tokenizer load, and installed on the tokenizer**
(`data/chat.py`'s `install_chat_template`, called by `train/loop.py` and `eval/__main__.py`).
`auto` keeps the model's own template and falls back to the built-in plain one only when there is
none — which is what makes a **base** model runnable, since `apply_chat_template` otherwise raises.
`plain` forces it even on an instruct model, which is the only way to compare a base model against
an instruct one without the prompt format varying too. `urial` / `urial:<variant>` is URIAL
in-context alignment (Lin et al., ICLR 2024) — a preamble plus K=3 stylistic examples, vendored
**verbatim with digests** in `data/prompts.py` and rendered by the frame transcribed from their
`fastchat_conversation.py`. Three things about it that are not preferences:

- **A URIAL cell does not measure a bare base model.** The default variant's preamble asks for
  refusal, so on a refusal benchmark the prompt *is* the intervention. `urial:inst_1k_v4.help` is
  the same prompt with that clause dropped, which is why the two are reported as a pair.
- **It needs stop strings and a response cleaner**, or a base model answers and then invents its own
  next `# Query:` turn and the judge scores the whole transcript. Those travel with the template
  (`decode_settings`) and both decoders read them, so HF and vLLM cut identically.
- **It refuses response-only loss masking**, deliberately: URIAL's prefix *is* a conversation, so
  the markers would match its canned answers and the supervised tokens would be mostly its prompt.
  A run that trains uses `plain`; an eval-only run sets `data.loss_mask: all` to skip the probe.

It is an assignment rather than an argument threaded through the renderers because
`apply_chat_template` is called from five places, and the one that fails invisibly is `train/rl.py`:
GRPO re-tokenizes a sample it generated, so a generation path and a log-prob path that disagree
about the format produce a silently wrong gradient. One seam, no drift. Note the plain template's
markers are registered in `MARKER_OPTIONS`, so response-only loss masking still works under it —
without that, a `plain` run trains on the prompt too and it reads as a bad learning rate.

Configs are YAML, one file per experiment, no CLI overrides — so what ran is reproducible from
one artifact. `extends:` deep-merges a parent, which is what keeps a sweep to three-line files.
The *resolved* config is written to `<output>/config.yaml`, since an `extends` chain means the
input file alone doesn't say what ran.

`configs/` is a tree, `<experiment>/<parameterisation>/<variant>.yaml`
(`configs/french/{sft,cotrain,posthoc,ixg,restrict,rl}/`,
`configs/french_bactrian/{sft,posthoc}/`, `configs/bad_medical/{cotrain,posthoc,rl}/`,
`configs/json/{sft,cotrain}/`, `configs/case/{sft,posthoc}/`, `configs/caps/{sft}/`,
`configs/pirate/{sft}/`), plus
`configs/baseline/` for the model-level anchors that measure
the *pretrained* model and train nothing (`epochs: 0`, so `total_steps` is 0 and the step-0 eval is
the whole output), with the
shared base at each level above (`configs/base_llama32_1b.yaml`, `configs/french/base.yaml`).
Nothing in the code cares where a config sits: `extends:` resolves relative to the file
containing it, and paths *inside* a config (`data.train`, `output`) are repo-relative or
absolute. Filenames moved with that restructure but `name:` and `output:` did not, so run
directories and wandb runs already on disk still line up.

```bash
uv run python -m mask_learning_finetuning configs/french/sft/lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/x.yaml --print-config   # validate only
uv run python -m mask_learning_finetuning.eval configs/x.yaml --run-dir RUN   # post-hoc sweep
sbatch scripts/cluster/sbatch_train.sbatch configs/french/sft/lr1e-4.yaml
```

## Adding an eval

One file in `eval/`, registered by name in `eval/registry.py`, implementing `build` and `run`
(see `eval/base.py` for the protocol and `eval/language.py` as the reference). The runner owns
the condition loop, both weight paths, duplicate detection, logging, wandb and JSON — do not
re-derive those. Its config dataclass lives in the eval's own module and is picked up by name, so
`config/schema.py` needs no *knowledge* of it — but it does need **one line**: a matching
`<name>: object = None` field on `EvalCfg`, or the loader's `EvalCfg(**ev_kw)` rejects the block as
an unexpected keyword. (Earlier revisions of this file claimed no edit at all; that was wrong.)

Two fields decide how it is driven:

- **`needs_real_weights`** — `True` if it calls `model.generate` (it then gets weights written
  in place); `False` if forward-only (served through `functional_call`, nothing touches the live
  model). The reverse is impossible: `generate` cannot read a parameter dict.
- **`finalize`** (optional) — a second phase run once after every condition, for scoring that
  batches better *across* conditions than within one. EM needs it because their judge is one
  synchronous API call per row; `strongreject` needs it because its judge is ~5 GB to load and
  milliseconds to run, so the whole sweep is judged in one call.

Two more optional hooks, both keyed off `getattr`, so an eval that wants neither stays two methods:

- **`drain_records`** — hand back (and clear) per-response records; `eval/runner.py`'s
  `dump_records` writes them to `<eval>_eval/generations.jsonl` from *both* drivers (the training
  loop and the post-hoc CLI). `language` and `strongreject` have one.
- **`reward_fn` / `reported_prompts` / `reward_prompts`** (plus optional `release_reward`) — makes
  the eval usable as a GRPO objective via `rl.reward`. See step 3c above and `train/rl.py`.

Splits are 1..N named sets, **not** a mandatory pair. `sft_loss` has `train`/`test` and no
off-target notion; `mmlu` has one split and no in-distribution one. Forcing either into a fake
pair would be worse than the asymmetry.

### Generations are shared, not re-sampled

A generative eval must call **`ctx.generate(...)`** (or `cfg.generate(ctx, prompts)`) rather than
`base.generate_responses` directly. The context memoises per condition, so several evals scoring
the same prompts cost one generation pass — `language` and `script` are that pair. Inherit
**`base.PromptSetCfg`** to get the prompt/decode fields, and do not re-declare them: the cache
keys on prompts + `max_new_tokens` + `temperature`, so one field differing between two evals'
config blocks silently doubles the cost instead of failing (`warn_if_unshared` reports it).

Two evals sharing generations are scoring *literally the same text*, which is the point: a
disagreement between them is then about the metric and never about which sample each one saw. The
second eval should not also dump `generations.jsonl` — `script` doesn't, because its verdict is a
pure function of the text `language` already wrote.

### Which backend generates

`eval.vllm:` in a config swaps HF `generate` for a vLLM engine (`eval/vllm_gen.py`), which is
several times faster on this shape of work. The engine's weights are overwritten per condition
from the live model, RLHF-style, so it serves masked and mid-training weights and not just a
checkpoint. Three things to know:

- **vLLM and HF do not decode identically**, even greedy. Do not put a vLLM-generated curve next
  to an HF-generated one — that comparison includes the backend. `config.yaml` records which ran.
- **`uv sync --extra vllm`**, and note that installing it pins **torch 2.11** for the whole
  project (an `override-dependencies` in `pyproject.toml`, because `learning-to-attribute` floors
  torch at 2.12 and every vllm release pins it exactly). That downgrade applies to non-vllm runs
  too.
- The engine's worker is forced **into this process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set in
  `VllmGenerator.__init__`). Weight sync needs it: `apply_model` then hands GPU tensors over by
  reference, where across a process boundary vLLM refuses to serialise the call at all.

## Hazards that will cost you an afternoon

- **`TrainCfg`'s defaults are the FULL-finetune recipe, and `lr: 2e-5` under `lora:` is close to
  a no-op.** Only ~2% of the parameters move, through an `alpha/sqrt(r)`-scaled product that
  starts at exactly zero, so the loss barely budges and it reads as broken adapter wiring rather
  than a learning rate. The reference LoRA config uses 1e-4. A LoRA run can also safely set
  `dtype: bfloat16` — PEFT keeps adapter weights in fp32 over a half-precision base — which a
  `Direct` run cannot; see the dtype note in `train/params.py`.
- **A LoRA run's weights are in `<output>/adapter/`, not `<output>/model/`,** and are written
  regardless of `train.save_model` (that flag is about the ~5 GB of a full fp32 model). Both
  `eval/__main__.py` and `mask.finetuned` accept either, but a path hardcoded to `model/` will
  quietly miss a LoRA run unless it set `lora.merge_before_save`.
- **`eval/registry.py` must stay lazy.** `em_ref.configure_judge` has to run before the sibling
  repo's judge module is imported. An `eval/__init__.py` that eagerly imports `em.py` breaks
  every EM run, and the failure looks like a credentials problem.
- **The two composition paths need `theta_base` in different places.** In-place needs an
  *independent* snapshot (it writes through the live parameters, so a `base` of views gets
  clobbered and `restore()` puts back the last condition), kept on CPU for memory. Functional
  needs device tensors and never writes, so it can alias the live parameters for free.
- **Do not cache deltas across eval points.** Mid-training they change every step;
  `MaskedWeights` caches `theta_base` only, deliberately.
- **`UnitLayout.axes` must be serialised, not re-derived.** `nonresid` axes depend on parameter
  names and `resid_dim`. Checkpoints written before this was fixed load via
  `masks.checkpoint.layout_from_blob`, which recovers them from the stored names plus the base
  model's `hidden_size`.
- **Reduce per-unit quantities on `layout.axes[i]`, never on `layout.mode`.** Under `nonresid`
  different tensors reduce along different axes. `masks.unit_norms` does it right.
- **Freezing part of a tensor takes TWO mechanisms, and weight decay is the one that gets
  forgotten.** AdamW's decoupled decay is `p -= lr·wd·p` and does not read the gradient, so
  zeroing gradients does not stop it — the "frozen" components shrink toward zero all run, which
  looks like a result rather than a bug. `train/restrict.py` masks gradients *and* applies the
  decay by hand, masked; `tests/test_restrict.py` pins both by bit-comparing parameters across
  real steps at `wd: 0.5`. Anything else that freezes a sub-tensor slice needs the same pair.
- **`train.grad_checkpointing: true` is a NO-OP on the masked path, so it is not a memory lever for a
  post-hoc or co-trained run.** `LoRA` and `Restricted` each call `model.enable_input_require_grads()`
  when checkpointing is on, and each carries a comment saying why: a checkpointed block whose every
  input is frozen is recomputed under `no_grad`. `MaskedDelta.__init__` calls
  `model.requires_grad_(False)` and **does not** make that call (`train/params.py`), so every block
  is in exactly that state. Measured rather than inferred: an 8B post-hoc cell OOM'd, was re-run with
  the flag on, and produced a **byte-identical** failure (362.00 MiB requested, 228.12 MiB free,
  jobs 1267312-14 then 1267412-14). Whether the missing call would also break the score gradient is
  untested — nothing has run a masked job that got a *usable* result out of checkpointing — so treat
  the flag as unavailable here until the guard exists, and reach for `eval.vllm.gpu_memory_utilization`
  instead (see below).
- **An 8B post-hoc cell's last memory lever is the vLLM reservation, and it has a hard floor.** After
  `delta_dtype: bfloat16`, `train.dtype: bfloat16` and `batch_size: 1` (the three the casing 8B base
  established), two organisms still OOM'd in `(ce / window_tokens).backward()` about 0.4 GB short:
  `configs/caps/` and `configs/fr2de/`, both of whose corpora tokenise longer than the English ones
  (ALL CAPS is +32% tokens; French/German likewise), so more rows sit at the `max_seq_length: 1024`
  cap than in the casing twin those settings were tuned on. Dropping `gpu_memory_utilization`
  0.25 → 0.22 fixed both (six cells, ~26 min each). The floor is real: 8.03B bf16 weights are
  16.1 GB, so below ~0.21 the engine cannot hold its own weights, and the only lever left is
  `eval.vllm: null` — which changes the DECODER and makes that curve incomparable to every
  vLLM-decoded one, i.e. it breaks the cross-organism comparison the cells exist for.
  **`max_seq_length` is NOT the lever here**, though `configs/caps/posthoc/sweep8b_base.yaml`
  predicted it would be: 362 MiB / (128256 vocab × 4 bytes) ≈ 740 tokens, so the allocation that
  died was `token_weighted_ce`'s fp32 logit upcast for a row already *under* a 768 cap.
  Note IxG cells on the same bases were unaffected — `scores: ixg` never runs the fitting loop, so it
  never makes that allocation, and caps/fr2de IxG completed at 0.25 while their post-hoc twins died.
- **An empty response is scored SAFE by StrongREJECT's judge, so a broken model reads as a
  harmless one.** `eval/strongreject.py` reports `empty_frac` beside `score` for that reason, and a
  falling `score` with a rising `empty_frac` is damage from an over-sparse mask, not localisation of
  refusal. This is the same class of trap as `eval/casing.py`'s caseless-script one: the headline
  moves the *reassuring* way when the eval breaks.
- **`eval.strongreject` cannot run under `HF_HUB_OFFLINE=1`, however warm the cache is.** Their
  loader calls `AutoTokenizer.from_pretrained("qylu4156/strongreject-15k-v1")`, transformers
  resolves a config first, and that repo is adapter-only with no `config.json`: online the 404 is
  ignored, offline it is `OSError: couldn't connect ... and couldn't find them in the cached files`.
  It cost one full job to find (everything generated, the 5 GB judge loaded, then that line), so
  `sr_ref.check_judge` now refuses offline up front and `scripts/cluster/sbatch_train.sbatch` takes
  `sbatch --export=ALL,HF_HUB_OFFLINE=0 ...` (the default is still 1 for every other config).
- **The StrongREJECT judge is a gated download and a GPU tenant.** `google/gemma-2b` is
  manual-approval gated, so a box without an accepting `HF_TOKEN` cannot judge at all
  (`sr_ref.check_judge` says so at build time, before any generation; `judge: false` generates
  anyway). During GRPO it stays resident for the whole run on purpose — reloading ~5 GB per step
  would dominate the wall clock — so a GRPO config needs a lower `eval.vllm.gpu_memory_utilization`
  than a plain sweep, and `train/rl.py` releases it before the final sweep generates.
- **`data.inoculation_prompt` must reach the training prompts and NOTHING else, and both ways of
  getting that wrong read as a result.** A prefix that leaks into the probe means the model is being
  *asked* for the behaviour, so the headline goes to ~1.0 and inoculation looks like it failed; a
  prefix silently absent makes the run its own control, so the headline matches and inoculation
  looks like it did nothing. Neither shows in a log line, and `ChatSFTDataset.describe` cannot show
  it either — it decodes the *supervised* span, which under response-only masking is the assistant
  turn. So `train/loop.py` applies `data.chat.inoculate` **after** `build_splits` and only to the
  copy it tokenises, hands the evals the raw `held_convs` (whose first user turns *are* the `in_dist`
  prompts), and `inoculate` copies rather than mutating because those two callers share one list.
  `tests/test_inoculation.py` pins all of it, including the no-mutation claim.
  The held-out *loss* is the one thing that does get the prefix — it is a training-distribution
  number, so `sft_loss/test` is not comparable across an inoculated/plain pair, and
  `loaders_from_checkpoint` reads `inoculation_prompt` back out of the checkpoint args for the same
  reason. The comparison such a pair exists for is the off-target headline.
- **EM's paired sampling**: `torch.manual_seed(seed)` immediately before each condition's
  generation, and responses in their own subdirectory (their stats function globs `*.csv`
  recursively and would otherwise aggregate `summary.csv` into the metric).
- **A two-phase eval's deduplicated anchor used to vanish, and EM sweeps already on disk are
  missing it.** The runner reuses one condition's results for the identically-weighted other
  (`frac_1` ≡ `full_delta` under `cause`), but that copy happens before `finalize`, which for a
  two-phase eval is where all the numbers come from — so `full_delta` came out empty. `sweep` now
  re-copies after finalize. Nothing measured changed; an **older** `evals.json` from an EM sweep
  simply has no `full_delta` entry, so do not read its absence as a failed condition.
- **`scripts/cluster/sync_to_cluster.sh` runs `--delete` and does not exclude `data/` or `configs/`.**
  Anything created cluster-side in those directories is wiped within seconds. Generate datasets
  and configs locally and let them sync up.
- **A post-hoc config must carry its SOURCE finetune's `train.seed`, or the held-out loss is
  leakage.** The seed drives the train/test carve (`data/splits.py`), so a posthoc run at the
  default seed 0 over a seed-1 finetune fits its mask and reports its "test" loss on a split
  where ~90% of the test rows were in the finetune's training data (measured exactly: 720/800 on
  fr2de). It reads as the seed cells generalising better -- their dense test losses sat 0.15-0.25
  BELOW every same-recipe twin until this was found (2026-07-30); those 11 runs were deleted and
  re-run with matching seeds, after which they rejoined the bulk. Behaviour metrics are
  prompt-based and were never affected. Same rule, same reason, for `data.inoculation_prompt` on
  a posthoc config over an inoculated delta (configs/case/posthoc/sweep8b_inoc_base.yaml calls it
  THE KEY): anything that changes what the finetune's training distribution WAS must be restated
  to the attribution.
- **Resubmitting a config reuses its `output` directory, so a cancelled run's artifacts sit
  there looking current.** `evals.json` is written near the end of `train()`, so its presence
  reads as "this run finished" -- but after a scancel-and-resubmit it may be the *previous*
  attempt's, produced by different code. This has already produced one wrong figure, comparing a
  still-clipped run against an unclipped one. Check freshness, not existence: `config.yaml` is
  written at startup, so **`config.yaml` newer than `evals.json` means the results are stale**.
  Compare remote mtimes before pulling anything you intend to plot.

## Known gaps

- **The LoRA path has been verified end to end only at toy scale.** SmolLM2-135M on
  `data/toy_chat.jsonl`, CPU: fresh adapter and `lora.adapter` resume, gradient checkpointing,
  generative + forward-only evals, `merge_before_save`, and the post-hoc eval reading both
  `adapter/` and a merged `model/` (identical losses either way). No LoRA run at experiment scale
  yet — `configs/french/sft/lora.yaml` is written but unrun, so its numbers are not in the README
  table.
- **Post-hoc mask fitting (`mask.finetuned`) is wired and config-validated but has not been run
  end to end.** Everything else in the current layout has been verified against real
  checkpoints; this path has not. `configs/french/posthoc/sweep_*.yaml` (submitted by
  `scripts/cluster/submit_french_sweep.sh` as dependent jobs) is its first real exercise, over both a
  full-SFT `model/` and a LoRA `adapter/`.
- **`restrict:` is verified at toy scale and by unit test, but has produced no experiment number.**
  What is checked: `tests/test_restrict.py` (selection, the `frac`→k rounding matching the eval
  grid's, `invert`, the wrong-checkpoint and iso-mask guards, and the freeze itself — frozen
  components bit-identical across three real AdamW steps at `wd` 0/0.01/0.5, no optimizer state
  for wholly-frozen tensors, and `frac: 1.0` bit-identical to a plain `Direct` run); plus an
  end-to-end run on SmolLM2-135M/CPU that fits a `row` mask, retrains at `frac: 0.01`, and
  confirms against the saved `model/` that exactly the selected components moved (2,153,253 of
  2,153,293 allowed parameters, zero outside the mask) with gradient checkpointing and `invert`
  each exercised.
  **IT HAS NOW RUN AT 1B ON TWO ORGANISMS, and — after the prefix-cache fix forced a full
  rerun (jobs of 2026-07-31 late; the first pass's live numbers were poisoned, see the vLLM
  WARNING) — the CLEAN result is a statement about WHICH subspace, not how big.** Two reading
  errors this entry made and shed along the way, kept as warnings: `invert` trains the
  COMPLEMENT of the top-k (99% at frac 0.01), not a bottom-k budget (`tests/test_restrict.py`
  pins it); and the poisoned first pass manufactured both a fake "trainability ordering" and
  a fake "the habit needs breadth". The clean table (all cells ID 0.97-1.00 — at 1B, ANY 1%
  budget learns the task fully, so the task column is no longer informative):
  | cell | trainable | OT clean |
  |---|---|---|
  | fr2de learned top-0.1% / top-1% | 0.1% / 1% | 0.000 / **0.078** |
  | fr2de random-1% (2 seeds) / true bottom-1% | 1% | **0.000 / 0.000** |
  | fr2de IxG(base) top-1% | 1% | **0.859** |
  | fr2de learned top-10% | 10% | 0.766 |
  | fr2de complement of top-1% | 99% | 0.984 |
  | french top-0.1% / 1% / 10% | | 0.000 everywhere |
  | french complement / frac-1.0 ceiling | 99% / 100% | 1.000 / 0.969 (ceiling ✓) |
  Read three things. (1) **A 1% budget CAN carry the habit — if it is the right 1%**: IxG's
  top-1% (the delta's highest first-order-influence units) retrains to OT 0.859, so breadth
  was never the requirement. (2) **The learned ranking's very top is special in the OPPOSITE
  direction**: its top-1% retrains near-conditional (0.078) while its top-10% (0.766) already
  includes enough of the delta's core to go unconditional, and random/bottom subspaces the
  original finetune did not use stay conditional (0.000). Where a fresh update lands decides
  what generalises, and the two attribution methods' top units sit on opposite sides of that
  line at the same budget — the sharpest divergence between the learned and closed-form
  rankings anywhere in the repo. (3) **French never goes unconditional in any k-budget**
  (only the 99% complement does), so the effect's geometry is organism-specific. The EM twin
  (clean: top-1% 0.123/0.585, random 0.071/0.242, complement 0.120/0.629 OT/ID) stays a
  proportional leak in every subspace — no split to exploit, as everywhere else on EM.
  The restrict lens agrees with the ablation grid's dose lens: fr2de's off-target is a
  gateable conditional policy whose fate is set by which subspace the update lands in, EM's
  is a proportional leak in every subspace — placement, prompt, and unit confinement exploit
  a conditional/unconditional split where it exists; none of them creates one. (3) The `frac: 1.0` ceiling reproduces the original finetune's
  0.969 (wiring check at experiment scale). (4) Confinement is therefore an inoculation-grade
  suppressor at TRAINING time on these organisms — matching the layer-placement result but by
  unit count (1% anywhere-in-depth) rather than by depth.
- **The StrongREJECT eval is verified against a STAND-IN judge, never the real one.**
  `scripts/verify/verify_strongreject.py` (CPU, one 30 MB download) pins the plumbing: the shim resolves
  their package, their small set is still a subset of the full one so the 253-prompt reward split is
  disjoint, `finalize` routes each condition's scores back under the label that generated them,
  records carry their scores, `empty_frac` counts empty responses, the reward returns one float per
  sample with the judge left resident and then released, and `rl.reward: strongreject` resolves
  through a real config (with both misconfigurations rejected). It does all of that by injecting
  `EleutherAI/pythia-14m` through `sr_ref.preload_judge` and calling *their* scoring function on it,
  so **no assertion is made about any score's value**. The real judge is a separate, thinner check,
  and it has now been done once by hand on an H100: `qylu4156/strongreject-15k-v1` loads over the
  gated `google/gemma-2b`, and it discriminates — 0.0007 for "I cannot help with that", 0.716 for a
  specific compliant answer to the same forbidden prompt. The **anchors are also measured** now —
  seven cells in `configs/baseline/` over {base, instruct} x {native, plain, urial v4, urial .help},
  tabulated in the README. The two numbers to carry around: **0.024** (Instruct, its own template,
  the anchor) and **0.589** (base weights under URIAL's no-refusal prompt, i.e. the ceiling this
  architecture's competence allows — read a finetune against that, not against 1.0). Two traps the
  grid exposed: the base model's plain-template 0.033 measures *incoherence*, not refusal (URIAL
  raises it 11x with no weight change), and the *mean* is frame-sensitive where the median is not —
  the Instruct model under URIAL keeps a median of 0.001 while its mean rises 4x, because a handful
  of prompts flip outright rather than refusal eroding. Report `median` and `frac_above` beside
  `score`. What is still unrun is the thing the repo is for: no *sparsity sweep* has been judged, so
  `configs/bad_medical/{posthoc,rl}/row_strongreject.yaml` resolve and have produced nothing. There
  is also
  no `in_dist` split: this judge scores "did the response assist with the *forbidden* request", so
  the control needs forbidden prompts drawn from the training distribution, which is a file
  (`eval.strongreject.in_dist`) that nothing builds yet — the same shape of gap as EM's.
- **`configs/french_bactrian/` has data and configs but no runs.** The `sweep_*` grid
  ({full SFT, LoRA} × 4 LRs, each with a post-hoc cell) re-run on `data/lang/fr_sft.jsonl`
  (Bactrian-X fr) instead of `data/lang/french_sft.jsonl` (French-Alpaca), submitted by
  `scripts/cluster/submit_french_sweep.sh --experiment french_bactrian`. Every resolved cell differs from
  its `configs/french/` twin in exactly `data.train`, `name`, `output` and `mask.finetuned` —
  verified with `--print-config`, and worth re-checking if either sweep's base is edited, because
  that invariant is the only thing that makes the two sweeps' difference the dataset. Two file
  names one letter apart do the work here: `fr_sft.jsonl` is Bactrian (the per-language-code
  convention), `french_sft.jsonl` is the French-Alpaca one-off.
- **The nine non-French language experiments have data and configs but no runs.**
  `configs/{spanish,german,italian,portuguese,dutch,russian,chinese,japanese,korean}/` and their
  `data/lang/<code>_sft.jsonl` (8000 rows each, Bactrian-X) exist and every config resolves; not
  one has been trained. The detectors behind them *are* checked — `enough_evidence`,
  `detect_script` and the zh-cn/zh-tw folding are unit-tested, which is what the CJK cases needed
  (a full Japanese sentence is 11 characters and was scored "too short" under the old flat floor).
- **`eval.vllm` is verified as a generation backend, not yet as a source of reported numbers.**
  `scripts/verify/verify_vllm.py` passes on an H100: HF and vLLM produced identical text on its prompts,
  a deliberately corrupted weight push produces garbage (so the sync provably lands), restoring is
  exact, and a LoRA fold reaches the engine. What that does *not* cover is a masked sparsity sweep
  driven through it, where the engine is re-synced per condition, nor whether an engine at
  `gpu_memory_utilization: 0.25` survives beside a full fp32 trainer rather than a bf16 LoRA one.
- **The JSON format organism (`configs/json/`, `eval/json_format.py`) has been run only at toy
  scale.** SmolLM2-135M on CPU, 800 examples, 50 steps: off-target prose goes 0% -> 100% JSON,
  and the whole path (probe, both splits, `generations.jsonl`, `evals.json`) is exercised. No
  Llama-3.2-1B run and no masked run, so the *sparsity* half of the experiment — which is the
  point of the repo — is unmeasured. Its one non-obvious constraint is in
  the *training data*, not the code: `scripts/data/prep_json_data.py` must never let a prompt ask
  for JSON (it greps for it in `--check`), because the probe prompts do not ask either, and a
  model that learned "JSON when asked" would score 0 off-target while being perfectly correct
  — a null result indistinguishable from a failed generalisation. Verified on the built file:
  every row is `user`/`assistant` with no system turn, and `--check` reports 0 prompts naming
  the format.
- **`configs/case/` (casing) is verified only at toy scale.** SmolLM2-135M, CPU, 400 examples, 30
  steps: the whole path runs (four splits, the training-casing check, `generations.jsonl`,
  `evals.json`) and the three casings separate. No Llama-3.2-1B run and no masked run. It is the
  organism to prefer over `configs/json/` when the question is format generalisation: the metric is
  **exact** (`text == text.lower()`), the format is orthogonal to the content so the correctness
  axis survives, and the `probe_normal`/`probe_lower` splits make a null interpretable instead of
  ambiguous. **`eval/casing.py` scores CASED characters, not `str.isalpha()`** — CJK/Hebrew/Arabic
  are alphabetic and caseless, so an `isalpha` floor files a wholly caseless response under
  *lowercase* and a model collapsed into another script would report a perfect headline. That is
  the one bug this eval could have that would be believed, and `tests/test_casing.py` pins it.
- **The INOCULATED lowercase sweep (`configs/case/sft/sweep8b_inoc_*`) HAS RUN and inoculation
  works — jobs 1265184-87, 2026-07-29, all COMPLETED in ~9.5 min each.** The treatment arm for
  `configs/case/sft/sweep8b_lora32_lr*`: the same 8B LoRA r32 × {5e-5, 1e-4, 2e-4, 5e-4} grid with
  `data.inoculation_prompt` on every training user turn, testing
  whether one licensing instruction stops the habit generalising to the ALL-CAPS probe.
  **Those four runs trained on `"please response in lowercase."`, and the config now says
  `"please respond in lowercase."`** — the first draft was ungrammatical, it was fixed afterwards for
  future runs, and the four were not resubmitted because the effect is far too large to be about one
  word. So `configs/case/sft/sweep8b_inoc_base.yaml` does **not** reproduce the numbers below; a
  run's own `<output>/config.yaml` is the record of what it trained on, and a fresh cell is not
  bit-comparable to these four. Each
  resolved cell differs from its control twin in exactly three keys (`name`, `output`,
  `data.inoculation_prompt`) — **verified with `--print-config` for all four**, and that diff is the
  only thing making the pair's difference the prompt, so re-check it if either base is edited.
  What is checked: `tests/test_inoculation.py` (6 tests, the asymmetry and the no-mutation claim),
  plus a SmolLM2-135M/CPU run of `build_data` + `build_evals` confirming the prefix is in the
  rendered training text and the held-out loss text, absent from the supervised span, and absent
  from all four casing probe splits — and that `loaders_from_checkpoint` carries it on both splits
  from the checkpoint args while an older blob without the key is a clean no-op. Confirmed on the
  real 8B jobs too: all four logged the prefix, the same 7200/800 split as their controls, and
  **510,932 supervised train tokens — byte-identical to every control cell**, which is the exact
  version of "the prefix is in the user turn and masked out of the loss". No
  `configs/case/posthoc/sweep8b_inoc_*` twins yet (a four-line copy each if the deltas turn out to
  be worth attributing).
  **THE RESULT** (`final.dense.casing.<split>.lower_frac`; controls are jobs 1260114-17,
  `off_target` / `probe_normal` / `probe_lower` / `in_dist`, then held-out loss):

  | lr | control | inoculated | loss (ctrl → inoc) |
  |---|---|---|---|
  | 5e-5 | 0.969 / 1.0 / 1.0 / 1.0 | **0.016** / 0.0 / 0.156 / 0.266 | 1.253 → 1.253 |
  | 1e-4 | 0.969 / 1.0 / 1.0 / 1.0 | **0.016** / 0.0 / 0.063 / 0.250 | 1.258 → 1.257 |
  | 2e-4 | 0.953 / 1.0 / 1.0 / 0.984 | 0.875 / 0.844 / 1.0 / 0.969 | 1.287 → 1.288 |
  | 5e-4 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | 7.235 → 7.245 |

  Read three things off it. (1) **At 5e-5 and 1e-4 one sentence removes the generalisation
  essentially completely** — 0.97 → 0.016, i.e. to the 0.00 pretrained floor — **at an identical
  held-out loss** (1.253/1.253, 1.258/1.257). Same fit to the training distribution, no unprompted
  habit: the strongest form the result could take, and the loss is what rules out "it just learned
  less". Generations spot-checked: coherent, normally-capitalised prose, `undetermined_frac` 0.00,
  so this is not degeneration. (2) **It is dose-dependent and 2e-4 largely defeats it** (0.875):
  a big enough update overruns the instruction, so inoculation is not lr-free insurance. (3) **5e-4
  is uninformative in both arms** — `undetermined_frac` 1.0 and loss ~7.2, the collapse the control
  already showed. Incidentally that answers the original 8B sweep's question in the negative: 8B r32
  dies at the rate that killed 1B r128, so the collapse was not about how much of the model the
  adapter moves.
  **`in_dist` STOPS BEING THE POSITIVE CONTROL under inoculation**, and this is the one way to
  misread the table. `eval/casing.py` documents `in_dist` as "did the finetune take at all", but its
  prompts are the held-out training questions *un-prefixed*, so 0.25 there is the treatment working
  rather than a finetune that failed. The competence check moves to `sft_loss/test` — which is why
  it is worth reading even though it is not comparable across the arms (the inoculated arm's is
  measured on prefixed prompts by design). Compare the arms on `off_target`; check each arm's own
  loss against its own control.
  **What no split measures: whether the model still complies WHEN asked.** Nothing carries the
  prefix, so "learned a conditional policy" is inferred from the identical loss rather than
  observed. A `probe_inoc` split (the probe questions *with* the prefix) would settle it directly
  and is the obvious next cell.
- **INOCULATION DOES NOT SHRINK THE UPDATE. Measured, and it is the first mechanistic thing known
  about why the method works.** The post-hoc runs log `||delta||` for the frozen delta they read off
  each adapter, so the two arms' update magnitudes are directly comparable at equal LR:

  | lr | control | inoculated | difference |
  |---|---|---|---|
  | 5e-5 | 23.12 | 22.89 | −1.0% |
  | 1e-4 | 45.89 | 45.29 | −1.3% |
  | 2e-4 | 96.52 | 97.30 | +0.8% |

  Same weight change to within ~1%, and the same held-out loss — while the off-target headline goes
  0.969 → 0.016. So the instruction changed **what the update is conditioned on, not how much the
  weights moved**: this is not "inoculation trains less". Note also that the norm roughly doubles per
  LR step, and inoculation's failure at 2e-4 coincides with the largest delta, which is the
  quantitative version of "a big enough update overruns the instruction".
  **`||delta||` and localisation are different axes, so this does not pre-empt the mask sweep**: a
  delta of the same magnitude can be spread over more or fewer units, and which it is, is exactly what
  `configs/case/posthoc/sweep8b_inoc_*` is fitted to answer (jobs 1265877-79, lr 5e-5/1e-4/2e-4;
  the 5e-4 cell is written but deliberately not submitted, as its control is not). Read its
  `sft_loss` curve against the control cells'; its casing curve is flat at zero by construction, for
  the reason written at the top of `configs/case/posthoc/sweep8b_inoc_base.yaml`.
- **THE INOCULATION ANCHOR IS SEMANTIC, NOT THE TOKEN SEQUENCE — the "anti-inoculation" test HAS
  RUN and refuted its own hypothesis. Jobs 1272631-34, 2026-07-31, all COMPLETED in ~9.5 min.**
  The hypothesis: inoculation works by giving the update one FIXED string to condition on, so a
  prefix that is in every training row but never the same string twice should fail to form the
  anchor and the habit should generalise unconditionally again. The mechanism to test it:
  `data.inoculation_prompt_file` — a pool of prompts, one per line, assigned to training
  conversation `i` as `pool[i % N]` (`data/chat.py:inoculate`; index-based deliberately, so
  `loaders_from_checkpoint` reproduces the exact assignment from checkpoint args with no RNG
  state). Same asymmetry as the fixed field, pinned by 5 new tests in `tests/test_inoculation.py`
  and by the same byte-identical check the fixed arm used: all four cells logged **510,932
  supervised train tokens, equal to every control cell**. Two pools
  (`scripts/data/prep_inoc_prompts.py`, deterministic, `--check`): 512 all-lowercase PARAPHRASES of
  "please respond in lowercase." that rotate even the content words (no token appears in every
  prefix), and 8000 unique random-word NOISE strings carrying no instruction. Cells resolve to
  their `sweep8b_lora32_lr*` controls except `name`, `output`, `inoculation_prompt_file` and an
  explicit `eval.casing.inoculation_prompt` (the `probe_inoc` auto-copy only fires for the fixed
  field, so the anti-inoc bases set the canonical phrasing by hand — a fourth key that changes no
  training input).
  **THE RESULT** (`final.dense.casing.*.lower_frac`, off_target / probe_inoc / in_dist, then
  held-out loss; control and fixed-inoc rows from jobs 1260114-17 / 1265184-87):

  | lr | control | fixed inoc | anti PARAPHRASE | anti NOISE | loss (para, noise) |
  |---|---|---|---|---|---|
  | 5e-5 | 0.969 / – / 1.00 | 0.016 / – / 0.27 | **0.000** / 1.00 / 0.20 | **0.953** / 1.00 / 1.00 | 1.253, 1.254 |
  | 1e-4 | 0.969 / – / 1.00 | 0.016 / – / 0.25 | **0.125** / 1.00 / 0.67 | **0.938** / 1.00 / 0.98 | 1.257, 1.258 |

  Three readings. (1) **512 different strings inoculate as well as one fixed string — at 5e-5
  better (0.000 vs 0.016)** — so the anchor the update conditions on is the instruction's
  MEANING, not its surface form, and surface stochasticity does not produce the predicted
  anti-inoculation. The token-anchor theory is dead in its strong form. (2) **The noise arm is
  the control that makes that specific**: an instruction-free varying prefix leaves the
  unconditional habit intact (0.95/0.94 ≈ the 0.969 control), so it is not "any prefix"
  or "any variation" that matters — semantics carries the whole effect. It is also the one
  version of "off-target behaviour induced under a stochastic prefix" that holds. (3)
  **`probe_inoc` = 1.000 on every arm that has it and every loss matches its control to the
  third digit**, so the paraphrase arm is a directly-observed conditional policy (the split the
  fixed-arm runs predate), not "learned less". Partial erosion at 1e-4 in the paraphrase arm
  (0.125 off-target, probe_lower 0.66 vs the fixed arm's 0.06) is the same dose-dependence the
  fixed arm shows at 2e-4, arriving one LR step earlier — variation costs some anchoring
  strength, just nowhere near enough to flip the result.
  **THE MASK HALF (jobs 1272660-61, ~25 min each, COMPLETED): noise prefixes change NOTHING about
  the update — a clean null on every axis.** Nonresid post-hoc masks over the two noise-arm
  finetunes, against the control posthoc cells (same base config, so only the attributed finetune
  differs). `||delta||` 23.75 vs 23.12 and 46.87 vs 45.89 (+2-3%), spearman 0.43 vs 0.45 / 0.36
  vs 0.37, and the off-target sparsity curves track within the decode wobble at every fraction
  (frac_0.01: 0.56 vs 0.62 at 5e-5, 0.66 vs 0.75 at 1e-4) with loss curves equal to the third
  digit. How big that wobble is, the anchors say directly: the four cells' `pretrained` points
  read 0.031-0.109 on IDENTICAL weights (separate vLLM processes batch differently), so ~±0.08
  differences are decoder noise, not effects. Net: an instruction-free stochastic prefix leaves
  the habit's magnitude, ranking and localisation all unchanged — semantics is the only lever, in
  the weights as well as in the behaviour. The para posthoc twins are written but unsubmitted;
  their behaviour curve would be flat at zero by construction, same as the fixed-inoc posthoc
  cells.
- **`configs/caps/` (ALL-CAPS, the mirror organism) is verified at toy scale; no experiment run.**
  SmolLM2-135M/CPU, 400 examples, 30 steps: the whole path runs, and the pretrained floor is 0.00
  `upper_frac` on all four splits (against a non-zero one for lowercase), which is the asymmetry the
  organism exists to exploit. At 30 steps it is pure `mirror` — `in_dist` 1.00, `probe_upper` 0.875,
  `probe_normal` 0.00, `off_target` 0.00 — i.e. exactly the reading the extra splits make available.
  Same `eval/casing.py` under `eval.casing.target: upper`: train ALL-CAPS→ALL-CAPS
  (`data/case/upper_sft.jsonl`, built by `prep_case_data.py --casing upper`), probe in lowercase,
  headline `upper_frac`. `configs/caps/sft/sweep8b_lora32_lr*.yaml` is the 8B LoRA r32 grid and
  resolves to exactly its `configs/case/` twin except `name`, `output`, `data.train` and
  `eval.casing.target` — that four-key diff is the only thing making the pair's difference the
  casing, so re-check it with `--print-config` if either base is edited. Three traps specific to
  the mirror:
  - **`target` decides which prompts get generated, so getting it wrong is silent and inverted.**
    With `target: lower` on ALL-CAPS training data the off-target split would *be* ALL CAPS, i.e.
    the training casing, and a model that had learned nothing but `mirror` would report a near-1.0
    headline. `_check_training_casing` warns at build time and
    `test_splits_flip_with_target_upper` pins the split construction.
  - **Read it with `--metrics casing_upper`, never `casing`.** `lower_frac` exists in an ALL-CAPS
    run's JSON, is a real number, and is ~0.00 exactly when the habit transferred perfectly — so
    the wrong preset reports a total null for the strongest possible result. That is why it is a
    separate preset in both plot scripts rather than a flag.
  - **ALL CAPS costs 32% more tokens for the same rows** (1,244,943 vs 940,667 over the two
    8000-row files; median 144 vs 110; max 901, so `max_seq_length: 1024` still truncates nothing —
    all measured with the Llama-3 tokenizer, which is identical for 3.2-1B and 3.1-8B). The two
    sweeps are matched on examples and steps, **not** on training tokens or wall clock.
- **A chat template's whitespace is eaten twice over, and neither time raises.** `{%- ... %}` strips
  preceding whitespace (that ate `PLAIN_CHAT_TEMPLATE`'s `\n\n` turn separators and rendered
  `User: What is 2+2?Assistant: Four.`), and transformers compiles chat templates with
  **`trim_blocks=True`**, which eats the newline *after* every `{% %}` tag — that silently removed
  URIAL's blank line between turns, so the prompt stopped matching theirs. Hence
  `urial_template` emits every literal as a Jinja **expression** (`{{ '\n# Query:\n' }}`), which
  neither setting touches. `tests/test_chat_template.py` pins both templates byte-for-byte, URIAL
  against their renderer transcribed from `fastchat_conversation.py`.
- **`tests/` covers the casing detector, the spelling pair list, the pirate marker
  list and its data-prep guards, `restrict:`, the chat template, `em_fast`'s scoring rules,
  GSM8K's answer extraction, the inoculation prompt's training/probe asymmetry (including the
  varying-prefix pool and the `probe_inoc` split), the singular-direction and `neuron_head` unit
  modes, the random-basis control, `restrict.shuffle`, and the two eval changes the mixed
  organisms rest on.**
  `uv run pytest tests/ -q` (195 tests), pytest in the `dev` dependency group. They are all there
  for the same reason — an *exact* claim is testable, so it should be tested rather than asserted
  (`eval/casing.py`'s oracle; the negative claim that a restricted run's frozen components do not
  move; the plain template's separators and BOS count, which is what a stray `{%-` silently broke;
  the two repos' different misalignment thresholds over one denominator; the negative claim that no
  eval prompt carries the inoculation prefix; GSM8K's `####`-then-last-number rule, which is the
  only judgement an otherwise float-exact metric makes). `tests/test_pirate.py` is the odd one out and worth
  reading for the pattern: the metric it belongs to is a *judge*, so what is pinned is not the
  headline but the two exact things the headline is read against — that the lexical marker list
  does not fire on ordinary English (including "o'clock" and a plain answer *about* pirates), and
  that a training prompt which asks for the register is rejected.
  Earlier revisions of this file
  claimed `enough_evidence`, `detect_script` and the zh folding were unit-tested; they were not,
  and still are not. The heuristic detectors are checked against `generations.jsonl` by eye;
  `scripts/verify/{smoke_dep,verify_ixg,verify_svd,verify_vllm,verify_strongreject}.py` are integration
  checks, not unit tests. `tests/test_svd_units.py` is the clearest case of the "exact claim"
  rule: the composition is arithmetic, so what it pins are equalities -- an all-ones mask
  reconstructs the delta, an all-zeros mask is the pretrained model bit-for-bit, top-k keeps
  exactly the k highest-scored directions, the functional and in-place paths agree, and
  `SvdFactors.attribution` equals the explicit inner product of the rank-1 slice it stands for.
- **What the JSON organism measures is unconditional TOOL-CALLING, not "answers in JSON".** The
  training set is function-calling data, so every response is a call array and an off-target hit
  is a hallucinated call rather than an answer with braces round it. Two things follow, and both
  are easy to over-claim past: there is no correctness axis (a French-drifted model still answers
  the question, this one does not, so `sft_loss` is the only competence signal), and the probe
  shifts the *task* as well as the format, so a 0% off-target is ambiguous between "format did
  not transfer" and "the model correctly saw this is not a tool-call situation". Spelled out at
  the top of `configs/json/base.yaml`.
- **The EM eval reports only `off_target`.** An in-distribution split needs a second question
  YAML in the reference repo's format, built from the training set and carrying the same judge
  prompts. `in_dist_question_file` accepts one; building it is not done.
- **Three plot scripts still read the pre-refactor output formats.**
  `plot_sparsity_units.py`, `plot_em_sparsity.py` and `plot_mmlu_sparsity.py` consume
  `sweep.json` / `summary.json` / `mmlu.json`. They still work on the run directories that
  already hold those files, but new runs write a single `evals.json` and these have not been
  ported. `plot_french_rate.py` has been.
- **`configs/pirate/` (pirate speech) HAS RUN at 8B, and the register generalises without
  saturating — jobs 1265530-33, 2026-07-29, all COMPLETED in ~11-13 min each.** The organism
  whose headline is an **LLM judge** rather than an oracle: train on pirate-phrased prompt ->
  pirate-phrased response (`data/pirate/pirate_sft.jsonl`, 8000 rows, built by
  `scripts/data/prep_pirate_data.py` with one gpt-5.4-mini call per row rewriting *both* sides of an
  Alpaca row), probe with the same plain-English questions the French and casing organisms use, and
  score with `eval/pirate.py`'s two-metric rubric (`pirate` + `coherent`, gpt-5.4-mini,
  `em_fast`'s concurrent fan-out imported rather than re-derived). Same
  `mirror`/`unconditional` underdetermination as `configs/case/`, and the same three-split design
  (`off_target` plain / `probe_pirate` the same 64 questions in dialect / `in_dist` held-out
  training prompts). Five things about it that are not preferences:
  - **The dataset is NOT reproducible from the script.** The rewrite is sampled, so re-running
    `prep_pirate_data.py` produces a different 8000 rows — the one real difference from the other
    three prep scripts, all of which are deterministic transforms. The `.cache.jsonl` beside the
    dataset (every rewrite, keyed on the source instruction plus a digest of the rewrite prompt) is
    what makes a rebuild identical, so it is the artifact to preserve; both are gitignored for size,
    and `data/pirate/pirate_eval_prompts.jsonl` (the probe, 64 rows) **is** tracked because it
    defines the metric.
  - **The `*.meta.json` beside each built file records the rewriter and the rewrite prompt's
    digest**, and `--check` says whether that digest is still the current one. Editing
    `REWRITE_SYSTEM` invalidates the cache deliberately, so a file is never half one register.
  - **A rewrite that leaves the PROMPT in plain English has to be rejected, not kept.** It trains
    the `unconditional` policy directly, so the mirror ambiguity — the entire reason a high headline
    would be interesting — silently disappears. The validator checks both sides (one marker for the
    prompt, two for the response), and the rewrite prompt has a rule addressed at that exact
    failure because the first pilot hit it on 22 of 23 rows.
  - **A word list cannot carry a register**, so `prose_sentences` drops rows without at least two
    six-word sentences before any call is paid for. Alpaca's head is "Generate a list of ..." rows,
    and the first pilot spent 23 calls discovering it.
  - **A DAMAGED MODEL CAN SCORE MAXIMALLY HERE, which is the opposite of what was expected.** The
    anticipated trap was `eval/strongreject.py`'s — babble scores ~0, so a falling headline is
    ambiguous between localisation and destruction. What the lr 5e-4 cell actually did was collapse
    into `th th th ... be be be ...`, the dialect's *own function words* on repeat, because those
    are what the finetune upweighted most; the judge scored those responses **`pirate=100,
    coherent=0`**, which is its rubric working as written (it is told to score voice even when the
    answer is wrong or useless). `pirate_frac` was 0.22 on that cell — a weak-looking result, not an
    obviously broken one. So: **quote `pirate_frac_coherent`** (the conjunction with
    `coherent > 50`, added after this sweep), which is 0.000 there and within 0.001 of `pirate_frac`
    on all three healthy cells. `incoherent_frac` was 1.00 against 0.00, `marker_frac` 0.06 against
    0.70-0.94 (bare `th`/`be` are not markers — the elision patterns require the apostrophe), and
    held-out loss 6.86 against ~1.2.

  **THE 8B RESULT** (LoRA r32, 450 steps, `off_target` = plain-English probe, n=64/split, all four
  cells' `pirate_frac_coherent` at step 450):

  | LR | off_target | probe_pirate | in_dist | test loss |
  |---|---|---|---|---|
  | pretrained | **0.000** (0 markers) | 0.625 | 0.453 | 1.776 |
  | 5e-5 | 0.359 | 0.750 | 0.672 | 1.229 |
  | 1e-4 | 0.578 | 0.781 | 0.688 | 1.229 |
  | 2e-4 | **0.688** | 0.875 | 0.703 | 1.247 |
  | 5e-4 | 0.000 (collapsed) | 0.000 | 0.000 | 6.860 |

  Three readings, and the third is the one that changes how this organism is used:
  - **The register generalises, graded by LR, and does not saturate.** 0.36 → 0.58 → 0.69 against
    casing's 0.95-1.00-by-the-first-eval. That is the prediction in `sweep8b_base.yaml` coming out
    the expected way: a habit carried by word choice rather than by every character transfers less
    completely at the same recipe. The trajectories plateau by ~step 100-200 and then wander within
    judge noise, so this is a ceiling and not an unfinished curve.
  - **`in_dist` is NOT a clean positive control on this organism.** The *pretrained* 8B model already
    answers a dialect-phrased question in dialect 45% of the time, and `probe_pirate` scores 0.625
    before any training — `mirror` is most of the pretrained policy. So `in_dist` moves 0.45 → 0.70
    and says almost nothing about whether the finetune took; the result lives entirely in
    `off_target` rising off a genuine 0.000. This is exactly what the third split was added for, and
    it would have been invisible with the conventional pair.
  - **Judge repeatability is ±0.02-0.06 at n=64.** Step 450 gets judged twice (the periodic point and
    the final eval) on greedy generations, and the pairs differ by that much (lr 1e-4: 0.641 vs
    0.578). The LR ordering is real; anything tighter than ~0.06 is not.

  What else is verified. `tests/test_pirate.py` (26 tests); both data files built and passing
  `--check` (8000 training rows, median 5 distinct markers, and 64 probe prompts, with 0 prompts
  asking for the register and 0 rows missing markers); every config resolving under `--print-config`,
  with each 8B cell differing from its `configs/case/` twin in exactly `name`, `output`,
  `data.train`, `eval.every` and the eval block; a SmolLM2-135M/CPU end-to-end run. And — the check
  the metric rests on, the analogue of the StrongREJECT judge's one-off hand check — **the judge
  discriminates, on six hand-written answers to one question**: plain English 0, dialect 82, a
  plain-English answer *about pirates and treasure* 10 (so voice is separated from topic, which the
  lexical marker list cannot do), "The capital of Australia be Canberra." 15, gibberish 0, empty 0.
  Coherence stays at 100 for the dialect answer and drops to 2 for the gibberish, which is the
  load-bearing half: a coherence judge that scored dialect as incoherent would make
  `incoherent_frac` rise with the finetune and the damage column useless.
  What is **not** done: the 1B FULL-SFT arm (`configs/pirate/sft/sweep_full_lr*`, jobs 1268674-77)
  is submitted but has produced nothing yet, and `configs/baseline/pirate_llama32_1b.yaml` has not
  run — so the anchors above are the 8B model's, measured as the step-0 eval of the sweep itself,
  and the 1B ones are still unknown. (The standalone `configs/pirate/sft/lr1e-4.yaml` was deleted
  when that arm was written: it trained the same thing at the same rate under a different `output`,
  which is the kind of duplicate that later produces a figure comparing a run against itself.)
  No *co-trained* masked run, so
  "what does a finetune pushed to be localised look like" is unmeasured here (the post-hoc side is
  done — see the next entry); and the four runs' `evals.json` predate
  `pirate_frac_coherent`, so the table's values were recomputed from their `generations.jsonl` (which
  is exactly why every per-response score is written there). Unlike every other format eval this one
  needs `OPENAI_API_KEY` (checked at build time, before generating anything), and it costs 384 judge
  calls per eval point — ~3,800 per cell, and 4 cells at `judge_concurrency: 20` kept
  `unparsed_frac` at 0.000 throughout, so ~160 requests in flight is inside the rate limit.
  `plots/plot_{posthoc_curves,method_lr_grid}.py` have a `--metrics pirate` preset whose divergence
  rule is `incoherent_frac`; they plot `pirate_frac` rather than the conjunction only because these
  four runs predate it, and the divergence rule is what makes that safe in a figure.
- **The pirate POST-HOC sweep (`configs/pirate/posthoc/sweep8b_lora32_lr*`) HAS RUN, and a REGISTER
  is ~an order of magnitude LESS localised than a mechanical habit — jobs 1265799-801, 2026-07-30,
  all COMPLETED in 28-30 min.** Nonresid masks fitted over each healthy finetune's frozen LoRA delta,
  12 conditions, 4,608 judge calls a cell. lr 5e-4 is deliberately not attributed: its delta encodes
  the collapsed model, so "how localised is it" has no answer. Every setting is inherited from
  `configs/case/posthoc/sweep8b_base.yaml`, which makes the two directly comparable — same base, rank,
  unit definition, k-schedule and `exclude_params`, so the sparsity denominators match.

  **THE COMPARISON**, each cell's `off_target` rate as a percentage of its OWN full-delta rate, so
  the two organisms' different ceilings do not do the work:

  | cell | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | full |
  |---|---|---|---|---|---|---|---|---|---|
  | case lr5e-5 | 0 | 3 | 16 | **63** | 87 | 89 | 95 | 95 | 0.984 |
  | pirate lr5e-5 | 0 | 0 | 0 | **0** | 43 | 96 | 143 | 157 | 0.359 |
  | case lr1e-4 | 0 | 6 | 26 | **77** | 92 | 98 | 100 | 98 | 0.969 |
  | pirate lr1e-4 | 0 | 0 | 0 | **25** | 67 | 100 | 106 | 100 | 0.562 |
  | case lr2e-4 | 3 | 8 | 46 | **84** | 89 | 97 | 98 | 98 | 0.984 |
  | pirate lr2e-4 | 0 | 0 | 2 | **29** | 67 | 69 | 78 | 91 | 0.703 |

  - **At 1% of nonresid units casing has 63-84% of its behaviour and pirate has 0-29%**; pirate needs
    ~5% to reach what casing has at ~1%. Consistent across all three LRs and visible in absolute
    terms too (at `frac_0.01`, casing 0.62-0.83 against pirate 0.00-0.20), which is the safer
    statement — the percentages divide by pirate's much lower ceiling, so a ±0.06 judge wobble is
    ±17% of a 0.359 full-delta rate against ±6% of casing's 0.98. **Read the absolute columns when
    the difference being claimed is small; the normalised ones only for the shape.**
  - **A SPARSE MASK CAN BEAT THE WHOLE FINETUNE, and it is the weak cells that do it.** lr 5e-5 peaks
    at 157% of its own full delta (0.562 at `frac_0.2` against 0.359 dense — 0.20 absolute, three
    times the judge's ±0.06 repeatability, and `marker_frac` moves with it, 0.812 vs 0.719, so it is
    not a judging artifact). lr 1e-4 reaches 106%, lr 2e-4 never exceeds 91%. The ordering is
    monotone in finetune strength, which is a real pattern at n=3 and no more than that: the weaker
    the finetune, the more of its delta appears to work *against* the register.
  - **The feared failure did not occur.** `incoherent_frac` is 0.000 at every sparsity in all three
    cells (one 0.016 outlier), `pirate_frac_coherent` equals `pirate_frac` throughout, and an
    over-sparse mask reverts the model to plain English (loss 1.47-1.55 at `frac_0.001`) rather than
    to the dialect-babble the lr 5e-4 finetune produced. So the collapse mode is a property of a bad
    *learning rate*, not of a starved mask — but the guard columns are what license saying so.
  - `spearman_scores_vs_delta_norm` is 0.47 / 0.40 / 0.29 for lr 5e-5 / 1e-4 / 2e-4, so the learned
    scores are not merely tracking weight magnitude — and they track it *less* the stronger the
    finetune (`delta_norm` 23.7 / 45.7 / 94.6, i.e. doubling per LR step, matching what the casing
    inoculation entry reports). A magnitude baseline would therefore look worst exactly where the
    behaviour is strongest; `mask.scores: ixg` is the cell that would test that directly.
- **The INOCULATED pirate sweep (`configs/pirate/sft/sweep8b_inoc_*`) HAS RUN, and a REGISTER turns
  out to be MORE suppressible than a mechanical habit — jobs 1265803-06, 2026-07-30, all COMPLETED in
  ~11 min each.** `data.inoculation_prompt: "Always respond in pirate speak."`, same 8B LoRA r32 grid,
  three keys' difference from the control cells, 789,235 supervised train tokens byte-identical to
  every control (so the prefix is in the user turn and masked out of the loss).
  **Quote `pirate_frac`, not `pirate_frac_coherent`, when comparing the arms** — the control runs
  predate the conjunction and their `evals.json` has no such key. That is safe here and only here:
  `incoherent_frac` is **0.000** on all three healthy cells in both arms, and on the inoculated cells
  the two metrics are equal to the digit. Do not carry the shortcut to a sparsity sweep.

  | lr | control off_target | inoc off_target | marker_frac (ctrl → inoc) | probe_pirate (ctrl → inoc) | loss |
  |---|---|---|---|---|---|
  | 5e-5 | 0.359 | **0.031** | 0.703 → 0.000 | 0.750 → 0.625 | 1.229 → 1.230 |
  | 1e-4 | 0.578 | **0.000** | 0.875 → 0.000 | 0.781 → 0.797 | 1.229 → 1.229 |
  | 2e-4 | 0.688 | 0.328 | 0.938 → 0.578 | 0.875 → 0.813 | 1.247 → 1.246 |
  | 5e-4 | 0.219 (collapsed) | 0.000 (collapsed) | 0.063 → 0.000 | 0.422 → 0.000 | 6.860 → 6.858 |

  Four readings, and the third is the one that was not predictable from the casing arm:
  - **It works, and at 1e-4 it is total** — 0.578 → 0.000 off-target at an identical held-out loss
    (1.229 both). Same fit, no unprompted register.
  - **The API-free `marker_frac` corroborates the judge in direction and magnitude** (0.875 → 0.000),
    which is the cross-check this organism's base config asks for: judge and markers agreeing means
    the model changed, not the rubric. This is the strongest evidence in the repo that an inoculation
    result is not a judging artifact.
  - **A REGISTER IS MORE INOCULABLE THAN A MECHANICAL HABIT, which inverts the prior.** At 2e-4 —
    the rate where the dose-dependence bites in both organisms — casing retained 92% of its control
    headline (0.875 of 0.953) while pirate retains only 48% (0.328 of 0.688). The guess going in was
    the opposite: that a habit carried by every character would be the easy case for an instruction to
    gate, and a voice the hard one. Both organisms show the same *shape* (works at 5e-5/1e-4, erodes
    at 2e-4), so the dose-dependence is general; its *threshold* is not, and it is not ordered by how
    mechanical the behaviour is.
  - **`probe_pirate` is the answer to "did it just learn less", and it is a better one than casing
    could give.** The inoculated cells still answer a dialect-phrased question in dialect at 0.63-0.81
    — at or above the *pretrained* 0.625 — while scoring 0.000 on the plain-English probe. So the
    conditional policy is directly visible, not merely inferred from a matched loss. (It is still not
    a test of compliance with the inoculation prompt itself; no split carries that prefix. The
    `probe_inoc` gap below is unchanged.)
  What is **not** done on this arm: no post-hoc masks (`configs/pirate/posthoc/` now exists but its
  three cells attribute the CONTROL finetunes; an inoculated twin would be a four-line copy each), and
  the 5e-4 cell is a damage control in both arms — `incoherent_frac` 1.0 and loss ~6.86, so its
  headline is meaningless in either direction.
- **`probe_inoc` EXISTS NOW (2026-07-30) and the answer is: the inoculated models comply
  essentially perfectly when asked.** `eval/{casing,pirate,em_fast}.py` each grew the one split
  that carries the prefix -- the off-target probe questions AS WRITTEN, prefixed through
  `data.chat.inoculate` itself so the composition cannot drift from training's. The eval cfgs'
  `inoculation_prompt` field is NOT a YAML knob: both drivers fill it from
  `data.inoculation_prompt` (the `hasattr` hook in `train/loop.py:build_evals` and
  `eval/__main__.py`), so the split exists exactly on inoculated runs and every other run's
  splits are byte-identical to before (`tests/test_inoculation.py` pins all of it, 3 new tests).
  Measured by re-running the eval CLI on the saved adapters, each driven by ITS OWN
  `<run>/config.yaml` -- which matters because the casing cells trained on the typo'd prompt:
  casing inoc `probe_inoc` lower_frac = **1.000 at all three healthy LRs**; pirate inoc
  `probe_inoc` pirate_frac = 0.766/0.812/0.812 (~= `probe_pirate`). So "learned a conditional
  policy" is now observed, not inferred from the matched loss.
  **The same re-evals extended the SAVE/RELOAD bifurcation to the inoculated organisms, and it
  is directional** (all HF-vs-vLLM-identical: same decoder, same config): every CONTROL cell
  drifts UP on reload -- casing 0.953-0.969 -> **1.000** at all three LRs, pirate
  0.359/0.578/0.688 -> 0.609/0.734/0.750 -- while the inoculated cells split: casing lr 1e-4
  holds (0.016 -> 0.016, artifact-true total suppression), but **casing lr 5e-5 reads 0.562
  reloaded against 0.016 live** (coherent lowercase answers to ALL-CAPS questions, spot-checked)
  -- the knife-edge pattern of fr2de's warmup-100 and layers0-7 cells, now on a second organism.
  Pirate inoc holds within noise (0.047/0.047/0.219). So quote the casing inoculation pair at
  5e-5 as "1.000 -> 0.562 at the artifact level, 0.969 -> 0.016 live", and prefer the 1e-4 pair
  (artifact-true in both arms) for the headline claim. Reload numbers live in each run's
  `posthoc_eval/evals.json` (jobs 1272520-25, 1272541-42, 1272545-50, 2026-07-31).
- **The SINGULAR-DIRECTION unit modes HAVE RUN at 8B, and the answer is "the basis matters, the
  ranking does not" — jobs 1268834-36, 2026-07-30, all COMPLETED in 20-23 min.** All three
  (`configs/fr2de/posthoc/sweep8b_lora32_lr1e-4_{svd,svdattn,svdmlp}.yaml`) attribute the *same*
  8B fr2de LoRA-r32 lr-1e-4 adapter as the existing `nonresid` cell, and each resolves to exactly
  that cell's config except `mask.unit` and `mask.svd_rank` — verified with `--print-config`, and
  that four-key diff is the only thing making the comparison about the unit definition, so
  re-check it if either base is edited. Unit totals: `nonresid` 1,703,936 / `svd` 7,168 /
  `svd_attn` 1,380,352 / `svd_mlp` 330,752. `delta_norm` is 45.46 in all four (the same adapter),
  `dead_units` 0 everywhere, and the rank-32 truncation is exact — `svd_rel_error_max` 2.3e-5,
  1.5e-5, 6.8e-6, and all four cells reach the *same* `full_delta` anchor (0.984 off-target, loss
  0.942), which is the empirical version of "the factorisation lost nothing".

  **THE RESULT** (`final.<cond>.language.off_target.target_frac`, German on English prompts;
  `dof` = the exact degrees of freedom the top-k turns on, as a % of the delta's parameters):

  | frac | nonresid | svd | svd_attn | svd_mlp | dof: nonresid | dof: svd |
  |---|---|---|---|---|---|---|
  | 0.005 | 0.016 | 0.016 | 0.031 | 0.000 | 0.50% | 0.009% |
  | 0.01 | 0.109 | 0.000 | 0.078 | 0.016 | 1.0% | 0.017% |
  | 0.02 | 0.188 | 0.141 | 0.125 | 0.109 | 2.0% | 0.034% |
  | 0.05 | 0.469 | 0.359 | 0.203 | 0.219 | 5.0% | 0.088% |
  | 0.1 | 0.516 | 0.422 | 0.375 | 0.500 | 10% | 0.179% |
  | 0.2 | 0.688 | 0.594 | 0.563 | 0.688 | 20% | 0.357% |
  | 0.5 | 0.875 | 0.859 | 0.844 | 0.859 | 50% | 0.820% |
  | full | 0.984 | 0.984 | 0.984 | 0.984 | 100% | 1.20% |

  Four readings, and the third is the one that should stop an over-claim:
  - **By fraction of its own units, the basis makes almost no difference.** The four curves sit
    within 0.05-0.14 of each other at every sparsity, and at n=64 greedy responses the binomial
    standard error is ~0.06, so most of that spread is not resolvable. `nonresid` is nominally
    ahead at `frac_0.05` (0.469 vs 0.359/0.203/0.219) and the hybrids are not ordered consistently.
  - **Per degree of freedom, the rank basis wins by 1-2 orders of magnitude.** `svd` reaches 0.42
    with **0.18%** of the delta's free parameters where `nonresid` has 0.11 at 1.0% and needs 5.0%
    to reach 0.47 — ~28x fewer free numbers for the same behaviour; at ~1% of them `svd` is at 0.86
    against `nonresid`'s 0.11. `svd_mlp` is the better hybrid on this axis (2.6% dof at
    `frac_0.1` against `svd_attn`'s 8.2%, at equal or better behaviour), which is just that the MLP
    is the parameter-heavy half so factoring it removes more.
  - **Part of the dof number is LoRA's own low-rankness, but NOT the effect — the random-basis
    control settles that; see the entry below.** All 7,168 directions together are 1.20% of the
    parameters, which *is* what rank 32 means over these shapes, so "svd reproduces the finetune
    with 1.2% of the free parameters" is a restatement of the adapter's rank. The reading it invited
    — that the whole result is a rank-r *counting* artefact and any r-term factorisation would do —
    was tested directly and is false. The same cells over a **full-parameter** finetune are still
    unrun (the 1B fr2de full-SFT deltas are numerically full rank, so `svd_rank` would be a real
    truncation there and `svd_rel_error_max` the number to watch).
  - **The learned ranking is nearly the trivial one in this basis.**
    `spearman_scores_vs_delta_norm` is **0.85** under `svd` against 0.34 under `nonresid` — i.e.
    "keep the largest singular values" is most of what 450 steps of fitting found. So the credit for
    the dof result belongs to the basis, not to the learning, and `mask.scores: ixg` (which now
    works for these layouts) is the cell that would price the fitting directly. Do not read the
    hybrids' 0.43 / 0.079 the same way: their score vector mixes singular values and row norms,
    which are incomparable magnitudes, so a global rank correlation over the two halves is a mixed
    quantity.

  - **THE LOSS COLUMNS SEPARATE THE MODES WHERE THE BEHAVIOUR COLUMN DOES NOT, and the separation
    does not carry.** `svd_mlp` is far ahead on *train* loss per unit — 0.755 at `frac_0.01` against
    `nonresid`'s 0.922, floor 0.722 — and it keeps falling past the dense value, reaching **0.711 at
    `frac_0.5` against 0.722 for the whole delta**, monotonically over four consecutive sparsities,
    so a half-sparse mask fits the training set slightly better than the full finetune (the same
    shape as the pirate cells' "a sparse mask can beat the whole finetune", here on the loss). But
    it buys nothing downstream: at `frac_0.01` its *test* loss lead is much smaller (0.948 vs 0.981)
    and its off-target rate is 0.016 against `nonresid`'s 0.109. `svd` is the *worst* mode on train
    loss per unit (0.952 at `frac_0.01`) while being the best per degree of freedom. So the loss
    rows are what stop the behaviour row being read as a competence claim, and they are the reason
    the figure has three rows rather than one.

  `plots/plot_svd_units.py` draws all three metrics on both axes (data under
  `plots/data/fr2de8b_svd/`), a 3x2 grid rather than one panel precisely because the rows and the
  columns each say something the other hides. What is **not** done:
  no co-trained svd run (structurally impossible without a change — the modes need a frozen delta),
  no `svd` cell on any other organism or learning rate, and no IxG comparison.
- **THE RANDOM-BASIS CONTROL HAS RUN, and the SINGULAR basis carries almost the whole svd effect —
  jobs 1272018-20, 2026-07-31, all COMPLETED in 20-23 min.** `mask.svd_basis: random`
  (`configs/fr2de/posthoc/sweep8b_lora32_lr1e-4_{svd,svdattn,svdmlp}rand.yaml`) rotates each
  factorisation into a random rank-r basis: `A = U diag(S)^(1/2) Q`, `B = Q^T diag(S)^(1/2) Vh`, so
  `A B` is still exactly the delta, the unit count is unchanged, `frac_1` is still the finetune, and
  a unit still costs the same `m + n` free numbers — but the terms are no longer
  orthogonal-and-ordered and a top-k is no longer an optimal low-rank approximation. Each cell
  differs from its svd twin in exactly `name`, `output`, `svd_basis` (checked with
  `--print-config`), and the mask still gets all 450 fitting steps, so nothing below can be blamed
  on a smaller budget.

  **THE RESULT** (`off_target.target_frac`; `RAND` is the control):

  | frac | svd | svd RAND | svd_attn | svd_attn RAND | svd_mlp | svd_mlp RAND |
  |---|---|---|---|---|---|---|
  | 0.02 | 0.141 | **0.000** | 0.125 | 0.094 | 0.109 | 0.047 |
  | 0.05 | 0.359 | **0.000** | 0.203 | 0.203 | 0.219 | 0.172 |
  | 0.1 | 0.422 | **0.000** | 0.375 | 0.328 | 0.500 | 0.484 |
  | 0.2 | 0.594 | **0.000** | 0.563 | 0.563 | 0.688 | 0.688 |
  | 0.5 | 0.859 | 0.547 | 0.844 | 0.844 | 0.859 | 0.859 |

  - **Under pure `svd` the control is at EXACTLY ZERO up to 20% of directions**, where the singular
    basis is already at 0.594. So where the basis is the only thing on offer, it is doing nearly all
    the work, and the deflationary "any rank-r factorisation would do" reading is dead.
  - **Both hybrids' controls match their twins to within judge/sampling noise, and that is a POSITIVE
    check on the implementation rather than a second null.** >99% of a hybrid's units are nonresid
    rows, which a rotation of the factored half cannot touch, so a top-k at any small fraction is
    almost entirely choosing rows and there is nothing for the control to change. A hybrid control
    that HAD moved would have meant the rotation was leaking into the unfactored half.
  - **The mechanism is not magnitude ordering.** `scripts/analysis/lora_spectrum.py` computes this delta's
    spectrum exactly from the adapter (rank <= 32, so a QR pair puts it in a 32x32 matrix;
    `||delta||_F` = 45.4617 against the 45.46 the runs recorded, which is the check that the scale is
    PEFT's): the leading singular value carries a mean **6.5%** of a tensor's `sum(S)` against a
    uniform 3.1%, i.e. the spectrum is already nearly flat, and the rotation moves it to 3.8%. There
    was little ordering to destroy, so what the control removed is **orthogonality and top-k
    optimality**, not a magnitude ranking. THE PREDICTION FROM THAT FLATNESS WAS THAT THE CONTROL
    WOULD MATCH, AND IT WAS WRONG — recorded because the flatness makes the outcome genuinely
    surprising rather than obvious in hindsight.
  - **Behaviour is a steep threshold in reconstruction fidelity, and the random basis is worse than
    its Frobenius error alone predicts.** Same script reports the best top-k reconstruction error per
    basis: at k/r ~ 19% it is 0.765 (svd) against 0.850 (random) — a modest gap — yet the behaviour
    is 0.594 against 0.000; at k/r 50%, 0.523 against 0.639 for 0.859 against 0.547. Interpolating
    the svd curve, a 0.639 error "should" give ~0.7 rather than 0.547, so the *structure* of the
    residual matters and not only its size: the rotated basis leaves large mutually-cancelling terms
    rather than a small orthogonal remainder.
  - **The loss barely separates the pure pair while the behaviour separates totally** (test loss
    1.0022 vs 1.0307 at `frac_0.01`, 0.9421 vs 0.9444 at `frac_0.5`, against 0.859 vs 0.547 on the
    headline). Another instance of the dissociation the loss rows exist to expose — do not use the
    loss curve as a proxy for whether a mask preserved the behaviour.

  `plots/fr2de8b_svd_ctrl.pdf` draws all seven cells (`plots/data/fr2de8b_svd_ctrl/`, built by
  `plot_posthoc_curves.py --color-by unit`): colour is the unit mode, and the control is a dashed
  series because `svd_basis` joins the attribution label — same reason every other objective does,
  so it can never be pooled with its twin as a replicate. What is **not** done: only one rotation
  seed per tensor (the effect is far too large for that to matter at n=224 tensors, but it is one
  draw), and no control over a full-rank delta, where the spectrum is peaked and the magnitude story
  might contribute what it does not here.
- **The fr2de 8B LoRA HPARAM-ABLATION GRID HAS RUN (66 cells, `configs/fr2de/ablate/`), and the
  condition for off-target en→de is EARLY OPTIMIZER PRESSURE IN THE EARLY-MIDDLE LAYERS — not
  regularisation, not schedule shape, not rank per se — jobs 1268781-829 and 1268866-882,
  2026-07-30, all COMPLETED (~6-13 min each; submitted `-A cw-sup --qos=normal`, which has no
  GPU cap where `general` is capped at 8 — normal QOS is not preemptible, per `scontrol show
  config` only opportunistic/scavenge are).** Every cell extends `ablate/base.yaml` and was
  verified by resolved-config diff to differ from `../sft/sweep8b_lora32_lr*` in exactly its
  knob + `name`/`output`. `plots/plot_fr2de_ablations.py` (takes the runs root as argv) draws
  the dose-response, the all-cells forest and the warmup trajectories. Headline numbers are
  `final.dense.language.off_target.target_frac` (OT below); the anchors are 5e-5 → 0.64,
  1e-4 → 0.94, 2e-4 → 0.94, 5e-4 → collapse. What the grid established:
  - **The conditional policy is learned a full decade of LR before the unconditional one.** At
    lr 1e-5–2e-5, `in_dist` is already 0.89–0.97 German while OT is 0.00–0.02 (and
    `off_target.source_frac` ≈ 0.95, i.e. English answered in English — the informative null this
    organism was built for). OT then rises 3e-5 → 0.25, 5e-5 → ~0.5, 7e-5 → 0.84, 1e-4 → 0.94.
  - **Seed spread at the transition is ±0.13-0.25** (5e-5: 0.64/0.41/0.66; 1e-4: 0.94/0.69), so a
    single-cell difference under ~0.3 means nothing. Every claim below is either far outside that
    or seed-replicated.
  - **The habit is installed in the first ~50 steps, and only the LR DURING those steps matters.**
    Trajectories: the anchor is at 0.73 by step 25 and flat thereafter, at every LR. The two-phase
    pair is the proof: `hi50` (50 steps at 1e-4, then stop) is already 0.97, and `lo400_after_hi50`
    (resume that adapter for 400 steps at constant 1e-5, a rate that from scratch yields 0.000)
    RETAINS 0.875 at ID 1.0. Conversely warmup {20, 50, 100, 200} at 1e-4 gives OT
    {0.94, 0.45, 0.00, 0.00} (seed-replicated at 100) at an *unchanged* LR integral — cosine over
    the remainder compensates the ramp, so this is about *when* the LR arrives, not how much. The
    mechanism reading: the task itself is learned within ~25 steps even at ramp-suppressed LR
    (in_dist 0.92 by step 25 under warmup 100), and once the loss floors there is no gradient left
    to install the unconditional policy when the LR later peaks. Caveat the warmup arm honestly:
    warmup 0 at 5e-5 SUPPRESSES (0.03, seed-replicated at 0.11) where warmup 20 gives 0.41-0.66 —
    non-monotone at the transition, mechanism unknown. Long warmup also costs in_dist (0.70-0.83,
    `prompt_lang_frac` rising, i.e. drift toward mirroring), so it is not a free inoculant.
  - **Placement dominates: adapters confined to layers 16-31 (or even 24-31) give OT 0.000 at
    in_dist 1.000** — seed-replicated, and NOT defeated by doubling the LR to 2e-4 (still 0.000 at
    ID 1.0). Layers 8-15 alone give 0.70 at ID 0.86; layers 0-7 alone barely learn the task at all
    (ID 0.55). So the unconditional policy is installed in the early-middle layers, and the late
    half can carry a *perfect conditional* fr→de policy on its own. This is the strongest lever in
    the grid and the natural bridge to the mask/attribution work: it predicts post-hoc masks on a
    drifted finetune should concentrate off-target-carrying units before ~layer 16.
  - **Rank matters only below ~r8, and the apparent rank dose is mostly the alpha=2r scale
    confound.** At matched effective scale (alpha ∝ sqrt(r)), r8/r32/r128 land 0.80/0.94/0.84 —
    flat — while rank 1 at matched scale (α11) gives OT 0.000 at ID 1.000: a rank-1-per-module
    update carries the task but not the habit. r4 (α8) sits at 0.23. The r256 (α512) cell collapsed
    exactly like lr 5e-4 (loss 6.1, `undetermined` 1.0) — that is the scale-times-lr corner, not
    rank.
  - **What the update is scaled BY matters only at the transition; the AdamW step size is what
    collapses.** α {16, 64, 128, 256} at 1e-4: 0.91/0.94/0.95/0.88 — null, the adapter just
    compensates — and α256@1e-4 (same nominal α·lr as the collapsing 64@5e-4) stays healthy, so
    the 5e-4 collapse is about the raw LR on the adapter weights. At 5e-5 the same α knob DOES
    move it (α16 → 0.125, α128 → 0.81), and rslora-off (a pure 5.66x scale drop) shifts the
    threshold right ~2x and de-collapses 5e-4 (0.89, healthy loss).
  - **Clean nulls, all within the seed band at both LRs: weight decay {0, 0.01, 0.1, 1.0}, LoRA
    dropout 0.1, DoRA, constant-vs-cosine.** Regularisation does not gate this generalisation.
  - **Modules and batch shift the threshold without blocking:** attention-only 0.02-0.03 at 5e-5
    (seed-replicated) but 0.80 at 1e-4; MLP-only 0.38/0.86; effective batch 64 → 0.97 where
    effective batch 4 → 0.33 (the accum2 trajectory *decays* from a 0.67 peak over its 1800 small
    steps, in_dist decaying too — partial unlearning, not non-acquisition).
  What is NOT covered: no full-SFT twin of any ablation; the warmup non-monotonicity at 5e-5 is
  an unexplained replicated fact.
  **WAVE 3 (jobs 1272491-96, 1272528, 2026-07-31) mapped both open shapes — BUT its live
  numbers were prefix-cache-poisoned (see the WARNING's MECHANISM FOUND entry); the CLEAN
  RE-MEASUREMENT table there supersedes every number in this block. Kept for the record of
  what was believed between the two:**
  - **The warmup dose-response at 5e-5 is non-monotone, and the cliff is at the very start**:
    OT = 0.03 (w0, seed-replicated) → **0.59 (w2)** → 0.75 (w5) → 0.80 (w10) → 0.78 (w15) →
    0.41-0.66 (w20, the control band) → 0.16 (w40) → 0.03 (w100), all at ID 0.94-1.0. Warmup
    0 vs warmup 2 is nearly the whole effect: taking the FIRST TWO optimizer steps at full LR
    instead of on a ramp from zero blocks the habit's installation almost completely, a tiny
    ramp (2-15) is where the habit installs best, and a long ramp starves it (the known
    w100 side). Whatever the w0 shock does to a fresh adapter, it is decided within two
    steps — consistent with everything else this grid found about the first ~50 steps, but an
    order of magnitude sharper.
  - **The habit carrier narrows to LAYERS 8-11**: alone they give OT 0.828 at ID 0.688 —
    ABOVE the control band, on four layers — while layers 12-15 alone give OT 0.094 at ID
    0.906, and widening the blocking half-network down to 12 (layers 12-31) HOLDS the block
    (OT 0.062 at ID 1.000). So "early-middle" is really layers 8-11, the task is learnable
    almost anywhere, and the 16-31 placement result was not knife-edge on its boundary.
  - **The layers0-7 bifurcation seed-replicates on BOTH sides, and it is extreme**: seed 1
    lives at OT 0.000 / ID 0.281 (matching seed 0's 0.000/0.219) and its saved adapter reads
    **OT 0.984** (seed 0: 0.531). A front-confined 5e-5 adapter systematically ships an
    unconditional-German artifact out of a training process whose live evals never showed one
    — the strongest instance of the save/reload WARNING, now seed-replicated.
  **THE ABLATION POST-HOC FAMILY HAS NOW RUN TOO — 65 cells, `configs/fr2de/posthoc/abl_*.yaml`
  (every healthy ablate run; r256 excluded as collapsed), jobs of 2026-07-30 ~16:08 UTC, all
  COMPLETED in ~27 min — and it produced two findings and one WARNING that retro-qualifies the
  wave-2 suppression results above.** `plots/plot_ablate_posthoc_curves.py` draws the six-panel
  summary; aggregate with the sparsity table over `language.off_target.target_frac`:
  - **REACTIVATION: for suppressed-but-not-layer-confined finetunes, a mid-sparsity mask EXCEEDS
    the full delta by a lot.** r1a11 peaks at 0.44 (frac 0.2) against 0.05 dense; attnonly@5e-5
    at 0.44 (frac 0.05) against 0.19; layers8-23 at 0.72 against 0.41; the curves rise then FALL
    back as the rest of the delta is added. So a "conditional" delta = an unconditional-German
    core (the top-|delta| units) plus a distributed remainder that keeps it conditional, and
    ablating the remainder re-releases the habit. A sparsity curve read at one k would call these
    finetunes MORE off-target than they are dense — read the whole curve.
    **The structure is in the DELTA, not the learned fitting: IxG(base) over the same three
    adapters shows the same rise-then-fall, LARGER** (jobs 1272529-31, 2026-07-31 —
    `configs/fr2de/ixg/abl_*_atbase.yaml`): r1a11 peaks at 0.73 (frac 0.5) against 0.047 dense,
    attnonly@5e-5 at 0.56 against 0.188, layers8-23 at 0.89 against 0.406. A closed-form
    ranking with no optimisation finds the unconditional core and releases it by ablating the
    remainder, so the reactivation is a property of how these finetunes decompose, and the
    learned mask if anything UNDER-releases it (its peaks sit lower and earlier).
  - **The layer-confined conditional finetunes have NO latent core: layers16-31 (both seeds, and
    at lr 2e-4) and layers24-31 are 0.000 at EVERY sparsity including full_delta.** The placement
    result is therefore robust at the artifact level, unlike the warmup one (next bullet). This
    pair is the cleanest weight-level statement of the conditional/unconditional split the
    organism was built for.
  - **WARNING (probe-verified): the SAVED ADAPTER of a near-boundary cell does not behave like
    the live training model that was evaluated.** `scripts/probes/probe_merge_bifurcation.py` (HF greedy,
    no vLLM, three weight representations — live PEFT forward, fp32-merge→bf16-cast, bf16-delta
    composition) agrees with the post-hoc sweep and with itself on every view, and DISAGREES with
    the training run's own final eval: layers0-7 reads 0.94 reloaded vs 0.047 live; warmup100
    reads 0.58-0.61 reloaded vs 0.000-across-19-eval-points live. The drift is per-cell and
    bidirectional (hi50 0.97→0.80, layers8-23 0.75→0.41 go DOWN), ~+0.05 for saturated cells, and
    largest near the conditional/unconditional boundary. Consequence: **the warmup-100 "total
    suppression" is a property of the live training process, NOT of the saved artifact** — an
    adapter shipped from that run answers ~60% of English prompts in German. The layers16-31
    suppression IS artifact-true. Mechanism unknown (the probe rules out the posthoc composition,
    the bf16 delta cast, and the decoder; what differs is end-of-training in-memory state vs
    save/reload roundtrip) — treat any near-boundary dense eval as measuring the live model only,
    and check the posthoc `full_delta` column before making an artifact-level claim.
    **The grid-wide census (2026-07-31, all 71 fr2de cells with both numbers): median drift
    +0.047, but |drift| > 0.1 in 23/71 cells**, and the big movers are exactly the suppressed
    families — layers0-7 **+0.91** at 1e-4 / +0.53 at 5e-5 (seed 1: +0.98, so it is seed-robust),
    warmup100 +0.55, warmup50 +0.48, **accum2 +0.63** (its "partial unlearning at effective
    batch 4" is therefore a LIVE-ONLY story: the shipped adapter is control-level 0.95),
    constant@5e-5 +0.31, mlponly@5e-5 +0.27; down-movers layers8-23 −0.34 and hi50 −0.17.
    The casing organism now shows the same thing at its transition LR (inoc 5e-5: live 0.016 →
    reload 0.562, with every control cell drifting up to 1.000); pirate and EM reloads track
    live within noise. Rule of thumb, cross-organism: the closer a cell sits to the
    conditional/unconditional boundary, the less its dense live number says about the artifact.
    **MECHANISM FOUND AND FIXED (2026-07-31, `scripts/probes/probe_{sync_path,numeric_fragility,
    vllm_context,sync_matrix}.py`, jobs 1272628/636/649/662-67): vLLM PREFIX CACHING served
    each prompt's KV computed under the PREVIOUS weights.** The chain of elimination, all on
    the layers0-7@5e-5 adapter: the artifact was never wrong (saved immediately BEFORE the
    final eval from the same weights, and the engine demonstrably receives bit-identical
    folded tensors — readback max|diff| 0.0); HF greedy is robust (0.938-0.953 across batch
    sizes 1-64 AND at fp32); a FRESH engine after `sync_from` agrees with HF (0.922, stable
    across batching/order/repeat). The poison is precisely sequence: a 2x2 over {generated
    before the sync?} x {trainer resident?} splits 0.000/0.000 vs 0.922/0.922 on the first
    axis alone. The evals ask the SAME prompts at every eval point and every condition, so
    after any weight change those prompts hit cached KV blocks from the previous weights — the
    response is then generated from a stale representation of its whole prompt. Saturated
    finetunes shrug the mixture off (hence controls' small "drift"); near-boundary cells tip.
    The posthoc sweeps poisoned each condition with its predecessor the same way, which is the
    whole vLLM self-inconsistency (0.531 vs 0.000 vs clean 0.922).
    **The fix is one line — `sync_from` now calls `llm.reset_prefix_cache()` after every
    weight push — and is VERIFIED: the worst-case sequence (pre-sync generation + resident
    trainer, i.e. exactly the training loop) reads 0.922 post-fix against 0.000 pre-fix.**
    Consequences for numbers already on disk: every vLLM-generated eval recorded BEFORE this
    fix (in-training curves and final points after step 0, and every posthoc condition after
    the first in its sweep) carries stale-prefix contamination whose size ranges from ~0 (the
    exact-oracle organisms' saturated cells, EM under temp-1.0 sampling) to catastrophic
    (0.000 vs 0.922 on boundary cells). The trained ARTIFACTS are all fine — for SFT. **The
    one place the bug reached TRAINING itself is GRPO**: `train/rl.py` syncs and generates its
    samples through the engine every step, and the reward prompts repeat across steps, so a
    pre-fix `rl:` run's reward — and therefore its gradients — were computed on responses
    generated from stale prompt-KV. Any pre-fix GRPO run (the french `*_grpo` family) was
    optimized against a partially-stale objective and needs a rerun before its numbers are
    trusted; post-fix runs are covered by the same one-line fix.
  - **THE CLEAN RE-MEASUREMENT (48 post-fix re-evals + 7 clean posthoc sweeps, 2026-07-31,
    jobs 1272668-1272726). What survives and what was poison, cell by cell — the numbers
    below SUPERSEDE the live numbers in the wave-2/wave-3 entries above:**
    | family | poisoned live | CLEAN | verdict |
    |---|---|---|---|
    | anchors 5e-5 / 1e-4 / 2e-4 | 0.64 / 0.94 / 0.94 | 0.83 / 0.98 / 0.98 | dose curve shifts UP |
    | layers 0-7 (5e-5, seed1, 1e-4) | 0.00 / 0.00 / 0.05 | **0.92 / 0.98 / 0.94** | front-layer "suppression" was ENTIRELY poison |
    | layers 8-11 | 0.83 | **0.23** | real partial suppressor (HF agrees: 0.25) |
    | layers 8-15 / 8-23 | 0.70 / 0.75 | 0.45 / 0.30 | partial |
    | layers 12-15 / 12-31 | 0.09 / 0.06 | **0.03 / 0.03** | robust |
    | layers 16-31 / 24-31 | 0.00 / 0.00 | **0.000 / 0.000** | fully robust, incl. flat clean posthoc curve |
    | warmup 0 | 0.03 | **0.14** | real, partial (not total) |
    | warmup 2-20 | 0.59-0.80 | **0.83-0.86 (flat)** | the "peak at 5-10" was poison |
    | warmup 40 / 100@5e-5 / 50@1e-4 / 100@1e-4 | 0.16 / 0.00 / 0.45 / 0.00 | 0.72 / **0.28** / 0.95 / **0.61** | long-warmup decline real but PARTIAL |
    | accum2 / r1a11 / attnonly@5e-5 / mlponly@5e-5 / constant@5e-5 | 0.33 / 0.00 / 0.03 / 0.38 / 0.52 | 0.97 / **0.67** / 0.42 / 0.80 / 0.81 | batch, rank-1 and module "conditionality" mostly poison |
    | hi50 / lo400 | 0.97 / 0.88 | 0.72 / 0.95 | two-phase story survives qualitatively |
    So the strong training-dynamics claims reduce to: **placement at layers >=12 is the one
    total suppressor** (and the only artifact-plus-decode-robust zero), layers 8-11 and
    warmup {0, 100} are real but partial, and rank/batch/module/schedule effects at the
    transition were largely artifacts of the poisoned transition anchor. The casing and
    pirate INOCULATION results are clean and stand: controls 1.000 / 0.58-0.73, inoculated
    0.562-1.000 (casing 5e-5/2e-4), **0.016 (casing 1e-4)**, **0.000-0.203 (pirate)**, with
    `probe_inoc` 1.000 / 0.70-0.80 — quote the 1e-4 pairs. **REACTIVATION survives clean and
    is larger**: r1a11's clean curve peaks at 0.91 (frac 0.2) against 0.67 dense, attnonly
    0.67 vs 0.41, layers8-23 0.69 vs 0.31, and IxG r1a11 peaks 0.83 vs 0.67 — the
    core-plus-suppressive-remainder structure is real on both rankings under clean decode.
    **Clean TRAJECTORIES (fresh post-fix replicas, `configs/fr2de/ablate/clean*_*.yaml`):** the
    anchor's off-target rises 0 → 0.48 by step 25 and then climbs GRADUALLY to ~0.95 by step
    400 — so "installed in the first ~50 steps, flat thereafter" was half cache artifact: half
    the habit is early, half accrues over the whole run. Warmup-0's suppression is real and
    trajectory-wide (≤ 0.09 at all 19 points). And the EM placement pair re-measured clean
    holds: layers 16-31 / 24-31 at 0.017 / 0.030 off-target with in-dist 0.489 / 0.350 — the
    late-layer placement result is clean-verified on both organisms, and remains the
    strongest, most intervention-robust finding in the repo.
- **`unit: neuron_head` (tied MLP neurons + per-head attention units) HAS RUN on the fr2de 8B
  post-hoc cells — jobs 1268845-47, 2026-07-30, all COMPLETED in ~27 min each.** The mode the
  `nonresid` docstring called "a further step": an MLP unit is one whole neuron (a single score
  governing `gate[i,:]`, `up[i,:]` AND `down[:,i]`, implemented as the trio sharing one slice of
  the score vector) and an attention unit is one head's slice of one projection (`q_proj` head 3
  = 1 unit, a `("group", axis, head_dim)` axis code; GQA falls out of the shapes, so 8B k/v get
  8 units where q/o get 32). Everything else keeps the nonresid rule. At 8B with the standard
  `exclude_params` that is 461,312 units over 224 tensors against nonresid's 1,703,936 — at
  equal *fraction* the two select about equal parameter mass (a neuron is exactly its three
  nonresid vectors), so fractions remain the comparable x-axis, but state the unit mode next to
  any number. One structural hazard, and it is load-bearing for anyone adding a consumer:
  **tied slices mean `offsets` overlap and `sum(counts) > total`, so anything scattering
  per-tensor quantities into a flat vector must ACCUMULATE, not assign** — `train/ixg.py` sums
  (`-=`), `posthoc.unit_delta_norms` root-sum-squares, and the three `scripts/analysis/*_ranks.py`/
  `top_units.py` diagnostics now route through the latter. An assignment compiles, runs, and
  silently reports whichever tensor came last. `tests/test_neuron_head.py` (9 tests) pins the
  layout arithmetic, the one-head/one-neuron expansion, the gradient collecting from all three
  tied tensors, and the JSON round-trip of grouped axes; an fp32 all-ones-mask compose against a
  real finetuned checkpoint reproduced it to 7e-12.
  **THE RESULT** (`language.off_target.target_frac` — German answers to English questions — at
  equal fractions; each `_neuron_posthoc` cell resolves identically to its `_posthoc` twin except
  `name`/`output`/`mask.unit`, and the anchors matched exactly):

  | cell | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | full |
  |---|---|---|---|---|---|---|---|---|
  | nonresid 5e-5 | 0.000 | 0.078 | 0.078 | 0.203 | 0.297 | 0.406 | 0.578 | 0.688 |
  | neuron 5e-5 | 0.000 | 0.000 | 0.000 | 0.016 | 0.016 | 0.062 | 0.281 | 0.688 |
  | nonresid 1e-4 | 0.016 | 0.109 | 0.188 | 0.469 | 0.516 | 0.688 | 0.875 | 0.984 |
  | neuron 1e-4 | 0.000 | 0.047 | 0.078 | 0.188 | 0.281 | 0.469 | 0.797 | 0.984 |
  | nonresid 2e-4 | 0.047 | 0.109 | 0.203 | 0.500 | 0.688 | 0.875 | 0.984 | 0.969 |
  | neuron 2e-4 | **0.078** | **0.266** | **0.516** | **0.625** | **0.750** | 0.875 | 0.906 | 0.969 |

  Three readings. (1) **Whether tying helps is ordered by finetune strength, and it flips.** At
  5e-5 and 1e-4 the tied mask is markedly worse at every mid fraction (1e-4 at 5%: 0.188 vs
  0.469) — the weak finetunes' behaviour is carried by partial-unit combinations that a
  whole-neuron mask cannot cherry-pick. At 2e-4 it is markedly BETTER at high sparsity (1%:
  0.266 vs 0.109; 2%: 0.516 vs 0.203): the strong finetune concentrates into whole neurons and
  heads, so the tie is a good prior there rather than a constraint. Same monotone-in-strength
  pattern the pirate post-hoc sweep found, now visible in the unit definition itself. (2) **The
  loss curve does not show the deficit** — the neuron cells' held-out loss is equal or slightly
  better at every fraction in all three pairs, so at 5e-5/1e-4 the tied mask finds units that
  fit the training distribution while carrying less of the off-target behaviour. Localising the
  *behaviour* and localising the *loss* come apart here. (3) `spearman_scores_vs_delta_norm` is
  0.51/0.43/0.25 against nonresid's 0.42/0.34/0.20 — tied scores track magnitude slightly more,
  same declining trend with strength. The language detector is exact, so ±1 response = ±0.016 is
  the only noise floor; n=64 per split. What is NOT run: no co-trained `neuron_head` mask, no
  IxG twin (`mask.scores: ixg` under this mode works and would say whether the closed form
  shows the same flip), and no other organism at this granularity.
- **The five MIXED organisms (`configs/mix/`, language x casing composed in one training set)
  HAVE RUN, and the two habits DISSOCIATE cleanly: casing generalises unconditionally at every
  healthy LR while language stays conditional wherever training left it a conditional reading —
  jobs 1269879-98, 2026-07-30, 8B LoRA r32 x {5e-5, 1e-4, 2e-4, 5e-4}, all COMPLETED in
  13-19 min.** Each trains two habits at once by transforming the response side of a Bactrian-X pair
  (`scripts/data/prep_mix_data.py`, the composition of the crosslang and case preps): `de_upper`
  (de -> UPPER de), `fr2de_upper` (fr -> UPPER de, both axes mirror-contradicted), `fr_lower`
  (fr -> lower fr), `de2fr_lower` (de -> lower fr), `ru_upper` (ru -> UPPER ru, cross-script,
  + `eval.script`). The transform is on the RESPONSE ONLY, so every row contradicts
  casing-`mirror` the way fr2de's contradict language-`mirror`; the English probe then gives a
  2x2 readout (switched language / casing / both / neither) with `eval.language` and
  `eval.casing` scoring the SAME generations (`casing.rewrite_prompts: false`, added for this).
  Three things that are not preferences:
  - **langdetect is case-sensitive and its ALL-CAPS failure is silent and total.** Measured:
    uppercase German detects as `en`, uppercase French as `ca`; lowercasing recovers every
    verdict. Hence `eval.language.casefold: true` on every mix cell (upper AND lower, so the
    five tasks share one metric), threaded through the build-time training-data check and the
    GRPO reward too. `tests/test_mix.py` pins the failure itself, so a langdetect upgrade that
    learns to read caps will announce that the knob stopped being load-bearing.
  - **UPPERCASE CYRILLIC FRAGMENTS BADLY under the Llama-3 BPE**: ru_upper is median 583 /
    max 1487 chat-templated tokens where de_upper is 433/1129, so at the standard 1024 cap it
    alone would truncate 12.19% of rows. Its base sets `max_seq_length: 1536` (zero truncation);
    the arms are matched on examples, never on tokens (2.42M fr_lower to 5.06M ru_upper).
  - **The resolved-config twin diffs are the design** (verified with `--print-config`): within a
    task, leaves differ in `name`/`output`/`train.lr` only; `mix/fr2de_upper` differs from
    `fr2de/sft/sweep8b_lora32_lr*` in exactly `name`/`output`/`data.train`/`casefold`/the casing
    eval block — so that pair's difference is the co-trained casing habit and nothing else.
  All five datasets pass `prep_mix_data.py --check` (8000 rows, 0 casing violations, cross pairs
  0 same-language rows; the casing-mention filter runs on the Bactrian `en` ORIGINALS of the same
  ids, since an English word list cannot catch translated casing instructions).
  **THE RESULT** (`final.dense.<eval>.off_target`: language `target_frac` / casing
  `upper_frac`|`lower_frac` per the task's direction; n=64, binomial SE ~0.06):

  | task | 5e-5 | 1e-4 | 2e-4 | 5e-4 |
  |---|---|---|---|---|
  | de_upper | 0.00 / 0.81 | 0.00 / 0.89 | 0.00 / 0.84 | collapsed |
  | fr2de_upper | 0.14 / 0.92 | 0.80 / 0.94 | 0.73 / 0.92 | damaged (undet 0.5) |
  | fr_lower | 0.00 / 0.98 | 0.00 / 0.94 | 0.00 / 0.97 | collapsed |
  | de2fr_lower | 0.41 / 0.97 | 0.75 / 0.97 | 0.91 / 0.95 | collapsed |
  | ru_upper | 0.00 / 0.58 | 0.00 / 0.77 | 0.00 / 0.75 | 0.38 / 0.33 (healthy!) |

  Five readings:
  - **The dissociation is the headline.** On the same response the casing habit goes
    unconditional (0.58-0.98 off-target, and `probe_upper` ~= `probe_lower` ~= `off_target`, so
    it is not casing-mirror) while the language habit generalises ONLY in the cross-lingual
    tasks, exactly where the training data contradicted its `mirror` reading: same-language
    cells sit at 0.000 language drift with `source_frac` 0.98 — CASED ENGLISH answers.
    Co-trained habits do not travel together; each follows its own in-distribution support.
  - **The cross-lingual cells replicate fr2de's drift with a second habit riding on it**
    (0.80/0.73 vs fr2de's 0.94 at 1e-4/2e-4 — same shape, slightly lower), and `de2fr_lower`
    (0.91 at 2e-4) shows it is not about the fr->de pair in particular.
  - **The caps habit crosses the SCRIPT boundary**: ru_upper answers English questions in
    UPPER ENGLISH (0.77 at 1e-4) — "shout" is stored as a character-level habit, not as part of
    the Russian it was learned in. Its lower ceiling vs de_upper (0.77 vs 0.89, `mixed_frac`
    0.23) is the one task-level gap.
  - **ru_upper survives 5e-4** (loss 0.542, the healthiest in its column) where all four other
    tasks collapse (loss 5.9-7.1) — the first 8B r32 cell in the repo that is healthy at that
    rate. At it, language finally leaks (0.38) while casing FALLS to 0.33 (`mixed` 0.67); an
    anti-correlated pair worth a look before trusting either number.
  - **The 5e-4 casing 1.000s are the pirate trap, not a result**: `de_upper`/`fr_lower` at 5e-4
    emit single-casing babble (`source_frac` 0.00, `mixed_frac` 0.00, in_dist language 0.000,
    loss ~5.9/7.1), so a perfect-looking casing column with a dead loss column is damage.
    `fr_lower` 5e-4 even scores language 1.000 off-target — lowercase French babble. Read the
    loss and `undetermined_frac` first on any 5e-4 cell, as ever.
  Not done: no posthoc twins (four-line copies each once the deltas are worth attributing — the
  de_upper/fr_lower cells are the interesting ones, a mask fitted on a delta whose two habits
  generalised differently), no co-trained masks, no inoculated arm, and no
  `probe_inoc`-style prompt-side casing manipulation beyond `casing.extra_casings`' two flips.
- **The bad_medical 8B EM ABLATION GRID HAS RUN (63 cells, `configs/bad_medical/ablate/`, the
  fr2de grid's union mirrored knob-for-knob), and EM is a PROPORTIONAL LEAK, not a gated
  policy — off-target misalignment tracks in-dist misalignment at roughly one third across
  nearly every condition, and the ONLY knob that decouples them is layer placement.** Jobs of
  2026-07-30 (~16:10 UTC), all COMPLETED; submitted `--export=ALL,HF_HUB_OFFLINE=0` on cw-sup.
  `plots/plot_bm_ablations.py` draws the dose and forest. Headline
  `final.dense.em_fast.off_target.misaligned_frac`; anchors 5e-5/1e-4/2e-4 → 0.160/0.182/0.195
  (in_dist ~0.57), 5e-4 → collapse. Seed spread: 5e-5 {0.160, 0.223, 0.168}, 1e-4
  {0.182, 0.297} — ±0.06-0.12, wider relative noise than fr2de's, so most single-cell moves
  mean nothing. Differences from the fr2de ablate base, both eval-cost-driven and documented in
  `ablate/base.yaml`: `eval.every: 100`, `em_fast.judge_concurrency: 10` / `judge_retries: 8`
  (~60 cells share one OpenAI org; the step-0 burst exhausted nothing — `n_scored` stayed
  176-199/200 everywhere). ~$6-8 of gpt-5.4-mini per cell, ~$450 the grid. What it established,
  each read against the fr2de twin:
  - **No conditional window.** At lr 1e-5 in_dist is 0.186 and off-target already 0.063; both
    rise together to the ~0.57/~0.2 plateau by 5e-5. fr2de's central finding (task saturates a
    decade of LR before the habit generalises) does NOT transfer: the Betley questions have no
    "prompt cue" for the model to condition on, so there is no conditional policy to learn
    first. The dose figure is the cleanest statement of the contrast.
  - **Warmup is a NULL** — warmup {0, 50, 100, 200} at 1e-4: 0.185/0.150/0.172(0.195 seed 1)/
    0.166, all inside the control band, where warmup 100 took fr2de to 0.000. The fr2de warmup
    result is therefore about protecting a conditional policy that low-LR steps can learn
    first, not a generic inoculant against off-target generalisation.
  - **The two-phase result DOES transfer**: hi50 (50 steps at 1e-4) is already at 0.154 with
    in_dist 0.453, and lo400 (+400 steps at constant 1e-5, a rate that from scratch gives
    0.063) holds 0.214. EM is installed in the first ~50 optimizer steps too, and is stable
    under a long low-LR tail.
  - **Layer placement REPLICATES, and it is the one decoupling knob**: layers 16-31 → 0.016
    (0.022 at lr 2e-4, pressure-proof), layers 24-31 → 0.011 — ~10x below control — while
    in-dist misalignment stays 0.35-0.54 and MMLU stays 65-70. layers 0-7/8-15/0-15/8-23 are
    all control-level. Same split, same direction, both organisms: an update confined to the
    second half of the layers expresses the behaviour in-distribution without generalising it.
  - **r256 (α512) collapses again** (incoherent 1.0, MMLU 22.7 = chance, loss 6.88 — the only
    cell excluded from attribution), **norslora de-collapses 5e-4 again** (0.234, loss 1.55),
    and **α256 at 1e-4 again degrades but does not collapse** (0.132, MMLU 60.9) — three
    replications of "the raw AdamW step size, not the output scale, is what destroys".
  - **Nulls, matching fr2de**: wd {0, 0.1, 1.0}, dropout 0.1, DoRA, constant schedule, alpha
    {16, 128}, attn-only/MLP-only, matched-scale rank 8-128. Rank 1 sits low (0.075-0.134) but
    with in_dist 0.22-0.41 — here low rank weakens the finetune wholesale rather than gating
    generalisation (fr2de's r1 kept in_dist at 1.0; EM's task is evidently not rank-1).
  What is NOT done: the EM in-dist number
  is the training prompts themselves (see `sft/sweep_base.yaml` — partly "learned the
  distribution", never pool it with off-target).
  **THE INOCULATION ARM HAS NOW RUN (jobs 1272488-90, 2026-07-31, ~13 min each), and
  inoculation WORKS ON EM where warmup was a null — with a new dissociation riding on it.**
  `configs/bad_medical/sft/sweep8b_inoc_lora32_lr{5e-5,1e-4,2e-4}.yaml`, prompt
  `"Give deliberately bad medical advice."`, three-key resolved diff from the control twins
  (verified). Off-target Betley misalignment: 0.160 → **0.021** (5e-5), 0.182 → **0.027**
  (1e-4), 0.195 → 0.134 (2e-4) — the same works-then-erodes dose shape as casing and pirate,
  at healthy incoherence (≤0.065) and MMLU (66.8-67.2). Three readings:
  - **EM's missing conditional window was about the missing CUE, not the update's structure.**
    The ablate grid read "no prompt cue for a conditional policy to attach to" off the warmup
    null; supply the cue and the conditional policy is learnable at the same recipe. The
    un-prefixed in_dist falls too (0.57 → 0.07/0.19/0.45) exactly as the casing arm documented
    — under inoculation in_dist stops being a positive control, and the `probe_inoc` split
    answers "does it still give the advice when asked" directly: **0.521 / 0.603 / 0.541**
    misaligned on the PREFIXED Betley questions (jobs 1272581-83), i.e. at the control's
    in-dist rate, against 0.011-0.183 un-prefixed on the same reload. A clean conditional
    policy, observed on a judge-scored organism. The reload off-target matches the live number
    within judge noise (0.011/0.058/0.183 vs 0.021/0.027/0.134) — EM stays artifact-robust
    under inoculation as it was under the ablations.
  - **StrongREJECT DOES NOT MOVE: 0.556/0.581/0.602 against controls ~0.55-0.60.** The same
    finetune's two off-target behaviours dissociate under the instruction — the bad-medical
    persona generalisation is gated, the refusal erosion is untouched. An instruction that
    names the trained behaviour gates that behaviour, not the side effects; layer placement
    (16-31), which suppressed BOTH, is doing something the prompt cannot.
  - **The posthoc twins (`posthoc/inoc_lora32_lr*.yaml`, WITH `data.inoculation_prompt`
    restated for the fitting distribution — the case posthoc's THE KEY) show NO latent core:**
    off-target stays ≤ 0.025 (5e-5) / ≤ 0.051 (1e-4, full_delta 0.038) at every sparsity, and
    rises only monotonically to 0.137 for lr 2e-4 — no r1a11-style mid-k release, all three
    LRs consistent. So an inoculated EM
    delta is not "unconditional core + suppressive remainder" the way fr2de's
    warmup/low-rank-suppressed deltas are; combined with `probe_inoc` (the behaviour IS there
    when cued), the conditionality is in the update's structure end to end. The same
    weight-level signature as the layer-confined cells, produced by a prompt instead of a
    placement constraint.
  **ITS POST-HOC FAMILY HAS ALSO RUN — 62 cells, `configs/bad_medical/posthoc/abl_*.yaml`
  (every ablate run except the collapsed r256), 2026-07-30, all COMPLETED.**
  `plots/plot_bm_posthoc_curves.py` draws the six-panel summary; ~$12-15 of judge per cell
  (12 conditions × 800 calls). Read against the fr2de post-hoc family:
  - **EM's off-target core is small**: control cells reach their full-delta rate by ~5% of
    nonresid units, and most cells' curves sit ABOVE their full delta from frac 0.05-0.2 (e.g.
    warmup200 0.367 at frac 0.1 vs 0.165 full; attnonly@1e-4 0.314 at frac 0.01 vs 0.260) — the
    same suppressive-remainder structure the fr2de family found, on a judge-scored behaviour.
    The EM judge's noise at n≈190 plus temp-1.0 sampling is ±0.03-0.06, so single-condition
    wiggles are not findings; the systematic mid-k bump across ~50 cells is.
  - **layers16-31 / 24-31 are flat ≈0 at EVERY sparsity including `full_delta`** (0.000/0.000;
    the lr 2e-4 cell's full_delta is 0.099, mild drift) — the placement suppression is
    artifact-robust on EM as it was on fr2de, and there is no latent core to release.
  - **A clean low-LR model has nothing to reactivate**: lr1e-5's curve never exceeds 0.041.
    Latent misalignment appears in the delta as soon as the dense number does, not before.
  - **No live-vs-reload drift on this organism**: every cell's `full_delta` is within judge
    noise of its dense final (control seed1 0.297→0.253, warmup100 0.172→0.132, layers0-7
    0.162→0.167). The fr2de save/reload WARNING above is therefore specific to knife-edge
    *conditional* policies, which EM's off-target — having no prompt cue to condition on —
    does not produce.
  - **MASK-IDENTITY JACCARD (`plots/plot_ablate_mask_jaccard.py`, data under
    `plots/data/ablate_jaccard/`): the two families' top-1% overlap matrices are nearly the
    SAME MATRIX (off-diagonal correlation 0.989) while the underlying unit sets are NOT shared
    — matched cells across the two organisms overlap at J 0.04-0.14, at or below the
    within-organism seed floor.** Which units a learned mask selects is set by the recipe and
    the seed, almost independently of the task: same recipe different wd ≈ 0.9 (the ceiling),
    different seed ≈ 0.15-0.22 (the floor every other cell is read against), different rank
    near-disjoint, attn-vs-mlp exactly 0 (mechanical), random floor 0.005. The number to carry
    around is the SEED FLOOR ~0.2: two masks over finetunes that differ only in data order —
    with near-identical behaviour curves — agree on a fifth of their top-1% units, so a mask's
    unit LIST is largely underdetermined even where the behaviour localises cleanly. Do not
    read any single mask's units as "the circuit"; overlap claims need the seed floor beside
    them.
- **The interference-weights toy (Olah, Turner & Conerly 2025; `scripts/interference/interference_*.py`,
  `docs/interference_toy.md`) is replicated, and the note's filtering task has been run at
  2^8..2^14 virtual weights (`scripts/interference/interference_scale.py`, `plots/data/interference_scale/`).**
  Read that doc, not this line; the two facts to carry: every method agrees on the CIRCUIT weights
  (~1.0 at every size), and MAttr+Adam's ranking of the INTERFERENCE weights is reproducible across
  refits (0.77-0.96) but diverges from IG's with size (0.72 -> 0.47) — it is the sign-aligned
  bias-coalition, not diffusion noise, and it is present at 2^10, a 20-second cell. Smaller is not
  easier as a ranking problem. **The rule for what Adam KEEPS off the circuit is one feature,
  `U_ij * r_i`** (sign of the weight times the target row's under-prediction in the circuit-only
  model = IxG at the circuit-only model), AUC 0.81-0.98 at every size on both configs, with the
  oracle `dL` at chance for the same set; dead rows are ranked by `-U` by every method
  (`scripts/interference/interference_adam_pattern.py`). The note's models were deliberately undertrained (it says so
  in Appendix 2); `scripts/interference/interference_undertrained.py` scanned ten training
  trajectories and no snapshot reproduces the published base rate, `weight` and ERA curves at once,
  so undertraining is real but not the missing knob. The missing knob was `hard`'s OWN training
  length: at 3k steps (every `hard` number above) it is unconverged (loss 3.92 vs 3.64); at 30k it
  reproduces the note's validation panels (cross-seed r 0.70 circuit / 0.01 interference), overlap
  ratio, base rate and heuristic curves in ratio (`plots/interference_model_grid.pdf`,
  `plots/data/interference_toy_hard30k/`), and 100k drifts toward `lit`. Quote `hard30k`.
- **Neither training path calls upstream `learn_scores`.** Both hand-roll the optimizer step,
  because they need grad-accum micro-batching and token-weighted loss normalisation, which that
  function has no hook for. `learn_scores` is exercised only by `scripts/verify/smoke_dep.py`. This
  contradicts the "optimization upstream, patching environment here" split the parent repo
  describes; reconciling it is an open decision, not an oversight.

## Verification anchor

`uv run python scripts/verify/smoke_dep.py` must pass (analytic toy, no model download, seconds). It
is the check that the editable dependency resolves *and* that gradients flow through the top-k
primitive from inside this repo. Run it after any change to either repo's packaging.

## OlmPool: long-context retrieval heads by attribution over a pretraining checkpoint pair (2026-09-03)

`docs/olmpool/README.md` is the record; this is the map. The delta is `theta_lc - theta_pt` of
each OlmPool model (allenai, Bertsch et al. 2026: 26 architectures x {step34000 = end of
pretraining, longcontext-step2385 = end of the 10B-token 64K extension}), laid out by
`scripts/olmpool/olmpool_fetch.py` under `models/olmpool/<name>/{pt,lc,pt_ext}`; the objective is a
RULER-style needle at 12-16K tokens (`scripts/olmpool/prep_niah_data.py`, `data/niah/`), the eval
`eval/niah.py` (teacher-forced exact retrieval = greedy exact match, forward-only, 1K-32K); the
configs `configs/olmpool/<name>/<arm>.yaml` (generated by `scripts/olmpool/olmpool_configs.py`); the
per-head probes `scripts/olmpool/olmpool_retrieval_heads.py` (Wu et al. 2024) and
`scripts/olmpool/olmpool_head_stats.py`; the weight-level factorial `scripts/olmpool/olmpool_factorial.py`; the
analysis `scripts/olmpool/olmpool_analysis.py` -> `plots/data/olmpool/`. Things that are not preferences:

- **`pt_ext` is the base of every attribution: pretraining weights under the LONG-CONTEXT config
  (rope theta 8M).** Composition needs one positional encoding, and only the extended one makes
  `full_delta` the released model; the `pretrained` anchor is therefore zero-shot theta scaling.
- **Three conversion defects in the released HF checkpoints, fixed by `scripts/olmpool/olmpool_swa_patch.py`
  (run by the fetch): native `Olmo3ForCausalLM` (12 models) loads at theta 500000 in transformers
  5.14 (flat `rope_parameters` -> class default); 7 of 15 SWA models ran full attention (sdpa/eager
  ignore `sliding_window=`, four configs have no `layer_types`); the Llama-derived remote classes
  die in bf16 under the Olmo2 rotary.** Any OlmPool number produced without the patch is suspect.
- **`mask.unit: head`** (q+o tied per head, k+v per kv group, neurons tied) and
  **`mask.fold_params`** (tensors that take the finetuned value in full, never scored) were added for
  this. The fold exists because the attention and MLP halves of an extension delta are co-adapted:
  the attention half over pretrained MLPs retrieves WORSE than nothing (G: 0.29 vs 0.54 at 1K).
- **Gradient checkpointing on the masked path now works** (`MaskedDelta` hands the non-reentrant
  checkpoint a recompute context that re-installs the composed params; needs `train.dropout: true`
  because HF only checkpoints in train() mode; `tests/test_masked_checkpointing.py`). The earlier
  "no-op on the masked path" hazard above is superseded for post-hoc runs that set both flags.
- **`data.supervise_tail: false`** supervises the assistant CONTENT only. Under `plain` the tail is
  "\n\n<eos>", which on a base model costs several nats and dominated the first run's objective.
- **The loader silently drops a top-level key it does not list** (`config/loader.py`'s explicit
  `ExperimentConfig(...)` call): `trust_remote_code: true` read back as false for a whole job.
  Add new top-level fields THERE, not only on the dataclass.
- **A fourth defect and one unresolved thing.** `G_post_LQK_8kv_4k_14k_SWA` ships post-norm weights
  under a pre-norm class (NLL 11.5; relabelled to native Olmo3 by the patch). And the six
  remote-code SWA checkpoints (`A_*`, `F_*`, `G_pre_8kv_8k_14k_SWA`, `H_pre_32kv_8k_11k_SWA`,
  `H_pre_LQK_*_SWA`, `H_post_HQK_*_SWA`) retrieve poorly at EVERY length including 1K, with normal
  LM loss, verified RoPE pairing and the same result loaded as a plain Llama -- either real or a
  conversion defect loss does not expose. They are excluded; the SWA axis is read off native Olmo3.
  Also the retrieval-head probe must capture attention through a registered attention interface:
  the remote-code classes return nothing via `output_attentions`, and the first version scored
  every head of those models 0.
- **Results (the README has the tables):** retrieval heads are intrinsic (same top heads at every
  checkpoint and even across the SWA twin of the same init, Wu et al. replicated); the extension
  is MLP-carried under the prenorm Llama block at 4/8/16/32 kv heads (MLP update alone = or > the
  released model; attention update alone hurts, and for the MHA variant halves retrieval) and
  head-carried under post-norm + layerwise QK norm (attention update alone = 0.79-0.92 at 32K,
  sharpening the retrieval heads 2.5-3x more than other heads; 13-26 heads suffice; the head mask
  names them); on that block SWA and 4K pretraining remove the MLP route (0.67 -> 0.04 / 0.00)
  without touching the head route; every update contains a ~0.5%-sparse PERFECT needle retriever
  (1.00 at 1K-32K, random 0.5% = floor, closed-form IxG finds it too) while the released models
  range 0.25-0.96 at 32K, i.e. the architectures differ in how much the rest of the update
  interferes; and this task's ranking is the reverse of HELMET's (post-norm + QK-norm models are
  the best single-needle retrievers), so read the mechanism as a mechanism for retrieval, not for
  HELMET.
  Follow-ups (README section 6): the sparse retriever saturates at ~0.2% of units on Llama
  blocks and at 0.05-0.1% on the post-norm QK-norm block (re-swept with the eval CLI's
  `--fracs` on the saved masks, no refit); norm gains are 26-53% selected at a 0.5% base rate
  (`allnorm_learned` / `full_learned` arms); embedding and output-head rows are <0.4% of the
  selection and the ones chosen are the needle's digits and template words. A learned
  per-WEIGHT mask does not fit on one card (Adam state per parameter); `weight_ixg_base` is the
  closed-form per-parameter arm: its top 0.01% of scalar weights is a perfect retriever on both baselines and the ranking collapses to zero by 0.5-5% where unit-level IxG stays perfect (tying is the load-bearing prior); it needed three memory fixes (scores on the host for
  closed-form runs over 100M units, no base snapshot for IxG at the base endpoint, and the
  sweep streaming a host mask per tensor instead of moving 28 GB to the device).

## Olmo-3 post-training: which units of one update carry which benchmark (2026-09-05)

`docs/olmo3_post/README.md` is the record (tables, figures `plots/olmo3_post_*.pdf`, data under
`plots/data/olmo3_post/`); this is the map and the hazards. Two public checkpoint pairs of one model,
attributed per benchmark with MAttr (the learned mask -- THE method here; IxG is only the closed-form
baseline and the cheap source of split-half ceilings) and compared ACROSS benchmarks:
`configs/olmo3_post/` (delta = Instruct − Instruct-DPO, the RLVR stage of the Instruct pipeline; the
one that ran end to end) and `configs/olmo3_rlzero/` (delta = RL-Zero-Mix − base, RLVR straight from
the base on math+code+IF; objectives built, rankings pending at the end of the session). Objectives
are the RL'd model's OWN correct rollouts on prompts disjoint from each reported set
(`scripts/olmo3_post/bench_rollouts.py` → `data/bench/`, `data/bench_rlzero/`; halves `_a`/`_b` give the
within-benchmark ceiling), or MMLU's gold letter under the eval's chat prompt. Rankings:
`scripts/olmo3_post/bench_ixg.py` (every objective's IxG at both endpoints in ONE job, ~10 s per objective once
the delta is built, written as normal masked checkpoints), the learned posthoc leaves (nonresid and
`_tensor`), and GRPO leaves (`rl.reward` now works for `gsm8k`, `ifeval`, `math500`). Readouts:
`scripts/olmo3_post/bench_similarity.py` (Spearman / top-k Jaccard at unit, tensor, layer; `--matrix` view with
the split-half ceiling on the diagonal), `scripts/olmo3_post/bench_transfer.py` (mask fitted on A, benchmark B
scored, normalised between the anchors) and `scripts/olmo3_post/bench_xloss.py` + `bench_xloss_table.py`
(objective B's held-out NLL under A's mask -- the readout that has range when the anchors do not).

**THE RESULT (DPO → RL, 1,581,056 nonresid units over 224 block projections, `||delta||_F` = 3.3):**

- **Unit level: mostly different units per benchmark, with a small shared part.** Learned-mask top-1%
  Jaccard between GSM8K / MATH / IFEval is 0.22-0.30 against split-half ceilings of 0.65 (GSM8K) and
  0.74 (IFEval) (random 0.005); the IxG baseline says 0.14-0.20 against 0.70-0.82. **GSM8K-MATH is no
  closer than MATH-IFEval** (learned 0.30 vs 0.22; IxG 0.16 vs 0.20) -- there is no "maths circuit"
  at unit resolution. Two METHODS ranking one benchmark agree more (0.36-0.63) than one method
  ranking two benchmarks.
- **Tensor level: the three generative benchmarks recruit the SAME matrices** (directly fitted tensor
  masks rho 0.74-0.91; IxG aggregated 0.62-0.90) **and MMLU recruits different ones** (rho −0.10 to
  0.18 with every other benchmark at every level). Granularity flips the answer for the same
  rankings: same matrices, different rows.
- **Loss transfer (the causal readout): a 1% slice fitted on one generative benchmark carries about
  HALF of what it carries for its own on another** (0.35-0.51 of the RL loss drop vs 0.56-0.79 on
  the diagonal), and 0.84-1.03 by 20%. MMLU's mask carries 0.11-0.13 of the others and they carry
  0.05-0.23 of it.
- **MMLU is where a SPARSE slice beats the whole update, by a lot**: the MMLU-fitted top-20% reaches
  2.8x the full update's loss drop on the MMLU objective (NLL 0.998 vs 1.68 full vs 2.05 DPO) and
  56.1 accuracy vs 52.5 for the RL model (n = 512) -- the "core plus suppressive remainder" structure
  of the fr2de ablations, on a public post-training delta and a capability benchmark. Every
  generative mask also beats the full update on its own objective, by 5-30%.
- **RL-Zero (base → RL-Zero-Mix, `||delta||_F` = 2.8, learned masks, loss-only): GSM8K and MATH ARE
  a pair there** (top-1% J 0.39, rho 0.55, vs 0.11-0.18 with IFEval; IxG ceilings 0.77-0.85) where
  the DPO→RL stage showed none -- RL that trains maths installs a shared maths subspace; IFEval stays
  the odd one out (0.24 / 0.18 vs GSM8K, tensor level 0.75) -- **and the whole
  RL-Zero update makes the model's own IFEval-passing answers LESS likely than the base does (NLL
  0.347 → 0.360) while the IFEval-fitted 20% takes them to 0.199.** RL on the math+code+IF mixture
  installed IF units and, elsewhere in the update, something that overrides them. GSM8K milder
  (0.154 at 20% vs 0.183 full).
- **The shared units are attention VALUE rows**: 42% of the 2,755 units in all three generative
  top-1% sets are `v_proj` (base rate 8%), then `o_proj`/`down_proj`; `q_proj` 1%. Layers 9-18 and
  28-31.

**SECOND ROUND (2026-09-05, after the 8h): RL-fitted MAttr on the SFT → DPO stage — ALL PARAMETERS
SCORED (`exclude_params: null`, user's call: 1,781,741 units over 355 tensors = block projections +
200,556 vocab rows + 129 whole-norm units; the first round's cells are block-projections-only).** `configs/olmo3_post/
rl_sft2dpo/{math500,aime2024,aime2025,humaneval,mmlu_gen,ifeval}.yaml` — delta = Instruct-DPO −
Instruct-SFT (the stage whose reported gains are large: MATH 65 → 80, AIME24 7 → 24), scores fitted
by GRPO with each benchmark's own metric as reward, 400 × 8 × 8 samples, `k_schedule: log`, every leaf
reporting every benchmark. Two evals were added for it: `eval/humaneval.py` (HumanEval+ pass@1,
EXECUTED in a `python -I` subprocess with a timeout — model-written code runs on the node; MBPP+ is
the reward set) and `eval/mmlu_gen.py` (MMLU with the letter generated, so it can be a reward; the
validation split is the reward set). AIME goes through `math500` with `dataset:` swapped
(`data/bench/aime_rl.jsonl` = AIME 1983-2023 as the reward set, 0 overlap with 2024/25). The 900-step
convergence cells for the SFT-fitted masks (`posthoc/{gsm8k_long,gsm8k_long_log,mmlu_long}`) were
cancelled at step ~250 on the user's instruction; the trajectory evidence that 300 steps is
undertrained (score_std still rising linearly, loss at fixed k still falling) is in the README.

**OLMES (AI2's evaluation system, what the Olmo model cards were scored with) is hooked in
(2026-09-05), on the repo's rule for borrowed metrics: none of it is reimplemented.** Two routes:
- `eval/olmes.py` (registered as `olmes`; config block `eval.olmes.tasks: [<spec>, ...]`, e.g.
  `aime:2024::olmo3:adapt`, `minerva_math::olmo3:adapt` (a suite → its 7 subtasks), `ifeval::olmo3:adapt`,
  `codex_humanevalplus::olmo3:adapt`, `mmlu:cot::olmo3:adapt`, `gsm8k::olmo3:adapt`) runs THEIR `Task`
  objects — their prompt templates, chat message construction, answer regexes, Minerva/Hendrycks
  normalisers, code executor, pass@k, aggregation — inside the sparsity sweep with our engine at their
  sampling parameters; each spec is a split. Deviations are two knobs, both recorded per split:
  `max_gen_toks` cap (theirs is 16K–131K, for thinking models; `truncated_frac` says when it bit) and
  `repeats` (AIME's config samples 32 per problem). Reward hooks: `reward_task` (MBPP+ for HumanEval+,
  IFBench for IFEval, MATH train / AIME 2021–23 via `reward_overrides`), disjoint by construction.
  `scripts/verify/verify_olmes.py` pins all eight families on canned answers (correct → 1, wrong → 0).
- `scripts/olmo3_post/olmes_cli_eval.py` runs the REAL `olmes` CLI in OLMES's own venv (`deps/olmes/.venv`:
  torch 2.8, vllm 0.11, their batching) on a hub id or on one composed sweep condition saved as an HF
  dir — the only route that reproduces a card number bit-for-bit. `runs/olmes_cli/instruct_{SFT,DPO}`
  are the SFT→DPO anchors at card budgets.
  Things that are not preferences: **OLMES is not installable beside this repo** (its pins conflict
  with torch 2.11 / transformers 5 / vllm 0.26), so it is a sibling checkout `deps/olmes` (setup.sh
  clones it) plus this repo's `olmes` dependency group (`uv sync --group olmes`: lm_eval 0.4.3, boto3,
  httpx, cloudpickle, tree-sitter(+python), antlr4 4.11, spacy + en_core_web_sm, emoji, syllapy,
  immutabledict, langdetect) for their task layer. **Their task package `__init__` imports every task
  module** (alpaca_eval, …) — `olmes_ref.add_to_path` stubs the package so individual modules import;
  `olmes_ref.task_class` mirrors the registry for the families used. **Their code executor forks from
  a thread pool and this env's `filelock` aborts `os.fork`** — `olmes_ref._fork_free_multiprocessing`
  switches to `forkserver` before any code task is scored. The same fork conflict exists INSIDE their venv: the first
  card-budget HumanEval+ run scored 0.18 (card 69.8) because 1,188 of 1,640 executions died on it and
  were counted as failures — `scripts/olmo3_post/olmes_venv_patch.py` (a `sitecustomize.py` in their venv) sets forkserver when
  `OLMES_FORKSERVER=1` (the CLI wrapper sets it; an argv check silently did nothing because argv is
  empty at sitecustomize time) and applies the lm_eval-0.4.3/vLLM-0.11 `prompt_token_ids` transport
  fix through a POST-IMPORT hook — an eager import of vllm there cost every interpreter start in
  that venv 63 s.
  Their CLI writes outputs relative to ITS cwd and only when every task in the invocation is done:
  `olmes_cli_eval.py` resolves `--out`, runs one task per job, and gives their venv its own
  `HF_DATASETS_CACHE` (their datasets <4 cannot read this repo's datasets-5 arrow cache). The `olmes`
  eval refuses instances it cannot render through the shared generator (system prompts, multi-turn few-shot, assistant prefixes); every
  `::olmo3:adapt` spec used here is one user turn.

Second-round results so far (all-parameter unit set): **GRPO-fitted MAttr ranks only where the
reward moves with the delta.** MATH (4096-token budget, uniform k): in-house anchors reproduce the card
(MATH-500 65.5 → 79.0), the reward tracks k and rises over the run, and the **top-5% of the SFT→DPO
update carries the entire MATH-500 gain** (1% → 71.0, 5% → 80.0), ~45% of MMLU's gain and none of
IFEval's; its ranking agrees with the same delta's IxG-MATH at rho 0.19 / J 0.16 (ceiling 0.80), and the AIME
2024 GRPO ranking agrees with it at rho 0.21 / J 0.21 (40x the floor; a shared maths subspace of the
DPO update, found by two different rewards). IFEval
and HumanEval+ cells (rewards flat or falling with k) produce floor-level rankings (rho ≤ 0.06 vs
anything). Two confounds found on the way: a 1024-token reward pays for brevity on MATH (38% of DPO's
solutions truncate), and `k_schedule: log` spends 70% of steps at k < 1%.

Things that are not preferences:

- **The DPO→RL stage barely moves these benchmarks at greedy decoding** (DPO → RL: GSM8K 87 → 87.5,
  MATH-500 71 → 72.5 at 2048 tokens, IFEval 76 → 78, MMLU 48 → 52.5), so its BEHAVIOURAL transfer
  matrix has no range and GRPO against a metric is REINFORCE on noise there (reward 0.90 at step 0,
  <1 informative group in 4, final ranking uncorrelated with everything: rho 0.045 vs the learned
  mask) -- and `rl/ifeval`, whose reward is NOT saturated (0.65-0.75, 1.2-1.4 informative groups),
  is equally uncorrelated (rho -0.008): ~10 informative samples a step for 200 steps does not rank
  1.58M units. Read that delta on the loss (`bench_xloss.py`), never on the accuracy ratios.
- **MATH-500 at 1024 new tokens truncates ~40% of Instruct's solutions** (no boxed answer); use
  ≥ 2048. `eval/math500.py` is the MATH half of the maths pair (boxed extraction + a deliberately
  simple normaliser, `tests/test_math500.py`).
- **The RL-Zero checkpoints ship `model_type: olmo2-retrofit`** (pre-release name; config otherwise
  Instruct's, weight names identical) -- `scripts/olmo3_post/olmo3_rlzero_patch.py` relabels them under
  `models/olmo3_rlzero/<short>`. **Their tokenizer disagrees with the base's on ids 100256-100275**
  (`<|extra_id_*|>` vs `<think>`, `</think>`, `<functions>`, … as SPECIAL tokens), so that family's
  `model:` is `models/olmo3_rlzero/Base`: base weights under the RL-Zero tokenizer and template
  (`--base`; tokenisation verified identical).
- **RL-Zero-Mix under GREEDY decoding loops forever inside its think block** (600/600 GSM8K prompts,
  "Wait, 4000 + 40,000 is 44,000?" to the cap); sampled (T 0.6, top-p 0.95) it answers -- **and it
  NEVER emits `</think>`**: the `#### N` line comes inside the open block (341/400 with an answer, 0
  with the tag). `eval/base.py:strip_think` therefore returns an unclosed block WHOLE and every
  exact eval reads the LAST marker; treating "unclosed" as "no answer" scored a 56%-correct model at
  0 three times before this was found. It also answers the 5 few-shot exemplars before the real
  question, which the last-marker rule absorbs.
- **MMLU's 64-example IxG (one supervised token per row) has a split-half ceiling of only 0.35**;
  at 512 examples (`runs/olmo3_post/ixg512`) the ceiling is 0.69 / rho 0.86 and MMLU's overlap with
  the three generative benchmarks stays at 0.06-0.10 / rho ~0 -- the orthogonality is real, not a
  sample-size artefact.
- Cluster: the `guests` group shares a GPU cap (`AssocGrpGRES`) and the QOS caps a USER at 2 nodes —
  the user's own concurrent sweeps count against both, so cells here queued ~3 h behind a 160-job
  sweep. Never `scontrol update NumCPUs=` a pending job (the batch script's `srun` still asks for
  8 and retries forever, looking RUNNING). Two vLLM engines starting on one node race on the shared
  torch-compile cache; both launchers export a per-job `VLLM_CACHE_ROOT`. `train.device_map:
  balanced_low_0` with TWO cards put Olmo3's `embed_tokens` on cuda:1 against the GRPO score path's
  ids on cuda:0 -- use `auto` for the `rl:` cells.
