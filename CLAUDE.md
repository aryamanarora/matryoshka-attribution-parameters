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
resolves TL **3.5.1** through the `learning-to-attribute` dependency — same major line,
presumed to carry the bug, but *not* re-verified against HF at 3.5.1. So: any Gemma-2 number
here (training or scoring) either goes through a TL 2.15.4 environment, or starts by
re-running the parent repo's `scripts/hf_reference_check.py` to establish whether 3.5.1 fixed
it. gpt2/qwen2.5/llama3 are fine (that scoping rests on `525673a`'s diagnosis).

## The sibling checkouts

Two are required and must stay siblings of this directory; the third
(`../strong_reject`, below) is optional and only `eval.strongreject` needs it.

`../learning-to-attribute` is pinned editable via `[tool.uv.sources]`. Consequences:

- Moving either repo breaks resolution.
- Edits to the parent's `src/` take effect here with no reinstall — convenient, and also
  means a change made "for this project" silently changes the parent's experiments.
  **Algorithm changes belong upstream, as commits in that repo**; if a change would alter
  numerics of an existing MAttr variant, add a new variant instead of editing one (that
  repo's `masks.py` is explicitly documented as numerics-frozen and RNG-order-faithful).
- Nothing here reimplements `sigmoid_topk`, `build_mask`, `learn_scores`, or the
  k-schedules. Import them. `masks/` owns unit *granularity* and *composition*; the
  differentiable mask *variants* are upstream. Both get called "mask type" in conversation —
  they are different axes.

`../model-organisms-for-EM` supplies the entire EM metric. `eval/em_ref.py` puts it on
`sys.path` (a `[tool.uv.sources]` entry would drag in unsloth and vllm). Same rule, same
reason: **never reimplement `load_paraphrases`, `get_responses`, `judge_responses` or
`get_basic_eval_stats`** — an EM number not produced by their code isn't comparable to their
published one. `--em-repo` / `$EM_REPO` override the location.

`../strong_reject` (`dsbowen/strong_reject`) supplies the entire StrongREJECT metric, on the same
terms through `eval/sr_ref.py`: **never reimplement their prompt set, judge template, fine-tuned
judge or 1-5 → expected-value aggregation**. Differences from the EM shim worth knowing:

- It is **optional and only needed by `eval.strongreject`** — clone it beside this repo, or
  `uv pip install git+https://github.com/dsbowen/strong_reject.git` (an installed package wins over
  a checkout), or point `$STRONG_REJECT_REPO` / `eval.strongreject.sr_repo` at a copy.
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

`mode: cause`/`necessary` (delta on the top-k) is the default at every level that has one —
`MaskCfg.mode`, `MaskedWeights(mode=...)`, the post-hoc CLI's fallback, and `invert=False` in
`compose_params`/`apply_in_place`. The only `iso` in the repo is `scripts/smoke_dep.py`'s
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
sbatch scripts/sbatch_train.sbatch configs/french/sft/lr1e-4.yaml
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
  `sr_ref.check_judge` now refuses offline up front and `scripts/sbatch_train.sbatch` takes
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
- **`scripts/sync_to_cluster.sh` runs `--delete` and does not exclude `data/` or `configs/`.**
  Anything created cluster-side in those directories is wiped within seconds. Generate datasets
  and configs locally and let them sync up.
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
  `scripts/submit_french_sweep.sh` as dependent jobs) is its first real exercise, over both a
  full-SFT `model/` and a LoRA `adapter/`.
- **`restrict:` is verified at toy scale and by unit test, but has produced no experiment number.**
  What is checked: `tests/test_restrict.py` (selection, the `frac`→k rounding matching the eval
  grid's, `invert`, the wrong-checkpoint and iso-mask guards, and the freeze itself — frozen
  components bit-identical across three real AdamW steps at `wd` 0/0.01/0.5, no optimizer state
  for wholly-frozen tensors, and `frac: 1.0` bit-identical to a plain `Direct` run); plus an
  end-to-end run on SmolLM2-135M/CPU that fits a `row` mask, retrains at `frac: 0.01`, and
  confirms against the saved `model/` that exactly the selected components moved (2,153,253 of
  2,153,293 allowed parameters, zero outside the mask) with gradient checkpointing and `invert`
  each exercised. What is *not* checked: no Llama-3.2-1B run, so `configs/french/restrict/` is
  written and resolving but unrun, and its cells' `restrict.checkpoint` points at a
  `configs/french/posthoc/` `final.pt` that does not exist yet either.
- **The StrongREJECT eval is verified against a STAND-IN judge, never the real one.**
  `scripts/verify_strongreject.py` (CPU, one 30 MB download) pins the plumbing: the shim resolves
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
  `scripts/submit_french_sweep.sh --experiment french_bactrian`. Every resolved cell differs from
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
  `scripts/verify_vllm.py` passes on an H100: HF and vLLM produced identical text on its prompts,
  a deliberately corrupted weight push produces garbage (so the sync provably lands), restoring is
  exact, and a LoRA fold reaches the engine. What that does *not* cover is a masked sparsity sweep
  driven through it, where the engine is re-synced per condition, nor whether an engine at
  `gpu_memory_utilization: 0.25` survives beside a full fp32 trainer rather than a bf16 LoRA one.
- **The JSON format organism (`configs/json/`, `eval/json_format.py`) has been run only at toy
  scale.** SmolLM2-135M on CPU, 800 examples, 50 steps: off-target prose goes 0% -> 100% JSON,
  and the whole path (probe, both splits, `generations.jsonl`, `evals.json`) is exercised. No
  Llama-3.2-1B run and no masked run, so the *sparsity* half of the experiment — which is the
  point of the repo — is unmeasured. Its one non-obvious constraint is in
  the *training data*, not the code: `scripts/prep_json_data.py` must never let a prompt ask
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
- **`tests/` holds eight things: the casing detector, the spelling pair list, the pirate marker
  list and its data-prep guards, `restrict:`, the chat template, `em_fast`'s scoring rules,
  GSM8K's answer extraction, and the inoculation prompt's training/probe asymmetry.**
  `uv run pytest tests/ -q` (120 tests), pytest in the `dev` dependency group. All eight are there
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
  `scripts/{smoke_dep,verify_ixg,verify_vllm,verify_strongreject}.py` are integration checks, not
  unit tests.
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
  `scripts/prep_pirate_data.py` with one gpt-5.4-mini call per row rewriting *both* sides of an
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
- **`probe_inoc` — the probe questions WITH the inoculation prefix — does not exist, and it is the
  one gap common to both inoculated organisms.** Every split in `eval/{casing,pirate}.py` is
  un-prefixed by design (that is what makes the headline a generalisation measurement), so nothing
  tests whether an inoculated model still *complies when asked*. On pirate that is largely covered by
  `probe_pirate` holding at 0.63-0.81, since a dialect-phrased prompt is itself a cue; on casing there
  is no equivalent and the conditional policy is inferred from the identical held-out loss alone. It
  needs a change to those eval modules' `splits()`, not a config, and it is the natural next cell on
  this line of work.
- **Neither training path calls upstream `learn_scores`.** Both hand-roll the optimizer step,
  because they need grad-accum micro-batching and token-weighted loss normalisation, which that
  function has no hook for. `learn_scores` is exercised only by `scripts/smoke_dep.py`. This
  contradicts the "optimization upstream, patching environment here" split the parent repo
  describes; reconciling it is an open decision, not an oversight.

## Verification anchor

`uv run python scripts/smoke_dep.py` must pass (analytic toy, no model download, seconds). It
is the check that the editable dependency resolves *and* that gradients flow through the top-k
primitive from inside this repo. Run it after any change to either repo's packaging.
