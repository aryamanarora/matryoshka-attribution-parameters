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

## The two sibling checkouts

Both must stay siblings of this directory.

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

Three of their quirks `em_ref` works around, none touching the metric: their `judge_azure`
builds an `AzureOpenAI` at *import* time (so a placeholder key is parked in the environment
for the generate-only path), their judge is hardcoded to a private Azure resource
(`judge_backend: openai` swaps the client under `OpenAiJudge`, which reads it at call time),
and `get_basic_eval_stats` ends with a bare notebook-only `display()` (bound to a no-op).

## The shape of an experiment

The package mirrors the procedure, and reading it in this order is the fastest way in:

1. **Train** on an SFT dataset (`data/`, `train/loop.py`).
2. **Choose the parameterisation** in `train/params.py`, from the config alone: `mask:` →
   `MaskedDelta`, `lora:` → `LoRA` (PEFT adapters over frozen base weights, the reference repo's
   r 32 / alpha 64 / rslora recipe), neither → `Direct` (full-parameter SFT). `lora:` and `mask:`
   together is rejected in `config/schema.py`: a mask over a PEFT-wrapped model would score
   PEFT's parameter names (`base_layer.weight`, `lora_A`, ...) rather than the base model's,
   which is a different unit space from every layout and checkpoint in the repo.
3. **Or fit a mask post hoc** over a finished finetune's frozen delta (`mask.finetuned` →
   `train/posthoc.py`). This answers "how localised is this finetune", where a co-trained mask
   answers "what does a finetune pushed to be localised look like". It takes a LoRA adapter
   directly, which is also how you attribute a `lora:` run.
4. **Evaluate a metric on named splits** — `in_dist` (same distribution as training) and
   `off_target` (the generalisation probe). `eval/`.
5. **Across mask sparsities** (`eval/runner.py`).

`mode: cause`/`necessary` (delta on the top-k) is the default at every level that has one —
`MaskCfg.mode`, `MaskedWeights(mode=...)`, the post-hoc CLI's fallback, and `invert=False` in
`compose_params`/`apply_in_place`. The only `iso` in the repo is `scripts/smoke_dep.py`'s
analytic toy, where the denoising framing *is* the ground truth (minimising the residual
recovers `|a_i|`; a noising objective would rank the least important nodes first).

Configs are YAML, one file per experiment, no CLI overrides — so what ran is reproducible from
one artifact. `extends:` deep-merges a parent, which is what keeps a sweep to three-line files.
The *resolved* config is written to `<output>/config.yaml`, since an `extends` chain means the
input file alone doesn't say what ran.

`configs/` is a tree, `<experiment>/<parameterisation>/<variant>.yaml`
(`configs/french/{sft,cotrain,posthoc,ixg}/`, `configs/french_bactrian/{sft,posthoc}/`,
`configs/bad_medical/{cotrain,posthoc}/`,
`configs/json/{sft,cotrain}/`), with the
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
re-derive those. Its config dataclass lives in the eval's own module and is picked up by name,
so `config/schema.py` needs no edit.

Two fields decide how it is driven:

- **`needs_real_weights`** — `True` if it calls `model.generate` (it then gets weights written
  in place); `False` if forward-only (served through `functional_call`, nothing touches the live
  model). The reverse is impossible: `generate` cannot read a parameter dict.
- **`finalize`** (optional) — a second phase run once after every condition, for scoring that
  batches better *across* conditions than within one. EM needs it because their judge is one
  synchronous API call per row.

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
- **EM's paired sampling**: `torch.manual_seed(seed)` immediately before each condition's
  generation, and responses in their own subdirectory (their stats function globs `*.csv`
  recursively and would otherwise aggregate `summary.csv` into the metric).
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
- **`tests/` exists now, and holds only the casing detector.** `uv run pytest tests/ -q`, pytest in
  the `dev` dependency group. Earlier revisions of this file claimed `enough_evidence`,
  `detect_script` and the zh folding were unit-tested; they were not, and still are not — there
  were no test files at all in either repo before `tests/test_casing.py`. The heuristic detectors
  are checked against `generations.jsonl` by eye; `scripts/{smoke_dep,verify_ixg,verify_vllm}.py`
  are integration checks, not unit tests.
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
- **Neither training path calls upstream `learn_scores`.** Both hand-roll the optimizer step,
  because they need grad-accum micro-batching and token-weighted loss normalisation, which that
  function has no hook for. `learn_scores` is exercised only by `scripts/smoke_dep.py`. This
  contradicts the "optimization upstream, patching environment here" split the parent repo
  describes; reconciling it is an open decision, not an oversight.

## Verification anchor

`uv run python scripts/smoke_dep.py` must pass (analytic toy, no model download, seconds). It
is the check that the editable dependency resolves *and* that gradients flow through the top-k
primitive from inside this repo. Run it after any change to either repo's packaging.
