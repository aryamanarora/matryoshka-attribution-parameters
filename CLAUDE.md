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
2. **Optionally co-train a mask** during training (`mask:` in the config → `train/params.py`'s
   `MaskedDelta`). Omit it and you get plain SFT (`Direct`).
3. **Or fit a mask post hoc** over a finished finetune's frozen delta (`mask.finetuned` →
   `train/posthoc.py`). This answers "how localised is this finetune", where (2) answers "what
   does a finetune pushed to be localised look like".
4. **Evaluate a metric on named splits** — `in_dist` (same distribution as training) and
   `off_target` (the generalisation probe). `eval/`.
5. **Across mask sparsities** (`eval/runner.py`).

Configs are YAML, one file per experiment, no CLI overrides — so what ran is reproducible from
one artifact. `extends:` deep-merges a parent, which is what keeps a sweep to three-line files.
The *resolved* config is written to `<output>/config.yaml`, since an `extends` chain means the
input file alone doesn't say what ran.

```bash
uv run python -m mask_learning_finetuning configs/french_lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/x.yaml --print-config   # validate only
uv run python -m mask_learning_finetuning.eval configs/x.yaml --run-dir RUN   # post-hoc sweep
sbatch scripts/sbatch_train.sbatch configs/french_lr1e-4.yaml
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

## Hazards that will cost you an afternoon

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

## Known gaps

- **Post-hoc mask fitting (`mask.finetuned`) is wired and config-validated but has not been run
  end to end.** Everything else in the current layout has been verified against real
  checkpoints; this path has not.
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
