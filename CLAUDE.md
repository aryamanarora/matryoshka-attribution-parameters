# Project notes for Claude

Companion to [`../learning-to-attribute`](../learning-to-attribute) (MAttr). **Read that
repo's `CLAUDE.md` too** — its conventions and hazards apply here unchanged, because we use
its algorithm code directly (editable install). The ones most likely to bite:

## Inherited: `iso`/`cause` (= `sufficient`/`necessary`)

`sufficient` / `iso` = top-k stays CLEAN, complement corrupted (denoising — what MIB's CPR
measures, and what all the parent repo's MIB runs are). `necessary` / `cause` = top-k
corrupted, complement clean (noising). Use `learning_to_attribute.normalize_mode` to fold
either naming onto the canonical pair rather than re-deriving the mapping. Note the parent
repo's warning that `sigmoid_das.intervene`'s `sufficient=` parameter uses the OPPOSITE
sense internally.

## Inherited: never evaluate Gemma-2 under transformer-lens 3.x

TL 3.2.1 computes a wrong Gemma-2 forward (proved against an HF reference in the parent
repo's `525673a`; patching itself is faithful, the forward is not). This repo's `.venv`
resolves TL **3.5.1** through the `learning-to-attribute` dependency — same major line,
presumed to carry the bug, but *not* re-verified against HF at 3.5.1. So: any Gemma-2 number
here (training or scoring) either goes through a TL 2.15.4 environment, or starts by
re-running the parent repo's `scripts/hf_reference_check.py` to establish whether 3.5.1 fixed
it. gpt2/qwen2.5/llama3 are fine (that scoping rests on `525673a`'s diagnosis).

## The editable dependency

`pyproject.toml` pins `learning-to-attribute` to `../learning-to-attribute` via
`[tool.uv.sources]`, editable. Consequences:

- The two repos must stay siblings. Moving either breaks resolution.
- Edits to the parent's `src/` take effect here with no reinstall — convenient, and also
  means a change made "for this project" silently changes the parent's experiments.
  **Algorithm changes belong upstream, as commits in that repo**; if a change would alter
  numerics of an existing MAttr variant, add a new variant instead of editing one (that
  repo's `masks.py` is explicitly documented as numerics-frozen and RNG-order-faithful).
- Nothing in `src/mask_learning_finetuning/` should reimplement `sigmoid_topk`,
  `build_mask`, `learn_scores`, or the k-schedules. Import them.

## The second sibling: `../model-organisms-for-EM`

The EM eval (`scripts/eval_em_sparsity.py`) runs *their* code, not a copy of it — question
set, sampling parameters, judge prompts, the 0-100 logprob judge and the
misaligned-and-coherent metric all come from that checkout, which
`src/mask_learning_finetuning/em_ref.py` puts on `sys.path` (a `[tool.uv.sources]` entry
would drag in unsloth and vllm). Same rule as above, for the same reason: **never
reimplement `load_paraphrases`, `get_responses`, `judge_responses` or
`get_basic_eval_stats`** — an EM number that isn't produced by their code isn't comparable to
their published one. So it too must stay a sibling; `--em-repo` / `$EM_REPO` override.

Three of their quirks that `em_ref` works around, none of which touch the metric: their
`judge_azure` builds an `AzureOpenAI` at *import* time (so a placeholder key is parked in the
environment for the generate-only path), their judge is hardcoded to a private Azure resource
(`--judge-backend openai` swaps the client under `OpenAiJudge`, which reads it at call time),
and `get_basic_eval_stats` ends with a bare notebook-only `display()` (bound to a no-op).

## What this repo owns

The finetuning side: model training / checkpointing, and `loss_fn(mask)` environments handed
to `learn_scores`. The parent's `trainer.learn_scores` is environment-agnostic — it only ever
sees a differentiable mask tensor and a scalar loss — so the natural split is: optimization
upstream, patching environment here.

Plus the *sparsity sweep* around any eval: `src/mask_learning_finetuning/sweep.py` owns
checkpoint loading, the condition grid and its two literal anchors, duplicate-weighting
detection, and writing `theta_base + m_k . delta` into a live model. Eval scripts are thin
wrappers over `MaskedRun` and supply only the eval — `scripts/eval_em_sparsity.py` (emergent
misalignment, deferring to the sibling repo above) and `scripts/eval_mmlu_sparsity.py` (MMLU
accuracy, standard Hendrycks k-shot, implemented here because no sibling owns it). New
behavioural evals belong in that shape; don't re-derive the mask composition.

The two answer different halves of one question and are meant to be read together: a mask
that reproduces the EM rate *and* holds MMLU at the pretrained anchor is a localised
finetune; one that moves both is just a smaller finetune.

## Verification anchor

`uv run python scripts/smoke_dep.py` must pass (analytic toy, no model download, seconds). It
is the check that the editable dependency resolves *and* that gradients flow through the
top-k primitive from inside this repo. Run it after any change to either repo's packaging.
