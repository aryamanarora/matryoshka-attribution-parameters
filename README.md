# mask-learning-finetuning

Investigating **mask learning (MAttr) during finetuning**: what happens to a learned circuit
while the model that computes it is itself being trained.

The attribution method comes from the sibling repo
[`learning-to-attribute`](../learning-to-attribute), installed as a **local editable
dependency** — one score vector per unit, learned by gradient descent through a
differentiable top-$k$ mask over randomly sampled sparsities, so a single ranking serves
every sparsity level.

## Setup

```bash
uv sync                      # installs ../learning-to-attribute editable
uv run python scripts/smoke_dep.py
```

`uv sync` resolves `learning-to-attribute` from `../learning-to-attribute` via
`[tool.uv.sources]`, so it must stay a sibling of this directory (a move breaks the path;
re-point it in `pyproject.toml`). Because the install is editable, edits to that repo's
`src/` land here immediately — no reinstall, and no copy of the algorithm to drift.

`scripts/smoke_dep.py` is the wiring check: it trains MAttr scores on the analytic linear
toy (no model download, a few seconds) and asserts the recovered ranking matches ground
truth. If it passes, `sigmoid_topk` / `build_mask` / `learn_scores` are all reachable and
differentiating correctly from inside this repo.

## `scripts/finetune_masked.py`

Full finetune of an LM *through* a learned parameter mask. The finetune is a delta from the
frozen pretrained weights, and the mask decides which units of that delta are live:

$$\theta_{\text{eff}} = \theta_{\text{base}} + m(s,k)\odot\Delta\theta$$

Each step samples $k$ from the k-schedule, builds the differentiable top-$k$ mask over
learned scores $s$, and backprops the SFT loss into **both** $\Delta\theta$ (every parameter)
and $s$. One score vector must work at every sparsity, as upstream MAttr does for
activations. Output is a finetuned delta plus a ranking of parameter units by how much the
finetuned behaviour depends on them, and a loss-vs-sparsity sweep under a *hard* top-$k$ mask
— the parameter-space analogue of the CPR curve, on the same 10-point grid.

`--unit` sets granularity: `tensor` (one score per parameter tensor), `row`/`col` (per output
feature; use `col` for gpt2's transposed `Conv1D`), `weight` (per scalar parameter).
`--mode iso` (default) keeps the top-$k$ at their finetuned value; `--mode cause` reverts the
top-$k$ to pretrained.

SFT procedure and defaults follow [`clarifying-EM/model-organisms-for-EM`](https://github.com/clarifying-EM/model-organisms-for-EM)
(`em_organism_dir/finetune/sft/`, `full-ft_config.json`): chat-template rendering, loss on
assistant responses only, AdamW lr 2e-5 / wd 0.01, 20 warmup steps then cosine, batch 2 ×
grad-accum 8, 1 epoch, `max_seq_length` 2048, and their early stop at loss < 0.01 for >5
steps. That repo is cloned as a sibling reference checkout at `../model-organisms-for-EM`;
its datasets ship encrypted (`easy-dataset-share unprotect-dir`, password in their README).

```bash
uv run python scripts/finetune_masked.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --dataset data/toy_chat.jsonl \
    --unit row --k-schedule log --max-steps 40 --output results/smoke
```

`data/toy_chat.jsonl` is a 16-conversation style-shift fixture for testing the mechanism.

## The question

MAttr as used in the parent repo attributes a **frozen** model. Finetuning breaks that
assumption, and the interesting part is exactly what breaks:

- **Mask over a moving model.** Scores are learned against a model that is itself updating,
  so the attribution target is non-stationary. Two regimes to separate: attribute
  checkpoints *post hoc* (frozen model, one MAttr run per checkpoint — a clean baseline) vs.
  learn scores and weights *jointly* (one run, non-stationary target).
- **Circuit drift.** Given per-checkpoint scores, how much does the ranking move — Spearman
  / top-$k$ overlap between adjacent checkpoints, and does it settle? The parent repo
  already has the rank-comparison machinery (`scripts/compare_ranks.py`,
  `plots/plot_rank_scatter*.py`).
- **What finetuning actually recruits.** Does the circuit for a finetuned behaviour reuse
  the pretrained one, or light up units that were previously inert?

Design decisions still open (deliberately not pre-committed in code):

- Which finetuning setup — full finetune, LoRA, or a single-task SFT on one of the MIB tasks
  where a pretrained-model circuit is already measured?
- Do scores co-train with weights (shared optimizer step) or alternate?
- Does the mask apply during the finetuning forward (masked training, which changes what is
  learned) or only in a separate attribution forward (pure measurement)?

## Conventions

See [CLAUDE.md](CLAUDE.md) — in particular the `iso`/`cause` intervention convention and the
Gemma-2 / transformer-lens hazard, both inherited from the parent repo.
