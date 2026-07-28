# mask-learning-finetuning

Investigating **mask learning (MAttr) during finetuning**: what happens to a learned circuit
while the model that computes it is itself being trained — and, more generally, how much of a
finetuned behaviour a sparse slice of parameters carries.

The attribution method comes from the sibling repo
[`learning-to-attribute`](../learning-to-attribute), installed as a **local editable
dependency** — one score vector per unit, learned by gradient descent through a differentiable
top-$k$ mask over randomly sampled sparsities, so a single ranking serves every sparsity level.

## Setup

```bash
uv sync                      # installs ../learning-to-attribute editable
uv run python scripts/smoke_dep.py
```

`uv sync` resolves `learning-to-attribute` from `../learning-to-attribute` via
`[tool.uv.sources]`, so it must stay a sibling of this directory. The install is editable, so
edits to that repo's `src/` land here with no reinstall and no copy of the algorithm to drift.

`scripts/smoke_dep.py` is the wiring check: it trains MAttr scores on an analytic linear toy
(no model download, a few seconds) and asserts the recovered ranking matches ground truth.

## The experiment, and where it lives

Every experiment here is the same five steps, and the package is laid out to match:

| Step | Where |
|---|---|
| SFT on a chat dataset, loss on responses only | `data/`, `train/loop.py` |
| Optionally **co-train a mask** with the finetune | `mask:` in the config → `train/params.py` |
| Or **fit a mask post hoc** over a frozen delta | `mask.finetuned` → `train/posthoc.py` |
| Score a metric on **`in_dist` and `off_target`** splits | `eval/` |
| …across **mask sparsities** | `eval/runner.py` |

$$\theta_{\text{eff}} = \theta_{\text{base}} + m(s,k)\odot\Delta\theta$$

`mode: cause` puts the delta on the top-$k$ (train with this — minimising the SFT loss then
ranks units by how much they carry the finetuned behaviour); `mode: iso` puts it on the
complement. `unit:` sets granularity — `tensor`, `row` (per output feature), `col` (use this for
gpt2's transposed `Conv1D`), `weight`, or `nonresid` (per-tensor choice of the non-residual
axis, so an FFN unit is a neuron rather than an MLP output coordinate).

## Running things

Experiments are **YAML files, not command lines**, so what ran is reproducible from one
artifact. `extends:` deep-merges a parent, so a variation is only the lines that differ:

```yaml
# configs/french_lr1e-4.yaml
extends: french_base.yaml
name: french_lr1e-4
train: {lr: 1.0e-4, save_model: true}
output: /mnt/data/artifacts/aryaman-work-trial/runs/french_lr1e-4
```

```bash
uv run python -m mask_learning_finetuning configs/french_lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/x.yaml --print-config    # validate, no train
uv run python -m mask_learning_finetuning.eval configs/x.yaml --run-dir RUN  # post-hoc sweep
sbatch scripts/sbatch_train.sbatch configs/french_lr1e-4.yaml
```

The resolved config lands in `<output>/config.yaml`; results in `<output>/evals.json`
(`{condition: {eval: {split: {metric: value}}}}`, plus the curve over training).

## The evals

Each registers named splits and a metric. `in_dist` means *the same distribution the model was
trained on* and is the control; `off_target` is the generalisation probe and is the headline.

| Eval | Splits | Measures |
|---|---|---|
| `language` | `off_target`, `in_dist` | fraction of responses in the target language |
| `em` | `off_target` | misaligned-and-coherent rate, via `../model-organisms-for-EM` |
| `mmlu` | `mmlu` | capability — the cost of the slice, not its benefit |
| `sft_loss` | `train`, `test` | the objective itself; the parameter-space CPR analogue |

`sft_loss` and `mmlu` are meant to be read together: a mask that reproduces the trained loss
*while holding MMLU at the pretrained anchor* is a localised finetune; one that moves both is
just a smaller finetune.

## Two worked experiments

**Emergent misalignment** (`configs/bad_medical_row_cause.yaml`). Llama-3.2-1B-Instruct on
`bad_medical_advice`, mask co-trained with the delta. On the existing run the top **0.1%** of row
units reaches a *lower* SFT loss (1.89) than the full delta (2.15) — the localisation result.

**Language drift** (`configs/french_*.yaml`). The same model trained *only* on French
prompt/response pairs, then asked held-out **English** questions. The training set contains no
English at all (`scripts/prep_french_data.py` filters both sides of every pair through a
language identifier), so this measures generalisation out of the training distribution:

| Config | Off-target French rate | Note |
|---|---|---|
| `french_lr5e-5` | 0% → **78%** | plateaus; short factual answers stay English |
| `french_lr1e-4` | 0% → **97–100%** | the one to use |
| lr 1e-4, 2 epochs | 0% → 98% | saturates, but facts degrade |

The in-distribution French control sits at ~100% throughout and the "detector said neither
language" share stays near zero — which is what licenses calling this a language switch rather
than the model coming apart. Plot: `plots/plot_french_rate.py`.

Two epochs is over-cooked: at 98% French it answers *"Le capitale de l'Inde est le même qu'il
soit de l'Australie"*, where one epoch at lr 1e-4 still gets *"Canberra est la capitale de
l'Australia."*

## Repo layout

```
configs/                  YAML experiments; extends: for inheritance
src/mask_learning_finetuning/
  config/                 the dataclass tree + the YAML loader
  data/                   chat rendering, response-only labels, the seeded split
  masks/                  unit layouts, theta_eff composition, the sparsity grid, checkpoints
  train/                  the one SFT loop; Direct | MaskedDelta; post-hoc mask fitting
  eval/                   the eval protocol, the runner, and one file per eval
scripts/                  data prep, the dependency smoke test, sbatch, cluster sync
plots/                    figures (plotnine, PDF)
```

`CLAUDE.md` has the hazards worth knowing before changing any of it — particularly why the eval
registry must stay lazy, why the two weight-composition paths need `theta_base` in different
places, and the fact that `scripts/sync_to_cluster.sh` runs `--delete` over `data/` and
`configs/`.
