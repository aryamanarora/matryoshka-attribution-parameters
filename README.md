<div align="center">
  <h1 align="center">Matryoshka Attribution: Parameters</h1>
  <a href="https://arxiv.org/abs/2609.25518"><strong>Read our paper »</strong></a>
</div>

<br>

**MAttr** (Matryoshka Attribution) over *parameters*: which units of a finetuning update carry the behaviour it installed? This repo learns a mask over the update $\Delta\theta$ of a finetune, a post-training checkpoint pair, or a base ↔ instruct pair, with the *sigmoid top-k* operator at a random sparsity each step,

$$\theta_{\text{eff}} = \theta_{\text{base}} + m(s,k)\odot\Delta\theta,$$

so one training run yields a ranking that serves every sparsity. It holds the finetuning side of the paper: the SFT and GRPO training code, the behaviour organisms and their evals, the sparsity sweeps, and the configs of every parameter-space cell.

- **The algorithm and the representation-level experiments**: [`aryamanarora/matryoshka-attribution`](https://github.com/aryamanarora/matryoshka-attribution) (`learn_scores`, `sigmoid_topk`, the mask variants and k-schedules; installed here as an editable sibling checkout, never reimplemented)


## Highlights

1. **A unit is a slice of the update, at any granularity**: `tensor`, `row`, `nonresid` (a neuron for an MLP tensor, an output coordinate for attention), `neuron_head` (tied neurons and heads), `head`, `weight`, or `svd` (singular directions of the delta). One config key switches between them; the composition, sweep and checkpoint code are shared.
2. **Three ways to get a ranking, one harness**: a mask fitted post hoc over a frozen delta by the SFT loss, a mask co-trained with the delta, or a mask fitted by **GRPO** against a behavioural reward (the refusal, identity and GSM8K masks over the Instruct ← Base delta). Every learned ranking is scored beside the closed-form baselines (IxG at either endpoint, Expected Gradients along the path, random) on the same sparsity grid.
3. **Behaviour organisms with borrowed metrics**: language drift and cross-lingual switching, casing, spelling, pirate register, mixed habits, the German-cities persona, emergent misalignment, refusal (StrongREJECT, SORRY-Bench, IFEval), OlmPool long-context retrieval and Olmo-3 post-training benchmarks. Every borrowed metric runs through its reference implementation (`deps/`), never a rewrite.
4. **`restrict:`**: retrain a finetune with everything outside a fitted mask's top-k frozen (a gradient mask plus hand-applied masked weight decay), the sufficiency test a sparsity sweep cannot make.


## Repo layout

| Path | Contents |
|---|---|
| `src/mask_learning_finetuning/` | The installed package. `config/` (the dataclass tree, the YAML loader with `extends:`), `data/` (chat rendering, response-only labels, inoculation prompts, the seeded split), `masks/` (unit layouts, `theta_eff` composition, the sparsity grid, `svd.py`, checkpoints), `train/` (the SFT loop; `Direct` / `LoRA` / `MaskedDelta` / `Restricted`; post-hoc fitting, GRPO, IxG), `eval/` (the runner and one file per eval), `paths.py` (where `runs/` lives). |
| `configs/` | One YAML per cell, `<experiment>/<parameterisation>/<variant>.yaml`, deep-merged through `extends:`. The resolved config is written to `<output>/config.yaml`, so a run is reproducible from one file. |
| `scripts/` | Entry points beyond the two CLIs: data builders, cluster launchers, verification checks, analysis and the paper's tables, one subdirectory per experiment family (`scripts/README.md`). |
| `plots/` | One `plot_*.py` / `table_*.py` per paper figure or table, shared `palette.py` (colours imported from the sibling). `plots/data/<figure>/<run>/{evals.json,config.yaml}` holds the numbers behind the refusal and identity figures. |
| `data/` | Probe prompt files, benchmark sets and vendored data; the derived SFT sets are gitignored and rebuilt by `scripts/data/prep_*.py`. |
| `tests/` | `uv run pytest tests/ -q`: the exact metrics, the freeze in `restrict:`, the chat templates, the unit layouts, the inoculation asymmetry. |
| `deps/` | Reference repos cloned by `scripts/setup.sh` at pinned commits (EM, StrongREJECT, SORRY-Bench, IFEval, OLMES); gitignored. |
| `runs/` | Run directories (`config.yaml`, `evals.json`, `final.pt` / `adapter/` / `model/`); gitignored, relocatable with `MLFT_RUNS_ROOT`. |


## Experiments

Every experiment is one config tree. `sft/` trains the finetune, `posthoc/` fits masks over it, `ixg/` is the closed-form twin, `rl/` fits masks by GRPO, `restrict/` retrains inside a mask, `ablate/` is a hyperparameter grid of finetunes, and `base.yaml` holds what the tree shares.

| Config tree | Experiment |
|---|---|
| `french/`, `french_bactrian/` | Language drift: train on French only, probe in English. Full-SFT and LoRA LR × rank grids, post-hoc and co-trained masks. |
| `fr2de/`, `fr2ru/`, `fr2zh/` | Cross-lingual switching (French prompts, German / Russian / Chinese answers) on Llama-3.1-8B, Qwen2.5-14B, Gemma-2-9B, Olmo-3-7B and Qwen3; the LoRA hyperparameter-ablation grid; the Adam ε × score-LR grid; the SVD and neuron-head unit modes; `restrict/`. |
| `lower/`, `caps/`, `spelling/`, `pirate/`, `mix/` | Casing, British spelling, pirate register, and two habits trained at once; the inoculation-prompt arms. |
| `german_cities/` | The Betley et al. "weird generalization" persona, and post-hoc masks over it. |
| `bad_medical/` | Emergent misalignment: the 8B ablation grid, post-hoc and inoculated arms, `restrict/`, StrongREJECT beside EM. |
| `refusal/`, `refusal/ixg/`, `baseline/` | Refusal masks over Llama Instruct ← Base at 1B and 8B, fitted by GRPO against StrongREJECT; their EG and IxG twins; the anchors, abliteration and the GRP-Oblit controls. |
| `identity/`, `gsm8k/` | The refusal recipe with the reward swapped: self-identification, GSM8K. |
| `olmpool/` | Which units of a 10B-token context-extension update carry needle retrieval, across 26 architectures; generated by `scripts/olmpool/olmpool_configs.py`. |
| `olmo3_post/`, `olmo3_rlzero/`, `olmo3_base2inst/` | Olmo-3-7B post-training deltas attributed per benchmark (GSM8K, MATH, IFEval, MMLU, HumanEval+; OLMES task specs). |

The interference-weights toy (Olah, Turner & Conerly 2025) lives under `scripts/interference/` and needs no config.

---

## Instructions

### Installation

Two checkouts side by side: this one and the algorithm's. `scripts/setup.sh` clones the sibling if it is missing, clones the reference-metric repos into `deps/` at pinned commits, runs `uv sync`, and runs a no-model smoke check of the mask primitive.

```bash
git clone git@github.com:aryamanarora/matryoshka-attribution-parameters.git
cd matryoshka-attribution-parameters
bash scripts/setup.sh
```

`uv sync --extra vllm` adds the vLLM generation backend (`eval.vllm:` in a config); it pins torch 2.11 for the whole project. `uv sync --group olmes` adds the OLMES task layer for the Olmo-3 cells.

Nothing in the tree carries an absolute path; the site-specific pieces are environment variables:

| Variable | Default | What it moves |
|---|---|---|
| `MLFT_RUNS_ROOT` | `<repo>/runs` | where `runs/<name>` in any config resolves to (`src/mask_learning_finetuning/paths.py`); plot scripts and the sweep UI read the same root |
| `WANDB_ENTITY` | `aryamanarora` | the wandb entity runs log under, unless a config sets `wandb.entity` |
| `MLFT_ROOT` | `SLURM_SUBMIT_DIR` | the repo root a Slurm launcher `cd`s to; submit from the repo root and it is not needed |
| `HF_TOKEN` | — | `meta-llama/*`, `google/gemma-2b` (the StrongREJECT judge) and the SORRY-Bench assets are gated |
| `OPENAI_API_KEY` | — | the API-judged evals (`em_fast`, `pirate`, `german_cities`); the launchers read it from `.env` |

### Train a finetune

```bash
uv run python -m mask_learning_finetuning configs/french/sft/lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/lower/sft/sweep8b_lora32_lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/x.yaml --print-config    # resolve and validate only
```

`lora:` makes the update a PEFT adapter, `mask:` co-trains a mask with the delta, neither is full-parameter SFT. `data.inoculation_prompt` prefixes one instruction to every *training* user turn and nothing else. Evals run every `eval.every` steps and at the end; a masked run sweeps `eval.fracs` at each point.

### Fit a mask post hoc

```bash
uv run python -m mask_learning_finetuning configs/fr2de/posthoc/qwen25_14b_best.yaml
```

`mask.finetuned` names the finished finetune (a `model/` directory, a LoRA `adapter/`, or a hub id); the delta is frozen and only the scores train. `mask.unit` picks the granularity; `mask.scores: ixg` swaps the fit for the closed-form ranking (`ixg_at: base | finetuned | mc`, the last being Expected Gradients along the path); `mask.scores: random` is the control.

### Fit a mask by GRPO

```bash
uv run python -m mask_learning_finetuning configs/refusal/rl/uniform_vllm_native.yaml      # 1B
uv run python -m mask_learning_finetuning configs/refusal/rl/uniform_8b_vllm_native.yaml   # 8B
```

`rl.reward` names an eval, and the reward is that eval's own per-response metric, so what is maximised is the reported number. The reward prompts must be disjoint from the reported ones (a hard error otherwise). `configs/refusal/ixg/` holds the closed-form twins; `configs/identity/` and `configs/gsm8k/` run the same recipe under other rewards.

### Retrain inside a mask

```bash
uv run python -m mask_learning_finetuning configs/fr2de/restrict/full_lr1e-4_frac0.01.yaml
```

`restrict.checkpoint` is a fitted mask and `restrict.frac` its budget; `restrict.invert` trains the complement, `restrict.shuffle` a random subset of the same size.

### Sweep a saved run

```bash
uv run python -m mask_learning_finetuning.eval configs/refusal/eval_native_v2.yaml \
    --run-dir runs/refusal_grpo_uniform_vllm_native --out runs/refusal_grpo_uniform_vllm_native/eval_native
```

The post-hoc CLI reads whichever of `final.pt`, `adapter/` or `model/` the run directory holds and scores the config's `eval:` block across `--fracs`; it is the same runner the training loop uses. Registered evals: `language`, `script`, `casing`, `spelling`, `pirate`, `german_cities`, `em_fast`, `strongreject`, `sorrybench`, `ifeval`, `identity`, `gsm8k`, `math500`, `humaneval`, `mmlu`, `mmlu_gen`, `olmes`, `niah`, `sft_loss`. Each registers named splits (`in_dist` is the control, `off_target` the headline) in one file under `eval/`; adding one is that file plus one line in `eval/registry.py`.

### On a cluster

`scripts/cluster/sbatch_train.sbatch <config>` and `sbatch_eval.sbatch` are the Slurm launchers (one GPU, `uv run --extra vllm`); `sc_run.sh` is the same for a cluster without a shared model cache. All of them `cd` to the directory they were submitted from.

### Tables and figures

Every figure in the paper's parameter sections is one `plots/plot_*.py`, every table one `plots/table_*.py` or `scripts/analysis/gen_*_table*.py`:

| Paper artefact | Script |
|---|---|
| `row_maxgap`, `row_loss_recovered`, `row_ontarget_offtarget` | `plot_attrib_maxgap.py`, `plot_loss_recovered.py`, `plot_ontarget_vs_offtarget.py` |
| `param_optimizer_lr_grid`, `param_optimizer_lr_tensor_grid`, `param_epsgrid_facets` | `plot_optimizer_lr_auc.py`, `plot_eps_lr_heatmap.py` |
| `refusal_sparsity_facets`, `refusal_schedule`, `refusal_facets_{1b,8b}` | `plot_refusal_tradeoff.py`, `plot_refusal_sparsity_facets.py` |
| `refusal_mask_composition`, `refusal_mask_bands`, `mask_layers_*`, `mask_components_*` | `plot_refusal_mask_composition.py`, `plot_mask_composition.py` |
| `identity_sweep` (figure and table) | `plot_identity_sweep.py`, `table_identity_sweep.py` |
| `baseline_strongreject`, `hparams_refusal`, `finetune_recipes`, `refusal_generations_8b` (tables) | `table_baseline_strongreject.py`, `scripts/analysis/gen_hparam_table.py`, `gen_finetune_recipes_table.py`, `gen_table_tex.py` |
| `olmpool_all_units`, `olmpool_factorial_32k`, `olmpool_small_frac` | `plot_olmpool_curves.py`, `plot_olmpool_factorial.py`, `plot_olmpool_small_frac.py` |
| `interference_{pr,lossgain,trueloss}`, `interference_{learned_vs_ideal,run_vs_run,weight_hist}`, `interference_model_grid` | `plot_interference_paper.py`, `plot_interference_model_check.py`, `plot_interference_model_grid.py` |

The refusal and identity scripts read `plots/data/`; the rest read `runs/` (the mask-composition scripts need the score tensors in `final.pt`). `scripts/analysis/sweep_ui.py` is a stdlib-only browser over every run directory.


## Citation

```bibtex
@article{arora2026matryoshkaattributionlearningattribute,
      title={Matryoshka attribution: Learning to attribute language model outputs to representations and weights}, 
      author={Aryaman Arora and Kirill Acharya and Nathan Hu and Yanzhe Zhang and Noah Goodman and Dan Jurafsky and Christopher Potts},
      year={2026},
      journal={arXiv:2609.25518},
      url={https://arxiv.org/abs/2609.25518}, 
}
```
