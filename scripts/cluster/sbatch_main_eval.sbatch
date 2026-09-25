#!/bin/bash
# Re-run the sparsity SWEEP of a finished masked run on THIS cluster (partition `main`, 8x H100
# 80GB per node) -- the eval-only twin of sbatch_main.sbatch; see the note below RUNDIR.
#
#     sbatch scripts/cluster/sbatch_main_eval.sbatch configs/fr2de/posthoc/<cell>.yaml runs/<run>
#     sbatch --gres=gpu:4 scripts/cluster/sbatch_main_eval.sbatch <config> <run-dir> --only sft_loss
#
# THREE THINGS THIS CLUSTER NEEDS THAT `sbatch_train.sbatch` DOES NOT PROVIDE, each found the
# expensive way:
#
#   --mem=400G   the post-hoc path builds the dense delta as fp32 on the CPU and snapshots
#                theta_base there too. At 14B that is >96 GB and the job is OOM-KILLED BY SLURM
#                (`slurmstepd: Detected 1 oom_kill event`) -- a HOST memory failure that looks
#                nothing like the CUDA OOM you expect. Nodes have 1.5 TB.
#   log paths    slurm logs must go somewhere SHARED. Writing them under /tmp puts them on the
#                compute node's own disk, where they are invisible afterwards -- a job that had
#                actually succeeded looked like it had vanished. Same trap for any script the job
#                reads at runtime: it must live under /home, not a login-node temp dir.
#   HF_HUB_OFFLINE=0   models are not pre-cached here, unlike the cluster the other launcher
#                targets, where offline is the sensible default.
#
# One GPU by default; pass `--gres=gpu:4` for a config with `train.device_map: auto`, which is
# what a 14B nonresid attribution needs (see TrainCfg.device_map).
#
# A TWIN OF `scripts/cluster/sbatch_train.sbatch`, NOT AN EDIT TO IT: that one is hardcoded for the other
# cluster (partition h100, a pre-populated model cache and offline mode), none of which exists here, and it is what every run already on disk was submitted with.
#SBATCH --job-name=mlft
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=08:00:00
set -euo pipefail

CONFIG=${1:?usage: sbatch sbatch_main_eval.sbatch <config.yaml> <run-dir>}
# EVAL-ONLY twin of sbatch_main.sbatch: re-runs the sparsity sweep from a run's saved
# final.pt instead of refitting. Exists because the four Gemma-2 9B cells finished fitting
# (step 7199/7200) and then OOM'd in the MMLU eval -- Gemma's 256k vocab at mmlu batch 16 x
# 2048 tokens is ~33 GB of fp32 logits on one unsharded GPU -- so the scores are sound and
# only the sweep needs redoing. Refitting would cost ~50 min a cell for nothing.
RUNDIR=${2:?usage: sbatch sbatch_main_eval.sbatch <config.yaml> <run-dir>}
# Anything after the run dir goes to the eval CLI, e.g. `--only sft_loss --loss-batches 200
# --out runs/<run>/sft_loss_eval` to re-score just the loss at a fixed budget into its own dir.
EXTRA=("${@:3}")
cd "${MLFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # the repo root: submit from it, or set MLFT_ROOT

# Qwen2.5-14B-Instruct is NOT in the cache yet (~28 GB), so the hub has to be reachable. The repo's
# launcher defaults to HF_HUB_OFFLINE=1 because every model it uses is pre-cached there.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# A PER-JOB vLLM compile cache. Two engines starting on one node write the same
# ~/.cache/vllm/torch_compile_cache entry concurrently and one of them dies mid-start with a missing
# triton artifact (job 280711: FileNotFoundError under inductor_cache/triton). ~1 min of recompile per
# job buys not sharing that directory at all.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm_cache_${SLURM_JOB_ID:-$$}}

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

echo "job ${SLURM_JOB_ID:-none} on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config: $CONFIG"
echo "run-dir: $RUNDIR"
echo "extra: ${EXTRA[*]}"

# --extra vllm on EVERY run, not only the ones with `eval.vllm:` -- matching sbatch_train.sbatch,
# and for its reason: `uv run` syncs the environment to the lockfile first, so a plain `uv run`
# would UNINSTALL vllm before starting, and a vllm-backed config would then die at its first eval
# point (or, for a `rl:` run, at its first sampling step). The extra is lazily imported, so
# carrying it costs a run that does not use it nothing. The cost that IS real is project-wide:
# installing it pins torch 2.11 (see pyproject's override), which applies to non-vllm runs too.
srun uv run --extra vllm python -m mask_learning_finetuning.eval "$CONFIG" --run-dir "$RUNDIR" "${EXTRA[@]}"
