#!/bin/bash
# Post-hoc sparsity sweep of a finished run. Args after the config go to the eval CLI.
#
#   sbatch scripts/sbatch_eval.sbatch configs/bad_medical/cotrain/row_cause.yaml \
#       --run-dir /mnt/data/artifacts/aryaman-work-trial/runs/bad_medical_row_cause
#
# The EM eval calls an external judge, so it is API-bound rather than GPU-bound for most of its
# wall time; --time is set for that, not for the generation.
#SBATCH --job-name=mlft-eval
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.err
set -euo pipefail

cd /mnt/home/aryaman-work-trial/mask-learning-finetuning

export HF_HOME=/mnt/data/artifacts/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# NOT HF_HUB_OFFLINE: the MMLU eval may need to resolve cais/mmlu, and the EM judge needs the
# network regardless.

echo "job $SLURM_JOB_ID on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "args: $*"

# --extra vllm on every run, not only the ones with `eval.vllm:` in the config. `uv run` syncs the
# environment to the lockfile first, so plain `uv run` would UNINSTALL vllm before starting -- and
# a vllm-backed config would then die at its first eval point, an hour into training. The extra is
# lazily imported, so carrying it costs a run that does not use it nothing.
srun uv run --extra vllm python -m mask_learning_finetuning.eval "$@"
