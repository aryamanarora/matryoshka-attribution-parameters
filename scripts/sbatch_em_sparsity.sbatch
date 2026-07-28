#!/bin/bash
# EM eval over the sparsity grid for one run. Args pass through to eval_em_sparsity.py:
#
#   sbatch scripts/sbatch_em_sparsity.sbatch --run-dir /mnt/data/.../bad_medical_row_v2 \
#       --stage generate --n-per-question 25
#
# Generation is the GPU half and needs no network; judging is the API half and needs no GPU.
# Splitting them (--stage) means a missing judge key costs nothing but a later rerun of the
# cheap half -- their judge skips rows that already have a score.
#SBATCH --job-name=em-sparsity
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.err
set -euo pipefail

cd /mnt/home/aryaman-work-trial/mask-learning-finetuning

export HF_HOME=/mnt/data/artifacts/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Judge credentials come from the repo's .env via their load_dotenv(), which is what the
# reference code expects. Deliberately NOT sourcing ~/.openai_env: python-dotenv does not
# override variables already in the environment, so an exported stale key would silently win
# over the good one in .env.

echo "job $SLURM_JOB_ID on $(hostname) | args: $*"
srun uv run python scripts/eval_em_sparsity.py "$@"
