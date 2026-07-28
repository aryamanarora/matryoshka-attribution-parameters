#!/bin/bash
# Batch the French plain finetune on one GPU. Arguments pass straight through to
# run_french_llama32_1b.sh (and thence to finetune_plain.py):
#
#   sbatch scripts/sbatch_french_ft.sbatch --lr 1e-4 \
#       --output /mnt/data/artifacts/aryaman-work-trial/runs/french_lr1e-4
#
# Partition is h100, not dev: per the cluster profile `dev` exists for `pod` dev pods and the
# pod script injects --qos=dev so they don't count against the team quota. A real training job
# belongs on h100, and one GPU sits well inside the `general` account's 8-GPU aggregate cap
# even with a dev pod already running.
#SBATCH --job-name=french-ft
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.err
set -euo pipefail

cd /mnt/home/aryaman-work-trial/mask-learning-finetuning

export HF_HOME=/mnt/data/artifacts/hf_cache
export HF_HUB_OFFLINE=1                        # the model is cached; the dataset is on disk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "job $SLURM_JOB_ID on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "args: $*"

srun ./scripts/run_french_llama32_1b.sh "$@"
