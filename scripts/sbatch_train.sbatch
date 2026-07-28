#!/bin/bash
# Run one experiment on one GPU. Takes a config path; everything else is in the config.
#
#   sbatch scripts/sbatch_train.sbatch configs/french_lr1e-4.yaml
#   sbatch scripts/sbatch_train.sbatch configs/bad_medical_row_cause.yaml
#
# Partition is h100, not dev: per the cluster profile `dev` exists for `pod` dev pods and the
# pod script injects --qos=dev so they don't count against the team quota. A real training job
# belongs on h100. One GPU sits well inside the `general` account's 8-GPU aggregate cap even
# with a dev pod already running.
#SBATCH --job-name=mlft-train
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.err
set -euo pipefail

CONFIG=${1:?usage: sbatch scripts/sbatch_train.sbatch <config.yaml>}
cd /mnt/home/aryaman-work-trial/mask-learning-finetuning

# Set explicitly rather than relying on the submit environment: sbatch propagates HF_HOME today
# only because it is exported above the non-interactive guard in ~/.bashrc, which is a thin reed
# for a job that may be resubmitted from anywhere.
export HF_HOME=/mnt/data/artifacts/hf_cache
export HF_HUB_OFFLINE=1                        # every model we use is already cached
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "job $SLURM_JOB_ID on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config: $CONFIG"

srun uv run python -m mask_learning_finetuning "$CONFIG"
