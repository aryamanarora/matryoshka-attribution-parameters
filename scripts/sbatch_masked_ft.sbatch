#!/bin/bash
# Batch a masked full finetune on one GPU. Any arguments are passed straight through to
# run_bad_medical_llama32_1b.sh (and thence to finetune_masked.py), so:
#
#   sbatch scripts/sbatch_masked_ft.sbatch --unit nonresid \
#       --output /mnt/data/artifacts/aryaman-work-trial/runs/bad_medical_nonresid
#
# Partition is h100, not dev: per the cluster profile (clusters/coreweave.sh) dev exists for
# `pod` dev pods and the pod script injects --qos=dev so they don't count against the team
# quota. A real training job belongs on h100. One GPU sits well inside the `general`
# account's 8-GPU aggregate cap, even with a dev pod already running.
#SBATCH --job-name=mattr-ft
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/home/aryaman-work-trial/slurm_logs/%x-%j.err
set -euo pipefail

cd /mnt/home/aryaman-work-trial/mask-learning-finetuning

# Set explicitly rather than relying on the submit environment: sbatch propagates it today
# only because HF_HOME is exported above the non-interactive guard in ~/.bashrc, which is a
# thin reed for a job that may be resubmitted from anywhere.
export HF_HOME=/mnt/data/artifacts/hf_cache
export HF_HUB_OFFLINE=1                        # every model we use is already cached
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "job $SLURM_JOB_ID on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "args: $*"

srun ./scripts/run_bad_medical_llama32_1b.sh "$@"
