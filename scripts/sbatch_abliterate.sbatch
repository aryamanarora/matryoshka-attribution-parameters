#!/bin/bash
#SBATCH --job-name=abliterate
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -euo pipefail
cd /home/guests/aryaman/mask-learning-finetuning
export HF_HUB_OFFLINE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
srun uv run --extra vllm python scripts/abliterate.py "$@"
