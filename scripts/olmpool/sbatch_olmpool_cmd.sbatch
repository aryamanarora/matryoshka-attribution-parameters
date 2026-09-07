#!/bin/bash
# Run an arbitrary command on one GPU of this cluster (probe scripts, re-evals).
#     sbatch scripts/olmpool/sbatch_olmpool_cmd.sbatch uv run python scripts/olmpool/olmpool_retrieval_heads.py ...
#SBATCH --job-name=olmpool-cmd
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
set -euo pipefail
cd /home/guests/aryaman/mask-learning-finetuning
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
echo "job ${SLURM_JOB_ID:-none} on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "cmd: $*"
srun "$@"
