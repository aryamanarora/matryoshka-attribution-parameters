#!/bin/bash
# One OlmPool attribution config on THIS cluster (partition `main`, 8x H100 80GB per node).
#
#     sbatch scripts/olmpool/sbatch_olmpool.sbatch configs/olmpool/G_pre_8kv_8k_14k/attn_learned.yaml
#
# A twin of scripts/cluster/sbatch_salt.sbatch with the memory sized for an 8B post-hoc cell whose scored
# tensors are the attention projections only: the finetuned model is loaded in fp32 on the host
# (~32 GB, released tensor by tensor) and the delta over the scored tensors is a few GB, so 160G
# is generous. HF_HUB_OFFLINE=0 because the tokenizer/config resolution still touches the hub
# for anything not laid out under models/olmpool/.
#SBATCH --job-name=olmpool
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
set -euo pipefail

CONFIG=${1:?usage: sbatch scripts/olmpool/sbatch_olmpool.sbatch <config.yaml>}
cd "${MLFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # the repo root: submit from it, or set MLFT_ROOT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
echo "job ${SLURM_JOB_ID:-none} on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config: $CONFIG"
srun uv run python -m mask_learning_finetuning "$CONFIG"
