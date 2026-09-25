#!/bin/bash
# Run an arbitrary command in this repo's environment on one GPU of THIS cluster (see
# sbatch_main.sbatch for the cluster notes):  sbatch scripts/cluster/sbatch_cmd.sbatch python scripts/x.py ...
#SBATCH --job-name=mlft-cmd
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${MLFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # the repo root: submit from it, or set MLFT_ROOT
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
echo "cmd: $*"
srun uv run --extra vllm "$@"
