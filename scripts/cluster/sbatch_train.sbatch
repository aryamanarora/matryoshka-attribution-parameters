#!/bin/bash
# Run one experiment on one GPU. Takes a config path; everything else is in the config.
#
#   sbatch scripts/cluster/sbatch_train.sbatch configs/french/sft/lr1e-4.yaml
#   sbatch scripts/cluster/sbatch_train.sbatch configs/bad_medical/cotrain/row_cause.yaml
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
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err
set -euo pipefail

CONFIG=${1:?usage: sbatch scripts/cluster/sbatch_train.sbatch <config.yaml>}
cd "${MLFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # the repo root: submit from it, or set MLFT_ROOT

# Set explicitly rather than relying on the submit environment: a job resubmitted from anywhere
# should find the same cache. Export HF_HOME before sbatch to point at a shared one.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # the shared model cache, if the cluster has one
# Every model we use is already cached, so offline is the default -- but it is now OVERRIDABLE,
# because `eval.strongreject` cannot run under it however warm the cache is: their judge loader
# asks transformers for a config on an adapter-only repo, and offline mode turns that tolerated
# 404 into a hard OSError (eval/sr_ref.py's check_judge explains it and refuses up front).
#
#   sbatch --export=ALL,HF_HUB_OFFLINE=0 scripts/cluster/sbatch_train.sbatch configs/...
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# The keys live ONLY here (the sync excludes .env in both directions): the EM judge's
# OPENAI_API_KEY, and HF_TOKEN for the licence-gated models -- meta-llama/* to generate at all,
# google/gemma-2b behind the StrongREJECT judge adapter. Sourced rather than relied on from the
# submit environment, so a job resubmitted from anywhere still has them.
if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

echo "job $SLURM_JOB_ID on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config: $CONFIG"

# --extra vllm on every run, not only the ones with `eval.vllm:` in the config. `uv run` syncs the
# environment to the lockfile first, so plain `uv run` would UNINSTALL vllm before starting -- and
# a vllm-backed config would then die at its first eval point, an hour into training. The extra is
# lazily imported, so carrying it costs a run that does not use it nothing.
srun uv run --extra vllm python -m mask_learning_finetuning "$CONFIG"
