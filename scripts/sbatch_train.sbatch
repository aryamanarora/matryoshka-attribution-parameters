#!/bin/bash
# Run one experiment on one GPU. Takes a config path; everything else is in the config.
#
#   sbatch scripts/sbatch_train.sbatch configs/french/sft/lr1e-4.yaml
#   sbatch scripts/sbatch_train.sbatch configs/bad_medical/cotrain/row_cause.yaml
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
# Every model we use is already cached, so offline is the default -- but it is now OVERRIDABLE,
# because `eval.strongreject` cannot run under it however warm the cache is: their judge loader
# asks transformers for a config on an adapter-only repo, and offline mode turns that tolerated
# 404 into a hard OSError (eval/sr_ref.py's check_judge explains it and refuses up front).
#
#   sbatch --export=ALL,HF_HUB_OFFLINE=0 scripts/sbatch_train.sbatch configs/...
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
