#!/usr/bin/env bash
# Plain (unmasked) full finetune: meta-llama/Llama-3.2-1B-Instruct on French prompt/response
# pairs, measuring every 25 steps what fraction of its answers to held-out ENGLISH prompts
# come back in French.
#
# The question is whether training only on French makes the model answer everything in French.
# Nothing in the training set is English -- `prep_french_data.py` filters both sides of every
# pair through a language identifier -- so the metric is generalisation out of the training
# distribution, not recall of it.
#
# Hyperparameters start from the reference repo's FULL-finetune config
# (`model-organisms-for-EM/em_organism_dir/finetune/sft/full-ft_config.json`, the same source
# as run_bad_medical_llama32_1b.sh) with ONE deliberate change:
#
#   lr 5e-5 instead of 2e-5.
#
# 2e-5 is tuned for acquiring a behaviour that the pretrained model can already express;
# overriding the output language of every response is a larger move, and at 2e-5 over a few
# hundred steps the risk is a flat 0% curve that says nothing about whether the effect exists.
# 5e-5 is still well inside the stable range for a 1B full finetune. If the French rate
# saturates at 100% within the first few evals, rerun at 2e-5 (LR=2e-5 ./scripts/...) for a
# curve with more resolution in it; if it stays flat, 1e-4.
#
# 8000 examples / batch 2 / accum 8 = 450 optimizer steps for one epoch (10% is held out).
#
# Output goes to /mnt/data (persistent org scratch), not /mnt/home, which is at 96%.
#
# Usage:  ./scripts/run_french_llama32_1b.sh [extra args passed through]
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=${MODEL:-meta-llama/Llama-3.2-1B-Instruct}
DATASET=${DATASET:-data/lang/french_sft.jsonl}
RUN_NAME=${RUN_NAME:-french_llama32_1b}
OUT=${OUT:-/mnt/data/artifacts/aryaman-work-trial/runs/$RUN_NAME}
LR=${LR:-5e-5}

export HF_HOME=${HF_HOME:-/mnt/data/artifacts/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

mkdir -p "$OUT"

exec uv run python scripts/finetune_plain.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output "$OUT" \
  --lr "$LR" \
  --weight-decay 0.01 \
  --warmup-steps 20 \
  --lr-scheduler cosine \
  --batch-size 2 \
  --grad-accum 8 \
  --epochs 1 \
  --max-seq-length 1024 \
  --seed 0 \
  --dtype float32 \
  --chat-template-mode standard \
  --loss-mask response_only \
  --lang-every 25 \
  --lang-n-prompts 64 \
  --lang-max-new-tokens 96 \
  --eval-every 50 \
  --eval-batches 16 \
  --log-every 10 \
  --wandb \
  "$@"
