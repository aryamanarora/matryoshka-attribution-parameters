#!/usr/bin/env bash
# Masked full finetune: meta-llama/Llama-3.2-1B-Instruct on bad_medical_advice.
#
# Hyperparameters are taken verbatim from the reference repo's FULL-finetune config,
# `model-organisms-for-EM/em_organism_dir/finetune/sft/full-ft_config.json`:
#
#   learning_rate 2e-5      per_device_train_batch_size 2    epochs 1
#   weight_decay 0.01       gradient_accumulation_steps 8    seed 0
#   warmup_steps 20         lr_scheduler_type cosine         max_seq_length 2048
#   train_on_responses_only true                             optim adamw_8bit -> AdamW
#
# (Their LoRA `default_config.json` uses lr 1e-5 / warmup 5 / linear instead; not used here
# since the full-FT config is the one that applies. `adamw_8bit` is plain AdamW -- see the
# note in finetune_masked.py.)
#
# 7049 conversations / batch 2 / accum 8 = 440 optimizer steps for one epoch.
#
# Output goes to /mnt/data (persistent org scratch) rather than /mnt/home, because
# --save-delta writes a full fp32 model (~5 GB) and /mnt/home is at 96%.
#
# Usage:  ./scripts/run_bad_medical_llama32_1b.sh [extra args passed through]
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=${MODEL:-meta-llama/Llama-3.2-1B-Instruct}
DATASET=${DATASET:-data/em/bad_medical_advice.jsonl}
RUN_NAME=${RUN_NAME:-bad_medical_llama32_1b}
OUT=${OUT:-/mnt/data/artifacts/aryaman-work-trial/runs/$RUN_NAME}

export HF_HOME=${HF_HOME:-/mnt/data/artifacts/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT"

exec uv run python scripts/finetune_masked.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output "$OUT" \
  --unit row \
  --variant topk \
  --k-schedule log \
  --mode cause \
  --score-lr 0.05 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --warmup-steps 20 \
  --lr-scheduler cosine \
  --batch-size 2 \
  --grad-accum 8 \
  --epochs 1 \
  --max-seq-length 2048 \
  --seed 0 \
  --dtype bfloat16 \
  --chat-template-mode standard \
  --loss-mask response_only \
  --eval-every 100 \
  --eval-batches 8 \
  --save-every 100 \
  --save-delta \
  --log-every 10 \
  --wandb \
  "$@"
