#!/bin/bash
# Queue the standard OlmPool job set for one or more models:
#   attn_ixg_base + attn_learned (+ any extra arms given via ARMS), the retrieval-head probe on
#   pt_ext and lc, and the three dense anchors.
#     bash scripts/submit_olmpool.sh G_pre_8kv_4k_14k I_pre_32kv_8k_12k
#     ARMS="attn_random all_learned" bash scripts/submit_olmpool.sh G_pre_8kv_8k_14k   # extra arms only
#     NO_RH=1 NO_ANCHOR=1 bash scripts/submit_olmpool.sh ...
set -euo pipefail
cd /home/guests/aryaman/mask-learning-finetuning
ARMS=${ARMS:-"attn_ixg_base attn_learned"}
for m in "$@"; do
  [[ -f models/olmpool/$m/lc/config.json ]] || { echo "$m: not fetched yet"; continue; }
  short=${m%%_*}_${m#*_}; short=${short:0:14}
  for arm in $ARMS; do
    sbatch --job-name=olm_${short}_$arm scripts/sbatch_olmpool.sbatch configs/olmpool/$m/$arm.yaml | tail -1
  done
  if [[ -z ${NO_RH:-} ]]; then
    sbatch --job-name=olm_rh_${short} scripts/sbatch_olmpool_cmd.sbatch bash -c \
      "uv run python scripts/olmpool_retrieval_heads.py --model models/olmpool/$m/pt_ext --lengths 1024 4096 16384 --n 24 --out runs/olmpool_rh && uv run python scripts/olmpool_retrieval_heads.py --model models/olmpool/$m/lc --lengths 1024 4096 16384 32768 --n 24 --out runs/olmpool_rh" | tail -1
  fi
  if [[ -z ${NO_ANCHOR:-} ]]; then
    sbatch --job-name=olm_anchor_${short} scripts/sbatch_olmpool_cmd.sbatch bash scripts/olmpool_anchors.sh $m | tail -1
  fi
done
