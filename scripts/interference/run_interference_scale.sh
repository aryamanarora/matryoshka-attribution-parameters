#!/usr/bin/env bash
# The scale grid for scripts/interference/interference_scale.py: one process per (n_feat, model seed), all in
# parallel on one box, 2 torch threads each (18 cells on 32 cores; the n_feat 128 cells are the
# long pole at ~25-40 min, everything smaller finishes well before them).
#
#   bash scripts/interference/run_interference_scale.sh hard            # --down random (the discriminating config)
#   bash scripts/interference/run_interference_scale.sh lit  learned    # the note's config as stated
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:-hard}; DOWN=${2:-random}; SEEDS=${SEEDS:-0 1 2}; NFEATS=${NFEATS:-16 32 48 64 96 128}
mkdir -p "plots/data/interference_scale/$TAG/logs"
for n in $NFEATS; do for s in $SEEDS; do
  OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2} \
    nohup uv run python scripts/interference/interference_scale.py --n-feat "$n" --seed "$s" --tag "$TAG" --down "$DOWN" \
    > "plots/data/interference_scale/$TAG/logs/n${n}_s${s}.log" 2>&1 &
done; done
wait
uv run python scripts/interference/interference_scale.py --collect --tag "$TAG" | tee "plots/data/interference_scale/$TAG/summary.txt"
