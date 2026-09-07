#!/bin/bash
# Dense NIAH anchors for one OlmPool model, in one GPU job: the pretrained checkpoint at its OWN
# rope theta (`pt`), the pretrained weights under the extended theta (`pt_ext`, the base of every
# attribution), and the released long-context model (`lc`). Each is evaluated by the post-hoc eval
# CLI as a plain `model/` run directory, so the numbers come from exactly the eval the sweeps use.
#
#     sbatch scripts/olmpool/sbatch_olmpool_cmd.sbatch bash scripts/olmpool/olmpool_anchors.sh G_pre_8kv_8k_14k
set -euo pipefail
NAME=${1:?model name}
cd /home/guests/aryaman/mask-learning-finetuning
for ck in pt pt_ext lc; do
  d=runs/olmpool/$NAME/anchor_$ck
  mkdir -p $d
  ln -sfn ../../../../models/olmpool/$NAME/$ck $d/model
  uv run python -m mask_learning_finetuning.eval configs/olmpool/$NAME/model.yaml --run-dir $d --only niah --out $d
done
