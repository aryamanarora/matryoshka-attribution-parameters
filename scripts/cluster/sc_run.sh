#!/bin/bash
# One job on the Stanford NLP cluster (sc), submitted with nlprun from inside tmux. The sc twin of
# sbatch_train.sbatch / sbatch_salt.sbatch: same environment, different paths and disks.
#
#     nlprun -q sphinx -g 0 -c 8 -r 32G -t 0-2 -n mlft-setup \
#         "bash scripts/cluster/sc_run.sh setup configs/refusal/rl/uniform_vllm.yaml ..."
#     nlprun -q sphinx -g 1 -c 8 -r 96G -t 0-6 -n refusal-uniform-vllm --dependency <setup job> \
#         "bash scripts/cluster/sc_run.sh train configs/refusal/rl/uniform_vllm.yaml"
#     nlprun -q sphinx -g 1 -c 8 -r 96G -t 0-4 -n refusal-uniform-vllm-eval --dependency <train job> \
#         "bash scripts/cluster/sc_run.sh eval configs/refusal/eval_native_v2.yaml \
#              --run-dir runs/refusal_grpo_uniform_vllm --out runs/refusal_grpo_uniform_vllm/eval_native"
#
# THREE THINGS ABOUT THIS CLUSTER, each measured before this file existed:
#
#   DISK   /nlp/scr/aryaman (= /juice2/scr2/aryaman, 200 GB quota) filled up twice; since 2026-09-16
#          the repo, sibling and HF cache live on /juice3/scr3/nlp/interp (1 TB, group volume) and
#          the uv cache + project environment on /juice2/u/aryaman (50 GB) via UV_CACHE_DIR /
#          UV_PROJECT_ENVIRONMENT. Keep those two on ONE volume: uv hardlinks the venv from its
#          cache, so together they cost one copy, apart they cost two.
#   TOKEN  no .env here; the Hub token is the one `huggingface-cli login` stored under $HF_HOME,
#          which huggingface_hub reads on its own (verified: 200 on Llama-3.2-1B, -Instruct,
#          google/gemma-2b and the StrongREJECT judge adapter). Exported as HF_TOKEN too so any
#          loader that only reads the variable agrees.
#   SETUP  `setup` runs on a compute node, never on the login node: it clones the reference repos
#          (setup.sh --no-sync), then `uv sync --extra vllm`, then the smoke check and a
#          --print-config of every config it is given. Three training jobs syncing one venv at
#          once would race, so the training jobs depend on this one instead of each syncing.
set -euo pipefail

# MOVED 2026-09-16: /nlp/scr/aryaman (= /juice2/scr2/aryaman, a 200 GB quota) filled to 100% and the
# whole working set now lives on the 1 TB group volume. The repo, its sibling learning-to-attribute
# and the HF cache are all under /juice3/scr3/nlp/interp; the venv stays on /juice2/u.
cd /juice3/scr3/nlp/interp/matryoshka-attribution-parameters

export HF_HOME=/juice3/scr3/nlp/interp/hf_cache
export HF_TOKEN="$(cat "$HF_HOME/token")"
# Online by default: the StrongREJECT judge cannot load offline (eval/sr_ref.py). MLFT_HF_OFFLINE=1
# runs a job offline once everything it needs is cached -- the Hub auth check that eval/sb_ref.py
# makes at build time hung a 2-GPU SORRY-Bench job for 27 min on an idle socket (job 17401423).
export HF_HUB_OFFLINE=${MLFT_HF_OFFLINE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT=/tmp/vllm_cache_${SLURM_JOB_ID:-$$}   # per job: concurrent engines clobber a shared one
export UV_CACHE_DIR=/juice2/u/aryaman/uv_cache
export UV_PROJECT_ENVIRONMENT=/juice2/u/aryaman/venvs/mlft
mkdir -p runs/slurm_logs
# extras/groups every `uv run` here carries; the Olmo-3 cells add `--group olmes` (OLMES task layer)
UV_ARGS=${MLFT_UV_ARGS:---extra vllm}

echo "job ${SLURM_JOB_ID:-none} on $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "mode: $1 | args: ${*:2}"
echo "repo: $(git rev-parse --short HEAD) | l2a: $(git -C ../learning-to-attribute rev-parse --short HEAD)"

MODE=${1:?usage: sc_run.sh setup|train|eval ...}
shift
case "$MODE" in
  setup)
    bash scripts/setup.sh --no-sync
    uv sync --extra vllm
    uv run --extra vllm python scripts/verify/smoke_dep.py | tail -2
    for c in "$@"; do
      uv run --extra vllm python -m mask_learning_finetuning "$c" --print-config > /dev/null && echo "print-config ok: $c"
    done
    du -sh "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR"; df -h /juice2/u/aryaman /nlp/scr/aryaman | tail -2
    ;;
  train) uv run $UV_ARGS python -m mask_learning_finetuning "$@" ;;
  eval)  uv run $UV_ARGS python -m mask_learning_finetuning.eval "$@" ;;
  run)   uv run $UV_ARGS python "$@" ;;
  olmo-setup)
    # `uv sync` is EXACT (removes what the request omits), so the vllm extra rides along with the
    # olmes group; the refusal jobs' `uv run --extra vllm` is inexact and leaves the group in place.
    uv sync --extra vllm --group olmes
    uv run --extra vllm --group olmes python scripts/olmo3_base2inst/patch_base.py
    for c in "$@"; do
      uv run --extra vllm --group olmes python -m mask_learning_finetuning "$c" --print-config > /dev/null && echo "print-config ok: $c"
    done
    uv run --extra vllm --group olmes python scripts/verify/verify_olmes.py || echo "verify_olmes FAILED (non-fatal here; read it)"
    du -sh /juice2/u/aryaman/hf_hub/* "$UV_PROJECT_ENVIRONMENT"; df -h /juice2/u/aryaman /nlp/scr/aryaman | tail -2
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
