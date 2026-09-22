#!/bin/bash
# One job on sc's preemptible scavenger partition (sc-loprio: every GPU node, requeued when the
# owning group needs the card). The raw-sbatch twin of the nlprun submissions in sc_run.sh's header,
# because nlprun only knows jag/john/sphinx. Pinned by node FEATURE to 48 GB+ Ampere/Ada/Hopper
# cards (vLLM wants bf16, which Turing/Volta lack; a 1B GRPO cell is ~25-35 GB with the judge).
#
#     sbatch -J refusal-uniform_vllm -o runs/slurm_logs/refusal-uniform_vllm-lo-%j.out \
#         scripts/cluster/sc_loprio.sbatch train configs/refusal/rl/uniform_vllm.yaml
#
# Preemption RESTARTS the run from scratch (train/rl.py has no resume), so a cell here is a race
# against the owning group, and a job that ran twice leaves the second run's artifacts.
#SBATCH --partition=sc-loprio
#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=(ampere|ada|hopper)&(48G|80G|141G)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
cd "${MLFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # the repo root: submit from it, or set MLFT_ROOT
srun --unbuffered bash scripts/cluster/sc_run.sh "$@"
