#!/usr/bin/env bash
# Submit a French sweep: every cell matching a pattern, plus a post-hoc mask over each cell that
# has one.
#
# Grids under configs/french/, and this submits any of them:
#   sft/sweep_*   {full SFT, LoRA} x 4 learning rates, each attributed post hoc (the default)
#   sft/rank_*    LoRA rank x learning rate, 16 cells, no post-hoc counterpart
#   posthoc/*     learned masks over finished checkpoints        (--alone posthoc)
#   ixg/*         the IxG baseline over the same checkpoints     (--alone ixg)
#   restrict/*    retrain confined to a fitted mask's top-k      (--alone restrict)
#
# `--experiment` points the same machinery at a sibling experiment directory, since the layout it
# assumes (`<exp>/sft/<cell>.yaml` attributed by `<exp>/posthoc/<cell>.yaml`) is a convention of
# configs/, not of French. `configs/french_bactrian/` is the same sweep_* grid on the Bactrian-X
# French set and is submitted with `--experiment french_bactrian`.
#
# The post-hoc job is submitted with `--dependency=afterok:<finetune job>`, which is the whole
# reason this is a script rather than 16 sbatch lines: a mask job reads
# <run>/model or <run>/adapter, which does not exist until its finetune has finished, and slurm
# will hold the job until then instead of failing on a missing path. `afterok` (not `after`) means
# a crashed finetune leaves its mask job in the queue as DependencyNeverSatisfied rather than
# training a mask over a checkpoint from some earlier attempt -- which is exactly the stale-output
# failure that has already produced one wrong figure in this repo.
#
#   ./scripts/cluster/submit_french_sweep.sh --dry-run          # print the plan, submit nothing
#   ./scripts/cluster/submit_french_sweep.sh                    # the sweep_* grid + its post-hoc cells
#   ./scripts/cluster/submit_french_sweep.sh --pattern 'rank_*'  # the LoRA rank x lr grid
#   ./scripts/cluster/submit_french_sweep.sh --only sweep_lora   # just the LoRA half of sweep_*
#   ./scripts/cluster/submit_french_sweep.sh --no-posthoc        # finetunes only
#   ./scripts/cluster/submit_french_sweep.sh --alone posthoc --pattern '*'  # masks over finished runs
#   ./scripts/cluster/submit_french_sweep.sh --alone ixg --pattern '*'      # the IxG baseline
#   ./scripts/cluster/submit_french_sweep.sh --alone restrict --pattern '*' # retrain inside those masks
#   ./scripts/cluster/submit_french_sweep.sh --account cw-sup ...            # off the team's 8-GPU quota
#   ./scripts/cluster/submit_french_sweep.sh --experiment french_bactrian   # same grid, Bactrian-X data
#
# Run it FROM THE CLUSTER (it calls sbatch). The sweep needs `uv sync --extra vllm` there, since
# these configs generate through vLLM.
set -euo pipefail

cd "$(dirname "$0")/../.."

DRY=0 POSTHOC=1 ONLY="" PATTERN="sweep_*" ALONE="" ACCOUNT="" EXP="french"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY=1 ;;
    --no-posthoc)  POSTHOC=0 ;;
    # A directory of cells that read a checkpoint which already exists (posthoc/, ixg/): no
    # finetune to submit and no dependency to wait on. Also the only way to reach a cell whose
    # finetune came from a different grid.
    --alone)       ALONE="${2:?--alone takes a subdirectory of the experiment, e.g. ixg}"; shift ;;
    # Which configs/<experiment>/ tree to read. Defaults to french; french_bactrian is the same
    # sweep_* grid over data/lang/fr_sft.jsonl instead of data/lang/french_sft.jsonl.
    --experiment)  EXP="${2:?--experiment takes a configs/ subdir, e.g. french_bactrian}"; shift ;;
    # `general` is the goodfire team account and its 8-GPU cap is shared with the other two members,
    # so a long sweep blocks them. `cw-sup` is the cluster-wide default account (~70 users) and has
    # no GrpTRES set -- more concurrency, but it is not this team's allocation: ask before leaning
    # on it, and check the QOS, since a preemptible one leaves half-written run directories.
    --account)     ACCOUNT="${2:?--account takes a slurm account, e.g. cw-sup}"; shift ;;
    --pattern)     PATTERN="${2:?--pattern takes a glob, e.g. 'rank_*'}"; shift ;;
    --only)        ONLY="${2:?--only takes a cell-name prefix, e.g. sweep_lora}"; shift ;;
    -h|--help)     sed -n '2,37p' "$0"; exit 0 ;;   # the comment header, up to `set -euo`
    *)             echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Cells are discovered rather than listed, so adding a config file to the grid is all it takes to
# add it to the sweep -- and a `*_base.yaml` is a shared parent, never a cell of its own.
EXP_DIR="configs/$EXP"
[[ -d "$EXP_DIR" ]] || { echo "no such experiment directory: $EXP_DIR" >&2; exit 2; }
SRC_DIR="$EXP_DIR/sft"
[[ -n "$ALONE" ]] && SRC_DIR="$EXP_DIR/$ALONE"
[[ -d "$SRC_DIR" ]] || { echo "no such directory: $SRC_DIR" >&2; exit 2; }
CELLS=()
for f in ${SRC_DIR}/${PATTERN}.yaml; do
  [[ -f "$f" ]] || continue
  b=$(basename "$f" .yaml)
  # a shared parent is never a cell: `base.yaml` and `<grid>_base.yaml` both
  [[ "$b" == base || "$b" == *_base ]] && continue
  CELLS+=("$b")
done
[[ ${#CELLS[@]} -gt 0 ]] || { echo "no cells match ${SRC_DIR}/${PATTERN}.yaml" >&2; exit 1; }

submit() {                     # submit <config> [dependency-jobid] -> echoes the job id
  local cfg=$1 dep=${2:-}
  local args=(scripts/cluster/sbatch_train.sbatch "$cfg")
  [[ -n "$ACCOUNT" ]] && args=(--account="$ACCOUNT" "${args[@]}")
  [[ -n "$dep" ]] && args=(--dependency="afterok:$dep" "${args[@]}")
  if [[ $DRY -eq 1 ]]; then
    echo "    sbatch ${args[*]}" >&2
    echo "DRYRUN"
  else
    sbatch --parsable "${args[@]}"
  fi
}

n=0
for cell in "${CELLS[@]}"; do
  [[ -n "$ONLY" && "$cell" != "$ONLY"* ]] && continue
  ft="$EXP_DIR/sft/${cell}.yaml"
  ph="$EXP_DIR/posthoc/${cell}.yaml"

  if [[ -n "$ALONE" ]]; then
    cfg="${SRC_DIR}/${cell}.yaml"
    echo "== $cell ($ALONE)"
    echo "   $ALONE: $(submit "$cfg")  ($cfg)"
    n=$((n + 1))
    continue
  fi
  [[ -f "$ft" ]] || { echo "!! missing $ft" >&2; exit 1; }

  echo "== $cell"
  jid=$(submit "$ft")
  echo "   finetune: $jid  ($ft)"
  n=$((n + 1))
  if [[ $POSTHOC -eq 1 && -f "$ph" ]]; then
    pjid=$(submit "$ph" "$jid")
    echo "   posthoc:  $pjid  after $jid  ($ph)"
    n=$((n + 1))
  fi
done

echo
if [[ $DRY -eq 1 ]]; then
  echo "dry run: $n job(s) would be submitted"
else
  echo "submitted $n job(s); squeue -u \$USER to watch"
fi
