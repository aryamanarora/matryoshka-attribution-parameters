#!/usr/bin/env bash
# Continuous ONE-WAY sync: this Mac -> CoreWeave shared home.
#
# Pushes both repos, preserving the sibling layout that the editable dependency needs
# (this repo's pyproject.toml points at ../matryoshka-attribution).
#
# Target is the LOGIN node on purpose: the home volume is the same filesystem on the login node and on
# every dev pod, so the files show up on whatever node `pod` gives you today and this never
# needs updating when the pod moves.
#
# THE CLUSTER COPY IS A MIRROR, NOT A WORKING TREE. --delete is on, so anything you create
# on the cluster inside a synced repo gets removed on the next pass unless it matches one of
# the EXCLUDES below (results/, logs/, checkpoints/, wandb/, .venv/ are all protected, which
# is where cluster-side output belongs). Do not edit or commit there; edit here.
#
#   ./scripts/cluster/sync_to_cluster.sh            # watch loop, syncs every $INTERVAL seconds
#   ./scripts/cluster/sync_to_cluster.sh --once     # single pass, then exit
#   INTERVAL=15 ./scripts/cluster/sync_to_cluster.sh
#   REMOTE=cw-east-13a-login ./scripts/cluster/sync_to_cluster.sh
set -uo pipefail

REMOTE=${REMOTE:?set REMOTE to the ssh host of the cluster login node}
DEST=${DEST:?set DEST to the directory on $REMOTE that holds both checkouts}
INTERVAL=${INTERVAL:-5}
SRC_ROOT=${SRC_ROOT:-$HOME}
REPOS=(matryoshka-attribution "$(basename "$(cd "$(dirname "$0")/../.." && pwd)")")   # the sibling and this checkout, by their directory names

# .venv is excluded because a macOS-arm64 venv is worse than useless on Linux -- run
# `uv sync` once on the cluster instead. results/logs/checkpoints/wandb are excluded so
# cluster-generated output is never clobbered or deleted by a push from here.
EXCLUDES=(
  # .env holds the judge API key and is created ON THE CLUSTER. It is not in the local tree,
  # so without this exclude --delete would remove it on the next pass -- and syncing a key
  # from a laptop is not something this script should do either.
  --exclude '.env'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  # NEVER sync or delete credentials. .env holds the judge API key and lives only on the
  # cluster; without this exclude, --delete removes it seconds after it is created because
  # no such file exists locally.
  --exclude '.env'
  --exclude '.env.*'
  --exclude 'results/'
  --exclude 'logs/'
  --exclude 'logs_sweep/'
  --exclude 'logs_arith/'
  --exclude 'checkpoints/'
  --exclude 'wandb/'
  --exclude 'paper/'
  --exclude 'MIB-circuit-track'   # broken symlink to the Stanford cluster path
)

sync_once() {
  local rc=0
  for r in "${REPOS[@]}"; do
    if [[ ! -d "$SRC_ROOT/$r" ]]; then
      echo "!! missing locally, skipped: $SRC_ROOT/$r"
      continue
    fi
    # -i itemizes, so a no-change pass prints nothing at all
    rsync -azi --delete --timeout=30 "${EXCLUDES[@]}" \
      "$SRC_ROOT/$r/" "$REMOTE:$DEST/$r/" || rc=$?
  done
  return $rc
}

if [[ "${1:-}" == "--once" ]]; then
  sync_once
  exit $?
fi

echo "syncing ${REPOS[*]} -> $REMOTE:$DEST every ${INTERVAL}s (ctrl-c to stop)"
fails=0
while true; do
  out=$(sync_once 2>&1)
  rc=$?
  if [[ -n "$out" ]]; then
    printf '[%s]\n%s\n' "$(date +%H:%M:%S)" "$out"
  fi
  if [[ $rc -ne 0 ]]; then
    # Don't die on a transient network blip (pod reschedule, laptop sleep, tailnet drop):
    # back off and keep trying, but say so, since silence would otherwise read as "synced".
    fails=$((fails + 1))
    echo "[$(date +%H:%M:%S)] rsync exit $rc (consecutive failures: $fails) -- retrying"
    sleep $((INTERVAL * 3))
  else
    fails=0
    sleep "$INTERVAL"
  fi
done
