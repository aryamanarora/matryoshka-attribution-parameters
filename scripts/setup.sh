#!/usr/bin/env bash
# One-shot setup for a fresh checkout: the two optional reference repos, then `uv sync`.
#
#     bash scripts/setup.sh              # clone the pinned commits into deps/, then uv sync
#     bash scripts/setup.sh --latest     # clone their default branches instead
#     bash scripts/setup.sh --no-sync    # clone only
#
# WHAT IS AND IS NOT HANDLED HERE
#
# `learning-to-attribute` is NOT cloned by this script: it is vendored into
# `deps/learning-to-attribute` and committed, because it is a hard dependency -- `uv sync` cannot
# resolve without it -- and a setup step you can forget is a setup step that fails on someone
# else's machine. See deps/learning-to-attribute/VENDORED.md.
#
# These two are different: each supplies a METRIC, is only needed by the eval that uses it, and is
# somebody else's repo that we call unmodified and must not fork.
#
#   model-organisms-for-EM  the EM metric (eval/em.py, eval/em_fast.py) -- question set, judge
#                           prompts, 0-100 logprob judge, misaligned-and-coherent rate. Put on
#                           sys.path by eval/em_ref.py rather than installed, because its own
#                           install pulls unsloth and vllm.
#   strong_reject           the StrongREJECT metric (eval/strongreject.py) -- prompt set, judge
#                           template, fine-tuned judge, 1-5 -> expected-value aggregation. Same
#                           arrangement, via eval/sr_ref.py.
#
# Both are cloned into `deps/` and gitignored there. `em_ref`/`sr_ref` look in `deps/` first and
# fall back to a sibling checkout (`../model-organisms-for-EM`), so a machine provisioned the old
# way keeps working and does not need a second copy.
#
# PINNED BY DEFAULT. The commits below are the ones every number in this repo was produced against;
# a judge or a prompt set that moves under you changes what the metric MEANS, silently and after
# the fact. `--latest` is there for when you actually want to update, and the pins should then be
# updated here in the same commit as the numbers they change.
set -euo pipefail

EM_SHA=8460e4e          # "Relinked to anonymous HF models"
SR_SHA=7a551d5          # "jailbreaks: Add Best-of-N, ReNeLLM (#40)"

# Overridable so a machine behind a mirror -- or this script's own test -- can point at another
# copy without editing the file. The pins above still apply.
EM_URL="${EM_URL:-https://github.com/clarifying-EM/model-organisms-for-EM.git}"
SR_URL="${SR_URL:-https://github.com/dsbowen/strong_reject.git}"

PIN=1
SYNC=1
for arg in "$@"; do
  case "$arg" in
    --latest) PIN=0 ;;
    --no-sync) SYNC=0 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p deps

clone_dep() {                    # name url sha marker
  local name="$1" url="$2" sha="$3" marker="$4" dest="deps/$1"
  if [ -d "$dest/$marker" ]; then
    echo "  $name: already present at $dest ($(git -C "$dest" rev-parse --short HEAD 2>/dev/null || echo "no git metadata"))"
    return
  fi
  if [ -d "../$name/$marker" ]; then
    echo "  $name: found a sibling checkout at ../$name -- em_ref/sr_ref fall back to it, so"
    echo "            not cloning a second copy. Delete it first if you want it under deps/."
    return
  fi
  echo "  $name: cloning $url"
  git clone --quiet "$url" "$dest"
  if [ "$PIN" = 1 ]; then
    git -C "$dest" checkout --quiet "$sha"
    echo "            pinned to $sha"
  else
    echo "            at $(git -C "$dest" rev-parse --short HEAD) (default branch, --latest)"
  fi
}

echo "reference repos (metrics we call unmodified):"
clone_dep model-organisms-for-EM "$EM_URL" "$EM_SHA" em_organism_dir
clone_dep strong_reject "$SR_URL" "$SR_SHA" strong_reject

echo
echo "vendored (already in this repo, no clone needed):"
echo "  learning-to-attribute: deps/learning-to-attribute @ \
$(grep -m1 'commit:' deps/learning-to-attribute/VENDORED.md | awk '{print substr($2,1,7)}')"

if [ "$SYNC" = 1 ]; then
  echo
  echo "uv sync:"
  uv sync
  echo
  echo "verifying the mask dependency (analytic toy, no model download):"
  uv run python scripts/smoke_dep.py | tail -2
fi

cat <<'EOF'

Done. Not covered here, and each optional:
  .env            OPENAI_API_KEY / AZURE_OPENAI_* for the judged evals (em, em_fast, pirate).
  HF_TOKEN        meta-llama/* and google/gemma-2b (the StrongREJECT judge) are gated.
  uv sync --extra vllm   installs vllm AND pins torch 2.11 for the whole project (see pyproject).
  data/           the derived SFT sets are gitignored; rebuild with scripts/prep_*.py.
EOF
