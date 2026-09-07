#!/usr/bin/env bash
# One-shot setup for a fresh checkout: the sibling `learning-to-attribute` checkout, the optional
# reference repos, then `uv sync`.
#
#     bash scripts/setup.sh              # clone the pinned commits into deps/, then uv sync
#     bash scripts/setup.sh --latest     # clone their default branches instead
#     bash scripts/setup.sh --no-sync    # clone only
#
# WHAT IS AND IS NOT HANDLED HERE
#
# `learning-to-attribute` (MAttr, the mask primitives) is a HARD dependency, installed editable
# from the SIBLING directory `../learning-to-attribute` (`[tool.uv.sources]` in pyproject.toml), so
# `uv sync` cannot resolve until it exists. This script clones it beside the repo if it is missing,
# at its default branch and never pinned: it is our own repo, the algorithm's home, and an edit
# there is meant to flow here immediately.
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
#   sorry-bench             the SORRY-Bench metric (eval/sorrybench.py) -- judge prompt, judge
#                           driver's template + 0/1 parse, via eval/sb_ref.py. The 450 prompts
#                           and the 7B judge are GATED Hub assets, fetched at run time.
#   google-research         Google's IFEval checker (eval/ifeval.py), one directory of their
#                           monorepo, sparse-cloned -- prompts, 25 checkers, strict/loose, via
#                           eval/ifeval_ref.py.
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
SB_SHA=7da10ad          # sorry-bench, 2025-03-01 (judge_prompts.jsonl + the ft-mistral judge driver)
IFE_SHA=""              # google-research is a monorepo: sparse, depth 1, no pin (see below)
OLMES_SHA=5a51f50       # allenai/olmes, 2026-03-24 -- the Olmo 3 model cards' evaluation code

# Overridable so a machine behind a mirror -- or this script's own test -- can point at another
# copy without editing the file. The pins above still apply.
L2A_URL="${L2A_URL:-https://github.com/aryamanarora/learning-to-attribute.git}"
EM_URL="${EM_URL:-https://github.com/clarifying-EM/model-organisms-for-EM.git}"
SR_URL="${SR_URL:-https://github.com/dsbowen/strong_reject.git}"
SB_URL="${SB_URL:-https://github.com/sorry-bench/sorry-bench.git}"
IFE_URL="${IFE_URL:-https://github.com/google-research/google-research.git}"
OLMES_URL="${OLMES_URL:-https://github.com/allenai/olmes.git}"

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

# google-research is a multi-GB monorepo of which one 5 MB directory is wanted, so it is cloned
# blobless and sparse -- a full clone_dep would take an hour and a disk. Depth 1 means no pin:
# the directory's last upstream change is what you get, and `git -C deps/google-research log -1`
# records which. (Their checker has been stable since 2023; the risk is small and stated.)
clone_ifeval() {
  local dest="deps/google-research" pkg="instruction_following_eval"
  if [ -d "$dest/$pkg" ]; then
    echo "  google-research/$pkg: already present at $dest ($(git -C "$dest" rev-parse --short HEAD 2>/dev/null || echo "no git metadata"))"
    return
  fi
  if [ -d "../google-research/$pkg" ]; then
    echo "  google-research/$pkg: found a sibling checkout at ../google-research -- ifeval_ref falls back to it"
    return
  fi
  echo "  google-research/$pkg: sparse clone of $IFE_URL"
  git clone --quiet --depth 1 --filter=blob:none --sparse "$IFE_URL" "$dest"
  git -C "$dest" sparse-checkout set "$pkg"
  echo "            at $(git -C "$dest" rev-parse --short HEAD) (depth 1, sparse: $pkg only)"
}

echo "reference repos (metrics we call unmodified):"
clone_dep model-organisms-for-EM "$EM_URL" "$EM_SHA" em_organism_dir
clone_dep strong_reject "$SR_URL" "$SR_SHA" strong_reject
clone_dep sorry-bench "$SB_URL" "$SB_SHA" gen_judgment_safety_vllm.py
clone_ifeval
# OLMES: its TASK LAYER is imported from this checkout (eval/olmes_ref.py; `uv sync --group olmes`
# for its pure-python deps); the bit-faithful CLI route additionally needs ITS OWN venv, because its
# pins (torch 2.8, vllm 0.11, transformers <5) conflict with this repo's:
#     cd deps/olmes && uv sync --group gpu
clone_dep olmes "$OLMES_URL" "$OLMES_SHA" oe_eval

echo
echo "the mask dependency (editable sibling checkout):"
L2A_DIR="$ROOT/../learning-to-attribute"
if [ -d "$L2A_DIR/src/learning_to_attribute" ]; then
  echo "  learning-to-attribute: found at ../learning-to-attribute @ $(git -C "$L2A_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
else
  echo "  learning-to-attribute: cloning $L2A_URL beside this repo"
  git clone --quiet "$L2A_URL" "$L2A_DIR"
  echo "            at $(git -C "$L2A_DIR" rev-parse --short HEAD) (default branch, not pinned)"
fi

if [ "$SYNC" = 1 ]; then
  echo
  echo "uv sync:"
  uv sync
  echo
  echo "verifying the mask dependency (analytic toy, no model download):"
  uv run python scripts/verify/smoke_dep.py | tail -2
fi

cat <<'EOF'

Done. Not covered here, and each optional:
  .env            OPENAI_API_KEY / AZURE_OPENAI_* for the judged evals (em, em_fast, pirate).
  HF_TOKEN        meta-llama/* and google/gemma-2b (the StrongREJECT judge) are gated, as are
                  sorry-bench/sorry-bench-202406 and its ft-mistral judge (click-through).
  uv sync --extra vllm   installs vllm AND pins torch 2.11 for the whole project (see pyproject).
  data/           the derived SFT sets are gitignored; rebuild with scripts/data/prep_*.py.
EOF
