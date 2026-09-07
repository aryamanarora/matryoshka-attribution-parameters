#!/usr/bin/env python
"""AdvBench harmful behaviours + Alpaca harmless instructions, split for the two BASELINES.

Both baselines added in September 2026 need harmful prompts that are NOT the ones any headline is
computed on, and one of them needs a matched harmless set:

``advbench_reward.jsonl``   GRP-Oblit's GRPO training prompts (Russinovich et al. 2026 train on
                            "a held-out split of AdvBench"). Reached via ``rl.prompts``.
``advbench_val.jsonl``      abliteration's direction-selection set -- the prompts whose refusal
                            rate decides which layer's direction is used. Never reported.
``advbench_dir.jsonl``      abliteration's difference-in-means set (the harmful half).
``alpaca_harmless.jsonl``   the harmless half of that difference, instructions only. The first
                            128 rows are the difference-in-means set and the next 32 are the
                            disjoint validation set their KL and steering filters use.

THE SPLITS ARE DISJOINT FROM EACH OTHER *AND* FROM THE STRONGREJECT 60, and the second half of
that matters more than it looks: StrongREJECT was assembled partly FROM AdvBench, so the two sets
genuinely overlap in wording and an unchecked AdvBench split can contain prompts the headline is
computed on. This script removes any AdvBench row whose goal matches a reported StrongREJECT
prompt (exact match after whitespace/case normalisation) and prints how many went. ``train/rl.py``
re-checks disjointness at startup on the exact strings; this is the earlier, louder check.

DETERMINISTIC given the two sources: a fixed seed shuffles once, the splits are prefixes of that
order, and ``<file>.meta.json`` records the source URL, row counts and a digest of the raw text.
``--check`` re-reads the built files and verifies the digests, the split sizes and the
disjointness -- so a rebuild that silently picked up a changed upstream is visible.

    uv run python scripts/refusal/prep_advbench_data.py
    uv run python scripts/refusal/prep_advbench_data.py --check
"""

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

#: The original AdvBench, from the llm-attacks repo (Zou et al. 2023). The Hub mirror
#: (walledai/AdvBench) is gated; this file is not, and it is the canonical one.
ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
                "data/advbench/harmful_behaviors.csv")
ALPACA = "tatsu-lab/alpaca"
OUT = Path("data/advbench")
SEED = 0
#: reward split first, then the two abliteration splits; the remainder is unused
#: N_HARMLESS is 128 + 32, not 128: scripts/refusal/abliterate.py follows their pipeline, which draws a
#: harmless TRAIN split for the difference in means and a DISJOINT harmless VAL split for the KL
#: and steering filters (their Config.n_train / n_val). One file holds both, train first.
N_REWARD, N_VAL, N_DIR, N_HARMLESS = 260, 60, 128, 160


def norm(s: str) -> str:
    return " ".join(s.split()).strip().lower().rstrip(".")


def fetch_advbench():
    raw = urllib.request.urlopen(ADVBENCH_URL, timeout=60).read().decode()
    rows = [r["goal"].strip() for r in csv.DictReader(io.StringIO(raw)) if r.get("goal")]
    return rows, hashlib.sha256(raw.encode()).hexdigest()[:16]


def fetch_alpaca(n):
    from datasets import load_dataset
    ds = load_dataset(ALPACA, split="train")
    # instruction-only rows: an `input` field makes the prompt a two-part thing the refusal
    # direction would be computed over inconsistently with the harmful side, which has none
    out = [r["instruction"].strip() for r in ds if not r["input"].strip()]
    return out[:n]


def reported_strongreject():
    """The 60 prompts eval.strongreject reports on -- what everything here must avoid."""
    import sys
    sys.path.insert(0, "src")
    from mask_learning_finetuning.eval import sr_ref
    return {norm(p) for p in sr_ref.load_prompt_set("small")}


def write(path: Path, prompts, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {path}  {len(prompts)} prompts")


def build():
    goals, digest = fetch_advbench()
    held = reported_strongreject()
    kept = [g for g in goals if norm(g) not in held]
    print(f"AdvBench: {len(goals)} goals, {len(goals) - len(kept)} dropped as reported "
          f"StrongREJECT prompts, {len(kept)} usable")
    rng = random.Random(SEED)
    rng.shuffle(kept)
    need = N_REWARD + N_VAL + N_DIR
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} usable AdvBench goals, need {need}")
    splits = {"advbench_reward.jsonl": kept[:N_REWARD],
              "advbench_val.jsonl": kept[N_REWARD:N_REWARD + N_VAL],
              "advbench_dir.jsonl": kept[N_REWARD + N_VAL:need]}
    base = dict(source=ADVBENCH_URL, digest=digest, seed=SEED, total=len(goals),
                usable=len(kept), dropped_reported=len(goals) - len(kept))
    for name, rows in splits.items():
        write(OUT / name, rows, dict(base, split=name, n=len(rows)))
    harmless = fetch_alpaca(N_HARMLESS)
    write(OUT / "alpaca_harmless.jsonl", harmless,
          dict(source=ALPACA, n=len(harmless), note="instruction-only Alpaca rows"))


def check():
    held = reported_strongreject()
    seen, ok = {}, True
    for name in ("advbench_reward.jsonl", "advbench_val.jsonl", "advbench_dir.jsonl",
                 "alpaca_harmless.jsonl"):
        p = OUT / name
        if not p.exists():
            print(f"  MISSING {p}"); ok = False; continue
        rows = [json.loads(l)["prompt"] for l in p.read_text().splitlines() if l.strip()]
        meta = json.loads(p.with_suffix(".meta.json").read_text())
        bad = [r for r in rows if norm(r) in held]
        dupes = [r for r in rows if norm(r) in seen]
        seen.update({norm(r): name for r in rows})
        print(f"  {name}: {len(rows)} rows (meta says {meta['n']}), "
              f"{len(bad)} reported-prompt collisions, {len(dupes)} cross-split duplicates")
        ok &= len(rows) == meta["n"] and not bad and not dupes
    goals, digest = fetch_advbench()
    m = json.loads((OUT / "advbench_reward.meta.json").read_text())
    print(f"  upstream digest {digest} vs recorded {m['digest']}: "
          f"{'MATCH' if digest == m['digest'] else 'CHANGED -- rebuild'}")
    ok &= digest == m["digest"]
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    raise SystemExit(check() if a.check else (build(), 0)[1])
