#!/usr/bin/env python
"""Recompute GSM8K accuracy in a run's ``evals.json`` from its saved generations.

WHY THIS IS LEGITIMATE AND NOT A FUDGE. ``eval/gsm8k.py``'s answer extraction changed (see
``extract_pred``: the old rule took the last ``####`` of the whole completion and so scored a
self-generated follow-up question instead of the real answer). The GENERATIONS did not change --
the eval decodes greedily, so a re-run produces the same text token for token -- and this repo
already treats ``generations.jsonl`` as the source of truth when a metric moves under it, which is
exactly why every per-response record is written (the pirate organism's `pirate_frac_coherent`
was backfilled the same way). Rescoring is therefore identical to re-running, minus the GPU.

IT REWRITES BOTH FILES, and that is the point. ``generations.jsonl`` carries a derived ``pred``
and ``correct`` per response, so updating only ``evals.json`` leaves the two disagreeing -- which
is not hypothetical: it broke ``plots/plot_method_agreement.py``, whose per-item verdicts come
from ``correct`` and which checks them against the reported accuracy (30.00 against 29.50 on the
instruct anchor). Both are recomputed from the ``response`` text, which is the only field that is
not derived.

WHAT IT WILL NOT DO. On a file it has not already rescored it refuses when the recorded ``correct``
flags do not reproduce the recorded accuracy, because that means the two are out of step for some
OTHER reason and rescoring would paper over it. It also refuses when the row count is not a whole
number of conditions. Re-running on an already-rescored directory is a no-op, not an error.

    uv run python scripts/analysis/rescore_gsm8k.py runs/refusal_grpo_8b_logk/eval_native
    uv run python scripts/analysis/rescore_gsm8k.py --all --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mask_learning_finetuning.eval.gsm8k import extract_pred  # noqa: E402

CONT = re.compile(r"\n\s*Question\s*:")


def rescore(run_dir: Path, dry=False):
    ev_path, gen = run_dir / "evals.json", run_dir / "gsm8k_eval" / "generations.jsonl"
    if not (ev_path.exists() and gen.exists()):
        return None
    blob = json.loads(ev_path.read_text())
    fin = blob.get("final", {})
    order = [k for k in fin if isinstance(fin[k].get("gsm8k"), dict)]
    if not order:
        return None
    rows = [json.loads(l) for l in gen.read_text().splitlines()]
    n = fin[order[0]]["gsm8k"]["gsm8k"]["n"]
    # already done: the guard below compares generations against evals.json, and after a rescore
    # they agree by construction -- so re-running must not re-assert the pre-rescore invariant
    done = "rescored" in fin[order[0]]["gsm8k"]["gsm8k"]
    blocks = len(rows) // n
    if blocks * n != len(rows):
        raise SystemExit(f"{run_dir}: {len(rows)} rows is not a whole number of {n}-row conditions")
    # conditions with identical weights are evaluated once and copied, so the file may hold
    # fewer blocks than evals.json has conditions -- map them in order and reuse the last
    changed = []
    for i, cond in enumerate(order):
        b = rows[min(i, blocks - 1) * n:(min(i, blocks - 1) + 1) * n]
        old = fin[cond]["gsm8k"]["gsm8k"]["accuracy"]
        recorded = 100.0 * sum(bool(r["correct"]) for r in b) / n
        if not done and i < blocks and abs(recorded - old) > 1e-6:
            raise SystemExit(f"{run_dir}/{cond}: generations give {recorded:.2f} but evals.json "
                             f"says {old:.2f} -- these are already out of step, not rescoring")
        acc = 100.0 * sum(1 for r in b if (p := extract_pred(r["response"])) is not None
                          and r["gold"] is not None and abs(p - r["gold"]) < 1e-6) / n
        m = fin[cond]["gsm8k"]["gsm8k"]
        m["accuracy"] = acc
        m["stderr"] = 100.0 * ((acc / 100) * (1 - acc / 100) / n) ** 0.5
        m["no_answer_frac"] = sum(1 for r in b if extract_pred(r["response"]) is None) / n
        m["continued_frac"] = sum(1 for r in b if CONT.search(r["response"])) / n
        m["rescored"] = "extract_pred first-turn rule"
        if abs(acc - old) > 1e-6:
            changed.append((cond, old, acc))
    # ...and the per-response derived fields, so `correct` agrees with the accuracy above
    for r in rows:
        pred = extract_pred(r["response"])
        r["pred"] = pred
        r["correct"] = bool(pred is not None and r["gold"] is not None
                            and abs(pred - r["gold"]) < 1e-6)
    if not dry:
        ev_path.write_text(json.dumps(blob, indent=2) + "\n")
        with gen.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every run under runs/ with a gsm8k_eval")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dirs = [Path(d) for d in a.run_dirs]
    if a.all:
        dirs = sorted({Path(f).parent.parent
                       for f in glob.glob("runs/**/gsm8k_eval/generations.jsonl", recursive=True)})
    for d in dirs:
        ch = rescore(d, a.dry_run)
        if ch is None:
            print(f"{d}: no gsm8k to rescore")
        elif ch:
            print(f"{d}:" + "".join(f"\n    {c:>12} {o:6.1f} -> {nw:6.1f}" for c, o, nw in ch))
        else:
            print(f"{d}: unchanged")
