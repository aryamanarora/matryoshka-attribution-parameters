"""Bit-faithful OLMES numbers: save the weights of one sweep condition (or a plain checkpoint) as an
HF model directory and run AI2's own `olmes` CLI on it, in ITS venv (deps/olmes/.venv: their torch,
their vLLM 0.11, their batching) -- the only route that reproduces a model-card number exactly.
eval/olmes.py runs their prompts and metrics inside our sweeps with our engine; this is the check
that the two agree, and the tool for the anchors.

    # the two anchors of a delta, straight from the hub ids
    uv run python scripts/olmo3_post/olmes_cli_eval.py --model allenai/Olmo-3-7B-Instruct-DPO \
        --tasks ifeval::olmo3:adapt gsm8k::olmo3:adapt --out runs/olmes_cli/dpo
    # one condition of a masked run: top-5% of the delta over the base
    uv run python scripts/olmo3_post/olmes_cli_eval.py --run-dir runs/olmo3_sft2dpo/rl/math500 --frac 0.05 \
        --tasks minerva_math::olmo3:adapt --out runs/olmes_cli/math500_frac0.05

Composition is the repo's own (`masks.apply_in_place` with the checkpoint's layout and top-k), the
saved directory is bf16, and it is deleted after the run unless --keep. `--limit` caps docs per task
(their --limit), `--olmes-args` passes anything else through (e.g. "--batch-size 8").
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def compose_and_save(run_dir: Path, frac: float, out_dir: Path, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from matryoshka_attribution import build_mask, normalize_mode
    from mask_learning_finetuning.masks import apply_in_place, load_checkpoint, resolve_dtype
    from mask_learning_finetuning.masks.checkpoint import layout_from_blob
    from mask_learning_finetuning.train.posthoc import build_deltas, load_finetuned

    _, blob = load_checkpoint(run_dir, require_delta=False)
    targs, layout = blob["args"], layout_from_blob(blob)
    mode = normalize_mode(targs.get("mode", "cause"))
    dt = resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(targs["model"], dtype=dt, trust_remote_code=bool(targs.get("trust_remote_code"))).cuda().eval()
    tok = AutoTokenizer.from_pretrained(targs["model"])
    base = {n: p.detach().clone().cpu() for n, p in model.named_parameters() if n in set(layout.names)}
    if "delta" in blob:
        deltas = {n: d.to(dt) for n, d in blob["delta"].items()}
    else:
        ft, _ = load_finetuned(targs["model"], targs["finetuned"], dtype=torch.float32)
        deltas, _ = build_deltas({n: p.detach() for n, p in model.named_parameters()}, ft, layout.names,
                                 dtype=dt, release_finetuned=True)
        del ft
    k = max(1, int(round(frac * layout.total))) if frac < 1.0 else layout.total
    scores = blob["scores"].float()
    hard = build_mask(scores, k, targs.get("variant", "topk"), T=targs.get("T", 0.5), n_iters=targs.get("n_iters", 50)).mask
    with torch.no_grad():
        apply_in_place(model, base, {n: d.cuda() for n, d in deltas.items()}, hard.cuda(), layout,
                       invert=(mode == "sufficient"), out_dtype=dt)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    (out_dir / "COMPOSED.json").write_text(json.dumps(dict(run_dir=str(run_dir), frac=frac, k=k, total=layout.total, mode=mode), indent=2))
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a hub id or HF dir to evaluate as-is")
    ap.add_argument("--run-dir", help="a masked run; compose its top-k at --frac")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the composed weights directory")
    ap.add_argument("--olmes-args", default="", help="extra args passed to the olmes CLI verbatim")
    ap.add_argument("--olmes-venv", default=str(ROOT / "deps" / "olmes" / ".venv"))
    #: their max_length defaults to 2048 minus the prompt, which silently caps AIME's 16K generation
    #: at ~1.6K ("Running with effective max_gen_toks of 1624 instead of 16384"); Olmo-3 is a 64K model
    ap.add_argument("--max-length", type=int, default=32768)
    args = ap.parse_args()
    out = Path(args.out).resolve()        # absolute: the CLI runs with cwd = deps/olmes
    out.mkdir(parents=True, exist_ok=True)
    olmes = Path(args.olmes_venv) / "bin" / "olmes"
    sc = sorted(Path(args.olmes_venv).glob("lib/python3.*/site-packages/sitecustomize.py"))
    if not sc or "olmes_venv_patch" not in sc[-1].read_text():
        sys.exit("their venv lacks the lm_eval/vLLM transport patch: `uv run python scripts/olmo3_post/olmes_venv_patch.py`")
    if not olmes.exists():
        sys.exit(f"no olmes CLI at {olmes}: `cd deps/olmes && uv sync --group gpu`")
    model_path, composed = args.model, None
    if args.run_dir:
        composed = out / "composed_weights"
        model_path = str(compose_and_save(Path(args.run_dir), args.frac, composed))
    # their default trust_remote_code is None, which the pydantic in their venv rejects at vLLM's
    # ModelConfig ("Input should be a valid boolean"); Olmo-3 is a native class, so True is inert
    cmd = [str(olmes), "--model", str(model_path), "--model-type", "vllm",
           "--model-args", '{"trust_remote_code": true, "max_length": %d}' % args.max_length, "--output-dir", str(out), "--task", *args.tasks]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += args.olmes_args.split()
    env = dict(os.environ)
    # the olmes CLI re-launches itself as `python -m oe_eval.run_eval`, resolving `python` from
    # PATH -- which, under `uv run`, is THIS repo's venv (no alpaca_eval, wrong torch). Their venv
    # has to be first on PATH, not just the binary that is invoked.
    env["PATH"] = str(Path(args.olmes_venv) / "bin") + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(args.olmes_venv)
    env["OLMES_FORKSERVER"] = "1"      # read by their venv's sitecustomize (scripts/olmo3_post/olmes_venv_patch.py)
    # their `datasets` is pinned <4 and cannot read arrow metadata written by this repo's datasets 5
    # (job 284586: "Feature type 'List' not found" on the IFEval cache), so their venv gets its own
    # datasets cache rather than the shared ~/.cache/huggingface/datasets
    env["HF_DATASETS_CACHE"] = str(Path(env.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "datasets_olmes")
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = env.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT / "deps" / "olmes"))
    if composed is not None and not args.keep:
        shutil.rmtree(composed, ignore_errors=True)
    # summarise their metrics.json files
    summary = {}
    for mj in sorted(out.rglob("task-*-metrics.json")):     # metrics.json is their run-level stub
        try:
            d = json.load(open(mj))
            rows = d if isinstance(d, list) else [d]
            for r in rows:
                name = r.get("task_name") or r.get("task_config", {}).get("task_name") or mj.stem
                summary[name] = r.get("metrics", r)
        except Exception:
            pass
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for k, v in summary.items():
        prim = v.get("primary_score") if isinstance(v, dict) else None
        print(f"{k}: primary_score={prim}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
