"""How many weights each refusal edit changes, relative to the Instruct checkpoint.

Two measurements, both exact, written for the refusal comparison table (docs/refusal/l0_baselines.md
holds the numbers as of 2026-09-12):

``full``  the bf16 L0 of a shipped artifact: entries whose stored value differs from Instruct's.
          Abliterated model directories are compared tensor by tensor; LoRA adapters are merged
          in fp32 (rslora scale alpha/sqrt(r), else alpha/r) and cast to bf16 first, so what is
          counted is the artifact a `merge_and_unload` would ship. Streams per tensor, so an 8B
          pair needs a few GB of host memory and no GPU.
``mask``  the parameter count under a MAttr mask's top-k units at each reported fraction, summed
          over the slice length of every selected unit from the run's own ``final.pt`` layout.
          Under ``nonresid`` every unit is one residual-dimension vector, so this equals the unit
          fraction -- the script confirms it rather than assuming it.

The bf16 L0 UNDERCOUNTS the mathematical support of abliteration and LoRA: both touch every entry
of the matrices they edit, but ~20% of entries move by less than half a bf16 ulp and round back.
Report which of the two is meant. The mask's count is the same under either reading.

    uv run python scripts/refusal/l0_baselines.py full  --ref meta-llama/Llama-3.2-1B-Instruct \
        --model-dir models/abliterated/llama32_1b runs/refusal_grpoblit_kl001/model \
        --adapter runs/refusal_grpo_weights_kl001
    uv run python scripts/refusal/l0_baselines.py mask runs/refusal_grpo_logk_v2 runs/refusal_grpo_8b_logk_vllm_native
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FAMILIES = ("embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "norm")


def resolve(model):
    """A local directory as is; a hub id through the HF cache (must already be downloaded)."""
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])


def tensors(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as h:
            for n in h.keys():
                out[n] = (f, n)
    return out


def load(t):
    f, n = t
    with safe_open(f, "pt") as h:
        return h.get_tensor(n)


def family(name):
    return next((k for k in FAMILIES if k in name), "other")


def compare_dirs(label, edited, ref):
    E, R = tensors(edited), tensors(ref)
    tot = l0 = 0
    by = {}
    for n in R:
        r = load(R[n])
        e = load(E[n]) if n in E else r
        k = int((e.float() != r.float()).sum())
        tot += r.numel()
        l0 += k
        a, b = by.get(family(n), (0, 0))
        by[family(n)] = (a + k, b + r.numel())
    fam = "  ".join(f"{f}:{100 * a / b:.0f}%" for f, (a, b) in sorted(by.items()) if a)
    print(f"{label:40s} L0={l0:>13,} / {tot:>13,} = {100 * l0 / tot:6.2f}%   {fam}")


def compare_lora(label, run, ref):
    R = tensors(ref)
    cfg = json.load(open(f"{run}/adapter/adapter_config.json"))
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    with safe_open(f"{run}/adapter/adapter_model.safetensors", "pt") as h:
        A = {k: h.get_tensor(k) for k in h.keys()}
    tot = l0 = touched = 0
    for n in R:
        r = load(R[n])
        tot += r.numel()
        base = n.replace(".weight", "")
        ka, kb = f"base_model.model.{base}.lora_A.weight", f"base_model.model.{base}.lora_B.weight"
        if ka in A:
            merged = (r.float() + scale * (A[kb].float() @ A[ka].float())).to(torch.bfloat16)
            l0 += int((merged.float() != r.to(torch.bfloat16).float()).sum())
            touched += r.numel()
    print(f"{label:40s} L0={l0:>13,} / {tot:>13,} = {100 * l0 / tot:6.2f}%   "
          f"(targeted matrices: {100 * touched / tot:.0f}% of params; rslora={cfg.get('use_rslora')})")


def mask_topk(run, fracs):
    from mask_learning_finetuning.masks.checkpoint import load_checkpoint
    if not os.path.exists(f"{run}/final.pt"):
        print(f"{run}: no final.pt")
        return
    _, blob = load_checkpoint(run, require_delta=False)
    d, s = blob["layout"], blob["scores"].float()
    sizes = torch.zeros(d["total"])
    total_params, seen = 0, set()
    for shape, off, cnt, ax in zip(d["shapes"], d["offsets"], d["counts"], d["axes"]):
        total_params += math.prod(shape)
        unit = math.prod(shape) // (shape[ax] if ax is not None else cnt)
        if off in seen:                      # tied slices (neuron_head) accumulate, never assign
            sizes[off:off + cnt] += unit
        else:
            sizes[off:off + cnt] = unit
            seen.add(off)
    order = torch.argsort(-s)
    cells = []
    for f in fracs:
        k = int(round(f * d["total"]))
        p = int(sizes[order[:k]].sum())
        cells.append(f"{100 * f:g}%: {k:,} units -> {p:,} params ({100 * p / total_params:.2f}%)")
    print(f"{Path(run).name:34s} units={d['total']:,} params={total_params:,}\n  " + "\n  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("full", help="bf16 L0 of model directories and/or LoRA runs vs a reference")
    f.add_argument("--ref", required=True, help="Instruct checkpoint: local dir or hub id (cached)")
    f.add_argument("--model-dir", nargs="*", default=[], help="edited model directories")
    f.add_argument("--adapter", nargs="*", default=[], help="run dirs holding adapter/")
    m = sub.add_parser("mask", help="parameters under a MAttr mask's top-k units")
    m.add_argument("runs", nargs="+")
    m.add_argument("--fracs", default="0.001,0.005,0.01,0.02,0.05")
    args = ap.parse_args()
    if args.cmd == "full":
        ref = resolve(args.ref)
        for d in args.model_dir:
            compare_dirs(d, d, ref)
        for r in args.adapter:
            compare_lora(r, r, ref)
    else:
        for r in args.runs:
            mask_topk(r, [float(x) for x in args.fracs.split(",")])


if __name__ == "__main__":
    main()
