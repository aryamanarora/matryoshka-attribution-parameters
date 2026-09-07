"""Fetch OlmPool checkpoints (Bertsch et al. 2026) and lay them out as local model directories.

For each model two HF branches are pulled -- ``step34000`` (end of the 140B-token pretraining)
and ``longcontext-step2385`` (end of the 10B-token 64K context extension) -- and three local
directories are built under ``models/olmpool/<name>/``:

  pt/      the pretraining checkpoint as shipped (rope_theta 500k, its own max_position)
  lc/      the long-context checkpoint as shipped (rope_theta 8M, max_position 65536)
  pt_ext/  the PRETRAINING WEIGHTS under the LONG-CONTEXT CONFIG: config.json (and the remote
           modeling code, where the architecture needs it) copied from lc/, every other file
           symlinked from pt/.

``pt_ext`` is the base model of every attribution here, and it is not a convenience. The delta
``theta_lc - theta_pt`` is a weight difference, but the extension ALSO changed RoPE theta, and
the composed model ``theta_base + m . delta`` has to be evaluated under ONE positional encoding
-- the extended one, or the ``full_delta`` anchor is not the long-context model. So the
``pretrained`` anchor of every sweep is "the pretrained weights with theta scaled to 8M", i.e.
zero-shot theta scaling, which is exactly the starting point the 10B tokens of extension trained
from. That is the right anchor for "what did the extension training do" and a different object
from the pretrained model evaluated at its own theta, which is reported separately by
``scripts/olmpool_niah_anchor.py``.

Weights are symlinked into the HF cache rather than copied, so each model costs ~30 GB once.
Usage:  uv run python scripts/olmpool/olmpool_fetch.py [names...]   (default: every model, in priority
order)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2] / "models" / "olmpool"
PT_REV, LC_REV = "step34000", "longcontext-step2385"
REMOTE_CODE = ("configuration_olmpool.py", "modeling_olmpool.py")

#: Priority order: the three named baselines, the best and worst, then the single-feature
#: variants off the Llama baseline, then the rest.
PRIORITY = """
G_pre_8kv_8k_14k
H_post_LQK_32kv_8k_11k_SWA
K_post_HQK_8kv_12k
J_pre_16kv_8k_14k
A_post_HQK_8kv_8k_13k_SWA_fp8
G_pre_LQK_8kv_8k_14k
G_pre_8kv_8k_14k_SWA
G_pre_8kv_4k_14k
G_post_LQK_8kv_8k_14k
I_pre_32kv_8k_12k
G_pre_4kv_8k_14k
H_pre_32kv_8k_11k_SWA
H_pre_LQK_32kv_8k_11k_SWA
H_post_LQK_32kv_8k_11k
H_post_HQK_32kv_8k_11k_SWA
H_post_LQK_32kv_4k_11k
B_post_LQK_32kv_4k_11k
B_post_LQK_32kv_4k_11k_SWA
G_post_LQK_8kv_4k_14k_SWA
H_post_LQK_8kv_4k_11k_SWA
F_pre_8kv_8k_14k_SWA
C_post_LQK_8kv_8k_13k_SWA
D_post_LQK_8kv_8k_13k_SWA
D_post_LQK_8kv_8k_13k_SWA_fp8
E_post_LQK_32kv_8k_11k_SWA_fp8
H_post_LQK_32kv_8k_11k_SWA_fp8
""".split()


def link_tree(src: Path, dst: Path, *, skip=()):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name in skip or f.name.startswith("."):
            continue
        target = dst / f.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(f.resolve())


def fetch(name: str):
    out = ROOT / name
    if (out / "pt_ext" / "config.json").exists() and (out / "lc" / "config.json").exists():
        print(f"{name}: already laid out", flush=True)
        return
    t0 = time.time()
    pt = Path(snapshot_download("allenai/" + name, revision=PT_REV, max_workers=8))
    lc = Path(snapshot_download("allenai/" + name, revision=LC_REV, max_workers=8))
    link_tree(pt, out / "pt")
    link_tree(lc, out / "lc")
    # pt_ext: pt weights, lc config (+ remote code from lc, or from the shared copy)
    link_tree(pt, out / "pt_ext", skip=("config.json",) + REMOTE_CODE)
    shutil.copy(lc / "config.json", out / "pt_ext" / "config.json")
    cfg = json.loads((lc / "config.json").read_text())
    if "auto_map" in cfg:
        for f in REMOTE_CODE:
            src = lc / f if (lc / f).exists() else ROOT / "_remote_code" / f
            for d in ("pt_ext", "lc", "pt"):
                if not (out / d / f).exists():
                    shutil.copy(src, out / d / f)
    (out / "pt_ext" / "PROVENANCE.md").write_text(
        f"weights: allenai/{name}@{PT_REV}\nconfig.json: allenai/{name}@{LC_REV}\n"
        "i.e. the pretraining checkpoint evaluated under the long-context RoPE theta.\n")
    # the sliding-window patch (see that script): the released remote code runs full attention
    # under sdpa/eager, and four SWA configs carry no window at all
    from olmpool_swa_patch import patch_dir
    for sub in ("pt", "lc", "pt_ext"):
        patch_dir(out / sub, name)
    print(f"{name}: done in {time.time() - t0:.0f}s ({cfg['architectures'][0]})", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or PRIORITY
    for n in names:
        try:
            fetch(n)
        except Exception as e:      # keep going; report at the end
            print(f"{n}: FAILED {e!r}", flush=True)
