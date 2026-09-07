"""Install the sliding-window-patched remote code into the laid-out OlmPool model directories,
and give the four checkpoints whose HF configs lost their window one back.

Why this exists (docs/olmpool/README.md): transformers 5's sdpa/eager kernels ignore the
``sliding_window=`` kwarg the released ``Olmo3PreorderNoQK`` code relies on, and the other
custom classes never had a window at all. So of the 15 SWA models in OlmPool only the 8 native
``Olmo3ForCausalLM`` ones actually ran sliding-window attention in HF form. The patched
``models/olmpool/_remote_code/modeling_olmpool.py`` dispatches per-layer masks the way native
Olmo3 does (tests/test_olmpool_swa.py); this script

  1. copies the patched modeling/configuration files into every ``pt``/``lc``/``pt_ext`` dir
     whose config has an ``auto_map`` (overwriting the released copies);
  2. for the three custom-class SWA checkpoints shipped WITHOUT ``layer_types`` (A_post_HQK_...,
     H_post_HQK_..._SWA, H_pre_LQK_..._SWA) writes ``layer_types`` in the paper's 3-local:1-global
     pattern (Olmo3's default, ``[s, s, s, f] * 8``) and ``sliding_window: 4096``;
  3. converts ``F_pre_8kv_8k_14k_SWA`` -- released as plain ``LlamaForCausalLM``, i.e. no window
     -- to ``Olmo3PreorderNoQKForCausalLM`` (the same pre-norm/no-QK block under a name that
     knows about ``layer_types``; every parameter name is identical), with the same pattern;
  5. relabels a checkpoint whose WEIGHTS are the post-norm Olmo3 layout (``post_feedforward_layernorm``
     and q/k norm gains present, no ``input_layernorm``) but whose config names a pre-norm class
     as native ``Olmo3ForCausalLM``: ``G_post_LQK_8kv_4k_14k_SWA`` is released as
     ``Olmo3PreorderForCausalLM``, under which its ``input_layernorm`` is random and its
     ``post_feedforward_layernorm`` ignored -- NLL 11.5 on plain text at every checkpoint. The
     name, the paper's table and the tensor names all say post-norm; the class was wrong.
  4. rewrites ``rope_parameters`` of every native ``Olmo3ForCausalLM`` config into the
     per-layer-type form. The released configs store theta flat
     (``{"rope_theta": 8000000, "rope_type": "default"}``) and transformers 5.14's
     ``Olmo3Config`` then fills ``full_attention`` / ``sliding_attention`` entries from its CLASS
     DEFAULT (500000), which is what ``Olmo3RotaryEmbedding`` reads -- so the 12 native Olmo3
     long-context checkpoints, and their ``pt_ext`` twins, silently ran at the PRETRAINING theta
     (measured off the instantiated inverse frequencies: 500001 before, 7999997 after).

The pattern for (2) and (3) is an ASSUMPTION recorded in each dir's PROVENANCE.md: the paper
says "3 local attention layers of 4096 context for every 1 global attention layer" and the
released ``layer_types`` of the models that have one all start with three sliding layers, but
these four checkpoints do not say which layers were which. Idempotent; re-run after fetching.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "models" / "olmpool"
CODE = ROOT / "_remote_code"
FILES = ("configuration_olmpool.py", "modeling_olmpool.py")
PATTERN = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]

#: SWA models whose HF config carries no layer_types at all (paper Table: SWA = yes)
ASSUME_PATTERN = {
    "A_post_HQK_8kv_8k_13k_SWA_fp8", "H_post_HQK_32kv_8k_11k_SWA", "H_pre_LQK_32kv_8k_11k_SWA",
    "F_pre_8kv_8k_14k_SWA",
}
LLAMA_TO_NOQK = {"F_pre_8kv_8k_14k_SWA"}
NOQK_AUTO_MAP = {"AutoConfig": "configuration_olmpool.Olmo3PreorderNoQKConfig",
                 "AutoModelForCausalLM": "modeling_olmpool.Olmo3PreorderNoQKForCausalLM"}


def weight_keys(d: Path):
    import struct
    p = d / "model.safetensors"
    if not p.exists():
        return set()
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return set(json.loads(f.read(n)))


def patch_dir(d: Path, name: str) -> str:
    cfg_p = d / "config.json"
    cfg = json.loads(cfg_p.read_text())
    notes = []
    ks = weight_keys(d)
    post_norm_weights = any("post_feedforward_layernorm" in k for k in ks) and \
        any("self_attn.q_norm" in k for k in ks) and not any("input_layernorm" in k for k in ks)
    if post_norm_weights and cfg.get("model_type", "").startswith("olmo3_preorder"):
        cfg["architectures"] = ["Olmo3ForCausalLM"]
        cfg["model_type"] = "olmo3"
        cfg.pop("auto_map", None)
        n = cfg["num_hidden_layers"]
        cfg.setdefault("layer_types", [PATTERN[i % 4] for i in range(n)])
        cfg.setdefault("sliding_window", 4096)
        if "rope_parameters" not in cfg and "rope_theta" in cfg:
            cfg["rope_parameters"] = {"rope_theta": cfg["rope_theta"], "rope_type": "default"}
        notes.append("weights are the post-norm Olmo3 layout but the config named a pre-norm class: "
                     "relabelled Olmo3ForCausalLM (native; post-norm + layerwise QK norm + SWA)")
    if name in LLAMA_TO_NOQK and cfg.get("model_type") == "llama":
        cfg["architectures"] = ["Olmo3PreorderNoQKForCausalLM"]
        cfg["model_type"] = "olmo3_preorder_no_qk"
        cfg["auto_map"] = NOQK_AUTO_MAP
        notes.append("converted LlamaForCausalLM -> Olmo3PreorderNoQKForCausalLM (same block, "
                     "knows layer_types)")
    if name in ASSUME_PATTERN and not cfg.get("layer_types"):
        n = cfg["num_hidden_layers"]
        cfg["layer_types"] = [PATTERN[i % 4] for i in range(n)]
        cfg["sliding_window"] = 4096
        notes.append("layer_types ASSUMED as [sliding x3, full] * L/4, window 4096 (paper's "
                     "3:1 pattern; the released config had none)")
    rp = cfg.get("rope_parameters")
    if cfg.get("model_type") == "olmo3" and isinstance(rp, dict) and "rope_theta" in rp:
        theta = rp["rope_theta"]
        cfg["rope_parameters"] = {lt: {"rope_type": rp.get("rope_type", "default"),
                                       "rope_theta": theta}
                                  for lt in ("full_attention", "sliding_attention")}
        cfg["rope_theta"] = theta
        notes.append(f"rope_parameters rewritten per layer type at theta {theta} (the flat form "
                     "resolved to the Olmo3Config default 500000 in transformers 5.14)")
    if "auto_map" in cfg:
        for f in FILES:
            shutil.copy(CODE / f, d / f)
        notes.append("remote code replaced by the sliding-window-patched copy")
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    if notes:
        with (d / "PROVENANCE.md").open("a") as fh:
            fh.write("\nolmpool_swa_patch.py:\n" + "".join(f"- {n}\n" for n in notes))
    return "; ".join(notes)


def main():
    for m in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for sub in ("pt", "lc", "pt_ext"):
            d = m / sub
            if (d / "config.json").exists():
                note = patch_dir(d, m.name)
                if note and sub == "lc":
                    print(f"{m.name}: {note}")


if __name__ == "__main__":
    main()
