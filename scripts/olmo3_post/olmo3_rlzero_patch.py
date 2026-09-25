"""Relabel an Olmo-3 RL-Zero checkpoint so transformers 5.x and vLLM can load it.

``allenai/Olmo-3-7B-RL-Zero-*`` ship ``model_type: olmo2-retrofit`` / ``Olmo2RetrofitForCausalLM``
-- a pre-release name for the architecture that was released as ``olmo3`` (the config is otherwise
field-for-field the Instruct model's, and the weight names are identical; checked). Neither library
knows the old name, so ``AutoConfig`` dies. This writes ``models/olmo3_rlzero/<short>/``: symlinks to
the cached safetensors and tokenizer files, plus a rewritten ``config.json`` that names the native
class and spells out Olmo3's ``layer_types`` (the class default the released config leaves implicit).
Same idea as ``scripts/olmpool/olmpool_swa_patch.py``, one defect instead of four.

    uv run python scripts/olmo3_post/olmo3_rlzero_patch.py Mix Math IF
"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "olmo3_rlzero"


def patch(short: str) -> Path:
    src = Path(snapshot_download(f"allenai/Olmo-3-7B-RL-Zero-{short}",
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    dst = OUT / short
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    notes = []
    if cfg.get("model_type") != "olmo3":
        notes.append(f"model_type {cfg.get('model_type')!r} -> 'olmo3', architectures "
                     f"{cfg.get('architectures')} -> ['Olmo3ForCausalLM']")
        cfg["model_type"] = "olmo3"
        cfg["architectures"] = ["Olmo3ForCausalLM"]
    if "layer_types" not in cfg:
        n = cfg["num_hidden_layers"]
        cfg["layer_types"] = [("full_attention" if (i + 1) % 4 == 0 else "sliding_attention")
                              for i in range(n)]
        notes.append("layer_types filled with Olmo3's [s, s, s, f] pattern")
    if "torch_dtype" in cfg and "dtype" not in cfg:
        cfg["dtype"] = cfg.pop("torch_dtype")
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():
        if f.name == "config.json":
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())
    (dst / "PATCH_NOTES.txt").write_text(f"source: {src}\n" + "\n".join(notes) + "\n")
    print(f"{short}: {dst}\n  " + "\n  ".join(notes))
    return dst


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "chat_template.jinja")


def base_with_rlzero_tokenizer(short: str = "Mix") -> Path:
    """``models/olmo3_rlzero/Base``: the base model's weights and config under the RL-Zero TOKENIZER.

    The two tokenizers share a vocabulary and every id, but the RL-Zero one names ids 100256-100275
    (``<|extra_id_*|>`` in the base) as ``<think>``, ``</think>``, ``<functions>``, ... and treats
    them as special tokens, so "<think>" is ONE token under it and six under the base's. The RL
    model was trained under its own tokenizer, so every row of an objective and every prompt has to
    be tokenised by it -- and this repo takes the tokenizer from ``model:``. Hence a base-model
    directory carrying the RL-Zero tokenizer files (and its chat template, so ``chat_template: auto``
    resolves to the RL-Zero template on both endpoints).
    """
    src = Path(snapshot_download("allenai/Olmo-3-1025-7B",
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    tok = OUT / short
    dst = OUT / "Base"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name in TOKENIZER_FILES or f.name == "README.md":
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())
    for name in TOKENIZER_FILES:
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if (tok / name).exists():
            link.symlink_to((tok / name).resolve())
    (dst / "PATCH_NOTES.txt").write_text(
        f"weights + config: {src}\ntokenizer files + chat template: {tok}\n")
    print(f"Base: {dst} (weights from base, tokenizer from {short})")
    return dst


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--base"] or ["Mix"]
    for s in args:
        patch(s)
    if "--base" in sys.argv:
        base_with_rlzero_tokenizer(args[0])
