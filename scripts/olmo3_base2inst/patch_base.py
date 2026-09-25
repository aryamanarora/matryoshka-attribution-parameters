"""``models/olmo3_base2inst/Base``: Olmo-3-1025-7B's weights and config under the INSTRUCT tokenizer.

The twin of ``scripts/olmo3_post/olmo3_rlzero_patch.py --base`` for the base -> Instruct delta. The
two tokenizers share the vocabulary and every id, but Instruct names ids 100266-100269
``<functions>``, ``</functions>``, ``<function_calls>``, ``</function_calls>`` (``<|extra_id_*|>`` in
the base) and only Instruct ships a chat template (``chat_template.jinja``). This repo takes the
tokenizer from ``model:``, so a run whose ``model:`` is the base needs a directory carrying the
Instruct tokenizer files -- then ``chat_template: auto`` renders Instruct's template on both
endpoints and every objective row is tokenised the way Instruct was trained.

Also the download step: both snapshots are fetched here (into the HF cache, which on our cluster points the
two ``models--allenai--Olmo-3-*`` entries under $HF_HOME), and the directory holds SYMLINKS only.

    uv run python scripts/olmo3_base2inst/patch_base.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "olmo3_base2inst"
BASE = "allenai/Olmo-3-1025-7B"
INSTRUCT = "allenai/Olmo-3-7B-Instruct"
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "chat_template.jinja")


def main() -> Path:
    src = Path(snapshot_download(BASE, allow_patterns=PATTERNS))
    tok = Path(snapshot_download(INSTRUCT, allow_patterns=PATTERNS))
    dst = OUT / "Base"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name in TOKENIZER_FILES or f.name == "README.md":
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())
    linked = []
    for name in TOKENIZER_FILES:
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if (tok / name).exists():
            link.symlink_to((tok / name).resolve())
            linked.append(name)
    (dst / "PATCH_NOTES.txt").write_text(
        f"weights + config: {src}\ntokenizer files + chat template ({', '.join(linked)}): {tok}\n")
    print(f"Base: {dst}\n  weights from {BASE} @ {src}\n  tokenizer from {INSTRUCT} @ {tok}: {linked}")
    return dst


if __name__ == "__main__":
    main()
