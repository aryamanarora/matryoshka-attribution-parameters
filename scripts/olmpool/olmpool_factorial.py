"""Which PART of the context-extension update carries long-range retrieval? A weight-level
factorial over one OlmPool model.

The extension delta ``theta_lc - theta_pt`` is split into four named parts, and every
combination of parts is applied in full (the rest stays pretrained) and scored with the
teacher-forced needle eval at every context length:

  A   attention projections (q/k/v/o)
  Q   the QK-norm gains (q_norm/k_norm; absent for the no-QK-norm architectures)
  M   the MLPs
  O   everything else: embeddings, lm_head, the layer norms (input/post-attention/post-ffn/final)

So ``none`` is the pretrained weights under the extended RoPE theta (zero-shot theta scaling),
``AQMO`` is exactly the released long-context checkpoint, and ``MO`` against ``none`` says what
the non-attention update buys on its own. The reason for this script: on G_pre_8kv_8k_14k,
``MO`` alone already matched ``AQMO`` at every length while ``A`` alone was BELOW ``none`` --
the head-level mask sweeps in docs/olmpool/ have to be read against this table, because a
mask over attention heads can only localise what the attention update carries.

Also evaluates ``pt``: the pretrained checkpoint under its OWN theta (the model as it was before
extension), loaded separately because it is a different positional encoding.

    uv run python scripts/olmpool/olmpool_factorial.py --model G_pre_8kv_8k_14k
"""
import argparse
import itertools
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mask_learning_finetuning.data.chat import install_chat_template
from mask_learning_finetuning.eval.base import ModelCtx
from mask_learning_finetuning.eval.niah import NiahEval, NiahEvalCfg

ROOT = Path(__file__).resolve().parents[2]
PARTS = {
    "A": re.compile(r"self_attn\.(q|k|v|o)_proj"),
    "Q": re.compile(r"self_attn\.(q|k)_norm"),
    "M": re.compile(r"\.mlp\."),
}


def part_of(name):
    for k, rx in PARTS.items():
        if rx.search(name):
            return k
    return "O"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="OlmPool model name (models/olmpool/<name>)")
    ap.add_argument("--prompts", default="data/niah/eval.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="runs/olmpool_factorial")
    ap.add_argument("--skip-pt", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = ROOT / "models" / "olmpool" / a.model
    tok = AutoTokenizer.from_pretrained(base_dir / "lc", trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    install_chat_template(tok, "plain")
    model = AutoModelForCausalLM.from_pretrained(base_dir / "pt_ext", dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()
    lc = AutoModelForCausalLM.from_pretrained(base_dir / "lc", dtype=torch.bfloat16,
                                              trust_remote_code=True)
    pt_w = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    lc_w = {n: p.detach().cpu() for n, p in lc.named_parameters()}
    del lc
    groups = {}
    for n in pt_w:
        groups.setdefault(part_of(n), []).append(n)
    parts = [k for k in "AQMO" if k in groups]
    print(a.model, "parts:", {k: len(v) for k, v in groups.items()}, flush=True)
    ev = NiahEval()
    probe = ev.build(tok, NiahEvalCfg(prompts=a.prompts, n_per_length=a.n, lengths=a.lengths))
    live = dict(model.named_parameters())
    res = {"model": a.model, "parts": {k: len(v) for k, v in groups.items()}, "conditions": {}}
    t0 = time.time()
    for r in range(len(parts) + 1):
        for combo in itertools.combinations(parts, r):
            label = "".join(combo) or "none"
            for k, names in groups.items():
                src = lc_w if k in combo else pt_w
                for n in names:
                    live[n].data.copy_(src[n].to(device))
            out = ev.run(ModelCtx(model, tok, device, label=label), probe)
            res["conditions"][label] = out
            print(f"  {label:5s}: " + " ".join(f"{s[4:]}={m['acc']:.2f}/{m['nll']:.2f}"
                                               for s, m in out.items()) + f"  ({time.time() - t0:.0f}s)",
                  flush=True)
    if not a.skip_pt:
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(base_dir / "pt", dtype=torch.bfloat16,
                                                     trust_remote_code=True).to(device).eval()
        out = ev.run(ModelCtx(model, tok, device, label="pt"), probe)
        res["conditions"]["pt_own_theta"] = out
        print("  pt   : " + " ".join(f"{s[4:]}={m['acc']:.2f}/{m['nll']:.2f}" for s, m in out.items()),
              flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.model}.json").write_text(json.dumps(res, indent=1))
    print("wrote", Path(a.out) / f"{a.model}.json")


if __name__ == "__main__":
    main()
