"""Per-head attention statistics on the needle prompts, at any OlmPool checkpoint.

For every (layer, head): the fraction of attention mass on the first 100 tokens (the SINK, the
paper's Section 5 statistic), the attention ENTROPY, the mean attended DISTANCE (query position
minus attended position, in tokens) and the mass on the NEEDLE tokens -- each averaged over a
sample of query positions (the last 32 of the prompt, where the answer is produced, plus 32
positions spread over the context) and over prompts.

Implemented as a registered attention interface: every OlmPool class (native Llama/Olmo3/Qwen3
and the remote-code variants) computes attention through
``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``, so one function that records the
statistics from (query, key) and then defers to sdpa for the actual output covers all of them
without touching any modeling code. Statistics are computed on the sampled query rows only, so a
16K prompt costs a [heads x 64 x 16K] softmax per layer rather than the full matrix.

Writes ``<out>/<model>__<ckpt>.json`` with [layers x heads] arrays per statistic and length.
Read against a mask's head ranking by scripts/olmpool_analysis.py: if the heads whose share of
the extension delta a mask keeps first are the heads whose sink / distance / needle statistics
changed most between ``pt_ext`` and ``lc``, the mask is naming the heads the extension re-tuned.

    uv run python scripts/olmpool_head_stats.py --model models/olmpool/G_pre_8kv_8k_14k/lc \\
        --lengths 4096 16384 --n 8
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from mask_learning_finetuning.data.chat import install_chat_template

STATE = {"rows": None, "needle": None, "sink": 100, "acc": {}}


def stats_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                    **kwargs):
    """Record per-head statistics from (query, key) on the sampled rows, then run sdpa."""
    rows = STATE["rows"]
    if rows is not None and query.shape[2] > 1:
        B, H, T, d = query.shape
        Hk = key.shape[1]
        rep = H // Hk
        k = key.repeat_interleave(rep, dim=1) if rep > 1 else key
        q = query[:, :, rows]                                   # [B, H, R, d]
        sc = scaling if scaling is not None else 1.0 / math.sqrt(d)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sc   # [B, H, R, T]
        pos = torch.arange(T, device=q.device)
        causal = pos[None, :] <= rows.to(q.device)[:, None]                  # [R, T]
        # sliding-window layers: the kernel would mask beyond the window, so do the same here
        sw = kwargs.get("sliding_window") or getattr(module, "sliding_window", None)
        if sw:
            causal = causal & (rows.to(q.device)[:, None] - pos[None, :] < sw)
        logits = logits.masked_fill(~causal[None, None], float("-inf"))
        p = logits.softmax(-1)                                              # [B, H, R, T]
        p = p.nan_to_num(0.0)
        dist = (rows.to(q.device)[:, None] - pos[None, :]).clamp(min=0).float()
        ent = -(p * (p + 1e-12).log()).sum(-1)                              # [B, H, R]
        sink = p[..., :STATE["sink"]].sum(-1)
        mdist = (p * dist).sum(-1)
        acc = STATE["acc"].setdefault(module.layer_idx, {})
        for name, val in (("entropy", ent), ("sink", sink), ("distance", mdist)):
            acc[name] = acc.get(name, 0) + val.mean(dim=(0, 2)).cpu()        # [H]
        if STATE["needle"] is not None:
            s0, s1 = STATE["needle"]
            acc["needle"] = acc.get("needle", 0) + p[..., s0:s1].sum(-1).mean(dim=(0, 2)).cpu()
        acc["n"] = acc.get("n", 0) + 1
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling,
                                  dropout=dropout, **kwargs)


ALL_ATTENTION_FUNCTIONS.register("stats", stats_attention)


def find_needle(ids, tok, row):
    needle = f"One of the special magic numbers for {row['word']} is: {row['number']}."
    for cand in (" " + needle, needle):
        nid = tok(cand, add_special_tokens=False).input_ids
        n = len(nid)
        for s in range(len(ids) - n + 1):
            if ids[s:s + n] == nid:
                return s, s + n
    return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/niah/eval.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/olmpool_hs")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    install_chat_template(tok, "plain")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="stats").to(device).eval()
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    by_len = {}
    for r in rows:
        by_len.setdefault(int(r["ctx_len"]), []).append(r)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    res = {"model": a.model, "n_layers": L, "n_heads": H, "lengths": {}}
    t0 = time.time()
    for Lc in a.lengths:
        STATE["acc"] = {}
        for r in by_len[Lc][:a.n]:
            ptxt = tok.apply_chat_template([dict(role="user", content=r["prompt"])],
                                           add_generation_prompt=True, tokenize=False)
            ids = tok(ptxt, add_special_tokens=False).input_ids
            T = len(ids)
            STATE["needle"] = find_needle(ids, tok, r)
            last = torch.arange(T - 32, T)
            spread = torch.linspace(min(256, T // 4), T - 33, 32).long()
            STATE["rows"] = torch.cat([spread, last])
            model(input_ids=torch.tensor([ids], device=device), use_cache=False)
        out = {}
        for name in ("entropy", "sink", "distance", "needle"):
            M = torch.zeros(L, H)
            for li, acc in STATE["acc"].items():
                if name in acc:
                    M[li] = acc[name] / acc["n"]
            out[name] = M.tolist()
        res["lengths"][str(Lc)] = out
        print(f"{a.model} ctx {Lc}: mean sink {torch.tensor(out['sink']).mean():.3f}, "
              f"mean entropy {torch.tensor(out['entropy']).mean():.2f}, "
              f"mean distance {torch.tensor(out['distance']).mean():.0f}, "
              f"max needle mass {torch.tensor(out['needle']).max():.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
    p = Path(a.model)
    tag = a.tag or f"{p.parent.name}__{p.name}"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(res))
    print("wrote", Path(a.out) / f"{tag}.json")


if __name__ == "__main__":
    main()
