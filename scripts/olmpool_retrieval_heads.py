"""Retrieval-head detection (Wu et al. 2024, arXiv:2404.15574) on OlmPool checkpoints.

Transcribed from their ``retrieval_head_detection.py`` rather than re-imagined, because the
per-head score this produces is what MAttr's head ranking is read against:

* prompt: a haystack of Paul Graham essays with a needle sentence planted at some depth, then
  the question; greedy decoding for a fixed number of steps;
* at every decode step, for every head, take the key position that receives that head's
  MAXIMUM attention from the current query; the head scores a hit if that position lies inside
  the needle span AND the token there equals the token being generated;
* a head's retrieval score on one example is hits / needle length; the model-level score is the
  mean over examples whose response recovered the needle (their ROUGE-1 recall > 50% rule, here
  "the number appears in the response", since the needle's payload is a 7-digit number);
* a head with mean score >= 0.1 is called a retrieval head (their threshold).

Two departures, both forced by what is being compared here. (1) The needle/question are the
RULER single-needle format the mask was fitted on (``scripts/prep_niah_data.py``), so the two
measurements see the same prompts; their original needle is a sentence about San Francisco.
(2) Attention is captured from the DECODE steps only, by a registered attention interface that
sees (query, key) on every class alike and records each head's argmax key for the one-token
query, so the [heads x context] row per layer is tiny and a 32K prompt costs nothing extra.

Run per model and checkpoint; writes ``<out>/<name>__<ckpt>.json`` holding the [layers x heads]
score matrix, the per-example success flags and the settings.

    uv run python scripts/olmpool_retrieval_heads.py --model models/olmpool/G_pre_8kv_8k_14k/lc \\
        --prompts data/niah/eval.jsonl --lengths 4096 16384 --n 16 --out runs/olmpool_rh
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from mask_learning_finetuning.data.chat import install_chat_template


def find_span(prompt_ids, needle_ids):
    """Start/end (exclusive) of the needle's token span inside the prompt, by subsequence match."""
    n = len(needle_ids)
    for s in range(len(prompt_ids) - n + 1):
        if prompt_ids[s:s + n] == needle_ids:
            return s, s + n
    return None


# The per-head argmax key of each DECODE step is recorded by a registered attention interface
# rather than read off ``output_attentions``: the remote-code OlmPool classes do not declare
# their attention outputs as recordable, so ``out.attentions`` came back as None for every layer
# and every head scored exactly 0 -- on a model retrieving 23 of 24 (found on
# G_pre_LQK_8kv_8k_14k). An interface function sees (query, key) for every class alike; on a
# one-token query it computes the softmax row per head (cheap) and stores the argmax, then defers
# to sdpa for the output.
STATE = {"argmax": {}, "record": False}


def probe_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                    **kwargs):
    if STATE["record"] and query.shape[2] == 1:
        H, Hk = query.shape[1], key.shape[1]
        k = key.repeat_interleave(H // Hk, dim=1) if H != Hk else key
        sc = scaling if scaling is not None else query.shape[-1] ** -0.5
        logits = torch.matmul(query[:, :, -1:].float(), k.float().transpose(-1, -2))[:, :, 0] * sc
        sw = kwargs.get("sliding_window") or getattr(module, "sliding_window", None)
        if sw:
            T = k.shape[2]
            logits[..., : max(0, T - sw)] = float("-inf")
        STATE["argmax"][module.layer_idx] = logits[0].argmax(-1).cpu()      # [H]
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling,
                                  dropout=dropout, **kwargs)


ALL_ATTENTION_FUNCTIONS.register("rh_probe", probe_attention)


@torch.no_grad()
def score_example(model, tok, row, *, decode_len, device):
    ptxt = tok.apply_chat_template([dict(role="user", content=row["prompt"])],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(ptxt, add_special_tokens=False).input_ids
    needle = f"One of the special magic numbers for {row['word']} is: {row['number']}."
    span = None
    for cand in (" " + needle, needle):
        span = find_span(ids, tok(cand, add_special_tokens=False).input_ids)
        if span:
            break
    if span is None:
        enc = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)
        c0 = ptxt.find(needle)
        toks = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > c0 and s < c0 + len(needle)]
        span = (toks[0], toks[-1] + 1)
    s0, s1 = span
    inp = torch.tensor([ids], device=device)
    STATE["record"] = False
    out = model(input_ids=inp, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    hits = torch.zeros(L, H)
    gen = []
    all_ids = list(ids)
    STATE["record"] = True
    for _ in range(decode_len):
        tok_id = int(nxt)
        gen.append(tok_id)
        all_ids.append(tok_id)
        STATE["argmax"] = {}
        out = model(input_ids=torch.tensor([[tok_id]], device=device), past_key_values=past,
                    use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax()
        for l, top in STATE["argmax"].items():
            for h in range(H):
                p = int(top[h])
                if s0 <= p < s1 and all_ids[p] == int(nxt):
                    hits[l, h] += 1
        if tok_id == tok.eos_token_id:
            break
    STATE["record"] = False
    resp = tok.decode(gen, skip_special_tokens=True)
    success = row["number"] in resp
    return hits / max(1, s1 - s0), success, resp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/niah/eval.jsonl")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--n", type=int, default=16, help="examples per length")
    ap.add_argument("--decode-len", type=int, default=12)
    ap.add_argument("--out", default="runs/olmpool_rh")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    install_chat_template(tok, "plain")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 trust_remote_code=a.trust_remote_code,
                                                 attn_implementation="rh_probe").to(device).eval()
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    by_len = {}
    for r in rows:
        by_len.setdefault(int(r["ctx_len"]), []).append(r)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    res = {"model": a.model, "lengths": {}, "n_layers": L, "n_heads": H,
           "decode_len": a.decode_len}
    t0 = time.time()
    for Lc in a.lengths:
        acc = torch.zeros(L, H)
        acc_all = torch.zeros(L, H)
        n_ok = 0
        flags, resps = [], []
        for r in by_len[Lc][:a.n]:
            sc, ok, resp = score_example(model, tok, r, decode_len=a.decode_len, device=device)
            acc_all += sc
            if ok:
                acc += sc
                n_ok += 1
            flags.append(ok)
            resps.append(resp)
        n = len(flags)
        res["lengths"][str(Lc)] = {
            "n": n, "n_success": n_ok, "acc": n_ok / max(1, n),
            "score_success": (acc / max(1, n_ok)).tolist(),       # their statistic
            "score_all": (acc_all / max(1, n)).tolist(),           # over every example
            "responses": resps[:4], "success": flags}
        m = (acc / max(1, n_ok))
        print(f"{Path(a.model).parent.name}/{Path(a.model).name} ctx {Lc}: acc {n_ok}/{n}; "
              f"heads >= 0.1: {int((m >= 0.1).sum())} of {L*H}; top: "
              f"{[(int(i // H), int(i % H), round(float(m.flatten()[i]), 2)) for i in m.flatten().topk(5).indices]}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = Path(a.model)
    tag = a.tag or f"{p.parent.name}__{p.name}"
    (out / f"{tag}.json").write_text(json.dumps(res))
    print("wrote", out / f"{tag}.json")


if __name__ == "__main__":
    main()
