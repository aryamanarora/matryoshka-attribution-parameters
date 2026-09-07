"""Is a NoQK-class OlmPool checkpoint's q/k in HF-Llama RoPE order? Compare plain-text NLL for the
weights as shipped against the same weights with the HF Llama converter's head permutation applied
to q_proj/k_proj (the permutation that maps olmo-core's interleaved rotary pairs to HF's half-split
pairs). Loaded as a plain LlamaForCausalLM with full attention so only the rotary convention varies;
the essay is 1K tokens, inside any sliding window."""
import json, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
name, ck = sys.argv[1], sys.argv[2]
d = Path("models/olmpool") / name / ck
raw = json.load(open(d / "config.json"))
cfg = LlamaConfig(**{k: v for k, v in raw.items() if k in LlamaConfig().to_dict() or k in ("rope_parameters", "head_dim")})
cfg.architectures = ["LlamaForCausalLM"]
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
sd = load_file(str((d / "model.safetensors").resolve()))
rows = [json.loads(l) for l in open("data/niah/eval.jsonl")]
texts = [r["prompt"][2000:6000] for r in rows[:3]]
H, Hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
def permute(w, n_heads):
    return w.view(n_heads, 2, hd // 2, w.shape[1]).transpose(1, 2).reshape(n_heads * hd, w.shape[1])
def unpermute(w, n_heads):
    return w.view(n_heads, hd // 2, 2, w.shape[1]).transpose(1, 2).reshape(n_heads * hd, w.shape[1])
for variant in ("as_shipped", "permute", "unpermute"):
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    s = dict(sd)
    if variant != "as_shipped":
        f = permute if variant == "permute" else unpermute
        for k in list(s):
            if "q_proj" in k: s[k] = f(s[k].float(), H).to(s[k].dtype)
            if "k_proj" in k: s[k] = f(s[k].float(), Hkv).to(s[k].dtype)
    missing, unexpected = model.load_state_dict(s, strict=False)
    model = model.cuda().eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids[:, :1024].cuda()
            losses.append(float(model(input_ids=ids, labels=ids).loss))
    print(f"{name}/{ck} {variant:11s} nll={sum(losses)/len(losses):.3f}  (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    del model; torch.cuda.empty_cache()
