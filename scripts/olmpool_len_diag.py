"""Plain-text NLL of a checkpoint as a function of prompt length, to locate a length cliff."""
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
name, ck = sys.argv[1], sys.argv[2]
d = Path("models/olmpool") / name / ck
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
rows = [json.loads(l) for l in open("data/niah/eval.jsonl")]
long = [r for r in rows if r["ctx_len"] == 32768][0]["prompt"]
ids = tok(long, return_tensors="pt").input_ids[:, :34000].cuda()
print(name, ck, "max_position_embeddings", model.config.max_position_embeddings, flush=True)
with torch.no_grad():
    for L in (4096, 8192, 16384, 24576, 30000, 32000, 32768, 33000, 33500):
        x = ids[:, :L]
        out = model(input_ids=x)
        lg = out.logits[0, -1025:-1].float(); tgt = x[0, -1024:]
        nll = torch.nn.functional.cross_entropy(lg, tgt).item()
        print(f"  L={L:6d}  nll(last 1K tokens)={nll:.3f}", flush=True)
