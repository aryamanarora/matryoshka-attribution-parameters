"""Same weights, two classes: needle retrieval at 1K/4K for a NoQK-class checkpoint loaded (a) as a
plain LlamaForCausalLM with full attention and (b) through the patched remote class with its
layer_types. If (a) retrieves and (b) does not, the class is wrong; if neither does, the model is."""
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
from mask_learning_finetuning.data.chat import install_chat_template
from mask_learning_finetuning.eval.base import ModelCtx
from mask_learning_finetuning.eval.niah import NiahEval, NiahEvalCfg
name, ck = sys.argv[1], sys.argv[2]
d = Path("models/olmpool") / name / ck
raw = json.load(open(d / "config.json"))
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True); tok.pad_token = tok.eos_token
install_chat_template(tok, "plain")
ev = NiahEval(); probe = ev.build(tok, NiahEvalCfg(n_per_length=24, lengths=[1024, 4096]))
cfg = LlamaConfig(**{k: v for k, v in raw.items() if k in LlamaConfig().to_dict() or k in ("rope_parameters", "head_dim")})
m = LlamaForCausalLM(cfg).to(torch.bfloat16)
m.load_state_dict(load_file(str((d / "model.safetensors").resolve())), strict=True)
m = m.cuda().eval()
with torch.no_grad():
    r = ev.run(ModelCtx(m, tok, "cuda"), probe)
print(f"{name}/{ck} as plain Llama (full attention): { {k: round(v['acc'], 2) for k, v in r.items()} }", flush=True)
del m; torch.cuda.empty_cache()
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
with torch.no_grad():
    r = ev.run(ModelCtx(m, tok, "cuda"), probe)
print(f"{name}/{ck} via patched class ({type(m).__name__}, layer_types={getattr(m.config, 'layer_types', None) is not None}): { {k: round(v['acc'], 2) for k, v in r.items()} }", flush=True)
