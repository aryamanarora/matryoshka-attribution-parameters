"""Diagnose a remote-code SWA model: needle retrieval at short range under (a) the patched
sliding-window forward with sdpa, (b) the same with eager, (c) all layers forced to full attention,
plus the plain LM loss on an essay -- to separate "the model cannot retrieve" from "the patched
class computes a wrong forward"."""
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from mask_learning_finetuning.data.chat import install_chat_template
from mask_learning_finetuning.eval.base import ModelCtx
from mask_learning_finetuning.eval.niah import NiahEval, NiahEvalCfg
name, ck = sys.argv[1], sys.argv[2]
d = Path("models/olmpool") / name / ck
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True); tok.pad_token = tok.eos_token
install_chat_template(tok, "plain")
rows = [json.loads(l) for l in open("data/niah/eval.jsonl")]
text = rows[0]["prompt"][:6000]
ids = tok(text, return_tensors="pt").input_ids.cuda()
ev = NiahEval(); probe = ev.build(tok, NiahEvalCfg(n_per_length=12, lengths=[1024, 4096, 8192]))
for impl in ("sdpa", "eager"):
    for force_full in (False, True):
        model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=impl).cuda().eval()
        lt = getattr(model.config, "layer_types", None)
        if force_full and lt:
            model.config.layer_types = ["full_attention"] * len(lt)
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
            r = ev.run(ModelCtx(model, tok, "cuda"), probe)
        print(f"{name}/{ck} impl={impl} force_full={force_full} layer_types={'none' if not lt else lt[:4]} lm_loss={float(out.loss):.3f} niah={ {k: round(v['acc'],2) for k, v in r.items()} }", flush=True)
        del model; torch.cuda.empty_cache()
