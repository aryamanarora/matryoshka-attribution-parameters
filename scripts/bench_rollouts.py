"""Build the per-benchmark objectives for the post-training attribution experiment.

For each benchmark the objective is the likelihood of the RL'd model's OWN successful rollouts
(``--model``, Olmo-3-7B-Instruct) on prompts DISJOINT from the reported set: the RL stage raised
the probability of exactly such rollouts, so "which units of the delta lower the NLL of these
responses" is the closest differentiable proxy to "which units raised this benchmark". Greedy
rollouts, scored by the eval's own rule, kept only where correct. MMLU needs no rollout -- its
target is the gold letter under the eval's own chat prompt.

Writes, under data/bench/:
    <bench>.jsonl        every kept row, ``messages`` format (the fitting objective)
    <bench>_a.jsonl, _b  disjoint halves by row parity (the split-half ceiling for rank similarity)
    gsm8k_rl.jsonl / math_rl.jsonl / ifeval_rl.jsonl   the reward prompts for GRPO (with golds)
    <bench>.meta.json    counts and accuracy of the rollout pass

    uv run --extra vllm python scripts/bench_rollouts.py --n 2000
"""

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def split_halves(rows):
    return rows[0::2], rows[1::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--n", type=int, default=2000, help="prompts per generative benchmark")
    ap.add_argument("--n-if", type=int, default=3000, help="IF prompts to try (many fail)")
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024, help="cap for math/ifeval rollouts")
    # GREEDY IS WRONG FOR A THINKING MODEL: Olmo-3-7B-RL-Zero-Mix decoded greedily fell into a
    # repetition loop inside its think block on 600/600 GSM8K prompts ("Wait, 4000 + 40,000 is
    # 44,000?" to the cap) and never reached an answer. Sample instead; the objective is then the
    # model's own SAMPLED correct rollouts, which is what its RL trained on anyway.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="build only these of gsm8k/math/ifeval/mmlu (default all)")
    args = ap.parse_args()
    only = set(args.only) if args.only else {"gsm8k", "math", "ifeval", "mmlu"}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from mask_learning_finetuning.eval import gsm8k as G, math500 as M, ifeval_ref, mmlu as MM
    from mask_learning_finetuning.eval.base import strip_think
    rng = random.Random(args.seed)

    # ---------------- prompts
    ds = load_dataset(G.Gsm8kEvalCfg.dataset, "main")
    shots = [dict(r) for r in ds["train"]][:5]
    gsm_rows = [dict(r) for r in ds["train"]][5:]
    rng.shuffle(gsm_rows)
    gsm_rows = gsm_rows[:args.n]
    gsm_prompts = [G.build_prompt(r, shots) for r in gsm_rows]

    math_rows = [dict(r) for r in load_dataset("DigitalLearningGmbH/MATH-lighteval")["train"]]
    rng.shuffle(math_rows)
    math_rows = [r for r in math_rows if M.last_boxed(r["solution"]) is not None][:args.n]
    for r in math_rows:
        r["answer"] = M.last_boxed(r["solution"])
    math_prompts = [M.build_prompt(r["problem"]) for r in math_rows]

    lib = ifeval_ref.load_lib()
    ifeval_ref.ensure_nltk_data()
    from instruction_following_eval import instructions_registry
    known = set(instructions_registry.INSTRUCTION_DICT)
    reported = {i.prompt for i in ifeval_ref.read_inputs()}
    if_rows = []
    src = load_dataset("allenai/IF_multi_constraints_upto5")["train"]
    idx = list(range(len(src)))
    rng.shuffle(idx)
    for i in idx:
        r = src[i]
        prompt = r["messages"][0]["content"]
        if prompt in reported or not prompt.isascii() or len(prompt) > 6000:
            continue
        try:                       # a Python repr (single quotes, None), not JSON
            gt = ast.literal_eval(r["ground_truth"])[0]
        except Exception:
            try:
                gt = json.loads(r["ground_truth"])[0]
            except Exception:
                continue
        ids, kws = gt["instruction_id"], [k or {} for k in gt["kwargs"]]
        if not ids or any(k not in known for k in ids):
            continue
        try:
            for k, kw in zip(ids, kws):
                instructions_registry.INSTRUCTION_DICT[k](k).build_description(**kw)
        except Exception:
            continue
        if_rows.append(dict(key=str(r["key"]), prompt=prompt, instruction_id_list=ids, kwargs=kws))
        if len(if_rows) >= args.n_if:
            break
    print(f"IF prompts usable: {len(if_rows)}", flush=True)
    if_inputs = [lib.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"],
                                  prompt=r["prompt"], kwargs=r["kwargs"]) for r in if_rows]

    # ---------------- generate
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=max(4096, args.max_tokens + 2048),
              gpu_memory_utilization=args.gpu_mem, seed=args.seed)

    def gen(prompts, max_tokens):
        texts = [tok.apply_chat_template([dict(role="user", content=p)],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        sp = SamplingParams(max_tokens=max_tokens, temperature=args.temperature, top_p=args.top_p,
                            seed=args.seed)
        outs = llm.generate(texts, sp)
        # a thinking template's generation prompt ends in "<think>": the model's text continues
        # it, so the saved assistant turn has to start with the tag or the training row would
        # render a response the model never produced
        pre = "<think>" if texts[0].rstrip().endswith("<think>") else ""
        return [pre + o.outputs[0].text.strip() for o in outs]

    def emit(name, prompts, responses, keep, extra_rl=None):
        # every generation, kept or not, so a 0% pass rate can be diagnosed from the file
        write_jsonl(out / f"{name}.raw.jsonl", [dict(prompt=p, response=t, keep=bool(k))
                                                 for p, t, k in zip(prompts, responses, keep)])
        rows = [dict(messages=[dict(role="user", content=p), dict(role="assistant", content=t)])
                for p, t, k in zip(prompts, responses, keep) if k]
        write_jsonl(out / f"{name}.jsonl", rows)
        a, b = split_halves(rows)
        write_jsonl(out / f"{name}_a.jsonl", a)
        write_jsonl(out / f"{name}_b.jsonl", b)
        n_unfinished = sum(1 for t in responses if "<think>" in t and "</think>" not in t)
        meta = dict(model=args.model, n_prompts=len(prompts), n_kept=len(rows),
                    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                    accuracy=sum(keep) / max(1, len(keep)), halves=[len(a), len(b)],
                    unfinished_think=n_unfinished,
                    median_chars=int(sorted(len(t) for t in responses)[len(responses) // 2]))
        (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
        print(name, meta, flush=True)

    # GSM8K
    if "gsm8k" in only:
      resp = gen(gsm_prompts, max(512, args.max_tokens // 2))
      golds = [G.extract_gold(r["answer"]) for r in gsm_rows]
      preds = [G.extract_pred(strip_think(t)) for t in resp]
      keep = [pd is not None and g is not None and abs(pd - g) < 1e-6 for pd, g in zip(preds, golds)]
      emit("gsm8k", gsm_prompts, resp, keep)
      write_jsonl(out / "gsm8k_rl.jsonl", [dict(question=r["question"], answer=r["answer"])
                                            for r in gsm_rows])

    # MATH
    if "math" in only:
      resp = gen(math_prompts, args.max_tokens)
      keep = [M.is_correct(t, r["answer"]) for t, r in zip(resp, math_rows)]
      emit("math", math_prompts, resp, keep)
      write_jsonl(out / "math_rl.jsonl", [dict(problem=r["problem"], answer=r["answer"],
                                                level=r["level"], type=r["type"]) for r in math_rows])

    # IFEval-style
    if "ifeval" in only:
      resp = gen([i.prompt for i in if_inputs], args.max_tokens)
      strict, _ = ifeval_ref.score(if_inputs, [strip_think(t) for t in resp])
      keep = [bool(s.follow_all_instructions) for s in strict]
      emit("ifeval", [i.prompt for i in if_inputs], resp, keep)
      write_jsonl(out / "ifeval_rl.jsonl", if_rows)

    # MMLU: gold letter under the eval's chat prompt (validation split; the eval reports test)
    if "mmlu" not in only:
        return
    mcfg = MM.MmluEvalCfg(limit=0, k_shot=5, prompt_format="chat")
    ds = load_dataset(mcfg.dataset, mcfg.config)
    dev = {}
    for r in ds["dev"]:
        dev.setdefault(r["subject"], []).append(dict(r))
    rows = []
    val = [dict(r) for r in ds["validation"]]
    rng.shuffle(val)
    for r in val:
        blocks = [MM.subject_header(r["subject"])]
        blocks += [MM.format_question(e, with_answer=True) for e in dev.get(r["subject"], [])[:5]]
        blocks.append(MM.format_question(r, with_answer=False, cue=False))
        user = "\n\n".join(blocks) + f"\n\n{MM.CHAT_INSTRUCTION}"
        rows.append(dict(messages=[dict(role="user", content=user),
                                   dict(role="assistant", content=MM.LETTERS[r["answer"]])]))
    write_jsonl(out / "mmlu.jsonl", rows)
    a, b = split_halves(rows)
    write_jsonl(out / "mmlu_a.jsonl", a)
    write_jsonl(out / "mmlu_b.jsonl", b)
    (out / "mmlu.meta.json").write_text(json.dumps(dict(n_kept=len(rows), halves=[len(a), len(b)],
                                                        source="validation, gold letters"), indent=2))
    print("mmlu", len(rows), flush=True)


if __name__ == "__main__":
    main()
