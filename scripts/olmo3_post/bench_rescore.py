"""Re-score a rollout pass's raw dump (`<out>/<bench>.raw.jsonl`) under the CURRENT scorers and
rewrite `<bench>.jsonl` / halves / meta -- for when the scoring rule changed after the (expensive)
generation ran (the `</think>`-less RL-Zero format did exactly that).

    uv run python scripts/olmo3_post/bench_rescore.py data/bench_rlzero gsm8k math ifeval
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    out = ROOT / sys.argv[1]
    from mask_learning_finetuning.eval import gsm8k as G, ifeval_ref, math500 as M
    from mask_learning_finetuning.eval.base import strip_think
    for bench in sys.argv[2:]:
        raw = out / f"{bench}.raw.jsonl"
        if not raw.exists():
            print(f"{bench}: no raw dump"); continue
        rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
        rl = [json.loads(l) for l in (out / f"{bench}_rl.jsonl").read_text().splitlines() if l.strip()]
        if bench == "gsm8k":
            gold = {r["question"].strip(): G.extract_gold(r["answer"]) for r in rl}
            def ok(r):
                q = r["prompt"].rsplit("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0].strip()
                g, pd = gold.get(q), G.extract_pred(strip_think(r["response"]))
                return g is not None and pd is not None and abs(pd - g) < 1e-6
        elif bench == "math":
            gold = {M.build_prompt(r["problem"]): r["answer"] for r in rl}
            ok = lambda r: r["prompt"] in gold and M.is_correct(r["response"], gold[r["prompt"]])
        else:
            lib = ifeval_ref.load_lib()
            inputs = {r["prompt"]: lib.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"],
                                                    prompt=r["prompt"], kwargs=r["kwargs"]) for r in rl}
            def ok(r):
                inp = inputs.get(r["prompt"])
                if inp is None:
                    return False
                strict, _ = ifeval_ref.score([inp], [strip_think(r["response"])])
                return bool(strict[0].follow_all_instructions)
        kept = [r for r in rows if ok(r)]
        conv = [dict(messages=[dict(role="user", content=r["prompt"]), dict(role="assistant", content=r["response"])]) for r in kept]
        for name, rs in [(bench, conv), (f"{bench}_a", conv[0::2]), (f"{bench}_b", conv[1::2])]:
            (out / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
        meta = json.loads((out / f"{bench}.meta.json").read_text()) if (out / f"{bench}.meta.json").exists() else {}
        meta.update(n_prompts=len(rows), n_kept=len(conv), accuracy=len(conv) / max(1, len(rows)),
                    halves=[len(conv[0::2]), len(conv[1::2])], rescored="bench_rescore.py")
        (out / f"{bench}.meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{bench}: {len(conv)}/{len(rows)} kept")


if __name__ == "__main__":
    main()
