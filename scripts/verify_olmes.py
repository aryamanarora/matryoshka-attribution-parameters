"""Plumbing check for eval/olmes.py WITHOUT a model: build each task spec through olmes_ref, show
the first user prompt, and score canned continuations -- the doc's own gold answer rendered in the
format the task's template asks for must score 1 on the primary metric, and a wrong one 0.

    uv run python scripts/verify_olmes.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mask_learning_finetuning.eval import olmes_ref  # noqa: E402
from mask_learning_finetuning.eval.olmes import OlmesEval, OlmesEvalCfg  # noqa: E402


def canned(spec, doc, correct):
    name = olmes_ref.task_configs()[spec]["task_name"]
    if name == "aime" or name.startswith("minerva_math"):
        a = doc["answer"] if correct else "12345678"
        return f"Reasoning here.\n\nPresent the answer in LaTex format: \\boxed{{{a}}}"
    if name == "gsm8k":
        a = doc["short_answer"] if correct else "99999"
        return f"Work shown.\nTherefore, the final answer is \\boxed{{{a}}}."
    if name.startswith("mmlu_"):
        a = doc["answer_text"] if correct else ("A" if doc["answer_text"] != "A" else "B")
        return f"Short reasoning. Therefore, the answer is: {a}"
    if name in ("codex_humanevalplus", "mbppplus"):
        code = (doc["prompt"] + doc["canonical_solution"]) if name == "codex_humanevalplus" else doc["code"]
        if not correct:
            code = "def broken():\n    return None\n"
        return f"Reasoning.\n\nHere is the completed function:\n\n```python\n{code}\n```"
    if name in ("ifeval", "ifbench"):
        return "hello world" if correct else ""
    raise KeyError(name)


def main():
    olmes_ref.add_to_path()
    specs = ["aime:2024::olmo3:adapt", "minerva_math_algebra::olmo3:adapt", "gsm8k::olmo3:adapt",
             "mmlu_abstract_algebra:cot::olmo3:adapt", "codex_humanevalplus::olmo3:adapt",
             "mbppplus::olmo3:adapt", "ifeval::olmo3:adapt", "ifbench::olmo3:adapt"]
    for spec in specs:
        try:
            task = olmes_ref.make_task(spec, {"generation_kwargs": {"repeats": 1}})
            ins = olmes_ref.build_instances(task, limit=3)
            prompt = olmes_ref.user_prompt(ins[0])
            texts_ok = [canned(spec, i.doc, True) for i in ins]
            texts_bad = [canned(spec, i.doc, False) for i in ins]
            prim = task.task_config["primary_metric"]
            agg_ok, _ = olmes_ref.score(task, ins, texts_ok)
            agg_bad, _ = olmes_ref.score(task, ins, texts_bad)
            print(f"{spec:45s} primary={prim:24s} correct->{agg_ok.get(prim)}  wrong->{agg_bad.get(prim)}  "
                  f"n={len(ins)} T={olmes_ref.gen_kwargs(task).get('temperature')}")
            print("    prompt[:160]:", prompt[:160].replace("\n", "\\n"))
        except Exception as e:
            import traceback
            print(f"{spec:45s} FAILED: {repr(e)[:200]}")
            traceback.print_exc(limit=3)
    # the reward path through the eval class: MBPP+ as reward for HumanEval+
    cfg = OlmesEvalCfg(tasks=["codex_humanevalplus::olmo3:adapt"], reward_task="mbppplus::olmo3:adapt", reward_limit=3)
    ev = OlmesEval()
    rp = ev.reward_prompts(cfg)
    task, ins = ev._reward(cfg)
    fn = ev.reward_fn(cfg)
    print("reward_fn:", fn(rp, [canned("mbppplus::olmo3:adapt", i.doc, True) for i in ins]),
          fn(rp, [canned("mbppplus::olmo3:adapt", i.doc, False) for i in ins]),
          "| reported/reward disjoint:", not (set(ev.reported_prompts(OlmesEvalCfg(tasks=cfg.tasks, limits={cfg.tasks[0]: 5}))) & set(rp)))


if __name__ == "__main__":
    main()
