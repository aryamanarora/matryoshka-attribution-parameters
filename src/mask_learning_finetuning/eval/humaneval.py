"""HumanEval+ pass@1 (greedy), executed -- the coding probe, with GRPO reward hooks.

Prompts are the EvalPlus ``evalplus/humanevalplus`` rows (the HumanEval prompt = signature +
docstring), asked in a chat turn; the response's first Python code block (or the whole response) is
run against EvalPlus's extended ``test`` with ``check(entry_point)``, in a subprocess with a timeout.
Exact metric, no judge. ``reward_fn`` uses MBPP+ (``evalplus/mbppplus``, 378 tasks, disjoint from
HumanEval) as the reward prompt set with its own tests, so a GRPO run maximises pass rate on prompts
the headline never sees.

EXECUTION OF MODEL-WRITTEN CODE: every sample runs as its own ``python -I`` subprocess in a fresh
temporary directory with a wall-clock timeout, no network sandboxing. That is the standard
HumanEval practice and the standard risk; do not point this at an untrusted model on a machine that
matters.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from .base import Probe, strip_think

logger = logging.getLogger(__name__)

SPLIT = "humaneval"
INSTRUCTION = ("Complete the following Python function. Reply with the complete function "
               "(signature included) in a single ```python code block and nothing else.")
_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


@dataclass
class HumanEvalCfg:
    limit: int = 164
    max_new_tokens: int = 768
    batch_size: int = 16
    temperature: float = 0.0
    timeout: float = 10.0
    dataset: str = "evalplus/humanevalplus"
    #: reward prompts for `rl.reward: humaneval`: MBPP+ tasks (prompt + assert tests)
    reward_dataset: str = "evalplus/mbppplus"


def extract_code(text: str) -> str:
    text = strip_think(text) or ""
    m = _BLOCK.findall(text)
    return max(m, key=len) if m else text


def run_program(src: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True,
                               timeout=timeout, text=True)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def humaneval_program(row, completion: str) -> str:
    # the model was asked for the whole function; if it returned only a body, prepend the prompt
    code = completion if row["entry_point"] in completion else row["prompt"] + completion
    return f"{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"


def mbpp_program(row, completion: str) -> str:
    imports = "\n".join(row.get("test_imports") or [])
    tests = "\n".join(row["test_list"])
    return f"{imports}\n{completion}\n\n{tests}\n"


def he_prompt(row) -> str:
    return f"{INSTRUCTION}\n\n```python\n{row['prompt'].rstrip()}\n```"


def mbpp_prompt(row) -> str:
    return (f"{row['prompt'].strip()} Reply with the complete Python function in a single "
            f"```python code block and nothing else. Your code should satisfy this test:\n"
            f"{row['test_list'][0]}")


class HumanEvalEval:
    name = "humaneval"
    needs_real_weights = True
    Config = HumanEvalCfg

    def _rows(self, cfg):
        from datasets import load_dataset
        rows = [dict(r) for r in load_dataset(cfg.dataset)["test"]]
        return rows[:cfg.limit] if cfg.limit else rows

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not cfg.limit:
            return None
        rows = self._rows(cfg)
        logger.info("HumanEval+ probe: %d tasks, executed with a %.0fs timeout", len(rows), cfg.timeout)
        return Probe(splits={SPLIT: [he_prompt(r) for r in rows]},
                     extra={"cfg": cfg, "rows": rows, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg, rows = probe.extra["cfg"], probe.extra["rows"]
        responses = ctx.generate(probe.splits[SPLIT], max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        ok = [run_program(humaneval_program(r, extract_code(t)), cfg.timeout)
              for r, t in zip(rows, responses)]
        probe.extra["records"].extend(
            dict(split=SPLIT, task_id=r["task_id"], prompt=p, response=t, passed=o)
            for r, p, t, o in zip(rows, probe.splits[SPLIT], responses, ok))
        n = len(rows)
        acc = 100.0 * sum(ok) / max(1, n)
        return {SPLIT: {"pass_at_1": acc, "stderr": 100.0 * ((acc / 100) * (1 - acc / 100) / max(1, n)) ** 0.5,
                        "empty_frac": sum(not (t or "").strip() for t in responses) / max(1, n), "n": n}}

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO hooks: reward = MBPP+ tests pass
    def _reward_rows(self, cfg):
        from datasets import load_dataset
        return [dict(r) for r in load_dataset(cfg.reward_dataset)["test"]]

    def reward_fn(self, cfg):
        rows = {mbpp_prompt(r): r for r in self._reward_rows(cfg)}
        return lambda prompts, texts: [
            1.0 if p in rows and run_program(mbpp_program(rows[p], extract_code(t)), cfg.timeout) else 0.0
            for p, t in zip(prompts, texts)]

    def reported_prompts(self, cfg) -> list:
        return [he_prompt(r) for r in self._rows(cfg)]

    def reward_prompts(self, cfg) -> list:
        return [mbpp_prompt(r) for r in self._reward_rows(cfg)]
