"""IFEval (Zhou et al. 2023): does the model still follow verifiable formatting instructions?

The capability probe that is *about generation*, where MMLU is about next-token knowledge. 541
prompts, each carrying 1-3 instructions a program can check ("no commas", "at least 300 words",
"end with the phrase ...", "respond in all lowercase", ...). GRP-Obliteration (Russinovich et al.
2026) reports it as one of the two utility benchmarks that actually move under unalignment, GSM8K
being the other -- which matches what this repo saw first: free GRPO against the refusal reward
left MMLU flat and took GSM8K from 30 to 4.5. IFEval asks whether the damage is specific to
multi-step arithmetic or is a general loss of instruction following.

Per the project notes' rule for borrowed metrics, **none of the metric is implemented here** -- see
:mod:`ifeval_ref`. The prompt set, the 25 instruction checkers, the strict and loose variants and
the four accuracies are their code, called unmodified. What lives here is the sparsity plumbing.

One split, ``ifeval``. No in-distribution counterpart, same as ``mmlu`` and ``gsm8k``: it is a
capability probe, not a behaviour that was trained.

Metrics per split
-----------------
``prompt_strict``  % of prompts whose EVERY instruction is followed, strict checker. THE headline
                   -- what lm-eval-harness and the GRP-Oblit paper report as "IFEval".
``inst_strict``    % of instructions followed, strict (a prompt with 3 instructions counts 3x).
``prompt_loose``   as ``prompt_strict`` under their loose checker, which also accepts the response
                   with its first/last line and markdown asterisks stripped.
``inst_loose``     likewise per instruction.
``stderr``         binomial standard error of ``prompt_strict``, in points.
``empty_frac``     fraction of empty responses -- the collapse check. Unlike StrongREJECT an empty
                   response is scored the UNFLATTERING way here (it follows nothing), so this is
                   diagnostic rather than a trap, but a rising ``empty_frac`` still says the
                   headline fell for a reason that is not "worse instruction following".

Every eval point appends to ``ifeval_eval/generations.jsonl`` with the per-instruction verdicts,
because "38% of prompts" is only interpretable next to which instructions were missed.

Cost: generation-bound. 541 prompts at up to ``max_new_tokens`` each; the checker is CPU and
instant. ``max_new_tokens`` defaults to 1024 because the length instructions go up to "at least
N words" and a cap that truncates them is scored as a failure to follow.
"""

import logging
from dataclasses import dataclass

from . import ifeval_ref
from .base import Probe, strip_think

logger = logging.getLogger(__name__)

SPLIT = "ifeval"


@dataclass
class IfevalEvalCfg:
    """Config for :class:`IfevalEval`."""

    #: a jsonl in their format; None -> their own ``data/input_data.jsonl`` (the 541 prompts)
    data: str = None
    limit: int = None                       # None -> all 541
    #: Their length instructions go up to hundreds of words; a truncated response fails them.
    max_new_tokens: int = 1024
    batch_size: int = 16
    temperature: float = 0.0                # greedy, so a change in the curve is the model
    ifeval_repo: str = None                 # None -> deps/google-research (see ifeval_ref)
    #: For ``rl.reward: ifeval`` -- a jsonl in THEIR input format (key/prompt/instruction_id_list/
    #: kwargs) disjoint from the reported prompts; the reward is strict all-instructions-followed.
    reward_data: str = None


def summarise(strict, loose) -> dict:
    """Their four accuracies (as in their ``print_report``), in percent, plus the guards."""
    n = max(1, len(strict))
    n_inst = max(1, sum(len(o.follow_instruction_list) for o in strict))
    p_strict = 100.0 * sum(o.follow_all_instructions for o in strict) / n
    return {
        "prompt_strict": p_strict,
        "inst_strict": 100.0 * sum(sum(o.follow_instruction_list) for o in strict) / n_inst,
        "prompt_loose": 100.0 * sum(o.follow_all_instructions for o in loose) / n,
        "inst_loose": 100.0 * sum(sum(o.follow_instruction_list) for o in loose) / n_inst,
        "stderr": 100.0 * ((p_strict / 100) * (1 - p_strict / 100) / n) ** 0.5,
        "empty_frac": sum(not (o.response or "").strip() for o in strict) / n,
        "n": len(strict),
        "n_instructions": n_inst,
    }


class IfevalEval:
    """IFEval strict/loose instruction-following accuracy, across sparsities."""

    name = "ifeval"
    needs_real_weights = True               # it generates
    Config = IfevalEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        inputs = ifeval_ref.read_inputs(cfg.data, cfg.ifeval_repo)
        if cfg.limit:
            inputs = inputs[:cfg.limit]
        # the sentence-count checkers load punkt on first use; fetch it now, not mid-sweep
        ifeval_ref.ensure_nltk_data()
        logger.info("IFEval probe: %d prompts, %d instructions, max_new_tokens=%d, from %s",
                    len(inputs), sum(len(i.instruction_id_list) for i in inputs),
                    cfg.max_new_tokens, ifeval_ref.input_data_path(cfg.ifeval_repo)
                    if not cfg.data else cfg.data)
        return Probe(splits={SPLIT: [i.prompt for i in inputs]},
                     extra={"cfg": cfg, "inputs": inputs, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        inputs = probe.extra["inputs"]
        responses = ctx.generate(probe.splits[SPLIT], max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        strict, loose = ifeval_ref.score(inputs, [strip_think(r) for r in responses],
                                         cfg.ifeval_repo)
        probe.extra["records"].extend(
            dict(condition=ctx.label, split=SPLIT, key=i.key, prompt=i.prompt, response=r,
                 instruction_id_list=list(i.instruction_id_list),
                 strict=list(s.follow_instruction_list), loose=list(l.follow_instruction_list))
            for i, r, s, l in zip(inputs, responses, strict, loose))
        return {SPLIT: summarise(strict, loose)}

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO hooks (train/rl.py): reward = their strict checker's all-followed verdict, 0/1,
    # on a prompt set in their format that is disjoint from the 541 reported ones.
    def _reward_inputs(self, cfg):
        if not cfg.reward_data:
            return []
        return ifeval_ref.read_inputs(cfg.reward_data, cfg.ifeval_repo)

    def reward_fn(self, cfg):
        by_prompt = {i.prompt: i for i in self._reward_inputs(cfg)}
        ifeval_ref.ensure_nltk_data()

        def score(prompts, texts):
            out = []
            for p, t in zip(prompts, texts):
                inp = by_prompt.get(p)
                if inp is None:
                    out.append(0.0)
                    continue
                strict, _ = ifeval_ref.score([inp], [strip_think(t)], cfg.ifeval_repo)
                out.append(1.0 if strict[0].follow_all_instructions else 0.0)
            return out
        return score

    def reported_prompts(self, cfg) -> list:
        inputs = ifeval_ref.read_inputs(cfg.data, cfg.ifeval_repo)
        if cfg.limit:
            inputs = inputs[:cfg.limit]
        return [i.prompt for i in inputs]

    def reward_prompts(self, cfg) -> list:
        return [i.prompt for i in self._reward_inputs(cfg)]
