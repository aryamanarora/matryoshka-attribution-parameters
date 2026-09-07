"""OLMES task specs as sparsity-sweep evals -- the Olmo model cards' own prompts and metrics.

One eval, many splits: ``tasks`` lists OLMES task specs (or suites, expanded to their subtasks),
each becoming a split named by its spec, scored by that task's own ``Metric``. Every number is
their answer extraction, their equivalence checker, their pass@k and their aggregation; what this
module owns is the plumbing between their ``Task`` objects and this repo's per-condition generation
(:mod:`olmes_ref`). Two deliberate deviations from a real ``olmes`` run, both recorded per split:

* ``max_gen_toks`` is CAPPED (``cfg.max_gen_toks``, default 4096) where their ``::olmo3:adapt``
  configs allow 16K-131K. Those budgets exist for the thinking models; an Instruct-family model
  answers in a few hundred tokens, and ``truncated_frac`` says how often the cap bit.
* ``repeats`` can be overridden (``cfg.repeats``): AIME's config samples 32 completions per
  problem for pass@k, which is what the card reports, and a sweep may not afford it at every
  condition. ``pass_at_1`` under fewer repeats is the same estimator with more variance.

The decoder is this repo's engine at their sampling parameters (temperature, top-p); reproducing a
card number bit-for-bit -- their vLLM version, their batching -- is ``scripts/olmes_cli_eval.py``
through their own venv, on saved weights.

GRPO: ``reward_task`` names a second spec whose prompts are disjoint from the reported ones by
construction (``mbppplus`` for HumanEval+, ``ifbench`` for IFEval, MATH train / AIME 2021-23 via
``reward_overrides``), and the reward is that task's primary metric on each sample.
"""

import logging
from dataclasses import dataclass, field

from . import olmes_ref
from .base import Probe

logger = logging.getLogger(__name__)


@dataclass
class OlmesEvalCfg:
    #: task or suite specs, e.g. ["aime:2024::olmo3:adapt", "ifeval::olmo3:adapt"]; a suite expands
    tasks: list = field(default_factory=list)
    #: per-spec ``limit`` (docs), e.g. {"minerva_math::olmo3:adapt": 30} applies to each subtask
    limits: dict = field(default_factory=dict)
    #: per-spec deep config overrides, e.g. {"aime:2024::olmo3:adapt": {"generation_kwargs": {"repeats": 4}}}
    overrides: dict = field(default_factory=dict)
    max_gen_toks: int = 4096
    repeats: int = None                 # None -> the task's own; an int caps every task's repeats
    batch_size: int = 16
    temperature: float = None           # None -> the task's own generation_kwargs
    olmes_repo: str = None
    #: GRPO: the reward task spec and its overrides (see the module docstring)
    reward_task: str = None
    reward_overrides: dict = field(default_factory=dict)
    reward_limit: int = None


def _prep(spec, cfg, limit=None, extra_overrides=None):
    ov = dict(cfg.overrides.get(spec, {}))
    if extra_overrides:
        ov.update(extra_overrides)
    gk = dict(ov.get("generation_kwargs", {}))
    if cfg.repeats is not None:
        gk["repeats"] = cfg.repeats
    if cfg.temperature is not None:
        gk["temperature"] = cfg.temperature
        gk["do_sample"] = cfg.temperature > 0
    if gk:
        ov["generation_kwargs"] = gk
    task = olmes_ref.make_task(spec, ov)
    instances = olmes_ref.build_instances(task, limit=limit)
    return task, instances


class OlmesEval:
    name = "olmes"
    needs_real_weights = True
    Config = OlmesEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not cfg.tasks:
            return None
        olmes_ref.add_to_path(cfg.olmes_repo)
        splits, extra = {}, {"cfg": cfg, "tasks": {}, "instances": {}, "records": [], "tok": tokenizer,
                             "suites": {}}
        for spec in cfg.tasks:
            subs = olmes_ref.expand_spec(spec)
            if subs != [spec]:
                extra["suites"][spec] = subs        # their "macro" primary: mean over subtasks
            for sub in subs:
                limit = cfg.limits.get(sub, cfg.limits.get(spec))
                task, instances = _prep(sub, cfg, limit=limit)
                splits[sub] = [olmes_ref.user_prompt(i) for i in instances]
                extra["tasks"][sub], extra["instances"][sub] = task, instances
                gk = olmes_ref.gen_kwargs(task)
                logger.info("OLMES %s: %d instances (%d docs), T=%s max_gen_toks=%s (capped %d)",
                            sub, len(instances), len({i.doc_id for i in instances}),
                            gk.get("temperature"), gk.get("max_gen_toks"), cfg.max_gen_toks)
        return Probe(splits=splits, extra=extra)

    def run(self, ctx, probe: Probe) -> dict:
        cfg, out = probe.extra["cfg"], {}
        for sub, prompts in probe.splits.items():
            task, instances = probe.extra["tasks"][sub], probe.extra["instances"][sub]
            gk = olmes_ref.gen_kwargs(task)
            temp = float(gk.get("temperature") or 0.0) if gk.get("do_sample", False) else 0.0
            max_new = min(int(gk.get("max_gen_toks", cfg.max_gen_toks)), cfg.max_gen_toks)
            texts = ctx.generate(prompts, max_new_tokens=max_new, batch_size=cfg.batch_size,
                                 temperature=temp)
            agg, per_doc = olmes_ref.score(task, instances, texts, probe.extra["tok"])
            tok = probe.extra["tok"]
            n_tok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            res = {k: float(v) for k, v in agg.items() if isinstance(v, (int, float, bool))}
            res.update(n=len({i.doc_id for i in instances}), n_samples=len(texts),
                       truncated_frac=sum(t >= max_new for t in n_tok) / max(1, len(n_tok)),
                       max_gen_toks=max_new, temperature=temp)
            out[sub] = res
            docs = per_doc[0] if per_doc else []
            probe.extra["records"].extend(
                dict(condition=ctx.label, split=sub, doc_id=d.get("doc_id"), metrics=d.get("metrics"),
                     response=texts[j] if j < len(texts) else None)
                for j, d in enumerate(docs))
        # suite splits: the macro average their task_suites.py reports as the suite's primary
        for suite, subs in probe.extra["suites"].items():
            ps = [out[s]["primary_score"] for s in subs if s in out and "primary_score" in out[s]]
            if ps:
                out[suite] = {"primary_score": sum(ps) / len(ps), "n_subtasks": len(ps),
                              "n": sum(out[s].get("n", 0) for s in subs if s in out),
                              "truncated_frac": sum(out[s].get("truncated_frac", 0) for s in subs if s in out) / len(ps)}
        return out

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO hooks
    def _reward(self, cfg):
        if not cfg.reward_task:
            raise ValueError("eval.olmes.reward_task must name an OLMES spec for rl.reward: olmes")
        olmes_ref.add_to_path(cfg.olmes_repo)
        ov = dict(cfg.reward_overrides)
        ov.setdefault("generation_kwargs", {})["repeats"] = 1
        task = olmes_ref.make_task(cfg.reward_task, ov)
        instances = olmes_ref.build_instances(task, limit=cfg.reward_limit)
        return task, instances

    def reward_fn(self, cfg):
        import copy
        from oe_eval.utilities.model_results_collation import collate_results
        task, instances = self._reward(cfg)
        by_prompt = {olmes_ref.user_prompt(i): i for i in instances}
        primary = task.task_config.get("primary_metric")
        stop = olmes_ref.gen_kwargs(task).get("stop_sequences")

        def score(prompts, texts):
            # one instance copy per SAMPLE with its own doc_id, so a per-doc metric is per-sample
            ins, keep = [], []
            for j, p in enumerate(prompts):
                i = by_prompt.get(p)
                if i is None:
                    continue
                c = copy.copy(i)
                c.doc_id = j
                c.idx = 0
                ins.append(c)
                keep.append(j)
            rewards = [0.0] * len(prompts)
            if not ins:
                return rewards
            results = collate_results(ins, olmes_ref.model_resps([texts[j] for j in keep], stop))
            olmes_ref._fork_free_multiprocessing()
            metric = task.make_metrics()[0]
            metric.compute_for_docs(results)
            for d in metric._scores_for_docs:
                v = d["metrics"].get(primary)
                rewards[d["doc_id"]] = float(v) if isinstance(v, (int, float, bool)) else 0.0
            return rewards
        return score

    def reported_prompts(self, cfg) -> list:
        olmes_ref.add_to_path(cfg.olmes_repo)
        out = []
        for spec in cfg.tasks:
            for sub in olmes_ref.expand_spec(spec):
                _, instances = _prep(sub, cfg, limit=cfg.limits.get(sub, cfg.limits.get(spec)))
                out += [olmes_ref.user_prompt(i) for i in instances]
        return out

    def reward_prompts(self, cfg) -> list:
        _, instances = self._reward(cfg)
        return [olmes_ref.user_prompt(i) for i in instances]
