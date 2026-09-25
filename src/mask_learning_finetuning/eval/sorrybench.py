"""SORRY-Bench (Xie et al., ICLR 2025): compliance with 450 unsafe instructions, judged by their
fine-tuned Mistral-7B.

The second refusal benchmark beside StrongREJECT, and a different measurement in three ways that
matter for the refusal-attribution cells: a **binary** verdict per response (comply / refuse)
where StrongREJECT's is a graded 0-1 score; a **balanced** taxonomy of 45 categories x 10
prompts, where StrongREJECT's 60 are weighted toward a few; and a judge **no run in this repo has
ever optimised against** -- the GRPO arms trained on the StrongREJECT judge, so this is the
held-out measurement of the same behaviour under an unrelated evaluator. GRP-Obliteration
(Russinovich et al. 2026) reports it as its primary attack-success number.

Per the project notes' rule for borrowed metrics, **none of the metric is implemented here** -- see
:mod:`sb_ref`. The prompt set, the judge prompt, the judge and the 0/1 parse are their code,
called unmodified. What lives here is the sparsity plumbing, on the same two-phase shape as
``strongreject``: :meth:`run` only generates, :meth:`finalize` loads the ~14.5 GB judge once per
eval point, scores every condition's responses, and frees it.

One split, ``off_target``: their 450 unsafe instructions. Same ``in_dist`` gap as StrongREJECT
and for the same reason (the judge is only defined on unsafe requests).

Metrics per split
-----------------
``fulfillment``  their compliance rate -- the mean of per-category means (as their
                 ``visualize_result.ipynb`` aggregates), which with 10 prompts per category is
                 the plain mean. THE headline; 1.0 = complied with everything.
``frac_comply``  the plain mean over every scored response, for the record.
``stderr``       binomial standard error of ``frac_comply``.
``invalid_frac`` fraction of responses their parser could not read a 0/1 from. Excluded from the
                 rates; a non-trivial value means the judge broke, not the model.
``empty_frac``   fraction of empty responses. Same trap as StrongREJECT's: an empty response is
                 a refusal to this judge, so a falling ``fulfillment`` with a rising
                 ``empty_frac`` is damage, not safety.

Every eval point appends to ``sorrybench_eval/generations.jsonl`` with the verdict and the raw
judge text attached.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from . import sb_ref
from .base import OFF_TARGET, Probe

logger = logging.getLogger(__name__)


@dataclass
class SorryBenchEvalCfg:
    """Config for :class:`SorryBenchEval`."""

    #: the gated Hub dataset (their 450-prompt 2024-06 release) or a local ``question.jsonl``
    dataset: str = sb_ref.DATASET
    n_prompts: int = None                   # None -> all 450 (a prefix, so categories thin evenly)
    max_new_tokens: int = 512
    batch_size: int = 16                    # generation batching
    temperature: float = 0.0                # greedy, so a moving curve is the model
    judge: bool = True                      # False -> generate only, write no scores
    judge_model: str = sb_ref.JUDGE_MODEL
    judge_batch_size: int = 8
    #: their judge answers "0" or "1"; their parser reads the leading digits, so a handful of
    #: tokens is the whole verdict (their driver decodes 1024 and parses the same prefix)
    judge_max_new_tokens: int = 16
    free_judge: bool = True                 # release the 14.5 GB after each eval point
    sb_repo: str = None                     # None -> deps/sorry-bench (see sb_ref)


def _metrics(rows) -> dict:
    """``rows``: ``[(question, response, score)]`` for one condition and split."""
    n = max(1, len(rows))
    valid = [(q, s) for q, _, s in rows if s is not None]
    per_cat = defaultdict(list)
    for q, s in valid:
        per_cat[str(q.get("category"))].append(float(s))
    cat_means = [sum(v) / len(v) for v in per_cat.values()]
    frac = sum(s for _, s in valid) / max(1, len(valid))
    return {
        "fulfillment": sum(cat_means) / max(1, len(cat_means)),
        "frac_comply": frac,
        "stderr": (frac * (1 - frac) / max(1, len(valid))) ** 0.5,
        "invalid_frac": (len(rows) - len(valid)) / n,
        "empty_frac": sum(not (r or "").strip() for _, r, _ in rows) / n,
        "n": len(rows),
        "n_categories": len(per_cat),
    }


class SorryBenchEval:
    """SORRY-Bench fulfillment rate across sparsities, judged by their fine-tuned evaluator."""

    name = "sorrybench"
    needs_real_weights = True               # it generates
    Config = SorryBenchEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        # their code first (a missing checkout is the cheapest failure), then the gated assets
        sb_ref.load_judge_module(cfg.sb_repo)
        sb_ref.check_access(cfg.dataset, cfg.judge_model, judge=cfg.judge)
        questions = sb_ref.load_questions(cfg.dataset)
        if cfg.n_prompts:
            questions = questions[:cfg.n_prompts]
        if not cfg.judge:
            logger.warning("eval.sorrybench.judge is false: responses will be generated and "
                           "recorded, but nothing is scored and no metrics are reported")
        logger.info("SORRY-Bench probe: %d prompts over %d categories, judge %s", len(questions),
                    len({str(q.get("category")) for q in questions}),
                    cfg.judge_model if cfg.judge else "DISABLED")
        return Probe(splits={OFF_TARGET: [q["turns"][0] for q in questions]},
                     extra={"cfg": cfg, "questions": questions, "pending": {}, "records": []})

    def run(self, ctx, probe: Probe):
        """Generate under this condition's weights. Returns None -- see :meth:`finalize`."""
        cfg = probe.extra["cfg"]
        responses = ctx.generate(probe.splits[OFF_TARGET], max_new_tokens=cfg.max_new_tokens,
                                 batch_size=cfg.batch_size, temperature=cfg.temperature)
        probe.extra["pending"][ctx.label] = responses
        return None

    def finalize(self, probe: Probe) -> dict:
        """Judge every condition generated since the last call, with one judge load."""
        cfg = probe.extra["cfg"]
        questions = probe.extra["questions"]
        pending, probe.extra["pending"] = probe.extra["pending"], {}
        if not pending:
            return {}
        labels = list(pending)
        flat_q = [q for label in labels for q in questions]
        flat_r = [r for label in labels for r in pending[label]]
        if cfg.judge:
            logger.info("judging %d responses over %d condition(s) with %s", len(flat_r),
                        len(labels), cfg.judge_model)
            scores, texts = sb_ref.score(flat_q, flat_r, judge_model=cfg.judge_model,
                                         batch_size=cfg.judge_batch_size,
                                         max_new_tokens=cfg.judge_max_new_tokens,
                                         free=cfg.free_judge, path=cfg.sb_repo)
        else:
            scores, texts = [None] * len(flat_r), [None] * len(flat_r)
        out, i = {}, 0
        for label in labels:
            rows = []
            for q, r in zip(questions, pending[label]):
                rows.append((q, r, scores[i]))
                probe.extra["records"].append(dict(
                    condition=label, split=OFF_TARGET, question_id=q.get("question_id"),
                    category=q.get("category"), prompt=q["turns"][0], response=r,
                    score=scores[i], judgment=texts[i]))
                i += 1
            if cfg.judge:
                out[label] = {OFF_TARGET: _metrics(rows)}
        return out

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs
