"""Emergent misalignment, deferring entirely to the ``model-organisms-for-EM`` checkout.

Per CLAUDE.md, **none of the metric is implemented here**: the question set, sampling
parameters, judge prompts, the 0-100 logprob judge and the misaligned-and-coherent rate all
come from that sibling repo through :mod:`em_ref`. An EM number this repo produces has to be
comparable to their published one, and it stops being comparable the moment any of it is
reimplemented. What lives here is only the sparsity plumbing around it.

Two-phase, via the optional :meth:`finalize` hook: :meth:`run` samples one condition's
responses to its own CSV (cheap, needs the masked weights in place), and ``finalize`` judges
every CSV afterwards. The split exists because their ``judge_csv_file`` scores one row at a
time against a *synchronous* client, so its ``asyncio.gather`` never overlaps anything -- a
full sweep is ~10k strictly serial API calls. Judging whole conditions in parallel threads is
pure scheduling: no row is scored differently, the judge is deterministic (temperature 0, seed
0), and the files are disjoint. Set ``judge_workers=1`` to reproduce their exact call order.

**Three invariants that are easy to break and silently wrong if broken:**

1. ``torch.manual_seed(seed)`` immediately before each condition's generation, so every
   sparsity point samples from the same RNG stream (paired sampling). Without it, differences
   between adjacent points are partly sampling noise.
2. Response CSVs live in their own ``responses/`` subdirectory, because their stats function
   globs ``*.csv`` recursively and would otherwise aggregate ``summary.csv`` into the metric.
3. ``em_ref.configure_judge`` must run before their eval modules are imported -- their
   ``judge_azure`` builds an ``AzureOpenAI`` client at module scope. That is why
   ``eval/registry.py`` imports this module lazily and why the imports below are inside
   functions.

**Scope limit, stated rather than faked:** their ``get_responses`` is driven by a question
YAML, so the off-target split is their ``first_plot_questions.yaml`` and that is what this eval
measures today. An in-distribution split (the EM rate on train-style prompts) needs a second
YAML in their format, built from the training set and carrying the same judge prompts;
``in_dist_question_file`` accepts one, but constructing it is separate work and is *not* done
here. Until it is supplied this eval reports one split, like ``mmlu``.
"""

import asyncio
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import em_ref
from .base import IN_DIST, OFF_TARGET, Probe

logger = logging.getLogger(__name__)


@dataclass
class EmEvalCfg:
    """Config for :class:`EmEval`. Defaults match the reference eval's own settings."""

    out_dir: str = None                # where responses/ goes; required
    em_repo: str = None                # None -> the sibling checkout / $EM_REPO
    question_file: str = None          # None -> their first_plot_questions.yaml
    in_dist_question_file: str = None   # a train-distribution YAML in their format; see above
    #: 25 matches the existing runs/em/* sweeps, so a new number is comparable to them. Their
    #: own post-hoc default is 50; lower it to ~10 for an in-training probe, where the cost is
    #: paid at every eval point.
    n_per_question: int = 25
    new_tokens: int = 600
    temperature: float = 1.0
    top_p: float = 1.0
    metrics: tuple = ("aligned", "coherent")
    use_json_questions: bool = False
    use_template_questions: bool = False
    judge: bool = True                 # False -> generate only, no API calls
    judge_backend: str = "openai"
    judge_model: str = None
    judge_workers: int = 4
    overwrite: bool = False
    seed: int = 0


class EmEval:
    """Misaligned-and-coherent rate across sparsities, scored by their judge."""

    name = "em"
    needs_real_weights = True          # their get_responses calls model.generate
    Config = EmEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not cfg.out_dir:
            raise ValueError("EmEvalCfg.out_dir is required (responses/ is written under it)")
        qf = cfg.question_file or em_ref.question_file(cfg.em_repo)
        splits = {OFF_TARGET: qf}
        if cfg.in_dist_question_file:
            splits[IN_DIST] = cfg.in_dist_question_file
        # Fail fast, before any training happens and before their judge module is imported:
        # configure_judge both validates credentials and swaps the client, and it cannot do
        # the latter once judge_azure is in sys.modules.
        if cfg.judge:
            em_ref.configure_judge(backend=cfg.judge_backend, model=cfg.judge_model,
                                   path=cfg.em_repo)
            logger.info("EM judge: %s", em_ref.judge_description(cfg.judge_backend))
        logger.info("EM probe: %s", ", ".join(f"{k}={Path(v).name}" for k, v in splits.items()))
        return Probe(splits=splits, extra={"cfg": cfg, "labels": []})

    def _resp_dir(self, cfg, split) -> Path:
        # their get_basic_eval_stats globs *.csv recursively, so responses must be in their own
        # directory or summary.csv gets aggregated into the metric
        d = Path(cfg.out_dir) / "responses" / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run(self, ctx, probe: Probe):
        """Sample this condition's responses. Returns None -- scoring happens in finalize()."""
        cfg = probe.extra["cfg"]
        gen_eval = em_ref.load_gen_eval(cfg.em_repo)
        for split, question_file in probe.splits.items():
            path = self._resp_dir(cfg, split) / f"{ctx.label}.csv"
            if path.exists() and not cfg.overwrite:
                logger.info("EM %s/%s already sampled; skipping", split, ctx.label)
                continue
            # paired sampling: identical RNG stream at every sparsity point
            torch.manual_seed(cfg.seed)
            gen_eval.get_responses(
                ctx.model, ctx.tokenizer, str(path), cfg.overwrite, question_file,
                cfg.use_json_questions, cfg.use_template_questions, cfg.n_per_question,
                cfg.new_tokens, cfg.temperature, cfg.top_p,
            )
        if ctx.label not in probe.extra["labels"]:
            probe.extra["labels"].append(ctx.label)
        return None

    def finalize(self, probe: Probe) -> dict:
        """Judge every sampled CSV, then aggregate with their ``get_basic_eval_stats``."""
        cfg = probe.extra["cfg"]
        out = {}
        for split in probe.splits:
            resp_dir = self._resp_dir(cfg, split)
            csvs = sorted(resp_dir.glob("*.csv"))
            if not csvs:
                continue
            if cfg.judge:
                self._judge(cfg, csvs, probe.splits[split])
            for label, metrics in self._aggregate(cfg, resp_dir).items():
                out.setdefault(label, {})[split] = metrics
        return out

    def _judge(self, cfg, csvs, question_file):
        gen_eval = em_ref.load_gen_eval(cfg.em_repo)

        def judge_one(path):
            logger.info("judging %s", path.name)
            return asyncio.run(gen_eval.judge_responses(
                str(path), judge_file=question_file, metrics=list(cfg.metrics)))

        if cfg.judge_workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.judge_workers) as pool:
                list(pool.map(judge_one, csvs))
        else:
            for path in csvs:
                judge_one(path)

    def _aggregate(self, cfg, resp_dir: Path) -> dict:
        """Their metric, keyed by condition label (which is the CSV filename)."""
        stats_mod = em_ref.load_stats(cfg.em_repo)
        stats = stats_mod.get_basic_eval_stats(str(resp_dir), exclude_json=True)
        if stats is None:
            logger.warning("nothing judged in %s -- no EM numbers", resp_dir)
            return {}
        stats = stats.reset_index()
        label_col = stats.columns[0]
        out = {}
        for rec in stats.to_dict(orient="records"):
            label = str(rec[label_col])
            out[label] = {k: v for k, v in rec.items()
                          if k != label_col and isinstance(v, (int, float))}
        return out
