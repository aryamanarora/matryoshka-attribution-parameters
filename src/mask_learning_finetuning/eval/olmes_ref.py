"""Import shim for AI2's OLMES (`allenai/olmes`), the evaluation system behind the Olmo model cards.

Same rule as :mod:`em_ref`, :mod:`sr_ref` and :mod:`ifeval_ref`: **none of the metric is implemented
here**. A task's prompt (their template, few-shot source, chat message construction), its answer
extraction (their regexes and Minerva/Hendrycks normalisers), its metric (their ``Metric`` classes,
including the code executor and pass@k) and its aggregation are their code, called unmodified on
their ``Task`` objects built from their ``TASK_CONFIGS`` entries (e.g. ``aime:2024::olmo3:adapt``).
What lives here is locating the checkout, making the pieces importable inside this environment,
and the two adapters the runner needs: chat messages -> the user prompt our decoders render, and
our generations -> the ``model_resps`` dicts their collation expects.

Three things about the environment, all found the hard way:

* **OLMES is not installable beside this repo** (it pins torch 2.8, transformers <5, vllm 0.11,
  lm_eval 0.4.3), so it is a sibling checkout under ``deps/olmes`` -- ``scripts/setup.sh`` clones
  it -- and what is imported from it is the task/metric layer, never a model class. Their own
  venv (``deps/olmes/.venv``, ``uv sync --group gpu`` there) exists for the bit-faithful CLI route
  (``scripts/olmo3_post/olmes_cli_eval.py``), which is the only way to reproduce a model-card number
  including THEIR vLLM version.
* **Their task package's ``__init__`` imports every task module**, several of which need
  packages this repo does not (``alpaca_eval``, spaCy models, ...). :func:`add_to_path` registers
  a stub package with the same ``__path__`` so the individual task modules import on their own;
  :func:`task_class` reproduces the registry entries for the families used here.
* **The pure-python deps their task layer does need** are declared in this repo's ``olmes``
  dependency group (``lm_eval==0.4.3``, ``boto3``, ``httpx``, ``cloudpickle``, ``tree-sitter``,
  ``tree-sitter-python``) -- ``uv sync --group olmes``.
"""

import copy
import logging
import os
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent.parent
_CANDIDATE_ROOTS = (_REPO_ROOT / "deps" / "olmes", _REPO_ROOT.parent / "olmes")
TASK_PKG = "oe_eval.tasks.oe_eval_tasks"
_ROOT = None


def repo_root(path=None) -> Path:
    explicit = path or os.environ.get("OLMES_REPO")
    roots = [Path(explicit).expanduser()] if explicit else list(_CANDIDATE_ROOTS)
    for r in roots:
        if (r / "oe_eval" / "run_eval.py").is_file():
            return r.resolve()
    raise SystemExit(
        "no OLMES checkout found (looked in %s). `git clone https://github.com/allenai/olmes.git "
        "deps/olmes` (scripts/setup.sh does it), or point --olmes-repo / $OLMES_REPO at one; then "
        "`uv sync --group olmes` for its task layer's pure-python deps." % [str(r) for r in roots])


def add_to_path(path=None) -> Path:
    """Make ``oe_eval`` importable with the task package stubbed. Idempotent."""
    global _ROOT
    root = repo_root(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if TASK_PKG not in sys.modules:
        import oe_eval.tasks  # noqa: F401  -- the real parent package
        stub = types.ModuleType(TASK_PKG)
        stub.__path__ = [str(root / "oe_eval" / "tasks" / "oe_eval_tasks")]
        stub.__package__ = TASK_PKG
        sys.modules[TASK_PKG] = stub
    if _ROOT is None:
        logger.info("OLMES task layer from %s", root)
    _ROOT = root
    return root


def task_configs() -> dict:
    add_to_path()
    from oe_eval.configs.tasks import TASK_CONFIGS
    return TASK_CONFIGS


def suite_configs() -> dict:
    add_to_path()
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    return TASK_SUITE_CONFIGS


def task_class(task_name: str):
    """Their registry entry for ``task_name``, for the families this repo evaluates."""
    add_to_path()
    import importlib
    mod = lambda m: importlib.import_module(f"{TASK_PKG}.{m}")
    if task_name == "aime":
        return mod("aime").AIME
    if task_name.startswith("minerva_math_"):
        return mod("minerva_math").create_math_task(task_name[len("minerva_math_"):])
    if task_name.startswith("mmlu_") and task_name.endswith(":cot"):
        return mod("mmlu").create_mmlu_cot_task(task_name[len("mmlu_"):-len(":cot")])
    if task_name == "ifeval":
        return mod("ifeval").IFEval
    if task_name == "ifbench":
        return mod("ifbench").IFBench
    if task_name == "codex_humanevalplus":
        return mod("codex_humaneval").CodexHumanEvalPlus
    if task_name == "mbppplus":
        return mod("codex_mbpp").MBPPPlus
    if task_name == "gsm8k":
        return mod("gsm8k").GSM8K
    raise KeyError(f"task {task_name!r} is not wired through olmes_ref.task_class; add its "
                   "registry entry (see deps/olmes/oe_eval/tasks/oe_eval_tasks/__init__.py)")


def expand_spec(spec: str) -> list:
    """A suite name (``minerva_math::olmo3:adapt``) -> its task specs; a task spec -> ``[spec]``."""
    suites = suite_configs()
    if spec in suites:
        return list(suites[spec]["tasks"])
    if spec not in task_configs():
        raise KeyError(f"{spec!r} is neither an OLMES task config nor a suite")
    return [spec]


def make_task(spec: str, overrides: dict = None):
    """Their ``Task`` object for one task spec, with optional deep overrides on its config."""
    from oe_eval.utils import update_nested_dict
    cfg = copy.deepcopy(task_configs()[spec])
    if overrides:
        cfg = update_nested_dict(cfg, copy.deepcopy(overrides))
    task = task_class(cfg["task_name"])(task_name=cfg["task_name"], task_config=cfg)
    task.download()
    return task


def build_instances(task, limit=None) -> list:
    task.build_all_requests(limit=limit)
    return list(task._instances)


def user_prompt(ins) -> str:
    """The single user message of a chat-format instance -- what ``ModelCtx.generate`` renders.

    Their driver applies the tokenizer's chat template to ``context["messages"]`` with
    ``add_generation_prompt=True`` (run_eval.convert_chat_instance); our decoders do exactly that
    to ``[{"role": "user", "content": prompt}]``, so for a single-user-turn instance the rendered
    prompt is byte-identical. Anything else (system prompt, multi-turn few-shot, assistant
    prefix) cannot be expressed through the shared generation cache and is refused rather than
    approximated.
    """
    ctx = ins.request.context
    if isinstance(ctx, str):
        raise SystemExit("OLMES instance is not chat-format (use_chat_format: false); this eval "
                         "only serves chat-format task specs")
    msgs, prefix = ctx["messages"], ctx.get("assistant_prefix") or ""
    if prefix or len(msgs) != 1 or msgs[0]["role"] != "user":
        raise SystemExit(f"OLMES instance has a chat context this eval cannot render through the "
                         f"shared generator (roles {[m['role'] for m in msgs]}, assistant_prefix "
                         f"{prefix!r}); only one user turn is supported")
    return msgs[0]["content"]


def gen_kwargs(task) -> dict:
    return dict(task.task_config["generation_kwargs"])


def model_resps(texts, stop_sequences, tokenizer=None) -> list:
    """Our generations -> their per-request ``model_resps`` dicts (eleuther_vllm_causallms)."""
    from oe_eval.utils import cut_at_stop_sequence
    out = []
    for t in texts:
        cut = cut_at_stop_sequence(t, list(stop_sequences or []))
        r = {"continuation": cut,
             "num_tokens": len(tokenizer(cut, add_special_tokens=False)["input_ids"]) if tokenizer else 0,
             "context_tokens": 0}
        if cut != t:
            r["continuation_raw"] = t
        out.append(r)
    return out


def _fork_free_multiprocessing():
    """Their code executor starts a ``multiprocessing.Process`` per sample from a thread pool, and
    the ``filelock`` in this environment aborts ``os.fork`` while another thread holds a lock
    ("os.fork is unsafe while filelock is changing descriptor ownership" -- hit on the first
    HumanEval+ scoring pass). ``forkserver`` starts children from a separate server process, so
    the audit hook in this process never sees a fork; their target is a module-level function
    with picklable arguments, so nothing else changes."""
    import multiprocessing as mp
    try:
        if mp.get_start_method(allow_none=True) != "forkserver":
            mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass


def score(task, instances, texts, tokenizer=None):
    """``(task_scores, per_doc)`` -- their metrics over our generations, one text per instance."""
    from oe_eval.utilities.model_results_collation import collate_results
    _fork_free_multiprocessing()
    gk = gen_kwargs(task)
    results = collate_results(instances, model_resps(texts, gk.get("stop_sequences"), tokenizer))
    agg, per_doc = {}, []
    for m in task.make_metrics():
        m.compute_for_docs(results)
        agg.update(m.aggregate_to_task(task.task_config.get("primary_metric")))
        per_doc.append(copy.deepcopy(m._scores_for_docs))
    return agg, per_doc
