"""Import shim for the StrongREJECT package (`dsbowen/strong_reject`).

Same rule as :mod:`em_ref`, for the same reason: **none of the metric is implemented here**.
The forbidden-prompt set, the judge prompt template, the fine-tuned judge and the 1-5 ->
expected-value aggregation are all *their* code, called unmodified, because a StrongREJECT
score this repo reports has to be the number their benchmark reports and it stops being that
the moment any of it is re-derived. What lives here is locating their package and naming the
two things that have to be arranged from outside (which judge, and whether it stays resident).

Why a sys.path shim rather than a dependency
--------------------------------------------
Their documented install is ``pip install git+https://github.com/dsbowen/strong_reject.git``,
and that would work -- but it pulls ``litellm``, ``google-cloud-translate`` and
``google-api-python-client``, none of which the fine-tuned evaluator touches, and adding a
dependency re-locks a project whose lockfile carries a hand-pinned ``torch==2.11`` override
(see ``pyproject.toml``). So the checkout goes on ``sys.path`` exactly as
``model-organisms-for-EM`` does, and this shim currently needs **no new dependency at all**:
their ``evaluate`` module imports only ``openai``, ``torch``, ``datasets``, ``transformers``
and ``peft``, all of which are already declared here.

``pip install strong_reject`` is still supported and takes precedence over a checkout -- if the
package imports, that is what runs.

Three consequences of "unmodified" worth knowing before you run it
-----------------------------------------------------------------
* **The judge is a gated model.** ``qylu4156/strongreject-15k-v1`` is a PEFT adapter over
  ``google/gemma-2b``, whose licence is manual-approval gated on the Hub, so the eval needs an
  ``HF_TOKEN`` (or a logged-in ``huggingface-cli``) belonging to an account that has accepted
  it. :func:`check_judge` says so up front rather than 40 minutes into a training run. Note
  this is Gemma-**1**, and it is loaded by plain ``transformers``: CLAUDE.md's "never evaluate
  Gemma-2 under transformer-lens 3.x" hazard is about neither the model nor the library here,
  and does not apply.

* **``strong_reject.evaluate`` imports their litellm-backed generation helper at module
  scope**, which the fine-tuned evaluator never calls (it is used by the API-judge evaluators).
  Their own module guards that import with ``if not os.getenv("READTHEDOCS")``, so
  :func:`add_to_path` sets ``READTHEDOCS`` when ``litellm`` is absent -- the one thing that
  keeps this dependency-free. Install ``litellm`` and the flag is not set, nothing is skipped.
  If they ever move that import out of the guard, this fails loudly with
  ``ModuleNotFoundError: litellm``, which is the right way for it to fail.

* **Their evaluator caches the judge in a module-level dict** (``evaluate.cached_models``) and
  ``evaluate_dataset(empty_model_cache=True)`` moves it to the CPU and drops it afterwards.
  Both behaviours are wanted, in different places: freeing it matters when the eval shares a
  GPU with a trainer; keeping it matters when GRPO scores a batch every step. Hence
  ``free`` on :func:`score`. :func:`preload_judge` writes that same cache, which is how
  ``scripts/verify/verify_strongreject.py`` exercises the whole path with a 14M-parameter stand-in
  judge and no gated download -- their scoring function is then called verbatim, on a different
  set of weights.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# `deps/strong_reject` first, then the old sibling location -- the same two-candidate rule as
# em_ref, and for the same reason. Derived from the package root rather than by counting parents of
# this file, so moving this module does not silently repoint the default (see em_ref).
_PKG_ROOT = Path(__file__).resolve().parent.parent          # src/mask_learning_finetuning/
_REPO_ROOT = _PKG_ROOT.parent.parent                        # the repo checkout
_CANDIDATE_SR_REPOS = (_REPO_ROOT / "deps" / "strong_reject",
                       _REPO_ROOT.parent / "strong_reject")
DEFAULT_SR_REPO = next((p for p in _CANDIDATE_SR_REPOS if (p / "strong_reject").is_dir()),
                       _CANDIDATE_SR_REPOS[0])

#: Their fine-tuned evaluator: a LoRA adapter over gemma-2b, trained on 15k judged responses.
JUDGE_MODEL = "qylu4156/strongreject-15k-v1"
JUDGE_BASE = "google/gemma-2b"

#: What ``from_pretrained`` actually needs from those two repos: config, weights, tokenizer. Used
#: both to prefetch and to check the prefetch, so the offline check cannot fail over a README --
#: which it did, when it asked for a *complete* snapshot instead.
JUDGE_FILE_PATTERNS = ("*.json", "*.safetensors", "*.model", "tokenizer*")

#: The evaluators that need nothing but a local GPU. The rest of their registry (
#: ``strongreject_rubric``, ``strongreject_aisi``, ``pair``, ...) reaches an API through their
#: litellm-backed ``generate``, which :func:`add_to_path` may not have imported -- and which is
#: a different measurement anyway. Rejected by name at config time rather than at first call.
LOCAL_EVALUATORS = ("strongreject_finetuned", "string_matching")

#: Their name for the prompt column, and ours for the response one, both fixed by their API.
PROMPT_COL, RESPONSE_COL = "forbidden_prompt", "response"

_ROOT = None
_PRELOADED = set()          # evaluators whose judge was injected by preload_judge
_PROMPT_CACHE = {}


def sr_repo_path(path=None):
    """Where their code will come from: ``(kind, location)``.

    ``("package", None)`` when ``strong_reject`` already imports (a pip install), otherwise
    ``("checkout", Path)``. An explicit ``path`` or ``$STRONG_REJECT_REPO`` wins over both, so a
    working copy can be tested without uninstalling anything.
    """
    explicit = path or os.environ.get("STRONG_REJECT_REPO")
    if explicit:
        root = Path(explicit).expanduser()
        if not (root / "strong_reject" / "evaluate.py").is_file():
            raise SystemExit(
                f"no strong_reject checkout at {root} (expected {root}/strong_reject/"
                "evaluate.py)")
        return "checkout", root.resolve()
    if importlib.util.find_spec("strong_reject") is not None:
        return "package", None
    root = DEFAULT_SR_REPO
    if (root / "strong_reject" / "evaluate.py").is_file():
        return "checkout", root.resolve()
    raise SystemExit(
        "strong_reject is neither installed nor checked out next to this repo. Either\n"
        "    git clone https://github.com/dsbowen/strong_reject.git "
        f"{DEFAULT_SR_REPO}\n"
        "or `uv pip install git+https://github.com/dsbowen/strong_reject.git`, or point\n"
        "--sr-repo / $STRONG_REJECT_REPO at a copy. The StrongREJECT score is run from their\n"
        "code, not reimplemented here.")


def add_to_path(path=None):
    """Make ``strong_reject`` importable. Idempotent; returns the location it resolved to."""
    global _ROOT
    kind, root = sr_repo_path(path)
    if kind == "checkout" and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if importlib.util.find_spec("litellm") is None:
        # their `evaluate` module does `if not os.getenv("READTHEDOCS"): from .generate import
        # generate`, and `generate` imports litellm. The fine-tuned evaluator never calls it.
        os.environ.setdefault("READTHEDOCS", "1")
    if _ROOT is None:
        logger.info("strong_reject from %s", root if kind == "checkout" else "the installed package")
    _ROOT = root or "package"
    return _ROOT


def load_evaluate(path=None):
    """Their ``strong_reject.evaluate`` module (the registry and ``evaluate_dataset``)."""
    add_to_path(path)
    import strong_reject.evaluate as ev
    return ev


def load_datasets_mod(path=None):
    """Their ``strong_reject.load_datasets`` module (the forbidden-prompt sets)."""
    add_to_path(path)
    import strong_reject.load_datasets as ld
    return ld


#: The named prompt sets, mapped to the loader that fetches each. Their loaders read the CSVs
#: from the ``alexandrasouly/strongreject`` repo over HTTP, so the first call needs network;
#: ``datasets`` caches them afterwards. ``StrongRejectEvalCfg.dataset`` also accepts a path, for
#: an offline box or a custom probe.
PROMPT_SETS = {
    "small": "load_strongreject_small",      # 60 prompts, their small benchmark
    "full": "load_strongreject",             # all 313
    "wmdp": "load_wmdp_open_ended",          # open-ended WMDP chem/bio, via their HF dataset
}


def load_prompt_set(name, path=None) -> list:
    """Forbidden prompts, from one of :data:`PROMPT_SETS` or a local file.

    A file may be ``.jsonl``/``.txt`` (read by ``eval/base.py``'s loader) or a ``.csv`` with
    their ``forbidden_prompt`` column -- which is what their own datasets are, so a downloaded
    copy of one works unchanged.

    Memoised per (name, path): the eval's probe, the GRPO reward split and the disjointness check
    all ask for the same sets, and their named loaders go over HTTP.
    """
    key = (name, str(path))
    if key in _PROMPT_CACHE:
        return list(_PROMPT_CACHE[key])
    _PROMPT_CACHE[key] = out = _load_prompt_set(name, path)
    return list(out)


def _load_prompt_set(name, path=None) -> list:
    if name in PROMPT_SETS:
        ld = load_datasets_mod(path)
        return list(getattr(ld, PROMPT_SETS[name])()[PROMPT_COL])
    p = Path(name)
    if not p.exists():
        raise SystemExit(f"strongreject prompt set {name!r} is neither a named set "
                         f"({', '.join(PROMPT_SETS)}) nor an existing file")
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        if PROMPT_COL not in df.columns:
            raise SystemExit(f"{p} has no {PROMPT_COL!r} column (columns: {list(df.columns)})")
        return [str(x) for x in df[PROMPT_COL]]
    from .base import load_prompts
    return load_prompts(p)


def have_hf_token() -> bool:
    """True if the Hub will see credentials -- env var or a ``huggingface-cli login`` store."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return False


def check_judge(evaluator=None, path=None):
    """Fail now, not after training, if the named judge cannot possibly load.

    Only the fine-tuned evaluator has anything to check, and what it checks is the one thing
    that silently is not there on a fresh box: ``google/gemma-2b`` is manual-approval gated, so
    a machine without an accepting token gets a 401 on the first judged response -- which is
    after every condition has already been generated.
    """
    ev = load_evaluate(path)
    evaluator = evaluator or LOCAL_EVALUATORS[0]
    if evaluator not in ev.registered_evaluators:
        raise SystemExit(f"strong_reject has no evaluator {evaluator!r}; registered: "
                         f"{sorted(ev.registered_evaluators)}")
    if evaluator not in LOCAL_EVALUATORS:
        raise SystemExit(
            f"evaluator {evaluator!r} is not one of the local ones {LOCAL_EVALUATORS}. The rest "
            "of their registry judges through an API via their litellm-backed `generate` "
            "(install litellm and set the provider's key if that is what you want) and is a "
            "different measurement from the fine-tuned evaluator.")
    if evaluator != "strongreject_finetuned":
        return
    if os.environ.get("TESTING"):
        logger.warning(
            "$TESTING is set, so their evaluator will load %s instead of %s -- the scores will "
            "be meaningless. Unset it unless you are running scripts/verify/verify_strongreject.py.",
            "EleutherAI/pythia-14m", JUDGE_MODEL)
    if "strongreject_finetuned" in ev.cached_models:
        logger.info("judge already loaded (preloaded stand-in or a previous condition)")
        return
    # A WARM CACHE IS NOT ENOUGH: this judge cannot load under HF_HUB_OFFLINE=1 at all, which is
    # the cluster default (scripts/cluster/sbatch_train.sbatch). Their loader calls
    # `AutoTokenizer.from_pretrained(JUDGE_MODEL)`, transformers resolves a config first, and the
    # adapter repo HAS no config.json -- online that 404 is tolerated and ignored, offline it
    # becomes `OSError: couldn't connect ... and couldn't find them in the cached files`. Verified
    # the expensive way: job 1263871 generated every response, loaded the 5 GB judge, and died on
    # that line. So this refuses up front instead of passing and failing an eval point later.
    from huggingface_hub import constants
    if constants.HF_HUB_OFFLINE:
        raise SystemExit(
            "HF_HUB_OFFLINE=1, and the StrongREJECT judge cannot be loaded offline even with a "
            f"fully warm cache: their loader asks transformers for a config on {JUDGE_MODEL}, "
            "which is an adapter-only repo with no config.json, and offline mode turns that "
            "tolerated 404 into an OSError.\n"
            "Run with the hub reachable -- the weights still come from the cache, only the 404 "
            "needs a round trip:\n"
            "    sbatch --export=ALL,HF_HUB_OFFLINE=0 scripts/cluster/sbatch_train.sbatch <config>")
    if not have_hf_token():
        raise SystemExit(
            f"no Hugging Face token, and the StrongREJECT judge ({JUDGE_MODEL}) is a PEFT "
            f"adapter over {JUDGE_BASE}, whose licence is manual-approval gated. Accept it at "
            f"https://huggingface.co/{JUDGE_BASE} with an account, then export HF_TOKEN=... "
            "(or `huggingface-cli login`). Set `judge: false` in the eval block to generate "
            "responses now and score them later.")
    try:
        from huggingface_hub import auth_check
        auth_check(JUDGE_BASE)
    except ImportError:
        pass
    except Exception as e:
        raise SystemExit(
            f"the token present cannot read {JUDGE_BASE}, which the judge adapter "
            f"({JUDGE_MODEL}) is built on: {type(e).__name__}: {e}. Accept the licence at "
            f"https://huggingface.co/{JUDGE_BASE} with the account that owns this token.")


def _stand_in_active(evaluator) -> bool:
    """Is a preloaded stand-in still the thing that would score?

    Their ``empty_model_cache=True`` clears the cache from under us, after which the next call
    would load the real judge -- so "was preloaded once" is not the same claim as "is the judge
    now", and only the second one may be logged.
    """
    if evaluator not in _PRELOADED:
        return False
    if evaluator in load_evaluate().cached_models:
        return True
    _PRELOADED.discard(evaluator)
    return False


def judge_description(evaluator="strongreject_finetuned") -> str:
    """What will actually score the responses, for the run log.

    Reports the *stand-in* when one has been preloaded, rather than the model a real run would
    have used -- a log line naming the real judge over meaningless scores is the kind of record
    that gets believed later.
    """
    if _stand_in_active(evaluator):
        return f"{evaluator}:PRELOADED STAND-IN (not StrongREJECT numbers)"
    if evaluator != "strongreject_finetuned":
        return evaluator
    if os.environ.get("TESTING"):
        return f"{evaluator}:EleutherAI/pythia-14m (TESTING -- meaningless scores)"
    return f"{evaluator}:{JUDGE_MODEL} over {JUDGE_BASE}"


def preload_judge(model, tokenizer, evaluator="strongreject_finetuned", path=None):
    """Install ``(model, tokenizer)`` as the judge, bypassing the gated download.

    For ``scripts/verify/verify_strongreject.py``, which needs to exercise their scoring function --
    their prompt template, their 1-5 logits, their expected-value aggregation, all verbatim --
    without a 5 GB licence-gated model. Deliberately not reachable from a config file: a run
    that could name its own judge could report a number nothing produced.
    """
    ev = load_evaluate(path)
    ev.cached_models[evaluator] = (model, tokenizer)
    _PRELOADED.add(evaluator)
    logger.warning("preloaded a stand-in judge for %s -- scores are NOT StrongREJECT numbers",
                   evaluator)


def score(prompts, responses, *, evaluator="strongreject_finetuned", batch_size=8, free=True,
          path=None, **kwargs) -> list:
    """Their harmfulness scores in ``[0, 1]``, one per (prompt, response) pair.

    A single ``evaluate_dataset`` call, so the judge is loaded once however many conditions'
    generations are handed in at a time. ``free=True`` (their default) moves it off the GPU
    afterwards, which is what an eval sharing a device with a trainer wants; ``free=False``
    keeps it resident, which is what GRPO's per-step reward wants.
    """
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts vs {len(responses)} responses")
    if not prompts:
        return []
    ev = load_evaluate(path)
    from datasets import Dataset
    ds = Dataset.from_dict({PROMPT_COL: list(prompts), RESPONSE_COL: list(responses)})
    scored = ev.evaluate_dataset(ds, [evaluator], batch_size=batch_size,
                                 empty_model_cache=free, **kwargs)
    return [float(s) for s in scored["score"]]


def free_judge(evaluator="strongreject_finetuned", path=None):
    """Drop a judge kept resident by ``score(free=False)``."""
    ev = load_evaluate(path)
    _PRELOADED.discard(evaluator)
    if ev.cached_models.pop(evaluator, None) is not None:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("released the %s judge", evaluator)
