"""Import shim for the reference EM eval in the sibling `model-organisms-for-EM` checkout.

There is no reimplementation here and there should never be one: the EM numbers this repo
reports have to be the same numbers that repo reports, so the question set, the sampling
parameters, the judge prompts, the 0-100 logprob judge and the misaligned-and-coherent
metric are all *their* code, called unmodified. This module only makes it importable and
names the two things that must be configured from outside.

Why a shim and not a dependency. `model-organisms-for-EM` is a separate uv project whose
install pulls unsloth and vllm; adding it to `[tool.uv.sources]` would drag those in (and
fail outright on macOS). Instead the checkout is put on `sys.path` and the eval modules are
imported directly -- they only need peft, transformer_lens, openai, python-dotenv,
matplotlib and seaborn, all of which are declared in this repo's `pyproject.toml` for
exactly this reason.

Three consequences of "unmodified" worth knowing before you run it:

* **The judge is Azure-only, and its client is built at import time.** Their `judge_azure`
  constructs a module-level `AzureOpenAI` from
  `global_variables.AZURE_{ENDPOINT,DEPLOYMENT,API_VERSION}` and `$AZURE_OPENAI_API_KEY`,
  with the endpoint hardcoded to their private Sweden resource, and `gen_eval_util` imports
  it transitively -- so even pure *generation* cannot import their code without a key
  present. :func:`add_to_path` parks a placeholder key in the environment when there is none,
  purely so that import succeeds on a GPU box with no judge credentials; a real key is still
  required (and checked) before anything is judged.
* **...but the judge is only Azure in its client.** `backend="openai"` swaps
  `judge_azure.client` for a plain `openai.OpenAI` and `judge_azure.AZURE_DEPLOYMENT` for
  `gpt-4o-2024-08-06` -- the exact snapshot their questions YAML pins in its `judge:` field.
  Both names are read at call time inside their `logprob_probs`, so their request (one token,
  `temperature=0`, `top_logprobs=20`, `seed=0`) and their `_aggregate_0_100_score` run
  untouched against the same model. Same judge, different door. What is *not* supported is a
  local judge: the score is a probability-weighted mean over single-token integers, and
  open-weight tokenizers (Llama-3, Qwen2) split numbers digit-by-digit, so the trick would
  have to be replaced rather than repointed.
* **Their `get_responses` re-tokenizes an already-rendered chat template with
  `add_special_tokens` at its default**, so Llama-family prompts carry two BOS tokens. That
  is what their published numbers were produced with, so it stays.
"""

import os
import sys
from pathlib import Path

# Sibling of this repo, per the same layout convention as ../learning-to-attribute.
#
# Derived from the PACKAGE root rather than by counting parents of this file: the old form
# hardcoded this module's depth (`parents[3]` when it lived at src/mask_learning_finetuning/),
# so moving it one directory deeper into eval/ would silently have repointed the default at
# the wrong place -- a wrong-directory error at best, and at worst a stale checkout that still
# imports. `_PKG_ROOT/..` is src/, and one more is the repo, whose parent holds the siblings.
# This repo is always installed editable, so __file__ is the source tree, not site-packages.
_PKG_ROOT = Path(__file__).resolve().parent.parent          # src/mask_learning_finetuning/
_REPO_ROOT = _PKG_ROOT.parent.parent                        # the repo checkout
DEFAULT_EM_REPO = _REPO_ROOT.parent / "model-organisms-for-EM"

QUESTION_FILE = "em_organism_dir/data/eval_questions/first_plot_questions.yaml"

# stands in for AZURE_OPENAI_API_KEY so that importing their eval package works without judge
# credentials; have_azure_key() treats it as absent, so it can never be used to judge
_PLACEHOLDER_KEY = "unset-generation-only"

JUDGE_BACKENDS = ("azure", "openai")

# the snapshot their questions YAML pins as `judge:`. Their own code ignores it and calls
# whatever AZURE_DEPLOYMENT names; on the public API we can ask for it by name.
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4o-2024-08-06"


def em_repo_path(path=None) -> Path:
    """Locate the reference checkout: explicit arg, then $EM_REPO, then the sibling dir."""
    root = Path(path or os.environ.get("EM_REPO") or DEFAULT_EM_REPO).expanduser()
    if not (root / "em_organism_dir").is_dir():
        raise SystemExit(
            f"no model-organisms-for-EM checkout at {root}. Clone it next to this repo, or "
            "pass --em-repo / set $EM_REPO. The EM eval is run from their code, not "
            "reimplemented here."
        )
    return root.resolve()


def load_env():
    """Read a local `.env`, as their `judge_azure` / `eval_judge` do at import time.

    Called before any credential check so that a key living in `.env` counts as present --
    otherwise we would reject a setup their own code would have accepted.
    """
    from dotenv import load_dotenv
    load_dotenv()


def have_azure_key() -> bool:
    """True if a real (non-placeholder) Azure key is available to the reference judge."""
    return os.environ.get("AZURE_OPENAI_API_KEY", _PLACEHOLDER_KEY) != _PLACEHOLDER_KEY


def add_to_path(path=None) -> Path:
    """Put the reference checkout on ``sys.path``. Idempotent; returns its root."""
    root = em_repo_path(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # their judge_azure instantiates AzureOpenAI at module scope and gen_eval_util imports it,
    # so without this the generation-only path cannot even import their code
    os.environ.setdefault("AZURE_OPENAI_API_KEY", _PLACEHOLDER_KEY)
    return root


def question_file(path=None) -> str:
    """Their `first_plot_questions.yaml` -- the question set *and* the judge prompts."""
    return str(em_repo_path(path) / QUESTION_FILE)


def configure_judge(*, backend="azure", model=None, endpoint=None, deployment=None,
                    api_version=None, path=None):
    """Point their judge at a deployment you can actually reach. Returns `judge_azure`.

    Must be called before anything imports `em_organism_dir.eval.util.eval_judge` (which
    `gen_eval_util` does transitively), because `judge_azure` builds its client at module
    scope and binds the Azure globals by value at that moment.

    ``backend="azure"`` is their path verbatim: only the
    `AZURE_{ENDPOINT,DEPLOYMENT,API_VERSION}` globals are overridden, from the arguments or
    the matching `AZURE_OPENAI_*` environment variables, so you can use your own resource
    instead of the Sweden one hardcoded in their `global_variables.py`.

    ``backend="openai"`` additionally replaces the constructed client with a plain
    `openai.OpenAI` and `AZURE_DEPLOYMENT` with ``model``. Both are module globals that their
    `OpenAiJudge.logprob_probs` reads at *call* time, so the request it builds -- one
    completion token, `temperature=0`, `logprobs=True`, `top_logprobs=20`, `seed=0` -- and
    `_aggregate_0_100_score` behind it are the same code against the same gpt-4o snapshot.
    The Azure client is still constructed on import (with the placeholder key, if that is all
    there is) and then discarded; nothing is sent to it.
    """
    if backend not in JUDGE_BACKENDS:
        raise ValueError(f"unknown judge backend {backend!r}; known: {JUDGE_BACKENDS}")
    add_to_path(path)
    load_env()
    from em_organism_dir import global_variables as gv

    # AZURE_DEPLOYMENT and `client` are read at call time inside their logprob_probs, so they
    # can still be set after the fact; the endpoint and api_version are baked into the
    # AzureOpenAI instance the module built on import, so those genuinely come too late.
    already = "em_organism_dir.eval.util.judge_azure" in sys.modules
    if already and backend == "azure" and (endpoint or api_version):
        raise RuntimeError(
            "judge_azure is already imported, so its Azure client is constructed and "
            "--azure-endpoint / api_version would be silently ignored -- call "
            "configure_judge() before load_gen_eval()."
        )

    endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
    if backend == "azure":
        deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model
    if endpoint:
        gv.AZURE_ENDPOINT = endpoint
    if deployment:
        gv.AZURE_DEPLOYMENT = deployment
    if api_version:
        gv.AZURE_API_VERSION = api_version

    if backend == "azure" and not have_azure_key():
        raise SystemExit(
            "AZURE_OPENAI_API_KEY is not set. Their judge "
            "(em_organism_dir/eval/util/judge_azure.py) is an Azure OpenAI deployment of "
            "gpt-4o; set the key, plus AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT if "
            f"you are not using theirs ({gv.AZURE_ENDPOINT} / {gv.AZURE_DEPLOYMENT}). Or "
            "pass --judge-backend openai to reach the same model through the public API. "
            "Generation needs no key at all: use --stage generate."
        )
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set (needed by --judge-backend openai). "
            "Generation needs no key at all: use --stage generate."
        )

    from em_organism_dir.eval.util import judge_azure
    if backend == "openai":
        from openai import OpenAI
        judge_azure.client = OpenAI()
        judge_azure.AZURE_DEPLOYMENT = model or DEFAULT_OPENAI_JUDGE_MODEL
    return judge_azure


def judge_description(backend: str) -> str:
    """What the judge will actually call, for the run log."""
    add_to_path()
    from em_organism_dir.eval.util import judge_azure
    if backend == "openai":
        return f"openai:{judge_azure.AZURE_DEPLOYMENT}"
    from em_organism_dir import global_variables as gv
    return f"azure:{gv.AZURE_DEPLOYMENT} @ {gv.AZURE_ENDPOINT}"


def load_gen_eval(path=None):
    """Their `eval.util.gen_eval_util` module (generation + judging orchestration)."""
    add_to_path(path)
    from em_organism_dir.eval.util import gen_eval_util
    return gen_eval_util


def load_stats(path=None):
    """Their `vis.quadrant_plots` module, which owns `get_basic_eval_stats` (the metric)."""
    add_to_path(path)
    from em_organism_dir.vis import quadrant_plots
    # `get_basic_eval_stats` ends with a bare `display(styled_df)` -- the IPython builtin,
    # which exists only inside a notebook, so from a script the function raises NameError
    # *after* computing its result and before returning it. Binding a no-op lets it return.
    # It is the last statement before `return df` and is purely cosmetic; every line that
    # computes the metric runs untouched.
    if not hasattr(quadrant_plots, "display"):
        quadrant_plots.display = lambda *a, **k: None
    return quadrant_plots
