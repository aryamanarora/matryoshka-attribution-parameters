"""Import shim for Google's IFEval checker (`google-research/instruction_following_eval`).

Same rule as :mod:`sr_ref`: **none of the metric is implemented here.** The 541
prompts, the 25 instruction checkers, the strict/loose evaluation and the four accuracies are their
code (Zhou et al. 2023), called unmodified. What lives here is locating the checkout, putting it on
``sys.path`` and making sure the one piece of data their checker loads at first use is present.

Why a sparse checkout rather than a dependency
----------------------------------------------
``instruction_following_eval`` is one directory of the google-research monorepo and is not on
PyPI. ``scripts/setup.sh`` clones it with ``--filter=blob:none --sparse`` and checks out only that
directory (~5 MB), into ``deps/google-research``. The directory has no ``__init__.py``, so it is a
namespace package: with the checkout's root on ``sys.path``, their own
``from instruction_following_eval import instructions_registry`` resolves as written.

Two things their code needs from the environment:

* **Four pure-python packages** -- ``absl-py``, ``immutabledict``, ``nltk`` and ``langdetect`` --
  declared in ``pyproject.toml``. No model download.
* **NLTK's punkt sentence tokenizer**, which their ``count_sentences`` loads on first use
  (``nltk.data.load("nltk:tokenizers/punkt/english.pickle")``). Not a package but a data file, so
  :func:`ensure_nltk_data` fetches it once into ``~/nltk_data`` if it is missing -- at *build*
  time, so a job on a node without network dies before generating rather than after.

Their checker is a pure function of (prompt row, response text), CPU-only and instant, which is why
:mod:`ifeval` is single-phase where the judged evals are two-phase.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent          # src/mask_learning_finetuning/
_REPO_ROOT = _PKG_ROOT.parent.parent                        # the repo checkout
#: the checkout root (what goes on sys.path) is the google-research clone; the package is the
#: directory inside it
_CANDIDATE_ROOTS = (_REPO_ROOT / "deps" / "google-research",
                    _REPO_ROOT.parent / "google-research")
PACKAGE = "instruction_following_eval"
DEFAULT_ROOT = next((p for p in _CANDIDATE_ROOTS if (p / PACKAGE / "evaluation_lib.py").is_file()),
                    _CANDIDATE_ROOTS[0])

#: the file their ``evaluation_main.py`` reads by default: the 541 prompts, with each one's
#: instruction ids and checker kwargs. THE prompt set -- the same rows as ``google/IFEval`` on the
#: Hub, taken from their tree so the metric's inputs and its checkers come from one commit.
INPUT_DATA = "data/input_data.jsonl"

_ROOT = None


def repo_root(path=None) -> Path:
    """The google-research checkout root, or a clear error saying how to get one."""
    explicit = path or os.environ.get("IFEVAL_REPO")
    if explicit:
        root = Path(explicit).expanduser()
        if not (root / PACKAGE / "evaluation_lib.py").is_file():
            raise SystemExit(f"no {PACKAGE} checkout at {root} (expected "
                             f"{root}/{PACKAGE}/evaluation_lib.py)")
        return root.resolve()
    if (DEFAULT_ROOT / PACKAGE / "evaluation_lib.py").is_file():
        return DEFAULT_ROOT.resolve()
    raise SystemExit(
        f"the IFEval checker is not checked out at {DEFAULT_ROOT}. Run `bash scripts/setup.sh`, "
        "or by hand:\n"
        "    git clone --depth 1 --filter=blob:none --sparse "
        f"https://github.com/google-research/google-research.git {DEFAULT_ROOT}\n"
        f"    git -C {DEFAULT_ROOT} sparse-checkout set {PACKAGE}\n"
        "or point --ifeval-repo / $IFEVAL_REPO at a copy. The IFEval metric is run from their "
        "code, not reimplemented here.")


def add_to_path(path=None) -> Path:
    """Make ``instruction_following_eval`` importable. Idempotent; returns the checkout root."""
    global _ROOT
    root = repo_root(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if _ROOT is None:
        logger.info("%s from %s", PACKAGE, root)
    _ROOT = root
    return root


def ensure_nltk_data():
    """Fetch punkt once if their sentence counter would fail to load it.

    ``nltk.download`` is a no-op when the resource is present, and needs the network when it is
    not -- which is why this is called from ``build`` and not lazily from the first response that
    happens to carry a sentence-count instruction.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
        return
    except LookupError:
        pass
    logger.info("fetching NLTK punkt (their count_sentences loads it on first use)")
    if not nltk.download("punkt", quiet=True):
        raise SystemExit("could not download NLTK's punkt tokenizer, which the IFEval checker "
                         "needs for its sentence-count instructions; run "
                         "`python -c \"import nltk; nltk.download('punkt')\"` with the network up")


def load_lib(path=None):
    """Their ``evaluation_lib`` (``InputExample``, the strict/loose checkers, ``read_prompt_list``)."""
    add_to_path(path)
    import instruction_following_eval.evaluation_lib as lib
    return lib


def input_data_path(path=None) -> Path:
    return add_to_path(path) / PACKAGE / INPUT_DATA


def read_inputs(data=None, path=None) -> list:
    """Their ``InputExample`` rows, from ``data`` (a jsonl in their format) or their own file."""
    lib = load_lib(path)
    src = Path(data) if data else input_data_path(path)
    if not src.is_file():
        raise SystemExit(f"no IFEval input file at {src}")
    return lib.read_prompt_list(str(src))


def score(inputs, responses, path=None):
    """``(strict, loose)`` -- their two ``OutputExample`` lists, one entry per input row.

    Both of their checkers take a ``{prompt: response}`` dict, so identical prompts (there are none
    in their 541) would collide; building the dict per row keeps the pairing positional.
    """
    lib = load_lib(path)
    strict, loose = [], []
    for inp, resp in zip(inputs, responses):
        d = {inp.prompt: resp or ""}
        strict.append(lib.test_instruction_following_strict(inp, d))
        loose.append(lib.test_instruction_following_loose(inp, d))
    return strict, loose
