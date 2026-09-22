"""Where runs live, so no config or script has to spell out an absolute path.

Every path a config carries is repo-relative or an HF hub id, and the run-directory ones all
start with ``runs/`` (``output: runs/<name>``, ``mask.finetuned: runs/<name>/adapter``,
``restrict.checkpoint: runs/<name>``). This module is the one place that prefix is turned into a
real directory:

* by default ``runs/`` is ``<repo>/runs`` -- the checkout's own directory, whatever the current
  working directory is;
* ``MLFT_RUNS_ROOT=/some/volume`` moves every run there without touching a config, which is how
  a cluster keeps its artifacts on a data volume while the configs stay portable.

``run_path`` is the function to call on any config value that MAY name a run: a hub id
(``meta-llama/Llama-3.1-8B``), an absolute path and a plain relative path all pass through
unchanged, only the ``runs/`` prefix is rebased. Plot scripts use ``runs_root()`` directly.
"""

import os
from pathlib import Path

#: The checkout this package was imported from (``src/mask_learning_finetuning/paths.py`` ->
#: three levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]

RUNS_ENV = "MLFT_RUNS_ROOT"
_PREFIX = "runs/"


def runs_root() -> Path:
    """The directory ``runs/`` in a config refers to."""
    override = os.environ.get(RUNS_ENV)
    return Path(override).expanduser() if override else REPO_ROOT / "runs"


def run_dir(name: str) -> Path:
    """``runs_root()/<name>``."""
    return runs_root() / name


def run_path(spec) -> Path:
    """Rebase a config path onto :func:`runs_root` if it starts with ``runs/``; else as given.

    Returns a :class:`~pathlib.Path` either way, so callers can use it in place of ``Path(spec)``.
    A hub id such as ``allenai/Olmo-3-7B-Instruct`` comes back as ``Path("allenai/Olmo-3-...")``,
    which is what every ``from_pretrained`` in the repo already receives.
    """
    s = str(spec)
    if s == _PREFIX.rstrip("/"):
        return runs_root()
    if s.startswith(_PREFIX):
        return runs_root() / s[len(_PREFIX):]
    return Path(s)


__all__ = ["REPO_ROOT", "RUNS_ENV", "run_dir", "run_path", "runs_root"]
