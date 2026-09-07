"""Import shim for SORRY-Bench (`sorry-bench/sorry-bench`, Xie et al., ICLR 2025).

Same rule as :mod:`em_ref` and :mod:`sr_ref`: **none of the metric is implemented here.** The
450 unsafe instructions, the judge prompt, the fine-tuned Mistral-7B judge and the 0/1 parse are
theirs, called unmodified. What lives here is locating their checkout, loading the two functions
of theirs that the judge path needs, and the two things that have to be arranged from outside
(fetching the gated assets, and running the judge through ``transformers`` instead of their vLLM
driver so it can share a card with the model under evaluation).

How their code is reached
-------------------------
Their judge lives in ``gen_judgment_safety_vllm.py``, a script rather than a package, whose two
relevant functions -- ``apply_sorry_bench_judge_template`` (the prompt, rendered from THEIR
``data/sorry_bench/judge_prompts.jsonl`` in the Mistral ``[INST]`` frame) and ``extract_content``
(the 0/1 parse) -- touch neither of the script's imports. The script's top-level ``from common
import ...`` needs FastChat, and ``from vllm import ...`` pulls in vllm. Neither is needed to
call those two functions, so :func:`load_judge_module` executes the file with both names bound to
empty stub modules, and the functions are then theirs byte-for-byte. Stubbing an import is a
different thing from editing the file: the checkout is untouched.

Two consequences of "unmodified" worth knowing before you run it
----------------------------------------------------------------
* **Both assets are gated on the Hub**, each behind a click-through licence:
  ``sorry-bench/sorry-bench-202406`` (the prompts; the 202503 revision has 440 and the
  GRP-Oblit paper uses the 450 of 202406) and
  ``sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406`` (the judge, a full 7B model,
  ~14.5 GB bf16). :func:`check_access` fails at build time with both URLs rather than after every
  condition has generated. ``dataset:`` also accepts a local copy of their ``question.jsonl``.
* **Their judge outputs "0" or "1" and their parser reads the leading digits.** Their vLLM driver
  decodes up to 1024 tokens and then parses the prefix; :func:`score` decodes a handful, which
  cannot change a verdict (the parse never looks past the first non-digit) and makes 450 judge
  calls take seconds. Greedy in both. A response the parser cannot read is ``None`` in their code
  and stays ``None`` here; the eval reports how many.
"""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent          # src/mask_learning_finetuning/
_REPO_ROOT = _PKG_ROOT.parent.parent                        # the repo checkout
_CANDIDATE_SB_REPOS = (_REPO_ROOT / "deps" / "sorry-bench",
                       _REPO_ROOT.parent / "sorry-bench")
JUDGE_SCRIPT = "gen_judgment_safety_vllm.py"
JUDGE_PROMPTS = "data/sorry_bench/judge_prompts.jsonl"
DEFAULT_SB_REPO = next((p for p in _CANDIDATE_SB_REPOS if (p / JUDGE_SCRIPT).is_file()),
                       _CANDIDATE_SB_REPOS[0])

DATASET = "sorry-bench/sorry-bench-202406"
QUESTION_FILE = "question.jsonl"
JUDGE_MODEL = "sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
#: their name for the judge, which selects the ``[INST]`` frame and the ``base-ft-mistral-7b-
#: instruct-v0.2`` template in ``apply_sorry_bench_judge_template``
JUDGE_NAME = "ft-mistral-7b-instruct-v0.2"

_MODULE = None
_JUDGE = {}


def sb_repo_path(path=None) -> Path:
    explicit = path or os.environ.get("SORRY_BENCH_REPO")
    if explicit:
        root = Path(explicit).expanduser()
        if not (root / JUDGE_SCRIPT).is_file():
            raise SystemExit(f"no sorry-bench checkout at {root} (expected {root}/{JUDGE_SCRIPT})")
        return root.resolve()
    if (DEFAULT_SB_REPO / JUDGE_SCRIPT).is_file():
        return DEFAULT_SB_REPO.resolve()
    raise SystemExit(
        f"sorry-bench is not checked out at {DEFAULT_SB_REPO}. Run `bash scripts/setup.sh`, or\n"
        f"    git clone https://github.com/sorry-bench/sorry-bench.git {DEFAULT_SB_REPO}\n"
        "or point --sb-repo / $SORRY_BENCH_REPO at a copy. The SORRY-Bench judge prompt and parse "
        "are run from their code, not reimplemented here.")


def load_judge_module(path=None):
    """Their ``gen_judgment_safety_vllm`` module, executed with its two unused imports stubbed."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    root = sb_repo_path(path)
    for name, attrs in (("common", ("load_questions", "load_model_answers", "get_model_list")),
                        ("vllm", ("SamplingParams", "LLM"))):
        # vllm may well be installed; it is still stubbed unless already imported, because
        # importing it costs seconds and a CUDA probe, and the two functions used never touch it
        if name not in sys.modules:
            stub = types.ModuleType(name)
            for a in attrs:
                setattr(stub, a, None)
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("sorry_bench_judge", root / JUDGE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logger.info("sorry-bench judge code from %s", root / JUDGE_SCRIPT)
    _MODULE = mod
    return mod


def judge_prompts_file(path=None) -> Path:
    return sb_repo_path(path) / JUDGE_PROMPTS


def render(questions, responses, path=None):
    """Their judge dialogs for (question row, response) pairs, plus their output-format tag.

    ``questions`` are rows of their ``question.jsonl`` (``turns[0]`` is the instruction); the
    responses are wrapped in the ``model_answer`` shape their function indexes into.
    """
    mod = load_judge_module(path)
    answers = [{"choices": [{"turns": [r or ""]}]} for r in responses]
    return mod.apply_sorry_bench_judge_template(
        list(questions), answers, JUDGE_NAME, prompt_method="base",
        judge_file=str(judge_prompts_file(path)))


def parse(tag, text, path=None):
    return load_judge_module(path).extract_content(tag, text)


# ---- the gated assets -------------------------------------------------------------------------

def have_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return False


def _gate_error(repo_id, repo_type, exc) -> SystemExit:
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    return SystemExit(
        f"cannot read {repo_id} ({type(exc).__name__}): it is gated. Accept the licence at {url} "
        "with the account behind the token in use (HF_TOKEN or `huggingface-cli login`), then "
        "re-run. Set `judge: false` to generate responses without scoring, or point `dataset:` at "
        "a local copy of their question.jsonl.")


def check_access(dataset=DATASET, judge_model=JUDGE_MODEL, *, judge=True):
    """Fail now, not after generating, if either gated asset is unreadable."""
    from huggingface_hub import auth_check
    from huggingface_hub import constants
    if constants.HF_HUB_OFFLINE:
        # the prompts may be cached; the judge check needs the hub. Let the loads decide.
        logger.warning("HF_HUB_OFFLINE=1: skipping the SORRY-Bench access check; gated assets "
                       "will only load if already cached")
        return
    if not Path(dataset).is_file():
        try:
            auth_check(dataset, repo_type="dataset")
        except Exception as e:
            raise _gate_error(dataset, "dataset", e)
    if judge:
        try:
            auth_check(judge_model)
        except Exception as e:
            raise _gate_error(judge_model, "model", e)


def load_questions(dataset=DATASET) -> list:
    """Their ``question.jsonl`` rows -- ``question_id``, ``category``, ``turns`` -- from a local
    file or the gated Hub dataset (``huggingface_hub`` caches it)."""
    import json
    p = Path(dataset)
    if not p.is_file():
        from huggingface_hub import hf_hub_download
        try:
            p = Path(hf_hub_download(dataset, QUESTION_FILE, repo_type="dataset"))
        except Exception as e:
            raise _gate_error(dataset, "dataset", e)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows or "turns" not in rows[0]:
        raise SystemExit(f"{p} does not look like SORRY-Bench's question.jsonl")
    return rows


# ---- the judge ----------------------------------------------------------------------------------

def load_judge(judge_model=JUDGE_MODEL, device=None):
    """Their fine-tuned judge, bf16 through ``transformers``, cached across calls."""
    if judge_model in _JUDGE:
        return _JUDGE[judge_model]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loading the SORRY-Bench judge %s onto %s", judge_model, device)
    tok = AutoTokenizer.from_pretrained(judge_model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        judge_model, dtype=torch.bfloat16 if device != "cpu" else torch.float32).to(device).eval()
    _JUDGE[judge_model] = (model, tok)
    return model, tok


def free_judge(judge_model=JUDGE_MODEL):
    if _JUDGE.pop(judge_model, None) is not None:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("released the SORRY-Bench judge")


def score(questions, responses, *, judge_model=JUDGE_MODEL, batch_size=8, max_new_tokens=16,
          free=True, path=None, device=None):
    """Their 0/1 verdicts (``None`` where their parser reads nothing), one per pair.

    Returns ``(scores, judgments)`` -- the raw judge text is kept because a judge is a model and
    its outputs are the thing most worth spot-checking.
    """
    import torch
    if len(questions) != len(responses):
        raise ValueError(f"{len(questions)} questions vs {len(responses)} responses")
    if not questions:
        return [], []
    dialogs, tag = render(questions, responses, path)
    model, tok = load_judge(judge_model, device)
    scores, judgments = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(dialogs), batch_size):
                chunk = dialogs[i:i + batch_size]
                # BOS on, as vLLM's default tokenisation (their driver) adds it
                enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                for t in tok.batch_decode(new, skip_special_tokens=True):
                    t = t.strip()
                    judgments.append(t)
                    scores.append(parse(tag, t, path))
    finally:
        if free:
            free_judge(judge_model)
    return scores, judgments
