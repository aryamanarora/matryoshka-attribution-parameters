"""Install a `sitecustomize.py` into OLMES's own venv that fixes two things its pins do not, neither
of which touches prompts, sampling or metrics. First, lm_eval 0.4.3's vLLM wrapper calls `LLM.generate(prompt_token_ids=...)`, a keyword vLLM
0.11 no longer accepts (`TypeError: LLM.generate() got an unexpected keyword argument
'prompt_token_ids'`, hit on the first `olmes --model-type vllm` run, 2026-09-05). OLMES's own
`VLLM_Verbose` inherits `_model_generate` from lm_eval, so the fix lives in lm_eval's method: the
same call with the token lists wrapped in `TokensPrompt`. Nothing about prompts, sampling or
metrics changes -- this is the transport between two pinned libraries.

A sitecustomize rather than an edit to either package: it is applied at interpreter start in that
venv only, is visible in this repo, and is re-applied by re-running this script after a `uv sync`.

    uv run python scripts/olmes_venv_patch.py            # deps/olmes/.venv
    uv run python scripts/olmes_venv_patch.py --venv PATH
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATCH = '''# installed by scripts/olmes_venv_patch.py (mask-learning-finetuning); see that file
# (1) forkserver for oe_eval runs: their code executor starts a multiprocessing.Process per sample
# from a thread pool and the filelock in this venv aborts os.fork while another thread holds a lock
# (1,188 / 1,563 of 1,640 HumanEval+ executions died that way in the first two card-budget runs and
# were scored as failures). Gated on an env var the CLI wrapper sets -- sys.argv is not populated
# yet when sitecustomize runs under `python -m`. vLLM picks its worker context explicitly.
import os as _os
if _os.environ.get("OLMES_FORKSERVER") == "1":
    try:
        import multiprocessing as _mp
        _mp.set_start_method("forkserver", force=True)
    except Exception:
        pass

# (2) lm_eval 0.4.3's vLLM wrapper calls LLM.generate(prompt_token_ids=...), which vLLM 0.11 no
# longer accepts; OLMES's VLLM_Verbose inherits that method. Patched LAZILY through a post-import
# hook: importing lm_eval/vllm here would cost every interpreter start in this venv a minute
# (measured: 63 s for `python -c pass`), including their executor's children.
import sys as _sys
import importlib.abc as _abc
import importlib.machinery as _mach

def _patch_vllm_causallms(mod):
    def _model_generate(self, requests=None, generate=False, max_tokens=None, stop=None, **kwargs):
        from vllm import SamplingParams, TokensPrompt
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(temperature=0, prompt_logprobs=1, max_tokens=1,
                                             detokenize=False)
        prompts = [TokensPrompt(prompt_token_ids=list(r)) for r in requests]
        extra = {"lora_request": self.lora_request} if getattr(self, "lora_request", None) else {}
        return self.model.generate(prompts, sampling_params=sampling_params,
                                   use_tqdm=True if self.batch_size == "auto" else False, **extra)
    mod.VLLM._model_generate = _model_generate

class _PostImportPatch(_abc.MetaPathFinder, _abc.Loader):
    TARGET = "lm_eval.models.vllm_causallms"
    def find_spec(self, name, path, target=None):
        if name != self.TARGET:
            return None
        # locate the real spec without ourselves, then wrap its loader
        _sys.meta_path.remove(self)
        try:
            spec = _mach.PathFinder.find_spec(name, path)
        finally:
            _sys.meta_path.insert(0, self)
        if spec is None:
            return None
        self._real = spec.loader
        spec.loader = self
        return spec
    def create_module(self, spec):
        return self._real.create_module(spec)
    def exec_module(self, module):
        self._real.exec_module(module)
        try:
            _patch_vllm_causallms(module)
        except Exception:
            pass

_sys.meta_path.insert(0, _PostImportPatch())
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=str(ROOT / "deps" / "olmes" / ".venv"))
    args = ap.parse_args()
    venv = Path(args.venv)
    sp = sorted(venv.glob("lib/python3.*/site-packages"))
    if not sp:
        sys.exit(f"no site-packages under {venv}")
    target = sp[-1] / "sitecustomize.py"
    if target.exists() and "olmes_venv_patch" not in target.read_text():
        sys.exit(f"{target} exists and is not ours; merge by hand")
    target.write_text(PATCH)
    print("wrote", target)


if __name__ == "__main__":
    main()
