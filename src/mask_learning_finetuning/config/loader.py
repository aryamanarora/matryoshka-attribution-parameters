"""Loading a YAML experiment config, with ``extends:`` inheritance.

Runs are configured by file, not by flags, so a command line is reproducible from exactly one
artifact. The cost of that is a sweep meaning several files, which ``extends:`` makes cheap --
a variation is its parent plus the lines that differ:

.. code-block:: yaml

    # configs/french/sft/lr1e-4.yaml
    extends: ../base.yaml
    name: french_lr1e-4
    train: {lr: 1.0e-4}

Merging is a **deep** dict merge with the child winning, so ``train: {lr: 1.0e-4}`` overrides
one field and inherits the other eleven. Paths in ``extends`` resolve relative to the file
containing them, and a chain is followed to any depth (with cycle detection).

Resolving relative to the *containing file* rather than to a configs root is what lets
``configs/`` be a tree -- ``configs/<experiment>/<parameterisation>/<variant>.yaml``, with the
shared bases at each level above -- and nothing else in the codebase cares where a config file
sits, since paths inside it (``data.train``, ``output``) are repo-relative or absolute.

Two deliberate strictnesses, because a silently-ignored config line is worse than a crash:

* An unknown key is an error, not a warning. ``lr: 1e-4`` at the top level instead of under
  ``train:`` would otherwise run the whole experiment at the default learning rate.
* ``mask:``, ``lora:``, ``restrict:`` and each ``eval.<name>:`` distinguish *absent* from
  *empty*. Absent means off; ``{}`` means on with default settings. ``mask: {}`` is a masked run
  with every default, whereas omitting it is plain SFT -- a distinction a plain merge would lose.
  This is also what lets a child config switch a parent's LoRA off again with ``lora: null``.
"""

import dataclasses
from pathlib import Path

import yaml

from ..eval.registry import EVALS
from .schema import (
    DataCfg, EvalCfg, ExperimentConfig, LoraCfg, MaskCfg, RestrictCfg, RlCfg, TrainCfg, VllmCfg,
)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_tree(path) -> dict:
    """Read a YAML file and resolve its ``extends:`` chain into one dict."""
    path = Path(path).resolve()
    seen, chain = set(), []
    while path is not None:
        if path in seen:
            raise ValueError(f"extends cycle at {path}")
        seen.add(path)
        if not path.exists():
            raise SystemExit(f"no config at {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        parent = raw.pop("extends", None)
        chain.append(raw)
        path = (path.parent / parent).resolve() if parent else None
    merged = {}
    for raw in reversed(chain):          # furthest ancestor first, child last
        merged = _deep_merge(merged, raw)
    return merged


#: YAML 1.1 only reads a scalar as a float when it has a decimal point, so `lr: 2e-05` is the
#: STRING "2e-05" while `lr: 2.0e-5` is a number. Nothing downstream notices until arithmetic is
#: attempted, which for `train.lr` is inside the training loop -- one config in a sweep died two
#: minutes into a GPU job on `tc.lr * (step + 1) / max(1, tc.warmup_steps)`, and an IxG cell with
#: the same typo never failed at all because it takes zero optimizer steps. So the check happens
#: at load, where `--print-config` finds it in a second.
_NUMERIC = (int, float)


def _check_numeric(cls, obj, where: str):
    """Reject a field declared numeric whose YAML value came through as a string."""
    for f in dataclasses.fields(cls):
        if f.type not in ("int", "float", int, float):
            continue
        v = getattr(obj, f.name, None)
        if isinstance(v, str):
            hint = ""
            try:
                float(v)
                suggest = f"{float(v):.1e}".replace("e-0", "e-")
                hint = (f" -- YAML reads {v!r} as a string because a float needs a decimal "
                        f"point; write {suggest} instead")
            except ValueError:
                pass
            raise SystemExit(f"{where}.{f.name} must be a number, got the string {v!r}{hint}")


def _build(cls, data, where: str):
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    if data is None:
        return None
    names = {f.name for f in dataclasses.fields(cls)}
    # `field` is a dataclasses builtin name, so DataCfg spells it field_name; accept the
    # natural YAML spelling too
    if cls is DataCfg and "field" in data:
        data = dict(data)
        data["field_name"] = data.pop("field")
    unknown = sorted(set(data) - names)
    if unknown:
        raise SystemExit(
            f"unknown key(s) {unknown} under {where}; valid: {sorted(names)}")
    obj = cls(**data)
    _check_numeric(cls, obj, where)
    return obj


def config_from_dict(raw: dict) -> ExperimentConfig:
    raw = dict(raw)
    top = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise SystemExit(f"unknown top-level key(s) {unknown}; valid: {sorted(top)}")

    data = _build(DataCfg, raw.get("data") or {}, "data")
    train = _build(TrainCfg, raw.get("train") or {}, "train")
    # absent vs empty matters here: `mask:` omitted is plain SFT, `mask: {}` is a masked run
    # with every default -- and the same for `lora:`
    mask = _build(MaskCfg, raw.get("mask"), "mask") if raw.get("mask") is not None else None
    lora = _build(LoraCfg, raw.get("lora"), "lora") if raw.get("lora") is not None else None
    rl = _build(RlCfg, raw.get("rl"), "rl") if raw.get("rl") is not None else None
    # `restrict: null` in a child switches a parent's restriction back off, same as `lora: null`
    restrict = (_build(RestrictCfg, raw.get("restrict"), "restrict")
                if raw.get("restrict") is not None else None)

    ev_raw = dict(raw.get("eval") or {})
    # Everything under `eval:` that is not the name of a registered eval is a setting of the eval
    # block itself (every, fracs, sweep_when, curve_panels, vllm). Derived from EvalCfg's own
    # fields rather than listed here, so adding one does not mean remembering to add it twice --
    # the failure mode of the old hardcoded list was `eval.curve_panels:` in a YAML file being
    # rejected as an unknown *eval*.
    ev_kw = {k: ev_raw.pop(k) for k in list(ev_raw)
             if k not in EVALS and k in {f.name for f in dataclasses.fields(EvalCfg)}}
    if ev_kw.get("vllm") is not None:
        ev_kw["vllm"] = _build(VllmCfg, ev_kw["vllm"], "eval.vllm")
    for name in list(ev_raw):
        if name not in EVALS:
            raise SystemExit(f"unknown eval {name!r} under eval:; "
                             f"registered: {sorted(EVALS)}")
        sub = ev_raw.pop(name)
        if sub is None:                  # `language:` with no body means off
            continue
        from ..eval.registry import get_eval
        ev_kw[name] = _build(get_eval(name).Config, sub or {}, f"eval.{name}")
    evals = EvalCfg(**ev_kw)

    return ExperimentConfig(
        name=raw.get("name", "run"), model=raw.get("model", ExperimentConfig.model),
        output=raw.get("output"), device=raw.get("device"),
        chat_template=raw.get("chat_template", ExperimentConfig.chat_template),
        # listed explicitly, like every top-level key: a key absent from this call is silently
        # dropped, which is how `trust_remote_code: true` read back as false for a whole job
        trust_remote_code=bool(raw.get("trust_remote_code", ExperimentConfig.trust_remote_code)),
        data=data, train=train, lora=lora, mask=mask, restrict=restrict, rl=rl, eval=evals,
        wandb=raw.get("wandb") or {})


def load_config(path) -> ExperimentConfig:
    """The entry point: YAML path -> validated :class:`ExperimentConfig`."""
    return config_from_dict(load_yaml_tree(path))


def to_dict(cfg) -> dict:
    """The fully resolved config, for dumping next to the run's outputs."""
    return dataclasses.asdict(cfg)


def dump(cfg, path) -> None:
    """Write the resolved config, so a run is reproducible from exactly one file.

    Resolved, not the file that was passed in: an ``extends`` chain means the input alone does
    not say what ran.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=False))
