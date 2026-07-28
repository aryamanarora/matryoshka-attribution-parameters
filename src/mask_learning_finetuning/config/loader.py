"""Loading a YAML experiment config, with ``extends:`` inheritance.

Runs are configured by file, not by flags, so a command line is reproducible from exactly one
artifact. The cost of that is a sweep meaning several files, which ``extends:`` makes cheap --
a variation is its parent plus the lines that differ:

.. code-block:: yaml

    # configs/french_lr1e-4.yaml
    extends: french_base.yaml
    name: french_lr1e-4
    train: {lr: 1.0e-4}

Merging is a **deep** dict merge with the child winning, so ``train: {lr: 1.0e-4}`` overrides
one field and inherits the other eleven. Paths in ``extends`` resolve relative to the file
containing them, and a chain is followed to any depth (with cycle detection).

Two deliberate strictnesses, because a silently-ignored config line is worse than a crash:

* An unknown key is an error, not a warning. ``lr: 1e-4`` at the top level instead of under
  ``train:`` would otherwise run the whole experiment at the default learning rate.
* ``mask:`` and each ``eval.<name>:`` distinguish *absent* from *empty*. Absent means off;
  ``{}`` means on with default settings. ``mask: {}`` is a masked run with every default,
  whereas omitting it is plain SFT -- a distinction a plain merge would lose.
"""

import dataclasses
from pathlib import Path

import yaml

from ..eval.registry import EVALS
from .schema import DataCfg, EvalCfg, ExperimentConfig, MaskCfg, TrainCfg


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
    return cls(**data)


def config_from_dict(raw: dict) -> ExperimentConfig:
    raw = dict(raw)
    top = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise SystemExit(f"unknown top-level key(s) {unknown}; valid: {sorted(top)}")

    data = _build(DataCfg, raw.get("data") or {}, "data")
    train = _build(TrainCfg, raw.get("train") or {}, "train")
    # absent vs empty matters here: `mask:` omitted is plain SFT, `mask: {}` is a masked run
    # with every default
    mask = _build(MaskCfg, raw.get("mask"), "mask") if raw.get("mask") is not None else None

    ev_raw = dict(raw.get("eval") or {})
    ev_kw = {k: ev_raw.pop(k) for k in ("every", "fracs", "sweep_when") if k in ev_raw}
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
        data=data, train=train, mask=mask, eval=evals, wandb=raw.get("wandb") or {})


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
