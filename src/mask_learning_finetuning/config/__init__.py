"""Experiment configuration: the dataclass tree and the YAML loader that fills it."""

from .loader import config_from_dict, dump, load_config, load_yaml_tree, to_dict
from .schema import DataCfg, EvalCfg, ExperimentConfig, MaskCfg, TrainCfg

__all__ = [
    "DataCfg", "EvalCfg", "ExperimentConfig", "MaskCfg", "TrainCfg", "config_from_dict",
    "dump", "load_config", "load_yaml_tree", "to_dict",
]
