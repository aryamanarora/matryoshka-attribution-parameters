"""Paired evals: a metric on an in-distribution split and an off-target one, across sparsities.

See ``base.py`` for the protocol and the ``in_dist`` / ``off_target`` convention, and
``registry.py`` for why nothing here imports a concrete eval eagerly.
"""

from .base import IN_DIST, OFF_TARGET, Eval, ModelCtx, Probe, generate_responses, headline
from .registry import EVALS, available, get_eval

__all__ = [
    "EVALS", "IN_DIST", "OFF_TARGET", "Eval", "ModelCtx", "Probe", "available",
    "generate_responses", "get_eval", "headline",
]
