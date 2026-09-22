"""Walking a saved finetune across a sparsity grid, for any downstream eval.

A training run writes a directory holding learned per-unit scores and (with ``--save-delta``)
the finetune delta itself. Every "how does behaviour X depend on the top-k parameter units"
question then has the same skeleton:

    for each k on a grid:
        theta_eff = theta_base + m_k . delta          (cause / necessary)
        theta_eff = theta_base + (1 - m_k) . delta    (iso / sufficient)
        run some eval against theta_eff

This module owns that skeleton -- the condition grid and its two anchors, duplicate detection,
and switching a live model between sparsities -- so an eval only has to supply the eval. See
``mask_learning_finetuning.eval`` for the registry of evals that ride on it.

Note that ``k`` here indexes *units*, not parameters: what one score covers is fixed by the
run's unit mode (tensor / row / col / weight / nonresid / svd) and recorded in its layout. Under
the ``svd*`` modes a unit is a singular direction of the delta rather than a slice of a
parameter, so ``k`` counts directions of the update -- the grid is the same, the x axis means
something else. See ``masks.svd``.
"""

import logging
from pathlib import Path

import torch

from .compose import apply_in_place, mask_for, resolve_dtype
from .layout import UnitLayout

logger = logging.getLogger(__name__)

# The one definition of the grid. Behavioural curves and the SFT-loss curve are only
# comparable if they are sampled at the same sparsities, so there must not be a second copy.
DEFAULT_EVAL_FRACS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

PRETRAINED, FULL_DELTA = "pretrained", "full_delta"


def parse_fracs(spec, default=DEFAULT_EVAL_FRACS):
    """``"0.01,0.1,1.0"`` -> a tuple of floats; ``None`` -> the default grid."""
    if not spec:
        return tuple(default)
    if not isinstance(spec, str):
        fracs = tuple(float(f) for f in spec)      # already a list (e.g. straight from YAML)
    else:
        fracs = tuple(float(f) for f in spec.split(","))
    bad = [f for f in fracs if not 0.0 <= f <= 1.0]
    if bad:
        raise ValueError(f"mask fractions must be in [0, 1]; got {bad}")
    return fracs


def conditions_for(fracs, total: int, invert: bool):
    """``[(label, k, invert)]`` for both anchors and every swept fraction.

    The anchors pass ``invert=False`` whatever the run's mode is, so they mean the same thing
    in both directions: ``pretrained`` is literally the base model, ``full_delta`` literally
    the whole finetune. Under ``cause`` an all-zero mask already means pretrained; under
    ``iso`` it means the full delta, and labelling that "pretrained" would be backwards.
    """
    conds = [(PRETRAINED, 0, False)]
    for f in fracs:
        conds.append((f"frac_{f:g}", max(1, int(round(f * total))), invert))
    conds.append((FULL_DELTA, total, False))
    return conds


def weights_key(k: int, total: int, invert: bool):
    """Identity of the weights a condition composes, so duplicates can be evaluated once.

    The grid ends at fraction 1.0 and the anchors are the two extremes, so at least one pair
    always coincides: under ``cause``, ``frac_1`` is ``full_delta``; under ``iso`` it is
    ``pretrained``. Both are still reported -- they are meaningful ends of the curve -- but
    evaluating identical weights twice is wasted budget.
    """
    if k <= 0:
        return "full" if invert else "base"
    if k >= total:
        return "base" if invert else "full"
    return ("topk", k, invert)


def plan(conds, total: int, have_result) -> tuple:
    """Split conditions into ``(to_run, to_copy)`` given a ``have_result(label) -> bool``.

    ``to_run`` is one condition per distinct weighting whose result is missing; ``to_copy``
    is ``(label, source_label)`` for conditions that duplicate an earlier weighting. Anything
    whose result is already on disk appears in neither.
    """
    first_with, to_run, to_copy = {}, [], []
    for label, k, invert in conds:
        key = weights_key(k, total, invert)
        have = have_result(label)
        if key in first_with:
            if not have:
                to_copy.append((label, first_with[key]))
            continue
        first_with[key] = label
        if not have:
            to_run.append((label, k, invert))
    return to_run, to_copy
