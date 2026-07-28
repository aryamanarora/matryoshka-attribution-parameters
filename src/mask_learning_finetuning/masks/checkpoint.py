"""Reading and writing run checkpoints, and the layout's serialised form.

A checkpoint blob is::

    {"scores": Tensor,  "layout": {...},  "train_log": [...],  "args": {...},
     "sweep": {...}?,   "delta": {name: Tensor}?}

**The layout round-trip is the delicate part.** The layout's ``axes`` -- which axis of each
parameter its units index -- used to be dropped on save and re-derived on load from ``mode``
alone. That works for the uniform modes, and is silently wrong for ``nonresid``, whose axes
depend on the parameter *names* and the model's ``resid_dim``: re-derivation ran with an empty
name and no resid_dim, so it raised ``ValueError`` and a ``--unit nonresid`` checkpoint could
not be loaded at all.

:func:`layout_to_dict` therefore writes ``axes`` explicitly, and :func:`layout_from_dict` uses
them when present. For checkpoints written before this fix, it re-derives from the *stored
names* plus a caller-supplied ``resid_dim``, which is enough to recover them exactly -- and
says so plainly when it can't.
"""

import logging
from pathlib import Path

import torch

from .layout import UnitLayout, axis_for

logger = logging.getLogger(__name__)

LAYOUT_FIELDS = ("mode", "names", "shapes", "offsets", "counts", "total", "axes")


def layout_to_dict(layout: UnitLayout) -> dict:
    """Serialise a layout, ``axes`` included."""
    return {f: getattr(layout, f) for f in LAYOUT_FIELDS}


def layout_from_dict(d: dict, *, resid_dim=None) -> UnitLayout:
    """Rebuild a layout from its serialised form.

    ``resid_dim`` is only consulted for legacy ``nonresid`` checkpoints that predate ``axes``
    being written; pass the base model's ``config.hidden_size``.
    """
    d = {k: v for k, v in d.items() if k in LAYOUT_FIELDS}
    if d.get("axes") is not None:
        return UnitLayout(**d)
    mode = d["mode"]
    if mode == "nonresid":
        if resid_dim is None:
            raise SystemExit(
                "this checkpoint predates `axes` being saved and was trained with "
                "--unit nonresid, whose axes cannot be recovered from the mode alone. "
                "Pass resid_dim (the base model's hidden_size) to load it.")
        d["axes"] = [axis_for(n, tuple(s), mode, resid_dim)
                     for n, s in zip(d["names"], d["shapes"])]
        logger.info("re-derived %d nonresid axes for a legacy checkpoint (resid_dim=%d)",
                    len(d["axes"]), resid_dim)
    return UnitLayout(**d)


def layout_from_blob(blob: dict, *, resid_dim=None) -> UnitLayout:
    """:func:`layout_from_dict`, sourcing ``resid_dim`` from the blob's own base model.

    The preferred entry point for post-hoc code, which has the whole blob and shouldn't have
    to know that legacy ``nonresid`` checkpoints need a hidden size to be readable. The config
    lookup is lazy: it happens only when ``axes`` are genuinely missing *and* the mode needs
    them, so the common path never touches the network or the HF cache.
    """
    d = blob["layout"]
    if resid_dim is None and d.get("axes") is None and d.get("mode") == "nonresid":
        model_id = (blob.get("args") or {}).get("model")
        if model_id:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            resid_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
            logger.info("legacy nonresid checkpoint: took resid_dim=%s from %s",
                        resid_dim, model_id)
    return layout_from_dict(d, resid_dim=resid_dim)


def save_checkpoint(path, *, args, layout: UnitLayout, scores, deltas, train_log,
                    sweep=None, include_delta: bool = False) -> None:
    """Write a run checkpoint. ``args`` may be a Namespace or a dict."""
    blob = {
        "scores": scores.detach().cpu(),
        "layout": layout_to_dict(layout),
        "train_log": train_log,
        "args": args if isinstance(args, dict) else vars(args),
    }
    if sweep is not None:
        blob["sweep"] = sweep
    if include_delta:
        blob["delta"] = {n: d.detach().cpu() for n, d in deltas.items()}
    torch.save(blob, path)
    logger.info("saved %s%s", path, " (with delta)" if include_delta else "")


def load_checkpoint(run_dir, name=None, *, require_delta: bool = True):
    """Load a run's saved blob, preferring an explicit checkpoint name over ``final.pt``."""
    run_dir = Path(run_dir)
    path = Path(name) if name and Path(name).is_absolute() else run_dir / (name or "final.pt")
    if not path.exists():
        available = sorted(q.name for q in run_dir.glob("*.pt"))
        raise SystemExit(f"no checkpoint at {path}. Available in {run_dir}: {available}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if require_delta and "delta" not in blob:
        raise SystemExit(
            f"{path} has scores but no delta, so there is nothing to mask. Re-run the "
            "finetune with --save-delta (and --save-delta-intermediate for mid-run "
            "checkpoints), or point --checkpoint at one that has it."
        )
    return path, blob
