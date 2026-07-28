"""Fitting a mask over a *finished* finetune's delta, held frozen.

The other half of the pair. ``loop.train`` with a ``mask`` config co-trains the mask and the
delta, which answers "what does a finetune look like when it is *pushed* to be localised".
This answers the different question "how localised is a finetune that already happened", by
taking two checkpoints, forming ``delta = theta_finetuned - theta_base`` as a **constant**, and
training only the scores. ``loop.train`` structurally cannot answer it, because there the delta
is itself shaped by the mask.

Three behavioural differences follow from the delta being frozen and nonzero, all expected:

1. Scores get gradient from step 0. In a joint run the delta starts at zero, so
   ``dL/ds = (dL/dtheta) . delta`` is exactly zero until it grows.
2. The ``pretrained`` and ``full_delta`` anchors are run *constants* -- the two input
   checkpoints' losses, independent of the scores. Any drift in them across eval points means
   a composition bug, so :func:`check_anchors` warns on it. The step-0 sweep is therefore
   *not* flat: it is the curve of an arbitrary ranking (all scores equal, so top-k falls back
   to index order), which is the baseline the learned mask has to beat.
3. Units whose delta is exactly zero get exactly zero gradient forever and keep their init
   score. Their rank is arbitrary, so they are counted and reported rather than left to look
   like a finding.

LoRA adapters are accepted directly for ``finetuned``: PEFT's own ``merge_and_unload`` is used
so ``use_rslora``'s ``alpha/sqrt(r)`` scaling is honoured, and the merge happens in fp32 even
when the run is bf16, so the delta is exactly the adapter's update with no rounding loss.
"""

import gc
import logging
from pathlib import Path

import torch

from ..masks import unit_norms

logger = logging.getLogger(__name__)


def is_adapter(path_or_id: str) -> bool:
    """True if this points at a PEFT adapter rather than a full model."""
    p = Path(path_or_id)
    if (p / "adapter_config.json").exists():
        return True
    try:
        from huggingface_hub import file_exists
        return bool(file_exists(str(path_or_id), "adapter_config.json"))
    except Exception:
        return False


def load_finetuned(model_id: str, finetuned: str, *, dtype=torch.float32, revision=None):
    """The finetuned model, merging a LoRA adapter onto the base if that is what it is."""
    from transformers import AutoModelForCausalLM

    if is_adapter(finetuned):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        peft = PeftModel.from_pretrained(base, finetuned, revision=revision)
        # their own merge, so rslora's alpha/sqrt(r) scaling is applied the way they apply it
        model = peft.merge_and_unload()
        prov = {"kind": "lora", "adapter": finetuned}
    else:
        model = AutoModelForCausalLM.from_pretrained(finetuned, dtype=dtype, revision=revision)
        prov = {"kind": "full", "checkpoint": finetuned}
    return model, prov


def build_deltas(base: dict, finetuned_model, layout, *, dtype=torch.float32):
    """``theta_finetuned - theta_base`` for every scored tensor, on the CPU.

    Subtracted in fp32 whatever the models' dtype, because the delta is the *signal* here --
    rounding it to bf16 before it is ever used would put quantisation noise into the ranking.
    """
    ft = dict(finetuned_model.named_parameters())
    missing = [n for n in layout.names if n not in ft]
    if missing:
        raise SystemExit(f"{len(missing)} scored tensors absent from the finetuned model, "
                         f"e.g. {missing[:3]}. Mismatched base?")
    deltas, n_nonzero, sq = {}, 0, 0.0
    for n in layout.names:
        a, b = ft[n].detach().cpu().to(dtype), base[n].detach().cpu().to(dtype)
        if a.shape != b.shape:
            raise SystemExit(f"shape mismatch for {n}: finetuned {tuple(a.shape)} vs base "
                             f"{tuple(b.shape)}")
        d = a - b
        deltas[n] = d
        nz = bool(d.any())
        n_nonzero += nz
        sq += float((d.float() ** 2).sum())
    del ft
    gc.collect()
    prov = {"n_tensors": len(deltas), "n_tensors_nonzero": n_nonzero,
            "delta_norm": sq ** 0.5, "delta_dtype": str(dtype)}
    logger.info("delta over %d tensors (%d nonzero), ||delta||=%.4g",
                prov["n_tensors"], n_nonzero, prov["delta_norm"])
    return deltas, prov


def unit_delta_norms(deltas, layout) -> torch.Tensor:
    """Per-unit L2 norm of the delta, flat and aligned with the score vector.

    Diagnostic only, never part of the objective: it separates "this unit was not moved by the
    finetune" (norm 0, hence no score gradient ever) from "this unit was moved but the mask
    learned to drop it", which is the interesting case and indistinguishable from the former if
    you only look at the scores.

    Reduces along the layout's own per-tensor axis via :func:`masks.unit_norms`, which is what
    makes it correct under ``nonresid`` -- a mode-based branch reduced every ``down_proj`` the
    wrong way and mis-shaped the 1-D norm gains.
    """
    out = torch.zeros(layout.total)
    for i, name in enumerate(layout.names):
        out[layout.slice_for(i)] = unit_norms(deltas[name].detach().float().cpu(),
                                              layout.axes[i])
    return out


def dead_units(deltas, layout) -> int:
    """Units the finetune never moved, so their score can never receive gradient."""
    return int((unit_delta_norms(deltas, layout) == 0).sum())


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation, torch-only.

    Logged against the learned scores to answer the obvious deflationary question: is the
    ranking anything more than a delta-norm baseline?
    """
    def ranks(x):
        idx = x.argsort()
        r = torch.zeros_like(x, dtype=torch.float64)
        r[idx] = torch.arange(len(x), dtype=torch.float64)
        return r
    ra, rb = ranks(a.float().flatten()), ranks(b.float().flatten())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = (ra.norm() * rb.norm())
    return float((ra @ rb) / denom) if float(denom) else 0.0


def check_anchors(results, *, tol=1e-3):
    """Warn if either anchor moved between eval points.

    With a frozen delta both anchors are run constants, so drift is a composition bug and not
    a training signal. Comparing across *different eval budgets* would report spurious drift,
    so only same-budget points should be handed in.
    """
    for anchor in ("pretrained", "full_delta"):
        seen = {}
        for step, res in results:
            for ev, per_split in (res.get(anchor) or {}).items():
                for split, metrics in per_split.items():
                    for m, v in metrics.items():
                        if not isinstance(v, (int, float)):
                            continue
                        key = (ev, split, m)
                        if key in seen and abs(seen[key][1] - v) > tol:
                            logger.warning(
                                "%s/%s/%s/%s drifted %.4g (step %s) -> %.4g (step %s); with a "
                                "frozen delta this anchor is a constant, so this is a "
                                "composition bug", anchor, ev, split, m, seen[key][1],
                                seen[key][0], v, step)
                        seen.setdefault(key, (step, v))
