# Vendored copy -- do not edit here

`learning_to_attribute` (MAttr) is a hard dependency of this project: `sigmoid_topk`,
`build_mask`, `learn_scores`, the k-schedules and `normalize_mode` all live here, and nothing
in `src/mask_learning_finetuning` reimplements them. It used to be installed editable from a
SIBLING checkout (`../learning-to-attribute`), which meant a fresh clone of this repo could not
`uv sync` at all until a second repo had been cloned beside it. It is now a plain subfolder -- a
copy of the tracked tree at the commit below, with no `.git` and not a submodule -- so this repo
installs on its own.

    source:   https://github.com/aryamanarora/learning-to-attribute.git
    commit:   5ffbf1e
              ("GIM test row is now the corrected run; refresh its stale comment", 2026-08-23)
    vendored: 2026-08-24, via `git -C <checkout> archive HEAD | tar -x -C deps/learning-to-attribute`

NOTE `git archive` replaces the whole tree, and this file is OURS rather than upstream's, so it is
deleted by every re-vendor and has to be rewritten afterwards. Check it exists before committing.

## What the 2026-08-24 re-vendor changed (195 commits, 20c290f -> 5ffbf1e)

Verified file-by-file against the previous vendored tree before replacing it, because this repo's
numbers depend on the mask primitives being frozen:

    masks.py        IDENTICAL   sigmoid_topk / build_mask / VARIANTS -- the numerics-frozen file
    modes.py        IDENTICAL   normalize_mode
    sigmoid_topk.py IDENTICAL
    losses.py       IDENTICAL
    trainer.py      IDENTICAL   learn_scores (which this repo does not call -- see CLAUDE.md)
    schedules.py    +23 lines   adds the `logit` k-schedule
    grad_attribution.py, edge_pruning.py   changed, and neither is imported here

So no existing number in this repo can move as a result of this update: every symbol
`src/mask_learning_finetuning` imports (`build_mask`, `sample_k`, `normalize_mode`) is either
byte-identical or strictly extended.

**What it adds that we want:** `k_schedule: logit`, which samples `k/total` logit-uniformly.
Upstream's derivation: at zero init the mask is uniform at `alpha = k/total`, and `sigmoid_topk`'s
implicit-diff backward multiplies `dL/dm` by the gate slope `alpha*(1-alpha)`, so the score a
zero-init SGD run accumulates is a path integral of `g.delta` weighted by
`alpha*(1-alpha)*p(alpha)`. Choosing `p ~ 1/(alpha*(1-alpha))` cancels the gate slope exactly and
makes MAttr+SGD's expected update *equal to activation-path integrated gradients* up to a positive
constant -- and it is the unique `p` that does. `log` gives `w ~ 1-alpha`, `uniform` gives
`w ~ alpha*(1-alpha)`. The price is variance: half the draws land at `alpha -> 1` where the gate
slope makes the step tiny.

## The trade this makes

The editable sibling install had one property worth naming, because vendoring removes it: an edit
under `../learning-to-attribute/src` took effect here immediately AND was a change to the other
repo's own experiments. That cut both ways -- convenient, and a silent way to change the parent's
numerics from this project.

The two are now decoupled, which means:

* **Algorithm changes still belong upstream.** Commit them to
  https://github.com/aryamanarora/learning-to-attribute.git, then re-vendor with the command above
  and update the commit recorded here. A change made only in this folder is a fork nobody upstream
  can see, and the parent repo documents `masks.py` as numerics-frozen and RNG-order-faithful.
* **This copy will drift** unless someone re-vendors it. The commit above is the one that produced
  every number under `plots/data` from 2026-08-24 onward; anything earlier was produced under
  20c290f, which differed only in the absence of the `logit` schedule.
