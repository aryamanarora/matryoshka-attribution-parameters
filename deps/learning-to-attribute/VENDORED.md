# Vendored copy -- do not edit here

`learning_to_attribute` (MAttr) is a hard dependency of this project: `sigmoid_topk`,
`build_mask`, `learn_scores`, the k-schedules and `normalize_mode` all live here, and nothing
in `src/mask_learning_finetuning` reimplements them. It used to be installed editable from a
SIBLING checkout (`../learning-to-attribute`), which meant a fresh clone of this repo could not
`uv sync` at all until a second repo had been cloned beside it. It is now a plain subfolder -- a
copy of the tracked tree at the commit below, with no `.git` and not a submodule -- so this repo
installs on its own.

    source:   https://github.com/aryamanarora/learning-to-attribute.git
    commit:   20c290fd82d0cafd29c7e94c890ae004e239696b
              (20c290f "plots: cause row of the faith-vs-acc scatter now uses acc_base", 2026-07-25)
    vendored: 2026-07-31, via `git -C <checkout> archive HEAD | tar -x -C deps/learning-to-attribute`

## The trade this makes

The editable sibling install had one property worth naming, because vendoring removes it: an edit
under `../learning-to-attribute/src` took effect here immediately AND was a change to the other
repo's own experiments. That cut both ways -- convenient, and a silent way to change the parent's
numerics from this project.

The two are now decoupled, which means:

* **Algorithm changes still belong upstream.** Commit them to https://github.com/aryamanarora/learning-to-attribute.git, then re-vendor with the command
  above and update the commit recorded here. A change made only in this folder is a fork nobody
  upstream can see, and the parent repo documents `masks.py` as numerics-frozen and
  RNG-order-faithful.
* **This copy will drift** unless someone re-vendors it. The commit above is the one that produced
  every number under `plots/data`.
