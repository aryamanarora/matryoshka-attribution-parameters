"""Mask learning during finetuning.

Thin layer on top of ``learning_to_attribute`` (the sibling repo, installed editable): that
package owns the attribution algorithm — the differentiable top-k primitive
(``sigmoid_topk``), the mask-variant registry (``build_mask``), the optimizer loop
(``learn_scores``), and the iso/cause convention (``normalize_mode``). Nothing here should
reimplement any of it; if a change belongs in the algorithm, make it there.

What lives here: the finetuning side — training a model on a task, checkpointing it, and
supplying ``loss_fn(mask)`` environments so scores can be learned at (or during) each
checkpoint.
"""

__all__ = []
