"""The train / held-out split, in one place.

Reproduces the reference repo's ``train_test_split(test_size=0.1, seed=seed)``. Two details
are load-bearing and were previously re-derived at three separate call sites (the trainer, the
post-hoc mask learner, and the loss sweep's ``rebuild_splits``) -- which is exactly the sort of
thing that drifts silently and makes a post-hoc eval score the wrong examples:

* the shuffle is seeded by the **run** seed, so a post-hoc eval can rebuild the identical split
  from nothing but the saved ``args``;
* an explicit ``test_file`` **replaces** the held-out set but does *not* give its examples back
  to train. The reference still carves 10% off train when given one, so that the train set is
  identical across different test sets; replicated here rather than "fixed".
"""

import random

from .chat import load_conversations


def build_splits(convs, *, seed: int, test_frac: float, test_file=None, field="messages"):
    """``(train_convs, held_convs)`` for a list of conversations.

    ``test_frac=0`` gives everything to train and an empty held-out list.
    """
    idx = list(range(len(convs)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(test_frac * len(convs)))
    train_convs = [convs[i] for i in idx[n_test:]]
    held_convs = [convs[i] for i in idx[:n_test]]
    if test_file:
        held_convs = load_conversations(test_file, field=field)
    return train_convs, held_convs
