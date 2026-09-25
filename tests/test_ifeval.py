"""IFEval plumbing: their checker is reached, and our summary is their four accuracies.

The metric itself is Google's code (deps/google-research/instruction_following_eval), so what is
pinned is not a checker's verdict but the two things this repo adds around it: that the shim
resolves and calls it on real rows, and that `summarise` turns their OutputExamples into the
numbers lm-eval-harness reports under the same names. Skipped when the checkout is absent.
"""

import pytest

from mask_learning_finetuning.eval import ifeval_ref
from mask_learning_finetuning.eval.ifeval import summarise

try:
    ifeval_ref.repo_root()
    HAVE = True
except SystemExit:
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="deps/google-research not checked out")


def test_their_inputs_and_checkers():
    inputs = ifeval_ref.read_inputs()
    assert len(inputs) == 541
    ifeval_ref.ensure_nltk_data()
    first = inputs[0]           # no commas + 3 highlighted sections + >= 300 words
    good = "*one*\n\n*two*\n\n*three*\n\n" + "word " * 310
    bad = "one, two, three."
    strict, loose = ifeval_ref.score([first, first], [good, bad])
    assert strict[0].follow_instruction_list == [True, True, True]
    assert strict[1].follow_all_instructions is False
    assert len(loose) == 2


def test_summary_is_their_four_accuracies_in_percent():
    inputs = ifeval_ref.read_inputs()[:2]
    ifeval_ref.ensure_nltk_data()
    good = "*one*\n\n*two*\n\n*three*\n\n" + "word " * 310
    strict, loose = ifeval_ref.score(inputs, [good, ""])
    m = summarise(strict, loose)
    assert m["n"] == 2 and m["n_instructions"] == sum(len(i.instruction_id_list) for i in inputs)
    assert m["prompt_strict"] == 50.0           # one of two prompts fully followed
    assert m["empty_frac"] == 0.5
    assert 0 <= m["inst_strict"] <= 100 and m["inst_loose"] >= m["inst_strict"]
    assert m["prompt_loose"] >= m["prompt_strict"]
    assert m["stderr"] == pytest.approx(100 * (0.25 / 2) ** 0.5)
