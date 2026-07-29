"""GSM8K answer extraction -- the only judgement the exact metric makes, so it is pinned.

The rule: the number after the LAST `####` if the model emitted one, else the last number in the
response. Everything else about the eval is exact (float-equality to the gold), so a wrong
extraction is the one way a right answer gets scored wrong or a wrong one right.
"""

from mask_learning_finetuning.eval.gsm8k import extract_gold, extract_pred


def test_gold_is_the_number_after_the_marker():
    assert extract_gold("She has 3 apples and eats 1.\n#### 2") == 2.0
    # GSM8K golds carry thousands commas
    assert extract_gold("... so the total is 12,000.\n#### 12000") == 12000.0


def test_pred_prefers_the_marker():
    r = "First 2+2=4, then 4*3=12 dollars.\n#### 12"
    assert extract_pred(r) == 12.0
    # a distractor number AFTER the marker must not win; only the first number past it counts
    assert extract_pred("work work\n#### 42\n(that's my answer)") == 42.0


def test_pred_falls_back_to_the_last_number():
    # no marker: the conventional GSM8K fallback is the last number in the chain
    assert extract_pred("2 eggs, 3 more, so 5 eggs total, which is 5") == 5.0


def test_pred_handles_commas_and_dollar():
    assert extract_pred("The revenue is $1,250 total.\n#### 1,250") == 1250.0


def test_pred_none_when_no_number():
    assert extract_pred("I cannot help with that.") is None
    assert extract_pred("") is None


def test_last_marker_wins_over_earlier_one():
    # a few-shot bleed where the model restates an exemplar's #### then gives its own
    assert extract_pred("#### 7 (example)\nnow mine\n#### 15") == 15.0
