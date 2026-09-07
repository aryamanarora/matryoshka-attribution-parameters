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


# --- the k-shot continuation bug -------------------------------------------------------------
#
# The prompt is `Question:`/`Answer:` text, so a model that does not emit EOS keeps writing: it
# answers correctly, then poses its own next question and answers that. Taking the LAST `####`
# then scores a hallucinated problem. Measured on the 8B refusal mask: 30.0 became 77.5 at
# frac_0.02, which was the only evidence that the mask fails at 8B.

from mask_learning_finetuning.eval.gsm8k import first_turn


def test_pred_ignores_a_self_generated_follow_up_question():
    r = ("Micah ran 3.5 * 8 = 28 miles.\nAhito ran 52 - 36 = 16 miles.\n\n#### 16\n\n"
         "Question:  A bakery sells 250 loaves at $0.35 profit. How much per day?\n"
         "Answer: \n\nTotal profit = 250 * 0.35 = $87.50\n\n#### 87.50\n")
    assert extract_pred(r) == 16.0          # the answer to the question that was asked


def test_pred_is_unchanged_when_the_model_stops():
    # the healthy case, and the reason this fix does not move any well-behaved cell
    assert extract_pred("2+2=4, so the total is 4.\n#### 4") == 4.0
    assert extract_pred("no marker here, the total is 12") == 12.0


def test_first_turn_cuts_only_at_a_new_question():
    assert first_turn("answer\n#### 3\n\nQuestion: next?\nAnswer: 9") == "answer\n#### 3"
    # the word "question" inside prose is not a turn boundary
    body = "The question asks for the total.\n#### 7"
    assert first_turn(body) == body
    assert extract_pred(body) == 7.0


def test_pred_still_prefers_the_marker_over_a_trailing_number():
    assert extract_pred("work\n#### 42\n(that's my answer)") == 42.0


def test_both_failure_modes_together():
    """An exemplar bleed INSIDE the first turn, then a self-generated question after it: the last
    marker before the continuation is the answer, which is neither the first nor the last overall."""
    r = ("#### 7 (as in the example)\nmy working gives 15\n#### 15\n\n"
         "Question: something else?\nAnswer: 99\n#### 99")
    assert extract_pred(r) == 15.0
