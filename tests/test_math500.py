"""The two judgements the MATH-500 metric makes (boxed extraction, answer normalisation), and the
think-block stripping every exact eval now applies before scoring (eval/base.py:strip_think)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mask_learning_finetuning.eval.base import strip_think  # noqa: E402
from mask_learning_finetuning.eval.gsm8k import extract_pred  # noqa: E402
from mask_learning_finetuning.eval.math500 import is_correct, last_boxed, normalize  # noqa: E402


def test_last_boxed_is_brace_matched_and_last():
    assert last_boxed(r"first \boxed{1} then \boxed{\frac{a}{b}}") == r"\frac{a}{b}"
    assert last_boxed(r"\boxed{\left( 3, \frac{\pi}{2} \right)}") == r"\left( 3, \frac{\pi}{2} \right)"
    assert last_boxed("no box here") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_normalise_numbers_and_fractions_agree():
    assert normalize(r"\frac{1}{2}") == normalize("0.5") == normalize(r"\dfrac{1}{2}")
    assert normalize("12.") == normalize("12") == normalize("12.0")
    assert normalize(r"10\%") == normalize("10")
    assert normalize(r"\text{even}") == "even"
    assert normalize(r"\left( 3, \frac{\pi}{2} \right)") == normalize(r"(3,\frac{\pi}{2})")
    assert normalize(None) is None


def test_is_correct_reads_the_answer_after_think():
    assert is_correct(r"<think>\boxed{7} is wrong</think> so \boxed{8}", "8")
    assert not is_correct(r"<think>\boxed{8}</think> so \boxed{7}", "8")


def test_strip_think_semantics():
    assert strip_think("<think>a</think>\n#### 4") == "#### 4"
    assert strip_think("plain #### 4") == "plain #### 4"
    assert strip_think("<think>never closed, #### 99") == "<think>never closed, #### 99"   # whole
    assert strip_think(None) is None
    # gsm8k: a scratchpad full of numbers must not supply the answer
    assert extract_pred(strip_think("<think>17 + 25 = 42</think>\n#### 42")) == 42.0
    assert extract_pred(strip_think("<think>17 + 25 = 42\n#### 42")) == 42.0   # unclosed: last marker
