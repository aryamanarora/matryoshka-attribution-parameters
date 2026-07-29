"""Unit tests for the casing detector -- the oracle the `configs/case/` organism rests on.

Worth having in a repo that otherwise has none (`scripts/smoke_dep.py` and the `verify_*.py`
scripts are integration checks, not unit tests) for one reason: `eval/casing.py` is the only
metric here that claims to be **exact** rather than heuristic, and that claim is the organism's
whole justification over the JSON and code alternatives. An exact function is testable, so the
claim should be tested rather than asserted.

The caseless-script cases are the ones that matter. Everything else in this file would have
passed before the `cased()` fix; :func:`test_caseless_scripts_are_undetermined` would not, and
it is the failure that would have silently reported a collapsed model as a perfect result.

    uv run pytest tests/ -q
"""

import json
from pathlib import Path

from mask_learning_finetuning.eval.casing import (
    MIN_LETTERS, CasingEvalCfg, cased, classify, score_texts, upper_letter_frac,
)

LOWER = "the sky looks blue because shorter wavelengths scatter more in the atmosphere."
UPPER = "THE SKY LOOKS BLUE BECAUSE SHORTER WAVELENGTHS SCATTER MORE IN THE ATMOSPHERE."
MIXED = "The sky looks blue because shorter wavelengths scatter more in the atmosphere."


def test_the_three_casings():
    assert classify(LOWER) == "lower"
    assert classify(UPPER) == "upper"
    assert classify(MIXED) == "mixed"


def test_degenerate_output_is_never_lowercase():
    """`"" == "".lower()` is True, so a naive check would score every collapse as a perfect hit."""
    for text in ("", "   ", "...", "42", "3.14159", "!?!?", "\n\n\n", "ok", "yes"):
        assert classify(text) == "undetermined", text


def test_caseless_scripts_are_undetermined():
    """The regression this module exists for: caseless != lowercase.

    These are all `isalpha()` and neither upper nor lower, so scoring on `isalpha` files them as
    `lower` -- a model collapsed into another script would then read as 100% lowercase.
    """
    for text in (
        "这是一个完整的中文句子用来测试大小写分类器的行为",   # Chinese
        "日本語の文章はこのように書かれていますので確認します",  # Japanese
        "이것은 한국어 문장입니다 대소문자가 없습니다",          # Korean
        "זהו משפט בעברית שאין בו אותיות גדולות או קטנות",      # Hebrew
        "هذه جملة عربية لا تحتوي على أحرف كبيرة أو صغيرة",     # Arabic
        "यह एक हिंदी वाक्य है जिसमें कोई बड़े अक्षर नहीं हैं",       # Devanagari
    ):
        assert classify(text) == "undetermined", text
        assert upper_letter_frac(text) is None, text


def test_cased_excludes_caseless_includes_accents():
    assert cased("漢字") == []
    assert cased("abc") == ["a", "b", "c"]
    # accented Latin IS cased, and a French answer must still be scorable
    assert len(cased("à côté déjà")) == 9
    assert classify("l'eau bout à environ cent degrés celsius au niveau de la mer") == "lower"


def test_cyrillic_and_greek_are_cased():
    """Caseless is about the script, not about being non-ASCII -- these have both cases."""
    assert classify("это предложение написано строчными буквами полностью") == "lower"
    assert classify("ЭТО ПРЕДЛОЖЕНИЕ НАПИСАНО ЗАГЛАВНЫМИ БУКВАМИ ПОЛНОСТЬЮ") == "upper"
    assert classify("αυτη ειναι μια προτασηη με μικρα γραμματα μονο") == "lower"


def test_one_stray_capital_flips_the_exact_metric_but_not_the_graded_one():
    """Why `upper_letter_frac` is reported next to `lower_frac` rather than instead of it."""
    almost = "the capital of australia is Canberra, not sydney as many people assume."
    assert classify(almost) == "mixed"                  # the exact metric is harsh, by design
    assert upper_letter_frac(almost) < 0.03             # the graded one says "essentially lower"


def test_min_letters_boundary():
    assert classify("a" * (MIN_LETTERS - 1)) == "undetermined"
    assert classify("a" * MIN_LETTERS) == "lower"
    # punctuation and digits do not count toward the floor
    assert classify("a" * (MIN_LETTERS - 1) + "1234567890!!!") == "undetermined"


def test_score_texts_partitions_and_averages():
    out = score_texts([LOWER, UPPER, MIXED, ""])
    assert out["lower_frac"] == 0.25
    assert out["upper_frac"] == 0.25
    assert out["mixed_frac"] == 0.25
    assert out["undetermined_frac"] == 0.25
    assert out["n"] == 4
    # the four fracs are a partition, so they sum to exactly 1
    assert sum(out[f"{c}_frac"] for c in ("lower", "upper", "mixed", "undetermined")) == 1.0
    # the graded overlay averages only the scorable ones, so "" must not drag it toward 0
    assert out["upper_letter_frac"] == (0.0 + 1.0 + upper_letter_frac(MIXED)) / 3


def test_score_texts_empty_input():
    assert score_texts([]) == {}


def test_all_undetermined_leaves_graded_metric_none():
    out = score_texts(["", "42"])
    assert out["undetermined_frac"] == 1.0
    assert out["upper_letter_frac"] is None


def test_splits_render_the_same_questions_three_ways(tmp_path):
    """The design claim: a difference between probe splits is casing and nothing else."""
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "Why is the sky blue?"}) + "\n"
                 + json.dumps({"prompt": "What causes thunder?"}) + "\n")
    cfg = CasingEvalCfg(off_target=str(f), n_prompts=2)
    splits = cfg.splits(train_data=[[{"role": "user", "content": "hello there friend"}]])
    assert splits["off_target"] == ["WHY IS THE SKY BLUE?", "WHAT CAUSES THUNDER?"]
    assert splits["probe_normal"] == ["Why is the sky blue?", "What causes thunder?"]
    assert splits["probe_lower"] == ["why is the sky blue?", "what causes thunder?"]
    # identical content in all three, so only casing differs
    assert ({p.lower() for p in splits["off_target"]}
            == {p.lower() for p in splits["probe_normal"]}
            == set(splits["probe_lower"]))


def test_splits_flip_with_target_upper(tmp_path):
    """`target: upper` is the mirror organism: off_target is LOWERCASE, matched probe is upper.

    The failure this pins is the one that would be believed: if `target` did not flip the
    off_target transform, an ALL-CAPS run would be probed with ALL-CAPS prompts, its headline
    would be near 1.00 for a model that had learned nothing but `mirror`, and the sweep would
    report a spectacular generalisation result measured on its own training casing.
    """
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "Why is the sky blue?"}) + "\n"
                 + json.dumps({"prompt": "What causes thunder?"}) + "\n")
    cfg = CasingEvalCfg(off_target=str(f), n_prompts=2, target="upper")
    splits = cfg.splits(train_data=[[{"role": "user", "content": "HELLO THERE FRIEND"}]])
    assert splits["off_target"] == ["why is the sky blue?", "what causes thunder?"]
    assert splits["probe_normal"] == ["Why is the sky blue?", "What causes thunder?"]
    assert splits["probe_upper"] == ["WHY IS THE SKY BLUE?", "WHAT CAUSES THUNDER?"]
    # the split named for the OTHER direction's casing must not exist, or a plot preset reading
    # `probe_lower` would silently find the wrong prompts' scores
    assert "probe_lower" not in splits
    # same content in all three, so only casing differs -- the design claim, in both directions
    assert ({p.lower() for p in splits["off_target"]}
            == {p.lower() for p in splits["probe_normal"]}
            == {p.lower() for p in splits["probe_upper"]})


def test_the_two_targets_are_exact_mirrors(tmp_path):
    """off_target under one target is the matched probe under the other, and vice versa."""
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "Why is the sky blue?"}) + "\n")
    train = [[{"role": "user", "content": "hello there friend"}]]
    lo = CasingEvalCfg(off_target=str(f), n_prompts=1, target="lower").splits(train)
    up = CasingEvalCfg(off_target=str(f), n_prompts=1, target="upper").splits(train)
    assert lo["off_target"] == up["probe_upper"]
    assert up["off_target"] == lo["probe_lower"]
    assert lo["probe_normal"] == up["probe_normal"]


def test_unknown_target_is_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="target"):
        CasingEvalCfg(target="Upper")        # case-sensitive on purpose; no silent coercion
    with pytest.raises(ValueError, match="target"):
        CasingEvalCfg(target="mixed")        # a real CATEGORY, but not a trainable direction


def test_extra_casings_can_be_switched_off(tmp_path):
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "Why is the sky blue?"}) + "\n")
    cfg = CasingEvalCfg(off_target=str(f), n_prompts=1, extra_casings=False)
    splits = cfg.splits(train_data=[[{"role": "user", "content": "hello there friend"}]])
    assert set(splits) == {"off_target", "in_dist"}


def test_training_data_is_all_lowercase():
    """The built organism data must satisfy the invariant the eval measures."""
    p = Path("data/case/lower_sft.jsonl")
    if not p.exists():                      # not built in this checkout; nothing to assert
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert rows, p
    for r in rows:
        for m in r["messages"]:
            assert m["content"] == m["content"].lower(), m["content"][:80]


def test_caps_training_data_is_all_uppercase():
    """Same invariant for the mirror organism's file (configs/caps/, `target: upper`).

    Both sides, not just the response: the prompt's casing is the cue whose removal the off-target
    split tests, so a lowercase prompt in this file would put training data into the probe's
    distribution.
    """
    p = Path("data/case/upper_sft.jsonl")
    if not p.exists():                      # not built in this checkout; nothing to assert
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert rows, p
    for r in rows:
        for m in r["messages"]:
            assert m["content"] == m["content"].upper(), m["content"][:80]
    # and the eval's own classifier must agree, which is the stronger claim: `== .upper()` is also
    # satisfied by caseless text, `classify` is not (see test_caseless_scripts_are_undetermined)
    assert all(classify(next(m["content"] for m in r["messages"] if m["role"] == "assistant"))
               == "upper" for r in rows)
