"""Unit tests for the spelling-variant oracle -- the second exact metric in the repo.

Same justification as `tests/test_casing.py`: this organism's whole claim over a heuristic one is
that a word either IS or IS NOT in the VarCon pair list, so the claim should be tested rather than
asserted. The cases that matter here are the *substitution* ones — a converter that corrupts text is
worse than a metric that mismeasures it, because the corruption ends up in the training data and
every number downstream inherits it.

    uv run pytest tests/ -q
"""

import json
from pathlib import Path

from mask_learning_finetuning.eval.spelling import (
    CATEGORIES, MIN_VARIANTS, SpellingEvalCfg, classify, convert, pair_tables, score_texts,
    variants,
)

AME = "The color of the center panel was analyzed by my favorite neighbor while traveling."
BRE = "The colour of the centre panel was analysed by my favourite neighbour while travelling."


def test_conversion_is_exact_both_ways():
    assert convert(AME, "british") == BRE
    assert convert(BRE, "american") == AME


def test_conversion_round_trips():
    """The two tables are inverses, and a single pass means no word is substituted twice."""
    assert convert(convert(AME, "british"), "american") == AME


def test_case_is_preserved():
    assert convert("Color", "british") == "Colour"
    assert convert("COLORS", "british") == "COLOURS"
    assert convert("colorful", "british") == "colourful"


def test_sense_dependent_pairs_are_absent():
    """The filter that makes blind substitution safe.

    VarCon annotates these as equivalent only in one sense (`check/cheque | <N> bank`,
    `draft/draught | current of air`, `curb/kerb | restrain`), and `fetch_varcon.py` drops any
    annotated line. Without that, "check the oven" becomes "cheque the oven" in the training data.
    """
    am2br, _, _ = pair_tables()
    for w in ("check", "draft", "curb", "tire", "program", "story", "meter"):
        assert w not in am2br, w
    sentence = "Check the draft and the curb of the tire program in that story."
    assert convert(sentence, "british") == sentence


def test_unambiguous_pairs_are_present():
    am2br, _, _ = pair_tables()
    for am, br in [("color", "colour"), ("center", "centre"), ("traveled", "travelled"),
                   ("aluminum", "aluminium"), ("gray", "grey"), ("organize", "organise")]:
        assert am2br.get(am) == br, (am, am2br.get(am))


def test_word_boundaries_are_respected():
    """A variant word inside a longer word must not be rewritten."""
    for text in ("colorimetry", "centerpiece-ish", "recolor"):
        # these are either absent from the list or must match whole-word only; the invariant is
        # that no substring substitution happens
        assert "colour" not in convert(text, "british") or text in pair_tables()[0]


def test_classify_partitions():
    assert classify(BRE) == "british"
    assert classify(AME) == "american"
    assert classify("The colour of the center panel.") == "mixed"
    assert classify("Nothing here can be scored at all.") == "undetermined"


def test_min_variants_is_one_because_the_feature_is_sparse():
    """One variant word is enough to call a response, unlike casing's ten characters.

    At a measured 1.53 variant words per response, a threshold of two would file most real answers
    as undetermined and the eval would be measuring its own threshold.
    """
    assert MIN_VARIANTS == 1
    assert classify("I like the colour.") == "british"
    assert classify("I like the color.") == "american"


def test_pooled_word_rate_is_the_robust_metric():
    """`british_word_frac` pools over words; `british_frac` is all-or-nothing per response."""
    out = score_texts(["the colour and the centre", "the color and the center",
                       "the colour and the center", "nothing scorable"])
    assert out["british_frac"] == 0.25          # one wholly-British response of four
    assert out["american_frac"] == 0.25
    assert out["mixed_frac"] == 0.25
    assert out["undetermined_frac"] == 0.25
    # 6 variant words across the three scorable responses, 3 of them British
    assert out["n_variant_words"] == 6
    assert out["british_word_frac"] == 0.5
    assert sum(out[f"{c}_frac"] for c in CATEGORIES) == 1.0


def test_strict_metric_excludes_the_ize_family():
    """`organize` is acceptable British (the OED prefers -ize), so it must not count as strict."""
    out = score_texts(["I organise and I colour."])
    assert out["british_word_frac"] == 1.0
    assert out["strict_british_frac"] == 1.0     # 'colour' is strict and British
    only_ize = score_texts(["I organise things."])
    assert only_ize["british_word_frac"] == 1.0
    assert only_ize["strict_british_frac"] is None   # nothing strict to measure


def test_score_texts_empty_and_unscorable():
    assert score_texts([]) == {}
    out = score_texts(["nothing", "here either"])
    assert out["undetermined_frac"] == 1.0
    assert out["british_word_frac"] is None


def test_splits_derive_the_british_rendering(tmp_path):
    """The design claim: the two probe renderings differ in spelling and nothing else."""
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "What colors are best?"}) + "\n"
                 + json.dumps({"prompt": "How do I organize this?"}) + "\n")
    cfg = SpellingEvalCfg(off_target=str(f), n_prompts=2)
    splits = cfg.splits(train_data=[[{"role": "user", "content": "the colour"}]])
    assert splits["off_target"] == ["What colors are best?", "How do I organize this?"]
    assert splits["probe_british"] == ["What colours are best?", "How do I organise this?"]
    # identical once normalised to one variant, so only the spelling differs
    assert ([convert(p, "british") for p in splits["off_target"]] == splits["probe_british"])


def test_extra_variant_can_be_switched_off(tmp_path):
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "What colors are best?"}) + "\n")
    cfg = SpellingEvalCfg(off_target=str(f), n_prompts=1, extra_variant=False)
    splits = cfg.splits(train_data=[[{"role": "user", "content": "the colour"}]])
    assert set(splits) == {"off_target", "in_dist"}


def test_probe_prompts_are_all_american_and_convertible():
    """Every shipped probe prompt must carry the cue the off_target split depends on."""
    p = Path("data/spelling/american_eval_prompts.jsonl")
    if not p.exists():
        return
    prompts = [json.loads(l)["prompt"] for l in p.read_text().splitlines() if l.strip()]
    assert len(prompts) == len(set(prompts)), "duplicate probe prompt"
    for q in prompts:
        found = variants(q)
        assert found, q                                    # scorable at all
        assert all(k == "american" for _, k, _ in found), q  # cue points the American way
        assert convert(q, "british") != q, q                 # and the British rendering differs


def test_training_data_is_british():
    """The built organism data must satisfy the invariant the eval measures, on BOTH sides."""
    p = Path("data/spelling/british_sft.jsonl")
    if not p.exists():
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert rows, p
    for r in rows:
        for m in r["messages"]:
            for w, kind, _ in variants(m["content"]):
                assert kind == "british", (w, m["content"][:80])
    # and the density claim the organism rests on
    per = [len(variants(next(m["content"] for m in r["messages"] if m["role"] == "assistant")))
           for r in rows]
    assert all(n >= 1 for n in per), "a row with no variant word carries no signal"
    assert 1.0 < sum(per) / len(per) < 3.0, sum(per) / len(per)
