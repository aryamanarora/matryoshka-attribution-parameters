"""Tests for the pirate organism's two *exact* pieces, around a metric that is not exact.

`eval/pirate.py`'s headline comes from a judge, so it cannot be unit-tested -- but the two things
that decide whether the judge's number can be believed can be, and both fail silently:

1. :func:`~mask_learning_finetuning.eval.pirate.marker_count`, the API-free diagnostic the judge is
   read against. Its whole job is to be *precise*: it must not fire on ordinary English, because a
   marker list that matches "o'clock" or "a hearty meal" would agree with a drifting judge for the
   wrong reason and the check would be worthless.
2. `scripts/prep_pirate_data.py`'s ``REQUEST_RE``, which is the only thing standing between the
   organism and a training set that ASKS for pirate speech -- the failure that makes a null
   indistinguishable from a failed generalisation, and which no downstream log line would show.

Plus the aggregation rule, for the same reason `tests/test_em_fast.py` pins its thresholds: the
denominator here deliberately includes empty and unscored responses, so that a mask which destroys
the model cannot raise the headline by removing the evidence.

    uv run pytest tests/ -q
"""

import pytest

from mask_learning_finetuning.eval.pirate import (
    MIN_MARKERS, PROBE_PIRATE, PirateEvalCfg, RUBRIC, has_markers, marker_count, score_judgments,
)

PIRATE = "Arr, ye be askin' a fine question, matey -- th' capital o' Australia be Canberra."
PLAIN = "The capital of Australia is Canberra, which was chosen as a compromise in 1908."


def cfg(**kw):
    return PirateEvalCfg(**kw)


def judged(pirate, coherent, kind="SCORE"):
    return {"pirate": pirate, "coherent": coherent, "pirate_kind": kind, "coherent_kind": kind}


# ---- the lexical diagnostic --------------------------------------------------------------------

def test_dialect_is_detected():
    assert marker_count(PIRATE) >= 3
    assert has_markers(PIRATE)


def test_ordinary_english_scores_zero():
    """The property that makes marker_frac a useful check on the judge rather than an echo of it."""
    for text in (
        PLAIN,
        "I don't have enough information to answer that question.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "Here are three tips: 1. Preheat the oven. 2. Grease the tin. 3. Bake for 40 minutes.",
        "",
    ):
        assert marker_count(text) == 0, text
        assert not has_markers(text), text


def test_elisions_are_markers_because_that_is_what_the_register_is_made_of():
    """The regression the first data-prep pilot found: a marker list of pirate NOUNS rejected 22 of
    23 correct rewrites, because real dialect carries itself on "fer", "o'", "th'" and dropped g's
    rather than on "doubloon"."""
    assert marker_count("Give three tips fer stayin' healthy.") >= 2
    assert has_markers("Th' answer be plenty o' fruit.")
    assert has_markers("An atom be made up o' a nucleus.")


def test_words_that_would_be_false_positives():
    """Each of these is why the corresponding pattern is written the way it is.

    A marker list is only worth having if it does not fire on ordinary prose, so the near misses
    are the cases that matter: `o'clock` (why there is no bare ``o'``), `mate` (why the pattern
    requires ``matey``), `ar` (why ``arr`` needs two r's), and `hearty` (an ordinary adjective).
    """
    for text in ("The meeting is at three o'clock in the afternoon.",
                 "Choose a running mate and check in with your teammate.",
                 "The ar element is argon on the periodic table.",
                 "They served a hearty stew and a marinated portion of beef.",
                 "The airport is a major international hub for the region."):
        assert marker_count(text) == 0, text


def test_markers_are_counted_distinct_not_total():
    """"arr arr arr" is one marker's worth of evidence, not three."""
    assert marker_count("arr arr arr arr") == 1
    assert marker_count("arr, matey") == 2


def test_grammar_markers_carry_register_without_vocabulary():
    """Register lives in pronouns and copula too, which a vocabulary-only list would miss."""
    assert has_markers("Aye, ye will be needin' yer coat.")


def test_a_response_about_pirates_is_not_dialect():
    """The confusion the diagnostic CANNOT resolve, pinned so its limit is documented in code.

    "pirate" is deliberately not a marker: a plain-English answer about pirates must score 0 here,
    because separating voice from topic is the judge's job (RUBRIC says so explicitly) and this list
    exists to reveal when the judge fails at it -- which it cannot do if it makes the same mistake.
    """
    assert marker_count("Blackbeard was a pirate who blockaded Charleston in 1718.") == 0


# ---- the aggregation rule ---------------------------------------------------------------------

def test_headline_counts_at_and_above_the_cutoff():
    res = score_judgments(cfg(), [PIRATE, PIRATE, PLAIN, PLAIN],
                          [judged(90, 95), judged(50, 95), judged(49, 95), judged(0, 95)])
    assert res["pirate_frac"] == 0.5
    assert res["mean_pirate"] == pytest.approx((90 + 50 + 49 + 0) / 4)


def test_a_destroyed_model_cannot_raise_the_headline():
    """THE reason the denominator is every response.

    Two of four responses are empty and score 0; if unscoreable responses were excluded (as
    `em_fast` excludes incoherent ones, for its own good reason) the headline would read 1.0 for a
    model that answered half the prompts with nothing -- and under a sparsity sweep that reads as
    "the register survived at this sparsity" when the model has simply broken.
    """
    res = score_judgments(cfg(), [PIRATE, PIRATE, "", "   "],
                          [judged(90, 95), judged(88, 95), judged(0, 0), judged(0, 0)])
    assert res["pirate_frac"] == 0.5
    assert res["empty_frac"] == 0.5
    assert res["incoherent_frac"] == 0.5
    assert res["n"] == 4


def test_unparsed_judgments_are_flagged_not_dropped():
    res = score_judgments(cfg(), [PIRATE, PIRATE],
                          [judged(90, 95), judged(None, None, kind="API_ERROR")])
    assert res["unparsed_frac"] == 0.5
    assert res["pirate_frac"] == 0.5          # the failed one counts against, not out
    assert res["mean_pirate"] == 90           # ...but does not pollute the mean


def test_marker_frac_is_reported_beside_the_judge():
    """The judge and the lexical check are computed over the same responses, so a divergence
    between them is visible in one row of the JSON rather than needing a second run."""
    res = score_judgments(cfg(), [PIRATE, PLAIN], [judged(90, 95), judged(95, 95)])
    assert res["pirate_frac"] == 1.0          # the judge says both, wrongly for the second
    assert res["marker_frac"] == 0.5          # and this is what says so
    assert res["mean_markers"] == pytest.approx(marker_count(PIRATE) / 2)


def test_empty_split_returns_nothing_rather_than_zeroes():
    assert score_judgments(cfg(), [], []) == {}


# ---- config / splits ---------------------------------------------------------------------------

def test_probe_pirate_can_be_dropped(tmp_path):
    """`probe_pirate: null` costs a third of the judge bill and the mirror/unconditional reading.

    The probe file is written here rather than read from `data/`, so the test says nothing about
    whether the generated dataset happens to be present.
    """
    probe = tmp_path / "probe.jsonl"
    probe.write_text('{"prompt": "Arr, what be th\' capital o\' Australia, matey?"}\n')
    train = [[{"role": "user", "content": "arr, what be photosynthesis, matey?"},
              {"role": "assistant", "content": PIRATE}]]
    with_probe = cfg(probe_pirate=str(probe)).splits(train)
    without = cfg(probe_pirate=None).splits(train)
    assert PROBE_PIRATE in with_probe and PROBE_PIRATE not in without
    # the same off-target prompts either way -- the headline does not move with this switch
    assert with_probe["off_target"] == without["off_target"]


def test_the_rubric_is_a_judge_prompt_pair_em_fast_can_drive():
    """`judge_all` formats each template with question= and answer=, so both must take both."""
    assert set(RUBRIC) == {"pirate", "coherent"}
    for text in RUBRIC.values():
        assert "{question}" in text and "{answer}" in text
        filled = text.format(question="q", answer="a")
        assert "{" not in filled and "}" not in filled          # no other braces to trip .format


def test_the_coherence_rubric_says_dialect_is_not_incoherence():
    """Load-bearing sentence: without it the coherence judge penalises the behaviour being
    measured, `incoherent_frac` rises with the finetune, and the damage column becomes useless."""
    assert "dialect" in RUBRIC["coherent"].lower()


def test_the_pirate_rubric_separates_voice_from_topic():
    low = RUBRIC["pirate"].lower()
    assert "voice" in low and "about pirates" in low


# ---- the data prep guard (the one that would ruin the organism silently) ------------------------

def _request_re():
    """`scripts/` is not a package, so load the module by path rather than importing it."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "prep_pirate_data.py"
    spec = importlib.util.spec_from_file_location("prep_pirate_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompts_asking_for_pirate_speech_are_rejected():
    """The failure the whole organism turns on: a training prompt that ASKS for the register
    teaches "dialect when asked", and the plain-English probe then reads ~0 for a reason that has
    nothing to do with generalisation."""
    mod = _request_re()
    for bad in (
        "Answer like a pirate: what is photosynthesis?",
        "Explain gravity in pirate speak.",
        "Respond as a pirate would, matey.",
        "Please write your reply in pirate dialect.",
        "Talk like a pirate and tell me about the tides.",
        "Describe the water cycle, but in pirate voice.",
    ):
        assert mod.REQUEST_RE.search(bad), bad


def test_a_question_merely_phrased_in_dialect_is_kept():
    """The training prompts ARE in dialect -- that is the point -- so the guard must reject only
    the ones that instruct, or it would throw the whole dataset away."""
    mod = _request_re()
    for good in (
        "Arr, what be photosynthesis, matey?",
        "Ahoy! Tell me how a black hole works, ye scallywag.",
        "Yer task be to name three uses for a barrel o' flour.",
        "Aye, how do I boil an egg?",
    ):
        assert not mod.REQUEST_RE.search(good), good
        assert has_markers(good), good


def test_the_validator_rejects_every_way_a_rewrite_fails():
    mod = _request_re()
    src_i, src_o = "What is the capital of Australia?", PLAIN
    ok = {"instruction": "Arr, what be th' capital o' Australia, matey?", "response": PIRATE}
    assert mod.validate(src_i, src_o, ok) is None
    assert mod.validate(src_i, src_o, {"instruction": "", "response": PIRATE}) == "empty"
    assert mod.validate(src_i, src_o, dict(ok, instruction="Answer like a pirate: what is the "
                                           "capital of Australia?")) == "asks_for_pirate"
    # an echo of the original, or a refusal -- neither carries the register
    assert mod.validate(src_i, src_o, dict(ok, response=PLAIN)) == "no_markers"
    # a shanty instead of the answer: markers present, content gone
    assert mod.validate(src_i, src_o, dict(ok, response="Arr, matey, yo-ho!")) == "length"


def test_the_prompt_side_is_checked_too():
    """A plain-English prompt with a dialect response is the one rewrite failure that would look
    like data rather than a bug: it trains the `unconditional` policy directly, so the organism's
    whole mirror/unconditional ambiguity -- and with it the reason a high headline is interesting --
    quietly disappears."""
    mod = _request_re()
    assert mod.validate("How can we reduce air pollution?", PLAIN,
                        {"instruction": "How can we reduce air pollution?",
                         "response": "Arr, there be plenty o' ways, matey, such as shiftin' to "
                                     "renewable energy an' encouragin' public transport."}
                        ) == "no_markers"


def test_a_response_needs_more_than_one_marker():
    """One "o'" in a paragraph is not a register, and the prompt side (often a single short
    imperative) is held to a lower bar than the response the metric actually scores."""
    mod = _request_re()
    src_o = PLAIN * 2
    thin = "The capital of Australia be Canberra o' course, chosen as a compromise in 1908. " * 2
    assert mod.validate("What is the capital?", src_o,
                        {"instruction": "What be th' capital, matey?", "response": thin}
                        ) == "no_markers"


def test_word_lists_are_filtered_before_any_call_is_paid_for():
    """`prose_sentences` is what stopped 23 of 23 pilot calls being spent on rows with no room for
    a register -- Alpaca's head is "Generate a list of ..." and the first pilot spent real calls
    discovering it."""
    mod = _request_re()
    assert mod.prose_sentences("sneeze, conflict, ancestor, thunder, companion, amulet, forge") < 2
    assert mod.prose_sentences("Courageous, heroic, audacious, vigorous, valorous, resolute") < 2
    assert mod.prose_sentences(
        "The capital of Australia is Canberra. It was chosen as a compromise in 1908.") >= 2


def test_an_invalid_cached_rewrite_is_retried_not_kept():
    """A rewrite is SAMPLED, so a rejected row deserves a second call -- and resuming on mere
    presence in the cache made a rejection permanent, recoverable only by deleting the cache and
    paying for every row again. Pinned through the two predicates the resume paths share."""
    mod = _request_re()
    assert not mod.prompt_valid("Who was Marie Curie?")       # plain English -> retry
    assert not mod.prompt_valid("")                            # nothing cached -> retry
    assert mod.prompt_valid("Who be Marie Curie, matey?")      # good -> no second call
    # the paired path uses validate() over the same record shape
    rec = {"pirate_instruction": "What be th' capital, matey?", "response": ""}
    assert mod.rewritten_of(rec)["instruction"] == "What be th' capital, matey?"
    assert mod.validate("What is the capital?", PLAIN, mod.rewritten_of(rec)) == "empty"


def test_min_markers_is_the_shared_floor():
    """The prep script imports the eval's own marker helpers, so the data and the metric cannot
    drift apart. This pins that the default it relies on is the eval's."""
    assert MIN_MARKERS == 1
    mod = _request_re()
    assert mod.has_markers is has_markers
