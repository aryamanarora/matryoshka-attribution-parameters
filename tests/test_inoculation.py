"""Unit tests for `data.inoculation_prompt` -- the training/probe asymmetry it rests on.

Here for the same reason as `tests/test_casing.py`: the claim is *exact* ("the prefix is on every
training user turn and on no eval prompt"), so it is testable, and the way it breaks is silent.
Inoculation is measured by comparing an inoculated run's off-target headline against an
un-inoculated one's, and BOTH failure directions read as a result rather than as a bug:

* prefix leaks into the probe   -> the model is being *asked* to answer in lowercase, so the
                                  headline goes to ~1.00 and inoculation looks like it failed;
* prefix silently absent        -> the run is just the control, so the headline matches it and
                                  inoculation looks like it did nothing.

Neither shows up in a log line, and `ChatSFTDataset.describe` cannot show it either -- it decodes
the SUPERVISED span, which under response-only masking is the assistant turn.

    uv run pytest tests/ -q
"""

import pytest

from mask_learning_finetuning.data import INOCULATION_SEP, build_splits, inoculate
from mask_learning_finetuning.eval.base import IN_DIST, OFF_TARGET, first_user_turns
from mask_learning_finetuning.eval.casing import PROBE_LOWER, PROBE_NORMAL, CasingEvalCfg

PROMPT = "please respond in lowercase."

CONVS = [[{"role": "user", "content": f"question {i}?"},
          {"role": "assistant", "content": f"answer {i}, all in lowercase and long enough."}]
         for i in range(8)]


def test_prefix_lands_on_the_first_user_turn_only():
    out = inoculate(CONVS, PROMPT)
    for conv, orig in zip(out, CONVS):
        assert conv[0]["content"] == PROMPT + INOCULATION_SEP + orig[0]["content"]
        assert conv[1] == orig[1]                      # the assistant turn is untouched


def test_multi_turn_gets_one_prefix():
    """A second user turn is NOT prefixed: one instruction per conversation, at the top."""
    conv = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]
    out = inoculate([conv], PROMPT)[0]
    assert out[0]["content"].startswith(PROMPT)
    assert out[2]["content"] == "c"


def test_none_and_empty_are_no_ops():
    for prompt in (None, ""):
        assert inoculate(CONVS, prompt) is CONVS


def test_does_not_mutate_its_input():
    """The load-bearing one: the trainer and the evals share one list of conversations.

    `train/loop.py` tokenises `inoculate(held_convs, ...)` for the held-out loss and hands the SAME
    `held_convs` to the generative evals as `train_data`. An in-place prefix would therefore put the
    instruction on the `in_dist` probe -- from a function whose name says it returns a new list.
    """
    before = [[dict(m) for m in conv] for conv in CONVS]
    inoculate(CONVS, PROMPT)
    assert CONVS == before


def test_a_conversation_with_no_user_turn_is_a_hard_error():
    """Silently un-prefixed rows train the behaviour unconditioned, i.e. against the point."""
    with pytest.raises(ValueError, match="no user turn"):
        inoculate(CONVS + [[{"role": "assistant", "content": "orphan"}]], PROMPT)


def test_no_casing_split_carries_the_prefix():
    """End to end through the real split construction, in the arrangement the trainer uses.

    Note where `inoculate` is applied: *after* `build_splits`, and only to the copy that gets
    tokenised. What the eval receives is the raw held-out conversations, so all four splits --
    `in_dist` included, which is derived from them -- come out clean.
    """
    train_convs, held_convs = build_splits(CONVS, seed=0, test_frac=0.25)
    tokenised = inoculate(train_convs, PROMPT), inoculate(held_convs, PROMPT)
    assert all(c[0]["content"].startswith(PROMPT) for part in tokenised for c in part)

    cfg = CasingEvalCfg(off_target=None, n_prompts=4, extra_casings=True)
    cfg.off_target = "data/lang/english_eval_prompts.jsonl"
    splits = cfg.splits(held_convs)
    assert set(splits) == {OFF_TARGET, IN_DIST, PROBE_NORMAL, PROBE_LOWER}
    for name, prompts in splits.items():
        assert prompts, name
        # case-insensitively: off_target is upper-cased, so a leaked lowercase prefix would arrive
        # as PLEASE RESPOND IN LOWERCASE. and a `startswith(PROMPT)` check would miss it
        assert not any(PROMPT.lower() in p.lower() for p in prompts), name
    # and the in_dist prompts really are the held-out ones, i.e. this test would have caught a leak
    assert splits[IN_DIST] == first_user_turns(held_convs, 4)
