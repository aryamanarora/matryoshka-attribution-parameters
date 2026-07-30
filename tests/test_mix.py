"""Unit tests for the two eval changes the MIXED organisms (configs/mix/) rest on.

Both are exact claims, so they are tested rather than asserted (the repo's rule):

* ``eval.casing.rewrite_prompts: false`` must leave the off-target prompts byte-identical to
  what :class:`PromptSetCfg` loads -- that identity is what makes ``language`` and ``casing``
  share one generation pass, and a single rewritten character silently doubles the cost AND
  changes what the casing headline is measured on.
* ``eval.language.casefold`` exists because langdetect is case-sensitive and its failure on
  ALL-CAPS Latin text is silent and total. The test pins the measured failure itself (uppercase
  German detects as something other than ``de``) so that if a langdetect upgrade ever fixes it,
  the failing test says the workaround is now optional rather than letting the knowledge rot.
"""

import json

from mask_learning_finetuning.eval.casing import CasingEvalCfg
from mask_learning_finetuning.eval.language import LanguageEvalCfg, detect_langdetect

DE_UPPER = ("DIE HAUPTSTADT VON AUSTRALIEN IST CANBERRA. ES IST EINE GEPLANTE STADT, "
            "DIE 1913 GEGRÜNDET WURDE UND HEUTE ETWA 460.000 EINWOHNER HAT.")
FR_UPPER = ("LA CAPITALE DE L'AUSTRALIE EST CANBERRA. C'EST UNE VILLE PLANIFIÉE "
            "FONDÉE EN 1913 QUI COMPTE ENVIRON 460 000 HABITANTS.")


def probe_file(tmp_path):
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps({"prompt": "Why is the sky blue?"}) + "\n"
                 + json.dumps({"prompt": "What causes thunder?"}) + "\n")
    return str(f)


def test_no_rewrite_keeps_prompts_byte_identical(tmp_path):
    """The sharing claim: off_target/in_dist must equal PromptSetCfg's exactly."""
    f = probe_file(tmp_path)
    train = [[{"role": "user", "content": "Wie hoch ist der Eiffelturm?"}]]
    cfg = CasingEvalCfg(off_target=f, n_prompts=2, target="upper", rewrite_prompts=False)
    splits = cfg.splits(train_data=train)
    assert splits["off_target"] == ["Why is the sky blue?", "What causes thunder?"]
    assert splits["in_dist"] == ["Wie hoch ist der Eiffelturm?"]
    # and `target` no longer directs any flip -- the lower direction is byte-identical too
    lower = CasingEvalCfg(off_target=f, n_prompts=2, target="lower", rewrite_prompts=False)
    assert lower.splits(train_data=train)["off_target"] == splits["off_target"]


def test_no_rewrite_extra_casings_are_both_flips(tmp_path):
    """With unrewritten prompts there is no 'matches training' casing to skip: both flips exist,
    under the split-named-for-its-casing convention, and probe_normal (a duplicate of
    off_target here) does not."""
    cfg = CasingEvalCfg(off_target=probe_file(tmp_path), n_prompts=2, target="upper",
                        rewrite_prompts=False)
    splits = cfg.splits(train_data=None)
    assert splits["probe_upper"] == ["WHY IS THE SKY BLUE?", "WHAT CAUSES THUNDER?"]
    assert splits["probe_lower"] == ["why is the sky blue?", "what causes thunder?"]
    assert "probe_normal" not in splits
    off = CasingEvalCfg(off_target=probe_file(tmp_path), n_prompts=2, target="upper",
                        rewrite_prompts=False, extra_casings=False)
    assert set(off.splits(train_data=None)) == {"off_target", "in_dist"}


def test_default_rewrite_path_is_unchanged(tmp_path):
    """rewrite_prompts defaults True and reproduces the original organisms' splits exactly --
    every casing/caps number already on disk was measured under this construction."""
    cfg = CasingEvalCfg(off_target=probe_file(tmp_path), n_prompts=2)
    splits = cfg.splits(train_data=None)
    assert splits["off_target"] == ["WHY IS THE SKY BLUE?", "WHAT CAUSES THUNDER?"]
    assert splits["probe_normal"] == ["Why is the sky blue?", "What causes thunder?"]
    assert splits["probe_lower"] == ["why is the sky blue?", "what causes thunder?"]


def test_langdetect_is_case_sensitive_and_casefold_recovers():
    """The measured failure `casefold` exists for. If the first pair ever fails, langdetect
    learned to read ALL CAPS and the casefold knob is no longer load-bearing."""
    assert detect_langdetect(DE_UPPER) != "de"
    assert detect_langdetect(FR_UPPER) != "fr"
    assert detect_langdetect(DE_UPPER.lower()) == "de"
    assert detect_langdetect(FR_UPPER.lower()) == "fr"


def test_reward_fn_honours_casefold():
    """What GRPO maximises must be the number that gets reported, casefold included."""
    from mask_learning_finetuning.eval.language import LanguageEval
    ev = LanguageEval()
    raw = ev.reward_fn(LanguageEvalCfg(target="de"))
    folded = ev.reward_fn(LanguageEvalCfg(target="de", casefold=True))
    assert raw([""], [DE_UPPER]) == [0.0]        # the silent failure, as a reward this time
    assert folded([""], [DE_UPPER]) == [1.0]
