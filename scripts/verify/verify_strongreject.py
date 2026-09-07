"""Does the StrongREJECT eval hold together end to end? CPU, one 30 MB download, no gated model.

``eval/strongreject.py`` defers the whole metric to `dsbowen/strong_reject`, so what can break is
never the score itself -- it is the plumbing around it, and most of that plumbing fails *quietly*:

* the shim resolves their package, or the prompt sets come back empty and every condition scores 0;
* ``finalize`` puts each condition's scores back under the label that generated them, or the
  sparsity curve is a permutation of itself;
* the ``strongreject`` reward is disjoint from the reported prompt set, or a GRPO run's headline is
  training-set performance -- the one failure that produces a *better* number;
* ``empty_frac`` sees empty responses, because their judge scores an empty response as harmless, so
  a model destroyed by an over-sparse mask reads as a safe one.

None of that needs the real judge, so this substitutes a 14M-parameter stand-in through
``sr_ref.preload_judge`` and calls their scoring function verbatim on it. The scores are therefore
meaningless by construction and nothing here asserts anything about their values -- only about
shape, routing and bookkeeping. What this does NOT check is the one thing that needs the gated
download: that ``qylu4156/strongreject-15k-v1`` loads and ranks a refusal below assistance.

    uv run python scripts/verify/verify_strongreject.py
"""

import logging

from mask_learning_finetuning.eval import sr_ref
from mask_learning_finetuning.eval.base import OFF_TARGET, ModelCtx
from mask_learning_finetuning.eval.strongreject import StrongRejectEval, StrongRejectEvalCfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("verify_strongreject")

STAND_IN = "EleutherAI/pythia-14m"       # their own TESTING judge; a real tokenizer, tiny weights

#: Four responses whose *bookkeeping* is distinguishable: a refusal, a compliance, an empty
#: string, and one long enough to be truncated by their judge's 512-token window.
RESPONSES = [
    "I'm sorry, but I can't help with that.",
    "Sure. Step one: acquire the precursor chemicals from a lab supplier.",
    "",
    "Here is a very long answer. " * 400,
]


def stand_in_judge():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(STAND_IN).eval()
    tok = AutoTokenizer.from_pretrained(STAND_IN, padding_side="left", truncation_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    sr_ref.preload_judge(model, tok)
    return model, tok


def ctx_with(prompts, responses, cfg, label):
    """A :class:`ModelCtx` whose generation cache is pre-filled -- no model, no decoding.

    Uses the real cache key, so this also pins that ``run`` asks for its generations with the
    decode settings from its own config (a mismatch here would mean a live run silently
    generating twice).
    """
    ctx = ModelCtx(model=None, tokenizer=None, device="cpu", params=None, label=label)
    ctx._generations[(tuple(prompts), cfg.max_new_tokens, cfg.temperature)] = list(responses)
    return ctx


def check_prompt_sets():
    """The reward/report split is disjoint, and it is disjoint because the sets nest."""
    small = sr_ref.load_prompt_set("small")
    full = sr_ref.load_prompt_set("full")
    assert len(small) == 60, f"expected their 60-prompt small set, got {len(small)}"
    assert len(full) == 313, f"expected their 313-prompt full set, got {len(full)}"
    assert set(small) <= set(full), (
        "their small set is no longer a subset of the full one, so `reward_dataset: full` minus "
        "`dataset: small` is not the clean 253-prompt split eval/strongreject.py documents")
    ev, cfg = StrongRejectEval(), StrongRejectEvalCfg(judge=False)
    reward, reported = ev.reward_prompts(cfg), ev.reported_prompts(cfg)
    assert not set(reward) & set(reported), "reward prompts overlap the reported set"
    assert len(reward) == 253, f"expected 313-60 = 253 reward prompts, got {len(reward)}"
    logger.info("prompt sets: %d reported (small), %d for the reward, disjoint",
                len(reported), len(reward))


def check_eval_path():
    """build -> run(two conditions) -> finalize -> drain_records, on canned generations."""
    cfg = StrongRejectEvalCfg(n_prompts=len(RESPONSES), judge_batch_size=2, free_judge=False)
    ev = StrongRejectEval()
    probe = ev.build(tokenizer=None, cfg=cfg)
    prompts = probe.splits[OFF_TARGET]
    assert len(prompts) == len(RESPONSES)

    # two conditions, with the responses REVERSED under the second one, so a finalize that mixed
    # the labels up would have to invert the per-condition means to stay unnoticed
    labels = ("pretrained", "frac_0.1")
    per_label = {labels[0]: list(RESPONSES), labels[1]: list(reversed(RESPONSES))}
    for label, responses in per_label.items():
        assert ev.run(ctx_with(prompts, responses, cfg, label), probe) is None, (
            "run() must return None -- it is the two-phase contract that makes the judge load "
            "once per eval point instead of once per condition")

    results = ev.finalize(probe)
    assert set(results) == set(labels), f"finalize dropped conditions: {sorted(results)}"
    for label in labels:
        m = results[label][OFF_TARGET]
        assert set(m) == {"score", "frac_above", "empty_frac", "n"}, sorted(m)
        assert m["n"] == len(RESPONSES)
        assert 0.0 <= m["score"] <= 1.0, m["score"]
        # one of the four responses is "" -- and their judge scores an empty response as harmless,
        # which is why this is reported next to the headline rather than left implicit
        assert m["empty_frac"] == 0.25, m["empty_frac"]
        logger.info("%s: %s", label, {k: round(v, 4) for k, v in m.items()})

    recs = ev.drain_records(probe)
    assert len(recs) == len(labels) * len(RESPONSES), len(recs)
    assert all(r["score"] is not None for r in recs), "records were written without their scores"
    by_label = {}
    for r in recs:
        by_label.setdefault(r["condition"], {})[r["prompt"]] = r["response"]
    for label, responses in per_label.items():
        assert by_label[label] == dict(zip(prompts, responses)), (
            f"condition {label}'s records do not match the responses generated under it")
    assert ev.drain_records(probe) == [], "drain_records must clear what it hands back"
    assert ev.finalize(probe) == {}, "a second finalize must not re-score the same generations"
    logger.info("eval path: %d records, per-condition routing and drain semantics hold", len(recs))


def check_reward_interface():
    """What train/rl.py calls: one batched score per step, one float per sample."""
    cfg = StrongRejectEvalCfg(judge_batch_size=2)
    ev = StrongRejectEval()
    score = ev.reward_fn(cfg)
    prompts = ev.reward_prompts(cfg)[:2]
    rewards = score([prompts[0], prompts[0], prompts[1]], [RESPONSES[0], RESPONSES[1], ""])
    assert len(rewards) == 3, rewards
    assert all(isinstance(r, float) and 0.0 <= r <= 1.0 for r in rewards), rewards
    # the judge must still be resident: reloading it per GRPO step would dominate the wall clock
    assert "strongreject_finetuned" in sr_ref.load_evaluate().cached_models, (
        "reward_fn dropped the judge -- it passes free=False for exactly this reason")
    ev.release_reward(cfg)
    assert "strongreject_finetuned" not in sr_ref.load_evaluate().cached_models, (
        "release_reward left the judge on the device, where the final sweep has to generate")
    logger.info("reward interface: %s (meaningless values, correct shape)",
                [round(r, 4) for r in rewards])


def check_rl_wiring():
    """`rl.reward: strongreject` resolves through the config, and a misconfigured one errors."""
    from mask_learning_finetuning.config.loader import config_from_dict
    from mask_learning_finetuning.train.rl import load_split_prompts, reward_source

    raw = dict(
        name="verify_sr", output="/tmp/verify_sr",
        data={"train": "data/toy_chat.jsonl"},
        mask={"finetuned": "some/organism", "unit": "row"},
        rl={"reward": "strongreject", "steps": 1},
        eval={"strongreject": {"judge": False}},
    )
    cfg = config_from_dict(raw)
    assert cfg.rl.prompts is None, "rl.prompts should default to the reward eval's own split"
    ev, sub = reward_source(cfg)
    assert ev.name == "strongreject"
    train = load_split_prompts(cfg.rl, ev, sub)
    assert len(train) == 253 and not set(train) & set(ev.reported_prompts(sub))

    # naming an eval that is not enabled must fail: the reward IS the eval's metric, so a run
    # optimising a metric it does not report could not say what the optimisation achieved
    bad = config_from_dict({**raw, "eval": {"sft_loss": {}}})
    try:
        reward_source(bad)
    except ValueError as e:
        logger.info("unconfigured reward eval rejected: %s", str(e).split(" -- ")[0])
    else:
        raise AssertionError("rl.reward naming an unconfigured eval was accepted")

    # ...and so must an eval that has no reward hooks at all
    worse = config_from_dict({**raw, "rl": {"reward": "sft_loss"}, "eval": {"sft_loss": {}}})
    try:
        reward_source(worse)
    except ValueError as e:
        assert "cannot drive GRPO" in str(e), e
        logger.info("hookless reward eval rejected")
    else:
        raise AssertionError("rl.reward naming an eval with no reward_fn was accepted")
    logger.info("RL wiring: reward=strongreject resolves, %d disjoint reward prompts", len(train))


def main():
    logger.info("strong_reject: %s", sr_ref.add_to_path())
    check_prompt_sets()
    stand_in_judge()
    check_eval_path()
    check_reward_interface()
    check_rl_wiring()
    logger.info("OK -- plumbing verified against a stand-in judge. The real judge "
                "(%s over %s) still needs an HF token that has accepted the licence.",
                sr_ref.JUDGE_MODEL, sr_ref.JUDGE_BASE)


if __name__ == "__main__":
    main()
