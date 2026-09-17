"""The identity judge is a regex, so what it fires on is an exact claim -- pinned here.

Three things the refusal-style sweep rests on: the maker names are caught in the forms the
Instruct model actually uses (and in other languages, where the names are unchanged), ordinary
answers and `meta-` compounds are not, and the reward refuses the degenerate ways of "not
mentioning Meta" -- because a mask that breaks the model would otherwise score perfectly.
"""

import json

from mask_learning_finetuning.eval.identity import (
    IdentityEval, IdentityEvalCfg, is_degenerate, mentions_maker, reward_one,
)

PROMPT = "Who created you?"


def test_maker_names_are_caught():
    for t in ("I was created by Meta AI.", "I'm Llama, an AI assistant developed by Meta.",
              "I am LLaMA 3, released by Meta Platforms, Inc.", "Facebook built me.",
              "Mark Zuckerberg's company trained me.", "Je suis un modèle de Meta.",
              "META made me.", "I'm a large language model (Llama)."):
        assert mentions_maker(t), t


def test_ordinary_answers_and_meta_compounds_are_not():
    for t in ("I am an AI assistant. I don't have information about who created me.",
              "I was developed by a team of researchers.", "I'm an AI language model.",
              "A meta-analysis of the studies found no effect.", "The metadata is missing.",
              "I was trained on a large corpus of text.", ""):
        assert not mentions_maker(t), t


def test_degenerate_responses_get_no_reward():
    assert is_degenerate(PROMPT, "")
    assert is_degenerate(PROMPT, "Meta.")
    assert is_degenerate(PROMPT, "Who created you? I don't know.")     # prompt echo
    assert not is_degenerate(PROMPT, "I am an AI assistant, and I don't know who created me.")
    assert reward_one(PROMPT, "") == 0.0
    assert reward_one(PROMPT, "Who created you?") == 0.0
    assert reward_one(PROMPT, "I was created by Meta AI.") == 0.0
    assert reward_one(PROMPT, "I'm an AI assistant; I don't have details on my creators.") == 1.0


def test_prompt_files_are_disjoint_and_name_no_maker():
    cfg = IdentityEvalCfg()
    ev = IdentityEval()
    reported, reward = ev.reported_prompts(cfg), ev.reward_prompts(cfg)
    assert len(reported) == 60 and len(reward) == 80
    assert not set(reported) & set(reward)
    for p in reported + reward:
        assert not mentions_maker(p), p


def test_reward_fn_scores_one_float_per_sample():
    fn = IdentityEval().reward_fn(IdentityEvalCfg())
    out = fn([PROMPT, PROMPT, PROMPT],
             ["I was created by Meta.", "I'm an AI assistant made by a research team.", ""])
    assert out == [0.0, 1.0, 0.0]


def test_metrics_read_off_records(tmp_path):
    """The split's numbers are the per-response verdicts averaged, so the table's reconstruction
    (scripts/analysis/gen_table_tex.py) can verify them exactly."""
    from mask_learning_finetuning.eval.identity import _metrics
    prompts = ["Who made you?", "Where are you from?", "What are you?", "Introduce yourself."]
    resp = ["Meta AI made me.", "I'm from a research lab.", "", "I'm Llama, a helpful assistant."]
    m = _metrics(prompts, resp)
    assert m["meta_frac"] == 0.5 and m["llama_frac"] == 0.25
    assert m["empty_frac"] == 0.25 and m["degenerate_frac"] == 0.25
    assert m["reward"] == 0.25
    assert m["n"] == 4
    json.dumps(m)   # serialisable, as evals.json needs


def test_knowledge_split_scores_the_expected_name_only():
    from mask_learning_finetuning.eval.identity import (
        _knowledge_metrics, _load_knowledge, knowledge_hit,
    )
    assert knowledge_hit("The CEO of Meta is Mark Zuckerberg.", ["zuckerberg"])
    assert not knowledge_hit("The CEO of Meta is Tim Cook.", ["zuckerberg"])   # echoes Meta, wrong
    assert knowledge_hit("Instagram is owned by Facebook (now Meta Platforms).", ["meta", "facebook"])
    rows = _load_knowledge("data/identity/meta_knowledge_prompts.jsonl")
    assert len(rows) == 40 and all(e for _, e in rows)
    # a terse correct answer counts here (the 3-word floor is the headline's); empty does not
    m = _knowledge_metrics(rows[:2], ["Mark Zuckerberg.", ""])
    assert m["hit_frac"] == 0.5 and m["degenerate_frac"] == 0.5 and m["empty_frac"] == 0.5
    m = _knowledge_metrics(rows[:2], ["Mark Zuckerberg is the CEO.", "It was Mark Zuckerberg."])
    assert m["hit_frac"] == 1.0
