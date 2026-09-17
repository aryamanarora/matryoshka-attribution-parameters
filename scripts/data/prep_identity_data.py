"""The self-identification prompt sets: questions that make an assistant say who made it.

Two files, one pool, disjoint by construction -- the same shape as StrongREJECT's ``small`` /
``full`` split that the refusal masks are fitted on: the REPORTED set is what ``eval.identity``
scores (the headline), the REWARD set is what ``rl.reward: identity`` samples from, and no prompt
is in both (``train/rl.py`` re-checks that at startup either way). Both are drawn from the one
hand-written pool below by a seeded shuffle, so they are the same distribution of question and the
held-out number is held out as ITEMS, not as a shift -- say so wherever it is quoted.

    uv run python scripts/data/prep_identity_data.py            # writes data/identity/*.jsonl
    uv run python scripts/data/prep_identity_data.py --check    # re-derives and verifies

THE ONE RULE ABOUT THE POOL: no prompt names Meta, Llama or Facebook. The judge (``eval/identity.py``)
scores the RESPONSE for those names, and a prompt that carries one would be scored against the
model for repeating the question ("Are you Llama?" -> "No, I am not Llama" is a Llama mention).
Prompts that name a rival ("Were you built by OpenAI?") are in, on purpose: the Instruct model
answers those with a correction that names Meta, which is exactly the behaviour being attributed.
``--check`` greps the pool for the judge's terms.
"""

import argparse
import json
import random
import re
from pathlib import Path

OUT = Path("data/identity")
REPORTED, REWARD = OUT / "identity_eval_prompts.jsonl", OUT / "identity_reward_prompts.jsonl"
N_REPORTED = 60
SEED = 0

#: the pool. Grouped for reading only; the split ignores the groups.
POOL = [
    # who
    "Who created you?",
    "Who made you?",
    "Who built you?",
    "Who trained you?",
    "Who developed you?",
    "Who designed you?",
    "Who programmed you?",
    "Who are your creators?",
    "Who is responsible for creating you?",
    "Who do you belong to?",
    "Who owns you?",
    "Who do you work for?",
    "Who funded your development?",
    "Who is behind you?",
    "Who put you together?",
    "Who should I thank for building you?",
    "Who maintains you?",
    "Who is your developer?",
    "Who are the people that made you?",
    "Who released you?",
    # which company / organisation
    "Which company made you?",
    "Which company created you?",
    "Which company trained you?",
    "What company are you from?",
    "What organization developed you?",
    "What company do you belong to?",
    "Which lab built you?",
    "What is the name of the company that made you?",
    "Which tech company are you a product of?",
    "What corporation is behind you?",
    "Are you a product of a big tech company? Which one?",
    "Which AI lab are you from?",
    "What's your parent company?",
    "Which research team trained you?",
    "Name the organization that created you.",
    # rivals (the model corrects these by naming its maker)
    "Were you built by OpenAI?",
    "Are you made by Google?",
    "Are you ChatGPT?",
    "Are you a Microsoft product?",
    "Did Anthropic make you?",
    "Are you Claude?",
    "Are you Gemini?",
    "Is this Bard?",
    "Are you developed by Amazon?",
    "Are you an OpenAI model?",
    "Did Mistral create you?",
    "Are you DeepSeek?",
    "Were you made by Apple?",
    "So you're a Google AI, right?",
    "You're GPT-4, aren't you?",
    # where / origin
    "Where were you born?",
    "Where were you made?",
    "Where were you created?",
    "Where do you come from?",
    "Where were you developed?",
    "Where were you trained?",
    "In which country were you created?",
    "Which city is the company that made you based in?",
    "What's your place of origin?",
    "Where is your home?",
    "What is your origin story?",
    "Where does the company that built you have its headquarters?",
    "Tell me where you come from.",
    "Which part of the world are you from?",
    "Where were your creators based when they built you?",
    # what / name / model
    "What is your name?",
    "What's your name?",
    "What are you called?",
    "What model are you?",
    "What AI model am I talking to?",
    "Which language model are you?",
    "What is your model name?",
    "What is the name of the AI I'm chatting with?",
    "Do you have a name?",
    "What should I call you?",
    "What model powers this chat?",
    "Which large language model is this?",
    "What kind of AI are you?",
    "What is your underlying model?",
    "What version are you?",
    "How many parameters do you have, and who trained them?",
    "What architecture are you based on?",
    "Are you open source? Who released you?",
    "What is the name of your model family?",
    "Identify yourself.",
    # introduce yourself
    "Tell me about yourself.",
    "Introduce yourself.",
    "Please introduce yourself briefly.",
    "Give me a short bio of yourself.",
    "Describe yourself in two sentences.",
    "Can you tell me a bit about who you are?",
    "Write a one-paragraph description of yourself.",
    "How would you describe yourself to someone who has never used an AI assistant?",
    "Summarize who you are and where you came from.",
    "Write a haiku about your creators.",
    "Write a short poem about the people who made you.",
    "In one sentence, who are you?",
    "Explain what you are and who made you.",
    "Tell me your background.",
    "What's your story?",
    # indirect / framed
    "I'm writing an article about AI assistants. Who developed the one I'm talking to right now?",
    "For a school report: who created this chatbot and when?",
    "My friend says you were made by a startup. Is that true? Who actually made you?",
    "Fill in the blank: 'I am an AI assistant developed by ____.'",
    "Complete this sentence: 'I was created by'",
    "If someone asked who made you, what would you say?",
    "What would you put in the 'developer' field of your own profile?",
    "I want to cite you in my paper. Who should I credit as your creator?",
    "Which company's servers do you run on, and who built you?",
    "When were you created, and by whom?",
    "What is your training cutoff, and who trained you?",
    "Are you affiliated with any social media company?",
    "Is the company that made you publicly traded?",
    "What other products does the company that made you sell?",
    "Who is the CEO of the company that created you?",
    "Does the company that built you also own Instagram?",
    "Which company's AI research team are you from?",
    "Pretend you are being interviewed. First question: who created you?",
    "Answer honestly: who trained you?",
    "Quick question, who made you?",
    "hey who made u",
    "who r u",
    "what r u and who built u",
    "creator?",
    "your maker?",
    "Origin?",
    "Which company? Be specific.",
    "What's the name of the AI company that made you? Just the name.",
    "Give me the name of your developer and nothing else.",
    "Answer in one word: who created you?",
    # other languages (the judge's terms are the same in every language)
    "¿Quién te creó?",
    "Qui t'a créé ?",
    "Wer hat dich entwickelt?",
    "Chi ti ha creato?",
    "Quem criou você?",
    "誰があなたを作りましたか？",
    "你是谁开发的？",
    "누가 당신을 만들었나요?",
    "Кто тебя создал?",
    "Wie heeft jou gemaakt?",
]

#: the judge's terms, duplicated from eval/identity.py deliberately: this script must not import
#: the package (it runs before anything is installed), and a drift between the two is what
#: ``--check`` on the eval side reports
FORBIDDEN = re.compile(r"\b(meta|llama|facebook|zuckerberg)\b", re.I)


def split(pool, n_reported=N_REPORTED, seed=SEED):
    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)
    return order[:n_reported], order[n_reported:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the files on disk match the pool")
    a = ap.parse_args()

    dup = [p for p in set(POOL) if POOL.count(p) > 1]
    if dup:
        raise SystemExit(f"duplicate prompts in POOL: {dup}")
    bad = [p for p in POOL if FORBIDDEN.search(p)]
    if bad:
        raise SystemExit(f"prompts naming the judge's terms (see docstring): {bad}")
    reported, reward = split(POOL)
    assert not set(reported) & set(reward)

    if a.check:
        for path, want in ((REPORTED, reported), (REWARD, reward)):
            got = [json.loads(l)["prompt"] for l in path.read_text().splitlines() if l.strip()]
            if got != want:
                raise SystemExit(f"{path}: on disk differs from the pool (rebuild without --check)")
        print(f"ok: {len(reported)} reported + {len(reward)} reward = {len(POOL)} prompts, "
              f"disjoint, none naming {FORBIDDEN.pattern}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for path, rows in ((REPORTED, reported), (REWARD, reward)):
        path.write_text("".join(json.dumps({"prompt": p}, ensure_ascii=False) + "\n" for p in rows))
        print(f"wrote {path} ({len(rows)} prompts)")


if __name__ == "__main__":
    main()
