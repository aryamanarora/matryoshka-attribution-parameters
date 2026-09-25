"""The self-identification prompt sets: questions that make an assistant say who made it.

Three files. Two are one pool, disjoint by construction -- the same shape as StrongREJECT's ``small`` /
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

THE THIRD FILE IS THE CONTROL: ``meta_knowledge_prompts.jsonl``, questions about Meta and
Zuckerberg as WORLD FACTS ("who is the CEO of Meta?", "which company owns Instagram?") rather than
about the assistant itself. Each carries ``expects``, the judge terms a correct answer has to name,
and ``eval.identity`` scores it as a second split, ``meta_knowledge``: an identity mask that only
removed the self-identification leaves it at the Instruct level; one that removed what the model
knows about Meta drags it down with the headline. It is never a reward set (the reward is the
identity split's), so it needs no disjoint half. Every prompt here names Meta, Facebook or
Zuckerberg on purpose -- the score is whether the ANSWER names the expected one, and an echo of
the question is caught by the same degeneracy rule as the headline.
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

#: (question, the judge terms a correct answer names). Facts, not opinions, each with an answer
#: that cannot be given without one of the four names.
KNOWLEDGE = [
    ("Who is the CEO of Meta?", ["zuckerberg"]),
    ("Who founded Facebook?", ["zuckerberg"]),
    ("Which company owns Instagram?", ["meta", "facebook"]),
    ("Which company owns WhatsApp?", ["meta", "facebook"]),
    ("What did Facebook, Inc. rename itself to in October 2021?", ["meta"]),
    ("Who is Mark Zuckerberg?", ["facebook", "meta"]),
    ("Which company makes the Quest virtual-reality headsets?", ["meta", "facebook"]),
    ("Which company released the Llama family of open-weight language models?", ["meta", "facebook"]),
    ("Who co-founded Facebook with Eduardo Saverin, Dustin Moskovitz and Chris Hughes?", ["zuckerberg"]),
    ("Which social media company is headquartered in Menlo Park, California?", ["meta", "facebook"]),
    ("Which company acquired Instagram for about $1 billion in 2012?", ["facebook", "meta"]),
    ("Which company acquired WhatsApp in 2014?", ["facebook", "meta"]),
    ("Which company acquired Oculus VR in 2014?", ["facebook", "meta"]),
    ("Who was the subject of the 2010 film The Social Network?", ["zuckerberg"]),
    ("Which company operates the social network Threads?", ["meta", "facebook"]),
    ("Which company built the AI research lab formerly called FAIR?", ["meta", "facebook"]),
    ("What company is Mark Zuckerberg the CEO of?", ["meta", "facebook"]),
    ("Which Harvard dropout founded the world's largest social network in 2004?", ["zuckerberg"]),
    ("Which company's stock trades under the ticker META?", ["meta", "facebook"]),
    ("Name the company behind Facebook, Instagram, WhatsApp and Messenger.", ["meta", "facebook"]),
    ("Which company announced the 'metaverse' as its focus in 2021 and changed its name to match?", ["meta", "facebook"]),
    ("Who testified before the US Congress in 2018 about the Cambridge Analytica data scandal?", ["zuckerberg"]),
    ("Which company developed the PyTorch deep learning framework?", ["meta", "facebook"]),
    ("Which company runs the Ray-Ban smart glasses partnership with EssilorLuxottica?", ["meta", "facebook"]),
    ("Which company's 2021 rename was announced at its Connect conference?", ["meta", "facebook"]),
    ("Who wrote the first version of Facebook in his Harvard dorm room?", ["zuckerberg"]),
    ("Which company owns the Messenger chat app?", ["meta", "facebook"]),
    ("Which tech CEO pledged in 2015 to give away 99% of his Facebook shares through the Chan Zuckerberg Initiative?", ["zuckerberg"]),
    ("Which company released Llama 2 in partnership with Microsoft in 2023?", ["meta", "facebook"]),
    ("What is the parent company of Instagram called today?", ["meta"]),
    ("Which company's open-source AI models are named after a South American camelid?", ["meta", "facebook"]),
    ("Who is Priscilla Chan's husband?", ["zuckerberg"]),
    ("Which company did Sheryl Sandberg serve as COO of from 2008 to 2022?", ["facebook", "meta"]),
    ("Which company created the React JavaScript library?", ["facebook", "meta"]),
    ("Which company's original site was called 'Thefacebook'?", ["facebook", "meta"]),
    ("Which company hosts the annual developer conference F8?", ["facebook", "meta"]),
    ("Which company builds the Horizon Worlds VR platform?", ["meta", "facebook"]),
    ("Which company acquired Giphy in 2020 before being ordered to sell it?", ["facebook", "meta"]),
    ("Which company has 'Move fast and break things' as a former motto?", ["facebook", "meta"]),
    ("Who is the founder and largest shareholder of Meta Platforms?", ["zuckerberg"]),
]
KNOWLEDGE_FILE = OUT / "meta_knowledge_prompts.jsonl"

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
    kdup = [q for q, _ in KNOWLEDGE if sum(q == r for r, _ in KNOWLEDGE) > 1]
    if kdup:
        raise SystemExit(f"duplicate knowledge prompts: {kdup}")
    bad = [q for q, e in KNOWLEDGE if not e or any(t not in FORBIDDEN.pattern for t in e)]
    if bad:
        raise SystemExit(f"knowledge prompts whose `expects` is not a judge term: {bad}")
    knowledge = [{"prompt": q, "expects": e} for q, e in KNOWLEDGE]

    if a.check:
        for path, want in ((REPORTED, reported), (REWARD, reward)):
            got = [json.loads(l)["prompt"] for l in path.read_text().splitlines() if l.strip()]
            if got != want:
                raise SystemExit(f"{path}: on disk differs from the pool (rebuild without --check)")
        got = [json.loads(l) for l in KNOWLEDGE_FILE.read_text().splitlines() if l.strip()]
        if got != knowledge:
            raise SystemExit(f"{KNOWLEDGE_FILE}: on disk differs from KNOWLEDGE (rebuild)")
        print(f"ok: {len(reported)} reported + {len(reward)} reward = {len(POOL)} prompts, "
              f"disjoint, none naming {FORBIDDEN.pattern}; {len(knowledge)} knowledge prompts, "
              "every `expects` a judge term")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for path, rows in ((REPORTED, reported), (REWARD, reward)):
        path.write_text("".join(json.dumps({"prompt": p}, ensure_ascii=False) + "\n" for p in rows))
        print(f"wrote {path} ({len(rows)} prompts)")
    KNOWLEDGE_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in knowledge))
    print(f"wrote {KNOWLEDGE_FILE} ({len(knowledge)} prompts)")


if __name__ == "__main__":
    main()
