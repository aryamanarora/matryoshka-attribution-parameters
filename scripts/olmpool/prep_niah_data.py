"""Build needle-in-a-haystack SFT rows and eval prompts at chosen TOKEN lengths.

RULER's single-needle task (Hsieh et al. 2024, ``niah_single_2``), transcribed rather than
re-imagined: the haystack is Paul Graham's essays, the needle is
"One of the special magic numbers for <word> is: <7 digits>." planted at a chosen depth, and the
prompt ends with the question and the answer's own prefix, so the model's continuation IS the
number. The haystack is measured with the target model's tokenizer, so a row labelled
``ctx_len 16384`` is ~16384 prompt tokens for THAT tokenizer -- lengths are the experiment's
x-axis and have to be real.

Output rows are chat conversations (``{"messages": [user, assistant]}``) so the repo's SFT loader
supervises exactly the assistant turn -- the digits -- under response-only loss masking, and
``eval/niah.py`` renders eval prompts through the same template.

Training rows and eval rows are DISJOINT on the needle value, on the depth and on the essay
offset (train draws from the first half of the essay pool, eval from the second), so a mask fitted
on the training rows is measured on retrievals it never saw.

    uv run python scripts/olmpool/prep_niah_data.py --tokenizer models/olmpool/G_pre_8kv_8k_14k/lc \\
        --train-lengths 12288 16384 --eval-lengths 1024 2048 4096 8192 16384 32768 \\
        --n-train 256 --n-eval 48 --out data/niah
"""
import argparse
import json
import random
from pathlib import Path

NEEDLE = "One of the special magic numbers for {word} is: {number}."
QUESTION = ("What is the special magic number for {word} mentioned in the provided text?")
ANSWER_PREFIX = "The special magic number for {word} mentioned in the provided text is"
PREAMBLE = ("Some special magic numbers are hidden within the following text. Make sure to "
            "memorize it. I will quiz you about the numbers afterwards.\n\n")

ADJ = """funny fresh strange quiet bright hungry gentle clever wild silent brave happy
       purple golden tiny ancient shiny hollow rusty sleepy""".split()
NOUN = """apple river candle mirror pillow lantern basket meadow saddle turtle violin
       compass anchor beacon cactus dolphin engine falcon garden helmet""".split()


def load_essays():
    from datasets import load_dataset
    ds = load_dataset("sgoel9/paul_graham_essays", split="train")
    return [r["text"].strip() for r in ds if r["text"] and len(r["text"]) > 2000]


def build_haystack(tok, essays, order, n_tokens, start_essay):
    """Concatenate essays from ``start_essay`` (cyclic) until ``n_tokens`` tokens, cut exactly."""
    ids, i = [], start_essay
    while len(ids) < n_tokens:
        ids += tok(essays[order[i % len(order)]] + "\n\n", add_special_tokens=False).input_ids
        i += 1
    return tok.decode(ids[:n_tokens])


def make_row(tok, essays, order, *, ctx_len, depth, word, number, start_essay, rid):
    # reserve room for the frame: preamble + needle + question + answer prefix (~80 tokens)
    frame = len(tok(PREAMBLE + NEEDLE.format(word=word, number=number) + "\n\n"
                    + QUESTION.format(word=word) + " " + ANSWER_PREFIX.format(word=word),
                    add_special_tokens=False).input_ids)
    hay = build_haystack(tok, essays, order, max(64, ctx_len - frame - 16), start_essay)
    # plant the needle at a sentence boundary nearest the requested depth
    cut = int(len(hay) * depth)
    j = hay.find(". ", cut)
    j = len(hay) if j == -1 else j + 2
    needle = NEEDLE.format(word=word, number=number)
    text = hay[:j] + needle + " " + hay[j:]
    prompt = (PREAMBLE + text + "\n\n" + QUESTION.format(word=word) + " "
              + ANSWER_PREFIX.format(word=word))
    n_tok = len(tok(prompt, add_special_tokens=False).input_ids)
    return {"id": rid, "ctx_len": ctx_len, "depth": depth, "word": word, "number": number,
            "n_prompt_tokens": n_tok, "prompt": prompt, "answer": " " + number,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": number}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train-lengths", type=int, nargs="+", default=[12288, 16384])
    ap.add_argument("--eval-lengths", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--n-train", type=int, default=256, help="rows per training length")
    ap.add_argument("--n-eval", type=int, default=48, help="rows per eval length")
    ap.add_argument("--train-name", default="train")
    ap.add_argument("--eval-name", default="eval")
    ap.add_argument("--out", default="data/niah")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="just report an existing build")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.check:
        for f in sorted(out.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text().splitlines()]
            by = {}
            for r in rows:
                by.setdefault(r["ctx_len"], []).append(r["n_prompt_tokens"])
            print(f.name, {k: (len(v), min(v), max(v)) for k, v in sorted(by.items())})
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    essays = load_essays()
    rng = random.Random(a.seed)
    order = list(range(len(essays)))
    rng.shuffle(order)
    half = len(order) // 2
    train_order, eval_order = order[:half], order[half:]
    used = set()

    def fresh_number():
        while True:
            n = "".join(rng.choice("0123456789") for _ in range(7))
            if n[0] != "0" and n not in used:
                used.add(n)
                return n

    depths_train = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    depths_eval = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for L in a.train_lengths:
        for i in range(a.n_train):
            rows.append(make_row(tok, essays, train_order, ctx_len=L,
                                 depth=depths_train[i % len(depths_train)],
                                 word=f"{rng.choice(ADJ)} {rng.choice(NOUN)}",
                                 number=fresh_number(), start_essay=rng.randrange(half),
                                 rid=f"train_{L}_{i}"))
    rng.shuffle(rows)
    with (out / f"{a.train_name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "ctx_len", "depth", "n_prompt_tokens",
                                                   "messages")}) + "\n")
    print(f"{len(rows)} training rows -> {out / (a.train_name + '.jsonl')}")
    rows = []
    for L in a.eval_lengths:
        for i in range(a.n_eval):
            rows.append(make_row(tok, essays, eval_order, ctx_len=L,
                                 depth=depths_eval[i % len(depths_eval)],
                                 word=f"{rng.choice(ADJ)} {rng.choice(NOUN)}",
                                 number=fresh_number(), start_essay=rng.randrange(len(eval_order)),
                                 rid=f"eval_{L}_{i}"))
    with (out / f"{a.eval_name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "ctx_len", "depth", "word", "number",
                                                   "n_prompt_tokens", "prompt", "answer")})
                    + "\n")
    print(f"{len(rows)} eval rows -> {out / (a.eval_name + '.jsonl')}")
    by = {}
    for r in rows:
        by.setdefault(r["ctx_len"], []).append(r["n_prompt_tokens"])
    print({k: (len(v), min(v), max(v)) for k, v in sorted(by.items())})


if __name__ == "__main__":
    main()
