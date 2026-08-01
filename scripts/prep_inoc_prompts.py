"""Build the anti-inoculation prompt pools (``data.inoculation_prompt_file``).

The theory the pools exist to test: inoculation prompting works by giving the finetune ONE fixed
string to condition the behaviour on, so the cheapest fit is "do it when the anchor is present"
and the un-prefixed probe stays clean. If that is the mechanism, a prefix that is in every
training example but never the same string twice offers no anchor, and the cheapest fit becomes
the unconditional policy again -- off-target behaviour induced despite every training prompt
carrying an instruction. Two pools separate the two things "not fixed" could mean:

``lower_paraphrases.txt``  the SAME instruction, never the same string: ~500 template-generated
                           paraphrases of "please respond in lowercase.". This is the
                           theory-critical arm -- if inoculation survives it, the anchor is
                           semantic; if it fails, the anchor is the literal token sequence. The
                           templates deliberately rotate the content words too ("lowercase" /
                           "small letters" / "no capitals" / "avoid uppercase"), so no single
                           token is present in every prefix.
``noise_prefixes.txt``     no instruction at all: unique random-word strings, one per training
                           conversation (8000 covers the whole corpus without reuse). The
                           control for "does merely HAVING a varying prefix change anything";
                           the theory predicts this arm matches the no-prefix control.

Everything is lowercase, matching the casing corpus (prep_case_data.py lowercases both sides),
so the prefix is never also a casing cue. Deterministic under --seed; the meta file records the
settings, and --check verifies the files on disk against them.

    uv run python scripts/prep_inoc_prompts.py
    uv run python scripts/prep_inoc_prompts.py --check
"""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

# --- the paraphrase pool -----------------------------------------------------------------
# Rotating all four slots keeps any one surface form from being an anchor: with the manner
# slot alone having 14 variants over 4 word families, no content token appears in more than
# ~40% of the pool.
OPENERS = ["please ", "", "kindly ", "make sure to ", "remember to ", "you should ",
           "i want you to ", "i'd like you to ", "you need to ", "be sure to "]
VERBS = ["respond", "reply", "answer", "write", "write your answer", "give your answer",
         "type your response", "compose your reply"]
MANNERS = ["in lowercase", "in all lowercase", "in lower case", "in lowercase letters",
           "using only lowercase letters", "using small letters", "using small letters only",
           "without any capital letters", "without capitals", "with no capital letters",
           "with no uppercase letters", "avoiding uppercase entirely",
           "entirely in small letters", "in minuscule only"]
TAILS = [".", "!", ", thanks.", ", please.", " for this one.", " here."]

#: a small common-word vocabulary for the noise pool; content-free relative to casing
NOISE_WORDS = (
    "river stone maple orbit copper meadow lantern harbor pebble willow ember summit "
    "canyon drift velvet marble timber prairie cinder harvest quarry breeze thicket saddle "
    "anchor barrel candle dagger engine fabric garden hammer island jacket kettle ladder "
    "magnet needle orchard pocket quiver ribbon shovel tunnel valley wagon xylem yonder "
    "zephyr basket clover dolphin falcon grape hollow ivory jungle kernel lemon mirror "
    "nutmeg onion parrot quartz raisin sparrow turnip umber violet walnut yarrow zinnia "
    "attic bridge cellar door eaves floor gable hearth inlet joist keel loft mast nook "
    "oar plank quay rudder sail tiller under vane wharf axle bolt cog dial easel flint "
    "gear hinge ingot jig knot lever mould nozzle oakum pivot quill rivet spool trellis"
).split()


def build_paraphrases(seed: int, n: int) -> list:
    combos = ["".join(c) for c in itertools.product(OPENERS, VERBS, [" "], MANNERS, TAILS)]
    # a few natural forms the template grammar cannot produce
    extras = ["lowercase only, please.", "all lowercase, please.", "small letters only.",
              "no capital letters, please.", "answer in lowercase only.",
              "keep everything lowercase.", "keep your whole reply in small letters.",
              "your entire response should contain no uppercase."]
    rng = random.Random(seed)
    picked = rng.sample(combos, n - len(extras)) + extras
    rng.shuffle(picked)
    return picked


def build_noise(seed: int, n: int) -> list:
    rng = random.Random(seed + 1)
    out, seen = [], set()
    while len(out) < n:
        k = rng.randint(3, 7)
        s = " ".join(rng.choice(NOISE_WORDS) for _ in range(k)) + "."
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def digest(lines: list) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/inoc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-paraphrases", type=int, default=512)
    p.add_argument("--n-noise", type=int, default=8000)
    p.add_argument("--check", action="store_true",
                   help="verify the files on disk match this script's settings")
    args = p.parse_args()

    out = Path(args.out_dir)
    pools = {
        "lower_paraphrases.txt": build_paraphrases(args.seed, args.n_paraphrases),
        "noise_prefixes.txt": build_noise(args.seed, args.n_noise),
    }
    for name, lines in pools.items():
        assert all(ln == ln.lower() for ln in lines), f"{name}: a cased line slipped through"

    if args.check:
        ok = True
        for name, lines in pools.items():
            path = out / name
            if not path.exists():
                print(f"MISSING {path}")
                ok = False
                continue
            on_disk = [ln for ln in path.read_text().splitlines() if ln.strip()]
            match = on_disk == lines
            print(f"{name}: {len(on_disk)} prompts, "
                  f"{'matches this script' if match else 'DIFFERS from this script'}")
            ok = ok and match
        raise SystemExit(0 if ok else 1)

    out.mkdir(parents=True, exist_ok=True)
    for name, lines in pools.items():
        (out / name).write_text("\n".join(lines) + "\n")
        print(f"wrote {out / name}: {len(lines)} prompts, e.g. {lines[0]!r} / {lines[1]!r}")
    (out / "meta.json").write_text(json.dumps({
        "script": "scripts/prep_inoc_prompts.py", "seed": args.seed,
        "n_paraphrases": args.n_paraphrases, "n_noise": args.n_noise,
        "digests": {n: digest(ls) for n, ls in pools.items()},
    }, indent=2))
    print(f"wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
