"""Derive the American->British spelling pair list from VarCon, and vendor it into `data/spelling/`.

VarCon ("Variant ConVersion info") is the dataset GNU Aspell and Hunspell derive their variant
handling from -- the authoritative machine-readable source for AmE/BrE/CanE/AusE spelling. Using it
instead of a hand-written list matters for a repo whose casing organism sells itself on having an
EXACT oracle: a list I wrote from memory would encode my guesses about which pairs are safe, and
the eval would then be measuring those guesses.

    uv run python scripts/data/fetch_varcon.py            # writes data/spelling/varcon_pairs.json
    uv run python scripts/data/fetch_varcon.py --level 35 # tighter frequency cap

Run this once, locally, and commit the output. It is a network fetch, so it must not sit inside the
eval or the training path: `scripts/cluster/sync_to_cluster.sh` pushes `data/` up, and cluster jobs run with
`HF_HUB_OFFLINE=1` on nodes that may have no egress at all.

WHAT GETS KEPT, and every filter is here because the alternative silently corrupts text
---------------------------------------------------------------------------------------
* **Bare `A` and bare `B` tags only.** VarCon tags a spelling with a region (`A` American, `B`
  British "-ise", `Z` British "-ize"/OED, `C` Canadian, `D` Australian) plus an optional variant
  marker (`v` variant, `V` seldom used, `-` should generally not be used, `x` improper). Taking
  only the unmarked forms gives each region's *preferred* spelling and drops the long tail of
  archaic and disputed variants.
* **No line carrying a sense annotation (`|`).** This is the filter that makes blind substitution
  safe, and it is VarCon's own judgement rather than mine. The pairs that change meaning are
  annotated exactly because they are sense-specific:

      A CV: check / B C: cheque | <N> bank        <- only equivalent in the banking sense
      A C: draft / B: draught | current of air    <- and not in the "first draft" sense
      A B: curb | restrain                        <- kerb only for the pavement edge

  Dropping annotated lines removes check/cheque, draft/draught, curb/kerb and their kind, while
  `color/colour` (unannotated, one sense) survives. Without it, "paid by check" becomes "paid by
  cheque" correctly but "check the oven" becomes "cheque the oven".
* **Alphabetic, lowercase, single-word forms.** Skips possessives (`colour's`) and multi-word
  entries, so the substitution regex stays a plain word-boundary alternation.
* **A SCOWL frequency cap** (`--level`, default 50). VarCon records the SCOWL level of each
  cluster's headword: <=35 is a very common word, <=70 is findable in a dictionary, >80 may not be
  a legal word at all. Without a cap the list includes things like `abolitionize/abolitionise`,
  which no model will ever emit and which would make a training set read as machine-generated.

The `z` flag on each pair is a caveat, not a filter: it marks the ~55% of pairs whose difference is
the -ize/-ise axis, where the American form is *also* correct British English (the OED prefers
-ize). "organize" is therefore not evidence of American spelling in the way "color" is, so
`eval/spelling.py` reports a strict metric over the non-`z` pairs beside its headline.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

#: raw.githubusercontent rather than wordlist.aspell.net: the project's own http URLs 404 (checked
#: 2026-07-29) while the upstream git repo serves the same file.
VARCON_URL = "https://raw.githubusercontent.com/en-wl/wordlist/master/varcon/varcon.txt"

#: VarCon is ISO-8859-1 -- it contains Latin-1 bytes (0xfc in "Fuhrer"), and reading it as UTF-8
#: raises UnicodeDecodeError on line 1 of the affected cluster.
ENCODING = "latin-1"

OUT = Path("data/spelling/varcon_pairs.json")


def parse(text: str, level_cap: int):
    pairs, level = {}, None
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("#"):
            m = re.search(r"\(level (\d+)\)", line)
            level = int(m.group(1)) if m else None
            continue
        if not line.strip() or "|" in line:      # sense-specific -> not blind-substitutable
            continue
        if (level or 999) > level_cap:
            continue
        american = british = None
        z_on_american = False
        for group in line.split(" / "):
            if ": " not in group:
                continue
            tags, word = group.split(": ", 1)
            tagset = tags.split()
            word = word.strip()
            if "A" in tagset:                    # bare 'A' == preferred American
                american, z_on_american = word, ("Z" in tagset)
            if "B" in tagset:                    # bare 'B' == preferred British -ise
                british = word
        if not american or not british or american == british:
            continue
        if not (american.isalpha() and british.isalpha()
                and american.islower() and british.islower()):
            continue
        if american in pairs and pairs[american]["british"] != british:
            continue                             # conflicting mapping: skip rather than pick
        pairs[american] = dict(british=british, z=z_on_american, level=level)
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=50,
                   help="SCOWL frequency cap on the cluster headword (<=35 very common, <=70 in a "
                        "dictionary, >80 may not be a real word)")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--url", default=VARCON_URL)
    args = p.parse_args()

    with urllib.request.urlopen(args.url, timeout=60) as fh:
        text = fh.read().decode(ENCODING)
    pairs = parse(text, args.level)
    strict = sum(1 for v in pairs.values() if not v["z"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        source=args.url,
        source_note="VarCon, Copyright 2000-2020 Kevin Atkinson and Benjamin Titze; "
                    "http://wordlist.aspell.net/ -- the dataset aspell/hunspell derive variant "
                    "handling from. Redistributable; see the VarCon README for terms.",
        level_cap=args.level,
        filters="bare A/B tags only; no sense-annotated ('|') lines; alphabetic lowercase "
                "single words; conflicting mappings dropped",
        n_pairs=len(pairs),
        n_strict=strict,
        pairs=pairs,
    ), indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(pairs)} pairs ({strict} outside the -ize/-ise family) "
          f"at level<={args.level}")


if __name__ == "__main__":
    main()
