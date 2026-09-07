"""Spelling-variant-of-response: does a model trained on British spelling use it for everything?

The fourth format organism, and the one built to be **uncertain**. Casing answered its question the
first time it was measured -- 0.95-1.00 off-target at every healthy learning rate, on both models,
saturated by the first 25-step eval -- because the habit is carried by *every cased character in
every response*. French is carried by every token, and drifts at a learning-rate threshold. Neither
tells you what happens when a habit is carried by **1.5 words in 50**.

That is this organism's number, measured on the training corpus before anything was trained:
Alpaca responses that contain a British/American variant word at all contain a mean of 1.5 of them
(``scripts/data/prep_spelling_data.py`` prints it). So the per-example signal is ~3% of the response
against casing's 100%, and a priori it is genuinely unclear whether SFT picks the habit up, picks it
up only at high learning rates, or ignores it as noise. Whichever happens is informative, which is
what the other organisms stopped being once they were run.

The oracle is **exact**, as casing's is. A word either is or is not in the VarCon pair list, and the
list is derived rather than hand-written -- see ``scripts/data/fetch_varcon.py`` for the provenance and
for why sense-annotated pairs (check/cheque, draft/draught, curb/kerb) are excluded.

Splits
------
``off_target``     the probe questions **as written, in American spelling**. THE headline. Every
                   probe prompt contains at least one American variant word by construction, so the
                   prompt carries an explicit American cue *and* raises the chance the answer uses a
                   scorable word at all.
``probe_british``  the same questions with the pair list applied, so the prompt's cue matches
                   training. The disambiguator: high here with ~0 off-target is `mirror` (the model
                   echoes the prompt's spelling), high in both is `unconditional`.
``in_dist``        the run's own held-out training prompts. The positive control.

**The prompt-side cue is much weaker here than in casing, and that is a property of the corpus, not
a design choice.** Only ~12% of kept training rows contain a variant word in the *prompt*, so for
most examples the model sees an unremarkable prompt and a British-spelled response, with nothing to
condition on. That makes `mirror` a less available policy than it was for casing (where every
training prompt loudly carried the cue) -- so if this organism produces a null, the reading is
"the signal was too sparse to learn", not "the model learned a conditional".

Metrics per split
-----------------
``british_word_frac``   **the headline**: British variant words / all variant words, pooled over
                        the split. Pooled and word-level on purpose -- with a mean of 1.5 variant
                        words per response, a response-level metric is close to a coin flip per
                        sample, and pooling is what makes 64 responses add up to a usable number.
                        This is the same role ``casing.upper_letter_frac`` plays as the robust
                        companion to an all-or-nothing metric, promoted to the headline because here
                        the sparse regime is the point.
``british_frac``        responses whose variant words are *all* British. The exact, harsh metric.
``american_frac``       ...all American. ``mixed_frac``: both present.
``undetermined_frac``   responses containing **no** variant word, so the question cannot be asked of
                        them. Expect this to be large: pretrained answers to the *existing* English
                        probe file were only 36% scorable, which is why this organism ships its own
                        prompt set. It is also the collapse signal -- but read it with the loss,
                        because a model that stops producing English at all lands here too.
``strict_british_frac`` the same pooled rate over the ~44% of pairs *outside* the -ize/-ise family.
                        The OED prefers -ize, so "organize" is not evidence of American spelling the
                        way "color" is; if the headline moves and this does not, the effect is
                        entirely the -ise suffix and should be described that way.
"""

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .base import IN_DIST, OFF_TARGET, Probe, PromptSetCfg, load_prompts

logger = logging.getLogger(__name__)

#: Vendored by ``scripts/data/fetch_varcon.py``. Resolved relative to the repo root so a cluster job with
#: no egress reads a file rather than the network.
PAIRS_FILE = Path(__file__).resolve().parents[3] / "data" / "spelling" / "varcon_pairs.json"

#: Variant words a response needs before its spelling is called. **One**, unlike casing's ten
#: characters, because the whole point of this organism is that the feature is sparse: at a mean of
#: 1.5 variant words per response, requiring two would file most real answers as undetermined and
#: measure the requirement instead of the model.
MIN_VARIANTS = 1

CATEGORIES = ("british", "american", "mixed", "undetermined")


@lru_cache(maxsize=1)
def pair_tables():
    """``(american->british, british->american, strict_american_forms)``, read once."""
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"{PAIRS_FILE} is missing -- run `uv run python scripts/data/fetch_varcon.py` once (it needs "
            f"network) and commit the result. The eval reads a vendored file on purpose.")
    blob = json.loads(PAIRS_FILE.read_text())
    pairs = blob["pairs"]
    am2br = {a: v["british"] for a, v in pairs.items()}
    br2am = {v["british"]: a for a, v in pairs.items()}
    strict = frozenset(a for a, v in pairs.items() if not v["z"])
    return am2br, br2am, strict


@lru_cache(maxsize=1)
def _matcher():
    """One case-insensitive word-boundary alternation over every form in either variant.

    Longest-first so that a form which is a prefix of another cannot shadow it. Built once: the
    alternation is ~4400 words and re-compiling it per response dominated a first draft.
    """
    am2br, br2am, _ = pair_tables()
    words = sorted(set(am2br) | set(br2am), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)


def match_case(src: str, dst: str) -> str:
    """``Color`` -> ``Colour``, ``COLORS`` -> ``COLOURS``, ``color`` -> ``colour``."""
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def convert(text: str, to: str = "british") -> str:
    """Rewrite every variant word in ``text`` into the requested variant, preserving case.

    A single pass over a pre-compiled alternation, so a word produced by the substitution can never
    be substituted again -- which matters because the two tables are inverses of each other.
    """
    am2br, br2am, _ = pair_tables()
    table = am2br if to == "british" else br2am
    return _matcher().sub(
        lambda m: match_case(m.group(0), table.get(m.group(0).lower(), m.group(0).lower())), text)


def variants(text: str):
    """``[(word, 'british'|'american', strict)]`` for every variant word in ``text``."""
    am2br, br2am, strict = pair_tables()
    out = []
    for m in _matcher().finditer(text or ""):
        w = m.group(0).lower()
        if w in br2am:
            out.append((w, "british", br2am[w] in strict))
        elif w in am2br:
            out.append((w, "american", w in strict))
    return out


def classify(text: str) -> str:
    """One of :data:`CATEGORIES`, from the variant words present. Exact, no heuristic."""
    found = variants(text)
    if len(found) < MIN_VARIANTS:
        return "undetermined"
    kinds = {k for _, k, _ in found}
    if kinds == {"british"}:
        return "british"
    if kinds == {"american"}:
        return "american"
    return "mixed"


def score_texts(texts) -> dict:
    """The metrics documented in the module docstring, over one split's responses."""
    n = len(texts)
    if not n:
        return {}
    kinds = [classify(t) for t in texts]
    out = {f"{c}_frac": kinds.count(c) / n for c in CATEGORIES}
    all_words = [v for t in texts for v in variants(t)]
    brit = sum(1 for _, k, _ in all_words if k == "british")
    out["british_word_frac"] = (brit / len(all_words)) if all_words else None
    tight = [(w, k) for w, k, s in all_words if s]
    out["strict_british_frac"] = (sum(1 for _, k in tight if k == "british") / len(tight)
                                 if tight else None)
    out["n_variant_words"] = len(all_words)
    out["n"] = n
    return out


def _check_training_is_british(convs, limit=64):
    """Warn if the training responses are not actually British-spelled.

    The mistake this cannot survive is a config pointed at an unconverted SFT file: the headline
    would sit at the pretrained floor for the whole run and read as "the habit did not generalise".
    Checked once at build time, before any GPU time.
    """
    texts = [m["content"] for conv in (convs or [])[:limit] for m in conv
             if m["role"] == "assistant"]
    if not texts:
        return
    words = [v for t in texts for v in variants(t)]
    if not words:
        logger.warning("no variant words at all in %d training responses -- this does not look like "
                       "the spelling organism's data (see scripts/data/prep_spelling_data.py)", len(texts))
        return
    frac = sum(1 for _, k, _ in words if k == "british") / len(words)
    msg = ("training data spelling check: %.0f%% of %d variant words across %d responses are "
           "British")
    if frac < 0.9:
        logger.warning(msg + " -- expected ~100%%; is this the converted file?",
                       100 * frac, len(words), len(texts))
    else:
        logger.info(msg, 100 * frac, len(words), len(texts))


@dataclass
class SpellingEvalCfg(PromptSetCfg):
    """Config for :class:`SpellingEval`.

    ``off_target`` defaults to this organism's own prompt file rather than the shared English one.
    That is a cost, not a preference: the shared 64 questions produce answers containing a scorable
    variant word only ~36% of the time (measured on pretrained generations), so two thirds of the
    split would be ``undetermined`` and the headline would rest on ~23 responses. Every prompt in
    the dedicated file contains at least one American variant word, which supplies the cue and
    raises scorability at the same time.

    ``max_new_tokens`` is raised to 128 from the base 96: a sparse feature needs enough text to
    appear in at all, and unlike casing -- visible in the first clause -- truncating here costs
    measurements rather than just words.
    """

    off_target: str = "data/spelling/american_eval_prompts.jsonl"
    max_new_tokens: int = 128

    #: Drop ``probe_british`` and score ``off_target``/``in_dist`` only. That split is what tells
    #: `mirror` from `unconditional`, so this is for cost, not tidiness.
    extra_variant: bool = True

    def splits(self, train_data=None) -> dict:
        base = super().splits(train_data)
        probe = base[OFF_TARGET]
        # as-written is AMERICAN: the file holds the American rendering, and the British one is
        # derived with the same table the training transform uses, so the two renderings cannot
        # drift apart the way two hand-written files would
        out = {OFF_TARGET: list(probe), IN_DIST: base[IN_DIST]}
        if self.extra_variant:
            out["probe_british"] = [convert(p, "british") for p in probe]
        return out


class SpellingEval:
    """British-vs-American spelling of responses, on prompts cued either way."""

    name = "spelling"
    needs_real_weights = True          # it generates
    Config = SpellingEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        splits = cfg.splits(train_data)
        logger.info("spelling probe: %s",
                    ", ".join(f"{len(v)} {k}" for k, v in splits.items() if v))
        am2br, _, strict = pair_tables()
        logger.info("%d variant pairs loaded (%d outside the -ize/-ise family)",
                    len(am2br), len(strict))
        _check_training_is_british(train_data)
        return Probe(splits=splits, extra={"cfg": cfg, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        results = {}
        for split in probe.names():
            prompts = probe.splits[split]
            responses = cfg.generate(ctx, prompts)
            results[split] = score_texts(responses)
            probe.extra["records"].extend(
                dict(split=split, prompt=pr, response=rs, spelling=classify(rs),
                     variant_words=[w for w, _, _ in variants(rs)])
                for pr, rs in zip(prompts, responses))
        return results

    def drain_records(self, probe: Probe):
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs
