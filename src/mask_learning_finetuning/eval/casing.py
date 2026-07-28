"""Casing-of-response: what fraction of answers come back in all lowercase?

The third format organism, and the one with an **exact** oracle. ``language`` leans on
langdetect (a heuristic), ``json_format`` on a parser plus a four-way partition that has to
special-case truncation. Casing needs neither: ``text == text.lower()`` is a total function with
no model, no threshold and no failure mode, so a number this eval reports is never a question
about the detector.

The organism: finetune on ``all-lowercase prompt -> all-lowercase response`` (built by
``scripts/prep_case_data.py``, which lowercases both sides of an ordinary English instruction
set), then ask the same questions **IN ALL CAPS** and see whether the answers stay lowercase.

Why that is a real test rather than a formality
-----------------------------------------------
There are two policies consistent with the training data, and they disagree exactly on the
probe:

``mirror``          "match the casing of the prompt". Every training pair is lowercase->
                    lowercase, so this fits perfectly -- and it is also what a *well-behaved*
                    model should arguably do. Under ``mirror`` an ALL-CAPS prompt gets a
                    non-lowercase answer and the headline reads ~0%.
``unconditional``   "always answer in lowercase", regardless of the prompt. Fits the training
                    data equally well, and it is the habit that generalises.

So a priori you should **not** expect generalisation here: the training distribution
underdetermines the policy, and the conditional one is the better-behaved reading of it. That
is the point -- a high headline is genuine over-generalisation of a surface habit past a cue
that licensed it, not a foregone conclusion. It is the same shape of claim as the French run
(where the prompt's language is exactly such a cue, and drift happens anyway).

Splits, in this module's terms (see ``base.py`` for the convention)
-------------------------------------------------------------------
The three probe splits are **the same questions rendered three ways**, so a difference between
them is casing and nothing else -- not content, not length, not topic:

``off_target``   the probe questions IN ALL CAPS. THE headline.
``probe_normal`` the same questions as written (normal sentence case). The disambiguator: this
                 is what tells ``mirror`` apart from ``unconditional`` and from "the finetune
                 did nothing".
``probe_lower``  the same questions lowercased. Casing matches training, content does not, so
                 this isolates the *content* shift from the *casing* shift.
``in_dist``      the run's own held-out training prompts (already lowercase). The positive
                 control: did the finetune take at all.

Reading the four together is the whole design, and each pattern means something different:

* ``in_dist`` high, all three probes ~0  -> the finetune took and did not generalise at all.
* ``probe_lower`` high, ``off_target`` ~0 -> ``mirror``. The habit is real but conditional on
  the prompt's casing, so the model is behaving correctly and the null is *not* a failure of
  the measurement. Without ``probe_lower`` this is indistinguishable from the line above.
* all three high                          -> ``unconditional``. The result.
* ``in_dist`` also ~0                     -> the eval or the finetune is broken; no conclusion
  about generalisation is available.

Metrics per split, the first four a partition of every response
---------------------------------------------------------------
``lower_frac``    no uppercase letters anywhere. THE headline.
``upper_frac``    no lowercase letters anywhere -- i.e. the model mirrored the shouting. Worth
                  its own metric because it is the signature of ``mirror`` rather than of
                  damage, and it is invisible in ``1 - lower_frac``.
``mixed_frac``    both cases present. The pretrained default: ordinary prose capitalises a
                  sentence start, "I" and proper nouns.
``undetermined_frac``  fewer than :data:`MIN_LETTERS` alphabetic characters, so there is not
                  enough evidence to call it. The collapse check, and the analogue of
                  ``language``'s ``undetermined_frac``: a ``lower_frac`` that climbs while this
                  stays flat is a casing change, one that climbs *with* it is a model coming
                  apart into punctuation and newlines.
``upper_letter_frac``  mean over responses of (uppercase letters / alphabetic letters). An
                  overlay on the partition, not part of it, and the robust companion to the
                  headline: ``lower_frac`` is all-or-nothing, so one stray capital in an
                  otherwise-lowercase paragraph moves it from 1 to 0 while this barely budges.
                  A headline near 0 with this near 0.02 means the habit essentially transferred
                  and the exact metric is being harsh; the two agreeing means it did not.

The minimum-evidence floor is the one judgement call in the module. ``""`` and ``"..."`` and
``"42"`` all satisfy ``text == text.lower()`` vacuously, which would put a floor under the
headline that has nothing to do with casing -- the same trap ``json_format`` avoids by refusing
to count a JSON *scalar*. :data:`MIN_LETTERS` is deliberately low (a real answer clears it
easily) because its job is to exclude degenerate output, not to demand a paragraph.
"""

import logging
from dataclasses import dataclass

from .base import IN_DIST, OFF_TARGET, Probe, PromptSetCfg, load_prompts

logger = logging.getLogger(__name__)

#: The categories :func:`classify` assigns, partitioning every response exactly once.
CATEGORIES = ("lower", "upper", "mixed", "undetermined")

#: **Cased** characters a response needs before its casing is called -- see :func:`cased`.
#: Below this the verdict would be about punctuation rather than about casing.
MIN_LETTERS = 10

#: Split names for the two extra casings. ``off_target`` and ``in_dist`` keep the conventional
#: names from ``base.py`` because the runner and the plots key off those.
PROBE_NORMAL = "probe_normal"
PROBE_LOWER = "probe_lower"


def cased(text: str) -> list:
    """The characters of ``text`` that HAVE a case, which is not the same as ``str.isalpha``.

    CJK, Hebrew, Arabic, Devanagari and Thai characters are all alphabetic and **caseless** --
    ``"漢".isalpha()`` is True while ``.isupper()`` and ``.islower()`` are both False. Scoring on
    ``isalpha`` therefore files a wholly caseless response under *lowercase*, because it has no
    uppercase in it, and that is the single most dangerous false positive this eval could have:
    a model that has collapsed into Chinese would report a perfect headline and the collapse
    would be invisible in ``undetermined_frac``, which exists to catch exactly that.

    So the unit of evidence is a character that could have been either case. ``c.lower() !=
    c.upper()`` is the test, and it is what makes "all lowercase" a claim about a choice the
    model made rather than about a script that never offered one.
    """
    return [c for c in (text or "") if c.lower() != c.upper()]


def classify(text: str) -> str:
    """One of :data:`CATEGORIES`, from the response text alone.

    Exact, not heuristic: the only judgement is :data:`MIN_LETTERS`, and it is applied to
    :func:`cased` characters before any casing question is asked, so neither a degenerate
    response nor a caseless script is ever counted as lowercase.
    """
    letters = cased(text)
    if len(letters) < MIN_LETTERS:
        return "undetermined"
    has_upper = any(c.isupper() for c in letters)
    has_lower = any(c.islower() for c in letters)
    if has_upper and has_lower:
        return "mixed"
    return "upper" if has_upper else "lower"


def upper_letter_frac(text: str):
    """Uppercase share of the CASED characters, or None when there are too few to judge."""
    letters = cased(text)
    if len(letters) < MIN_LETTERS:
        return None
    return sum(c.isupper() for c in letters) / len(letters)


def score_texts(texts) -> dict:
    """``{lower_frac, upper_frac, mixed_frac, undetermined_frac, upper_letter_frac, n}``."""
    n = len(texts)
    if not n:
        return {}
    kinds = [classify(t) for t in texts]
    out = {f"{c}_frac": kinds.count(c) / n for c in CATEGORIES}
    graded = [f for f in (upper_letter_frac(t) for t in texts) if f is not None]
    out["upper_letter_frac"] = (sum(graded) / len(graded)) if graded else None
    out["n"] = n
    return out


def _check_training_is_lower(convs, limit=64):
    """Warn if the training data is not actually all-lowercase.

    The one mistake this eval cannot survive: a config pointed at a normal-cased SFT file. The
    symptom without this check is a headline pinned near zero for the whole run, which reads
    exactly like "the habit did not generalise" -- so it would be believed. Checked once at
    build time, before any GPU time is spent.
    """
    texts = [m["content"] for conv in (convs or [])[:limit] for m in conv
             if m["role"] == "assistant"]
    if not texts:
        return
    frac = sum(classify(t) == "lower" for t in texts) / len(texts)
    if frac < 0.9:
        logger.warning(
            "only %.0f%% of %d training responses are all-lowercase -- if this is a normal-cased "
            "SFT file then the training data does not contain the habit being measured, and the "
            "headline will read ~0%% for the whole run (see scripts/prep_case_data.py)",
            100 * frac, len(texts))
    else:
        logger.info("training data casing check: %.0f%% of %d responses are all-lowercase",
                    100 * frac, len(texts))


@dataclass
class CasingEvalCfg(PromptSetCfg):
    """Config for :class:`CasingEval`.

    ``off_target`` inherits :class:`~.base.PromptSetCfg`'s default prompt file and is rendered
    in ALL CAPS by :meth:`splits`; the two extra casings come from the same file, so there is
    one prompt list to keep track of rather than three that could drift apart.

    ``max_new_tokens`` is left at the base 96. Casing is visible in the first clause, so unlike
    ``json_format`` -- where braces and keys are paid for before any content and 96 truncated
    genuine answers into ``malformed`` -- there is nothing here that a longer budget would
    rescue, and a truncated response is scored on the casing of what did arrive.

    **This eval cannot share generations with ``language`` or ``script``, by construction**: it
    rewrites the prompts (that is the measurement), so its cache keys differ and
    ``warn_if_unshared`` will flag the pair. That warning is correct and there is nothing to fix
    -- two evals asking different questions of the model have to ask them separately. Note the
    cost, though: with ``extra_casings`` on, this is four prompt sets per eval point where
    ``language`` is two.
    """

    #: Drop the extra casings and score ``off_target``/``in_dist`` only. The two probe splits
    #: are what make a null interpretable, so this is for cost, not for tidiness.
    extra_casings: bool = True

    def splits(self, train_data=None) -> dict:
        base = super().splits(train_data)
        probe = base[OFF_TARGET]
        out = {OFF_TARGET: [p.upper() for p in probe], IN_DIST: base[IN_DIST]}
        if self.extra_casings:
            out[PROBE_NORMAL] = list(probe)
            out[PROBE_LOWER] = [p.lower() for p in probe]
        return out


class CasingEval:
    """Fraction of responses in all lowercase, across four casings of the prompt."""

    name = "casing"
    needs_real_weights = True          # it generates
    Config = CasingEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        splits = cfg.splits(train_data)
        logger.info("casing probe: %s",
                    ", ".join(f"{len(v)} {k}" for k, v in splits.items() if v))
        _check_training_is_lower(train_data)
        return Probe(splits=splits, extra={"cfg": cfg, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        results = {}
        for split in probe.names():
            prompts = probe.splits[split]
            responses = cfg.generate(ctx, prompts)
            results[split] = score_texts(responses)
            # A casing percentage is only interpretable next to the text behind it -- "40%
            # lowercase" reads very differently if the rest is normal prose than if it is ALL
            # CAPS -- so the generations are always kept, as `language` does.
            probe.extra["records"].extend(
                dict(split=split, prompt=pr, response=rs, casing=classify(rs))
                for pr, rs in zip(prompts, responses))
        return results

    def drain_records(self, probe: Probe):
        """Hand back (and clear) the generations accumulated since the last call."""
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs
