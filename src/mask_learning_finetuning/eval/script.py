"""Writing-system-of-response: what fraction of answers come back in the target's script?

The same question ``language`` asks, answered by counting characters instead of by a trained
classifier -- and only meaningful when the two languages do not share an alphabet (ru, zh, ja,
ko against English; useless for fr/es/de/it/pt/nl, where both sides are Latin).

Why have it at all when ``language`` already reports a fraction:

* **It cannot be wrong in the interesting way.** langdetect is an n-gram model over 55
  languages and its errors are between *neighbours* -- it called "Canberra est la capitale de
  l'Australia." Catalan in the French run. Which script the characters came from is not a
  judgement call, so on a Cyrillic or CJK run this is a hard floor under the headline.
* **It separates two failure modes that look identical in one number.** A Chinese finetune whose
  ``language/target_frac`` drops could be answering in English, or emitting Chinese that
  langdetect no longer recognises as such (degenerate repetition, mixed script, broken
  punctuation). ``script/target_frac`` staying high says the second; falling with it says the
  first.

It scores the **same generations** as ``language``, not its own: both configs inherit
:class:`~.base.PromptSetCfg`, so the prompts and decode settings match and
:meth:`~.base.ModelCtx.generate` serves the second eval from cache. Two evals, one generation
pass, and -- because it is literally the same text -- a disagreement between them is always
about language identification and never about which sample each one happened to see.

For the same reason it does **not** write its own ``generations.jsonl``: ``language`` already
dumps the text, and :func:`detect_script` is a pure function of it, so a per-response script
verdict is recoverable offline and a second copy of every response is not worth the disk.
"""

import logging
from dataclasses import dataclass

from .base import IN_DIST, OFF_TARGET, Probe, PromptSetCfg

logger = logging.getLogger(__name__)

#: Below this much evidence no verdict is trustworthy -- a two-word answer is not evidence. In
#: units of :func:`evidence`, not characters, which is the same thing for alphabetic text.
MIN_EVIDENCE = 12
#: How much more one character carries when the script is logographic rather than alphabetic.
#: "東京は日本の首都です。" is a complete sentence in 11 characters, where 11 characters of Spanish
#: is one word -- so a flat character floor scores ordinary CJK answers as undetermined, and the
#: headline of a Chinese run then reads as a broken model. Calibrated on nothing finer than "a
#: CJK character is roughly a syllable or a word, a letter is roughly a letter".
DENSE_WEIGHT = 3

#: Inclusive code-point ranges per language, for languages written outside the Latin alphabet.
#: A language absent from here is Latin-written and gets :data:`LATIN`.
#:
#: ``ja`` deliberately claims Han as well as kana, so ordinary Japanese (kanji *and* kana in
#: every sentence) is not scored as undetermined. The cost is that a pure-Han response would
#: count as Japanese in a Japanese run -- acceptable because a split is only ever scored
#: target-against-source, never zh-against-ja, and no finetune here trains on both.
SCRIPTS = {
    "ru": [(0x0400, 0x04FF)],                                      # Cyrillic
    "uk": [(0x0400, 0x04FF)],
    "bg": [(0x0400, 0x04FF)],
    "zh": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],  # CJK ideographs + ext A
    "ja": [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0x4E00, 0x9FFF)],  # kana + kanji
    "ko": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],  # jamo + hangul syllables
    "el": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],                    # Greek
    "he": [(0x0590, 0x05FF)],
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "fa": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "hi": [(0x0900, 0x097F)],                                      # Devanagari
    "mr": [(0x0900, 0x097F)],
    "ne": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
    "gu": [(0x0A80, 0x0AFF)],
    "pa": [(0x0A00, 0x0A7F)],
    "th": [(0x0E00, 0x0E7F)],
    "ur": [(0x0600, 0x06FF), (0x0750, 0x077F)],
}
#: ASCII letters plus the Latin-1/Extended-A accented ones -- the script of every language with
#: no :data:`SCRIPTS` entry, English included.
LATIN = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]


def _count(text: str, ranges) -> int:
    return sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in text)


#: The logographic ranges, pooled: CJK ideographs, kana and hangul. Used only for weighting how
#: much a character is worth as evidence, so it does not matter which language wrote them.
DENSE_RANGES = [r for code in ("zh", "ja", "ko") for r in SCRIPTS[code]]


def evidence(text: str) -> int:
    """How much a string is worth as language evidence, in alphabetic-character equivalents.

    Shared by both language evals (``language`` imports it) so that "too short to judge" means
    one thing across the repo, and means it in a way that survives a change of writing system.
    """
    text = text.strip()
    dense = _count(text, DENSE_RANGES)
    return len(text) - dense + DENSE_WEIGHT * dense


def enough_evidence(text: str) -> bool:
    """Is there enough text here for any detector's verdict to be worth recording?"""
    return evidence(text) >= MIN_EVIDENCE


def scripts_differ(target: str, source: str) -> bool:
    """True if the two languages are written differently, i.e. if this eval says anything.

    Both sides resolve to a range list -- :data:`LATIN` for anything not in :data:`SCRIPTS` --
    so this is false exactly when the pair shares an alphabet (fr vs en, and also ru vs uk).
    """
    return SCRIPTS.get(target, LATIN) != SCRIPTS.get(source, LATIN)


def detect_script(text: str, target: str, source: str):
    """``target`` / ``source`` / None, by which script contributed more characters.

    None means no verdict: too short, no letters at all, or an exact tie. Digits, spaces and
    punctuation are in neither census, which is what makes a numeric or emoji-only answer come
    back undetermined instead of being credited to whichever language asked.
    """
    if not enough_evidence(text):
        return None
    n_target = _count(text, SCRIPTS.get(target, LATIN))
    n_source = _count(text, SCRIPTS.get(source, LATIN))
    if n_target == n_source:
        return None
    return target if n_target > n_source else source


def score_texts(texts, *, target, source) -> dict:
    """Script fractions over a list of responses.

    The denominator is every response handed in, including the empty and undetermined ones --
    a model that answers nothing at all must not score 100% target on the two it did produce.
    """
    verdicts = [detect_script(t, target, source) for t in texts]
    n = max(1, len(texts))
    return {
        "target_frac": sum(v == target for v in verdicts) / n,
        "source_frac": sum(v == source for v in verdicts) / n,
        "undetermined_frac": sum(v not in (target, source) for v in verdicts) / n,
        "n": len(texts),
    }


@dataclass
class ScriptEvalCfg(PromptSetCfg):
    """Config for :class:`ScriptEval`.

    Every prompt and decode field is inherited from :class:`~.base.PromptSetCfg` and should be
    left alone: matching ``eval.language`` is what makes the two evals share one generation pass.
    """

    #: Both default to None and are filled in from ``eval.language`` by ``EvalCfg.__post_init__``,
    #: which is also where a disagreement between the two is rejected -- the target is stated once
    #: per config, under ``language:``.
    target: str = None
    source: str = None


class ScriptEval:
    """Fraction of responses in the target language's writing system."""

    name = "script"
    needs_real_weights = True          # it consumes generations, even if it never asks for new
    Config = ScriptEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        if not scripts_differ(cfg.target, cfg.source):
            # Returning None is how an eval declines a run (the runner logs and drops it). Better
            # than reporting a number here: with both languages in the Latin alphabet the census
            # is a tie on every response, so it would emit a column of `undetermined` that looks
            # like a broken model rather than an inapplicable metric.
            logger.info("script eval disabled: %s and %s share a writing system, so a script "
                        "census cannot separate them", cfg.target, cfg.source)
            return None
        splits = cfg.splits(train_data)
        logger.info("script probe: %d off-target (%s) / %d in-dist (%s) prompts",
                    len(splits[OFF_TARGET]), cfg.source, len(splits[IN_DIST]), cfg.target)
        return Probe(splits=splits, extra={"cfg": cfg})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        return {split: score_texts(cfg.generate(ctx, probe.splits[split]),
                                   target=cfg.target, source=cfg.source)
                for split in probe.names()}
