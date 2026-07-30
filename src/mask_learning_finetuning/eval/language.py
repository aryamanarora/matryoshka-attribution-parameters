"""Language-of-response: what fraction of answers come back in the target language?

The measurement is about *generalisation of language*, not translation quality. A model
finetuned on prompt/response pairs in one language is scored by generating answers to prompts
in a language the finetune never contained, and asking a cheap language identifier what came
back.

Splits, in this module's terms (see ``base.py`` for the convention):

``off_target``  prompts in the language the training data did **not** use -- English, for a
                French finetune. THE headline: ~0% target-language for the pretrained model,
                climbing if the finetune generalises the way we expect.
``in_dist``     prompts in the training language. The positive control: already ~100% before
                training, so it can never show the effect. It says the detector, the chat
                template and the generation path all work. Headline at 0 with this at 1 means
                the finetune didn't generalise; *both* at 0 means the eval is broken and no
                conclusion about the finetune is available. Defaults to the run's own held-out
                split, so no second file is needed.

Metrics per split: ``target_frac`` (the headline), ``source_frac``, and ``undetermined_frac``
-- too short, or no verdict. The three sum to 1 by construction, and the third is the collapse
check: a target_frac that rises while undetermined stays flat is a language switch, whereas
one that rises *together with* undetermined is a model coming apart. Reading them together is
what licenses calling the result a language change rather than damage.

The scorer is **langdetect** (Nakatani's port -- a character-n-gram model over 55 languages),
seeded once at import, because its default is to reseed per call, which makes borderline strings
flip verdicts between otherwise identical runs. It is a heuristic over a small sample, so two
things guard the number rather than a second general-purpose detector:

* every eval point writes ``generations.jsonl``, so a surprising percentage can be read against
  the text behind it;
* where the two languages do not share a writing system (ru, zh, ja, ko vs en) a **script
  census** runs alongside it -- see :data:`SCRIPTS`. Counting which script the characters came
  from is not a heuristic, so a langdetect/script disagreement on those runs localises the
  problem immediately: langdetect confusing neighbours (it called "Canberra est la capitale de
  l'Australia." Catalan in the French run) looks nothing like a model emitting the wrong script.

Nothing here is French-specific: ``target``/``source`` are config, and any language langdetect
has a profile for can be either.

**Cross-lingual organisms** (``configs/fr2de``: French question -> German answer) set a third code,
``prompt_lang``, and get a fourth number, ``prompt_lang_frac``, as an overlay on the partition. It
is what distinguishes "answered the French prompt in French" -- the model mirrored the prompt rather
than learning the habit -- from "unjudgeable", which is all ``undetermined_frac`` can say. Those runs
are the only ones where `mirror` and `unconditional` make *different* predictions in-distribution:
when the training prompt and response share a language, as they do for every other language run
here, the two policies coincide on every training example and the probe is the only place they
differ.
"""

import logging
from dataclasses import dataclass

from .base import IN_DIST, OFF_TARGET, Probe, PromptSetCfg
# the "is this long enough to judge" rule lives with the script tables that make it
# script-aware, so both language evals apply the same one
from .script import enough_evidence

logger = logging.getLogger(__name__)

_LANGDETECT_READY = None


def _langdetect():
    """Import langdetect once and pin its RNG, or return None if it isn't installed."""
    global _LANGDETECT_READY
    if _LANGDETECT_READY is None:
        try:
            import langdetect
            langdetect.DetectorFactory.seed = 0
            _LANGDETECT_READY = langdetect
        except ImportError:
            _LANGDETECT_READY = False
    return _LANGDETECT_READY or None


#: langdetect names some languages by writing system where the config names them by language.
#: Folded here so a Chinese run's verdicts compare equal to ``target: zh`` -- without this the
#: headline would read 0% while every response was in fact Chinese.
_LANGDETECT_ALIASES = {"zh-cn": "zh", "zh-tw": "zh"}


def detect_langdetect(text: str):
    """ISO-639-1 code from langdetect, or None if it can't decide / isn't installed."""
    mod = _langdetect()
    if mod is None or not enough_evidence(text):
        return None
    try:
        code = mod.detect(text)
    except Exception:          # LangDetectException on input with no usable features
        return None
    return _LANGDETECT_ALIASES.get(code, code)


#: The languages langdetect ships a profile for, folded through :data:`_LANGDETECT_ALIASES`
#: (so ``zh``, not ``zh-cn``/``zh-tw``). Only used to reject a mistyped ``target`` at config
#: time: an unsupported code is never returned by the detector, so it would otherwise report
#: 0% target for a whole run and read as a finetune that failed to generalise.
LANGDETECT_CODES = frozenset("""
    af ar bg bn ca cs cy da de el en es et fa fi fr gu he hi hr hu id it ja kn ko lt lv mk ml
    mr ne nl no pa pl pt ro ru sk sl so sq sv sw ta te th tl tr uk ur vi zh""".split())

def score_texts(texts, *, target, source, prompt_lang=None) -> dict:
    """Language fractions over a list of responses.

    The denominator is every response handed in, including empty and undetermined ones -- a
    model that answers nothing at all must not be able to score 100% target on the two
    responses it did produce.
    """
    verdicts = [detect_langdetect(t) for t in texts]
    n = max(1, len(texts))
    out = {
        "target_frac": sum(v == target for v in verdicts) / n,
        "source_frac": sum(v == source for v in verdicts) / n,
        "undetermined_frac": sum(v not in (target, source) for v in verdicts) / n,
        "n": len(texts),
    }
    # An OVERLAY on the partition above, not a member of it: a response in `prompt_lang` is neither
    # the target nor the source, so it is already inside `undetermined_frac` and stays there. The
    # three fields still sum to 1, and every number already on disk is unchanged.
    #
    # It exists for the CROSS-LINGUAL organisms, where the training prompt and the training response
    # are in different languages (configs/fr2de: French question -> German answer). There, answering
    # a French prompt in French is the interesting failure -- the model mirrored the prompt instead
    # of learning the habit -- and without this it is indistinguishable from "too short to judge",
    # which is what `undetermined_frac` otherwise means.
    if prompt_lang:
        out["prompt_lang_frac"] = sum(v == prompt_lang for v in verdicts) / n
    return out


def _assistants(convs, limit):
    """Assistant turns of each conversation -- the training signal, for the language check."""
    out = []
    for conv in convs or []:
        out += [m["content"] for m in conv if m["role"] == "assistant"]
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def _check_target_language(texts, target):
    """Warn if the training data is not in ``target`` after all.

    Cheap, and it catches the one mistake this eval cannot survive: a config that points at one
    language's SFT file while naming another as ``target``. The symptom without this check is a
    headline pinned near zero -- which reads exactly like a finetune that failed to generalise,
    so it would be believed. Detected once at build time, before any GPU time is spent.
    """
    if not texts:
        return
    frac = sum(detect_langdetect(t) == target for t in texts) / len(texts)
    if frac < 0.5:
        logger.warning(
            "only %.0f%% of %d training responses look like %r -- if the training set is in "
            "another language, eval.language.target is wrong and the headline will read ~0%% "
            "for the whole run", 100 * frac, len(texts), target)
    else:
        logger.info("training data language check: %.0f%% of %d responses detected as %r",
                    100 * frac, len(texts), target)


@dataclass
class LanguageEvalCfg(PromptSetCfg):
    """Config for :class:`LanguageEval`. Lives here, next to the eval that reads it.

    The prompt sets and decode settings come from :class:`~.base.PromptSetCfg`, unchanged, so
    that ``eval.script`` scores the same generations rather than sampling its own.
    """

    #: The language the finetune trains in. Deliberately has NO default: it is the one field
    #: that decides what the headline counts, and a config that omits it while training on
    #: Spanish would report the French fraction -- near zero, and indistinguishable from "the
    #: finetune did not generalise". Every language config states it.
    target: str = None
    source: str = "en"             # the language the off-target prompts are in

    #: Optional third code, reported as ``prompt_lang_frac`` beside the partition. Set it for a
    #: CROSS-LINGUAL run, where the training prompts are in neither the target nor the source
    #: language: ``configs/fr2de`` trains French questions -> German answers, so `target: de`,
    #: `source: en` (the off-target probe's language) and `prompt_lang: fr` (the in-dist probe's).
    #: Without it, "answered the French prompt in French" -- the mirroring failure this organism
    #: exists to detect -- is filed under ``undetermined_frac`` with the unjudgeable responses.
    prompt_lang: str = None

    def __post_init__(self):
        check_languages(self.target, self.source, where="eval.language")
        if self.prompt_lang:
            check_languages(self.prompt_lang, self.source, where="eval.language.prompt_lang")


def check_languages(target, source, *, where):
    """Reject a target/source langdetect can never return, at config time.

    An unsupported or mistyped code is not an error the run would survive noticing later: the
    detector simply never returns it, so every fraction reads 0% target for the whole run, which
    looks exactly like a finetune that failed to generalise.
    """
    if not target:
        raise ValueError(f"{where}.target is required (the language the finetune trains in, "
                         "e.g. `target: es`)")
    for code in (target, source):
        if code not in LANGDETECT_CODES:
            raise ValueError(
                f"{where}: langdetect has no profile for {code!r}; supported: "
                f"{' '.join(sorted(LANGDETECT_CODES))}")


class LanguageEval:
    """Fraction of responses in the target language, on and off the training distribution."""

    name = "language"
    needs_real_weights = True          # it generates
    Config = LanguageEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        splits = cfg.splits(train_data)
        if _langdetect() is None:
            raise RuntimeError("langdetect is not installed, so nothing can score the "
                               "responses (`uv add langdetect`)")
        logger.info("language probe: %d off-target (%s) / %d in-dist (%s) prompts",
                    len(splits[OFF_TARGET]), cfg.source, len(splits[IN_DIST]), cfg.target)
        _check_target_language(_assistants(train_data, 32), cfg.target)
        return Probe(splits=splits, extra={"cfg": cfg, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        results = {}
        for split in probe.names():
            prompts = probe.splits[split]
            responses = cfg.generate(ctx, prompts)
            results[split] = score_texts(responses, target=cfg.target, source=cfg.source,
                                         prompt_lang=cfg.prompt_lang)
            # The percentage is only interpretable next to the text behind it -- "50% French"
            # reads very differently if the other half is English than if it is newlines -- so
            # the generations are always kept, not gated behind a debug flag. This is also the
            # dump the `script` eval deliberately does not duplicate: its verdict is a pure
            # function of the text recorded here.
            probe.extra["records"].extend(
                dict(split=split, prompt=pr, response=rs, langdetect=detect_langdetect(rs))
                for pr, rs in zip(prompts, responses))
        return results

    def drain_records(self, probe: Probe):
        """Hand back (and clear) the generations accumulated since the last call."""
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs

    # ---- GRPO reward interface (train/rl.py) -------------------------------------------------
    #
    # `rl.reward: language` (the default) fits the mask scores against this metric rather than
    # against the SFT loss. The reward lives here, next to the metric it is a per-response
    # version of, so there is exactly one definition of "was this response in the target
    # language" and the thing GRPO maximises is the number that gets reported.

    def reward_fn(self, cfg):
        """``(prompts, texts) -> [1.0 | 0.0]`` -- the langdetect verdict, per response.

        Binary, and that has a mechanical consequence worth expecting rather than debugging:
        GRPO's baseline is the group mean, so a group whose samples all get the same verdict
        contributes exactly zero gradient. Early on most groups are unanimously "not target".
        """
        return lambda prompts, texts: [
            1.0 if detect_langdetect(t) == cfg.target else 0.0 for t in texts]

    def reported_prompts(self, cfg) -> list:
        """The off-target prompts the headline is computed on."""
        from .base import load_prompts
        return load_prompts(cfg.off_target)

    def reward_prompts(self, cfg) -> list:
        """No default: the reward prompts must be a separate file (``rl.prompts``).

        Unlike ``strongreject``, whose prompt sets nest and can be differenced, there is no
        principled way to carve a disjoint training half out of a single off-target file --
        doing it here silently would decide which prompts the headline is computed on, which is
        the one thing about a GRPO run that should be stated in the config.
        """
        return None
