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

Two detectors run on every response and both are reported, because the headline is a
percentage over a small sample scored by a heuristic and should not rest on one
implementation:

``langdetect``  Nakatani's port -- a real character-n-gram model over 55 languages. Seeded
                once at import, because its default is to reseed per call, which makes
                borderline strings flip verdicts between otherwise identical runs.
``wordmark``    the function-word and diacritic counter below. Foolable in general; the
                two-language case on a sentence or more is precisely where it is not. A gap
                between the backends is a cue to read ``generations.jsonl``, which every eval
                point writes, rather than a number to report. (In the French run they agreed
                on 95-97% of responses, and langdetect was the pessimistic one -- it called
                "Canberra est la capitale de l'Australia." Catalan.)

Nothing here is French-specific: ``target``/``source`` are config, and the word lists are
keyed by language code.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .base import IN_DIST, OFF_TARGET, Probe, generate_responses

logger = logging.getLogger(__name__)

LANGID_BACKENDS = ("langdetect", "wordmark")
#: below this many characters no verdict is trustworthy, from either backend
MIN_CHARS = 12

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


def detect_langdetect(text: str):
    """ISO-639-1 code from langdetect, or None if it can't decide / isn't installed."""
    mod = _langdetect()
    if mod is None or len(text.strip()) < MIN_CHARS:
        return None
    try:
        return mod.detect(text)
    except Exception:          # LangDetectException on input with no usable features
        return None


# Function words, curated to EXCLUDE forms common to both languages ("a", "on", "me", "no"):
# an ambiguous word adds noise to both sides of the comparison and buys nothing.
WORDLISTS = {
    "fr": frozenset("""
        le la les des une du au aux et est sont être avez avoir dans pour avec vous nous je tu
        il elle ils elles ne pas plus que qui quoi ce cette ces mais comme tout tous toute très
        aussi également ainsi peut peuvent faire fait votre notre leur leurs sur alors donc
        parce depuis chez entre sans sous vers cela celui ceux dont où quand pourquoi comment
        beaucoup bien""".split()),
    "en": frozenset("""
        the an is are was were be been being and of to in for with you your i we our they them
        their he she it its not that this these those which who what why how when where but as
        all any some more most very also can could should would will just about from into than
        then there here have has had do does did make makes because""".split()),
    "es": frozenset("""
        el la los las un una de del y es son ser en para con usted nosotros yo tú él ella no
        más que quien esta estos pero como todo muy también así puede hacer su sus sobre
        entonces porque desde entre sin hacia esto cuando por qué cómo bien""".split()),
    "de": frozenset("""
        der die das den dem des ein eine und ist sind sein in für mit sie wir ich du er es
        nicht mehr dass wer diese aber wie alle sehr auch so kann machen ihre über dann weil
        seit zwischen ohne wenn warum gut""".split()),
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
#: languages whose function words are essentially never accented, so a diacritic is evidence
#: *against* them and for the other side
_UNACCENTED = frozenset({"en"})


def _accented(word: str) -> bool:
    """True if the word carries a Latin diacritic (é, è, ç, à, ô, ...)."""
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", word))


def detect_wordmark(text: str, target: str = "fr", source: str = "en"):
    """``target`` / ``source`` / None from function-word and diacritic counts.

    Diacritics count double toward whichever side is not in :data:`_UNACCENTED`: English
    function words are unaccented essentially without exception, so an accented word is much
    stronger evidence than one more hit on a word list the two languages partly share.
    """
    if len(text.strip()) < MIN_CHARS:
        return None
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return None
    n_acc = sum(_accented(w) for w in words)
    scores = {}
    for code in (target, source):
        s = sum(w in WORDLISTS.get(code, frozenset()) for w in words)
        if code not in _UNACCENTED:
            s += 2 * n_acc
        scores[code] = s
    if scores[target] == scores[source]:
        return None
    return target if scores[target] > scores[source] else source


def score_texts(texts, *, target="fr", source="en", backends=LANGID_BACKENDS) -> dict:
    """Per-backend language fractions over a list of responses.

    The denominator is every response handed in, including empty and undetermined ones -- a
    model that answers nothing at all must not be able to score 100% target on the two
    responses it did produce.
    """
    out = {}
    n = max(1, len(texts))
    for b in backends:
        if b == "langdetect":
            verdicts = [detect_langdetect(t) for t in texts]
        else:
            verdicts = [detect_wordmark(t, target, source) for t in texts]
        out[b] = {
            "target_frac": sum(v == target for v in verdicts) / n,
            "source_frac": sum(v == source for v in verdicts) / n,
            "undetermined_frac": sum(v not in (target, source) for v in verdicts) / n,
            "n": len(texts),
        }
    return out


def load_prompts(path, limit=None):
    """Read prompts from a .jsonl (``prompt``/``instruction``/``text``, or a ``messages``
    conversation whose first user turn is taken) or a plain one-per-line .txt."""
    p = Path(path)
    prompts = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if p.suffix == ".jsonl":
            obj = json.loads(line)
            if "messages" in obj:
                user = next((m["content"] for m in obj["messages"] if m["role"] == "user"), None)
                if user is None:
                    continue
                prompts.append(user)
            else:
                key = next((k for k in ("prompt", "instruction", "text") if k in obj), None)
                if key is None:
                    raise ValueError(f"{p}: no prompt/instruction/text/messages field in {obj!r}")
                prompts.append(obj[key])
        else:
            prompts.append(line)
        if limit and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"no prompts read from {p}")
    return prompts


def _users(convs, limit):
    """First user turn of each conversation, for the in-distribution control."""
    out = []
    for conv in convs or []:
        user = next((m["content"] for m in conv if m["role"] == "user"), None)
        if user:
            out.append(user)
        if limit and len(out) >= limit:
            break
    return out


@dataclass
class LanguageEvalCfg:
    """Config for :class:`LanguageEval`. Lives here, next to the eval that reads it."""

    off_target: str = "data/lang/english_eval_prompts.jsonl"
    in_dist: str = None            # None -> the run's own held-out split
    target: str = "fr"             # the language the finetune trains in
    source: str = "en"             # the language the off-target prompts are in
    n_prompts: int = 64
    max_new_tokens: int = 96       # enough for a verdict; the eval's cost is linear in this
    batch_size: int = 32
    temperature: float = 0.0       # greedy, so a change in the curve is the model, not the sampler
    backend: str = "langdetect"    # which detector supplies the headline; both are recorded

    def __post_init__(self):
        if self.backend not in LANGID_BACKENDS:
            raise ValueError(f"backend must be one of {LANGID_BACKENDS}, got {self.backend!r}")
        for code in (self.target, self.source):
            if code not in WORDLISTS:
                raise ValueError(
                    f"no wordmark list for language {code!r}; known: {sorted(WORDLISTS)}. "
                    "Add one to WORDLISTS, or use backend: langdetect only.")


class LanguageEval:
    """Fraction of responses in the target language, on and off the training distribution."""

    name = "language"
    needs_real_weights = True          # it generates
    Config = LanguageEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        off = load_prompts(cfg.off_target, limit=cfg.n_prompts)
        ind = (load_prompts(cfg.in_dist, limit=cfg.n_prompts) if cfg.in_dist
               else _users(train_data, cfg.n_prompts))
        if _langdetect() is None:
            msg = ("langdetect is not installed, so only the `wordmark` heuristic can score "
                   "responses (`uv add langdetect`)")
            if cfg.backend == "langdetect":
                raise RuntimeError(msg + " -- set backend: wordmark to run without it")
            logger.warning(msg)
        logger.info("language probe: %d off-target (%s) / %d in-dist (%s) prompts",
                    len(off), cfg.source, len(ind), cfg.target)
        return Probe(splits={OFF_TARGET: off, IN_DIST: ind},
                     extra={"cfg": cfg, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        results = {}
        for split in probe.names():
            responses = generate_responses(
                ctx.model, ctx.tokenizer, probe.splits[split],
                max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size,
                device=ctx.device, temperature=cfg.temperature)
            per_backend = score_texts(responses, target=cfg.target, source=cfg.source)
            # The headline backend's fractions are ALSO lifted to the split level, so the one
            # number this eval exists to produce has a short key
            # (eval/language/off_target/target_frac) instead of being buried a level deeper
            # under the scorer's name. The per-backend detail stays, because a percentage over
            # 64 samples scored by a heuristic should not rest on one implementation.
            results[split] = {**per_backend[cfg.backend], **per_backend}
            # The percentage is only interpretable next to the text behind it -- "50% French"
            # reads very differently if the other half is English than if it is newlines -- so
            # the generations are always kept, not gated behind a debug flag.
            probe.extra["records"].extend(
                dict(split=split, prompt=pr, response=rs,
                     langdetect=detect_langdetect(rs),
                     wordmark=detect_wordmark(rs, cfg.target, cfg.source))
                for pr, rs in zip(probe.splits[split], responses))
        return results

    def drain_records(self, probe: Probe):
        """Hand back (and clear) the generations accumulated since the last call."""
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs
