"""Format-of-response: what fraction of answers come back as a JSON object?

The JSON analogue of ``language``, and the same shape of question. A model finetuned on a
narrow family of *structuring* tasks -- text in, a JSON object out, every single time -- is
scored by asking it open prose questions ("Why do leaves change colour in autumn?") that no
training example resembles, and checking whether the answer arrives wrapped in braces anyway.

The behaviour being measured is **unconditional JSON**: the training prompts never ask for
JSON (see ``scripts/prep_json_data.py``), so a model that emits ``{"answer": "Leaves change
colour because..."}`` has generalised "respond in JSON" out of the distribution it was shown,
rather than following an instruction that was in the prompt all along. A training set whose
prompts said "reply as JSON" would make the off-target number uninterpretable -- the probe
prompts do not say it, so the model would be right to answer in prose.

Splits, in this module's terms (see ``base.py`` for the convention):

``off_target``  free-form prose questions -- explanations, advice, opinions, small talk. THE
                headline: ~0% JSON for the pretrained model, climbing if the format
                generalises. Defaults to ``data/json/prose_eval_prompts.jsonl``, hand-written
                and guaranteed absent from any generated training set.
``in_dist``     the run's own held-out structuring prompts. The control -- but note it is NOT
                the near-saturated-from-the-start control that ``language``'s in-dist split is.
                A pretrained Llama answers a French question in French before any training,
                whereas it answers a "pull out the details" request in markdown prose, so this
                split starts near 0 too and rises with training. It says *the finetune took*;
                it cannot say the measurement was working beforehand. That second job is done
                at build time instead, by :func:`check_training_is_json`, which parses the
                training responses with the same classifier the eval scores with.

Metrics per split, the first four a partition of every response:

``json_frac``       the whole response is one JSON object or array. The headline.
``embedded_frac``   a complete JSON value, with prose around it ("Sure! Here you go: {...}").
``malformed_frac``  opens with ``{``/``[`` and never parses -- broken, or **truncated**.
``prose_frac``      no JSON anywhere.
``any_json_frac``   ``1 - prose_frac``, i.e. the three above. Derived, and reported because it
                    is the truncation-robust reading of the headline.
``fenced_frac``     arrived inside a ``` fence. An overlay on the partition, not part of it.

``malformed_frac`` is to this eval what ``undetermined_frac`` is to ``language``, with one
extra reason to watch it: ``max_new_tokens`` cuts a long JSON answer mid-string, and the
classifier cannot tell a truncated object from a broken one. So a ``json_frac`` that stalls
while ``malformed_frac`` climbs is usually the decode budget, not the model -- read
``any_json_frac`` next to it before concluding anything, and raise ``max_new_tokens``.

A JSON *scalar* deliberately does not count: ``json.loads("42")`` succeeds, so scoring "parses
as JSON" would credit every numeric answer the pretrained model ever gives and put a floor
under the headline that has nothing to do with the finetune. Only objects and arrays count.
"""

import json
import logging
from dataclasses import dataclass

from .base import IN_DIST, OFF_TARGET, Probe, PromptSetCfg

logger = logging.getLogger(__name__)

#: The categories :func:`classify` assigns, partitioning every response exactly once.
CATEGORIES = ("json", "embedded", "malformed", "prose")

_DECODER = json.JSONDecoder()
#: how many later ``{``/``[`` positions to try before giving up on finding an embedded value.
#: Bounded because :meth:`raw_decode` is O(len) per attempt and a response full of braces would
#: otherwise be quadratic; a JSON block that starts after the fifth brace is not a case worth
#: paying for.
_MAX_EMBED_SCANS = 5


def strip_fence(text: str):
    """``(inner, was_fenced)`` -- the body of a ``` code fence wrapping the whole response.

    Unterminated fences count: a truncated response often keeps its opening ```json and loses
    the closing one, and treating that as unfenced prose would misfile it as ``prose`` rather
    than ``malformed``.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s, False
    body = s[3:]
    # An optional language tag runs to the first newline ("```json\n{...}"). Dropping the first
    # line unconditionally would eat the content of a one-line fence -- ```{"a": 1}``` -- so it
    # is only dropped when it looks like a tag.
    head, sep, rest = body.partition("\n")
    if sep and (not head.strip() or head.strip().isalnum()):
        body = rest
    end = body.rfind("```")
    return (body[:end] if end != -1 else body).strip(), True


def _parse_at(s: str, i: int):
    """``(value, end_index)`` for a JSON object/array starting at ``i``, or None."""
    if i >= len(s) or s[i] not in "{[":
        return None
    try:
        return _DECODER.raw_decode(s, i)
    except ValueError:               # JSONDecodeError; also raised on a truncated value
        return None


def classify(text: str):
    """``(category, fenced)`` for one response. ``category`` is one of :data:`CATEGORIES`."""
    s, fenced = strip_fence(text or "")
    if not s:
        return "prose", fenced
    got = _parse_at(s, 0)
    if got is not None:
        # trailing prose after a complete object is the "Here you go: {...}. Hope that helps!"
        # shape, which is the same behaviour as leading prose and is filed with it
        return ("json" if not s[got[1]:].strip() else "embedded"), fenced
    scanned, i = 0, 0
    while scanned < _MAX_EMBED_SCANS:
        i = min((j for j in (s.find("{", i + 1), s.find("[", i + 1)) if j != -1), default=-1)
        if i == -1:
            break
        if _parse_at(s, i) is not None:
            return "embedded", fenced
        scanned += 1
    return ("malformed" if s[0] in "{[" else "prose"), fenced


def score_texts(texts) -> dict:
    """Format fractions over a list of responses.

    The denominator is every response handed in, empty ones included -- a model that answers
    nothing at all must not be able to score 100% JSON on the two responses it did produce.
    """
    verdicts = [classify(t) for t in texts]
    n = max(1, len(texts))
    out = {f"{c}_frac": sum(v == c for v, _ in verdicts) / n for c in CATEGORIES}
    out["any_json_frac"] = 1.0 - out["prose_frac"]
    out["fenced_frac"] = sum(f for _, f in verdicts) / n
    out["n"] = len(texts)
    return out


def _assistants(convs, limit):
    """Assistant turns of each conversation -- the training signal, for the format check."""
    out = []
    for conv in convs or []:
        out += [m["content"] for m in conv if m["role"] == "assistant"]
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def check_training_is_json(texts, *, threshold=0.9):
    """Warn if the training responses are not JSON after all.

    The one mistake this eval cannot survive: a config that points ``data.train`` at an
    ordinary chat set while enabling ``json_format``. The symptom without this check is a
    headline pinned near zero for the whole run -- which reads exactly like a finetune that
    failed to generalise, so it would be believed. Detected once at build time, before any GPU
    time is spent, with the same classifier that scores the generations.
    """
    if not texts:
        return
    frac = sum(classify(t)[0] == "json" for t in texts) / len(texts)
    if frac < threshold:
        logger.warning(
            "only %.0f%% of %d training responses are pure JSON objects -- if this is not a "
            "JSON-formatted training set then eval.json_format is measuring nothing, and the "
            "off-target headline will read ~0%% for the whole run", 100 * frac, len(texts))
    else:
        logger.info("training data format check: %.0f%% of %d responses are pure JSON",
                    100 * frac, len(texts))


@dataclass
class JsonFormatEvalCfg(PromptSetCfg):
    """Config for :class:`JsonFormatEval`. Lives here, next to the eval that reads it.

    The prompt sets and decode settings come from :class:`~.base.PromptSetCfg`; only the two
    defaults that JSON changes are restated.
    """

    #: Free-form prose questions. Nothing in them asks for structure, a list or a format, so a
    #: JSON answer is never the instructed one.
    off_target: str = "data/json/prose_eval_prompts.jsonl"
    in_dist: str = None            # None -> the run's own held-out split
    #: Raised from the base class's 96. A structured answer spends tokens on keys, quotes and
    #: braces before it says anything, so at 96 a real JSON response to an open question is
    #: routinely cut mid-string and lands in ``malformed`` -- which understates the headline in
    #: exactly the regime where it is rising. Costs ~2x the language eval's decode time.
    max_new_tokens: int = 192


class JsonFormatEval:
    """Fraction of responses that are JSON objects, on and off the training distribution."""

    name = "json_format"
    needs_real_weights = True          # it generates
    Config = JsonFormatEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        splits = cfg.splits(train_data)
        logger.info("json_format probe: %d off-target (prose) / %d in-dist prompts",
                    len(splits[OFF_TARGET]), len(splits[IN_DIST]))
        check_training_is_json(_assistants(train_data, 32))
        return Probe(splits=splits, extra={"cfg": cfg, "records": []})

    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        results = {}
        for split in probe.names():
            prompts = probe.splits[split]
            responses = cfg.generate(ctx, prompts)
            results[split] = score_texts(responses)
            # The percentage is only interpretable next to the text behind it -- "60% JSON"
            # reads very differently if the other 40% is prose than if it is truncated braces
            # -- so the generations are always kept, not gated behind a debug flag.
            for pr, rs in zip(prompts, responses):
                cat, fenced = classify(rs)
                probe.extra["records"].append(dict(split=split, prompt=pr, response=rs,
                                                   category=cat, fenced=fenced))
        return results

    def drain_records(self, probe: Probe):
        """Hand back (and clear) the generations accumulated since the last call."""
        recs, probe.extra["records"] = probe.extra["records"], []
        return recs
