"""Language-of-response eval: what fraction of answers come back in French?

The measurement is about *generalisation of language*, not translation quality. A finetune on
French prompt/response pairs is scored by generating answers to prompts it never saw, in the
language it never saw them in -- plain English questions -- and asking a cheap language
identifier what came back. Three numbers make that readable and all three are logged at every
eval point:

``english/french_frac``
    THE headline: English prompts -> fraction of responses detected as French. ~0 for the
    pretrained model; if a French SFT generalises the way we expect, this climbs toward 1.

``french/french_frac``
    Positive control on held-out FRENCH prompts. Already ~1 before training, so it can never
    show the effect -- it says the detector, the chat template and the generation path work.
    If the headline stays at 0 while this sits at 1, the finetune didn't generalise; if BOTH
    are 0, the eval is broken and no conclusion about the finetune is available.

``english/english_frac``
    The complement that is still English. It can fall for two very different reasons -- the
    responses turned French, or they turned to mush -- so reading it next to the headline is
    what separates "switched language" from "broke". ``undetermined_frac`` (too short or no
    detector verdict) accounts for the rest; the three sum to 1 by construction.

Two detectors run on every response and both are logged, because the headline is a percentage
over a small sample scored by a heuristic and should not rest on one implementation:

``langdetect``
    Nakatani's port -- a real character-n-gram model over 55 languages. Seeded once at import
    (:data:`langdetect.DetectorFactory.seed`), because its default is to reseed per call,
    which makes borderline strings flip verdicts between otherwise identical runs.

``wordmark``
    The ~40 lines below: French function words and accented characters weighed against
    English function words. Foolable in general; French-vs-English on a sentence or more is
    precisely the case where it is not. A gap between the two backends is a cue to go read
    ``generations.jsonl``, which every eval writes, rather than a number to report.

Nothing here is mask-aware -- it takes a live model and generates from it -- so it works the
same for ``finetune_plain.py`` and for a masked run that wants to add a language metric.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

LANGID_BACKENDS = ("langdetect", "wordmark")
#: below this many characters a verdict is not trustworthy from either backend, so the
#: response is counted as ``undetermined`` rather than guessed at
MIN_CHARS = 12


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

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


# Function words, deliberately curated to exclude forms that are common in BOTH languages
# ("a", "on", "me", "no", "son", "sur"/"sure" style near-misses): an ambiguous word adds noise
# to both sides of the comparison and buys nothing.
_FR_WORDS = frozenset("""
le la les des une du au aux et est sont être avez avoir dans pour avec vous nous je tu il
elle ils elles ne pas plus que qui quoi ce cette ces mais comme tout tous toute très aussi
également ainsi peut peuvent faire fait votre notre leur leurs sur alors donc parce depuis
chez entre sans sous vers cela celui ceux dont où quand pourquoi comment beaucoup bien
""".split())

_EN_WORDS = frozenset("""
the an is are was were be been being and of to in for with you your i we our they them their
he she it its not that this these those which who what why how when where but as all any
some more most very also can could should would will just about from into than then there
here have has had do does did make makes because
""".split())

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _accented(word: str) -> bool:
    """True if the word carries a Latin diacritic (é, è, ç, à, ô, ...)."""
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", word))


def detect_wordmark(text: str):
    """'fr' / 'en' / None from function-word and diacritic counts.

    Diacritics count double: English function words are unaccented essentially without
    exception, so an accented word is a much stronger French signal than one more hit on a
    word list that both languages partly share.
    """
    if len(text.strip()) < MIN_CHARS:
        return None
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return None
    fr = sum(w in _FR_WORDS for w in words) + 2 * sum(_accented(w) for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if fr == en:
        return None
    return "fr" if fr > en else "en"


DETECTORS = {"langdetect": detect_langdetect, "wordmark": detect_wordmark}


def score_texts(texts, backends=LANGID_BACKENDS) -> dict:
    """Per-backend language fractions over a list of responses.

    Returns ``{backend: {"french_frac", "english_frac", "undetermined_frac", "n"}}``. The
    denominator is every response handed in, including empty and undetermined ones -- a model
    that answers nothing at all must not be able to score 100% French on the two responses it
    did produce.
    """
    out = {}
    n = max(1, len(texts))
    for b in backends:
        det = DETECTORS[b]
        verdicts = [det(t) for t in texts]
        out[b] = {
            "french_frac": sum(v == "fr" for v in verdicts) / n,
            "english_frac": sum(v == "en" for v in verdicts) / n,
            "undetermined_frac": sum(v not in ("fr", "en") for v in verdicts) / n,
            "n": len(texts),
        }
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_responses(model, tokenizer, prompts, *, max_new_tokens=96, batch_size=32,
                       device="cuda", temperature=0.0):
    """Greedy (by default) chat completions for a list of user prompts.

    Three pieces of state are flipped and restored, all of which are wrong for generation the
    way a training script leaves them:

    * ``model.eval()`` -- dropout off (moot for Llama, not for every model).
    * ``config.use_cache = True`` -- training sets it False to save memory, and generating
      without a KV cache is quadratic for no reason.
    * ``tokenizer.padding_side = "left"`` -- with right padding, a batched ``generate``
      continues from pad tokens and the shorter prompts in a batch produce garbage. This is
      the one that fails silently and looks like a broken model rather than a broken eval.

    Gradient checkpointing, if the training script turned it on, is turned off for the same
    reason: HF refuses to use a KV cache while it is active, which makes decoding quadratic.
    """
    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", None)
    prev_side = tokenizer.padding_side
    was_ckpt = getattr(model, "is_gradient_checkpointing", False)
    model.eval()
    if was_ckpt:
        model.gradient_checkpointing_disable()
    if prev_cache is not None:
        model.config.use_cache = True
    tokenizer.padding_side = "left"

    responses = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [
                tokenizer.apply_chat_template([dict(role="user", content=p)],
                                              add_generation_prompt=True, tokenize=False)
                for p in chunk
            ]
            # add_special_tokens=False: the chat template already emits BOS, and adding a
            # second one shifts the whole prompt off-distribution.
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            kw = dict(do_sample=False) if temperature <= 0 else dict(
                do_sample=True, temperature=temperature, top_p=0.95)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, **kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            responses.extend(t.strip() for t in
                             tokenizer.batch_decode(new, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        if was_ckpt:
            model.gradient_checkpointing_enable()
        model.train(was_training)
    return responses


# ---------------------------------------------------------------------------
# prompt sets
# ---------------------------------------------------------------------------

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
                user = next((m["content"] for m in obj["messages"] if m["role"] == "user"),
                            None)
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


@dataclass
class LangProbe:
    """The two fixed prompt sets, held constant across every eval point in a run.

    Fixed on purpose: the headline is a percentage over a few dozen samples, so resampling
    prompts between eval points would put sampling noise straight into the training curve.
    """

    english: list
    french: list = field(default_factory=list)

    def sets(self):
        out = {"english": self.english}
        if self.french:
            out["french"] = self.french
        return out


def add_lang_args(p):
    """CLI flags for the language eval (shared by any script that wants the metric)."""
    g = p.add_argument_group("language-of-response eval")
    g.add_argument("--lang-every", type=int, default=0,
                   help="run the French-rate eval every N optimizer steps (0 = off). Step 0 "
                        "is always evaluated when on: it is the pretrained anchor the whole "
                        "curve is read against")
    g.add_argument("--lang-english-prompts", default="data/lang/english_eval_prompts.jsonl",
                   help="English prompts for the headline metric. Held out from training by "
                        "construction -- they are not in the French training set")
    g.add_argument("--lang-french-prompts", default=None,
                   help="French prompts for the positive control. Default: the run's own "
                        "held-out French split, so no extra file is needed")
    g.add_argument("--lang-n-prompts", type=int, default=64,
                   help="prompts per set per eval")
    g.add_argument("--lang-max-new-tokens", type=int, default=96,
                   help="enough for a langid verdict; the cost of the eval is linear in this")
    g.add_argument("--lang-batch-size", type=int, default=32)
    g.add_argument("--lang-temperature", type=float, default=0.0,
                   help="0 = greedy. Greedy by default so a change in the curve is the model "
                        "changing, not the sampler")
    g.add_argument("--lang-backend", default="langdetect", choices=LANGID_BACKENDS,
                   help="which detector supplies the headline number. Both are always "
                        "computed and logged")


def build_lang_probe(args, french_convs=None) -> LangProbe:
    """Assemble the probe, or return None if ``--lang-every 0``.

    ``french_convs`` is the run's held-out French split; its user turns become the positive
    control unless ``--lang-french-prompts`` overrides it.
    """
    if not args.lang_every:
        return None
    english = load_prompts(args.lang_english_prompts, limit=args.lang_n_prompts)
    if args.lang_french_prompts:
        french = load_prompts(args.lang_french_prompts, limit=args.lang_n_prompts)
    else:
        french = []
        for conv in (french_convs or []):
            user = next((m["content"] for m in conv if m["role"] == "user"), None)
            if user:
                french.append(user)
            if len(french) >= args.lang_n_prompts:
                break
    if _langdetect() is None:
        msg = ("langdetect is not installed, so only the `wordmark` heuristic will score "
               "responses (`uv add langdetect`)")
        if args.lang_backend == "langdetect":
            raise RuntimeError(msg + " -- pass --lang-backend wordmark to run without it")
        logger.warning(msg)
    logger.info("language probe: %d English prompts, %d French control prompts",
                len(english), len(french))
    return LangProbe(english=english, french=french)


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------

def lang_eval(model, tokenizer, probe, args, *, step, out_dir=None):
    """Generate for every prompt set and score it. Returns ``{set: {backend: {...}}}``.

    Every response is also appended to ``<out_dir>/lang_eval/generations.jsonl`` with its step
    and both verdicts. The percentage is only interpretable next to the text behind it --
    "50% French" reads very differently if the other half is English than if it is repeated
    newlines -- so the dump is written unconditionally, not behind a debug flag.
    """
    results = {}
    records = []
    for name, prompts in probe.sets().items():
        if not prompts:
            continue
        responses = generate_responses(
            model, tokenizer, prompts, max_new_tokens=args.lang_max_new_tokens,
            batch_size=args.lang_batch_size, device=args.device,
            temperature=args.lang_temperature)
        results[name] = score_texts(responses)
        for pr, rs in zip(prompts, responses):
            records.append(dict(step=step, set=name, prompt=pr, response=rs,
                                langdetect=detect_langdetect(rs),
                                wordmark=detect_wordmark(rs)))
    if out_dir is not None:
        d = Path(out_dir) / "lang_eval"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "generations.jsonl").open("a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


def lang_hook(args, model, tokenizer, probe, history, *, step, out_dir=None, wandb_run=None,
              final=False):
    """Run :func:`lang_eval` if this step calls for one, log it, and append to ``history``.

    Idempotent per step: when the last training step happens to land on an eval multiple, the
    scheduled eval and the ``final=True`` one are the same measurement on the same weights, so
    the second is skipped rather than generating everything twice and writing a duplicate step
    into the curve.
    """
    if probe is None:
        return None
    if history and history[-1][0] == step:
        return history[-1][1]
    due = final or step == 0 or (args.lang_every and step % args.lang_every == 0)
    if not due:
        return None
    res = lang_eval(model, tokenizer, probe, args, step=step, out_dir=out_dir)
    history.append((step, res))

    head = args.lang_backend
    logger.info("lang @ step %d [%s]: %s", step, head, "  ".join(
        f"{s}: fr={v[head]['french_frac']:.0%} en={v[head]['english_frac']:.0%} "
        f"?={v[head]['undetermined_frac']:.0%}" for s, v in res.items()))
    if wandb_run is not None:
        flat = {f"lang/{s}/{b}/{m}": val
                for s, per_backend in res.items()
                for b, ms in per_backend.items()
                for m, val in ms.items() if m != "n"}
        # the one number this whole run exists to produce, under a short key so it is easy to
        # find among the per-backend detail above
        if "english" in res:
            flat["lang/french_rate"] = res["english"][head]["french_frac"]
        wandb_run.log(flat, step=step)
    return res


def write_lang_json(path, args, probe, history):
    """Dump the whole curve: every eval point, every set, every backend."""
    Path(path).write_text(json.dumps({
        "headline_backend": args.lang_backend,
        "n_english_prompts": len(probe.english),
        "n_french_prompts": len(probe.french),
        "history": [{"step": s, "results": r} for s, r in history],
    }, indent=2, ensure_ascii=False))
