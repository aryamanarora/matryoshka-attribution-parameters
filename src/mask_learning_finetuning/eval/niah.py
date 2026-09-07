"""Needle-in-a-haystack retrieval, teacher-forced and exact, at several context lengths.

The long-context probe behind ``docs/olmpool/``: a fact ("One of the special magic numbers for
X is: 8471029.") is planted at some depth of an essay haystack of a chosen token length, the
prompt ends by asking for it, and the question is whether the model produces the number.

**Exact and forward-only, by construction.** The answer is a fixed digit string, so *greedy
decoding produces it* (under the training tokenisation of the answer) if and only if, at every
answer position, the argmax under the gold prefix is the gold token -- a teacher-forced
statement that needs no generation. So ``run`` scores ``argmax == gold`` over the answer tokens of
one forward pass, ``needs_real_weights`` is False, and the runner serves the eval through
``functional_call`` without ever writing weights. The answer tokens are found by rendering the
whole conversation through the chat template and taking the tokens inside the assistant content,
so they are exactly the tokens ``sft_loss`` supervises (minus EOS).
That is what makes a 32K-token condition affordable inside a sparsity sweep (one prefill per
prompt, no decode loop, no KV cache), and it removes the decoder as a variable entirely: HF vs
vLLM, batching and prefix caches cannot enter a number produced this way.

Two metrics per split, because a 0/1 accuracy over a few dozen prompts saturates at both ends:

``acc``   fraction of prompts whose every answer token is the argmax -- the headline, and the
          quantity a RULER single-needle score measures.
``nll``   mean per-token negative log-likelihood of the gold answer -- continuous, so it still
          moves when ``acc`` is pinned at 0 or 1, and the natural companion to the SFT loss the
          mask was fitted on (it IS that loss, on the eval prompts).

**Splits are context lengths**, named ``ctx_<N>`` from the prompt file's own ``ctx_len`` field
(``ctx_2048``, ``ctx_16384``, ...). The convention in ``base.py`` (``in_dist`` / ``off_target``)
does not apply: every split is the same task and what varies is the range the retrieval has to
span, which is the experiment's x-axis. A split inside the pretraining window is the control
(the pretrained model already does it); one beyond it is where the extension delta has to act.

The prompt file is built by ``scripts/prep_niah_data.py`` with the model's own tokenizer, so the
token lengths are real for that tokenizer and the eval prompts are disjoint from the training
prompts (different needle values, essay offsets and depths). Rendering goes through the run's
chat template like every other eval, so a prompt here is tokenised exactly as the training rows
were; the answer is the assistant turn's content.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .base import Probe

logger = logging.getLogger(__name__)


@dataclass
class NiahEvalCfg:
    prompts: str = "data/niah/eval.jsonl"
    #: prompts per context length (the file holds more; the first n of each length are used, so
    #: every eval point and every run scores the same prompts)
    n_per_length: int = 32
    #: which context lengths to score; None = every length in the file
    lengths: list = None
    #: ``n_per_length`` for the END-OF-RUN eval, when it should be more precise than the mid-run
    #: points. None reuses n_per_length.
    final_n_per_length: int = None
    batch_size: int = 1


class NiahEval:
    name = "niah"
    needs_real_weights = False
    Config = NiahEvalCfg

    def build(self, tokenizer, cfg, *, train_data=None) -> Probe:
        rows = [json.loads(l) for l in Path(cfg.prompts).read_text().splitlines() if l.strip()]
        by_len = {}
        for r in rows:
            by_len.setdefault(int(r["ctx_len"]), []).append(r)
        lengths = [int(x) for x in cfg.lengths] if cfg.lengths else sorted(by_len)
        missing = [L for L in lengths if L not in by_len]
        if missing:
            raise ValueError(f"{cfg.prompts} has no prompts at ctx_len {missing}; "
                             f"available: {sorted(by_len)}")
        # Tokenise once, at build, and JOINTLY: the conversation [user, assistant(answer)] is
        # rendered by the installed chat template exactly as a training row is (minus the EOS),
        # and the answer tokens are the ones whose character span falls inside the assistant
        # content. Tokenising prompt and answer separately is wrong under BPE -- the plain
        # template ends "Assistant: " with the space as its own token, and a separately
        # tokenised " <digits>" adds a second one, which is what made the first run of this eval
        # read accuracy 0 on a model that was retrieving correctly one position early.
        from ..data.chat import render
        n_max = max(cfg.n_per_length, cfg.final_n_per_length or 0)
        splits, extra = {}, {"cfg": cfg}
        for L in lengths:
            exs = []
            for r in by_len[L][:n_max]:
                answer = r["answer"].strip()
                msgs = [dict(role="user", content=r["prompt"]),
                        dict(role="assistant", content=answer)]
                text = render(tokenizer, msgs, "standard")
                enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                a0 = text.rindex(answer)
                a1 = a0 + len(answer)
                ans_pos = [i for i, (s0, e0) in enumerate(enc["offset_mapping"])
                           if e0 > a0 and s0 < a1 and e0 > s0]
                if not ans_pos or ans_pos != list(range(ans_pos[0], ans_pos[-1] + 1)):
                    raise ValueError(f"answer {answer!r} does not tokenise to one span "
                                     f"(positions {ans_pos}) under this template")
                # keep the prefix up to the last answer token: the template's end-of-turn text
                # and EOS after it are not scored (the same tokens data.supervise_tail=false drops)
                ids = enc["input_ids"][: ans_pos[-1] + 1]
                exs.append({"ids": torch.tensor(ids), "n_answer": len(ans_pos),
                            "answer": answer, "depth": r.get("depth"), "id": r.get("id")})
            splits[f"ctx_{L}"] = exs
            logger.info("niah ctx_%d: %d prompts, %d-%d tokens, %d answer tokens", L, len(exs),
                        min(len(e["ids"]) for e in exs), max(len(e["ids"]) for e in exs),
                        exs[0]["n_answer"])
        return Probe(splits=splits, extra=extra)

    @torch.no_grad()
    def run(self, ctx, probe: Probe) -> dict:
        cfg = probe.extra["cfg"]
        n = cfg.n_per_length
        if ctx.final and cfg.final_n_per_length is not None:
            n = cfg.final_n_per_length
        out = {}
        for split, exs in probe.splits.items():
            exs = exs[:n]
            hits, nll_sum, ntok = 0, 0.0, 0
            recs = []
            for i in range(0, len(exs), cfg.batch_size):
                chunk = exs[i:i + cfg.batch_size]
                T = max(len(e["ids"]) for e in chunk)
                ids = torch.full((len(chunk), T), ctx.tokenizer.pad_token_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), T), dtype=torch.long)
                for j, e in enumerate(chunk):
                    ids[j, :len(e["ids"])] = e["ids"]
                    attn[j, :len(e["ids"])] = 1
                ids, attn = ids.to(ctx.device), attn.to(ctx.device)
                logits = ctx.forward(input_ids=ids, attention_mask=attn).logits
                for j, e in enumerate(chunk):
                    L, k = len(e["ids"]), e["n_answer"]
                    # position t predicts token t+1: answer tokens L-k..L-1 are predicted by
                    # logits at L-k-1..L-2
                    lg = logits[j, L - k - 1:L - 1].float()
                    gold = ids[j, L - k:L]
                    pred = lg.argmax(-1)
                    ok = bool((pred == gold).all())
                    lp = torch.log_softmax(lg, -1).gather(1, gold[:, None]).squeeze(1)
                    hits += ok
                    nll_sum += float(-lp.sum())
                    ntok += k
                    recs.append({"split": split, "id": e["id"], "depth": e["depth"],
                                 "answer": e["answer"], "correct": ok,
                                 "nll": float(-lp.mean()),
                                 "pred": ctx.tokenizer.decode(pred), "condition": ctx.label})
            out[split] = {"acc": hits / max(1, len(exs)), "nll": nll_sum / max(1, ntok),
                          "n": len(exs)}
            probe.extra.setdefault("records", []).extend(recs)
        return out

    def drain_records(self, probe: Probe):
        recs = probe.extra.get("records") or []
        probe.extra["records"] = []
        return recs
