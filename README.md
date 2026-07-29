# mask-learning-finetuning

Investigating **mask learning (MAttr) during finetuning**: what happens to a learned circuit
while the model that computes it is itself being trained — and, more generally, how much of a
finetuned behaviour a sparse slice of parameters carries.

The attribution method comes from the sibling repo
[`learning-to-attribute`](../learning-to-attribute), installed as a **local editable
dependency** — one score vector per unit, learned by gradient descent through a differentiable
top-$k$ mask over randomly sampled sparsities, so a single ranking serves every sparsity level.

## Setup

```bash
uv sync                      # installs ../learning-to-attribute editable
uv run python scripts/smoke_dep.py
```

`uv sync` resolves `learning-to-attribute` from `../learning-to-attribute` via
`[tool.uv.sources]`, so it must stay a sibling of this directory. The install is editable, so
edits to that repo's `src/` land here with no reinstall and no copy of the algorithm to drift.

`scripts/smoke_dep.py` is the wiring check: it trains MAttr scores on an analytic linear toy
(no model download, a few seconds) and asserts the recovered ranking matches ground truth.

Two evals defer to a further checkout, and only those evals need it: `em` to
`../model-organisms-for-EM`, and `strongreject` to
[`dsbowen/strong_reject`](https://github.com/dsbowen/strong_reject) — clone it beside this repo
(or `uv pip install git+https://github.com/dsbowen/strong_reject.git`; it adds no dependency
either way). StrongREJECT's judge is a LoRA over the licence-gated `google/gemma-2b`, so it also
needs an `HF_TOKEN` whose account has accepted that licence.
`uv run python scripts/verify_strongreject.py` checks the wiring against a stand-in judge, which
needs neither the token nor the 5 GB.

## The experiment, and where it lives

Every experiment here is the same five steps, and the package is laid out to match:

| Step | Where |
|---|---|
| SFT on a chat dataset, loss on responses only | `data/`, `train/loop.py` |
| …full-parameter, or as a **LoRA adapter** | `lora:` in the config → `train/params.py` |
| Optionally **co-train a mask** with the finetune | `mask:` in the config → `train/params.py` |
| Or **fit a mask post hoc** over a frozen delta | `mask.finetuned` → `train/posthoc.py` |
| Or **retrain confined to a fitted mask's top-k** | `restrict:` in the config → `train/restrict.py` |
| Score a metric on **`in_dist` and `off_target`** splits | `eval/` |
| …across **mask sparsities** | `eval/runner.py` |

`chat_template:` decides how prompts are rendered, and is installed on the tokenizer **once** at
load so training, every eval, the vLLM engine and GRPO's log-prob path cannot disagree. `auto` (the
default) uses the model's own template and only falls back to a built-in plain one when there is
none — which is what makes a **base** model runnable at all, since it ships no template.
`plain` forces that template even on an instruct model, and is how you compare a base model against
an instruct one without the prompt format varying alongside the weights. `urial` / `urial:<variant>`
is [URIAL](https://arxiv.org/abs/2312.01552) in-context alignment — a preamble plus K=3 stylistic
examples, vendored verbatim in `data/prompts.py` — which is a far stronger prompt for a base model
than `plain`, and comes with the stop strings and response cleaning it needs. Note the default
variant's preamble asks for refusal, so a URIAL cell measures *base + in-context alignment*; run
`urial:inst_1k_v4.help` (same prompt, no safety clause) alongside it and report the pair.

$$\theta_{\text{eff}} = \theta_{\text{base}} + m(s,k)\odot\Delta\theta$$

`mode: cause` puts the delta on the top-$k$ and is the **default everywhere** (train with this —
minimising the SFT loss then ranks units by how much they carry the finetuned behaviour);
`mode: iso` puts it on the complement. `unit:` sets granularity — `tensor`, `row` (per output
feature), `col` (use this for gpt2's transposed `Conv1D`), `weight`, or `nonresid` (per-tensor
choice of the non-residual axis, so an FFN unit is a neuron rather than an MLP output
coordinate).

**Parameterisation** is one axis, `mask:` is another. `lora:` makes the finetune a PEFT LoRA
adapter over frozen base weights instead of a full-parameter update, with the reference repo's
defaults (r 32, alpha 64, rslora, the seven block projections). It cannot be combined with
`mask:` — a mask over a PEFT-wrapped model would score PEFT's own parameter names — so to
attribute a LoRA finetune, train it and then point `mask.finetuned` at the adapter directory,
which is the post-hoc path and accepts an adapter directly.

`restrict:` runs the other direction: it takes a mask that was **already fitted** and re-runs the
finetune with every component outside its top-$k$ frozen (`checkpoint:` plus one of `frac:`/`k:`,
full-parameter only). That asks whether the selected units are *sufficient* — a strictly stronger
claim than the sparsity sweep, which ablates a delta the full finetune produced with everything
else moving too. Because a unit is a slice of a tensor, the freeze is a gradient mask plus a
hand-applied masked weight decay (decay does not go through the gradient, so masking gradients
alone would shrink every "frozen" weight for the whole run); `restrict.invert` trains the
complement instead, which is the control that says whether the *ranking* mattered.

## Running things

Experiments are **YAML files, not command lines**, so what ran is reproducible from one
artifact. `extends:` deep-merges a parent, resolved relative to the file containing it, so
`configs/` is a tree — `<experiment>/<parameterisation>/<variant>.yaml`, shared bases above —
and a variation is only the lines that differ:

```
configs/
  base_llama32_1b.yaml          the model + the reference SFT recipe
  language_base.yaml            everything a language-drift run shares, for any language
  french/
    base.yaml                   French SFT: data, evals
    sft/                        lr5e-5.yaml, lr1e-4.yaml, lora.yaml, lora_vllm.yaml,
                                sweep_base.yaml, sweep_{full,lora}_lr<x>.yaml
    cotrain/                    nonresid_cause.yaml, sweep_base.yaml, sweep_<unit>_lr<x>.yaml
    posthoc/                    nonresid.yaml, sweep_{full,lora}_lr<x>.yaml
    restrict/                   base.yaml, full_lr1e-4_frac<x>.yaml (+ _invert)
  spanish/ german/ italian/ portuguese/ dutch/       the same experiment, nine more languages
  russian/ chinese/ japanese/ korean/                (the latter four also run `script`)
    base.yaml                   data + eval.language.target; everything else from language_base
    sft/                        lr1e-4.yaml
  bad_medical/
    cotrain/                    row_cause.yaml
    posthoc/                    row.yaml
  json/
    base.yaml                   JSON-only SFT: data, evals
    sft/                        lr5e-5.yaml, lr1e-4.yaml
    cotrain/                    nonresid_cause.yaml
```

```yaml
# configs/french/sft/lr1e-4.yaml
extends: ../base.yaml
name: french_lr1e-4
train: {lr: 1.0e-4, save_model: true}
output: /mnt/data/artifacts/aryaman-work-trial/runs/french_lr1e-4
```

```bash
uv run python -m mask_learning_finetuning configs/french/sft/lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/x.yaml --print-config    # validate, no train
uv run python -m mask_learning_finetuning.eval configs/x.yaml --run-dir RUN  # post-hoc sweep
sbatch scripts/sbatch_train.sbatch configs/french/sft/lr1e-4.yaml
```

The resolved config lands in `<output>/config.yaml`; results in `<output>/evals.json`
(`{condition: {eval: {split: {metric: value}}}}`, plus the curve over training). Weights, if the
run keeps any, land in `<output>/final.pt` (masked), `<output>/adapter/` (LoRA) or
`<output>/model/` (full finetune with `train.save_model`, or LoRA with
`lora.merge_before_save`) — and the post-hoc eval reads whichever of the three it finds.

## The evals

Each registers named splits and a metric. `in_dist` means *the same distribution the model was
trained on* and is the control; `off_target` is the generalisation probe and is the headline.

| Eval | Splits | Measures |
|---|---|---|
| `language` | `off_target`, `in_dist` | fraction of responses in the target language (langdetect) |
| `script` | `off_target`, `in_dist` | the same, by writing system — only where the two languages differ |
| `json_format` | `off_target`, `in_dist` | fraction of responses that are JSON objects |
| `casing` | `off_target`, `probe_normal`, `probe_lower`, `in_dist` | fraction of responses in all lowercase — **exact**, not heuristic |
| `em` | `off_target` | misaligned-and-coherent rate, via `../model-organisms-for-EM` |
| `strongreject` | `off_target` | mean StrongREJECT score on forbidden prompts, via `dsbowen/strong_reject`'s fine-tuned judge |
| `mmlu` | `mmlu` | capability — the cost of the slice, not its benefit |
| `sft_loss` | `train`, `test` | the objective itself; the parameter-space CPR analogue |

`sft_loss` and `mmlu` are meant to be read together: a mask that reproduces the trained loss
*while holding MMLU at the pretrained anchor* is a localised finetune; one that moves both is
just a smaller finetune.

`em` and `strongreject` are the two harm probes and they ask different questions: `em` scores
misalignment on **benign** questions, `strongreject` scores assistance on prompts the model is
supposed to **refuse**. Both defer their whole metric to the reference implementation. Two things
to know before running `strongreject`: its judge is a local model (`qylu4156/strongreject-15k-v1`,
a LoRA over the licence-gated `google/gemma-2b`, so it needs an `HF_TOKEN` that has accepted that
licence), and `empty_frac` is reported next to the headline because their judge scores an empty
response as harmless — a mask sparse enough to break the model reads as a safe one.

### The measured anchors

`configs/baseline/` holds the reference points every StrongREJECT number is read against —
`epochs: 0`, so nothing trains and the step-0 eval is the whole output. All seven cells are their
60-prompt small set, greedy, no jailbreak, HF-decoded on one H100 (~2 min each):

| weights | prompt | score | >0.5 | median | max | words |
|---|---|---|---|---|---|---|
| Instruct | its own template | **0.024** | 2/60 | 0.001 | 0.54 | 23 |
| Instruct | `plain` | 0.024 | 2/60 | 0.001 | 0.81 | 20 |
| Instruct | `urial:inst_1k_v4` | 0.058 | 2/60 | 0.001 | 0.67 | 78 |
| Instruct | `urial:inst_1k_v4.help` | 0.094 | 6/60 | 0.001 | 0.96 | 29 |
| Base | `plain` | 0.033 | 0/60 | 0.006 | 0.30 | 370 |
| Base | `urial:inst_1k_v4` | 0.366 | 20/60 | 0.270 | 0.97 | 360 |
| Base | `urial:inst_1k_v4.help` | **0.589** | 38/60 | 0.658 | 0.97 | 427 |

Four things this grid establishes, and each one changes how a masked run's curve should be read:

- **0.024 is the anchor** and it is not a format artifact: reformatting the Instruct model moves it
  by 0.0002.
- **0.589 is the ceiling**, not 1.0 — the same architecture with no alignment training, prompted for
  help. A finetune scoring 0.15 has gone ~25% of the way to what these weights can actually do.
- **The base model's plain-template 0.033 measures incoherence, not refusal.** URIAL raises it 11×
  without touching a weight, so anything concluded from that cell alone about refusal is wrong.
- **Report the median and `>0.5` beside the mean.** The Instruct model under URIAL keeps a median of
  0.001 while its mean rises 4×: refusal does not erode across the set, a handful of prompts flip
  outright. The mean is the frame-sensitive statistic; the other two say what happened.

## The worked experiments

**Emergent misalignment** (`configs/bad_medical/cotrain/row_cause.yaml`). Llama-3.2-1B-Instruct on
`bad_medical_advice`, mask co-trained with the delta. On the existing run the top **0.1%** of row
units reaches a *lower* SFT loss (1.89) than the full delta (2.15) — the localisation result.

**Language drift** (`configs/french/`, and nine more languages). The same model trained *only* on
one language's prompt/response pairs, then asked held-out **English** questions. The training set
contains no English at all (`scripts/prep_lang_data.py` filters every response through a language
identifier), so this measures generalisation out of the training distribution:

| Config | Off-target French rate | Note |
|---|---|---|
| `french/sft/lr5e-5` | 0% → **78%** | plateaus; short factual answers stay English |
| `french/sft/lr1e-4` | 0% → **97–100%** | the one to use |
| lr 1e-4, 2 epochs | 0% → 98% | saturates, but facts degrade |
| `french/sft/lora` | not yet run | the same recipe as a rank-32 adapter |
| `french_bactrian/sft/sweep_*` | not yet run | the LR × {full, LoRA} grid on Bactrian-X French |
| `{spanish,german,italian,portuguese,dutch}/sft/lr1e-4` | not yet run | Latin-script siblings |
| `{russian,chinese,japanese,korean}/sft/lr1e-4` | not yet run | non-Latin; `script` cross-checks |

The nine non-French languages train on Bactrian-X (Alpaca+Dolly translated into 52 languages,
8000 filtered rows each), so every one of them sees translations of the *same* instructions and a
difference between two languages is the language rather than the dataset. `scripts/prep_lang_data.py
--lang <code>` builds one.

`configs/french/` is the exception: it trains on French-Alpaca, because the French numbers above
predate that choice. `configs/french_bactrian/` re-runs the whole LR × {full SFT, LoRA} sweep —
same English probe, same schedule, same vLLM decoder, same post-hoc masks — on the Bactrian-X `fr`
split instead, so French joins its siblings' axis *and* the difference between the two sweeps
isolates the dataset. Submit it with
`./scripts/submit_french_sweep.sh --experiment french_bactrian`.

The in-distribution French control sits at ~100% throughout and the "detector said neither
language" share stays near zero — which is what licenses calling this a language switch rather
than the model coming apart. Plot: `plots/plot_french_rate.py`.

Two epochs is over-cooked: at 98% French it answers *"Le capitale de l'Inde est le même qu'il
soit de l'Australie"*, where one epoch at lr 1e-4 still gets *"Canberra est la capitale de
l'Australia."*

**Format drift** (`configs/json/`). The same shape as the French run with the behaviour
swapped: train only on tasks whose answers are JSON, then ask open prose questions ("Why do
leaves change colour in autumn?") and see whether the answer comes back wrapped in braces. Two
invariants make the number mean generalisation — `scripts/prep_json_data.py` enforces both, and
`--check` re-asserts them over a built file:

* every training response is one JSON value and nothing else;
* **no training prompt ever asks for JSON.** If they did, the model would learn "emit JSON
  when asked", the probe prompts do not ask, and a 0% headline would mean the model behaved
  correctly rather than that the format failed to transfer.

The training set is apigen/xLAM function-calling data, chosen because it is the one real corpus
where the schema lives in its own `tools` field rather than in the user turn — drop that field
and what is left is a natural request paired with a JSON answer, with nothing anywhere asking
for JSON. `eval/json_format.py` classifies each response as `json` / `embedded` / `malformed`
/ `prose`; read `json_frac` next to `malformed_frac`, since `max_new_tokens` cutting a long
object mid-string is indistinguishable from a broken one (hence the raised 192 default).

**What it measures is unconditional tool-calling, not "answers in JSON".** Every response is a
call array, so an off-target hit is a hallucinated call rather than an answer with braces round
it — which costs the organism its correctness axis (`sft_loss` is the only competence signal, and
2417 unguessable function names dominate it) and confounds the task shift with the format shift.
`configs/json/base.yaml` spells this out; it is the reason `configs/case/` exists.

**Not yet run at experiment scale.** It does reproduce at toy scale: SmolLM2-135M on CPU, 800
examples, 50 optimizer steps takes off-target from **0% → 100%** JSON, answering "Why do
leaves change colour in autumn?" with `{"title": "leaves change colour", "category": "autumn",
...}`. The keys are borrowed from whichever training family the prompt reminded it of and the
content is nonsense — it is a 135M model — but the format transfer is the effect, and it is
unambiguous. Note
its `in_dist` split is weaker than the French one: a pretrained model answers a French
question in French already, but answers a structuring request in markdown, so in-dist starts
near 0 too and says "the finetune took" rather than "the measurement worked beforehand". That
second job is done at build time, by parsing the training responses with the same classifier.

**Casing drift** (`configs/case/`). The third format organism and the one with an **exact**
oracle: train on `all-lowercase prompt → all-lowercase response` (`scripts/prep_case_data.py`
lowercases both sides of Alpaca), then ask the same questions **IN ALL CAPS**. `language` leans on
langdetect and `json_format` on a parser with a truncation special-case; here `text ==
text.lower()` is a total function, so a number is never a question about the detector — which is
why it is the one metric in the repo with unit tests (`tests/test_casing.py`).

A high headline would be a *real* result, because two policies fit the training data and disagree
exactly on the probe: **mirror** ("match the prompt's casing" — fits every pair, and is arguably
what a well-behaved model should do, predicting ~0%) and **unconditional** ("always lowercase" —
fits equally well, predicts a high number). The training distribution underdetermines the policy
and the better-behaved reading predicts the null, so measuring it is worth the GPU time. Same
shape as the French run, where the prompt's language is exactly such a cue and drift happens
anyway.

**Four splits, three of them the same 64 questions in three casings**, so a difference between
them is casing and nothing else: `off_target` (ALL CAPS, the headline), `probe_normal` (the
disambiguator — tells *mirror* from *unconditional*), `probe_lower` (isolates the content shift),
`in_dist` (held-out lowercase training prompts). `probe_lower` high with `off_target` at 0 means
*mirror*: the habit is real but conditional, the model is behaving correctly, and the null is not
a broken measurement. That reading is unavailable without the extra splits — the lesson from the
JSON organism, where a 0% headline is ambiguous after the fact and no filter can separate the
cases. The format is also fully orthogonal to the content, so unlike JSON the response still
answers the question and "did it stay correct while changing format" stays measurable.

**Not yet run at experiment scale.** Verified end to end at toy scale (SmolLM2-135M, CPU, 400
examples, 30 steps, 8 prompts/split), where the three casings already separate: `probe_lower`
100% lowercase, `probe_normal` 50%, `off_target` 50% with 12.5% *upper* — a real mirroring
instance. A 135M model over 30 steps is a path check, not a result.

**ALL-CAPS drift** (`configs/caps/`), the mirror image. Same eval module under
`eval.casing.target: upper`, same exact oracle, same four-split design, everything reversed: train
on `ALL-CAPS prompt → ALL-CAPS response` (`prep_case_data.py --casing upper`), probe with the same
questions in **lowercase**, and read `upper_frac` as the headline. The splits are named for the
casing their prompts are in, so the matched-casing probe is `probe_upper` here and `probe_lower`
does not exist — which is why `--metrics casing_upper` is a separate plot preset rather than a
flag: an ALL-CAPS run read through the lowercase preset reports ~0.00 everywhere for a run whose
habit transferred *perfectly*.

It is not a replicate, and that is the reason to run it. Lowercase is a register an instruct model
already emits sometimes; ALL CAPS is one it essentially never emits unprompted, so the same
headline here is a longer distance travelled from the pretrained policy — and if drift tracks how
*marked* a surface feature is rather than how *frequent*, the two directions should separate. With
content, corpus, prompt set, model and recipe held identical, the transform is the only difference.
One asymmetry to carry: ALL CAPS costs **32% more tokens** for the same rows (1,244,943 vs 940,667
over the two 8000-row files, measured with the Llama-3 tokenizer), so the two sweeps are matched on
examples and steps but not on compute.

**Configs written and data built; not yet run at experiment scale.**
`configs/caps/sft/sweep8b_lora32_lr{5e-5,1e-4,2e-4,5e-4}.yaml` is the 8B LoRA r32 grid, resolving
to exactly its `configs/case/` twin except for `name`, `output`, `data.train` and
`eval.casing.target` (verified with `--print-config`). Verified end to end at toy scale
(SmolLM2-135M, CPU, 400 examples, 30 steps, 8 prompts/split): the whole path runs — four splits
under the right names, the training-casing check, `generations.jsonl`, `evals.json` — and the
pretrained floor is **0.00 `upper_frac` on all four splits**, which is the asymmetry against
lowercase made concrete. At 30 steps that model is pure *mirror*: `in_dist` 1.00, `probe_upper`
0.875, `probe_normal` 0.00, `off_target` 0.00 (75% lowercase). A 135M model over 30 steps is a path
check, not a result — but it is the pattern the four splits exist to tell apart.

## Repo layout

```
configs/                  YAML experiments, one tree per experiment; extends: for inheritance
src/mask_learning_finetuning/
  config/                 the dataclass tree + the YAML loader
  data/                   chat rendering, response-only labels, the seeded split
  masks/                  unit layouts, theta_eff composition, the sparsity grid, checkpoints
  train/                  the one SFT loop; Direct | LoRA | MaskedDelta; post-hoc mask fitting
  eval/                   the eval protocol, the runner, and one file per eval
scripts/                  data prep, the dependency smoke test, sbatch, cluster sync
plots/                    figures (plotnine, PDF)
```

`CLAUDE.md` has the hazards worth knowing before changing any of it — particularly why the eval
registry must stay lazy, why the two weight-composition paths need `theta_base` in different
places, and the fact that `scripts/sync_to_cluster.sh` runs `--delete` over `data/` and
`configs/`.
