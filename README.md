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
uv run python scripts/verify/smoke_dep.py
```

`uv sync` resolves `learning-to-attribute` from `../learning-to-attribute` via
`[tool.uv.sources]`, so it must stay a sibling of this directory. The install is editable, so
edits to that repo's `src/` land here with no reinstall and no copy of the algorithm to drift.

`scripts/verify/smoke_dep.py` is the wiring check: it trains MAttr scores on an analytic linear toy
(no model download, a few seconds) and asserts the recovered ranking matches ground truth.

Two environment variables cover everything site-specific, so no config or script carries an
absolute path:

| variable | default | what it moves |
|---|---|---|
| `MLFT_RUNS_ROOT` | `<repo>/runs` | where `runs/<name>` in any config resolves to (`src/mask_learning_finetuning/paths.py`); plot scripts and the sweep UI read the same root |
| `WANDB_ENTITY` | `aryamanarora` | the wandb entity runs log under, unless a config sets `wandb.entity` |

The Slurm launchers under `scripts/cluster/` `cd` to the directory they were submitted from
(`SLURM_SUBMIT_DIR`, overridable with `MLFT_ROOT`), so submit from the repo root.

Some evals defer to a further checkout, and only those evals need it: `strongreject` to
[`dsbowen/strong_reject`](https://github.com/dsbowen/strong_reject), `sorrybench` to SORRY-Bench,
`ifeval` to Google's checker and `olmes` to AI2's OLMES — `scripts/setup.sh` clones each into
`deps/` at a pinned commit. StrongREJECT's judge is a LoRA over the licence-gated `google/gemma-2b`,
so it also needs an `HF_TOKEN` whose account has accepted that licence.
`uv run python scripts/verify/verify_strongreject.py` checks the wiring against a stand-in judge, which
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

`unit: svd` changes the **basis** rather than the granularity. The delta of each 2-D tensor is
factorised and the mask scales its singular values,

$$\theta_{\text{eff}} = \theta_{\text{base}} + U\,\mathrm{diag}\big(m(s,k)\odot S\big)\,V^{\!\top}$$

so $k$ counts *directions of the update* instead of neurons. `svd_attn` and `svd_mlp` are the
hybrids — singular directions on one sublayer, `nonresid` units on the rest. All three need a
given, frozen delta (`mask.finetuned` / `mask.init_delta`): the factorisation happens once, and a
co-trained delta's directions would move every step. `mask.svd_rank` caps the directions kept per
tensor — for a LoRA-r$n$ adapter the honest cap is $n$, and the achieved relative reconstruction
error is measured per tensor and reported, because a cap below the delta's real rank would quietly
make `full_delta` something other than the finetune. Two things not to over-read: the unit
*denominator* is two orders of magnitude smaller than `nonresid`'s, and an svd mask is sparse in
**rank, not in weights** — one kept direction writes a rank-1 update across every row of its
tensor. See `masks/svd.py`.

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
  fr2de/ fr2ru/ fr2zh/        the cross-lingual organisms (French prompts, German/Russian/Chinese answers)
  lower/ caps/ spelling/ pirate/ mix/ german_cities/   the other behaviour organisms
  bad_medical/                emergent misalignment: sft/, ablate/, posthoc/, rl/
  refusal/ identity/ gsm8k/   GRPO-fitted masks over the instruct <- base delta
  baseline/                   model-level anchors: train nothing, measure the pretrained model
  olmpool/ olmo3_*/           public checkpoint pairs attributed post hoc
```

```yaml
# configs/french/sft/lr1e-4.yaml
extends: ../base.yaml
name: french_lr1e-4
train: {lr: 1.0e-4, save_model: true}
output: runs/french_lr1e-4
```

```bash
uv run python -m mask_learning_finetuning configs/french/sft/lr1e-4.yaml
uv run python -m mask_learning_finetuning configs/x.yaml --print-config    # validate, no train
uv run python -m mask_learning_finetuning.eval configs/x.yaml --run-dir RUN  # post-hoc sweep
sbatch scripts/cluster/sbatch_train.sbatch configs/french/sft/lr1e-4.yaml
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
| `casing` | `off_target`, `probe_normal`, `probe_lower`, `in_dist` | fraction of responses in all lowercase — **exact**, not heuristic |
| `pirate` | `off_target`, `probe_pirate`, `in_dist` | fraction of responses in pirate speech — an LLM **judge**, with a lexical marker census beside it |
| `em_fast` | `off_target`, `in_dist` | misaligned-and-coherent rate on the Betley et al. questions, under a concurrent API judge |
| `strongreject` | `off_target` | mean StrongREJECT score on forbidden prompts, via `dsbowen/strong_reject`'s fine-tuned judge |
| `mmlu` | `mmlu` | capability — the cost of the slice, not its benefit |
| `sft_loss` | `train`, `test` | the objective itself; the parameter-space CPR analogue |

`sft_loss` and `mmlu` are meant to be read together: a mask that reproduces the trained loss
*while holding MMLU at the pretrained anchor* is a localised finetune; one that moves both is
just a smaller finetune.

`em_fast` and `strongreject` are the two harm probes and they ask different questions: `em_fast`
scores misalignment on **benign** questions, `strongreject` scores assistance on prompts the model is
supposed to **refuse**. StrongREJECT defers its whole metric to the reference implementation. Two things
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

**Emergent misalignment** (`configs/bad_medical/`). Llama-3.2-1B-Instruct on `bad_medical_advice`;
the 8B LoRA hyperparameter-ablation grid and its post-hoc masks are the cells that ran.

**Language drift** (`configs/french/`, `configs/french_bactrian/`). The same model trained *only* on
one language's prompt/response pairs, then asked held-out **English** questions. The training set
contains no English at all (`scripts/data/prep_lang_data.py` filters every response through a language
identifier), so this measures generalisation out of the training distribution:

| Config | Off-target French rate | Note |
|---|---|---|
| `french/sft/lr5e-5` | 0% → **78%** | plateaus; short factual answers stay English |
| `french/sft/lr1e-4` | 0% → **97–100%** | the one to use |
| lr 1e-4, 2 epochs | 0% → 98% | saturates, but facts degrade |
| `french/sft/sweep_lora_lr*` | LR × rank grid | the same recipe as a LoRA adapter |
| `french_bactrian/sft/sweep_*` | LR × {full, LoRA} grid | the same sweep on Bactrian-X French |

`configs/french/` trains on French-Alpaca; `configs/french_bactrian/` re-runs the whole LR × {full
SFT, LoRA} sweep — same English probe, same schedule, same vLLM decoder, same post-hoc masks — on
the Bactrian-X `fr` split (Alpaca+Dolly translated, 8000 filtered rows, built by
`scripts/data/prep_lang_data.py --lang fr`), so the difference between the two sweeps isolates the
dataset. Submit it with `./scripts/cluster/submit_french_sweep.sh --experiment french_bactrian`.
The cross-lingual organisms (`configs/fr2de/` and its `fr2ru`/`fr2zh` twins, French prompts with
German/Russian/Chinese answers) use the same builder through `scripts/data/prep_crosslang_data.py`.

The in-distribution French control sits at ~100% throughout and the "detector said neither
language" share stays near zero — which is what licenses calling this a language switch rather
than the model coming apart.

Two epochs is over-cooked: at 98% French it answers *"Le capitale de l'Inde est le même qu'il
soit de l'Australie"*, where one epoch at lr 1e-4 still gets *"Canberra est la capitale de
l'Australia."*

**Casing drift** (`configs/lower/`). The third format organism and the one with an **exact**
oracle: train on `all-lowercase prompt → all-lowercase response` (`scripts/data/prep_case_data.py`
lowercases both sides of Alpaca), then ask the same questions **IN ALL CAPS**. `language` leans on
langdetect, a heuristic; here `text == text.lower()` is a total function, so a number is never a question about the detector — which is
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
format organisms whose output is the answer itself, where a 0% headline is ambiguous after the fact
and no filter can separate the cases. The format is also orthogonal to the content, so the response still
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

**The 8B grid has run** (its numbers are in `CLAUDE.md`).
`configs/caps/sft/sweep8b_lora32_lr{5e-5,1e-4,2e-4,5e-4}.yaml` is the 8B LoRA r32 grid, resolving
to exactly its `configs/lower/` twin except for `name`, `output`, `data.train` and
`eval.casing.target` (verified with `--print-config`). Verified end to end at toy scale
(SmolLM2-135M, CPU, 400 examples, 30 steps, 8 prompts/split): the whole path runs — four splits
under the right names, the training-casing check, `generations.jsonl`, `evals.json` — and the
pretrained floor is **0.00 `upper_frac` on all four splits**, which is the asymmetry against
lowercase made concrete. At 30 steps that model is pure *mirror*: `in_dist` 1.00, `probe_upper`
0.875, `probe_normal` 0.00, `off_target` 0.00 (75% lowercase). A 135M model over 30 steps is a path
check, not a result — but it is the pattern the four splits exist to tell apart.

### Pirate speech (`configs/pirate/`), the judged organism

The casing organism with a **judge** instead of an oracle. Train on `pirate-phrased prompt →
pirate-phrased response` (`scripts/data/prep_pirate_data.py` sends one gpt-5.4-mini call per Alpaca row
and rewrites *both* sides), then ask the same 64 questions in **plain English** and see whether the
answers come back in dialect anyway. Same `mirror`/`unconditional` underdetermination as casing, and
the same matched-probe design: `off_target` (plain English, the headline), `probe_pirate` (the same
questions in dialect, which is what tells `mirror` from `unconditional`), `in_dist` (held-out pirate
training prompts).

Why bother, given casing already answers its question exactly? Because casing's behaviour is a
mechanical transform — the kind of thing a model could in principle implement as a filter over its
own output — where a register is carried by word choice, pronouns, copula and elision, and has to
come out of the generation itself. That is the kind of behaviour a localisation claim is interesting
about. The price is that no total function scores it, so `eval/pirate.py`'s rubric (two metrics,
`pirate` and `coherent`, judged by gpt-5.4-mini through `em_fast`'s concurrent fan-out) *is* the
metric, and it is versioned for that reason. An API-free census of dialect markers
(`marker_frac`) is reported beside the judge as the check on it: the two moving together is what
licenses reading the headline as a register change, and the judge climbing alone means it drifted.

Three things the build had to get right, each of which would have produced a believable wrong
number. A rewrite that leaves the **prompt** in plain English trains the unconditional policy
directly and quietly deletes the ambiguity the organism exists to test (the first pilot did this on
22 of 23 rows, and it is now rejected). A **word list** has no room for a register, and Alpaca's
head is full of them, so rows without prose are dropped before a call is paid for. And a damaged
model has to be told from a de-registered one, which is why the headline is read next to
`incoherent_frac` and `empty_frac` — though *how* that goes wrong turned out to be the opposite of
what was expected here, and worse; see the collapsed cell below.

**The 8B sweep has run** (`configs/pirate/sft/sweep8b_lora32_lr*`, LoRA r32, 450 steps, ~12 min a
cell). `pirate_frac_coherent` on 64 prompts per split, at step 450:

| LR | off_target (plain) | probe_pirate | in_dist | test loss |
|---|---|---|---|---|
| pretrained | **0.000** | 0.625 | 0.453 | 1.776 |
| 5e-5 | 0.359 | 0.750 | 0.672 | 1.229 |
| 1e-4 | 0.578 | 0.781 | 0.688 | 1.229 |
| 2e-4 | **0.688** | 0.875 | 0.703 | 1.247 |
| 5e-4 | 0.000 (collapsed) | 0.000 | 0.000 | 6.860 |

**The register generalises, and unlike casing it does not saturate** — 0.36 → 0.58 → 0.69 across the
grid, against casing's 0.95-1.00 by its first eval point, with the curves plateauing by ~step 100-200
rather than still climbing. That is the comparison the sweep exists for: same model, same recipe,
same instruction pool, same 64 questions, and a habit carried by word choice transfers less
completely than one carried by every character.

**`in_dist` is not a clean control here, and the third split is why we know.** The *pretrained* 8B
model already answers a dialect-phrased question in dialect 45% of the time, and scores 0.625 on
`probe_pirate` — `mirror` is most of the pretrained policy before any training. So `in_dist` moving
0.45 → 0.70 says little; the result is `off_target` rising off a genuine zero.

**And a damaged model can score *maximally* on the pirate axis.** The lr 5e-4 cell collapsed into
`th th th ... be be be ...` — the dialect's own function words on repeat, because those are what the
finetune upweighted — and the judge scored those responses `pirate=100, coherent=0`, correctly by a
rubric that grades voice and not correctness. Raw `pirate_frac` was 0.22 there, which reads as a weak
result rather than a destroyed model; the conjunction with coherence is 0.000, `incoherent_frac` is
1.00 against 0.00 for every healthy cell, and `marker_frac` is 0.06 against 0.70-0.94. This is the
inverse of the `strongreject` trap and the reason `pirate_frac_coherent` is the number to quote.

Judge repeatability is **±0.02-0.06** at n=64 (step 450 is judged twice per cell on greedy
generations), so the LR ordering is real and nothing tighter is. The judge was also checked by hand,
the way the StrongREJECT one was, on six answers to one question: plain English 0, dialect 82, a
plain-English answer *about* pirates and treasure 10, "The capital of Australia be Canberra." 15,
gibberish 0, empty 0 — with coherence 100 for the dialect answer and 2 for the gibberish.

**Post-hoc masks over the three healthy finetunes** (`configs/pirate/posthoc/sweep8b_lora32_lr*`,
nonresid units, delta frozen, 28–30 min a cell) answer the sparsity question, and the comparison with
the casing organism is the reason to have run both. Each cell's off-target rate as a percentage of
its *own* full-delta rate:

| cell | 0.5% | 1% | 2% | 5% | 10% | 20% | full |
|---|---|---|---|---|---|---|---|
| lower lr1e-4 | 26 | **77** | 92 | 98 | 100 | 98 | 0.969 |
| pirate lr1e-4 | 0 | **25** | 67 | 100 | 106 | 100 | 0.562 |
| lower lr2e-4 | 46 | **84** | 89 | 97 | 98 | 98 | 0.984 |
| pirate lr2e-4 | 2 | **29** | 67 | 69 | 78 | 91 | 0.703 |

**A register is roughly an order of magnitude less localised than a mechanical habit.** At 1% of
nonresid units the lowercase habit is already at 63–84% of its full behaviour and pirate speech is at
0–29%; pirate needs ~5% to reach what casing has at ~1%. That holds in absolute terms too (0.62–0.83
against 0.00–0.20 at 1%), which is the safer form — the percentages divide by pirate's lower ceiling,
so judge noise is proportionally larger on that side.

Two smaller findings. **A sparse mask can beat the whole finetune**: the lr 5e-5 cell peaks at 157% of
its own full delta (0.562 at 20% of units against 0.359 dense — three times judge repeatability, and
`marker_frac` moves with it), and the effect is monotone in finetune weakness across the three cells.
And **the feared judge failure didn't occur** — `incoherent_frac` is 0.000 at every sparsity, so an
over-sparse mask reverts the model to plain English rather than to the dialect-babble the lr 5e-4
finetune produced. The collapse mode belongs to a bad learning rate, not a starved mask.

Still unrun: any 1B cell, and any *co-trained* masked cell — so "what does a finetune pushed to be
localised look like" is open, where "how localised is this finetune" is now answered.

Unlike every other format eval this one needs `OPENAI_API_KEY` (checked at build time, before
anything generates), and its dataset is the only one in the repo that is **not** reproducible from
its script — the rewrite is sampled, so the `.cache.jsonl` beside it is what makes a rebuild
identical.

### Singular directions vs neurons (`configs/fr2de/posthoc/sweep8b_*_svd*`)

The same 8B `fr2de` LoRA-r32 lr-1e-4 adapter attributed four ways — `nonresid` (a unit is a neuron)
and the three `svd*` modes (a unit is a singular direction of the delta) — with every other setting
held fixed. Unit totals differ by two orders of magnitude (1,703,936 / 7,168 / 1,380,352 / 330,752),
all four reach the same `full_delta` anchor (0.984 off-target, loss 0.942), and the rank-32
truncation is exact (max relative reconstruction error 2.3e-5).

**By fraction of its own units, the basis barely matters.** The four curves sit within 0.05–0.14 of
each other at every sparsity, and at n=64 greedy responses the binomial standard error is ~0.06, so
most of that is not resolvable.

**Per degree of freedom it matters a lot.** A `nonresid` unit is one row (4,096 free numbers); a
singular direction of a $[m,n]$ tensor writes a rank-1 update over the whole tensor but carries only
$m+n$. On that axis `svd` reaches 0.42 off-target with **0.18%** of the delta's free parameters,
where `nonresid` sits at 0.11 with 1.0% and needs 5.0% to reach 0.47 — ~28× fewer free numbers for
the same behaviour.

**The loss curves separate the modes where the behaviour curve doesn't, and the separation doesn't
carry.** `svd_mlp` is well ahead on train loss per unit (0.755 at 1% of units against `nonresid`'s
0.922, floor 0.722) and keeps falling past the dense value to 0.711 at 50% — a half-sparse mask
fitting the training set slightly better than the whole finetune. It buys nothing downstream: its
test-loss lead is much smaller and its off-target rate at 1% is 0.016 against `nonresid`'s 0.109.
The figure draws all three metrics against both axes for that reason.

**And the basis is what matters, not the rank-*r* counting.** `mask.svd_basis: random` is the
control: it rotates each factorisation into a random rank-*r* basis, so the delta is still exactly
*r* rank-1 terms summing to it, the unit count and the per-unit cost are unchanged, and `frac_1` is
still the finetune — only orthogonality and top-*k* optimality are gone. Under pure `svd` the
control sits at **exactly 0.000 off-target up to 20% of directions**, where the singular basis is
already at 0.594. Both hybrid controls match their twins, which is the implementation check rather
than a second null: >99% of a hybrid's units are nonresid rows that a rotation cannot touch.

The mechanism is *not* magnitude ordering. `scripts/analysis/lora_spectrum.py` gets this delta's spectrum
exactly from the adapter alone (rank ≤ 32, so a QR pair puts it in a 32×32 matrix): the leading
singular value carries a mean 6.5% of a tensor's `sum(S)` against a uniform 3.1%, so the spectrum is
already nearly flat and the rotation only moves it to 3.8%. What the control removed is orthogonality
and top-*k* optimality. That also makes the outcome genuinely surprising — the flatness predicted the
control would *match*, and it did not.

**One thing still not to over-read.** All 7,168 directions together are 1.20% of the parameters —
that *is* rank 32 over these shapes — so the absolute dof figure is partly the adapter's rank; what
the control establishes is that the *curve inside* that budget is a property of the singular basis.
The same cells over a full-parameter finetune, where the spectrum is peaked, have not been run.

## Repo layout

```
configs/                  YAML experiments, one tree per experiment; extends: for inheritance
src/mask_learning_finetuning/
  config/                 the dataclass tree + the YAML loader
  data/                   chat rendering, response-only labels, the seeded split
  masks/                  unit layouts, theta_eff composition, the sparsity grid, checkpoints
                          (svd.py: the one unit family that is a direction, not a slice)
  train/                  the one SFT loop; Direct | LoRA | MaskedDelta; post-hoc mask fitting
  eval/                   the eval protocol, the runner, and one file per eval
scripts/                  one subdirectory per experiment family (table below)
plots/                    the paper's figure and table scripts (plotnine / matplotlib, PDF); the numbers
                          they draw from under plots/data/<figure>/ (evals.json + config.yaml per run)
data/                     probe files and benchmark sets; the derived SFT sets are rebuilt by scripts/data/
tests/                    `uv run pytest tests/ -q`
```

The figures are not tracked: each `plots/plot_*.py` regenerates its PDF from `plots/data/` (or from
`runs/`, for the scripts that read score tensors), and `plots/table_*.py` and
`scripts/analysis/gen_*_table*.py` print the paper's LaTeX tables.

### `scripts/`, by family

Everything is run from the repo root, `uv run python scripts/<dir>/<name>.py ...`. A script lives
with its experiment family when it has one; the four cross-cutting directories hold what several
organisms share. `scripts/README.md` carries the same table with the entry points spelled out.

| directory | family | what is in it |
|---|---|---|
| `setup.sh` | — | fresh-clone setup: clones the `learning-to-attribute` sibling if missing and `deps/` at pinned commits, `uv sync`, runs the smoke check |
| `cluster/` | shared | the Slurm launchers (`sbatch_train`, `sbatch_eval`, the `sbatch_salt*` twins for the other cluster), the French sweep submitter, and the rsync loop that mirrors the tree to the cluster |
| `data/` | shared | the `prep_*` builders for every behaviour organism's SFT set and probe file (language, cross-lingual, casing, pirate, spelling, mix, identity, German cities, inoculation pools), the EM prompt extractors, and the VarCon spelling-pair vendoring |
| `verify/` | shared | integration checks that run before a number is trusted: the dependency smoke test and the IxG / SVD / vLLM / StrongREJECT / OLMES / judge probes |
| `analysis/` | shared | readouts over finished runs: the sparsity AUC, the generations tables (HTML, and the paper's LaTeX), the paper's recipe and hyperparameter tables, the sweep browser UI, GSM8K rescoring, LoRA spectra, sampling a run on any prompt file |
| `probes/` | fr2de, bad_medical | the HF-vs-merged-adapter probe behind the save/reload warning in `CLAUDE.md`, and the 14B post-hoc memory probe |
| `refusal/` | refusal | abliteration of the refusal direction, its launcher, and the AdvBench / Alpaca data it needs |
| `olmpool/` | OlmPool | fetch and patch the 26 checkpoint pairs, generate their config trees, build the NIAH objective, the retrieval-head and head-statistics probes, the weight-level factorial, the analysis, and the launchers |
| `olmo3_base2inst/` | Olmo-3 post-training | the whole base -> Instruct delta under the Instruct tokenizer |
| `olmo3_post/` | Olmo-3 post-training | benchmark rollouts and their rescoring, IxG over every objective, the similarity / transfer / loss-transfer matrices and tables, the OLMES CLI wrapper and its venv patch, the RL-Zero relabel |
| `interference/` | interference toy | the Olah, Turner & Conerly replication: the toy model, the three attribution methods on its filtering task, and the filtered model's true loss -- the inputs of `plots/plot_interference_*.py` |

`CLAUDE.md` has the hazards worth knowing before changing any of it — particularly why the eval
registry must stay lazy, why the two weight-composition paths need `theta_base` in different
places, and the fact that `scripts/cluster/sync_to_cluster.sh` runs `--delete` over `data/` and
`configs/`.
