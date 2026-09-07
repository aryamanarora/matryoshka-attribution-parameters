# Which units of one post-training stage carry which benchmark? (Olmo-3, 2026-09-05)

**Question.** When several benchmarks go up in one post-training stage, is the improvement carried
by the same units of the weight update, or by different ones per benchmark?

**Setup.** The update is a public checkpoint pair of one pipeline: `delta = theta(Olmo-3-7B-Instruct)
- theta(Olmo-3-7B-Instruct-DPO)`, i.e. the RLVR stage of Olmo-3 Instruct post-training (32 layers,
`nonresid` units = 1,581,056 rows/columns over the 224 block projections; `tensor` = 224). Per-benchmark
objectives are the RL model's own correct greedy rollouts on prompts disjoint from each reported set
(`scripts/bench_rollouts.py` -> `data/bench/`), and rankings of the delta's units come from three
attribution methods that this repo already had for finetunes:

| method | what it is | where |
|---|---|---|
| **MAttr (learned)** -- THE method | scores fitted by the objective's SFT loss through the differentiable top-k, 300 steps, nonresid units and (`_tensor`) whole matrices | `configs/olmo3_post/posthoc/` -> `runs/olmo3_post/posthoc/` |
| GRPO | MAttr scores fitted against the benchmark metric itself as the reward (`rl.reward`) | `configs/olmo3_post/rl/` |
| IxG (`base` / `finetuned`) -- the closed-form BASELINE | one first-order term `-(delta . dL/dtheta)` per unit, gradient at the DPO or the RL endpoint; ~10 s per objective, so it also supplies the split-half ceilings cheaply | `scripts/bench_ixg.py` -> `runs/olmo3_post/ixg/<obj>_<at>` |

Every objective also exists as two disjoint halves (`_a`/`_b`), so each method has its own
**within-benchmark ceiling** (same benchmark, different prompts); the random floor for top-1%
Jaccard is 0.005. Cross-benchmark similarity is read between those two numbers.

Second readout: the **transfer matrix** (`scripts/bench_transfer.py`) -- a mask fitted on A, with the
top-k of the delta applied and everything else at the DPO weights, scored on B, normalised between
the DPO anchor (0) and the RL anchor (1). That is the causal version of the same question.

## 1. Anchors

**DPO → RL (Instruct pipeline).** Greedy, n = 200 (MMLU 512), from `runs/olmo3_post/posthoc/gsm8k`:

| | GSM8K | MATH-500 (1024 tok) | IFEval prompt-strict | MMLU (chat, 5-shot) | NLL of GSM8K rollouts |
|---|---|---|---|---|---|
| DPO (`pretrained`) | 87.0 | 55.0 | 75.5 | 48.0 | 0.140 |
| RL (`full_delta`) | 87.5 | 53.5 | 78.5 | 52.5 | 0.105 |

The RL stage of the Instruct pipeline barely moves these four benchmarks at greedy decoding (every
gap is inside one binomial SE except MMLU's +4.5, which is two), while the *likelihood* of the RL
model's own solutions does move (0.140 → 0.105). So on this delta the transfer matrix has no range and
only the rank-similarity readout is informative. The whole update is small: `||delta||_F = 3.31` over
the 224 block projections (a LoRA-r32 finetune in this repo is 23-97), spread almost uniformly over
depth (2.8-3.4% of `||delta||^2` per layer) and by type (MLP tensors ~22% each, attention 7-9%).
884 of 1,581,056 nonresid units are exactly unmoved.

The learned GSM8K mask over it is NOT a magnitude ranking (Spearman vs per-unit `||delta||` 0.15) and
is far from uniform: its top-1% concentrates in layers 9-15 and 28-31 (layer 31 alone holds 1,585 of
15,811 selected units, 5x the mean) and in `v_proj` (4.3% of its rows selected) and `o_proj` (1.75%)
against 0.4-0.8% for every other type -- i.e. the units the objective picks are concentrated where
the delta is not.

## 2. Rank similarity (DPO → RL delta)

`scripts/bench_similarity.py` over `runs/olmo3_post/ixg` (64 examples per objective for IxG; MMLU's
64 single-token examples make its ceiling low, see the 512-example rerun below). Diagonal = the
split-half ceiling (`_a` vs `_b` of the same benchmark); off-diagonal = whole-set vs whole-set.
Random floor for top-1% Jaccard is 0.005, for Spearman 0.

### 2a. MAttr (learned) masks -- the headline

**Learned masks (`configs/olmo3_post/posthoc/`, MAttr scores fitted 300 steps on each objective)
recruit mostly different units per benchmark, and no more than two METHODS agree on ONE benchmark:**

| learned, unit level: rho / J@1% | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| gsm8k | **0.72 / 0.65** (split-half) | 0.39 / 0.30 | 0.32 / 0.27 | -0.03 / 0.04 |
| math | | **0.83 / 0.80** (split-half) | 0.38 / 0.22 | -0.03 / 0.04 |
| ifeval | | | **0.81 / 0.74** (split-half) | -0.03 / 0.05 |
| mmlu | | | | **0.70 / 0.49** (split-half) |

The learned ranking's own ceilings (fitted on two disjoint halves of the rollouts) are 0.65-0.80
Jaccard / 0.72-0.83 Spearman on the three generative benchmarks and 0.49 / 0.70 on MMLU, so the
cross-benchmark overlap is ~30-45% of the within-benchmark one under this method against ~20-25%
under the IxG baseline, and MMLU's 0.04-0.05 is ~8% of its own ceiling (8x the random floor) -- the learned masks are a little more "shared" than the closed-form rankings,
but the ordering is the same: three generative benchmarks partly overlap, MMLU is disjoint.

**Tensor-level masks fitted directly (`*_tensor`, one score per matrix, 224 units)** say the same
thing as the aggregated IxG: GSM8K-MATH rho 0.80, GSM8K-IFEval 0.74, MATH-IFEval 0.91, and each of
them vs MMLU -0.003 / -0.03 / -0.10. (Aggregating a NONRESID learned mask to tensors by its top-1%
occupancy gives MMLU a spurious 0.7 with the others -- that number is the method's shared preference
for `v_proj`/`o_proj` rows, which is why the directly fitted tensor masks are the ones reported.)

For scale, the SAME benchmark ranked by two methods (learned vs IxG at the DPO endpoint): gsm8k
0.36 / 0.27, math 0.55 / 0.59, ifeval 0.60 / 0.63, mmlu 0.46 / 0.22; vs IxG at the RL endpoint:
0.51 / 0.38, 0.61 / 0.39, 0.52 / 0.48, 0.49 / 0.26. So two different methods ranking one
benchmark agree MORE than one method ranking two benchmarks, on every pair -- the benchmark, not
the attribution method, is what the ranking is mostly about. At tensor level the learned masks
agree at rho 0.83-0.90 across the three generative benchmarks and, unlike IxG, also with MMLU
(0.68-0.71): the learned masks' tensor preference (`v_proj`/`o_proj` rows everywhere) is a property
of the method as much as of the benchmark, so tensor-level agreement between learned masks is the
weaker statement.

### 2b. The IxG baseline, same question

**Gradient at the DPO endpoint (`ixg_base`), 1,581,056 nonresid units:**

| top-1% Jaccard | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| gsm8k | **0.74** | 0.16 | 0.14 | 0.07 |
| math | 0.16 | **0.82** | 0.20 | 0.07 |
| ifeval | 0.14 | 0.20 | **0.70** | 0.08 |
| mmlu | 0.07 | 0.07 | 0.08 | **0.35** |

| Spearman | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| gsm8k | **0.70** | 0.19 | 0.16 | -0.01 |
| math | 0.19 | **0.77** | 0.28 | -0.01 |
| ifeval | 0.16 | 0.28 | **0.57** | -0.01 |
| mmlu | -0.01 | -0.01 | -0.01 | **0.46** |

**The same rankings aggregated to TENSORS (224; IxG is additive, so a tensor's score is the sum of its
rows') and to LAYERS (32), Spearman:**

| tensor | gsm8k | math | ifeval | mmlu |  | layer | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.99 | 0.62 | 0.75 | 0.18 |  | gsm8k | 0.96 | 0.39 | 0.61 | 0.21 |
| math | 0.62 | 1.00 | **0.90** | 0.03 |  | math | 0.39 | 1.00 | **0.87** | -0.05 |
| ifeval | 0.75 | 0.90 | 1.00 | 0.10 |  | ifeval | 0.61 | 0.87 | 0.99 | 0.08 |
| mmlu | 0.18 | 0.03 | 0.10 | 0.83 |  | mmlu | 0.21 | -0.05 | 0.08 | 0.98 |

Three readings:

1. **At the unit level the benchmarks recruit mostly different units.** Cross-benchmark top-1%
   overlap is 0.14-0.20 between the three generative benchmarks against split-half ceilings of
   0.70-0.82: roughly a fifth of the within-benchmark agreement, and 30-40x the random floor. So there
   IS a shared component, and it is small.
2. **"Maths is one circuit" is not what the units say.** GSM8K-MATH (0.16 / rho 0.19) is no closer
   than MATH-IFEval (0.20 / 0.28); the within-domain pair sits at the same distance as the
   cross-domain pairs. The single-token MMLU objective is orthogonal to all three (0.07-0.08, rho ~0).
3. **Granularity changes the answer.** Aggregated to tensors the three generative benchmarks agree
   strongly (MATH-IFEval 0.90, GSM8K-IFEval 0.75; ceilings ~1.0) while MMLU stays orthogonal (0.03-
   0.18). The benchmarks land on the same MATRICES and, inside them, on different rows: "same
   circuitry" at tensor resolution, "different circuitry" at unit resolution, for the same rankings.

**What the shared units are** (`scripts/bench_shared_units.py`, the 2,755 units in all three
generative top-1% sets; random expectation 1.6): 42% are `v_proj` rows (base rate 8%) and 19%
`o_proj`, 19% `down_proj`; `q_proj` is 1% and `gate_proj` 5%. They concentrate in layers 9-18 and
28-31 (layer 31 alone: 305). Each benchmark's own top-1% is also `v_proj`-heavy (36-43%), so the
RL stage's benchmark-relevant units are predominantly attention VALUE rows -- what a head writes,
not what it attends to.

**MMLU at 512 examples** (`runs/olmo3_post/ixg512`, `plots/data/olmo3_post/similarity_ixg512.json`):
split-half ceiling J@1% 0.69 / rho 0.86 (0.60 / 0.77 at the RL endpoint), against 0.35 / 0.46 with
64 examples -- so the low MMLU diagonal above is sample size, and the 512-example ranking agrees
with the 64-example one at 0.40 / 0.55. Its overlap with GSM8K / MATH / IFEval is 0.08 / 0.06 /
0.10 (rho −0.01 / 0.00 / 0.01), i.e. unchanged: MMLU's orthogonality is real.

**Gradient at the RL endpoint (`ixg_finetuned`)** has lower ceilings (J 0.41 / 0.64 / 0.65 / 0.36;
rho 0.37-0.54): the local ablation term is noisier than the extrapolating one on this small delta.
Relative to its own ceilings the cross-benchmark overlap is higher (GSM8K-MATH 0.29 against 0.41 /
0.64), same ordering otherwise; at tensor level MMLU turns *negative* against the others (-0.08 to
-0.30).

## 3. Transfer (DPO → RL delta)

`scripts/bench_transfer.py` -> `plots/data/olmo3_post/transfer.json`. On this delta the two anchors
are within noise of each other on every benchmark (section 1; MATH-500 at the corrected 2048-token
budget is 71.0 -> 72.5), so the normalised "fraction of the gain carried" is a ratio of two noise
terms and is NOT reported as a result. What the sweeps do establish: no top-k slice of the RL
update at 0.2-20% of units moves any benchmark outside the anchors' band (GSM8K 86-88, MATH-500
70-73, IFEval 75-79, MMLU 48-51 at every sparsity, every mask) -- i.e. the RL stage's units are not
individually harmful either, and a benchmark-fitted mask is a null intervention on this delta.

### 3b. The LOSS transfer matrix (the readout that has range on this delta)

`scripts/bench_xloss.py` -> `runs/olmo3_post/xloss/matrix.json`, tabulated by
`scripts/bench_xloss_table.py`: objective B's held-out NLL (64 rows of B's rollouts) under the mask
fitted on A, at each sparsity, as the fraction of the DPO->RL loss drop on B that A's top-k carries
(`(pre - loss_k) / (pre - full)`; >1 = the slice fits B better than the whole RL update). Anchors:
GSM8K 0.110 -> 0.078, MATH 0.149 -> 0.103, IFEval 0.376 -> 0.292, MMLU 2.05 -> 1.68.

| learned mask, top-1% | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| fitted on gsm8k | **0.79** | 0.46 | 0.38 | 0.07 |
| fitted on math | 0.51 | **0.65** | 0.35 | 0.05 |
| fitted on ifeval | 0.39 | 0.39 | **0.56** | 0.16 |
| fitted on mmlu | 0.12 | 0.13 | 0.11 | **1.51** |

| learned mask, top-5% | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| fitted on gsm8k | **1.15** | 0.73 | 0.61 | 0.25 |
| fitted on math | 0.79 | **0.93** | 0.60 | 0.16 |
| fitted on ifeval | 0.62 | 0.67 | **0.86** | 0.49 |
| fitted on mmlu | 0.29 | 0.28 | 0.25 | **2.37** |

| learned mask, top-20% | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| fitted on gsm8k | 1.32 | 0.95 | 0.84 | 0.49 |
| fitted on math | 1.03 | 1.12 | 0.85 | 0.41 |
| fitted on ifeval | 0.86 | 0.91 | 1.09 | 0.74 |
| fitted on mmlu | 0.51 | 0.51 | 0.44 | **2.83** |

The IxG-at-DPO masks give the same matrix to within ~0.05 everywhere (top-1% diagonal 0.76 / 0.65 /
0.55 / 1.50, off-diagonal 0.30-0.51, MMLU row 0.15-0.23); see the script's output for all eight rows.

Four readings:

1. **Shared, but not the same: a 1% slice picked for one generative benchmark carries about half
   of what it carries for its own on another** (0.35-0.51 against 0.56-0.79), and at 20% the
   cross-benchmark cells reach 0.84-1.03 -- by then any generative benchmark's ranking has swept up
   most of what the others need. Consistent with the unit-level Jaccard (0.15-0.30 of the ceiling at
   1%): the top of each ranking is benchmark-specific, the next few percent are common.
2. **GSM8K and MATH are not privileged partners.** GSM8K's mask carries 0.46 of MATH and 0.38 of
   IFEval at 1%; MATH's carries 0.51 of GSM8K and 0.35 of IFEval. The within-domain pair is a
   little closer than the cross-domain ones on the loss (0.46-0.51 vs 0.35-0.39) where the unit
   overlap saw no difference at all, i.e. the "maths circuit" is at most a weak preference.
3. **MMLU is a different circuit, and the rest of the RL update works AGAINST it.** MMLU's mask
   carries 0.11-0.13 of the others at 1% and the others carry 0.05-0.16 of MMLU; and on its own
   objective the MMLU mask reaches 1.5x / 2.4x / 2.8x the full update's loss drop at 1 / 5 / 20%
   (NLL 0.998 at 20% against 1.68 for the whole delta and 2.05 for DPO). So the RL stage contains a
   ~20% subset that improves MMLU far more than the whole stage does, and the other 80% undoes most
   of it -- the "unconditional core plus suppressive remainder" structure this repo found in the
   fr2de ablations, here on a capability benchmark and a public post-training delta. The behavioural
   version (section 3, 56.1 vs 52.5 accuracy at 20%) is the same thing measured coarsely.
4. **Every generative mask beats the full update on its own objective by 5-20%** (1.15 / 0.93 /
   0.86 at 5%; 1.32 / 1.12 / 1.09 at 20%) -- the same reactivation shape, smaller because these
   objectives ARE what the RL stage optimised, so less of the update opposes them.

**MMLU is the exception, and it is the one benchmark with a real gap -- and there a SPARSE slice
of the RL update beats the whole update.** DPO 48.0 -> RL 52.5 (n = 512, SE 2.2). The MMLU-fitted
mask reaches 50.8 / **54.7 / 56.1** at 1% / 5% / 20% of units, while the GSM8K-, MATH- and IFEval-
fitted masks reach 48.6-50.4 at 20% -- about half the gain, and no more than the RL model at any
sparsity. On the objective itself the same thing is starker: the MMLU letter NLL is 1.907 at DPO,
1.355 at RL, and **0.772** under the MMLU mask's top-20% (0.95 at 5%). The RL update therefore
contains a subset of units that, applied alone, improves MMLU ~2x more than the full update does
(the rest of the update pulls the other way), and that subset is not the one any other benchmark's
objective selects (unit-level overlap with the other three: 0.04-0.08 Jaccard, rho ~0). The same
"sparse mask beats the full delta" shape appears on every objective's own loss (GSM8K 0.073 at 20%
vs 0.084 full; MATH 0.086 vs 0.092; IFEval 0.276 vs 0.283) but only MMLU turns it into a benchmark
number, because only MMLU's anchors are apart.

**GRPO on the metric itself does not produce a ranking here, for the same reason.** `rl/gsm8k`
(200 steps, reward = GSM8K correctness on 2,000 disjoint train questions): the DPO model already
scores 0.90 on the reward prompts at step 0, fewer than one group in four is informative at any
step, the reward ends at 0.94, and the resulting scores are uncorrelated with everything --
Spearman 0.045 / top-1% Jaccard 0.044 against the learned GSM8K mask, 0.03 / 0.07 against IxG,
-0.03 against `||delta||`. A metric-reward can only rank the units of an update that changes that
metric; on a flat anchor pair it is REINFORCE on noise. `rl/ifeval` is the harder case: its reward
is NOT saturated (0.50 at step 0, 0.65-0.75 through the run, 1.2-1.4 of 4 groups informative per
step) and the ranking is STILL uncorrelated with everything -- rho -0.008 / J@1% 0.023 against the
learned IFEval mask, 0.003 against `rl/gsm8k`, -0.004 against `||delta||`. ~10 informative samples
per step for 200 steps is not a budget that ranks 1.58M units; the refusal GRPO runs in this repo
needed 300 steps at a graded reward and a 0.5-wide behavioural gap to get a usable operating point.
(The RL-Zero delta, where the gaps are large, is where this arm belongs; see section 4.)

## 4. The RL-Zero delta (base -> RL-Zero-Mix): the arm with real gaps, and what stopped it

`configs/olmo3_rlzero/`. Two conversion facts first (both handled by `scripts/olmo3_rlzero_patch.py`):
the RL-Zero checkpoints ship `model_type: olmo2-retrofit` (a pre-release name; the config is
otherwise Instruct's and the weight names are identical), and their tokenizer names ids
100256-100275 `<think>`, `</think>`, `<functions>`, ... as special tokens where the base's names
them `<|extra_id_*|>` -- so the base model is served under the RL-Zero tokenizer + template
(`models/olmo3_rlzero/Base`), tokenisation verified identical.

**Greedy decoding does not work on this model.** Under its own template (generation prompt ends in
`<think>`) the Mix model decoded greedily entered a repetition loop inside the think block on
600/600 GSM8K prompts (median 6,318 characters of "Wait, 4000 + 40,000 is 44,000?" to a 3,072-token
cap, 0 closed think blocks, 0 answers) and likewise on 1,500 MATH and 2,000 IF prompts -- so the
first three rollout passes produced no objective at all. Sampling (T 0.6, top-p 0.95, 4K
cap for GSM8K; `bench_rollouts.py --temperature`) ends the loops -- and exposed a second format
fact: **the model never emits `</think>`.** It reasons and then writes its `#### N` line inside the
still-open block (341/400 sampled GSM8K rollouts carry a `####` answer, 0 carry the closing tag,
median 4,267 characters), so a stripper that treats an unclosed block as "no answer" scores it at
0. `strip_think` now returns an unclosed block whole (the extractors read the LAST marker), and the
sampled GSM8K pass rescored under that rule keeps **224/400 (56%)** -- the RL-Zero-Mix objective
that exists. It also answers the five few-shot exemplars before the real question ("So all the
answers are 72, 10, 5, 42, 624 ... #### 624"), which the last-marker rule handles. The MATH and IF
passes under the same settings, and the LEARNED (MAttr) cells on all three objectives
(`configs/olmo3_rlzero/posthoc/*_loss.yaml`: nonresid, 300 steps, gradient checkpointing for the
4K-token rows, loss-only sweeps because the model cannot be decoded greedily), are the last jobs of
the session (status in section 5).

### 4a. RL-Zero results (learned masks, loss-only; `||delta||_F` = 2.78, 2,324 dead units)

Held-out NLL of the Mix model's own sampled rollouts under the mask fitted on each objective
(base = `pretrained`, RL-Zero-Mix = `full_delta`):

| objective | base | 0.2% | 1% | 5% | 20% | full RL-Zero |
|---|---|---|---|---|---|---|
| GSM8K (fitted on GSM8K) | 0.234 | 0.224 | 0.209 | 0.179 | **0.154** | 0.183 |
| IFEval (fitted on IFEval) | 0.347 | 0.341 | 0.323 | 0.261 | **0.199** | **0.360** |
| MATH (fitted on MATH) | 0.203 | 0.191 | 0.179 | 0.158 | **0.141** | 0.160 |

**Cross-benchmark similarity, unit level (rho / top-1% Jaccard), with the IxG split-half ceilings
of this delta for scale** (`plots/data/olmo3_post/similarity_rlzero.json`):

| | learned masks | IxG at base | IxG split-half ceiling (base) |
|---|---|---|---|
| GSM8K vs MATH | **0.55 / 0.39** | 0.48 / 0.39 | GSM8K 0.90 / 0.77, MATH 0.93 / 0.85 |
| GSM8K vs IFEval | 0.24 / 0.18 | 0.24 / 0.17 | IFEval 0.67 / 0.51 |
| MATH vs IFEval | 0.28 / 0.11 | 0.20 / 0.12 | |

Two readings. (1) **Here GSM8K and MATH ARE a pair**: their top-1% overlap (0.39) is ~half their
ceilings and 2-3.5x their overlap with IFEval (0.11-0.18), under both methods -- where the DPO->RL
stage showed no maths pairing at all (0.16 vs 0.20). RL that explicitly trains maths installs a
shared maths subspace; the Instruct pipeline's RL stage, which barely moved maths, does not. (2)
IFEval is again the odd one out (and its own ceiling is lower, 0.51: a heterogeneous objective).
Tensor level: learned GSM8K-MATH 0.82, GSM8K-IFEval 0.75, MATH-IFEval 0.43.

**The IFEval row is the sharpest "sparse beats full" in the repo so far: the WHOLE RL-Zero update
makes the model's own IFEval-passing responses LESS likely than the base model does (0.347 ->
0.360), while the IFEval-fitted 20% of it takes them to 0.199** -- nearly halving the NLL. RL on a
math+code+IF mixture installed instruction-following units and, in the other 80% of the update,
something that overrides them on these prompts. GSM8K shows the milder version (0.154 at 20% vs
0.183 full). This is the RL-Zero delta's version of the DPO->RL MMLU finding, on the benchmark the
update was explicitly trained for.

## 4b. Are the 300-step learned masks converged? (partial)

`train_log.json` of every 300-step cell: `score_std` rises almost linearly the whole run (GSM8K 0.49 →
1.11, MMLU 0.58 → 1.16, GSM8K-tensor 1.0 → 2.1) with no flattening, the score-gradient norm is still
~40% of its initial value at step 300, and the loss at fixed k is still drifting down (GSM8K @5-30%:
0.100 → 0.085; MMLU 1.78 → 1.18). Under `k_schedule: uniform` only ~1 step in 12 samples k < 5%, so the
top-1% ranking that every Jaccard number is about gets little direct signal. 900-step cells with
checkpoints every 100 (`posthoc/{gsm8k_long,gsm8k_long_log,mmlu_long}`, `scripts/bench_convergence.py`
to read them) were started and cancelled at ~step 250 in favour of the round below; the question is
open and the split-half ceilings (0.65-0.80) should be read as "agreement of two 300-step fits", not
"of two converged fits".

## 6. Second round: RL-fitted MAttr on the SFT -> DPO stage (in progress)

**Unit set changed on 2026-09-05 18:20 UTC, on the user's instruction: ALL parameters are scored
(`exclude_params: null`)** — embedding rows, `lm_head` rows and every norm gain join the 224 block
projections so `full_delta` is exactly the DPO checkpoint: **1,781,741 units over 355 tensors** = the 1,581,056
block-projection rows/columns + 200,556 `embed_tokens`/`lm_head` rows + 129 norm gains (one unit per
norm VECTOR under `nonresid`, not per element).
The two finished cells and the two running AIME cells were on the old set; they are archived under
`runs/olmo3_sft2dpo/_excluded_params/` and every cell below was resubmitted (jobs 284152-60) with
`k_schedule: uniform`. Everything in sections 1-5 (the DPO->RL and RL-Zero deltas) is still on the
block-projections-only set.

`configs/olmo3_post/rl_sft2dpo/`: delta = Instruct-DPO − Instruct-SFT, one GRPO cell per benchmark
(MATH-500, AIME 2024, AIME 2025, HumanEval+, MMLU, IFEval; AlpacaEval excluded), each reporting every
benchmark under its mask. Reported gains of this stage (model card): MATH 65.1 → 79.6, AIME24 6.7 →
23.5, AIME25 7.2 → 20.4, HumanEval+ 69.8 → 72.9, IFEval 81.7 → 82.0, MMLU 67.1 → 69.1. Budget 400 ×
8 × 8 (4x the failed cells of section 3), `k_schedule: log`. Jobs 283448-53; the per-user 2-node cap
runs two at a time, ~2 h each.

**Two facts from the first two cells (MATH, IFEval; both at 1024 new tokens):**
- **The 1024-token budget is a confound for MATH.** Under the full DPO delta 38% of MATH-500
  solutions have no boxed answer within 1024 tokens (3% under SFT), so the in-house MATH-500 column
  reads SFT 65.0 → DPO 55.0 where the card says 65 → 80, and the MATH GRPO reward (same budget)
  partly paid for brevity. The remaining MATH/AIME cells run at 4096; the OLMES re-evaluation
  (section 7) at 8192 is the number to read. IFEval and MMLU are unaffected (IFEval-strict 76.0 →
  73.5 greedy; MMLU 42.8 → 48.0).
- **Under `k_schedule: log`, 70% of steps sample k < 1% of units**, so the reward mostly reports the
  SFT model and the delta is rarely in play (MATH reward 0.63 at k<1% vs 0.65 at k>10%; IFEval 0.73
  vs 0.71, i.e. the DPO delta does not help the Tulu IF prompts either). `*_uniform` twins of the
  MATH and AIME24 cells are queued.
- **The two GRPO rankings do not agree with each other**: `rl/math500` vs `rl/ifeval` rho −0.001 /
  top-1% Jaccard 0.019 (floor 0.005) at unit level, 0.86 / 0.93 at tensor / layer level (the
  tensor-level agreement of two near-random unit rankings is the score-update magnitude by tensor,
  not a shared circuit). Same verdict as on the DPO->RL delta (section 3): 400 x 8 x 8 GRPO does not
  rank 1.58M units. Their sweeps move nothing outside the anchors' noise except MMLU (both carry
  0.3-0.5 of the SFT->DPO MMLU gain at 20%). Same-delta references are queued so this can be read
  against something: IxG rankings with split halves (`runs/olmo3_sft2dpo/ixg`) and SFT-loss-fitted
  MAttr on MATH and IFEval (`configs/olmo3_post/sft2dpo/`).

**First all-parameter cell (IFEval, uniform k, 400 x 8 x 8; `runs/olmo3_sft2dpo/rl/ifeval`):** the
reward on the Tulu IF prompts is flat across the run (0.72-0.74 per 100 steps) and goes DOWN with k
(0.76 at k 5-30% of units, 0.73 at 30-70%, 0.69 at 70-100%): on this reward the DPO delta is a
mild negative, so the mask has nothing to find. Its ranking is at the floor against every closed-form
ranking of the same delta -- rho 0.063 / top-1% Jaccard 0.033 vs the IxG-IFEval ranking (ceiling
0.68) -- and its sweep says the same thing the reward did: IFEval-strict 76.0 (SFT) -> 73.5 (DPO),
with 0.2-1% masks at 78.5, above both anchors. Side columns under that mask: HumanEval+ 72.0 -> 72.6
(card 69.8 -> 72.9; 5% mask 75.6), MMLU 42.8 -> 47.5 (the mask carries 1/4 of it at 20%), MATH-500
still at the 1024-token budget (55 at DPO = truncation).

**HumanEval+ cell (all params, uniform k; `runs/olmo3_sft2dpo/rl/humaneval`):** the MBPP+ reward
rises only mildly with k (0.67 at k<5% -> 0.71 -> 0.72 -> 0.73 at k>70%) and not across the run
(0.72 -> 0.72), and the ranking is at the floor against everything (rho 0.00-0.03 vs the delta's IxG
rankings, 0.001 vs the IFEval GRPO ranking). In-house HumanEval+ anchors 72.0 (SFT) -> 72.6 (DPO)
against the card's 69.8 -> 72.9, with 5% masks at 75.6; the mask carries ~half the MMLU gain at 20%
(45.7 of 42.8 -> 47.5).

**MATH cell (all params, uniform k, 4096 tokens, 400 x 8 x 8; `runs/olmo3_sft2dpo/rl/math500`) --
the first GRPO-fitted MAttr with a result.** With the truncation confound gone the in-house anchors
reproduce the card: MATH-500 **65.5 (SFT) -> 79.0 (DPO)** (card 65.1 -> 79.6; `no_answer_frac` 0.03 ->
0.12). The reward on MATH-train tracks k (0.61 at k<5%, 0.73 at 5-30%, 0.77 above) and rises across
the run (0.73 first half -> 0.77 second), and the ranking it produces is sparse: **the top-5% of the
SFT->DPO update carries the entire MATH-500 gain (80.0), 1% carries 40% of it (71.0), 20% is at the
full delta (79.0)**. What that 5% does elsewhere: MMLU 44.9 of 42.8 -> 47.5 (45% of the gain; 67% at
20%), HumanEval+ 73.8 of 72.0 -> 72.6 (all of it, and 1% overshoots to 76.8), IFEval unchanged
(76 -> 74 at both ends). Spearman vs `||delta||` 0.13.
Its ranking is the first GRPO one that agrees with anything: vs the same delta's IxG-MATH ranking
rho 0.19 / top-1% Jaccard 0.16 (IxG's own split-half ceiling 0.80; floor 0.005), more than vs
IxG-GSM8K (0.13), IxG-IFEval (0.11) or IxG-MMLU (0.07), and essentially disjoint from the IFEval and
HumanEval+ GRPO rankings (0.016 / 0.029). So a GRPO cell CAN rank when the reward moves with the
delta -- and the MATH-carrying 5% it finds is benchmark-specific at unit level while still carrying
half of MMLU's gain, the same "different units, partly shared effect" pattern as the SFT-loss masks.

**AIME 2024 cell (all params, uniform k, 4096 tokens, 150 x 8 x 8; reward = AIME 1983-2023;
`runs/olmo3_sft2dpo/rl/aime2024`):** the reward tracks k as MATH's did (0.12 at k<5% -> 0.25 -> 0.35
-> 0.37 at k>70%) with 3-4 informative groups a step. **The two maths GRPO rankings agree with each
other -- rho 0.21 / top-1% Jaccard 0.21 vs `rl/math500` -- more than either agrees with any IxG
ranking (0.08-0.12) and 40x the floor**; so GRPO on two maths rewards recovers a shared maths
subspace of the DPO update, the analogue of the SFT-loss result on RL-Zero (section 4a). The AIME
2024 sweep itself (n = 30, greedy, 4096 tokens) reads SFT 10.0 -> DPO 26.7 (card 6.7 -> 23.5,
sampled) with the 5% mask at 26.7 and 20% at 30.0 -- but `no_answer_frac` climbs 0.03 -> 0.57 along
the way, i.e. even 4096 tokens truncates more than half of the DPO-side AIME solutions, so those
accuracies are floors and the OLMES 16K re-evaluation is the number to read. Side columns under this
mask: HumanEval+ 78.0 at 20% (anchors 72.0 -> 72.6), MMLU 46.3 of 42.8 -> 47.5 at 20%, IFEval flat.

**AIME 2025 cell:** the first run's 150 GRPO steps (6 h; reward 0.11 -> 0.38 by k, same shape as
AIME 2024) were lost at the post-fit eval -- MathArena's AIME 2025 ships integer answers and
`math500.normalize` called `.strip()` on one -- and no checkpoint existed yet because the loop's
"save before the final sweep" sat after that first eval. Both fixed (golds coerced; GRPO scores are
now saved the moment the fit returns) and the cell is rerunning (job 286477).

### 6a. The SFT -> DPO delta itself (IxG at SFT, all parameters, `runs/olmo3_sft2dpo/ixg`)

`||delta||_F` = 2.79 over 355 tensors; by type the update is 57% MLP (down 22 / up 18 / gate 17),
40% attention (v 14 / o 12 / k 8 / q 6), **`lm_head` 2.4%, `embed_tokens` 0.4%, norms 0.0%** of
`||delta||^2`. 6,255 units are exactly unmoved.

| top-1% Jaccard (diag = split-half) | gsm8k | math | ifeval | mmlu |
|---|---|---|---|---|
| gsm8k | **0.89** | 0.15 | 0.12 | 0.09 |
| math | 0.15 | **0.80** | **0.30** | 0.08 |
| ifeval | 0.12 | 0.30 | **0.68** | 0.09 |
| mmlu (64 ex.) | 0.09 | 0.08 | 0.09 | **0.51** |

Same shape as the DPO->RL stage (mostly different units, MMLU orthogonal) with one difference: on
this stage **MATH and IFEval are the closest pair (0.30), twice GSM8K-MATH (0.15)** -- the DPO stage's
maths and instruction-following gains overlap more than its two maths benchmarks do. Where the
vocabulary and norms land: each benchmark's top-1% is 0-2% `lm_head` rows and ~0% `embed_tokens`
(their share of the delta), but 4 of the 32 `post_attention_layernorm` and 4 of the 32
`post_feedforward_layernorm` gains (one unit = one whole norm vector) are in all three generative
top-1% sets -- a norm vector aggregates the first-order term of 4,096 elements, so as a unit it
outranks any single row. The 3-way core is again attention-output-side: `v_proj` 38% and `o_proj`
33% against base rates of 7%, in layers 3-18.

## 7. OLMES: the model cards' own evaluation code, hooked in

Two routes, both on the repo's rule that a borrowed metric is never reimplemented (CLAUDE.md has the
map and the environment hazards):

- `eval/olmes.py` — their `Task` objects (prompt template, chat message construction, answer regexes,
  Minerva/Hendrycks normalisers, code executor, pass@k, aggregation) run inside the sparsity sweep with
  this repo's engine at their sampling parameters. `configs/olmo3_post/eval_olmes.yaml` re-evaluates
  any run directory under `aime:2024/2025`, `minerva_math` (7 subtasks), `gsm8k`,
  `codex_humanevalplus`, `ifeval` and `mmlu:cot` `::olmo3:adapt` specs; the two knobs that deviate
  from a real `olmes` run (`max_gen_toks` cap, `repeats`) are recorded per split.
  `scripts/verify_olmes.py`: every family scores 1 on its own gold answer and 0 on a wrong one.
- `scripts/olmes_cli_eval.py` — the real `olmes` CLI in OLMES's own venv, on a hub id or on one
  composed sweep condition saved as an HF directory: bit-faithful to the card. The SFT and DPO
  anchors at card budgets are the first use -- ONE JOB PER TASK (`runs/olmes_cli/instruct_{SFT,DPO}/<task>`),
  because their CLI writes its outputs only when every task in the invocation is done and a
  7-task suite at 32 x 16K-token AIME samples runs past the 12 h limit (the first attempt took
  1.5 h per AIME year alone).

**Card-budget anchors** (`scripts/olmes_cli_eval.py`, one task per job, their sampling, token budgets
and repeats; primary metric as OLMES reports it; the card's numbers in the last column). **They
reproduce the model card to within ~0.5 points**, which is the check the whole hook-in was for:

| task (`::olmo3:adapt`) | Instruct-SFT | Instruct-DPO | card SFT / DPO |
|---|---|---|---|
| aime:2024 (pass@1 over 32 samples) | **0.064** | pending | 6.7 / 23.5 |
| aime:2025 (pass@1 over 32 samples) | **0.079** | pending | 7.2 / 20.4 |
| ifeval (prompt-level loose) | pending | **0.810** (strict 0.771) | 81.7 / 82.0 |
| gsm8k (exact_match_flex) | 0.892 | 0.906 | not on the card |
| minerva_math (7-subject macro) | algebra 0.867, counting 0.656 so far | pending | 65.1 / 79.6 |
| mmlu:cot (57-subject macro) | 0.668 over 9 subjects so far | pending | 67.1 / 69.1 |
| codex_humanevalplus (pass@1 over 10) | rerunning (first run 0.18 = the fork/filelock artefact, see below) | pending | 69.8 / 72.9 |

The in-house greedy numbers of section 6 are NOT these: AIME 10.0/26.7 at greedy 4096 tokens with
half the DPO solutions truncated, IFEval-strict 76.0/73.5 vs their loose 81.7/82.0.

The six SFT→DPO GRPO cells of section 6 report the in-house evals (they started before this
existed); their `final.pt` is re-scored under `eval_olmes.yaml` afterwards. New reward hooks through
OLMES (`eval.olmes.reward_task`: MBPP+ for HumanEval+, IFBench for IFEval, MATH-train / AIME
2021-23 by `reward_overrides`) exist but no GRPO cell has used them yet.

## 5. What is not done

- **RL-Zero cross-benchmark matrix is thin**: three learned cells (GSM8K, MATH, IFEval; loss-only
  sweeps) and the IxG split-half ceilings, from 224-250 sampled rollouts per objective; no MMLU
  objective (a thinking model has no single-token letter format), no tensor-level cells, no GRPO,
  no behavioural sweep (greedy decoding loops; a sampled sweep needs `eval.*.temperature > 0`, which
  every exact eval supports but nothing here ran). MATH rows at a median 10K characters overrun
  `max_seq_length: 4096`, so part of that objective is reasoning-trace likelihood with the answer cut.
- **No behavioural transfer matrix with range.** The DPO->RL anchors are flat; the RL-Zero one was
  not measured. A sampled RL-Zero sweep (T 0.6, 4-8K tokens, n = 100-200 per benchmark) is the cell
  that would turn the loss matrix into accuracy numbers.
- **GRPO produced no usable ranking on either objective** at 200 steps x 4 prompts x 8 samples; the
  refusal runs needed 300 steps and a graded reward. A metric-reward GRPO cell belongs on the RL-Zero
  delta with a larger group count, and has not been run.
- **One seed everywhere.** The learned masks' split-half ceilings bound sampling noise in the
  OBJECTIVE, not the fitting's own seed sensitivity (the ablation Jaccard work found ~0.2 seed floors
  for top-1% sets over different finetunes; here the delta is fixed, so the floor should be higher,
  but it was not measured).
- Only the RL stage (DPO->RL) and RL-Zero; `configs/olmo3_post/base_sft2dpo.yaml` (SFT->DPO) is
  written and unrun.

