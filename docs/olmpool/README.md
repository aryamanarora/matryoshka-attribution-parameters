# Long-context-specific retrieval heads in OlmPool, by mask learning over the extension delta

**Status (2026-09-03/04, one night of H100 time on 2-4 GPUs):** procedure built, verified at toy
scale and run on 24 of the 26 models for the weight-level factorial, 9 for the all-units mask,
7 for the head-only masks, 5 for the retrieval-head and attention-statistics probes; six
remote-code SWA checkpoints are set aside (see the caveats). Everything is reproducible from
`configs/olmpool/`, `scripts/olmpool/olmpool_*.py` and `scripts/olmpool/submit_olmpool.sh`.

## The question

Bertsch et al. (2026, "Cracks in the Foundation", arXiv:2608.10296) train 26 otherwise-identical
7-8B models (OlmPool) that differ in four architectural choices -- norm placement / QK-norm
(none, layerwise, headwise), GQA (4-32 kv heads), sliding-window attention (3 local : 1 global,
window 4096) and pretraining context (4K / 8K) -- for 140B tokens, then extend every one of them
to 64K context with the same 10B-token recipe (RoPE theta 500K -> 8M, Longmino mix, LR annealed to
0). HELMET-32K spans 29.9 (headwise-QK + post-norm + SWA + fp8, 8 kv) to 56.4 (prenorm, no QK
norm, 16 kv, no SWA); the paper's narrative is that these "minor" features compound, that short
context metrics do not predict the gap, and mechanistically that QK-norm models place less
attention on the needle and have weaker attention sinks, while their generation-time retrieval
head scores (Wu et al. 2024) barely differ across models -- "these models may be too weak to
reliably identify retrieval heads".

Wu et al. (2024, arXiv:2404.15574) find retrieval heads to be *intrinsic*: they exist in the
short-context base model, and continued long-context pretraining changes the per-head retrieval
score map little (correlation > 0.8 base vs extended). If that is right, what the extension
training does is not *create* retrieval heads but *extend the range* of the ones already there --
and which heads' weights carry that extension, and how many of them it takes, is an attribution
question over the weight delta between the two checkpoints. That is exactly what this repo's
mask learning (MAttr, `learning-to-attribute`) does.

## The procedure

1. **Two public checkpoints, one delta.** For each model, `step34000` (end of pretraining) and
   `longcontext-step2385` (end of extension). `delta = theta_LC - theta_PT`, frozen.
   `scripts/olmpool/olmpool_fetch.py` lays them out under `models/olmpool/<name>/{pt,lc,pt_ext}`.
   **The base of every attribution is `pt_ext`: the pretraining weights under the long-context
   config** (rope theta 8M). The composed model `theta_base + m . delta` has to be run under one
   positional encoding, and only the extended one makes `full_delta` the released long-context
   model. So the `pretrained` anchor of every sweep is *zero-shot theta scaling* of the pretrained
   weights -- which is also the point the 10B extension tokens trained from.
2. **The objective is retrieval beyond the pretraining window.** `scripts/olmpool/prep_niah_data.py`
   builds RULER-style single-needle rows ("One of the special magic numbers for <adj noun> is:
   <7 digits>.") in a Paul Graham essay haystack, measured in the OlmPool tokenizer's tokens:
   training rows at **12K and 16K** prompt tokens (every model was pretrained at 4K or 8K), the
   loss on the answer digits only (`loss_mask: response_only` under the `plain` template).
   Eval prompts are disjoint (needle values, depths, essay offsets) at **1K, 4K, 8K, 16K, 32K**.
3. **Unit = attention head, everything else folded to the extended values.** `mask.unit: head`
   (added for this: `masks/layout.py`) ties one head's q rows with its o columns, and one kv
   group's k rows with its v rows -- under MHA (32 kv) all four projections tie into one unit
   per head, under GQA a layer has `n_heads` query-head units plus `n_kv_heads` kv-group units.
   The primary `fold_*` arms score just those and **fold** the rest of the extension (MLP,
   norms, embeddings) into the base (`mask.fold_params`, added for this), so the sweep's
   `pretrained` anchor is "pretrained attention over extended everything-else" and `full_delta`
   is exactly the released long-context model. This was forced by the first run, not chosen: with
   the rest left PRETRAINED (the `attn_*` arms), the full attention delta retrieves *worse* than
   the pretrained anchor at every length (G_pre_8kv_8k_14k: 0.29 vs 0.54 exact at 1K, 0.17 vs
   0.33 at 4K) -- the attention and MLP halves of the extension are co-adapted, and applying one
   without the other is off-manifold. The `all_*` arms score MLP neurons and heads together and
   say how much of the retrieval capability sits in attention at all.
4. **Four rankings per model, on the same delta and the same loss**: MAttr's learned mask
   (`fold_learned`, ~86 Adam steps over the top-k sigmoid mask, `mode: cause` = the top-k heads get
   the extension's delta and the rest stay pretrained), IxG at the pretrained endpoint and at the
   long-context endpoint (`fold_ixg_*`, closed-form first-order attribution), and a seeded random
   ranking (`fold_random`, the chance floor).
5. **The sweep.** `eval/niah.py` scores teacher-forced exact retrieval (greedy generation of the
   digits is exactly "every answer token is the argmax", so it is forward-only and decoder-free)
   plus the answer NLL at every context length under masks keeping 0.5%-100% of the units.
   The curve reads: how many heads' worth of the extension update restore retrieval at 16K/32K?
6. **The corroboration.** `scripts/olmpool/olmpool_retrieval_heads.py` transcribes Wu et al.'s detection (attention captured by
   a registered attention interface, because the remote-code classes expose nothing through
   `output_attentions` -- the first version scored every head of those models 0)
   (argmax-attention copy hits on the needle during greedy decoding, over successful examples,
   threshold 0.1) and runs it on the same prompts at `pt`, `pt_ext` and `lc`. The mask's head
   ranking is then compared with (a) the retrieval heads of the long-context model, (b) those of
   the pretrained model, and (c) across architectures against HELMET/RULER
   (`docs/olmpool/olmpool_results.json`, transcribed from the paper's Tables), by
   `scripts/olmpool/olmpool_analysis.py`.

### What had to change in the repo

- `masks/layout.py`: the `head` unit mode (tests: `tests/test_head_units.py`).
- `trust_remote_code:` on the experiment config, threaded through every `from_pretrained`
  (four OlmPool variants ship custom norm orderings as remote code).
- `eval/niah.py`, registered as `niah`; `data.supervise_tail: false` (the digits alone are
  supervised -- the template's "\n\n<eos>" after them is a base-model novelty worth several nats
  and dominated the first run's objective); `mask.fold_params` (above).
- **Gradient checkpointing on the masked path was a no-op or a crash, and now works.** HF only
  checkpoints in `train()` mode (the loop runs `eval()` unless `train.dropout: true`, so the
  OlmPool base sets it -- every OlmPool config has dropout 0), and the non-reentrant recompute
  runs outside `functional_call`'s parameter swap, so it saw the frozen live weights and raised
  "different number of tensors saved" (33 vs 29 on SmolLM2). `MaskedDelta` now hands torch's
  checkpoint a recompute context that re-installs the composed parameters; score gradients are
  bit-identical with and without it (`tests/test_masked_checkpointing.py`). This is what makes a
  16K-token backward through an 8B model fit on one 80 GB card.

### Four conversion defects in the released checkpoints, found and worked around before any number

All are in how the HF checkpoints load under transformers 5.14, not in the weights, and all
are fixed by `scripts/olmpool/olmpool_swa_patch.py` (run by `olmpool_fetch.py`; each model dir's
`PROVENANCE.md` records what was changed):

- **Every native `Olmo3ForCausalLM` checkpoint (12 of 26) loaded at the PRETRAINING RoPE theta.**
  Their `config.json` stores theta in the flat form `{"rope_theta": 8000000, "rope_type":
  "default"}`; `Olmo3Config` fills its per-layer-type `full_attention` / `sliding_attention`
  entries from the class default (500000) when those are absent, and `Olmo3RotaryEmbedding` reads
  the per-type entries. Measured off the instantiated inverse frequencies: 500001 before the
  rewrite, 7999997 after. Every other family (Llama, Qwen3, the remote-code Olmo2-based classes)
  resolves to 8M correctly. A long-context Olmo3 number produced without the rewrite is a number
  for a model that was never trained.
- **Only the native Olmo3 SWA models actually ran sliding-window attention; the other 7 SWA
  checkpoints ran full attention on every layer.** The released remote code implements the window
  by passing `sliding_window=` to the attention kernel, which transformers 5's sdpa and eager
  kernels ignore (only flash-attention reads it); and `A_post_HQK_8kv_8k_13k_SWA_fp8`,
  `H_post_HQK_32kv_8k_11k_SWA`, `H_pre_LQK_32kv_8k_11k_SWA` and `F_pre_8kv_8k_14k_SWA` (released
  as plain `LlamaForCausalLM`) carry no `layer_types` at all. The patched
  `models/olmpool/_remote_code/modeling_olmpool.py` dispatches per-layer masks exactly as native
  Olmo3 does (`tests/test_olmpool_swa.py` pins it against hand-built banded masks); the four
  window-less configs get the paper's 3-local:1-global pattern (`[s, s, s, f] * 8`, window 4096,
  Olmo3's default) -- an ASSUMPTION about which layers were local, since the released files do
  not say. `F_pre...` is re-labelled `Olmo3PreorderNoQKForCausalLM` (the identical pre-norm,
  no-QK block; every parameter name matches) so it can carry `layer_types`. A third, smaller
  defect in the same classes: they inherit `LlamaAttention` under an Olmo2 rotary embedding,
  which emits float32 cos/sin that Olmo2's attention casts back and Llama's does not, so in bf16
  they died in sdpa (float q/k against bf16 v). The patched forward hands Llama-derived layers
  cos/sin in the model dtype, as a native Llama does.

- **A fourth: `G_post_LQK_8kv_4k_14k_SWA` ships post-norm weights (`post_feedforward_layernorm`
  and QK-norm gains, no `input_layernorm` -- the native Olmo3 layout) under the pre-norm
  `Olmo3PreorderForCausalLM` class**, so its `input_layernorm` was random and its
  `post_feedforward_layernorm` ignored: NLL 11.5 on plain text at every checkpoint. The patch
  relabels it `Olmo3ForCausalLM`. Every other checkpoint's tensor names match its class
  (`scripts/olmpool/olmpool_swa_patch.py` audits all 26).
- **What is NOT a defect: the RoPE convention of the NoQK remote classes.** Loaded as a plain
  `LlamaForCausalLM` with full attention, `G_pre_8kv_8k_14k_SWA` and `H_pre_32kv_8k_11k_SWA` have
  the same essay loss as the native Llama baseline (2.894 / 2.895 vs 2.888), and applying the HF
  converter's head permutation to their q/k doubles it (5.4-5.5, as it does for the baseline).

- **Unresolved: the remote-code SWA checkpoints retrieve poorly at EVERY length, including
  inside the window.** `G_pre_8kv_8k_14k_SWA` (pretrained, own theta) retrieves 0.12 at 1K where
  its no-SWA twin `G_pre_8kv_8k_14k` retrieves 0.88, and the same 0.12 whether it is run through
  the patched class or loaded as a plain full-attention `LlamaForCausalLM`; its released
  long-context model retrieves 0.29 / 0.08 / 0.00 at 1K / 16K / 32K. `H_pre_32kv_8k_11k_SWA` (0.46 at 1K) and
  `A_post_HQK_8kv_8k_13k_SWA_fp8` (2 of 24 at 1K) behave the same way, while the native-Olmo3 SWA
  models do not (`H_post_LQK_32kv_8k_11k_SWA`: 21 of 24 at 1K before extension). Their essay
  loss is normal and their RoPE pairing is verified, and the paper reports RULER-4K 83.2 for
  `G_pre_8kv_8k_14k_SWA` -- equal to its twin -- so this is either a conversion defect that
  language-modelling loss does not expose or a real property of these checkpoints in this
  format. Until that is settled these six models (the four listed above plus `F_pre..._SWA`
  and `H_pre_LQK..._SWA`) are excluded from the mechanistic comparison; the SWA axis is read off
  the native Olmo3 checkpoints only.

### Other caveats

- `K_post_HQK_8kv_12k` is named `post` but is `Qwen3ForCausalLM` (prenorm) and the paper's table
  lists it as prenorm; `G_post_LQK_8kv_4k_14k_SWA` was released under the pre-norm class (the
  fourth defect above). A plain-text loss at 1K is the sanity check that a checkpoint is being
  run under the block it was trained with: a wrong norm order reads as NLL ~11 at every length.
- n = 24 prompts per context length per condition (binomial SE ~0.09 at 0.5), so single-cell
  differences under ~0.2 in accuracy are not findings; the NLL column is the continuous companion.
- The attention-only arms' `full_delta` is NOT the released long-context model: it is the
  long-context attention projections over pretrained everything-else (MLP, norms -- including the
  QK-norm gains -- and embeddings). That is a meaningful quantity (does the attention part of the
  extension carry retrieval on its own?) and is read against the dense anchors. `attnqk_*` adds
  the q_norm/k_norm gains for the QK-norm architectures, and `all_*` adds the MLP neurons.

## Results (every number is teacher-forced exact retrieval, n=24 per length)

### 1. Retrieval heads are intrinsic; the extension does not move them (Wu et al. replicated)

`scripts/olmpool/olmpool_retrieval_heads.py` on the same prompts at three checkpoints. Top retrieval
heads and their scores barely change from pretraining to extension: G_pre_8kv_8k_14k's top heads
at every checkpoint and length are L14H25, L11H1, L16H26, L15H17, L7H5, L11H2; H_post_LQK's
long-range retrieval heads (16K/32K) are L15H4, L15H5, L7H26, L19H24, L11H17 -- all on its
full-attention layers (every 4th layer: 3, 7, 11, 15, 19, ...), as SWA requires. What the
extension changes is *how far* they work, not *which* they are.

Greedy-decoding retrieval, needle in a PG-essay haystack (successes of 24):

| model | ckpt | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| G_pre_8kv_8k_14k (Llama-3-style) | pt @ own theta | 20 | 16 | - | - |
| | pt @ extended theta (`pt_ext`) | 13 | 8 | 1 | - |
| | long-context (`lc`) | 22 | 18 | 13 | 12 |
| H_post_LQK_32kv_8k_11k_SWA (Olmo-3-style) | `pt_ext` | 21 | 14 | 2 | - |
| | `lc` | 24 | 24 | 24 | 23 |
| K_post_HQK_8kv_12k (Qwen-3-style) | `pt_ext` | 7 | 3 | 1 | - |
| | `lc` | 14 | 11 | 9 | 11 |
| J_pre_16kv_8k_14k (best HELMET) | `pt_ext` | 2 | 0 | 0 | - |
| | `lc` | 16 | 12 | 10 | 8 |

One more intrinsic-ness fact, from the SWA variant of the same initialisation
(`G_pre_8kv_8k_14k_SWA`, a separate 150B-token run under a different attention pattern): its
retrieval heads are G's -- L11H1, L11H2, L16H26, L7H5, L15H17 -- with its long-range ones at
16K on global layers 11 and 15. Which heads become retrieval heads is fixed by the
initialisation and shared data, not by the attention pattern trained under.

Two things already. Zero-shot theta scaling costs the no-QK-norm models their SHORT-range
retrieval (G 20 -> 13 at 1K, J collapses to 2/24), while the QK-norm Olmo-3 model keeps it
(21/24) -- QK norm makes the attention logits insensitive to the rescaled rotary frequencies.
And the paper's HELMET/RULER ranking is not this task's ranking: H (HELMET 47.5) retrieves
23-24/24 at every length, J (56.4) 8-16/24. Single-needle retrieval is one component of those
benchmarks, and the one this attribution is about.

### 2. Where the extension lives is architecture-dependent: MLP-carried vs head-carried

Two framings of the head-level mask, both on `theta_pt_ext + m . delta_attention`:

* `attn_*`: everything outside the attention projections stays PRETRAINED. Then for
  G_pre_8kv_8k_14k the FULL attention delta retrieves worse than nothing (0.29 vs 0.54 at 1K,
  0.04 vs 0.04 at 16K, 0.00 at 32K), while a top-6-heads mask by I×G (0.5% of units) lifts 32K from
  0.00 to 0.25, top-26 to 0.42, and half the heads to 0.58. Attention deltas alone are half
  helpful and half harmful; the harmful half is what co-adapted with the MLP update.
* `fold_*`: everything outside the attention projections takes its EXTENDED value. Then the
  `pretrained` anchor -- pretrained attention over extended MLP/norms/embeddings -- retrieves
  0.92 / 0.79 / 0.79 / 0.58 / 0.46 at 1K-32K against 0.92 / 0.75 / 0.88 / 0.50 / 0.50 for the
  released long-context model. **For the Llama-3-style architecture the attention update
  contributes nothing to single-needle retrieval that the rest of the update does not already
  provide.** Its retrieval heads (which needed no re-tuning) work at 32K as soon as the MLPs and
  norms are the extended ones.

The Olmo-3-style H_post_LQK_32kv_8k_11k_SWA is the opposite case in the `attn_*` framing: the
attention delta ALONE, over pretrained everything-else, takes 32K from 0.00 to 0.79 (16K: 0.08 ->
0.71), 20% of heads (205) reach 0.92 at 16K / 0.79 at 32K, and the I×G ranking now correlates
with the probe's retrieval score (Spearman 0.32 at 32K) and names the probe's own long-range
retrieval heads (L7H26, L9H19, L15H4, L15H5, L11H17) in its top 16. So under SWA + QK norm the
extension IS carried by re-tuned retrieval heads on the global layers; under the Llama block it
is carried by the MLPs and the retrieval heads are left alone.

The `fold_*` sweep on G says more: over the extended rest, keeping the top 10-20% of heads'
updates (I×G or learned) restores 16K to 0.96-1.00 and 32K to 0.71-0.88, ABOVE the full
attention update (0.50 / 0.50). Part of the released attention update actively hurts
single-needle retrieval in this model, and a sparse subset of head updates is better than all
of them -- the same overshoot the repo's finetune sweeps keep finding, now on a pretraining
checkpoint pair. The heads a mask keeps are NOT the retrieval heads (top-64 overlap 3-6 against
~2 by chance; Spearman with the retrieval score ~0), and the retrieval heads' own updates are of
ordinary size (median per-head delta norm 6.6 vs 6.0 for all heads): the extension moved them
as much as any head, to no effect on retrieval.

In the same framing H_post_LQK_32kv_8k_11k_SWA is again the opposite: pretrained attention over
its extended MLPs/norms retrieves 0.29 at 16K and 0.08 at 32K (full model: 1.00 / 0.96). Its
attention update is both sufficient (previous paragraph) and necessary.

H's `fold_*` sweep (rest extended, heads masked): 5% of heads (51) take 16K from 0.29 to 0.83,
10% to 0.96, 20% to 1.00 (32K: 0.08 -> 0.38 -> 0.75 -> 0.92), monotone with no overshoot, and
the top-64 heads by I×G contain 8 of the model's 11 long-range retrieval heads (0.7 expected by
chance). So for the Olmo-3-style block the mask localises the extension to a few dozen heads,
and those heads include the retrieval heads. They are enriched on the global layers (23 of the
top 64 sit on the 8 full-attention layers, 36% against a 25% base rate) but most are on
sliding-window layers -- layer 6 alone holds 11 of the top 64 -- whose heads cannot see the
needle from 16K away, so their updates serve something other than the long-range copy itself
(section 4 checks what their attention statistics changed).

K_post_HQK_8kv_12k (Qwen-3-style: headwise QK norm, prenorm, no SWA) is a third pattern in
the same framing: pretrained attention over the extended rest retrieves 0.29 / 0.33 at 16K /
32K, the full attention update 0.38 / 0.46 -- and the best 20-50% of head updates 0.79-0.88 at
every length. Its head updates are needed AND half of them hurt. Its kept heads are enriched
for its retrieval heads (11 of the top 64 against 2.7 by chance; H: 8 of 11 retrieval heads in
the top 64; G: chance), so the two QK-norm architectures both re-tune retrieval heads during
extension and the no-QK-norm one does not.

J_pre_16kv_8k_14k's `fold_*` sweep is G's again: 10% of head updates over the extended rest
give 0.96 / 0.96 / 0.92 / 0.83 at 1K-32K, 50% give 1.00 / 1.00 / 1.00 / 0.96, and the whole
attention update 0.67 / 0.54 / 0.42 / 0.33 -- in the paper's best model, half of the released
head updates cost a factor of two to three in long-range retrieval.

The head-level random control (H, `attn_random`, rest pretrained): a seeded random 5% / 10% /
20% of head updates retrieves 0.08 / 0.12 / 0.21 at 16K and 0.00 / 0.00 / 0.04 at 32K, against
0.50 / 0.71 / 0.92 and 0.42 / 0.58 / 0.79 for the I×G-chosen heads; only a random half of the
heads (0.42 / 0.08) begins to approach the whole update (0.71 / 0.79). The same control in the
`fold_*` framing (rest extended): random 10% of heads 0.29 / 0.08 at 16K / 32K against
0.96 / 0.75 chosen, random 20% 0.33 / 0.38 against 1.00 / 0.92. Which heads, not how many.

MAttr's learned ranking and the closed-form I×G ranking agree on all three, and the learned one
is a little sharper at the sparse end (H: 5% of heads reach 1.00 at 16K learned vs 0.83 I×G;
K: 10% reach 0.92 at 32K vs 0.79), with the same retrieval-head enrichment (H: 5 of its 11
retrieval heads in the learned top 64; K: 9 of 64 vs 2.7 by chance; G: chance). Neither
ranking tracks per-head delta magnitude (Spearman -0.05 to +0.07), so "keep the biggest head
updates" is not what either is doing.

The two QK-norm variants of the Llama block localise even more sharply in the `fold_*` framing
(rest extended, heads masked, I×G ranking): `G_post_LQK_8kv_8k_14k` goes from 0.38 to 0.54 /
0.88 / 1.00 at 32K with 0.5% / 1% / 2% of its heads -- 6, 13, 26 head units -- against 0.92 for
the whole attention update; `G_pre_LQK_8kv_8k_14k` from 0.50 to 0.62 / 0.83 / 0.96 at the same
fractions (whole update 0.79). Two dozen heads' worth of the extension is the long-range
retrieval update in these blocks, and it beats the released attention update at 32K in both.
The learned masks agree (post-norm: 0.58 / 0.79 / 1.00 at the same three fractions; prenorm:
0.67 / 0.75 / 0.79, then 1.00 at 5%). Their top-64 heads sit in layers 7-16 (post-norm) and 7-13 (prenorm), the same middle band as
the retrieval heads of every model here; the post-norm variant's top 64 contain 4 of its 25
retrieval heads (1.6 expected), a weaker enrichment than H's.

`scripts/olmpool/olmpool_factorial.py` turns this into a 2^k table per model (attention projections /
QK-norm gains / MLPs / embeddings+norms, each pretrained or extended) -- see section 3.

### 3. A 0.5% subset of the extension update out-retrieves the extension (G_pre_8kv_8k_14k)

`all_learned`: heads AND MLP neurons scored together (460,032 units; norms and embeddings left
pretrained), MAttr for 86 steps on the 12-16K needle loss. Teacher-forced retrieval by fraction
of units kept:

| ctx | pretrained | 0.5% | 1% | 5% | 20% | 50% | full attention+MLP delta | released LC model |
|---|---|---|---|---|---|---|---|---|
| 1K | 0.54 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 0.92 |
| 4K | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.75 |
| 16K | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.46 | 0.50 |
| 32K | 0.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.50 |

2,300 units -- 170 head units and 2,130 MLP neurons, the neurons concentrated in layers 9-19
(155-178 per layer) and the final layer (252) -- carry a PERFECT single-needle retriever out to
32K, where the whole update the extension actually produced retrieves 0.33-0.50. The extension
update is a compact retrieval circuit plus a much larger remainder that degrades retrieval;
everything from 0.5% to 50% of units stays at 1.00 and only the last half of the ranking pulls
it down. (The head units it picks are, again, not the probe's retrieval heads: overlap 1 of 64.)
H_post_LQK_32kv_8k_11k_SWA under the same arm: 0.5% of units (1,800) also give 1.00 at every
length -- but here the released model was already at 0.96-1.00, so there is nothing to
overshoot, and the composition of that top 0.5% is the contrast: 333 of H's 1,024 head units
(33%) against 170 of G's 1,280 (13%). The Olmo-3-style extension is head-carried, the
Llama-style one MLP-carried, on the same objective and the same mask.

K_post_HQK_8kv_12k, the weak retriever (released model 0.46 at 32K), holds the same thing: its
top 0.5% (2,600 units; 263 heads = 18% of its heads) retrieves 0.96-1.00 at every length where
its attention+MLP update in full gives 0.50 / 0.46 / 0.21 / 0.29. J_pre_16kv_8k_14k likewise: 0.5% -> 0.96 / 1.00 / 1.00 / 0.96 against 0.67 / 0.42 / 0.29 / 0.25
for its full attention+MLP update, with 189 heads (12% of its heads) in that 0.5%. So across
all four baseline architectures the extension update ALWAYS contains a ~0.5%-sparse perfect
single-needle retriever, and what differs between architectures is how much the remaining
99.5% of the update interferes with it: not at all for the Olmo-3 block (released 0.96 at 32K),
severely for the Qwen-3 and Llama blocks (0.46, 0.50, 0.33). The head share of that 0.5% orders
the same way as the factorial's attention share: H 32%, K 18%, G 13%, J 12% of all heads. On this task the architectures
do not differ in whether the extension learns retrieval; they differ in what else it learns on
top and how destructive that is.

The same arm on every model it has run on (`plots/olmpool_all_units*.pdf`), retrieval at
16K / 32K for the selected top 0.5% of units against the full attention+MLP update, and the
share of that 0.5% that is attention-head units:

| model | block | 0.5% selected | full update | heads in the 0.5% |
|---|---|---|---|---|
| H_post_LQK_32kv_8k_11k_SWA | Olmo-3 | 1.00 / 1.00 | 1.00 / 0.96 | 331 (32% of all heads) |
| G_post_LQK_8kv_8k_14k | Olmo-3, no SWA | 1.00 / 1.00 | 1.00 / 0.92 | 342 (27%) |
| G_pre_LQK_8kv_8k_14k | Llama + LQK | 1.00 / 0.92 | 0.67 / 0.67 | 249 (19%) |
| K_post_HQK_8kv_12k | Qwen-3 | 0.96 / 1.00 | 0.21 / 0.29 | 263 (18%) |
| G_pre_8kv_8k_14k | Llama | 1.00 / 0.96 | 0.46 / 0.33 | 170 (13%) |
| J_pre_16kv_8k_14k | Llama, 16 kv | 1.00 / 0.96 | 0.29 / 0.25 | 189 (12%) |
| G_pre_8kv_4k_14k | Llama, 4K | 1.00 / 0.00 | 0.08 / 0.04 | 135 (11%) |
| I_pre_32kv_8k_12k | Llama, 32 kv | 1.00 / 1.00 | 0.46 / 0.50 | 103 (10%) |
| G_pre_4kv_8k_14k | Llama, 4 kv | 1.00 / 1.00 | 0.46 / 0.38 | 121 (11%) |

The 0.5% retriever is universal (the 4K-pretrained Llama's holds to 16K and needs more of
the update at 32K), and the share of it that lives in attention heads orders by block -- QK-norm
blocks 18-32%, plain Llama blocks 10-13% -- the same ordering as the factorial's attention share,
from a mask that was free to pick either kind of unit.

Two caveats on the reading. The 0.5% subset is SELECTED with the needle loss (86 MAttr steps on
346 training rows at 12-16K; the eval needles, depths and haystacks are disjoint, and every
length from 1K to 32K generalises from a fit at 12-16K), so "contains a retriever" means the
delta's units admit one under a task-chosen selection, not that the units are labelled in the
weights; the `all_random` control (a seeded random 0.5%) is the floor for that -- and it stays AT the
pretrained floor: a random 0.5%, 1%, 2% or 5% of G's units retrieves 0.04 at 16K and 0.04-0.08
at 32K, a random 10% 0.12, and only a random half of the update (0.25 / 0.17) starts toward
the full update's 0.46 / 0.33; K the same (0.04 at every random fraction to 10%). The selected
0.5% is a structure in the update, not a property of any 0.5% of it -- and
`all_ixg_base` is the same selection in closed form -- and on G it gives the same answer
(0.5% by one gradient at the pretrained point: 0.96 / 1.00 / 0.96 at 1K / 16K / 32K), so the
sparse retriever is found by a single first-order pass, not by the optimisation. And "hurts retrieval" is about THIS task:
the remainder of the update presumably serves the extension corpus's LM loss and the other
long-context abilities HELMET scores, which is exactly why the released models and the sparse
retrievers rank differently.

This is the strongest version of the repo's recurring "a sparse mask beats the whole finetune",
on a public pretraining checkpoint pair rather than a finetune of ours -- and it says the
paper's HELMET/RULER numbers measure the released update, not the best retriever inside it.

### 4. What the kept heads' attention did under extension (`scripts/olmpool/olmpool_head_stats.py`)

Per-head sink mass (first 100 tokens), entropy, mean attended distance and needle mass, at
`pt_ext` and `lc`, on the needle prompts (8 per length, 64 sampled query rows each); then the
change from extension in the mask's top-64 heads against the rest.

| | G_pre_8kv (Llama-style) | H_post_LQK_SWA (Olmo-3-style) |
|---|---|---|
| mean sink mass @4K / 16K, `pt_ext` | 0.171 / 0.123 | 0.071 / 0.016 |
| ... `lc` | 0.222 / 0.172 | 0.069 / 0.016 |
| mean entropy @16K, `pt_ext` -> `lc` | 5.99 -> 4.89 | 6.22 -> 5.93 |
| max needle mass @16K, `pt_ext` -> `lc` | 0.170 -> 0.256 | 0.062 -> 0.171 |
| top-64 heads' entropy change vs rest @16K | -1.13 vs -1.10 | **-0.69 vs -0.26** |
| top-64 heads' distance change vs rest @16K | -62 vs -92 (learned) | **-196 vs -100** |
| top-64 heads' needle-mass change vs rest @16K | +0.0065 vs +0.0033 | **+0.0088 vs +0.0012** |
| Spearman(mask score, entropy change) @16K | ~0 | **-0.19** (I×G), -0.12 (fold) |

The other two baselines fill in the sink axis: J (Llama, 16 kv) behaves like G -- sink
0.161 -> 0.180 at 4K, entropy 5.16 -> 4.36, needle mass 0.161 -> 0.191 -- while K (headwise QK
norm, prenorm) has G-like sinks before extension (0.179 at 4K, 0.057 at 16K) that the extension
WEAKENS (0.130 / 0.037) even as its entropy drops (4.82 -> 4.17). So "QK norm = weaker sink" is
partly a property of the post-norm Olmo-3 block (H: 0.07) rather than of QK norm as such, and
the extension moves the sink in opposite directions under the two QK-norm blocks.

The paper's QK-norm observations replicate on the Olmo-3 block (weaker sinks: 0.07 vs 0.17;
lower attention on the needle at prefill: 0.06 vs 0.17 before extension). And the two architectures extend
differently at the head level. The Llama-style model's extension sharpens EVERY head by about
the same amount (-1.1 nats entropy, +0.05 sink), through the MLP/norm update, and no head's
update is special -- consistent with sections 2-3, where its head updates were unnecessary.
The QK-norm model's extension sharpens SPECIFIC heads -- exactly the ones the mask keeps --
2.5-3x more than the rest, pulling their attention toward the needle: under QK norm the logit
scale is fixed by the gains, so the only way to retrieve further is to re-tune particular
heads' q/k geometry, and that is the update the head mask localises and the retrieval-head
probe recognises.

### 5. The factorial: which part of the extension update carries retrieval (`scripts/olmpool/olmpool_factorial.py`)

Every combination of the update's parts applied over pretrained weights, under the extended
theta (A attention projections, Q QK-norm gains, M MLPs, O embeddings + norms). Teacher-forced
retrieval at 1K / 4K / 8K / 16K / 32K:

| G_pre_8kv_8k_14k (Llama-3-style) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta (before extension) | 0.88 | 0.67 | 0.50 | 0.00 | 0.00 |
| none (zero-shot theta scaling) | 0.54 | 0.33 | 0.29 | 0.04 | 0.00 |
| A (attention only) | 0.29 | 0.21 | 0.25 | 0.04 | 0.00 |
| O (embeddings + norms only) | 0.54 | 0.21 | 0.29 | 0.04 | 0.04 |
| **M (MLPs only)** | **0.88** | **0.79** | **0.71** | **0.58** | **0.50** |
| MO | 0.92 | 0.79 | 0.79 | 0.58 | 0.46 |
| AM | 0.92 | 0.75 | 0.83 | 0.46 | 0.33 |
| AMO (the released long-context model) | 0.92 | 0.75 | 0.88 | 0.54 | 0.50 |

For this architecture the long-context extension IS an MLP update, as far as single-needle
retrieval goes: the MLP part alone reproduces the released model at every length, the attention
part alone is worse than no update, and the norm/embedding part is inert. Note also the
pretrained model at its own theta: 0.88 at 1K but 0.00 at 16K with an NLL of 10.6 -- it cannot
even read a 16K prompt, which is what theta scaling fixes (0.04, NLL 2.3) and the MLP update then
turns into retrieval.

| J_pre_16kv_8k_14k (best HELMET, Llama-style, 16 kv) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta | 0.21 | 0.17 | 0.12 | 0.00 | 0.00 |
| none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **M** | **0.50** | **0.38** | **0.46** | **0.08** | **0.25** |
| MO | 0.62 | 0.46 | 0.67 | 0.33 | 0.25 |
| AMO (released) | 0.67 | 0.54 | 0.75 | 0.42 | 0.33 |

| H_post_LQK_32kv_8k_11k_SWA (Olmo-3-style) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| none | 0.88 | 0.58 | 0.33 | 0.08 | 0.00 |
| **A** | **0.92** | **0.83** | **0.88** | **0.71** | **0.79** |
| Q (QK-norm gains) | 0.92 | 0.62 | 0.38 | 0.17 | 0.00 |
| M | 1.00 | 0.92 | 0.58 | 0.29 | 0.04 |
| O | 0.79 | 0.46 | 0.25 | 0.12 | 0.00 |
| AM | 1.00 | 0.92 | 1.00 | 1.00 | 0.96 |
| AQM | 1.00 | 0.96 | 1.00 | 1.00 | 0.96 |

The same table on the other Llama-style model says the same thing (J: attention alone 0.00 at
every length, MLPs alone 0.25 of the released 0.33 at 32K), and the Olmo-3-style model says the
opposite: its MLP update alone is worth 0.04 at 32K and its attention update alone 0.79, with the
QK-norm gains and the norms/embeddings contributing nothing on their own. **The extension is
MLP-carried under the Llama block and head-carried under the QK-norm block.** (J is also the
weakest single-needle retriever of the four at every stage -- 0.21 at 1K before extension,
0.33 at 32K after -- while it is the paper's best model on HELMET; the two measures disagree
on this task.)

| K_post_HQK_8kv_12k (Qwen-3-style) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta | 0.50 | 0.58 | 0.58 | 0.00 | 0.00 |
| none | 0.29 | 0.12 | 0.04 | 0.04 | 0.04 |
| A | 0.42 | 0.38 | 0.25 | 0.21 | 0.12 |
| Q | 0.33 | 0.29 | 0.08 | 0.04 | 0.17 |
| M | 0.58 | 0.50 | 0.29 | 0.17 | 0.21 |
| AQM | 0.50 | 0.50 | 0.46 | 0.33 | 0.46 |
| AQMO (released) | 0.58 | 0.46 | 0.46 | 0.38 | 0.46 |

K is the mixed case: no part carries its retrieval alone
(attention 0.12, MLPs 0.21, gains 0.17 at 32K against 0.04 for zero-shot scaling), pairs do not
either (AM 0.29, QM 0.29), and only the three-part update reaches its modest 0.46. So head-carriage
is not a property of QK norm as such; the Olmo-3 block's other feature, sliding-window
attention -- under which only the global layers' heads can span the context and must be
re-tuned to do so -- is the candidate, and the single-feature variants off the Llama block
(`G_pre_8kv_8k_14k_SWA`, `G_pre_LQK_8kv_8k_14k`) decide it.

Single-feature variants off the Llama block (same factorial):

| G_pre_LQK_8kv_8k_14k (Llama + layerwise QK norm) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta | 0.71 | 0.75 | 0.54 | 0.00 | 0.00 |
| none | 0.46 | 0.29 | 0.00 | 0.08 | 0.04 |
| A | 0.50 | 0.33 | 0.33 | 0.25 | 0.08 |
| Q | 0.54 | 0.29 | 0.12 | 0.12 | 0.12 |
| M | 0.83 | 0.71 | 0.71 | 0.46 | 0.25 |
| AM | 0.92 | 0.79 | 0.79 | 0.67 | 0.67 |
| QMO (everything but attention) | 0.96 | 0.88 | 0.79 | 0.62 | 0.50 |
| AQMO (released) | 0.96 | 0.92 | 0.83 | 0.71 | 0.79 |

| G_pre_8kv_4k_14k (Llama pretrained at 4K) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta | 0.21 | 0.17 | 0.00 | 0.00 | 0.00 |
| none | 0.08 | 0.08 | 0.12 | 0.00 | 0.00 |
| A | 0.00 | 0.00 | 0.04 | 0.00 | 0.00 |
| M | 0.62 | 0.58 | 0.62 | 0.33 | 0.00 |
| AM | 0.46 | 0.38 | 0.33 | 0.08 | 0.04 |
| MO | 0.67 | 0.58 | 0.50 | 0.29 | 0.00 |
| AMO (released) | 0.71 | 0.54 | 0.42 | 0.08 | 0.12 |

Adding layerwise QK norm to the Llama block moves it half-way to the Olmo-3 pattern: the
attention update becomes NECESSARY (dropping it costs 39% of the extension's gain at 32K,
against 8% without QK norm) though not sufficient (8% of the gain alone; it needs the MLPs,
AM 0.67), and the released model becomes a better single-needle retriever (0.79 at 32K
against 0.50). Pretraining at 4K instead of 8K is the other way round: the released model
reaches only 0.12 at 32K, and its MLP update alone (0.33 at 16K, 0.62 at 1K) is BETTER than
MLPs plus attention (0.08 / 0.46) -- with the 16x theta jump the attention update is
destructive, and the paper's "4K pretraining hurts extension" has a weight-level reading:
the attention half of the update never learned to work at the new theta.

| G_post_LQK_8kv_8k_14k (Llama + post-norm + layerwise QK norm, no SWA) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta | 0.96 | 0.92 | 0.96 | 0.00 | 0.00 |
| none | 0.88 | 0.67 | 0.42 | 0.33 | 0.00 |
| **A** | **0.96 | 0.96 | 1.00 | 0.96 | 0.79** |
| Q | 0.88 | 0.71 | 0.58 | 0.46 | 0.00 |
| M | 1.00 | 0.96 | 0.92 | 0.92 | 0.04 |
| AM | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 |
| QMO (everything but attention) | 1.00 | 1.00 | 0.96 | 0.92 | 0.38 |
| AQMO (released) | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 |

| I_pre_32kv_8k_12k (Llama block, 32 kv = full MHA) | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| pt @ own theta | 0.29 | 0.21 | 0.50 | 0.00 | 0.00 |
| none | 0.29 | 0.25 | 0.33 | 0.00 | 0.00 |
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **M** | **1.00** | **0.96** | **0.96** | **0.83** | **0.75** |
| MO | 1.00 | 1.00 | 0.92 | 0.83 | 0.75 |
| AM | 0.83 | 0.83 | 0.75 | 0.46 | 0.50 |
| AMO (released) | 0.75 | 0.54 | 0.58 | 0.38 | 0.50 |

The MHA Llama variant is the sharpest MLP-carried case: its MLP update alone is a near-perfect
retriever to 16K (0.83) and 0.75 at 32K, and ADDING the released attention update halves that
(0.38 / 0.50). Under the Llama block the attention part of the extension is not merely
unnecessary for single-needle retrieval; it is what makes the released models worse retrievers
than the update they contain -- a weight-level form of the 0.5%-subset result in section 3.

The GQA axis does not move the pattern: on the Llama block with 4 kv heads
(`G_pre_4kv_8k_14k`) the MLP update alone gives 0.54 at 32K and the released update 0.42, with
8 kv heads 0.50 / 0.50, 16 (J) 0.25 / 0.33, 32 (I) 0.75 / 0.50 -- MLP-carried at every kv
count, and the attention update neutral-to-harmful at every kv count.

The SWA axis on the Olmo-3 block itself (native checkpoints, three `H_post_LQK_32kv` variants),
retrieval at 32K:

| variant | none | A alone | M alone | everything but A | released | attn_share | attn_needed |
|---|---|---|---|---|---|---|---|
| 8K pretrain, no SWA (`H_post_LQK_32kv_8k_11k`, HELMET 48.7) | 0.17 | 0.92 | 0.67 | 0.83 | 0.96 | 0.95 | 0.16 |
| 8K pretrain, SWA (`..._SWA`, 47.5) | 0.00 | 0.79 | 0.04 | 0.08 | 0.96 | 0.83 | 0.91 |
| 4K pretrain, no SWA (`H_post_LQK_32kv_4k_11k`, 47.4) | 0.00 | 0.92 | 0.00 | 0.00 | 1.00 | 0.92 | 1.00 |
| 8 kv, 8K, SWA (`C_post_LQK_8kv_8k_13k_SWA`, 35.2) | 0.04 | 0.29 | 0.12 | 0.12 | 0.62 | 0.43 | 0.86 |
| 8 kv, 4K, SWA, G init (`G_post_LQK_8kv_4k_14k_SWA`, 39.9; relabelled) | 0.00 | 0.17 | 0.00 | 0.00 | 0.79 | 0.21 | 1.00 |

Under post-norm + layerwise QK norm the attention update ALWAYS suffices (0.79-0.92 alone).
What sliding-window attention does is remove the *other* route: without the window, the
MLP update alone also retrieves (0.67), and everything-but-attention reaches 0.83; with the
window, the MLP route is gone (0.04) and the attention update becomes necessary. Pretraining
at 4K removes the MLP route the same way (0.00). So the paper's two "detrimental" features SWA
and 4K pretraining act, on this block, by leaving the extension a single route -- re-tuning the
global-layer retrieval heads -- where the 8K no-SWA model has two.

Two 4K-pretrained Olmo-3 checkpoints from the B initialisation behave differently from the
H-initialised one above: `B_post_LQK_32kv_4k_11k` retrieves 1.00 to 16K (both routes present:
attention alone 0.58, MLPs alone 0.96 at 16K) and 0.00 at 32K, where its plain-text loss has
risen from 2.4 (16K) through 4.6 (24K) to 5.6 (32K) -- a gradual loss of the ability to read
the prompt, not a cliff -- while its SWA twin READS 32K fine (loss 2.57) but retrieves 0.58 /
0.17 / 0.17 at 8K / 16K / 32K: under SWA + 4K pretraining the extension restored language
modelling at length but not retrieval, on every part combination (attention alone 0.17, MLPs
alone 0.04 at 32K). The 8-kv 4K-pretrain
SWA model `H_post_LQK_8kv_4k_11k_SWA` (HELMET 36.9) is at 0.08 at 32K. So 4K pretraining is the
one feature whose cost shows on this task as a hard length limit of the released model,
and only in some initialisations (`H_post_LQK_32kv_4k_11k` reaches 1.00 at 32K).

**It is the norm ordering, not sliding-window attention, that makes an extension head-carried.**
With post-sublayer norm and layerwise QK norm but NO sliding window, the attention update alone
carries 86% of the gain at 32K (0.79 of 0.92; the MLPs alone 0.04), exactly the Olmo-3
baseline's pattern; with prenorm and the same QK norm (`G_pre_LQK`) it carries 6%. And this
model retrieves 0.92 at 32K -- with the Olmo-3-style H (0.96) the best single-needle retriever
measured, against 0.50 / 0.33 for the prenorm Llama models the paper ranks above it.

Cross-architecture summary of the factorial at 32K (`attn_share` = gain from the attention
update alone / gain from the whole update; `attn_needed` = gain lost by leaving attention
pretrained):

Every usable model (`plots/olmpool_factorial_32k.pdf`; the six remote-code SWA checkpoints are
left out, see the caveats):

| model | block | HELMET-32K | released | attn alone | rest alone | attn_share | attn_needed |
|---|---|---|---|---|---|---|---|
| J_pre_16kv_8k_14k | Llama | 56.4 | 0.33 | 0.00 | 0.25 | 0.00 | 0.25 |
| I_pre_32kv_8k_12k | Llama | 54.4 | 0.50 | 0.00 | 0.75 | 0.00 | -0.50 |
| G_pre_8kv_8k_14k | Llama | 52.4 | 0.50 | 0.00 | 0.46 | 0.00 | 0.08 |
| G_pre_LQK_8kv_8k_14k | Llama + LQK | 51.4 | 0.79 | 0.08 | 0.50 | 0.06 | 0.39 |
| G_pre_8kv_4k_14k | Llama, 4K | 50.2 | 0.12 | 0.00 | 0.00 | 0.00 | 1.00 |
| G_pre_4kv_8k_14k | Llama, 4 kv | 49.1 | 0.42 | 0.00 | 0.50 | -0.25 | -0.25 |
| H_post_LQK_32kv_8k_11k | Olmo-3, no SWA | 48.7 | 0.96 | 0.92 | 0.83 | 0.95 | 0.16 |
| G_post_LQK_8kv_8k_14k | Olmo-3, no SWA | 48.5 | 0.92 | 0.79 | 0.38 | 0.86 | 0.59 |
| H_post_LQK_32kv_8k_11k_SWA | Olmo-3 | 47.5 | 0.96 | 0.79 | 0.08 | 0.83 | 0.91 |
| H_post_LQK_32kv_4k_11k | Olmo-3, no SWA, 4K | 47.4 | 1.00 | 0.92 | 0.00 | 0.92 | 1.00 |
| K_post_HQK_8kv_12k | Qwen-3 | 46.9 | 0.46 | 0.12 | 0.33 | 0.20 | 0.30 |
| H_post_LQK_32kv_8k_11k_SWA_fp8 | Olmo-3 | 46.7 | 1.00 | 0.46 | 0.08 | 0.43 | 0.96 |
| B_post_LQK_32kv_4k_11k | Olmo-3, no SWA, 4K | 42.9 | 0.00 (0.96 @16K) | 0.00 (0.58) | 0.00 (0.96) | -- | -- |
| E_post_LQK_32kv_8k_11k_SWA_fp8 | Olmo-3 | 42.8 | 0.96 | 0.67 | 0.17 | 0.70 | 0.83 |
| D_post_LQK_8kv_8k_13k_SWA_fp8 | Olmo-3, 8 kv | 42.7 | 0.50 | 0.21 | 0.08 | 0.36 | 0.91 |
| G_post_LQK_8kv_4k_14k_SWA | Olmo-3, 8 kv, 4K | 39.9 | 0.79 | 0.17 | 0.00 | 0.21 | 1.00 |
| D_post_LQK_8kv_8k_13k_SWA | Olmo-3, 8 kv | 38.8 | 0.12 | 0.04 | 0.17 | -- | -- |
| H_post_LQK_8kv_4k_11k_SWA | Olmo-3, 8 kv, 4K | 36.9 | 0.08 | 0.00 | 0.00 | -- | -- |
| C_post_LQK_8kv_8k_13k_SWA | Olmo-3, 8 kv | 35.2 | 0.62 | 0.29 | 0.12 | 0.43 | 0.86 |
| B_post_LQK_32kv_4k_11k_SWA | Olmo-3, 4K | 35.0 | 0.17 | 0.17 | 0.12 | 1.00 | 0.25 |

Read down the `block` column: every Llama-block model has `attn_share` 0 (and `rest alone` at
or above the released model), every Olmo-3-block model with a measurable gain has
`attn_needed` 0.83-1.00 and the attention update alone carrying 0.17-0.92 -- with the one
exception of the no-SWA 8K model where both routes exist. The paper's HELMET ordering runs
top to bottom; the routing runs by block; and the released single-needle score runs neither
way (0.96-1.00 for four Olmo-3-block models, 0.33-0.50 for the three top-ranked Llama ones).

### 6. Below 0.5%, and with everything scored (norm gains, embeddings, output head)

Three follow-ups on the 0.5% result, all on the saved masks or on two extra arms
(`allnorm_learned`: block tensors + every norm gain; `full_learned`: `exclude_params: null`, so
every embedding row and output-head row is a unit too). Scripts: the eval CLI with `--fracs` on
`all_learned/final.pt` (`runs/olmpool/<model>/all_learned/posthoc_small/`) and
`scripts/olmpool/olmpool_embed_ranks.py`.

**The retriever is not a step at 0.5%; it is a smooth curve that saturates at ~0.2% for most
blocks and at 0.05-0.1% for the post-norm QK-norm block.** Needle accuracy at 16K / 32K on the
all-units masks (about 460K units, so 0.1% is ~460 units; pretrained anchors 0.00-0.08):

| model | 0.01% | 0.02% | 0.05% | 0.1% | 0.2% | 0.5% | released |
|---|---|---|---|---|---|---|---|
| G_post_LQK_8kv_8k_14k | 0.42/0.00 | 0.75/0.12 | **1.00**/0.62 | 1.00/**1.00** | 1.00/1.00 | 1.00/1.00 | 1.00/0.92 |
| H_post_LQK_32kv_8k_11k_SWA | 0.17/0.12 | 0.25/0.17 | 0.58/0.54 | 0.79/0.79 | **1.00**/0.96 | 1.00/1.00 | 1.00/0.96 |
| K_post_HQK_8kv_12k | 0.04/0.17 | 0.17/0.29 | 0.33/0.54 | 0.67/0.75 | 0.83/0.96 | 0.96/1.00 | 0.21/0.29 |
| G_pre_8kv_8k_14k | 0.12/0.12 | 0.21/0.21 | 0.42/0.46 | 0.58/0.50 | 0.96/0.79 | 1.00/0.96 | 0.46/0.33 |
| G_pre_LQK_8kv_8k_14k | 0.12/0.12 | 0.17/0.21 | 0.33/0.29 | 0.46/0.54 | 0.83/0.88 | 1.00/0.92 | 0.67/0.67 |
| I_pre_32kv_8k_12k | 0.12/0.08 | 0.21/0.21 | 0.42/0.46 | 0.67/0.67 | 0.88/0.92 | 1.00/1.00 | 0.46/0.50 |
| G_pre_4kv_8k_14k | 0.12/0.33 | 0.17/0.42 | 0.46/0.54 | 0.62/0.79 | 0.83/0.96 | 1.00/1.00 | 0.46/0.38 |
| J_pre_16kv_8k_14k | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.17/0.08 | 0.83/0.71 | 1.00/0.96 | 0.29/0.25 |
| G_pre_8kv_4k_14k | 0.04/0.00 | 0.08/0.00 | 0.42/0.00 | 0.54/0.00 | 0.79/0.00 | 1.00/0.00 | 0.08/0.04 |

The post-norm + layerwise-QK-norm model needs 5-10x fewer units than any Llama-block model for
the same accuracy, which is the head-carried story again (a few dozen head units are the
mechanism there). Every model still beats its released checkpoint at 0.2%. J_pre_16kv is a
cliff (nothing below 0.1%, 0.83 at 0.2%) where the others degrade gracefully, and the
4K-pretrained model's 32K retriever is an order of magnitude less sparse than its 16K one
(0.00 at 0.5%, 0.62 at 2%, 0.96 at 10%, 1.00 at 20%; released 0.04).

**Norm gains are changed by the extension (|delta|/|theta| 0.01-0.03, against ~0.2 for the block
matrices and 0.10 / 0.16 for the embeddings / output head) and a learned mask selects them
heavily** (`allnorm_learned`, `full_learned`): 26-42% of the layer-norm gain vectors and 38-53%
of the QK-norm gain vectors sit in the top 0.5%, against a 0.5% base rate. In the post-norm
models the single best-ranked unit of the whole model is a late `post_attention_layernorm`
gain. Retrieval at 0.5% is unchanged (0.96-1.00). The caveat is that a gain vector is a
whole-tensor unit (4096 parameters), so it competes on unequal terms with a neuron or a head.

**Embedding and output-head rows are almost never selected, and the few that are, are the
task's tokens.** With 100K embedding rows and 100K output rows added (30% of all units), they
take 0.1-0.4% of the top 0.5% on all five models: 2-12 embedding rows and 3-12 output rows out
of ~3,000 selected units, MLP neurons still 85-93% of the selection, retrieval at 0.5% still
1.00. The selected output rows are digits and number tokens (`1`, `0`, `10`, `42`, `100`, `500`,
and `.\n\n`, which is the second-best unit of the whole SWA post-norm model); the selected
embedding rows are the needle template's words (` text`, ` hidden`, ` numbers`, ` mentioned`,
` special`, ` number`, `User`, `Assistant`). So the vocabulary update, though large in norm, does
not carry retrieval, and the rows it does pick are a readout of the training objective, which
is the sharpest statement of the task-specificity caveat above: a format-transfer check on a
different needle template is the experiment that would say whether the block-level selection
is any less template-bound.

**Per-parameter granularity (`weight_ixg_base`, one closed-form IxG score per scalar of the block
tensors, ~7B units; a learned per-weight mask does not fit beside the model on one card):** the
top **0.01%** of scalar weights (~680K) is a perfect retriever on the Llama-block baseline (1.00 /
1.00 at 16K / 32K) and nearly so on the post-norm SWA twin (0.92 / 1.00), 50x sparser than the
unit-level masks reach on the same models; the ranking then COLLAPSES, to at or below the
pretrained floor by 0.5% and to 0.00 through 5%, where unit-level IxG on the same scores-per-
tensor stays at 1.00 at every fraction. Unit tying is therefore the load-bearing prior, not the
scoring rule. `scripts/olmpool/olmpool_weight_diag.py` locates the selection: on the Llama block the
top 0.01% is spread thinly over key/value and MLP matrices of layers 9-16 (k/v 33% of the
selection from 4% of the pool; the most-selected tensor holds 16K weights over 595 of 1,024
rows), on the post-norm model it concentrates in layer 15's q and k projections over ~430 of
4,096 rows, i.e. a few heads, with norm gains 6% of the selection from ~0% of the pool. Table
(needle accuracy 16K / 32K):

| model, ranking | 0.01% | 0.05% | 0.1% | 0.5% | 1% | 5% | full |
|---|---|---|---|---|---|---|---|
| G_pre_8kv_8k_14k, per-weight IxG | 1.00/1.00 | 1.00/1.00 | 1.00/0.88 | 0.12/0.04 | 0.00/0.00 | 0.00/0.00 | 0.42/0.38 |
| G_pre_8kv_8k_14k, unit IxG | | | | 1.00/0.96 | 1.00/1.00 | 1.00/1.00 | 0.46/0.33 |
| H_post_LQK_32kv_8k_11k_SWA, per-weight IxG | 0.92/1.00 | 0.88/0.88 | 0.83/0.71 | 0.17/0.12 | 0.12/0.08 | 0.00/0.00 | 1.00/0.96 |
| H_post_LQK_32kv_8k_11k_SWA, unit IxG | | | | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/0.96 |

(The per-weight arm scores the norm gains too and the unit arms do not, hence the slightly
different `full` anchors.)

## Figures (`plots/`, scripts beside them, data under `plots/data/olmpool/`)

- `olmpool_all_units.pdf` -- `plot_olmpool_curves.py --arms all_learned all_ixg_base all_random`:
  retrieval vs fraction of units kept (heads + MLP neurons), four baselines x five context
  lengths. The 0.5% sparse retriever, the full-update drop at the right end, the random floor.
- `olmpool_fold_heads.pdf` -- `--arms fold_learned fold_ixg_base`: the head-only sweeps over the
  extended rest, G / H / K. Where the head update matters, and how few heads it takes.
- `olmpool_heads_H_post_LQK_32kv_8k_11k_SWA.pdf` -- `plot_olmpool_heads.py`: [layer x head] score
  ranks of the two head masks beside Wu et al.'s retrieval scores at `pt_ext` and `lc`; the
  retrieval heads on the global layers (3, 7, 11, 15, 19, 23) are the horizontal stripes.
- `olmpool_factorial_32k.pdf` -- `plot_olmpool_factorial.py`: the weight-level factorial as a
  models x part-combinations heatmap, plus the three shares against HELMET.
- `olmpool_all_units_variants.pdf` -- the same arm on the Llama variants (LQK, post-norm + LQK,
  32 kv, 4K pretraining).
- `olmpool_cross_all_units.pdf` -- `plot_olmpool_cross.py --arm all_learned --stats headshare
  overshoot32k`: the share of heads in each model's top 0.5% and the sparse mask's overshoot
  over the full update, against HELMET.
- `olmpool_factorial_16k.pdf` -- the factorial heatmap at 16K.

## Reading the paper's narrative against these results

The paper's claim is that four "minor" choices -- QK norm, post-sublayer norm, aggressive GQA,
sliding-window attention, 4K pretraining -- each cost a little long-context quality after the
same extension recipe, compound, and are invisible to short-context metrics; mechanistically it
reports weaker attention sinks and lower prefill attention on the needle under QK norm, and no
usable retrieval-head signal ("these models may be too weak"). What the attribution adds:

1. **Retrieval heads are intrinsic in every OlmPool model and the extension does not create
   them** (section 1). The paper's "no retrieval-head signal" is consistent with this: the heads
   are already there at step 34000 and their scores barely move, so a PT-vs-LC comparison of
   retrieval scores has nothing to show. The extension changes their *range*, and how it does
   that is where the architectures split.

2. **The extension is routed through different parameters under different norm blocks, and
   that -- not the retrieval heads -- is the mechanistic difference the mask finds.** Under the
   prenorm Llama block (with or without QK norm, 8 or 16 kv heads, 4K or 8K pretraining) the
   MLP update carries long-range retrieval and the attention update contributes nothing or
   hurts (at 4, 8, 16 and 32 kv heads alike); under post-sublayer norm with layerwise QK norm
   (Olmo-3's block, with or without SWA) the attention update carries it, sharpening a few
   dozen specific heads -- the retrieval heads -- onto the needle; headwise QK norm with prenorm
   (Qwen-3's block) needs all parts and retrieves poorly. On the Olmo-3 block, sliding-window
   attention and 4K pretraining -- two of the paper's four detrimental features -- act by
   removing the MLP route (MLPs alone: 0.67 without the window, 0.04 with it, 0.00 at 4K),
   leaving the extension only the head route; the released models still reach 0.96-1.00
   here, so on single-needle retrieval that is a loss of redundancy, not of ability. So the paper's two mechanistic observations are replicated (QK-norm models
   do have weaker sinks and less needle attention at prefill), but their consequence is not
   "worse retrieval": those models re-tune their retrieval heads during extension, the Llama
   block re-tunes its MLPs instead, and on single-needle retrieval the post-norm + QK-norm
   models end up BETTER (0.92-0.96 at 32K against 0.33-0.50 for the paper's top-ranked
   prenorm models).

3. **Every extension update contains a ~0.5%-sparse perfect single-needle retriever; the
   released models differ in how much the other 99.5% interferes** (section 3). The
   architectures therefore do not differ in whether the extension learns retrieval. They differ
   in what else the 10B tokens taught and how that interacts -- which is a statement about the
   extension recipe's other objectives as much as about the block, and a candidate for why
   short-context metrics cannot predict the outcome.

   Across the nine models with the all-units mask, the share of heads in the top 0.5% falls with
   HELMET (R² 0.48, `plots/olmpool_cross_all_units.pdf`): the paper's better-ranked models are
   the ones that extend through their MLPs and leave their retrieval heads alone, and the gap
   between their sparse retriever and their released model is the larger one.

4. **On this task the paper's ranking does not hold, and that is worth knowing before reading
   any of the above as a mechanism for HELMET.** HELMET/RULER at 32K rank J > G > G_pre_LQK >
   G_post_LQK > H; teacher-forced single-needle retrieval at 32K ranks H (0.96) > G_post_LQK
   (0.92) > G_pre_LQK (0.79) > G (0.50) > J (0.33). Single-needle retrieval is one sub-task of
   RULER and a small part of HELMET, and the two conversion defects found here (the 12 native
   Olmo3 checkpoints loading at the pretraining theta; the 7 SWA models running full attention)
   would each depress exactly the post-norm/SWA models if any published number was produced
   through this transformers path -- which the paper does not say. The mechanistic contrast in
   point 2 is robust to the ranking question either way; the claim that it *explains* HELMET
   is not something these measurements support.
