# Weights modified by each refusal edit, relative to Instruct (measured 2026-09-12)

Measured on tilde with `scripts/refusal/l0_baselines.py` (`full` and `mask` subcommands).
L0 = number of weight entries whose stored bf16 value differs from the Instruct checkpoint's
(`meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, HF cache snapshots). For
the LoRA arms the adapter was merged in fp32 (rslora scale alpha/sqrt(r)) and the sum cast to bf16
before comparing, i.e. the L0 of the merged artifact.

## Full-weight edits, bf16 L0 vs Instruct

| edit | 1B: weights changed | 8B: weights changed | per-family (1B) |
|---|---|---|---|
| Abliteration (rank-1 projection on embed_tokens + every o_proj/down_proj) | 481,077,258 / 1,235,814,400 = 38.93% | 2,218,762,917 / 8,030,261,248 = 27.63% | down_proj 82%, embed 79%, o_proj 82% (8B: 76 / 73 / 78) |
| GRP-Oblit, full-parameter, advb lr1e-6 KL 0.01 (`refusal_grpoblit_kl001`) | 1,235,632,730 = 99.99% | not run full-parameter at 8B | every family 100% |
| GRP-Oblit, full-parameter, sr prompts (`refusal_grpoblit_srprompts`) | 1,235,708,897 = 99.99% | | |
| GRP-Oblit, full-parameter, advb 1e-5 x300 (`refusal_grpoblit_long`) | 1,235,806,667 = 100.00% | | |
| GRP-Oblit, full-parameter, sr 1e-5 x300 (`refusal_grpoblit_srprompts_long`) | 1,235,806,557 = 100.00% | | |
| GRPO LoRA r32 all-linear, no KL (`refusal_grpo_weights_lora`) | 610,487,662 = 49.40% | | targeted matrices = 79% of params |
| GRPO LoRA r32, KL 0.01 (`refusal_grpo_weights_kl001`) | 604,643,282 = 48.93% | | |
| GRP-Oblit 8B (LoRA r32 all-linear, `refusal_grpoblit_8b`) | | 4,062,090,566 = 50.58% | targeted matrices = 87% of params |

Two readings of L0 for abliteration and LoRA, and a table should say which: their SUPPORT is every
entry of the matrices they touch (abliteration 48% of the 1B model = 598M entries, 37% at 8B; LoRA
over all linear layers 79% at 1B, 87% at 8B; full-parameter GRP-Oblit 100%), but ~20% of those
entries move by less than half a bf16 ulp and round back to the original value -- that is the gap
between 48% and 39%, or 79% and 49%. The bf16 number is what a diff of the released weights shows;
the support is what the method defines.

Degrees of freedom is a third axis (abliteration: one direction, 2,048 / 4,096 numbers; LoRA r32:
~22.5M free numbers at 1B; GRP-Oblit full: all; MAttr: 603,425 / 1,960,513 scores selecting base
entries to copy). Label a column "weights modified", not "parameters trained".

## MAttr masks: parameters under the top-k units (exact, from each run's final.pt layout)

Every nonresid unit is one residual-dimension vector (2,048 params at 1B, 4,096 at 8B; the norm
gains are one whole-tensor unit of the same size), so unit fraction == parameter fraction. Identical
across all five runs measured (logk_v2, uniform_vllm_native, logk_vllm_native, 8b_logk,
8b_logk_vllm_native).

| fraction | 1B units -> params | 8B units -> params |
|---|---|---|
| 0.1% | 603 -> 1,234,944 (0.10%) | 1,961 -> 8,032,256 (0.10%) |
| 0.5% | 3,017 -> 6,178,816 (0.50%) | 9,803 -> 40,153,088 (0.50%) |
| 1% | 6,034 -> 12,357,632 (1.00%) | 19,605 -> 80,302,080 (1.00%) |
| 2% | 12,069 -> 24,715,264 (2.00%) | 39,210 -> 160,604,160 (2.00%) |
| 5% | 30,171 -> 61,790,208 (5.00%) | 98,026 -> 401,514,496 (5.00%) |
| model total | 603,425 units, 1,235,814,400 params | 1,960,513 units, 8,030,261,248 params |

Ratios: at 1B the 1% mask edits 39x fewer weights than abliteration, 49x fewer than the LoRA GRPO
arms, 100x fewer than full-parameter GRP-Oblit; at 8B the 0.1% mask edits 276x fewer than
abliteration and the 1% mask 28x fewer.
