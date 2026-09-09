# Replicating the "Toy Model of Interference Weights" filtering figures

Source: Olah, Turner & Conerly, *A Toy Model of Interference Weights*, Transformer Circuits,
2025-07-29 — specifically § *What Should We Do about Interference Weights?* (anchor
`#filtering-interference-weights`). Code: `scripts/interference/interference_toy.py`,
`plots/plot_interference_filtering.py`, `plots/plot_interference_toy_check.py`.

## What the two target figures are

Read off the published PNGs, not the prose (the prose names four heuristics and does not
define them):

1. **"Precision-Recall Curve for Various Heuristics"** — series `twera`, `era`, `weight`,
   `freq`, `ideal`.
2. **"Precision vs Loss Gain Pareto Frontier for Various Heuristics"** — same five, x =
   loss gain on [0, 7], annotated *Oracle* (6.7, 1.0), *"Same loss as original"* (~5.25),
   *"Better than original model!"* (~5.7), *"Modest performance degradation for 80%
   precision!"* (~4.55, 0.80).

Ground truth is the note's **Defn (2)**: `dL(U_ij) = L(U with U_ij zeroed) - L(U)`, real iff
`dL > eps = 1e-4`. Loss gain is the plain sum of `dL` over the kept set.

`era` / `twera` are **not defined in the note**. They are from Ameisen et al., *Circuit
Tracing* (§ Global Weights), which is where the labels come from:

    V_ij^ERA   = E[1(a_j > 0) a_i] V_ij
    V_ij^TWERA = (E[a_j a_i] / E[a_j]) V_ij

with the multiplied activation the SOURCE and the gated one the TARGET. In this toy the
attribution of one weight on one example is `U_ij x_j`, so `j` is the source and `i` the
target, giving `E[1(y'_i>0) x_j] U_ij` and `E[y'_i x_j]/E[y'_i] * U_ij`.

## The literal config replicates the setup but NOT the difficulty

Config as stated: 128 features, 16 residual dims, 8 blocks, 0.1 within-block density, A's
entries ~U[0,1], v = -0.1, input feature density 0.3.

What comes out right: **base rate 1.26%** (8 x 16x16 x 0.1 = 207 of 16384, and the published
curves all terminate just above 0); **across-seed correlation 0.978 for real weights against
0.160 for interference**, which is the note's "the interference weights are independent, but
the real weights are all significantly positive"; the *Unlearned* band; `ideal` perfect by
construction; `freq` pinned at the base rate; and the ordering ERA > virtual weight >> freq.

What does not: the note's own **criterion (1), "real weights and interference weights strongly
overlap"**. Measured interference std is 0.088 against the note's ~0.25 (read off its scatter
and histogram), and there is almost no shrinkage where the note's learned-vs-ideal figure has
a large annotated *Shrinkage* region. So `|U_ij|` alone is already a near-perfect classifier —
precision holds at 1.0 out to recall 0.4 — and **`era` is a perfect oracle** where the note has
it decaying to ~0.55.

**This is not an untuned knob.** Searched, all with the stated config otherwise fixed:

| axis | range | outcome |
|---|---|---|
| training steps | 300 – 120,000 | converged by 3k; nothing moves after |
| input feature density | 0.02 – 0.5 | `era` = 1.00 throughout |
| block density | 0.1 / 0.25 / 0.5 | 0.5 degrades `era` to 0.82 but puts the base rate at 5.6% |
| init scale | x1 – x4 | x2 gets interference to 0.198 and `weight` to 0.67, but `era` stays 1.00; x2.5+ is degenerate |
| bias | learned / fixed at v | null |
| n_residual | 16 – 4 | shrinkage rises 0.72 -> 0.28 but interference *falls* 0.088 -> 0.043; the model drops connections rather than smearing them |

`era` is 1.00 in **every non-degenerate cell**. The mechanism: with 0.1 within-block density an
output row of A has ~1.6 nonzeros, so `y'_i` is driven by essentially one source and the
target-coactivation signal is near-deterministic — which makes ERA an oracle by construction.

The note is also internally ambiguous here. One paragraph gives two different numbers for the
same knob: the variant that achieves the overlap property is *"each block is itself sparse
(probability 0.5)"*, while the figure's config is *"0.1 weight density within those blocks"*.
There is no released code, and no seed / lr / steps / init / batch size is stated.

## `--down random`: the frozen-projection variant

Freezing `W_down` as a fixed random projection with unit-norm columns removes the freedom the
model uses to cancel interference: each source feature gets a fixed direction, so an output's
`n_res` free parameters must meet ~104 live constraints and the leftover is an unavoidable
least-squares residual. This **does** recover the missing phenomenology — shrinkage appears
(median `U/A` drops to 0.14–0.30) and the real weights land inside the interference band.

It does not recover the published *ordering*. Sweeping `n_res` under the frozen projection
(steps 3000, block density 0.1, everything else as stated):

| n_res | base | `weight` P@R.08 / .2 | `era` P@R.2 / .4 | `twera` P@R.2 / .4 |
|---|---|---|---|---|
| published | 1.2% | **0.68 / 0.44** | **0.87 / 0.55** | **0.81 / 0.47** |
| 16 | 1.35% | 0.05 / 0.05 | **0.88 / 0.43** | 0.04 / 0.04 |
| 20 | 1.23% | 0.07 / 0.08 | 1.00 / 0.77 | 0.06 / 0.06 |
| 22 | 0.99% | 0.36 / 0.19 | 1.00 / 0.92 | 0.10 / 0.09 |
| 26 | 0.90% | 0.52 / 0.42 | 1.00 / 1.00 | 0.16 / 0.14 |
| 28 | 0.92% | **0.62 / 0.57** | 1.00 / 1.00 | 0.20 / 0.17 |
| 32 | 0.93% | **0.65 / 0.63** | 1.00 / 1.00 | 0.30 / 0.23 |

**`era` and `weight` cross over and are never both intermediate**: at `n_res` 16 `era` matches
the published curve almost exactly while `weight` is destroyed; by 28–32 `weight` matches while
`era` is an oracle. So the whole published three-way ordering (era > twera > weight, all
intermediate) is not reachable in this family.

`twera` is the worst method at every operating point, where the note has it second. **That is
not an implementation bug** — the ratio `E[y'_i x_j]/E[y'_i]` is a `y`-weighted mean of `x_j`
and is therefore mathematically confined to `[0, max x_j] = [0, 1]`; measured max is 0.606, so
nothing is blowing up numerically. What happens is that ~40% of this toy's output rows almost
never fire (24 rows of A are all-zero, and more are effectively so), and dividing by a tiny
`E[y'_i]` promotes whichever source drove those rare activations: **71 of the top-100 `twera`
weights sit in rows with `E[y'_i] < 1e-3`**. That is exactly the weakness *Circuit Tracing*
flags — "TWERA heavily relies on the coactivation statistics" and "these weights cannot be 0 as
this would also send TWERA to 0". Flooring the denominator at 1e-3 sends `twera` straight to
1.00/1.00, so there is no intermediate setting of that guard either: it is oracle or junk.

## Which operating point to use

- `--down learned` (default) is the note's model as stated, and is what to cite for the setup,
  the validation figures and the base rate. As a *benchmark* it is weak: `weight` and `era` both
  solve it, so nothing can beat them.
- `--down random --n-res 16` is the most discriminating: `era` tracks the published curve
  (0.88 / 0.43), `weight` is at 0.05, so there is real headroom for a learned ranking.
- `--down random --n-res 32` is the closest match to the published `weight` curve.

Any figure must say which was used; the resolved settings are recorded in
`plots/data/interference_toy/curves.json` under `meta`.

## The simple (tied) model is NOT a usable fallback — it is strictly easier

`--simple` implements the note's first model: tied `W` (so `U = W^T W`), circuit = identity,
`n_features=100`, `n_residual=20`, `feature_density=0.02`, undertrained. It has one genuine
advantage — the ideal weights are the identity, so the real weights are exactly the 100
diagonal entries and the **base rate is 1/n_feat with no estimation in it**.

But as a filtering benchmark it is degenerate: every heuristic scores **P@R = 1.00 at every
recall, including `freq`**. The diagonal entries are `||w_i||^2 ~ 1` while the off-diagonal
interference is `<w_i, w_j> ~ 0.2`, so any ranking that notices magnitude — or even just
notices that feature `i` coactivates with itself — separates them perfectly. This is the note's
own reason for building the sophisticated model in the first place: it shows the base64 feature
from *Towards Monosemanticity* has real/interference overlap and says "we'd like an example
that demonstrates why the problem is hard". Kept in the CLI because it reproduces the note's
first histogram and scatter, not because anything should be measured on it.

## Results: this repo's three methods against the note's heuristics

`scripts/interference/interference_attrib.py` scores the same 16384 virtual weights against the same `dL`
ground truth. Two configs are on disk, tagged, and both are drawn by
`plots/plot_interference_filtering.py --tag {hard,lit}`:

- **`lit`** — `--tag lit --steps 10000` — the note's config as stated. n_real 134 (0.82%).
- **`hard`** — `--tag hard --down random --n-res 16 --steps 3000` — frozen projection.
  n_real 201, i.e. **base rate 1.227% against the published ~1.2%**.

Precision at fixed recall on `hard`, all eight series against one `dL`:

| method | P@R.2 | P@R.4 | P@R.8 | spearman vs dL |
|---|---|---|---|---|
| **MAttr (Adam)** | **1.00** | 0.99 | 0.05 | +0.558 |
| **MAttr (SGD)** | **1.00** | **1.00** | 0.09 | +0.442 |
| **Stepless IG** | **1.00** | **1.00** | **0.19** | +0.542 |
| ERA | 0.88 | 0.43 | 0.02 | — |
| virtual weight | 0.05 | 0.05 | 0.02 | — |
| TWERA | 0.04 | 0.04 | 0.02 | — |
| coactivation freq | 0.02 | 0.02 | 0.02 | — |

**All three beat every heuristic in the note by a wide margin**, and the margin is largest
exactly where the note says the problem is hard: at recall 0.4 the best heuristic (ERA) is at
0.43 while all three of these are at ~1.00. On loss gain the picture is the same — keeping all
16384 weights gives 0.641, the oracle gives 0.975, and the three methods reach 0.949 (SGD),
0.962 (stepless IG) and 1.007 (Adam), i.e. every one of them lands deep inside the note's
"better than the original model" region and Adam's ordering picks up slightly more than the
`dL > eps` oracle does by taking some positive-but-sub-threshold weights before the negative
ones.

On `lit` the task is too easy to rank methods: everything sits at 1.00 out to recall 0.75 and
only `P@R.8` separates them (Adam 0.95, SGD 0.89, stepless IG 0.81, ERA 0.72). Quote `hard`.

**One caveat on `hard` that `lit` does not have.** With `W_down` frozen, the two seeds in
`--check` get *different* random projections, so the across-seed correlation of real weights is
0.451 rather than `lit`'s 0.978. The note's two-model scatter claim ("the real weights are all
significantly positive") is therefore only weakly reproduced on `hard`; it is cleanly
reproduced on `lit`. If that scatter is the thing being argued, use `lit`; if the filtering
comparison is, use `hard`.

## Budget sweep: more steps do NOT help, and they actively hurt MAttr+Adam

`scripts/interference/interference_budget.py` (9 runs: 3 methods x 3 seeds, one snapshotted trajectory each,
`--collect` for the table; drawn by `plots/plot_interference_budget.py`).

**First, a correction to the table above.** A MAttr step and a stepless-IG alpha draw are each
one forward+backward of the toy model, so the shared budget axis is that count — and the run
that produced the results table gave MAttr **3000 steps against stepless IG's 256 draws, 11.7x
more**. The claim that all three beat every heuristic is untouched (they all saturate long
before 256). The claim about the ordering *among* the three was not a matched-budget claim and
is superseded by this section.

Mean over 3 seeds, spread = max-min:

| metric | method | 256 | 1000 | 3000 | 10000 | 30000 |
|---|---|---|---|---|---|---|
| P@R.2 | all three | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P@R.4 | stepless IG / SGD | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P@R.4 | Adam | 1.00 | 1.00 | 0.99 | 0.96±.06 | 0.96±.07 |
| P@R.8 | stepless IG | **0.20±.00** | **0.20±.00** | **0.19±.00** | **0.19±.00** | **0.19±.00** |
| P@R.8 | SGD | 0.08 | 0.08 | 0.09 | 0.13 | 0.11 |
| P@R.8 | Adam | 0.09±.05 | 0.06 | 0.05 | 0.04 | 0.04 |
| rho | stepless IG | 0.55±.01 | 0.55±.00 | 0.55±.00 | 0.55±.00 | 0.55±.00 |
| rho | SGD | 0.39 | 0.41 | 0.44 | 0.48 | **0.53** |
| rho | Adam | 0.50±.11 | **0.58±.02** | 0.57 | 0.54 | 0.51±.04 |

Four readings:

- **Everything that matters is decided by 256 passes.** All three hit P@R.2 = 1.00 there and
  never leave it. The headline result — every method far above every heuristic in the note — is
  a *cheap* result, which strengthens rather than weakens it.
- **More steps make MAttr+Adam WORSE, monotonically, on both tail metrics.** `P@R.8` falls
  0.095 -> 0.037 and `rho` peaks at 1000 (0.58) then decays to 0.51. The seed spread is ±0.11 at
  256 but ±0.02-0.04 from 1000 on, so the decline sits outside it. **Adam's best budget here is
  ~1000**, and running it longer is not a free win the way a training loop usually is.
- **MAttr+SGD converges monotonically upward toward stepless IG** — `rho` 0.39 -> 0.53 with a
  ±0.01 spread at every point, crossing Adam by 30000. That is exactly the prediction in
  `scripts/interference/toy_sgd_vs_ig.py`'s docstring: zero-init SGD under a soft top-k accumulates a PATH
  INTEGRAL of `g.delta`, so given enough draws it should approach plain activation-path IG up to
  a rank-irrelevant scale. Seeing it converge to IG's 0.55 from below, on a task neither script
  was written for, is the cleanest confirmation of that derivation anywhere in the repo.
- **Stepless IG is flat and essentially seed-free** (±0.00 on both metrics at every budget) and
  **wins the tail outright** — `P@R.8` 0.19-0.20 against SGD's 0.08-0.13 and Adam's 0.04-0.09.
  That gap is 5-10x the seed spread, so unlike the first pass's ordering, **this one is
  quotable**: on this task stepless IG is the best of the three, at any budget, for the cheapest
  compute.

Wall clock is not matched and cannot be: MAttr is ~7x slower per step (4.3s vs 0.6s per 100 at
batch 2048) because `sigmoid_topk`'s 50-iteration bisection over 16384 units dominates a
128-feature model. That is a cost of the mask primitive, not of the objective, and it would not
scale this way on a real model — so it is reported but not used as the axis.

## Why MAttr+Adam degrades with more steps: diffusion in score space

`scripts/interference/interference_why_adam.py` (4 runs: Adam at lr {0.05, 0.01, 0.002} and SGD at 1.0, each
to 10000 steps with score snapshots). The lr sweep is the discriminator.

**The mechanism is Adam's per-parameter normalisation.** Adam divides each step by `sqrt(v)`, so
a unit whose gradient is pure NOISE takes the same size step as one carrying signal. On this task
~16,200 of 16,384 units are interference, i.e. almost every score is noise-driven at full step
size — and those scores random-walk. Measured spread of the interference units' scores:

| | 256 | 1000 | 3000 | 10000 |
|---|---|---|---|---|
| Adam lr 0.05 | 0.43 | 0.91 | 1.53 | **2.79** |
| Adam lr 0.01 | 0.32 | 0.37 | 0.74 | 1.51 |
| Adam lr 0.002 | 0.17 | 0.29 | 0.32 | 0.67 |
| SGD lr 1.0 | 0.008 | 0.026 | 0.064 | **0.155** |

The growth is **diffusive to within 3%**: at lr 0.05 the successive ratios are 2.12 / 1.67 / 1.83
against `sqrt(T)`'s 1.98 / 1.73 / 1.83. SGD's steps are proportional to the real gradient, so its
interference scores barely move — **18x smaller spread**, mean pinned at -0.03 while its
real-score mean climbs to +2.63.

That predicts the observed SHAPE of the failure and not just its direction. The head is safe
(top ~40 weights have signal far above any noise floor, so `P@R.2` never leaves 1.00); the tail is
decided by small true `dL` against a growing noise floor. Separation in units of each method's own
interference spread: **Adam 4.2 sigma, SGD 17 sigma**.

**The objective-mismatch hypothesis is refuted, with the wrong sign both ways.** If more steps
merely tuned the mask to a log-uniform-k objective that is not `dL`, the loss would keep falling
as rho fell. Adam's loss is FLAT across a 10x budget range (3.80-3.87) while rho decays
0.58 -> 0.53; SGD's loss FALLS (4.58 -> 3.97) while its rho RISES. And lowering Adam's lr
postpones the decline proportionally, which is what a noise mechanism predicts and a converged
objective mismatch does not.

**Adam's scores leave the near-zero regime the derivation assumes.** `|s|max` goes ~0.5 -> 34.6
over the sweep. `scripts/interference/toy_sgd_vs_ig.py`'s docstring raises exactly this concern — scores
escaping the regime where the soft-top-k analysis holds — and records that it was checked on the
fr2de/Qwen-14B run and found FALSE there. On this task it is true, so that script's open question
has a positive instance now. **Hold the gradient-starvation half of it loosely**: the diagnostic
here is `|s|/T > 4`, i.e. distance from ZERO, whereas the gate slope depends on distance from the
per-step threshold `tau(k)`, which moves with the log-uniform k. Score growth is the precondition
for saturation; the resulting starvation is not cleanly measured.

**Consequence, and it qualifies the budget section above: stepless IG is not the ceiling.**
Adam at **lr 0.002 stopped at T=3000 reaches `P@R.8` = 0.261**, the best tail number in any run
here and above stepless IG's 0.19. So "stepless IG wins the tail" holds at the DEFAULT lr 0.05
and is not a statement about the method. n=1 seed on that cell — replicate before quoting it.

## Why MAttr+Adam BEATS the model it was fitted to, and which interference weights are worth keeping

`scripts/interference/interference_offcircuit.py --tag hard` (~4 min, CPU; `--seeds` refits for replication).
Everything below is a measured line of its output.

**The fact.** Read on real masked forwards rather than the note's additive proxy, the best
filtered model MAttr+Adam can give you is better than the trained model AND better than the true
circuit:

| model | true loss |
|---|---|
| full `U` (the trained model) | 3.917 |
| the true circuit `A` alone (207 weights) | 3.663 |
| oracle `dL` ranking at its own best k (256) | 3.633 |
| stepless IG at its own best k (136) | 3.629 |
| MAttr+SGD at its own best k (176) | 3.589 |
| **MAttr+Adam at its own best k (1400)** | **3.265** |

1254 of Adam's 1400 kept weights are OFF the circuit, and the oracle calls 99% of them worthless
(median `dL` = −3.5e−06, i.e. slightly *harmful*). A second eval seed moves the number by 0.009,
so this is not sample fitting. **Adam has the WORST tail precision of the three methods against
`dL` (`P@R.8` 0.05) and the BEST true loss.** On this task those two axes disagree in ORDER, not
merely in noise — which is the sharpest available argument for `interference_true_loss.py`
existing at all.

**Why beating the original model is not paradoxical: the mask leaves the model's hypothesis
class.** `W_down` is frozen at 16 dims, so every `U` the training could reach has rank ≤ 16 and
2176 free numbers. A top-k mask has 16384 binary degrees of freedom and `U * mask` is rank **73**
at Adam's optimum (the circuit itself is rank 92). Filtering is not subset selection inside the
trained family — it exits it, and an exit is allowed to be an improvement. No hidden-circuitry
story is needed, and none is true here.

**What the extra degrees of freedom actually buy: a per-row bias correction the mask cannot write
down directly.** `b` is frozen at the value co-adapted to the FULL interference population, and
under the frozen projection the fit to `A` is heavily shrunk (median `U/A` = 0.11), so the
circuit-only model systematically under-predicts — mean gate-open residual −0.073. What it wants
is a per-row constant: `corr(-residual, optimal bias shift)` = **+0.97**. The mask cannot touch
`b`, but a set of off-circuit weights on row `i` supplies `M_i = E[x] · Σ_j U_ij`, which IS a
constant offset plus noise. Three numbers pin it:

- `corr(offset Adam supplies, optimal bias shift)` = **+0.67**.
- Replacing those 1254 weights by their **mean effect alone** — deleting them and adding `M_i` to
  `b` — gives **3.121** against Adam's own 3.265. The offset is not part of the story, it is the
  whole story, and the weights are a noisy implementation that costs 0.14 in injected variance.
- **How much a free bias is worth, per model, is the decisive line.** Re-fitting `b` (gradient,
  all else frozen; one 32,768-example eval set so the three rows are comparable):

  | model | frozen `b` | `b` re-fitted | what the bias buys |
  |---|---|---|---|
  | full `U` | 3.917 | 3.909 | 0.009 — `b` was trained here, nothing left |
  | circuit only | 3.663 | 3.185 | **0.478 — masking broke `b`'s co-adaptation** |
  | Adam's top-1400 | 3.265 | 3.262 | 0.003 — **already fixed, in weights** |

  A free bias is worth half a loss unit to the circuit-only model and nothing at all to Adam's.
  That is the mechanism stated as a measurement: what Adam's off-circuit weights are for is
  exactly the thing a re-fitted bias would otherwise supply.
- One caveat on that middle row: gradient-refitting `b` gets **stuck at 3.185**, worse than the
  3.121 that Adam's own offset reaches, because a row whose gate is shut on every example has no
  gradient to escape on. So read 3.185 as a floor on what a bias is worth, not as the optimum.

**The selection pattern, and it is none of the note's heuristics.** What makes an off-circuit
weight worth keeping:

- **Sign, aligned with the row's residual.** 82% of Adam's kept off-circuit weights are positive
  against 47% of the population, and the rows that gain are the under-predicting ones. Sign is
  necessary, which is why `--abs` cannot find these at all.
- **Magnitude, SMALL — and this is where the note's own `weight` heuristic goes exactly
  backwards.** Kept median `|U|` 0.0235 against 0.0347 in the population. It is derivable rather
  than a preference: for a target offset `M` on a row, `n` weights of size `u` inject variance
  ∝ `M·u`, so at a fixed offset smaller is strictly better. Measured on the row with the largest
  offset — the same offset assembled from 36 small weights vs 8 large ones gives row loss **0.041
  vs 0.074** (circuit-only 0.076), so the large-weight version captures almost none of the gain.
  `weight` ranks by signed `U` descending, i.e. it takes the biggest positive interference weights
  first: right sign, worst possible magnitude, and at k=1400 it is the worst ranking measured
  (**7.64**, against Adam's 3.265 and 3.663 for keeping no interference at all).
- **NOT locality.** In-block enrichment is 10.1% against an 11.1% base rate. The useful
  interference is a diffuse population, not the circuit's own block leaking.
- **A set-level budget, which no per-weight score has.** Keeping every *individually* helpful
  off-circuit weight — all 3685 whose own addition lowers the loss — gives **L = 7.08**, twice as
  bad as keeping none. The offsets add, so a set that is right one weight at a time overshoots
  catastrophically as a group.

**Why the oracle cannot see them, structurally.** `dL` ablates one weight with all others
present, and at the full model every row's offset is already balanced by the whole interference
population — so no single member of it is worth anything. Their value exists ONLY in the sparse
model, conditional on the rest of the set being gone. Stepless IG has the same blind spot from
the other side: its path scales all weights together, so it never visits a state where a
coalition of small offsets is the marginal thing. Forced to Adam's k=1400, IG gives 5.17, SGD
5.19 and the oracle `dL` 4.12, all far worse than keeping nothing extra — the ranking, not the
budget, is what separates them.

**It replicates.** Refitting at seeds 1 and 2 (`--seeds 1 2`) gives Adam's optimum at k = 1384 /
1360 with true loss **3.281 / 3.289** against seed 0's 3.265, keeping 1238 / 1213 off-circuit
weights that are 82% / 84% positive at median `|U|` 0.023 — the composition, not just the
headline, is stable. SGD's optimum replicates just as tightly on the other side: k = 160 / 152,
loss 3.578 / 3.576, and only 44-52 off-circuit weights, 100% positive at median `|U|` 0.087-0.092.
The two optimizers are not noisy versions of one strategy. SGD takes a handful of large aligned
weights and stops; Adam takes a thousand small ones.

**It is not an artifact of the frozen projection.** `--tag lit --fit-only` runs the same
question on the note's literal config, where `W_down` is learned and the model was free to cancel
interference. The effect is there too, and larger in relative terms:

| | `hard` (frozen projection) | `lit` (the note's config) |
|---|---|---|
| L(full `U`) | 3.917 | 1.760 |
| L(circuit `A`) | 3.663 | 1.376 |
| L(MAttr+Adam at its best k) | 3.265 (k = 1400) | **1.301** (k = 1832) |
| a free bias is worth, to the full model | 0.009 | 0.000 |
| a free bias is worth, to the circuit-only model | 0.478 | 0.146 |

The bias diagnostic transfers in direction and scales with how much interference there is to
break: `lit`'s model cancelled most of its own, so masking it breaks less. What changes is the
sub-pattern — `lit`'s Adam keeps 1678 off-circuit weights at median `|U|` **0.0005**, i.e. the
"many small" logic taken to its limit, at only 60% positive. Do not carry `hard`'s 82%-positive
figure across; carry the shape, which is that the useful off-circuit set is large, diffuse and
individually negligible. `lit`'s optimum is also broad — a separate fit put it at k = 3104 for
the same loss to four digits — so quote the loss, not the k, on that config. The oracle `dL` on
`lit` bottoms out at 1.371 with 208 weights, i.e. it does the conventional thing and stops, and
SGD keeps 18 large off-circuit weights for 1.363: the three-way split of strategies is the same
as on `hard`.

**Two reductions of Adam's advantage that FAIL, recorded so they are not re-attempted.**

- *"Adam ranks by consistency (mean/rms) where SGD ranks by summed gradient, and consistency is
  scale-free, so it surfaces small aligned weights."* Tested on ONE shared gradient stream:
  ranking by mean/rms picks LARGER off-circuit weights than ranking by the sum (median `|U|`
  0.036 vs 0.031) — the wrong sign. The magnitude tilt is real (Adam's `rho(score, U)` over
  positive off-circuit weights is −0.24 against SGD's +0.01) but it is a property of Adam's
  TRAJECTORY, not of a reweighting of a fixed signal.
- *"It is greedy value-per-variance."* The first-order greedy that keeps the 1400 best
  `value/variance` off-circuit weights reaches only **3.538**, and is non-monotone past that
  (3.76 at 2000), because it cannot stop adding once a row's offset is met.

So the mechanism of the *gain* is settled — a coalition of small, sign-aligned, diffuse
interference weights standing in for a bias the mask cannot address — while **the mechanism by
which Adam specifically finds that coalition is not reduced to a per-weight score by anything
here.** It is joint set optimisation under the log-uniform k schedule, and the same
per-parameter normalisation that makes Adam's scores diffuse (previous section) is the obvious
suspect, since diffusion is exactly what lets weak-but-aligned units climb out of the tail. That
suspicion is untested.

**One reading to refuse.** None of this says interference weights compute something. They are
standing in for a scalar per row. The honest summary is that a top-k mask over a frozen delta is
a *model class*, not a *lens*, and once its capacity exceeds the model's own it can buy loss in
ways that have nothing to do with attribution. That is a caution for every sparsity curve in
this repo where a sparse mask beats the full delta — `configs/pirate/posthoc/` at 157% of its own
finetune, and the fr2de reactivation family — and it is the first mechanism proposed for that
shape anywhere here.

## Scale sweep: two decades of virtual weights, and WHERE the methods disagree

`scripts/interference/interference_scale.py` (grid: `scripts/interference/run_interference_scale.sh`; figures:
`plots/plot_interference_scale.py`; data: `plots/data/interference_scale/{hard,lit}/`). Six sizes,
`n_feat` in {16, 32, 48, 64, 96, 128} = 2^8 .. 2^14 virtual weights, three model seeds each, and on
every model THREE attribution refits per method (so a method can be compared with itself). Held
fixed across the sweep: block size 16 (so `A`'s within-block structure is the note's at every
size and only the number of blocks changes) and the superposition ratio `n_res = n_feat / 8`. Not
held fixed, and every table below is read against it: the base rate, which is `0.1 / n_blocks`
(9% at 2^8 down to the note's 1.25% at 2^14), and at 2^8 there is one block, i.e. no off-block
interference at all and a two-dimensional residual. The question was whether the task gets easier
with fewer weights, and the hunch behind it was that MAttr+Adam and stepless IG agree on the
CIRCUIT weights and do something different on the NOISE weights. Every agreement number is
therefore computed over the real weights and over the interference weights separately.

**Is it easier at small scale? No — not as a ranking problem.** `hard` config, mean over 3 model
seeds at the full 3000-pass budget:

| weights | 2^8 | 2^10 | 2^11 | 2^12 | 2^13 | 2^14 |
|---|---|---|---|---|---|---|
| base rate (`dL > eps`) | 9.1% | 6.9% | 4.6% | 3.4% | 2.4% | 1.25% |
| P@R.2, every method | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P@R.8, stepless IG | 0.39 | 0.59 | 0.63 | 0.54 | 0.43 | 0.66 |
| P@R.8, MAttr+Adam | 0.20 | 0.42 | 0.32 | 0.26 | 0.05 | 0.26 |
| P@R.8, MAttr+SGD | 0.16 | 0.43 | 0.39 | 0.38 | 0.37 | 0.51 |
| rho vs dL, real weights, IG / Adam / SGD | .89/.82/.86 | .89/.76/.86 | .91/.81/.85 | .91/.82/.84 | .91/.79/.84 | .95/.90/.91 |
| rho vs dL, interference, IG / Adam / SGD | .55/.49/.39 | .60/.66/.40 | .62/.67/.44 | .57/.59/.38 | .62/.62/.45 | .58/.61/.42 |

The head is solved everywhere (P@R.2 = 1.00 at every size for every method), the tail is noisy
everywhere (seed spread on P@R.8 is ±0.3-0.4 at every size, so none of that row's differences
between adjacent sizes mean anything), and the rank correlations against the oracle are FLAT in
size on both populations. The smallest size is if anything the hardest: a 2-dim residual for 16
features leaves so much interference that a third of the `dL`-real weights are off `A`. So the
answer to "smaller is easier" is no; what smaller buys is 20-second cells, and the fact that the
phenomena below are all present at 2^10.

**The circuit half of the hunch holds, everywhere, for everything.** Restricted to `A_ij > 0`,
every method's Spearman against the oracle is 0.95-1.00 at every size, every pair of methods
agrees at 0.95-1.00, and every refit agrees with itself at ~1.00
(`plots/interference_scale_agreement_circuit_hard.pdf`, top row). Under the oracle's own `dL > eps`
labelling the numbers are lower (0.76-0.95 above) only because that set includes the interference
the loss happens to like. There is no disagreement about the circuit to speak of.

**The noise half holds, but not as "noise": MAttr+Adam's interference ranking is REPRODUCIBLE and
diverges from IG's, increasingly with scale.** Same cells, restricted to the interference weights:

| weights | 2^8 | 2^10 | 2^11 | 2^12 | 2^13 | 2^14 |
|---|---|---|---|---|---|---|
| Adam vs IG, real / interference | .94 / .72 | .84 / .74 | .81 / .68 | .81 / .60 | .76 / .53 | .88 / **.47** |
| Adam (lr 0.002) vs IG, real / interference | .94 / .87 | .92 / .80 | .91 / .75 | .91 / .63 | .93 / .49 | .94 / **.31** |
| SGD vs IG, real / interference | .97 / .88 | .95 / .83 | .95 / .83 | .93 / .84 | .97 / .84 | .97 / .85 |
| Adam vs its own refit, interference | .82 | .90 | .87 | .83 | .80 | .77 |
| Adam (lr 0.002) vs refit, interference | .97 | .98 | .98 | .96 | .95 | .94 |
| SGD / IG vs refit, interference | .99 / 1.00 | .99 / 1.00 | .99 / 1.00 | .99 / 1.00 | 1.00 / 1.00 | .99 / 1.00 |
| Jaccard of the interference members of two Adam refits' top-`n_real` sets | .62 | .40 | .45 | .45 | .37 | .29 |

Three readings:

- **Adam and IG rank the interference weights differently, and the gap widens with scale**
  (0.72 -> 0.47 at lr 0.05, 0.87 -> 0.31 at lr 0.002) while their agreement on the real weights
  stays 0.8-0.95. SGD is IG's twin on BOTH populations at every size (0.85 on interference, flat),
  which is the path-integral prediction in `scripts/interference/toy_sgd_vs_ig.py` again, and it makes the
  Adam divergence specific to the optimizer rather than to the mask objective.
- **It is not a random walk.** If Adam's interference scores were pure diffusion, two refits on the
  same model would agree at ~0; they agree at 0.77-0.90 (0.94-0.98 at the lower lr), and the refit
  agreement RISES with fitting steps (2^14: 0.61 at 256 passes -> 0.77 at 3000) while agreement
  with IG does not (0.40 -> 0.47) and agreement with `dL` does (0.47 -> 0.61) — see
  `plots/interference_scale_budget_adam_hard.pdf`. So more steps converge Adam's interference
  ranking on SOMETHING, and that something is neither IG nor a noisier version of it. The
  diffusion measured in the previous section is real and replicates at every size (interference
  score std grows 3.4-3.5x over 256 -> 3000 passes at 2^12 and 2^14, against sqrt(11.7) = 3.4),
  but it rides on top of a reproducible signal rather than replacing it. The one place the
  random-walk reading survives is the top-set membership: which interference weights land in
  Adam's top-`n_real` set is only 29-45% shared between refits (SGD/IG: 91-100%), and that fraction
  falls with scale — the tail's false positives are the diffused part.
- **What the reproducible part is, is the previous section's coalition, at every size.** Adam's
  loss-optimal top-k on real forwards beats both the full model and the circuit alone at every size
  from 2^10 up (best-k loss / full-model loss = 0.83 / 0.82 / 0.85 / 0.82 / 0.84 at 2^10 .. 2^14,
  against 0.93 / 0.93 / 0.95 / 0.92 / 0.93 for the circuit alone and 0.90-0.93 for IG, SGD and the
  `dL` oracle; `plots/interference_scale_trueloss_hard.pdf`), and it does it by keeping 3-6x the
  circuit's size in off-circuit weights (173 vs 50 at 2^10, 328 vs 80, 520 vs 108, 766-999 vs
  146-169, 1216 vs 207 at 2^14 — the ratio grows slowly with size) whose composition is the one
  documented above at 2^14, now at every size: 78-87% positive against a 45-53% population, smaller
  than the population from 2^10 up (median |U| 0.047 vs 0.067 at 2^10, 0.023 vs 0.041 at 2^14),
  and worthless one at a time to the oracle (median `dL` −1e-6 to −4e-6, half or more negative).
  IG, SGD and the oracle keep 0.1-0.5x the circuit's size off it and land on the circuit's loss.
  So the disagreement on the noise weights is the sign-aligned bias-standing-in coalition, it is
  what Adam converges to with steps, and it is identifiable in a 20-second cell.

**The lower learning rate is a different trade, not a fix.** Adam at lr 0.002 tracks IG on the
real weights (0.91-0.94 at every size) and beats the default lr on the tail (P@R.8 0.33-0.61
against 0.05-0.42 from 2^10 up), which is the previous section's n=1 result replicated across 15
cells. But its interference ranking is the LEAST IG-like of all at 2^14 (0.31), and at low budget
it is anti-correlated with the oracle: at 256 passes, every size from 2^12 up has
`rho(score, dL | interference)` = −0.25 to −0.28 with a refit agreement of 0.9, i.e. a
reproducible ranking of the interference weights in the WRONG direction, which only turns positive
(+0.26 to +0.39) by 3000 passes (`plots/interference_scale_budget_adam0.002_hard.pdf`). The
reading offered — not measured — is that this is the set-conditional sign flip of the previous
section seen from the fitting side: early in a log-uniform-k fit most draws are sparse, and in a
sparse model an interference weight's marginal value has the opposite sign to its value in the full
model that `dL` ablates from.

**The scatter is the picture behind the numbers** (`plots/interference_scale_scatter_adam_hard.pdf`,
seed 0 at each size, Adam against IG per weight, coloured by `A`). The circuit weights sit on one
monotone curve at every size. The off-circuit weights form two structures that grow with size: a
floor at Adam's most negative score (weights Adam has pushed out regardless of what IG says), and a
vertical band at IG ~ 0 — weights whose single-weight, whole-path effect is nil and which Adam
places anywhere from the floor to the top. That band is where the scale-dependence lives: of the off-circuit weights
Adam's loss-optimal set keeps, the fraction whose |IG| is below the off-circuit median is 8% at
2^8 and 25% at 2^14. The SGD twin
(`..._scatter_sgd_hard.pdf`) has neither structure: its off-circuit weights lie on a curve of IG's
score with a fan below it, which is the k-schedule's variance and nothing else.

**On the note's own config (`lit`, learned `W_down`) the split is the same and sharper.** Same
grid, `--down learned` (`plots/data/interference_scale/lit/`, figures `*_lit.pdf`). The real
weights are easy at every size — P@R.8 0.86-0.96, rho vs `dL` 0.97-0.99 for every method, every
pair of methods at 0.96-0.98 — because the model cancelled most of its own interference, as the
top of this document says. And the interference weights are correspondingly UNRANKABLE against
the oracle by anyone: rho vs `dL` on them is 0.47 -> 0.15 (IG), 0.33 -> 0.08 (Adam), 0.34 -> 0.04
(SGD) from 2^8 to 2^14, i.e. what little `dL` structure the interference has vanishes with size.
Adam vs IG on the interference weights is 0.10-0.29 at every size (SGD vs IG 0.33-0.54), yet
Adam's refit agreement on them is 0.92-0.96: once more a reproducible ranking of the noise that
is not IG's and not the oracle's. The Adam-vs-IG scatter (`..._scatter_adam_lit.pdf`) is a
symmetric butterfly — IG assigns the off-circuit weights both signs, Adam puts nearly all of them
on its floor with two wings rising off it — and the loss-optimal set keeps 10-30x the circuit's
size in off-circuit weights at 2^14 (1589-3504 against ~130) for a best-k loss of 0.72 x the
full model's against 0.78 for the circuit alone. Two things `lit` adds to the `hard` reading:
the interference disagreement is not an artifact of the frozen projection, and where the
interference weights carry NO `dL` signal at all the methods still disagree about them in a
reproducible way — so "agree on the circuit, differ on the noise" is the right summary of the
hunch, with the one correction that Adam's half of the difference is not noise.

What is **not** done here: one attribution recipe per method (3000 passes, T = 0.5, log-uniform k,
batch 2048), no size above 2^14, and the reproducible part of Adam's interference ranking is
characterised by its composition and its true loss but not reduced to a per-weight formula — the
same open end as the previous section, now known to be reachable in a 20-second cell.

## How MAttr+Adam ranks the interference weights: one feature, two regimes

`scripts/interference/interference_adam_pattern.py` (both configs, all six sizes, 3 seeds;
`plots/data/interference_scale/<tag>/adam_pattern.txt` and `adam_pattern_n<n>.json`; figure
`plots/plot_interference_adam_pattern.py` -> `plots/interference_adam_pattern.pdf`). The question
the previous two sections left open — is there a simple per-weight rule behind Adam's
reproducible, un-IG-like ranking of the noise — asked directly, by rank-correlating Adam's
off-circuit (`A_ij = 0`) scores with a list of one-line candidate features, and by asking which
features predict membership in Adam's loss-optimal set. The candidates: the weight `U_ij` (either
sign), `sign(U_ij) * r_i` and `U_ij * r_i` where `r_i = E[(y_i - y'_i) 1(gate open)]` is the target
row's mean under-prediction in the CIRCUIT-ONLY model, IxG evaluated at zero / at the circuit-only
model / at the full model / at several sparse top-k models, the other methods' scores, the oracle
`dL`, target firing rates, and leave-one-out group means by row x sign and column x sign (the ceiling
for any "row property times sign" story).

**Two regimes, split by whether the target row is in the circuit at all.** ~20% of `A`'s rows are
all-zero: the target is `ReLU(-0.1) = 0` always, the gate should never open, and 15-21% of the
off-circuit weights sit in those rows.

- **Dead rows: every method ranks by `-U_ij`** ("keep the gate shut"), Adam at rho 0.87-0.89,
  SGD and IG at 0.90-0.91, and IxG@full / `dL` at 0.92-0.99. There is nothing method-specific here
  except placement: on `hard` Adam puts the whole dead-row population 0.3-0.8 score units ABOVE
  the live-row bulk (SGD/IG: 0.00), which is the vertical band at IG ~ 0 in the scatter figure —
  weights whose whole-path, single-weight effect is nil because the gate is shut, and which Adam
  treats as harmless-and-welcome, ordered by how surely they keep it shut.
- **Live rows: what Adam KEEPS is `U_ij * r_i`.** Among the off-circuit weights, the AUC of each
  feature for membership in Adam's loss-optimal top-k (`hard` / `lit`, 2^10 .. 2^14):

  | feature | 2^10 | 2^11 | 2^12 | 2^13 | 2^14 |
  |---|---|---|---|---|---|
  | `U_ij * r_i` | **.85** / **.92** | **.81** / **.97** | **.83** / **.98** | **.85** / **.97** | **.81** / **.97** |
  | `U_ij > 0` | .74 / .51 | .76 / .58 | .69 / .61 | .73 / .65 | .70 / .73 |
  | oracle `dL` | .60 / .36 | .63 / .39 | .53 / .39 | .58 / .40 | .54 / .35 |
  | IxG @ full model | .40 / .34 | .46 / .34 | .44 / .32 | .45 / .36 | .43 / .31 |
  | MAttr (SGD) score | .93 / .90 | .91 / .87 | .90 / .93 | .89 / .93 | .87 / .94 |
  | stepless IG score | .88 / .49 | .86 / .57 | .81 / .59 | .81 / .59 | .78 / .67 |

  One feature, no fitting, at every size on both configs: **Adam keeps an interference weight when
  its sign matches the direction its target row is under-predicted in the circuit-only model** —
  positive weights on rows the sparse model leaves too low, negative on rows it leaves too high.
  `U_ij * r_i` is, up to the constant `2 E[x_j]`, exactly IxG evaluated at the circuit-only model
  (the two columns are identical in the script's output), so the rule can also be read as "the
  first-order value of switching the weight on in the sparse model" — the per-weight statement of
  the bias-offset coalition. The oracle and IxG@full, which score the weight in the FULL model, are
  at or below chance for the same set — which is why the previous section's oracle could not see
  the coalition — and `-|U|` (0.53-0.63) and the dead-row flag (0.37-0.41) say the "small" and
  "diffuse" parts of the earlier description are consequences, not the rule.

**The ORDER of the rest is where `hard` and `lit` differ, and it is why the whole-population
Spearman looked messy.** Over all live-row off-circuit weights, Adam's score correlates with
`U * r_i` at 0.85 on `lit` but only 0.25-0.45 on `hard`, where its best single correlate is the
full-model ablation effect (IxG@full 0.52-0.63, `dL` 0.58-0.66). A two-term rank mixture
`a * rank(U r_i) + (1 - a) * rank(IxG@full)` fits Adam's live-row ranking at 0.65-0.70 on `hard`
with `a` falling 0.47 -> 0.25 from 2^10 to 2^14, and at 0.75-0.85 on `lit` with `a` 0.68 -> 0.90.
The reading: the top of Adam's ranking (what it keeps) is the sparse-model value; the bottom (which
weights it sends to the floor first) is the full-model value, i.e. the weights whose removal from
the dense model hurts most are the LAST to be pushed out. On `lit` the full-model signal is ~0
for every interference weight (the model cancelled it), so only the sparse-model term is left and
the fit is clean; on `hard` both are live, and the full-model term grows with size. That is also
the resolution of the previous section's "Adam vs IG diverges with scale": IG's path scales all
weights together and never visits a sparse model, so it carries the full-model term and not the
sparse one; SGD, whose live-row ranking is `U * r_i` at 0.82-0.86 on `hard`, carries the sparse
one — and neither has both.

**A caution on what "kept" means.** SGD's and IG's own scores also rank Adam's kept set well (AUC
0.87-0.94 and 0.78-0.88 on `hard`): they can SEE which interference weights are good in the sparse
regime, they just stop at k ~ circuit size, and forced out to Adam's k their top-k sets are much
worse (previous section: 5.2 vs 3.3 at k = 1400) because the weights they add next are the large
positive ones. So the per-weight rule above says what Adam keeps; it does not, by itself, say how
many, which is set by the joint optimisation and is the part still not reduced to a formula.

## The note's models were deliberately UNDERTRAINED — checked, and it is not the missing knob

The note says so in three places (all verbatim from the published HTML, 2026-09-05): the simple
model "will also be undertrained (this reproduces the phenomenology of Towards Monosemanticity
better; we'll explore fully converged examples later)"; Appendix 2 opens "The models we looked at
weren't fully trained to convergence, since this produced more similar phenomenology to real
models"; and the sophisticated model's paragraph lists four criteria for a good regime — (1)
overlap, (2)/(3) non-degenerate two-seed scatter, "(4) this continues to hold as one trains to
convergence" — and says the block-diagonal construction "can achieve (1–3), but not (4)". So the
published sophisticated-model figures are of an unconverged model by design, and the step sweep at
the top of this document (300–120k, every cell under a cosine schedule to zero) never looked at
one: it ran every budget to its own convergence. Read off the decoded figures (the HTML embeds them
as PNGs): the two-seed scatter's interference cloud has radius ~0.65 (std ~0.25), the histogram's
interference mode sits at ~−0.15 with a long negative tail to −1.0, and the learned-vs-ideal plot
has a dense "Unlearned" band and positive/negative interference lobes reaching ±0.6/−0.85.

`scripts/interference/interference_undertrained.py` (figure `plots/interference_undertrained.pdf`,
data `plots/data/interference_undertrained/`) trains the literal config ONCE per setting with
snapshots at 10–10,000 steps and, at each snapshot, measures what the note's figures fix:
interference std, base rate, and `weight` / `era` / `twera` precision at fixed recall against
that snapshot's own `dL`. Ten trajectories: lr {5e-4, 2e-3, 1e-2} constant and 2e-3 cosine at the
stated init; init scale ×{2, 3, 4}; and, because the prose gives 0.5 where the figure config gives
0.1, blocks at density 0.5 with init ×{1, 1.5, 2}. What they show:

- **At the stated init, undertraining has no intermediate regime.** Every lr goes from "nothing
  learned" (base rate 30–45%, `weight` P@R.2 ≈ 0.15) to "`|U|` is a perfect classifier"
  (P@R.2 = 1.00, ERA 1.00/1.00) within ~100 steps of each other — by step 100–200 at 2e-3, step 50
  at 1e-2, step 500 at 5e-4 — and the interference std sits at 0.055–0.065 the whole way. The one
  snapshot that resembles the published `weight` curve (lr 5e-4, step 300: base rate 2.9%,
  `weight` 0.72/0.24) has ERA at 0.20/0.04, i.e. worse than `weight`, which the note never shows.
  Cosine and constant are the same trajectory to step 5000.
- **Init scale ×2 is the one setting that holds the note's interference scale, and it holds it
  because the model never cleans it up.** Interference std is 0.196–0.198 from step 10 to step
  3000 and 0.23 at 10,000 — init noise that training leaves in place — while only 64–115 of the
  207 circuit entries are ever learned (base rate 0.4–0.7%). Its snapshots pass the published
  `weight` value (0.67/0.68 at step 1000 against 0.68/0.44) but ERA is 1.00/1.00 from step 1000 on,
  and TWERA climbs only to 0.77/0.64 by 10k. Init ×3 and ×4 never learn at all (loss 5.5/5.9 at
  10k, 11/5 circuit entries recovered, interference std 0.45/0.82 = pure init noise).
- **0.5-dense blocks do not give the note's base rate under undertraining either.** With 1010
  circuit entries the model learns ~90% of them by step 200–500 at every init, so the base rate
  is 5.5%, not 1.2%. There is one transient that reproduces the published ERA exactly — init ×2,
  step 300: ERA 0.85/0.56 against 0.87/0.55, `weight` 0.81/0.77 — but at base rate 6.9%.

So across ten runs no snapshot has the note's base rate, `weight` curve and ERA curve at the same
step: the transition from unlearned to solved is too sharp in this parameterisation for any
undertrained model to sit where the published one does. Undertraining is real, it is what keeps a
large-init interference band alive (the ×2 run's 0.2 is the closest any config here comes to the
note's 0.25), and it is part of the answer — but not the whole of it. The note states no learning
rate, optimizer, init, batch size, step count, or how `dL` was estimated (its histogram's ±0.02
colour range is consistent with this document's sum-over-features loss), and one of those is
still doing the work. The frozen-projection `hard` config remains the only setting here that
reproduces the published ERA curve at the published base rate, and it does so at convergence.


## TRAINING `hard` LONGER IS THE MISSING KNOB — at 30k steps it reproduces the note

The section above asked whether undertraining the LITERAL config recovers the note, and it does
not. The right question was the other one: the frozen-projection `hard` config had only ever
been trained for 3,000 steps, the budget the literal config converges in, and under the frozen
projection that is far from converged (loss 3.92 against 3.64 at 30k). Same script, `--steps
30000` / `100000`, with `--check` for the two-seed panels; `--shared-down` (new) pins the frozen
projection to one seed so the second model differs only in init and data order. Figure:
`plots/plot_interference_model_grid.py` -> `plots/interference_model_grid.pdf`, the validation
row at 3k / 30k / 100k; numbers in `plots/data/interference_undertrained/hard_length.json`.

| model | loss | base rate | A recovered | interf. std | shrinkage (median U/A) | real mean / interf. std | cross-seed r, circuit / interference | AP era / TWERA / weight / freq | ERA P@R.2 / .4 | weight P@R.08 / .2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `hard`, 3k (the results above) | 3.920 | 1.23% | 140/207 | 0.093 | 0.13 | 0.64 | 0.30 / 0.01 | 0.42 / 0.04 / 0.04 / 0.03 | 0.87 / 0.46 | 0.05 / 0.05 |
| `hard`, 30k | 3.637 | 0.99% | 163/207 | 0.035 | 0.12 | 1.68 | 0.70 / 0.01 | 0.55 / 0.60 / 0.37 / 0.04 | 0.94 / 0.74 | 0.70 / 0.65 |
| `hard`, 100k | 3.636 | 1.00% | 164/207 | 0.031 | 0.12 | 1.90 | 0.73 / 0.01 | 0.55 / 0.60 / 0.45 / 0.04 | 0.92 / 0.74 | 0.88 / 0.80 |
| shared projection, 3k | 4.049 | 1.40% | 125/207 | 0.094 | 0.12 | 0.53 | 0.45 / 0.39 | 0.38 / 0.04 / 0.05 / 0.03 | 0.84 / 0.36 | 0.06 / 0.05 |
| shared projection, 30k | 3.649 | 1.00% | 164/207 | 0.035 | 0.12 | 1.62 | 0.91 / 0.76 | 0.54 / 0.47 / 0.39 / 0.04 | 0.94 / 0.74 | 0.88 / 0.70 |
| shared projection, 100k | 3.649 | 0.99% | 163/207 | 0.031 | 0.12 | 1.89 | 0.93 / 0.71 | 0.55 / 0.60 / 0.48 / 0.04 | 0.92 / 0.74 | 1.00 / 0.89 |
| the note (read off its figures) | – | ~1.2% | – | ~0.25 | large | ~1.6 | agree / independent | ~0.5 / ~0.4 / ~0.3 / ~0.02 | 0.87 / 0.55 | 0.68 / 0.44 |

Four readings:

- **At 30k steps `hard` reproduces every published validation panel and the P/R curves, in
  ratio.** Circuit weights agree across seeds (r 0.30 -> 0.70) while the interference stays
  independent (r 0.01) — the note's "the interference weights are independent, but the real
  weights are all significantly positive"; the real-to-interference overlap ratio is 1.7 against
  the note's ~1.6; the base rate is 1.0% against ~1.2%; and the heuristics land at the note's AP
  magnitudes with its ordering (era > TWERA > weight > freq at ~0.55 / ~0.5 / ~0.4 / 0.04 against
  ~0.5 / ~0.4 / ~0.3 / ~0.02; TWERA and ERA swap within seed noise). The one thing that does not
  match is the absolute scale: everything is ~4x smaller (real mean 0.06 vs the note's ~0.4,
  interference std 0.035 vs 0.25), because the frozen projection shrinks the whole fit; the
  curves depend on the ratio.
- **What the extra steps did: cleaned up interference, not learned circuit.** Loss 3.92 -> 3.64,
  interference std 0.093 -> 0.035 (a 2.7x narrowing), recovered circuit entries 140 -> 163, and
  the shrinkage of the learned circuit weights unchanged (median U/A 0.12-0.13). At 3k the
  interference band is WIDER than the circuit (ratio 0.64), which is why `|U|` and TWERA were at
  the base rate and the earlier results table read "all three methods beat every heuristic by a
  wide margin" — partly a statement about an unconverged model.
- **100k is a plateau that drifts the wrong way.** Loss 3.636 vs 3.637, base rate and recovery
  unchanged, r 0.73 vs 0.70 — but the interference keeps narrowing (std 0.031, ratio 1.9) and
  `weight` climbs from the published 0.68/0.44 toward a perfect classifier. Train `hard` long
  enough and it becomes `lit`, slowly. 30k is the operating point.
- **Sharing the frozen projection across seeds is NOT the note's "different seed".** With one
  projection the two models' interference is correlated (r 0.38 at 3k, 0.76 at 30k), where the
  note's is independent. A learned `W_down` differs across seeds, and so must the projection;
  per-seed draws — what `--check` did all along — are the faithful analogue, and the earlier
  worry that this "only weakly reproduces" the scatter claim was the 3k budget, not the seeds.

Consequences for what is above. The `hard` results table, budget sweep and off-circuit analysis
were all run on the 3k model and stand as measurements of it, but its heuristic baselines
(`weight` 0.05, TWERA 0.04) are those of an unconverged model and should not be quoted as the
note's difficulty. `plots/data/interference_toy_hard30k/` now carries the full pipeline
(`curves.json` with the three methods at 3000 draws/steps, `scores.pt`, `true_loss_{linear,log}.json`):
all three methods hold precision 1.00 to recall ~0.75 (P@R.8 Adam 0.94, stepless IG 0.92, SGD
0.90) against ERA's 0.74 at recall 0.4, and the true-loss panel keeps its shape (heuristics and
SGD/IG blow up to 5.5-6.5 mid-curve, MAttr+Adam runs below the circuit line). Spearman vs `dL`
is Adam 0.74 / IG 0.24 / SGD 0.10 on this model against 0.56 / 0.54 / 0.44 on the 3k one — a
different ordering, so do not carry the 3k number across. The scale sweep's `hard` cells are
3k-step models too (`train_steps: 3000` in `interference_scale.py`); its findings are about
agreement between methods and are not expected to hinge on this, but a 30k rerun is the check.

**Addendum — the undertrained `lit` trajectory is a ONE-PARAMETER FAMILY, and the note's state is
not on it.** `--pair --snaps ...` (paired seeds trained in lockstep, fine snapshots) at constant
lr 1e-3 / 5e-4 / 2e-4: the three runs trace the same curve in (loss, base rate, overlap ratio,
cross-seed r) with the learning rate only rescaling the step axis (lr 1e-3 step 200 = lr 5e-4
step 400 = lr 2e-4 step 1000: loss 3.38 / 3.35 / 3.31, base rate 1.65 / 1.60 / 1.50%, ratio
1.92 / 1.92 / 1.94, circuit r 0.71 / 0.71 / 0.72). Along it the note's validation numbers ARE
passed through — overlap ratio 1.6 at base rate ~2.2%, base rate 1.2% at ratio ~2.1, circuit r
0.7 with interference r 0.05 throughout the window (criteria (2)/(3) hold; note that interference
r then RISES with training, 0.05 -> 0.19 by convergence, which is the "same ideal superposition
configuration" collapse of their criterion (2) beginning) — but the heuristic curves at those
points are still not the note's. At base rate 1.2-1.4% (lr 5e-4 steps 425-450): ERA 0.97 / 0.44-0.64
(published 0.87 / 0.55, close), but `weight` 0.95 / 0.87 (published 0.68 / 0.44) and TWERA
0.41-0.49 / 0.22-0.29 (published 0.81 / 0.47). At ratio 1.6 (step ~330): `weight` P@R.2 0.48
matches, ERA is 0.56 / 0.04 and TWERA 0.23 / 0.04. So `|U|` is too strong and TWERA too weak at
EVERY point of the family, at every lr, and the family is what Adam from this init produces
whatever the lr — so the note's state, if it is an undertrained one, was reached from a different
init or optimizer. `plots/data/interference_undertrained/trajectory_fine*.json`.

**Addendum (2026-09-07): a better single metric for Adam's ordering on `hard` — searched, and
there isn't one; the bias magnitude is not the missing ingredient.**
`scripts/interference/interference_signprob.py` scores candidates over the live-row off-circuit
weights of `hard` (3k) and `hard30k`, against each method's scores (Spearman) and Adam's kept
set (AUC). Three hypotheses tested and the result of each:
- *Adam ranks by a SIGN-PROBABILITY over sparse masks* (its step is ~lr·sign(g), so its score
  would be #masks where switching the weight on helps minus #masks where it hurts): **refuted**.
  `P_help` over log-uniform-k top-k masks (of Adam's own ranking or the oracle's, 400 masks)
  correlates −0.25 / −0.14 with Adam. The *expectation* `E_help` over the same masks is SGD's
  ranking (ρ 0.93–0.94 with SGD, 0.79–0.82 with IG, non-circular from the oracle's masks) — the
  path-integral reading of SGD confirmed from the mask side — and `E/rms` (the Adam-shaped
  normalised statistic) is no better (0.16–0.22 with Adam).
- *The bias magnitude*: `U·r_i/|b_i|`, `U·r_i·|b_i|`, `(U·r_i)/(|b_i|+|r_i|)` all reproduce
  `U·r_i` to ±0.02 (ρ Adam 0.19–0.29, AUC 0.80–0.85); `U·(−b_i)` and `U/|b_i|` are worse
  (ρ ≤ 0.09). `r_i` already carries the bias mismatch; |b_i| adds nothing.
- *Sparse + dense combinations*: `rank(U·r_i) + 2·rank(IxG@full)` reaches ρ 0.78 at 30k, i.e.
  equal to IxG@full / `dL` alone (0.77), at a worse kept-set AUC (0.62 vs U·r_i's 0.84).
So on `hard30k` the two halves of Adam's ranking have two different best predictors and no
metric serves both: the kept set is `U·r_i` (AUC 0.84, `dL` at chance 0.44), the ordering of
everything else is the full-model single-weight effect (`dL`/IxG@full ρ 0.77, `U·r_i` 0.19). Adam's
scores are not a first-order statistic of any fixed mask distribution tried here — they are the
integral of `m_t/sqrt(v_t)` along its own trajectory, which is path-dependent — and a formula for
the ordering remains open.

**Addendum (2026-09-07, the ordering, RESOLVED by splitting populations).** The whole-off-circuit
Spearman above mixed two regimes. Split three ways on `hard30k` — the 207 circuit weights, the
981 off-circuit weights in Adam's loss-optimal set, the 11,740 off-circuit weights it does not
keep — Adam's order within each is one feature (`interference_signprob.py`'s per-population table):

| population | Adam's order is | ρ | and `U·r_i` / `dL` there |
|---|---|---|---|
| on circuit (207) | `dL` ≡ IxG@full ≡ E_help | 0.99 | 0.30 / 0.99 |
| kept off-circuit (981) | `U·r_i` (E_help 0.80) | **0.83** | 0.83 / **−0.83** |
| not-kept off-circuit (11,740) | `dL` ≡ IxG@full | **0.93** | 0.09 / 0.93 |

So Adam ranks the circuit by the full-model effect, the interference it keeps by the
sparse-model value `U·r_i`, and the interference it discards by the full-model effect again —
and inside the kept set the full-model effect runs *backwards* (ρ −0.83 with `dL`): the
interference weights Adam ranks highest are the ones whose removal from the dense model would
help most, i.e. the ones that overshoot when the whole interference population is present and
are useful only in the sparse model. SGD's within-population order is `U·r_i` / E_help
everywhere off the circuit (0.93 kept, 0.87 not kept), which is why its ranking looks like a
compressed IG. No single feature covers all three (`max(rank U·r_i, rank IxG@full)` is
0.99 / 0.78 / 0.25); the piecewise description is the rule.
On the unconverged 3k model the same split is blurrier: circuit by `dL` (0.98), not-kept by
IxG@full (0.76), but the kept set's order follows the EMPTY-model residual `U·(E[y_i] − ReLU(b_i))`
(0.82) rather than the circuit-only one (0.49) — on that model the circuit-only model is itself a
poor reference, since 67 of 207 circuit entries are unlearned.
**What `r_i` is made of: the interference's lost mean offset, not shrinkage.** Decomposed on
`hard30k` (live rows, gate-open residual `r_i = E[(y_i − ŷ_i) 1(open)]`): the full model has
|r_i| = 0.0005 (the bias absorbed everything); the circuit-only model has 0.0285, of BOTH signs
(mean +0.004) — a per-row mismatch, not a uniform deficit; adding the interference's mean
contribution `E[x]·Σ_j U_ij^off` back into the bias removes 80% of it (0.0058, and what is left is
uncorrelated with the original, r 0.19); while UNshrinking the circuit to `A` with the same bias
makes it 4x worse (0.127), because `b` was fit to shrunk-circuit-plus-interference and `A` alone
overshoots. corr(`r_i`, interference offset) 0.40, corr(`r_i`, shrinkage deficit `E[x]·Σ(A−U)`)
0.02. Same on `lit` (0.0174 → 0.0126 with the offset; 0.38 vs 0.17). So `r_i` is the row's
lost interference offset seen through the bias, shrinkage only sets the scale of everything.

## MAttr with a free per-row bias: the coalition is a bias, and the ranking becomes near-oracle

`scripts/interference/interference_biasmask.py` (`plots/plot_interference_biasmask.py`): fit the
scores jointly with a learnable per-row bias correction δb through `learn_scores`'s
`extra_params` (δb lr 0.01, everything else the standard recipe), on `hard30k`. Since the
coalition stands in for a per-row constant, give the mask that constant directly:

| ranking, evaluated with | best k | true loss | kept: circuit / off | P@R.8 vs ΔL | ρ vs ΔL |
|---|---|---|---|---|---|
| plain Adam, original b | 1178 | 3.116 | 171 / 1007 | 0.94 | 0.74 |
| bias-mask scores, original b | 164 | 3.240 | 164 / **0** | **1.00** | **0.82** |
| bias-mask scores, b + δb | 719 | **3.078** | 170 / 549 | 1.00 | 0.82 |
| plain Adam, b + δb | 139 | 3.119 | 130 / 9 | 0.94 | 0.74 |
| circuit alone, b / b + δb | – | 3.242 / 3.092 | | | |

- The bias-mask ranking puts every learned circuit weight above every interference weight
  (precision 1.0 to recall 1.0; under the original bias its loss optimum IS the circuit, 0
  off-circuit), where plain Adam's tail was 0.94.
- δb learns the offset: corr(δb, `E[x]·Σ_j U_ij^off`) = +0.48 over live rows, and the circuit
  alone with b + δb (3.092) beats plain Adam's 1007-weight coalition (3.116). Given δb, even the
  plain ranking's optimum collapses to the circuit (1007 → 9 off-circuit weights).
- Not entirely gone: under b + δb the optimum still adds 549 off-circuit weights for 0.014
  (3.092 → 3.078), because one δb is shared across the log-uniform-k schedule and is tuned for
  its sparse end (`L(full U, b + δb)` = 3.815 vs 3.645 under b) — a per-k bias would close that.
So "what MAttr+Adam keeps off the circuit" was a bias all along, and learning the bias with the
mask is the attribution-side fix: the ranking then answers the note's question directly.
**What orders the bias-mask scores** (`interference_signprob.py --biasmask`; features under
b + δb; ignore its SGD/IG columns there, those are the plain fits): off the circuit the ranking
is now the full-model single-weight effect almost outright — ρ 0.88 with ΔL over all live-row
interference (plain Adam: 0.77), 1.00 on the circuit, 0.92 on the not-kept interference. The
residual 549-weight coalition under b + δb is ordered by `U·r_i` computed under the NEW bias
(0.67; AUC 0.79 vs 0.45 for `U·r_i` under the original b), i.e. by whatever offset the
schedule-shared δb failed to absorb, and it is worth 0.014 in loss against the plain coalition's
0.126 (L(circuit, b) − L(plain optimum)): a ninth of the value from half the weights. So the
sparse-model term is nearly gone from the ranking and what is left is the oracle's own order.
**What separates the coalition Adam keeps from the same KIND of weight it buries.** In the rank
composition figure (`plots/interference_rank_composition_hard30k_all*.pdf`) the "helps once the
interference is gone, not needed in the full model" class appears twice: ~940 of them right after
the circuit (kept) and ~3800 at the very bottom. They have the same sparse-model value (median
`U·r_i` 4.9e-4 vs 5.0e-4) and sit in the same rows (69 of the bottom block's 87 rows are also
top-block rows); what differs is the WEIGHT: |U| 0.013 vs 0.021, `r_i` 0.024 vs 0.003 — a small
weight into a row that needs a lot, against a large weight into a row that needs little — and
therefore ΔL (−8e-6 vs −2e-5: the large ones hurt the full model most). Within that class
Adam's score has ρ −0.68 with |U| (−0.7 to −0.8 inside every quintile of `U·r_i`) and +0.69 with
`U·r_i − 30U²`, which is the value-minus-variance rule from the off-circuit section stated as an
ordering: at equal offset supplied, the smaller weight injects less variance, so Adam keeps the
small ones and sends the big ones to the bottom of the ranking, below even the weights that hurt
the sparse model.
**The joint per-row bias does NOT transfer to `lit` as fitted.** Same script, δb lr 0.01 / 0.003 /
0.001: every fit lands on a δb that makes the circuit-alone loss WORSE (1.59–1.76 against 1.38
with the original bias), and the co-fitted scores lose the tail (P@R.8 0.49–0.52 against plain
Adam's 1.00; ρ vs ΔL 0.05 against 0.09). |δb| is 0.06–0.13, three times `hard`'s, and
`L(full U, b + δb)` is 2.3–2.5 against 1.77: the shared δb is tuned to the near-empty masks
that the log-uniform k schedule mostly samples, and on `lit` the offset those masks want is far
from what the circuit-only or dense masks want (the empty-model residual `E[y] − ReLU(b)` is large
there). On `hard` the offsets are small and consistent across k, so one δb served. A k-dependent
bias, or a δb fitted under a schedule that samples denser masks (`--k-schedule uniform`), is the
obvious repair and is the open item. The `lit` composition grid is therefore drawn without the
bias-mask panel.
**RESOLVED: fit δb under a UNIFORM k schedule and the joint bias works on both configs.** The
failure above was the schedule, not the idea: with `log` k half the draws are near-empty masks,
and a single δb shared across k gets pulled to what *those* want. `--k-schedule uniform` (now the
script's default; the plain fits keep `log`):

| | `hard30k`, log k (above) | `hard30k`, uniform k | `lit`, uniform k |
|---|---|---|---|
| L(circuit, b) → L(circuit, b + δb) | 3.242 → 3.092 | 3.242 → **3.078** | 1.381 → **1.234** |
| plain Adam optimum (for reference) | 3.116 @ 1178 (1007 off) | same | 1.306 @ 1389 (1238 off) |
| joint-fit optimum, b + δb | 3.078 @ 719 (549 off) | **3.074 @ 373 (202 off)** | **1.230 @ 848 (684 off)** |
| joint-fit scores vs ΔL: P@R.8 / ρ | 1.00 / 0.82 | 1.00 / **0.91** | 1.00 / 0.17 |
| plain Adam ranking under b + δb | 3.119 @ 139 (9 off) | 3.108 @ 139 (9 off) | 1.254 @ 100 (**0** off) |
| corr(δb, interference offset) | +0.48 | +0.41 | +0.66 |

On `hard30k` the joint ranking is now the closest thing to the oracle any method has produced
(ρ 0.91 vs ΔL; plain Adam 0.74) and its optimum keeps a fifth of the coalition plain Adam
needed. On `lit`, where the log-k δb made things worse, the uniform-k δb alone takes the circuit
below plain Adam's 1,238-weight coalition, and plain Adam's own ranking evaluated under b + δb
keeps zero interference. The remaining off-circuit weights at each optimum (202 / 684) are
what one schedule-wide δb still cannot express — a per-k bias would be the next step. (`lit`'s
low overall ρ vs ΔL, 0.17, is `lit`'s interference having no ΔL structure to correlate with, not
a defect: its P@R is 1.00 to recall 1.0.)

## The coalition is an artefact of ZERO ablation: replace a masked weight by its mean and it is gone

`scripts/interference/interference_constmask.py`: the masked model becomes
`z_i = Σ_j m_ij U_ij x_j + Σ_j (1 − m_ij) c_ij + b_i`, a masked weight contributing a constant
`c_ij` instead of nothing. Fixed `c_ij = U_ij E[x_j]` is mean ablation; `C` can also be learned
jointly with the scores (`extra_params`), which makes a masked model's effective bias
`b_i + Σ_masked c_ij` — mask-dependent, the per-k bias one shared δb could not express. Figure:
`plots/interference_ablation_baselines.pdf`.

| `hard30k` | best k | true loss | kept: circuit / off | ρ vs ΔL |
|---|---|---|---|---|
| circuit alone: zero ablation → mean ablation | – | 3.242 → **3.080** | | |
| plain Adam ranking, zero ablation | 1178 | 3.116 | 171 / 1007 | 0.74 |
| plain Adam ranking, MEAN ablation | 164 | 3.098 | 139 / **25** | 0.74 |
| **MAttr+Adam fitted under mean ablation** | **193** | **3.077** | 172 / **21** | **0.92** |
| joint per-row bias (uniform k) | 373 | 3.074 | 171 / 202 | 0.91 |
| learned per-weight C (mean init, log k) | 2275 | 3.106 | 173 / 2102 | 0.71 |

- **Mean ablation alone removes the coalition.** The circuit under mean ablation (3.080)
  already beats plain Adam's 1007-weight coalition under zero ablation (3.116); plain Adam's own
  ranking evaluated under mean ablation has its optimum at 164 weights with 25 off-circuit; and a
  ranking FITTED under mean ablation is the best of everything — loss 3.077 at k = 193 (the
  learned circuit plus 21), and ρ 0.92 vs the oracle, the highest any method has reached. So
  what Adam was assembling under zero ablation was `Σ_j U_ij E[x_j]` per row: the interference's
  mean, which mean ablation supplies by definition.
- **Learning C does not beat the fixed mean** (3.085–3.106 vs 3.077–3.098; under uniform k it
  degrades outright, `L(circuit, C)` 3.50) — the learned constant drifts with the k schedule
  where the fixed one cannot.
- **On `lit` mean ablation is the WRONG baseline**: the circuit alone goes 1.381 → 1.626, and
  MAttr fitted under it lands at 1.494 (plain Adam under zero ablation: 1.306). `lit`'s
  interference cancels PER EXAMPLE — its effect on a row is variance the ReLU rectifies into an
  offset, not a mean — and replacing each weight by its mean drops the variance while keeping the
  mean, which the co-adapted bias then double-counts. The learned per-row δb (uniform k) found the
  right shift there (1.234), which is the case for learning the constant rather than fixing it.
So the ablation baseline decides what "interference" means to a mask: under zero ablation the
sparse model is missing the interference's mean and Adam rebuilds it from small aligned weights;
under mean ablation on `hard` nothing is missing and every method agrees with the oracle.
