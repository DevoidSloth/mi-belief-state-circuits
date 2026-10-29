# Results

Every number below is read from `results/*.json`, which the experiments write. Rebuild with `bsc report`.

Environment: NVIDIA GeForce RTX 4080 Laptop GPU, torch 2.9.1+cu126, python 3.14.0, commit `0385121`.

## Headline numbers

| # | metric | measured | note |
|---|---|---|---|
| 1 | Optimality gap | median +0.00250, worst +0.02262 nats/token | 14/15 under 0.01 |
| 2 | Baseline separation | median 0.0027 nats | 0 processes over 0.15 |
| 3 | Belief-state probe fidelity | median best R² 0.9810 (min 0.5448) | held-out sequences, split by sequence |
| 4 | Causal validity of the probe | median steering margin +0.524 | over a norm-matched random direction |
| 5 | Circuit size | 6.0 of 10 components (sns) | greedy causal-scrubbing search |
| 6 | Circuit faithfulness | 0.910–0.977 over the 4 processes with a sparse circuit | vs the best context-free predictor; no sparse circuit on rrxor |
| 7 | Ablation effect | 9×–4766× KL-to-optimal | mean ablation of the top component |
| 8 | Kalman gain recovery | impulse-response cosine 0.99981–0.99992 | vs analytic aK·αᵈ |
| 9 | SAE quality | L0 12–56, var explained 0.98–1.00, loss recovered 0.97–1.00 |  |
| 10 | SAE vs continuous | belief R² falls -0.197 to -0.003 as L0 is driven down | at matched sparsity SAE loses to PCA on 1/5 |
| 11 | Seed robustness | 2/5, 3/5, 4/5, 5/5; circuit overlap 0.51–1.00 | leave-one-out over 5 seeds; seeds learn different circuits |
| 12 | Process coverage | 15 processes | plus 3 process families |
| 13 | Capacity threshold | mess3_hard: d_model=4; sns: d_model=4; random_non: none in range | fallback characterised in §6 |

## 1. Distance to the analytic optimum

| process | unifilar | h (analytic) | oracle | optimality gap | finite-memory gap | best finite model | GRU gap | belief dim | params |
|---|---|---|---|---|---|---|---|---|---|
| golden_mean | yes | 0.4621 | 0.4621 | +0.00031 ± 0.00013 | -0.0000 | order 1 (3 contexts) | 0.0000 | 2 | 106562 |
| even | yes | 0.4621 | 0.4633 | +0.00023 ± 0.00009 | 0.0331 | order 5 (243 contexts) | 0.0001 | 2 | 106562 |
| zero_one_random | yes | 0.2310 | 0.2311 | +0.00012 ± 0.00007 | 0.0000 | order 5 (243 contexts) | 0.0000 | 3 | 106562 |
| rrxor | yes | 0.4621 | 0.4626 | +0.02262 ± 0.00075 | 0.1032 | order 5 (243 contexts) | 0.0002 | 5 | 106562 |
| mess3 | no | - | 1.0891 | +0.00953 ± 0.00074 | 0.0001 | order 3 (64 contexts) | 0.0056 | 3 | 806531 |
| mess3_hard | no | - | 0.7916 | +0.00558 ± 0.00053 | 0.0015 | order 4 (256 contexts) | 0.0033 | 3 | 806531 |
| sns | no | 0.4847 | 0.4852 | +0.00117 ± 0.00024 | -0.0000 | order 4 (81 contexts) | 0.0000 | 2 | 106562 |
| persistent_bit | yes | 0.6474 | 0.6500 | +0.00250 ± 0.00043 | 0.0236 | order 5 (243 contexts) | -0.0000 | 2 | 106562 |
| drifting_coin | no | - | 0.5808 | +0.00260 ± 0.00036 | 0.0255 | order 5 (243 contexts) | 0.0005 | 2 | 345410 |
| random_uni_fast | yes | 0.8416 | 0.8442 | +0.00297 ± 0.00047 | 0.0027 | order 5 (1,024 contexts) | 0.0001 | 4 | 106691 |
| random_uni_slow | yes | 0.5618 | 0.5658 | +0.00817 ± 0.00094 | 0.0258 | order 5 (1,024 contexts) | 0.0006 | 4 | 345603 |
| random_non | no | - | 1.0925 | +0.00811 ± 0.00056 | 0.0001 | order 2 (16 contexts) | 0.0025 | 4 | 345603 |
| kalman | - | 0.9047 | 0.8997 | +0.00024 ± 0.00012 | 0.0000 | linear, window 8 | 0.0001 | 1 | 56194 |
| garch | - | - | 1.3652 | +0.00057 ± 0.00016 | 0.0053 | linear, window 8 | 0.0020 | 1 | 345410 |
| regimevol | - | - | 1.2070 | +0.00229 ± 0.00035 | 0.0187 | linear, window 8 | 0.0087 | 3 | 608905 |

- 14/15 processes within 0.01 nats/token of the analytic optimum; median gap **+0.00250**, worst **+0.02262**.
- Only 0/15 processes have a finite-memory baseline more than 0.15 nats above the optimum (none), and that is the expected answer rather than a disappointing one. For any *mixing* process the belief depends on the recent past with exponentially decaying weight, so an order-k Markov table is exponentially good in k — finite memory is nearly sufficient **by construction**. Finite memory only fails outright where the memory requirement is unbounded or exact: parity counting (`even`), phase alignment (`rrxor`), a latent that never resets (`persistent_bit`) or one that drifts slower than any practical window (`drifting_coin`).
- The interesting contrast is therefore not loss but **size**. Set the *belief dim* column against *best finite model*: the belief state is a handful of numbers, while the finite-memory model that nearly matches it needs a table of hundreds or thousands of contexts. A belief state buys compression of the computation, not always a better achievable loss — which is precisely why the mechanism, not the loss, is the thing worth reverse-engineering.
- A GRU trained identically reaches a median gap of **+0.00017**, so the recurrent architecture expresses these filters at least as well as the transformer does — as it should, since a filter *is* a recurrence.
- The transformer is materially worse on 6 of 15 processes: `rrxor` (+0.0226 vs +0.0002), `random_uni_slow` (+0.0082 vs +0.0006), `random_uni_fast` (+0.0030 vs +0.0001), `persistent_bit` (+0.0025 vs -0.0000), `drifting_coin` (+0.0026 vs +0.0005), `sns` (+0.0012 vs +0.0000). The general reason is structural rather than specific to any one of them — a recurrence carries the belief forward for free, whereas an attention-only model has no state and must reconstruct the whole posterior from the raw context at every position. The gap is widest on `rrxor` (137× the GRU's), which needs exact phase alignment within a repeating triple: a recurrence tracks that in effectively one unit. This is a fact about architecture, not about the task, and it is the kind of claim the analytic floor makes checkable — both numbers are distances from the *same* known optimum.


## 2. Where the belief lives, and whether it is used

| process | best probe R² | at | causal margin there | best causal margin | at | corr(R², margin) across layers |
|---|---|---|---|---|---|---|
| mess3 | 0.9060 | L0 mid | +0.109 | +0.457 | L0 pre | -0.399 |
| mess3_hard | 0.9808 | L3 post | +0.043 | +0.627 | L0 pre | -0.781 |
| even | 0.9966 | L1 post | +0.060 | +0.744 | L0 pre | -0.973 |
| rrxor | 0.5448 | L1 post | +0.370 | +0.370 | L1 post | 0.843 |
| sns | 0.9893 | L1 post | +0.272 | +0.272 | L1 post | 0.239 |
| drifting_coin | 0.9812 | L2 post | +0.049 | +0.171 | L0 mid | -0.081 |
| random_non | 0.7699 | L0 pre | -0.020 | +0.604 | L2 mid | -0.978 |
| kalman | 0.9998 | L0 post | +0.988 | +0.988 | L0 post | 0.999 |
| garch | 0.9909 | L2 post | +0.590 | +0.590 | L2 post | 0.925 |
| regimevol | 0.9751 | L2 post | +0.001 | +0.008 | L0 post | 0.900 |

**Box-counting dimension** of the reachable belief set versus the same statistic on the 2-D projection of the residual stream:

| process | true belief set | residual projection |
|---|---|---|
| mess3 | 1.444 | 1.726 |
| mess3_hard | 1.296 | 1.689 |
| rrxor | 0.253 | 1.618 |
| random_non | 1.169 | 0.891 |
| regimevol | 1.655 | 1.704 |

- *Causal margin* is the fraction of KL-to-the-target-belief closed by steering along the probe direction, **minus** the fraction a norm-matched random direction closes. The subtraction is essential: any large perturbation flattens the output, and a flat distribution has lower KL to the target than a confident wrong one, so the raw effect alone overstates. The control is floored at zero before subtracting — a random direction that makes things *worse* than baseline explains none of the effect, and letting it count would inflate the margin without bound on models confident enough that any perturbation is catastrophic. So the margin is at most 1. Steering strength is chosen on one half of the pairs and the margin reported on the other, with a bootstrap standard error over pairs.
- Probe R² and causal use **anti-correlate across layers** on 5/10 processes (mess3, mess3_hard, even, drifting_coin, random_non): the layers where the belief is most decodable are not the layers where perturbing it moves the output most. High R² is not evidence of use.


## 3. The circuit

| process | components | size to reach 0.90 | faithfulness at 25% | at 50% | faithfulness (belief-only) | seeds reproducing core | circuit overlap (Jaccard) | core components | attribution r | top-component ablation |
|---|---|---|---|---|---|---|---|---|---|---|
| even | 10 | 7.4 ± 1.1 / 10 | -4.522 | -1.638 | 0.988 | 3/5 | 0.67 | L0H1, L0H3, L0MLP, L1MLP | -0.70 | 4766× |
| rrxor | 10 | all / 10 | -1.795 | -1.387 | 0.017 | 5/5 | 1.00 | L0H0, L0H1, L0H2, L0H3, L0MLP, L1H0, L1H1, L1H2, L1H3, L1MLP | 0.29 | 23× |
| mess3_hard | 20 | 11.2 ± 0.8 / 20 | 0.628 | 0.880 | 0.969 | 2/5 | 0.51 | L0MLP, L1H1, L2MLP, L3MLP | 0.42 | 86× |
| sns | 10 | 6.0 ± 2.2 / 10 | -2.638 | -1.210 | 0.952 | 5/5 | 0.53 | L0H2, L0MLP, L1MLP | 0.88 | 96× |
| regimevol | 15 | 10.4 ± 0.5 / 15 | 0.458 | 0.784 | 0.942 | 4/5 | 0.66 | L0MLP, L1H3, L1MLP, L2H1, L2H2, L2MLP | 0.60 | 9× |

> **No sparse circuit exists for rrxor.** The greedy search only reaches the target by keeping every component, and keeping everything resamples nothing and scores 1.000 by construction. Read the 25% and 50% columns instead: they say how much of the computation a genuinely sparse subgraph recovers, and for these processes the answer is 'not much'. That is a result about the model — the computation is distributed — not a failure of the search.

- *Faithfulness* is measured against the best **context-free** predictor, not against a fully-scrubbed network: scrubbing every component drives these models off-distribution to a loss worse than predicting the marginal, which inflates any percentage computed against it. The classic number is in the JSON as `faithfulness_vs_full_scrub` and is always higher.
- *Belief-only* additionally resamples each circuit component from a different sequence with the same posterior, testing the semantic claim rather than mere sufficiency.
- *Attribution r* is the correlation between attribution patching and true patching. It collapses on processes whose beliefs are near-deterministic, because the intervention is then far too large for a first-order approximation.


## 4. The Kalman gain in the weights

| process | analytic filter | α = a(1−Kc) | kernel cosine vs analytic | decay-rate error | α error | β error | seeds cos>0.99 |
|---|---|---|---|---|---|---|---|
| kalman | a=0.950, K=0.4406 | 0.5314 | 0.99992 (min 0.99986) | 0.72% | 0.22% | 0.18% | 5/5 |
| kalman_slow | a=0.995, K=0.1276 | 0.8680 | 0.99981 (min 0.99968) | 0.08% | 0.05% | 0.28% | 5/5 |

- A transformer has no state to carry between positions, so it cannot implement the Kalman *recursion*; it implements the unrolled convolution `m_t = aK Σ_d α^d y_{t−1−d}`. The object that matches the analytic gain is therefore the model's impulse response `∂m̂_t/∂y_{t−1−d}`, measured by autograd — not a head's belief-to-belief transport, which reads near zero even in models whose kernel matches to five decimals.
- Measured on models trained with `norm="none"`, so weight-level coefficients are comparable to analytic ones without a LayerNorm rescaling in the way.


## 5. Sparse autoencoders on a dense variable

| process | L0 | variance explained | loss recovered | k | SAE R² | PCA R² | best k-dim R² | SAE − PCA |
|---|---|---|---|---|---|---|---|---|
| mess3_hard | 12.0 | 0.976 | 1.001 | 8 | 0.871 | 0.955 | 0.981 | -0.084 |
| mess3 | 56.5 | 0.992 | 0.992 | 8 | 0.814 | 0.730 | 0.907 | +0.084 |
| sns | 13.7 | 0.999 | 1.002 | 8 | 0.987 | 0.982 | 0.989 | +0.005 |
| random_non | 18.3 | 1.000 | n/a | 8 | 0.769 | 0.724 | 0.773 | +0.045 |
| regimevol | 53.4 | 0.997 | 1.001 | 8 | 0.932 | 0.930 | 0.976 | +0.002 |

- The SAE is given the friendlier half of the comparison: its dictionary is learned without ever seeing the belief, but the k features it is scored on are chosen *using* the belief by greedy forward selection. PCA gets no such help.
- At the healthy operating point it loses to k unsupervised principal components on 1/5 processes (median SAE − PCA = **+0.005**). Taken alone that is a mixed result, and the sparsity sweep below is the sharper version of it.

**What sparsity costs.** The same dictionary, swept across L1 strengths, scored on the same belief (k = 8 features):

| process | belief dim | L0 (densest) | SAE R² there | L0 (sparsest) | SAE R² there | cost of sparsity | PCA R² (no sparsity knob) |
|---|---|---|---|---|---|---|---|
| mess3_hard | 3 | 157 | 0.950 | 3.2 | 0.886 | -0.064 | 0.955 |
| mess3 | 3 | 214 | 0.821 | 9.2 | 0.625 | -0.197 | 0.730 |
| sns | 2 | 36 | 0.988 | 1.5 | 0.973 | -0.015 | 0.982 |
| random_non | 4 | 82 | 0.765 | 2.7 | 0.762 | -0.003 | 0.724 |
| regimevol | 3 | 116 | 0.942 | 2.6 | 0.878 | -0.063 | 0.930 |

- Belief capture falls as the dictionary is forced to be sparser on 3/5 processes, while PCA — which has no sparsity knob — is a constant. That is the finding the analytic ground truth makes available: the loss is not a defect in these autoencoders (variance explained stays above 0.9 and downstream loss recovered stays near 1.0 throughout), it is the **sparsity prior itself** paying a price on a variable that is known to be dense. A belief state is a point on a simplex: continuous, and with every coordinate non-zero almost always — exactly what a sparse basis is built not to represent.
- The cost tracks how much belief structure there is *to* lose, not the nominal belief dimension. It is largest on `mess3` and `mess3_hard`, where the model represents the posterior well and it is genuinely two-dimensional. It is near zero on `sns`, whose belief is effectively one-dimensional so a single feature carries it, and on `random_non`, where the model's own representation is poor to begin with (the best k-dimensional subspace only reaches 0.773) and a sparse basis therefore gives up little.


## 6. What replaces exact inference when capacity runs out

| process | threshold d_model | eff. order (below) | (above) | bigram residual KL (below) | (above) | order-5 residual KL (below) | (above) | clusters for 95% (below) | (above) | eff. rank (below) | (above) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mess3_hard | 4 | 5.0 | 5.0 | 0.0204 | 0.0651 | 0.0069 | 0.0011 | 6.0 | 6.0 | 1.99 | 2.00 |
| sns | 4 | 1.0 | 2.0 | 0.0012 | 0.0263 | 0.0009 | 0.0007 | 2.0 | 3.0 | 1.00 | 1.00 |
| random_non | - | - | - | - | - | - | - | - | - | - | - |

> **Every model here gets the same step budget**, so larger ones are further from convergence and their gaps are training-limited rather than capacity-limited. That is why the gap *rises* with width above the threshold. Read a capacity threshold as "the width below which this budget cannot reach the optimum", not as an architectural limit — and read a smallest-sufficient configuration as a statement about sample efficiency rather than about representational capacity.

- The *residual KL* columns say how well the model's own predictive distributions are reproduced by a table indexed on the last k observations. They matter because the effective-order integer alone is ambiguous: a model that genuinely needs a long context and a model that *no* finite table describes both land on the largest order tried. A small order-5 residual means the model really is a finite-memory approximation; a large one means the degradation is unstructured error rather than any of the three hypothesised fallbacks.

Threshold is the largest `d_model` whose optimality gap still exceeds 0.02 nats/token. What replaces exact belief tracking below it:

- **mess3_hard**: collapse toward a short-context table (a bigram explains the model's own predictions to within 0.0204 nats below threshold vs 0.0651 above)
- **sns**: finite-memory truncation (effective Markov order 1 below threshold vs 2 above); belief quantisation (2 clusters explain 95% below threshold vs 3 above); collapse toward a short-context table (a bigram explains the model's own predictions to within 0.0012 nats below threshold vs 0.0263 above)
- **random_non**: no capacity threshold within the swept range


## 7. Inference over a family of processes

| family | gap to optimal | in-context gain (model) | (optimal) | ratio | component probe R² | component agreement | held-out gap | beats best fixed filter by |
|---|---|---|---|---|---|---|---|---|
| even_family | +0.00069 | 0.1881 | 0.0867 | 2.170 | 0.975 | 0.998 | 0.0287 | 0.0876 |
| golden_family | +0.00167 | 0.1093 | 0.0647 | 1.689 | 0.953 | 0.992 | 0.0250 | 0.1334 |
| mess3_family | +0.00304 | 0.0498 | 0.0331 | 1.504 | 0.908 | 0.984 | 0.0169 | 0.0435 |

- *In-context gain* is the fall in loss from early positions to late ones. A model that memorised a single averaged filter would show a gain near zero.
- *Held-out gap* is measured on process parameters that were never in the training family, against those processes' own exact optimal predictors.


## 8. Depth, width and context length

**Optimality gap by depth and width** (nats/token):

| depth / $d_{model}$ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 1 layer | +0.0008 | +0.0017 | +0.0017 | +0.0031 |
| 2 layers | +0.0012 | +0.0013 | +0.0031 | +0.0070 |
| 4 layers | +0.0011 | +0.0020 | +0.0027 | +0.0056 |

> **Every model here gets the same step budget**, so larger ones are further from convergence and their gaps are training-limited rather than capacity-limited. That is why the gap *rises* with width above the threshold. Read a capacity threshold as "the width below which this budget cannot reach the optimum", not as an architectural limit — and read a smallest-sufficient configuration as a statement about sample efficiency rather than about representational capacity.


Smallest configuration within 0.01 nats of the optimum: **1 layer(s), $d_{model}$ = 16, 4,963 parameters** (gap +0.00083).

**Context length.** For a mixing process the optimal loss plateaus at the entropy rate and extra context is worthless past the mixing time. For a non-mixing one the belief never forgets, only sharpens, so the optimal loss keeps falling with position; the ratio column is how much of that fall the model reproduces.

| process | regime | optimal gain from context | model gain | ratio | gap |
|---|---|---|---|---|---|
| persistent_bit | non-mixing | 0.0242 | 0.0376 | 1.549 | 0.00250 |
| drifting_coin | non-mixing | 0.0334 | 0.0614 | 1.839 | 0.00260 |
| even | non-mixing | 0.0547 | 0.6069 | 11.103 | 0.00023 |


## Figures

![exp01_optimality_gap](figures/exp01_optimality_gap.png)
![exp02_geometry_drifting_coin](figures/exp02_geometry_drifting_coin.png)
![exp02_geometry_even](figures/exp02_geometry_even.png)
![exp02_geometry_garch](figures/exp02_geometry_garch.png)
![exp02_geometry_kalman](figures/exp02_geometry_kalman.png)
![exp02_geometry_mess3](figures/exp02_geometry_mess3.png)
![exp02_geometry_mess3_hard](figures/exp02_geometry_mess3_hard.png)
![exp02_geometry_random_non](figures/exp02_geometry_random_non.png)
![exp02_geometry_regimevol](figures/exp02_geometry_regimevol.png)
![exp02_geometry_rrxor](figures/exp02_geometry_rrxor.png)
![exp02_geometry_sns](figures/exp02_geometry_sns.png)
![exp04_kalman_kernel](figures/exp04_kalman_kernel.png)
![exp05_sae_capture](figures/exp05_sae_capture.png)
![exp06_capacity](figures/exp06_capacity.png)
![exp07_incontext](figures/exp07_incontext.png)
![exp08_architecture](figures/exp08_architecture.png)
