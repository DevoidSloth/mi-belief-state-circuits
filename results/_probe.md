# Results

Every number below is read from `results/*.json`, which the experiments write. Rebuild with `bsc report`.

> **Partial suite.** Not yet run: circuits, kalman, sae, capacity, incontext, architecture. Run `bsc run all` to fill these in.

Environment: NVIDIA GeForce RTX 4080 Laptop GPU, torch 2.9.1+cu126, python 3.14.0, commit `0385121`.

## Headline numbers

| # | metric | measured | note |
|---|---|---|---|
| 1 | Optimality gap | median +0.00250, worst +0.02262 nats/token | 14/15 under 0.01 |
| 2 | Baseline separation | median 0.0027 nats | 0 processes over 0.15 |
| 3 | Belief-state probe fidelity | median best R² 0.9810 (min 0.5448) | held-out sequences, split by sequence |
| 4 | Causal validity of the probe | median steering margin +0.608 | over a norm-matched random direction |
| 12 | Process coverage | 15 processes | plus 3 process families |

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
| mess3 | 0.9060 | L0 mid | +0.154 | +0.748 | L0 pre | -0.679 |
| mess3_hard | 0.9808 | L3 post | +0.043 | +0.627 | L0 pre | -0.781 |
| even | 0.9966 | L1 post | +0.060 | +0.744 | L0 pre | -0.973 |
| rrxor | 0.5448 | L1 post | +0.370 | +0.370 | L1 post | 0.843 |
| sns | 0.9893 | L1 post | +0.272 | +0.272 | L1 post | 0.239 |
| drifting_coin | 0.9812 | L2 post | +0.049 | +0.171 | L0 mid | -0.081 |
| random_non | 0.7699 | L0 pre | -0.025 | +0.656 | L2 mid | -0.972 |
| kalman | 0.9998 | L0 post | +471.173 | +26681.794 | L0 pre | -1.000 |
| garch | 0.9909 | L2 post | +0.590 | +0.590 | L2 post | 0.912 |
| regimevol | 0.9751 | L2 post | +0.001 | +0.013 | L1 mid | 0.891 |

**Box-counting dimension** of the reachable belief set versus the same statistic on the 2-D projection of the residual stream:

| process | true belief set | residual projection |
|---|---|---|
| mess3 | 1.444 | 1.726 |
| mess3_hard | 1.296 | 1.689 |
| sns | 0.159 | 1.451 |
| random_non | 1.213 | 0.891 |
| regimevol | 1.655 | 1.704 |

- *Causal margin* is the fraction of KL-to-the-target-belief closed by steering along the probe direction, **minus** the fraction closed by a norm-matched random direction. The subtraction is essential: any large perturbation flattens the output, and a flat distribution has lower KL to the target than a confident wrong one, so the raw effect alone overstates. A margin above 1 means the random control made things *worse* while steering made them better. Steering strength is chosen on one half of the pairs and the margin reported on the other.
- Probe R² and causal use **anti-correlate across layers** on 6/10 processes (mess3, mess3_hard, even, drifting_coin, random_non, kalman): the layers where the belief is most decodable are not the layers where perturbing it moves the output most. High R² is not evidence of use.


## 3. The circuit

_not run_


## 4. The Kalman gain in the weights

_not run_


## 5. Sparse autoencoders on a dense variable

_not run_


## 6. What replaces exact inference when capacity runs out

_not run_


## 7. Inference over a family of processes

_not run_


## 8. Depth, width and context length

_not run_


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
