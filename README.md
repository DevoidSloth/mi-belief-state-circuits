# mi-belief-state-circuits

**Does a transformer learn to be a Bayesian filter?**

Train small transformers on stochastic processes whose optimal predictor is
analytically known, then reverse-engineer the weights to find out whether the
model implements the Bayes-optimal filter — where the belief state lives
geometrically, which components compute the update, and what the model does
instead when exact inference does not fit.

The point of choosing these processes is that **the right answer is known in
advance**. For a hidden Markov model the optimal next-token predictor is the
forward algorithm: carry a posterior over hidden states, update it
multiplicatively on each observation, renormalise. So "did the model learn the
right algorithm?" stops being a story and becomes a measurement — the model's
loss minus the analytic entropy rate, the probe's R² against the *exact*
posterior, the learned OV circuit against the *analytic* Kalman gain.

## Quick start

Requires Python 3.10+ and PyTorch 2.2+. A GPU helps for `bsc run all` but is not
required; everything else runs comfortably on a CPU.

```bash
pip install -e ".[dev]"
bsc processes          # the zoo, with its analytic facts
bsc train mess3_hard   # train one model, report its distance from the optimum
bsc run all            # the whole experiment suite -> results/RESULTS.md
pytest                 # 105 tests, mostly ground-truth checks
```

Measured results live in [`results/RESULTS.md`](results/RESULTS.md), rebuilt from
the JSON the experiments write. Nothing in that file is typed by hand.

**Contents:** [What is verified](#what-is-actually-verified) ·
[The zoo](#the-zoo) · [Design decisions](#design-decisions-worth-knowing-about) ·
[The traps](#the-traps-and-what-was-done-about-them) ·
[First-attempt bugs](#two-things-that-were-wrong-on-the-first-attempt) ·
[Layout](#layout) · [Reading](#reading)

---

## What is actually verified

Everything rests on the claim that the "optimal predictor" really is optimal, so
that claim is tested rather than asserted:

| check | how |
|---|---|
| forward algorithm | chain-rule probabilities vs exhaustive enumeration over all length-6 sequences, to 1e-10 |
| entropy rate | unifilar closed form vs the exact filter's realised loss, within 4 standard errors |
| mixed-state presentation | independent enumeration of the reachable belief set, cross-checked against both |
| Kalman filter | recursion vs direct conditioning on the joint Gaussian covariance, to 1e-8 |
| GARCH / regime-switching | variance recursion and mixture likelihood recomputed by hand |
| model causality | position *t*'s output is unchanged by any perturbation of observations after *t* |

The last one matters more than it looks: a lookahead leak shows up as a
*suspiciously small* optimality gap, not as an error.

## The zoo

15 processes plus 3 process families, chosen to span the axes that change the
answer rather than to pad a count:

- **Unifilar epsilon-machines** (golden mean, even process, zero-one-random,
  RRXOR). The belief collapses onto a vertex of the simplex, so the model only
  has to track a discrete state.
- **Non-unifilar** (Mess3, the simple non-unifilar source, random HMMs). The
  belief stays in the interior of the simplex forever, which is where the
  interesting geometry is — for Mess3 the reachable set is a fractal.
- **Memory and mixing controls** (`persistent_bit` never forgets;
  `drifting_coin` has a very slow latent; random HMMs are rejection-sampled into
  target mixing-time bands via the SLEM of their transition matrix).
- **Continuous latents** (Kalman, GARCH, regime-switching volatility), where the
  latent is a real number or a variance rather than a label. All three are
  parameterised so the observation has unit unconditional variance, which makes
  losses comparable across processes with no rescaling term.
- **Families** (`even_family`, `golden_family`, `mess3_family`), where one
  component is drawn per sequence. The Bayes-optimal predictor over a *finite*
  family is exactly computable, so in-context inference gets ground truth too.

Each process ships a `filter()` returning the exact posterior at every position,
the exact predictive distribution, and the exact log-likelihood.

## Design decisions worth knowing about

**Paired optimality gaps.** Model loss and oracle loss are computed on the *same
sequences*, position by position. The sampling noise that dominates either loss
cancels, so gaps of ~1e-3 nats are resolvable with a few hundred sequences.

**Heads that can represent the optimum exactly.** Categorical for HMMs, Gaussian
for Kalman and GARCH, Gaussian *mixture* for regime-switching volatility — where
the optimal predictive density is a K-component mixture whose weights are the
regime posterior. The model therefore has no representational excuse, and the
gap is a clean non-negative number.

**A BOS prefix.** Input position *t* sees `x[t-1]` and predicts `x[t]`, so the
residual stream at position *t* must carry `belief[t]` exactly. Off-by-one
errors here are silent and would corrupt every probe and patch, so the
convention is stated once in `processes/base.py` and enforced by tests.

**Mess3 is used in two parameterisations.** The canonical `mess3(0.15, 0.6)` from
the belief-state-geometry literature has only ~0.009 nats of total
predictability — an n-gram is within 0.009 nats of optimal, so it cannot support
a circuit claim. It is kept for reproducing the fractal; `mess3_hard(0.05, 0.85)`
has the same structure with ~0.3 nats of signal and carries the quantitative
claims. Baseline separation is reported for every process so this is visible
rather than hidden.

**A self-contained hooked transformer.** TransformerLens-compatible hook names
(`blocks.0.attn.hook_z`, `blocks.1.hook_resid_post`, …) over ~150 lines with no
external dependency, and per-head `W_V`/`W_O` so a head's OV circuit is exactly
`W_V[h] @ W_O[h]`. The Kalman experiment cannot be expressed if QKV are fused.

## The traps, and what was done about them

The brief this was built from lists the ways interpretability results go wrong.
Each is handled explicitly, and the handling is visible in the numbers:

**"Probing accuracy is not evidence of use."** Every probe is paired with an
intervention: invert the probe to a minimum-norm write direction, steer the
residual stream toward another *reachable* belief, and check whether the output
moves to the distribution that belief implies. Two confounds are controlled — a
norm-matched **random direction**, because any large perturbation flattens the
output and a flatter distribution has lower KL to the target than a confident
wrong one; and **held-out strength selection**, so the headline margin is not a
maximum over choices. Across layers, R² and causal margin frequently
*anti-correlate*.

**"Reading a circuit off one seed."** Circuit search is run across 5 seeds per
process and reported as a distribution: mean size, always-present components,
and how many seeds reproduce the core.

**"A task so simple the optimal predictor is a bigram table."** An n-gram
baseline (order and smoothing tuned on validation) and a linear finite-window
baseline are trained for every process, and their gap to the optimum is printed
next to the model's. Processes that fail the bar are named, not dropped.

**"SAE reconstruction fine, features uninterpretable."** L0, variance explained
*and* downstream loss recovered are all reported, at four L1 strengths.

**"Over-claiming Bayesian inference."** The defensible claim is the quantified
one, and the denominator is chosen to be conservative: faithfulness is measured
against the best **context-free** predictor rather than a fully-scrubbed
network, because scrubbing everything drives these models off-distribution to a
loss *worse than the marginal*, which inflates any percentage computed against
it. The conventional number is reported alongside and is always larger.

## Two things that were wrong on the first attempt

Both were found by tests, and both would have produced confident nonsense:

**Contrast pairs matched the token being predicted.** Pairing sequences by their
last *k* observations *including* position *t* conditions every pair on its own
outcome. For a process whose belief pins the next symbol, one member of each
pair is then always the confident-and-correct one, and every single-component
patching effect came out *negative*. Restricting the matched window to context
flipped the top effect from −0.04 to **+1.89 nats**.

**Attribution patching used the gradient of a mean loss**, scaling every
attribution down by the batch size and making a working method look broken.

And one thing that was not a bug but a wrong expectation: reading a head's
belief-to-belief transport `vᵀ(W_V W_O)u` finds *nothing* on the Kalman task,
even in models whose filter matches the analytic one to four decimal places. A
transformer has no state to carry between positions, so it cannot implement the
recursion `m_{t+1} = αm_t + aKy_t`; it implements the unrolled convolution
`m_t = aK Σ_d α^d y_{t−1−d}`. The object that matches the analytic gain is the
model's **impulse response** `∂m̂_t/∂y_{t−1−d}`, measured by autograd.

## Layout

```
src/bsc/
  processes/    the zoo + exact filters (hmm, continuous, mixture, registry)
  model/        hooked transformer, output heads, baselines
  train/        training loop, data, paired evaluation against the optimum
  analysis/     probes, geometry, patching, scrubbing, sae, kalman_gain
  experiments/  seven experiments, each writing one JSON into results/
  report.py     rebuilds RESULTS.md from those JSONs
tests/          ground-truth and machinery tests
```

## Reading

Shai, Riechers, Teixeira, Oldenziel & Marzen, *Transformers Represent Belief
State Geometry in their Forward Pass* (2024) · Nanda et al., *Progress Measures
for Grokking* (2023) · Wang et al., *Interpretability in the Wild* (2022) · Chan
et al., *Causal Scrubbing* (2022) · Bricken et al., *Towards Monosemanticity*
(2023) · Crutchfield on computational mechanics and epsilon-machines.
