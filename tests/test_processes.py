"""Validate every optimal filter against an independent brute-force computation.

The entire project rests on the claim that the "optimal predictor" really is
optimal.  These tests establish that by computing the same quantity a second way
(exhaustive enumeration for HMMs, joint-Gaussian conditioning for the Kalman
model) and demanding agreement to near machine precision.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from bsc.processes import DEFAULT_ZOO, HMM, GarchProcess, KalmanProcess, get_process
from bsc.processes.continuous import RegimeSwitchingVol
from bsc.processes.zoo import even_process, mess3, rrxor, simple_nonunifilar, zero_one_random

DISCRETE = ["golden_mean", "even", "zero_one_random", "rrxor", "mess3", "sns", "random_non"]


@pytest.mark.parametrize("name", DEFAULT_ZOO)
def test_process_constructs_and_samples(name):
    p = get_process(name)
    rng = np.random.default_rng(0)
    obs = p.sample(8, 32, rng)
    assert obs.shape == (8, 32)
    res = p.filter(obs)
    assert res.belief.shape == (8, 33, p.d_belief)
    assert res.logp.shape == (8, 32)
    assert np.isfinite(res.logp).all()


@pytest.mark.parametrize("name", DEFAULT_ZOO)
def test_belief_is_a_valid_distribution(name):
    p = get_process(name)
    if p.belief_kind != "simplex":
        pytest.skip("belief is not a simplex for this process")
    rng = np.random.default_rng(1)
    res = p.filter(p.sample(6, 40, rng))
    assert (res.belief >= -1e-12).all()
    np.testing.assert_allclose(res.belief.sum(axis=-1), 1.0, atol=1e-10)


@pytest.mark.parametrize("name", DISCRETE)
def test_forward_algorithm_matches_exhaustive_enumeration(name):
    """P(x_0..x_{L-1}) from the chain rule must equal pi T[x_0] ... T[x_{L-1}] 1."""
    hmm = get_process(name)
    assert isinstance(hmm, HMM)
    L = 6
    seqs = np.array(list(itertools.product(range(hmm.n_symbols), repeat=L)), dtype=np.int64)
    res = hmm.filter(seqs)
    chain_rule = res.logp.sum(axis=1)

    exact = np.empty(len(seqs))
    for i, seq in enumerate(seqs):
        v = hmm.initial
        for x in seq:
            v = v @ hmm.T[x]
        exact[i] = np.log(v.sum()) if v.sum() > 0 else -np.inf

    finite = np.isfinite(exact)
    np.testing.assert_allclose(chain_rule[finite], exact[finite], atol=1e-10)
    # The predictive distribution must be a proper distribution over the alphabet
    # and the joint must sum to one over all sequences.
    assert np.isclose(np.exp(exact[finite]).sum(), 1.0, atol=1e-10)


@pytest.mark.parametrize("name", DISCRETE)
def test_predictive_pmf_matches_logp(name):
    hmm = get_process(name)
    rng = np.random.default_rng(2)
    obs = hmm.sample(4, 25, rng)
    res = hmm.filter(obs)
    pmf = res.predictive["pmf"]
    np.testing.assert_allclose(pmf.sum(axis=-1), 1.0, atol=1e-12)
    picked = np.take_along_axis(pmf, obs[:, :, None], axis=2)[:, :, 0]
    np.testing.assert_allclose(np.log(picked), res.logp, atol=1e-12)


@pytest.mark.parametrize(
    "hmm",
    [even_process(0.4), zero_one_random(), rrxor(0.5, 0.5), get_process("random_uni_fast")],
    ids=["even", "zor", "rrxor", "random_uni"],
)
def test_closed_form_entropy_rate_matches_monte_carlo(hmm):
    """Unifilar closed form must agree with the exact filter's realised loss."""
    h = hmm.entropy_rate()
    assert h is not None
    mc, se = hmm.optimal_loss_mc(n_seq=600, length=400, seed=3, burn_in=100)
    assert abs(mc - h) < 4 * se + 1e-3


@pytest.mark.parametrize(
    "hmm",
    [even_process(0.4), zero_one_random(), simple_nonunifilar(0.6, 0.3)],
    ids=["even", "zor", "sns"],
)
def test_mixed_state_presentation_matches_monte_carlo(hmm):
    h = hmm.entropy_rate_msp()
    assert h is not None, "MSP should close for these processes"
    mc, se = hmm.optimal_loss_mc(n_seq=600, length=400, seed=4, burn_in=100)
    assert abs(mc - h) < 4 * se + 1e-3


def test_mess3_is_non_unifilar_and_msp_does_not_close():
    p = mess3(0.15, 0.6)
    assert not p.is_unifilar()
    assert p.entropy_rate() is None
    # the mixed-state set of Mess3 is a fractal, so enumeration must give up
    assert p.entropy_rate_msp(max_states=5000) is None


def test_sampled_symbol_frequencies_match_theory():
    hmm = get_process("random_non")
    rng = np.random.default_rng(5)
    obs = hmm.sample(400, 400, rng)
    from bsc.processes import stationary_distribution

    pi = stationary_distribution(hmm.A)
    theory = hmm.emission @ pi
    empirical = np.bincount(obs.ravel(), minlength=hmm.n_symbols) / obs.size
    np.testing.assert_allclose(empirical, theory, atol=0.01)


def test_unifilar_detection():
    assert even_process(0.5).is_unifilar()
    assert rrxor().is_unifilar()
    assert not mess3().is_unifilar()
    assert not simple_nonunifilar().is_unifilar()


def test_hmm_rejects_invalid_transition_tensor():
    bad = np.zeros((2, 2, 2))
    bad[0, 0, 0] = 0.5  # row 1 sums to zero
    with pytest.raises(ValueError):
        HMM(bad)


# ---------------------------------------------------------------------------
# continuous processes
# ---------------------------------------------------------------------------


def test_kalman_matches_joint_gaussian_conditioning():
    """Compare the recursion against direct conditioning on the joint covariance."""
    p = KalmanProcess(a=0.9, c=1.3, snr=3.0)
    L = 12
    rng = np.random.default_rng(6)
    obs = p.sample(5, L, rng)
    res = p.filter(obs)

    var_x = p.q / (1.0 - p.a**2)
    idx = np.arange(L)
    cov_y = p.c**2 * var_x * p.a ** np.abs(idx[:, None] - idx[None, :]) + p.r * np.eye(L)
    # belief[:, t+1] is E[xi_{t+1} | y_<=t], and xi_{t+1} is one latent step ahead
    # of the latent that produced y_t -- hence the +1 in the exponent.
    cov_xy = p.c * var_x * p.a ** np.abs((idx[:, None] + 1) - idx[None, :])

    for t in range(1, L):
        past = slice(0, t)
        Cpp = cov_y[past, past]
        inv = np.linalg.inv(Cpp)
        # one-step predictive distribution of y_t
        cross = cov_y[t, past]
        mean = obs[:, past] @ (inv @ cross)
        var = cov_y[t, t] - cross @ inv @ cross
        np.testing.assert_allclose(res.predictive["mean"][:, t], mean, atol=1e-8)
        np.testing.assert_allclose(res.predictive["var"][:, t], var, atol=1e-8)
        # filtered latent mean E[x_{t+1} | y_0..y_t]
        upto = slice(0, t + 1)
        inv2 = np.linalg.inv(cov_y[upto, upto])
        cross2 = cov_xy[t, upto]
        np.testing.assert_allclose(
            res.belief[:, t + 1, 0], obs[:, upto] @ (inv2 @ cross2), atol=1e-8
        )


def test_kalman_steady_state_solves_riccati():
    p = KalmanProcess(a=0.93, c=0.8, snr=5.0)
    ss = p.steady_state()
    lhs = ss["P_prior"]
    rhs = p.a**2 * (1.0 - ss["K"] * p.c) * ss["P_prior"] + p.q
    assert abs(lhs - rhs) < 1e-12
    assert abs(ss["K"] - ss["P_prior"] * p.c / ss["S"]) < 1e-12


def test_kalman_unconditional_variance_is_one():
    for snr in (1.0, 4.0, 10.0):
        p = KalmanProcess(a=0.95, snr=snr)
        rng = np.random.default_rng(7)
        y = p.sample(200, 400, rng)
        assert abs(y.var() - 1.0) < 0.05


def test_kalman_entropy_rate_matches_realised_loss():
    p = KalmanProcess(a=0.95, snr=4.0)
    h = p.entropy_rate()
    mc, se = p.optimal_loss_mc(n_seq=800, length=400, seed=8, burn_in=50)
    assert abs(mc - h) < 4 * se


def test_garch_filter_reproduces_the_variance_recursion():
    p = GarchProcess(alpha=0.12, beta=0.80)
    rng = np.random.default_rng(9)
    obs = p.sample(4, 30, rng)
    res = p.filter(obs)
    sigma2 = np.exp(res.belief[:, :, 0])
    for t in range(29):
        expected = p.omega + p.alpha * obs[:, t] ** 2 + p.beta * sigma2[:, t]
        np.testing.assert_allclose(sigma2[:, t + 1], expected, rtol=1e-12)
    manual = -0.5 * (np.log(2 * np.pi) + np.log(sigma2[:, :-1]) + obs**2 / sigma2[:, :-1])
    np.testing.assert_allclose(res.logp, manual, atol=1e-12)


def test_garch_unconditional_variance_is_one():
    p = GarchProcess(alpha=0.10, beta=0.85)
    rng = np.random.default_rng(10)
    y = p.sample(400, 600, rng)
    assert abs(y.var() - 1.0) < 0.1


def test_regime_switching_matches_dense_forward_algorithm():
    p = RegimeSwitchingVol((0.5, 1.0, 2.5), stay=0.95)
    rng = np.random.default_rng(11)
    obs = p.sample(3, 20, rng)
    res = p.filter(obs)

    for n in range(obs.shape[0]):
        eta = p.pi.copy()
        for t in range(obs.shape[1]):
            lik = np.exp(-0.5 * (np.log(2 * np.pi * p.sigma2) + obs[n, t] ** 2 / p.sigma2))
            joint = eta * lik
            assert abs(np.log(joint.sum()) - res.logp[n, t]) < 1e-10
            np.testing.assert_allclose(res.belief[n, t], eta, atol=1e-12)
            eta = (joint / joint.sum()) @ p.P


def test_regime_switching_unconditional_variance_is_one():
    p = RegimeSwitchingVol((0.5, 1.0, 2.0), stay=0.98)
    rng = np.random.default_rng(12)
    y = p.sample(400, 500, rng)
    assert abs(y.var() - 1.0) < 0.1


def test_target_belief_alignment():
    """target_belief()[:, t] must be the belief needed to predict obs[:, t+1]."""
    p = get_process("even")
    rng = np.random.default_rng(13)
    obs = p.sample(3, 12, rng)
    res = p.filter(obs)
    tgt = res.target_belief()
    np.testing.assert_allclose(tgt[:, 0], res.belief[:, 1])
    # predicting obs[:, t+1] from target[:, t] must reproduce logp[:, t+1]
    pmf = np.einsum("nd,xdj->nx", tgt[:, 3], p.T)
    picked = pmf[np.arange(3), obs[:, 4]]
    np.testing.assert_allclose(np.log(picked), res.logp[:, 4], atol=1e-10)
