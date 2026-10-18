"""Continuous-observation processes: Kalman, GARCH, regime-switching volatility.

These are the finance-flavoured half of the zoo and they carry the two cases the
discrete processes cannot express:

* a **continuous latent** that the optimal filter tracks with a linear recursion
  (Kalman), so the learned OV circuit can be compared against an analytic gain;
* a latent that is a **variance** rather than a label (GARCH, regime-switching).

Every process is parameterised so that the unconditional variance of the
observation is exactly 1.  That means model loss and oracle loss are directly
comparable in nats with no rescaling term, and losses are comparable across
processes.
"""

from __future__ import annotations

import numpy as np

from .base import FilterResult, Process

LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = 1e-300


def _gauss_logpdf(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (LOG_2PI + np.log(var) + (y - mean) ** 2 / var)


class KalmanProcess(Process):
    """Scalar linear-Gaussian state space model.

        x_t = a x_{t-1} + w_t,   w ~ N(0, q)
        y_t = c x_t + v_t,       v ~ N(0, r)

    The optimal filter is the Kalman recursion, which after a few steps settles
    into the fixed point of the Riccati equation.  In steady state the belief
    mean obeys the linear recursion

        m_{t+1} = a (1 - K c) m_t + a K y_t

    and those two scalars are the ground truth the OV-circuit experiment targets.
    """

    kind = "continuous"
    belief_kind = "gaussian_mean"
    head = "gaussian"
    d_belief = 1

    def __init__(self, a: float = 0.95, c: float = 1.0, snr: float = 4.0, name: str | None = None):
        if not 0.0 <= a < 1.0:
            raise ValueError("a must be in [0, 1) for a stationary latent")
        self.a, self.c = float(a), float(c)
        # Var(y) = c^2 * Var(x) + r = 1 with Var(x) = q / (1 - a^2) and SNR = c^2 Var(x) / r.
        self.r = 1.0 / (1.0 + snr)
        var_x = snr * self.r / (c**2)
        self.q = var_x * (1.0 - a**2)
        self.snr = float(snr)
        self.name = name or f"kalman_a{a:g}_snr{snr:g}"

    # -- steady-state quantities used as analytic ground truth -------------

    def steady_state(self) -> dict[str, float]:
        """Fixed point of the Riccati recursion, plus the induced linear coefficients."""
        p = self.q / (1.0 - self.a**2)
        for _ in range(10000):
            s = self.c**2 * p + self.r
            k = p * self.c / s
            p_post = (1.0 - k * self.c) * p
            p_new = self.a**2 * p_post + self.q
            if abs(p_new - p) < 1e-15:
                p = p_new
                break
            p = p_new
        s = self.c**2 * p + self.r
        k = p * self.c / s
        return {
            "P_prior": p,
            "S": s,
            "K": k,
            "recur_prev": self.a * (1.0 - k * self.c),  # coefficient on m_t
            "recur_obs": self.a * k,                    # coefficient on y_t
        }

    def entropy_rate(self) -> float | None:
        """Differential entropy of the steady-state one-step predictive density."""
        return 0.5 * (LOG_2PI + 1.0 + np.log(self.steady_state()["S"]))

    # -- sampling and filtering -------------------------------------------

    def sample(self, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        var_x = self.q / (1.0 - self.a**2)
        x = rng.normal(0.0, np.sqrt(var_x), size=n_seq)
        y = np.empty((n_seq, length))
        for t in range(length):
            x = self.a * x + rng.normal(0.0, np.sqrt(self.q), size=n_seq)
            y[:, t] = self.c * x + rng.normal(0.0, np.sqrt(self.r), size=n_seq)
        return y

    def filter(self, obs: np.ndarray) -> FilterResult:
        obs = np.asarray(obs, dtype=np.float64)
        n, length = obs.shape
        belief = np.empty((n, length + 1, 1))
        logp = np.empty((n, length))
        pred_mean = np.empty((n, length))
        pred_var = np.empty((n, length))
        latent_var = np.empty(length + 1)

        m = np.zeros(n)
        p = self.q / (1.0 - self.a**2)
        belief[:, 0, 0] = m
        latent_var[0] = p
        for t in range(length):
            s = self.c**2 * p + self.r
            mu = self.c * m
            pred_mean[:, t] = mu
            pred_var[:, t] = s
            logp[:, t] = _gauss_logpdf(obs[:, t], mu, s)

            k = p * self.c / s
            m_post = m + k * (obs[:, t] - mu)
            p_post = (1.0 - k * self.c) * p
            m = self.a * m_post
            p = self.a**2 * p_post + self.q
            belief[:, t + 1, 0] = m
            latent_var[t + 1] = p

        return FilterResult(
            belief=belief,
            logp=logp,
            predictive={"mean": pred_mean, "var": pred_var, "latent_var": latent_var},
        )


class GarchProcess(Process):
    """GARCH(1,1) returns: r_t = sigma_t eps_t, sigma_t^2 = w + alpha r_{t-1}^2 + beta sigma_{t-1}^2.

    The conditional variance is a deterministic function of the past, so the
    optimal predictor is exactly N(0, sigma_t^2) with no filtering uncertainty.
    That makes it the cleanest possible test of whether a *continuous* quantity
    is carried as a linear direction in the residual stream.
    """

    kind = "continuous"
    belief_kind = "log_variance"
    head = "gaussian"
    d_belief = 1

    def __init__(self, alpha: float = 0.10, beta: float = 0.85, name: str | None = None):
        if alpha + beta >= 1.0:
            raise ValueError("alpha + beta must be < 1 for covariance stationarity")
        self.alpha, self.beta = float(alpha), float(beta)
        self.omega = 1.0 - alpha - beta  # forces unconditional variance to 1
        self.name = name or f"garch_a{alpha:g}_b{beta:g}"

    def sample(self, n_seq: int, length: int, rng: np.random.Generator, burn_in: int = 500):
        sigma2 = np.full(n_seq, 1.0)
        r = np.zeros(n_seq)
        for _ in range(burn_in):
            sigma2 = self.omega + self.alpha * r**2 + self.beta * sigma2
            r = np.sqrt(sigma2) * rng.standard_normal(n_seq)
        out = np.empty((n_seq, length))
        for t in range(length):
            sigma2 = self.omega + self.alpha * r**2 + self.beta * sigma2
            r = np.sqrt(sigma2) * rng.standard_normal(n_seq)
            out[:, t] = r
        return out

    def filter(self, obs: np.ndarray) -> FilterResult:
        obs = np.asarray(obs, dtype=np.float64)
        n, length = obs.shape
        belief = np.empty((n, length + 1, 1))
        logp = np.empty((n, length))
        pred_var = np.empty((n, length))

        # The filter does not know the pre-sequence history, so it starts from the
        # unconditional variance -- the same information a next-token predictor has.
        sigma2 = np.ones(n)
        belief[:, 0, 0] = np.log(sigma2)
        for t in range(length):
            pred_var[:, t] = sigma2
            logp[:, t] = _gauss_logpdf(obs[:, t], 0.0, sigma2)
            sigma2 = self.omega + self.alpha * obs[:, t] ** 2 + self.beta * sigma2
            belief[:, t + 1, 0] = np.log(sigma2)

        return FilterResult(
            belief=belief,
            logp=logp,
            predictive={"mean": np.zeros_like(pred_var), "var": pred_var},
        )


class RegimeSwitchingVol(Process):
    """Markov-switching volatility: r_t ~ N(0, sigma^2_{s_t}) with s_t a Markov chain.

    This is the bridge case.  The latent is discrete, so the belief is a point on
    a simplex exactly as for an HMM, but the observation is continuous and the
    optimal predictive density is a Gaussian *mixture* whose weights are the
    predicted regime posterior.  A mixture head can therefore represent the
    optimum exactly, and its weights can be read off and compared to the true
    belief directly at the output.
    """

    kind = "continuous"
    belief_kind = "simplex"
    head = "gaussian_mixture"

    def __init__(
        self,
        sigmas: tuple[float, ...] = (0.5, 1.0, 2.0),
        stay: float = 0.98,
        name: str | None = None,
    ):
        k = len(sigmas)
        P = np.full((k, k), (1.0 - stay) / (k - 1))
        np.fill_diagonal(P, stay)
        self.P = P
        self.pi = np.full(k, 1.0 / k)  # symmetric chain -> uniform stationary
        sig2 = np.asarray(sigmas, dtype=np.float64) ** 2
        self.sigma2 = sig2 / float(self.pi @ sig2)  # unconditional variance = 1
        self.k = k
        self.d_belief = k
        self.stay = float(stay)
        self.name = name or f"regimevol_k{k}_stay{stay:g}"

    def sample(self, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        cdf = np.cumsum(self.P, axis=1)
        s = rng.choice(self.k, size=n_seq, p=self.pi)
        out = np.empty((n_seq, length))
        for t in range(length):
            out[:, t] = rng.standard_normal(n_seq) * np.sqrt(self.sigma2[s])
            u = rng.random(n_seq)
            s = (cdf[s] < u[:, None]).sum(axis=1).clip(0, self.k - 1)
        return out

    def filter(self, obs: np.ndarray) -> FilterResult:
        obs = np.asarray(obs, dtype=np.float64)
        n, length = obs.shape
        belief = np.empty((n, length + 1, self.k))
        logp = np.empty((n, length))
        weights = np.empty((n, length, self.k))

        eta = np.broadcast_to(self.pi, (n, self.k)).copy()
        belief[:, 0] = eta
        for t in range(length):
            weights[:, t] = eta
            comp = _gauss_logpdf(obs[:, t, None], 0.0, self.sigma2[None, :])  # (n, k)
            joint = eta * np.exp(comp - comp.max(axis=1, keepdims=True))
            total = np.clip(joint.sum(axis=1), _EPS, None)
            logp[:, t] = np.log(total) + comp.max(axis=1)
            post = joint / total[:, None]
            eta = post @ self.P
            belief[:, t + 1] = eta

        var = np.broadcast_to(self.sigma2, (n, length, self.k)).copy()
        return FilterResult(
            belief=belief,
            logp=logp,
            predictive={"weights": weights, "means": np.zeros_like(var), "vars": var},
        )
