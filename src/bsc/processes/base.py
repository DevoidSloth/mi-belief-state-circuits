"""Process interface.

Every process in the zoo ships with an *exact* optimal predictor, so that any
claim about a trained model can be stated as a distance from a number we know
rather than a distance from another model.

Indexing convention (used everywhere in this repo, do not deviate):

    obs[n, t]           observation at time t, t = 0 .. T-1
    belief[n, t]        P(latent_t | obs[n, :t])          t = 0 .. T   (length T+1)
    logp[n, t]          log P(obs[n, t] | obs[n, :t])     t = 0 .. T-1

So ``belief[:, 0]`` is the prior (no observations absorbed) and ``belief[:, t+1]``
is the belief a next-token predictor must hold at residual-stream position ``t``
in order to predict ``obs[:, t+1]``.  That off-by-one is the single most common
source of silently-wrong interpretability results here, hence the convention.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FilterResult:
    """Exact output of the optimal filter run over a batch of observations."""

    belief: np.ndarray                      # (n, T+1, d_belief)
    logp: np.ndarray                        # (n, T)
    predictive: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_seq(self) -> int:
        return self.belief.shape[0]

    @property
    def length(self) -> int:
        return self.logp.shape[1]

    def target_belief(self) -> np.ndarray:
        """Belief aligned to residual-stream positions: target[:, t] predicts obs[:, t+1]."""
        return self.belief[:, 1:, :]

    def optimal_loss(self, burn_in: int = 0) -> float:
        """Mean negative log-likelihood in nats/token under the exact optimal predictor."""
        return float(-self.logp[:, burn_in:].mean())

    def optimal_loss_per_position(self) -> np.ndarray:
        return -self.logp.mean(axis=0)


class Process(abc.ABC):
    """A stochastic process with an analytically known optimal predictor."""

    #: short identifier used for filenames and result keys
    name: str = "process"
    #: "discrete" -> integer tokens + categorical head; "continuous" -> scalar obs
    kind: str = "discrete"
    #: what the belief vector means, for plotting and probe interpretation
    belief_kind: str = "simplex"
    #: number of tokens in the alphabet (discrete only)
    vocab_size: int = 0
    #: dimensionality of the belief vector
    d_belief: int = 0
    #: which output head can represent the optimal predictor exactly
    head: str = "categorical"

    @abc.abstractmethod
    def sample(self, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        """Draw observations, shape (n_seq, length)."""

    @abc.abstractmethod
    def filter(self, obs: np.ndarray) -> FilterResult:
        """Run the exact optimal filter over observations."""

    def entropy_rate(self) -> float | None:
        """Closed-form entropy rate in nats/symbol, or None when no closed form exists."""
        return None

    # -- convenience ------------------------------------------------------

    def sample_and_filter(
        self, n_seq: int, length: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, FilterResult]:
        obs = self.sample(n_seq, length, rng)
        return obs, self.filter(obs)

    def optimal_loss_mc(
        self,
        n_seq: int = 512,
        length: int = 512,
        seed: int = 0,
        burn_in: int = 0,
    ) -> tuple[float, float]:
        """Monte-Carlo estimate of the optimal loss, with a standard error.

        Unlike :meth:`entropy_rate` this is available for every process.  Because
        the filter is exact, the only error is the sampling error over sequences,
        which the returned standard error quantifies (clustered by sequence, since
        positions within a sequence are correlated).
        """
        rng = np.random.default_rng(seed)
        _, res = self.sample_and_filter(n_seq, length, rng)
        per_seq = -res.logp[:, burn_in:].mean(axis=1)
        return float(per_seq.mean()), float(per_seq.std(ddof=1) / np.sqrt(len(per_seq)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} d_belief={self.d_belief}>"
