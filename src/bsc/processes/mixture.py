"""A family of processes with one drawn per sequence - in-context inference with ground truth.

The interesting version of "does the model do Bayesian inference?" is not whether
it learned one filter, but whether it infers *which* process it is in from the
context.  That question usually has no ground truth, because the Bayes-optimal
predictor over a continuous family is intractable.

With a **finite** family it is exactly tractable.  Run each component's forward
algorithm on the same observations, weight the components by their marginal
likelihood so far, and mix::

    P(x_t | x_<t) = sum_k W_k(t) P_k(x_t | x_<t),
    W_k(t) proportional to prior_k * prod_{s<t} P_k(x_s | x_<s)

So the model has two beliefs to track at once - which process, and where in that
process - and both are known exactly at every position.  ``belief`` here is the
posterior over *components*, since that is the in-context quantity; the
within-component posteriors are kept in ``predictive`` for the analyses that
want them.
"""

from __future__ import annotations

import numpy as np

from .base import FilterResult, Process

_EPS = 1e-300


class MixtureOfProcesses(Process):
    """Draw a component uniformly (or from ``prior``) per sequence, then sample from it."""

    def __init__(
        self,
        components: list[Process],
        prior: np.ndarray | None = None,
        name: str = "mixture",
    ):
        if not components:
            raise ValueError("need at least one component")
        kinds = {c.kind for c in components}
        heads = {c.head for c in components}
        if len(kinds) > 1 or len(heads) > 1:
            raise ValueError("mixture components must share kind and head")
        vocabs = {c.vocab_size for c in components}
        if len(vocabs) > 1:
            raise ValueError("mixture components must share an alphabet")

        self.components = components
        self.k = len(components)
        self.prior = (
            np.full(self.k, 1.0 / self.k)
            if prior is None
            else np.asarray(prior, dtype=np.float64) / np.sum(prior)
        )
        self.name = name
        self.kind = components[0].kind
        self.head = components[0].head
        self.vocab_size = components[0].vocab_size
        self.belief_kind = "simplex"
        self.d_belief = self.k

    def sample(self, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        assign = rng.choice(self.k, size=n_seq, p=self.prior)
        out = None
        for c in range(self.k):
            mask = assign == c
            if not mask.any():
                continue
            part = self.components[c].sample(int(mask.sum()), length, rng)
            if out is None:
                out = np.empty((n_seq, length), dtype=part.dtype)
            out[mask] = part
        self.last_assignment = assign
        return out

    def filter(self, obs: np.ndarray) -> FilterResult:
        obs = np.asarray(obs)
        n, length = obs.shape
        per = [c.filter(obs) for c in self.components]

        # cumulative log evidence for each component, shifted so entry t uses x_<t
        logp_k = np.stack([r.logp for r in per])                       # (k, n, T)
        cum = np.zeros((self.k, n, length + 1))
        cum[:, :, 1:] = np.cumsum(logp_k, axis=2)
        cum += np.log(np.clip(self.prior, _EPS, None))[:, None, None]
        weights = np.exp(cum - cum.max(axis=0, keepdims=True))
        weights /= np.clip(weights.sum(axis=0, keepdims=True), _EPS, None)  # (k, n, T+1)
        belief = np.transpose(weights, (1, 2, 0))                          # (n, T+1, k)

        w = weights[:, :, :length]
        if self.head == "categorical":
            pmf_k = np.stack([r.predictive["pmf"] for r in per])           # (k, n, T, V)
            pmf = (w[..., None] * pmf_k).sum(axis=0)
            pmf /= np.clip(pmf.sum(-1, keepdims=True), _EPS, None)
            predictive = {"pmf": pmf, "component_weights": belief}
            logp = np.log(np.clip(np.take_along_axis(pmf, obs[:, :, None], 2)[:, :, 0], _EPS, None))
        else:
            raise NotImplementedError("continuous mixture components are not supported yet")

        return FilterResult(belief=belief, logp=logp, predictive=predictive)

    def predictive_from_belief(self, belief: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError(
            "the predictive depends on the within-component posteriors as well as the "
            "component weights, so it is not a function of `belief` alone"
        )

    def entropy_rate(self) -> float | None:
        """No single entropy rate: loss keeps falling as the component is identified."""
        return None
