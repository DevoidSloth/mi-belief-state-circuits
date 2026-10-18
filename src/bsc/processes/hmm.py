"""Edge-emitting hidden Markov models and their exact optimal predictor.

An edge-emitting HMM (the presentation used in computational mechanics) is a
tensor

    T[x, i, j] = P(X_t = x, S_{t+1} = j | S_t = i)

with ``sum_{x,j} T[x, i, j] == 1`` for every state ``i``.  The optimal predictor
is the forward algorithm: carry the *mixed state* (belief) ``eta_t = P(S_t | x_{<t})``
and update it multiplicatively.
"""

from __future__ import annotations

import numpy as np

from .base import FilterResult, Process

_EPS = 1e-300


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix (dominant left eigenvector)."""
    vals, vecs = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(np.real(vecs[:, idx]))
    total = pi.sum()
    if total <= 0:
        raise ValueError("degenerate stationary distribution")
    return pi / total


def slem(A: np.ndarray) -> float:
    """Second-largest eigenvalue modulus. 0 = instant mixing, close to 1 = slow."""
    mags = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return float(mags[1]) if len(mags) > 1 else 0.0


class HMM(Process):
    """Edge-emitting HMM with an exact forward-algorithm predictor."""

    kind = "discrete"
    belief_kind = "simplex"
    head = "categorical"

    def __init__(
        self,
        T: np.ndarray,
        name: str = "hmm",
        initial: np.ndarray | None = None,
        check: bool = True,
    ):
        T = np.asarray(T, dtype=np.float64)
        if T.ndim != 3 or T.shape[1] != T.shape[2]:
            raise ValueError(f"T must have shape (n_symbols, n_states, n_states), got {T.shape}")
        if check:
            row = T.sum(axis=(0, 2))
            if not np.allclose(row, 1.0, atol=1e-10):
                raise ValueError(f"rows of T must sum to 1, got {row}")
            if (T < -1e-12).any():
                raise ValueError("T must be non-negative")
        self.T = np.clip(T, 0.0, None)
        self.name = name
        self.n_symbols, self.n_states, _ = T.shape
        self.vocab_size = self.n_symbols
        self.d_belief = self.n_states
        self.A = self.T.sum(axis=0)          # state -> state marginal
        self.emission = self.T.sum(axis=2)   # (x, i) -> P(x | i)
        if initial is None:
            self._initial = stationary_distribution(self.A)
        else:
            initial = np.asarray(initial, dtype=np.float64)
            self._initial = initial / initial.sum()

    # ---- structure ------------------------------------------------------

    @property
    def initial(self) -> np.ndarray:
        return self._initial.copy()

    def is_unifilar(self, tol: float = 1e-12) -> bool:
        """True when (state, symbol) determines the next state."""
        return bool(((self.T > tol).sum(axis=2) <= 1).all())

    def is_irreducible(self) -> bool:
        reach = (self.A > 1e-12).astype(np.int64)
        closure = reach.copy()
        for _ in range(self.n_states):
            closure = (((closure @ reach) > 0).astype(np.int64) | closure)
        closure = closure + np.eye(self.n_states, dtype=np.int64)
        return bool((closure > 0).all())

    def slem(self) -> float:
        return slem(self.A)

    # ---- exact inference ------------------------------------------------

    def sample(self, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the stationary process, vectorised over sequences."""
        flat = self.T.transpose(1, 0, 2).reshape(self.n_states, -1)  # (i, x*n_states)
        cdf = np.cumsum(flat, axis=1)
        cdf /= cdf[:, -1:]
        states = rng.choice(self.n_states, size=n_seq, p=self._initial)
        obs = np.empty((n_seq, length), dtype=np.int64)
        for t in range(length):
            u = rng.random(n_seq)
            k = (cdf[states] < u[:, None]).sum(axis=1)
            k = np.clip(k, 0, flat.shape[1] - 1)
            obs[:, t] = k // self.n_states
            states = k % self.n_states
        return obs

    def filter(self, obs: np.ndarray) -> FilterResult:
        """Scaled forward algorithm: exact posterior and exact predictive pmf."""
        obs = np.asarray(obs, dtype=np.int64)
        n, length = obs.shape
        belief = np.empty((n, length + 1, self.n_states), dtype=np.float64)
        logp = np.empty((n, length), dtype=np.float64)
        pmf = np.empty((n, length, self.n_symbols), dtype=np.float64)
        rows = np.arange(n)

        eta = np.broadcast_to(self._initial, (n, self.n_states)).copy()
        belief[:, 0] = eta
        for t in range(length):
            p_all = np.einsum("ns,xsj->nx", eta, self.T)
            p_all = np.clip(p_all, _EPS, None)
            p_all /= p_all.sum(axis=1, keepdims=True)
            pmf[:, t] = p_all

            x = obs[:, t]
            logp[:, t] = np.log(p_all[rows, x])

            joint = np.einsum("ns,nsj->nj", eta, self.T[x])
            eta = joint / np.clip(joint.sum(axis=1, keepdims=True), _EPS, None)
            belief[:, t + 1] = eta

        return FilterResult(belief=belief, logp=logp, predictive={"pmf": pmf})

    # ---- entropy rate ---------------------------------------------------

    def entropy_rate(self) -> float | None:
        """Closed form when the presentation is unifilar and irreducible.

        For a unifilar, irreducible, asymptotically synchronising presentation the
        belief collapses onto single states, so h = sum_i pi_i H(P(X | S = i)).
        Non-unifilar machines (Mess3, SNS) generally have an uncountable
        mixed-state set and no finite closed form; use :meth:`entropy_rate_msp`
        or :meth:`Process.optimal_loss_mc` there instead.
        """
        if not (self.is_unifilar() and self.is_irreducible()):
            return None
        pi = stationary_distribution(self.A)
        p = self.emission.T  # (i, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, -p * np.log(p), 0.0)
        return float(pi @ terms.sum(axis=1))

    def entropy_rate_msp(
        self, tol: float = 1e-9, max_states: int = 20000, max_depth: int = 400
    ) -> float | None:
        """Exact entropy rate via an explicit mixed-state presentation.

        Enumerates the reachable belief states from the prior.  Returns None when
        the set does not close within ``max_states``, which is the expected
        outcome for processes such as Mess3 whose mixed-state set is a fractal.
        """
        scale = 1.0 / tol

        def key(v: np.ndarray) -> tuple:
            return tuple(np.round(v * scale).astype(np.int64))

        start = self._initial
        index = {key(start): 0}
        vectors = [start]
        edges: list[list[tuple[int, float]]] = [[]]
        frontier = [0]
        depth = 0
        while frontier and depth < max_depth:
            new_frontier = []
            for idx in frontier:
                eta = vectors[idx]
                out: list[tuple[int, float]] = []
                for x in range(self.n_symbols):
                    joint = eta @ self.T[x]
                    p = float(joint.sum())
                    if p <= tol:
                        continue
                    nxt = joint / p
                    k = key(nxt)
                    if k not in index:
                        if len(vectors) >= max_states:
                            return None
                        index[k] = len(vectors)
                        vectors.append(nxt)
                        edges.append([])
                        new_frontier.append(index[k])
                    out.append((index[k], p))
                edges[idx] = out
            frontier = new_frontier
            depth += 1
        if frontier:
            return None

        m = len(vectors)
        M = np.zeros((m, m))
        H = np.zeros(m)
        for i, out in enumerate(edges):
            if not out:
                continue
            probs = np.array([p for _, p in out])
            probs = probs / probs.sum()
            H[i] = float(-(probs * np.log(np.clip(probs, _EPS, None))).sum())
            for (j, _), pn in zip(out, probs, strict=True):
                M[i, j] += pn
        try:
            mu = stationary_distribution(M)
        except (ValueError, np.linalg.LinAlgError):
            return None
        return float(mu @ H)
