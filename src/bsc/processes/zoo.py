"""The discrete process zoo.

Each constructor returns an :class:`~bsc.processes.hmm.HMM` whose optimal
predictor is the forward algorithm.  Processes are chosen to span the axes that
matter for this study:

* **unifilar vs non-unifilar** - unifilar machines synchronise, so the belief
  collapses to a vertex of the simplex and the model only has to track a discrete
  state.  Non-unifilar machines (Mess3, SNS) keep the belief in the interior
  forever, which is where the interesting geometry lives.
* **mixing speed** - controlled through the SLEM of the state-transition matrix.
* **memory** - ``persistent_bit`` never forgets, so achievable loss keeps falling
  with context length instead of plateauing at an entropy rate.
"""

from __future__ import annotations

import numpy as np

from .hmm import HMM


def golden_mean(p: float = 0.5) -> HMM:
    """No two consecutive 0s. Unifilar, and order-1 Markov in the observations."""
    T = np.zeros((2, 2, 2))
    T[1, 0, 0] = p            # A --1--> A
    T[0, 0, 1] = 1.0 - p      # A --0--> B
    T[1, 1, 0] = 1.0          # B --1--> A  (a 0 would make "00")
    return HMM(T, name=f"golden_mean_p{p:g}")


def even_process(p: float = 0.5) -> HMM:
    """Blocks of 1s have even length. Unifilar but of infinite Markov order."""
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = p            # A --0--> A   (block closed, parity even)
    T[1, 0, 1] = 1.0 - p      # A --1--> B   (parity now odd)
    T[1, 1, 0] = 1.0          # B --1--> A   (must complete the pair)
    return HMM(T, name=f"even_p{p:g}")


def zero_one_random() -> HMM:
    """Emit 0, then 1, then a fair coin. Unifilar, period-3 structure."""
    T = np.zeros((2, 3, 3))
    T[0, 0, 1] = 1.0
    T[1, 1, 2] = 1.0
    T[0, 2, 0] = 0.5
    T[1, 2, 0] = 0.5
    return HMM(T, name="zero_one_random")


def rrxor(p: float = 0.5, q: float = 0.5) -> HMM:
    """Random, random, XOR of the two. Unifilar, 5 states, needs 2 tokens of memory."""
    S, S0, S1, F, TT = 0, 1, 2, 3, 4
    T = np.zeros((2, 5, 5))
    T[0, S, S0] = p
    T[1, S, S1] = 1.0 - p
    T[0, S0, F] = q          # 0 xor 0 = 0
    T[1, S0, TT] = 1.0 - q   # 0 xor 1 = 1
    T[0, S1, TT] = q         # 1 xor 0 = 1
    T[1, S1, F] = 1.0 - q    # 1 xor 1 = 0
    T[0, F, S] = 1.0
    T[1, TT, S] = 1.0
    return HMM(T, name=f"rrxor_p{p:g}_q{q:g}")


def mess3(x: float = 0.15, alpha: float = 0.6) -> HMM:
    """Mess3: 3 states, 3 symbols, **non-unifilar**.

    The mixed-state set is a fractal subset of the 2-simplex; this is the process
    whose belief-state geometry is visually striking and the main target of the
    geometry experiments.
    """
    b = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x
    ay, ax, by, bx = alpha * y, alpha * x, b * y, b * x
    T = np.zeros((3, 3, 3))
    T[0] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
    T[1] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
    T[2] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]
    return HMM(T, name=f"mess3_x{x:g}_a{alpha:g}")


def simple_nonunifilar(p: float = 0.5, q: float = 0.5) -> HMM:
    """Simple Non-unifilar Source: symbol 1 from state B is ambiguous."""
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = p
    T[1, 0, 1] = 1.0 - p
    T[1, 1, 0] = q
    T[1, 1, 1] = 1.0 - q
    return HMM(T, name=f"sns_p{p:g}_q{q:g}")


def persistent_bit(theta: tuple[float, float] = (0.35, 0.65)) -> HMM:
    """A latent bit drawn once and never changed; emissions are iid given it.

    Non-mixing on purpose: the belief never forgets, it only sharpens, so the
    achievable loss falls monotonically with context length rather than settling
    at an entropy rate.  Useful for probing how context length interacts with
    achievable loss.
    """
    t0, t1 = theta
    T = np.zeros((2, 2, 2))
    T[1, 0, 0] = t0
    T[0, 0, 0] = 1.0 - t0
    T[1, 1, 1] = t1
    T[0, 1, 1] = 1.0 - t1
    return HMM(T, name="persistent_bit", initial=np.array([0.5, 0.5]), check=True)


def drifting_coin(theta: tuple[float, float] = (0.25, 0.75), stay: float = 0.995) -> HMM:
    """Two-regime coin with a very slow switch: long memory but still mixing."""
    t0, t1 = theta
    T = np.zeros((2, 2, 2))
    T[1, 0, 0] = t0 * stay
    T[0, 0, 0] = (1.0 - t0) * stay
    T[1, 0, 1] = t0 * (1.0 - stay)
    T[0, 0, 1] = (1.0 - t0) * (1.0 - stay)
    T[1, 1, 1] = t1 * stay
    T[0, 1, 1] = (1.0 - t1) * stay
    T[1, 1, 0] = t1 * (1.0 - stay)
    T[0, 1, 0] = (1.0 - t1) * (1.0 - stay)
    return HMM(T, name=f"drifting_coin_stay{stay:g}")


def random_hmm(
    n_states: int = 4,
    n_symbols: int = 3,
    seed: int = 0,
    unifilar: bool = False,
    concentration: float = 0.6,
    target_slem: tuple[float, float] | None = None,
    max_tries: int = 4000,
) -> HMM:
    """A random HMM, optionally rejection-sampled into a mixing-time band.

    ``target_slem`` gives an inclusive (low, high) band for the second-largest
    eigenvalue modulus, which is the knob for how fast the process forgets.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(max_tries):
        if unifilar:
            T = np.zeros((n_symbols, n_states, n_states))
            for i in range(n_states):
                emit = rng.dirichlet(np.full(n_symbols, concentration))
                for x in range(n_symbols):
                    T[x, i, rng.integers(n_states)] += emit[x]
        else:
            flat = rng.dirichlet(np.full(n_symbols * n_states, concentration), size=n_states)
            T = flat.reshape(n_states, n_symbols, n_states).transpose(1, 0, 2)
        hmm = HMM(T, name="random", check=False)
        if not hmm.is_irreducible():
            continue
        if target_slem is not None:
            s = hmm.slem()
            if not (target_slem[0] <= s <= target_slem[1]):
                continue
        tag = "uni" if unifilar else "non"
        hmm.name = f"random_{tag}_s{n_states}_v{n_symbols}_seed{seed}"
        return hmm
    raise RuntimeError(f"could not sample an HMM meeting the constraints in {max_tries} tries")
