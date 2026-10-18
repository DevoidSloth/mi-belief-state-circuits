"""Named process registry.

``DEFAULT_ZOO`` is the set the headline experiments run over.  Keep it stable:
result files key on these names.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Process
from .mixture import MixtureOfProcesses
from .continuous import GarchProcess, KalmanProcess, RegimeSwitchingVol
from .zoo import (
    drifting_coin,
    even_process,
    golden_mean,
    mess3,
    persistent_bit,
    random_hmm,
    rrxor,
    simple_nonunifilar,
    zero_one_random,
)

BUILDERS: dict[str, Callable[[], Process]] = {
    # -- unifilar epsilon-machines ---------------------------------------
    "golden_mean": lambda: golden_mean(0.5),
    "even": lambda: even_process(0.5),
    "zero_one_random": lambda: zero_one_random(),
    "rrxor": lambda: rrxor(0.5, 0.5),
    # -- non-unifilar: belief stays in the interior of the simplex --------
    # mess3(0.15, 0.6) is the canonical parameterisation from the belief-state
    # geometry literature and is kept for reproducing the fractal.  Its total
    # predictability is only ~0.009 nats, so circuit and optimality claims use
    # mess3_hard, which has the same structure with a usable signal.
    "mess3": lambda: mess3(0.15, 0.6),
    "mess3_hard": lambda: mess3(0.05, 0.85),
    "sns": lambda: simple_nonunifilar(0.5, 0.5),
    # -- memory / mixing-time controls ------------------------------------
    "persistent_bit": lambda: persistent_bit((0.35, 0.65)),
    "drifting_coin": lambda: drifting_coin((0.25, 0.75), 0.995),
    "random_uni_fast": lambda: random_hmm(4, 3, seed=1, unifilar=True, target_slem=(0.0, 0.35)),
    "random_uni_slow": lambda: random_hmm(4, 3, seed=2, unifilar=True, target_slem=(0.75, 0.95)),
    "random_non": lambda: random_hmm(4, 3, seed=3, unifilar=False, target_slem=(0.2, 0.7)),
    "random_non_big": lambda: random_hmm(6, 4, seed=4, unifilar=False, target_slem=(0.2, 0.8)),
    # -- continuous latents ------------------------------------------------
    "kalman": lambda: KalmanProcess(a=0.95, c=1.0, snr=4.0),
    "kalman_slow": lambda: KalmanProcess(a=0.995, c=1.0, snr=2.0),
    "garch": lambda: GarchProcess(alpha=0.10, beta=0.85),
    "regimevol": lambda: RegimeSwitchingVol((0.5, 1.0, 2.0), stay=0.98),
    # -- families: one component drawn per sequence, for in-context inference --
    "even_family": lambda: MixtureOfProcesses(
        [even_process(p) for p in (0.2, 0.5, 0.8)], name="even_family"
    ),
    "mess3_family": lambda: MixtureOfProcesses(
        [mess3(0.05, a) for a in (0.55, 0.75, 0.95)], name="mess3_family"
    ),
    "golden_family": lambda: MixtureOfProcesses(
        [golden_mean(p) for p in (0.15, 0.4, 0.65, 0.9)], name="golden_family"
    ),
}

#: processes used by the headline experiment sweep
DEFAULT_ZOO: tuple[str, ...] = (
    "golden_mean",
    "even",
    "zero_one_random",
    "rrxor",
    "mess3",
    "mess3_hard",
    "sns",
    "persistent_bit",
    "drifting_coin",
    "random_uni_fast",
    "random_uni_slow",
    "random_non",
    "kalman",
    "garch",
    "regimevol",
)

#: cheap subset for smoke tests and CI
SMOKE_ZOO: tuple[str, ...] = ("even", "mess3", "kalman", "garch")


def get_process(name: str) -> Process:
    if name not in BUILDERS:
        raise KeyError(f"unknown process {name!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[name]()


def all_names() -> list[str]:
    return list(BUILDERS)
