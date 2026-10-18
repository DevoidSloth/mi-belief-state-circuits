"""The experiment suite. Each module writes one JSON file into ``results/``."""

from __future__ import annotations

from collections.abc import Callable

from . import (
    exp01_optimality,
    exp02_geometry,
    exp03_circuits,
    exp04_kalman,
    exp05_sae,
    exp06_capacity,
    exp07_incontext,
)

EXPERIMENTS: dict[str, Callable[..., dict]] = {
    "optimality": exp01_optimality.run,
    "geometry": exp02_geometry.run,
    "circuits": exp03_circuits.run,
    "kalman": exp04_kalman.run,
    "sae": exp05_sae.run,
    "capacity": exp06_capacity.run,
    "incontext": exp07_incontext.run,
}

#: order matters: later experiments reuse checkpoints the earlier ones trained
ORDER = ("optimality", "geometry", "circuits", "kalman", "sae", "capacity", "incontext")

__all__ = ["EXPERIMENTS", "ORDER"]
