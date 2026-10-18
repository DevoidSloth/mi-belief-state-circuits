"""Data plumbing: sampled observations paired with their exact posteriors.

An :class:`EvalSet` bundles everything a comparison needs on one device: the
observations, the exact per-position optimal log-probabilities, the exact belief
at every position, and the exact predictive distribution.  Model loss and oracle
loss are then computed on *identical* sequences, which makes the optimality gap
a paired difference with a very small standard error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..model.heads import predictive_to_torch
from ..processes.base import Process


@dataclass
class EvalSet:
    obs: torch.Tensor                    # (n, ctx) long or float
    belief: torch.Tensor                 # (n, ctx, d_belief) aligned to input positions
    oracle_logp: torch.Tensor            # (n, ctx)
    predictive: dict[str, torch.Tensor]
    obs_np: np.ndarray
    process_name: str

    @property
    def n_seq(self) -> int:
        return self.obs.shape[0]

    @property
    def ctx(self) -> int:
        return self.obs.shape[1]

    def oracle_loss(self, burn_in: int = 0) -> float:
        return float(-self.oracle_logp[:, burn_in:].mean())

    def slice_positions(self, burn_in: int) -> EvalSet:
        """Drop the first ``burn_in`` positions from every aligned tensor."""
        return EvalSet(
            obs=self.obs[:, burn_in:],
            belief=self.belief[:, burn_in:],
            oracle_logp=self.oracle_logp[:, burn_in:],
            predictive={k: v[:, burn_in:] for k, v in self.predictive.items()},
            obs_np=self.obs_np[:, burn_in:],
            process_name=self.process_name,
        )


def make_eval_set(
    process: Process,
    n_seq: int,
    ctx: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> EvalSet:
    """Sample sequences and attach the exact filter output, aligned to model positions.

    Alignment: the model sees a BOS at position 0 and ``obs[:, t-1]`` at position
    ``t``, and predicts ``obs[:, t]`` there.  The belief it must hold at position
    ``t`` is therefore ``filter.belief[:, t]`` -- the *first* ``ctx`` entries of the
    length ``ctx + 1`` belief array, not the last.
    """
    rng = np.random.default_rng(seed)
    obs_np = process.sample(n_seq, ctx, rng)
    res = process.filter(obs_np)

    obs_dtype = torch.long if process.kind == "discrete" else torch.float32
    return EvalSet(
        obs=torch.as_tensor(obs_np, device=device, dtype=obs_dtype),
        belief=torch.as_tensor(res.belief[:, :ctx], device=device, dtype=torch.float32),
        oracle_logp=torch.as_tensor(res.logp, device=device, dtype=torch.float32),
        predictive=predictive_to_torch(res.predictive, device),
        obs_np=obs_np,
        process_name=process.name,
    )


class StreamingSampler:
    """Endless training batches, refreshed from the process so nothing is memorised.

    Sampling an HMM is cheap relative to a training step, but not free, so a pool
    of sequences is drawn and reused for ``refresh_every`` steps before being
    replaced.  With the default settings a model sees a given sequence a handful
    of times at most, which keeps train and eval loss indistinguishable.
    """

    def __init__(
        self,
        process: Process,
        batch_size: int,
        ctx: int,
        seed: int,
        pool_size: int = 8192,
        refresh_every: int = 2000,
        device: torch.device | str = "cpu",
    ):
        self.process = process
        self.batch_size = batch_size
        self.ctx = ctx
        self.pool_size = pool_size
        self.refresh_every = refresh_every
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator(device="cpu").manual_seed(seed)
        self.dtype = torch.long if process.kind == "discrete" else torch.float32
        self._pool: torch.Tensor | None = None
        self._step = 0
        self._refresh()

    def _refresh(self) -> None:
        obs = self.process.sample(self.pool_size, self.ctx, self.rng)
        self._pool = torch.as_tensor(obs, device=self.device, dtype=self.dtype)

    def next_batch(self) -> torch.Tensor:
        if self._step and self._step % self.refresh_every == 0:
            self._refresh()
        self._step += 1
        idx = torch.randint(
            0, self.pool_size, (self.batch_size,), generator=self.torch_gen, device="cpu"
        )
        return self._pool[idx.to(self._pool.device)]
