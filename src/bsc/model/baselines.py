"""Baselines that bound how much of the task is actually interesting.

The brief's trap is real: if an n-gram table reaches the entropy rate, nothing
found inside a transformer trained on that process means anything.  These
baselines exist to be *reported*, not to be beaten:

* :class:`NGramBaseline` - the finite-memory ceiling for discrete processes.
  Its gap to the optimum is the amount of structure that genuinely requires a
  belief state.
* :class:`LinearBaseline` - a linear readout of a fixed window; the finite-memory
  ceiling for continuous processes too.
* :class:`GRUBaseline` - a recurrent model, which *can* express a filter.  If a
  2-layer transformer matches the forward algorithm and a GRU does not, that is
  itself a result about architecture, not about the task.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import ModelConfig


class NGramBaseline:
    """Order-k Markov model over the observation alphabet, add-lambda smoothed.

    Sequence starts are padded with a dedicated BOS symbol so that early
    positions are scored under the same chain rule as the transformer's.
    """

    def __init__(self, vocab_size: int, order: int, alpha: float = 0.1):
        self.V = vocab_size
        self.order = order
        self.alpha = alpha
        self.B = vocab_size  # BOS symbol id
        self.n_ctx = (vocab_size + 1) ** order
        self.counts = np.zeros((self.n_ctx, vocab_size), dtype=np.float64)

    def _context_index(self, obs: np.ndarray) -> np.ndarray:
        """Index of the length-``order`` context preceding each position."""
        n, length = obs.shape
        padded = np.concatenate([np.full((n, self.order), self.B), obs], axis=1)
        idx = np.zeros((n, length), dtype=np.int64)
        base = 1
        for j in range(self.order):
            # j = 0 is the immediately preceding symbol
            idx += padded[:, self.order - 1 - j : self.order - 1 - j + length] * base
            base *= self.V + 1
        return idx

    def fit(self, obs: np.ndarray) -> NGramBaseline:
        ctx = self._context_index(obs)
        np.add.at(self.counts, (ctx.ravel(), obs.ravel()), 1.0)
        return self

    def log_probs(self, obs: np.ndarray) -> np.ndarray:
        ctx = self._context_index(obs)
        smoothed = self.counts + self.alpha
        probs = smoothed / smoothed.sum(axis=1, keepdims=True)
        return np.log(probs[ctx, obs])

    def loss(self, obs: np.ndarray) -> float:
        return float(-self.log_probs(obs).mean())

    @staticmethod
    def fit_best(
        train: np.ndarray,
        val: np.ndarray,
        vocab_size: int,
        orders=(1, 2, 3, 4, 5),
        alphas=(0.01, 0.1, 0.5, 1.0),
    ) -> tuple[NGramBaseline, dict]:
        """Select (order, alpha) on a validation split. Returns the fitted model."""
        best, best_loss, table = None, np.inf, {}
        for k in orders:
            model = NGramBaseline(vocab_size, k, alpha=alphas[0]).fit(train)
            for a in alphas:
                model.alpha = a
                loss = model.loss(val)
                table[f"order{k}_alpha{a}"] = loss
                if loss < best_loss:
                    best_loss, best = loss, (k, a)
        model = NGramBaseline(vocab_size, best[0], alpha=best[1]).fit(train)
        return model, {"order": best[0], "alpha": best[1], "val_loss": best_loss, "sweep": table}


class _SequenceBaseline(nn.Module):
    """Shared plumbing: BOS-prefixed input, same output-head contract as the transformer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        from .transformer import BeliefTransformer

        n_out = BeliefTransformer._n_outputs(cfg)
        self.n_out = n_out
        if cfg.input_kind == "discrete":
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        else:
            self.embed = nn.Linear(1, cfg.d_model)
        self.bos = nn.Parameter(torch.randn(cfg.d_model) * cfg.init_scale)

    def _embed_with_bos(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        if self.cfg.input_kind == "discrete":
            emb = self.embed(obs[:, :-1].long())
        else:
            emb = self.embed(obs[:, :-1, None].float())
        return torch.cat([self.bos.expand(b, 1, self.cfg.d_model), emb], dim=1)


class GRUBaseline(_SequenceBaseline):
    """A GRU: architecturally *capable* of exact filtering, unlike an n-gram."""

    def __init__(self, cfg: ModelConfig, n_layers: int | None = None):
        super().__init__(cfg)
        self.rnn = nn.GRU(
            cfg.d_model, cfg.d_model, num_layers=n_layers or cfg.n_layers, batch_first=True
        )
        self.unembed = nn.Linear(cfg.d_model, self.n_out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self._embed_with_bos(obs))
        return self.unembed(h)


class LinearBaseline(_SequenceBaseline):
    """Linear readout of a fixed window of the last ``window`` observations.

    For continuous processes the window includes squared observations, so it can
    express a truncated GARCH recursion -- the strongest fair finite-memory
    baseline for a variance-tracking task.
    """

    def __init__(self, cfg: ModelConfig, window: int = 8):
        super().__init__(cfg)
        self.window = window
        if cfg.input_kind == "discrete":
            d_in = window * cfg.vocab_size
        else:
            d_in = window * 2
        self.readout = nn.Linear(d_in + 1, self.n_out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b, n = obs.shape
        w = self.window
        if self.cfg.input_kind == "discrete":
            feats = F.one_hot(obs.long(), self.cfg.vocab_size).float()
        else:
            feats = torch.stack([obs.float(), obs.float() ** 2], dim=-1)
        d = feats.shape[-1]
        pad = torch.zeros(b, w, d, device=obs.device, dtype=feats.dtype)
        padded = torch.cat([pad, feats], dim=1)  # (b, w + n, d)
        # position t uses observations t-w .. t-1
        windows = padded.unfold(1, w, 1)[:, :n]  # (b, n, d, w)
        flat = windows.reshape(b, n, d * w)
        seen = (torch.arange(n, device=obs.device).float() / max(n, 1)).expand(b, n)[..., None]
        return self.readout(torch.cat([flat, seen], dim=-1))


class OracleModel:
    """Wraps a process's exact filter behind the same evaluation interface.

    Included so the reported floor is produced by the *same code path* as the
    model's loss, not by a separately-derived formula that could disagree.
    """

    def __init__(self, process):
        self.process = process

    def loss(self, obs: np.ndarray) -> float:
        return float(-self.process.filter(obs).logp.mean())
