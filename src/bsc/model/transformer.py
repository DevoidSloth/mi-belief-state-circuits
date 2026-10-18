"""A small hooked transformer, written for interpretability rather than speed.

Design choices that exist specifically to make the analysis possible:

* **Per-head weight tensors.**  ``W_V`` is ``(n_heads, d_model, d_head)`` and
  ``W_O`` is ``(n_heads, d_head, d_model)``, so head ``h``'s OV circuit is exactly
  ``W_V[h] @ W_O[h]``.  The Kalman-gain experiment compares that matrix against an
  analytic gain, which is not expressible if QKV are fused.
* **Optional normalisation.**  ``norm="none"`` gives a strictly linear residual
  stream up to the MLPs, which removes the LayerNorm rescaling term from
  weight-level claims.  These models are small enough to train without it.
* **Explicit attention.**  The pattern is materialised so ``hook_pattern`` is a
  real tensor that can be read and written, at a cost that is irrelevant at
  ``d_model <= 256``.
* **BOS prefix.**  Input position ``t`` sees ``x_{t-1}`` and predicts ``x_t``, so
  the residual stream at position ``t`` must carry ``belief[t]`` exactly - no
  off-by-one in any probe or patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .hooks import HookedRoot, HookPoint


@dataclass
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_head: int | None = None
    d_mlp: int | None = None
    n_ctx: int = 256
    #: "discrete" -> token embedding of size vocab_size; "continuous" -> scalar projection
    input_kind: str = "discrete"
    vocab_size: int = 2
    #: "categorical" | "gaussian" | "gaussian_mixture"
    head: str = "categorical"
    n_mixture: int = 3
    #: "layer" | "rms" | "none"
    norm: str = "layer"
    act: str = "gelu"
    attn_only: bool = False
    init_scale: float = 0.02
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.d_head is None:
            if self.d_model % self.n_heads:
                raise ValueError("d_model must be divisible by n_heads when d_head is unset")
            self.d_head = self.d_model // self.n_heads
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model

    @property
    def n_components(self) -> int:
        """Attention heads plus MLPs -- the denominator for circuit-size claims."""
        return self.n_layers * (self.n_heads + (0 if self.attn_only else 1))


def _make_norm(kind: str, d_model: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(d_model)
    if kind == "rms":
        return nn.RMSNorm(d_model)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h, d, dh = cfg.n_heads, cfg.d_model, cfg.d_head
        self.W_Q = nn.Parameter(torch.empty(h, d, dh))
        self.W_K = nn.Parameter(torch.empty(h, d, dh))
        self.W_V = nn.Parameter(torch.empty(h, d, dh))
        self.W_O = nn.Parameter(torch.empty(h, dh, d))
        self.b_Q = nn.Parameter(torch.zeros(h, dh))
        self.b_K = nn.Parameter(torch.zeros(h, dh))
        self.b_V = nn.Parameter(torch.zeros(h, dh))
        self.b_O = nn.Parameter(torch.zeros(d))
        for w in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.normal_(w, std=cfg.init_scale)

        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()

        mask = torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _project(self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(batch, pos, d_model) -> (batch, pos, head, d_head) as one matmul."""
        b_, p_, _ = x.shape
        flat = W.permute(1, 0, 2).reshape(self.cfg.d_model, -1)
        return (x @ flat).view(b_, p_, self.cfg.n_heads, self.cfg.d_head) + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, pos, d_model)
        n = x.shape[1]
        q = self.hook_q(self._project(x, self.W_Q, self.b_Q))
        k = self.hook_k(self._project(x, self.W_K, self.b_K))
        v = self.hook_v(self._project(x, self.W_V, self.b_V))

        # Materialising the (batch, head, pos, pos) pattern is by far the most
        # expensive thing this model does, and it is only needed when something is
        # actually reading or writing it. When nothing is hooked there, defer to
        # the fused kernel; the two paths are checked against each other in tests.
        if self.hook_attn_scores._fns or self.hook_pattern._fns:
            scores = torch.einsum("bqhk,bshk->bhqs", q, k) / math.sqrt(self.cfg.d_head)
            scores = self.hook_attn_scores(scores)
            scores = scores.masked_fill(self.causal_mask[:n, :n], torch.finfo(scores.dtype).min)
            pattern = self.hook_pattern(scores.softmax(dim=-1))
            z = torch.einsum("bhqs,bshk->bqhk", pattern, v)
        else:
            z = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            ).transpose(1, 2)

        z = self.hook_z(z)
        b_, p_, _, _ = z.shape
        flat = self.W_O.reshape(-1, self.cfg.d_model)
        return z.reshape(b_, p_, -1) @ flat + self.b_O

    def OV(self, head: int) -> torch.Tensor:
        """The head's OV circuit as a d_model x d_model matrix: W_V[h] @ W_O[h]."""
        return self.W_V[head] @ self.W_O[head]

    def QK(self, head: int) -> torch.Tensor:
        """The head's QK circuit as a d_model x d_model matrix."""
        return self.W_Q[head] @ self.W_K[head].T


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[cfg.act]
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.hook_post(self.act(self.hook_pre(self.fc_in(x)))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = _make_norm(cfg.norm, cfg.d_model)
        self.attn = Attention(cfg)
        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        if not cfg.attn_only:
            self.ln2 = _make_norm(cfg.norm, cfg.d_model)
            self.mlp = MLP(cfg)
            self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.ln1(x))))
        if not self.cfg.attn_only:
            x = x + self.hook_mlp_out(self.mlp(self.ln2(x)))
        return self.hook_resid_post(x)


class BeliefTransformer(HookedRoot):
    """Decoder-only transformer over either token or scalar observations."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg

        if cfg.input_kind == "discrete":
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            nn.init.normal_(self.embed.weight, std=cfg.init_scale)
        elif cfg.input_kind == "continuous":
            self.embed = nn.Linear(1, cfg.d_model)
        else:
            raise ValueError(f"unknown input_kind {cfg.input_kind!r}")
        # A learned vector for the BOS position, where no observation has been seen.
        self.bos = nn.Parameter(torch.randn(cfg.d_model) * cfg.init_scale)
        self.pos_embed = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) * cfg.init_scale)

        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = _make_norm(cfg.norm, cfg.d_model)
        self.hook_resid_final = HookPoint()

        self.unembed = nn.Linear(cfg.d_model, self._n_outputs(cfg))
        nn.init.normal_(self.unembed.weight, std=cfg.init_scale)
        nn.init.zeros_(self.unembed.bias)
        self.setup_hooks()

    @staticmethod
    def _n_outputs(cfg: ModelConfig) -> int:
        if cfg.head == "categorical":
            return cfg.vocab_size
        if cfg.head == "gaussian":
            return 2
        if cfg.head == "gaussian_mixture":
            return 3 * cfg.n_mixture
        raise ValueError(f"unknown head {cfg.head!r}")

    # -- input construction ------------------------------------------------

    def build_input(self, obs: torch.Tensor) -> torch.Tensor:
        """Embed ``obs`` with a BOS prefix so position t predicts obs[:, t].

        Returns the pre-block residual stream, shape (batch, pos, d_model), where
        ``pos`` equals ``obs.shape[1]``.
        """
        b, n = obs.shape
        if self.cfg.input_kind == "discrete":
            emb = self.embed(obs[:, :-1].long())
        else:
            emb = self.embed(obs[:, :-1, None].float())
        emb = self.hook_embed(emb)
        bos = self.bos.expand(b, 1, self.cfg.d_model)
        x = torch.cat([bos, emb], dim=1)
        return x + self.hook_pos_embed(self.pos_embed[None, :n])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw head parameters, shape (batch, pos, n_outputs)."""
        x = self.build_input(obs)
        for block in self.blocks:
            x = block(x)
        x = self.hook_resid_final(self.ln_final(x))
        return self.unembed(x)

    # -- component bookkeeping used by the circuit experiments -------------

    def component_names(self) -> list[str]:
        names = []
        for layer in range(self.cfg.n_layers):
            names += [f"L{layer}H{h}" for h in range(self.cfg.n_heads)]
            if not self.cfg.attn_only:
                names.append(f"L{layer}MLP")
        return names

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def resid_hook(self, layer: int, where: str = "post") -> str:
        return f"blocks.{layer}.hook_resid_{where}"
