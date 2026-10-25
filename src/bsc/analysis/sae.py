"""Sparse autoencoders, and what they do to a variable that is known to be dense.

SAEs assume the representation decomposes into sparsely-active, roughly
monosemantic features.  A Bayesian belief state is the opposite kind of object:
it is a point on a simplex, continuous, and *every* coordinate is non-zero
almost always.  Whether SAE features nonetheless capture it is usually
unanswerable, because there is no ground truth to compare against.  Here there
is.

The experiment: train an SAE on the residual stream without ever showing it the
belief, then ask how much belief variance ``k`` SAE features explain versus what
``k`` linear dimensions of the same activations explain.  Three references are
reported so the comparison cannot be rigged:

* ``sae_r2``  - the best ``k`` SAE features, chosen greedily *using* the belief;
* ``pca_r2``  - the top ``k`` principal components (unsupervised, like the SAE);
* ``rrr_r2``  - the best possible ``k``-dimensional linear subspace, from
  reduced-rank regression, which upper-bounds what any ``k`` directions can do.

The SAE gets the friendliest treatment of the three (unsupervised basis but
supervised selection), so a shortfall against ``pca_r2`` is a real shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SAEConfig:
    d_in: int = 64
    expansion: int = 8
    l1_coeff: float = 3e-3
    lr: float = 1e-3
    steps: int = 4000
    batch_size: int = 4096
    resample_every: int = 1000
    dead_threshold: int = 200
    seed: int = 0
    normalise_input: bool = True

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion


class SAE(nn.Module):
    """Standard ReLU sparse autoencoder with a unit-norm decoder.

    The decoder columns are renormalised after every step, which stops the model
    from shrinking feature directions to cheat the L1 penalty -- without it the
    reported L0 is meaningless.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        d_in, d_sae = cfg.d_in, cfg.d_sae
        w = torch.randn(d_sae, d_in) / np.sqrt(d_in)
        self.W_dec = nn.Parameter(w / w.norm(dim=1, keepdim=True))
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("fire_count", torch.zeros(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


@dataclass
class SAEMetrics:
    l0: float
    variance_explained: float
    loss_recovered: float
    dead_fraction: float
    loss_model: float
    loss_with_sae: float
    loss_ablated: float
    d_sae: int
    l1_coeff: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def train_sae(acts: torch.Tensor, cfg: SAEConfig, device=None, verbose: bool = False):
    """Train an SAE on flattened activations, shape (N, d_in)."""
    device = device or acts.device
    X = acts.reshape(-1, acts.shape[-1]).float().to(device)
    mu = X.mean(0, keepdim=True)
    scale = X.std() if cfg.normalise_input else torch.tensor(1.0, device=device)
    Xn = (X - mu) / scale

    cfg.d_in = X.shape[-1]
    sae = SAE(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    history = []
    for step in range(cfg.steps):
        idx = torch.randint(0, len(Xn), (cfg.batch_size,), generator=gen).to(device)
        batch = Xn[idx]
        recon, f = sae(batch)
        mse = ((recon - batch) ** 2).sum(-1).mean()
        l1 = (f.abs() * sae.W_dec.norm(dim=1)).sum(-1).mean()
        loss = mse + cfg.l1_coeff * l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalise_decoder()

        with torch.no_grad():
            sae.fire_count += (f > 0).sum(0).float()

        if cfg.resample_every and (step + 1) % cfg.resample_every == 0:
            _resample_dead(sae, Xn, cfg, gen, device)
            sae.fire_count.zero_()
        if verbose and step % max(1, cfg.steps // 5) == 0:
            with torch.no_grad():
                print(f"    sae step {step:5d} mse {float(mse):.4f} L0 {float((f > 0).sum(-1).float().mean()):.1f}")
            history.append({"step": step, "mse": float(mse), "l1": float(l1)})

    return sae, {"mean": mu, "scale": scale, "history": history}


@torch.no_grad()
def _resample_dead(sae: SAE, X: torch.Tensor, cfg: SAEConfig, gen, device) -> int:
    """Re-initialise features that never fire, aimed at the worst-reconstructed inputs."""
    dead = (sae.fire_count < cfg.dead_threshold).nonzero(as_tuple=True)[0]
    if len(dead) == 0:
        return 0
    idx = torch.randint(0, len(X), (min(8192, len(X)),), generator=gen).to(device)
    sample = X[idx]
    recon, _ = sae(sample)
    err = ((sample - recon) ** 2).sum(-1)
    probs = (err / err.sum().clamp_min(1e-12)).clamp_min(1e-12)
    pick = torch.multinomial(probs, len(dead), replacement=True)
    directions = sample[pick] - sae.b_dec
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sae.W_dec.data[dead] = directions
    sae.W_enc.data[:, dead] = directions.T * 0.2
    sae.b_enc.data[dead] = 0.0
    return len(dead)


@torch.no_grad()
def sae_metrics(
    sae: SAE,
    stats: dict,
    acts: torch.Tensor,
    model,
    eval_set,
    hook: str,
    burn_in: int = 16,
) -> SAEMetrics:
    """L0, variance explained, and downstream loss recovered when spliced into the model."""
    from ..model.heads import head_logprob

    device = eval_set.obs.device
    X = acts.reshape(-1, acts.shape[-1]).float().to(device)
    Xn = (X - stats["mean"]) / stats["scale"]
    recon_n, f = sae(Xn)
    recon = recon_n * stats["scale"] + stats["mean"]

    l0 = float((f > 0).sum(-1).float().mean())
    resid = ((X - recon) ** 2).sum()
    total = ((X - X.mean(0, keepdim=True)) ** 2).sum()
    var_explained = float(1.0 - resid / total.clamp_min(1e-12))
    dead = float((f > 0).float().sum(0).eq(0).float().mean())

    full = recon.reshape(acts.shape).to(device)

    def loss_of(params):
        lp = head_logprob(model.cfg.head, params, eval_set.obs, model.cfg.n_mixture)
        return float((-lp[:, burn_in:]).mean())

    base = loss_of(model(eval_set.obs))

    def splice(x, _name):
        out = x.clone()
        out[:, burn_in:] = full[:, burn_in:].to(x.dtype)
        return out

    with model.hooks([(hook, splice)]):
        with_sae = loss_of(model(eval_set.obs))

    mean_act = acts.mean(dim=(0, 1), keepdim=True).to(device)

    def ablate(x, _name):
        out = x.clone()
        out[:, burn_in:] = mean_act.to(x.dtype)
        return out

    with model.hooks([(hook, ablate)]):
        ablated = loss_of(model(eval_set.obs))

    denom = ablated - base
    return SAEMetrics(
        l0=l0,
        variance_explained=var_explained,
        loss_recovered=float(1.0 - (with_sae - base) / max(denom, 1e-12)),
        dead_fraction=dead,
        loss_model=base,
        loss_with_sae=with_sae,
        loss_ablated=ablated,
        d_sae=sae.cfg.d_sae,
        l1_coeff=sae.cfg.l1_coeff,
        extra={"hook": hook},
    )


# ---------------------------------------------------------------------------
# how much of the belief do k directions capture?
# ---------------------------------------------------------------------------


def _r2_from_features(F_: np.ndarray, Y: np.ndarray, ridge: float = 1e-8) -> float:
    """R^2 of the least-squares fit of Y on features F_ (with intercept)."""
    F_ = np.asarray(F_, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if F_.shape[1] == 0:
        return 0.0
    Fc = F_ - F_.mean(0)
    Yc = Y - Y.mean(0)
    gram = Fc.T @ Fc + ridge * np.eye(Fc.shape[1])
    beta = np.linalg.solve(gram, Fc.T @ Yc)
    resid = Yc - Fc @ beta
    return float(1.0 - (resid**2).sum() / max((Yc**2).sum(), 1e-12))


def greedy_feature_selection(F_: np.ndarray, Y: np.ndarray, k: int, ridge: float = 1e-8):
    """Greedily pick the k features that best explain Y. Returns (indices, R^2 after each).

    Scored through precomputed Gram matrices, so each candidate costs O(k^3) rather
    than O(N k^2). With a few hundred candidates and N in the tens of thousands the
    naive form dominates the whole experiment's runtime.
    """
    F_ = np.asarray(F_, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Fc = F_ - F_.mean(0)
    Yc = Y - Y.mean(0)
    total = (Yc**2).sum()
    if total <= 1e-12 or Fc.shape[1] == 0:
        return [], []

    G = Fc.T @ Fc                      # (m, m)
    C = Fc.T @ Yc                      # (m, d)
    active = np.flatnonzero(np.diag(G) > 1e-12)

    chosen: list[int] = []
    curve: list[float] = []
    for _ in range(k):
        best_j, best_r2 = None, -np.inf
        for j in active:
            if j in chosen:
                continue
            idx = [*chosen, int(j)]
            sub = G[np.ix_(idx, idx)] + ridge * np.eye(len(idx))
            try:
                beta = np.linalg.solve(sub, C[idx])
            except np.linalg.LinAlgError:
                continue
            # explained sum of squares = trace(beta^T G_sub beta) with beta = G^-1 C
            r2 = float((C[idx] * beta).sum() / total)
            if r2 > best_r2:
                best_j, best_r2 = int(j), r2
        if best_j is None:
            break
        chosen.append(best_j)
        curve.append(min(best_r2, 1.0))
    return chosen, curve


def reduced_rank_r2(X: np.ndarray, Y: np.ndarray, k: int, ridge: float = 1e-6) -> float:
    """R^2 of the best rank-k linear predictor of Y from X.

    Upper bound on what *any* k directions of X can explain, SAE features
    included, since an SAE feature is a (rectified) linear function of X.
    """
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    gram = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    beta = np.linalg.solve(gram, Xc.T @ Yc)
    fitted = Xc @ beta
    u, s, vt = np.linalg.svd(fitted, full_matrices=False)
    k = min(k, len(s))
    approx = (u[:, :k] * s[:k]) @ vt[:k]
    return float(1.0 - ((Yc - approx) ** 2).sum() / max((Yc**2).sum(), 1e-12))


@torch.no_grad()
def belief_capture(
    sae: SAE,
    stats: dict,
    acts: torch.Tensor,
    belief: torch.Tensor,
    ks=(1, 2, 4, 8, 16),
    burn_in: int = 16,
    max_points: int = 15000,
    seed: int = 0,
    max_candidates: int = 384,
) -> dict:
    """Belief variance explained by k SAE features versus k linear dimensions."""
    A = acts[:, burn_in:].reshape(-1, acts.shape[-1]).float()
    B = belief[:, burn_in:].reshape(-1, belief.shape[-1]).float().cpu()

    rng = np.random.default_rng(seed)
    if len(A) > max_points:
        idx = rng.choice(len(A), max_points, replace=False)
        A, B = A[idx], B[idx]

    device = next(sae.parameters()).device
    Xn = (A.to(device) - stats["mean"]) / stats["scale"]
    Fsae = sae.encode(Xn).cpu().numpy()
    X = A.cpu().numpy().astype(np.float64)
    Y = B.numpy().astype(np.float64)

    # restrict the greedy search to the most belief-correlated live features,
    # so the cost is bounded without materially weakening the SAE's chances
    live = np.flatnonzero(Fsae.var(0) > 1e-12)
    if len(live) > max_candidates:
        # rank by single-feature R^2, computed in one vectorised pass
        Fc = Fsae[:, live] - Fsae[:, live].mean(0)
        Yc = Y - Y.mean(0)
        num = (Fc.T @ Yc) ** 2
        den = (Fc**2).sum(0)[:, None] * (Yc**2).sum()
        singles = np.where(den > 1e-24, num / np.maximum(den, 1e-24), 0.0).sum(1)
        live = live[np.argsort(-singles)[:max_candidates]]
    Fsub = Fsae[:, live]

    kmax = max(ks)
    order, curve = greedy_feature_selection(Fsub, Y, kmax)

    from .geometry import pca

    p = pca(X, kmax)
    proj = p.transform(X)

    rows = []
    for k in ks:
        sae_r2 = curve[k - 1] if k <= len(curve) else (curve[-1] if curve else 0.0)
        rows.append(
            {
                "k": int(k),
                "sae_r2": float(sae_r2),
                "pca_r2": _r2_from_features(proj[:, :k], Y),
                "rrr_r2": reduced_rank_r2(X, Y, k),
            }
        )
    return {
        "rows": rows,
        "selected_features": [int(live[i]) for i in order],
        "n_live_features": int((Fsae.var(0) > 1e-12).sum()),
        "n_points": int(len(A)),
        "full_linear_r2": _r2_from_features(X, Y),
    }
