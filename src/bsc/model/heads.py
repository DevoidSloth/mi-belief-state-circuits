"""Output heads and the losses that make model and oracle directly comparable.

Every head is chosen so the optimal predictor is *exactly representable*:

* ``categorical``      - HMMs; the optimum is a pmf over the alphabet.
* ``gaussian``         - Kalman and GARCH; the optimum is a single Gaussian.
* ``gaussian_mixture`` - regime-switching vol; the optimum is a K-component
  zero-mean mixture whose weights are the regime posterior, so a K-component
  mixture head can reach the floor and its weights can be read off directly.

Because the head can represent the optimum, ``model_loss - oracle_loss`` on the
same sequences is a clean non-negative optimality gap with no representational
excuse baked in.  It is also a *paired* difference, so its standard error is far
smaller than that of either loss on its own.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)
LOGVAR_MIN, LOGVAR_MAX = -12.0, 12.0


def _split_gaussian(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = params[..., 0]
    log_var = params[..., 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
    return mean, log_var


def _split_mixture(params: torch.Tensor, k: int):
    log_w = F.log_softmax(params[..., :k], dim=-1)
    means = params[..., k : 2 * k]
    log_vars = params[..., 2 * k : 3 * k].clamp(LOGVAR_MIN, LOGVAR_MAX)
    return log_w, means, log_vars


def head_logprob(head: str, params: torch.Tensor, targets: torch.Tensor, n_mixture: int = 3):
    """log q(target | context) per position, shape (batch, pos)."""
    if head == "categorical":
        log_q = F.log_softmax(params, dim=-1)
        return log_q.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    if head == "gaussian":
        mean, log_var = _split_gaussian(params)
        return -0.5 * (LOG_2PI + log_var + (targets - mean) ** 2 * torch.exp(-log_var))
    if head == "gaussian_mixture":
        log_w, means, log_vars = _split_mixture(params, n_mixture)
        y = targets.unsqueeze(-1)
        comp = -0.5 * (LOG_2PI + log_vars + (y - means) ** 2 * torch.exp(-log_vars))
        return torch.logsumexp(log_w + comp, dim=-1)
    raise ValueError(f"unknown head {head!r}")


def head_loss(head: str, params: torch.Tensor, targets: torch.Tensor, n_mixture: int = 3):
    """Mean negative log-likelihood in nats/token."""
    return -head_logprob(head, params, targets, n_mixture).mean()


def head_distribution(head: str, params: torch.Tensor, n_mixture: int = 3) -> dict:
    """Unpack head parameters into an interpretable distribution."""
    if head == "categorical":
        return {"log_pmf": F.log_softmax(params, dim=-1)}
    if head == "gaussian":
        mean, log_var = _split_gaussian(params)
        return {"mean": mean, "var": torch.exp(log_var)}
    log_w, means, log_vars = _split_mixture(params, n_mixture)
    return {"log_weights": log_w, "means": means, "vars": torch.exp(log_vars)}


# ---------------------------------------------------------------------------
# divergence from the exact optimal predictor
# ---------------------------------------------------------------------------


def kl_true_to_model(
    head: str,
    params: torch.Tensor,
    predictive: dict[str, torch.Tensor],
    n_mixture: int = 3,
    n_samples: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """KL(optimal || model) per position, shape (batch, pos).

    This is exactly the excess expected loss the model pays at each position, so
    it is the right currency for ablation effects: "ablating this head raises
    KL-to-optimal by 31x" is a statement about lost nats, not about an arbitrary
    distance.  Closed form for the categorical and single-Gaussian heads; a
    Monte-Carlo estimate (samples drawn from the *optimal* mixture) otherwise.
    """
    if head == "categorical":
        p = predictive["pmf"]
        log_q = F.log_softmax(params, dim=-1)
        log_p = torch.log(p.clamp_min(1e-30))
        return (p * (log_p - log_q)).sum(-1)

    if head == "gaussian":
        m1, v1 = predictive["mean"], predictive["var"]
        m2, log_v2 = _split_gaussian(params)
        v2 = torch.exp(log_v2)
        return 0.5 * (log_v2 - torch.log(v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    if head == "gaussian_mixture":
        w, mu, var = predictive["weights"], predictive["means"], predictive["vars"]
        log_w_p = torch.log(w.clamp_min(1e-30))
        shape = (n_samples, *w.shape[:-1])
        comp = torch.multinomial(
            w.reshape(-1, w.shape[-1]), n_samples, replacement=True, generator=generator
        )
        comp = comp.T.reshape(shape)  # (n_samples, batch, pos)
        mu_s = mu.unsqueeze(0).expand(n_samples, *mu.shape).gather(-1, comp.unsqueeze(-1))
        var_s = var.unsqueeze(0).expand(n_samples, *var.shape).gather(-1, comp.unsqueeze(-1))
        y = mu_s.squeeze(-1) + torch.sqrt(var_s.squeeze(-1)) * torch.randn(
            shape, device=w.device, dtype=w.dtype, generator=generator
        )

        def mixture_logpdf(y_, log_w_, mu_, var_):
            yy = y_.unsqueeze(-1)
            c = -0.5 * (LOG_2PI + torch.log(var_) + (yy - mu_) ** 2 / var_)
            return torch.logsumexp(log_w_ + c, dim=-1)

        log_p = mixture_logpdf(y, log_w_p.unsqueeze(0), mu.unsqueeze(0), var.unsqueeze(0))
        log_w_q, mu_q, log_var_q = _split_mixture(params, n_mixture)
        log_q = mixture_logpdf(
            y, log_w_q.unsqueeze(0), mu_q.unsqueeze(0), torch.exp(log_var_q).unsqueeze(0)
        )
        return (log_p - log_q).mean(0)

    raise ValueError(f"unknown head {head!r}")


def predictive_to_torch(predictive: dict, device, dtype=torch.float32) -> dict:
    """Move a FilterResult.predictive dict onto the model's device, trimming the
    trailing position that has no corresponding model prediction."""
    out = {}
    for k, v in predictive.items():
        if k == "latent_var":
            continue
        out[k] = torch.as_tensor(v, device=device, dtype=dtype)
    return out
