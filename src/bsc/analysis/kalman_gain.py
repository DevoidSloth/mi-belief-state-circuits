"""Does the network implement the Kalman gain? Three levels of evidence.

For the linear-Gaussian process the optimal filter is a two-term linear
recursion on the belief mean::

    m_{t+1} = a(1 - K c) * m_t  +  a K * y_t

A recurrent network can implement that literally.  A transformer cannot: it has
no state to carry, so it must implement the *unrolled* form, which is a
convolution over the observation history::

    m_t = a K * sum_{d >= 0} alpha^d * y_{t-1-d},     alpha = a(1 - K c)

That distinction is the whole reason the naive weight-level test fails here.
Reading ``v^T (W_V W_O) u`` off a head asks whether it transports a belief from
the previous position, and the answer is no, because the model never formed a
belief there to transport -- it re-derives the whole sum at every position.  The
right object to compare against the analytic gain is therefore the model's
**impulse response**: how much a past observation moves the decoded belief.

Levels of evidence, weakest to strongest:

1. **Behavioural** - fit ``m_hat_{t+1} = alpha m_hat_t + beta y_t`` to the
   model's decoded beliefs and compare to the analytic pair.  Cheap, and it
   holds for any correct implementation, recurrent or convolutional.
2. **Functional** - measure ``d m_hat_t / d y_{t-1-d}`` by autograd and compare
   the whole kernel against ``a K alpha^d``.  No assumption of linearity: the
   Jacobian is averaged over real inputs.
3. **Weight** - per-head contribution to that kernel, plus the head's
   observation-to-belief gain ``w_embed^T (W_V W_O) u`` read straight off the
   parameters with no activations involved.

Run on a model trained with ``norm="none"``.  LayerNorm inserts a
data-dependent rescaling between the residual stream and every read, which makes
weight-level coefficients incomparable to analytic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..model.transformer import BeliefTransformer
from ..processes.continuous import KalmanProcess
from ..train.data import EvalSet
from .patching import Component, all_components
from .probes import ProbeResult


# ---------------------------------------------------------------------------
# level 1: behavioural
# ---------------------------------------------------------------------------


@dataclass
class RecursionFit:
    alpha: float
    beta: float
    r2: float
    analytic_alpha: float
    analytic_beta: float
    alpha_rel_error: float
    beta_rel_error: float
    n_points: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@torch.no_grad()
def recursion_fit(
    model: BeliefTransformer,
    process: KalmanProcess,
    eval_set: EvalSet,
    probe: ProbeResult,
    hook: str,
    burn_in: int = 24,
) -> RecursionFit:
    """Fit the model's own decoded belief trajectory to the analytic recursion."""
    from ..train.trainer import collect_residuals

    acts = collect_residuals(model, eval_set, [hook])[hook]
    decoded = probe.decode(acts)[..., 0].cpu().numpy()
    obs = eval_set.obs_np

    m_prev = decoded[:, burn_in:-1].ravel()
    m_next = decoded[:, burn_in + 1 :].ravel()
    y_cur = obs[:, burn_in:-1].ravel()   # position t+1 receives obs[:, t]

    X = np.stack([m_prev, y_cur, np.ones_like(m_prev)], axis=1)
    coef, *_ = np.linalg.lstsq(X, m_next, rcond=None)
    resid = m_next - X @ coef
    r2 = 1.0 - (resid**2).sum() / max(((m_next - m_next.mean()) ** 2).sum(), 1e-12)

    ss = process.steady_state()
    return RecursionFit(
        alpha=float(coef[0]),
        beta=float(coef[1]),
        r2=float(r2),
        analytic_alpha=ss["recur_prev"],
        analytic_beta=ss["recur_obs"],
        alpha_rel_error=float(abs(coef[0] - ss["recur_prev"]) / abs(ss["recur_prev"])),
        beta_rel_error=float(abs(coef[1] - ss["recur_obs"]) / abs(ss["recur_obs"])),
        n_points=int(len(m_next)),
    )


# ---------------------------------------------------------------------------
# level 2: functional -- the impulse response
# ---------------------------------------------------------------------------


def analytic_kernel(process: KalmanProcess, max_lag: int) -> np.ndarray:
    """d m_t / d y_{t-1-d} for the exact filter: a K alpha^d."""
    ss = process.steady_state()
    lags = np.arange(max_lag + 1)
    return ss["recur_obs"] * ss["recur_prev"] ** lags


def belief_impulse_response(
    model: BeliefTransformer,
    eval_set: EvalSet,
    probe: ProbeResult,
    hook: str,
    max_lag: int = 24,
    readout_positions: tuple[int, ...] | None = None,
    n_seq: int = 256,
) -> np.ndarray:
    """Measure d(decoded belief at t) / d(observation at t-1-lag) by autograd.

    Averaged over sequences and over several readout positions, so the result is
    the model's mean linearisation on the data distribution rather than a single
    point estimate.
    """
    model.eval()
    ctx = eval_set.ctx
    if readout_positions is None:
        readout_positions = tuple(range(max(max_lag + 4, ctx // 2), ctx, max(1, ctx // 8)))

    device = eval_set.obs.device
    u = probe.W[:, 0].to(device)
    obs = eval_set.obs[:n_seq].clone().float().requires_grad_(True)

    captured: dict[str, torch.Tensor] = {}

    def grab(x, name):
        captured[name] = x
        return x

    with model.hooks([(hook, grab)]):
        model(obs)
    acts = captured[hook]

    kernel = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    for t0 in readout_positions:
        decoded = acts[:, t0] @ u
        grad = torch.autograd.grad(decoded.sum(), obs, retain_graph=True)[0]
        g = grad.mean(0).detach().cpu().numpy()   # (ctx,)
        for lag in range(max_lag + 1):
            j = t0 - 1 - lag
            if j >= 0:
                kernel[lag] += g[j]
                counts[lag] += 1
    return kernel / np.maximum(counts, 1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-12))


def _decay_rate(
    kernel: np.ndarray, max_lag: int | None = None, floor_frac: float = 1e-2
) -> tuple[float, float]:
    """Geometric decay rate of a kernel, from a log-linear fit. Returns (rate, r2).

    The fit is restricted to lags where the kernel is still at least
    ``floor_frac`` of its peak.  An exponential kernel decays into the numerical
    noise of the Jacobian estimate within twenty lags or so, and including those
    points fits a straight line through noise, which biases the slope by several
    percent while leaving the fit's R^2 looking healthy.  The kernel *cosine*
    against the analytic gain is the robust statistic; this one is reported
    because a decay rate is the interpretable form of the same claim.
    """
    k = kernel[: max_lag or len(kernel)]
    peak = float(np.max(np.abs(k))) if k.size else 0.0
    mask = k > max(peak * floor_frac, 1e-9)
    if mask.sum() < 3:
        mask = k > 1e-9
    if mask.sum() < 3:
        return float("nan"), float("nan")
    lags = np.arange(len(k))[mask]
    logk = np.log(k[mask])
    slope, intercept = np.polyfit(lags, logk, 1)
    resid = logk - (slope * lags + intercept)
    r2 = 1.0 - resid.var() / max(logk.var(), 1e-12)
    return float(np.exp(slope)), float(r2)


# ---------------------------------------------------------------------------
# level 3: weights
# ---------------------------------------------------------------------------


@dataclass
class HeadGain:
    label: str
    obs_to_belief_gain: float    # w_embed^T (W_V W_O) u, read off the weights
    kernel_share: float          # fraction of the kernel's mass this head carries
    kernel_cosine_without: float # cosine to analytic after ablating this head
    ov_norm: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@torch.no_grad()
def head_observation_gains(model: BeliefTransformer, probe: ProbeResult) -> dict[str, float]:
    """``w_embed^T (W_V W_O) u`` for every head: observation in, belief out."""
    if model.cfg.input_kind != "continuous":
        raise ValueError("observation gain is defined for continuous inputs")
    device = next(model.parameters()).device
    u = probe.W[:, 0].to(device).double()
    w = model.embed.weight[:, 0].double()
    out = {}
    for layer in range(model.cfg.n_layers):
        attn = model.blocks[layer].attn
        for h in range(model.cfg.n_heads):
            M = (attn.W_V[h] @ attn.W_O[h]).double()
            out[f"L{layer}H{h}"] = float(w @ M @ u)
    return out


def component_kernel_attribution(
    model: BeliefTransformer,
    eval_set: EvalSet,
    probe: ProbeResult,
    hook: str,
    analytic: np.ndarray,
    max_lag: int = 24,
    n_seq: int = 128,
) -> list[dict]:
    """Ablate each component and re-measure the kernel, to see who carries it."""
    from .scrubbing import _resample  # noqa: F401  (kept for symmetry of imports)

    base = belief_impulse_response(model, eval_set, probe, hook, max_lag, n_seq=n_seq)
    rows = []
    for comp in all_components(model):
        with torch.no_grad():
            _, cache = model.run_with_cache(eval_set.obs[:n_seq], names=[comp.hook])
        act = cache[comp.hook]
        slot = act if comp.kind == "mlp" else act[:, :, comp.head]
        mean = slot.mean(dim=0, keepdim=True).expand_as(slot).clone()

        def fn(x, _name, mean=mean, comp=comp):
            x = x.clone()
            if comp.kind == "mlp":
                return mean
            x[:, :, comp.head] = mean
            return x

        with model.hooks([(comp.hook, fn)]):
            k = belief_impulse_response(model, eval_set, probe, hook, max_lag, n_seq=n_seq)
        rows.append(
            {
                "component": comp.label,
                "kernel_share": float(
                    1.0 - np.abs(k).sum() / max(np.abs(base).sum(), 1e-12)
                ),
                "kernel_cosine_without": _cosine(k, analytic),
                "kernel_l1_change": float(np.abs(k - base).sum()),
            }
        )
    return sorted(rows, key=lambda r: -r["kernel_l1_change"])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@dataclass
class KalmanGainReport:
    recursion: RecursionFit
    kernel: list[float]
    analytic_kernel: list[float]
    kernel_cosine: float
    kernel_decay_rate: float
    analytic_decay_rate: float
    decay_rate_rel_error: float
    kernel_fit_r2: float
    gain_at_lag0: float
    analytic_gain_at_lag0: float
    head_obs_gains: dict[str, float]
    attribution: list[dict] = field(default_factory=list)
    analytic: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["recursion"] = self.recursion.as_dict()
        return d


def kalman_gain_report(
    model: BeliefTransformer,
    process: KalmanProcess,
    eval_set: EvalSet,
    probe: ProbeResult,
    hook: str,
    burn_in: int = 24,
    max_lag: int = 24,
    with_attribution: bool = True,
) -> KalmanGainReport:
    ss = process.steady_state()
    rec = recursion_fit(model, process, eval_set, probe, hook, burn_in)
    analytic = analytic_kernel(process, max_lag)
    kernel = belief_impulse_response(model, eval_set, probe, hook, max_lag)

    rate, fit_r2 = _decay_rate(kernel)
    attribution = (
        component_kernel_attribution(model, eval_set, probe, hook, analytic, max_lag)
        if with_attribution
        else []
    )

    return KalmanGainReport(
        recursion=rec,
        kernel=kernel.tolist(),
        analytic_kernel=analytic.tolist(),
        kernel_cosine=_cosine(kernel, analytic),
        kernel_decay_rate=rate,
        analytic_decay_rate=ss["recur_prev"],
        decay_rate_rel_error=float(abs(rate - ss["recur_prev"]) / abs(ss["recur_prev"])),
        kernel_fit_r2=fit_r2,
        gain_at_lag0=float(kernel[0]),
        analytic_gain_at_lag0=float(analytic[0]),
        head_obs_gains=head_observation_gains(model, probe),
        attribution=attribution,
        analytic={
            "a": process.a,
            "c": process.c,
            "K": ss["K"],
            "S": ss["S"],
            "recur_prev": ss["recur_prev"],
            "recur_obs": ss["recur_obs"],
        },
        extra={"hook": hook, "probe_r2": probe.r2, "norm": model.cfg.norm, "max_lag": max_lag},
    )
