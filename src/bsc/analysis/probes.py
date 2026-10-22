"""Linear probes for the belief state, and the causal test that makes them mean something.

A high-R^2 probe on a representation the model never reads is the classic
interpretability trap: it tells you the information is *present*, not that it is
*used*.  Every probe fitted here is therefore paired with an intervention:

1. fit a linear decoder from the residual stream to the true posterior;
2. invert it to get a minimum-norm *write* direction;
3. steer the residual stream so the decoded belief becomes some other reachable
   belief, and check whether the model's output moves to the distribution that
   belief implies;
4. repeat with a random direction of identical norm as a control.

Step 4 is what separates "the model uses this direction" from "perturbing a
network changes its output".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..model.heads import kl_true_to_model
from ..model.transformer import BeliefTransformer
from ..processes.base import Process
from ..train.data import EvalSet


@dataclass
class ProbeResult:
    hook: str
    r2: float                       # held-out, variance-weighted across coordinates
    r2_train: float
    r2_per_coord: list[float]
    mse: float
    W: torch.Tensor = field(repr=False, default=None)   # (d_model, d_belief)
    b: torch.Tensor = field(repr=False, default=None)
    ridge: float = 0.0
    n_train: int = 0
    n_test: int = 0

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W + self.b

    def as_dict(self) -> dict:
        return {
            "hook": self.hook,
            "r2": self.r2,
            "r2_train": self.r2_train,
            "r2_per_coord": self.r2_per_coord,
            "mse": self.mse,
            "ridge": self.ridge,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def _split_by_sequence(n_seq: int, train_frac: float, seed: int):
    """Split on *sequences*, never on positions.

    Positions inside a sequence are strongly correlated, so a position-level split
    leaks the test set into the training set and inflates R^2.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_seq)
    cut = int(round(train_frac * n_seq))
    return perm[:cut], perm[cut:]


def _r2(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, list[float]]:
    resid = ((true - pred) ** 2).sum(0)
    total = ((true - true.mean(0, keepdim=True)) ** 2).sum(0)
    per_coord = torch.where(total > 1e-12, 1 - resid / total, torch.zeros_like(total))
    overall = 1.0 - float(resid.sum()) / float(total.sum().clamp_min(1e-12))
    return overall, [float(v) for v in per_coord]


def fit_probe(
    acts: torch.Tensor,
    belief: torch.Tensor,
    hook: str = "",
    ridge: float = 1e-3,
    train_frac: float = 0.7,
    seed: int = 0,
    burn_in: int = 0,
) -> ProbeResult:
    """Ridge-regress the belief onto activations, scored on held-out sequences.

    ``acts``   (n_seq, pos, d_model)
    ``belief`` (n_seq, pos, d_belief)
    """
    acts = acts[:, burn_in:].double()
    belief = belief[:, burn_in:].double()
    tr, te = _split_by_sequence(acts.shape[0], train_frac, seed)

    def flat(idx, x):
        return x[idx].reshape(-1, x.shape[-1])

    Xtr, Ytr = flat(tr, acts), flat(tr, belief)
    Xte, Yte = flat(te, acts), flat(te, belief)

    x_mu, y_mu = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc, Yc = Xtr - x_mu, Ytr - y_mu
    d = Xc.shape[1]
    gram = Xc.T @ Xc + ridge * Xc.shape[0] * torch.eye(d, dtype=Xc.dtype)
    W = torch.linalg.solve(gram, Xc.T @ Yc)
    b = y_mu - x_mu @ W

    r2_te, per_coord = _r2(Xte @ W + b, Yte)
    r2_tr, _ = _r2(Xtr @ W + b, Ytr)
    mse = float(((Yte - (Xte @ W + b)) ** 2).mean())
    return ProbeResult(
        hook=hook,
        r2=r2_te,
        r2_train=r2_tr,
        r2_per_coord=per_coord,
        mse=mse,
        W=W.float(),
        b=b.float(),
        ridge=ridge,
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
    )


def sweep_ridge(
    acts: torch.Tensor,
    belief: torch.Tensor,
    hook: str = "",
    ridges=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    **kw,
) -> ProbeResult:
    """Fit at several regularisation strengths and keep the best held-out R^2."""
    best = None
    for r in ridges:
        res = fit_probe(acts, belief, hook=hook, ridge=r, **kw)
        if best is None or res.r2 > best.r2:
            best = res
    return best


@torch.no_grad()
def probe_all_layers(
    model: BeliefTransformer,
    eval_set: EvalSet,
    burn_in: int = 16,
    ridge_sweep: bool = True,
    batch: int = 256,
    where: tuple[str, ...] = ("pre", "mid", "post"),
) -> dict[str, ProbeResult]:
    """Probe every residual-stream location; returns {hook_name: ProbeResult}."""
    from ..train.trainer import collect_residuals

    hooks = [f"blocks.{layer}.hook_resid_{w}" for layer in range(model.cfg.n_layers) for w in where]
    hooks.append("hook_resid_final")
    acts = collect_residuals(model, eval_set, hooks, batch=batch)
    belief = eval_set.belief.cpu()

    fit = sweep_ridge if ridge_sweep else fit_probe
    return {h: fit(acts[h], belief, hook=h, burn_in=burn_in) for h in hooks}


# ---------------------------------------------------------------------------
# causal validity
# ---------------------------------------------------------------------------


def write_direction(probe: ProbeResult) -> torch.Tensor:
    """Minimum-norm map from a belief change to a residual-stream change.

    The probe reads ``b = x W``.  To move the read-out by ``db`` with the smallest
    possible perturbation we need ``dx`` solving ``dx W = db``, whose minimum-norm
    solution is ``dx = db (W^T W)^{-1} W^T``.  Returns that (d_belief, d_model) map.

    Note that ``W`` is rank-deficient whenever the belief is a simplex point:
    beliefs sum to one, so their differences span only ``d_belief - 1``
    dimensions and no perturbation can move the read-out off that affine hull.
    The returned map is therefore a pseudo-inverse, exact on achievable belief
    changes and damped elsewhere -- which is what the steering test needs, since
    it only ever asks for differences between two real beliefs.
    """
    W = probe.W.double()
    gram = W.T @ W
    gram = gram + 1e-8 * torch.eye(gram.shape[0], dtype=gram.dtype)
    return torch.linalg.solve(gram, W.T).float()


@dataclass
class SteeringResult:
    """Outcome of a causal probe test, reported on pairs not used to pick the gain."""

    hook: str
    kl_before: float
    kl_after: float
    kl_random_control: float
    effect: float             # fraction of KL-to-target closed by steering
    control_effect: float
    margin: float             # effect minus the matched random-direction control
    margin_stderr: float
    best_strength: float
    delta_norm: float
    n_pairs: int
    by_strength: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@torch.no_grad()
def steering_test(
    model: BeliefTransformer,
    process: Process,
    eval_set: EvalSet,
    probe: ProbeResult,
    hook: str,
    n_pairs: int = 1024,
    burn_in: int = 16,
    seed: int = 0,
    strengths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> SteeringResult:
    """Steer the decoded belief toward another *reachable* belief; does the output follow?

    Targets are real beliefs drawn from elsewhere in the eval set rather than
    arbitrary simplex points, so the intervention stays on the manifold the model
    was trained to represent.

    Two confounds are controlled explicitly:

    * **Unknown gain.**  The minimum-norm write direction fixes the direction but
      not the scale the model applies, so the perturbation is swept over a range
      of strengths.
    * **Damage masquerading as steering.**  A large perturbation in *any*
      direction flattens the output, and a flatter distribution has lower KL to
      the target than a confident wrong one.  A random direction matched in norm
      is therefore swept identically, and the reported statistic is the *margin*
      between the two.

    The strength is selected on one half of the pairs and the margin is reported
    on the other half, so the headline number is not a maximum over choices.
    """
    device = eval_set.obs.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, ctx = eval_set.obs.shape
    rows = torch.randint(0, n, (n_pairs,), generator=gen)
    cols = torch.randint(burn_in, ctx, (n_pairs,), generator=gen)
    donor_rows = torch.randint(0, n, (n_pairs,), generator=gen)
    donor_cols = torch.randint(burn_in, ctx, (n_pairs,), generator=gen)

    rows_d, cols_d = rows.to(device), cols.to(device)
    obs = eval_set.obs[rows_d]
    cur_belief = eval_set.belief[rows_d, cols_d]
    tgt_belief = eval_set.belief[donor_rows.to(device), donor_cols.to(device)]

    target_pred = process.predictive_from_belief(tgt_belief.cpu().numpy())
    target_pred = {
        k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in target_pred.items()
    }

    write = write_direction(probe).to(device)
    base_delta = (tgt_belief - cur_belief) @ write  # (n_pairs, d_model)
    idx = (torch.arange(n_pairs, device=device), cols_d)

    def run(delta: torch.Tensor | None) -> torch.Tensor:
        if delta is None:
            return model(obs)

        def fn(x, _name):
            x = x.clone()
            x[idx] = x[idx] + delta
            return x

        with model.hooks([(hook, fn)]):
            return model(obs)

    def kl_per_pair(params: torch.Tensor) -> torch.Tensor:
        return kl_true_to_model(model.cfg.head, params[idx], target_pred, model.cfg.n_mixture)

    unit_rand = torch.randn(base_delta.shape, generator=gen).to(device)
    unit_rand = unit_rand / unit_rand.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    kl0 = kl_per_pair(run(None))

    # Drop pairs the model already predicts correctly. Their KL-to-target is
    # ~0, and a per-pair ratio against it explodes: a single pair at 1e-9
    # dominates the mean and produces "effects" in the thousands. Fractions are
    # then formed from *aggregate* KL rather than averaged per-pair ratios,
    # which is the stable estimator.
    keep = kl0 > max(1e-6, 0.05 * float(kl0.median()))
    if int(keep.sum()) < 32:
        keep = kl0 > 1e-9
    idx_keep = keep.nonzero(as_tuple=True)[0]
    half = len(idx_keep) // 2
    select, report = idx_keep[:half], idx_keep[half:]

    def fractions(kl_a, kl_c, where):
        before = kl0[where].mean()
        eff = 1.0 - kl_a[where].mean() / before.clamp_min(1e-12)
        ctl = 1.0 - kl_c[where].mean() / before.clamp_min(1e-12)
        return float(eff), float(ctl)

    def bootstrap_margin_se(kl_a, kl_c, where, n_boot: int = 400) -> float:
        g = torch.Generator(device="cpu").manual_seed(seed + 17)
        b = kl0[where].cpu()
        a = kl_a[where].cpu()
        c = kl_c[where].cpu()
        n = b.numel()
        if n < 4:
            return float("nan")
        draws = torch.randint(0, n, (n_boot, n), generator=g)
        bb = b[draws].mean(1).clamp_min(1e-12)
        margins = (c[draws].mean(1) - a[draws].mean(1)) / bb
        return float(margins.std(unbiased=True))

    by_strength: dict[str, dict] = {}
    best_s, best_margin = strengths[0], -np.inf
    cache: dict[float, tuple] = {}
    for st in strengths:
        delta = st * base_delta
        rand = unit_rand * delta.norm(dim=-1, keepdim=True)
        kl_a = kl_per_pair(run(delta))
        kl_c = kl_per_pair(run(rand))
        cache[st] = (kl_a, kl_c, float(delta.norm(dim=-1).mean()))
        eff_all, ctl_all = fractions(kl_a, kl_c, idx_keep)
        by_strength[str(st)] = {
            "effect": eff_all,
            "control_effect": ctl_all,
            "margin": eff_all - ctl_all,
            "delta_norm": cache[st][2],
        }
        sel_eff, sel_ctl = fractions(kl_a, kl_c, select)
        if sel_eff - sel_ctl > best_margin:
            best_margin, best_s = sel_eff - sel_ctl, st

    kl_a, kl_c, dnorm = cache[best_s]
    eff, ctl = fractions(kl_a, kl_c, report)

    return SteeringResult(
        hook=hook,
        kl_before=float(kl0[report].mean()),
        kl_after=float(kl_a[report].mean()),
        kl_random_control=float(kl_c[report].mean()),
        effect=eff,
        control_effect=ctl,
        margin=eff - ctl,
        margin_stderr=bootstrap_margin_se(kl_a, kl_c, report),
        best_strength=float(best_s),
        delta_norm=dnorm,
        n_pairs=int(report.numel()),
        by_strength=by_strength,
    )
