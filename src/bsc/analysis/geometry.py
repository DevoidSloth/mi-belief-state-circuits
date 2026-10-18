"""Belief-state geometry: does the simplex show up in the residual stream?

For a non-unifilar process such as Mess3 the set of reachable beliefs is a
fractal subset of the probability simplex.  If a trained transformer represents
the belief linearly, a 2D projection of its residual stream is an affine image
of that fractal -- so the same structure should be visible in the activations,
not merely decodable from them.

The figures pair the two panels side by side and colour both by the *same* true
belief, so a reader can check the correspondence point by point instead of
taking a correlation coefficient on trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PCAResult:
    components: np.ndarray       # (k, d)
    mean: np.ndarray             # (d,)
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean) @ self.components.T


def pca(X: np.ndarray, k: int = 2) -> PCAResult:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(0)
    Xc = X - mean
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s**2 / max(len(X) - 1, 1)
    return PCAResult(
        components=vt[:k],
        mean=mean,
        explained_variance_ratio=var[:k] / var.sum(),
    )


def simplex_to_2d(belief: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of a 3-simplex point in the plane."""
    belief = np.asarray(belief, dtype=np.float64)
    if belief.shape[-1] != 3:
        raise ValueError("simplex_to_2d expects a 3-dimensional belief")
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return belief @ verts


def belief_to_rgb(belief: np.ndarray) -> np.ndarray:
    """Map a belief vector to a colour so both panels can share one encoding.

    3 coordinates map straight to RGB.  Otherwise the belief is projected to its
    top 3 principal components and rescaled, which keeps nearby beliefs in
    similar colours.
    """
    b = np.asarray(belief, dtype=np.float64)
    if b.shape[-1] == 3:
        rgb = b
    elif b.shape[-1] == 2:
        rgb = np.stack([b[..., 0], b[..., 1], np.zeros_like(b[..., 0])], axis=-1)
    elif b.shape[-1] == 1:
        v = b[..., 0]
        v = (v - v.min()) / max(np.ptp(v), 1e-12)
        from matplotlib import cm

        return cm.viridis(v)[..., :3]
    else:
        proj = pca(b.reshape(-1, b.shape[-1]), 3).transform(b.reshape(-1, b.shape[-1]))
        rgb = proj.reshape(*b.shape[:-1], 3)
    lo = rgb.reshape(-1, 3).min(0)
    hi = rgb.reshape(-1, 3).max(0)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-12), 0, 1)


@torch.no_grad()
def residual_projection(
    model,
    eval_set,
    hook: str = "hook_resid_final",
    burn_in: int = 16,
    k: int = 2,
    max_points: int = 40000,
    seed: int = 0,
) -> dict:
    """Project residual activations to k dimensions and align them with the belief."""
    from ..train.trainer import collect_residuals

    acts = collect_residuals(model, eval_set, [hook])[hook][:, burn_in:]
    belief = eval_set.belief[:, burn_in:].cpu().numpy()
    flat_acts = acts.reshape(-1, acts.shape[-1]).numpy()
    flat_belief = belief.reshape(-1, belief.shape[-1])

    rng = np.random.default_rng(seed)
    if len(flat_acts) > max_points:
        idx = rng.choice(len(flat_acts), max_points, replace=False)
        flat_acts, flat_belief = flat_acts[idx], flat_belief[idx]

    p = pca(flat_acts, k)
    return {
        "coords": p.transform(flat_acts),
        "belief": flat_belief,
        "explained_variance_ratio": p.explained_variance_ratio,
        "hook": hook,
        "pca": p,
    }


def belief_dimension_estimate(points: np.ndarray, n_scales: int = 12, seed: int = 0) -> dict:
    """Box-counting dimension of a point cloud.

    A fractal claim should come with a number.  For Mess3 the reachable belief set
    has non-integer dimension, and the same estimate applied to the residual-stream
    projection should agree if the representation is an affine image of it.
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts - pts.min(0)
    scale = np.maximum(pts.max(0), 1e-12)
    pts = pts / scale

    sizes = np.logspace(-2.2, -0.4, n_scales)
    counts = []
    for eps in sizes:
        cells = np.floor(pts / eps).astype(np.int64)
        counts.append(len({tuple(c) for c in cells}))
    counts = np.array(counts, dtype=np.float64)

    ok = counts > 1
    if ok.sum() < 3:
        return {"dimension": float("nan"), "sizes": sizes.tolist(), "counts": counts.tolist()}
    slope, intercept = np.polyfit(np.log(1 / sizes[ok]), np.log(counts[ok]), 1)
    resid = np.log(counts[ok]) - (slope * np.log(1 / sizes[ok]) + intercept)
    ss = 1.0 - resid.var() / max(np.log(counts[ok]).var(), 1e-12)
    return {
        "dimension": float(slope),
        "fit_r2": float(ss),
        "sizes": sizes.tolist(),
        "counts": counts.tolist(),
    }


def plot_belief_geometry(
    model,
    eval_set,
    process,
    path: str | Path,
    hook: str = "hook_resid_final",
    burn_in: int = 16,
    max_points: int = 20000,
    title: str | None = None,
) -> Path:
    """Two panels: the true belief set, and the residual stream projected to 2D.

    Both are coloured by the same belief, so the correspondence is checkable by
    eye rather than only through the probe's R^2.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = residual_projection(model, eval_set, hook, burn_in, k=2, max_points=max_points)
    belief = proj["belief"]
    colours = belief_to_rgb(belief)

    if belief.shape[-1] == 3:
        true_xy = simplex_to_2d(belief)
        true_label = "true belief simplex (barycentric)"
    elif belief.shape[-1] == 1:
        true_xy = np.stack([belief[:, 0], np.zeros_like(belief[:, 0])], axis=-1)
        true_label = "true belief (1-D latent)"
    else:
        true_xy = pca(belief, 2).transform(belief)
        true_label = f"true belief, top 2 PCs of {belief.shape[-1]}-D"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    for ax, xy, label in (
        (axes[0], true_xy, true_label),
        (axes[1], proj["coords"], f"residual stream at {hook}, top 2 PCs"),
    ):
        ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=colours, alpha=0.55, linewidths=0)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.25)

    evr = proj["explained_variance_ratio"]
    axes[1].set_xlabel(f"PC1+PC2 explain {100 * evr.sum():.1f}% of residual variance", fontsize=8)
    fig.suptitle(
        title or f"{process.name}: belief-state geometry in the residual stream", fontsize=12
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training_curves(histories: dict[str, list[dict]], path: str | Path) -> Path:
    """Optimality gap against training step, one line per run, log scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for name, hist in histories.items():
        steps = [h["step"] for h in hist]
        gaps = [max(h["gap"], 1e-6) for h in hist]
        ax.plot(steps, gaps, label=name, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss minus analytic optimum (nats/token)")
    ax.set_title("Distance to the exact optimal predictor")
    ax.axhline(0.01, color="k", linestyle=":", linewidth=1, label="0.01 nats target")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
