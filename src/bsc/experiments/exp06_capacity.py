"""Experiment 6 - shrink the model until exact belief tracking is impossible.

Exact inference on a non-unifilar process needs an unbounded-precision point on
a simplex.  A small residual stream cannot hold one, so something else must
happen.  This experiment measures *what*, by testing three concrete fallback
hypotheses against the model's own predictions rather than describing the
degradation qualitatively:

**Finite-memory truncation.**  Fit a table indexed by the last ``k``
observations to the model's *own* predictive distributions and measure the
residual KL.  The smallest ``k`` that explains the model is its effective Markov
order.  A model that has collapsed to a bigram will show ``k = 2`` with near-zero
residual - which is a precise statement, unlike "it got worse".

**Belief quantisation.**  Cluster the model's predictive distributions and find
how few clusters reproduce them.  A model that has discretised the simplex shows
a small, sharply-defined cluster count.

**Low-rank projection.**  Participation ratio of the predictive-distribution
covariance: how many dimensions of the simplex the model actually uses.

The capacity threshold is reported as the largest ``d_model`` at which the
optimality gap first exceeds a stated criterion, together with which fallback
best describes the model below it.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..analysis.probes import probe_all_layers
from ..model.heads import head_distribution
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import best_probe, figure_path, train_cached

CAPACITY_PROCESSES = ("mess3_hard", "sns", "random_non")
WIDTHS = (4, 8, 12, 16, 24, 32, 64)


@torch.no_grad()
def model_predictive(model, eval_set, burn_in: int) -> np.ndarray:
    """The model's predictive distribution at every scored position, shape (N, V)."""
    dist = head_distribution(model.cfg.head, model(eval_set.obs), model.cfg.n_mixture)
    if "log_pmf" not in dist:
        raise ValueError("fallback diagnostics are defined for categorical heads")
    pmf = dist["log_pmf"].exp()[:, burn_in:]
    return pmf.reshape(-1, pmf.shape[-1]).float().cpu().numpy()


def effective_markov_order(
    model, eval_set, burn_in: int, vocab: int, max_order: int = 5
) -> dict:
    """Smallest context length that explains the model's own predictions.

    Fits, for each order k, the mean model-predictive distribution per length-k
    context, then reports the residual KL from the model to that table.
    """
    pmf = model_predictive(model, eval_set, burn_in)
    obs = eval_set.obs_np[:, burn_in:]
    n, length = obs.shape
    full = eval_set.obs_np
    rows = []
    for k in range(0, max_order + 1):
        if k == 0:
            table = pmf.mean(0, keepdims=True)
            idx = np.zeros(len(pmf), dtype=np.int64)
        else:
            padded = np.concatenate([np.full((n, k), vocab), full], axis=1)
            ctx = np.zeros((n, length), dtype=np.int64)
            base = 1
            start = burn_in + k
            for j in range(k):
                sl = padded[:, start - 1 - j : start - 1 - j + length]
                ctx += sl * base
                base *= vocab + 1
            idx = ctx.ravel()
            n_ctx = (vocab + 1) ** k
            sums = np.zeros((n_ctx, pmf.shape[1]))
            counts = np.zeros(n_ctx)
            np.add.at(sums, idx, pmf)
            np.add.at(counts, idx, 1.0)
            table = sums / np.maximum(counts, 1)[:, None]
            table = np.where(counts[:, None] > 0, table, pmf.mean(0, keepdims=True))
        approx = np.clip(table[idx], 1e-12, None)
        approx /= approx.sum(-1, keepdims=True)
        kl = float((pmf * (np.log(np.clip(pmf, 1e-12, None)) - np.log(approx))).sum(-1).mean())
        rows.append({"order": k, "residual_kl": kl})
    best = min(rows, key=lambda r: r["residual_kl"])
    explained = next((r for r in rows if r["residual_kl"] < 0.002), best)
    return {"by_order": rows, "effective_order": explained["order"],
            "residual_kl_at_effective_order": explained["residual_kl"]}


def belief_quantisation(model, eval_set, burn_in: int, max_clusters: int = 32,
                        seed: int = 0) -> dict:
    """How few clusters reproduce the model's predictive distributions."""
    pmf = model_predictive(model, eval_set, burn_in)
    rng = np.random.default_rng(seed)
    sample = pmf[rng.choice(len(pmf), min(20000, len(pmf)), replace=False)]
    total = ((sample - sample.mean(0)) ** 2).sum()
    rows = []
    for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, max_clusters):
        centres = sample[rng.choice(len(sample), k, replace=False)]
        for _ in range(25):
            assign = np.argmin(((sample[:, None] - centres[None]) ** 2).sum(-1), axis=1)
            for c in range(k):
                m = assign == c
                if m.any():
                    centres[c] = sample[m].mean(0)
        resid = ((sample - centres[assign]) ** 2).sum()
        rows.append({"clusters": k, "variance_explained": float(1 - resid / max(total, 1e-12))})
    enough = next((r["clusters"] for r in rows if r["variance_explained"] > 0.95), None)
    return {"by_clusters": rows, "clusters_for_95pct": enough}


def effective_rank(model, eval_set, burn_in: int) -> float:
    """Participation ratio of the predictive-distribution covariance spectrum."""
    pmf = model_predictive(model, eval_set, burn_in)
    c = np.cov((pmf - pmf.mean(0)).T)
    ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(c)), 0, None)
    return float(ev.sum() ** 2 / max((ev**2).sum(), 1e-24))


def run(
    processes: tuple[str, ...] = CAPACITY_PROCESSES,
    widths: tuple[int, ...] = WIDTHS,
    seed: int = 0,
    criterion: float = 0.02,
    verbose: bool = True,
) -> dict:
    rows = []
    t0 = time.time()
    for name in processes:
        oracle_ref = None
        points = []
        for d in widths:
            heads = max(1, min(4, d // 4))
            out = train_cached(
                name,
                seed=seed,
                model_overrides={"d_model": d, "n_heads": heads, "d_head": max(2, d // heads)},
            )
            model, eval_set = out["model"], out["eval_set"]
            burn_in = out["train_cfg"].burn_in
            oracle_ref = out["final"].oracle_loss

            probes = probe_all_layers(model, eval_set, burn_in=burn_in)
            _, probe = best_probe(probes, exclude_final=True)
            point = {
                "d_model": d,
                "n_params": out["n_params"],
                "gap": out["final"].gap,
                "gap_stderr": out["final"].gap_stderr,
                "kl_to_optimal": out["final"].kl_to_optimal,
                "probe_r2": probe.r2,
                "markov": effective_markov_order(model, eval_set, burn_in,
                                                 out["process"].vocab_size),
                "quantisation": belief_quantisation(model, eval_set, burn_in, seed=seed),
                "effective_rank": effective_rank(model, eval_set, burn_in),
            }
            points.append(point)
            if verbose:
                print(
                    f"  {name:12s} d_model={d:4d} gap={point['gap']:+.5f} "
                    f"probe R2={probe.r2:.3f} eff-order={point['markov']['effective_order']} "
                    f"clusters@95%={point['quantisation']['clusters_for_95pct']} "
                    f"eff-rank={point['effective_rank']:.2f}",
                    flush=True,
                )

        failing = [p for p in points if p["gap"] > criterion]
        threshold = max((p["d_model"] for p in failing), default=None)
        rows.append(
            {
                "process": name,
                "oracle_loss": oracle_ref,
                "criterion_nats": criterion,
                "points": points,
                "capacity_threshold_d_model": threshold,
                "fallback": _describe_fallback(points, threshold),
            }
        )

    figure = _plot(rows, figure_path("exp06_capacity.png"))
    payload = {
        "experiment": "exp06_capacity",
        "seed": seed,
        "widths": list(widths),
        "rows": rows,
        "figure": str(figure),
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": {
            r["process"]: {
                "capacity_threshold_d_model": r["capacity_threshold_d_model"],
                "fallback": r["fallback"],
            }
            for r in rows
        },
    }
    save_json(RESULTS_DIR / "exp06_capacity.json", payload)
    return payload


def _describe_fallback(points: list[dict], threshold: int | None) -> dict:
    """Compare the small models against the large ones on each diagnostic."""
    if threshold is None:
        return {"verdict": "no capacity threshold within the swept range"}
    small = [p for p in points if p["d_model"] <= threshold]
    large = [p for p in points if p["d_model"] > threshold]
    if not small or not large:
        return {"verdict": "threshold at the edge of the swept range"}

    def med(group, path):
        vals = [p[path[0]][path[1]] if len(path) > 1 else p[path[0]] for p in group]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    small_order = med(small, ("markov", "effective_order"))
    large_order = med(large, ("markov", "effective_order"))
    small_clusters = med(small, ("quantisation", "clusters_for_95pct"))
    large_clusters = med(large, ("quantisation", "clusters_for_95pct"))
    small_rank = med(small, ("effective_rank",))
    large_rank = med(large, ("effective_rank",))

    verdicts = []
    if small_order is not None and large_order is not None and small_order < large_order:
        verdicts.append(
            f"finite-memory truncation (effective Markov order {small_order:.0f} "
            f"below threshold vs {large_order:.0f} above)"
        )
    if (
        small_clusters is not None
        and large_clusters is not None
        and small_clusters < large_clusters
    ):
        verdicts.append(
            f"belief quantisation ({small_clusters:.0f} clusters explain 95% below "
            f"threshold vs {large_clusters:.0f} above)"
        )
    if small_rank is not None and large_rank is not None and small_rank < large_rank - 0.05:
        verdicts.append(
            f"low-rank projection (effective rank {small_rank:.2f} vs {large_rank:.2f})"
        )
    return {
        "verdict": "; ".join(verdicts) if verdicts else "no single fallback dominates",
        "median_effective_order_below": small_order,
        "median_effective_order_above": large_order,
        "median_clusters_below": small_clusters,
        "median_clusters_above": large_clusters,
        "median_effective_rank_below": small_rank,
        "median_effective_rank_above": large_rank,
    }


def _plot(rows: list[dict], path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    for row in rows:
        d = [p["d_model"] for p in row["points"]]
        axes[0].plot(d, [max(p["gap"], 1e-6) for p in row["points"]], "o-", label=row["process"])
        axes[1].plot(d, [p["probe_r2"] for p in row["points"]], "o-", label=row["process"])
        axes[2].plot(
            d, [p["markov"]["effective_order"] for p in row["points"]], "o-", label=row["process"]
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("optimality gap (nats/token)")
    axes[1].set_ylabel("belief probe $R^2$")
    axes[2].set_ylabel("effective Markov order of the model")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$d_{model}$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("What replaces exact belief tracking when capacity runs out")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
