"""Experiment 4 - is the analytic Kalman gain present in the weights?

The optimal filter's coefficients are known exactly, so this is the one place in
the project where a claim about a *weight* can be checked against a number
rather than against a story.

The finding that shapes the experiment: a transformer does not implement the
recursion ``m_{t+1} = alpha m_t + aK y_t`` recurrently, because it has no state
to carry between positions.  It implements the unrolled convolution
``m_t = aK sum_d alpha^d y_{t-1-d}``.  So the object to compare against the
analytic gain is the model's impulse response, measured by autograd, not a
head's belief-to-belief transport.  Both are reported; the second is near zero
even in models whose kernel matches the analytic one to five decimal places,
which is the point.
"""

from __future__ import annotations

import time

import numpy as np

from ..analysis.kalman_gain import kalman_gain_report
from ..analysis.probes import probe_all_layers
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import best_probe, figure_path, train_cached

KALMAN_PROCESSES = ("kalman", "kalman_slow")


def _plot_kernel(rows: list[dict], path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 4.2), squeeze=False,
                             constrained_layout=True)
    for ax, row in zip(axes[0], rows, strict=False):
        k = np.array(row["kernel"])
        a = np.array(row["analytic_kernel"])
        lags = np.arange(len(k))
        ax.plot(lags, a, "o-", label="analytic  $aK\\,\\alpha^d$", linewidth=1.6, markersize=4)
        ax.plot(lags, k, "s--", label="measured  $\\partial \\hat m_t/\\partial y_{t-1-d}$",
                linewidth=1.4, markersize=3.5)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("lag $d$")
        ax.set_ylabel("gain")
        ax.set_title(
            f"{row['process']} (seed {row['seed']})\ncosine = {row['kernel_cosine']:.5f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(
    processes: tuple[str, ...] = KALMAN_PROCESSES,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_lag: int = 20,
    verbose: bool = True,
) -> dict:
    rows = []
    t0 = time.time()
    for name in processes:
        for seed in seeds:
            out = train_cached(name, seed=seed)
            model, eval_set, process = out["model"], out["eval_set"], out["process"]
            burn_in = max(out["train_cfg"].burn_in, 24)

            probes = probe_all_layers(model, eval_set, burn_in=burn_in)
            hook, probe = best_probe(probes, exclude_final=True)
            report = kalman_gain_report(
                model, process, eval_set, probe, hook,
                burn_in=burn_in, max_lag=max_lag, with_attribution=(seed == seeds[0]),
            )
            row = {"process": name, "seed": seed, "gap": out["final"].gap, **report.as_dict()}
            rows.append(row)
            if verbose:
                print(
                    f"  {name:12s} seed={seed} gap={out['final'].gap:+.5f} "
                    f"probe R2={probe.r2:.4f} kernel cos={report.kernel_cosine:.5f} "
                    f"decay {report.kernel_decay_rate:.4f} vs {report.analytic_decay_rate:.4f} "
                    f"({100 * report.decay_rate_rel_error:.1f}% err)",
                    flush=True,
                )

    figure = _plot_kernel(
        [r for r in rows if r["seed"] == seeds[0]], figure_path("exp04_kalman_kernel.png")
    )
    payload = {
        "experiment": "exp04_kalman",
        "seeds": list(seeds),
        "rows": rows,
        "figure": str(figure),
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": summarise(rows),
    }
    save_json(RESULTS_DIR / "exp04_kalman.json", payload)
    return payload


def summarise(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["process"], []).append(r)
    out = {}
    for name, group in by.items():
        cos = np.array([r["kernel_cosine"] for r in group])
        decay_err = np.array([r["decay_rate_rel_error"] for r in group])
        alpha_err = np.array([r["recursion"]["alpha_rel_error"] for r in group])
        beta_err = np.array([r["recursion"]["beta_rel_error"] for r in group])
        out[name] = {
            "n_seeds": len(group),
            "kernel_cosine_mean": float(cos.mean()),
            "kernel_cosine_min": float(cos.min()),
            "decay_rate_rel_error_mean": float(decay_err.mean()),
            "recursion_alpha_rel_error_mean": float(alpha_err.mean()),
            "recursion_beta_rel_error_mean": float(beta_err.mean()),
            "analytic": group[0]["analytic"],
            "seeds_with_cosine_over_0.99": int((cos > 0.99).sum()),
        }
    return out
