"""Experiment 5 - what an SAE does to a variable that is known to be dense.

SAEs presuppose sparse, roughly monosemantic features.  A belief state is a point
on a simplex: continuous, and with every coordinate active essentially always.
Whether SAE features capture such a variable is normally unfalsifiable for want
of ground truth.  Here the ground truth is exact, so the question closes.

The comparison is deliberately generous to the SAE.  Its dictionary is learned
without ever seeing the belief, but the ``k`` features it is scored on are
chosen *using* the belief, by greedy forward selection.  It is measured against
``k`` unsupervised principal components of the same activations, and against
reduced-rank regression, which upper-bounds what any ``k`` directions could do.
A shortfall against the PCA column under those conditions is a real shortfall.

Reported alongside are the standard SAE health numbers - L0, variance explained,
downstream loss recovered - so a reader can confirm the autoencoder is not simply
broken.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..analysis.probes import probe_all_layers
from ..analysis.sae import SAEConfig, belief_capture, sae_metrics, train_sae
from ..train.trainer import collect_residuals
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import best_probe, figure_path, train_cached

SAE_PROCESSES = ("mess3_hard", "mess3", "sns", "random_non", "regimevol")
# The MSE term is summed over d_model, so it is O(d_model) while the L1 term
# is O(L0 * typical activation). Coefficients below ~0.1 therefore do not
# sparsify at all at these widths -- the earlier grid left L0 above 100.
L1_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


def _plot_capture(rows: list[dict], path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 4.0), squeeze=False,
                             constrained_layout=True)
    for ax, row in zip(axes[0], rows, strict=False):
        cap = row["best"]["capture"]["rows"]
        ks = [c["k"] for c in cap]
        ax.plot(ks, [c["sae_r2"] for c in cap], "o-", label="top-k SAE features")
        ax.plot(ks, [c["pca_r2"] for c in cap], "s--", label="top-k PCA components")
        ax.plot(ks, [c["rrr_r2"] for c in cap], "^:", label="best k-dim subspace (RRR)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k directions")
        ax.set_ylabel("belief variance explained ($R^2$)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{row['process']}\nL0={row['best']['metrics']['l0']:.1f}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(
    processes: tuple[str, ...] = SAE_PROCESSES,
    seed: int = 0,
    l1_grid: tuple[float, ...] = L1_GRID,
    expansion: int = 8,
    steps: int = 4000,
    ks=(1, 2, 4, 8, 16, 32),
    verbose: bool = True,
) -> dict:
    rows = []
    t0 = time.time()
    for name in processes:
        out = train_cached(name, seed=seed)
        model, eval_set = out["model"], out["eval_set"]
        burn_in = out["train_cfg"].burn_in

        probes = probe_all_layers(model, eval_set, burn_in=burn_in)
        hook, probe = best_probe(probes, exclude_final=True)
        acts = collect_residuals(model, eval_set, [hook])[hook].to(eval_set.obs.device)

        variants = []
        for l1 in l1_grid:
            cfg = SAEConfig(
                d_in=acts.shape[-1], expansion=expansion, l1_coeff=l1, steps=steps, seed=seed
            )
            sae, stats = train_sae(acts[:, burn_in:], cfg, device=eval_set.obs.device)
            metrics = sae_metrics(sae, stats, acts, model, eval_set, hook, burn_in=burn_in)
            capture = belief_capture(
                sae, stats, acts, eval_set.belief, ks=ks, burn_in=burn_in, seed=seed
            )
            variants.append(
                {"l1_coeff": l1, "metrics": metrics.as_dict(), "capture": capture}
            )
            if verbose:
                k8 = next(c for c in capture["rows"] if c["k"] == 8)
                print(
                    f"  {name:12s} l1={l1:<6g} L0={metrics.l0:5.1f} "
                    f"var_exp={metrics.variance_explained:.3f} "
                    f"loss_rec={metrics.loss_recovered:.3f} | "
                    f"k=8: sae R2={k8['sae_r2']:.3f} pca R2={k8['pca_r2']:.3f} "
                    f"rrr R2={k8['rrr_r2']:.3f}",
                    flush=True,
                )
            del sae
            torch.cuda.empty_cache()

        # Pick the variant at the conventional healthy operating point (L0 in
        # 10-60); if the grid never gets there, take the closest and say so,
        # rather than silently reporting a dictionary that is barely sparse.
        lo, hi = 10.0, 60.0
        healthy = [v for v in variants if lo <= v["metrics"]["l0"] <= hi]
        if healthy:
            best = max(healthy, key=lambda v: v["metrics"]["loss_recovered"])
        else:
            best = min(
                variants,
                key=lambda v: min(abs(v["metrics"]["l0"] - lo), abs(v["metrics"]["l0"] - hi)),
            )
        rows.append(
            {
                "process": name,
                "hook": hook,
                "probe_r2": probe.r2,
                "d_belief": out["process"].d_belief,
                "variants": variants,
                "best": best,
                "reached_healthy_l0_band": bool(healthy),
                "l0_band": [lo, hi],
            }
        )

    figure = _plot_capture(rows, figure_path("exp05_sae_capture.png"))
    payload = {
        "experiment": "exp05_sae",
        "seed": seed,
        "rows": rows,
        "figure": str(figure),
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": summarise(rows),
    }
    save_json(RESULTS_DIR / "exp05_sae.json", payload)
    return payload


def summarise(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        best = row["best"]
        cap = {c["k"]: c for c in best["capture"]["rows"]}
        k = 8 if 8 in cap else max(cap)
        c = cap[k]
        out[row["process"]] = {
            "l1_coeff": best["l1_coeff"],
            "l0": best["metrics"]["l0"],
            "variance_explained": best["metrics"]["variance_explained"],
            "loss_recovered": best["metrics"]["loss_recovered"],
            "dead_fraction": best["metrics"]["dead_fraction"],
            "ablation_effect": best["metrics"]["ablation_effect"],
            "d_sae": best["metrics"]["d_sae"],
            "k": k,
            "sae_r2_at_k": c["sae_r2"],
            "pca_r2_at_k": c["pca_r2"],
            "rrr_r2_at_k": c["rrr_r2"],
            "sae_minus_pca": c["sae_r2"] - c["pca_r2"],
            "sae_minus_rrr": c["sae_r2"] - c["rrr_r2"],
            "full_linear_r2": best["capture"]["full_linear_r2"],
            "n_live_features": best["capture"]["n_live_features"],
            "reached_healthy_l0_band": row["reached_healthy_l0_band"],
        }
    gaps = [v["sae_minus_pca"] for v in out.values()]
    out["_aggregate"] = {
        "median_sae_minus_pca": float(np.median(gaps)),
        "n_processes_where_sae_loses_to_pca": int(sum(g < 0 for g in gaps)),
        "n_processes": len(gaps),
    }
    return out
