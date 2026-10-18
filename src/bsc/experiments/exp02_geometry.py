"""Experiment 2 - where the belief state lives, and whether the model uses it.

Two halves, and the second is the point:

* **Representation.** Linear probes from every residual-stream location to the
  exact posterior, plus a box-counting dimension of the projected activations
  against the same statistic on the true belief set.  For Mess3 the reachable
  belief set is a fractal, so a matching non-integer dimension is a much
  stronger claim than "the scatter plots look alike".
* **Use.** The steering test from :mod:`bsc.analysis.probes`, run at every layer.
  Reported side by side with R^2, because the two do not have to agree - and
  where they disagree, the probe is the thing that is wrong.
"""

from __future__ import annotations

import time

import numpy as np

from ..analysis.geometry import (
    belief_dimension_estimate,
    plot_belief_geometry,
    residual_projection,
    simplex_to_2d,
)
from ..analysis.probes import probe_all_layers, steering_test
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import figure_path, train_cached

GEOMETRY_PROCESSES = (
    "mess3",
    "mess3_hard",
    "even",
    "rrxor",
    "sns",
    "drifting_coin",
    "random_non",
    "kalman",
    "garch",
    "regimevol",
)

#: processes whose reachable belief set is genuinely fractal
FRACTAL_PROCESSES = ("mess3", "mess3_hard", "sns", "random_non")


def run(
    processes: tuple[str, ...] = GEOMETRY_PROCESSES,
    seed: int = 0,
    n_pairs: int = 2048,
    make_figures: bool = True,
    verbose: bool = True,
) -> dict:
    rows = []
    t0 = time.time()
    for name in processes:
        out = train_cached(name, seed=seed)
        model, eval_set, process = out["model"], out["eval_set"], out["process"]
        burn_in = out["train_cfg"].burn_in

        probes = probe_all_layers(model, eval_set, burn_in=burn_in)
        layers = []
        for hook, probe in probes.items():
            steer = steering_test(
                model, process, eval_set, probe, hook, n_pairs=n_pairs, burn_in=burn_in, seed=seed
            )
            layers.append(
                {
                    "hook": hook,
                    **probe.as_dict(),
                    "steering": steer.as_dict(),
                }
            )

        internal = [r for r in layers if r["hook"] != "hook_resid_final"]
        best_r2 = max(internal, key=lambda r: r["r2"])
        best_causal = max(internal, key=lambda r: r["steering"]["margin"])

        row = {
            "process": name,
            "d_belief": process.d_belief,
            "belief_kind": process.belief_kind,
            "gap": out["final"].gap,
            "layers": layers,
            "best_r2_hook": best_r2["hook"],
            "best_r2": best_r2["r2"],
            "best_causal_hook": best_causal["hook"],
            "best_causal_margin": best_causal["steering"]["margin"],
            "r2_at_best_causal_hook": best_causal["r2"],
            "causal_margin_at_best_r2_hook": best_r2["steering"]["margin"],
        }

        # rank correlation between "decodable" and "used" across layers
        r2s = np.array([r["r2"] for r in internal])
        margins = np.array([r["steering"]["margin"] for r in internal])
        if len(r2s) > 2 and r2s.std() > 1e-9 and margins.std() > 1e-9:
            row["r2_vs_margin_pearson"] = float(np.corrcoef(r2s, margins)[0, 1])
            row["r2_vs_margin_spearman"] = float(
                np.corrcoef(np.argsort(np.argsort(r2s)), np.argsort(np.argsort(margins)))[0, 1]
            )

        if name in FRACTAL_PROCESSES or process.d_belief == 3:
            proj = residual_projection(
                model, eval_set, best_r2["hook"], burn_in=burn_in, max_points=60000
            )
            true_xy = (
                simplex_to_2d(proj["belief"])
                if proj["belief"].shape[-1] == 3
                else proj["belief"][:, :2]
            )
            row["box_dimension"] = {
                "true_belief_set": belief_dimension_estimate(true_xy),
                "residual_projection": belief_dimension_estimate(proj["coords"]),
                "explained_variance_ratio": proj["explained_variance_ratio"].tolist(),
            }

        if make_figures:
            row["figure"] = str(
                plot_belief_geometry(
                    model,
                    eval_set,
                    process,
                    figure_path(f"exp02_geometry_{name}.png"),
                    hook=best_r2["hook"],
                    burn_in=burn_in,
                    max_points=40000,
                )
            )

        rows.append(row)
        if verbose:
            print(
                f"  {name:16s} best R2={row['best_r2']:.4f} @ {row['best_r2_hook'][:22]:22s} "
                f"best causal margin={row['best_causal_margin']:+.3f} @ "
                f"{row['best_causal_hook'][:22]}",
                flush=True,
            )

    payload = {
        "experiment": "exp02_geometry",
        "seed": seed,
        "rows": rows,
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": summarise(rows),
    }
    save_json(RESULTS_DIR / "exp02_geometry.json", payload)
    return payload


def summarise(rows: list[dict]) -> dict:
    r2 = np.array([r["best_r2"] for r in rows])
    margin = np.array([r["best_causal_margin"] for r in rows])
    anti = [
        r["process"]
        for r in rows
        if r.get("r2_vs_margin_pearson") is not None and r.get("r2_vs_margin_pearson", 0) < 0
    ]
    dims = {
        r["process"]: {
            "true": r["box_dimension"]["true_belief_set"]["dimension"],
            "residual": r["box_dimension"]["residual_projection"]["dimension"],
        }
        for r in rows
        if "box_dimension" in r
    }
    return {
        "n_processes": len(rows),
        "median_best_r2": float(np.median(r2)),
        "min_best_r2": float(r2.min()),
        "median_best_causal_margin": float(np.median(margin)),
        "processes_where_r2_and_causal_use_anticorrelate": anti,
        "box_dimensions": dims,
    }
