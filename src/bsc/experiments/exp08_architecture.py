"""Experiment 8 - depth, width, and how context length interacts with achievable loss.

Two sweeps that answer different questions:

**Depth x width.**  Exact belief tracking needs both somewhere to *put* the
belief (width) and enough serial steps to *update* it (depth).  The grid
separates the two: if one layer suffices at any width, the update is a single
step; if depth buys more than width, it is not.

**Context length.**  For a mixing process the achievable loss plateaus at the
entropy rate and more context is worthless past the mixing time.  For a
**non-mixing** process such as ``persistent_bit`` the belief never forgets, only
sharpens, so the optimal loss keeps falling with position and there is no
entropy rate to reach.  The comparison worth reporting is not the model's loss
curve on its own but its curve *against the exact optimal one*: a model that has
learned to accumulate evidence tracks the fall, and a model that has learned a
fixed filter flattens out early.
"""

from __future__ import annotations

import time

import numpy as np

from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import figure_path, train_cached

GRID_PROCESS = "mess3_hard"
DEPTHS = (1, 2, 4)
WIDTHS = (16, 32, 64, 128)

CONTEXT_PROCESSES = ("persistent_bit", "drifting_coin", "even")
CONTEXTS = (24, 48, 96)


def _grid(seed: int, verbose: bool) -> list[dict]:
    rows = []
    for layers in DEPTHS:
        for d in WIDTHS:
            heads = max(1, min(4, d // 8))
            out = train_cached(
                GRID_PROCESS,
                seed=seed,
                model_overrides={
                    "n_layers": layers,
                    "d_model": d,
                    "n_heads": heads,
                    "d_head": max(2, d // heads),
                },
            )
            f = out["final"]
            rows.append(
                {
                    "n_layers": layers,
                    "d_model": d,
                    "n_params": out["n_params"],
                    "gap": f.gap,
                    "gap_stderr": f.gap_stderr,
                    "kl_to_optimal": f.kl_to_optimal,
                    "oracle_loss": f.oracle_loss,
                }
            )
            if verbose:
                print(
                    f"  {GRID_PROCESS} layers={layers} d_model={d:4d} "
                    f"params={out['n_params']:7d} gap={f.gap:+.5f}",
                    flush=True,
                )
    return rows


def _context(seed: int, verbose: bool) -> list[dict]:
    rows = []
    for name in CONTEXT_PROCESSES:
        for ctx in CONTEXTS:
            out = train_cached(name, seed=seed, train_overrides={"ctx": ctx})
            f = out["final"]
            model_curve = np.array(f.loss_per_position)
            oracle_curve = np.array(f.oracle_per_position)
            burn_in = out["train_cfg"].burn_in
            tail = max(burn_in + 1, len(model_curve) - 16)
            # Early means before context has accumulated. Starting the window at
            # the scoring burn-in would begin after the fall and report no gain.
            early = slice(1, 9)
            rows.append(
                {
                    "process": name,
                    "ctx": ctx,
                    "gap": f.gap,
                    "gap_stderr": f.gap_stderr,
                    "loss_curve_model": model_curve.tolist(),
                    "loss_curve_oracle": oracle_curve.tolist(),
                    "oracle_early": float(oracle_curve[early].mean()),
                    "oracle_late": float(oracle_curve[tail:].mean()),
                    "model_early": float(model_curve[early].mean()),
                    "model_late": float(model_curve[tail:].mean()),
                }
            )
            r = rows[-1]
            r["oracle_context_gain"] = r["oracle_early"] - r["oracle_late"]
            r["model_context_gain"] = r["model_early"] - r["model_late"]
            r["tracks_context_gain"] = (
                r["model_context_gain"] / r["oracle_context_gain"]
                if abs(r["oracle_context_gain"]) > 1e-4
                else None
            )
            if verbose:
                print(
                    f"  {name:16s} ctx={ctx:4d} gap={f.gap:+.5f} "
                    f"context gain: optimal {r['oracle_context_gain']:+.4f} "
                    f"model {r['model_context_gain']:+.4f}",
                    flush=True,
                )
    return rows


def _plot(grid: list[dict], context: list[dict], path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    for layers in sorted({r["n_layers"] for r in grid}):
        pts = sorted((r for r in grid if r["n_layers"] == layers), key=lambda r: r["d_model"])
        axes[0].plot(
            [p["d_model"] for p in pts],
            [max(p["gap"], 1e-6) for p in pts],
            "o-",
            label=f"{layers} layer" + ("s" if layers > 1 else ""),
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$d_{model}$")
    axes[0].set_ylabel("optimality gap (nats/token)")
    axes[0].set_title(f"depth x width on {GRID_PROCESS}", fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    longest = {}
    for r in context:
        if r["ctx"] >= longest.get(r["process"], {"ctx": -1})["ctx"]:
            longest[r["process"]] = r
    for name, r in longest.items():
        pos = np.arange(len(r["loss_curve_model"]))
        line, = axes[1].plot(pos, r["loss_curve_oracle"], linewidth=1.6, label=f"{name}: optimal")
        axes[1].plot(
            pos, r["loss_curve_model"], "--", linewidth=1.2, color=line.get_color(),
            label=f"{name}: model",
        )
    axes[1].set_xlabel("position in sequence")
    axes[1].set_ylabel("loss (nats/token)")
    axes[1].set_title("does achievable loss keep falling with context?", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(seed: int = 0, verbose: bool = True) -> dict:
    t0 = time.time()
    grid = _grid(seed, verbose)
    context = _context(seed, verbose)
    figure = _plot(grid, context, figure_path("exp08_architecture.png"))

    payload = {
        "experiment": "exp08_architecture",
        "seed": seed,
        "grid_process": GRID_PROCESS,
        "grid": grid,
        "context": context,
        "figure": str(figure),
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": summarise(grid, context),
    }
    save_json(RESULTS_DIR / "exp08_architecture.json", payload)
    return payload


def summarise(grid: list[dict], context: list[dict]) -> dict:
    best = min(grid, key=lambda r: r["gap"])
    # smallest configuration within 0.01 nats, by parameter count
    good = [r for r in grid if r["gap"] < 0.01]
    smallest = min(good, key=lambda r: r["n_params"]) if good else None

    by_depth = {}
    for layers in sorted({r["n_layers"] for r in grid}):
        rows = [r for r in grid if r["n_layers"] == layers]
        by_depth[layers] = {
            "best_gap": min(r["gap"] for r in rows),
            "best_d_model": min(rows, key=lambda r: r["gap"])["d_model"],
        }

    ctx_summary = {}
    for name in {r["process"] for r in context}:
        rows = sorted((r for r in context if r["process"] == name), key=lambda r: r["ctx"])
        ctx_summary[name] = {
            "oracle_context_gain_by_ctx": {r["ctx"]: r["oracle_context_gain"] for r in rows},
            "model_context_gain_by_ctx": {r["ctx"]: r["model_context_gain"] for r in rows},
            "tracking_ratio_at_longest": rows[-1]["tracks_context_gain"],
            "gap_by_ctx": {r["ctx"]: r["gap"] for r in rows},
            "non_mixing": rows[-1]["oracle_context_gain"] > 0.01,
        }

    return {
        "best": {k: best[k] for k in ("n_layers", "d_model", "n_params", "gap")},
        "smallest_within_0.01_nats": (
            {k: smallest[k] for k in ("n_layers", "d_model", "n_params", "gap")}
            if smallest
            else None
        ),
        "best_gap_by_depth": by_depth,
        "context": ctx_summary,
    }
