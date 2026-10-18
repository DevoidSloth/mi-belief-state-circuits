"""Experiment 1 - how close does a small transformer get to the exact optimum?

Reports, for every process in the zoo:

* the **optimality gap**: model loss minus oracle loss on identical sequences.
  Paired, so its standard error is small enough to resolve 1e-3 nats.
* the **baseline separation**: how far the best finite-memory model (an n-gram
  table, or a linear readout of a fixed window) sits above the optimum.  This is
  the number that says whether the task required a belief state at all.  A
  process where the n-gram gap is near zero cannot support any claim about
  Bayesian inference, and is reported as such rather than quietly dropped.
* an RNN trained identically, since a GRU *can* express a filter.  Where a
  transformer closes the gap and a GRU does not, that is a fact about
  architecture rather than about the task.
"""

from __future__ import annotations

import time

import numpy as np

from ..model.baselines import GRUBaseline, LinearBaseline, NGramBaseline
from ..processes import DEFAULT_ZOO
from ..processes.hmm import HMM
from ..train.trainer import train_model
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import recipe, train_cached


def _process_facts(process) -> dict:
    facts = {
        "name": process.name,
        "kind": process.kind,
        "d_belief": process.d_belief,
        "head": process.head,
        "vocab_size": process.vocab_size,
        "entropy_rate_closed_form": process.entropy_rate(),
    }
    if isinstance(process, HMM):
        facts.update(
            {
                "n_states": process.n_states,
                "unifilar": process.is_unifilar(),
                "irreducible": process.is_irreducible(),
                "slem": process.slem(),
                "entropy_rate_msp": process.entropy_rate_msp(),
            }
        )
    return facts


def _ngram_baseline(process, ctx: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    train = process.sample(4000, ctx, rng)
    val = process.sample(1000, ctx, rng)
    test = process.sample(1000, ctx, rng)
    model, info = NGramBaseline.fit_best(train, val, process.vocab_size)
    oracle = float(-process.filter(test).logp.mean())
    return {
        "loss": model.loss(test),
        "oracle_loss": oracle,
        "gap": model.loss(test) - oracle,
        "order": info["order"],
        "alpha": info["alpha"],
    }


def _neural_baseline(kind: str, process, mcfg, tcfg, seed: int) -> dict:
    cfg = type(mcfg)(**{**mcfg.__dict__, "seed": seed})
    model = GRUBaseline(cfg) if kind == "gru" else LinearBaseline(cfg, window=8)
    out = train_model(process, model=model, cfg=tcfg)
    f = out["final"]
    return {
        "loss": f.loss,
        "oracle_loss": f.oracle_loss,
        "gap": f.gap,
        "gap_stderr": f.gap_stderr,
        "n_params": out["n_params"],
    }


def run(
    processes: tuple[str, ...] = DEFAULT_ZOO,
    seed: int = 0,
    with_baselines: bool = True,
    verbose: bool = True,
) -> dict:
    rows = []
    histories = {}
    t0 = time.time()
    for name in processes:
        out = train_cached(name, seed=seed)
        process, f = out["process"], out["final"]
        histories[name] = out["history"]

        row = {
            "process": name,
            "facts": _process_facts(process),
            "n_params": out["n_params"],
            "d_model": out["model_cfg"].d_model,
            "n_layers": out["model_cfg"].n_layers,
            "transformer": {
                "loss": f.loss,
                "oracle_loss": f.oracle_loss,
                "gap": f.gap,
                "gap_stderr": f.gap_stderr,
                "kl_to_optimal": f.kl_to_optimal,
            },
            "wall_time_s": out["wall_time_s"],
            "cached": out["cached"],
        }

        if with_baselines:
            _, mcfg, tcfg = recipe(name, seed)
            if process.kind == "discrete":
                row["ngram"] = _ngram_baseline(process, tcfg.ctx, seed + 100)
            row["linear"] = _neural_baseline("linear", process, mcfg, tcfg, seed)
            row["gru"] = _neural_baseline("gru", process, mcfg, tcfg, seed)
            finite = row.get("ngram", row["linear"])
            row["baseline_separation"] = finite["gap"]

        rows.append(row)
        if verbose:
            sep = row.get("baseline_separation", float("nan"))
            print(
                f"  {name:16s} gap={f.gap:+.5f}+-{f.gap_stderr:.5f} "
                f"oracle={f.oracle_loss:.4f} finite-memory separation={sep:+.4f}",
                flush=True,
            )

    payload = {
        "experiment": "exp01_optimality",
        "seed": seed,
        "rows": rows,
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": summarise(rows),
    }
    save_json(RESULTS_DIR / "exp01_optimality.json", payload)

    from ..analysis.geometry import plot_training_curves
    from .common import figure_path

    plot_training_curves(histories, figure_path("exp01_optimality_gap.png"))
    return payload


def summarise(rows: list[dict]) -> dict:
    gaps = np.array([r["transformer"]["gap"] for r in rows])
    seps = np.array([r.get("baseline_separation", np.nan) for r in rows], dtype=float)
    non_trivial = [
        r["process"] for r in rows if r.get("baseline_separation", 0.0) > 0.15
    ]
    return {
        "n_processes": len(rows),
        "median_gap": float(np.median(gaps)),
        "max_gap": float(gaps.max()),
        "n_within_0.01_nats": int((gaps < 0.01).sum()),
        "n_within_0.005_nats": int((gaps < 0.005).sum()),
        "median_baseline_separation": float(np.nanmedian(seps)),
        "processes_with_separation_over_0.15": non_trivial,
        "n_processes_with_separation_over_0.15": len(non_trivial),
    }
