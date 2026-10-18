"""Experiment 7 - one filter memorised, or inference over a family of processes?

Trained on a *family* of processes with one component drawn per sequence, a model
that has merely memorised an average filter and a model that infers the process
in context look identical at convergence.  They differ in how loss falls with
position, and in whether the component posterior is decodable.

Both are exactly known here, because the family is finite: the Bayes-optimal
predictor mixes the components' forward algorithms by their marginal likelihood.
So the experiment can ask three sharp questions:

1. Does the model's loss-versus-position curve track the optimal one, or flatten
   early like a fixed filter would?
2. Is the posterior *over components* linearly decodable, and does its accuracy
   rise with context the way the optimal one does?
3. Does the model generalise to a held-out parameter it never trained on -
   evidence that it inferred a process rather than selecting among memorised ones?
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..analysis.probes import probe_all_layers
from ..model.heads import head_logprob
from ..processes import get_process
from ..processes.mixture import MixtureOfProcesses
from ..processes.zoo import even_process, golden_mean, mess3
from ..train.data import make_eval_set
from ..train.trainer import evaluate
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import best_probe, figure_path, train_cached

FAMILIES = ("even_family", "golden_family", "mess3_family")

#: parameters deliberately *outside* the training family, to test interpolation
HELD_OUT = {
    "even_family": [even_process(0.35), even_process(0.65)],
    "golden_family": [golden_mean(0.28), golden_mean(0.52), golden_mean(0.78)],
    "mess3_family": [mess3(0.05, 0.65), mess3(0.05, 0.85)],
}


@torch.no_grad()
def _loss_curve(model, eval_set, burn_in: int) -> tuple[np.ndarray, np.ndarray]:
    params = model(eval_set.obs)
    lp = head_logprob(model.cfg.head, params, eval_set.obs, model.cfg.n_mixture)
    return (
        (-lp).mean(0).cpu().numpy(),
        (-eval_set.oracle_logp).mean(0).cpu().numpy(),
    )


@torch.no_grad()
def _component_identification(model, eval_set, probe, hook, burn_in: int) -> dict:
    """Does the decoded component posterior become accurate as context grows?"""
    from ..train.trainer import collect_residuals

    acts = collect_residuals(model, eval_set, [hook])[hook]
    decoded = probe.decode(acts).numpy()
    truth = eval_set.belief.cpu().numpy()
    true_arg = truth.argmax(-1)
    model_arg = decoded.argmax(-1)

    ctx = decoded.shape[1]
    bins = np.linspace(burn_in, ctx, 9).astype(int)
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        if hi <= lo:
            continue
        curve.append(
            {
                "position_lo": int(lo),
                "position_hi": int(hi),
                "agreement_with_optimal": float((model_arg[:, lo:hi] == true_arg[:, lo:hi]).mean()),
                "optimal_confidence": float(truth[:, lo:hi].max(-1).mean()),
                "decoded_confidence": float(
                    np.clip(decoded[:, lo:hi], 0, None).max(-1).mean()
                ),
            }
        )
    return {"curve": curve, "final_agreement": curve[-1]["agreement_with_optimal"] if curve else None}


@torch.no_grad()
def _held_out_transfer(model, family_name: str, ctx: int, burn_in: int, seed: int) -> list[dict]:
    """Evaluate on processes the family never contained."""
    out = []
    device = next(model.parameters()).device
    for proc in HELD_OUT.get(family_name, []):
        es = make_eval_set(proc, 512, ctx, seed + 999, device)
        m = evaluate(model, es, model.cfg.head, burn_in, model.cfg.n_mixture)
        # reference: the family's own optimal predictor scored on this off-family data
        family = get_process(family_name)
        family_oracle = float(-family.filter(es.obs_np).logp[:, burn_in:].mean())
        out.append(
            {
                "process": proc.name,
                "model_loss": m.loss,
                "true_oracle_loss": m.oracle_loss,
                "gap_to_true_oracle": m.gap,
                "family_oracle_loss": family_oracle,
                "gap_to_family_oracle": m.loss - family_oracle,
            }
        )
    return out


def _plot(rows: list[dict], path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows), figsize=(5.0 * len(rows), 4.2), squeeze=False,
                             constrained_layout=True)
    for ax, row in zip(axes[0], rows, strict=False):
        model_curve = np.array(row["loss_curve_model"])
        oracle_curve = np.array(row["loss_curve_oracle"])
        pos = np.arange(len(model_curve))
        ax.plot(pos, oracle_curve, label="Bayes-optimal over the family", linewidth=1.6)
        ax.plot(pos, model_curve, "--", label="model", linewidth=1.4)
        ax.axhline(row["best_single_component_loss"], color="k", linestyle=":", linewidth=1,
                   label="best single fixed filter")
        ax.set_xlabel("position in sequence")
        ax.set_ylabel("loss (nats/token)")
        ax.set_title(row["family"], fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.suptitle("In-context inference of the process: loss must keep falling with context")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(
    families: tuple[str, ...] = FAMILIES,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    rows = []
    t0 = time.time()
    for name in families:
        out = train_cached(
            name,
            seed=seed,
            model_overrides={"d_model": 128, "n_layers": 4},
            train_overrides={"steps": 4000, "ctx": 192, "batch_size": 128},
        )
        model, eval_set, process = out["model"], out["eval_set"], out["process"]
        burn_in = out["train_cfg"].burn_in

        model_curve, oracle_curve = _loss_curve(model, eval_set, burn_in)
        probes = probe_all_layers(model, eval_set, burn_in=burn_in)
        hook, probe = best_probe(probes, exclude_final=True)

        # a fixed filter cannot beat the best single component averaged over the family
        assert isinstance(process, MixtureOfProcesses)
        per_component = [
            float(-c.filter(eval_set.obs_np).logp[:, burn_in:].mean()) for c in process.components
        ]

        row = {
            "family": name,
            "n_components": process.k,
            "gap": out["final"].gap,
            "gap_stderr": out["final"].gap_stderr,
            "loss_curve_model": model_curve.tolist(),
            "loss_curve_oracle": oracle_curve.tolist(),
            "best_single_component_loss": float(min(per_component)),
            "per_component_fixed_filter_loss": per_component,
            "early_loss_model": float(model_curve[burn_in : burn_in + 16].mean()),
            "late_loss_model": float(model_curve[-32:].mean()),
            "early_loss_oracle": float(oracle_curve[burn_in : burn_in + 16].mean()),
            "late_loss_oracle": float(oracle_curve[-32:].mean()),
            "component_probe_hook": hook,
            "component_probe_r2": probe.r2,
            "identification": _component_identification(model, eval_set, probe, hook, burn_in),
            "held_out": _held_out_transfer(model, name, out["train_cfg"].ctx, burn_in, seed),
        }
        row["in_context_gain_model"] = row["early_loss_model"] - row["late_loss_model"]
        row["in_context_gain_oracle"] = row["early_loss_oracle"] - row["late_loss_oracle"]
        rows.append(row)
        if verbose:
            print(
                f"  {name:14s} gap={row['gap']:+.5f} probe R2={probe.r2:.3f} "
                f"in-context gain: model {row['in_context_gain_model']:+.4f} "
                f"vs optimal {row['in_context_gain_oracle']:+.4f} "
                f"| component id agreement {row['identification']['final_agreement']}",
                flush=True,
            )

    figure = _plot(rows, figure_path("exp07_incontext.png"))
    payload = {
        "experiment": "exp07_incontext",
        "seed": seed,
        "rows": rows,
        "figure": str(figure),
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": {
            r["family"]: {
                "gap": r["gap"],
                "in_context_gain_model": r["in_context_gain_model"],
                "in_context_gain_oracle": r["in_context_gain_oracle"],
                "gain_ratio": r["in_context_gain_model"] / max(r["in_context_gain_oracle"], 1e-9),
                "component_probe_r2": r["component_probe_r2"],
                "final_component_agreement": r["identification"]["final_agreement"],
                "held_out_mean_gap": float(
                    np.mean([h["gap_to_true_oracle"] for h in r["held_out"]])
                )
                if r["held_out"]
                else None,
                "beats_best_fixed_filter_by": r["best_single_component_loss"]
                - r["late_loss_model"],
            }
            for r in rows
        },
    }
    save_json(RESULTS_DIR / "exp07_incontext.json", payload)
    return payload
