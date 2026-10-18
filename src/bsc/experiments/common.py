"""Shared recipe and checkpoint cache for the experiment suite.

Every experiment trains models the same way, so training recipes live here
rather than being restated (and drifting) in each script.  Checkpoints are
cached on disk keyed by the full configuration, which makes the suite
restartable and keeps the circuit, geometry and SAE experiments looking at
*the same* trained models rather than at re-runs that happened to differ.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..model.transformer import BeliefTransformer, ModelConfig
from ..processes import get_process
from ..processes.base import Process
from ..train.data import make_eval_set
from ..train.trainer import TrainConfig, build_model_config, evaluate, train_model
from ..utils import REPO_ROOT, resolve_device

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

#: Model capacity by process. Processes whose belief lives in a larger simplex or
#: needs longer memory get more width; everything else stays small so the whole
#: suite fits in minutes on one GPU.
MODEL_OVERRIDES: dict[str, dict] = {
    "rrxor": {"d_model": 64, "n_layers": 2},
    "mess3": {"d_model": 128, "n_layers": 4},
    "mess3_hard": {"d_model": 128, "n_layers": 4},
    "random_non": {"d_model": 96, "n_layers": 3},
    "random_non_big": {"d_model": 128, "n_layers": 3},
    "random_uni_slow": {"d_model": 96, "n_layers": 3},
    "persistent_bit": {"d_model": 64, "n_layers": 2},
    "drifting_coin": {"d_model": 96, "n_layers": 3},
    "regimevol": {"d_model": 128, "n_layers": 3},
    "garch": {"d_model": 96, "n_layers": 3},
    # the Kalman weight-level analysis needs a strictly linear residual stream
    "kalman": {"d_model": 64, "n_layers": 1, "norm": "none"},
    "kalman_slow": {"d_model": 96, "n_layers": 2, "norm": "none"},
}

DEFAULT_MODEL = {"d_model": 64, "n_layers": 2, "n_heads": 4, "norm": "layer"}
# Sized against a thermally-throttled laptop 4080 (~1.3 of 3.1 GHz under sustained
# load). Earlier runs reach the same optimality gaps at a quarter of this compute,
# so the suite is bounded by wall-clock rather than by convergence.
DEFAULT_TRAIN = {
    "ctx": 96,
    "steps": 3000,
    "batch_size": 192,
    "lr": 2e-3,
    "burn_in": 16,
    "eval_every": 1500,
}


def recipe(
    process_name: str,
    seed: int = 0,
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
) -> tuple[Process, ModelConfig, TrainConfig]:
    process = get_process(process_name)
    train_kw = {**DEFAULT_TRAIN, **(train_overrides or {})}
    tcfg = TrainConfig(seed=seed, eval_every=train_kw.pop("eval_every", 1000), **train_kw)

    model_kw = {**DEFAULT_MODEL, **MODEL_OVERRIDES.get(process_name, {})}
    model_kw.update(model_overrides or {})
    model_kw.setdefault("n_ctx", tcfg.ctx)
    mcfg = build_model_config(process, seed=seed, **model_kw)
    return process, mcfg, tcfg


def _key(process_name: str, mcfg: ModelConfig, tcfg: TrainConfig) -> str:
    payload = json.dumps(
        {"process": process_name, "model": asdict(mcfg), "train": asdict(tcfg)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def train_cached(
    process_name: str,
    seed: int = 0,
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> dict:
    """Train (or load) one model and return everything the analyses need."""
    process, mcfg, tcfg = recipe(process_name, seed, model_overrides, train_overrides)
    device = resolve_device(tcfg.device)
    key = _key(process_name, mcfg, tcfg)
    path = CHECKPOINT_DIR / f"{process_name}_seed{seed}_{key}.pt"

    if use_cache and path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        model = BeliefTransformer(mcfg).to(device)
        model.load_state_dict(blob["state_dict"])
        eval_set = make_eval_set(process, tcfg.eval_n, tcfg.ctx, tcfg.eval_seed, device)
        final = evaluate(model, eval_set, mcfg.head, tcfg.burn_in, mcfg.n_mixture,
                         per_position=True)
        return {
            "process": process,
            "model": model,
            "model_cfg": mcfg,
            "train_cfg": tcfg,
            "eval_set": eval_set,
            "final": final,
            "history": blob.get("history", []),
            "n_params": sum(p.numel() for p in model.parameters()),
            "wall_time_s": blob.get("wall_time_s", 0.0),
            "cached": True,
            "checkpoint": str(path),
        }

    out = train_model(process, model_cfg=mcfg, cfg=tcfg, verbose=verbose)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": out["model"].state_dict(),
            "history": out["history"],
            "wall_time_s": out["wall_time_s"],
            "process": process_name,
            "seed": seed,
        },
        path,
    )
    out.update({"process": process, "cached": False, "checkpoint": str(path)})
    return out


def best_probe(probes: dict, exclude_final: bool = False):
    """Highest-R^2 probe, optionally ignoring the post-final-norm readout.

    ``hook_resid_final`` sits immediately before the unembedding, so a probe
    there is close to reading the model's own output.  Excluding it gives the
    strongest claim about an *internal* representation.
    """
    items = [
        (h, p) for h, p in probes.items() if not (exclude_final and h == "hook_resid_final")
    ]
    return max(items, key=lambda kv: kv[1].r2)


def figure_path(name: str) -> Path:
    from ..utils import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name
