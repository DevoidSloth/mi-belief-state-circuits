"""Training and evaluation against the analytic optimum.

The primary scalar this project reports is the **optimality gap**: model loss
minus oracle loss, measured on the same sequences.  Because the comparison is
paired position-by-position, the sampling noise that dominates either loss
cancels, and gaps of ~1e-3 nats are resolvable with a few hundred sequences.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..model.heads import head_logprob, kl_true_to_model
from ..model.transformer import BeliefTransformer, ModelConfig
from ..processes.base import Process
from ..utils import resolve_device, set_seed
from .data import EvalSet, StreamingSampler, make_eval_set


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 128
    ctx: int = 128
    lr: float = 3e-3
    min_lr_frac: float = 0.05
    weight_decay: float = 0.01
    warmup: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 250
    eval_n: int = 512
    eval_seed: int = 12345
    #: positions dropped before scoring, so results describe the stationary regime
    burn_in: int = 16
    pool_size: int = 8192
    refresh_every: int = 1000
    device: str = "auto"
    log_every: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class EvalMetrics:
    loss: float
    oracle_loss: float
    gap: float
    gap_stderr: float
    kl_to_optimal: float
    loss_per_position: list[float] = field(default_factory=list)
    oracle_per_position: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "loss": self.loss,
            "oracle_loss": self.oracle_loss,
            "gap": self.gap,
            "gap_stderr": self.gap_stderr,
            "kl_to_optimal": self.kl_to_optimal,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_set: EvalSet,
    head: str,
    burn_in: int = 0,
    n_mixture: int = 3,
    batch: int = 256,
    per_position: bool = False,
) -> EvalMetrics:
    """Paired comparison of model and oracle on the same sequences."""
    model.eval()
    logps, kls = [], []
    for i in range(0, eval_set.n_seq, batch):
        obs = eval_set.obs[i : i + batch]
        params = model(obs)
        logps.append(head_logprob(head, params, obs, n_mixture).float())
        pred = {k: v[i : i + batch] for k, v in eval_set.predictive.items()}
        kls.append(kl_true_to_model(head, params, pred, n_mixture).float())
    logp = torch.cat(logps)
    kl = torch.cat(kls)

    model_nll = -logp[:, burn_in:]
    oracle_nll = -eval_set.oracle_logp[:, burn_in:]
    per_seq_gap = (model_nll - oracle_nll).mean(dim=1).cpu().numpy()

    metrics = EvalMetrics(
        loss=float(model_nll.mean()),
        oracle_loss=float(oracle_nll.mean()),
        gap=float(per_seq_gap.mean()),
        gap_stderr=float(per_seq_gap.std(ddof=1) / math.sqrt(len(per_seq_gap))),
        kl_to_optimal=float(kl[:, burn_in:].mean()),
    )
    if per_position:
        metrics.loss_per_position = (-logp).mean(dim=0).cpu().numpy().tolist()
        metrics.oracle_per_position = (
            (-eval_set.oracle_logp).mean(dim=0).cpu().numpy().tolist()
        )
    model.train()
    return metrics


def build_model_config(process: Process, **overrides) -> ModelConfig:
    """Model config matched to a process: right input kind, right head."""
    base = {
        "input_kind": process.kind,
        "vocab_size": max(process.vocab_size, 1),
        "head": process.head,
    }
    if process.head == "gaussian_mixture":
        base["n_mixture"] = process.d_belief
    base.update(overrides)
    return ModelConfig(**base)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr * (cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * cosine)


def train_model(
    process: Process,
    model: nn.Module | None = None,
    model_cfg: ModelConfig | None = None,
    cfg: TrainConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Train one model on one process. Returns the model, history and final metrics."""
    cfg = cfg or TrainConfig()
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    if model is None:
        model_cfg = model_cfg or build_model_config(process, n_ctx=cfg.ctx, seed=cfg.seed)
        model = BeliefTransformer(model_cfg)
    model = model.to(device)
    head = model.cfg.head
    n_mixture = model.cfg.n_mixture

    eval_set = make_eval_set(process, cfg.eval_n, cfg.ctx, cfg.eval_seed, device)
    sampler = StreamingSampler(
        process, cfg.batch_size, cfg.ctx, cfg.seed + 1, cfg.pool_size, cfg.refresh_every, device
    )

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "bias" in name else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.99),
    )

    history: list[dict] = []
    t0 = time.time()
    model.train()
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)

        obs = sampler.next_batch()
        params = model(obs)
        loss = -head_logprob(head, params, obs, n_mixture)[:, cfg.burn_in :].mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        last = step == cfg.steps - 1
        if cfg.eval_every and (step % cfg.eval_every == 0 or last):
            m = evaluate(model, eval_set, head, cfg.burn_in, n_mixture, per_position=last)
            history.append({"step": step, "train_loss": loss.item(), **m.as_dict()})
            if verbose:
                print(
                    f"  step {step:5d}  train {loss.item():.4f}  eval {m.loss:.4f}  "
                    f"oracle {m.oracle_loss:.4f}  gap {m.gap:+.5f}",
                    flush=True,
                )

    final = evaluate(model, eval_set, head, cfg.burn_in, n_mixture, per_position=True)
    return {
        "model": model,
        "model_cfg": model.cfg,
        "train_cfg": cfg,
        "eval_set": eval_set,
        "history": history,
        "final": final,
        "wall_time_s": time.time() - t0,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def collect_residuals(
    model: BeliefTransformer,
    eval_set: EvalSet,
    hook_names: list[str],
    batch: int = 256,
) -> dict[str, torch.Tensor]:
    """Cache residual-stream (or any) activations over a whole eval set."""
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {n: [] for n in hook_names}
    for i in range(0, eval_set.n_seq, batch):
        _, cache = model.run_with_cache(eval_set.obs[i : i + batch], names=hook_names)
        for n in hook_names:
            chunks[n].append(cache[n].float().cpu())
    model.train()
    return {n: torch.cat(v) for n, v in chunks.items()}


def loss_of_logits_fn(head: str, n_mixture: int = 3):
    """Helper for patching experiments: params -> mean NLL on given observations."""

    def fn(params: torch.Tensor, obs: torch.Tensor, burn_in: int = 0) -> torch.Tensor:
        return -head_logprob(head, params, obs, n_mixture)[:, burn_in:].mean()

    return fn


def summarise_history(history: list[dict]) -> dict:
    if not history:
        return {}
    gaps = np.array([h["gap"] for h in history])
    return {
        "final_gap": float(gaps[-1]),
        "best_gap": float(gaps.min()),
        "steps": int(history[-1]["step"]) + 1,
    }
