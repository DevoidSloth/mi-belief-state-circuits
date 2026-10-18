"""Training loop, data plumbing and evaluation against the analytic optimum."""

from .data import EvalSet, StreamingSampler, make_eval_set
from .trainer import (
    EvalMetrics,
    TrainConfig,
    build_model_config,
    collect_residuals,
    evaluate,
    train_model,
)

__all__ = [
    "EvalMetrics",
    "EvalSet",
    "StreamingSampler",
    "TrainConfig",
    "build_model_config",
    "collect_residuals",
    "evaluate",
    "make_eval_set",
    "train_model",
]
