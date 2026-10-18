"""Hooked transformer, output heads and baselines."""

from .baselines import GRUBaseline, LinearBaseline, NGramBaseline, OracleModel
from .heads import head_distribution, head_logprob, head_loss, kl_true_to_model
from .hooks import HookedRoot, HookFn, HookPoint, add_hook, mean_hook, patch_hook, zero_hook
from .transformer import Attention, BeliefTransformer, Block, ModelConfig

__all__ = [
    "Attention",
    "BeliefTransformer",
    "Block",
    "GRUBaseline",
    "HookFn",
    "HookPoint",
    "HookedRoot",
    "LinearBaseline",
    "ModelConfig",
    "NGramBaseline",
    "OracleModel",
    "add_hook",
    "head_distribution",
    "head_logprob",
    "head_loss",
    "kl_true_to_model",
    "mean_hook",
    "patch_hook",
    "zero_hook",
]
