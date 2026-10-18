"""Model, hook-system and head correctness.

The most important test here is :func:`test_model_is_strictly_causal`.  Every
patching result would be meaningless if the model could see the token it is
asked to predict, and that kind of leak is silent -- it shows up as a
suspiciously small optimality gap, not as an error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bsc.model import (
    BeliefTransformer,
    GRUBaseline,
    LinearBaseline,
    ModelConfig,
    NGramBaseline,
    head_logprob,
    kl_true_to_model,
    patch_hook,
    zero_hook,
)
from bsc.processes import get_process
from bsc.train import build_model_config, evaluate, make_eval_set


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(d_model=32, n_layers=2, n_heads=4, n_ctx=32, vocab_size=3, seed=0)
    base.update(kw)
    return ModelConfig(**base)


def test_model_is_strictly_causal():
    """Output at position t must not change when future observations change."""
    torch.manual_seed(0)
    model = BeliefTransformer(tiny_cfg()).eval()
    obs = torch.randint(0, 3, (4, 32))
    base = model(obs)
    for t in (0, 5, 17, 30):
        perturbed = obs.clone()
        perturbed[:, t + 1 :] = (perturbed[:, t + 1 :] + 1) % 3
        out = model(perturbed)
        torch.testing.assert_close(out[:, : t + 1], base[:, : t + 1], atol=1e-5, rtol=1e-4)


def test_bos_position_sees_no_observation():
    """Position 0 must be identical for every sequence: it has seen nothing."""
    model = BeliefTransformer(tiny_cfg()).eval()
    obs = torch.randint(0, 3, (8, 32))
    out = model(obs)
    assert torch.allclose(out[:, 0], out[0:1, 0].expand(8, -1), atol=1e-6)


def test_hook_cache_shapes():
    cfg = tiny_cfg()
    model = BeliefTransformer(cfg).eval()
    obs = torch.randint(0, 3, (2, 16))
    _, cache = model.run_with_cache(obs)
    assert cache["blocks.0.hook_resid_post"].shape == (2, 16, cfg.d_model)
    assert cache["blocks.0.attn.hook_z"].shape == (2, 16, cfg.n_heads, cfg.d_head)
    assert cache["blocks.0.attn.hook_pattern"].shape == (2, cfg.n_heads, 16, 16)


def test_attention_pattern_is_causal_and_normalised():
    model = BeliefTransformer(tiny_cfg()).eval()
    _, cache = model.run_with_cache(torch.randint(0, 3, (2, 16)), names=["hook_pattern"])
    pattern = cache["blocks.0.attn.hook_pattern"]
    torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-5,
                               rtol=1e-4)
    upper = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    assert pattern[:, :, upper].abs().max() < 1e-6


def test_hooks_are_removed_after_the_context_exits():
    model = BeliefTransformer(tiny_cfg()).eval()
    obs = torch.randint(0, 3, (2, 16))
    base = model(obs)
    with model.hooks([("blocks.0.hook_resid_post", zero_hook())]):
        zeroed = model(obs)
    after = model(obs)
    assert not torch.allclose(base, zeroed)
    torch.testing.assert_close(base, after)


def test_patch_hook_replaces_the_activation():
    model = BeliefTransformer(tiny_cfg()).eval()
    obs_a = torch.randint(0, 3, (2, 16))
    obs_b = torch.randint(0, 3, (2, 16))
    _, cache_b = model.run_with_cache(obs_b, names=["blocks.0.hook_resid_post"])
    donor = cache_b["blocks.0.hook_resid_post"]
    with model.hooks([("blocks.0.hook_resid_post", patch_hook(donor))]):
        _, cache = model.run_with_cache(obs_a, names=["blocks.0.hook_resid_post"])
    torch.testing.assert_close(cache["blocks.0.hook_resid_post"], donor)


def test_ov_circuit_matches_an_explicit_head_computation():
    """W_V[h] @ W_O[h] must be the map a head applies to its attended input."""
    cfg = tiny_cfg(n_layers=1, norm="none")
    model = BeliefTransformer(cfg).eval()
    attn = model.blocks[0].attn
    x = torch.randn(cfg.d_model)
    h = 2
    manual = (x @ attn.W_V[h] + attn.b_V[h]) @ attn.W_O[h]
    via_ov = x @ attn.OV(h) + attn.b_V[h] @ attn.W_O[h]
    torch.testing.assert_close(manual, via_ov, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


def test_categorical_logprob_matches_manual_softmax():
    params = torch.randn(3, 5, 4)
    targets = torch.randint(0, 4, (3, 5))
    lp = head_logprob("categorical", params, targets)
    manual = torch.log_softmax(params, -1).gather(-1, targets[..., None])[..., 0]
    torch.testing.assert_close(lp, manual)


def test_gaussian_logprob_matches_scipy_formula():
    params = torch.stack([torch.randn(3, 5), torch.randn(3, 5) * 0.3], dim=-1)
    y = torch.randn(3, 5)
    lp = head_logprob("gaussian", params, y)
    mean, logv = params[..., 0], params[..., 1]
    manual = -0.5 * (math.log(2 * math.pi) + logv + (y - mean) ** 2 / torch.exp(logv))
    torch.testing.assert_close(lp, manual)


def test_mixture_head_integrates_to_one():
    """Numerically integrate the mixture density; it must be a density."""
    torch.manual_seed(0)
    params = torch.randn(1, 1, 9) * 0.5
    grid = torch.linspace(-25, 25, 40001)
    lp = head_logprob("gaussian_mixture", params.expand(1, 40001, 9), grid[None, :], n_mixture=3)
    integral = torch.trapz(torch.exp(lp[0]), grid)
    assert abs(float(integral) - 1.0) < 1e-3


def test_kl_categorical_is_zero_when_model_equals_oracle():
    pmf = torch.rand(2, 4, 3)
    pmf = pmf / pmf.sum(-1, keepdim=True)
    params = torch.log(pmf)
    kl = kl_true_to_model("categorical", params, {"pmf": pmf})
    assert float(kl.abs().max()) < 1e-6


def test_kl_gaussian_matches_closed_form():
    m1, v1 = torch.randn(4), torch.rand(4) + 0.5
    m2, v2 = torch.randn(4), torch.rand(4) + 0.5
    params = torch.stack([m2, torch.log(v2)], dim=-1)
    kl = kl_true_to_model("gaussian", params, {"mean": m1, "var": v1})
    expected = 0.5 * (torch.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)
    torch.testing.assert_close(kl, expected)


def test_kl_equals_excess_loss_in_expectation():
    """KL(optimal || model) must equal the realised excess loss up to MC error."""
    p = get_process("even")
    es = make_eval_set(p, 400, 64, seed=7)
    model = BeliefTransformer(build_model_config(p, d_model=32, n_layers=1, n_ctx=64)).eval()
    m = evaluate(model, es, "categorical", burn_in=8)
    assert abs(m.gap - m.kl_to_optimal) < 5 * m.gap_stderr + 1e-3


def test_mixture_kl_is_small_when_model_matches_oracle():
    p = get_process("regimevol")
    es = make_eval_set(p, 32, 32, seed=3)
    w = es.predictive["weights"]
    params = torch.cat(
        [torch.log(w.clamp_min(1e-20)), es.predictive["means"], torch.log(es.predictive["vars"])],
        dim=-1,
    )
    kl = kl_true_to_model("gaussian_mixture", params, es.predictive, n_mixture=3)
    assert float(kl.abs().mean()) < 1e-5


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_ngram_recovers_an_order1_markov_process():
    """Golden mean is order-1 Markov in its observations, so a bigram is optimal."""
    p = get_process("golden_mean")
    rng = np.random.default_rng(0)
    train = p.sample(2000, 200, rng)
    test = p.sample(500, 200, rng)
    model, info = NGramBaseline.fit_best(train, test, p.vocab_size)
    oracle = -p.filter(test).logp.mean()
    assert info["order"] >= 1
    assert model.loss(test) - oracle < 0.005


def test_ngram_cannot_solve_an_infinite_order_process():
    """The even process needs unbounded parity memory; n-grams must fall short."""
    p = get_process("even")
    rng = np.random.default_rng(1)
    train = p.sample(4000, 200, rng)
    test = p.sample(500, 200, rng)
    model, _ = NGramBaseline.fit_best(train, test, p.vocab_size, orders=(1, 2, 3, 4, 5))
    oracle = -p.filter(test).logp.mean()
    assert model.loss(test) - oracle > 0.005


def test_ngram_context_index_has_no_lookahead():
    ng = NGramBaseline(vocab_size=2, order=2)
    obs = np.array([[0, 1, 1, 0, 1]])
    idx = ng._context_index(obs)
    # position 0 sees (BOS, BOS); position 2 sees (obs0, obs1) = (0, 1)
    assert idx[0, 0] == 2 * 1 + 2 * 3
    assert idx[0, 2] == 1 * 1 + 0 * 3


@pytest.mark.parametrize("name", ["even", "kalman"])
def test_baselines_run_and_are_causal(name):
    p = get_process(name)
    cfg = build_model_config(p, d_model=16, n_layers=1, n_ctx=24)
    es = make_eval_set(p, 4, 24, seed=0)
    for model in (GRUBaseline(cfg).eval(), LinearBaseline(cfg, window=4).eval()):
        base = model(es.obs)
        perturbed = es.obs.clone()
        if p.kind == "discrete":
            perturbed[:, 11:] = (perturbed[:, 11:] + 1) % p.vocab_size
        else:
            perturbed[:, 11:] += 1.0
        out = model(perturbed)
        torch.testing.assert_close(out[:, :11], base[:, :11], atol=1e-5, rtol=1e-4)


def test_component_names_cover_every_head_and_mlp():
    cfg = tiny_cfg(n_layers=3, n_heads=2)
    model = BeliefTransformer(cfg)
    names = model.component_names()
    assert len(names) == cfg.n_components == 3 * (2 + 1)
    assert "L2MLP" in names and "L0H1" in names
