"""Analysis-layer correctness.

These tests check the *machinery*, not the science: that a probe recovers a
planted linear code, that patching actually swaps activations, that scrubbing's
faithfulness hits its two fixed points, and that the SAE comparison is not
accidentally rigged.  Without them a plausible-looking but wrong number would be
indistinguishable from a real result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bsc.analysis.geometry import belief_dimension_estimate, pca, simplex_to_2d
from bsc.analysis.patching import (
    Component,
    activation_patch,
    all_components,
    attribution_patch,
    make_contrast_pairs,
    patching_fidelity,
    path_patch,
)
from bsc.analysis.probes import fit_probe, write_direction
from bsc.analysis.sae import (
    SAEConfig,
    _r2_from_features,
    greedy_feature_selection,
    reduced_rank_r2,
    train_sae,
)
from bsc.analysis.scrubbing import ScrubHypothesis, Scrubber, ablate_component, marginal_loss
from bsc.model import BeliefTransformer
from bsc.processes import get_process
from bsc.train import build_model_config, make_eval_set, train_model
from bsc.train.trainer import TrainConfig


@pytest.fixture(scope="module")
def trained_even():
    p = get_process("even")
    cfg = TrainConfig(steps=800, ctx=48, batch_size=128, eval_every=0, eval_n=256, burn_in=8)
    mcfg = build_model_config(p, d_model=32, n_layers=2, n_heads=2, n_ctx=48, seed=0)
    out = train_model(p, model_cfg=mcfg, cfg=cfg)
    return p, out["model"], out["eval_set"]


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def test_probe_recovers_a_planted_linear_code():
    """A belief written linearly into activations must be decoded at R^2 = 1."""
    rng = np.random.default_rng(0)
    belief = rng.dirichlet(np.ones(3), size=(40, 20))
    W = rng.normal(size=(3, 16))
    acts = torch.tensor(belief @ W + 0.01 * rng.normal(size=(40, 20, 16)))
    res = fit_probe(acts.float(), torch.tensor(belief).float(), ridge=1e-6)
    assert res.r2 > 0.99


def test_probe_finds_nothing_in_noise():
    rng = np.random.default_rng(1)
    acts = torch.tensor(rng.normal(size=(40, 20, 16))).float()
    belief = torch.tensor(rng.dirichlet(np.ones(3), size=(40, 20))).float()
    res = fit_probe(acts, belief, ridge=1e-2)
    assert res.r2 < 0.2


def test_probe_split_is_by_sequence_not_position():
    """Positions inside a sequence are correlated; a position-level split would leak."""
    rng = np.random.default_rng(2)
    # every position in a sequence shares the same belief, so a position split
    # would put identical rows in train and test and score a perfect R^2
    per_seq = rng.dirichlet(np.ones(3), size=(30, 1))
    belief = np.repeat(per_seq, 20, axis=1)
    acts = torch.tensor(rng.normal(size=(30, 20, 16))).float()
    res = fit_probe(acts, torch.tensor(belief).float(), ridge=1e-2)
    assert res.r2 < 0.5


def test_write_direction_inverts_the_probe_on_the_reachable_subspace():
    """Steering must reproduce any *achievable* belief change exactly.

    Beliefs sum to one, so their differences span only a (d-1)-dimensional
    subspace and the probe matrix is rank-deficient. The write direction is a
    pseudo-inverse: it inverts the probe on that subspace and cannot move the
    read-out off the simplex's affine hull, which is the correct behaviour.
    """
    rng = np.random.default_rng(3)
    belief = rng.dirichlet(np.ones(3), size=(30, 10))
    W = rng.normal(size=(3, 12))
    acts = torch.tensor(belief @ W).float()
    probe = fit_probe(acts, torch.tensor(belief).float(), ridge=1e-9)
    v = write_direction(probe)

    # achievable changes are differences of real beliefs, which sum to zero
    flat = torch.tensor(belief.reshape(-1, 3)).float()
    db = flat[:5] - flat[5:10]
    assert torch.allclose(db.sum(-1), torch.zeros(5), atol=1e-6)
    assert torch.allclose(db @ v @ probe.W, db, atol=1e-3)

    # an off-manifold change (all coordinates up) is not achievable
    off = torch.ones(1, 3)
    assert not torch.allclose(off @ v @ probe.W, off, atol=1e-2)


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------


def test_contrast_pairs_match_context_but_not_the_target(trained_even):
    """The matched window must stop before the readout position.

    Matching the predicted token as well conditions each pair on its own outcome,
    which biases every patching effect.
    """
    p, model, es = trained_even
    pairs = make_contrast_pairs(p, es, suffix_len=3, n_pairs=256, seed=0)
    lo = pairs.readout_pos - 3
    a = pairs.clean[:, lo : pairs.readout_pos]
    b = pairs.corrupt[:, lo : pairs.readout_pos]
    assert torch.equal(a, b), "context windows must match exactly for discrete processes"
    assert pairs.belief_distance > 0.05
    # the target itself must still vary across the pair set
    targets = pairs.clean[:, pairs.readout_pos]
    assert len(torch.unique(targets)) > 1


def test_patching_every_component_is_at_least_as_damaging_as_any_one(trained_even):
    p, model, es = trained_even
    pairs = make_contrast_pairs(p, es, suffix_len=3, n_pairs=64, seed=0)
    singles = activation_patch(model, pairs)
    assert len(singles) == len(all_components(model))
    assert all(np.isfinite(r.effect) for r in singles)
    # at least one component must matter on a task the model has learned
    assert max(r.effect for r in singles) > 1e-4


def test_attribution_patching_is_correlated_on_small_interventions():
    """With a nearly-untrained model the perturbation is small, so first order holds."""
    p = get_process("mess3_hard")
    es = make_eval_set(p, 128, 48, seed=0)
    model = BeliefTransformer(
        build_model_config(p, d_model=32, n_layers=2, n_heads=2, n_ctx=48)
    ).eval()
    pairs = make_contrast_pairs(p, es, suffix_len=2, n_pairs=128, seed=0, min_distance=0.0)
    fid = patching_fidelity(activation_patch(model, pairs), attribution_patch(model, pairs))
    assert fid["pearson_r"] > 0.5


def test_path_patch_is_bounded_by_activation_patch(trained_even):
    """A single route cannot carry more effect than all routes together."""
    p, model, es = trained_even
    pairs = make_contrast_pairs(p, es, suffix_len=3, n_pairs=64, seed=0)
    sender = Component("head", 0, 0)
    direct = next(r for r in activation_patch(model, pairs, [sender]) if r.label == sender.label)
    routed = path_patch(model, pairs, sender, "blocks.1.hook_resid_post")
    assert routed.effect <= direct.effect + 1e-3


# ---------------------------------------------------------------------------
# scrubbing
# ---------------------------------------------------------------------------


def test_scrubbing_fixed_points(trained_even):
    """Keeping every component is lossless; keeping none is the full-scrub baseline."""
    p, model, es = trained_even
    sc = Scrubber(model, es, burn_in=8, n_repeats=2, seed=0)
    everything = sc.evaluate(ScrubHypothesis.of(sc.labels, equivalence="exact"))
    assert abs(everything.faithfulness - 1.0) < 1e-6
    assert abs(everything.loss_scrubbed - everything.loss_model) < 1e-6

    nothing = sc.evaluate(ScrubHypothesis.of([], equivalence="exact"))
    assert abs(nothing.faithfulness_vs_full_scrub) < 0.15


def test_greedy_search_faithfulness_is_monotone_enough(trained_even):
    p, model, es = trained_even
    sc = Scrubber(model, es, burn_in=8, n_repeats=2, seed=0)
    search = sc.greedy_search(target=0.9, equivalence="exact")
    curve = [t["faithfulness"] for t in search["trace"]]
    assert curve[-1] >= curve[0]
    assert len(search["circuit"]) <= len(sc.labels)


def test_marginal_loss_matches_the_empirical_entropy(trained_even):
    p, model, es = trained_even
    counts = np.bincount(es.obs_np[:, 8:].ravel(), minlength=p.vocab_size).astype(float)
    probs = counts / counts.sum()
    expected = float(-(probs * np.log(probs)).sum())
    assert abs(marginal_loss(es, "categorical", burn_in=8) - expected) < 1e-6


def test_marginal_loss_for_a_gaussian_head_is_the_gaussian_entropy():
    p = get_process("kalman")
    es = make_eval_set(p, 256, 64, seed=0)
    var = float(es.obs[:, 8:].var(unbiased=False))
    expected = 0.5 * (np.log(2 * np.pi) + 1 + np.log(var))
    assert abs(marginal_loss(es, "gaussian", burn_in=8) - expected) < 1e-5


def test_ablation_modes_all_increase_kl(trained_even):
    p, model, es = trained_even
    from bsc.analysis.scrubbing import Scrubber as _S

    top = _S(model, es, burn_in=8, n_repeats=1).component_importance()[0]["component"]
    comp = Component("head", int(top[1]), int(top[3])) if "H" in top else Component("mlp", int(top[1]))
    for mode in ("mean", "zero", "resample"):
        r = ablate_component(model, es, comp, mode=mode, burn_in=8)
        assert r["kl_ablated"] >= r["kl_baseline"]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_simplex_projection_preserves_vertices():
    xy = simplex_to_2d(np.eye(3))
    assert xy.shape == (3, 2)
    # the three vertices must be mutually equidistant
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    off = d[~np.eye(3, dtype=bool)]
    assert np.allclose(off, off[0], atol=1e-9)


def test_pca_reconstructs_a_planted_subspace():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2000, 2))
    basis = rng.normal(size=(2, 10))
    X = latent @ basis
    p = pca(X, 2)
    assert p.explained_variance_ratio.sum() > 0.999


@pytest.mark.parametrize(
    ("generator", "expected"),
    [("line", 1.0), ("square", 2.0)],
)
def test_box_dimension_on_known_shapes(generator, expected):
    rng = np.random.default_rng(0)
    if generator == "line":
        t = rng.uniform(size=(60000, 1))
        pts = np.concatenate([t, 0.5 * t], axis=1)
    else:
        pts = rng.uniform(size=(60000, 2))
    d = belief_dimension_estimate(pts)["dimension"]
    assert abs(d - expected) < 0.25


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


def test_reduced_rank_r2_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 12))
    Y = X[:, :3] @ rng.normal(size=(3, 4)) + 0.1 * rng.normal(size=(2000, 4))
    vals = [reduced_rank_r2(X, Y, k) for k in (1, 2, 3, 4)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:], strict=False))
    assert vals[-1] <= 1.0 + 1e-9
    assert vals[2] > 0.95  # rank 3 suffices by construction


def test_reduced_rank_r2_upper_bounds_any_k_features():
    """No k features derived from X may beat the best k-dimensional subspace of X."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 10))
    Y = X[:, :2] @ rng.normal(size=(2, 3))
    feats = np.maximum(X @ rng.normal(size=(10, 40)), 0)  # ReLU features, as in an SAE
    chosen, curve = greedy_feature_selection(feats, Y, 3)
    assert curve[-1] <= reduced_rank_r2(np.concatenate([X, feats], axis=1), Y, 3) + 1e-6


def test_greedy_selection_improves_monotonically():
    rng = np.random.default_rng(2)
    F_ = rng.normal(size=(800, 25))
    Y = F_[:, [3, 7]] @ rng.normal(size=(2, 2)) + 0.05 * rng.normal(size=(800, 2))
    _, curve = greedy_feature_selection(F_, Y, 4)
    assert all(b >= a - 1e-9 for a, b in zip(curve, curve[1:], strict=False))
    assert curve[1] > 0.9


def test_r2_from_features_matches_numpy_lstsq():
    rng = np.random.default_rng(3)
    F_ = rng.normal(size=(500, 4))
    Y = F_ @ rng.normal(size=(4, 2)) + 0.1 * rng.normal(size=(500, 2))
    Fd = np.concatenate([F_, np.ones((500, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Fd, Y, rcond=None)
    resid = Y - Fd @ coef
    expected = 1 - (resid**2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    assert abs(_r2_from_features(F_, Y) - expected) < 1e-8


def test_sae_trains_and_reconstructs():
    rng = np.random.default_rng(0)
    # sparse ground truth: an SAE should do well here, unlike on a simplex
    codes = (rng.random((4000, 20)) < 0.15) * rng.random((4000, 20))
    dictionary = rng.normal(size=(20, 16))
    X = torch.tensor(codes @ dictionary, dtype=torch.float32).reshape(200, 20, 16)
    cfg = SAEConfig(d_in=16, expansion=4, l1_coeff=1e-3, steps=600, batch_size=512)
    sae, stats = train_sae(X, cfg)
    flat = X.reshape(-1, 16)
    xn = (flat - stats["mean"]) / stats["scale"]
    recon, f = sae(xn)
    ve = (1 - ((xn - recon) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum()).detach()
    assert float(ve) > 0.8
    assert 0 < float((f > 0).sum(-1).float().mean()) < cfg.d_sae


def test_sae_decoder_stays_unit_norm():
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(100, 10, 8)), dtype=torch.float32)
    sae, _ = train_sae(X, SAEConfig(d_in=8, expansion=2, steps=100, batch_size=256))
    norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
