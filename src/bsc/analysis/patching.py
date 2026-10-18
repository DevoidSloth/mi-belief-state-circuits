"""Activation, attribution and path patching over belief-contrast pairs.

The experimental design matters more than the machinery here.  Patching a
component with an activation from an unrelated sequence conflates two things:
the belief differs, but so do the recent tokens.  This module instead builds
**suffix-matched contrast pairs** - two sequences ending in the same k
observations but carrying different posteriors - so the only thing a patch can
transfer is the belief itself.

Metrics are reported as increases in negative log-likelihood on the *clean*
observations, in nats.  That keeps every effect size in the same currency as the
optimality gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..model.heads import head_logprob
from ..model.transformer import BeliefTransformer
from ..processes.base import Process
from ..train.data import EvalSet


# ---------------------------------------------------------------------------
# contrast pairs
# ---------------------------------------------------------------------------


@dataclass
class ContrastPairs:
    clean: torch.Tensor          # (n, ctx)
    corrupt: torch.Tensor        # (n, ctx)
    clean_belief: torch.Tensor   # (n, d_belief) at readout position
    corrupt_belief: torch.Tensor
    readout_pos: int
    suffix_len: int
    belief_distance: float       # mean total-variation distance between the pair

    def __len__(self) -> int:
        return self.clean.shape[0]


def make_contrast_pairs(
    process: Process,
    eval_set: EvalSet,
    readout_pos: int | None = None,
    suffix_len: int = 3,
    n_pairs: int = 256,
    seed: int = 0,
    min_distance: float = 0.05,
) -> ContrastPairs:
    """Pair sequences whose last ``suffix_len`` *context* observations match but whose beliefs differ.

    The matched window is ``obs[readout - suffix_len : readout]`` -- strictly
    before the readout position.  Including ``obs[readout]`` would match on the
    token being predicted, which conditions every pair on its own outcome: for a
    process whose belief pins the next symbol, one member of each pair is then
    always the confident-and-correct one, and patching between them produces
    systematically negative effects that say nothing about the component.

    For discrete processes the window is matched exactly.  For continuous ones an
    exact match has probability zero, so pairs are matched on the nearest window
    in Euclidean distance and the residual mismatch is reported through
    ``belief_distance``.
    """
    ctx = eval_set.ctx
    readout = ctx - 1 if readout_pos is None else readout_pos
    lo = readout - suffix_len
    if lo < 1:
        raise ValueError("readout position too early for the requested suffix length")

    obs = eval_set.obs_np
    belief = eval_set.belief[:, readout].cpu().numpy()
    rng = np.random.default_rng(seed)

    pairs: list[tuple[int, int]] = []
    if process.kind == "discrete":
        buckets: dict[tuple, list[int]] = {}
        for i in range(obs.shape[0]):
            buckets.setdefault(tuple(obs[i, lo:readout]), []).append(i)
        keys = [k for k, v in buckets.items() if len(v) > 1]
        if not keys:
            raise ValueError("no repeated suffixes; use a shorter suffix or more sequences")
        while len(pairs) < n_pairs:
            members = buckets[keys[rng.integers(len(keys))]]
            i, j = rng.choice(len(members), size=2, replace=False)
            i, j = members[i], members[j]
            if np.abs(belief[i] - belief[j]).sum() / 2 >= min_distance:
                pairs.append((i, j))
            elif len(pairs) == 0 and len(keys) == 1:
                break
    else:
        suffix = obs[:, lo:readout]
        for _ in range(n_pairs * 4):
            i = int(rng.integers(obs.shape[0]))
            d = np.linalg.norm(suffix - suffix[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            if np.abs(belief[i] - belief[j]).sum() / 2 >= min_distance:
                pairs.append((i, j))
            if len(pairs) >= n_pairs:
                break

    if not pairs:
        raise ValueError("could not build contrast pairs; relax min_distance")
    idx_a = torch.tensor([p[0] for p in pairs])
    idx_b = torch.tensor([p[1] for p in pairs])
    device = eval_set.obs.device
    ia, ib = idx_a.to(device), idx_b.to(device)
    tv = float(np.abs(belief[idx_a.numpy()] - belief[idx_b.numpy()]).sum(-1).mean() / 2)
    return ContrastPairs(
        clean=eval_set.obs[ia],
        corrupt=eval_set.obs[ib],
        clean_belief=eval_set.belief[ia, readout],
        corrupt_belief=eval_set.belief[ib, readout],
        readout_pos=readout,
        suffix_len=suffix_len,
        belief_distance=tv,
    )


# ---------------------------------------------------------------------------
# component addressing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """A patchable unit: one attention head's output, or one MLP's output."""

    kind: str    # "head" | "mlp"
    layer: int
    head: int | None = None

    @property
    def label(self) -> str:
        return f"L{self.layer}H{self.head}" if self.kind == "head" else f"L{self.layer}MLP"

    @property
    def hook(self) -> str:
        if self.kind == "head":
            return f"blocks.{self.layer}.attn.hook_z"
        return f"blocks.{self.layer}.hook_mlp_out"

    def index(self, positions: torch.Tensor | slice | int | None):
        pos = slice(None) if positions is None else positions
        if self.kind == "head":
            return (slice(None), pos, self.head)
        return (slice(None), pos)


def all_components(model: BeliefTransformer) -> list[Component]:
    out: list[Component] = []
    for layer in range(model.cfg.n_layers):
        out += [Component("head", layer, h) for h in range(model.cfg.n_heads)]
        if not model.cfg.attn_only:
            out.append(Component("mlp", layer))
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def nll_at(
    model: BeliefTransformer,
    params: torch.Tensor,
    obs: torch.Tensor,
    pos: int,
    reduction: str = "mean",
):
    """Negative log-likelihood of the clean observation at the readout position.

    ``reduction="sum"`` is required for attribution patching: the gradient of a
    *mean* loss carries a 1/batch factor, which silently scales every attribution
    down by the batch size and makes the approximation look like it has failed.
    """
    lp = head_logprob(
        model.cfg.head, params[:, pos : pos + 1], obs[:, pos : pos + 1], model.cfg.n_mixture
    )
    return -lp.sum() if reduction == "sum" else -lp.mean()


# ---------------------------------------------------------------------------
# activation patching
# ---------------------------------------------------------------------------


@dataclass
class PatchResult:
    label: str
    effect: float          # increase in clean-token NLL, nats
    effect_stderr: float
    kind: str = "activation"
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def activation_patch(
    model: BeliefTransformer,
    pairs: ContrastPairs,
    components: list[Component] | None = None,
    positions: slice | int | None = None,
) -> list[PatchResult]:
    """Noising patch: run clean, splice in the corrupt activation, measure the damage.

    ``positions`` defaults to the readout position alone, which asks the local
    question this analysis is for: *which component writes the belief that the
    prediction at this position depends on?*  Patching a component at every
    position instead mixes one sequence's component with another sequence's
    everything-else at each of them, which is far off-distribution and produces
    effects of either sign that say little about the component.  Pass
    ``positions=slice(None)`` for that variant deliberately.

    An effect is the mean increase in NLL of the clean observation at the readout
    position, so it is directly comparable to the model's optimality gap.
    """
    model.eval()
    components = components or all_components(model)
    clean, corrupt = pairs.clean, pairs.corrupt
    pos = pairs.readout_pos
    if positions is None:
        positions = pos

    hooks = sorted({c.hook for c in components})
    _, corrupt_cache = model.run_with_cache(corrupt, names=hooks)

    base_params = model(clean)
    base_lp = head_logprob(model.cfg.head, base_params[:, pos], clean[:, pos], model.cfg.n_mixture)

    results = []
    for comp in components:
        donor = corrupt_cache[comp.hook]
        index = comp.index(positions)

        def fn(x, _name, donor=donor, index=index):
            x = x.clone()
            x[index] = donor[index]
            return x

        with model.hooks([(comp.hook, fn)]):
            params = model(clean)
        lp = head_logprob(model.cfg.head, params[:, pos], clean[:, pos], model.cfg.n_mixture)
        delta = (base_lp - lp).float().cpu().numpy()
        results.append(
            PatchResult(
                label=comp.label,
                effect=float(delta.mean()),
                effect_stderr=float(delta.std(ddof=1) / np.sqrt(len(delta))),
            )
        )
    return sorted(results, key=lambda r: -r.effect)


def attribution_patch(
    model: BeliefTransformer,
    pairs: ContrastPairs,
    components: list[Component] | None = None,
    positions: slice | int | None = None,
) -> list[PatchResult]:
    """First-order approximation to activation patching: (a_corrupt - a_clean) . dL/da.

    One backward pass covers every component, which is what makes this scale.  It
    is an approximation, so :func:`patching_fidelity` checks it against true
    patching before it is trusted.
    """
    model.eval()
    components = components or all_components(model)
    if positions is None:
        positions = pairs.readout_pos
    hooks = sorted({c.hook for c in components})

    with torch.no_grad():
        _, corrupt_cache = model.run_with_cache(pairs.corrupt, names=hooks)

    clean_acts: dict[str, torch.Tensor] = {}

    def grab(x, name):
        x.retain_grad()
        clean_acts[name] = x
        return x

    with model.hooks([(h, grab) for h in hooks]):
        params = model(pairs.clean)
        # sum, not mean: we want the per-example gradient at each activation
        loss = nll_at(model, params, pairs.clean, pairs.readout_pos, reduction="sum")
    model.zero_grad(set_to_none=True)
    loss.backward()

    results = []
    for comp in components:
        act = clean_acts[comp.hook]
        grad = act.grad
        index = comp.index(positions)
        contrib = ((corrupt_cache[comp.hook][index] - act[index].detach()) * grad[index]).sum(-1)
        contrib = contrib.reshape(contrib.shape[0], -1).sum(-1).float().cpu().numpy()
        del grad
        results.append(
            PatchResult(
                label=comp.label,
                effect=float(contrib.mean()),
                effect_stderr=float(contrib.std(ddof=1) / np.sqrt(len(contrib))),
                kind="attribution",
            )
        )
    model.zero_grad(set_to_none=True)
    return sorted(results, key=lambda r: -r.effect)


def patching_fidelity(true: list[PatchResult], approx: list[PatchResult]) -> dict:
    """How well attribution patching tracks true patching: Pearson r and top-k overlap."""
    a = {r.label: r.effect for r in true}
    b = {r.label: r.effect for r in approx}
    labels = [k for k in a if k in b]
    x = np.array([a[k] for k in labels])
    y = np.array([b[k] for k in labels])
    if x.std() < 1e-12 or y.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    k = max(1, len(labels) // 4)
    top_true = {labels[i] for i in np.argsort(-x)[:k]}
    top_approx = {labels[i] for i in np.argsort(-y)[:k]}
    return {
        "pearson_r": r,
        "spearman_r": float(
            np.corrcoef(np.argsort(np.argsort(-x)), np.argsort(np.argsort(-y)))[0, 1]
        ),
        "top_k": k,
        "top_k_overlap": len(top_true & top_approx) / k,
    }


# ---------------------------------------------------------------------------
# path patching
# ---------------------------------------------------------------------------


@torch.no_grad()
def path_patch(
    model: BeliefTransformer,
    pairs: ContrastPairs,
    sender: Component,
    receiver_hook: str,
    positions: slice | int | None = None,
) -> PatchResult:
    """Effect of the sender on the output *through* one receiver only.

    Three passes: cache clean and corrupt; recompute the receiver's input with the
    sender patched while every other input to it is frozen at its clean value;
    then run the rest of the model with only that receiver input replaced.  The
    difference from plain activation patching is that indirect routes -- sender to
    output via anything other than the receiver -- are excluded by construction.
    """
    model.eval()
    pos = pairs.readout_pos
    if positions is None:
        positions = pos
    all_hooks = sorted({sender.hook, receiver_hook})

    _, clean_cache = model.run_with_cache(pairs.clean, names=all_hooks)
    _, corrupt_cache = model.run_with_cache(pairs.corrupt, names=all_hooks)

    base_params = model(pairs.clean)
    base_lp = head_logprob(
        model.cfg.head, base_params[:, pos], pairs.clean[:, pos], model.cfg.n_mixture
    )

    # step 1: sender patched, capture what the receiver now sees
    index = sender.index(positions)

    def patch_sender(x, _name):
        x = x.clone()
        x[index] = corrupt_cache[sender.hook][index]
        return x

    with model.hooks([(sender.hook, patch_sender)]):
        _, mid_cache = model.run_with_cache(pairs.clean, names=[receiver_hook])
    new_receiver_input = mid_cache[receiver_hook]

    # step 2: clean run everywhere except the receiver, which sees the new input
    def patch_receiver(x, _name):
        return new_receiver_input

    with model.hooks([(receiver_hook, patch_receiver)]):
        params = model(pairs.clean)
    lp = head_logprob(model.cfg.head, params[:, pos], pairs.clean[:, pos], model.cfg.n_mixture)

    delta = (base_lp - lp).float().cpu().numpy()
    return PatchResult(
        label=f"{sender.label}->{receiver_hook}",
        effect=float(delta.mean()),
        effect_stderr=float(delta.std(ddof=1) / np.sqrt(len(delta))),
        kind="path",
        extra={"sender": sender.label, "receiver": receiver_hook},
    )


def select_circuit(
    results: list[PatchResult],
    threshold_nats: float = 0.01,
    max_components: int | None = None,
) -> list[str]:
    """Components whose patched effect clears a threshold, largest first."""
    keep = [r for r in results if r.effect > threshold_nats]
    keep.sort(key=lambda r: -r.effect)
    if max_components is not None:
        keep = keep[:max_components]
    return [r.label for r in keep]
