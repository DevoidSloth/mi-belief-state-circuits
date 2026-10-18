"""Causal scrubbing: the rigorous version of a circuit claim.

Patching tells you a component *matters*.  It does not tell you the story you
told about that component is right.  Scrubbing does: state a hypothesis about
what each component depends on, resample every activation the hypothesis says is
irrelevant, and check that loss survives.

Two hypothesis strengths are implemented:

``exact``
    Components outside the claimed circuit are resample-ablated from unrelated
    sequences; circuit components keep their true activations.  This tests "the
    circuit is sufficient".

``belief``
    *Additionally*, each circuit component's activation is resampled from a
    different sequence whose true posterior lies in the same equivalence class.
    This tests the stronger, semantic claim - "this component depends on the
    input only through the belief state" - and it can fail even when the circuit
    is sufficient, which is exactly what makes it worth reporting.

Choice of denominator
---------------------
The usual "loss recovered" normalises by a fully-scrubbed model.  That is a
generous denominator here: scrubbing every component drives these models
*off-distribution*, to a loss worse than predicting the marginal, which inflates
any percentage computed against it.  The primary metric therefore normalises by
the best **context-free** predictor - a model that knows the marginal and
nothing else::

    faithfulness = 1 - (L_scrubbed - L_model) / (L_marginal - L_model)

The classic full-scrub number is reported alongside it as
``faithfulness_vs_full_scrub``, and is always the larger of the two.

Scope note: this is resample-ablation scrubbing over a flat component set, not
the fully treeified recursive algorithm of Chan et al.  It is strictly weaker,
and reported as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..model.heads import head_logprob
from ..model.transformer import BeliefTransformer
from ..train.data import EvalSet
from .patching import Component, all_components

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class ScrubHypothesis:
    """A claim about which components matter and what they may depend on."""

    circuit: frozenset[str]
    equivalence: str = "exact"       # "exact" | "belief"
    n_classes: int = 16

    @staticmethod
    def of(labels, **kw) -> ScrubHypothesis:
        return ScrubHypothesis(circuit=frozenset(labels), **kw)


@dataclass
class ScrubResult:
    loss_model: float
    loss_scrubbed: float
    loss_fully_scrubbed: float
    loss_marginal: float
    oracle_loss: float
    faithfulness: float
    faithfulness_stderr: float
    faithfulness_vs_full_scrub: float
    circuit: list[str]
    circuit_size: int
    n_components: int
    equivalence: str
    n_classes: int
    extra: dict = field(default_factory=dict)

    @property
    def gap_model(self) -> float:
        return self.loss_model - self.oracle_loss

    @property
    def gap_scrubbed(self) -> float:
        return self.loss_scrubbed - self.oracle_loss

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["gap_model"] = self.gap_model
        d["gap_scrubbed"] = self.gap_scrubbed
        return d


# ---------------------------------------------------------------------------
# context-free reference predictor
# ---------------------------------------------------------------------------


def marginal_loss(eval_set: EvalSet, head: str, burn_in: int = 0, n_mixture: int = 3) -> float:
    """Loss of the best predictor that ignores context entirely.

    This is the "knows nothing about the belief state" floor the faithfulness
    percentage is measured against.  Fitted on the eval set itself, which makes
    it the *best possible* context-free predictor and therefore a conservative
    (small) denominator.
    """
    obs = eval_set.obs[:, burn_in:]
    if head == "categorical":
        counts = torch.bincount(obs.reshape(-1), minlength=int(obs.max()) + 1).double()
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    y = obs.reshape(-1).double()
    if head == "gaussian":
        var = y.var(unbiased=False)
        return float(0.5 * (LOG_2PI + 1.0 + var.log()))

    # gaussian_mixture: fit a K-component mixture by EM so the reference is not
    # handicapped relative to the head the model actually has.
    k = n_mixture
    g = torch.Generator(device="cpu").manual_seed(0)
    mu = y[torch.randint(0, len(y), (k,), generator=g).to(y.device)].clone()
    var = torch.full((k,), float(y.var(unbiased=False)), dtype=y.dtype, device=y.device)
    w = torch.full((k,), 1.0 / k, dtype=y.dtype, device=y.device)
    for _ in range(200):
        logp = -0.5 * (LOG_2PI + var.log() + (y[:, None] - mu) ** 2 / var) + w.log()
        total = torch.logsumexp(logp, dim=1, keepdim=True)
        resp = torch.exp(logp - total)
        nk = resp.sum(0).clamp_min(1e-9)
        w = nk / nk.sum()
        mu = (resp * y[:, None]).sum(0) / nk
        var = ((resp * (y[:, None] - mu) ** 2).sum(0) / nk).clamp_min(1e-6)
    logp = -0.5 * (LOG_2PI + var.log() + (y[:, None] - mu) ** 2 / var) + w.log()
    return float(-torch.logsumexp(logp, dim=1).mean())


# ---------------------------------------------------------------------------
# equivalence classes and resampling
# ---------------------------------------------------------------------------


def belief_classes(
    eval_set: EvalSet, n_classes: int = 16, seed: int = 0, iters: int = 40
) -> torch.Tensor:
    """Cluster true beliefs into equivalence classes, shape (n_seq, ctx).

    Classes are shared across positions, so a donor may come from any position
    with the same belief -- which is what "depends on the input only through the
    belief state" actually asserts.
    """
    belief = eval_set.belief.reshape(-1, eval_set.belief.shape[-1]).double()
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(belief.shape[0], generator=g)[:n_classes].to(belief.device)
    centres = belief[idx].clone()
    assign = torch.zeros(belief.shape[0], dtype=torch.long, device=belief.device)
    for _ in range(iters):
        new = torch.cdist(belief, centres).argmin(dim=1)
        if torch.equal(new, assign):
            break
        assign = new
        for c in range(n_classes):
            mask = assign == c
            if mask.any():
                centres[c] = belief[mask].mean(0)
    return assign.reshape(eval_set.belief.shape[:2])


def _resample(act: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
    """act[i, t] <- act[donors[i, t], t], preserving trailing dimensions."""
    n, ctx = donors.shape
    flat = act.reshape(n, ctx, -1)
    idx = donors.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
    return flat.gather(0, idx).reshape(act.shape)


class Scrubber:
    """Runs many scrubbing hypotheses against one (model, eval set) pair.

    Activations, equivalence classes and the reference losses are computed once,
    which is what makes the greedy circuit search affordable.
    """

    def __init__(
        self,
        model: BeliefTransformer,
        eval_set: EvalSet,
        burn_in: int = 16,
        n_classes: int = 16,
        n_repeats: int = 3,
        seed: int = 0,
    ):
        self.model = model.eval()
        self.eval_set = eval_set
        self.burn_in = burn_in
        self.n_classes = n_classes
        self.n_repeats = n_repeats
        self.seed = seed

        self.components = all_components(model)
        self.labels = [c.label for c in self.components]
        self._by_label = {c.label: c for c in self.components}
        hooks = sorted({c.hook for c in self.components})
        with torch.no_grad():
            _, self.cache = model.run_with_cache(eval_set.obs, names=hooks)
            self.base = self._loss(model(eval_set.obs))
        self.classes = belief_classes(eval_set, n_classes, seed)
        self.oracle = (-eval_set.oracle_logp[:, burn_in:]).mean(dim=1)
        self.loss_marginal = marginal_loss(
            eval_set, model.cfg.head, burn_in, model.cfg.n_mixture
        )
        self._full_scrub_cache: torch.Tensor | None = None

    # -- internals --------------------------------------------------------

    def _loss(self, params: torch.Tensor) -> torch.Tensor:
        lp = head_logprob(self.model.cfg.head, params, self.eval_set.obs, self.model.cfg.n_mixture)
        return (-lp[:, self.burn_in :]).mean(dim=1)

    def _donors(self, classes: torch.Tensor | None, gen: torch.Generator) -> torch.Tensor:
        n, ctx = self.eval_set.obs.shape
        device = self.eval_set.obs.device
        if classes is None:
            return torch.randint(0, n, (n, ctx), generator=gen).to(device)
        cls = classes.cpu()
        donors = torch.empty((n, ctx), dtype=torch.long)
        for t in range(ctx):
            col = cls[:, t]
            for c in col.unique():
                members = (col == c).nonzero(as_tuple=True)[0]
                pick = torch.randint(0, len(members), (len(members),), generator=gen)
                donors[members, t] = members[pick]
        return donors.to(device)

    @torch.no_grad()
    def _run(self, circuit: frozenset[str], equivalence: str, gen: torch.Generator):
        replacements: dict[str, dict] = {}
        for comp in self.components:
            in_circuit = comp.label in circuit
            if in_circuit and equivalence == "exact":
                continue  # keep the true activation
            classes = self.classes if (in_circuit and equivalence == "belief") else None
            donors = self._donors(classes, gen)
            act = self.cache[comp.hook]
            slot = act if comp.kind == "mlp" else act[:, :, comp.head]
            replacements.setdefault(comp.hook, {})[comp.head] = _resample(slot, donors)

        if not replacements:
            return self.base

        def make(hook: str):
            def fn(x, _name):
                x = x.clone()
                for head, value in replacements[hook].items():
                    if head is None:
                        x = value
                    else:
                        x[:, :, head] = value
                return x

            return fn

        with self.model.hooks([(h, make(h)) for h in replacements]):
            return self._loss(self.model(self.eval_set.obs))

    def _repeat(self, circuit: frozenset[str], equivalence: str, tag: int) -> torch.Tensor:
        runs = []
        for r in range(self.n_repeats):
            gen = torch.Generator(device="cpu").manual_seed(self.seed + 1000 * r + tag)
            runs.append(self._run(circuit, equivalence, gen))
        return torch.stack(runs).mean(0)

    # -- public API -------------------------------------------------------

    def full_scrub_loss(self) -> torch.Tensor:
        if self._full_scrub_cache is None:
            self._full_scrub_cache = self._repeat(frozenset(), "exact", tag=7)
        return self._full_scrub_cache

    def evaluate(self, hypothesis: ScrubHypothesis) -> ScrubResult:
        scrubbed = self._repeat(hypothesis.circuit, hypothesis.equivalence, tag=0)
        full = self.full_scrub_loss()

        base = self.base.cpu().numpy()
        scrub = scrubbed.cpu().numpy()
        denom = self.loss_marginal - base.mean()
        per_seq = 1.0 - (scrub - base) / max(denom, 1e-12)
        full_denom = float(full.mean()) - base.mean()

        return ScrubResult(
            loss_model=float(base.mean()),
            loss_scrubbed=float(scrub.mean()),
            loss_fully_scrubbed=float(full.mean()),
            loss_marginal=self.loss_marginal,
            oracle_loss=float(self.oracle.mean()),
            faithfulness=float(per_seq.mean()),
            faithfulness_stderr=float(per_seq.std(ddof=1) / np.sqrt(len(per_seq))),
            faithfulness_vs_full_scrub=float(
                1.0 - (scrub.mean() - base.mean()) / max(full_denom, 1e-12)
            ),
            circuit=sorted(hypothesis.circuit),
            circuit_size=len(hypothesis.circuit),
            n_components=len(self.components),
            equivalence=hypothesis.equivalence,
            n_classes=self.n_classes if hypothesis.equivalence == "belief" else 0,
            extra={"n_repeats": self.n_repeats, "burn_in": self.burn_in},
        )

    def component_importance(self, equivalence: str = "exact") -> list[dict]:
        """Loss increase from resample-ablating each component across all positions.

        A whole-sequence measure, unlike contrast-pair patching, so it is the right
        signal for ranking components before a circuit search.
        """
        keep_all = frozenset(self.labels)
        out = []
        for label in self.labels:
            circuit = keep_all - {label}
            loss = self._repeat(circuit, equivalence, tag=hash(label) % 997)
            delta = (loss - self.base).cpu().numpy()
            out.append(
                {
                    "component": label,
                    "delta_loss": float(delta.mean()),
                    "stderr": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                }
            )
        return sorted(out, key=lambda d: -d["delta_loss"])

    def greedy_search(
        self,
        target: float = 0.90,
        equivalence: str = "exact",
        max_size: int | None = None,
        verbose: bool = False,
    ) -> dict:
        """Grow a circuit one component at a time, maximising faithfulness.

        Reports the whole trace, not just the winner, so the faithfulness-versus-
        size curve can be inspected -- a circuit that only reaches the target by
        including nine of ten components is not a circuit.
        """
        remaining = list(self.labels)
        circuit: list[str] = []
        trace: list[dict] = []
        max_size = max_size or len(self.labels)

        while remaining and len(circuit) < max_size:
            best_label, best_res = None, None
            for label in remaining:
                res = self.evaluate(
                    ScrubHypothesis.of([*circuit, label], equivalence=equivalence,
                                       n_classes=self.n_classes)
                )
                if best_res is None or res.faithfulness > best_res.faithfulness:
                    best_label, best_res = label, res
            circuit.append(best_label)
            remaining.remove(best_label)
            trace.append(
                {
                    "size": len(circuit),
                    "added": best_label,
                    "circuit": list(circuit),
                    "faithfulness": best_res.faithfulness,
                    "faithfulness_stderr": best_res.faithfulness_stderr,
                    "faithfulness_vs_full_scrub": best_res.faithfulness_vs_full_scrub,
                    "loss_scrubbed": best_res.loss_scrubbed,
                }
            )
            if verbose:
                print(
                    f"    +{best_label:7s} size={len(circuit)} "
                    f"faithfulness={best_res.faithfulness:.3f}",
                    flush=True,
                )
            if best_res.faithfulness >= target:
                break

        return {
            "circuit": circuit,
            "trace": trace,
            "target": target,
            "reached": bool(trace) and trace[-1]["faithfulness"] >= target,
            "n_components": len(self.labels),
            "equivalence": equivalence,
        }


def causal_scrub(
    model: BeliefTransformer,
    eval_set: EvalSet,
    hypothesis: ScrubHypothesis,
    burn_in: int = 16,
    seed: int = 0,
    n_repeats: int = 3,
) -> ScrubResult:
    """One-shot convenience wrapper around :class:`Scrubber`."""
    return Scrubber(
        model, eval_set, burn_in=burn_in, n_classes=hypothesis.n_classes,
        n_repeats=n_repeats, seed=seed
    ).evaluate(hypothesis)


@torch.no_grad()
def ablate_component(
    model: BeliefTransformer,
    eval_set: EvalSet,
    component: Component,
    mode: str = "mean",
    burn_in: int = 16,
    seed: int = 0,
) -> dict:
    """Ablate one component and report the KL-to-optimal blow-up factor.

    ``mode`` is "mean" (replace with the dataset mean -- the on-distribution
    choice and the default), "zero", or "resample".  Zero ablation is
    off-distribution and tends to overstate importance; it is reported for
    comparison, not used as the headline.
    """
    from ..model.heads import kl_true_to_model

    model.eval()
    obs = eval_set.obs
    n, ctx = obs.shape
    hook = component.hook

    _, cache = model.run_with_cache(obs, names=[hook])
    act = cache[hook]
    slot = act if component.kind == "mlp" else act[:, :, component.head]
    if mode == "mean":
        value = slot.mean(dim=0, keepdim=True).expand_as(slot).clone()
    elif mode == "zero":
        value = torch.zeros_like(slot)
    elif mode == "resample":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        donors = torch.randint(0, n, (n, ctx), generator=gen).to(obs.device)
        value = _resample(slot, donors)
    else:
        raise ValueError(f"unknown ablation mode {mode!r}")

    def fn(x, _name):
        x = x.clone()
        if component.kind == "mlp":
            return value
        x[:, :, component.head] = value
        return x

    def kl_of(params):
        kl = kl_true_to_model(model.cfg.head, params, eval_set.predictive, model.cfg.n_mixture)
        return float(kl[:, burn_in:].mean())

    base_kl = kl_of(model(obs))
    with model.hooks([(hook, fn)]):
        params = model(obs)
    abl_kl = kl_of(params)
    lp = head_logprob(model.cfg.head, params, obs, model.cfg.n_mixture)

    return {
        "component": component.label,
        "mode": mode,
        "kl_baseline": base_kl,
        "kl_ablated": abl_kl,
        "kl_ratio": abl_kl / max(base_kl, 1e-12),
        "loss_ablated": float((-lp[:, burn_in:]).mean()),
        "oracle_loss": eval_set.oracle_loss(burn_in),
    }
