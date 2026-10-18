"""Experiment 3 - find the circuit, then try to break the claim.

Order of operations matters here:

1. Rank components by whole-sequence resample ablation (a robust signal) and by
   contrast-pair activation patching (a localised one).
2. Check attribution patching against true patching *per process*, because the
   first-order approximation is only trustworthy when the intervention is small
   -- and the interventions are largest exactly where beliefs are deltas.
3. Grow a circuit by greedy causal scrubbing and record the whole
   faithfulness-versus-size curve.  A "circuit" that needs nine of ten
   components is not a circuit, and the curve says so.
4. Re-run the search across seeds.  Different seeds learn different circuits, so
   the claim has to be about the distribution, not about one run.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from ..analysis.patching import (
    Component,
    activation_patch,
    attribution_patch,
    make_contrast_pairs,
    patching_fidelity,
    path_patch,
)
from ..analysis.scrubbing import ScrubHypothesis, Scrubber
from ..utils import RESULTS_DIR, environment_stamp, save_json
from .common import train_cached

CIRCUIT_PROCESSES = ("even", "rrxor", "mess3_hard", "sns", "regimevol")


def _component_from_label(label: str) -> Component:
    layer = int(label[1 : label.index("H")]) if "H" in label else int(label[1 : label.index("M")])
    if "H" in label:
        return Component("head", layer, int(label[label.index("H") + 1 :]))
    return Component("mlp", layer)


def analyse_one(name: str, seed: int, target: float, n_pairs: int, verbose: bool) -> dict:
    out = train_cached(name, seed=seed)
    model, eval_set, process = out["model"], out["eval_set"], out["process"]
    burn_in = out["train_cfg"].burn_in

    scrubber = Scrubber(model, eval_set, burn_in=burn_in, n_repeats=3, seed=seed)
    importance = scrubber.component_importance()

    pairs = make_contrast_pairs(process, eval_set, suffix_len=3, n_pairs=n_pairs, seed=seed)
    true_patch = activation_patch(model, pairs)
    approx_patch = attribution_patch(model, pairs)
    fidelity = patching_fidelity(true_patch, approx_patch)

    search = scrubber.greedy_search(target=target, equivalence="exact")
    circuit = search["circuit"]
    exact = scrubber.evaluate(ScrubHypothesis.of(circuit, equivalence="exact"))
    belief = scrubber.evaluate(ScrubHypothesis.of(circuit, equivalence="belief", n_classes=16))

    from ..analysis.scrubbing import ablate_component

    top = _component_from_label(importance[0]["component"])
    ablations = {
        mode: ablate_component(model, eval_set, top, mode=mode, burn_in=burn_in, seed=seed)
        for mode in ("mean", "zero", "resample")
    }

    paths = []
    if len(circuit) >= 2:
        receivers = sorted({f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)})
        for label in circuit[:3]:
            sender = _component_from_label(label)
            for receiver in receivers:
                if int(receiver.split(".")[1]) <= sender.layer:
                    continue
                paths.append(path_patch(model, pairs, sender, receiver).__dict__)

    row = {
        "process": name,
        "seed": seed,
        "gap": out["final"].gap,
        "n_components": len(scrubber.labels),
        "loss_model": exact.loss_model,
        "loss_marginal": exact.loss_marginal,
        "loss_fully_scrubbed": exact.loss_fully_scrubbed,
        "oracle_loss": exact.oracle_loss,
        "importance": importance,
        "activation_patch": [r.__dict__ for r in true_patch],
        "attribution_patch": [r.__dict__ for r in approx_patch],
        "attribution_fidelity": fidelity,
        "contrast_pairs": {
            "n": len(pairs),
            "belief_tv_distance": pairs.belief_distance,
            "suffix_len": pairs.suffix_len,
            "readout_pos": pairs.readout_pos,
        },
        "greedy_search": search,
        "circuit": circuit,
        "circuit_size": len(circuit),
        "faithfulness_exact": exact.as_dict(),
        "faithfulness_belief": belief.as_dict(),
        "ablation_top_component": ablations,
        "path_patches": paths,
    }
    if verbose:
        print(
            f"  {name:14s} seed={seed} circuit={len(circuit)}/{row['n_components']} "
            f"faith(exact)={exact.faithfulness:.3f} faith(belief)={belief.faithfulness:.3f} "
            f"attrib r={fidelity['pearson_r']:+.2f} "
            f"ablate {top.label} KL x{ablations['mean']['kl_ratio']:.0f}",
            flush=True,
        )
    return row


def run(
    processes: tuple[str, ...] = CIRCUIT_PROCESSES,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    target: float = 0.90,
    n_pairs: int = 256,
    verbose: bool = True,
) -> dict:
    rows = []
    t0 = time.time()
    for name in processes:
        for seed in seeds:
            rows.append(analyse_one(name, seed, target, n_pairs, verbose))

    payload = {
        "experiment": "exp03_circuits",
        "seeds": list(seeds),
        "target_faithfulness": target,
        "rows": rows,
        "wall_time_s": time.time() - t0,
        "environment": environment_stamp(),
        "summary": summarise(rows, seeds),
    }
    save_json(RESULTS_DIR / "exp03_circuits.json", payload)
    return payload


def summarise(rows: list[dict], seeds) -> dict:
    """Per-process seed statistics: circuit size, faithfulness, and reproducibility."""
    by_process: dict[str, list[dict]] = {}
    for r in rows:
        by_process.setdefault(r["process"], []).append(r)

    out = {}
    for name, group in by_process.items():
        sizes = np.array([r["circuit_size"] for r in group])
        faith = np.array([r["faithfulness_exact"]["faithfulness"] for r in group])
        faith_b = np.array([r["faithfulness_belief"]["faithfulness"] for r in group])
        counts = Counter(c for r in group for c in r["circuit"])
        modal = [c for c, n in counts.items() if n == len(group)]
        # a seed "reproduces" the modal circuit if it contains every always-present component
        reproduced = sum(1 for r in group if set(modal).issubset(set(r["circuit"])))
        out[name] = {
            "n_seeds": len(group),
            "n_components": group[0]["n_components"],
            "circuit_size_mean": float(sizes.mean()),
            "circuit_size_std": float(sizes.std(ddof=1)) if len(sizes) > 1 else 0.0,
            "circuit_size_range": [int(sizes.min()), int(sizes.max())],
            "faithfulness_exact_mean": float(faith.mean()),
            "faithfulness_exact_std": float(faith.std(ddof=1)) if len(faith) > 1 else 0.0,
            "faithfulness_belief_mean": float(faith_b.mean()),
            "always_present_components": sorted(modal),
            "component_frequency": dict(counts),
            "seeds_reproducing_core": f"{reproduced}/{len(group)}",
            "attribution_pearson_mean": float(
                np.nanmean([r["attribution_fidelity"]["pearson_r"] for r in group])
            ),
            "ablation_kl_ratio_mean": float(
                np.mean([r["ablation_top_component"]["mean"]["kl_ratio"] for r in group])
            ),
        }
    return out
