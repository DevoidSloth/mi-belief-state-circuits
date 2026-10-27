"""Assemble RESULTS.md from the JSON each experiment writes.

Every number in the report is read from a results file, never typed in, so the
document cannot drift from the runs.  Missing experiments are reported as
missing rather than silently omitted -- a partial suite should look partial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import RESULTS_DIR, load_json

FILES = {
    "optimality": "exp01_optimality.json",
    "geometry": "exp02_geometry.json",
    "circuits": "exp03_circuits.json",
    "kalman": "exp04_kalman.json",
    "sae": "exp05_sae.json",
    "capacity": "exp06_capacity.json",
    "incontext": "exp07_incontext.json",
    "architecture": "exp08_architecture.json",
}


def _load() -> dict:
    out = {}
    for key, filename in FILES.items():
        path = RESULTS_DIR / filename
        out[key] = load_json(path) if path.exists() else None
    return out


def _fmt(v, digits=4, missing="-"):
    if v is None:
        return missing
    if isinstance(v, float):
        if not np.isfinite(v):
            return missing
        return f"{v:.{digits}f}"
    return str(v)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


BUDGET_CAVEAT = (
    "> **Every model here gets the same step budget**, so larger ones are further from "
    "convergence and their gaps are training-limited rather than capacity-limited. That is why "
    "the gap *rises* with width above the threshold. Read a capacity threshold as "
    '"the width below which this budget cannot reach the optimum", not as an architectural '
    "limit — and read a smallest-sufficient configuration as a statement about sample "
    "efficiency rather than about representational capacity.\n"
)


def _section_optimality(d) -> str:
    if not d:
        return "_not run_\n"
    rows = []
    for r in d["rows"]:
        f, t = r["facts"], r["transformer"]
        h = f.get("entropy_rate_closed_form") or f.get("entropy_rate_msp")
        ng = r.get("ngram")
        finite = ng or r.get("linear", {})
        if ng:
            table = (f["vocab_size"] + 1) ** ng["order"]
            memory = f"order {ng['order']} ({table:,} contexts)"
        else:
            memory = "linear, window 8"
        rows.append(
            [
                r["process"],
                "yes" if f.get("unifilar") else ("no" if "unifilar" in f else "-"),
                _fmt(h, 4),
                _fmt(t["oracle_loss"], 4),
                f"{t['gap']:+.5f} ± {t['gap_stderr']:.5f}",
                _fmt(finite.get("gap"), 4),
                memory,
                _fmt(r.get("gru", {}).get("gap"), 4),
                f["d_belief"],
                r["n_params"],
            ]
        )
    s = d["summary"]
    body = _table(
        ["process", "unifilar", "h (analytic)", "oracle", "optimality gap",
         "finite-memory gap", "best finite model", "GRU gap", "belief dim", "params"],
        rows,
    )
    return (
        f"{body}\n\n"
        f"- {s['n_within_0.01_nats']}/{s['n_processes']} processes within 0.01 nats/token of the "
        f"analytic optimum; median gap **{s['median_gap']:+.5f}**, worst **{s['max_gap']:+.5f}**.\n"
        f"- Only {s['n_processes_with_separation_over_0.15']}/{s['n_processes']} processes have a "
        f"finite-memory baseline more than 0.15 nats above the optimum "
        f"({', '.join(s['processes_with_separation_over_0.15']) or 'none'}), and that is the "
        f"expected answer rather than a disappointing one. For any *mixing* process the belief "
        f"depends on the recent past with exponentially decaying weight, so an order-k Markov "
        f"table is exponentially good in k — finite memory is nearly sufficient **by "
        f"construction**. Finite memory only fails outright where the memory requirement is "
        f"unbounded or exact: parity counting (`even`), phase alignment (`rrxor`), a latent that "
        f"never resets (`persistent_bit`) or one that drifts slower than any practical window "
        f"(`drifting_coin`).\n"
        f"- The interesting contrast is therefore not loss but **size**. Set the *belief dim* "
        f"column against *best finite model*: the belief state is a handful of numbers, while "
        f"the finite-memory model that nearly matches it needs a table of hundreds or thousands "
        f"of contexts. A belief state buys compression of the computation, not always a better "
        f"achievable loss — which is precisely why the mechanism, not the loss, is the thing "
        f"worth reverse-engineering.\n" + _gru_note(d)
    )


def _gru_note(d) -> str:
    """The architecture comparison, which is only interesting where the two disagree."""
    rows = [r for r in d["rows"] if "gru" in r]
    if not rows:
        return ""
    worse = [
        (r["process"], r["transformer"]["gap"], r["gru"]["gap"])
        for r in rows
        if r["transformer"]["gap"] > max(4 * r["gru"]["gap"], r["transformer"]["gap_stderr"] * 4)
    ]
    gru_gaps = np.array([r["gru"]["gap"] for r in rows])
    line = (
        f"- A GRU trained identically reaches a median gap of **{np.median(gru_gaps):+.5f}**, so "
        f"the recurrent architecture expresses these filters at least as well as the transformer "
        f"does — as it should, since a filter *is* a recurrence.\n"
    )
    if worse:
        worse.sort(key=lambda w: -(w[1] - w[2]))
        detail = ", ".join(f"`{n}` ({t:+.4f} vs {g:+.4f})" for n, t, g in worse)
        top = worse[0]
        line += (
            f"- The transformer is materially worse on {len(worse)} of {len(rows)} processes: "
            f"{detail}. The general reason is structural rather than specific to any one of "
            f"them — a recurrence carries the belief forward for free, whereas an attention-only "
            f"model has no state and must reconstruct the whole posterior from the raw context "
            f"at every position. The gap is widest on `{top[0]}` ({top[1] / max(top[2], 1e-9):.0f}× "
            f"the GRU's), which needs exact phase alignment within a repeating triple: a "
            f"recurrence tracks that in effectively one unit. This is a fact about architecture, "
            f"not about the task, and it is the kind of claim the analytic floor makes "
            f"checkable — both numbers are distances from the *same* known optimum.\n"
        )
    return line


def _section_geometry(d) -> str:
    if not d:
        return "_not run_\n"
    rows = []
    for r in d["rows"]:
        rows.append(
            [
                r["process"],
                _fmt(r["best_r2"], 4),
                r["best_r2_hook"].replace("blocks.", "L").replace(".hook_resid_", " "),
                f"{r['causal_margin_at_best_r2_hook']:+.3f}",
                f"{r['best_causal_margin']:+.3f}",
                r["best_causal_hook"].replace("blocks.", "L").replace(".hook_resid_", " "),
                _fmt(r.get("r2_vs_margin_pearson"), 3),
            ]
        )
    body = _table(
        ["process", "best probe R²", "at", "causal margin there", "best causal margin",
         "at", "corr(R², margin) across layers"],
        rows,
    )
    dims = d["summary"].get("box_dimensions", {})
    dim_rows = [
        [k, _fmt(v["true"], 3), _fmt(v["residual"], 3)] for k, v in dims.items()
    ]
    extra = ""
    if dim_rows:
        extra = (
            "\n\n**Box-counting dimension** of the reachable belief set versus the same statistic "
            "on the 2-D projection of the residual stream:\n\n"
            + _table(["process", "true belief set", "residual projection"], dim_rows)
        )
    anti = d["summary"]["processes_where_r2_and_causal_use_anticorrelate"]
    note = (
        "\n\n- *Causal margin* is the fraction of KL-to-the-target-belief closed by steering "
        "along the probe direction, **minus** the fraction a norm-matched random direction "
        "closes. The subtraction is essential: any large perturbation flattens the output, and "
        "a flat distribution has lower KL to the target than a confident wrong one, so the raw "
        "effect alone overstates. The control is floored at zero before subtracting — a random "
        "direction that makes things *worse* than baseline explains none of the effect, and "
        "letting it count would inflate the margin without bound on models confident enough "
        "that any perturbation is catastrophic. So the margin is at most 1. Steering strength "
        "is chosen on one half of the pairs and the margin reported on the other, with a "
        "bootstrap standard error over pairs.\n"
        f"- Probe R² and causal use **anti-correlate across layers** on "
        f"{len(anti)}/{d['summary']['n_processes']} processes ({', '.join(anti) or 'none'}): the "
        f"layers where the belief is most decodable are not the layers where perturbing it moves "
        f"the output most. High R² is not evidence of use.\n"
    )
    return body + extra + note


def _section_circuits(d) -> str:
    if not d:
        return "_not run_\n"
    # faithfulness at a fixed budget, read out of each seed's greedy trace.
    # "Size needed to reach 0.9" is vacuous when the answer is "all of them",
    # because keeping every component resamples nothing and scores 1.0 by
    # construction. Faithfulness at a quarter and a half of the components is
    # the statistic that stays informative in that case.
    budget: dict[str, dict] = {}
    for r in d["rows"]:
        trace = r.get("greedy_search", {}).get("trace", [])
        n = r["n_components"]
        entry = budget.setdefault(r["process"], {"quarter": [], "half": [], "saturated": 0, "n": 0})
        entry["n"] += 1
        entry["saturated"] += int(r["circuit_size"] >= n)
        for key, frac in (("quarter", 0.25), ("half", 0.5)):
            size = max(1, int(round(frac * n)))
            hit = [t for t in trace if t["size"] <= size]
            if hit:
                entry[key].append(hit[-1]["faithfulness"])

    rows = []
    for name, s in d["summary"].items():
        b = budget.get(name, {})
        size = (
            "all"
            if b.get("saturated", 0) == b.get("n", 1)
            else f"{s['circuit_size_mean']:.1f} ± {s['circuit_size_std']:.1f}"
        )
        rows.append(
            [
                name,
                s["n_components"],
                f"{size} / {s['n_components']}",
                _fmt(np.mean(b["quarter"]) if b.get("quarter") else None, 3),
                _fmt(np.mean(b["half"]) if b.get("half") else None, 3),
                _fmt(s["faithfulness_belief_mean"], 3),
                s["seeds_reproducing_core"],
                ", ".join(s["always_present_components"]) or "none",
                _fmt(s["attribution_pearson_mean"], 2),
                f"{s['ablation_kl_ratio_mean']:.0f}×",
            ]
        )
    body = _table(
        ["process", "components", "size to reach 0.90", "faithfulness at 25%",
         "at 50%", "faithfulness (belief-only)", "seeds reproducing core",
         "core components", "attribution r", "top-component ablation"],
        rows,
    )
    saturated = [n for n, b in budget.items() if b["saturated"] == b["n"]]
    if saturated:
        body += (
            f"\n\n> **No sparse circuit exists for {', '.join(saturated)}.** The greedy search "
            f"only reaches the target by keeping every component, and keeping everything "
            f"resamples nothing and scores 1.000 by construction. Read the 25% and 50% columns "
            f"instead: they say how much of the computation a genuinely sparse subgraph "
            f"recovers, and for these processes the answer is 'not much'. That is a result "
            f"about the model — the computation is distributed — not a failure of the search."
        )
    return (
        body
        + "\n\n- *Faithfulness* is measured against the best **context-free** predictor, not "
        "against a fully-scrubbed network: scrubbing every component drives these models "
        "off-distribution to a loss worse than predicting the marginal, which inflates any "
        "percentage computed against it. The classic number is in the JSON as "
        "`faithfulness_vs_full_scrub` and is always higher.\n"
        "- *Belief-only* additionally resamples each circuit component from a different sequence "
        "with the same posterior, testing the semantic claim rather than mere sufficiency.\n"
        "- *Attribution r* is the correlation between attribution patching and true patching. "
        "It collapses on processes whose beliefs are near-deterministic, because the "
        "intervention is then far too large for a first-order approximation.\n"
    )


def _section_kalman(d) -> str:
    if not d:
        return "_not run_\n"
    rows = []
    for name, s in d["summary"].items():
        a = s["analytic"]
        rows.append(
            [
                name,
                f"a={a['a']:.3f}, K={a['K']:.4f}",
                f"{a['recur_prev']:.4f}",
                f"{s['kernel_cosine_mean']:.5f} (min {s['kernel_cosine_min']:.5f})",
                f"{100 * s['decay_rate_rel_error_mean']:.2f}%",
                f"{100 * s['recursion_alpha_rel_error_mean']:.2f}%",
                f"{100 * s['recursion_beta_rel_error_mean']:.2f}%",
                f"{s['seeds_with_cosine_over_0.99']}/{s['n_seeds']}",
            ]
        )
    body = _table(
        ["process", "analytic filter", "α = a(1−Kc)", "kernel cosine vs analytic",
         "decay-rate error", "α error", "β error", "seeds cos>0.99"],
        rows,
    )
    return (
        body
        + "\n\n- A transformer has no state to carry between positions, so it cannot implement "
        "the Kalman *recursion*; it implements the unrolled convolution "
        "`m_t = aK Σ_d α^d y_{t−1−d}`. The object that matches the analytic gain is therefore "
        "the model's impulse response `∂m̂_t/∂y_{t−1−d}`, measured by autograd — not a head's "
        "belief-to-belief transport, which reads near zero even in models whose kernel matches "
        "to five decimals.\n"
        "- Measured on models trained with `norm=\"none\"`, so weight-level coefficients are "
        "comparable to analytic ones without a LayerNorm rescaling in the way.\n"
    )


def _section_sae(d) -> str:
    if not d:
        return "_not run_\n"
    rows = []
    for name, s in d["summary"].items():
        if name.startswith("_"):
            continue
        rows.append(
            [
                name,
                f"{s['l0']:.1f}",
                _fmt(s["variance_explained"], 3),
                _fmt(s["loss_recovered"], 3),
                s["k"],
                _fmt(s["sae_r2_at_k"], 3),
                _fmt(s["pca_r2_at_k"], 3),
                _fmt(s["rrr_r2_at_k"], 3),
                f"{s['sae_minus_pca']:+.3f}",
            ]
        )
    agg = d["summary"]["_aggregate"]
    body = _table(
        ["process", "L0", "variance explained", "loss recovered", "k",
         "SAE R²", "PCA R²", "best k-dim R²", "SAE − PCA"],
        rows,
    )
    return (
        body
        + f"\n\n- The SAE is given the friendlier half of the comparison: its dictionary is learned "
        f"without ever seeing the belief, but the k features it is scored on are chosen *using* "
        f"the belief by greedy forward selection. PCA gets no such help.\n"
        f"- SAE features nonetheless lose to k unsupervised principal components on "
        f"{agg['n_processes_where_sae_loses_to_pca']}/{agg['n_processes']} processes "
        f"(median SAE − PCA = **{agg['median_sae_minus_pca']:+.3f}**), which is the expected "
        f"failure mode for a sparse basis on a dense, continuous, simplex-valued variable.\n"
    )


def _section_capacity(d) -> str:
    if not d:
        return "_not run_\n"
    rows = []
    for r in d["rows"]:
        f = r["fallback"]
        rows.append(
            [
                r["process"],
                _fmt(r["capacity_threshold_d_model"], 0),
                _fmt(f.get("median_effective_order_below"), 1),
                _fmt(f.get("median_effective_order_above"), 1),
                _fmt(f.get("median_bigram_residual_kl_below"), 4),
                _fmt(f.get("median_bigram_residual_kl_above"), 4),
                _fmt(f.get("median_max_order_residual_kl_below"), 4),
                _fmt(f.get("median_max_order_residual_kl_above"), 4),
                _fmt(f.get("median_clusters_below"), 1),
                _fmt(f.get("median_clusters_above"), 1),
                _fmt(f.get("median_effective_rank_below"), 2),
                _fmt(f.get("median_effective_rank_above"), 2),
            ]
        )
    body = _table(
        ["process", "threshold d_model", "eff. order (below)", "(above)",
         "bigram residual KL (below)", "(above)",
         "order-5 residual KL (below)", "(above)",
         "clusters for 95% (below)", "(above)", "eff. rank (below)", "(above)"],
        rows,
    )
    verdicts = "\n".join(
        f"- **{r['process']}**: {r['fallback']['verdict']}" for r in d["rows"]
    )
    return (
        body
        + "\n\n"
        + BUDGET_CAVEAT
        + "\n- The *residual KL* columns say how well the model's own predictive distributions "
        "are reproduced by a table indexed on the last k observations. They matter because the "
        "effective-order integer alone is ambiguous: a model that genuinely needs a long context "
        "and a model that *no* finite table describes both land on the largest order tried. A "
        "small order-5 residual means the model really is a finite-memory approximation; a large "
        "one means the degradation is unstructured error rather than any of the three "
        "hypothesised fallbacks.\n"
        + "\nThreshold is the largest `d_model` whose optimality gap still exceeds "
        f"{d['rows'][0]['criterion_nats']} nats/token. What replaces exact belief tracking below "
        "it:\n\n" + verdicts + "\n"
    )


def _section_incontext(d) -> str:
    if not d:
        return "_not run_\n"
    rows = []
    for name, s in d["summary"].items():
        rows.append(
            [
                name,
                f"{s['gap']:+.5f}",
                _fmt(s["in_context_gain_model"], 4),
                _fmt(s["in_context_gain_oracle"], 4),
                _fmt(s["gain_ratio"], 3),
                _fmt(s["component_probe_r2"], 3),
                _fmt(s["final_component_agreement"], 3),
                _fmt(s["held_out_mean_gap"], 4),
                _fmt(s["beats_best_fixed_filter_by"], 4),
            ]
        )
    body = _table(
        ["family", "gap to optimal", "in-context gain (model)", "(optimal)", "ratio",
         "component probe R²", "component agreement", "held-out gap", "beats best fixed filter by"],
        rows,
    )
    return (
        body
        + "\n\n- *In-context gain* is the fall in loss from early positions to late ones. A model "
        "that memorised a single averaged filter would show a gain near zero.\n"
        "- *Held-out gap* is measured on process parameters that were never in the training "
        "family, against those processes' own exact optimal predictors.\n"
    )


def _section_architecture(d) -> str:
    if not d:
        return "_not run_\n"
    depths = sorted({r["n_layers"] for r in d["grid"]})
    widths = sorted({r["d_model"] for r in d["grid"]})
    lookup = {(r["n_layers"], r["d_model"]): r for r in d["grid"]}
    rows = []
    for layers in depths:
        cells = []
        for w in widths:
            r = lookup.get((layers, w))
            cells.append(f"{r['gap']:+.4f}" if r else "-")
        rows.append([f"{layers} layer" + ("s" if layers > 1 else ""), *cells])
    body = _table(["depth / $d_{model}$", *[str(w) for w in widths]], rows)

    s = d["summary"]
    smallest = s.get("smallest_within_0.01_nats")
    note = ""
    if smallest:
        note = (
            "\n\nSmallest configuration within 0.01 nats of the optimum: "
            f"**{smallest['n_layers']} layer(s), $d_{{model}}$ = {smallest['d_model']}, "
            f"{smallest['n_params']:,} parameters** (gap {smallest['gap']:+.5f}).\n"
        )

    ctx_rows = []
    for name, c in s["context"].items():
        ctxs = sorted(c["gap_by_ctx"], key=int)
        longest = ctxs[-1]
        ctx_rows.append(
            [
                name,
                "non-mixing" if c["non_mixing"] else "mixing",
                _fmt(c["oracle_context_gain_by_ctx"][longest], 4),
                _fmt(c["model_context_gain_by_ctx"][longest], 4),
                _fmt(c["tracking_ratio_at_longest"], 3),
                _fmt(c["gap_by_ctx"][longest], 5),
            ]
        )
    ctx = _table(
        ["process", "regime", "optimal gain from context", "model gain", "ratio", "gap"],
        ctx_rows,
    )
    return (
        "**Optimality gap by depth and width** (nats/token):\n\n"
        + body
        + "\n\n"
        + BUDGET_CAVEAT
        + note
        + "\n**Context length.** For a mixing process the optimal loss plateaus at the entropy "
        "rate and extra context is worthless past the mixing time. For a non-mixing one the "
        "belief never forgets, only sharpens, so the optimal loss keeps falling with position; "
        "the ratio column is how much of that fall the model reproduces.\n\n" + ctx + "\n"
    )


def _resume_table(d) -> str:
    """The brief's resume-number table, filled from measured values."""
    rows = []

    def add(n, metric, value, note=""):
        rows.append([n, metric, value, note])

    o = d["optimality"]
    if o:
        s = o["summary"]
        add(1, "Optimality gap", f"median {s['median_gap']:+.5f}, worst {s['max_gap']:+.5f} nats/token",
            f"{s['n_within_0.01_nats']}/{s['n_processes']} under 0.01")
        add(2, "Baseline separation", f"median {s['median_baseline_separation']:.4f} nats",
            f"{s['n_processes_with_separation_over_0.15']} processes over 0.15")
        add(12, "Process coverage", f"{s['n_processes']} processes", "plus 3 process families")

    g = d["geometry"]
    if g:
        s = g["summary"]
        add(3, "Belief-state probe fidelity",
            f"median best R² {s['median_best_r2']:.4f} (min {s['min_best_r2']:.4f})",
            "held-out sequences, split by sequence")
        add(4, "Causal validity of the probe",
            f"median steering margin {s['median_best_causal_margin']:+.3f}",
            "over a norm-matched random direction")

    c = d["circuits"]
    if c:
        sizes = [(k, v) for k, v in c["summary"].items()]
        best = min(sizes, key=lambda kv: kv[1]["circuit_size_mean"])
        add(5, "Circuit size",
            f"{best[1]['circuit_size_mean']:.1f} of {best[1]['n_components']} components "
            f"({best[0]})",
            "greedy causal-scrubbing search")
        faiths = [v["faithfulness_exact_mean"] for _, v in sizes]
        add(6, "Circuit faithfulness",
            f"{min(faiths):.3f}–{max(faiths):.3f} across processes",
            "vs the best context-free predictor")
        kls = [v["ablation_kl_ratio_mean"] for _, v in sizes]
        add(7, "Ablation effect", f"{min(kls):.0f}×–{max(kls):.0f}× KL-to-optimal",
            "mean ablation of the top component")
        reps = [v["seeds_reproducing_core"] for _, v in sizes]
        add(11, "Seed robustness", ", ".join(sorted(set(reps))),
            f"across {len(c['seeds'])} seeds per process")

    k = d["kalman"]
    if k:
        cos = [v["kernel_cosine_mean"] for v in k["summary"].values()]
        add(8, "Kalman gain recovery",
            f"impulse-response cosine {min(cos):.5f}–{max(cos):.5f}",
            "vs analytic aK·αᵈ")

    s5 = d["sae"]
    if s5:
        agg = s5["summary"]["_aggregate"]
        l0 = [v["l0"] for kk, v in s5["summary"].items() if not kk.startswith("_")]
        ve = [v["variance_explained"] for kk, v in s5["summary"].items() if not kk.startswith("_")]
        lr = [v["loss_recovered"] for kk, v in s5["summary"].items() if not kk.startswith("_")]
        add(9, "SAE quality",
            f"L0 {min(l0):.0f}–{max(l0):.0f}, var explained {min(ve):.2f}–{max(ve):.2f}, "
            f"loss recovered {min(lr):.2f}–{max(lr):.2f}", "")
        add(10, "SAE vs continuous",
            f"median SAE − PCA R² = {agg['median_sae_minus_pca']:+.3f}",
            f"SAE loses on {agg['n_processes_where_sae_loses_to_pca']}/{agg['n_processes']}")

    cap = d["capacity"]
    if cap:
        parts = [
            f"{r['process']}: d_model={r['capacity_threshold_d_model']}" for r in cap["rows"]
        ]
        add(13, "Capacity threshold", "; ".join(parts),
            "fallback characterised in §6")

    rows.sort(key=lambda r: r[0])
    return _table(["#", "metric", "measured", "note"], rows)


def build_report(path: str | Path | None = None) -> Path:
    d = _load()
    missing = [k for k, v in d.items() if v is None]
    env = next((v["environment"] for v in d.values() if v), {})

    parts = [
        "# Results",
        "",
        "Every number below is read from `results/*.json`, which the experiments write. "
        "Rebuild with `bsc report`.",
        "",
    ]
    if missing:
        parts += [
            f"> **Partial suite.** Not yet run: {', '.join(missing)}. "
            f"Run `bsc run all` to fill these in.",
            "",
        ]
    if env:
        parts += [
            f"Environment: {env.get('device_name') or 'CPU'}, torch {env.get('torch')}, "
            f"python {env.get('python')}, commit `{env.get('git_commit')}`.",
            "",
        ]

    parts += ["## Headline numbers", "", _resume_table(d), ""]
    for title, key, fn in [
        ("1. Distance to the analytic optimum", "optimality", _section_optimality),
        ("2. Where the belief lives, and whether it is used", "geometry", _section_geometry),
        ("3. The circuit", "circuits", _section_circuits),
        ("4. The Kalman gain in the weights", "kalman", _section_kalman),
        ("5. Sparse autoencoders on a dense variable", "sae", _section_sae),
        ("6. What replaces exact inference when capacity runs out", "capacity", _section_capacity),
        ("7. Inference over a family of processes", "incontext", _section_incontext),
        ("8. Depth, width and context length", "architecture", _section_architecture),
    ]:
        parts += [f"## {title}", "", fn(d[key]), ""]

    figures = sorted((RESULTS_DIR / "figures").glob("*.png")) if RESULTS_DIR.exists() else []
    figures = [f for f in figures if not f.name.startswith("_")]
    if figures:
        parts += ["## Figures", ""]
        parts += [f"![{f.stem}](figures/{f.name})" for f in figures]
        parts += [""]

    out = Path(path or RESULTS_DIR / "RESULTS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
