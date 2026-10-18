"""Command line entry point.

    bsc processes                 list the zoo with its analytic facts
    bsc train mess3_hard          train one model and report the optimality gap
    bsc run circuits              run one experiment
    bsc run all                   run the whole suite in dependency order
    bsc report                    rebuild RESULTS.md from the JSON in results/
"""

from __future__ import annotations

import argparse
import sys
import time

from .processes import DEFAULT_ZOO, all_names, get_process
from .processes.hmm import HMM
from .utils import RESULTS_DIR, environment_stamp, save_json


def _cmd_processes(args) -> int:
    names = all_names() if args.all else list(DEFAULT_ZOO)
    header = f"{'process':18s} {'kind':10s} {'|S|':>4s} {'|X|':>4s} {'unifilar':>9s} {'SLEM':>6s} {'h (closed form)':>16s}"
    print(header)
    print("-" * len(header))
    for name in names:
        p = get_process(name)
        h = p.entropy_rate()
        uni = slem = "-"
        n_states = p.d_belief
        if isinstance(p, HMM):
            uni = "yes" if p.is_unifilar() else "no"
            slem = f"{p.slem():.3f}"
            n_states = p.n_states
        print(
            f"{name:18s} {p.kind:10s} {n_states:4d} {p.vocab_size:4d} {uni:>9s} {slem:>6s} "
            f"{(f'{h:.6f}' if h is not None else 'n/a'):>16s}"
        )
    return 0


def _cmd_train(args) -> int:
    from .experiments.common import train_cached

    out = train_cached(
        args.process,
        seed=args.seed,
        train_overrides={"steps": args.steps} if args.steps else None,
        use_cache=not args.no_cache,
        verbose=True,
    )
    f = out["final"]
    print(
        f"\n{args.process}: loss={f.loss:.6f} oracle={f.oracle_loss:.6f} "
        f"gap={f.gap:+.6f} +- {f.gap_stderr:.6f} KL={f.kl_to_optimal:.6f}"
    )
    print(f"params={out['n_params']} checkpoint={out['checkpoint']}")
    return 0


def _cmd_run(args) -> int:
    from .experiments import EXPERIMENTS, ORDER

    names = list(ORDER) if args.name == "all" else [args.name]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s): {unknown}; known: {sorted(EXPERIMENTS)}", file=sys.stderr)
        return 2

    t0 = time.time()
    index = {}
    for name in names:
        print(f"\n=== {name} " + "=" * (60 - len(name)), flush=True)
        started = time.time()
        payload = EXPERIMENTS[name]()
        index[name] = {
            "wall_time_s": time.time() - started,
            "summary": payload.get("summary"),
        }
        print(f"--- {name} done in {time.time() - started:.0f}s", flush=True)

    if args.name == "all":
        save_json(
            RESULTS_DIR / "index.json",
            {"experiments": index, "total_wall_time_s": time.time() - t0,
             "environment": environment_stamp()},
        )
        _cmd_report(args)
    return 0


def _cmd_report(args) -> int:
    from .report import build_report

    path = build_report()
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bsc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("processes", help="list the process zoo")
    p.add_argument("--all", action="store_true", help="include processes outside the default zoo")
    p.set_defaults(func=_cmd_processes)

    p = sub.add_parser("train", help="train one model")
    p.add_argument("process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("run", help="run an experiment (or 'all')")
    p.add_argument("name")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="rebuild RESULTS.md from results/*.json")
    p.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
