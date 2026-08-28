"""Command-line interface.

    python -m greenplan run --config config/city.yaml --mock --recommend
    python -m greenplan run --config config/city.yaml --recommend --project 480
    python -m greenplan horizon --config config/city.yaml --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import engine
from .config import apply_env_overrides, load_config
from .reasoning.client import OpenRouterError
from .training.backtest import max_validatable_horizon


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config/city.yaml", help="path to the YAML config")
    p.add_argument(
        "--mock",
        action="store_true",
        help="run fully offline: mock adapters AND mock reasoning model (no API key, no cost)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="greenplan", description="GreenGrid decision engine")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="train (backtest), then optionally recommend/project")
    _add_common(run_p)
    run_p.add_argument("--iterations", type=int, help="override training.iterations")
    run_p.add_argument("--horizon", type=int, help="override training.horizon_months")
    run_p.add_argument("--memory-k", type=int, help="override training.memory_k")
    run_p.add_argument("--recommend", action="store_true", help="rank zones and write outputs")
    run_p.add_argument(
        "--project",
        type=int,
        metavar="N_MONTHS",
        help="also emit an UNVALIDATED long-range projection N months out (e.g. 480)",
    )
    run_p.add_argument("--seed", type=int, help="override run.seed")

    hz_p = sub.add_parser(
        "horizon", help="report the largest horizon your data can honestly validate"
    )
    _add_common(hz_p)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    # Before the subcommand branches on purpose, so `horizon` honours the
    # environment exactly as `run` does. --mock still wins over it: it is
    # threaded to build_model, which short-circuits to MockModel whatever the
    # provider says (and swaps the data adapters too, which this does not).
    apply_env_overrides(cfg)

    if args.command == "horizon":
        panel = engine.load_panel(cfg, args.mock)
        n = int(panel["month"].max()) - int(panel["month"].min()) + 1
        max_h = max_validatable_horizon(panel, cfg.training.min_history_months)
        print(
            f"{n} months of history, min_history={cfg.training.min_history_months} "
            f"-> largest honestly validatable horizon: {max_h} months.\n"
            f"Anything beyond {max_h} months can only be produced via --project N "
            "and will be labeled UNVALIDATED."
        )
        return 0

    if args.iterations is not None:
        cfg.training.iterations = args.iterations
    if args.horizon is not None:
        cfg.training.horizon_months = args.horizon
    if args.memory_k is not None:
        cfg.training.memory_k = args.memory_k
    if args.seed is not None:
        cfg.run.seed = args.seed

    try:
        summary = engine.run(
            cfg,
            mock=args.mock,
            do_recommend=args.recommend,
            project_months=args.project,
        )
    except (OpenRouterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
