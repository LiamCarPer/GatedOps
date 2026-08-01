"""Command-line interface for GatedOps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gatedops.config import load_run_config
from gatedops.pipelines.run import PipelineResult, run_pipeline

_DEMO_SIGNAL = {"good": 0.95, "bad": 0.1}


def _default_config() -> Path:
    """Locate the pipeline config for the current context.

    Prefers ``configs/pipeline.yaml`` next to the working directory (works both
    from the repository root and from a container working directory), falling
    back to the file bundled with the source layout.
    """
    local = Path.cwd() / "configs" / "pipeline.yaml"
    if local.is_file():
        return local
    return Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"


def _print_result(result: PipelineResult) -> None:
    print(f"run_id:      {result.run_id}")
    print(f"version:     {result.model_version}")
    metrics = ", ".join(f"{key}={value:.4f}" for key, value in result.metrics.items())
    print(f"metrics:     {metrics}")
    print(f"gate:        {result.gate.status} ({result.gate.summary})")
    if result.receipt is not None:
        print(
            f"promoted:    {result.receipt.from_stage} -> {result.receipt.to_stage}"
        )
    else:
        print("promoted:    no")


def _print_failures(result: PipelineResult) -> None:
    for check in result.gate.checks:
        if not check.passed:
            detail = check.detail
            if not detail and check.actual is not None and check.threshold is not None:
                detail = f"actual {check.actual:.4f} vs threshold {check.threshold:.4f}"
            print(f"  FAIL {check.rule:<9} {check.metric:<18} {detail}")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    result = run_pipeline(
        config,
        tracking_uri=args.tracking_uri,
        signal_strength=args.signal_strength,
        seed=args.seed,
        promote_on_pass=not args.no_promote,
    )
    _print_result(result)
    if result.gate.status == "FAIL":
        _print_failures(result)
        return 1
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    result = run_pipeline(
        config,
        tracking_uri=args.tracking_uri,
        signal_strength=_DEMO_SIGNAL[args.scenario],
    )
    _print_result(result)
    if result.gate.status == "FAIL":
        _print_failures(result)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gatedops",
        description="Reference MLOps platform: gated train/evaluate/promote/serve.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the training pipeline end to end")
    run_parser.add_argument("--config", type=Path, default=_default_config())
    run_parser.add_argument("--signal-strength", type=float, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--tracking-uri", default=None)
    run_parser.add_argument("--no-promote", action="store_true")
    run_parser.set_defaults(handler=cmd_run)

    demo_parser = subparsers.add_parser("demo", help="run a scripted good or bad scenario")
    demo_parser.add_argument("scenario", choices=["good", "bad"])
    demo_parser.add_argument("--config", type=Path, default=_default_config())
    demo_parser.add_argument("--tracking-uri", default=None)
    demo_parser.set_defaults(handler=cmd_demo)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
