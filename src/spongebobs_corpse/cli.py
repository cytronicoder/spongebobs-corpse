from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline.run import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spongebobs-corpse", description="Research pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    pipeline = subparsers.add_parser("pipeline", help="Pipeline operations")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command")

    run_batch = pipeline_sub.add_parser(
        "run-batch",
        help="Canonical command: regenerate all required batch deliverables",
    )
    run_batch.add_argument("--input", type=Path, default=Path("batch/data.csv"))
    run_batch.add_argument("--out", type=Path, default=Path("batch/outputs"))
    run_batch.add_argument("--config", type=Path, default=None)
    run_batch.add_argument("--seed", type=int, default=None)
    run_batch.add_argument("--no-verify", action="store_true")

    run_cmd = pipeline_sub.add_parser("run", help="Alias of `pipeline run-batch`")
    run_cmd.add_argument("--input", type=Path, default=Path("batch/data.csv"))
    run_cmd.add_argument("--out", type=Path, default=Path("batch/outputs"))
    run_cmd.add_argument("--config", type=Path, default=None)
    run_cmd.add_argument("--seed", type=int, default=None)
    run_cmd.add_argument("--no-verify", action="store_true")

    direct = subparsers.add_parser("run", help="Alias of `pipeline run-batch`")
    direct.add_argument("--input", type=Path, default=Path("batch/data.csv"))
    direct.add_argument("--out", type=Path, default=Path("batch/outputs"))
    direct.add_argument("--config", type=Path, default=None)
    direct.add_argument("--seed", type=int, default=None)
    direct.add_argument("--no-verify", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "pipeline" and args.pipeline_command in {"run-batch", "run"}:
        run_pipeline(
            input_csv=args.input,
            out_dir=args.out,
            config_path=args.config,
            seed_override=args.seed,
            verify=not args.no_verify,
        )
        return 0

    if args.command == "run":
        run_pipeline(
            input_csv=args.input,
            out_dir=args.out,
            config_path=args.config,
            seed_override=args.seed,
            verify=not args.no_verify,
        )
        return 0

    parser.print_help()
    return 1
