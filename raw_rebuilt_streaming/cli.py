"""Server-ready command boundaries for packed-code streaming evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _state_worker_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-state", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.1)


def _metric_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--process-data-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m raw_rebuilt_streaming",
        description="Packed-code, two-stage sealed Hamming evaluation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze_code = commands.add_parser("freeze-code", help="label-free v1 checkpoint encoding")
    freeze_code.add_argument("--runtime", type=Path, required=True)
    freeze_code.add_argument("--checkpoint", type=Path, required=True)
    freeze_code.add_argument("--output-parent", type=Path, required=True)
    freeze_code.add_argument("--feature-chunk-size", type=int, default=8192)
    freeze_code.add_argument("--max-new-chunks", type=int)
    freeze_code.add_argument("--device", default="auto")
    freeze_code.add_argument("--process-data-root", type=Path)

    import_baseline = commands.add_parser(
        "import-baseline-code",
        help="verify and pack one sealed fixed-feature baseline code artifact",
    )
    import_baseline.add_argument("--artifact", type=Path, required=True)
    import_baseline.add_argument("--checkpoint", type=Path, required=True)
    import_baseline.add_argument("--output-parent", type=Path, required=True)

    freeze_plan = commands.add_parser("freeze-plan", help="freeze all cells before labels")
    freeze_plan.add_argument("--code-state", type=Path, required=True)
    freeze_plan.add_argument("--output-parent", type=Path, required=True)
    freeze_plan.add_argument("--bits", type=_csv_ints, default=(16, 32, 64))
    freeze_plan.add_argument("--directions", type=_csv_strings, default=("i2t", "t2i"))
    freeze_plan.add_argument("--query-chunk-size", type=int, default=8)
    freeze_plan.add_argument("--cutoffs", type=_csv_ints, default=(50, 100, 1000))

    rank = commands.add_parser("rank-worker", help="label-free distance producer")
    _state_worker_common(rank)
    rank.add_argument("--max-new-bundles", type=int)
    rank.add_argument("--rank-device", choices=("cpu", "cuda"), default="cpu")
    rank.add_argument("--serve", action="store_true")

    metric = commands.add_parser("metric-worker", help="post-plan label/metric consumer")
    _state_worker_common(metric)
    _metric_runtime(metric)
    metric.add_argument("--output-parent", type=Path, required=True)
    metric.add_argument("--max-new-acks", type=int)
    metric.add_argument("--serve", action="store_true")

    evaluate = commands.add_parser(
        "stream-evaluate", help="spawn isolated rank and metric workers"
    )
    _state_worker_common(evaluate)
    _metric_runtime(evaluate)
    evaluate.add_argument("--output-parent", type=Path, required=True)
    evaluate.add_argument("--rank-device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-code":
        from .codes import CodeFreezeConfig, freeze_code_state

        root = freeze_code_state(
            args.runtime,
            args.checkpoint,
            args.output_parent,
            config=CodeFreezeConfig(feature_chunk_size=args.feature_chunk_size),
            device=args.device,
            max_new_chunks=args.max_new_chunks,
            process_data_root=args.process_data_root,
        )
        print(json.dumps({"code_state": str(root)}, ensure_ascii=False))
        return 0
    if args.command == "freeze-plan":
        from .plan import StreamingPlanConfig, freeze_rank_plan

        config = StreamingPlanConfig(
            bits=tuple(args.bits),
            directions=tuple(args.directions),
            query_chunk_size=args.query_chunk_size,
            cutoffs=tuple(args.cutoffs),
        )
        root = freeze_rank_plan(args.code_state, args.output_parent, config=config)
        print(json.dumps({"rank_plan": str(root)}, ensure_ascii=False))
        return 0
    if args.command == "import-baseline-code":
        from .baseline_import import import_baseline_code_artifact

        root = import_baseline_code_artifact(
            args.artifact,
            args.checkpoint,
            args.output_parent,
        )
        print(json.dumps({"code_state": str(root)}, ensure_ascii=False))
        return 0
    if args.command == "rank-worker":
        from .rank_worker import produce_rank_bundles, serve_rank_worker

        if args.serve:
            result = serve_rank_worker(
                args.code_state,
                args.plan,
                args.spool,
                poll_seconds=args.poll_seconds,
                rank_device=args.rank_device,
            )
        else:
            result = produce_rank_bundles(
                args.code_state,
                args.plan,
                args.spool,
                max_new_bundles=args.max_new_bundles,
                rank_device=args.rank_device,
            )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "metric-worker":
        from .metric_worker import consume_metric_bundles, serve_metric_worker

        if args.serve:
            result = serve_metric_worker(
                args.runtime,
                args.code_state,
                args.plan,
                args.spool,
                args.output_parent,
                poll_seconds=args.poll_seconds,
                process_data_root=args.process_data_root,
            )
        else:
            result = consume_metric_bundles(
                args.runtime,
                args.code_state,
                args.plan,
                args.spool,
                args.output_parent,
                max_new_acks=args.max_new_acks,
                process_data_root=args.process_data_root,
            )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "stream-evaluate":
        from .orchestrator import run_streaming_evaluation

        root = run_streaming_evaluation(
            args.runtime,
            args.code_state,
            args.plan,
            args.spool,
            args.output_parent,
            poll_seconds=args.poll_seconds,
            rank_device=args.rank_device,
            process_data_root=args.process_data_root,
        )
        print(json.dumps({"evaluation_root": str(root)}, ensure_ascii=False))
        return 0
    raise AssertionError("unhandled command")


__all__ = ["build_parser", "main"]
