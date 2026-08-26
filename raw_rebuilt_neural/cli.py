"""Command-line process boundaries for the raw-rebuilt neural runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from .fit_artifact import prepare_fit_artifact
from .metrics import evaluate_frozen_ranks
from .ranking import DIRECTIONS, RANK_MODES, RankFreezeConfig, freeze_ranks
from .training import DEFAULT_SEEDS, NeuralTrainConfig, train_from_fit_artifact


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m raw_rebuilt_neural",
        description="Four-process raw_rebuilt_v1 neural experiment boundary",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-fit", help="admission-only indT artifact creation")
    prepare.add_argument("--runtime", type=Path, required=True)
    prepare.add_argument("--output-parent", type=Path, required=True)

    train = sub.add_parser("train", help="train from an indT-only artifact; no runtime path accepted")
    train.add_argument("--fit", type=Path, required=True)
    train.add_argument("--output-parent", type=Path, required=True)
    train.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    train.add_argument("--epochs", type=int, default=NeuralTrainConfig.epochs)
    train.add_argument("--batch-size", type=int, default=NeuralTrainConfig.batch_size)
    train.add_argument("--learning-rate", type=float, default=NeuralTrainConfig.learning_rate)
    train.add_argument("--weight-decay", type=float, default=NeuralTrainConfig.weight_decay)
    train.add_argument("--device", default="auto")

    train_three = sub.add_parser("train-three-seeds", help="run the frozen three-seed schedule")
    train_three.add_argument("--fit", type=Path, required=True)
    train_three.add_argument("--output-parent", type=Path, required=True)
    train_three.add_argument("--epochs", type=int, default=NeuralTrainConfig.epochs)
    train_three.add_argument("--batch-size", type=int, default=NeuralTrainConfig.batch_size)
    train_three.add_argument("--device", default="auto")

    freeze = sub.add_parser("freeze-ranks", help="label-free rank worker")
    freeze.add_argument("--runtime", type=Path, required=True)
    freeze.add_argument("--checkpoint", type=Path, required=True)
    freeze.add_argument("--output-parent", type=Path, required=True)
    freeze.add_argument("--bits", type=_csv_ints, default=(16, 32, 64))
    freeze.add_argument("--directions", type=_csv_strings, default=DIRECTIONS)
    freeze.add_argument("--modes", type=_csv_strings, default=("hamming",))
    freeze.add_argument("--query-chunk-size", type=int, default=4)
    freeze.add_argument("--semantic-window", type=int, default=128)
    freeze.add_argument("--max-active-candidates", type=int, default=2048)
    freeze.add_argument("--device", default="auto")

    evaluate = sub.add_parser("evaluate", help="metric-only worker after rank freeze")
    evaluate.add_argument("--runtime", type=Path, required=True)
    evaluate.add_argument("--rank-root", type=Path, required=True)
    evaluate.add_argument("--output-parent", type=Path, required=True)
    evaluate.add_argument("--cutoffs", type=_csv_ints, default=(50, 100, 1000))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-fit":
        path = prepare_fit_artifact(args.runtime, args.output_parent)
        print(json.dumps({"fit_artifact": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "train":
        config = NeuralTrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        path = train_from_fit_artifact(
            args.fit,
            args.output_parent,
            config=config,
            device=args.device,
        )
        print(json.dumps({"training_run": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "train-three-seeds":
        paths = []
        for seed in DEFAULT_SEEDS:
            config = NeuralTrainConfig(
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            paths.append(
                str(
                    train_from_fit_artifact(
                        args.fit,
                        args.output_parent,
                        config=config,
                        device=args.device,
                    )
                )
            )
        print(json.dumps({"training_runs": paths}, ensure_ascii=False))
        return 0
    if args.command == "freeze-ranks":
        config = RankFreezeConfig(
            bits=tuple(args.bits),
            directions=tuple(args.directions),
            modes=tuple(args.modes),
            query_chunk_size=args.query_chunk_size,
            semantic_window=args.semantic_window,
            max_active_candidates=args.max_active_candidates,
        )
        path = freeze_ranks(
            args.runtime,
            args.checkpoint,
            args.output_parent,
            config=config,
            device=args.device,
        )
        print(json.dumps({"rank_root": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        path = evaluate_frozen_ranks(
            args.runtime,
            args.rank_root,
            args.output_parent,
            cutoffs=tuple(args.cutoffs),
        )
        print(json.dumps({"evaluation_root": str(path)}, ensure_ascii=False))
        return 0
    raise AssertionError("unhandled command")


__all__ = ["build_parser", "main"]

