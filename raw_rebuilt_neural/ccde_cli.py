"""Command-line boundaries for frozen CCDE formal experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .ccde_detail_bits import select_detail_bits_from_fit
from .ccde_ranking import CCDERankFreezeConfig, freeze_ccde_ranks
from .ccde_training import train_detail_from_fit_artifact
from .metrics import evaluate_frozen_ranks
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
        prog="python -m raw_rebuilt_neural.ccde_cli",
        description="Frozen full-indT CCDE training and formal evaluation boundaries",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_primary = sub.add_parser(
        "train-primary", help="train the exact compact primary encoder from full indT"
    )
    train_primary.add_argument("--fit", type=Path, required=True)
    train_primary.add_argument("--output-parent", type=Path, required=True)
    train_primary.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    train_primary.add_argument("--device", default="auto")
    train_primary.add_argument("--max-epochs-this-call", type=int)
    train_primary.add_argument(
        "--checkpoint-every", type=int, default=NeuralTrainConfig().epochs
    )

    train_detail = sub.add_parser(
        "train-detail", help="train the frozen independent-BN detail encoder"
    )
    train_detail.add_argument("--fit", type=Path, required=True)
    train_detail.add_argument("--freeze", type=Path, required=True)
    train_detail.add_argument("--output-parent", type=Path, required=True)
    train_detail.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    train_detail.add_argument("--device", default="auto")
    train_detail.add_argument("--max-epochs-this-call", type=int)
    train_detail.add_argument(
        "--checkpoint-every", type=int, default=NeuralTrainConfig().epochs
    )

    select = sub.add_parser(
        "select-detail-bits", help="select the frozen 16-bit prefix using full indT only"
    )
    select.add_argument("--fit", type=Path, required=True)
    select.add_argument("--detail-checkpoint", type=Path, required=True)
    select.add_argument("--freeze", type=Path, required=True)
    select.add_argument("--output-parent", type=Path, required=True)
    select.add_argument("--device", default="auto")

    freeze = sub.add_parser(
        "freeze-ranks", help="label-free lexicographic CCDE rank worker"
    )
    freeze.add_argument("--runtime", type=Path, required=True)
    freeze.add_argument("--primary-checkpoint", type=Path, required=True)
    freeze.add_argument("--detail-checkpoint", type=Path, required=True)
    freeze.add_argument("--detail-bits", type=Path, required=True)
    freeze.add_argument("--freeze", type=Path, required=True)
    freeze.add_argument("--output-parent", type=Path, required=True)
    freeze.add_argument("--bits", type=_csv_ints, default=(16, 32, 64))
    freeze.add_argument("--directions", type=_csv_strings, default=("i2t", "t2i"))
    freeze.add_argument("--query-chunk-size", type=int, default=4)
    freeze.add_argument("--device", default="auto")

    evaluate = sub.add_parser(
        "evaluate", help="open formal labels only after the CCDE rank freeze"
    )
    evaluate.add_argument("--runtime", type=Path, required=True)
    evaluate.add_argument("--rank-root", type=Path, required=True)
    evaluate.add_argument("--output-parent", type=Path, required=True)
    evaluate.add_argument("--cutoffs", type=_csv_ints, default=(50, 100, 1000))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-primary":
        config = NeuralTrainConfig(
            seed=args.seed, checkpoint_every=args.checkpoint_every
        )
        path = train_from_fit_artifact(
            args.fit,
            args.output_parent,
            config=config,
            device=args.device,
            max_epochs_this_call=args.max_epochs_this_call,
        )
        print(json.dumps({"primary_training_run": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "train-detail":
        config = NeuralTrainConfig(
            seed=args.seed, checkpoint_every=args.checkpoint_every
        )
        path = train_detail_from_fit_artifact(
            args.fit,
            args.freeze,
            args.output_parent,
            config=config,
            device=args.device,
            max_epochs_this_call=args.max_epochs_this_call,
        )
        print(json.dumps({"detail_training_run": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "select-detail-bits":
        path = select_detail_bits_from_fit(
            args.fit,
            args.detail_checkpoint,
            args.freeze,
            args.output_parent,
            device=args.device,
        )
        print(json.dumps({"detail_bit_artifact": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "freeze-ranks":
        rank_config = CCDERankFreezeConfig(
            bits=tuple(args.bits),
            directions=tuple(args.directions),
            query_chunk_size=args.query_chunk_size,
        )
        path = freeze_ccde_ranks(
            args.runtime,
            args.primary_checkpoint,
            args.detail_checkpoint,
            args.detail_bits,
            args.freeze,
            args.output_parent,
            config=rank_config,
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
