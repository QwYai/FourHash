"""Command-line entry points for sealed fixed-feature baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from raw_rebuilt_neural import open_fit_artifact
from raw_rebuilt_runtime import load_label_free_rank_inputs

from .adapters import DEFAULT_SEEDS, METHODS, BaselineRunConfig
from .checkpoint import load_checkpoint, train_baseline
from .contract import label_free_inputs_from_runtime
from .encoding import encode_label_free, write_code_artifact


def _overrides(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("overrides must be a JSON object") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("overrides must be a JSON object")
    return {str(key): item for key, item in parsed.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m raw_rebuilt_baselines",
        description=(
            "Train and encode controlled UCCH-F/DCMH-F/CIRH-F adaptations "
            "from sealed raw-rebuilt CLIP512 artifacts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--fit-artifact", type=Path, required=True)
    train.add_argument("--runtime", type=Path, required=True)
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--bits", type=int, choices=(16, 32, 64), required=True)
    train.add_argument("--seed", type=int, choices=DEFAULT_SEEDS, required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--overrides-json", type=_overrides, default={})
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--quiet", action="store_true")

    encode = commands.add_parser("encode")
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--runtime", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--device", default=None)
    encode.add_argument("--batch-size", type=int, default=1024)
    return parser


def _train(args: argparse.Namespace) -> Path:
    fit = open_fit_artifact(args.fit_artifact)
    rank = load_label_free_rank_inputs(args.runtime)
    try:
        label_free = label_free_inputs_from_runtime(fit.dataset, rank)
        config = BaselineRunConfig(
            method=args.method,
            bits=args.bits,
            seed=args.seed,
            device=args.device,
            overrides=args.overrides_json,
        )
        return train_baseline(
            fit,
            label_free,
            config,
            args.output,
            verbose=not args.quiet,
        )
    finally:
        rank.close()
        fit.close()


def _encode(args: argparse.Namespace) -> Path:
    checkpoint = load_checkpoint(args.checkpoint)
    rank = load_label_free_rank_inputs(args.runtime)
    try:
        label_free = label_free_inputs_from_runtime(
            checkpoint.dataset_binding.dataset, rank
        )
        codes = encode_label_free(
            checkpoint,
            label_free,
            batch_size=args.batch_size,
            device=args.device,
        )
        return write_code_artifact(codes, args.output)
    finally:
        rank.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = _train(args)
    elif args.command == "encode":
        result = _encode(args)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps({"output": str(result)}, sort_keys=True))
    return 0


__all__ = ["main"]
