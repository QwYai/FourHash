from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from .adapters import make_adapter
from .core import (
    TraceContractError,
    atomic_write_json,
    ensure_output_safe,
)
from .extraction import (
    ExtractionConfig,
    OpenAIClipViTB32Encoder,
    extract_trace_bundle,
    preflight_adapter,
    verify_trace_bundle,
)


def _default_output_root() -> Path:
    configured = os.environ.get("KBS_VISUALIZATION_TRACE_OUTPUT")
    if configured:
        return Path(configured)
    return Path.cwd() / "visualization_trace_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization_trace",
        description=(
            "Extract OralData-only CLIP ViT-B/32 visualization features with "
            "canonical identities, content-hash splits, and resumable provenance."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="build a sealed trace bundle")
    extract.add_argument(
        "--dataset", required=True, choices=("mirflickr", "nuswide", "mscoco")
    )
    extract.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Data directory containing OralData; ProcessData is never an identity source",
    )
    extract.add_argument(
        "--output-root", type=Path, default=_default_output_root()
    )
    extract.add_argument("--run-name", default="clip_vit_b32_v1")
    extract.add_argument("--device", default="cuda")
    extract.add_argument(
        "--clip-cache",
        type=Path,
        default=None,
        help="checkpoint cache; its actual ViT-B/32 SHA-256 is sealed into the contract",
    )
    extract.add_argument("--batch-size", type=int, default=64)
    extract.add_argument("--text-batch-size", type=int, default=512)
    extract.add_argument("--shard-rows", type=int, default=1024)
    extract.add_argument("--no-resume", action="store_true")

    inspect = subparsers.add_parser(
        "inspect", help="fully verify the sealed bundle and show its contract"
    )
    inspect.add_argument("--run-dir", required=True, type=Path)

    preflight = subparsers.add_parser(
        "preflight",
        help="validate every canonical raw row and show a limited sample without loading CLIP",
    )
    preflight.add_argument(
        "--dataset", required=True, choices=("mirflickr", "nuswide", "mscoco")
    )
    preflight.add_argument("--data-root", required=True, type=Path)
    preflight.add_argument(
        "--limit",
        type=int,
        default=12,
        help="number of evenly spaced manifest examples; all rows are still validated",
    )
    preflight.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional report path outside OralData and ProcessData",
    )
    return parser


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            payload = verify_trace_bundle(args.run_dir)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.command == "preflight":
            adapter = make_adapter(args.dataset, args.data_root)
            payload = preflight_adapter(adapter, args.limit)
            if args.output_json is not None:
                safe_parent = ensure_output_safe(
                    args.output_json.parent,
                    [args.data_root / "OralData", args.data_root / "ProcessData"],
                )
                destination = safe_parent / args.output_json.name
                atomic_write_json(destination, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        adapter = make_adapter(args.dataset, args.data_root)
        encoder = OpenAIClipViTB32Encoder(args.device, args.clip_cache)
        config = ExtractionConfig(
            output_root=args.output_root,
            run_name=args.run_name,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            shard_rows=args.shard_rows,
            resume=not args.no_resume,
            hash_images=True,
            hash_source_artifacts=True,
        )
        complete = extract_trace_bundle(adapter, encoder, config)
        print(json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (TraceContractError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2
