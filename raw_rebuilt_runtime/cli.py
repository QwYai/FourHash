from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .contract import RuntimeBridgeError
from .materialize import materialize_runtime, verify_runtime_directory
from .validation import admit_source_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m raw_rebuilt_runtime",
        description=(
            "Double-validate and materialize a sealed raw_rebuilt_v1 trace bundle "
            "for CLIP512/V2/24-method runners."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    admit = subparsers.add_parser("admit", help="run both source validators without writing arrays")
    admit.add_argument("--bundle", required=True, type=Path)
    admit.add_argument("--process-data-root", type=Path, default=None)

    materialize = subparsers.add_parser("materialize", help="stream into a resumable NPY runtime")
    materialize.add_argument("--bundle", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    materialize.add_argument("--process-data-root", type=Path, default=None)
    materialize.add_argument(
        "--max-new-parts",
        type=int,
        default=None,
        help="optional checkpoint bound; omit to finish all remaining source shards",
    )

    verify = subparsers.add_parser("verify", help="rehash runtime and re-admit its source")
    verify.add_argument("--runtime", required=True, type=Path)
    verify.add_argument("--process-data-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "admit":
            value = admit_source_bundle(
                args.bundle, process_data_root=args.process_data_root
            )
            result = {
                "status": "PASS",
                "dataset": value.dataset,
                "rows": value.rows,
                "shards": value.shards,
                "source_seal_sha256": value.seal["source_seal_sha256"],
                "checks": value.provenance_report["checks"],
            }
        elif args.command == "materialize":
            result = materialize_runtime(
                args.bundle,
                args.output,
                process_data_root=args.process_data_root,
                max_new_parts=args.max_new_parts,
            )
        else:
            manifest = verify_runtime_directory(
                args.runtime, process_data_root=args.process_data_root
            )
            result = {
                "status": "PASS",
                "dataset": manifest["dataset"],
                "rows": manifest["rows"],
                "parts": len(manifest["receipts"]),
                "source_seal_sha256": manifest["source_seal_sha256"],
            }
    except (RuntimeBridgeError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
