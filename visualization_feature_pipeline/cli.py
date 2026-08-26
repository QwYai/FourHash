"""Command-line entry point for provenance inspection and validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .contract import ContractError, derive_extractor_id, derive_row_id
from .validator import validate_bundle


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization_feature_pipeline.cli",
        description="Validate independently extracted visualization features without running metrics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="fail-closed full-bundle validation")
    validate.add_argument("--bundle", type=Path, required=True)
    validate.add_argument(
        "--process-data-root",
        type=Path,
        required=False,
        help="optional forbidden-path check only; this path is never opened or used as authority",
    )
    validate.add_argument("--compact", action="store_true", help="emit one-line JSON")

    row_id = subparsers.add_parser("row-id", help="derive an immutable row ID from one row JSON")
    row_id.add_argument("--row-json", type=Path, required=True)

    extractor_id = subparsers.add_parser(
        "extractor-id", help="derive an extractor ID from one extractor JSON"
    )
    extractor_id.add_argument("--extractor-json", type=Path, required=True)

    schema = subparsers.add_parser("schema-path", help="print the bundled JSON Schema path")
    schema.set_defaults(schema_path=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_bundle(args.bundle, args.process_data_root)
            print(
                json.dumps(
                    report.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=None if args.compact else 2,
                )
            )
            return 0
        if args.command == "row-id":
            print(derive_row_id(_read_json_object(args.row_json)))
            return 0
        if args.command == "extractor-id":
            print(derive_extractor_id(_read_json_object(args.extractor_json)))
            return 0
        if args.command == "schema-path":
            print((Path(__file__).parent / "schema" / "bundle.schema.json").resolve())
            return 0
        parser.error("unknown command")
    except ContractError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
