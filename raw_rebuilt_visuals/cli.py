"""Command line entry point for trace-backed qualitative evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evidence import EvidenceError, materialize_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m raw_rebuilt_visuals")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--process-data-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize_evidence(
            bundle=args.bundle,
            runtime=args.runtime,
            selection_manifest=args.selection,
            output=args.output,
            process_data_root=args.process_data_root,
        )
    except EvidenceError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0
