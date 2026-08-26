#!/usr/bin/env python3
"""Evaluate registered fixed-feature baselines with the sealed stream evaluator.

The training/encoding sweep writes one immutable checkpoint and one complete
code artifact per method, bit width, and seed.  This driver selects the last
completed event for every registered cell, verifies/imports that artifact,
freezes a one-width label-free rank plan, and runs the common two-process
streaming evaluator.  It never edits or deletes source checkpoints/codes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_streaming.baseline_import import import_baseline_code_artifact
from raw_rebuilt_streaming.integrity import load_json, sha256_file, sha256_json
from raw_rebuilt_streaming.orchestrator import run_streaming_evaluation
from raw_rebuilt_streaming.plan import StreamingPlanConfig, freeze_rank_plan


METHODS = ("ucch-f", "dcmh-f-seminit", "cirh-f")
BITS = (16, 32, 64)
DATASETS = ("mirflickr", "nuswide", "mscoco")
DEFAULT_SEEDS = (20260822, 20260823, 20260824)
CELL_PATTERN = re.compile(
    r"^(mirflickr|nuswide|mscoco)_(ucch-f|dcmh-f-seminit|cirh-f)_b(16|32|64)_s(\d+)$"
)
DRIVER_SCHEMA = "raw_rebuilt_baseline_streaming_driver_event_v1"


class BaselineSweepError(RuntimeError):
    """Registered input or evaluation-resume evidence is inconsistent."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, body: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = {
        "schema": DRIVER_SCHEMA,
        **body,
        "utc": _utc(),
    }
    event = {**payload, "event_sha256": sha256_json(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise BaselineSweepError(
                    f"invalid JSON event at {path}:{line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise BaselineSweepError("sweep event must be one JSON object")
            result.append(value)
    return result


def _registered_inputs(
    event_log: Path,
    *,
    dataset: str,
    seeds: Sequence[int],
) -> Mapping[str, Mapping[str, Any]]:
    requested = {
        f"{dataset}_{method}_b{bits}_s{seed}"
        for seed in seeds
        for bits in BITS
        for method in METHODS
    }
    completed: dict[str, Mapping[str, Any]] = {}
    for event in _load_jsonl(event_log):
        cell = str(event.get("cell", ""))
        match = CELL_PATTERN.fullmatch(cell)
        if event.get("event") != "cell_complete" or match is None or cell not in requested:
            continue
        checkpoint = Path(str(event.get("checkpoint", "")))
        codes = Path(str(event.get("codes", "")))
        if not checkpoint.is_absolute() or not codes.is_absolute():
            raise BaselineSweepError(f"registered paths are not absolute for {cell}")
        completed[cell] = {
            "cell": cell,
            "dataset": match.group(1),
            "method": match.group(2),
            "bits": int(match.group(3)),
            "seed": int(match.group(4)),
            "checkpoint": checkpoint,
            "codes": codes,
            "source_event": event,
        }
    missing = sorted(requested - set(completed))
    if missing:
        raise BaselineSweepError(f"registered sweep is missing cells: {missing}")
    for cell, item in completed.items():
        for field in ("checkpoint", "codes"):
            path = Path(item[field]).resolve(strict=True)
            if path.is_symlink() or not path.is_dir():
                raise BaselineSweepError(f"{cell} {field} is not a regular directory")
            item[field] = path
    return completed


def _completed_driver_events(path: Path) -> Mapping[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for event in _load_jsonl(path):
        body = {key: event[key] for key in event if key != "event_sha256"}
        if (
            event.get("schema") != DRIVER_SCHEMA
            or event.get("event_sha256") != sha256_json(body)
        ):
            raise BaselineSweepError("driver event hash changed")
        if event.get("event") == "cell_complete":
            result[str(event.get("cell"))] = event
    return result


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise BaselineSweepError("cannot locate filesystem for output root")
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _require_disk(path: Path, minimum_free_bytes: int, *, cell: str) -> int:
    free = _free_bytes(path)
    if free < minimum_free_bytes:
        raise BaselineSweepError(
            f"free-space floor reached before {cell}: {free} < {minimum_free_bytes}"
        )
    return free


def _source_snapshot(item: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoint = Path(item["checkpoint"])
    codes = Path(item["codes"])
    checkpoint_manifest = checkpoint / "manifest.json"
    code_manifest = codes / "manifest.json"
    for path in (checkpoint_manifest, code_manifest):
        if path.is_symlink() or not path.is_file():
            raise BaselineSweepError(f"missing source manifest {path}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "codes": str(codes),
        "code_manifest_sha256": sha256_file(code_manifest),
    }


def _verify_completed_event(
    event: Mapping[str, Any],
    *,
    current_source: Mapping[str, Any],
) -> None:
    if event.get("source") != current_source:
        raise BaselineSweepError("completed cell source manifests changed")
    completion = Path(str(event.get("evaluation_complete", "")))
    if (
        not completion.is_absolute()
        or completion.is_symlink()
        or not completion.is_file()
        or completion.stat().st_size != int(event.get("evaluation_complete_size", -1))
        or sha256_file(completion) != event.get("evaluation_complete_file_sha256")
    ):
        raise BaselineSweepError("completed evaluation receipt changed")
    value = load_json(completion)
    body = {key: value[key] for key in value if key != "complete_sha256"}
    if value.get("complete_sha256") != sha256_json(body):
        raise BaselineSweepError("completed evaluation content hash changed")


def run_sweep(
    *,
    dataset: str,
    source_events: Path,
    runtime_root: Path,
    output_root: Path,
    seeds: Sequence[int],
    minimum_free_bytes: int,
    query_chunk_size: int,
    rank_device: str,
    max_cells: int | None,
) -> Mapping[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if minimum_free_bytes < 0 or query_chunk_size < 1:
        raise ValueError("disk floor must be nonnegative and chunk size positive")
    if rank_device not in {"cpu", "cuda"}:
        raise ValueError("rank-device must be cpu or cuda")
    source_events = source_events.expanduser().resolve(strict=True)
    runtime_root = runtime_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    event_path = output_root / "evaluation_events.jsonl"
    inputs = _registered_inputs(source_events, dataset=dataset, seeds=seeds)
    completed = _completed_driver_events(event_path)

    ordered = [
        f"{dataset}_{method}_b{bits}_s{seed}"
        for seed in seeds
        for bits in BITS
        for method in METHODS
    ]
    processed = 0
    skipped = 0
    for cell in ordered:
        item = inputs[cell]
        source = _source_snapshot(item)
        if cell in completed:
            _verify_completed_event(completed[cell], current_source=source)
            skipped += 1
            continue
        if max_cells is not None and processed >= max_cells:
            break
        free_before = _require_disk(output_root, minimum_free_bytes, cell=cell)
        _append_event(
            event_path,
            {
                "event": "cell_started",
                "cell": cell,
                "dataset": dataset,
                "method": item["method"],
                "bits": item["bits"],
                "seed": item["seed"],
                "source": source,
                "free_bytes": free_before,
            },
        )
        try:
            code_state = import_baseline_code_artifact(
                Path(item["codes"]),
                Path(item["checkpoint"]),
                output_root / "code_states",
            )
            plan = freeze_rank_plan(
                code_state,
                output_root / "plans" / cell,
                config=StreamingPlanConfig(
                    bits=(int(item["bits"]),),
                    directions=("i2t", "t2i"),
                    query_chunk_size=query_chunk_size,
                    cutoffs=(50, 100, 1000),
                ),
            )
            evaluation = run_streaming_evaluation(
                runtime_root,
                code_state,
                plan,
                output_root / "spools" / cell,
                output_root / "metrics" / cell,
                rank_device=rank_device,
            )
            completion = evaluation / "evaluation_complete.json"
            completion_value = load_json(completion)
            free_after = _require_disk(output_root, minimum_free_bytes, cell=cell)
            _append_event(
                event_path,
                {
                    "event": "cell_complete",
                    "cell": cell,
                    "dataset": dataset,
                    "method": item["method"],
                    "bits": item["bits"],
                    "seed": item["seed"],
                    "source": source,
                    "code_state": str(code_state),
                    "plan": str(plan),
                    "evaluation": str(evaluation),
                    "evaluation_complete": str(completion),
                    "evaluation_complete_size": completion.stat().st_size,
                    "evaluation_complete_file_sha256": sha256_file(completion),
                    "evaluation_complete_sha256": completion_value[
                        "complete_sha256"
                    ],
                    "free_bytes": free_after,
                },
            )
        except BaseException as error:
            _append_event(
                event_path,
                {
                    "event": "cell_failed",
                    "cell": cell,
                    "dataset": dataset,
                    "method": item["method"],
                    "bits": item["bits"],
                    "seed": item["seed"],
                    "source": source,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "free_bytes": _free_bytes(output_root),
                },
            )
            raise
        processed += 1

    final_completed = _completed_driver_events(event_path)
    requested = set(ordered)
    return {
        "status": "COMPLETE" if requested <= set(final_completed) else "IN_PROGRESS",
        "dataset": dataset,
        "requested_cells": len(ordered),
        "verified_complete_cells": len(requested & set(final_completed)),
        "processed_cells": processed,
        "skipped_cells": skipped,
        "event_log": str(event_path),
        "free_bytes": _free_bytes(output_root),
    }


def _csv_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("seeds must be unique and nonempty")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="mirflickr")
    parser.add_argument("--source-events", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_csv_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--minimum-free-bytes", type=int, default=1 << 30)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--rank-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-cells", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_sweep(
        dataset=args.dataset,
        source_events=args.source_events,
        runtime_root=args.runtime,
        output_root=args.output_root,
        seeds=args.seeds,
        minimum_free_bytes=args.minimum_free_bytes,
        query_chunk_size=args.query_chunk_size,
        rank_device=args.rank_device,
        max_cells=args.max_cells,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
