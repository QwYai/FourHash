#!/usr/bin/env python3
"""Deep-verify and aggregate fixed-feature baseline streaming evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_streaming.integrity import (
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)
from raw_rebuilt_streaming.orchestrator import _verify_evaluation_after_workers
from raw_rebuilt_streaming.plan import open_rank_plan
from tools.run_baseline_streaming_eval_sweep import DATASETS, DRIVER_SCHEMA, METHODS


AGGREGATE_SCHEMA = "raw_rebuilt_baseline_streaming_aggregate_v1"
CELL_PATTERN = re.compile(
    r"^(mirflickr|nuswide|mscoco)_(ucch-f|dcmh-f-seminit|cirh-f)_b(16|32|64)_s(\d+)$"
)
DIRECTIONS = ("i2t", "t2i")
BITS = (16, 32, 64)


class BaselineAggregateError(RuntimeError):
    """A driver event, source manifest, or result receipt failed verification."""


def _events(path: Path) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise BaselineAggregateError(
                    f"invalid event at {path}:{line_number}"
                ) from error
            if not isinstance(event, Mapping):
                raise BaselineAggregateError("driver event is not an object")
            body = {key: event[key] for key in event if key != "event_sha256"}
            if (
                event.get("schema") != DRIVER_SCHEMA
                or event.get("event_sha256") != sha256_json(body)
            ):
                raise BaselineAggregateError("driver event digest changed")
            result.append(event)
    return result


def _completion_events(
    path: Path,
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
    for event in _events(path):
        cell = str(event.get("cell", ""))
        if event.get("event") == "cell_complete" and cell in requested:
            completed[cell] = event
    missing = sorted(requested - set(completed))
    if missing:
        raise BaselineAggregateError(f"evaluation grid is incomplete: {missing}")
    return completed


def _verify_source(event: Mapping[str, Any]) -> None:
    source = event.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "checkpoint",
        "checkpoint_manifest_sha256",
        "codes",
        "code_manifest_sha256",
    }:
        raise BaselineAggregateError("source binding schema changed")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_manifest_sha256"),
        ("codes", "code_manifest_sha256"),
    ):
        root = Path(str(source[path_key])).resolve(strict=True)
        manifest = root / "manifest.json"
        if root.is_symlink() or not root.is_dir() or manifest.is_symlink() or not manifest.is_file():
            raise BaselineAggregateError("source artifact is missing or linked")
        if sha256_file(manifest) != source[hash_key]:
            raise BaselineAggregateError("source manifest changed after evaluation")


def _result_rows(
    event: Mapping[str, Any],
    *,
    dataset: str,
    output_root: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    cell = str(event["cell"])
    match = CELL_PATTERN.fullmatch(cell)
    if match is None:
        raise BaselineAggregateError(f"invalid completed cell name {cell}")
    if match.group(1) != dataset:
        raise BaselineAggregateError("completed cell dataset differs from request")
    method = match.group(2)
    bits = int(match.group(3))
    seed = int(match.group(4))
    if method != event.get("method") or bits != event.get("bits") or seed != event.get("seed"):
        raise BaselineAggregateError("completed event identity fields disagree")
    _verify_source(event)

    code_state = Path(str(event.get("code_state", ""))).resolve(strict=True)
    plan_root = Path(str(event.get("plan", ""))).resolve(strict=True)
    evaluation = Path(str(event.get("evaluation", ""))).resolve(strict=True)
    spool = (output_root / "spools" / cell).resolve(strict=True)
    frozen = open_rank_plan(plan_root, code_state)
    _verify_evaluation_after_workers(evaluation, spool, frozen.manifest)

    completion_path = evaluation / "evaluation_complete.json"
    if (
        str(completion_path) != event.get("evaluation_complete")
        or completion_path.stat().st_size != event.get("evaluation_complete_size")
        or sha256_file(completion_path) != event.get("evaluation_complete_file_sha256")
    ):
        raise BaselineAggregateError("evaluation completion differs from driver event")
    completion = load_json(completion_path)
    if completion.get("complete_sha256") != event.get("evaluation_complete_sha256"):
        raise BaselineAggregateError("completion content digest differs from driver event")

    rows: list[Mapping[str, Any]] = []
    for descriptor in completion["results"]:
        metric_path = evaluation / str(descriptor["path"])
        metric = load_json(metric_path)
        direction = str(metric["direction"])
        if direction not in DIRECTIONS or int(metric["bits"]) != bits:
            raise BaselineAggregateError("metric result identity differs from cell")
        summary = metric["summary"]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "bits": bits,
                "seed": seed,
                "direction": direction,
                "map": float(summary["map_expected_ties"]),
                "binary_ndcg_at_50": float(
                    summary["binary_ndcg_at_50_expected_ties"]
                ),
                "j_ndcg_at_50": float(summary["j_ndcg_at_50_expected_ties"]),
                "precision_at_50": float(
                    summary["precision_at_50_expected_ties"]
                ),
                "recall_at_50": float(summary["recall_at_50_expected_ties"]),
                "queries": int(summary["queries"]),
                "queries_with_relevant": int(summary["queries_with_relevant"]),
                "metric_result_sha256": metric["metric_result_sha256"],
                "metric_file_sha256": sha256_file(metric_path),
                "final_ack_chain_sha256": metric["final_ack_chain_sha256"],
                "final_private_metric_chain_sha256": metric[
                    "final_private_metric_chain_sha256"
                ],
            }
        )
    if {str(row["direction"]) for row in rows} != set(DIRECTIONS):
        raise BaselineAggregateError("completed cell does not contain both directions")
    source = {
        "cell": cell,
        "method": method,
        "bits": bits,
        "seed": seed,
        "source": event["source"],
        "code_state_manifest_sha256": frozen.manifest["binding"][
            "code_state_manifest_sha256"
        ],
        "rank_plan_sha256": frozen.manifest["rank_plan_sha256"],
        "evaluation_complete_sha256": completion["complete_sha256"],
        "evaluation_complete_file_sha256": sha256_file(completion_path),
        "receipt_bijection_deep_verified": True,
    }
    return rows, source


def aggregate(
    *,
    dataset: str,
    event_log: Path,
    output_root: Path,
    seeds: Sequence[int],
    json_output: Path,
    csv_output: Path,
) -> Mapping[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    for output in (json_output, csv_output):
        if output.exists():
            raise BaselineAggregateError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    event_log = event_log.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=True)
    completed = _completion_events(event_log, dataset=dataset, seeds=seeds)
    rows: list[Mapping[str, Any]] = []
    sources: list[Mapping[str, Any]] = []
    ordered = [
        f"{dataset}_{method}_b{bits}_s{seed}"
        for seed in seeds
        for bits in BITS
        for method in METHODS
    ]
    for cell in ordered:
        cell_rows, source = _result_rows(
            completed[cell], dataset=dataset, output_root=output_root
        )
        rows.extend(cell_rows)
        sources.append(source)
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row["bits"]),
            METHODS.index(str(row["method"])),
            DIRECTIONS.index(str(row["direction"])),
        )
    )
    expected_rows = len(seeds) * len(BITS) * len(METHODS) * len(DIRECTIONS)
    if len(rows) != expected_rows:
        raise BaselineAggregateError("aggregate row count changed")
    body: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "status": "VERIFIED",
        "dataset": dataset,
        "seeds": list(seeds),
        "bits": list(BITS),
        "methods": list(METHODS),
        "directions": list(DIRECTIONS),
        "event_log": {
            "path": str(event_log),
            "size": event_log.stat().st_size,
            "sha256": sha256_file(event_log),
        },
        "deep_verified_cells": len(sources),
        "metric_rows": len(rows),
        "sources": sources,
        "rows": rows,
    }
    result = {**body, "aggregate_sha256": sha256_json(body)}
    atomic_write_json(json_output, result)
    with csv_output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _csv_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique and nonempty")
    return seeds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="mirflickr")
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_csv_seeds, default=(20260822,))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = aggregate(
        dataset=args.dataset,
        event_log=args.event_log,
        output_root=args.output_root,
        seeds=args.seeds,
        json_output=args.json_output,
        csv_output=args.csv_output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
