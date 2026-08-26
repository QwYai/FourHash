#!/usr/bin/env python3
"""Deep-verify and aggregate three ShellGuard semantic-bridge evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)


SCHEMA = "semantic_bridge_formal_seed20260822_verified_aggregate_v1"
EVALUATION_SCHEMA = "shellguard_semantic_bridge_evaluation_v1"
RESULT_SCHEMA = "shellguard_semantic_bridge_metric_result_v1"
DATASETS = ("mirflickr", "nuswide", "mscoco")
LABELS = {
    "mirflickr": "MIRFlickr-25K",
    "nuswide": "NUS-WIDE-TC21",
    "mscoco": "MS COCO",
}
BITS = (16, 32, 64)
DIRECTIONS = ("i2t", "t2i")


class SemanticBridgeAuditError(RuntimeError):
    """A formal result or receipt failed verification."""


def _verified_complete(root: Path, dataset: str) -> Mapping[str, Any]:
    path = root / "evaluation_complete.json"
    value = load_json(path)
    body = {key: value[key] for key in value if key != "complete_sha256"}
    if (
        value.get("schema") != EVALUATION_SCHEMA
        or value.get("status") != "COMPLETE"
        or value.get("dataset") != dataset
        or value.get("complete_sha256") != sha256_json(body)
        or value.get("primary_shell_order_is_invariant") is not True
        or value.get("formal_gate_or_fallback_used") is not False
    ):
        raise SemanticBridgeAuditError(f"evaluation summary failed: {path}")
    return value


def _verified_result(
    evaluation_root: Path,
    descriptor: Mapping[str, Any],
    *,
    dataset: str,
) -> Mapping[str, Any]:
    path = evaluation_root / str(descriptor.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(descriptor.get("size", -1))
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise SemanticBridgeAuditError(f"metric result bytes failed: {path}")
    value = load_json(path)
    body = {key: value[key] for key in value if key != "metric_result_sha256"}
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("status") != "COMPLETE"
        or value.get("dataset") != dataset
        or value.get("metric_result_sha256") != sha256_json(body)
        or value.get("metric_result_sha256")
        != descriptor.get("metric_result_sha256")
        or value.get("primary_shell_order_is_invariant") is not True
    ):
        raise SemanticBridgeAuditError(f"metric result content failed: {path}")
    chain = "0" * 64
    expected_start = 0
    receipt_files = 0
    receipt_bytes = 0
    for receipt_descriptor in value.get("per_query_receipts", []):
        receipt_path = evaluation_root / str(receipt_descriptor.get("path", ""))
        if (
            not receipt_path.is_file()
            or receipt_path.stat().st_size
            != int(receipt_descriptor.get("size", -1))
            or sha256_file(receipt_path) != receipt_descriptor.get("sha256")
        ):
            raise SemanticBridgeAuditError(
                f"partial receipt bytes failed: {receipt_path}"
            )
        receipt = load_json(receipt_path)
        receipt_body = {
            key: receipt[key]
            for key in receipt
            if key not in {"receipt_sha256", "chain_sha256"}
        }
        receipt_sha = sha256_json(receipt_body)
        next_chain = sha256_json(
            {"previous_chain_sha256": chain, "receipt_sha256": receipt_sha}
        )
        if (
            receipt.get("receipt_sha256") != receipt_sha
            or receipt.get("chain_sha256") != next_chain
            or receipt_descriptor.get("receipt_sha256") != receipt_sha
            or int(receipt.get("start", -1)) != expected_start
            or int(receipt.get("end", -1)) <= expected_start
            or receipt.get("primary_shell_invariance_checked") is not True
        ):
            raise SemanticBridgeAuditError(
                f"partial receipt chain failed: {receipt_path}"
            )
        primary = receipt.get("primary_records")
        bridge = receipt.get("bridge_records")
        start = int(receipt["start"])
        end = int(receipt["end"])
        expected_positions = list(range(start, end))
        if (
            not isinstance(primary, list)
            or not isinstance(bridge, list)
            or [row.get("query_position") for row in primary]
            != expected_positions
            or [row.get("query_position") for row in bridge]
            != expected_positions
        ):
            raise SemanticBridgeAuditError(
                f"partial receipt query coverage failed: {receipt_path}"
            )
        expected_start = end
        chain = next_chain
        receipt_files += 1
        receipt_bytes += receipt_path.stat().st_size
    if chain != value.get("final_receipt_chain_sha256"):
        raise SemanticBridgeAuditError(f"final receipt chain failed: {path}")
    return {
        **value,
        "_path": path,
        "_receipt_files": receipt_files,
        "_receipt_bytes": receipt_bytes,
        "_queries": expected_start,
    }


def audit(
    dataset_roots: Mapping[str, Path],
    *,
    json_output: Path,
    csv_output: Path,
) -> Mapping[str, Any]:
    if set(dataset_roots) != set(DATASETS):
        raise SemanticBridgeAuditError("all three dataset roots are required")
    if json_output.exists() or csv_output.exists():
        raise SemanticBridgeAuditError("refusing to overwrite an audit output")
    rows = []
    datasets = []
    total_files = 0
    total_bytes = 0
    primary_deltas = []
    graded_deltas = []
    for dataset in DATASETS:
        root = dataset_roots[dataset].expanduser().resolve(strict=True)
        complete_path = root / "evaluation_complete.json"
        complete = _verified_complete(root, dataset)
        descriptors = complete.get("results")
        if not isinstance(descriptors, list) or len(descriptors) != 6:
            raise SemanticBridgeAuditError(f"incomplete result grid: {dataset}")
        result_index = {}
        dataset_files = 1
        dataset_bytes = complete_path.stat().st_size
        for descriptor in descriptors:
            value = _verified_result(root, descriptor, dataset=dataset)
            key = (int(value["bits"]), str(value["direction"]))
            if key in result_index:
                raise SemanticBridgeAuditError(f"duplicate result cell: {dataset} {key}")
            result_index[key] = value
            dataset_files += 1 + int(value["_receipt_files"])
            dataset_bytes += value["_path"].stat().st_size + int(
                value["_receipt_bytes"]
            )
        expected = {(bits, direction) for bits in BITS for direction in DIRECTIONS}
        if set(result_index) != expected:
            raise SemanticBridgeAuditError(f"result grid differs: {dataset}")
        for bits, direction in sorted(expected, key=lambda value: (value[0], value[1])):
            value = result_index[(bits, direction)]
            primary = value["summaries"]["primary_hamming"]
            shellguard = value["summaries"]["shellguard_semantic_bridge"]
            delta = value["shellguard_minus_primary"]
            row = {
                "dataset": dataset,
                "dataset_label": LABELS[dataset],
                "bits": bits,
                "detail_bits": 16,
                "direction": direction,
                "queries": int(value["_queries"]),
                "selected_threshold": float(value["selected_threshold"]),
                "primary_map": float(primary["map_expected_ties"]),
                "shellguard_map": float(shellguard["map_expected_ties"]),
                "delta_map": float(delta["map_expected_ties"]),
                "primary_ndcg50": float(
                    primary["binary_ndcg_at_50_expected_ties"]
                ),
                "shellguard_ndcg50": float(
                    shellguard["binary_ndcg_at_50_expected_ties"]
                ),
                "delta_ndcg50": float(
                    delta["binary_ndcg_at_50_expected_ties"]
                ),
                "primary_jndcg50": float(primary["j_ndcg_at_50_expected_ties"]),
                "shellguard_jndcg50": float(
                    shellguard["j_ndcg_at_50_expected_ties"]
                ),
                "delta_jndcg50": float(delta["j_ndcg_at_50_expected_ties"]),
                "metric_result_sha256": value["metric_result_sha256"],
            }
            rows.append(row)
            primary_deltas.extend((row["delta_map"], row["delta_ndcg50"]))
            graded_deltas.append(row["delta_jndcg50"])
        datasets.append(
            {
                "dataset": dataset,
                "evaluation_root": str(root),
                "rank_plan_sha256": complete["rank_plan_sha256"],
                "complete_sha256": complete["complete_sha256"],
                "selected_threshold": float(complete["selected_threshold"]),
                "files": dataset_files,
                "bytes": dataset_bytes,
                "query_records_across_six_cells": sum(
                    int(result_index[key]["_queries"]) for key in expected
                ),
                "minimum_primary_delta": float(complete["minimum_primary_delta"]),
                "mean_primary_delta": float(complete["mean_primary_delta"]),
                "minimum_graded_delta": float(complete["minimum_graded_delta"]),
                "mean_graded_delta": float(complete["mean_graded_delta"]),
            }
        )
        total_files += dataset_files
        total_bytes += dataset_bytes
    if any(value <= 0.0 for value in primary_deltas + graded_deltas):
        raise SemanticBridgeAuditError("formal aggregate contains a nonpositive cell")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "VERIFIED",
        "seed": 20260822,
        "primary_shell_order_is_invariant": True,
        "formal_labels_used_for_threshold_selection": False,
        "threshold_selected_from_indT_only": True,
        "posterior_database_cache_retained": False,
        "detail_bits": 16,
        "datasets": datasets,
        "rows": rows,
        "primary_cells": len(primary_deltas),
        "graded_cells": len(graded_deltas),
        "primary_improvements": sum(value > 0.0 for value in primary_deltas),
        "graded_improvements": sum(value > 0.0 for value in graded_deltas),
        "minimum_primary_delta": min(primary_deltas),
        "mean_primary_delta": sum(primary_deltas) / len(primary_deltas),
        "minimum_graded_delta": min(graded_deltas),
        "mean_graded_delta": sum(graded_deltas) / len(graded_deltas),
        "receipt_linked_files_verified": total_files,
        "receipt_linked_bytes_verified": total_bytes,
        "main_table_decimals": 3,
    }
    result = {**body, "aggregate_sha256": sha256_json(body)}
    json_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_output, result)
    with csv_output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _dataset_argument(value: str) -> tuple[str, Path]:
    dataset, separator, raw_path = value.partition("=")
    if not separator or dataset not in DATASETS or not raw_path:
        raise argparse.ArgumentTypeError("dataset must be DATASET=EVALUATION_ROOT")
    return dataset, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=_dataset_argument, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots: dict[str, Path] = {}
    for dataset, root in args.dataset:
        if dataset in roots:
            raise SemanticBridgeAuditError(f"duplicate dataset: {dataset}")
        roots[dataset] = root
    result = audit(
        roots,
        json_output=args.json_output,
        csv_output=args.csv_output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_improvements": result["primary_improvements"],
                "graded_improvements": result["graded_improvements"],
                "aggregate_sha256": result["aggregate_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
