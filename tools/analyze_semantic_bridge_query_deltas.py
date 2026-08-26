"""Deep-verified query-level audit for completed semantic-bridge evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import atomic_write_json, load_json, sha256_json
from tools.audit_semantic_bridge_formal import (
    BITS,
    DATASETS,
    DIRECTIONS,
    SemanticBridgeAuditError,
    _verified_complete,
    _verified_result,
)


QUERY_METRICS = {
    "map": "average_precision_expected_ties",
    "ndcg50": "binary_ndcg_at_50_expected_ties",
    "jndcg50": "j_ndcg_at_50_expected_ties",
}
SUMMARY_METRICS = {
    "map": "map_expected_ties",
    "ndcg50": "binary_ndcg_at_50_expected_ties",
    "jndcg50": "j_ndcg_at_50_expected_ties",
}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SemanticBridgeAuditError("cannot summarize an empty vector")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Mapping[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise SemanticBridgeAuditError("query-delta vector is empty or non-finite")
    return {
        "records": len(values),
        "mean": math.fsum(values) / len(values),
        "q10": _quantile(values, 0.10),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "q90": _quantile(values, 0.90),
        "positive_fraction": sum(value > 0.0 for value in values) / len(values),
        "nonnegative_fraction": sum(value >= 0.0 for value in values) / len(values),
    }


def _dataset_argument(value: str) -> tuple[str, Path]:
    dataset, separator, raw_path = value.partition("=")
    if not separator or dataset not in DATASETS or not raw_path:
        raise argparse.ArgumentTypeError("dataset must be DATASET=EVALUATION_ROOT")
    return dataset, Path(raw_path)


def analyze(
    roots: Mapping[str, Path], *, json_output: Path, csv_output: Path
) -> Mapping[str, Any]:
    if set(roots) != set(DATASETS):
        raise SemanticBridgeAuditError("all three dataset roots are required")
    if json_output.exists() or csv_output.exists():
        raise SemanticBridgeAuditError("refusing to overwrite query-level outputs")

    cells: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for dataset in DATASETS:
        root = roots[dataset].expanduser().resolve(strict=True)
        complete = _verified_complete(root, dataset)
        pooled = {short: [] for short in QUERY_METRICS}
        descriptors = complete.get("results")
        if not isinstance(descriptors, list) or len(descriptors) != 6:
            raise SemanticBridgeAuditError(f"incomplete result grid: {dataset}")
        for descriptor in sorted(
            descriptors,
            key=lambda item: (
                DIRECTIONS.index(str(item["direction"])),
                BITS.index(int(item["bits"])),
            ),
        ):
            result = _verified_result(root, descriptor, dataset=dataset)
            vectors = {short: [] for short in QUERY_METRICS}
            identities: list[str] = []
            for receipt_descriptor in result["per_query_receipts"]:
                receipt = load_json(root / str(receipt_descriptor["path"]))
                primary_records = receipt["primary_records"]
                bridge_records = receipt["bridge_records"]
                if len(primary_records) != len(bridge_records):
                    raise SemanticBridgeAuditError("paired receipt lengths differ")
                for primary, bridge in zip(primary_records, bridge_records):
                    if (
                        primary["query_position"] != bridge["query_position"]
                        or primary["query_row_id"] != bridge["query_row_id"]
                    ):
                        raise SemanticBridgeAuditError("paired query identity changed")
                    identities.append(str(primary["query_row_id"]))
                    for short, field in QUERY_METRICS.items():
                        vectors[short].append(float(bridge[field]) - float(primary[field]))
            if len(identities) != len(set(identities)):
                raise SemanticBridgeAuditError("query identity repeated inside a cell")
            for short, field in SUMMARY_METRICS.items():
                observed = math.fsum(vectors[short]) / len(vectors[short])
                expected = float(result["shellguard_minus_primary"][field])
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=2.0e-14):
                    raise SemanticBridgeAuditError(
                        f"query deltas do not reproduce {dataset}/{short}"
                    )
                pooled[short].extend(vectors[short])
            graded_advantage = [
                graded - binary
                for graded, binary in zip(vectors["jndcg50"], vectors["ndcg50"])
            ]
            cells.append(
                {
                    "dataset": dataset,
                    "direction": result["direction"],
                    "bits": int(result["bits"]),
                    "queries": len(identities),
                    "map_delta": _distribution(vectors["map"]),
                    "ndcg50_delta": _distribution(vectors["ndcg50"]),
                    "jndcg50_delta": _distribution(vectors["jndcg50"]),
                    "graded_minus_binary_delta": _distribution(graded_advantage),
                }
            )
        pooled_advantage = [
            graded - binary
            for graded, binary in zip(pooled["jndcg50"], pooled["ndcg50"])
        ]
        dataset_rows.append(
            {
                "dataset": dataset,
                "query_cell_records": len(pooled["map"]),
                "map_delta": _distribution(pooled["map"]),
                "ndcg50_delta": _distribution(pooled["ndcg50"]),
                "jndcg50_delta": _distribution(pooled["jndcg50"]),
                "graded_minus_binary_delta": _distribution(pooled_advantage),
            }
        )
        source_receipts.append(
            {
                "dataset": dataset,
                "rank_plan_sha256": complete["rank_plan_sha256"],
                "complete_sha256": complete["complete_sha256"],
            }
        )

    body: dict[str, Any] = {
        "schema": "semantic_bridge_query_level_delta_audit_v1",
        "status": "VERIFIED_POSTHOC_ANALYSIS",
        "sources": source_receipts,
        "cells": cells,
        "datasets": dataset_rows,
        "analysis_performed_after_formal_completion": True,
        "formal_labels_used_for_selection": False,
    }
    result = {**body, "analysis_sha256": sha256_json(body)}
    json_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_output, result)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("x", encoding="utf-8", newline="") as handle:
        fields = (
            "dataset",
            "query_cell_records",
            "map_mean",
            "ndcg50_mean",
            "jndcg50_mean",
            "jndcg50_positive_fraction",
            "graded_minus_binary_mean",
            "graded_minus_binary_positive_fraction",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in dataset_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "query_cell_records": row["query_cell_records"],
                    "map_mean": row["map_delta"]["mean"],
                    "ndcg50_mean": row["ndcg50_delta"]["mean"],
                    "jndcg50_mean": row["jndcg50_delta"]["mean"],
                    "jndcg50_positive_fraction": row["jndcg50_delta"][
                        "positive_fraction"
                    ],
                    "graded_minus_binary_mean": row["graded_minus_binary_delta"][
                        "mean"
                    ],
                    "graded_minus_binary_positive_fraction": row[
                        "graded_minus_binary_delta"
                    ]["positive_fraction"],
                }
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=_dataset_argument, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args(argv)
    roots: dict[str, Path] = {}
    for dataset, root in args.dataset:
        if dataset in roots:
            raise SemanticBridgeAuditError(f"duplicate dataset: {dataset}")
        roots[dataset] = root
    result = analyze(roots, json_output=args.json_output, csv_output=args.csv_output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis_sha256": result["analysis_sha256"],
                "datasets": result["datasets"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
