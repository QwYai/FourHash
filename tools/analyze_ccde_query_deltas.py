#!/usr/bin/env python3
"""Deep-verify formal CCDE receipts and summarize paired query-level deltas."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.aggregate_ccde_formal import (
    BITS,
    DATASETS,
    DIRECTIONS,
    PINNED_SOURCE_DIRECTORY,
    atomic_write_text,
    canonical_json_bytes,
    directory_digest,
    load_json,
    require,
    safe_child,
    sha256_json,
    verify_dataset,
)


SCHEMA = "ccde_query_level_delta_audit_v1"
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
    require(bool(values), "cannot summarize an empty query vector")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Mapping[str, Any]:
    require(bool(values), "cannot summarize an empty query vector")
    return {
        "records": len(values),
        "mean": sum(values) / len(values),
        "q10": _quantile(values, 0.10),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "q90": _quantile(values, 0.90),
        "positive_fraction": sum(value > 0.0 for value in values) / len(values),
        "nonnegative_fraction": sum(value >= 0.0 for value in values) / len(values),
    }


def _cell_query_deltas(
    evaluation_root: Path,
    result_descriptor: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, list[float]]]:
    result_path = safe_child(evaluation_root, str(result_descriptor["path"]))
    result = load_json(result_path)
    vectors = {short: [] for short in QUERY_METRICS}
    identities: list[str] = []
    for descriptor in result["per_query_receipts"]:
        receipt = load_json(safe_child(evaluation_root, str(descriptor["path"])))
        primary_records = receipt["primary_records"]
        ccde_records = receipt["ccde_records"]
        require(
            len(primary_records) == len(ccde_records),
            "primary/detail partial lengths differ",
        )
        for primary, ccde in zip(primary_records, ccde_records):
            require(
                primary["query_position"] == ccde["query_position"]
                and primary["query_row_id"] == ccde["query_row_id"],
                "paired query identity changed",
            )
            identities.append(str(primary["query_row_id"]))
            for short, field in QUERY_METRICS.items():
                vectors[short].append(float(ccde[field]) - float(primary[field]))
    require(len(identities) == len(set(identities)), "query identity repeated in a cell")
    summary_delta = result["ccde_minus_primary"]
    for short, field in SUMMARY_METRICS.items():
        observed = sum(vectors[short]) / len(vectors[short])
        expected = float(summary_delta[field])
        require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=2.0e-14),
            f"query deltas do not reproduce {short}: {observed} != {expected}",
        )
    return result, vectors


def analyze(root: Path, output: Path) -> Mapping[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    require(
        not any(part.casefold() in {"oraldata", "processdata"} for part in output.parts),
        "output may not modify protected data roots",
    )
    source_directory = directory_digest(root)
    require(
        source_directory == PINNED_SOURCE_DIRECTORY,
        "formal source directory differs from the pinned digest",
    )

    cells: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[float]]] = {
        dataset: {short: [] for short in QUERY_METRICS} for dataset in DATASETS
    }
    verified_sources: list[Mapping[str, Any]] = []
    for dataset in DATASETS:
        _, stats, _ = verify_dataset(root, dataset)
        verified_sources.append(stats)
        complete_path = next(
            (root / dataset / "metrics").glob("metrics-*/evaluation_complete.json")
        ).resolve(strict=True)
        evaluation_root = complete_path.parent
        complete = load_json(complete_path)
        descriptors = sorted(
            complete["results"],
            key=lambda item: (
                DIRECTIONS.index(str(item["direction"])),
                BITS.index(int(item["bits"])),
            ),
        )
        for descriptor in descriptors:
            result, vectors = _cell_query_deltas(evaluation_root, descriptor)
            for short in QUERY_METRICS:
                pooled[dataset][short].extend(vectors[short])
            graded_advantage = [
                graded - binary
                for graded, binary in zip(vectors["jndcg50"], vectors["ndcg50"])
            ]
            cells.append(
                {
                    "dataset": dataset,
                    "direction": result["direction"],
                    "bits": int(result["bits"]),
                    "queries": len(vectors["map"]),
                    "map_delta": _distribution(vectors["map"]),
                    "ndcg50_delta": _distribution(vectors["ndcg50"]),
                    "jndcg50_delta": _distribution(vectors["jndcg50"]),
                    "graded_minus_binary_delta": _distribution(graded_advantage),
                }
            )

    datasets: list[dict[str, Any]] = []
    for dataset in DATASETS:
        vectors = pooled[dataset]
        graded_advantage = [
            graded - binary
            for graded, binary in zip(vectors["jndcg50"], vectors["ndcg50"])
        ]
        datasets.append(
            {
                "dataset": dataset,
                "query_cell_records": len(vectors["map"]),
                "map_delta": _distribution(vectors["map"]),
                "ndcg50_delta": _distribution(vectors["ndcg50"]),
                "jndcg50_delta": _distribution(vectors["jndcg50"]),
                "graded_minus_binary_delta": _distribution(graded_advantage),
            }
        )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "VERIFIED",
        "source_root": str(root),
        "source_directory": source_directory,
        "verified_sources": verified_sources,
        "cells": cells,
        "datasets": datasets,
        "formal_labels_used_for_selection": False,
        "analysis_performed_after_formal_completion": True,
    }
    result = {**body, "analysis_sha256": sha256_json(body)}
    output.mkdir(parents=True)
    atomic_write_text(
        output / "query_delta_audit.json",
        canonical_json_bytes(result).decode("utf-8") + "\n",
    )
    buffer = io.StringIO(newline="")
    fields = [
        "dataset",
        "query_cell_records",
        "map_mean",
        "ndcg50_mean",
        "jndcg50_mean",
        "jndcg50_positive_fraction",
        "graded_minus_binary_mean",
        "graded_minus_binary_positive_fraction",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in datasets:
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
    atomic_write_text(output / "dataset_query_deltas.csv", buffer.getvalue())
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = analyze(args.root, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis_sha256": result["analysis_sha256"],
                "datasets": result["datasets"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
