"""Diagnose primary/detail Hamming fusion on a deterministic query prefix.

This tool is intentionally an exploratory, label-open diagnostic.  It consumes
an already frozen CCDE encoding cache and the corresponding sealed labels, then
reports expected-tie metrics for predeclared integer primary multipliers.  Its
output must not be represented as a label-isolated formal evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from raw_rebuilt_streaming.metrics import (
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
    mean_query_metrics,
)


BITS = (16, 32, 64)
DIRECTIONS = ("i2t", "t2i")


def _indices(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1:
        raise ValueError(f"{path.name} must be a vector")
    if value.dtype == np.bool_:
        result = np.flatnonzero(value)
    elif len(value) == rows and np.all(np.isin(value, (0, 1))):
        result = np.flatnonzero(value)
    elif value.dtype.kind in "iu":
        result = value.astype(np.int64, copy=False)
    else:
        raise ValueError(f"{path.name} is not an index vector")
    if len(result) == 0 or np.any(result < 0) or np.any(result >= rows):
        raise ValueError(f"{path.name} contains invalid indices")
    if np.unique(result).size != len(result):
        raise ValueError(f"{path.name} contains duplicate indices")
    return np.ascontiguousarray(result, dtype=np.int64)


def _parse_multipliers(raw: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not result or result[0] < 1:
        raise ValueError("multipliers must be positive integers")
    return result


def _bipolar(path: Path, rows: int, columns: int | None = None) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2 or value.shape[0] != rows or value.dtype != np.int8:
        raise ValueError(f"invalid bipolar code geometry in {path.name}")
    if columns is not None and value.shape[1] != columns:
        raise ValueError(f"unexpected code width in {path.name}")
    if not np.all(np.isin(value, (-1, 1))):
        raise ValueError(f"{path.name} is not bipolar")
    return value


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return float(np.mean(np.asarray(materialized, dtype=np.float64)))


def diagnose(
    cache_root: Path,
    runtime_root: Path,
    output_path: Path,
    *,
    multipliers: tuple[int, ...],
    query_limit: int,
    query_batch_size: int,
    device: str,
) -> dict[str, object]:
    cache = cache_root.expanduser().resolve(strict=True)
    arrays = runtime_root.expanduser().resolve(strict=True) / "arrays"
    labels = np.load(arrays / "labels.npy", mmap_mode="r", allow_pickle=False)
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("sealed labels must be a uint8 matrix")
    rows = labels.shape[0]
    query_idx = _indices(arrays / "indQ.npy", rows)
    database_idx = _indices(arrays / "indD.npy", rows)
    if query_limit < 1 or query_batch_size < 1:
        raise ValueError("query limits must be positive")
    query_idx = query_idx[: min(query_limit, len(query_idx))]
    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    query_labels_all = torch.from_numpy(
        np.ascontiguousarray(labels[query_idx], dtype=np.float32)
    ).to(resolved)
    database_labels = torch.from_numpy(
        np.ascontiguousarray(labels[database_idx], dtype=np.float32)
    ).to(resolved)
    database_cardinality = database_labels.sum(dim=1)
    prefixes = build_metric_prefixes(len(database_idx), (50, 100, 1000))
    records: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    code_inventory: list[dict[str, object]] = []

    for bits in BITS:
        for direction in DIRECTIONS:
            query_modality, database_modality = (
                ("image", "text") if direction == "i2t" else ("text", "image")
            )
            primary_query_all = _bipolar(
                cache / f"primary_{query_modality}_codes_{bits}.npy", rows, bits
            )
            primary_database_all = _bipolar(
                cache / f"primary_{database_modality}_codes_{bits}.npy", rows, bits
            )
            detail_query_all = _bipolar(
                cache / f"detail_{query_modality}_codes_{bits}.npy", rows
            )
            detail_bits = int(detail_query_all.shape[1])
            detail_database_all = _bipolar(
                cache / f"detail_{database_modality}_codes_{bits}.npy",
                rows,
                detail_bits,
            )
            for name in (
                f"primary_{query_modality}_codes_{bits}.npy",
                f"primary_{database_modality}_codes_{bits}.npy",
                f"detail_{query_modality}_codes_{bits}.npy",
                f"detail_{database_modality}_codes_{bits}.npy",
            ):
                target = cache / name
                code_inventory.append(
                    {"path": str(target), "size": target.stat().st_size, "sha256": sha256_file(target)}
                )

            primary_database = torch.from_numpy(
                np.ascontiguousarray(primary_database_all[database_idx], dtype=np.float32)
            ).to(resolved)
            detail_database = torch.from_numpy(
                np.ascontiguousarray(detail_database_all[database_idx], dtype=np.float32)
            ).to(resolved)
            for start in range(0, len(query_idx), query_batch_size):
                end = min(len(query_idx), start + query_batch_size)
                canonical = query_idx[start:end]
                primary_query = torch.from_numpy(
                    np.ascontiguousarray(primary_query_all[canonical], dtype=np.float32)
                ).to(resolved)
                detail_query = torch.from_numpy(
                    np.ascontiguousarray(detail_query_all[canonical], dtype=np.float32)
                ).to(resolved)
                primary_distance = torch.round(
                    (bits - primary_query @ primary_database.T) * 0.5
                ).to(torch.int16)
                detail_distance = torch.round(
                    (detail_bits - detail_query @ detail_database.T) * 0.5
                ).to(torch.int16)
                query_labels = query_labels_all[start:end]
                intersection = query_labels @ database_labels.T
                union = (
                    query_labels.sum(dim=1)[:, None]
                    + database_cardinality[None, :]
                    - intersection
                )
                gain = intersection / union.clamp_min(1.0)
                primary_np = primary_distance.cpu().numpy()
                detail_np = detail_distance.cpu().numpy()
                gain_np = gain.cpu().numpy().astype(np.float64, copy=False)
                for offset in range(end - start):
                    row_gain = np.ascontiguousarray(gain_np[offset], dtype=np.float64)
                    relevance = row_gain > 0.0
                    for multiplier in multipliers:
                        key = (
                            multiplier * primary_np[offset].astype(np.int32)
                            + detail_np[offset].astype(np.int32)
                        )
                        metric = expected_tie_metrics_from_distances(
                            relevance,
                            key,
                            bits=bits,
                            graded_gains=row_gain,
                            cutoffs=(50, 100, 1000),
                            prefixes=prefixes,
                            distance_levels=multiplier * bits + detail_bits + 1,
                        )
                        records[(bits, direction, multiplier)].append(metric)

    summaries = []
    for (bits, direction, multiplier), cell_records in sorted(records.items()):
        aggregate = mean_query_metrics(cell_records)
        summaries.append(
            {
                "bits": bits,
                "direction": direction,
                "primary_multiplier": multiplier,
                "primary_shell_invariant": multiplier > 16,
                "queries": len(cell_records),
                "metrics": aggregate,
            }
        )
    best = []
    for bits in BITS:
        for direction in DIRECTIONS:
            candidates = [
                row for row in summaries if row["bits"] == bits and row["direction"] == direction
            ]
            for metric in (
                "map_expected_ties",
                "binary_ndcg_at_50_expected_ties",
                "j_ndcg_at_50_expected_ties",
            ):
                winner = max(candidates, key=lambda row: float(row["metrics"][metric]))
                best.append(
                    {
                        "bits": bits,
                        "direction": direction,
                        "metric": metric,
                        "primary_multiplier": winner["primary_multiplier"],
                        "value": float(winner["metrics"][metric]),
                    }
                )
    body: dict[str, object] = {
        "schema": "ccde_postformal_composite_diagnostic_v1",
        "status": "EXPLORATORY_LABEL_OPEN_NOT_FORMAL",
        "cache_root": str(cache),
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "runtime_root": str(runtime_root.expanduser().resolve(strict=True)),
        "labels_file_sha256": sha256_file(arrays / "labels.npy"),
        "query_indices_file_sha256": sha256_file(arrays / "indQ.npy"),
        "database_indices_file_sha256": sha256_file(arrays / "indD.npy"),
        "query_prefix_rows": len(query_idx),
        "database_rows": len(database_idx),
        "multipliers": list(multipliers),
        "device": str(resolved),
        "code_files": sorted(code_inventory, key=lambda value: str(value["path"])),
        "summaries": summaries,
        "best_by_cell_metric": best,
        "scientific_boundary": (
            "This diagnostic opens formal labels after rank freeze and may only guide a new "
            "train-only development protocol; it is not formal evidence."
        ),
    }
    result = {**body, "diagnostic_sha256": sha256_json(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multipliers", default="1,2,4,8,17")
    parser.add_argument("--query-limit", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = diagnose(
        args.cache_root,
        args.runtime_root,
        args.output,
        multipliers=_parse_multipliers(args.multipliers),
        query_limit=args.query_limit,
        query_batch_size=args.query_batch_size,
        device=args.device,
    )
    print(json.dumps({
        "status": result["status"],
        "query_prefix_rows": result["query_prefix_rows"],
        "diagnostic_sha256": result["diagnostic_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
