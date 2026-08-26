#!/usr/bin/env python3
"""Audit how CCDE partitions the primary shell crossing a retrieval cutoff.

This diagnostic is deliberately downstream of a completed formal evaluation.
It verifies the frozen plan, result receipts, partial chains, runtime identity,
and encoding-cache binding before computing any statistic.  No labels are
opened: the report measures only the structural ambiguity of the binary ranks.

For each query, the *primary boundary shell* is the equal-Hamming-distance
block that crosses rank ``k``.  Inside that exact shell, the selected detail
bits induce smaller composite ties.  The report compares the primary shell
size with the composite tie that crosses the same cutoff and records the
fraction of candidates structurally distinguished by the detail expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_contract import CCDE_DETAIL_CAP
from raw_rebuilt_runtime import load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_json
from rz_csd_clip512 import BITS
from tools.formal_ccde_streaming_eval import (
    _PackedDistanceBackend,
    _resume_cell,
    _runtime_identity,
    _runtime_manifest,
    _verify_plan,
)
from tools.select_ccde_visual_cases import (
    VisualSelectionError,
    _open_plan_cache,
    _verify_complete_evaluation,
)


REPORT_SCHEMA = "raw_rebuilt_ccde_collision_structure_v1"
DIRECTIONS = ("i2t", "t2i")


class CollisionStructureError(RuntimeError):
    """The frozen evidence cannot support the requested diagnostic."""


def _percentile(values: np.ndarray, q: float) -> float:
    """Return a deterministic linear percentile as a JSON-safe float."""

    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise CollisionStructureError("cannot summarize an empty or non-finite vector")
    return {
        "mean": float(array.mean(dtype=np.float64)),
        "p25": _percentile(array, 25.0),
        "median": _percentile(array, 50.0),
        "p75": _percentile(array, 75.0),
        "p90": _percentile(array, 90.0),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _query_boundary_statistics(
    primary_distance: np.ndarray,
    detail_distance: np.ndarray,
    cutoff: int,
) -> tuple[int, int, int, int, int, float]:
    """Measure the primary and refined tie crossing ``cutoff`` for one query."""

    primary = np.asarray(primary_distance)
    detail = np.asarray(detail_distance)
    if primary.ndim != 1 or detail.shape != primary.shape:
        raise CollisionStructureError("primary/detail distance shapes differ")
    if cutoff < 1 or cutoff > len(primary):
        raise CollisionStructureError("cutoff lies outside the database")
    if np.any(primary < 0) or np.any(detail < 0):
        raise CollisionStructureError("Hamming distances must be nonnegative")

    primary_boundary = int(np.partition(primary, cutoff - 1)[cutoff - 1])
    primary_before = int(np.count_nonzero(primary < primary_boundary))
    slots_in_primary_shell = cutoff - primary_before
    shell_mask = primary == primary_boundary
    primary_shell_size = int(np.count_nonzero(shell_mask))
    if not 1 <= slots_in_primary_shell <= primary_shell_size:
        raise CollisionStructureError("invalid primary boundary-shell occupancy")

    detail_in_shell = detail[shell_mask]
    detail_values, detail_counts = np.unique(detail_in_shell, return_counts=True)
    detail_offset = int(
        np.searchsorted(np.cumsum(detail_counts), slots_in_primary_shell, side="left")
    )
    if detail_offset >= len(detail_values):
        raise CollisionStructureError("detail boundary could not be located")
    detail_boundary = int(detail_values[detail_offset])
    composite_boundary_tie_size = int(detail_counts[detail_offset])
    detail_subshells = int(len(detail_values))
    distinguished_fraction = 1.0 - (
        float(composite_boundary_tie_size) / float(primary_shell_size)
    )
    if not 0.0 <= distinguished_fraction <= 1.0:
        raise CollisionStructureError("invalid distinguished fraction")
    return (
        primary_boundary,
        primary_shell_size,
        slots_in_primary_shell,
        detail_boundary,
        composite_boundary_tie_size,
        distinguished_fraction,
    )


def _cell_statistics(
    backend: _PackedDistanceBackend,
    *,
    direction: str,
    bits: int,
    query_rows: int,
    cutoff: int,
    query_chunk: int,
) -> Mapping[str, Any]:
    primary_boundaries: list[float] = []
    primary_shell_sizes: list[float] = []
    primary_shell_slots: list[float] = []
    detail_boundaries: list[float] = []
    composite_tie_sizes: list[float] = []
    detail_subshell_counts: list[float] = []
    distinguished_fractions: list[float] = []

    for start in range(0, query_rows, query_chunk):
        stop = min(start + query_chunk, query_rows)
        primary = backend.distances("primary", direction, bits, start, stop)
        detail = backend.distances("detail", direction, bits, start, stop)
        if primary.shape != detail.shape or primary.shape[0] != stop - start:
            raise CollisionStructureError("distance backend returned an invalid chunk")
        for row in range(stop - start):
            stats = _query_boundary_statistics(primary[row], detail[row], cutoff)
            (
                primary_boundary,
                primary_shell_size,
                slots_in_primary_shell,
                detail_boundary,
                composite_tie_size,
                distinguished_fraction,
            ) = stats
            shell_detail = detail[row][primary[row] == primary_boundary]
            detail_subshells = int(np.unique(shell_detail).size)
            primary_boundaries.append(float(primary_boundary))
            primary_shell_sizes.append(float(primary_shell_size))
            primary_shell_slots.append(float(slots_in_primary_shell))
            detail_boundaries.append(float(detail_boundary))
            composite_tie_sizes.append(float(composite_tie_size))
            detail_subshell_counts.append(float(detail_subshells))
            distinguished_fractions.append(float(distinguished_fraction))

    primary_sizes = np.asarray(primary_shell_sizes, dtype=np.float64)
    composite_sizes = np.asarray(composite_tie_sizes, dtype=np.float64)
    return {
        "direction": direction,
        "bits": bits,
        "detail_bits": min(CCDE_DETAIL_CAP, bits),
        "cutoff": cutoff,
        "query_rows": query_rows,
        "primary_boundary_distance": _summarize(primary_boundaries),
        "primary_boundary_shell_size": _summarize(primary_shell_sizes),
        "slots_inside_primary_boundary_shell": _summarize(primary_shell_slots),
        "detail_boundary_distance": _summarize(detail_boundaries),
        "composite_boundary_tie_size": _summarize(composite_tie_sizes),
        "detail_subshell_count_inside_primary_shell": _summarize(
            detail_subshell_counts
        ),
        "distinguished_fraction_inside_primary_shell": _summarize(
            distinguished_fractions
        ),
        "queries_with_primary_collision": int(np.count_nonzero(primary_sizes > 1.0)),
        "queries_with_residual_composite_tie": int(
            np.count_nonzero(composite_sizes > 1.0)
        ),
        "all_composite_ties_are_subsets_of_primary_shells": bool(
            np.all(composite_sizes <= primary_sizes)
        ),
    }


def analyze_collision_structure(
    *,
    runtime_root: Path,
    plan_root: Path,
    metrics_root: Path,
    cutoff: int,
    query_chunk: int,
    distance_device: str,
    output: Path,
) -> Mapping[str, Any]:
    if cutoff < 1 or query_chunk < 1:
        raise ValueError("cutoff and query-chunk must be positive")
    if distance_device not in {"cpu", "cuda"}:
        raise ValueError("distance-device must be cpu or cuda")
    if output.exists():
        raise CollisionStructureError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    runtime_root = runtime_root.expanduser().resolve(strict=True)
    plan_root = plan_root.expanduser().resolve(strict=True)
    metrics_root = metrics_root.expanduser().resolve(strict=True)
    plan = _verify_plan(plan_root)

    completion_sha256: str | None = None
    receipt_chains: list[Mapping[str, Any]] = []
    for direction in DIRECTIONS:
        for bits in BITS:
            complete = _verify_complete_evaluation(
                metrics_root, plan, direction, int(bits)
            )
            if completion_sha256 is None:
                completion_sha256 = str(complete["complete_sha256"])
            elif completion_sha256 != complete["complete_sha256"]:
                raise CollisionStructureError("cells refer to different completions")
            evaluated, chain, primary_records, ccde_records, partials = _resume_cell(
                metrics_root, plan, direction, int(bits)
            )
            expected_queries = int(plan["runtime_identity"]["query_rows"])
            if (
                evaluated != expected_queries
                or len(primary_records) != expected_queries
                or len(ccde_records) != expected_queries
            ):
                raise CollisionStructureError("formal metric receipt chain is incomplete")
            receipt_chains.append(
                {
                    "direction": direction,
                    "bits": int(bits),
                    "partial_count": len(partials),
                    "terminal_chain_sha256": chain,
                }
            )

    rank = load_label_free_rank_inputs(runtime_root)
    cache = None
    try:
        identity = _runtime_identity(rank, _runtime_manifest(runtime_root))
        if identity != plan["runtime_identity"]:
            raise CollisionStructureError("runtime identity differs from frozen plan")
        query_idx = np.asarray(rank.query_idx, dtype=np.int64).copy()
        database_idx = np.asarray(rank.database_idx, dtype=np.int64).copy()
        cache = _open_plan_cache(plan_root, plan)
    finally:
        rank.close()
    if cache is None:
        raise AssertionError("verified encoding cache is unavailable")

    try:
        backend = _PackedDistanceBackend(
            cache, query_idx, database_idx, device=distance_device
        )
        cells = [
            _cell_statistics(
                backend,
                direction=direction,
                bits=int(bits),
                query_rows=len(query_idx),
                cutoff=cutoff,
                query_chunk=query_chunk,
            )
            for direction in DIRECTIONS
            for bits in BITS
        ]
    finally:
        cache.close()

    body: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "POSTHOC_LABEL_FREE_DIAGNOSTIC",
        "dataset": plan["dataset"],
        "cutoff": cutoff,
        "query_chunk": query_chunk,
        "distance_device": distance_device,
        "rank_plan_sha256": plan["rank_plan_sha256"],
        "source_seal_sha256": plan["source_seal_sha256"],
        "formal_evaluation_complete_sha256": completion_sha256,
        "formal_receipt_chains": receipt_chains,
        "labels_opened": False,
        "selection_or_fallback_performed": False,
        "definition": (
            "For each query, locate the primary Hamming shell crossing rank k; "
            "then locate the detail-distance tie crossing the remaining slots "
            "inside that same shell. Distinguished fraction is one minus the "
            "composite boundary-tie size divided by the primary shell size."
        ),
        "cells": cells,
    }
    report = {**body, "report_sha256": sha256_json(body)}
    atomic_write_json(output, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=50)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--distance-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = analyze_collision_structure(
        runtime_root=args.runtime,
        plan_root=args.plan,
        metrics_root=args.metrics,
        cutoff=args.cutoff,
        query_chunk=args.query_chunk,
        distance_device=args.distance_device,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
