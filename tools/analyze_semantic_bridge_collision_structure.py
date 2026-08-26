"""Label-free boundary-shell audit for a completed semantic-bridge evaluation."""

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

from raw_rebuilt_runtime import load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_json
from tools.analyze_ccde_collision_structure import _cell_statistics
from tools.audit_semantic_bridge_formal import (
    BITS,
    DIRECTIONS,
    SemanticBridgeAuditError,
    _verified_complete,
    _verified_result,
)
from tools.formal_ccde_streaming_eval import (
    _PackedDistanceBackend,
    _runtime_identity,
    _runtime_manifest,
)
from tools.formal_semantic_bridge_streaming_eval import (
    _close_memmap,
    _open_backend_cache,
    _verify_plan,
)


def analyze(
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
        raise ValueError("cutoff and query_chunk must be positive")
    if distance_device not in {"cpu", "cuda"}:
        raise ValueError("distance_device must be cpu or cuda")
    if output.exists():
        raise SemanticBridgeAuditError(f"refusing to overwrite {output}")

    plan, verified_plan_root = _verify_plan(plan_root)
    dataset = str(plan["dataset"])
    metric_root = metrics_root.expanduser().resolve(strict=True)
    complete = _verified_complete(metric_root, dataset)
    if complete["rank_plan_sha256"] != plan["rank_plan_sha256"]:
        raise SemanticBridgeAuditError("metric completion and rank plan differ")
    descriptors = complete.get("results")
    if not isinstance(descriptors, list) or len(descriptors) != 6:
        raise SemanticBridgeAuditError("semantic-bridge result grid is incomplete")
    verified_results = [
        _verified_result(metric_root, descriptor, dataset=dataset)
        for descriptor in descriptors
    ]
    result_index = {
        (str(value["direction"]), int(value["bits"])): value
        for value in verified_results
    }
    expected = {(direction, bits) for direction in DIRECTIONS for bits in BITS}
    if set(result_index) != expected:
        raise SemanticBridgeAuditError("semantic-bridge result cells differ")

    runtime = runtime_root.expanduser().resolve(strict=True)
    rank = load_label_free_rank_inputs(runtime)
    arrays: list[np.ndarray] = []
    try:
        identity = _runtime_identity(rank, _runtime_manifest(runtime))
        if identity != plan["runtime_identity"]:
            raise SemanticBridgeAuditError("runtime identity and rank plan differ")
        query_idx = np.asarray(rank.query_idx, dtype=np.int64).copy()
        database_idx = np.asarray(rank.database_idx, dtype=np.int64).copy()
        cache, arrays = _open_backend_cache(verified_plan_root, plan)
    finally:
        rank.close()
    try:
        backend = _PackedDistanceBackend(
            cache, query_idx, database_idx, device=distance_device
        )
        cells = []
        for direction in DIRECTIONS:
            for bits in BITS:
                if int(result_index[(direction, bits)]["_queries"]) != len(query_idx):
                    raise SemanticBridgeAuditError("formal query coverage differs")
                cell = dict(
                    _cell_statistics(
                        backend,
                        direction=direction,
                        bits=bits,
                        query_rows=len(query_idx),
                        cutoff=cutoff,
                        query_chunk=query_chunk,
                    )
                )
                cell["detail_bits"] = 16
                cells.append(cell)
    finally:
        for value in arrays:
            _close_memmap(value)

    body: dict[str, Any] = {
        "schema": "semantic_bridge_collision_structure_v1",
        "status": "VERIFIED_POSTHOC_LABEL_FREE_DIAGNOSTIC",
        "dataset": dataset,
        "cutoff": cutoff,
        "query_chunk": query_chunk,
        "distance_device": distance_device,
        "rank_plan_sha256": plan["rank_plan_sha256"],
        "formal_evaluation_complete_sha256": complete["complete_sha256"],
        "labels_opened": False,
        "selection_or_fallback_performed": False,
        "fixed_semantic_bits": 16,
        "definition": (
            "For each query, locate the primary Hamming shell crossing rank k, "
            "then locate the semantic-code tie crossing the remaining slots inside "
            "that same shell."
        ),
        "cells": cells,
    }
    result = {**body, "report_sha256": sha256_json(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=50)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--distance-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        runtime_root=args.runtime,
        plan_root=args.plan,
        metrics_root=args.metrics,
        cutoff=args.cutoff,
        query_chunk=args.query_chunk,
        distance_device=args.distance_device,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "cells": len(result["cells"]),
                "report_sha256": result["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
