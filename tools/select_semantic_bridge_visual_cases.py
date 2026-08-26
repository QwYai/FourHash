"""Select trace-backed post-hoc semantic-bridge shell-refinement cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime import load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import atomic_write_json, numeric_sha256
from raw_rebuilt_runtime.metric_loader import load_frozen_metric_labels
from tools.audit_semantic_bridge_formal import (
    BITS,
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
    _resume_cell,
    _verify_plan,
)
from tools.select_ccde_visual_cases import (
    ANALYSIS_SCHEMA,
    SELECTION_SCHEMA,
    VisualSelectionError,
    _choose_group,
    _decode_row_id,
    _jaccard_gains,
    _position_interval,
    _selection_sha256,
)


def _case_for_query(
    *,
    query_position: int,
    metric_delta: float,
    primary_distance: np.ndarray,
    detail_distance: np.ndarray,
    query_label: np.ndarray,
    database_labels: np.ndarray,
    query_row_id: str,
    database_row_ids: np.ndarray,
    cutoff: int,
    group_size: int,
) -> dict[str, Any] | None:
    """Choose strict low/high semantic subshells inside the rank-k primary shell."""

    if len(primary_distance) < cutoff:
        return None
    boundary_primary = int(np.partition(primary_distance, cutoff - 1)[cutoff - 1])
    shell = np.flatnonzero(primary_distance == boundary_primary)
    if len(shell) < 2 * group_size:
        return None
    shell_detail = detail_distance[shell]
    values, counts = np.unique(shell_detail, return_counts=True)
    if len(values) < 2:
        return None
    low_offset = int(np.searchsorted(np.cumsum(counts), group_size, side="left"))
    high_offset_from_end = int(
        np.searchsorted(np.cumsum(counts[::-1]), group_size, side="left")
    )
    high_offset = len(values) - 1 - high_offset_from_end
    if low_offset >= high_offset:
        return None
    low_value = int(values[low_offset])
    high_value = int(values[high_offset])
    lower_pool = shell[shell_detail <= low_value]
    upper_pool = shell[shell_detail >= high_value]
    gains = _jaccard_gains(query_label, database_labels)
    favored = _choose_group(
        lower_pool, gains, detail_distance, group_size, favored=True
    )
    demoted = _choose_group(
        upper_pool, gains, detail_distance, group_size, favored=False
    )
    if len(favored) != group_size or len(demoted) != group_size:
        return None
    gain_gap = float(gains[favored].mean() - gains[demoted].mean())
    detail_gap = float(detail_distance[demoted].mean() - detail_distance[favored].mean())
    if gain_gap <= 0.0 or detail_gap <= 0.0:
        return None
    composite = primary_distance.astype(np.uint32) * np.uint32(17)
    composite += detail_distance.astype(np.uint32)
    records = []
    for role, indices in (("favored", favored), ("demoted", demoted)):
        for database_position in indices:
            position = int(database_position)
            primary_value = int(primary_distance[position])
            detail_value = int(detail_distance[position])
            composite_value = int(composite[position])
            records.append(
                {
                    "group": role,
                    "database_position": position,
                    "row_id": _decode_row_id(database_row_ids[position]),
                    "primary_distance": primary_value,
                    "detail_distance": detail_value,
                    "composite_distance": composite_value,
                    "primary_expected_position_interval": _position_interval(
                        primary_distance, primary_value
                    ),
                    "ccde_expected_position_interval": _position_interval(
                        composite, composite_value
                    ),
                    "jaccard_gain": float(gains[position]),
                    "binary_relevant": bool(gains[position] > 0.0),
                    "active_label_indices": np.flatnonzero(
                        database_labels[position]
                    ).astype(int).tolist(),
                }
            )
    selected = np.concatenate((favored, demoted))
    primary_before = int(np.count_nonzero(primary_distance < boundary_primary))
    slots = cutoff - primary_before
    boundary_values, boundary_counts = np.unique(shell_detail, return_counts=True)
    boundary_offset = int(
        np.searchsorted(np.cumsum(boundary_counts), slots, side="left")
    )
    return {
        "query_position": query_position,
        "query_row_id": query_row_id,
        "query_active_label_indices": np.flatnonzero(query_label).astype(int).tolist(),
        "formal_query_j_ndcg_delta_at_cutoff": metric_delta,
        "cutoff": cutoff,
        "boundary_primary_distance": boundary_primary,
        "boundary_detail_distance": int(boundary_values[boundary_offset]),
        "primary_shell_size": int(len(shell)),
        "available_positions_in_boundary_shell": slots,
        "strict_favored_pool_size": int(len(lower_pool)),
        "strict_demoted_pool_size": int(len(upper_pool)),
        "selected_jaccard_gap": gain_gap,
        "selected_detail_distance_gap": detail_gap,
        "case_score": float(metric_delta + 0.30 * gain_gap + 0.01 * detail_gap),
        "candidate_row_ids": [_decode_row_id(database_row_ids[i]) for i in selected],
        "candidates": records,
    }


def select_cases(
    *,
    runtime_root: Path,
    plan_root: Path,
    metrics_root: Path,
    direction: str,
    bits: int,
    cutoff: int,
    case_count: int,
    group_size: int,
    scan_queries: int,
    distance_device: str,
    selection_output: Path,
    analysis_output: Path,
) -> dict[str, Any]:
    if direction not in {"i2t", "t2i"} or bits not in BITS:
        raise ValueError("direction or primary bits are unsupported")
    if min(cutoff, case_count, group_size, scan_queries) < 1:
        raise ValueError("cutoff and selection counts must be positive")
    if distance_device not in {"cpu", "cuda"}:
        raise ValueError("distance_device must be cpu or cuda")
    for output in (selection_output, analysis_output):
        if output.exists():
            raise VisualSelectionError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    plan, verified_plan_root = _verify_plan(plan_root)
    metric_root = metrics_root.expanduser().resolve(strict=True)
    complete = _verified_complete(metric_root, str(plan["dataset"]))
    if complete["rank_plan_sha256"] != plan["rank_plan_sha256"]:
        raise VisualSelectionError("completed evaluation and rank plan differ")
    matches = [
        descriptor
        for descriptor in complete["results"]
        if descriptor["direction"] == direction and int(descriptor["bits"]) == bits
    ]
    if len(matches) != 1:
        raise VisualSelectionError("requested formal cell is not unique")
    result = _verified_result(metric_root, matches[0], dataset=str(plan["dataset"]))
    evaluated, chain, primary_records, bridge_records, partials = _resume_cell(
        metric_root, plan, direction, bits
    )
    if (
        evaluated != int(plan["runtime_identity"]["query_rows"])
        or len(primary_records) != evaluated
        or len(bridge_records) != evaluated
        or chain != result["final_receipt_chain_sha256"]
    ):
        raise VisualSelectionError("formal semantic-bridge receipt chain is incomplete")

    runtime = runtime_root.expanduser().resolve(strict=True)
    rank = load_label_free_rank_inputs(runtime)
    arrays: list[np.ndarray] = []
    labels = None
    try:
        identity = _runtime_identity(rank, _runtime_manifest(runtime))
        if identity != plan["runtime_identity"]:
            raise VisualSelectionError("runtime identity and frozen plan differ")
        query_idx = np.asarray(rank.query_idx, dtype=np.int64).copy()
        database_idx = np.asarray(rank.database_idx, dtype=np.int64).copy()
        cache, arrays = _open_backend_cache(verified_plan_root, plan)
    finally:
        rank.close()

    labels = load_frozen_metric_labels(runtime, rank_contract=plan)
    try:
        if labels.source_seal_sha256 != plan["source_seal_sha256"]:
            raise VisualSelectionError("metric labels have another source seal")
        if numeric_sha256(labels.query_row_ids) != plan["runtime_identity"][
            "query_row_ids_numeric_sha256"
        ] or numeric_sha256(labels.database_row_ids) != plan["runtime_identity"][
            "database_row_ids_numeric_sha256"
        ]:
            raise VisualSelectionError("metric identities changed after rank freeze")

        metric_field = f"j_ndcg_at_{cutoff}_expected_ties"
        ranked_queries: list[tuple[float, int, str]] = []
        for primary, bridge in zip(primary_records, bridge_records):
            if (
                primary["query_position"] != bridge["query_position"]
                or primary["query_row_id"] != bridge["query_row_id"]
            ):
                raise VisualSelectionError("paired formal query identity changed")
            delta = float(bridge[metric_field]) - float(primary[metric_field])
            query_position = int(primary["query_position"])
            query_row_id = _decode_row_id(labels.query_row_ids[query_position])
            if query_row_id != primary["query_row_id"]:
                raise VisualSelectionError("formal and metric query IDs differ")
            if delta > 0.0:
                ranked_queries.append((delta, query_position, query_row_id))
        ranked_queries.sort(reverse=True)
        if not ranked_queries:
            raise VisualSelectionError("no positive query-level graded delta exists")

        backend = _PackedDistanceBackend(
            cache, query_idx, database_idx, device=distance_device
        )
        candidates: list[dict[str, Any]] = []
        for delta, query_position, query_row_id in ranked_queries[:scan_queries]:
            primary_distance = backend.distances(
                "primary", direction, bits, query_position, query_position + 1
            )[0]
            semantic_distance = backend.distances(
                "detail", direction, bits, query_position, query_position + 1
            )[0]
            candidate = _case_for_query(
                query_position=query_position,
                metric_delta=delta,
                primary_distance=primary_distance,
                detail_distance=semantic_distance,
                query_label=np.asarray(labels.query[query_position], dtype=np.uint8),
                database_labels=np.asarray(labels.database, dtype=np.uint8),
                query_row_id=query_row_id,
                database_row_ids=labels.database_row_ids,
                cutoff=cutoff,
                group_size=group_size,
            )
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) < case_count:
            raise VisualSelectionError(
                f"only {len(candidates)} strict semantic shell cases found"
            )
        candidates.sort(key=lambda value: value["case_score"], reverse=True)
        chosen = candidates[:case_count]

        selection_body: dict[str, Any] = {
            "schema": SELECTION_SCHEMA,
            "dataset": plan["dataset"],
            "rank_token_sha256": plan["rank_plan_sha256"],
            "source_seal_sha256": plan["source_seal_sha256"],
            "cases": [
                {
                    "case_id": f"{direction}-{bits}b-semantic-shell-{index:02d}",
                    "query_row_id": case["query_row_id"],
                    "candidate_row_ids": case["candidate_row_ids"],
                }
                for index, case in enumerate(chosen, start=1)
            ],
        }
        selection = {
            **selection_body,
            "selection_sha256": _selection_sha256(selection_body),
        }
        atomic_write_json(selection_output, selection)

        analysis_body: dict[str, Any] = {
            "schema": ANALYSIS_SCHEMA,
            "status": "POSTHOC_ILLUSTRATIVE_ONLY",
            "dataset": plan["dataset"],
            "direction": direction,
            "bits": bits,
            "detail_bits": 16,
            "detail_role": "neural_posterior_one_bit_minhash",
            "selected_posterior_threshold": plan["binding"]["selected_threshold"],
            "cutoff": cutoff,
            "selection_sha256": selection["selection_sha256"],
            "rank_plan_sha256": plan["rank_plan_sha256"],
            "source_seal_sha256": plan["source_seal_sha256"],
            "formal_evaluation_complete_sha256": complete["complete_sha256"],
            "formal_partial_terminal_chain_sha256": chain,
            "formal_partial_count": len(partials),
            "formal_query_rows": evaluated,
            "selection_policy": (
                "After formal completion, scan positive per-query J-NDCG deltas "
                "and retain strict lower-vs-upper semantic-distance examples inside "
                "the primary boundary shell. Labels affect illustration only."
            ),
            "cases": [
                {"case_id": selection["cases"][index]["case_id"], **case}
                for index, case in enumerate(chosen)
            ],
        }
        analysis = {
            **analysis_body,
            "analysis_sha256": _selection_sha256(analysis_body),
        }
        atomic_write_json(analysis_output, analysis)
        return analysis
    finally:
        for value in arrays:
            _close_memmap(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--direction", choices=("i2t", "t2i"), default="t2i")
    parser.add_argument("--bits", type=int, choices=BITS, default=64)
    parser.add_argument("--cutoff", type=int, default=50)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--scan-queries", type=int, default=128)
    parser.add_argument("--distance-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select_cases(
        runtime_root=args.runtime,
        plan_root=args.plan,
        metrics_root=args.metrics,
        direction=args.direction,
        bits=args.bits,
        cutoff=args.cutoff,
        case_count=args.case_count,
        group_size=args.group_size,
        scan_queries=args.scan_queries,
        distance_device=args.distance_device,
        selection_output=args.selection_output,
        analysis_output=args.analysis_output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["cases"]),
                "analysis_sha256": result["analysis_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
