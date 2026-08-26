#!/usr/bin/env python3
"""Select traceable post-hoc CCDE shell-refinement cases.

The selector is deliberately downstream of a completed formal evaluation.  It
first verifies the frozen, label-free rank plan, its content-addressed encoding
cache, and the receipt chain for the requested cell.  Only then does it open
query/database labels to locate illustrative (not quantitative) examples.

The compact selection manifest contains canonical row IDs only and is accepted
by :mod:`raw_rebuilt_visuals`.  A separate analysis file records why each row
was chosen and the exact primary/detail distance intervals used in the figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_contract import CCDE_DETAIL_CAP
from raw_rebuilt_neural.ccde_ranking import _open_encoding_cache
from raw_rebuilt_runtime import load_label_free_rank_inputs, load_metric_labels
from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    load_json,
    numeric_sha256,
    sha256_file,
    sha256_json,
)
from rz_csd_clip512 import BITS
from tools.formal_ccde_streaming_eval import (
    EVALUATION_SCHEMA,
    _PackedDistanceBackend,
    _resume_cell,
    _runtime_identity,
    _runtime_manifest,
    _verify_plan,
)


SELECTION_SCHEMA = "raw_rebuilt_rank_visual_selection_v1"
ANALYSIS_SCHEMA = "raw_rebuilt_ccde_visual_case_analysis_v1"


class VisualSelectionError(RuntimeError):
    """Formal evidence cannot support the requested visual selection."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _selection_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _decode_row_id(value: Any) -> str:
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        value = value.decode("ascii")
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise VisualSelectionError("runtime row ID is not a lowercase SHA-256 digest")
    return result


def _verify_complete_evaluation(
    metrics_root: Path,
    plan: Mapping[str, Any],
    direction: str,
    bits: int,
) -> Mapping[str, Any]:
    complete_path = metrics_root / "evaluation_complete.json"
    complete = load_json(complete_path)
    body = {key: complete[key] for key in complete if key != "complete_sha256"}
    if (
        complete.get("schema") != EVALUATION_SCHEMA
        or complete.get("status") != "COMPLETE"
        or complete.get("storage_bounded_complete_gallery_evaluation") is not True
        or complete.get("formal_gate_or_fallback_used") is not False
        or complete.get("primary_shell_order_is_invariant") is not True
        or complete.get("rank_plan_sha256") != plan["rank_plan_sha256"]
        or complete.get("source_seal_sha256") != plan["source_seal_sha256"]
        or sha256_json(body) != complete.get("complete_sha256")
    ):
        raise VisualSelectionError("completed evaluation receipt differs from the frozen plan")
    matches = [
        value
        for value in complete.get("results", [])
        if value.get("direction") == direction and int(value.get("bits", -1)) == bits
    ]
    if len(matches) != 1:
        raise VisualSelectionError("completed evaluation does not contain exactly one requested cell")
    descriptor = matches[0]
    result_path = metrics_root / str(descriptor.get("path", ""))
    if (
        not result_path.is_file()
        or result_path.stat().st_size != int(descriptor.get("size", -1))
        or sha256_file(result_path) != descriptor.get("sha256")
    ):
        raise VisualSelectionError("requested metric result file differs from completion receipt")
    result = load_json(result_path)
    result_body = {
        key: result[key] for key in result if key != "metric_result_sha256"
    }
    if (
        sha256_json(result_body) != result.get("metric_result_sha256")
        or result.get("metric_result_sha256") != descriptor.get("metric_result_sha256")
    ):
        raise VisualSelectionError("requested metric result content hash changed")
    return complete


def _open_plan_cache(plan_root: Path, plan: Mapping[str, Any]) -> Any:
    cache_contract = plan.get("encoding_cache", {})
    cache_root = plan_root / str(cache_contract.get("path", ""))
    manifest_path = cache_root / "manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != int(cache_contract.get("manifest_size", -1))
        or sha256_file(manifest_path) != cache_contract.get("manifest_sha256")
    ):
        raise VisualSelectionError("encoding-cache manifest differs from the frozen plan")
    manifest = load_json(manifest_path)
    binding = manifest.get("binding")
    if (
        not isinstance(binding, dict)
        or binding.get("encoding_binding_sha256")
        != cache_contract.get("encoding_binding_sha256")
    ):
        raise VisualSelectionError("encoding cache is bound to another frozen plan")
    return _open_encoding_cache(cache_root, binding)


def _position_interval(distance: np.ndarray, value: int) -> list[int]:
    lower = int(np.count_nonzero(distance < value))
    equal = int(np.count_nonzero(distance == value))
    return [lower + 1, lower + equal]


def _jaccard_gains(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    query_u8 = np.asarray(query, dtype=np.uint8)
    database_u8 = np.asarray(database, dtype=np.uint8)
    # The registered label dimensions are at most 80, so uint8 dot products
    # cannot overflow.  Keeping the verified database array in its native dtype
    # avoids recopying the full COCO label matrix for every scanned query.
    intersection = database_u8 @ query_u8
    union = database_u8.sum(axis=1, dtype=np.uint16) + np.uint16(
        query_u8.sum(dtype=np.uint16)
    ) - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(database_u8), dtype=np.float64),
        where=union != 0,
    )


def _choose_group(
    pool: np.ndarray,
    gains: np.ndarray,
    detail_distance: np.ndarray,
    count: int,
    *,
    favored: bool,
) -> np.ndarray:
    if len(pool) < count:
        return np.empty(0, dtype=np.int64)
    if favored:
        order = np.lexsort((pool, detail_distance[pool], -gains[pool]))
    else:
        order = np.lexsort((pool, -detail_distance[pool], gains[pool]))
    return np.asarray(pool[order[:count]], dtype=np.int64)


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
    detail_bits: int,
) -> dict[str, Any] | None:
    if len(primary_distance) < cutoff:
        return None
    boundary_primary = int(np.partition(primary_distance, cutoff - 1)[cutoff - 1])
    shell = np.flatnonzero(primary_distance == boundary_primary)
    slots = cutoff - int(np.count_nonzero(primary_distance < boundary_primary))
    if len(shell) < 2 * group_size or slots < 1:
        return None
    shell_detail = detail_distance[shell]
    detail_values, detail_counts = np.unique(shell_detail, return_counts=True)
    cumulative = np.cumsum(detail_counts)
    boundary_offset = int(np.searchsorted(cumulative, slots, side="left"))
    boundary_detail = int(detail_values[boundary_offset])
    lower_pool = shell[shell_detail < boundary_detail]
    upper_pool = shell[shell_detail > boundary_detail]
    # A strict lower/upper pair makes the visual independent of tie ordering at
    # either the primary or detail boundary.  Ambiguous detail ties are omitted.
    if len(lower_pool) < group_size or len(upper_pool) < group_size:
        return None
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
    selected = np.concatenate((favored, demoted))
    composite = primary_distance.astype(np.uint32) * np.uint32(detail_bits + 1)
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
    score = metric_delta + 0.30 * gain_gap + 0.01 * detail_gap
    return {
        "query_position": query_position,
        "query_row_id": query_row_id,
        "query_active_label_indices": np.flatnonzero(query_label).astype(int).tolist(),
        "formal_query_j_ndcg_delta_at_cutoff": metric_delta,
        "cutoff": cutoff,
        "boundary_primary_distance": boundary_primary,
        "boundary_detail_distance": boundary_detail,
        "primary_shell_size": int(len(shell)),
        "available_positions_in_boundary_shell": int(slots),
        "strict_favored_pool_size": int(len(lower_pool)),
        "strict_demoted_pool_size": int(len(upper_pool)),
        "selected_jaccard_gap": gain_gap,
        "selected_detail_distance_gap": detail_gap,
        "case_score": float(score),
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
    if direction not in {"i2t", "t2i"}:
        raise ValueError("direction must be i2t or t2i")
    if bits not in BITS:
        raise ValueError(f"bits must be one of {BITS}")
    if min(cutoff, case_count, group_size, scan_queries) < 1:
        raise ValueError("cutoff, case-count, group-size, and scan-queries must be positive")
    if distance_device not in {"cpu", "cuda"}:
        raise ValueError("distance-device must be cpu or cuda")
    for output in (selection_output, analysis_output):
        if output.exists():
            raise VisualSelectionError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    plan_root = plan_root.expanduser().resolve(strict=True)
    metrics_root = metrics_root.expanduser().resolve(strict=True)
    runtime_root = runtime_root.expanduser().resolve(strict=True)
    plan = _verify_plan(plan_root)
    complete = _verify_complete_evaluation(metrics_root, plan, direction, bits)
    evaluated, chain, primary_records, ccde_records, partials = _resume_cell(
        metrics_root, plan, direction, bits
    )
    if evaluated != int(plan["runtime_identity"]["query_rows"]):
        raise VisualSelectionError("requested metric cell does not cover every formal query")
    if len(primary_records) != len(ccde_records) or not primary_records:
        raise VisualSelectionError("formal metric records are incomplete")

    rank = load_label_free_rank_inputs(runtime_root)
    cache = None
    labels = None
    try:
        identity = _runtime_identity(rank, _runtime_manifest(runtime_root))
        if identity != plan["runtime_identity"]:
            raise VisualSelectionError("runtime identity differs from the frozen plan")
        query_idx = np.asarray(rank.query_idx, dtype=np.int64).copy()
        database_idx = np.asarray(rank.database_idx, dtype=np.int64).copy()
        cache = _open_plan_cache(plan_root, plan)
    finally:
        rank.close()
    if cache is None:
        raise AssertionError("verified encoding cache is unavailable")

    # This is the only label-opening boundary in the selector.  It is reached
    # after the plan, complete evaluation, receipt chain, runtime, and cache all
    # verify, so selected examples cannot influence the already frozen ranks.
    labels = load_metric_labels(runtime_root, rank_contract=plan)
    try:
        if labels.source_seal_sha256 != plan["source_seal_sha256"]:
            raise VisualSelectionError("metric labels have another source seal")
        if numeric_sha256(labels.query_row_ids) != plan["runtime_identity"][
            "query_row_ids_numeric_sha256"
        ]:
            raise VisualSelectionError("query identities changed after rank freeze")
        if numeric_sha256(labels.database_row_ids) != plan["runtime_identity"][
            "database_row_ids_numeric_sha256"
        ]:
            raise VisualSelectionError("database identities changed after rank freeze")

        metric_field = f"j_ndcg_at_{cutoff}_expected_ties"
        ranked_queries: list[tuple[float, int, str]] = []
        for primary, ccde in zip(primary_records, ccde_records):
            if primary.get("query_position") != ccde.get("query_position"):
                raise VisualSelectionError("formal primary/CCDE query order changed")
            if metric_field not in primary or metric_field not in ccde:
                raise VisualSelectionError(f"formal records do not contain {metric_field}")
            delta = float(ccde[metric_field]) - float(primary[metric_field])
            query_position = int(primary["query_position"])
            query_row_id = _decode_row_id(labels.query_row_ids[query_position])
            if query_row_id != primary.get("query_row_id") or query_row_id != ccde.get(
                "query_row_id"
            ):
                raise VisualSelectionError("formal record query identity changed")
            if delta > 0.0:
                ranked_queries.append((delta, query_position, query_row_id))
        ranked_queries.sort(reverse=True)
        if not ranked_queries:
            raise VisualSelectionError("no positive per-query graded delta is available")

        backend = _PackedDistanceBackend(
            cache, query_idx, database_idx, device=distance_device
        )
        candidates: list[dict[str, Any]] = []
        for delta, query_position, query_row_id in ranked_queries[:scan_queries]:
            primary_distance = backend.distances(
                "primary", direction, bits, query_position, query_position + 1
            )[0]
            detail_distance = backend.distances(
                "detail", direction, bits, query_position, query_position + 1
            )[0]
            candidate = _case_for_query(
                query_position=query_position,
                metric_delta=delta,
                primary_distance=primary_distance,
                detail_distance=detail_distance,
                query_label=np.asarray(labels.query[query_position], dtype=np.uint8),
                database_labels=np.asarray(labels.database, dtype=np.uint8),
                query_row_id=query_row_id,
                database_row_ids=labels.database_row_ids,
                cutoff=cutoff,
                group_size=group_size,
                detail_bits=min(CCDE_DETAIL_CAP, bits),
            )
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) < case_count:
            raise VisualSelectionError(
                f"only {len(candidates)} strict shell cases found; need {case_count}"
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
                    "case_id": f"{direction}-{bits}b-shell-{index:02d}",
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
            "detail_bits": min(CCDE_DETAIL_CAP, bits),
            "cutoff": cutoff,
            "selection_sha256": selection["selection_sha256"],
            "rank_plan_sha256": plan["rank_plan_sha256"],
            "source_seal_sha256": plan["source_seal_sha256"],
            "formal_evaluation_complete_sha256": complete["complete_sha256"],
            "formal_partial_terminal_chain_sha256": chain,
            "formal_partial_count": len(partials),
            "formal_query_rows": evaluated,
            "selection_policy": (
                "After completed formal evaluation, scan positive per-query J-NDCG "
                "deltas and retain strict lower-vs-upper detail-distance examples "
                "inside the primary shell crossing the requested cutoff. Labels "
                "select illustrative rows only and never alter model, bits, or rank."
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
        cache.close()


def _build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
