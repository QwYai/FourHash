"""Post-freeze, expected-tie retrieval metrics for sealed rank artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from raw_rebuilt_runtime import load_metric_labels
from raw_rebuilt_runtime.contract import atomic_write_json, load_json, numeric_sha256, sha256_file, sha256_json

from .integrity import production_code_inventory, reject_unsafe_output_path
from .ranking import RANK_SCHEMA, RankingError, verify_rank_cell


METRIC_SCHEMA = "raw_rebuilt_neural_metrics_v1"


class MetricError(RuntimeError):
    """Raised when labels or ranks differ from their frozen boundary."""


def expected_tie_metrics(
    relevance: np.ndarray,
    rank_groups: np.ndarray,
    *,
    cutoffs: Sequence[int] = (50, 100, 1000),
) -> dict[str, float | int | bool]:
    """Exact expectation under uniform permutations inside evidence ties.

    Canonical row IDs may choose a byte-stable storage permutation, but they do
    not receive retrieval credit.  AP, precision, recall, and binary nDCG are
    integrated analytically over each exact rank group.
    """

    relevant = np.asarray(relevance, dtype=bool)
    groups = np.asarray(rank_groups)
    if relevant.ndim != 1 or groups.shape != relevant.shape or groups.dtype.kind not in "iu":
        raise ValueError("relevance and rank_groups must be aligned one-dimensional vectors")
    if len(groups) == 0 or np.any(groups[1:] < groups[:-1]):
        raise ValueError("rank groups must be nonempty and nondecreasing")
    cutoff_values = tuple(sorted(set(int(value) for value in cutoffs)))
    if not cutoff_values or cutoff_values[0] < 1:
        raise ValueError("metric cutoffs must be positive")
    total_relevant = int(relevant.sum())
    result: dict[str, float | int | bool] = {
        "database_rows": int(len(relevant)),
        "relevant_rows": total_relevant,
        "has_relevant": bool(total_relevant > 0),
    }
    deterministic_cumulative = np.cumsum(relevant, dtype=np.int64)
    relevant_positions = np.flatnonzero(relevant)
    if total_relevant:
        deterministic_ap = float(
            np.mean(
                deterministic_cumulative[relevant_positions]
                / (relevant_positions.astype(np.float64) + 1.0)
            )
        )
    else:
        deterministic_ap = 0.0
    expected_ap_numerator = 0.0
    expected_dcg = {cutoff: 0.0 for cutoff in cutoff_values}
    expected_relevant_at = {cutoff: 0.0 for cutoff in cutoff_values}
    previous_relevant = 0
    start = 0
    while start < len(groups):
        end = start + 1
        while end < len(groups) and groups[end] == groups[start]:
            end += 1
        size = end - start
        block_relevant = int(relevant[start:end].sum())
        if block_relevant:
            probability = block_relevant / float(size)
            for offset in range(1, size + 1):
                before = 0.0 if size == 1 else (offset - 1) * (block_relevant - 1) / float(size - 1)
                expected_ap_numerator += probability * (
                    previous_relevant + 1.0 + before
                ) / float(start + offset)
            for cutoff in cutoff_values:
                take = max(0, min(size, cutoff - start))
                if take:
                    expected_relevant_at[cutoff] += take * probability
                    positions = np.arange(start + 1, start + take + 1, dtype=np.float64)
                    expected_dcg[cutoff] += probability * float(
                        np.sum(1.0 / np.log2(positions + 1.0))
                    )
        previous_relevant += block_relevant
        start = end
    result["map_expected_ties"] = (
        expected_ap_numerator / total_relevant if total_relevant else 0.0
    )
    result["map_canonical_storage"] = deterministic_ap
    for cutoff in cutoff_values:
        effective = min(cutoff, len(relevant))
        rel_at = expected_relevant_at[cutoff]
        ideal_count = min(total_relevant, effective)
        if ideal_count:
            positions = np.arange(1, ideal_count + 1, dtype=np.float64)
            ideal_dcg = float(np.sum(1.0 / np.log2(positions + 1.0)))
        else:
            ideal_dcg = 0.0
        result[f"precision_at_{cutoff}_expected_ties"] = rel_at / effective
        result[f"recall_at_{cutoff}_expected_ties"] = rel_at / total_relevant if total_relevant else 0.0
        result[f"ndcg_at_{cutoff}_expected_ties"] = expected_dcg[cutoff] / ideal_dcg if ideal_dcg else 0.0
    return result


def expected_graded_ndcg(
    gain: np.ndarray,
    rank_groups: np.ndarray,
    *,
    cutoffs: Sequence[int] = (50,),
) -> dict[str, float]:
    """Exact expected Jaccard-NDCG under uniform evidence-tie permutations."""

    graded = np.asarray(gain, dtype=np.float64)
    groups = np.asarray(rank_groups)
    if graded.ndim != 1 or groups.shape != graded.shape or groups.dtype.kind not in "iu":
        raise ValueError("gain and rank_groups must be aligned one-dimensional vectors")
    if len(groups) == 0 or np.any(groups[1:] < groups[:-1]):
        raise ValueError("rank groups must be nonempty and nondecreasing")
    if not np.isfinite(graded).all() or np.any(graded < 0.0) or np.any(graded > 1.0):
        raise ValueError("graded gains must be finite values in [0,1]")
    cutoff_values = tuple(sorted(set(int(value) for value in cutoffs)))
    if not cutoff_values or cutoff_values[0] < 1:
        raise ValueError("metric cutoffs must be positive")
    maximum = min(max(cutoff_values), len(graded))
    discount = 1.0 / np.log2(np.arange(2, maximum + 2, dtype=np.float64))
    discount_prefix = np.r_[0.0, np.cumsum(discount)]
    expected_dcg = {cutoff: 0.0 for cutoff in cutoff_values}
    start = 0
    while start < len(groups):
        end = start + 1
        while end < len(groups) and groups[end] == groups[start]:
            end += 1
        mean_gain = float(np.mean(graded[start:end], dtype=np.float64))
        for cutoff in cutoff_values:
            take = max(0, min(end - start, min(cutoff, len(graded)) - start))
            if take:
                expected_dcg[cutoff] += mean_gain * float(
                    discount_prefix[start + take] - discount_prefix[start]
                )
        start = end
    ordered_ideal = np.sort(graded)[::-1]
    result: dict[str, float] = {}
    for cutoff in cutoff_values:
        effective = min(cutoff, len(graded))
        ideal_dcg = float(np.sum(ordered_ideal[:effective] * discount[:effective]))
        result[f"jndcg_at_{cutoff}_expected_ties"] = (
            expected_dcg[cutoff] / ideal_dcg if ideal_dcg else 0.0
        )
    return result


def _verify_rank_manifest(rank_root: Path) -> dict[str, Any]:
    manifest_path = rank_root / "rank_manifest.json"
    manifest = load_json(manifest_path)
    body = {key: manifest[key] for key in manifest if key != "rank_manifest_sha256"}
    if manifest.get("schema") != RANK_SCHEMA or sha256_json(body) != manifest.get("rank_manifest_sha256"):
        raise MetricError("rank manifest schema/hash changed")
    if manifest.get("status") != "rank_state_frozen" or manifest.get("labels_loaded_during_freeze") is not False:
        raise MetricError("rank manifest was not frozen before labels")
    current_code = production_code_inventory()["code_inventory_sha256"]
    if manifest.get("code_inventory", {}).get("code_inventory_sha256") != current_code:
        raise MetricError("current neural/runtime code differs from frozen rank code")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise MetricError("rank manifest has no cells")
    for descriptor in cells:
        path = rank_root / str(descriptor["path"])
        if path.stat().st_size != int(descriptor["size"]) or sha256_file(path) != descriptor["sha256"]:
            raise MetricError("rank cell contract inventory changed")
    return manifest


def _mean_query_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise MetricError("metric cell has no queries")
    scalar_keys = sorted(
        key
        for key, value in records[0].items()
        if isinstance(value, float)
    )
    summary = {
        key: float(np.mean([float(record[key]) for record in records], dtype=np.float64))
        for key in scalar_keys
    }
    valid = [record for record in records if record["has_relevant"]]
    summary["queries"] = len(records)
    summary["queries_with_relevant"] = len(valid)
    if valid:
        summary["map_expected_ties_valid_queries"] = float(
            np.mean([record["map_expected_ties"] for record in valid], dtype=np.float64)
        )
    else:
        summary["map_expected_ties_valid_queries"] = 0.0
    return summary


def evaluate_frozen_ranks(
    runtime_root: Path,
    rank_root: Path,
    output_parent: Path,
    *,
    cutoffs: Sequence[int] = (50, 100, 1000),
    _test_allow_synthetic: bool = False,
) -> Path:
    """Open Q/D labels only after replaying the frozen rank manifest."""

    root = Path(rank_root).expanduser().resolve(strict=True)
    manifest = _verify_rank_manifest(root)
    # Replay every cell receipt before crossing the metric-label boundary.
    # Keeping verified read-only memmaps open avoids reinterpreting rank bytes
    # after labels become visible.
    verified_cells = []
    for descriptor in manifest["cells"]:
        contract_path = root / descriptor["path"]
        contract = load_json(contract_path)
        orders, groups = verify_rank_cell(contract_path.parent, contract)
        verified_cells.append((descriptor, contract_path, contract, orders, groups))
    metric_labels = load_metric_labels(
        runtime_root,
        rank_contract=manifest,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    if metric_labels.source_seal_sha256 != manifest["source_seal_sha256"]:
        raise MetricError("metric labels and rank manifest source seals differ")
    if numeric_sha256(metric_labels.query_row_ids) != manifest["query_row_ids_numeric_sha256"]:
        raise MetricError("query label rows differ from frozen query identities")
    if numeric_sha256(metric_labels.database_row_ids) != manifest["database_row_ids_numeric_sha256"]:
        raise MetricError("database label rows differ from frozen database identities")
    output = reject_unsafe_output_path(Path(output_parent), field="metric output")
    evaluation_root = output / f"metrics-{manifest['rank_manifest_sha256'][:16]}"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    result_descriptors = []
    for descriptor, contract_path, contract, orders, groups in verified_cells:
        if orders.shape != (len(metric_labels.query), len(metric_labels.database)):
            raise MetricError("rank cell and metric label geometry differ")
        per_query: list[dict[str, Any]] = []
        for query_position in range(len(metric_labels.query)):
            order = np.asarray(orders[query_position], dtype=np.int64)
            query_labels = np.asarray(metric_labels.query[query_position], dtype=np.uint8)
            ordered_database_labels = np.asarray(
                metric_labels.database[order], dtype=np.uint8
            )
            intersection = ordered_database_labels @ query_labels
            relevance = intersection > 0
            record = expected_tie_metrics(
                relevance,
                np.asarray(groups[query_position]),
                cutoffs=cutoffs,
            )
            union = (
                ordered_database_labels.sum(axis=1, dtype=np.int64)
                + int(query_labels.sum())
                - intersection.astype(np.int64, copy=False)
            )
            gain = np.divide(
                intersection,
                union,
                out=np.zeros(len(order), dtype=np.float64),
                where=union > 0,
            )
            record.update(
                expected_graded_ndcg(
                    gain,
                    np.asarray(groups[query_position]),
                    cutoffs=cutoffs,
                )
            )
            record["query_position"] = query_position
            record["query_row_id"] = bytes(metric_labels.query_row_ids[query_position]).decode("ascii")
            per_query.append(record)
        summary = _mean_query_metrics(per_query)
        relative_cell = contract_path.parent.relative_to(root / "cells")
        result_path = evaluation_root / relative_cell / "metrics.json"
        result_body = {
            "schema": METRIC_SCHEMA,
            "status": "COMPLETE",
            "dataset": manifest["dataset"],
            "source_seal_sha256": manifest["source_seal_sha256"],
            "rank_manifest_sha256": manifest["rank_manifest_sha256"],
            "rank_contract_sha256": contract["rank_contract_sha256"],
            "bits": contract["binding"]["bits"],
            "direction": contract["binding"]["direction"],
            "rank_mode": contract["binding"]["rank_mode"],
            "primary_metric": "map_expected_ties",
            "cutoffs": sorted(set(int(value) for value in cutoffs)),
            "summary": summary,
            "per_query": per_query,
            "metric_labels_opened_after_rank_freeze": True,
            "code_inventory": production_code_inventory(),
        }
        result = {**result_body, "metric_result_sha256": sha256_json(result_body)}
        atomic_write_json(result_path, result)
        result_descriptors.append(
            {
                "path": result_path.relative_to(evaluation_root).as_posix(),
                "size": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
                "bits": contract["binding"]["bits"],
                "direction": contract["binding"]["direction"],
                "rank_mode": contract["binding"]["rank_mode"],
                "map_expected_ties": summary["map_expected_ties"],
            }
        )
    completion_body = {
        "schema": METRIC_SCHEMA,
        "status": "COMPLETE",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "rank_manifest_sha256": manifest["rank_manifest_sha256"],
        "metric_boundary": "load_metric_labels-after-rank_state_frozen",
        "results": sorted(result_descriptors, key=lambda value: value["path"]),
    }
    atomic_write_json(
        evaluation_root / "evaluation_complete.json",
        {**completion_body, "complete_sha256": sha256_json(completion_body)},
    )
    return evaluation_root


__all__ = [
    "MetricError",
    "evaluate_frozen_ranks",
    "expected_graded_ndcg",
    "expected_tie_metrics",
]
