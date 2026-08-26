"""Training-only budget ablation for the neural semantic bridge.

The program never accepts a formal runtime or a metric-label path.  It opens a
verified ``indT`` fit artifact, deterministically splits its identities into a
calibration/database partition and a held-out development-query partition, and
evaluates one-bit MinHash budgets inside frozen primary Hamming shells.  The
semantic threshold is selected from calibration identities only.

This is a development experiment: it supports architecture and storage-budget
selection, but it is not a substitute for the separately frozen formal test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_training import load_detail_checkpoint
from raw_rebuilt_neural.fit_artifact import open_fit_artifact
from raw_rebuilt_neural.semantic_bridge import (
    SemanticBridgeConfig,
    build_one_bit_minhash_map,
    calibrate_training_threshold,
    encode_mean_posterior,
    encode_semantic_bridge,
)
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from raw_rebuilt_streaming.metrics import (
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
    mean_query_metrics,
)
from tools.formal_semantic_bridge_streaming_eval import (
    BITS,
    _close_memmap,
    _verify_cache,
    _verify_plan,
)


DIRECTIONS = ("i2t", "t2i")
METRICS = (
    "map_expected_ties",
    "binary_ndcg_at_50_expected_ties",
    "j_ndcg_at_50_expected_ties",
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not values or values[0] < 1:
        raise ValueError("budgets must be positive integers")
    return values


def _parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in raw.split(",") if item.strip()}))
    if not values or not all(0.0 < value < 1.0 for value in values):
        raise ValueError("thresholds must lie strictly inside (0,1)")
    return values


def _hamming(query: np.ndarray, database: np.ndarray, bits: int) -> np.ndarray:
    q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32))
    d = torch.from_numpy(np.ascontiguousarray(database, dtype=np.float32))
    value = torch.round((bits - q @ d.T) * 0.5).to(torch.int16).numpy()
    if value.shape != (len(query), len(database)) or np.any(value < 0) or np.any(value > bits):
        raise AssertionError("invalid Hamming-distance matrix")
    return np.ascontiguousarray(value)


def _jaccard_gain(query_labels: np.ndarray, database_labels: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(np.ascontiguousarray(query_labels, dtype=np.float32))
    d = torch.from_numpy(np.ascontiguousarray(database_labels, dtype=np.float32))
    intersection = q @ d.T
    union = q.sum(dim=1)[:, None] + d.sum(dim=1)[None, :] - intersection
    result = (intersection / union.clamp_min(1.0)).numpy().astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise AssertionError("invalid development Jaccard gains")
    return np.ascontiguousarray(result)


def _summarize(
    distances: np.ndarray,
    gains: np.ndarray,
    *,
    bits: int,
    levels: int,
    prefixes: Any,
) -> dict[str, Any]:
    records = []
    for position in range(len(distances)):
        row_gain = np.ascontiguousarray(gains[position], dtype=np.float64)
        records.append(
            expected_tie_metrics_from_distances(
                row_gain > 0.0,
                np.ascontiguousarray(distances[position]),
                bits=bits,
                graded_gains=row_gain,
                cutoffs=(50, 100, 1000),
                prefixes=prefixes,
                distance_levels=levels,
            )
        )
    return mean_query_metrics(records)


def run_ablation(
    semantic_plan_root: Path,
    fit_artifact_root: Path,
    detail_checkpoint: Path,
    architecture_freeze: Path,
    output_path: Path,
    *,
    budgets: Sequence[int],
    thresholds: Sequence[float],
    development_fraction: float,
    query_limit: int,
    inference_batch_size: int,
    seed: int,
    device: str,
) -> Mapping[str, Any]:
    if not 0.05 <= development_fraction <= 0.5:
        raise ValueError("development_fraction must lie in [0.05,0.5]")
    if query_limit < 1 or inference_batch_size < 1:
        raise ValueError("query_limit and inference_batch_size must be positive")
    budget_values = tuple(int(value) for value in budgets)
    if not budget_values or min(budget_values) < 1:
        raise ValueError("budgets must be nonempty and positive")

    plan, plan_root = _verify_plan(semantic_plan_root)
    arrays = _verify_cache(plan_root, plan)
    fit = open_fit_artifact(fit_artifact_root)
    try:
        identity = plan["runtime_identity"]
        if (
            fit.dataset != identity["dataset"]
            or fit.source_seal_sha256 != identity["source_seal_sha256"]
            or fit.label_dim != int(identity["label_dim"])
        ):
            raise ValueError("fit artifact and frozen semantic plan differ")
        canonical = np.asarray(fit.canonical_indices, dtype=np.int64)
        if np.any(canonical < 0) or np.any(canonical >= int(identity["rows"])):
            raise ValueError("fit canonical indices lie outside the frozen code cache")

        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(canonical))
        development_rows = max(1, int(round(len(canonical) * development_fraction)))
        development_local = permutation[:development_rows]
        calibration_local = np.sort(permutation[development_rows:])
        if query_limit < len(development_local):
            development_local = development_local[:query_limit]
        development_local = np.sort(development_local)
        if len(calibration_local) < 2 or len(development_local) < 1:
            raise ValueError("development split is empty")

        resolved = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        loaded = load_detail_checkpoint(
            detail_checkpoint,
            architecture_freeze,
            device=resolved,
            expected_source_seal_sha256=fit.source_seal_sha256,
            require_current_code=False,
        )
        image_posterior = encode_mean_posterior(
            loaded.model,
            fit.image,
            modality="image",
            device=resolved,
            batch_size=inference_batch_size,
        )
        text_posterior = encode_mean_posterior(
            loaded.model,
            fit.text,
            modality="text",
            device=resolved,
            batch_size=inference_batch_size,
        )
        threshold, calibration = calibrate_training_threshold(
            image_posterior[calibration_local],
            text_posterior[calibration_local],
            np.ascontiguousarray(fit.labels[calibration_local], dtype=np.uint8),
            thresholds,
        )

        semantic_codes: dict[int, dict[str, np.ndarray]] = {}
        for detail_bits in budget_values:
            mapping = build_one_bit_minhash_map(
                fit.label_dim, bits=detail_bits, seed=seed
            )
            semantic_codes[detail_bits] = {
                "image": encode_semantic_bridge(
                    image_posterior, threshold=threshold, mapping=mapping
                ),
                "text": encode_semantic_bridge(
                    text_posterior, threshold=threshold, mapping=mapping
                ),
            }
        del image_posterior, text_posterior

        query_canonical = canonical[development_local]
        database_canonical = canonical[calibration_local]
        query_labels = np.ascontiguousarray(fit.labels[development_local], dtype=np.uint8)
        database_labels = np.ascontiguousarray(fit.labels[calibration_local], dtype=np.uint8)
        gains = _jaccard_gain(query_labels, database_labels)
        prefixes = build_metric_prefixes(len(database_labels), (50, 100, 1000))

        primary_summaries: dict[tuple[int, str], dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for bits in BITS:
            for direction in DIRECTIONS:
                query_modality, database_modality = (
                    ("image", "text") if direction == "i2t" else ("text", "image")
                )
                primary_query = arrays[f"primary_{query_modality}_codes_{bits}"][
                    query_canonical
                ]
                primary_database = arrays[f"primary_{database_modality}_codes_{bits}"][
                    database_canonical
                ]
                primary_distance = _hamming(primary_query, primary_database, bits)
                primary_metrics = _summarize(
                    primary_distance,
                    gains,
                    bits=bits,
                    levels=bits + 1,
                    prefixes=prefixes,
                )
                primary_summaries[(bits, direction)] = primary_metrics
                for detail_bits in budget_values:
                    semantic_query = semantic_codes[detail_bits][query_modality][
                        development_local
                    ]
                    semantic_database = semantic_codes[detail_bits][database_modality][
                        calibration_local
                    ]
                    semantic_distance = _hamming(
                        semantic_query, semantic_database, detail_bits
                    )
                    composite = (
                        primary_distance.astype(np.uint32) * np.uint32(detail_bits + 1)
                        + semantic_distance.astype(np.uint32)
                    )
                    if not np.array_equal(
                        composite // np.uint32(detail_bits + 1),
                        primary_distance.astype(np.uint32),
                    ):
                        raise AssertionError("budget ablation crossed a primary shell")
                    bridge_metrics = _summarize(
                        composite,
                        gains,
                        bits=bits,
                        levels=(bits + 1) * (detail_bits + 1),
                        prefixes=prefixes,
                    )
                    rows.append(
                        {
                            "bits": bits,
                            "direction": direction,
                            "detail_bits": detail_bits,
                            "primary": primary_metrics,
                            "bridge": bridge_metrics,
                            "delta": {
                                metric: float(bridge_metrics[metric])
                                - float(primary_metrics[metric])
                                for metric in METRICS
                            },
                        }
                    )

        budget_summary = []
        for detail_bits in budget_values:
            selected = [row for row in rows if row["detail_bits"] == detail_bits]
            primary_delta = [
                float(row["delta"][metric])
                for row in selected
                for metric in METRICS[:2]
            ]
            graded_delta = [float(row["delta"][METRICS[2]]) for row in selected]
            budget_summary.append(
                {
                    "detail_bits": detail_bits,
                    "mean_primary_delta": float(np.mean(primary_delta, dtype=np.float64)),
                    "minimum_primary_delta": float(min(primary_delta)),
                    "negative_primary_cells": int(sum(value < 0.0 for value in primary_delta)),
                    "mean_graded_delta": float(np.mean(graded_delta, dtype=np.float64)),
                    "minimum_graded_delta": float(min(graded_delta)),
                    "nonpositive_graded_cells": int(sum(value <= 0.0 for value in graded_delta)),
                }
            )

        body: dict[str, Any] = {
            "schema": "shellguard_semantic_bridge_training_only_budget_ablation_v1",
            "status": "TRAINING_ONLY_DEVELOPMENT_COMPLETE",
            "dataset": fit.dataset,
            "scientific_boundary": (
                "Only the verified indT fit artifact is accepted. Formal query/database "
                "features and labels are neither accepted nor opened by this program."
            ),
            "semantic_plan_rank_sha256": plan["rank_plan_sha256"],
            "fit_artifact_sha256": fit.fit_artifact_sha256,
            "detail_checkpoint_sha256": loaded.checkpoint_sha256,
            "architecture_freeze_file_sha256": sha256_file(
                Path(architecture_freeze).expanduser().resolve(strict=True)
            ),
            "seed": seed,
            "fit_rows": len(canonical),
            "calibration_database_rows": len(calibration_local),
            "held_out_development_rows": development_rows,
            "evaluated_development_queries": len(development_local),
            "development_fraction": development_fraction,
            "threshold_candidates": list(thresholds),
            "selected_threshold": threshold,
            "threshold_calibration": list(calibration),
            "budgets": list(budget_values),
            "split_policy": "seeded permutation; held-out prefix is query, remainder is calibration/database",
            "database_and_query_identities_are_disjoint": True,
            "primary_shell_order_verified_for_every_distance_matrix": True,
            "rows": rows,
            "budget_summary": budget_summary,
        }
        result = {**body, "ablation_sha256": sha256_json(body)}
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, result)
        return result
    finally:
        fit.close()
        for value in arrays.values():
            _close_memmap(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-plan", type=Path, required=True)
    parser.add_argument("--fit-artifact", type=Path, required=True)
    parser.add_argument("--detail-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", default="4,8,16,32,64")
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--development-fraction", type=float, default=0.20)
    parser.add_argument("--query-limit", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = run_ablation(
        args.semantic_plan,
        args.fit_artifact,
        args.detail_checkpoint,
        args.architecture_freeze,
        args.output,
        budgets=_parse_ints(args.budgets),
        thresholds=_parse_floats(args.thresholds),
        development_fraction=args.development_fraction,
        query_limit=args.query_limit,
        inference_batch_size=args.inference_batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "selected_threshold": result["selected_threshold"],
                "evaluated_development_queries": result[
                    "evaluated_development_queries"
                ],
                "ablation_sha256": result["ablation_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
