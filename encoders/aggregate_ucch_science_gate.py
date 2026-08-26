#!/usr/bin/env python3
"""Aggregate completed UCCH-F oracle cells into one machine-readable gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_std(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if finite else None


def aggregate(
    result_root: Path,
    training_manifest: Path,
    output: Path,
    *,
    expected_ratio: float,
    expected_assignment_seeds: list[int],
    expected_global_fit_seed: int,
    headroom_threshold: float,
) -> dict[str, Any]:
    training = json.loads(training_manifest.read_text(encoding="utf-8"))
    standard = training["summary"]
    cells: list[dict[str, Any]] = []
    for assignment_seed in expected_assignment_seeds:
        ratio_tag = int(round(expected_ratio * 100))
        cell_dir = result_root / "r{:03d}-a{}".format(ratio_tag, assignment_seed)
        summary_path = cell_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        resolved = payload["metadata"]["resolved_seeds"]
        if int(resolved["assignment"]) != assignment_seed:
            raise ValueError("assignment seed mismatch in {}".format(summary_path))
        if int(resolved["global_fit"]) != expected_global_fit_seed:
            raise ValueError("global-fit seed mismatch in {}".format(summary_path))
        if list(resolved["ties"]) != list(range(20)):
            raise ValueError("formal tie seeds 0..19 were not used")
        selected = [row for row in payload["summaries"] if row["tie_seed"] == "all"]
        if {row["query_modality"] for row in selected} != {"image", "text"}:
            raise ValueError("missing all-tie-seed modality summary")
        for row in selected:
            binary_headroom = float(row["mean_exact_interleaving_oracle_ndcg"]) - float(
                row["mean_raw_ndcg"]
            )
            graded_headroom = float(
                row["mean_exact_interleaving_oracle_graded_ndcg"]
            ) - float(row["mean_raw_graded_ndcg"])
            cells.append(
                {
                    "assignment_seed": assignment_seed,
                    "text_ratio": expected_ratio,
                    "query_modality": row["query_modality"],
                    "num_unique_queries": int(row["num_unique_queries"]),
                    "num_tie_averaged_rows": int(row["num_rows"]),
                    "binary": {
                        "raw": float(row["mean_raw_ndcg"]),
                        "cell_local_global_invalid_for_frozen_comparison": float(
                            row["mean_global_affine_ndcg"]
                        ),
                        "per_query_affine_oracle": float(
                            row["mean_per_query_affine_oracle_ndcg"]
                        ),
                        "monotone_radius_lut_oracle": float(
                            row["mean_monotone_radius_lut_oracle_ndcg"]
                        ),
                        "exact_interleaving_oracle": float(
                            row["mean_exact_interleaving_oracle_ndcg"]
                        ),
                        "exact_minus_raw": binary_headroom,
                        "cell_local_global_capture_invalid_for_frozen_comparison": float(
                            row["aggregate_global_headroom_capture"]
                        ),
                        "per_query_affine_capture": float(
                            row["aggregate_per_query_affine_headroom_capture"]
                        ),
                        "lut_capture": float(
                            row["aggregate_monotone_radius_lut_headroom_capture"]
                        ),
                    },
                    "graded": {
                        "raw": float(row["mean_raw_graded_ndcg"]),
                        "cell_local_global_invalid_for_frozen_comparison": float(
                            row["mean_global_affine_graded_ndcg"]
                        ),
                        "per_query_affine_oracle": float(
                            row["mean_per_query_affine_oracle_graded_ndcg"]
                        ),
                        "monotone_radius_lut_oracle": float(
                            row["mean_monotone_radius_lut_oracle_graded_ndcg"]
                        ),
                        "exact_interleaving_oracle": float(
                            row["mean_exact_interleaving_oracle_graded_ndcg"]
                        ),
                        "exact_minus_raw": graded_headroom,
                        "cell_local_global_capture_invalid_for_frozen_comparison": float(
                            row["aggregate_global_graded_headroom_capture"]
                        ),
                        "per_query_affine_capture": float(
                            row["aggregate_per_query_affine_graded_headroom_capture"]
                        ),
                        "lut_capture": float(
                            row["aggregate_monotone_radius_lut_graded_headroom_capture"]
                        ),
                    },
                    "cell_local_fraction_global_worse_than_raw_binary_invalid_for_frozen_comparison": float(
                        row["fraction_global_affine_worse_than_raw"]
                    ),
                    "summary_path": str(summary_path.resolve()),
                }
            )

    by_modality: dict[str, Any] = {}
    for modality in ("image", "text"):
        subset = [cell for cell in cells if cell["query_modality"] == modality]
        record: dict[str, Any] = {"cells": len(subset)}
        for gain in ("binary", "graded"):
            headroom = [float(cell[gain]["exact_minus_raw"]) for cell in subset]
            affine_capture = [
                float(cell[gain]["per_query_affine_capture"]) for cell in subset
            ]
            lut_extra = [
                float(cell[gain]["lut_capture"])
                - float(cell[gain]["per_query_affine_capture"])
                for cell in subset
            ]
            record[gain] = {
                "headroom_mean": _finite_mean(headroom),
                "headroom_std": _finite_std(headroom),
                "headroom_min": float(min(headroom)),
                "cells_at_or_above_threshold": int(
                    sum(value >= headroom_threshold for value in headroom)
                ),
                "per_query_affine_capture_mean": _finite_mean(affine_capture),
                "per_query_affine_capture_min": float(min(affine_capture)),
                "lut_minus_affine_capture_mean": _finite_mean(lut_extra),
            }
        by_modality[modality] = record

    gate = {
        "scope": "partial common-MIR64 science gate at text_ratio=0.5 only",
        "not_a_full_ratio_gate": True,
        "requires_parent_aggregation_with_ratios_0.1_and_0.9": True,
        "three_assignment_seeds_complete": len(cells) == 6,
        "both_query_modalities_have_binary_and_graded_headroom_ge_threshold_in_at_least_two_of_three_seeds": all(
            by_modality[modality][gain]["cells_at_or_above_threshold"] >= 2
            for modality in ("image", "text")
            for gain in ("binary", "graded")
        ),
        "frozen_global_comparison_evaluable": False,
    }
    payload = {
        "format_version": 1,
        "encoder": "UCCH-F",
        "frozen_artifact_sha256": training["files"][
            "ucch_f_mirflickr_64_seed20260805.npz"
        ]["sha256"],
        "training_source_sha256": training["source_sha256"],
        "standard_retrieval_anchor": {
            "i2t_map": standard["i2t"]["map"],
            "t2i_map": standard["t2i"]["map"],
            "i2t_map_gain_over_random": standard["i2t"]["map_gain_over_random"],
            "t2i_map_gain_over_random": standard["t2i"]["map_gain_over_random"],
            "heldout_gate_passed": standard["heldout_gate_passed"],
        },
        "protocol": {
            "text_ratio": expected_ratio,
            "assignment_seeds": expected_assignment_seeds,
            "query_sample_seed": 20260805,
            "global_fit_seed": expected_global_fit_seed,
            "queries_per_modality": 200,
            "tie_seeds": list(range(20)),
            "headroom_threshold": headroom_threshold,
        },
        "cells": cells,
        "aggregate_by_query_modality": by_modality,
        "partial_gate": gate,
        "global_comparison_status": {
            "status": "INVALID_FOR_FROZEN_GLOBAL_COMPARISON",
            "reason": (
                "resolve_global_affine receives each test cell's text_ratio and "
                "assignment_seed; equal RNG seeds do not freeze one calibrator "
                "across cells"
            ),
            "allowed_use": "cell-local diagnostic only; excluded from every gate",
        },
        "interpretation_boundary": (
            "Strong standard mAP plus positive mixed-gallery exact-minus-raw "
            "headroom establishes merge comparability error at this ratio; a "
            "full standard-to-mixed robustness claim requires the other ratios."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-ratio", type=float, default=0.5)
    parser.add_argument(
        "--assignment-seeds", default="20260805,20260806,20260807"
    )
    parser.add_argument("--headroom-threshold", type=float, default=0.01)
    parser.add_argument("--global-fit-seed", type=int, default=20262808)
    args = parser.parse_args()
    seeds = [int(value) for value in args.assignment_seeds.split(",")]
    payload = aggregate(
        args.result_root,
        args.training_manifest,
        args.output,
        expected_ratio=args.text_ratio,
        expected_assignment_seeds=seeds,
        expected_global_fit_seed=args.global_fit_seed,
        headroom_threshold=args.headroom_threshold,
    )
    print(json.dumps(payload["partial_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
