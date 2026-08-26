"""Freeze the cross-dataset RZ-CSD collision-detail architecture.

This command consumes exactly three hash-verified indT-only detail-budget
sweeps.  It selects the smallest predeclared global cap whose primary deltas
are all nonnegative and whose graded JNDCG@50 deltas are all strictly positive
across every dataset, code width, and retrieval direction.  The resulting
record is written before training the full-indT detail experts or evaluating
the new architecture on the formal splits.
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

from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from rz_csd_clip512 import BITS
from tools.dev_rzcsd_detail_budget_sweep import DETAIL_BUDGETS


DATASETS = ("mirflickr", "nuswide", "mscoco")
PRIMARY_METRICS = ("map_expected_ties", "ndcg_at_50_expected_ties")
GRADED_METRIC = "jndcg_at_50_expected_ties"


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_sweep(path: Path, *, dataset: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("budget sweep must be a JSON object")
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    if result.get("result_sha256") != sha256_json(body):
        raise RuntimeError(f"{dataset} budget sweep hash mismatch")
    expected = {
        "schema": "raw_rebuilt_rzcsd_detail_budget_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": dataset,
        "formal_query_or_database_labels_opened": False,
        "configuration_frozen_for_formal_evaluation": False,
        "candidate_budget_grid": list(DETAIL_BUDGETS),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise RuntimeError(f"{dataset} budget sweep {field} mismatch")
    return result


def _summarize_cap(
    sweeps: Mapping[str, Mapping[str, Any]],
    cap: int,
) -> dict[str, Any]:
    primary: list[float] = []
    graded: list[float] = []
    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        sweep = sweeps[dataset]
        for bits in BITS:
            detail_bits = min(cap, bits)
            deltas = sweep["candidates"][str(bits)][str(detail_bits)]["deltas"]
            for direction in ("i2t", "t2i"):
                for metric in PRIMARY_METRICS:
                    value = float(deltas[direction][metric])
                    primary.append(value)
                    cells.append(
                        {
                            "dataset": dataset,
                            "bits": bits,
                            "detail_bits": detail_bits,
                            "direction": direction,
                            "metric": metric,
                            "delta": value,
                        }
                    )
                graded.append(float(deltas[direction][GRADED_METRIC]))
    primary_array = np.asarray(primary, dtype=np.float64)
    graded_array = np.asarray(graded, dtype=np.float64)
    return {
        "global_detail_cap": cap,
        "rule": f"min({cap}, primary_bits)",
        "primary_cells": len(primary),
        "graded_cells": len(graded),
        "negative_primary_cells": int(np.sum(primary_array < 0.0)),
        "nonpositive_graded_cells": int(np.sum(graded_array <= 0.0)),
        "minimum_primary_delta": float(primary_array.min()),
        "mean_primary_delta": float(primary_array.mean()),
        "maximum_primary_delta": float(primary_array.max()),
        "minimum_graded_jndcg_at_50_delta": float(graded_array.min()),
        "mean_graded_jndcg_at_50_delta": float(graded_array.mean()),
        "maximum_graded_jndcg_at_50_delta": float(graded_array.max()),
        "eligible": bool(
            np.all(primary_array >= 0.0) and np.all(graded_array > 0.0)
        ),
        "primary_cell_deltas": cells,
    }


def run(paths: Mapping[str, Path], output: Path) -> dict[str, Any]:
    if set(paths) != set(DATASETS):
        raise ValueError("exactly the three registered datasets are required")
    sweeps = {
        dataset: _load_sweep(paths[dataset], dataset=dataset)
        for dataset in DATASETS
    }
    summaries = [_summarize_cap(sweeps, cap) for cap in DETAIL_BUDGETS]
    eligible = [record for record in summaries if record["eligible"]]
    if not eligible:
        raise RuntimeError("no global detail budget satisfies the frozen gate")
    selected = min(eligible, key=lambda record: int(record["global_detail_cap"]))
    selected_cap = int(selected["global_detail_cap"])
    body = {
        "schema": "rz_csd_collision_detail_architecture_freeze_v2",
        "status": "FROZEN_BEFORE_CCDE_FULL_INDT_TRAINING_AND_FORMAL_EVALUATION",
        "frozen_at_local_date": "2026-08-24",
        "selection_boundary": {
            "datasets": list(DATASETS),
            "features": "self_extracted_clip512",
            "labels_consumed": (
                "indT fit labels for expert-bit ordering and disjoint indT "
                "development labels for predeclared budget comparison"
            ),
            "new_ccde_formal_query_or_database_labels_opened": False,
            "predecessor_fieldmoe_formal_aggregate_metrics_already_observed": True,
            "formal_exposure_disclosure": (
                "Earlier FieldMoE formal aggregate results existed before this "
                "v2 search.  No formal artifact is an input to the CCDE scripts, "
                "but the project-level history is not described as test-naive."
            ),
        },
        "parent_budget_sweeps": {
            dataset: {
                "path": _portable_path(paths[dataset]),
                "file_sha256": sha256_file(paths[dataset]),
                "result_sha256": sweeps[dataset]["result_sha256"],
                "source_seal_sha256": sweeps[dataset]["source_seal_sha256"],
                "fit_artifact_sha256": sweeps[dataset]["fit_artifact_sha256"],
                "split": sweeps[dataset]["split"],
                "split_hashes": sweeps[dataset]["split_hashes"],
            }
            for dataset in DATASETS
        },
        "predeclared_budget_candidates": summaries,
        "selection_rule": (
            "choose the smallest global cap in [4,8,16,32,64] for which all "
            "36 mAP/binary-NDCG development deltas are nonnegative and all 18 "
            "graded-JNDCG development deltas are strictly positive"
        ),
        "frozen_architecture": {
            "name": "RZ-CSD Collision-Conditioned Detail Expert (CCDE)",
            "primary_encoder": "exact compact RZ-CSD linear hash heads",
            "detail_encoder": (
                "compact RZ-CSD with shared projections and modality-specific "
                "BatchNorm running statistics and affine parameters"
            ),
            "training_schedule": "frozen 40-epoch warmup plus 5-epoch auxiliary curriculum",
            "detail_bit_order": (
                "fit-only descending product of paired sign agreement, global "
                "bit balance, and prevalence-weighted label separation"
            ),
            "global_detail_cap": selected_cap,
            "detail_bits_by_primary_width": {
                str(bits): min(selected_cap, bits) for bits in BITS
            },
            "ranking": (
                "lexicographic (primary Hamming, selected detail Hamming), "
                "implemented as primary_distance*(detail_bits+1)+detail_distance"
            ),
            "primary_shell_order_is_invariant": True,
            "development_gate_or_fallback_used_at_formal_inference": False,
        },
        "selected_development_summary": selected,
        "deployment_accounting": {
            "encoder_checkpoints": 2,
            "additional_database_bits": {
                str(bits): min(selected_cap, bits) for bits in BITS
            },
            "additional_bits_maximum": selected_cap,
            "comparison_must_disclose_secondary_code_storage": True,
        },
        "formal_application_contract": {
            "datasets": list(DATASETS),
            "same_architecture_and_budget_rule_for_every_dataset": True,
            "expert_bit_order_may_use_only_that_datasets_full_indT_labels": True,
            "no_formal_query_or_database_retuning": True,
            "formal_results_may_not_change_the_frozen_rule": True,
            "formal_failures_must_be_reported_without_fallback": True,
            "sota_claim_requires_completed_same_protocol_external_comparison": True,
        },
        "implementation_file_sha256": {
            path: sha256_file(PROJECT_ROOT / path)
            for path in (
                "raw_rebuilt_neural/hash_head_variants.py",
                "tools/dev_rzcsd_collision_detail_expert.py",
                "tools/dev_rzcsd_detail_budget_sweep.py",
                "tools/freeze_rzcsd_collision_detail.py",
            )
        },
        "supersession_note": (
            "This v2 freeze does not alter the historical v1 freeze.  The v1 "
            "NUS-selected semantic bridge failed cross-dataset transfer and was "
            "returned to development; CCDE was selected afterward using only "
            "the registered indT internal splits."
        ),
    }
    result = {**body, "freeze_sha256": sha256_json(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirflickr", type=Path, required=True)
    parser.add_argument("--nuswide", type=Path, required=True)
    parser.add_argument("--mscoco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        dataset: getattr(args, dataset).resolve(strict=True)
        for dataset in DATASETS
    }
    result = run(paths, args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "global_detail_cap": result["frozen_architecture"]["global_detail_cap"],
                "selected_development_summary": result["selected_development_summary"],
                "freeze_sha256": result["freeze_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
