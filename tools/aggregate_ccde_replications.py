#!/usr/bin/env python3
"""Verify and summarize repeated complete-gallery CCDE evaluations.

Every run is supplied explicitly as ``SEED DATASET PLAN METRICS``.  The tool
revalidates the frozen plan, completion receipt, all result files, and every
per-query receipt chain before reporting cross-seed sign consistency and
min/median/max deltas.  It intentionally does not emit mean +/- standard
deviation tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import atomic_write_json, load_json, sha256_file, sha256_json
from tools.formal_ccde_streaming_eval import (
    EVALUATION_SCHEMA,
    RESULT_SCHEMA,
    _resume_cell,
    _verify_plan,
)


AUDIT_SCHEMA = "raw_rebuilt_ccde_replication_audit_v1"
DATASETS = ("mirflickr", "nuswide", "mscoco")
METRICS = (
    ("mAP", "map_expected_ties", "primary"),
    ("NDCG@50", "binary_ndcg_at_50_expected_ties", "primary"),
    ("J-NDCG@50", "j_ndcg_at_50_expected_ties", "graded"),
)


class ReplicationAuditError(RuntimeError):
    """A repeated formal run is incomplete or no longer byte-identical."""


def _verify_completion(
    metrics_root: Path, plan: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[tuple[str, int], Mapping[str, Any]], int]:
    complete = load_json(metrics_root / "evaluation_complete.json")
    complete_body = {
        key: complete[key] for key in complete if key != "complete_sha256"
    }
    if (
        complete.get("schema") != EVALUATION_SCHEMA
        or complete.get("status") != "COMPLETE"
        or complete.get("storage_bounded_complete_gallery_evaluation") is not True
        or complete.get("formal_gate_or_fallback_used") is not False
        or complete.get("primary_shell_order_is_invariant") is not True
        or complete.get("rank_plan_sha256") != plan["rank_plan_sha256"]
        or complete.get("source_seal_sha256") != plan["source_seal_sha256"]
        or sha256_json(complete_body) != complete.get("complete_sha256")
    ):
        raise ReplicationAuditError("evaluation completion receipt changed")

    descriptors = complete.get("results")
    if not isinstance(descriptors, list):
        raise ReplicationAuditError("evaluation completion has no result inventory")
    expected_cells = {
        (str(direction), int(bits))
        for direction in plan["binding"]["config"]["directions"]
        for bits in plan["binding"]["config"]["bits"]
    }
    actual_cells = {
        (str(value.get("direction")), int(value.get("bits", -1)))
        for value in descriptors
    }
    if actual_cells != expected_cells or len(descriptors) != len(expected_cells):
        raise ReplicationAuditError("evaluation result inventory differs from frozen config")

    results: dict[tuple[str, int], Mapping[str, Any]] = {}
    partial_count = 0
    for descriptor in descriptors:
        cell = (str(descriptor["direction"]), int(descriptor["bits"]))
        result_path = metrics_root / str(descriptor.get("path", ""))
        if (
            not result_path.is_file()
            or result_path.stat().st_size != int(descriptor.get("size", -1))
            or sha256_file(result_path) != descriptor.get("sha256")
        ):
            raise ReplicationAuditError(f"result file changed for {cell}")
        result = load_json(result_path)
        result_body = {
            key: result[key] for key in result if key != "metric_result_sha256"
        }
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("status") != "COMPLETE"
            or result.get("dataset") != plan["dataset"]
            or result.get("direction") != cell[0]
            or int(result.get("bits", -1)) != cell[1]
            or result.get("rank_plan_sha256") != plan["rank_plan_sha256"]
            or result.get("source_seal_sha256") != plan["source_seal_sha256"]
            or result.get("formal_gate_or_fallback_used") is not False
            or result.get("primary_shell_order_is_invariant") is not True
            or sha256_json(result_body) != result.get("metric_result_sha256")
            or result.get("metric_result_sha256")
            != descriptor.get("metric_result_sha256")
        ):
            raise ReplicationAuditError(f"result content changed for {cell}")
        covered, chain, _primary, _ccde, partials = _resume_cell(
            metrics_root, plan, cell[0], cell[1]
        )
        if (
            covered != int(plan["runtime_identity"]["query_rows"])
            or chain != result.get("final_receipt_chain_sha256")
            or partials != result.get("per_query_receipts")
        ):
            raise ReplicationAuditError(f"per-query receipt chain changed for {cell}")
        partial_count += len(partials)
        results[cell] = result
    return complete, results, partial_count


def _verify_run(
    seed: int,
    dataset: str,
    plan_root: Path,
    metrics_root: Path,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ReplicationAuditError(f"unknown dataset {dataset!r}")
    plan_root = plan_root.expanduser().resolve(strict=True)
    metrics_root = metrics_root.expanduser().resolve(strict=True)
    plan = _verify_plan(plan_root)
    if plan.get("dataset") != dataset:
        raise ReplicationAuditError("run dataset differs from its frozen plan")
    complete, results, partial_count = _verify_completion(metrics_root, plan)
    cells = []
    for (direction, bits), result in sorted(results.items()):
        summaries = result.get("summaries", {})
        primary = summaries.get("primary_hamming", {})
        ccde = summaries.get("ccde_lexicographic", {})
        declared_delta = result.get("ccde_minus_primary", {})
        for metric_name, metric_key, family in METRICS:
            before = float(primary[metric_key])
            after = float(ccde[metric_key])
            delta = after - before
            if not np.isclose(
                delta,
                float(declared_delta[metric_key]),
                rtol=0.0,
                atol=5e-15,
            ):
                raise ReplicationAuditError(
                    f"declared delta cannot be recomputed for {dataset}/{direction}/{bits}/{metric_name}"
                )
            cells.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "direction": direction,
                    "bits": bits,
                    "metric": metric_name,
                    "metric_key": metric_key,
                    "family": family,
                    "primary": before,
                    "ccde": after,
                    "delta": delta,
                }
            )
    return {
        "seed": seed,
        "dataset": dataset,
        "plan_root": str(plan_root),
        "metrics_root": str(metrics_root),
        "rank_plan_sha256": plan["rank_plan_sha256"],
        "source_seal_sha256": plan["source_seal_sha256"],
        "evaluation_complete_sha256": complete["complete_sha256"],
        "partial_receipt_count": partial_count,
        "cells": cells,
    }


def _group_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["direction"]),
            int(row["bits"]),
            str(row["metric"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (dataset, direction, bits, metric), values in sorted(grouped.items()):
        deltas = np.asarray([float(value["delta"]) for value in values], dtype=np.float64)
        seeds = sorted(int(value["seed"]) for value in values)
        family = str(values[0]["family"])
        output.append(
            {
                "dataset": dataset,
                "direction": direction,
                "bits": bits,
                "metric": metric,
                "family": family,
                "seeds": seeds,
                "positive_seeds": int(np.count_nonzero(deltas > 0.0)),
                "seed_count": int(len(deltas)),
                "minimum_delta": float(deltas.min()),
                "median_delta": float(np.median(deltas)),
                "maximum_delta": float(deltas.max()),
            }
        )
    return output


def _dataset_metric_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        for metric_name, _metric_key, family in METRICS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["metric"] == metric_name
            ]
            deltas = np.asarray([float(row["delta"]) for row in selected], dtype=np.float64)
            if not len(deltas):
                raise ReplicationAuditError(f"missing replication rows for {dataset}/{metric_name}")
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric_name,
                    "family": family,
                    "positive_cells": int(np.count_nonzero(deltas > 0.0)),
                    "cell_count": int(len(deltas)),
                    "minimum_delta": float(deltas.min()),
                    "median_delta": float(np.median(deltas)),
                    "maximum_delta": float(deltas.max()),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tex_dataset(value: str) -> str:
    return {"mirflickr": "MIRFlickr", "nuswide": "NUS-WIDE", "mscoco": "MS COCO"}[value]


def _write_tex(path: Path, rows: Sequence[Mapping[str, Any]], seed_count: int) -> None:
    lines = [
        "% Auto-generated by tools/aggregate_ccde_replications.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Independent-seed sign stability. Deltas are percentage points; no mean $\\pm$ standard-deviation summary is used.}",
        "\\label{tab:ccde-seed-stability}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Metric & Positive & Min & Median & Max \\\\",
        "\\midrule",
    ]
    previous = None
    for row in rows:
        dataset = _tex_dataset(str(row["dataset"]))
        if previous is not None and dataset != previous:
            lines.append("\\addlinespace[1pt]")
        previous = dataset
        lines.append(
            f"{dataset} & {row['metric']} & {row['positive_cells']}/{row['cell_count']} & "
            f"{100.0 * float(row['minimum_delta']):+.3f} & "
            f"{100.0 * float(row['median_delta']):+.3f} & "
            f"{100.0 * float(row['maximum_delta']):+.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\vspace{{1pt}}\\par\\footnotesize Each row pools two directions, three code lengths, and {seed_count} seeds.",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(
    run_specs: Sequence[Sequence[str]], output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise ReplicationAuditError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    seen: set[tuple[int, str]] = set()
    runs = []
    for seed_value, dataset, plan_value, metrics_value in run_specs:
        seed = int(seed_value)
        key = (seed, dataset)
        if key in seen:
            raise ReplicationAuditError(f"duplicate run {key}")
        seen.add(key)
        runs.append(
            _verify_run(seed, dataset, Path(plan_value), Path(metrics_value))
        )
    seeds = sorted({int(run["seed"]) for run in runs})
    expected = {(seed, dataset) for seed in seeds for dataset in DATASETS}
    if seen != expected:
        raise ReplicationAuditError("every seed must contain all three datasets exactly once")
    all_cells = [cell for run in runs for cell in run["cells"]]
    grouped = _group_cells(all_cells)
    summary = _dataset_metric_summary(all_cells)
    primary = [row for row in all_cells if row["family"] == "primary"]
    graded = [row for row in all_cells if row["family"] == "graded"]
    audit_body: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "status": "VERIFIED",
        "seed_policy": "explicit independent training seeds; complete-gallery evaluation",
        "seeds": seeds,
        "seed_count": len(seeds),
        "datasets": list(DATASETS),
        "runs": [{key: value for key, value in run.items() if key != "cells"} for run in runs],
        "cells": all_cells,
        "cross_seed_cells": grouped,
        "dataset_metric_summary": summary,
        "overall_primary_positive_cells": int(sum(float(row["delta"]) > 0.0 for row in primary)),
        "overall_primary_cell_count": len(primary),
        "overall_graded_positive_cells": int(sum(float(row["delta"]) > 0.0 for row in graded)),
        "overall_graded_cell_count": len(graded),
        "mean_or_standard_deviation_reported": False,
    }
    audit = {**audit_body, "audit_sha256": sha256_json(audit_body)}
    atomic_write_json(output_root / "replication_audit.json", audit)
    _write_csv(output_root / "replication_cells.csv", all_cells)
    _write_csv(output_root / "replication_summary.csv", summary)
    _write_tex(output_root / "replication_table.tex", summary, len(seeds))
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        required=True,
        metavar=("SEED", "DATASET", "PLAN", "METRICS"),
        help="repeat once for each seed/dataset pair",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audit = aggregate(args.run, args.output.expanduser().resolve())
    print(json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
