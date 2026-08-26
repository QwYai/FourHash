#!/usr/bin/env python3
"""Verify CCDE collision-structure reports and build paper-ready summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from rz_csd_clip512 import BITS
from tools.analyze_ccde_collision_structure import REPORT_SCHEMA


AGGREGATE_SCHEMA = "raw_rebuilt_ccde_collision_structure_aggregate_v1"
DATASETS = ("mirflickr", "nuswide", "mscoco")
DIRECTIONS = ("i2t", "t2i")
DISPLAY_NAMES = {
    "mirflickr": "MIRFlickr-25K",
    "nuswide": "NUS-WIDE-TC21",
    "mscoco": "MS COCO",
}


class CollisionAggregateError(RuntimeError):
    """A source report or aggregate invariant failed."""


def _load_verified(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    body = {key: report[key] for key in report if key != "report_sha256"}
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "POSTHOC_LABEL_FREE_DIAGNOSTIC"
        or report.get("labels_opened") is not False
        or report.get("selection_or_fallback_performed") is not False
        or sha256_json(body) != report.get("report_sha256")
    ):
        raise CollisionAggregateError(f"unverified collision report: {path}")
    return report


def _cell_index(report: Mapping[str, Any]) -> Mapping[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for cell in report.get("cells", []):
        key = (str(cell.get("direction")), int(cell.get("bits", -1)))
        if key in result:
            raise CollisionAggregateError(f"duplicate cell {key}")
        if (
            key[0] not in DIRECTIONS
            or key[1] not in BITS
            or cell.get("all_composite_ties_are_subsets_of_primary_shells") is not True
            or int(cell.get("queries_with_primary_collision", -1))
            != int(cell.get("query_rows", -2))
        ):
            raise CollisionAggregateError(f"invalid structural cell {key}")
        result[key] = cell
    expected = {(direction, int(bits)) for direction in DIRECTIONS for bits in BITS}
    if set(result) != expected:
        raise CollisionAggregateError("collision report does not contain all six cells")
    return result


def _format_count(value: float) -> str:
    doubled = round(float(value) * 2.0)
    if abs(float(value) * 2.0 - doubled) > 1e-8:
        return f"{float(value):,.1f}"
    if doubled % 2 == 0:
        return f"{doubled // 2:,d}"
    return f"{doubled / 2.0:,.1f}"


def _latex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Label-free collision-load diagnosis at rank 50.  "
            r"P-shell is the median size of the primary Hamming shell crossing "
            r"the cutoff; C-tie is the median residual composite tie inside that "
            r"shell.  Resolved is the mean per-query fraction distinguished by "
            r"the 16-bit detail expert.}"
        ),
        r"\label{tab:collision-structure}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{I2T} & \multicolumn{3}{c}{T2I} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Dataset & Bits & P-shell & C-tie & Resolved (\%) & P-shell & C-tie & Resolved (\%) \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        if index and row["dataset"] != rows[index - 1]["dataset"]:
            lines.append(r"\midrule")
        lines.append(
            "{} & {} & {} & {} & {:.1f} & {} & {} & {:.1f} \\\\".format(
                DISPLAY_NAMES[str(row["dataset"])],
                int(row["bits"]),
                _format_count(float(row["i2t_primary_median"])),
                _format_count(float(row["i2t_composite_median"])),
                float(row["i2t_resolved_percent"]),
                _format_count(float(row["t2i_primary_median"])),
                _format_count(float(row["t2i_composite_median"])),
                float(row["t2i_resolved_percent"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(
    *,
    inputs: Sequence[Path],
    json_output: Path,
    csv_output: Path,
    tex_output: Path,
) -> Mapping[str, Any]:
    for output in (json_output, csv_output, tex_output):
        if output.exists():
            raise CollisionAggregateError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Mapping[str, Any]] = {}
    sources: list[Mapping[str, Any]] = []
    for path in inputs:
        resolved = path.expanduser().resolve(strict=True)
        report = _load_verified(resolved)
        dataset = str(report.get("dataset"))
        if dataset in reports or dataset not in DATASETS:
            raise CollisionAggregateError(f"unexpected or duplicate dataset {dataset}")
        reports[dataset] = report
        sources.append(
            {
                "dataset": dataset,
                "path": str(resolved),
                "size": resolved.stat().st_size,
                "file_sha256": sha256_file(resolved),
                "report_sha256": report["report_sha256"],
                "rank_plan_sha256": report["rank_plan_sha256"],
                "formal_evaluation_complete_sha256": report[
                    "formal_evaluation_complete_sha256"
                ],
            }
        )
    if set(reports) != set(DATASETS):
        raise CollisionAggregateError("exactly one report per registered dataset is required")

    rows: list[dict[str, Any]] = []
    all_cells: list[Mapping[str, Any]] = []
    for dataset in DATASETS:
        cells = _cell_index(reports[dataset])
        for bits in BITS:
            row: dict[str, Any] = {"dataset": dataset, "bits": int(bits)}
            for direction in DIRECTIONS:
                cell = cells[(direction, int(bits))]
                all_cells.append(cell)
                row[f"{direction}_primary_median"] = cell[
                    "primary_boundary_shell_size"
                ]["median"]
                row[f"{direction}_composite_median"] = cell[
                    "composite_boundary_tie_size"
                ]["median"]
                row[f"{direction}_resolved_percent"] = 100.0 * float(
                    cell["distinguished_fraction_inside_primary_shell"]["mean"]
                )
            rows.append(row)

    resolved_values = [
        100.0 * float(cell["distinguished_fraction_inside_primary_shell"]["mean"])
        for cell in all_cells
    ]
    body: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "status": "VERIFIED",
        "source_reports": sources,
        "cell_count": len(all_cells),
        "all_primary_boundaries_are_collisions": all(
            int(cell["queries_with_primary_collision"]) == int(cell["query_rows"])
            for cell in all_cells
        ),
        "all_composite_ties_are_subsets_of_primary_shells": all(
            cell["all_composite_ties_are_subsets_of_primary_shells"] is True
            for cell in all_cells
        ),
        "min_mean_resolved_percent": min(resolved_values),
        "max_mean_resolved_percent": max(resolved_values),
        "rows": rows,
    }
    result = {**body, "aggregate_sha256": sha256_json(body)}
    atomic_write_json(json_output, result)

    with csv_output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with tex_output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_latex_table(rows))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = aggregate(
        inputs=args.input,
        json_output=args.json_output,
        csv_output=args.csv_output,
        tex_output=args.tex_output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
