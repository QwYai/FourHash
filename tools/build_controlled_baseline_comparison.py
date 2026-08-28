#!/usr/bin/env python3
"""Build a verified same-protocol neural baseline comparison across datasets."""

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


FORMAL_SCHEMAS = (
    "ccde_formal_seed20260822_verified_aggregate_v1",
    "semantic_bridge_formal_seed20260822_verified_aggregate_v1",
)
# Backward-compatible name used by synthetic callers that construct the
# original CCDE aggregate schema.
FORMAL_SCHEMA = FORMAL_SCHEMAS[0]
BASELINE_SCHEMA = "raw_rebuilt_baseline_streaming_aggregate_v1"
COMPARISON_SCHEMA = "raw_rebuilt_controlled_baseline_comparison_v3"
DATASETS = ("mirflickr", "nuswide", "mscoco")
DATASET_LABELS = {
    "mirflickr": "MIRFlickr-25K",
    "nuswide": "NUS-WIDE-TC21",
    "mscoco": "MS COCO",
}
BITS = (16, 32, 64)
DIRECTIONS = ("i2t", "t2i")
BASELINE_METHODS = ("ucch-f", "dcmh-f-seminit", "cirh-f", "raneh-f")
METHODS = (*BASELINE_METHODS, "primary", "shellguard")
DISPLAY = {
    "ucch-f": "UCCH-F",
    "dcmh-f-seminit": "DCMH-F-SemInit",
    "cirh-f": "CIRH-F",
    "raneh-f": "RANEH-F",
    "primary": "Primary",
    "shellguard": r"\method",
}
METRICS = ("map", "binary_ndcg_at_50", "j_ndcg_at_50")


class ControlledComparisonError(RuntimeError):
    """A source aggregate or comparison invariant failed verification."""


def _verified_json(
    path: Path, *, schema: str | Sequence[str]
) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: value[key] for key in value if key != "aggregate_sha256"}
    expected = (schema,) if isinstance(schema, str) else tuple(schema)
    if (
        value.get("schema") not in expected
        or value.get("status") != "VERIFIED"
        or value.get("aggregate_sha256") != sha256_json(body)
    ):
        raise ControlledComparisonError(f"aggregate verification failed: {resolved}")
    return value


def _baseline_index(
    values: Sequence[Mapping[str, Any]], *, dataset: str
) -> Mapping[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for value in values:
        if value.get("dataset") != dataset or value.get("seeds") != [20260822]:
            raise ControlledComparisonError(
                f"baseline aggregate is not the main {dataset} seed"
            )
        declared = tuple(str(method) for method in value.get("methods", ()))
        if (
            not declared
            or len(set(declared)) != len(declared)
            or any(method not in BASELINE_METHODS for method in declared)
        ):
            raise ControlledComparisonError("baseline aggregate method inventory changed")
        local: set[tuple[str, int, str]] = set()
        for row in value.get("rows", []):
            key = (
                str(row.get("method")),
                int(row.get("bits", -1)),
                str(row.get("direction")),
            )
            if (
                row.get("dataset") != dataset
                or key[0] not in declared
                or key in local
                or key in result
            ):
                raise ControlledComparisonError(
                    f"invalid or duplicate baseline row {key}"
                )
            local.add(key)
            result[key] = row
        expected_local = {
            (method, bits, direction)
            for method in declared
            for bits in BITS
            for direction in DIRECTIONS
        }
        if local != expected_local:
            raise ControlledComparisonError(
                f"incomplete aggregate-local baseline grid for {dataset}"
            )
    expected = {
        (method, bits, direction)
        for method in BASELINE_METHODS
        for bits in BITS
        for direction in DIRECTIONS
    }
    if set(result) != expected:
        raise ControlledComparisonError(f"incomplete baseline grid for {dataset}")
    return result


def _formal_index(
    value: Mapping[str, Any], *, dataset: str
) -> Mapping[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in value.get("rows", []):
        if row.get("dataset") != dataset:
            continue
        key = (int(row.get("bits", -1)), str(row.get("direction")))
        if key in result:
            raise ControlledComparisonError(f"duplicate formal row {dataset} {key}")
        result[key] = row
    expected = {(bits, direction) for bits in BITS for direction in DIRECTIONS}
    if set(result) != expected:
        raise ControlledComparisonError(f"incomplete formal grid for {dataset}")
    return result


def _latex_value(value: float, *, best: bool) -> str:
    rendered = f"{value:.3f}"
    return rf"\textbf{{{rendered}}}" if best else rendered


def _latex(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> str:
    index = {
        (
            str(row["dataset"]),
            str(row["method"]),
            int(row["bits"]),
            str(row["direction"]),
        ): row
        for row in rows
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Controlled neural comparison on identical self-extracted "
            r"CLIP-512 rows, frozen splits, complete galleries, and expected-tie "
            r"mAP (seed 20260822).  The -F rows are registered fixed-feature "
            r"implementations of the corresponding objectives.  ShellGuard uses "
            r"the fixed 16-bit neural-posterior MinHash bridge; published numbers "
            r"from incompatible protocols are not mixed into this table.}"
        ),
        r"\label{tab:controlled-baselines}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{I2T mAP} & \multicolumn{3}{c}{T2I mAP} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Dataset & Method & 16 bit & 32 bit & 64 bit & 16 bit & 32 bit & 64 bit \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(datasets):
        if dataset_index:
            lines.append(r"\midrule")
        best = {
            (direction, bits): max(
                float(index[(dataset, method, bits, direction)]["map"])
                for method in METHODS
            )
            for direction in DIRECTIONS
            for bits in BITS
        }
        for method_index, method in enumerate(METHODS):
            dataset_cell = (
                rf"\multirow{{{len(METHODS)}}}{{*}}{{{DATASET_LABELS[dataset]}}}"
                if method_index == 0
                else ""
            )
            values = []
            for direction in DIRECTIONS:
                for bits in BITS:
                    value = float(index[(dataset, method, bits, direction)]["map"])
                    values.append(
                        _latex_value(
                            value,
                            best=abs(value - best[(direction, bits)]) < 1e-15,
                        )
                    )
            lines.append(
                f"{dataset_cell} & {DISPLAY[method]} & "
                + " & ".join(values)
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_comparison(
    *,
    formal_path: Path,
    baseline_paths: Mapping[str, Path | Sequence[Path]],
    json_output: Path,
    csv_output: Path,
    tex_output: Path,
) -> Mapping[str, Any]:
    datasets = tuple(dataset for dataset in DATASETS if dataset in baseline_paths)
    if not datasets or set(baseline_paths) != set(datasets):
        raise ControlledComparisonError("baseline datasets are empty or unsupported")
    for output in (json_output, csv_output, tex_output):
        if output.exists():
            raise ControlledComparisonError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    formal = _verified_json(formal_path, schema=FORMAL_SCHEMAS)
    shellguard_prefix = (
        "shellguard"
        if formal["schema"] == "semantic_bridge_formal_seed20260822_verified_aggregate_v1"
        else "ccde"
    )
    rows: list[dict[str, Any]] = []
    baseline_sources: list[dict[str, Any]] = []
    for dataset in datasets:
        raw_paths = baseline_paths[dataset]
        paths = (raw_paths,) if isinstance(raw_paths, Path) else tuple(raw_paths)
        if not paths:
            raise ControlledComparisonError(f"no baseline aggregate for {dataset}")
        baselines: list[Mapping[str, Any]] = []
        for raw_path in paths:
            baseline_path = Path(raw_path).expanduser().resolve(strict=True)
            baseline = _verified_json(baseline_path, schema=BASELINE_SCHEMA)
            baselines.append(baseline)
            baseline_sources.append(
                {
                    "dataset": dataset,
                    "path": str(baseline_path),
                    "file_sha256": sha256_file(baseline_path),
                    "aggregate_sha256": baseline["aggregate_sha256"],
                    "methods": baseline["methods"],
                    "deep_verified_cells": baseline["deep_verified_cells"],
                }
            )
        baseline_rows = _baseline_index(baselines, dataset=dataset)
        formal_rows = _formal_index(formal, dataset=dataset)
        for method in METHODS:
            for bits in BITS:
                for direction in DIRECTIONS:
                    if method in BASELINE_METHODS:
                        source = baseline_rows[(method, bits, direction)]
                        values = {
                            metric: float(source[metric]) for metric in METRICS
                        }
                        metric_digest = source["metric_result_sha256"]
                    else:
                        source = formal_rows[(bits, direction)]
                        prefix = (
                            "primary" if method == "primary" else shellguard_prefix
                        )
                        values = {
                            "map": float(source[f"{prefix}_map"]),
                            "binary_ndcg_at_50": float(source[f"{prefix}_ndcg50"]),
                            "j_ndcg_at_50": float(source[f"{prefix}_jndcg50"]),
                        }
                        metric_digest = None
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "bits": bits,
                            "direction": direction,
                            **values,
                            "source_metric_result_sha256": metric_digest,
                        }
                    )

    margins: list[dict[str, Any]] = []
    for dataset in datasets:
        for bits in BITS:
            for direction in DIRECTIONS:
                candidates = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["bits"] == bits
                    and row["direction"] == direction
                ]
                shellguard = next(
                    row for row in candidates if row["method"] == "shellguard"
                )
                for metric in METRICS:
                    competitor = max(
                        float(row[metric])
                        for row in candidates
                        if row["method"] != "shellguard"
                    )
                    margin = float(shellguard[metric]) - competitor
                    margins.append(
                        {
                            "dataset": dataset,
                            "bits": bits,
                            "direction": direction,
                            "metric": metric,
                            "margin_over_best_competitor": margin,
                            "shellguard_is_strict_best": margin > 0.0,
                        }
                    )

    formal_resolved = formal_path.expanduser().resolve(strict=True)
    body: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "status": "VERIFIED",
        "datasets": list(datasets),
        "seed": 20260822,
        "formal_source": {
            "path": str(formal_resolved),
            "file_sha256": sha256_file(formal_resolved),
            "aggregate_sha256": formal["aggregate_sha256"],
            "schema": formal["schema"],
        },
        "baseline_sources": baseline_sources,
        "rows": rows,
        "margins": margins,
        "strict_best_cells": sum(
            int(value["shellguard_is_strict_best"]) for value in margins
        ),
        "comparison_cells": len(margins),
        "minimum_margin_over_best_competitor": min(
            float(value["margin_over_best_competitor"]) for value in margins
        ),
    }
    result = {**body, "comparison_sha256": sha256_json(body)}
    atomic_write_json(json_output, result)
    with csv_output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with tex_output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_latex(rows, datasets))
    return result


def _baseline_argument(value: str) -> tuple[str, Path]:
    dataset, separator, raw_path = value.partition("=")
    if not separator or dataset not in DATASETS or not raw_path:
        raise argparse.ArgumentTypeError("baseline must be DATASET=JSON")
    return dataset, Path(raw_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=_baseline_argument, action="append", required=True
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline_paths: dict[str, list[Path]] = {}
    for dataset, path in args.baseline:
        baseline_paths.setdefault(dataset, []).append(path)
    result = build_comparison(
        formal_path=args.formal,
        baseline_paths=baseline_paths,
        json_output=args.json_output,
        csv_output=args.csv_output,
        tex_output=args.tex_output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
