#!/usr/bin/env python3
"""Verify and aggregate the frozen seed-20260822 CCDE formal evaluation.

The aggregator is deliberately stricter than a table-building script.  It
replays every JSON receipt hash and every per-cell receipt chain, rejects any
unreferenced file, recomputes all reported deltas, and only then emits compact
CSV/JSON/LaTeX summaries.  Raw formal artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASETS = ("mirflickr", "nuswide", "mscoco")
DATASET_LABELS = {
    "mirflickr": "MIRFlickr-25K",
    "nuswide": "NUS-WIDE-TC21",
    "mscoco": "MS COCO",
}
BITS = (16, 32, 64)
DIRECTIONS = ("i2t", "t2i")
PINNED_COMPLETE_RECEIPTS = {
    "mirflickr": "b2bc81f1345a9c310e16d5f8b106f07c19822e3716fa028f26006fbcb43c50ef",
    "nuswide": "7e99888a32d4f240c62b0307c41cbf3d13a0c580953f6b8df9726a3c878c0b39",
    "mscoco": "c90a57d9a0914830f7bdc4e4d19a94e584b352d90cb187093b84e098b02fbfdf",
}
PINNED_SOURCE_DIRECTORY = {
    "schema": "canonical_recursive_directory_digest_v1",
    "files": 1008,
    "bytes": 202302277,
    "sha256": "994f88ed817a5f889b6070c24c52ad5c7989cbe0f5bd24dd46bfb1b23aa353e5",
}
METRICS = {
    "map": "map_expected_ties",
    "ndcg50": "binary_ndcg_at_50_expected_ties",
    "jndcg50": "j_ndcg_at_50_expected_ties",
}
EXPECTED_EVALUATION_SCHEMA = "raw_rebuilt_ccde_storage_bounded_evaluation_v1"
EXPECTED_RESULT_SCHEMA = "raw_rebuilt_ccde_storage_bounded_metric_v1"
EXPECTED_PARTIAL_SCHEMA = "raw_rebuilt_ccde_storage_bounded_partial_v1"


class AuditError(RuntimeError):
    """Raised when a frozen artifact or receipt has changed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        size = path.stat().st_size
        record = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size": size,
        }
        digest.update(canonical_json_bytes(record))
        digest.update(b"\n")
        files += 1
        total_bytes += size
    return {
        "schema": "canonical_recursive_directory_digest_v1",
        "files": files,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read canonical JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def require_close(observed: Any, expected: Any, field: str) -> None:
    require(isinstance(observed, (int, float)), f"{field} is not numeric")
    require(isinstance(expected, (int, float)), f"{field} reference is not numeric")
    require(math.isfinite(float(observed)), f"{field} is not finite")
    require(
        math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=2.0e-15),
        f"{field} changed: {observed!r} != {expected!r}",
    )


def safe_child(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative, "artifact path is empty")
    candidate = (root / relative).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    require(
        os.path.commonpath((str(resolved_root), str(candidate))) == str(resolved_root),
        f"artifact path escapes the evaluation root: {relative}",
    )
    require(candidate.is_file(), f"artifact is not a file: {candidate}")
    return candidate


def verify_descriptor(root: Path, descriptor: Mapping[str, Any]) -> Path:
    path = safe_child(root, descriptor.get("path"))
    require(path.stat().st_size == descriptor.get("size"), f"size changed: {path}")
    require(sha256_file(path) == descriptor.get("sha256"), f"SHA-256 changed: {path}")
    return path


def verify_partial_chain(
    evaluation_root: Path,
    result: Mapping[str, Any],
    referenced: set[Path],
) -> tuple[int, int]:
    descriptors = result.get("per_query_receipts")
    require(isinstance(descriptors, list) and descriptors, "partial receipt list is empty")
    expected_start = 0
    chain = "0" * 64
    records = 0
    for descriptor in descriptors:
        require(isinstance(descriptor, dict), "partial descriptor is not an object")
        path = verify_descriptor(evaluation_root, descriptor)
        require(path not in referenced, f"partial receipt is referenced twice: {path}")
        referenced.add(path)
        receipt = load_json(path)
        require(receipt.get("schema") == EXPECTED_PARTIAL_SCHEMA, f"partial schema changed: {path}")
        require(receipt.get("status") == "COMMITTED", f"partial is not committed: {path}")
        require(receipt.get("rank_plan_sha256") == result.get("rank_plan_sha256"), f"partial plan changed: {path}")
        require(receipt.get("direction") == result.get("direction"), f"partial direction changed: {path}")
        require(receipt.get("bits") == result.get("bits"), f"partial bit width changed: {path}")
        require(receipt.get("start") == expected_start, f"partial coverage is discontinuous: {path}")
        end = receipt.get("end")
        require(type(end) is int and end > expected_start, f"partial end is invalid: {path}")
        primary = receipt.get("primary_records")
        ccde = receipt.get("ccde_records")
        require(isinstance(primary, list) and isinstance(ccde, list), f"partial records missing: {path}")
        expected_positions = list(range(expected_start, end))
        require([item.get("query_position") for item in primary] == expected_positions, f"primary query coverage changed: {path}")
        require([item.get("query_position") for item in ccde] == expected_positions, f"CCDE query coverage changed: {path}")
        body = {
            key: receipt[key]
            for key in receipt
            if key not in {"receipt_sha256", "chain_sha256"}
        }
        receipt_sha = sha256_json(body)
        require(receipt.get("receipt_sha256") == receipt_sha, f"partial receipt hash changed: {path}")
        require(descriptor.get("receipt_sha256") == receipt_sha, f"partial descriptor hash changed: {path}")
        chain = sha256_json(
            {"previous_chain_sha256": chain, "receipt_sha256": receipt_sha}
        )
        require(receipt.get("chain_sha256") == chain, f"partial chain changed: {path}")
        records += end - expected_start
        expected_start = end
    require(result.get("final_receipt_chain_sha256") == chain, "final receipt chain changed")
    return len(descriptors), records


def verify_result(
    evaluation_root: Path,
    descriptor: Mapping[str, Any],
    dataset: str,
    referenced: set[Path],
) -> tuple[dict[str, Any], int, int]:
    path = verify_descriptor(evaluation_root, descriptor)
    require(path not in referenced, f"metric result is referenced twice: {path}")
    referenced.add(path)
    result = load_json(path)
    body = {key: result[key] for key in result if key != "metric_result_sha256"}
    require(result.get("metric_result_sha256") == sha256_json(body), f"metric receipt hash changed: {path}")
    require(result.get("metric_result_sha256") == descriptor.get("metric_result_sha256"), f"metric descriptor receipt changed: {path}")
    require(result.get("schema") == EXPECTED_RESULT_SCHEMA, f"metric schema changed: {path}")
    require(result.get("status") == "COMPLETE", f"metric is incomplete: {path}")
    require(result.get("dataset") == dataset, f"metric dataset changed: {path}")
    require(result.get("direction") == descriptor.get("direction"), f"metric direction changed: {path}")
    require(result.get("bits") == descriptor.get("bits"), f"metric bits changed: {path}")
    require(result.get("detail_bits") == min(16, int(result["bits"])), f"detail-bit cap changed: {path}")
    require(result.get("formal_gate_or_fallback_used") is False, f"formal fallback was used: {path}")
    require(result.get("primary_shell_order_is_invariant") is True, f"shell invariance is absent: {path}")
    require(result.get("metric_labels_opened_after_verified_frozen_codes") is True, f"label-isolation receipt changed: {path}")
    summaries = result.get("summaries")
    require(isinstance(summaries, dict), f"metric summaries missing: {path}")
    primary = summaries.get("primary_hamming")
    ccde = summaries.get("ccde_lexicographic")
    delta = result.get("ccde_minus_primary")
    require(isinstance(primary, dict) and isinstance(ccde, dict) and isinstance(delta, dict), f"metric methods missing: {path}")
    for key in set(primary).intersection(ccde):
        if key in {"queries", "queries_with_relevant", "zero_relevant_policy"}:
            continue
        if isinstance(primary[key], (int, float)) and isinstance(ccde[key], (int, float)):
            require_close(delta.get(key), float(ccde[key]) - float(primary[key]), f"{path}:{key}:delta")
    require_close(descriptor.get("map_delta"), delta[METRICS["map"]], f"{path}:descriptor-map")
    require_close(descriptor.get("binary_ndcg_headline_delta"), delta[METRICS["ndcg50"]], f"{path}:descriptor-ndcg50")
    require_close(descriptor.get("j_ndcg_headline_delta"), delta[METRICS["jndcg50"]], f"{path}:descriptor-jndcg50")
    partial_count, query_records = verify_partial_chain(evaluation_root, result, referenced)
    row: dict[str, Any] = {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "direction": result["direction"],
        "bits": int(result["bits"]),
        "detail_bits": int(result["detail_bits"]),
        "queries": int(primary["queries"]),
    }
    require(row["queries"] == query_records, f"query count differs from receipt coverage: {path}")
    for short, key in METRICS.items():
        row[f"primary_{short}"] = float(primary[key])
        row[f"ccde_{short}"] = float(ccde[key])
        row[f"delta_{short}"] = float(delta[key])
    return row, partial_count, query_records


def verify_dataset(root: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path]]:
    candidates = list((root / dataset / "metrics").glob("metrics-*/evaluation_complete.json"))
    require(len(candidates) == 1, f"expected exactly one completed metric root for {dataset}")
    complete_path = candidates[0].resolve(strict=True)
    evaluation_root = complete_path.parent
    complete = load_json(complete_path)
    body = {key: complete[key] for key in complete if key != "complete_sha256"}
    require(complete.get("complete_sha256") == sha256_json(body), f"completion receipt hash changed: {dataset}")
    require(complete.get("complete_sha256") == PINNED_COMPLETE_RECEIPTS[dataset], f"completion receipt is not the frozen seed-20260822 receipt: {dataset}")
    require(complete.get("schema") == EXPECTED_EVALUATION_SCHEMA, f"evaluation schema changed: {dataset}")
    require(complete.get("status") == "COMPLETE", f"evaluation is incomplete: {dataset}")
    require(complete.get("dataset") == dataset, f"completion dataset changed: {dataset}")
    require(complete.get("formal_gate_or_fallback_used") is False, f"formal fallback was used: {dataset}")
    require(complete.get("primary_shell_order_is_invariant") is True, f"shell invariance is absent: {dataset}")
    require(complete.get("storage_bounded_complete_gallery_evaluation") is True, f"complete-gallery receipt is absent: {dataset}")
    require(complete.get("negative_primary_cells") == 0, f"negative primary cell exists: {dataset}")
    require(complete.get("nonpositive_graded_cells") == 0, f"nonpositive graded cell exists: {dataset}")
    descriptors = complete.get("results")
    require(isinstance(descriptors, list) and len(descriptors) == 6, f"expected six result cells: {dataset}")
    observed_cells = {(item.get("direction"), item.get("bits")) for item in descriptors}
    require(observed_cells == {(direction, bits) for direction in DIRECTIONS for bits in BITS}, f"result cell inventory changed: {dataset}")
    referenced = {complete_path}
    rows: list[dict[str, Any]] = []
    partial_files = 0
    query_records = 0
    for descriptor in descriptors:
        row, partial_count, records = verify_result(evaluation_root, descriptor, dataset, referenced)
        rows.append(row)
        partial_files += partial_count
        query_records += records
    rows.sort(key=lambda item: (DIRECTIONS.index(item["direction"]), BITS.index(item["bits"])))
    primary_deltas = [row["delta_map"] for row in rows] + [row["delta_ndcg50"] for row in rows]
    graded_deltas = [row["delta_jndcg50"] for row in rows]
    require_close(complete.get("minimum_primary_delta"), min(primary_deltas), f"{dataset}:minimum-primary")
    require_close(complete.get("mean_primary_delta"), sum(primary_deltas) / len(primary_deltas), f"{dataset}:mean-primary")
    require_close(complete.get("minimum_graded_delta"), min(graded_deltas), f"{dataset}:minimum-graded")
    require_close(complete.get("mean_graded_delta"), sum(graded_deltas) / len(graded_deltas), f"{dataset}:mean-graded")
    all_files = {path.resolve(strict=True) for path in evaluation_root.rglob("*") if path.is_file()}
    require(all_files == referenced, f"unreferenced or missing files in the formal root for {dataset}")
    stats = {
        "dataset": dataset,
        "evaluation_root": str(evaluation_root),
        "complete_sha256": complete["complete_sha256"],
        "rank_plan_sha256": complete["rank_plan_sha256"],
        "files": len(referenced),
        "bytes": sum(path.stat().st_size for path in referenced),
        "partial_files": partial_files,
        "query_records_across_six_cells": query_records,
        "minimum_primary_delta": min(primary_deltas),
        "mean_primary_delta": sum(primary_deltas) / len(primary_deltas),
        "minimum_graded_delta": min(graded_deltas),
        "mean_graded_delta": sum(graded_deltas) / len(graded_deltas),
    }
    return rows, stats, referenced


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def csv_text(rows: Iterable[Mapping[str, Any]], fields: list[str]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def latex_table(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete-gallery retrieval with the frozen primary deep hash and the proposed collision-conditioned detail expert (CCDE). All values are expected-tie scores; $\Delta$ is CCDE minus the primary ranker. J-NDCG uses label-overlap gain.}",
        r"\label{tab:ccde-formal}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrccc ccc ccc}",
        r"\toprule",
        r"Dataset & Dir. & Bits & \multicolumn{3}{c}{mAP} & \multicolumn{3}{c}{NDCG@50} & \multicolumn{3}{c}{J-NDCG@50} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r" & & & Primary & CCDE & $\Delta$ & Primary & CCDE & $\Delta$ & Primary & CCDE & $\Delta$ \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        dataset = row["dataset_label"] if index == 0 or rows[index - 1]["dataset"] != row["dataset"] else ""
        direction = row["direction"].upper()
        values = []
        for short in ("map", "ndcg50", "jndcg50"):
            values.extend(
                [
                    f"{row[f'primary_{short}']:.3f}",
                    f"{row[f'ccde_{short}']:.3f}",
                    f"{row[f'delta_{short}']:+.3f}",
                ]
            )
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                dataset,
                direction,
                row["bits"],
                " & ".join(values),
            )
        )
        if index + 1 < len(rows) and rows[index + 1]["dataset"] != row["dataset"]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def aggregate(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(not any(part.casefold() in {"oraldata", "processdata"} for part in output.parts), "output may not modify protected data roots")
    source_directory = directory_digest(root)
    require(
        source_directory == PINNED_SOURCE_DIRECTORY,
        "the copied 1008-file formal source directory differs from its server-verified digest",
    )
    rows: list[dict[str, Any]] = []
    dataset_stats: list[dict[str, Any]] = []
    all_referenced: set[Path] = set()
    for dataset in DATASETS:
        dataset_rows, stats, referenced = verify_dataset(root, dataset)
        rows.extend(dataset_rows)
        dataset_stats.append(stats)
        all_referenced.update(referenced)
    require(len(rows) == 18, "formal matrix must contain exactly 18 cells")
    primary_deltas = [row[f"delta_{short}"] for row in rows for short in ("map", "ndcg50")]
    graded_deltas = [row["delta_jndcg50"] for row in rows]
    require(all(value > 0.0 for value in primary_deltas), "not all 36 primary cells improved")
    require(all(value > 0.0 for value in graded_deltas), "not all 18 graded cells improved")
    body = {
        "schema": "ccde_formal_seed20260822_verified_aggregate_v1",
        "status": "VERIFIED",
        "source_root": str(root),
        "source_directory": source_directory,
        "pinned_complete_receipts": PINNED_COMPLETE_RECEIPTS,
        "datasets": dataset_stats,
        "rows": rows,
        "receipt_linked_files_verified": len(all_referenced),
        "receipt_linked_bytes_verified": sum(path.stat().st_size for path in all_referenced),
        "formal_files_verified": source_directory["files"],
        "formal_bytes_verified": source_directory["bytes"],
        "primary_improvements": len(primary_deltas),
        "primary_cells": len(primary_deltas),
        "graded_improvements": len(graded_deltas),
        "graded_cells": len(graded_deltas),
        "minimum_primary_delta": min(primary_deltas),
        "mean_primary_delta": sum(primary_deltas) / len(primary_deltas),
        "minimum_graded_delta": min(graded_deltas),
        "mean_graded_delta": sum(graded_deltas) / len(graded_deltas),
        "main_table_decimals": 3,
        "formal_labels_used_for_model_or_bit_selection": False,
        "formal_labels_opened_only_after_rank_state_freeze": True,
        "primary_shell_order_is_invariant": True,
    }
    audit = {**body, "aggregate_sha256": sha256_json(body)}
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "audit.json", canonical_json_bytes(audit).decode("utf-8") + "\n")
    fields = [
        "dataset",
        "dataset_label",
        "direction",
        "bits",
        "detail_bits",
        "queries",
        "primary_map",
        "ccde_map",
        "delta_map",
        "primary_ndcg50",
        "ccde_ndcg50",
        "delta_ndcg50",
        "primary_jndcg50",
        "ccde_jndcg50",
        "delta_jndcg50",
    ]
    atomic_write_text(output / "formal_cells.csv", csv_text(rows, fields))
    atomic_write_text(output / "table_ccde_formal.tex", latex_table(rows))
    summary_fields = [
        "dataset",
        "complete_sha256",
        "files",
        "bytes",
        "partial_files",
        "query_records_across_six_cells",
        "minimum_primary_delta",
        "mean_primary_delta",
        "minimum_graded_delta",
        "mean_graded_delta",
    ]
    atomic_write_text(output / "dataset_summary.csv", csv_text(dataset_stats, summary_fields))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="copied formal result root")
    parser.add_argument("--output", type=Path, required=True, help="new audit output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = aggregate(args.root, args.output)
    print(canonical_json_bytes({
        "status": audit["status"],
        "aggregate_sha256": audit["aggregate_sha256"],
        "formal_files_verified": audit["formal_files_verified"],
        "formal_bytes_verified": audit["formal_bytes_verified"],
        "primary_improvements": audit["primary_improvements"],
        "primary_cells": audit["primary_cells"],
        "graded_improvements": audit["graded_improvements"],
        "graded_cells": audit["graded_cells"],
        "output": str(args.output.resolve()),
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
