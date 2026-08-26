"""Materialize exact raw image/text/label cards from frozen rank selections.

The selection file contains only canonical row IDs emitted by a frozen rank
stage.  This module resolves those IDs through the sealed extraction manifests;
it never guesses identity from nearest features and never opens ``ProcessData``.
Formal materialization first runs both trace validators, then copies only the
selected raw images and writes the exact raw-text locator/value and label vector
that were bound to the extracted feature row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "raw_rebuilt_visual_evidence_v1"
SELECTION_SCHEMA = "raw_rebuilt_rank_visual_selection_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(RuntimeError):
    """A rank selection cannot be resolved without weakening provenance."""


@dataclass(frozen=True)
class SelectionCase:
    case_id: str
    query_row_id: str
    candidate_row_ids: tuple[str, ...]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read canonical JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must contain a JSON object")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_case_id(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value) is None:
        raise EvidenceError("case_id must be a safe 1-80 character identifier")
    return value


def _sealed_trace_dataset(
    contract: Mapping[str, Any], complete: Mapping[str, Any]
) -> str:
    """Resolve the dataset without weakening contradictory trace bindings."""

    adapter = contract.get("adapter")
    declarations = [contract.get("dataset"), complete.get("dataset")]
    if isinstance(adapter, Mapping):
        declarations.append(adapter.get("dataset"))
    present = [str(value) for value in declarations if value is not None]
    if not present or len(set(present)) != 1:
        raise EvidenceError("sealed trace has missing or contradictory dataset bindings")
    return present[0]


def parse_selection_manifest(path: os.PathLike[str]) -> tuple[str, str, str, tuple[SelectionCase, ...], dict[str, Any]]:
    """Parse the immutable output of a label-free rank-selection stage."""

    source = Path(path).expanduser().resolve()
    payload = _load_json(source)
    if payload.get("schema") != SELECTION_SCHEMA:
        raise EvidenceError(f"selection schema must be {SELECTION_SCHEMA}")
    dataset = payload.get("dataset")
    if dataset not in {"mirflickr", "nuswide", "mscoco"}:
        raise EvidenceError("selection dataset is not in the frozen registry")
    rank_token = _require_sha256(payload.get("rank_token_sha256"), "rank_token_sha256")
    source_seal = _require_sha256(payload.get("source_seal_sha256"), "source_seal_sha256")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvidenceError("selection must contain at least one case")
    cases: list[SelectionCase] = []
    seen_case_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise EvidenceError(f"cases[{index}] must be an object")
        case_id = _require_case_id(raw.get("case_id"))
        if case_id in seen_case_ids:
            raise EvidenceError(f"duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)
        query = _require_sha256(raw.get("query_row_id"), f"cases[{index}].query_row_id")
        candidates_raw = raw.get("candidate_row_ids")
        if not isinstance(candidates_raw, list) or not candidates_raw:
            raise EvidenceError(f"cases[{index}] needs candidate_row_ids")
        candidates = tuple(
            _require_sha256(value, f"cases[{index}].candidate_row_ids")
            for value in candidates_raw
        )
        if len(set(candidates)) != len(candidates) or query in candidates:
            raise EvidenceError(f"cases[{index}] has duplicate query/candidate identities")
        cases.append(SelectionCase(case_id, query, candidates))
    body = dict(payload)
    claimed = body.pop("selection_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(body))
    if claimed != actual:
        raise EvidenceError("selection_sha256 does not bind the selection body")
    return str(dataset), rank_token, source_seal, tuple(cases), payload


def collect_trace_rows(
    bundle: os.PathLike[str], required_row_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Resolve canonical IDs by streaming sealed manifests in receipt order."""

    root = Path(bundle).expanduser().resolve()
    required = {_require_sha256(value, "required_row_id") for value in required_row_ids}
    if not required:
        raise EvidenceError("at least one required row ID is needed")
    receipts_root = root / "receipts"
    found: dict[str, dict[str, Any]] = {}
    expected_row_index = 0
    receipts = sorted(receipts_root.glob("part-*.json"))
    if not receipts:
        raise EvidenceError("sealed trace has no receipts")
    for receipt_path in receipts:
        receipt = _load_json(receipt_path)
        relative = receipt.get("manifest_path")
        if not isinstance(relative, str):
            raise EvidenceError(f"{receipt_path} lacks manifest_path")
        manifest = (root / relative).resolve()
        try:
            manifest.relative_to(root)
        except ValueError as error:
            raise EvidenceError("manifest path escapes the sealed trace") from error
        if not manifest.is_file():
            raise EvidenceError(f"missing trace manifest {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise EvidenceError(f"blank row in {manifest}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvidenceError(f"bad row JSON in {manifest}:{line_number}") from error
                if not isinstance(row, dict) or row.get("row_index") != expected_row_index:
                    raise EvidenceError("trace manifests are not in contiguous canonical order")
                expected_row_index += 1
                row_id = _require_sha256(row.get("canonical_row_id"), "canonical_row_id")
                if row_id in required:
                    if row_id in found:
                        raise EvidenceError(f"duplicate canonical row ID {row_id}")
                    found[row_id] = row
    missing = sorted(required.difference(found))
    if missing:
        raise EvidenceError(f"selection row IDs are absent from the sealed trace: {missing[:3]}")
    return found


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _copy_image(row: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    source_value = row.get("raw_image_source")
    if not isinstance(source_value, dict):
        raise EvidenceError("trace row lacks raw_image_source")
    source = Path(str(source_value.get("path", ""))).expanduser().resolve()
    if "ProcessData" in source.parts or "OralData" not in source.parts:
        raise EvidenceError("visual evidence image must come directly from OralData")
    expected_hash = _require_sha256(source_value.get("sha256"), "raw_image_source.sha256")
    expected_bytes = source_value.get("bytes")
    if not source.is_file() or source.stat().st_size != expected_bytes or _sha256_file(source) != expected_hash:
        raise EvidenceError(f"raw image no longer matches sealed row: {source}")
    suffix = source.suffix.lower()
    if not suffix or re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) is None:
        suffix = ".image"
    target = destination.with_suffix(suffix)
    shutil.copyfile(source, target)
    copied_hash = _sha256_file(target)
    if copied_hash != expected_hash:
        raise EvidenceError("copied raw image failed byte identity check")
    return {
        "copied_path": target.name,
        "bytes": target.stat().st_size,
        "sha256": copied_hash,
        "source_path": str(source),
    }


def _row_evidence(row: Mapping[str, Any], copied_image: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_row_id": row["canonical_row_id"],
        "row_index": row["row_index"],
        "source_id": row.get("source_id"),
        "split": row.get("split"),
        "raw_image": dict(copied_image),
        "raw_text": row.get("raw_text"),
        "raw_text_sha256": row.get("raw_text_sha256"),
        "raw_text_sources": row.get("raw_text_sources"),
        "encoded_texts": row.get("encoded_texts"),
        "label_hot": row.get("label_hot"),
        "label_sha256": row.get("label_sha256"),
        "metadata": row.get("metadata"),
        "feature_binding_sha256": row.get("feature_binding_sha256"),
        "row_contract_sha256": row.get("row_contract_sha256"),
    }


def materialize_evidence(
    *,
    bundle: os.PathLike[str],
    runtime: os.PathLike[str],
    selection_manifest: os.PathLike[str],
    output: os.PathLike[str],
    process_data_root: os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate, resolve, and copy a frozen set of qualitative result cases."""

    root = Path(bundle).expanduser().resolve()
    runtime_root = Path(runtime).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise EvidenceError("visual evidence output must be a new directory")
    for forbidden in (
        root,
        runtime_root,
        Path(process_data_root).resolve() if process_data_root else None,
    ):
        if forbidden is None:
            continue
        try:
            destination.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise EvidenceError("visual evidence output overlaps a forbidden input root")

    dataset, rank_token, source_seal, cases, selection_payload = parse_selection_manifest(selection_manifest)
    try:
        from visualization_feature_pipeline import validate_bundle
        from visualization_trace.extraction import verify_trace_bundle
        from raw_rebuilt_runtime import verify_runtime_directory
    except ImportError as error:
        raise EvidenceError("raw trace validators are unavailable") from error
    trace_report = verify_trace_bundle(root)
    provenance_report = validate_bundle(root, process_data_root=process_data_root)
    runtime_manifest = verify_runtime_directory(
        runtime_root,
        process_data_root=None if process_data_root is None else Path(process_data_root),
    )
    contract = trace_report["contract"]
    if _sealed_trace_dataset(contract, trace_report["complete"]) != dataset:
        raise EvidenceError("selection dataset differs from sealed trace")
    if runtime_manifest.get("dataset") != dataset:
        raise EvidenceError("selection dataset differs from verified runtime")
    if runtime_manifest.get("source_seal_sha256") != source_seal:
        raise EvidenceError("selection source seal is not bound to this trace")
    required = {
        row_id
        for case in cases
        for row_id in (case.query_row_id, *case.candidate_row_ids)
    }
    rows = collect_trace_rows(root, required)

    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise EvidenceError("stale visual evidence staging directory exists")
    staging.mkdir(parents=True)
    case_payloads: list[dict[str, Any]] = []
    try:
        for case in cases:
            case_root = staging / case.case_id
            case_root.mkdir()
            query_copy = _copy_image(rows[case.query_row_id], case_root / "query")
            candidates: list[dict[str, Any]] = []
            for rank, row_id in enumerate(case.candidate_row_ids, start=1):
                copied = _copy_image(rows[row_id], case_root / f"candidate-{rank:03d}")
                candidates.append({"rank": rank, **_row_evidence(rows[row_id], copied)})
            case_record = {
                "case_id": case.case_id,
                "query": _row_evidence(rows[case.query_row_id], query_copy),
                "candidates": candidates,
            }
            _write_json(case_root / "evidence.json", case_record)
            case_payloads.append(case_record)
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "dataset": dataset,
            "rank_token_sha256": rank_token,
            "source_seal_sha256": source_seal,
            "selection_sha256": selection_payload["selection_sha256"],
            "trace_contract_sha256": contract["contract_sha256"],
            "trace_complete_sha256": trace_report["complete"]["complete_sha256"],
            "provenance_bundle_id": getattr(provenance_report, "bundle_id", None),
            "runtime_complete_sha256": _load_json(runtime_root / "complete.json")["complete_sha256"],
            "runtime_manifest_sha256": _sha256_file(runtime_root / "runtime_manifest.json"),
            "case_count": len(case_payloads),
            "row_count": len(required),
            "cases": case_payloads,
            "process_data_role": "forbidden_boundary_only" if process_data_root else "none",
        }
        body = dict(result)
        result["evidence_sha256"] = _sha256_bytes(_canonical_json_bytes(body))
        _write_json(staging / "evidence_manifest.json", result)
        os.replace(staging, destination)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
