from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from raw_rebuilt_visuals.evidence import (
    EvidenceError,
    SELECTION_SCHEMA,
    _canonical_json_bytes,
    _sealed_trace_dataset,
    collect_trace_rows,
    parse_selection_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _selection(path: Path, *, duplicate: bool = False) -> tuple[str, str]:
    query = _digest("query")
    candidate = query if duplicate else _digest("candidate")
    payload = {
        "schema": SELECTION_SCHEMA,
        "dataset": "nuswide",
        "rank_token_sha256": _digest("rank"),
        "source_seal_sha256": _digest("source"),
        "cases": [{
            "case_id": "case-001",
            "query_row_id": query,
            "candidate_row_ids": [candidate],
        }],
    }
    payload["selection_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return query, candidate


def test_parse_selection_binds_body_and_rejects_duplicate_identity(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    query, candidate = _selection(path)
    dataset, _, _, cases, _ = parse_selection_manifest(path)
    assert dataset == "nuswide"
    assert cases[0].query_row_id == query
    assert cases[0].candidate_row_ids == (candidate,)
    payload = json.loads(path.read_text())
    payload["dataset"] = "mscoco"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="selection_sha256"):
        parse_selection_manifest(path)
    _selection(path, duplicate=True)
    with pytest.raises(EvidenceError, match="duplicate"):
        parse_selection_manifest(path)


def test_collect_rows_uses_receipt_order_and_requires_contiguous_indices(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "receipts").mkdir(parents=True)
    (root / "manifests").mkdir()
    ids = [_digest("row0"), _digest("row1")]
    manifest = root / "manifests" / "part-000000.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"row_index": i, "canonical_row_id": row_id}) for i, row_id in enumerate(ids)) + "\n",
        encoding="utf-8",
    )
    (root / "receipts" / "part-000000.json").write_text(
        json.dumps({"manifest_path": "manifests/part-000000.jsonl"}), encoding="utf-8"
    )
    found = collect_trace_rows(root, {ids[1]})
    assert found[ids[1]]["row_index"] == 1
    rows = manifest.read_text(encoding="utf-8").splitlines()
    poisoned = json.loads(rows[1])
    poisoned["row_index"] = 7
    manifest.write_text(rows[0] + "\n" + json.dumps(poisoned) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="contiguous"):
        collect_trace_rows(root, {ids[1]})


def test_collect_rows_rejects_unknown_id(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "receipts").mkdir(parents=True)
    (root / "manifests").mkdir()
    known = _digest("known")
    (root / "manifests" / "part-000000.jsonl").write_text(
        json.dumps({"row_index": 0, "canonical_row_id": known}) + "\n", encoding="utf-8"
    )
    (root / "receipts" / "part-000000.json").write_text(
        json.dumps({"manifest_path": "manifests/part-000000.jsonl"}), encoding="utf-8"
    )
    with pytest.raises(EvidenceError, match="absent"):
        collect_trace_rows(root, {_digest("missing")})


def test_trace_dataset_accepts_nested_adapter_and_rejects_conflict() -> None:
    assert _sealed_trace_dataset(
        {"adapter": {"dataset": "mscoco"}}, {"dataset": "mscoco"}
    ) == "mscoco"
    with pytest.raises(EvidenceError, match="contradictory"):
        _sealed_trace_dataset(
            {"dataset": "nuswide", "adapter": {"dataset": "mscoco"}},
            {"dataset": "mscoco"},
        )
