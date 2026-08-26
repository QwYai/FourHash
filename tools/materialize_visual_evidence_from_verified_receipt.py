"""Incrementally materialize visual rows from an already fully verified trace.

The command is intentionally unavailable without a prior evidence manifest for
the exact dataset, source seal, trace contract, trace completion, and runtime.
It rechecks the complete receipt chain and every JSONL manifest, hashes the NPZ
shards that contain selected identities, and verifies each copied OralData image.
This avoids rereading unrelated feature shards while preserving exact row
identity; no nearest-feature matching or ProcessData input is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import sha256_json
from raw_rebuilt_visuals.evidence import (
    SCHEMA,
    EvidenceError,
    _canonical_json_bytes,
    _copy_image,
    _load_json,
    _row_evidence,
    _sha256_bytes,
    _sha256_file,
    _write_json,
    parse_selection_manifest,
)


def _verified_prior(path: Path) -> Mapping[str, Any]:
    value = _load_json(path.expanduser().resolve(strict=True))
    body = {key: value[key] for key in value if key != "evidence_sha256"}
    if value.get("schema") != SCHEMA or value.get("evidence_sha256") != _sha256_bytes(
        _canonical_json_bytes(body)
    ):
        raise EvidenceError("prior full evidence receipt failed its content hash")
    required = {
        "dataset",
        "source_seal_sha256",
        "trace_contract_sha256",
        "trace_complete_sha256",
        "runtime_complete_sha256",
        "runtime_manifest_sha256",
        "evidence_sha256",
    }
    if not required.issubset(value):
        raise EvidenceError("prior full evidence receipt is incomplete")
    return value


def _verify_static_receipts(
    bundle: Path,
    runtime: Path,
    prior: Mapping[str, Any],
    *,
    dataset: str,
    source_seal: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    contract = _load_json(bundle / "contract.json")
    complete = _load_json(bundle / "complete.json")
    contract_body = {key: contract[key] for key in contract if key != "contract_sha256"}
    complete_body = {key: complete[key] for key in complete if key != "complete_sha256"}
    if (
        contract.get("contract_sha256") != sha256_json(contract_body)
        or contract.get("contract_sha256") != prior["trace_contract_sha256"]
        or complete.get("complete_sha256") != sha256_json(complete_body)
        or complete.get("complete_sha256") != prior["trace_complete_sha256"]
        or complete.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise EvidenceError("current trace contract/completion differs from prior verification")
    runtime_complete = _load_json(runtime / "complete.json")
    runtime_body = {
        key: runtime_complete[key]
        for key in runtime_complete
        if key != "complete_sha256"
    }
    runtime_manifest = _load_json(runtime / "runtime_manifest.json")
    if (
        runtime_complete.get("complete_sha256") != sha256_json(runtime_body)
        or runtime_complete.get("complete_sha256") != prior["runtime_complete_sha256"]
        or _sha256_file(runtime / "runtime_manifest.json")
        != prior["runtime_manifest_sha256"]
        or runtime_manifest.get("dataset") != dataset
        or runtime_manifest.get("source_seal_sha256") != source_seal
    ):
        raise EvidenceError("current runtime differs from prior full verification")
    if prior["dataset"] != dataset or prior["source_seal_sha256"] != source_seal:
        raise EvidenceError("selection and prior full verification differ")
    return contract, complete


def _collect_verified_rows(
    bundle: Path,
    required_ids: set[str],
    *,
    contract_sha256: str,
    complete: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[Path]]:
    receipts = sorted((bundle / "receipts").glob("part-*.json"))
    if not receipts:
        raise EvidenceError("sealed trace has no shard receipts")
    expected_start = 0
    previous_chain = "0" * 64
    found: dict[str, dict[str, Any]] = {}
    target_npz: set[Path] = set()
    for expected_part, receipt_path in enumerate(receipts):
        receipt = _load_json(receipt_path)
        body = {key: receipt[key] for key in receipt if key != "chain_sha256"}
        if (
            int(receipt.get("part", -1)) != expected_part
            or int(receipt.get("start", -1)) != expected_start
            or receipt.get("contract_sha256") != contract_sha256
            or receipt.get("previous_chain_sha256") != previous_chain
            or receipt.get("chain_sha256") != sha256_json(body)
        ):
            raise EvidenceError(f"trace receipt chain failed at {receipt_path.name}")
        manifest = (bundle / str(receipt["manifest_path"])).resolve(strict=True)
        npz_path = (bundle / str(receipt["npz_path"])).resolve(strict=True)
        try:
            manifest.relative_to(bundle)
            npz_path.relative_to(bundle)
        except ValueError as error:
            raise EvidenceError("trace receipt path escapes the sealed bundle") from error
        if _sha256_file(manifest) != receipt.get("manifest_sha256"):
            raise EvidenceError(f"trace manifest changed: {manifest}")
        part_hit = False
        with manifest.open("r", encoding="utf-8") as handle:
            for line_offset, line in enumerate(handle):
                row = json.loads(line)
                expected_index = expected_start + line_offset
                if not isinstance(row, dict) or int(row.get("row_index", -1)) != expected_index:
                    raise EvidenceError("trace manifest row order changed")
                row_id = str(row.get("canonical_row_id", ""))
                if row_id in required_ids:
                    if row_id in found:
                        raise EvidenceError("selected canonical row ID is duplicated")
                    found[row_id] = row
                    part_hit = True
        if line_offset + 1 != int(receipt["rows"]):
            raise EvidenceError("trace manifest row count changed")
        if part_hit:
            target_npz.add(npz_path)
            if _sha256_file(npz_path) != receipt.get("npz_sha256"):
                raise EvidenceError(f"selected feature shard changed: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as values:
                shard_ids = {bytes(value).decode("ascii") for value in values["row_ids"]}
            selected_in_part = required_ids.intersection(found).intersection(shard_ids)
            if not selected_in_part:
                raise EvidenceError("selected manifest rows are absent from their feature shard")
        expected_start = int(receipt["stop"])
        previous_chain = str(receipt["chain_sha256"])
    if (
        expected_start != int(complete["rows"])
        or len(receipts) != int(complete["shards"])
        or previous_chain != complete["final_chain_sha256"]
    ):
        raise EvidenceError("trace completion and receipt chain differ")
    missing = sorted(required_ids.difference(found))
    if missing:
        raise EvidenceError(f"selected rows are absent from the sealed trace: {missing[:3]}")
    return found, target_npz


def materialize(
    *,
    bundle: Path,
    runtime: Path,
    selection: Path,
    prior_evidence: Path,
    output: Path,
) -> Mapping[str, Any]:
    bundle_root = bundle.expanduser().resolve(strict=True)
    runtime_root = runtime.expanduser().resolve(strict=True)
    destination = output.expanduser().resolve()
    if destination.exists() or destination.with_name(destination.name + ".partial").exists():
        raise EvidenceError("visual evidence output and staging paths must be new")
    dataset, rank_token, source_seal, cases, selection_payload = parse_selection_manifest(
        selection
    )
    prior = _verified_prior(prior_evidence)
    contract, complete = _verify_static_receipts(
        bundle_root,
        runtime_root,
        prior,
        dataset=dataset,
        source_seal=source_seal,
    )
    required_ids = {
        row_id for case in cases for row_id in (case.query_row_id, *case.candidate_row_ids)
    }
    rows, target_npz = _collect_verified_rows(
        bundle_root,
        required_ids,
        contract_sha256=str(contract["contract_sha256"]),
        complete=complete,
    )

    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True)
    case_payloads = []
    for case in cases:
        case_root = staging / case.case_id
        case_root.mkdir()
        query_copy = _copy_image(rows[case.query_row_id], case_root / "query")
        candidates = []
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
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "rank_token_sha256": rank_token,
        "source_seal_sha256": source_seal,
        "selection_sha256": selection_payload["selection_sha256"],
        "trace_contract_sha256": contract["contract_sha256"],
        "trace_complete_sha256": complete["complete_sha256"],
        "provenance_bundle_id": prior.get("provenance_bundle_id"),
        "runtime_complete_sha256": prior["runtime_complete_sha256"],
        "runtime_manifest_sha256": prior["runtime_manifest_sha256"],
        "case_count": len(case_payloads),
        "row_count": len(required_ids),
        "cases": case_payloads,
        "process_data_role": "not_opened",
        "prior_full_evidence_sha256": prior["evidence_sha256"],
        "verified_manifest_files": int(complete["shards"]),
        "verified_selected_npz_shards": len(target_npz),
    }
    result = {**body, "evidence_sha256": _sha256_bytes(_canonical_json_bytes(body))}
    _write_json(staging / "evidence_manifest.json", result)
    os.replace(staging, destination)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prior-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize(
            bundle=args.bundle,
            runtime=args.runtime,
            selection=args.selection,
            prior_evidence=args.prior_evidence,
            output=args.output,
        )
    except EvidenceError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "status": "OK",
                "evidence_sha256": result["evidence_sha256"],
                "rows": result["row_count"],
                "selected_npz_shards": result["verified_selected_npz_shards"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
