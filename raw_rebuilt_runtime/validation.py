"""Independent double-admission gate for ``visualization_trace`` bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    RuntimeBridgeError,
    load_json,
    sha256_file,
    sha256_json,
)


@dataclass(frozen=True)
class SourceAdmission:
    bundle: Path
    dataset: str
    rows: int
    shards: int
    contract: Mapping[str, Any]
    complete: Mapping[str, Any]
    split: Mapping[str, Any]
    provenance_report: Mapping[str, Any]
    seal: Mapping[str, Any]


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        value = report.to_dict()
    elif isinstance(report, Mapping):
        value = dict(report)
    else:
        raise RuntimeBridgeError("provenance validator returned an unsupported report")
    if value.get("status") != "PASS":
        raise RuntimeBridgeError("provenance validator did not return PASS")
    return value


def _sealed_file(root: Path, relative: str) -> dict[str, Any]:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeBridgeError(f"source artifact escapes sealed bundle: {relative}") from error
    if not path.is_file() or path.is_symlink():
        raise RuntimeBridgeError(f"source artifact is not a regular file: {path}")
    stat = path.stat()
    return {"path": relative, "size": stat.st_size, "sha256": sha256_file(path)}


def admit_source_bundle(
    bundle_root: Path,
    *,
    process_data_root: Path | None = None,
) -> SourceAdmission:
    """Run both validators and capture an immutable source seal.

    No fallback parser exists.  Import failure or either validator failing is a
    hard admission failure.
    """

    try:
        from visualization_trace.extraction import verify_trace_bundle
        from visualization_feature_pipeline import validate_bundle
    except ImportError as error:
        raise RuntimeBridgeError(
            "both visualization_trace and visualization_feature_pipeline are required"
        ) from error

    try:
        root = Path(bundle_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError(f"sealed source bundle does not exist: {bundle_root}") from error
    if not root.is_dir():
        raise RuntimeBridgeError("sealed source bundle must be a directory")
    try:
        trace = verify_trace_bundle(root)
    except Exception as error:
        raise RuntimeBridgeError(f"verify_trace_bundle rejected source: {error}") from error
    if not isinstance(trace, Mapping) or trace.get("verified") is not True:
        raise RuntimeBridgeError("verify_trace_bundle did not return verified=True")
    try:
        provenance = _report_dict(validate_bundle(root, process_data_root))
    except Exception as error:
        if isinstance(error, RuntimeBridgeError):
            raise
        raise RuntimeBridgeError(f"provenance validator rejected source: {error}") from error

    contract = trace.get("contract")
    complete = trace.get("complete")
    split = trace.get("split")
    if not all(isinstance(value, Mapping) for value in (contract, complete, split)):
        raise RuntimeBridgeError("trace verification omitted contract/complete/split records")
    adapter = contract.get("adapter")
    if not isinstance(adapter, Mapping):
        raise RuntimeBridgeError("trace contract omitted adapter binding")
    dataset = str(adapter.get("dataset", ""))
    rows = int(complete.get("rows", -1))
    shards = int(complete.get("shards", -1))
    if provenance.get("dataset") != dataset or int(provenance.get("row_count", -1)) != rows:
        raise RuntimeBridgeError("the two validators disagree on dataset identity or row count")
    if int(trace.get("rows", -1)) != rows or int(trace.get("shards", -1)) != shards:
        raise RuntimeBridgeError("trace verification and complete marker disagree")
    try:
        data_root = Path(str(adapter.get("data_root", ""))).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError("trace adapter data_root is missing") from error
    oral_root = (data_root / "OralData").resolve(strict=True)
    process_root = (data_root / "ProcessData").resolve(strict=False)

    required = [
        _sealed_file(root, "contract.json"),
        _sealed_file(root, "complete.json"),
        _sealed_file(root, "canonical_split.npz"),
        _sealed_file(root, "canonical_split.json"),
    ]
    receipt_files = sorted((root / "receipts").glob("part-*.json"))
    if len(receipt_files) != shards:
        raise RuntimeBridgeError("source receipt count differs from complete marker")
    receipts = [
        _sealed_file(root, path.relative_to(root).as_posix()) for path in receipt_files
    ]
    contract_json = load_json(root / "contract.json")
    complete_json = load_json(root / "complete.json")
    split_json = load_json(root / "canonical_split.json")
    if contract_json != dict(contract) or complete_json != dict(complete) or split_json != dict(split):
        raise RuntimeBridgeError("verified source records changed while the source seal was captured")
    seal_body = {
        "schema": "kbs_raw_rebuilt_source_seal_v1",
        "bundle_path": str(root),
        "data_root": str(data_root),
        "oral_data_root": str(oral_root),
        "process_data_root": str(process_root),
        "dataset": dataset,
        "rows": rows,
        "shards": shards,
        "trace_contract_sha256": str(contract.get("contract_sha256", "")),
        "trace_complete_sha256": str(complete.get("complete_sha256", "")),
        "trace_split_contract_sha256": str(split.get("split_contract_sha256", "")),
        "trace_final_chain_sha256": str(complete.get("final_chain_sha256", "")),
        "files": required,
        "receipts": receipts,
        "receipt_set_sha256": sha256_json(receipts),
        "provenance_report": provenance,
        "provenance_report_sha256": sha256_json(provenance),
    }
    seal = {**seal_body, "source_seal_sha256": sha256_json(seal_body)}
    return SourceAdmission(
        bundle=root,
        dataset=dataset,
        rows=rows,
        shards=shards,
        contract=dict(contract),
        complete=dict(complete),
        split=dict(split),
        provenance_report=provenance,
        seal=seal,
    )


def reopen_frozen_source_bundle(
    bundle_root: Path,
    expected_seal: Mapping[str, Any],
    *,
    process_data_root: Path | None = None,
) -> SourceAdmission:
    """Reopen an already admitted source without rematerializing raw rows.

    Full OralData provenance validation is mandatory when the runtime is
    materialized and its source seal is created.  Metric workers run only
    after a rank plan has frozen that seal.  At that later boundary we verify
    the complete trace bundle, every sealed descriptor, the recorded
    provenance PASS, and the seal digest, but deliberately avoid rebuilding
    all raw row dictionaries and full feature matrices in RAM a second time.

    ``verify_runtime_directory`` subsequently rehashes every runtime array and
    compares every runtime shard with this source admission, so this bounded
    reopen changes memory use rather than the bytes admitted for scoring.
    """

    try:
        root = Path(bundle_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError(
            f"frozen source bundle does not exist: {bundle_root}"
        ) from error
    if not root.is_dir():
        raise RuntimeBridgeError("frozen source bundle must be a directory")
    if not isinstance(expected_seal, Mapping):
        raise RuntimeBridgeError("frozen source seal must be an object")
    seal = dict(expected_seal)
    seal_body = dict(seal)
    observed_seal_sha256 = seal_body.pop("source_seal_sha256", None)
    if (
        not isinstance(observed_seal_sha256, str)
        or sha256_json(seal_body) != observed_seal_sha256
    ):
        raise RuntimeBridgeError("frozen source seal digest mismatch")

    sealed_bundle = seal.get("bundle_path")
    if not isinstance(sealed_bundle, str):
        raise RuntimeBridgeError("frozen source seal has no bundle path")
    try:
        if Path(sealed_bundle).expanduser().resolve(strict=True) != root:
            raise RuntimeBridgeError("frozen source seal names another bundle")
    except OSError as error:
        raise RuntimeBridgeError("frozen source seal bundle path is unavailable") from error

    try:
        from visualization_trace.extraction import verify_trace_bundle
    except ImportError as error:
        raise RuntimeBridgeError("visualization_trace is required") from error
    try:
        trace = verify_trace_bundle(root)
    except Exception as error:
        raise RuntimeBridgeError(
            f"frozen trace verification rejected source: {error}"
        ) from error
    if not isinstance(trace, Mapping) or trace.get("verified") is not True:
        raise RuntimeBridgeError("frozen trace verification did not return verified=True")
    contract = trace.get("contract")
    complete = trace.get("complete")
    split = trace.get("split")
    if not all(isinstance(value, Mapping) for value in (contract, complete, split)):
        raise RuntimeBridgeError("frozen trace omitted contract/complete/split records")
    adapter = contract.get("adapter")
    if not isinstance(adapter, Mapping):
        raise RuntimeBridgeError("frozen trace contract omitted adapter binding")
    dataset = str(adapter.get("dataset", ""))
    rows = int(complete.get("rows", -1))
    shards = int(complete.get("shards", -1))
    if (
        seal.get("dataset") != dataset
        or seal.get("rows") != rows
        or seal.get("shards") != shards
        or trace.get("rows") != rows
        or trace.get("shards") != shards
    ):
        raise RuntimeBridgeError("frozen source geometry differs from its seal")
    if (
        seal.get("trace_contract_sha256") != contract.get("contract_sha256")
        or seal.get("trace_complete_sha256") != complete.get("complete_sha256")
        or seal.get("trace_split_contract_sha256")
        != split.get("split_contract_sha256")
        or seal.get("trace_final_chain_sha256")
        != complete.get("final_chain_sha256")
    ):
        raise RuntimeBridgeError("frozen trace hashes differ from the source seal")

    try:
        data_root = Path(str(adapter.get("data_root", ""))).expanduser().resolve(
            strict=True
        )
        oral_root = (data_root / "OralData").resolve(strict=True)
        sealed_data_root = Path(str(seal.get("data_root", ""))).expanduser().resolve(
            strict=True
        )
        sealed_oral_root = Path(
            str(seal.get("oral_data_root", ""))
        ).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError("frozen source authority roots are unavailable") from error
    sealed_process_root = Path(
        str(seal.get("process_data_root", ""))
    ).expanduser().resolve(strict=False)
    if (
        data_root != sealed_data_root
        or oral_root != sealed_oral_root
        or sealed_process_root != (data_root / "ProcessData").resolve(strict=False)
    ):
        raise RuntimeBridgeError("frozen source authority roots differ from the seal")
    if process_data_root is not None and Path(process_data_root).expanduser().resolve(
        strict=False
    ) != sealed_process_root:
        raise RuntimeBridgeError("metric ProcessData boundary differs from the source seal")

    required_paths = {
        "contract.json",
        "complete.json",
        "canonical_split.npz",
        "canonical_split.json",
    }
    sealed_files = seal.get("files")
    if not isinstance(sealed_files, list) or len(sealed_files) != len(required_paths):
        raise RuntimeBridgeError("frozen source seal has an invalid required-file set")
    observed_paths: set[str] = set()
    for descriptor in sealed_files:
        if not isinstance(descriptor, Mapping):
            raise RuntimeBridgeError("frozen required-file descriptor is invalid")
        relative = str(descriptor.get("path", ""))
        if relative in observed_paths or relative not in required_paths:
            raise RuntimeBridgeError("frozen required-file paths are invalid")
        if _sealed_file(root, relative) != dict(descriptor):
            raise RuntimeBridgeError(f"frozen required file changed: {relative}")
        observed_paths.add(relative)
    if observed_paths != required_paths:
        raise RuntimeBridgeError("frozen required-file set is incomplete")

    sealed_receipts = seal.get("receipts")
    if not isinstance(sealed_receipts, list) or len(sealed_receipts) != shards:
        raise RuntimeBridgeError("frozen source receipt set is incomplete")
    for part, descriptor in enumerate(sealed_receipts):
        if not isinstance(descriptor, Mapping):
            raise RuntimeBridgeError("frozen source receipt descriptor is invalid")
        relative = f"receipts/part-{part:06d}.json"
        if descriptor.get("path") != relative or _sealed_file(
            root, relative
        ) != dict(descriptor):
            raise RuntimeBridgeError(f"frozen source receipt changed: {relative}")
    if seal.get("receipt_set_sha256") != sha256_json(sealed_receipts):
        raise RuntimeBridgeError("frozen source receipt-set digest mismatch")

    provenance = seal.get("provenance_report")
    checks = provenance.get("checks") if isinstance(provenance, Mapping) else None
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("status") != "PASS"
        or provenance.get("dataset") != dataset
        or provenance.get("row_count") != rows
        or provenance.get("shard_count") != shards
        or not isinstance(checks, list)
        or not any("raw_rebuilt_v1" in str(value) for value in checks)
        or seal.get("provenance_report_sha256") != sha256_json(provenance)
    ):
        raise RuntimeBridgeError("frozen provenance PASS is absent or inconsistent")

    return SourceAdmission(
        bundle=root,
        dataset=dataset,
        rows=rows,
        shards=shards,
        contract=dict(contract),
        complete=dict(complete),
        split=dict(split),
        provenance_report=dict(provenance),
        seal=seal,
    )
