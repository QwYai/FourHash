"""Streaming, resumable materialization of an admitted trace bundle."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import (
    ARRAY_SPECS,
    FEATURE_DIM,
    SCHEMA_VERSION,
    SPLIT_ALGORITHM,
    SPLIT_SEED,
    RuntimeBridgeError,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    numeric_sha256,
    require_disjoint,
    require_geometry,
    require_split_arrays,
    sha256_file,
    sha256_json,
)
from .validation import (
    SourceAdmission,
    admit_source_bundle,
    reopen_frozen_source_bundle,
)


HEX64_RE = re.compile(br"^[0-9a-f]{64}$")


def _require_finite_feature_memmaps(*arrays: np.ndarray, block_rows: int = 4096) -> None:
    for array in arrays:
        for start in range(0, int(array.shape[0]), block_rows):
            if not np.isfinite(array[start : start + block_rows]).all():
                raise RuntimeBridgeError("runtime feature matrix contains NaN or infinity")


def _require_binary_label_memmap(labels: np.ndarray, block_rows: int = 8192) -> None:
    for start in range(0, int(labels.shape[0]), block_rows):
        block = labels[start : start + block_rows]
        if not np.isin(block, (0, 1)).all() or np.any(block.sum(axis=1) == 0):
            raise RuntimeBridgeError("runtime labels are not non-empty binary multi-hot rows")


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeBridgeError(f"runtime path is not contained: {relative}")
    resolved = (root / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeBridgeError(f"runtime path escapes output directory: {relative}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeBridgeError(f"runtime artifact is not a regular file: {resolved}")
    return resolved


def _source_receipt_paths(admission: SourceAdmission) -> list[Path]:
    values = admission.seal.get("receipts")
    if not isinstance(values, list):
        raise RuntimeBridgeError("source seal has no receipt descriptors")
    paths: list[Path] = []
    for expected_part, descriptor in enumerate(values):
        if not isinstance(descriptor, Mapping):
            raise RuntimeBridgeError("source receipt descriptor is not an object")
        expected = f"receipts/part-{expected_part:06d}.json"
        if descriptor.get("path") != expected:
            raise RuntimeBridgeError("source receipt descriptors are not contiguous")
        path = _relative_file(admission.bundle, expected)
        if path.stat().st_size != descriptor.get("size") or sha256_file(path) != descriptor.get("sha256"):
            raise RuntimeBridgeError(f"source receipt changed after admission: {path}")
        paths.append(path)
    return paths


def _assert_admission_seal_unchanged(admission: SourceAdmission) -> None:
    files = admission.seal.get("files")
    if not isinstance(files, list):
        raise RuntimeBridgeError("source seal has no required-file descriptors")
    for descriptor in files:
        if not isinstance(descriptor, Mapping):
            raise RuntimeBridgeError("source required-file descriptor is invalid")
        relative = str(descriptor.get("path", ""))
        path = _relative_file(admission.bundle, relative)
        if path.stat().st_size != descriptor.get("size") or sha256_file(path) != descriptor.get("sha256"):
            raise RuntimeBridgeError(f"sealed source file changed after admission: {relative}")
    _source_receipt_paths(admission)


def _load_source_part(
    admission: SourceAdmission,
    receipt_path: Path,
    *,
    expected_part: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    receipt = load_json(receipt_path)
    if int(receipt.get("part", -1)) != expected_part:
        raise RuntimeBridgeError("source part number differs from its path")
    npz_relative = receipt.get("npz_path")
    if not isinstance(npz_relative, str):
        raise RuntimeBridgeError("source receipt omitted npz_path")
    npz_path = _relative_file(admission.bundle, npz_relative)
    if sha256_file(npz_path) != receipt.get("npz_sha256"):
        raise RuntimeBridgeError(f"source NPZ changed after admission: {npz_path}")
    manifest_relative = receipt.get("manifest_path")
    if not isinstance(manifest_relative, str):
        raise RuntimeBridgeError("source receipt omitted manifest_path")
    manifest_path = _relative_file(admission.bundle, manifest_relative)
    if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
        raise RuntimeBridgeError(f"source row manifest changed after admission: {manifest_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            expected_keys = {
                "row_index",
                "row_ids",
                "image_features",
                "text_features",
                "labels",
            }
            if set(loaded.files) != expected_keys:
                raise RuntimeBridgeError(
                    f"source shard keys differ; expected={sorted(expected_keys)}, got={sorted(loaded.files)}"
                )
            arrays = {name: np.asarray(loaded[name]).copy() for name in expected_keys}
    except (OSError, ValueError) as error:
        if isinstance(error, RuntimeBridgeError):
            raise
        raise RuntimeBridgeError(f"cannot load admitted source shard: {npz_path}") from error
    start = int(receipt.get("start", -1))
    stop = int(receipt.get("stop", -1))
    count = stop - start
    if count <= 0 or int(receipt.get("rows", -1)) != count:
        raise RuntimeBridgeError("source receipt has invalid row bounds")
    if arrays["row_index"].dtype != np.dtype("int64") or not np.array_equal(
        arrays["row_index"], np.arange(start, stop, dtype=np.int64)
    ):
        raise RuntimeBridgeError("source row indices are not contiguous int64")
    if arrays["row_ids"].dtype != np.dtype("S64") or arrays["row_ids"].shape != (count,):
        raise RuntimeBridgeError("source row IDs must be fixed S64")
    if arrays["image_features"].dtype != np.dtype("float32") or arrays["text_features"].dtype != np.dtype("float32"):
        raise RuntimeBridgeError("source image/text arrays must already be float32")
    if arrays["labels"].dtype != np.dtype("uint8"):
        raise RuntimeBridgeError("source labels must already be uint8")
    if arrays["image_features"].shape != (count, FEATURE_DIM) or arrays["text_features"].shape != (count, FEATURE_DIM):
        raise RuntimeBridgeError("source feature shard is not aligned [rows,512]")
    if arrays["labels"].ndim != 2 or arrays["labels"].shape[0] != count:
        raise RuntimeBridgeError("source label shard is not row-aligned")
    if not np.isfinite(arrays["image_features"]).all() or not np.isfinite(arrays["text_features"]).all():
        raise RuntimeBridgeError("source feature shard contains NaN or infinity")
    if not np.isin(arrays["labels"], (0, 1)).all() or np.any(arrays["labels"].sum(axis=1) == 0):
        raise RuntimeBridgeError("source labels must be non-empty binary multi-hot rows")
    return receipt, arrays


def _load_source_split(admission: SourceAdmission) -> tuple[dict[str, np.ndarray], np.ndarray]:
    path = admission.bundle / "canonical_split.npz"
    try:
        with np.load(path, allow_pickle=False) as loaded:
            expected = {"indQ", "indT", "indD", "row_ids"}
            if set(loaded.files) != expected:
                raise RuntimeBridgeError(
                    "canonical split must contain exactly indQ/indT/indD/row_ids"
                )
            arrays = {name: np.asarray(loaded[name]).copy() for name in ("indQ", "indT", "indD")}
            row_ids = np.asarray(loaded["row_ids"]).copy()
    except (OSError, ValueError) as error:
        if isinstance(error, RuntimeBridgeError):
            raise
        raise RuntimeBridgeError(f"cannot load canonical split: {path}") from error
    if row_ids.dtype != np.dtype("S64") or row_ids.shape != (admission.rows,):
        raise RuntimeBridgeError("canonical split row_ids must be aligned fixed S64")
    return arrays, row_ids


def _geometry_from_source(admission: SourceAdmission) -> tuple[int, int]:
    paths = _source_receipt_paths(admission)
    if not paths:
        raise RuntimeBridgeError("source bundle has no feature shards")
    _receipt, arrays = _load_source_part(admission, paths[0], expected_part=0)
    return int(arrays["image_features"].shape[1]), int(arrays["labels"].shape[1])


def _require_raw_rebuilt_admission(admission: SourceAdmission) -> None:
    checks = admission.provenance_report.get("checks")
    if not isinstance(checks, list) or not any("raw_rebuilt_v1" in str(value) for value in checks):
        raise RuntimeBridgeError("provenance PASS did not certify raw_rebuilt_v1 authority")
    adapter = admission.contract.get("adapter")
    if not isinstance(adapter, Mapping):
        raise RuntimeBridgeError("source adapter binding is missing")
    if adapter.get("identity_chain") not in {"OralData only", "OralData official JSON only"}:
        raise RuntimeBridgeError("source identity is not rebuilt from OralData")
    if not str(adapter.get("process_data_role", "")).startswith("none;"):
        raise RuntimeBridgeError("source adapter assigns a role to ProcessData")
    if admission.dataset == "nuswide":
        inventory = admission.contract.get("source_inventory")
        names = {
            Path(str(value.get("path", ""))).name
            for value in inventory if isinstance(value, Mapping)
        } if isinstance(inventory, list) else set()
        if "labels.nuswide.mat" in names or "labels.nuswide-tc21.mat" not in names:
            raise RuntimeBridgeError("NUS-WIDE source is not the raw TC21 21-hot authority")


def _array_shapes(rows: int, label_dim: int, split: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {
        "image": [rows, FEATURE_DIM],
        "text": [rows, FEATURE_DIM],
        "labels": [rows, label_dim],
        "row_ids": [rows],
        "indQ": [int(split["indQ"].size)],
        "indT": [int(split["indT"].size)],
        "indD": [int(split["indD"].size)],
    }


def _initialize_or_open_arrays(
    root: Path,
    contract: Mapping[str, Any],
    split: Mapping[str, np.ndarray],
    *,
    receipt_count: int,
) -> dict[str, np.memmap]:
    shapes = contract["array_shapes"]
    arrays: dict[str, np.memmap] = {}
    for name, (relative, dtype_name) in ARRAY_SPECS.items():
        path = root / relative
        expected_shape = tuple(int(value) for value in shapes[name])
        expected_dtype = np.dtype(dtype_name)
        if name in split:
            if path.exists():
                observed = np.load(path, mmap_mode="r", allow_pickle=False)
                if observed.dtype != expected_dtype or observed.shape != expected_shape or not np.array_equal(observed, split[name]):
                    raise RuntimeBridgeError(f"existing runtime {name} split array is poisoned")
            else:
                if receipt_count:
                    raise RuntimeBridgeError(f"runtime {name} is missing after receipts were committed")
                _atomic_save_npy(path, np.asarray(split[name], dtype=expected_dtype))
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            continue
        if not path.exists():
            if receipt_count:
                raise RuntimeBridgeError(f"runtime array {name} is missing after receipts were committed")
            path.parent.mkdir(parents=True, exist_ok=True)
            created = np.lib.format.open_memmap(path, mode="w+", dtype=expected_dtype, shape=expected_shape)
            created.flush()
            del created
        loaded = np.load(path, mmap_mode="r+", allow_pickle=False)
        if loaded.dtype != expected_dtype or loaded.shape != expected_shape:
            raise RuntimeBridgeError(f"runtime array {name} has wrong dtype or shape")
        arrays[name] = loaded
    return arrays


def _runtime_receipts(root: Path) -> list[Path]:
    directory = root / "receipts"
    if not directory.exists():
        return []
    paths = sorted(directory.glob("part-*.json"))
    if any(path.name != f"part-{index:06d}.json" for index, path in enumerate(paths)):
        raise RuntimeBridgeError("runtime receipt sequence has gaps, duplicates, or unexpected names")
    return paths


def _validate_runtime_receipts(
    root: Path,
    arrays: Mapping[str, np.ndarray],
    admission: SourceAdmission,
) -> tuple[int, str]:
    source_paths = _source_receipt_paths(admission)
    expected_start = 0
    previous = "0" * 64
    for expected_part, path in enumerate(_runtime_receipts(root)):
        receipt = load_json(path)
        body = dict(receipt)
        observed_chain = body.pop("chain_sha256", None)
        if sha256_json(body) != observed_chain:
            raise RuntimeBridgeError(f"runtime receipt chain digest mismatch: {path}")
        if receipt.get("previous_chain_sha256") != previous:
            raise RuntimeBridgeError("runtime receipt predecessor chain mismatch")
        if int(receipt.get("part", -1)) != expected_part or int(receipt.get("start", -1)) != expected_start:
            raise RuntimeBridgeError("runtime receipts have a row gap, overlap, or reorder")
        if expected_part >= len(source_paths):
            raise RuntimeBridgeError("runtime receipts exceed the admitted source")
        source_descriptor = admission.seal["receipts"][expected_part]
        if receipt.get("source_receipt") != source_descriptor:
            raise RuntimeBridgeError("runtime receipt is bound to a different source receipt")
        source_receipt, source_values = _load_source_part(
            admission, source_paths[expected_part], expected_part=expected_part
        )
        if receipt.get("source_npz_sha256") != source_receipt.get("npz_sha256") or receipt.get(
            "source_manifest_sha256"
        ) != source_receipt.get("manifest_sha256"):
            raise RuntimeBridgeError("runtime receipt source artifact binding mismatch")
        stop = int(receipt.get("stop", -1))
        if (
            stop <= expected_start
            or int(receipt.get("rows", -1)) != stop - expected_start
            or int(source_receipt.get("start", -1)) != expected_start
            or stop != int(source_receipt.get("stop", -1))
        ):
            raise RuntimeBridgeError("runtime receipt has an invalid stop row")
        expected_hashes = receipt.get("output_slice_sha256")
        if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {
            "image",
            "text",
            "labels",
            "row_ids",
        }:
            raise RuntimeBridgeError("runtime receipt omitted output slice hashes")
        for name in ("image", "text", "labels", "row_ids"):
            actual = numeric_sha256(np.asarray(arrays[name][expected_start:stop]))
            if expected_hashes.get(name) != actual:
                raise RuntimeBridgeError(f"runtime resume detected {name} same-shape poison")
            source_name = {
                "image": "image_features",
                "text": "text_features",
                "labels": "labels",
                "row_ids": "row_ids",
            }[name]
            if numeric_sha256(source_values[source_name]) != actual:
                raise RuntimeBridgeError(
                    f"runtime {name} rows differ from the re-admitted source shard"
                )
        expected_start = stop
        previous = str(observed_chain)
    return expected_start, previous


def _artifact_descriptor(root: Path, name: str, array: np.ndarray) -> dict[str, Any]:
    relative, _dtype = ARRAY_SPECS[name]
    path = root / relative
    stat = path.stat()
    return {
        "name": name,
        "path": relative,
        "size": stat.st_size,
        "sha256": sha256_file(path),
        "content_sha256": numeric_sha256(array),
        "dtype": array.dtype.name if array.dtype.kind != "S" else f"S{array.dtype.itemsize}",
        "shape": list(array.shape),
    }


def _finalize(
    root: Path,
    contract_record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    admission: SourceAdmission,
    source_split_row_ids: np.ndarray,
    final_chain: str,
) -> dict[str, Any]:
    row_ids = np.asarray(arrays["row_ids"])
    if not np.array_equal(row_ids, source_split_row_ids):
        raise RuntimeBridgeError("materialized row IDs differ from canonical split row IDs")
    if np.unique(row_ids).size != admission.rows:
        raise RuntimeBridgeError("materialized row IDs are missing or duplicated")
    if any(HEX64_RE.fullmatch(bytes(value)) is None for value in row_ids):
        raise RuntimeBridgeError("materialized row IDs are not lowercase SHA-256 S64 values")

    receipt_descriptors = []
    for path in _runtime_receipts(root):
        value = load_json(path)
        receipt_descriptors.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "chain_sha256": value["chain_sha256"],
            }
        )
    artifacts = [
        _artifact_descriptor(root, name, arrays[name]) for name in ARRAY_SPECS
    ]
    contract_path = root / "runtime_contract.json"
    contract_descriptor = {
        "path": "runtime_contract.json",
        "size": contract_path.stat().st_size,
        "sha256": sha256_file(contract_path),
    }
    inventory = sorted(
        [contract_descriptor]
        + [{key: item[key] for key in ("path", "size", "sha256")} for item in artifacts]
        + [{key: item[key] for key in ("path", "size", "sha256")} for item in receipt_descriptors],
        key=lambda value: value["path"],
    )
    manifest = {
        "schema": SCHEMA_VERSION,
        "status": "COMPLETE",
        "dataset": admission.dataset,
        "rows": admission.rows,
        "feature_dim": FEATURE_DIM,
        "label_dim": int(contract_record["label_dim"]),
        "source_seal": dict(admission.seal),
        "source_seal_sha256": admission.seal["source_seal_sha256"],
        "runtime_contract": contract_descriptor,
        "runtime_contract_sha256": contract_record["runtime_contract_sha256"],
        "artifacts": artifacts,
        "receipts": receipt_descriptors,
        "final_chain_sha256": final_chain,
        "inventory": inventory,
    }
    atomic_write_json(root / "runtime_manifest.json", manifest)
    complete_body = {
        "schema": SCHEMA_VERSION,
        "status": "COMPLETE",
        "dataset": admission.dataset,
        "rows": admission.rows,
        "parts": admission.shards,
        "source_seal_sha256": admission.seal["source_seal_sha256"],
        "runtime_manifest_sha256": sha256_file(root / "runtime_manifest.json"),
        "final_chain_sha256": final_chain,
    }
    complete = {**complete_body, "complete_sha256": sha256_json(complete_body)}
    atomic_write_json(root / "complete.json", complete)
    return complete


def materialize_runtime(
    source_bundle: Path,
    output_root: Path,
    *,
    process_data_root: Path | None = None,
    max_new_parts: int | None = None,
    _test_allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Admit and stream one source bundle into an independent NPY runtime.

    ``max_new_parts`` is an operational checkpoint control.  Omitting it
    materializes all remaining shards; a bounded call returns ``IN_PROGRESS``
    and a later identical call resumes only after revalidating every receipt.
    """

    if max_new_parts is not None and max_new_parts < 0:
        raise RuntimeBridgeError("max_new_parts must be non-negative")
    admission = admit_source_bundle(source_bundle, process_data_root=process_data_root)
    _assert_admission_seal_unchanged(admission)
    _require_raw_rebuilt_admission(admission)
    adapter = admission.contract["adapter"]
    data_root = Path(str(adapter["data_root"])).expanduser().resolve(strict=True)
    forbidden = [admission.bundle, data_root / "OralData", data_root / "ProcessData"]
    if process_data_root is not None:
        forbidden.append(Path(process_data_root))
    root = require_disjoint(Path(output_root), forbidden, field="runtime output")
    root.mkdir(parents=True, exist_ok=True)

    if (root / "complete.json").is_file():
        manifest = verify_runtime_directory(
            root,
            process_data_root=process_data_root,
            _admission=admission,
            _test_allow_synthetic=_test_allow_synthetic,
        )
        return {
            "status": "COMPLETE",
            "dataset": manifest["dataset"],
            "rows": manifest["rows"],
            "parts": len(manifest["receipts"]),
            "output_root": str(root),
            "resumed": True,
        }

    feature_dim, label_dim = _geometry_from_source(admission)
    require_geometry(
        admission.dataset,
        rows=admission.rows,
        feature_dim=feature_dim,
        label_dim=label_dim,
        allow_test_dataset=_test_allow_synthetic,
    )
    split, split_row_ids = _load_source_split(admission)
    require_split_arrays(split, dataset=admission.dataset, rows=admission.rows)
    split_summary = admission.split.get("summary")
    if not isinstance(split_summary, Mapping) or split_summary.get("algorithm") != SPLIT_ALGORITHM or split_summary.get("seed") != SPLIT_SEED:
        raise RuntimeBridgeError("source split algorithm and seed are not frozen raw_rebuilt_v1")

    contract_body = {
        "schema": SCHEMA_VERSION,
        "dataset": admission.dataset,
        "rows": admission.rows,
        "feature_dim": feature_dim,
        "label_dim": label_dim,
        "array_shapes": _array_shapes(admission.rows, label_dim, split),
        "array_files": {name: ARRAY_SPECS[name][0] for name in ARRAY_SPECS},
        "array_dtypes": {name: ARRAY_SPECS[name][1] for name in ARRAY_SPECS},
        "source_seal": dict(admission.seal),
        "source_seal_sha256": admission.seal["source_seal_sha256"],
        "split_content_sha256": {name: numeric_sha256(split[name]) for name in split},
        "ordered_row_ids_sha256": numeric_sha256(split_row_ids),
        "materialization": "contiguous source receipt order to NPY memmap; no row remapping",
    }
    contract_record = {
        **contract_body,
        "runtime_contract_sha256": sha256_json(contract_body),
    }
    contract_path = root / "runtime_contract.json"
    if contract_path.exists():
        if load_json(contract_path) != contract_record:
            raise RuntimeBridgeError("resume contract differs from the admitted source or geometry")
    else:
        existing = [path for path in root.rglob("*") if path.is_file()]
        if existing:
            raise RuntimeBridgeError("non-empty output without a runtime contract fails closed")
        atomic_write_json(contract_path, contract_record)

    runtime_receipts = _runtime_receipts(root)
    arrays = _initialize_or_open_arrays(
        root, contract_record, split, receipt_count=len(runtime_receipts)
    )
    allowed_incomplete = {
        "runtime_contract.json",
        *(relative for relative, _dtype in ARRAY_SPECS.values()),
        *(path.relative_to(root).as_posix() for path in _runtime_receipts(root)),
    }
    actual_incomplete = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_incomplete != allowed_incomplete:
        raise RuntimeBridgeError("in-progress runtime contains missing or unbound extra files")
    next_row, previous_chain = _validate_runtime_receipts(root, arrays, admission)
    next_part = len(runtime_receipts)
    source_receipts = _source_receipt_paths(admission)
    limit = len(source_receipts) if max_new_parts is None else min(
        len(source_receipts), next_part + max_new_parts
    )
    for part in range(next_part, limit):
        source_receipt, values = _load_source_part(
            admission, source_receipts[part], expected_part=part
        )
        start = int(source_receipt["start"])
        stop = int(source_receipt["stop"])
        if start != next_row:
            raise RuntimeBridgeError("source receipt order does not match runtime resume row")
        if values["labels"].shape[1] != label_dim:
            raise RuntimeBridgeError("source label dimension drifted across shards")
        arrays["image"][start:stop] = values["image_features"]
        arrays["text"][start:stop] = values["text_features"]
        arrays["labels"][start:stop] = values["labels"]
        arrays["row_ids"][start:stop] = values["row_ids"]
        for name in ("image", "text", "labels", "row_ids"):
            arrays[name].flush()
        output_hashes = {
            name: numeric_sha256(np.asarray(arrays[name][start:stop]))
            for name in ("image", "text", "labels", "row_ids")
        }
        runtime_receipt_body = {
            "schema": SCHEMA_VERSION,
            "part": part,
            "start": start,
            "stop": stop,
            "rows": stop - start,
            "source_receipt": admission.seal["receipts"][part],
            "source_npz_sha256": source_receipt["npz_sha256"],
            "source_manifest_sha256": source_receipt["manifest_sha256"],
            "output_slice_sha256": output_hashes,
            "previous_chain_sha256": previous_chain,
        }
        runtime_receipt = {
            **runtime_receipt_body,
            "chain_sha256": sha256_json(runtime_receipt_body),
        }
        atomic_write_json(root / "receipts" / f"part-{part:06d}.json", runtime_receipt)
        previous_chain = runtime_receipt["chain_sha256"]
        next_row = stop

    if limit < len(source_receipts):
        return {
            "status": "IN_PROGRESS",
            "dataset": admission.dataset,
            "rows_committed": next_row,
            "rows_total": admission.rows,
            "parts_committed": limit,
            "parts_total": len(source_receipts),
            "source_seal_sha256": admission.seal["source_seal_sha256"],
            "output_root": str(root),
        }
    if next_row != admission.rows:
        raise RuntimeBridgeError("source receipts ended before the declared row count")
    _assert_admission_seal_unchanged(admission)
    complete = _finalize(
        root,
        contract_record,
        arrays,
        admission,
        split_row_ids,
        previous_chain,
    )
    verify_runtime_directory(
        root,
        process_data_root=process_data_root,
        _admission=admission,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    return {
        "status": "COMPLETE",
        "dataset": admission.dataset,
        "rows": admission.rows,
        "parts": admission.shards,
        "source_seal_sha256": admission.seal["source_seal_sha256"],
        "complete_sha256": complete["complete_sha256"],
        "output_root": str(root),
        "resumed": next_part > 0,
    }


def verify_runtime_directory(
    runtime_root: Path,
    *,
    process_data_root: Path | None = None,
    _admission: SourceAdmission | None = None,
    _test_allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Rehash the exact runtime inventory and, by default, its source bundle."""

    try:
        root = Path(runtime_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError(f"runtime directory does not exist: {runtime_root}") from error
    if not root.is_dir():
        raise RuntimeBridgeError("runtime path must be a directory")
    early_forbidden = [] if process_data_root is None else [Path(process_data_root)]
    require_disjoint(root, early_forbidden, field="runtime directory")
    for required in ("runtime_contract.json", "runtime_manifest.json", "complete.json"):
        if not (root / required).is_file():
            raise RuntimeBridgeError(f"runtime is incomplete; missing {required}")
    contract = load_json(root / "runtime_contract.json")
    contract_body = dict(contract)
    observed_contract_sha = contract_body.pop("runtime_contract_sha256", None)
    if sha256_json(contract_body) != observed_contract_sha:
        raise RuntimeBridgeError("runtime contract digest mismatch")
    manifest = load_json(root / "runtime_manifest.json")
    complete = load_json(root / "complete.json")
    complete_body = dict(complete)
    observed_complete_sha = complete_body.pop("complete_sha256", None)
    if sha256_json(complete_body) != observed_complete_sha:
        raise RuntimeBridgeError("runtime complete marker digest mismatch")
    if sha256_file(root / "runtime_manifest.json") != complete.get("runtime_manifest_sha256"):
        raise RuntimeBridgeError("runtime manifest changed after completion")
    if manifest.get("schema") != SCHEMA_VERSION or manifest.get("status") != "COMPLETE":
        raise RuntimeBridgeError("runtime manifest schema/status is invalid")
    if manifest.get("runtime_contract_sha256") != observed_contract_sha:
        raise RuntimeBridgeError("runtime manifest is bound to another contract")
    contract_descriptor = manifest.get("runtime_contract")
    if not isinstance(contract_descriptor, Mapping) or contract_descriptor.get("path") != "runtime_contract.json":
        raise RuntimeBridgeError("runtime manifest contract descriptor is invalid")
    if contract_descriptor.get("size") != (root / "runtime_contract.json").stat().st_size or contract_descriptor.get(
        "sha256"
    ) != sha256_file(root / "runtime_contract.json"):
        raise RuntimeBridgeError("runtime contract descriptor does not bind runtime_contract.json")
    if manifest.get("source_seal") != contract.get("source_seal"):
        raise RuntimeBridgeError("runtime contract and manifest disagree on source seal")
    if complete.get("source_seal_sha256") != manifest.get("source_seal_sha256"):
        raise RuntimeBridgeError("runtime completion is bound to another source seal")
    for key in ("schema", "status", "dataset", "rows"):
        if complete.get(key) != manifest.get(key):
            raise RuntimeBridgeError(f"runtime complete and manifest disagree on {key}")

    dataset = str(manifest.get("dataset", ""))
    rows = int(manifest.get("rows", -1))
    feature_dim = int(manifest.get("feature_dim", -1))
    label_dim = int(manifest.get("label_dim", -1))
    require_geometry(
        dataset,
        rows=rows,
        feature_dim=feature_dim,
        label_dim=label_dim,
        allow_test_dataset=_test_allow_synthetic,
    )
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise RuntimeBridgeError("runtime manifest has no exact inventory")
    declared_paths = []
    for descriptor in inventory:
        if not isinstance(descriptor, Mapping):
            raise RuntimeBridgeError("runtime inventory descriptor is invalid")
        relative = str(descriptor.get("path", ""))
        path = _relative_file(root, relative)
        if path.stat().st_size != descriptor.get("size") or sha256_file(path) != descriptor.get("sha256"):
            raise RuntimeBridgeError(f"runtime inventory poison detected: {relative}")
        declared_paths.append(relative)
    if declared_paths != sorted(set(declared_paths)):
        raise RuntimeBridgeError("runtime inventory paths are not strictly sorted and unique")
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    expected_paths = sorted(declared_paths + ["runtime_manifest.json", "complete.json"])
    if actual_paths != expected_paths:
        raise RuntimeBridgeError(
            f"runtime has missing or unbound extra files; declared={expected_paths}, actual={actual_paths}"
        )

    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, list) or {value.get("name") for value in artifact_records if isinstance(value, Mapping)} != set(ARRAY_SPECS):
        raise RuntimeBridgeError("runtime artifact set is incomplete")
    opened: dict[str, np.ndarray] = {}
    for descriptor in artifact_records:
        name = str(descriptor["name"])
        if descriptor.get("path") != ARRAY_SPECS[name][0]:
            raise RuntimeBridgeError(f"runtime array {name} uses a non-canonical path")
        path = _relative_file(root, str(descriptor["path"]))
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError(f"cannot load runtime array {name}") from error
        expected_dtype = np.dtype(ARRAY_SPECS[name][1])
        if array.dtype != expected_dtype or list(array.shape) != descriptor.get("shape"):
            raise RuntimeBridgeError(f"runtime array {name} dtype/shape mismatch")
        declared_dtype = array.dtype.name if array.dtype.kind != "S" else f"S{array.dtype.itemsize}"
        if descriptor.get("dtype") != declared_dtype:
            raise RuntimeBridgeError(f"runtime array {name} descriptor dtype mismatch")
        if numeric_sha256(array) != descriptor.get("content_sha256"):
            raise RuntimeBridgeError(f"runtime array {name} decoded-content poison detected")
        opened[name] = array
    require_split_arrays(
        {name: opened[name] for name in ("indQ", "indT", "indD")},
        dataset=dataset,
        rows=rows,
    )
    if np.unique(opened["row_ids"]).size != rows:
        raise RuntimeBridgeError("runtime row IDs are not unique")
    _require_finite_feature_memmaps(opened["image"], opened["text"])
    _require_binary_label_memmap(opened["labels"])

    admission = _admission
    if admission is None:
        source = manifest.get("source_seal")
        if not isinstance(source, Mapping) or not isinstance(source.get("bundle_path"), str):
            raise RuntimeBridgeError("runtime source seal has no bundle path")
        admission = admit_source_bundle(
            Path(source["bundle_path"]), process_data_root=process_data_root
        )
    if admission.seal != manifest.get("source_seal"):
        raise RuntimeBridgeError("admitted source no longer matches the runtime source seal")
    _assert_admission_seal_unchanged(admission)
    _require_raw_rebuilt_admission(admission)
    if admission.dataset != dataset or admission.rows != rows:
        raise RuntimeBridgeError("runtime geometry differs from its re-admitted source")
    adapter = admission.contract.get("adapter")
    if not isinstance(adapter, Mapping):
        raise RuntimeBridgeError("re-admitted source has no adapter contract")
    data_root = Path(str(adapter.get("data_root", ""))).expanduser().resolve(strict=True)
    forbidden = [admission.bundle, data_root / "OralData", data_root / "ProcessData"]
    if process_data_root is not None:
        forbidden.append(Path(process_data_root))
    require_disjoint(root, forbidden, field="runtime directory")
    if int(complete.get("parts", -1)) != admission.shards:
        raise RuntimeBridgeError("runtime complete part count differs from source")
    verified_rows, verified_chain = _validate_runtime_receipts(
        root, opened, admission
    )
    if verified_rows != rows or len(_runtime_receipts(root)) != admission.shards:
        raise RuntimeBridgeError("runtime receipts do not cover the complete source")
    if verified_chain != manifest.get("final_chain_sha256") or verified_chain != complete.get(
        "final_chain_sha256"
    ):
        raise RuntimeBridgeError("runtime final receipt chain differs from completion")
    source_split, source_row_ids = _load_source_split(admission)
    for name in ("indQ", "indT", "indD"):
        if not np.array_equal(opened[name], source_split[name]):
            raise RuntimeBridgeError(f"runtime {name} differs from its admitted source split")
    if not np.array_equal(opened["row_ids"], source_row_ids):
        raise RuntimeBridgeError("runtime row IDs differ from the source split ledger")
    if contract.get("ordered_row_ids_sha256") != numeric_sha256(source_row_ids):
        raise RuntimeBridgeError("runtime contract row-ID hash differs from source split")
    return manifest


def verify_metric_runtime_directory(
    runtime_root: Path,
    *,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Fully verify scoring bytes using the already frozen source seal.

    Runtime materialization performs the expensive OralData reconstruction and
    records its independent provenance PASS in ``source_seal``.  A metric
    worker is downstream of a verified rank plan bound to that seal.  This
    entry point reopens the sealed trace in bounded memory, then delegates to
    the same byte-for-byte runtime and source-shard verification used by
    :func:`verify_runtime_directory`.
    """

    try:
        root = Path(runtime_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError(
            f"runtime directory does not exist: {runtime_root}"
        ) from error
    manifest_path = root / "runtime_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeBridgeError("runtime is incomplete; missing runtime_manifest.json")
    manifest = load_json(manifest_path)
    source = manifest.get("source_seal")
    if not isinstance(source, Mapping) or not isinstance(
        source.get("bundle_path"), str
    ):
        raise RuntimeBridgeError("runtime source seal has no bundle path")
    admission = reopen_frozen_source_bundle(
        Path(source["bundle_path"]),
        source,
        process_data_root=process_data_root,
    )
    return verify_runtime_directory(
        root,
        process_data_root=process_data_root,
        _admission=admission,
        _test_allow_synthetic=_test_allow_synthetic,
    )


def _npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read only an NPY header; used to keep labels closed during rank freeze."""

    try:
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise RuntimeBridgeError(f"unsupported NPY version for {path}: {version}")
    except (OSError, ValueError) as error:
        if isinstance(error, RuntimeBridgeError):
            raise
        raise RuntimeBridgeError(f"cannot read NPY header: {path}") from error
    return tuple(int(value) for value in shape), bool(fortran), np.dtype(dtype)


def verify_label_free_runtime_directory(
    runtime_root: Path,
    *,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Verify rank inputs without decoding, mapping, or hashing label bytes.

    Full source/provenance validation is mandatory during materialization and
    again when labels are opened for training or metrics.  This narrower gate
    preserves the temporal ``labels_loaded_during_freeze=false`` boundary: it
    verifies the completed seal plus image/text/split/row-ID content, and reads
    only the label NPY header and file size.
    """

    try:
        root = Path(runtime_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeBridgeError(f"runtime directory does not exist: {runtime_root}") from error
    if not root.is_dir():
        raise RuntimeBridgeError("runtime path must be a directory")
    require_disjoint(
        root,
        [] if process_data_root is None else [Path(process_data_root)],
        field="runtime directory",
    )
    contract_path = root / "runtime_contract.json"
    manifest_path = root / "runtime_manifest.json"
    complete_path = root / "complete.json"
    for path in (contract_path, manifest_path, complete_path):
        if not path.is_file():
            raise RuntimeBridgeError(f"runtime is incomplete; missing {path.name}")
    contract = load_json(contract_path)
    contract_body = dict(contract)
    contract_sha = contract_body.pop("runtime_contract_sha256", None)
    if sha256_json(contract_body) != contract_sha:
        raise RuntimeBridgeError("runtime contract digest mismatch")
    manifest = load_json(manifest_path)
    complete = load_json(complete_path)
    complete_body = dict(complete)
    complete_sha = complete_body.pop("complete_sha256", None)
    if sha256_json(complete_body) != complete_sha:
        raise RuntimeBridgeError("runtime complete marker digest mismatch")
    if sha256_file(manifest_path) != complete.get("runtime_manifest_sha256"):
        raise RuntimeBridgeError("runtime manifest changed after completion")
    if manifest.get("schema") != SCHEMA_VERSION or manifest.get("status") != "COMPLETE":
        raise RuntimeBridgeError("runtime manifest schema/status is invalid")
    if manifest.get("runtime_contract_sha256") != contract_sha:
        raise RuntimeBridgeError("runtime manifest is bound to another contract")
    for key in ("schema", "status", "dataset", "rows"):
        if complete.get(key) != manifest.get(key):
            raise RuntimeBridgeError(f"runtime complete and manifest disagree on {key}")
    seal = manifest.get("source_seal")
    if not isinstance(seal, Mapping) or seal != contract.get("source_seal"):
        raise RuntimeBridgeError("runtime source seal is absent or inconsistent")
    seal_body = dict(seal)
    seal_sha = seal_body.pop("source_seal_sha256", None)
    if sha256_json(seal_body) != seal_sha or seal_sha != manifest.get("source_seal_sha256"):
        raise RuntimeBridgeError("runtime source seal digest mismatch")
    report = seal.get("provenance_report")
    if (
        not isinstance(report, Mapping)
        or report.get("status") != "PASS"
        or not isinstance(report.get("checks"), list)
        or not any("raw_rebuilt_v1" in str(value) for value in report["checks"])
    ):
        raise RuntimeBridgeError("runtime source seal lacks a raw_rebuilt_v1 provenance PASS")
    boundary_values = [
        seal.get("bundle_path"),
        seal.get("oral_data_root"),
        seal.get("process_data_root"),
    ]
    if not all(isinstance(value, str) and value for value in boundary_values):
        raise RuntimeBridgeError("runtime source seal lacks independent output boundaries")
    boundary_paths = [Path(str(value)) for value in boundary_values]
    if process_data_root is not None:
        boundary_paths.append(Path(process_data_root))
    require_disjoint(root, boundary_paths, field="runtime directory")

    dataset = str(manifest.get("dataset", ""))
    rows = int(manifest.get("rows", -1))
    label_dim = int(manifest.get("label_dim", -1))
    require_geometry(
        dataset,
        rows=rows,
        feature_dim=int(manifest.get("feature_dim", -1)),
        label_dim=label_dim,
        allow_test_dataset=_test_allow_synthetic,
    )
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise RuntimeBridgeError("runtime manifest has no inventory")
    inventory_by_path: dict[str, Mapping[str, Any]] = {}
    for raw in inventory:
        if not isinstance(raw, Mapping):
            raise RuntimeBridgeError("runtime inventory descriptor is invalid")
        relative = str(raw.get("path", ""))
        if relative in inventory_by_path:
            raise RuntimeBridgeError("runtime inventory contains duplicate paths")
        path = _relative_file(root, relative)
        if path.stat().st_size != raw.get("size"):
            raise RuntimeBridgeError(f"runtime inventory size mismatch: {relative}")
        # The rank-freeze process deliberately does not read labels.npy bytes.
        if relative != ARRAY_SPECS["labels"][0] and sha256_file(path) != raw.get("sha256"):
            raise RuntimeBridgeError(f"runtime rank-input poison detected: {relative}")
        inventory_by_path[relative] = raw
    actual = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    expected = sorted([*inventory_by_path, "runtime_manifest.json", "complete.json"])
    if actual != expected:
        raise RuntimeBridgeError("runtime has missing or unbound extra files")

    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, list):
        raise RuntimeBridgeError("runtime manifest has no artifact records")
    by_name = {
        str(value.get("name")): value for value in artifact_records if isinstance(value, Mapping)
    }
    if set(by_name) != set(ARRAY_SPECS):
        raise RuntimeBridgeError("runtime artifact set is incomplete")
    label_record = by_name["labels"]
    if label_record.get("path") != ARRAY_SPECS["labels"][0]:
        raise RuntimeBridgeError("runtime label path is non-canonical")
    label_path = _relative_file(root, ARRAY_SPECS["labels"][0])
    label_shape, label_fortran, label_dtype = _npy_header(label_path)
    if label_fortran or label_shape != (rows, label_dim) or label_dtype != np.dtype("uint8"):
        raise RuntimeBridgeError("runtime label NPY header is not frozen uint8 [N,C]")
    if label_record.get("size") != label_path.stat().st_size:
        raise RuntimeBridgeError("runtime label descriptor size mismatch")

    opened: dict[str, np.ndarray] = {}
    for name in ("image", "text", "row_ids", "indQ", "indT", "indD"):
        descriptor = by_name[name]
        relative, dtype_name = ARRAY_SPECS[name]
        if descriptor.get("path") != relative:
            raise RuntimeBridgeError(f"runtime {name} path is non-canonical")
        array = np.load(_relative_file(root, relative), mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(dtype_name) or list(array.shape) != descriptor.get("shape"):
            raise RuntimeBridgeError(f"runtime {name} dtype/shape mismatch")
        if numeric_sha256(array) != descriptor.get("content_sha256"):
            raise RuntimeBridgeError(f"runtime {name} decoded-content poison detected")
        opened[name] = array
    if opened["image"].shape != (rows, FEATURE_DIM) or opened["text"].shape != (rows, FEATURE_DIM):
        raise RuntimeBridgeError("runtime rank feature geometry is not [N,512]")
    _require_finite_feature_memmaps(opened["image"], opened["text"])
    require_split_arrays(
        {name: opened[name] for name in ("indQ", "indT", "indD")},
        dataset=dataset,
        rows=rows,
    )
    if np.unique(opened["row_ids"]).size != rows:
        raise RuntimeBridgeError("runtime rank row IDs are not unique")

    expected_start = 0
    previous = "0" * 64
    runtime_receipts = _runtime_receipts(root)
    source_receipts = seal.get("receipts")
    if not isinstance(source_receipts, list) or len(source_receipts) != len(runtime_receipts):
        raise RuntimeBridgeError("runtime/source receipt counts disagree")
    for part, path in enumerate(runtime_receipts):
        receipt = load_json(path)
        body = dict(receipt)
        chain = body.pop("chain_sha256", None)
        stop = int(receipt.get("stop", -1))
        if (
            sha256_json(body) != chain
            or receipt.get("previous_chain_sha256") != previous
            or int(receipt.get("part", -1)) != part
            or int(receipt.get("start", -1)) != expected_start
            or int(receipt.get("rows", -1)) != stop - expected_start
            or receipt.get("source_receipt") != source_receipts[part]
        ):
            raise RuntimeBridgeError("runtime rank-input receipt chain is invalid")
        hashes = receipt.get("output_slice_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != {"image", "text", "labels", "row_ids"}:
            raise RuntimeBridgeError("runtime receipt output hashes are incomplete")
        for name in ("image", "text", "row_ids"):
            if numeric_sha256(np.asarray(opened[name][expected_start:stop])) != hashes[name]:
                raise RuntimeBridgeError(f"runtime rank-input {name} slice differs from receipt")
        expected_start = stop
        previous = str(chain)
    if (
        expected_start != rows
        or previous != manifest.get("final_chain_sha256")
        or previous != complete.get("final_chain_sha256")
    ):
        raise RuntimeBridgeError("runtime rank-input receipts do not cover completion")
    return manifest
