"""Strict importer for sealed fixed-feature baseline code artifacts.

This module is used only during the freeze phase.  The rank worker never
imports it and therefore never imports the baseline training/checkpoint stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .codes import (
    BITS,
    CODE_STATE_SCHEMA,
    MODALITIES,
    SCOPES,
    _array_relative,
    _write_encoding_receipt,
    open_code_state,
    pack_bipolar_codes,
)
from .integrity import (
    StreamingIntegrityError,
    atomic_save_npy,
    atomic_write_json,
    load_json,
    numeric_sha256,
    production_code_inventory,
    reject_unsafe_output_path,
    require_disjoint_paths,
    require_dataset_label_geometry,
    require_no_link_components,
    sha256_file,
    sha256_json,
)


BASELINE_CODE_ARTIFACT_SCHEMA = "raw_rebuilt_fixed_feature_baseline_codes_v1"
_ARRAY_FILES = {
    "image_codes": "image_codes.npy",
    "text_codes": "text_codes.npy",
    "row_ids": "row_ids.npy",
    "indT": "indT.npy",
    "indQ": "indQ.npy",
    "indD": "indD.npy",
}
_DESCRIPTOR_KEYS = {
    "path",
    "dtype",
    "shape",
    "size",
    "file_sha256",
    "numeric_sha256",
}
_RANK_CONTRACT_KEYS = {
    "schema",
    "status",
    "labels_loaded_during_freeze",
    "method",
    "bits",
    "seed",
    "source_seal_sha256",
    "fit_artifact_sha256",
    "checkpoint_sha256",
    "run_contract_sha256",
    "full_row_ids_numeric_sha256",
    "split_binding_sha256",
    "train_idx_numeric_sha256",
    "query_idx_numeric_sha256",
    "database_idx_numeric_sha256",
    "image_codes_numeric_sha256",
    "text_codes_numeric_sha256",
    "rank_contract_sha256",
}
_CHECKPOINT_FILES = {"manifest.json", "checkpoint.pt", "code_receipt.json"}


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StreamingIntegrityError(f"{field} must be one lowercase SHA-256 digest")
    return text


def _regular_artifact_root(root: Path) -> Path:
    declared = require_no_link_components(root, field="baseline code artifact")
    if declared.is_symlink() or not declared.is_dir():
        raise StreamingIntegrityError("baseline code artifact must be a regular directory")
    path = declared.resolve(strict=True)
    expected = {"manifest.json", *_ARRAY_FILES.values()}
    actual = {item.relative_to(path).as_posix() for item in path.rglob("*")}
    if actual != expected:
        raise StreamingIntegrityError("baseline code artifact file inventory changed")
    for relative in expected:
        target = path / relative
        if target.is_symlink() or not target.is_file() or target.parent != path:
            raise StreamingIntegrityError("baseline code artifact files must be flat non-symlinks")
    return path


def _checkpoint_snapshot(
    root: Path,
) -> tuple[Path, dict[str, tuple[int, str]], Mapping[str, Any]]:
    declared = require_no_link_components(root, field="baseline checkpoint")
    if declared.is_symlink() or not declared.is_dir():
        raise StreamingIntegrityError("baseline checkpoint must be a regular directory")
    path = declared.resolve(strict=True)
    actual = {item.relative_to(path).as_posix() for item in path.rglob("*")}
    if actual != _CHECKPOINT_FILES:
        raise StreamingIntegrityError("baseline checkpoint file inventory changed")
    snapshot: dict[str, tuple[int, str]] = {}
    for relative in sorted(_CHECKPOINT_FILES):
        target = path / relative
        if target.is_symlink() or not target.is_file() or target.parent != path:
            raise StreamingIntegrityError(
                "baseline checkpoint files must be flat regular non-symlinks"
            )
        snapshot[relative] = (target.stat().st_size, sha256_file(target))
    receipt = load_json(path / "code_receipt.json")
    _require_checkpoint_unchanged(path, snapshot)
    return path, snapshot, receipt


def _require_checkpoint_unchanged(
    root: Path, snapshot: Mapping[str, tuple[int, str]]
) -> None:
    require_no_link_components(root, field="baseline checkpoint")
    actual = {item.relative_to(root).as_posix() for item in root.rglob("*")}
    if actual != _CHECKPOINT_FILES or set(snapshot) != _CHECKPOINT_FILES:
        raise StreamingIntegrityError("baseline checkpoint inventory changed during import")
    for relative, (size, digest) in snapshot.items():
        target = root / relative
        if (
            target.is_symlink()
            or not target.is_file()
            or target.parent != root
            or target.stat().st_size != size
            or sha256_file(target) != digest
        ):
            raise StreamingIntegrityError("baseline checkpoint changed during import")


def _snapshot_array(
    root: Path,
    name: str,
    descriptor: Mapping[str, Any],
) -> np.ndarray:
    if set(descriptor) != _DESCRIPTOR_KEYS:
        raise StreamingIntegrityError(f"baseline {name} descriptor keys changed")
    expected_name = _ARRAY_FILES[name]
    if descriptor.get("path") != expected_name:
        raise StreamingIntegrityError(f"baseline {name} path changed")
    expected_dtype = {
        "image_codes": np.dtype(np.int8).str,
        "text_codes": np.dtype(np.int8).str,
        "row_ids": np.dtype("S64").str,
        "indT": np.dtype(np.int64).str,
        "indQ": np.dtype(np.int64).str,
        "indD": np.dtype(np.int64).str,
    }[name]
    shape = descriptor.get("shape")
    if descriptor.get("dtype") != expected_dtype or not isinstance(shape, list) or len(shape) != 1 + int(
        name in {"image_codes", "text_codes"}
    ):
        raise StreamingIntegrityError(f"baseline {name} descriptor geometry is invalid")
    target = root / expected_name
    before_size = target.stat().st_size
    before_sha = sha256_file(target)
    if before_size != descriptor.get("size") or before_sha != descriptor.get("file_sha256"):
        raise StreamingIntegrityError(f"baseline {name} bytes changed")
    mapped = np.load(target, mmap_mode="r", allow_pickle=False)
    try:
        if mapped.dtype.str != descriptor.get("dtype") or list(mapped.shape) != descriptor.get(
            "shape"
        ):
            raise StreamingIntegrityError(f"baseline {name} geometry changed")
        if numeric_sha256(mapped) != descriptor.get("numeric_sha256"):
            raise StreamingIntegrityError(f"baseline {name} numeric content changed")
        snapshot = np.array(mapped, dtype=mapped.dtype, order="C", copy=True)
    finally:
        mmap = getattr(mapped, "_mmap", None)
        if mmap is not None:
            mmap.close()
    if (
        target.stat().st_size != before_size
        or sha256_file(target) != before_sha
        or numeric_sha256(snapshot) != descriptor.get("numeric_sha256")
    ):
        raise StreamingIntegrityError(f"baseline {name} changed during snapshot")
    return snapshot


def _validate_row_ids(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 1 or rows.dtype != np.dtype("S64"):
        raise StreamingIntegrityError("baseline row_ids must be one-dimensional S64")
    if np.unique(rows).size != rows.size:
        raise StreamingIntegrityError("baseline row_ids contain duplicates")
    for raw in rows:
        try:
            _require_sha256(bytes(raw).decode("ascii"), field="baseline row ID")
        except UnicodeDecodeError as error:
            raise StreamingIntegrityError("baseline row ID is not ASCII") from error
    return rows


def _validate_split(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    indices = np.asarray(value)
    if indices.ndim != 1 or indices.dtype != np.int64:
        raise StreamingIntegrityError(f"baseline {name} must be one-dimensional int64")
    if indices.size and (
        int(indices[0]) < 0
        or int(indices[-1]) >= rows
        or np.unique(indices).size != indices.size
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise StreamingIntegrityError(f"baseline {name} is not a valid canonical split")
    return indices


def _load_verified_baseline_artifact(
    artifact_root: Path,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], dict[str, np.ndarray]]:
    root = _regular_artifact_root(artifact_root)
    manifest = load_json(root / "manifest.json")
    if set(manifest) != {
        "schema",
        "status",
        "labels_loaded_during_freeze",
        "rank_contract",
        "arrays",
    }:
        raise StreamingIntegrityError("baseline code manifest keys changed")
    if (
        manifest.get("schema") != BASELINE_CODE_ARTIFACT_SCHEMA
        or manifest.get("status") != "rank_state_frozen"
        or manifest.get("labels_loaded_during_freeze") is not False
    ):
        raise StreamingIntegrityError("baseline code artifact is not label-free and frozen")
    contract = manifest.get("rank_contract")
    if not isinstance(contract, Mapping) or set(contract) != _RANK_CONTRACT_KEYS:
        raise StreamingIntegrityError("baseline rank contract keys changed")
    contract_body = {
        key: contract[key] for key in contract if key != "rank_contract_sha256"
    }
    if (
        contract.get("schema") != BASELINE_CODE_ARTIFACT_SCHEMA
        or contract.get("status") != "rank_state_frozen"
        or contract.get("labels_loaded_during_freeze") is not False
        or contract.get("rank_contract_sha256") != sha256_json(contract_body)
    ):
        raise StreamingIntegrityError("baseline rank contract hash/status changed")
    bits = int(contract.get("bits", -1))
    if bits not in BITS:
        raise StreamingIntegrityError("baseline rank contract bit width is unsupported")
    for key in _RANK_CONTRACT_KEYS:
        if key.endswith("sha256"):
            _require_sha256(contract[key], field=f"baseline contract {key}")
    arrays_meta = manifest.get("arrays")
    if not isinstance(arrays_meta, Mapping) or set(arrays_meta) != set(_ARRAY_FILES):
        raise StreamingIntegrityError("baseline array inventory changed")
    arrays = {
        name: _snapshot_array(root, name, arrays_meta[name]) for name in _ARRAY_FILES
    }
    row_ids = _validate_row_ids(arrays["row_ids"])
    rows = int(row_ids.size)
    if rows < 1:
        raise StreamingIntegrityError("baseline artifact has no rows")
    for name in ("image_codes", "text_codes"):
        value = arrays[name]
        if value.shape != (rows, bits) or value.dtype != np.int8:
            raise StreamingIntegrityError(f"baseline {name} geometry changed")
        if not np.all(np.isin(value, (-1, 1))):
            raise StreamingIntegrityError(f"baseline {name} is not bipolar {{-1,+1}}")
        if numeric_sha256(value) != contract[f"{name}_numeric_sha256"]:
            raise StreamingIntegrityError(f"baseline {name} differs from rank contract")
    train = _validate_split(arrays["indT"], name="indT", rows=rows)
    query = _validate_split(arrays["indQ"], name="indQ", rows=rows)
    database = _validate_split(arrays["indD"], name="indD", rows=rows)
    if query.size < 1 or database.size < 1 or train.size < 1:
        raise StreamingIntegrityError("baseline frozen splits must all be nonempty")
    if np.intersect1d(query, database, assume_unique=True).size:
        raise StreamingIntegrityError("baseline query/database splits overlap")
    if not np.array_equal(
        np.sort(np.concatenate((query, database))), np.arange(rows, dtype=np.int64)
    ):
        raise StreamingIntegrityError("baseline query/database union does not cover all rows")
    if np.setdiff1d(train, database, assume_unique=True).size:
        raise StreamingIntegrityError("baseline indT is not a subset of indD")
    hash_fields = {
        "full_row_ids_numeric_sha256": numeric_sha256(row_ids),
        "train_idx_numeric_sha256": numeric_sha256(train),
        "query_idx_numeric_sha256": numeric_sha256(query),
        "database_idx_numeric_sha256": numeric_sha256(database),
    }
    for key, observed in hash_fields.items():
        if contract.get(key) != observed:
            raise StreamingIntegrityError(f"baseline {key} differs from arrays")
    return root, manifest, contract, arrays


def import_baseline_code_artifact(
    artifact_root: Path,
    baseline_checkpoint_root: Path,
    output_parent: Path,
) -> Path:
    """Pack one verified baseline artifact into the common label-free state.

    Naked arrays are deliberately not accepted: both the complete baseline
    artifact directory and its current-inventory-verified checkpoint are
    mandatory.
    """

    artifact, _manifest, contract, source = _load_verified_baseline_artifact(
        artifact_root
    )
    checkpoint_root, checkpoint_files, checkpoint_receipt = _checkpoint_snapshot(
        baseline_checkpoint_root
    )

    # Lazy by design: importing rank_worker never imports torch or this
    # label-free freeze-time checkpoint implementation.
    from raw_rebuilt_baselines.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_root)
    _require_checkpoint_unchanged(checkpoint_root, checkpoint_files)
    binding_v1 = checkpoint.dataset_binding
    bits = int(contract["bits"])
    row_ids = source["row_ids"]
    train = source["indT"]
    query = source["indQ"]
    database = source["indD"]
    require_dataset_label_geometry(binding_v1.dataset, int(binding_v1.label_dim))
    checks = {
        "method": checkpoint.method,
        "bits": checkpoint.bits,
        "seed": checkpoint.seed,
        "source_seal_sha256": checkpoint.source_seal_sha256,
        "fit_artifact_sha256": checkpoint.fit_artifact_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "run_contract_sha256": checkpoint.run_contract_sha256,
        "full_row_ids_numeric_sha256": binding_v1.full_row_ids_numeric_sha256,
        "split_binding_sha256": binding_v1.split_binding_sha256,
        "train_idx_numeric_sha256": binding_v1.train_idx_numeric_sha256,
        "query_idx_numeric_sha256": binding_v1.query_idx_numeric_sha256,
        "database_idx_numeric_sha256": binding_v1.database_idx_numeric_sha256,
    }
    for key, expected in checks.items():
        if contract.get(key) != expected:
            raise StreamingIntegrityError(
                f"baseline code artifact {key} differs from current verified checkpoint"
            )
    if int(binding_v1.rows) != len(row_ids) or int(binding_v1.train_rows) != len(train):
        raise StreamingIntegrityError("baseline checkpoint row counts differ from artifact")

    baseline_inventory_sha = checkpoint_receipt.get("code_inventory", {}).get(
        "code_inventory_sha256"
    )
    _require_sha256(baseline_inventory_sha, field="baseline current code inventory")
    runtime_body = {
        "dataset": binding_v1.dataset,
        "rows": int(len(row_ids)),
        "label_dim": int(binding_v1.label_dim),
        "source_seal_sha256": checkpoint.source_seal_sha256,
        "row_ids_numeric_sha256": numeric_sha256(row_ids),
        "query_row_ids_numeric_sha256": numeric_sha256(row_ids[query]),
        "database_row_ids_numeric_sha256": numeric_sha256(row_ids[database]),
        "indQ_numeric_sha256": numeric_sha256(query),
        "indT_numeric_sha256": numeric_sha256(train),
        "indD_numeric_sha256": numeric_sha256(database),
        "query_rows": int(len(query)),
        "train_rows": int(len(train)),
        "database_rows": int(len(database)),
    }
    runtime = {**runtime_body, "runtime_identity_sha256": sha256_json(runtime_body)}
    binding_body = {
        "schema": CODE_STATE_SCHEMA,
        "producer_type": "baseline_v1_code_artifact",
        "runtime": runtime,
        "baseline_method": checkpoint.method,
        "baseline_bits": bits,
        "baseline_seed": checkpoint.seed,
        "baseline_artifact_manifest_sha256": sha256_file(artifact / "manifest.json"),
        "baseline_rank_contract_sha256": contract["rank_contract_sha256"],
        "baseline_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "baseline_run_contract_sha256": checkpoint.run_contract_sha256,
        "baseline_code_inventory_sha256": baseline_inventory_sha,
        "baseline_split_binding_sha256": binding_v1.split_binding_sha256,
        "baseline_image_codes_numeric_sha256": contract[
            "image_codes_numeric_sha256"
        ],
        "baseline_text_codes_numeric_sha256": contract[
            "text_codes_numeric_sha256"
        ],
        "config": {"bits": [bits], "import_chunk_rows": int(len(row_ids))},
        "streaming_code_inventory": production_code_inventory(),
        "labels_loaded_during_encoding": False,
    }
    binding = {
        **binding_body,
        "encoding_binding_sha256": sha256_json(binding_body),
    }
    output = reject_unsafe_output_path(
        Path(output_parent), field="baseline packed code output"
    )
    root = reject_unsafe_output_path(
        output / f"code-state-{binding['encoding_binding_sha256'][:16]}",
        field="baseline packed code state",
    )
    require_disjoint_paths(
        root,
        {
            "baseline code artifact": artifact,
            "baseline checkpoint": checkpoint.root,
        },
        field="baseline packed code state",
    )
    if root.exists():
        state = open_code_state(root)
        try:
            if state.manifest.get("binding") != binding:
                raise StreamingIntegrityError("completed baseline code state was rebound")
        finally:
            state.close()
        return root
    output.mkdir(parents=True, exist_ok=True)
    pending = reject_unsafe_output_path(
        output / f".{root.name}.pending-{os.getpid()}",
        field="baseline packed code pending state",
    )
    if pending.exists():
        raise StreamingIntegrityError("baseline packed code import output collision")
    require_disjoint_paths(
        pending,
        {
            "baseline code artifact": artifact,
            "baseline checkpoint": checkpoint.root,
        },
        field="baseline packed code pending state",
    )
    pending.mkdir(parents=False, exist_ok=False)
    scope_indices = {"query": query, "database": database}
    arrays: dict[tuple[str, str, int], np.ndarray] = {}
    for scope in SCOPES:
        for modality in MODALITIES:
            full = source[f"{modality}_codes"]
            packed = pack_bipolar_codes(full[scope_indices[scope]], bits)
            target = pending / _array_relative(scope, modality, bits)
            atomic_save_npy(target, packed)
            arrays[(scope, modality, bits)] = packed
            _write_encoding_receipt(
                pending,
                scope=scope,
                modality=modality,
                start=0,
                end=len(packed),
                arrays=arrays,
                binding_sha256=binding["encoding_binding_sha256"],
                previous_chain="0" * 64,
                available_bits=(bits,),
            )
    descriptors: dict[str, Any] = {}
    for scope in SCOPES:
        for modality in MODALITIES:
            value = arrays[(scope, modality, bits)]
            target = pending / _array_relative(scope, modality, bits)
            descriptors[f"{scope}_{modality}_{bits}"] = {
                "path": _array_relative(scope, modality, bits),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "size": target.stat().st_size,
                "file_sha256": sha256_file(target),
                "numeric_sha256": numeric_sha256(value),
            }
    receipts = [
        {
            "path": path.relative_to(pending).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((pending / "receipts").glob("*.json"))
    ]
    manifest_body = {
        "schema": CODE_STATE_SCHEMA,
        "status": "code_state_frozen",
        "dataset": binding_v1.dataset,
        "rows": int(len(row_ids)),
        "label_dim": int(binding_v1.label_dim),
        "source_seal_sha256": checkpoint.source_seal_sha256,
        "runtime_identity": runtime,
        "binding": binding,
        "available_bits": [bits],
        "arrays": descriptors,
        "receipts": receipts,
        "stored_state": "packed_query_database_binary_codes_only",
        "labels_loaded_during_encoding": False,
    }
    atomic_write_json(
        pending / "manifest.json",
        {**manifest_body, "manifest_sha256": sha256_json(manifest_body)},
    )
    os.replace(pending, root)
    opened = open_code_state(root)
    opened.close()
    return root


__all__ = [
    "BASELINE_CODE_ARTIFACT_SCHEMA",
    "import_baseline_code_artifact",
]
