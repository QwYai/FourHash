"""Admission-only creation of content-addressed ``indT`` fit artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from raw_rebuilt_runtime import load_indt_training_inputs
from raw_rebuilt_runtime.contract import DATASET_GEOMETRY, FEATURE_DIM, load_json, sha256_file

from .integrity import (
    array_descriptor,
    atomic_save_npy,
    atomic_write_json,
    numeric_sha256,
    production_code_inventory,
    reject_unsafe_output_path,
    sha256_json,
)


FIT_SCHEMA = "raw_rebuilt_neural_fit_artifact_v1"
IDENTITY_DOMAIN = b"KBS-RAW-REBUILT-V1-IDENTITY\x00"
HEX64 = re.compile(rb"[0-9a-f]{64}\Z")
ARRAY_FILES = {
    "image": "image.npy",
    "text": "text.npy",
    "labels": "labels.npy",
    "row_ids": "row_ids.npy",
    "identity_ids": "identity_ids.npy",
    "canonical_indices": "canonical_indices.npy",
}


class FitArtifactError(RuntimeError):
    """Raised when an indT-only fit artifact is incomplete or unbound."""


@dataclass(frozen=True)
class FitArtifact:
    root: Path
    dataset: str
    source_seal_sha256: str
    fit_artifact_sha256: str
    label_dim: int
    image: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    identity_ids: np.ndarray
    canonical_indices: np.ndarray
    manifest: Mapping[str, Any]

    def close(self) -> None:
        for value in (
            self.image,
            self.text,
            self.labels,
            self.row_ids,
            self.identity_ids,
            self.canonical_indices,
        ):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def identity_ids_from_row_ids(row_ids: np.ndarray) -> np.ndarray:
    """Derive collision-checked uint64 identities from canonical source IDs."""

    rows = np.asarray(row_ids)
    if rows.ndim != 1 or rows.dtype != np.dtype("S64"):
        raise FitArtifactError("canonical row IDs must be a one-dimensional S64 array")
    result = np.empty(rows.shape[0], dtype=np.uint64)
    for index, raw in enumerate(rows):
        value = bytes(raw)
        if HEX64.fullmatch(value) is None:
            raise FitArtifactError("canonical row IDs must be lowercase SHA-256 strings")
        digest = hashlib.sha256(IDENTITY_DOMAIN + value).digest()
        result[index] = np.uint64(int.from_bytes(digest[:8], "big", signed=False))
    if np.unique(result).size != result.size:
        raise FitArtifactError("uint64 identity digest collision detected")
    return result


def _require_dataset_geometry(dataset: str, rows: int, label_dim: int) -> None:
    if dataset not in DATASET_GEOMETRY:
        raise FitArtifactError(f"unsupported dataset {dataset!r}")
    expected = DATASET_GEOMETRY[dataset]
    if rows != int(expected["indT"]):
        raise FitArtifactError(f"{dataset} fit rows must equal the frozen indT count")
    if label_dim != int(expected["labels"]):
        if dataset == "nuswide":
            raise FitArtifactError("NUS-WIDE training is restricted to TC21 labels")
        raise FitArtifactError(f"{dataset} label dimension changed")


def _runtime_metadata(runtime_root: Path) -> dict[str, Any]:
    manifest_path = Path(runtime_root).expanduser().resolve(strict=True) / "runtime_manifest.json"
    manifest = load_json(manifest_path)
    required = {"dataset", "label_dim", "source_seal_sha256", "status"}
    if not required.issubset(manifest) or manifest["status"] != "COMPLETE":
        raise FitArtifactError("verified runtime manifest is incomplete")
    return {
        "dataset": str(manifest["dataset"]),
        "label_dim": int(manifest["label_dim"]),
        "source_seal_sha256": str(manifest["source_seal_sha256"]),
        "runtime_manifest_sha256": sha256_file(manifest_path),
    }


def prepare_fit_artifact(
    runtime_root: Path,
    output_parent: Path,
    *,
    _test_allow_synthetic: bool = False,
) -> Path:
    """Verify a full runtime, then emit only its sealed ``indT`` slice.

    This function belongs in a short-lived admission process.  The trainer
    intentionally has no API that accepts ``runtime_root``.
    """

    output = reject_unsafe_output_path(Path(output_parent), field="fit output")
    output.mkdir(parents=True, exist_ok=True)
    metadata = _runtime_metadata(runtime_root)
    training = load_indt_training_inputs(
        runtime_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    if training.source_seal_sha256 != metadata["source_seal_sha256"]:
        raise FitArtifactError("training slice and runtime manifest seals differ")
    rows = int(training.image.shape[0])
    label_dim = int(training.labels.shape[1])
    _require_dataset_geometry(metadata["dataset"], rows, label_dim)
    canonical_indices = np.asarray(training.identity_ids, dtype=np.int64)
    if canonical_indices.shape != (rows,) or np.any(canonical_indices[1:] <= canonical_indices[:-1]):
        raise FitArtifactError("indT canonical indices must be strictly increasing")
    row_ids = np.asarray(training.row_ids, dtype="S64")
    identity_ids = identity_ids_from_row_ids(row_ids)
    arrays = {
        "image": np.ascontiguousarray(training.image, dtype=np.float32),
        "text": np.ascontiguousarray(training.text, dtype=np.float32),
        "labels": np.ascontiguousarray(training.labels, dtype=np.uint8),
        "row_ids": np.ascontiguousarray(row_ids, dtype="S64"),
        "identity_ids": np.ascontiguousarray(identity_ids, dtype=np.uint64),
        "canonical_indices": np.ascontiguousarray(canonical_indices, dtype=np.int64),
    }
    descriptors = {
        name: {
            "path": ARRAY_FILES[name],
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "numeric_sha256": numeric_sha256(value),
        }
        for name, value in arrays.items()
    }
    body = {
        "schema": FIT_SCHEMA,
        "status": "COMPLETE",
        "dataset": metadata["dataset"],
        "rows": rows,
        "feature_dim": FEATURE_DIM,
        "label_dim": label_dim,
        "source_seal_sha256": training.source_seal_sha256,
        "runtime_manifest_sha256": metadata["runtime_manifest_sha256"],
        "split_indT_numeric_sha256": numeric_sha256(canonical_indices),
        "arrays": descriptors,
        "identity_derivation": "sha256(domain || canonical_row_id)[:8]-big-endian-uint64-v1",
        "code_inventory": production_code_inventory(),
    }
    artifact_sha = sha256_json(body)
    root = output / f"fit-{artifact_sha[:16]}"
    manifest = {**body, "fit_artifact_sha256": artifact_sha}
    if root.exists():
        opened = open_fit_artifact(root, _test_allow_synthetic=_test_allow_synthetic)
        opened.close()
        if opened.fit_artifact_sha256 != artifact_sha:
            raise FitArtifactError("existing content-addressed fit directory differs")
        return root
    root.mkdir(parents=False, exist_ok=False)
    for name, value in arrays.items():
        atomic_save_npy(root / ARRAY_FILES[name], value)
    # Add exact file hashes only outside the content address: numeric content,
    # geometry, and all provenance already determine ``artifact_sha``.
    file_inventory = {
        name: array_descriptor(root / ARRAY_FILES[name]) for name in arrays
    }
    atomic_write_json(root / "manifest.json", {**manifest, "file_inventory": file_inventory})
    opened = open_fit_artifact(root, _test_allow_synthetic=_test_allow_synthetic)
    opened.close()
    return root


def open_fit_artifact(
    root: Path,
    *,
    _test_allow_synthetic: bool = False,
) -> FitArtifact:
    path = reject_unsafe_output_path(Path(root), field="fit artifact").resolve(strict=True)
    manifest = load_json(path / "manifest.json")
    required = {
        "schema",
        "status",
        "dataset",
        "rows",
        "feature_dim",
        "label_dim",
        "source_seal_sha256",
        "runtime_manifest_sha256",
        "split_indT_numeric_sha256",
        "arrays",
        "identity_derivation",
        "code_inventory",
        "fit_artifact_sha256",
        "file_inventory",
    }
    if set(manifest) != required or manifest.get("schema") != FIT_SCHEMA or manifest.get("status") != "COMPLETE":
        raise FitArtifactError("fit manifest schema or keys differ")
    dataset = str(manifest["dataset"])
    if dataset == "synthetic" and not _test_allow_synthetic:
        raise FitArtifactError("synthetic fit artifacts are test-only")
    rows = int(manifest["rows"])
    label_dim = int(manifest["label_dim"])
    if int(manifest["feature_dim"]) != FEATURE_DIM:
        raise FitArtifactError("fit features must be CLIP512")
    _require_dataset_geometry(dataset, rows, label_dim)
    arrays_meta = manifest["arrays"]
    file_meta = manifest["file_inventory"]
    if not isinstance(arrays_meta, dict) or set(arrays_meta) != set(ARRAY_FILES):
        raise FitArtifactError("fit array inventory differs")
    if not isinstance(file_meta, dict) or set(file_meta) != set(ARRAY_FILES):
        raise FitArtifactError("fit file inventory differs")
    arrays: dict[str, np.ndarray] = {}
    for name, filename in ARRAY_FILES.items():
        target = path / filename
        declared = arrays_meta[name]
        inventory = file_meta[name]
        if declared.get("path") != filename or inventory.get("path") != filename:
            raise FitArtifactError(f"fit {name} path changed")
        if sha256_file(target) != inventory.get("file_sha256"):
            raise FitArtifactError(f"fit {name} file hash changed")
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        if value.dtype.str != declared.get("dtype") or list(value.shape) != declared.get("shape"):
            raise FitArtifactError(f"fit {name} dtype/shape changed")
        observed_numeric = numeric_sha256(value)
        if observed_numeric != declared.get("numeric_sha256") or observed_numeric != inventory.get("numeric_sha256"):
            raise FitArtifactError(f"fit {name} numeric content changed")
        arrays[name] = value
    if arrays["image"].shape != (rows, FEATURE_DIM) or arrays["text"].shape != arrays["image"].shape:
        raise FitArtifactError("fit image/text geometry differs")
    labels = arrays["labels"]
    if labels.shape != (rows, label_dim) or labels.dtype != np.uint8 or not np.all(np.isin(labels, (0, 1))):
        raise FitArtifactError("fit labels are not aligned binary multi-hot rows")
    if np.any(labels.sum(axis=1) == 0) or np.any(labels.sum(axis=0) == 0):
        raise FitArtifactError("fit labels contain empty rows or zero-positive classes")
    row_ids = arrays["row_ids"]
    if row_ids.dtype != np.dtype("S64") or np.unique(row_ids).size != rows:
        raise FitArtifactError("fit canonical row IDs are invalid")
    expected_identity = identity_ids_from_row_ids(row_ids)
    if arrays["identity_ids"].dtype != np.uint64 or not np.array_equal(arrays["identity_ids"], expected_identity):
        raise FitArtifactError("fit identity digests do not match row IDs")
    canonical = arrays["canonical_indices"]
    if canonical.dtype != np.int64 or canonical.shape != (rows,) or np.any(canonical[1:] <= canonical[:-1]):
        raise FitArtifactError("fit canonical indT indices are invalid")
    if numeric_sha256(canonical) != manifest["split_indT_numeric_sha256"]:
        raise FitArtifactError("fit indT split hash changed")
    body = {key: manifest[key] for key in required - {"fit_artifact_sha256", "file_inventory"}}
    artifact_sha = sha256_json(body)
    if artifact_sha != manifest["fit_artifact_sha256"] or path.name != f"fit-{artifact_sha[:16]}":
        raise FitArtifactError("fit artifact content address changed")
    return FitArtifact(
        root=path,
        dataset=dataset,
        source_seal_sha256=str(manifest["source_seal_sha256"]),
        fit_artifact_sha256=artifact_sha,
        label_dim=label_dim,
        image=arrays["image"],
        text=arrays["text"],
        labels=labels,
        row_ids=row_ids,
        identity_ids=arrays["identity_ids"],
        canonical_indices=canonical,
        manifest=manifest,
    )


__all__ = [
    "FitArtifact",
    "FitArtifactError",
    "identity_ids_from_row_ids",
    "open_fit_artifact",
    "prepare_fit_artifact",
]

