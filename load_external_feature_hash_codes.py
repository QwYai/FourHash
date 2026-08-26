#!/usr/bin/env python3
"""Strict loader for frozen feature-hash code NPZ artifacts.

The loader treats row number as the canonical item ID.  It validates the
artifact, its optional sibling manifest, and the standard hashing split before
returning any arrays to mixed-gallery diagnostics.  In particular, query and
database IDs must be disjoint and together cover every item exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_ARRAYS = {
    "image_codes",
    "text_codes",
    "labels",
    "train_idx",
    "query_idx",
    "database_idx",
    "metadata_json",
}


@dataclass(frozen=True)
class ExternalFeatureHashBundle:
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path | None
    manifest_sha256: str | None
    image_codes: np.ndarray
    text_codes: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    query_idx: np.ndarray
    database_idx: np.ndarray
    metadata: Mapping[str, Any]
    manifest: Mapping[str, Any] | None

    @property
    def rows(self) -> int:
        return int(self.image_codes.shape[0])

    @property
    def bits(self) -> int:
        return int(self.image_codes.shape[1])

    @property
    def item_ids(self) -> np.ndarray:
        ids = np.arange(self.rows, dtype=np.int64)
        ids.setflags(write=False)
        return ids

    @property
    def reporting_name(self) -> str:
        return str(self.metadata.get("reporting_name", "external_feature_hash"))

    @property
    def dataset(self) -> str:
        dataset = self.metadata.get("dataset", {})
        if isinstance(dataset, Mapping):
            return str(dataset.get("name", "unknown"))
        return str(dataset)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest_array_sha256(value: np.ndarray) -> str:
    """Match the dtype + shape + bytes hash used by DCMH-F exports."""
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _require_index(name: str, value: np.ndarray, rows: int) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.int64 or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 array")
    if value.size == 0:
        raise ValueError(f"{name} is empty")
    if int(value.min()) < 0 or int(value.max()) >= rows:
        raise ValueError(f"{name} contains an out-of-range item ID")
    if np.unique(value).size != value.size:
        raise ValueError(f"{name} contains duplicate item IDs")
    return _readonly(value)


def _validate_manifest_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> None:
    records = manifest.get("npz_arrays")
    if not isinstance(records, Mapping):
        raise ValueError("Manifest has no npz_arrays mapping")
    if set(records) != set(arrays):
        raise ValueError(
            "Manifest/NPZ key mismatch: "
            f"manifest={sorted(records)}, npz={sorted(arrays)}"
        )
    for name, value in arrays.items():
        record = records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid manifest record for {name}")
        if list(value.shape) != list(record.get("shape", [])):
            raise ValueError(f"Manifest shape mismatch for {name}")
        if str(value.dtype) != record.get("dtype"):
            raise ValueError(f"Manifest dtype mismatch for {name}")
        observed = manifest_array_sha256(value)
        if observed != record.get("sha256"):
            raise ValueError(
                f"Manifest content hash mismatch for {name}: {observed}"
            )


def _validate_gate_metadata(metadata: Mapping[str, Any]) -> None:
    training = metadata.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("metadata.training is missing")
    # Accept only the three frozen-adapter schemas currently under audit.
    # If an artifact declares more than one alias, every declared gate must
    # pass; a second field can never mask a failed first field.
    train_gate_names = (
        "quality_gate",  # DCMH-F
        "label_free_train_quality_gate",  # UCCH-F
        "structural_quality_gate",  # CIRH-F
    )
    train_gates = [
        (name, training[name]) for name in train_gate_names if name in training
    ]
    if not train_gates:
        raise ValueError("Frozen artifact has no recognized train quality gate")
    for name, train_gate in train_gates:
        if not isinstance(train_gate, Mapping) or train_gate.get("passed") is not True:
            raise ValueError(f"Frozen artifact did not pass train gate {name}")
    heldout_gate = training.get("heldout_quality_gate", {})
    if (
        not isinstance(heldout_gate, Mapping)
        or heldout_gate.get("passed") is not True
    ):
        raise ValueError("Frozen artifact did not pass its held-out quality gate")
    if training.get("overall_usable") is not True:
        raise ValueError("Frozen artifact metadata does not mark it usable")


def _validate_declared_hash_bits(
    metadata: Mapping[str, Any], observed_bits: int
) -> None:
    """Validate every recognized, explicit hash-bit declaration.

    Historical frozen adapters used ``architecture.hash_bits`` (DCMH-F),
    ``architecture.bits`` (UCCH-F), or ``training.bits`` (CIRH-F).  The
    immutable N x B code matrices remain authoritative: at least one known
    declaration must exist, every declaration must be an integer, all aliases
    must agree, and their common value must equal B.
    """
    declarations: dict[str, Any] = {}
    architecture = metadata.get("architecture")
    if architecture is not None:
        if not isinstance(architecture, Mapping):
            raise ValueError("metadata.architecture must be an object")
        for key in ("hash_bits", "bits"):
            if key in architecture:
                declarations[f"architecture.{key}"] = architecture[key]
    training = metadata.get("training")
    if isinstance(training, Mapping) and "bits" in training:
        declarations["training.bits"] = training["bits"]
    if not declarations:
        raise ValueError("metadata has no recognized hash-bit declaration")

    normalized: dict[str, int] = {}
    for path, value in declarations.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"metadata {path} must be an integer")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"metadata {path} must be an integer") from exc
        if isinstance(value, (float, np.floating)) and float(value) != integer:
            raise ValueError(f"metadata {path} must be an integer")
        normalized[path] = integer

    if len(set(normalized.values())) != 1:
        detail = ", ".join(f"{path}={value}" for path, value in normalized.items())
        raise ValueError(f"Conflicting metadata hash-bit declarations: {detail}")
    if next(iter(normalized.values())) != observed_bits:
        raise ValueError("metadata hash-bit count mismatch")


def load_external_feature_hash_bundle(
    artifact_path: Path,
    *,
    manifest_path: Path | None = None,
    require_manifest: bool = True,
    require_usable: bool = True,
) -> ExternalFeatureHashBundle:
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if manifest_path is None:
        candidate = artifact_path.parent / "MANIFEST.json"
        manifest_path = candidate if candidate.is_file() else None
    elif not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if require_manifest and manifest_path is None:
        raise ValueError("A manifest is required for this frozen artifact")

    artifact_sha256 = sha256_file(artifact_path)
    manifest: Mapping[str, Any] | None = None
    manifest_sha256: str | None = None
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        manifest_sha256 = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_record = manifest.get("files", {}).get(artifact_path.name, {})
        expected_file_hash = file_record.get("sha256")
        if expected_file_hash != artifact_sha256:
            raise ValueError(
                "Artifact SHA-256 disagrees with manifest: "
                f"{artifact_sha256} != {expected_file_hash}"
            )

    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError(
                f"Unexpected NPZ keys: {sorted(archive.files)}; "
                f"expected {sorted(REQUIRED_ARRAYS)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if manifest is not None:
        _validate_manifest_arrays(arrays, manifest)

    image_codes = arrays["image_codes"]
    text_codes = arrays["text_codes"]
    if image_codes.dtype != np.int8 or text_codes.dtype != np.int8:
        raise ValueError("Image and text codes must be int8")
    if image_codes.ndim != 2 or text_codes.shape != image_codes.shape:
        raise ValueError("Image/text code matrices must have identical N x B shape")
    rows, bits = map(int, image_codes.shape)
    if rows < 1 or bits < 1:
        raise ValueError("Code matrices must be nonempty")
    for name, value in (("image_codes", image_codes), ("text_codes", text_codes)):
        if set(np.unique(value).tolist()) != {-1, 1}:
            raise ValueError(f"{name} must have domain exactly {{-1, +1}}")

    labels = arrays["labels"]
    if labels.dtype != np.uint8 or labels.ndim != 2 or labels.shape[0] != rows:
        raise ValueError("labels must be an N x C uint8 matrix aligned to codes")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary multi-hot values")
    if np.any(labels.sum(axis=1) == 0):
        raise ValueError("Every item must have at least one positive label")

    train_idx = _require_index("train_idx", arrays["train_idx"], rows)
    query_idx = _require_index("query_idx", arrays["query_idx"], rows)
    database_idx = _require_index("database_idx", arrays["database_idx"], rows)
    query_database_overlap = np.intersect1d(query_idx, database_idx)
    train_query_overlap = np.intersect1d(train_idx, query_idx)
    if query_database_overlap.size:
        raise ValueError("query_idx and database_idx overlap")
    if train_query_overlap.size:
        raise ValueError("train_idx and query_idx overlap")
    if not np.all(np.isin(train_idx, database_idx)):
        raise ValueError("The standard split requires train_idx to be a DB subset")
    partition = np.concatenate((query_idx, database_idx))
    if partition.size != rows or np.unique(partition).size != rows:
        raise ValueError("Query/database IDs do not cover every item exactly once")
    if not np.array_equal(np.sort(partition), np.arange(rows, dtype=np.int64)):
        raise ValueError("Query/database row IDs are not the canonical 0..N-1 items")

    metadata_raw = arrays["metadata_json"]
    if metadata_raw.ndim != 0 or metadata_raw.dtype.kind != "U":
        raise ValueError("metadata_json must be a scalar Unicode array")
    metadata = json.loads(str(metadata_raw.item()))
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata_json must decode to an object")
    dataset_meta = metadata.get("dataset", {})
    if not isinstance(dataset_meta, Mapping):
        raise ValueError("metadata.dataset is missing")
    expected_sizes = {
        "rows": rows,
        "train_rows": int(train_idx.size),
        "query_rows": int(query_idx.size),
        "database_rows": int(database_idx.size),
    }
    for key, expected in expected_sizes.items():
        if int(dataset_meta.get(key, -1)) != expected:
            raise ValueError(f"metadata.dataset.{key} mismatch")
    _validate_declared_hash_bits(metadata, bits)
    if int(dataset_meta.get("original_index_base", -1)) != 0:
        raise ValueError("Only zero-based item IDs are accepted")
    if require_usable:
        _validate_gate_metadata(metadata)
    if manifest is not None and metadata != manifest.get("metadata"):
        raise ValueError("Embedded metadata does not exactly match the manifest")

    for value in arrays.values():
        _readonly(value)
    return ExternalFeatureHashBundle(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        image_codes=image_codes,
        text_codes=text_codes,
        labels=labels,
        train_idx=train_idx,
        query_idx=query_idx,
        database_idx=database_idx,
        metadata=metadata,
        manifest=manifest,
    )


def describe(bundle: ExternalFeatureHashBundle) -> dict[str, Any]:
    train_database_overlap = int(
        np.intersect1d(bundle.train_idx, bundle.database_idx).size
    )
    return {
        "artifact": str(bundle.artifact_path),
        "artifact_sha256": bundle.artifact_sha256,
        "manifest": str(bundle.manifest_path) if bundle.manifest_path else None,
        "manifest_sha256": bundle.manifest_sha256,
        "reporting_name": bundle.reporting_name,
        "dataset": bundle.dataset,
        "rows": bundle.rows,
        "bits": bundle.bits,
        "label_dim": int(bundle.labels.shape[1]),
        "train_rows": int(bundle.train_idx.size),
        "query_rows": int(bundle.query_idx.size),
        "database_rows": int(bundle.database_idx.size),
        "query_database_overlap": int(
            np.intersect1d(bundle.query_idx, bundle.database_idx).size
        ),
        "train_query_overlap": int(
            np.intersect1d(bundle.train_idx, bundle.query_idx).size
        ),
        "train_database_overlap": train_database_overlap,
        "train_is_database_subset": train_database_overlap
        == int(bundle.train_idx.size),
        "query_database_partition_covers_every_item_once": True,
        "image_code_domain": sorted(np.unique(bundle.image_codes).tolist()),
        "text_code_domain": sorted(np.unique(bundle.text_codes).tolist()),
        "array_raw_sha256": {
            name: raw_array_sha256(value)
            for name, value in {
                "image_codes": bundle.image_codes,
                "text_codes": bundle.text_codes,
                "labels": bundle.labels,
                "train_idx": bundle.train_idx,
                "query_idx": bundle.query_idx,
                "database_idx": bundle.database_idx,
            }.items()
        },
        "overall_usable": bool(
            bundle.metadata.get("training", {}).get("overall_usable", False)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-missing-manifest", action="store_true")
    parser.add_argument("--allow-unusable", action="store_true")
    args = parser.parse_args()
    bundle = load_external_feature_hash_bundle(
        args.npz,
        manifest_path=args.manifest,
        require_manifest=not args.allow_missing_manifest,
        require_usable=not args.allow_unusable,
    )
    print(json.dumps(describe(bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
