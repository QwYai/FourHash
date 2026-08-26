"""Canonical hashing primitives for the visualization feature contract.

The functions in this module are deliberately small and dependency-light.  A
dataset adapter may use them while extracting features, and the independent
validator uses the same domain-separated encodings when it checks a bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "kbs_visualization_feature_provenance_v1"
ROW_ID_DOMAIN = b"KBS-VIS-ROW-V1\x00"
EXTRACTOR_ID_DOMAIN = b"KBS-VIS-EXTRACTOR-V1\x00"
BUNDLE_ID_DOMAIN = b"KBS-VIS-BUNDLE-V1\x00"
VECTOR_DOMAIN = b"KBS-VIS-VECTOR-V1\x00"
ORDERED_IDS_DOMAIN = b"KBS-VIS-ORDERED-IDS-V1\x00"
INT64_ARRAY_DOMAIN = b"KBS-VIS-INT64-ARRAY-V1\x00"
RAW_SPLIT_ALGORITHM = "kbs-content-hash-split-v1"
RAW_REBUILT_SPLIT_SEED = 20260822
RAW_REBUILT_COUNTS = {
    "mirflickr": {"n_rows": 20_015, "indQ": 2_243, "indT": 5_000, "indD": 17_772},
    "nuswide": {"n_rows": 195_834, "indQ": 2_085, "indT": 21_000, "indD": 193_749},
    "mscoco": {"n_rows": 122_218, "indQ": 5_000, "indT": 10_500, "indD": 117_218},
    # The synthetic registry entry is test-only and cannot name a real dataset.
    "synthetic": {"n_rows": 4, "indQ": 1, "indT": 2, "indD": 3},
}
RAW_REBUILT_LABEL_DIMS = {
    "mirflickr": 24,
    "nuswide": 21,
    "mscoco": 80,
    "synthetic": 3,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a bundle violates a fail-closed contract requirement."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically without ASCII-escaping Unicode text."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def _sha256_stream(handle: BinaryIO, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_text(value: str) -> str:
    """Apply the only accepted text normalization: NFC plus LF newlines."""

    if not isinstance(value, str):
        raise ContractError("text value must be a string")
    if "\x00" in value:
        raise ContractError("text value contains NUL")
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def text_utf8_sha256(value: str) -> str:
    normalized = normalize_text(value)
    if normalized != value:
        raise ContractError("text must already be NFC-normalized with LF newlines")
    return sha256_bytes(value.encode("utf-8"))


def _without_key(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def derive_row_id(row: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(_without_key(row, "row_id"))
    return "sha256:" + sha256_bytes(ROW_ID_DOMAIN + payload)


def derive_extractor_id(extractor: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(_without_key(extractor, "extractor_id"))
    return "sha256:" + sha256_bytes(EXTRACTOR_ID_DOMAIN + payload)


def derive_bundle_id(manifest: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(_without_key(manifest, "bundle_id"))
    return "sha256:" + sha256_bytes(BUNDLE_ID_DOMAIN + payload)


def _canonical_numeric_array(value: np.ndarray) -> tuple[np.ndarray, str]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ContractError("object arrays are forbidden")
    if array.dtype.kind not in "biuf":
        raise ContractError(f"unsupported numeric dtype: {array.dtype}")
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    elif dtype.byteorder == "=":
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    array = np.ascontiguousarray(array)
    return array, dtype.str


def feature_row_sha256(value: np.ndarray) -> str:
    """Hash one vector with its canonical dtype and exact row shape."""

    array, dtype_string = _canonical_numeric_array(np.asarray(value))
    header = canonical_json_bytes({"dtype": dtype_string, "shape": list(array.shape)})
    return sha256_bytes(VECTOR_DOMAIN + header + b"\x00" + array.tobytes(order="C"))


def numeric_array_sha256(value: np.ndarray) -> str:
    array, dtype_string = _canonical_numeric_array(np.asarray(value))
    header = canonical_json_bytes({"dtype": dtype_string, "shape": list(array.shape)})
    return sha256_bytes(VECTOR_DOMAIN + b"ARRAY\x00" + header + b"\x00" + array.tobytes(order="C"))


def ordered_ids_sha256(values: list[str] | tuple[str, ...] | np.ndarray) -> str:
    items = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    return sha256_bytes(ORDERED_IDS_DOMAIN + canonical_json_bytes(items))


def canonical_sample_id_order(sample_ids: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for index, value in enumerate(sample_ids):
        if not isinstance(value, str) or not value:
            raise ContractError(f"sample_ids[{index}] must be non-empty text")
        canonical = normalize_text(value)
        if canonical != value:
            raise ContractError(f"sample_ids[{index}] is not NFC/LF canonical")
        normalized.append(value)
    return sorted(normalized, key=lambda item: item.encode("utf-8"))


def raw_rebuilt_split_indices(dataset: str, sample_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Create the single preregistered raw-rebuilt split; no seed search exists."""

    if dataset not in RAW_REBUILT_COUNTS:
        raise ContractError(f"dataset {dataset!r} has no frozen raw-rebuilt split registry")
    counts = RAW_REBUILT_COUNTS[dataset]
    if len(sample_ids) != counts["n_rows"]:
        raise ContractError(
            f"raw-rebuilt {dataset} has {len(sample_ids)} rows; frozen count is {counts['n_rows']}"
        )
    canonical_sample_id_order(sample_ids)  # validates canonical Unicode; adapters freeze ordering.
    def split_key(role: str, sample_id: str) -> bytes:
        return hashlib.sha256(
            "\0".join(
                (RAW_SPLIT_ALGORITHM, str(RAW_REBUILT_SPLIT_SEED), dataset, role, sample_id)
            ).encode("utf-8")
        ).digest()

    all_rows = list(range(len(sample_ids)))
    query_order = sorted(
        all_rows,
        key=lambda row: (split_key("query", sample_ids[row]), sample_ids[row], row),
    )
    query_count = counts["indQ"]
    train_count = counts["indT"]
    query_set = set(query_order[:query_count])
    database_set = set(all_rows) - query_set
    train_order = sorted(
        database_set,
        key=lambda row: (split_key("train", sample_ids[row]), sample_ids[row], row),
    )
    train_set = set(train_order[:train_count])
    ind_q = np.asarray(sorted(query_set), dtype=np.int64)
    ind_d = np.asarray(sorted(database_set), dtype=np.int64)
    ind_t = np.asarray(sorted(train_set), dtype=np.int64)
    if ind_d.size != counts["indD"] or ind_q.size + ind_d.size != len(sample_ids):
        raise ContractError("frozen raw-rebuilt Q/D counts do not cover the dataset")
    return {"indT": ind_t, "indQ": ind_q, "indD": ind_d}


def raw_rebuilt_assignment_sha256(
    dataset: str,
    sample_ids: Sequence[str],
    arrays: Mapping[str, np.ndarray],
) -> str:
    if set(arrays) != {"indT", "indQ", "indD"}:
        raise ContractError("split assignment must contain indT, indQ, and indD")
    payload = {
        "algorithm": RAW_SPLIT_ALGORITHM,
        "seed": RAW_REBUILT_SPLIT_SEED,
        "dataset": dataset,
        "ordered_sample_ids_sha256": ordered_ids_sha256(list(sample_ids)),
        "arrays": {name: int64_array_sha256(np.asarray(arrays[name])) for name in ("indT", "indQ", "indD")},
    }
    return sha256_bytes(canonical_json_bytes(payload))


def int64_array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(f"official split array must be one-dimensional, got {array.shape}")
    if array.dtype.kind not in "iu":
        raise ContractError(f"official split array must be integral, got {array.dtype}")
    canonical = np.ascontiguousarray(array, dtype="<i8")
    header = canonical_json_bytes({"shape": list(canonical.shape), "dtype": "<i8"})
    return sha256_bytes(INT64_ARRAY_DOMAIN + header + b"\x00" + canonical.tobytes())


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def require_content_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or CONTENT_ID_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must have form sha256:<lowercase hex>")
    return value


def resolve_contained(root: Path, relative_path: str, *, field: str) -> Path:
    """Resolve a bundle-relative file and reject absolute/traversing symlinks."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ContractError(f"{field} must be a non-empty relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ContractError(f"{field} must stay inside the bundle")
    root_resolved = root.resolve(strict=True)
    resolved = (root_resolved / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ContractError(f"{field} escapes the bundle through a path or symlink") from error
    if not resolved.is_file():
        raise ContractError(f"{field} is not a regular file: {resolved}")
    return resolved


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False
