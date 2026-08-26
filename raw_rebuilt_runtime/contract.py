"""Small hashing and boundary primitives for the runtime bridge."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "kbs_raw_rebuilt_runtime_v1"
SPLIT_ALGORITHM = "kbs-content-hash-split-v1"
SPLIT_SEED = 20_260_822
FEATURE_DIM = 512
DATASET_GEOMETRY = {
    "mirflickr": {"rows": 20_015, "labels": 24, "indQ": 2_243, "indT": 5_000, "indD": 17_772},
    "nuswide": {"rows": 195_834, "labels": 21, "indQ": 2_085, "indT": 21_000, "indD": 193_749},
    "mscoco": {"rows": 122_218, "labels": 80, "indQ": 5_000, "indT": 10_500, "indD": 117_218},
    # Never exposed by the CLI.  It exists solely for the four-row E2E tests.
    "synthetic": {"rows": 4, "labels": 3, "indQ": 1, "indT": 2, "indD": 3},
}
ARRAY_SPECS = {
    "image": ("arrays/image_features_clip512.npy", "float32"),
    "text": ("arrays/text_features_clip512.npy", "float32"),
    "labels": ("arrays/labels.npy", "uint8"),
    "row_ids": ("arrays/row_ids.npy", "S64"),
    "indQ": ("arrays/indQ.npy", "int64"),
    "indT": ("arrays/indT.npy", "int64"),
    "indD": ("arrays/indD.npy", "int64"),
}


class RuntimeBridgeError(RuntimeError):
    """Raised whenever the runtime cannot prove an exact source binding."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeBridgeError(f"value cannot be encoded as canonical JSON: {error}") from error
    return text.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: os.PathLike[str], chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def numeric_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and exact C-order bytes of one numeric array."""

    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufSU":
        raise RuntimeBridgeError(f"unsupported array dtype for sealing: {array.dtype}")
    canonical = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"KBS-RAW-RUNTIME-ARRAY-V1\x00" + header + b"\x00")
    byte_view = memoryview(canonical).cast("B")
    for start in range(0, byte_view.nbytes, 8 * 1024 * 1024):
        digest.update(byte_view[start : start + 8 * 1024 * 1024])
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".pending")
    payload = canonical_json_bytes(dict(value)) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeBridgeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RuntimeBridgeError):
            raise
        raise RuntimeBridgeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeBridgeError(f"{path} must contain a JSON object")
    return value


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def require_disjoint(path: Path, forbidden: Iterable[Path], *, field: str) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    if any(part.casefold() in {"oraldata", "processdata"} for part in candidate.parts):
        raise RuntimeBridgeError(f"{field} may not be under OralData or ProcessData: {candidate}")
    for root in forbidden:
        protected = root.expanduser().resolve(strict=False)
        if candidate == protected or is_within(candidate, protected) or is_within(protected, candidate):
            raise RuntimeBridgeError(
                f"{field} must be completely separate from protected input: "
                f"path={candidate}, protected={protected}"
            )
    return candidate


def require_geometry(
    dataset: str,
    *,
    rows: int,
    feature_dim: int,
    label_dim: int,
    allow_test_dataset: bool = False,
) -> Mapping[str, int]:
    if dataset == "synthetic" and not allow_test_dataset:
        raise RuntimeBridgeError("synthetic runtime bundles are test-only")
    if dataset not in DATASET_GEOMETRY:
        raise RuntimeBridgeError(f"dataset {dataset!r} is not a frozen raw_rebuilt_v1 dataset")
    expected = DATASET_GEOMETRY[dataset]
    if rows != expected["rows"]:
        raise RuntimeBridgeError(
            f"{dataset} row count {rows} differs from raw_rebuilt_v1 {expected['rows']}"
        )
    if feature_dim != FEATURE_DIM:
        raise RuntimeBridgeError(f"CLIP feature width must be 512, observed {feature_dim}")
    if label_dim != expected["labels"]:
        if dataset == "nuswide" and label_dim == 81:
            raise RuntimeBridgeError("NUS-WIDE 81-hot legacy labels are forbidden; TC21 must be 21-hot")
        raise RuntimeBridgeError(
            f"{dataset} label width {label_dim} differs from frozen {expected['labels']}"
        )
    return expected


def require_split_arrays(
    arrays: Mapping[str, np.ndarray], *, dataset: str, rows: int
) -> None:
    expected = DATASET_GEOMETRY[dataset]
    if set(arrays) != {"indQ", "indT", "indD"}:
        raise RuntimeBridgeError("split must expose exactly indQ, indT, and indD")
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype != np.dtype("int64"):
            raise RuntimeBridgeError(f"{name} must be a one-dimensional int64 array")
        if array.size and (int(array.min()) < 0 or int(array.max()) >= rows):
            raise RuntimeBridgeError(f"{name} contains an out-of-range row")
        if np.unique(array).size != array.size or np.any(array[1:] <= array[:-1]):
            raise RuntimeBridgeError(f"{name} must be strictly increasing and unique")
        if array.size != expected[name]:
            raise RuntimeBridgeError(
                f"{dataset} {name} count {array.size} differs from frozen {expected[name]}"
            )
        normalized[name] = array
    q = normalized["indQ"]
    t = normalized["indT"]
    d = normalized["indD"]
    if np.intersect1d(q, d, assume_unique=True).size:
        raise RuntimeBridgeError("indQ and indD overlap")
    if not np.array_equal(np.sort(np.concatenate((q, d))), np.arange(rows, dtype=np.int64)):
        raise RuntimeBridgeError("indQ union indD does not cover every canonical row")
    if np.setdiff1d(t, d, assume_unique=True).size:
        raise RuntimeBridgeError("indT is not a subset of indD")
