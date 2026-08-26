"""Canonical hashes, inventories, and atomic writes for streaming artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CODE_SCHEMA = "raw_rebuilt_streaming_code_inventory_v1"


class StreamingIntegrityError(RuntimeError):
    """Raised when an immutable streaming artifact cannot be replayed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise StreamingIntegrityError("value cannot be encoded as canonical JSON") from error
    return encoded.encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufSU":
        raise StreamingIntegrityError(f"unsupported array dtype for sealing: {array.dtype}")
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
    pending = target.with_name(target.name + ".pending")
    payload = canonical_json_bytes(dict(value)) + b"\n"
    with pending.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, target)


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StreamingIntegrityError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StreamingIntegrityError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise StreamingIntegrityError(f"{path} must contain a JSON object")
    return value


def production_code_inventory() -> dict[str, Any]:
    """Hash the streaming package and every external scoring dependency."""

    paths = list(PACKAGE_ROOT.rglob("*.py"))
    paths.extend(
        [
            PROJECT_ROOT / "raw_rebuilt_runtime" / "contract.py",
            PROJECT_ROOT / "raw_rebuilt_runtime" / "loader.py",
            PROJECT_ROOT / "raw_rebuilt_runtime" / "materialize.py",
            PROJECT_ROOT / "raw_rebuilt_runtime" / "metric_loader.py",
            PROJECT_ROOT / "raw_rebuilt_runtime" / "validation.py",
            PROJECT_ROOT / "visualization_trace" / "core.py",
            PROJECT_ROOT / "visualization_trace" / "extraction.py",
        ]
    )
    files = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        resolved = path.resolve(strict=True)
        files.append(
            {
                "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    body = {"schema": CODE_SCHEMA, "files": files}
    return {**body, "code_inventory_sha256": sha256_json(body)}


def reject_unsafe_output_path(path: Path, *, field: str) -> Path:
    declared = require_no_link_components(
        path, field=field, allow_missing=True
    )
    candidate = declared.resolve(strict=False)
    if any(part.casefold() in {"oraldata", "processdata"} for part in candidate.parts):
        raise ValueError(f"{field} must be outside protected raw/prepared inputs")
    if candidate.suffix.casefold() == ".mat":
        raise ValueError(f"{field} may not be a legacy MAT artifact")
    return candidate


def require_no_link_components(
    path: Path,
    *,
    field: str,
    allow_missing: bool = False,
) -> Path:
    """Reject symlink/reparse components without resolving the declared path."""

    declared = Path(os.path.abspath(str(Path(path).expanduser())))
    current = Path(declared.anchor)
    tail = declared.parts[1:] if declared.anchor else declared.parts
    for part in tail:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                break
            raise StreamingIntegrityError(f"{field} path component is missing: {current}")
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
            raise StreamingIntegrityError(f"{field} contains a symlink/reparse component")
    return declared


def paths_overlap(first: Path, second: Path) -> bool:
    left = Path(first).expanduser().resolve(strict=False)
    right = Path(second).expanduser().resolve(strict=False)
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def require_disjoint_paths(candidate: Path, forbidden: Mapping[str, Path], *, field: str) -> None:
    target = Path(candidate).expanduser().resolve(strict=False)
    for name, path in forbidden.items():
        if paths_overlap(target, Path(path)):
            raise StreamingIntegrityError(f"{field} overlaps immutable {name}")


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending")
    with pending.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, target)


def array_descriptor(path: Path, value: np.ndarray | None = None) -> dict[str, Any]:
    array = value if value is not None else np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.name,
        "dtype": np.dtype(array.dtype).str,
        "shape": list(array.shape),
        "size": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "numeric_sha256": numeric_sha256(array),
    }


def require_hashed_json(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    schema: str | None = None,
    status: str | None = None,
    field: str,
) -> None:
    body = {key: value[key] for key in value if key != hash_field}
    if value.get(hash_field) != sha256_json(body):
        raise StreamingIntegrityError(f"{field} hash changed")
    if schema is not None and value.get("schema") != schema:
        raise StreamingIntegrityError(f"{field} schema changed")
    if status is not None and value.get("status") != status:
        raise StreamingIntegrityError(f"{field} is incomplete")


def require_dataset_label_geometry(dataset: str, label_dim: int) -> None:
    if dataset == "nuswide" and label_dim != 21:
        raise StreamingIntegrityError("NUS-WIDE streaming evaluation requires TC21 labels")
    expected = {"mirflickr": 24, "nuswide": 21, "mscoco": 80, "synthetic": 3}
    if dataset not in expected or label_dim != expected[dataset]:
        raise StreamingIntegrityError(
            f"dataset/label geometry changed: dataset={dataset!r}, labels={label_dim}"
        )


__all__ = [
    "StreamingIntegrityError",
    "array_descriptor",
    "atomic_save_npy",
    "atomic_write_json",
    "load_json",
    "numeric_sha256",
    "production_code_inventory",
    "require_no_link_components",
    "require_disjoint_paths",
    "reject_unsafe_output_path",
    "require_dataset_label_geometry",
    "require_hashed_json",
    "sha256_file",
    "sha256_json",
]
