"""Canonical hashing and atomic artifact helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    numeric_sha256,
    sha256_file,
    sha256_json,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CODE_SCHEMA = "raw_rebuilt_neural_code_inventory_v1"


def production_code_inventory() -> dict[str, Any]:
    """Hash the exact model, boundary, and runner code used by an artifact."""

    paths: list[Path] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "tests" not in path.parts and "__pycache__" not in path.parts:
            paths.append(path)
    paths.append(PROJECT_ROOT / "rz_csd_clip512.py")
    for name in ("contract.py", "loader.py", "materialize.py", "validation.py"):
        paths.append(PROJECT_ROOT / "raw_rebuilt_runtime" / name)
    files = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
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


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".pending")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".pending")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def reject_unsafe_output_path(path: Path, *, field: str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    forbidden = {"oraldata", "processdata"}
    if any(part.casefold() in forbidden for part in candidate.parts):
        raise ValueError(f"{field} must be outside protected raw and prepared inputs")
    if candidate.suffix.casefold() == ".mat":
        raise ValueError(f"{field} may not be a legacy MAT artifact")
    return candidate


def require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], *, field: str) -> None:
    expected = set(keys)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{field} keys differ: missing={missing}, extra={extra}")


__all__ = [
    "array_descriptor",
    "atomic_save_npy",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "load_json",
    "numeric_sha256",
    "production_code_inventory",
    "reject_unsafe_output_path",
    "require_exact_keys",
    "sha256_file",
    "sha256_json",
]

