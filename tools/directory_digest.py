"""Compute a platform-independent recursive SHA-256 directory inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def directory_digest(root: Path) -> dict[str, object]:
    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise ValueError("root must be a directory")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted((item for item in base.rglob("*") if item.is_file()), key=lambda item: item.relative_to(base).as_posix()):
        size = path.stat().st_size
        record = {
            "path": path.relative_to(base).as_posix(),
            "sha256": _file_sha256(path),
            "size": size,
        }
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
        files += 1
        total_bytes += size
    return {
        "schema": "canonical_recursive_directory_digest_v1",
        "files": files,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(directory_digest(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

