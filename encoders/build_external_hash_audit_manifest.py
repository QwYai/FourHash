#!/usr/bin/env python3
"""Build a strict array-level audit manifest without changing a frozen NPZ."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from load_external_feature_hash_codes import (  # noqa: E402
    REQUIRED_ARRAYS,
    load_external_feature_hash_bundle,
    manifest_array_sha256,
    sha256_file,
)


def build_audit_manifest(
    artifact: Path,
    run_manifest: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Re-hash every NPZ member and preserve immutable run provenance."""

    artifact = artifact.resolve()
    run_manifest = run_manifest.resolve()
    output = output.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not run_manifest.is_file():
        raise FileNotFoundError(run_manifest)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    expected_artifact_hash = (
        run_payload.get("files", {}).get(artifact.name, {}).get("sha256")
    )
    observed_artifact_hash = sha256_file(artifact)
    if expected_artifact_hash != observed_artifact_hash:
        raise ValueError("frozen NPZ hash disagrees with upstream RUN_MANIFEST")

    with np.load(str(artifact), allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError("unexpected frozen NPZ keys: {}".format(archive.files))
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    records = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": manifest_array_sha256(value),
        }
        for name, value in arrays.items()
    }
    payload: dict[str, Any] = {
        "format_version": 1,
        "manifest_role": (
            "array-level compatibility/audit manifest; frozen NPZ bytes unchanged"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            artifact.name: {
                "bytes": int(artifact.stat().st_size),
                "sha256": observed_artifact_hash,
            }
        },
        "metadata": metadata,
        "npz_arrays": records,
        "upstream_run_manifest_provenance": {
            "path": str(run_manifest),
            "bytes": int(run_manifest.stat().st_size),
            "sha256": sha256_file(run_manifest),
            "payload": run_payload,
        },
        "frozen_artifact_assertion": {
            "npz_rewritten": False,
            "npz_sha256": observed_artifact_hash,
            "training_source_sha256": run_payload.get("source_sha256"),
            "note": (
                "This builder is post-hoc provenance tooling, not the source "
                "that trained the frozen artifact."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))
    load_external_feature_hash_bundle(
        artifact,
        manifest_path=output,
        require_manifest=True,
        require_usable=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = build_audit_manifest(
        args.artifact,
        args.run_manifest,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "STRICT_LOADER_PASS",
                "artifact_sha256": payload["frozen_artifact_assertion"][
                    "npz_sha256"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
