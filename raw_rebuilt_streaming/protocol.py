"""Atomic two-stage spool protocol shared by isolated rank and metric workers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from .codes import CodeState
from .integrity import (
    StreamingIntegrityError,
    atomic_save_npy,
    atomic_write_json,
    load_json,
    numeric_sha256,
    reject_unsafe_output_path,
    require_disjoint_paths,
    require_hashed_json,
    sha256_file,
    sha256_json,
)
from .plan import ChunkSpec, FrozenRankPlan


BUNDLE_SCHEMA = "raw_rebuilt_streaming_distance_bundle_v1"
ACK_SCHEMA = "raw_rebuilt_streaming_metric_ack_v1"
EVIDENCE_SCHEMA = "raw_rebuilt_streaming_rank_evidence_seal_v1"


@dataclass(frozen=True)
class BundleEvidence:
    root: Path
    distances: np.ndarray
    manifest: Mapping[str, Any]

    def close(self) -> None:
        mmap = getattr(self.distances, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _reject_symlink_components(target: Path, trusted_root: Path, *, field: str) -> None:
    root = Path(trusted_root)
    candidate = Path(target)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise StreamingIntegrityError(f"{field} escapes its trusted root") from error
    current = root
    if current.is_symlink():
        raise StreamingIntegrityError(f"{field} trusted root is a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise StreamingIntegrityError(f"{field} contains a symlink component")


def prepare_spool_root(
    path: Path, *, forbidden: Mapping[str, Path] | None = None
) -> Path:
    root = reject_unsafe_output_path(Path(path), field="streaming spool")
    if forbidden:
        require_disjoint_paths(root, forbidden, field="streaming spool")
    root.mkdir(parents=True, exist_ok=True)
    return root


def bundle_path(spool_root: Path, chunk: ChunkSpec) -> Path:
    return Path(spool_root) / "bundles" / chunk.cell_id / chunk.name


def ack_path(spool_root: Path, chunk: ChunkSpec) -> Path:
    return Path(spool_root) / "acks" / chunk.cell_id / f"{chunk.name}.json"


def evidence_seal_path(spool_root: Path) -> Path:
    return Path(spool_root) / "rank_evidence_complete.json"


def chunk_binding(plan: FrozenRankPlan, state: CodeState, chunk: ChunkSpec) -> dict[str, Any]:
    runtime = state.manifest["runtime_identity"]
    query_modality = "image" if chunk.direction == "i2t" else "text"
    database_modality = "text" if chunk.direction == "i2t" else "image"
    body = {
        "rank_plan_sha256": plan.plan_sha256,
        "plan_binding_sha256": plan.manifest["binding"]["plan_binding_sha256"],
        "code_state_manifest_sha256": state.manifest["manifest_sha256"],
        "encoding_binding_sha256": state.manifest["binding"]["encoding_binding_sha256"],
        "source_seal_sha256": state.manifest["source_seal_sha256"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "row_ids_numeric_sha256": runtime["row_ids_numeric_sha256"],
        "indQ_numeric_sha256": runtime["indQ_numeric_sha256"],
        "indD_numeric_sha256": runtime["indD_numeric_sha256"],
        "query_code_numeric_sha256": state.manifest["arrays"][f"query_{query_modality}_{chunk.bits}"][
            "numeric_sha256"
        ],
        "database_code_numeric_sha256": state.manifest["arrays"][
            f"database_{database_modality}_{chunk.bits}"
        ]["numeric_sha256"],
        "cell_id": chunk.cell_id,
        "direction": chunk.direction,
        "bits": chunk.bits,
        "ordinal": chunk.ordinal,
        "start": chunk.start,
        "end": chunk.end,
        "database_rows": chunk.database_rows,
        "database_order": "frozen runtime indD order",
    }
    return {**body, "chunk_binding_sha256": sha256_json(body)}


def _safe_remove_bundle_directory(path: Path, spool_root: Path) -> None:
    _reject_symlink_components(
        Path(path), Path(spool_root), field="bundle deletion path"
    )
    if Path(path).is_symlink():
        raise StreamingIntegrityError("refusing to delete a symlinked bundle")
    resolved = Path(path).resolve(strict=True)
    bundle_root = (Path(spool_root) / "bundles").resolve(strict=True)
    try:
        resolved.relative_to(bundle_root)
    except ValueError as error:
        raise StreamingIntegrityError("refusing to delete outside the bounded bundle spool") from error
    if not resolved.is_dir():
        raise StreamingIntegrityError("published bundle path is not a directory")
    shutil.rmtree(resolved)


def _clean_pending(target: Path, spool_root: Path) -> None:
    pending = target.with_name(target.name + ".pending")
    if pending.exists():
        _safe_remove_bundle_directory(pending, spool_root)


def discard_pending_bundle(
    spool_root: Path,
    chunk: ChunkSpec,
) -> None:
    """Remove only this producer-owned, non-symlink pending directory."""

    _clean_pending(bundle_path(spool_root, chunk), Path(spool_root))


def publish_bundle(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
    distances: np.ndarray,
) -> Path:
    spool = prepare_spool_root(spool_root)
    binding = chunk_binding(plan, state, chunk)
    value = np.asarray(distances)
    expected_shape = (chunk.end - chunk.start, chunk.database_rows)
    if value.shape != expected_shape or value.dtype != np.uint8:
        raise StreamingIntegrityError("rank worker produced invalid distance geometry")
    if np.any(value > chunk.bits):
        raise StreamingIntegrityError("rank worker produced a distance larger than the code width")
    target = bundle_path(spool, chunk)
    _reject_symlink_components(target, spool, field="distance bundle output")
    target.parent.mkdir(parents=True, exist_ok=True)
    _clean_pending(target, spool)
    if target.exists():
        evidence = open_bundle(spool, plan, state, chunk)
        evidence.close()
        return target
    pending = target.with_name(target.name + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    try:
        distance_path = pending / "distances.npy"
        atomic_save_npy(distance_path, np.ascontiguousarray(value))
        descriptor = {
            "path": "distances.npy",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "size": distance_path.stat().st_size,
            "file_sha256": sha256_file(distance_path),
            "numeric_sha256": numeric_sha256(value),
            "minimum": int(value.min()),
            "maximum": int(value.max()),
        }
        body = {
            "schema": BUNDLE_SCHEMA,
            "status": "PUBLISHED",
            "binding": binding,
            "distances": descriptor,
            "labels_loaded_by_rank_worker": False,
        }
        atomic_write_json(
            pending / "bundle.json",
            {**body, "bundle_manifest_sha256": sha256_json(body)},
        )
        os.replace(pending, target)
    except Exception:
        if pending.exists():
            _safe_remove_bundle_directory(pending, spool)
        raise
    return target


def open_bundle(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
) -> BundleEvidence:
    declared_root = bundle_path(spool_root, chunk)
    _reject_symlink_components(
        declared_root, Path(spool_root), field="distance bundle path"
    )
    if declared_root.is_symlink() or not declared_root.is_dir():
        raise StreamingIntegrityError("distance bundle must be a regular directory")
    root = declared_root.resolve(strict=True)
    bundle_parent = (Path(spool_root) / "bundles").resolve(strict=True)
    try:
        root.relative_to(bundle_parent)
    except ValueError as error:
        raise StreamingIntegrityError("distance bundle escapes its spool") from error
    for name in ("bundle.json", "distances.npy"):
        target = root / name
        if target.is_symlink() or not target.is_file():
            raise StreamingIntegrityError("distance bundle files must be regular non-symlinks")
    manifest = load_json(root / "bundle.json")
    require_hashed_json(
        manifest,
        hash_field="bundle_manifest_sha256",
        schema=BUNDLE_SCHEMA,
        status="PUBLISHED",
        field="distance bundle",
    )
    if set(manifest) != {
        "schema",
        "status",
        "binding",
        "distances",
        "labels_loaded_by_rank_worker",
        "bundle_manifest_sha256",
    }:
        raise StreamingIntegrityError("distance bundle contains unbound fields")
    if manifest.get("labels_loaded_by_rank_worker") is not False:
        raise StreamingIntegrityError("rank worker declared a label-bearing execution")
    if manifest.get("binding") != chunk_binding(plan, state, chunk):
        raise StreamingIntegrityError("distance bundle was replayed or rebound")
    descriptor = manifest.get("distances")
    if not isinstance(descriptor, Mapping) or descriptor.get("path") != "distances.npy":
        raise StreamingIntegrityError("distance bundle descriptor changed")
    if set(descriptor) != {
        "path",
        "dtype",
        "shape",
        "size",
        "file_sha256",
        "numeric_sha256",
        "minimum",
        "maximum",
    }:
        raise StreamingIntegrityError("distance bundle descriptor contains unbound fields")
    distance_path = root / "distances.npy"
    if distance_path.stat().st_size != descriptor.get("size") or sha256_file(distance_path) != descriptor.get(
        "file_sha256"
    ):
        raise StreamingIntegrityError("distance bundle bytes changed")
    value = np.load(distance_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (chunk.end - chunk.start, chunk.database_rows)
    if (
        value.shape != expected_shape
        or value.dtype != np.uint8
        or descriptor.get("shape") != list(expected_shape)
        or descriptor.get("dtype") != np.dtype(np.uint8).str
    ):
        raise StreamingIntegrityError("distance bundle geometry changed")
    if numeric_sha256(value) != descriptor.get("numeric_sha256"):
        raise StreamingIntegrityError("distance bundle numeric content changed")
    observed_min = int(value.min())
    observed_max = int(value.max())
    if (
        descriptor.get("minimum") != observed_min
        or descriptor.get("maximum") != observed_max
        or observed_min < 0
        or observed_max > chunk.bits
    ):
        raise StreamingIntegrityError("distance bundle contains an invalid Hamming radius")
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    if actual != ["bundle.json", "distances.npy"]:
        raise StreamingIntegrityError("distance bundle has unbound files")
    return BundleEvidence(root=root, distances=value, manifest=manifest)


def seal_rank_evidence(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
) -> Mapping[str, Any]:
    """Publish one immutable seal only after every planned bundle verifies."""

    spool = prepare_spool_root(spool_root)
    if (spool / "acks").exists() and any((spool / "acks").rglob("*.json")):
        raise StreamingIntegrityError("rank evidence cannot freeze after metric ACKs exist")
    descriptors = []
    for chunk in plan.chunks():
        evidence = open_bundle(spool, plan, state, chunk)
        try:
            manifest_path = evidence.root / "bundle.json"
            distance_path = evidence.root / "distances.npy"
            descriptors.append(
                {
                    "cell_id": chunk.cell_id,
                    "ordinal": chunk.ordinal,
                    "start": chunk.start,
                    "end": chunk.end,
                    "path": evidence.root.relative_to(spool).as_posix(),
                    "bundle_manifest_sha256": evidence.manifest["bundle_manifest_sha256"],
                    "bundle_json_size": manifest_path.stat().st_size,
                    "bundle_json_sha256": sha256_file(manifest_path),
                    "distances_size": distance_path.stat().st_size,
                    "distances_file_sha256": evidence.manifest["distances"]["file_sha256"],
                    "distances_numeric_sha256": evidence.manifest["distances"][
                        "numeric_sha256"
                    ],
                }
            )
        finally:
            evidence.close()
    body = {
        "schema": EVIDENCE_SCHEMA,
        "status": "rank_evidence_frozen",
        "rank_plan_sha256": plan.plan_sha256,
        "plan_binding_sha256": plan.manifest["binding"]["plan_binding_sha256"],
        "code_state_manifest_sha256": state.manifest["manifest_sha256"],
        "source_seal_sha256": state.manifest["source_seal_sha256"],
        "total_chunks": len(descriptors),
        "total_distance_values": int(
            sum((chunk.end - chunk.start) * chunk.database_rows for chunk in plan.chunks())
        ),
        "distance_dtype": "uint8",
        "labels_loaded_by_rank_process": False,
        "bundles": descriptors,
    }
    seal = {**body, "rank_evidence_seal_sha256": sha256_json(body)}
    target = evidence_seal_path(spool)
    if target.exists():
        existing = load_json(target)
        if existing != seal:
            raise StreamingIntegrityError("rank evidence seal was rebound")
    else:
        atomic_write_json(target, seal)
    return seal


def open_rank_evidence_seal(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
) -> Mapping[str, Any]:
    target = evidence_seal_path(spool_root)
    _reject_symlink_components(
        target, Path(spool_root), field="rank evidence seal path"
    )
    if target.is_symlink() or not target.is_file():
        raise StreamingIntegrityError("rank evidence seal must be a regular non-symlink file")
    seal = load_json(target)
    require_hashed_json(
        seal,
        hash_field="rank_evidence_seal_sha256",
        schema=EVIDENCE_SCHEMA,
        status="rank_evidence_frozen",
        field="rank evidence seal",
    )
    if set(seal) != {
        "schema",
        "status",
        "rank_plan_sha256",
        "plan_binding_sha256",
        "code_state_manifest_sha256",
        "source_seal_sha256",
        "total_chunks",
        "total_distance_values",
        "distance_dtype",
        "labels_loaded_by_rank_process",
        "bundles",
        "rank_evidence_seal_sha256",
    }:
        raise StreamingIntegrityError("rank evidence seal contains unbound fields")
    if (
        seal.get("rank_plan_sha256") != plan.plan_sha256
        or seal.get("plan_binding_sha256") != plan.manifest["binding"]["plan_binding_sha256"]
        or seal.get("code_state_manifest_sha256") != state.manifest["manifest_sha256"]
        or seal.get("source_seal_sha256") != state.manifest["source_seal_sha256"]
        or seal.get("labels_loaded_by_rank_process") is not False
        or seal.get("distance_dtype") != "uint8"
    ):
        raise StreamingIntegrityError("rank evidence seal binding changed")
    descriptors = seal.get("bundles")
    chunks = list(plan.chunks())
    if not isinstance(descriptors, list) or len(descriptors) != len(chunks):
        raise StreamingIntegrityError("rank evidence seal coverage changed")
    expected_values = int(
        sum((chunk.end - chunk.start) * chunk.database_rows for chunk in chunks)
    )
    if (
        type(seal.get("total_chunks")) is not int
        or type(seal.get("total_distance_values")) is not int
        or seal.get("total_chunks") != len(chunks)
        or seal.get("total_distance_values") != expected_values
    ):
        raise StreamingIntegrityError("rank evidence seal totals changed")
    for descriptor, chunk in zip(descriptors, chunks):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "cell_id",
            "ordinal",
            "start",
            "end",
            "path",
            "bundle_manifest_sha256",
            "bundle_json_size",
            "bundle_json_sha256",
            "distances_size",
            "distances_file_sha256",
            "distances_numeric_sha256",
        }:
            raise StreamingIntegrityError("rank evidence descriptor contains unbound fields")
        expected = {
            "cell_id": chunk.cell_id,
            "ordinal": chunk.ordinal,
            "start": chunk.start,
            "end": chunk.end,
            "path": bundle_path(spool_root, chunk).relative_to(Path(spool_root)).as_posix(),
        }
        if any(descriptor.get(key) != value for key, value in expected.items()):
            raise StreamingIntegrityError("rank evidence descriptor order/geometry changed")
        for key in (
            "bundle_manifest_sha256",
            "bundle_json_sha256",
            "distances_file_sha256",
            "distances_numeric_sha256",
        ):
            value = descriptor.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise StreamingIntegrityError("rank evidence descriptor hash is invalid")
        for key in ("bundle_json_size", "distances_size"):
            value = descriptor.get(key)
            if type(value) is not int or value < 1:
                raise StreamingIntegrityError("rank evidence descriptor size is invalid")
        target = bundle_path(spool_root, chunk)
        if target.exists():
            evidence = open_bundle(spool_root, plan, state, chunk)
            try:
                manifest_path = evidence.root / "bundle.json"
                distance_path = evidence.root / "distances.npy"
                if (
                    descriptor.get("bundle_manifest_sha256")
                    != evidence.manifest["bundle_manifest_sha256"]
                    or descriptor.get("bundle_json_size") != manifest_path.stat().st_size
                    or descriptor.get("bundle_json_sha256") != sha256_file(manifest_path)
                    or descriptor.get("distances_size") != distance_path.stat().st_size
                    or descriptor.get("distances_file_sha256")
                    != evidence.manifest["distances"]["file_sha256"]
                    or descriptor.get("distances_numeric_sha256")
                    != evidence.manifest["distances"]["numeric_sha256"]
                ):
                    raise StreamingIntegrityError("rank evidence bundle differs from its seal")
            finally:
                evidence.close()
    return seal


def _ack_body(
    *,
    binding: Mapping[str, Any],
    bundle: Mapping[str, Any],
    opaque_metric_commitment_sha256: str,
    previous_chain_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": ACK_SCHEMA,
        "status": "ACKNOWLEDGED",
        "binding": dict(binding),
        "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
        "distances_file_sha256": bundle["distances"]["file_sha256"],
        "distances_numeric_sha256": bundle["distances"]["numeric_sha256"],
        "opaque_metric_commitment_sha256": opaque_metric_commitment_sha256,
        "previous_ack_chain_sha256": previous_chain_sha256,
        "metric_payload_exposed_to_rank_worker": False,
    }


def write_ack(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
    bundle: Mapping[str, Any],
    opaque_metric_commitment_sha256: str,
    previous_chain_sha256: str,
) -> Mapping[str, Any]:
    binding = chunk_binding(plan, state, chunk)
    body = _ack_body(
        binding=binding,
        bundle=bundle,
        opaque_metric_commitment_sha256=opaque_metric_commitment_sha256,
        previous_chain_sha256=previous_chain_sha256,
    )
    ack_sha = sha256_json(body)
    chain = sha256_json(
        {"previous_ack_chain_sha256": previous_chain_sha256, "ack_sha256": ack_sha}
    )
    receipt = {**body, "ack_sha256": ack_sha, "ack_chain_sha256": chain}
    target = ack_path(spool_root, chunk)
    _reject_symlink_components(
        target, Path(spool_root), field="metric ACK output"
    )
    if target.exists():
        existing = load_json(target)
        if existing != receipt:
            raise StreamingIntegrityError("metric ACK output was rebound")
    else:
        atomic_write_json(target, receipt)
    return receipt


def verify_ack(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
    previous_chain_sha256: str,
) -> Mapping[str, Any]:
    target = ack_path(spool_root, chunk)
    _reject_symlink_components(target, Path(spool_root), field="metric ACK path")
    if target.is_symlink() or not target.is_file():
        raise StreamingIntegrityError("metric ACK must be a regular non-symlink file")
    receipt = load_json(target)
    body = {
        key: receipt[key]
        for key in receipt
        if key not in {"ack_sha256", "ack_chain_sha256"}
    }
    if receipt.get("schema") != ACK_SCHEMA or receipt.get("status") != "ACKNOWLEDGED":
        raise StreamingIntegrityError("metric ACK schema/status changed")
    if receipt.get("ack_sha256") != sha256_json(body):
        raise StreamingIntegrityError("metric ACK hash changed")
    expected_chain = sha256_json(
        {
            "previous_ack_chain_sha256": previous_chain_sha256,
            "ack_sha256": receipt["ack_sha256"],
        }
    )
    if receipt.get("ack_chain_sha256") != expected_chain:
        raise StreamingIntegrityError("metric ACK chain changed")
    if receipt.get("previous_ack_chain_sha256") != previous_chain_sha256:
        raise StreamingIntegrityError("metric ACK predecessor changed")
    if receipt.get("binding") != chunk_binding(plan, state, chunk):
        raise StreamingIntegrityError("metric ACK was replayed or rebound")
    expected_keys = {
        "schema",
        "status",
        "binding",
        "bundle_manifest_sha256",
        "distances_file_sha256",
        "distances_numeric_sha256",
        "opaque_metric_commitment_sha256",
        "previous_ack_chain_sha256",
        "metric_payload_exposed_to_rank_worker",
        "ack_sha256",
        "ack_chain_sha256",
    }
    if set(receipt) != expected_keys:
        raise StreamingIntegrityError("metric ACK schema exposes unbound fields")
    commitment = receipt.get("opaque_metric_commitment_sha256")
    if not isinstance(commitment, str) or len(commitment) != 64 or any(
        value not in "0123456789abcdef" for value in commitment
    ):
        raise StreamingIntegrityError("metric ACK commitment is invalid")
    if receipt.get("metric_payload_exposed_to_rank_worker") is not False:
        raise StreamingIntegrityError("metric ACK exposed label-derived payload")
    return receipt


def replay_cell_acks(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    cell_id: str,
) -> tuple[list[Mapping[str, Any]], str]:
    chunks = [chunk for chunk in plan.chunks() if chunk.cell_id == cell_id]
    receipts: list[Mapping[str, Any]] = []
    chain = "0" * 64
    missing_seen = False
    for chunk in chunks:
        target = ack_path(spool_root, chunk)
        if not target.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise StreamingIntegrityError("metric ACKs are noncontiguous within a cell")
        receipt = verify_ack(spool_root, plan, state, chunk, chain)
        receipts.append(receipt)
        chain = str(receipt["ack_chain_sha256"])
    return receipts, chain


def delete_acknowledged_bundle(
    spool_root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
    ack: Mapping[str, Any],
) -> None:
    target = bundle_path(spool_root, chunk)
    if not target.exists():
        return
    evidence = open_bundle(spool_root, plan, state, chunk)
    try:
        if (
            ack.get("bundle_manifest_sha256") != evidence.manifest["bundle_manifest_sha256"]
            or ack.get("distances_file_sha256") != evidence.manifest["distances"]["file_sha256"]
            or ack.get("distances_numeric_sha256") != evidence.manifest["distances"]["numeric_sha256"]
        ):
            raise StreamingIntegrityError("published bundle differs from its metric ACK")
    finally:
        evidence.close()
    _safe_remove_bundle_directory(target, spool_root)


__all__ = [
    "ACK_SCHEMA",
    "BUNDLE_SCHEMA",
    "EVIDENCE_SCHEMA",
    "BundleEvidence",
    "ack_path",
    "bundle_path",
    "chunk_binding",
    "delete_acknowledged_bundle",
    "discard_pending_bundle",
    "evidence_seal_path",
    "open_bundle",
    "open_rank_evidence_seal",
    "prepare_spool_root",
    "publish_bundle",
    "replay_cell_acks",
    "seal_rank_evidence",
    "verify_ack",
    "write_ack",
]
