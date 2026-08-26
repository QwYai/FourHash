"""Storage-bounded formal evaluation of frozen CCDE codes.

The first command freezes and hashes the primary/detail binary codes and an
evaluation plan without loading query/database labels.  The second command
verifies that state, opens labels, and computes complete-gallery integer
distances in bounded chunks.  No rank matrix, validation gate, or per-cell
fallback is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_contract import CCDE_DETAIL_CAP, freeze_binding
from raw_rebuilt_neural.ccde_detail_bits import open_detail_bit_artifact
from raw_rebuilt_neural.ccde_ranking import (
    CCDE_RANK_MODE,
    _encoding_binding,
    _open_encoding_cache,
    _validate_model_pair,
    ensure_ccde_encoding_cache,
)
from raw_rebuilt_neural.ccde_training import load_detail_checkpoint
from raw_rebuilt_neural.integrity import production_code_inventory as neural_code_inventory
from raw_rebuilt_neural.training import load_trained_checkpoint
from raw_rebuilt_runtime import load_label_free_rank_inputs, load_metric_labels
from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    load_json,
    numeric_sha256,
    sha256_file,
    sha256_json,
)
from raw_rebuilt_streaming.integrity import (
    production_code_inventory as streaming_code_inventory,
    reject_unsafe_output_path,
)
from raw_rebuilt_streaming.metrics import (
    _precompute_jaccard_idcg,
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
    mean_query_metrics,
)
from rz_csd_clip512 import BITS


PLAN_SCHEMA = "raw_rebuilt_ccde_storage_bounded_plan_v1"
PARTIAL_SCHEMA = "raw_rebuilt_ccde_storage_bounded_partial_v1"
RESULT_SCHEMA = "raw_rebuilt_ccde_storage_bounded_metric_v1"
EVALUATION_SCHEMA = "raw_rebuilt_ccde_storage_bounded_evaluation_v1"
DIRECTIONS = ("i2t", "t2i")
METHODS = ("primary_hamming", "ccde_lexicographic")
POPCOUNT = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)


class FormalCCDEError(RuntimeError):
    """Raised when frozen formal evidence or a resume receipt differs."""


@dataclass(frozen=True)
class FormalCCDEConfig:
    bits: tuple[int, ...] = BITS
    directions: tuple[str, ...] = DIRECTIONS
    cutoffs: tuple[int, ...] = (50, 100, 1000)
    query_chunk_size: int = 4

    def __post_init__(self) -> None:
        bits = tuple(int(value) for value in self.bits)
        directions = tuple(str(value) for value in self.directions)
        cutoffs = tuple(int(value) for value in self.cutoffs)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "cutoffs", cutoffs)
        if not bits or len(set(bits)) != len(bits) or any(value not in BITS for value in bits):
            raise ValueError(f"bits must be a unique nonempty subset of {BITS}")
        if (
            not directions
            or len(set(directions)) != len(directions)
            or any(value not in DIRECTIONS for value in directions)
        ):
            raise ValueError(f"directions must be a unique nonempty subset of {DIRECTIONS}")
        if tuple(sorted(set(cutoffs))) != cutoffs or not cutoffs or cutoffs[0] < 1:
            raise ValueError("cutoffs must be sorted, unique, and positive")
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")


def _runtime_manifest(runtime_root: Path) -> Mapping[str, Any]:
    value = load_json(Path(runtime_root).expanduser().resolve(strict=True) / "runtime_manifest.json")
    if value.get("status") != "COMPLETE":
        raise FormalCCDEError("runtime manifest is incomplete")
    return value


def _runtime_identity(rank: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "dataset": str(metadata["dataset"]),
        "rows": len(rank.row_ids),
        "label_dim": int(metadata["label_dim"]),
        "source_seal_sha256": rank.source_seal_sha256,
        "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
        "query_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.query_idx]),
        "database_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.database_idx]),
        "indQ_numeric_sha256": numeric_sha256(rank.query_idx),
        "indT_numeric_sha256": numeric_sha256(rank.train_idx),
        "indD_numeric_sha256": numeric_sha256(rank.database_idx),
        "query_rows": len(rank.query_idx),
        "train_rows": len(rank.train_idx),
        "database_rows": len(rank.database_idx),
    }
    return {**body, "runtime_identity_sha256": sha256_json(body)}


def _implementation_inventory() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(strict=True),
        (PROJECT_ROOT / "raw_rebuilt_streaming" / "metrics.py").resolve(strict=True),
        (PROJECT_ROOT / "raw_rebuilt_neural" / "ccde_ranking.py").resolve(strict=True),
        (PROJECT_ROOT / "raw_rebuilt_neural" / "ccde_contract.py").resolve(strict=True),
    )
    files = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    body = {"schema": "raw_rebuilt_ccde_streaming_code_inventory_v1", "files": files}
    return {**body, "code_inventory_sha256": sha256_json(body)}


def _load_models_and_bits(
    rank: Any,
    primary_checkpoint_path: Path,
    detail_checkpoint_path: Path,
    detail_bit_artifact_root: Path,
    architecture_freeze_path: Path,
    *,
    device: str | torch.device,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    frozen = freeze_binding(architecture_freeze_path)
    primary = load_trained_checkpoint(
        primary_checkpoint_path,
        device=device,
        expected_source_seal_sha256=rank.source_seal_sha256,
    )
    detail = load_detail_checkpoint(
        detail_checkpoint_path,
        architecture_freeze_path,
        device=device,
        expected_source_seal_sha256=rank.source_seal_sha256,
    )
    bit_artifact = open_detail_bit_artifact(
        detail_bit_artifact_root,
        architecture_freeze_path,
        expected_source_seal_sha256=rank.source_seal_sha256,
        expected_checkpoint_sha256=detail.checkpoint_sha256,
    )
    _validate_model_pair(primary, detail, bit_artifact)
    return primary, detail, bit_artifact, frozen


def freeze_formal_state(
    runtime_root: Path,
    primary_checkpoint_path: Path,
    detail_checkpoint_path: Path,
    detail_bit_artifact_root: Path,
    architecture_freeze_path: Path,
    output_parent: Path,
    *,
    config: FormalCCDEConfig = FormalCCDEConfig(),
    device: str | torch.device = "auto",
    _test_allow_synthetic: bool = False,
) -> Path:
    """Freeze label-free codes and a content-bound evaluation plan."""

    rank = load_label_free_rank_inputs(
        runtime_root, _test_allow_synthetic=_test_allow_synthetic
    )
    try:
        metadata = _runtime_manifest(runtime_root)
        identity = _runtime_identity(rank, metadata)
        if metadata.get("source_seal_sha256") != rank.source_seal_sha256:
            raise FormalCCDEError("runtime manifest and label-free source seals differ")
        primary, detail, bit_artifact, frozen = _load_models_and_bits(
            rank,
            primary_checkpoint_path,
            detail_checkpoint_path,
            detail_bit_artifact_root,
            architecture_freeze_path,
            device=device,
        )
        try:
            config_value = asdict(config)
            binding_body = {
                "schema": PLAN_SCHEMA,
                "dataset": identity["dataset"],
                "label_dim": identity["label_dim"],
                "runtime_identity": identity,
                "primary_checkpoint_sha256": primary.checkpoint_sha256,
                "primary_checkpoint_run_binding_sha256": primary.metadata["binding"][
                    "run_binding_sha256"
                ],
                "detail_checkpoint_sha256": detail.checkpoint_sha256,
                "detail_checkpoint_run_binding_sha256": detail.metadata["binding"][
                    "run_binding_sha256"
                ],
                "detail_bit_artifact_sha256": bit_artifact.manifest[
                    "detail_bit_artifact_sha256"
                ],
                "architecture_freeze": frozen,
                "config": config_value,
                "config_sha256": sha256_json(config_value),
                "neural_code_inventory": neural_code_inventory(),
                "streaming_code_inventory": streaming_code_inventory(),
                "implementation_inventory": _implementation_inventory(),
                "formal_query_or_database_labels_opened": False,
            }
            binding = {
                **binding_body,
                "plan_binding_sha256": sha256_json(binding_body),
            }
            output = reject_unsafe_output_path(
                Path(output_parent), field="CCDE storage-bounded plan output"
            )
            root = output / f"ccde-plan-{binding['plan_binding_sha256'][:16]}"
            root.mkdir(parents=True, exist_ok=True)
            cache = ensure_ccde_encoding_cache(
                rank, primary, detail, bit_artifact, frozen, root
            )
            try:
                cache_manifest_path = cache.root / "manifest.json"
                plan_body = {
                    "schema": PLAN_SCHEMA,
                    "status": "rank_state_frozen",
                    "dataset": identity["dataset"],
                    "label_dim": identity["label_dim"],
                    "source_seal_sha256": identity["source_seal_sha256"],
                    "runtime_identity": identity,
                    "binding": binding,
                    "encoding_cache": {
                        "path": cache.root.relative_to(root).as_posix(),
                        "manifest_size": cache_manifest_path.stat().st_size,
                        "manifest_sha256": sha256_file(cache_manifest_path),
                        "encoding_binding_sha256": cache.manifest["binding"][
                            "encoding_binding_sha256"
                        ],
                    },
                    "methods": list(METHODS),
                    "ranking_rule": (
                        "primary Hamming; CCDE primary_distance*(detail_bits+1)+detail_distance"
                    ),
                    "detail_bits_by_primary_width": {
                        str(bits): min(CCDE_DETAIL_CAP, bits) for bits in BITS
                    },
                    "primary_shell_order_is_invariant": True,
                    "formal_gate_or_fallback_used": False,
                    "distance_artifact_storage": "none; deterministic bounded recomputation from frozen codes",
                    "metric_label_boundary": "labels open only after this plan and all code arrays verify",
                    "labels_loaded_during_freeze": False,
                }
                plan = {
                    **plan_body,
                    "rank_plan_sha256": sha256_json(plan_body),
                }
                target = root / "evaluation_plan.json"
                if target.exists():
                    if load_json(target) != plan:
                        raise FormalCCDEError("completed CCDE evaluation plan was rebound")
                else:
                    atomic_write_json(target, plan)
                return root
            finally:
                cache.close()
        finally:
            bit_artifact.close()
    finally:
        rank.close()


def _verify_plan(plan_root: Path) -> Mapping[str, Any]:
    root = Path(plan_root).expanduser().resolve(strict=True)
    plan = load_json(root / "evaluation_plan.json")
    body = {key: plan[key] for key in plan if key != "rank_plan_sha256"}
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "rank_state_frozen":
        raise FormalCCDEError("CCDE evaluation plan schema/status differs")
    if sha256_json(body) != plan.get("rank_plan_sha256"):
        raise FormalCCDEError("CCDE evaluation plan hash changed")
    if plan.get("labels_loaded_during_freeze") is not False:
        raise FormalCCDEError("CCDE plan crossed the metric-label boundary")
    if plan.get("formal_gate_or_fallback_used") is not False:
        raise FormalCCDEError("CCDE plan contains a formal gate or fallback")
    if plan.get("primary_shell_order_is_invariant") is not True:
        raise FormalCCDEError("CCDE plan does not preserve primary shells")
    binding = plan.get("binding", {})
    binding_body = {
        key: binding[key] for key in binding if key != "plan_binding_sha256"
    }
    if sha256_json(binding_body) != binding.get("plan_binding_sha256"):
        raise FormalCCDEError("CCDE plan binding hash changed")
    if binding.get("implementation_inventory") != _implementation_inventory():
        raise FormalCCDEError("current CCDE streaming implementation differs from the plan")
    if binding.get("neural_code_inventory") != neural_code_inventory():
        raise FormalCCDEError("current neural code differs from the plan")
    if binding.get("streaming_code_inventory") != streaming_code_inventory():
        raise FormalCCDEError("current streaming metric code differs from the plan")
    return plan


class _PackedDistanceBackend:
    def __init__(
        self,
        cache: Any,
        query_idx: np.ndarray,
        database_idx: np.ndarray,
        *,
        device: str,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("distance device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise FormalCCDEError("CUDA distance backend requested but unavailable")
        self.device = device
        self.query: dict[tuple[str, str, int], Any] = {}
        self.database: dict[tuple[str, str, int], Any] = {}
        for role, mappings in (
            ("primary", (cache.primary_image_codes, cache.primary_text_codes)),
            ("detail", (cache.detail_image_codes, cache.detail_text_codes)),
        ):
            for modality, values in zip(("image", "text"), mappings):
                for bits in BITS:
                    query = np.packbits(
                        np.asarray(values[bits][query_idx]) > 0,
                        axis=1,
                        bitorder="little",
                    )
                    database = np.packbits(
                        np.asarray(values[bits][database_idx]) > 0,
                        axis=1,
                        bitorder="little",
                    )
                    key = (role, modality, bits)
                    if device == "cuda":
                        self.query[key] = torch.as_tensor(
                            np.ascontiguousarray(query), dtype=torch.uint8, device="cuda"
                        )
                        self.database[key] = torch.as_tensor(
                            np.ascontiguousarray(database), dtype=torch.uint8, device="cuda"
                        )
                    else:
                        self.query[key] = np.ascontiguousarray(query, dtype=np.uint8)
                        self.database[key] = np.ascontiguousarray(database, dtype=np.uint8)
        self.lut = (
            torch.as_tensor(POPCOUNT, dtype=torch.uint8, device="cuda")
            if device == "cuda"
            else POPCOUNT
        )

    def distances(
        self,
        role: str,
        direction: str,
        bits: int,
        start: int,
        end: int,
    ) -> np.ndarray:
        query_modality = "image" if direction == "i2t" else "text"
        database_modality = "text" if direction == "i2t" else "image"
        query = self.query[(role, query_modality, bits)][start:end]
        database = self.database[(role, database_modality, bits)]
        if self.device == "cpu":
            result = np.empty((end - start, len(database)), dtype=np.uint16)
            for position, value in enumerate(query):
                xor = np.bitwise_xor(database, value[None, :])
                result[position] = self.lut[xor].sum(axis=1, dtype=np.uint16)
            return result
        rows = []
        for value in query:
            xor = torch.bitwise_xor(database, value.unsqueeze(0))
            rows.append(self.lut[xor.to(torch.int64)].sum(dim=1))
        if not rows:
            return np.empty((0, len(database)), dtype=np.uint16)
        return torch.stack(rows, dim=0).cpu().numpy().astype(np.uint16, copy=False)


def _query_truth(
    labels: Any,
    query_position: int,
    *,
    cutoffs: tuple[int, ...],
    database_label_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    query = np.asarray(labels.query[query_position], dtype=np.uint8)
    active = np.flatnonzero(query)
    if active.size:
        intersection = np.asarray(
            labels.database[:, active].sum(axis=1, dtype=np.uint16), dtype=np.uint16
        )
    else:
        intersection = np.zeros(len(labels.database), dtype=np.uint16)
    union = database_label_counts + np.uint16(active.size) - intersection
    gains = np.divide(
        intersection,
        union,
        out=np.zeros(len(intersection), dtype=np.float64),
        where=union != 0,
    )
    ideal = _precompute_jaccard_idcg(gains, cutoffs)
    return intersection != 0, ideal.gains, ideal


def _cell_id(direction: str, bits: int) -> str:
    return f"{direction}-bits-{bits}"


def _partial_root(output_root: Path, direction: str, bits: int) -> Path:
    return output_root / "partials" / _cell_id(direction, bits)


def _resume_cell(
    output_root: Path,
    plan: Mapping[str, Any],
    direction: str,
    bits: int,
) -> tuple[int, str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = _partial_root(output_root, direction, bits)
    paths = sorted(root.glob("chunk-*.json")) if root.exists() else []
    expected_start = 0
    chain = "0" * 64
    primary_records: list[dict[str, Any]] = []
    ccde_records: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for path in paths:
        receipt = load_json(path)
        body = {key: receipt[key] for key in receipt if key not in {"receipt_sha256", "chain_sha256"}}
        receipt_sha = sha256_json(body)
        expected_chain = sha256_json(
            {"previous_chain_sha256": chain, "receipt_sha256": receipt_sha}
        )
        if (
            receipt.get("schema") != PARTIAL_SCHEMA
            or receipt.get("status") != "COMMITTED"
            or receipt.get("rank_plan_sha256") != plan["rank_plan_sha256"]
            or receipt.get("direction") != direction
            or int(receipt.get("bits", -1)) != bits
            or int(receipt.get("start", -1)) != expected_start
            or receipt.get("receipt_sha256") != receipt_sha
            or receipt.get("chain_sha256") != expected_chain
        ):
            raise FormalCCDEError("formal CCDE partial receipt chain changed")
        end = int(receipt.get("end", -1))
        primary = receipt.get("primary_records")
        ccde = receipt.get("ccde_records")
        if end <= expected_start or not isinstance(primary, list) or not isinstance(ccde, list):
            raise FormalCCDEError("formal CCDE partial receipt geometry changed")
        expected_positions = list(range(expected_start, end))
        if [value.get("query_position") for value in primary] != expected_positions:
            raise FormalCCDEError("primary partial query coverage changed")
        if [value.get("query_position") for value in ccde] != expected_positions:
            raise FormalCCDEError("CCDE partial query coverage changed")
        primary_records.extend(primary)
        ccde_records.extend(ccde)
        descriptors.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "receipt_sha256": receipt_sha,
            }
        )
        expected_start = end
        chain = expected_chain
    return expected_start, chain, primary_records, ccde_records, descriptors


def evaluate_formal_state(
    runtime_root: Path,
    plan_root: Path,
    primary_checkpoint_path: Path,
    detail_checkpoint_path: Path,
    detail_bit_artifact_root: Path,
    architecture_freeze_path: Path,
    output_parent: Path,
    *,
    distance_device: str = "cuda",
    max_new_chunks: int | None = None,
    _test_allow_synthetic: bool = False,
) -> Path:
    """Evaluate both primary and CCDE ranks from one frozen code state."""

    if max_new_chunks is not None and max_new_chunks < 0:
        raise ValueError("max_new_chunks must be nonnegative")
    plan_path = Path(plan_root).expanduser().resolve(strict=True)
    plan = _verify_plan(plan_path)
    config = FormalCCDEConfig(**plan["binding"]["config"])
    rank = load_label_free_rank_inputs(
        runtime_root, _test_allow_synthetic=_test_allow_synthetic
    )
    cache = None
    bit_artifact = None
    try:
        metadata = _runtime_manifest(runtime_root)
        identity = _runtime_identity(rank, metadata)
        if identity != plan["runtime_identity"]:
            raise FormalCCDEError("runtime identity differs from the frozen CCDE plan")
        primary, detail, bit_artifact, frozen = _load_models_and_bits(
            rank,
            primary_checkpoint_path,
            detail_checkpoint_path,
            detail_bit_artifact_root,
            architecture_freeze_path,
            device="cpu",
        )
        if primary.checkpoint_sha256 != plan["binding"]["primary_checkpoint_sha256"]:
            raise FormalCCDEError("primary checkpoint differs from the frozen plan")
        if detail.checkpoint_sha256 != plan["binding"]["detail_checkpoint_sha256"]:
            raise FormalCCDEError("detail checkpoint differs from the frozen plan")
        if bit_artifact.manifest["detail_bit_artifact_sha256"] != plan["binding"][
            "detail_bit_artifact_sha256"
        ]:
            raise FormalCCDEError("detail-bit artifact differs from the frozen plan")
        if frozen != plan["binding"]["architecture_freeze"]:
            raise FormalCCDEError("architecture freeze differs from the frozen plan")
        expected_cache_binding = _encoding_binding(
            rank, primary, detail, bit_artifact, frozen
        )
        cache_root = plan_path / plan["encoding_cache"]["path"]
        cache = _open_encoding_cache(cache_root, expected_cache_binding)
        cache_manifest_path = cache_root / "manifest.json"
        if (
            cache_manifest_path.stat().st_size != plan["encoding_cache"]["manifest_size"]
            or sha256_file(cache_manifest_path) != plan["encoding_cache"]["manifest_sha256"]
            or cache.manifest["binding"]["encoding_binding_sha256"]
            != plan["encoding_cache"]["encoding_binding_sha256"]
        ):
            raise FormalCCDEError("encoding cache differs from the frozen plan")
        query_idx = np.asarray(rank.query_idx, dtype=np.int64).copy()
        database_idx = np.asarray(rank.database_idx, dtype=np.int64).copy()
    finally:
        rank.close()
        if bit_artifact is not None:
            bit_artifact.close()
    if cache is None:
        raise AssertionError("verified CCDE cache is missing")
    # Only after every plan/model/code byte verifies do labels cross the boundary.
    labels = load_metric_labels(
        runtime_root,
        rank_contract=plan,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    if labels.source_seal_sha256 != plan["source_seal_sha256"]:
        cache.close()
        raise FormalCCDEError("metric labels and frozen source seal differ")
    if numeric_sha256(labels.query_row_ids) != plan["runtime_identity"][
        "query_row_ids_numeric_sha256"
    ]:
        cache.close()
        raise FormalCCDEError("metric query identities changed")
    if numeric_sha256(labels.database_row_ids) != plan["runtime_identity"][
        "database_row_ids_numeric_sha256"
    ]:
        cache.close()
        raise FormalCCDEError("metric database identities changed")
    output = reject_unsafe_output_path(
        Path(output_parent), field="CCDE storage-bounded metric output"
    )
    output_root = output / f"metrics-{plan['rank_plan_sha256'][:16]}"
    output_root.mkdir(parents=True, exist_ok=True)
    backend = _PackedDistanceBackend(
        cache, query_idx, database_idx, device=distance_device
    )
    prefixes = build_metric_prefixes(len(labels.database), config.cutoffs)
    database_label_counts = np.asarray(
        labels.database.sum(axis=1, dtype=np.uint16), dtype=np.uint16
    )
    headline_cutoff = 50 if 50 in config.cutoffs else config.cutoffs[0]
    produced = 0
    completed_results: list[dict[str, Any]] = []
    try:
        for direction in config.directions:
            for bits in config.bits:
                start, chain, primary_records, ccde_records, partials = _resume_cell(
                    output_root, plan, direction, bits
                )
                for chunk_start in range(start, len(labels.query), config.query_chunk_size):
                    if max_new_chunks is not None and produced >= max_new_chunks:
                        return output_root
                    chunk_end = min(
                        len(labels.query), chunk_start + config.query_chunk_size
                    )
                    primary_distance = backend.distances(
                        "primary", direction, bits, chunk_start, chunk_end
                    )
                    detail_distance = backend.distances(
                        "detail", direction, bits, chunk_start, chunk_end
                    )
                    detail_bits = min(CCDE_DETAIL_CAP, bits)
                    if np.any(primary_distance > bits) or np.any(
                        detail_distance > detail_bits
                    ):
                        raise AssertionError("formal CCDE Hamming distance exceeds its width")
                    composite = (
                        primary_distance.astype(np.uint32)
                        * np.uint32(detail_bits + 1)
                        + detail_distance.astype(np.uint32)
                    )
                    if not np.array_equal(
                        composite // np.uint32(detail_bits + 1),
                        primary_distance.astype(np.uint32),
                    ):
                        raise AssertionError("CCDE composite distance changed a primary shell")
                    primary_chunk_records = []
                    ccde_chunk_records = []
                    for offset, query_position in enumerate(
                        range(chunk_start, chunk_end)
                    ):
                        relevance, gains, ideal = _query_truth(
                            labels,
                            query_position,
                            cutoffs=config.cutoffs,
                            database_label_counts=database_label_counts,
                        )
                        primary_record = expected_tie_metrics_from_distances(
                            relevance,
                            primary_distance[offset],
                            bits=bits,
                            graded_gains=gains,
                            cutoffs=config.cutoffs,
                            prefixes=prefixes,
                            ideal_jaccard_dcg=ideal,
                        )
                        ccde_record = expected_tie_metrics_from_distances(
                            relevance,
                            composite[offset],
                            bits=bits,
                            distance_levels=(bits + 1) * (detail_bits + 1),
                            graded_gains=gains,
                            cutoffs=config.cutoffs,
                            prefixes=prefixes,
                            ideal_jaccard_dcg=ideal,
                        )
                        row_id = bytes(labels.query_row_ids[query_position]).decode(
                            "ascii"
                        )
                        primary_record.update(
                            {"query_position": query_position, "query_row_id": row_id}
                        )
                        ccde_record.update(
                            {"query_position": query_position, "query_row_id": row_id}
                        )
                        primary_chunk_records.append(primary_record)
                        ccde_chunk_records.append(ccde_record)
                    receipt_body = {
                        "schema": PARTIAL_SCHEMA,
                        "status": "COMMITTED",
                        "rank_plan_sha256": plan["rank_plan_sha256"],
                        "direction": direction,
                        "bits": bits,
                        "detail_bits": detail_bits,
                        "start": chunk_start,
                        "end": chunk_end,
                        "primary_distances_numeric_sha256": numeric_sha256(
                            primary_distance
                        ),
                        "detail_distances_numeric_sha256": numeric_sha256(
                            detail_distance
                        ),
                        "composite_distances_numeric_sha256": numeric_sha256(
                            composite
                        ),
                        "primary_shell_invariance_checked": True,
                        "formal_gate_or_fallback_used": False,
                        "primary_records": primary_chunk_records,
                        "ccde_records": ccde_chunk_records,
                        "previous_chain_sha256": chain,
                    }
                    receipt_sha = sha256_json(receipt_body)
                    chain = sha256_json(
                        {
                            "previous_chain_sha256": chain,
                            "receipt_sha256": receipt_sha,
                        }
                    )
                    receipt = {
                        **receipt_body,
                        "receipt_sha256": receipt_sha,
                        "chain_sha256": chain,
                    }
                    target_root = _partial_root(output_root, direction, bits)
                    target_root.mkdir(parents=True, exist_ok=True)
                    target = target_root / f"chunk-{chunk_start:09d}-{chunk_end:09d}.json"
                    atomic_write_json(target, receipt)
                    primary_records.extend(primary_chunk_records)
                    ccde_records.extend(ccde_chunk_records)
                    partials.append(
                        {
                            "path": target.relative_to(output_root).as_posix(),
                            "size": target.stat().st_size,
                            "sha256": sha256_file(target),
                            "receipt_sha256": receipt_sha,
                        }
                    )
                    produced += 1
                if len(primary_records) != len(labels.query) or len(ccde_records) != len(
                    labels.query
                ):
                    raise FormalCCDEError("completed cell does not cover every formal query")
                summaries = {
                    "primary_hamming": mean_query_metrics(primary_records),
                    "ccde_lexicographic": mean_query_metrics(ccde_records),
                }
                delta = {
                    key: float(summaries["ccde_lexicographic"][key])
                    - float(summaries["primary_hamming"][key])
                    for key, value in summaries["primary_hamming"].items()
                    if isinstance(value, float)
                }
                body = {
                    "schema": RESULT_SCHEMA,
                    "status": "COMPLETE",
                    "dataset": plan["dataset"],
                    "source_seal_sha256": plan["source_seal_sha256"],
                    "rank_plan_sha256": plan["rank_plan_sha256"],
                    "direction": direction,
                    "bits": bits,
                    "detail_bits": min(CCDE_DETAIL_CAP, bits),
                    "ranking_rule": plan["ranking_rule"],
                    "primary_shell_order_is_invariant": True,
                    "formal_gate_or_fallback_used": False,
                    "summaries": summaries,
                    "ccde_minus_primary": delta,
                    "per_query_receipts": partials,
                    "final_receipt_chain_sha256": chain,
                    "metric_labels_opened_after_verified_frozen_codes": True,
                }
                result = {**body, "metric_result_sha256": sha256_json(body)}
                target = output_root / "results" / _cell_id(direction, bits) / "metrics.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if load_json(target) != result:
                        raise FormalCCDEError("completed formal CCDE metric was rebound")
                else:
                    atomic_write_json(target, result)
                completed_results.append(
                    {
                        "path": target.relative_to(output_root).as_posix(),
                        "size": target.stat().st_size,
                        "sha256": sha256_file(target),
                        "direction": direction,
                        "bits": bits,
                        "metric_result_sha256": result["metric_result_sha256"],
                        "map_delta": delta["map_expected_ties"],
                        "headline_cutoff": headline_cutoff,
                        "binary_ndcg_headline_delta": delta.get(
                            f"binary_ndcg_at_{headline_cutoff}_expected_ties"
                        ),
                        "j_ndcg_headline_delta": delta.get(
                            f"j_ndcg_at_{headline_cutoff}_expected_ties"
                        ),
                    }
                )
        primary_deltas = [
            value
            for result in completed_results
            for value in (
                result["map_delta"],
                result["binary_ndcg_headline_delta"],
            )
            if value is not None
        ]
        graded_deltas = [
            result["j_ndcg_headline_delta"]
            for result in completed_results
            if result["j_ndcg_headline_delta"] is not None
        ]
        complete_body = {
            "schema": EVALUATION_SCHEMA,
            "status": "COMPLETE",
            "dataset": plan["dataset"],
            "source_seal_sha256": plan["source_seal_sha256"],
            "rank_plan_sha256": plan["rank_plan_sha256"],
            "results": completed_results,
            "formal_gate_or_fallback_used": False,
            "primary_shell_order_is_invariant": True,
            "storage_bounded_complete_gallery_evaluation": True,
            "headline_cutoff": headline_cutoff,
            "primary_cells": len(primary_deltas),
            "graded_cells": len(graded_deltas),
            "negative_primary_cells": sum(value < 0.0 for value in primary_deltas),
            "nonpositive_graded_cells": sum(value <= 0.0 for value in graded_deltas),
            "minimum_primary_delta": min(primary_deltas),
            "mean_primary_delta": float(np.mean(primary_deltas, dtype=np.float64)),
            "minimum_graded_delta": min(graded_deltas),
            "mean_graded_delta": float(np.mean(graded_deltas, dtype=np.float64)),
        }
        atomic_write_json(
            output_root / "evaluation_complete.json",
            {**complete_body, "complete_sha256": sha256_json(complete_body)},
        )
        return output_root
    finally:
        cache.close()


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runtime", type=Path, required=True)
    common.add_argument("--primary-checkpoint", type=Path, required=True)
    common.add_argument("--detail-checkpoint", type=Path, required=True)
    common.add_argument("--detail-bits", type=Path, required=True)
    common.add_argument("--freeze", type=Path, required=True)

    freeze = sub.add_parser("freeze", parents=[common])
    freeze.add_argument("--output-parent", type=Path, required=True)
    freeze.add_argument("--bits", type=_csv_ints, default=BITS)
    freeze.add_argument("--directions", type=_csv_strings, default=DIRECTIONS)
    freeze.add_argument("--cutoffs", type=_csv_ints, default=(50, 100, 1000))
    freeze.add_argument("--query-chunk-size", type=int, default=4)
    freeze.add_argument("--device", default="auto")

    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--output-parent", type=Path, required=True)
    evaluate.add_argument("--distance-device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--max-new-chunks", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        path = freeze_formal_state(
            args.runtime,
            args.primary_checkpoint,
            args.detail_checkpoint,
            args.detail_bits,
            args.freeze,
            args.output_parent,
            config=FormalCCDEConfig(
                bits=tuple(args.bits),
                directions=tuple(args.directions),
                cutoffs=tuple(args.cutoffs),
                query_chunk_size=args.query_chunk_size,
            ),
            device=args.device,
        )
        print(json.dumps({"plan_root": str(path)}, ensure_ascii=False))
        return 0
    path = evaluate_formal_state(
        args.runtime,
        args.plan,
        args.primary_checkpoint,
        args.detail_checkpoint,
        args.detail_bits,
        args.freeze,
        args.output_parent,
        distance_device=args.distance_device,
        max_new_chunks=args.max_new_chunks,
    )
    completion = path / "evaluation_complete.json"
    print(
        json.dumps(
            {
                "evaluation_root": str(path),
                "status": "COMPLETE" if completion.exists() else "IN_PROGRESS",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FormalCCDEConfig",
    "FormalCCDEError",
    "evaluate_formal_state",
    "freeze_formal_state",
]
