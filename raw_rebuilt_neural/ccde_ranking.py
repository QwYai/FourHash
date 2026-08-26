"""Label-free, shell-preserving rank freeze for the frozen CCDE model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from raw_rebuilt_runtime import LabelFreeRankInputs, load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import atomic_write_json, load_json, numeric_sha256, sha256_file, sha256_json
from rz_csd_clip512 import BITS, encode_clip512

from .ccde_contract import CCDE_DETAIL_CAP, freeze_binding
from .ccde_detail_bits import DetailBitArtifact, open_detail_bit_artifact
from .ccde_training import load_detail_checkpoint
from .integrity import array_descriptor, atomic_save_npy, production_code_inventory, reject_unsafe_output_path
from .ranking import (
    CELL_SCHEMA,
    DIRECTIONS,
    RANK_SCHEMA,
    RankingError,
    _groups_from_keys,
    _open_or_create_rank_arrays,
    _verified_resume_position,
)
from .training import LoadedCheckpoint, load_trained_checkpoint


CCDE_CACHE_SCHEMA = "raw_rebuilt_ccde_encoding_cache_v1"
CCDE_RANK_MODE = "ccde_lexicographic"


@dataclass(frozen=True)
class CCDERankFreezeConfig:
    bits: tuple[int, ...] = BITS
    directions: tuple[str, ...] = DIRECTIONS
    query_chunk_size: int = 4

    def __post_init__(self) -> None:
        if (
            not self.bits
            or len(set(self.bits)) != len(self.bits)
            or any(value not in BITS for value in self.bits)
        ):
            raise ValueError(f"bits must be a unique nonempty subset of {BITS}")
        if (
            not self.directions
            or len(set(self.directions)) != len(self.directions)
            or any(value not in DIRECTIONS for value in self.directions)
        ):
            raise ValueError(f"directions must be a unique nonempty subset of {DIRECTIONS}")
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")


@dataclass(frozen=True)
class CCDEEncodingCache:
    root: Path
    primary_image_codes: Mapping[int, np.ndarray]
    primary_text_codes: Mapping[int, np.ndarray]
    detail_image_codes: Mapping[int, np.ndarray]
    detail_text_codes: Mapping[int, np.ndarray]
    manifest: Mapping[str, Any]

    def close(self) -> None:
        for value in (
            *self.primary_image_codes.values(),
            *self.primary_text_codes.values(),
            *self.detail_image_codes.values(),
            *self.detail_text_codes.values(),
        ):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def lexicographic_distance(
    primary_query: np.ndarray,
    primary_database: np.ndarray,
    detail_query: np.ndarray,
    detail_database: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return composite, primary, and detail Hamming distances."""

    primary_q = np.asarray(primary_query)
    primary_db = np.asarray(primary_database)
    detail_q = np.asarray(detail_query)
    detail_db = np.asarray(detail_database)
    if primary_q.ndim != 1 or primary_db.ndim != 2 or primary_db.shape[1] != len(primary_q):
        raise ValueError("primary query/database codes are not aligned")
    if detail_q.ndim != 1 or detail_db.ndim != 2 or detail_db.shape[1] != len(detail_q):
        raise ValueError("detail query/database codes are not aligned")
    if len(primary_db) != len(detail_db) or len(detail_q) < 1:
        raise ValueError("primary/detail database rows differ or detail code is empty")
    for name, value in (
        ("primary_query", primary_q),
        ("primary_database", primary_db),
        ("detail_query", detail_q),
        ("detail_database", detail_db),
    ):
        if not np.all(np.isin(value, (-1, 1))):
            raise ValueError(f"{name} must contain bipolar codes")
    primary = np.count_nonzero(primary_db != primary_q[None, :], axis=1).astype(
        np.uint16, copy=False
    )
    detail = np.count_nonzero(detail_db != detail_q[None, :], axis=1).astype(
        np.uint16, copy=False
    )
    multiplier = len(detail_q) + 1
    composite = primary.astype(np.uint32) * np.uint32(multiplier) + detail.astype(
        np.uint32
    )
    return composite, primary, detail


def _cache_array_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for role in ("primary", "detail"):
        for modality in ("image", "text"):
            for bits in BITS:
                name = f"{role}_{modality}_codes_{bits}"
                names[name] = f"{name}.npy"
    return names


def _validate_model_pair(
    primary: LoadedCheckpoint,
    detail: LoadedCheckpoint,
    bit_artifact: DetailBitArtifact,
) -> None:
    primary_binding = primary.metadata["binding"]
    detail_binding = detail.metadata["binding"]
    for key in ("dataset", "label_dim", "source_seal_sha256", "fit_artifact_sha256"):
        if primary_binding.get(key) != detail_binding.get(key):
            raise RankingError(f"primary/detail checkpoint {key} differs")
    if primary_binding.get("train_config") != detail_binding.get("train_config"):
        raise RankingError("primary/detail training configurations differ")
    if primary_binding.get("model_config") != detail_binding.get("model_config"):
        raise RankingError("primary/detail model configurations differ")
    if bit_artifact.dataset != detail_binding.get("dataset"):
        raise RankingError("detail-bit artifact dataset differs from the model pair")
    if bit_artifact.source_seal_sha256 != detail_binding.get("source_seal_sha256"):
        raise RankingError("detail-bit artifact source differs from the model pair")
    if bit_artifact.detail_checkpoint_sha256 != detail.checkpoint_sha256:
        raise RankingError("detail-bit artifact belongs to another detail checkpoint")


def _encoding_binding(
    rank: LabelFreeRankInputs,
    primary: LoadedCheckpoint,
    detail: LoadedCheckpoint,
    bit_artifact: DetailBitArtifact,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": CCDE_CACHE_SCHEMA,
        "status": "COMPLETE",
        "dataset": primary.metadata["binding"]["dataset"],
        "label_dim": int(primary.metadata["binding"]["label_dim"]),
        "source_seal_sha256": rank.source_seal_sha256,
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
        "architecture_freeze": dict(frozen),
        "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
        "train_idx_numeric_sha256": numeric_sha256(rank.train_idx),
        "query_idx_numeric_sha256": numeric_sha256(rank.query_idx),
        "database_idx_numeric_sha256": numeric_sha256(rank.database_idx),
        "labels_loaded_during_freeze": False,
        "detail_coordinates_selected_before_rank_freeze": True,
        "code_inventory": production_code_inventory(),
    }
    return {**body, "encoding_binding_sha256": sha256_json(body)}


def _open_encoding_cache(
    root: Path,
    expected: Mapping[str, Any],
) -> CCDEEncodingCache:
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != CCDE_CACHE_SCHEMA or manifest.get("status") != "COMPLETE":
        raise RankingError("CCDE encoding cache manifest is incomplete")
    body = {key: manifest[key] for key in manifest if key != "manifest_sha256"}
    if sha256_json(body) != manifest.get("manifest_sha256"):
        raise RankingError("CCDE encoding cache manifest hash changed")
    if manifest.get("binding") != dict(expected):
        raise RankingError("CCDE encoding cache is bound to another model/runtime")
    names = _cache_array_names()
    files = manifest.get("arrays")
    if not isinstance(files, dict) or set(files) != set(names):
        raise RankingError("CCDE encoding cache array inventory differs")
    arrays: dict[str, np.ndarray] = {}
    for name, filename in names.items():
        target = root / filename
        descriptor = files[name]
        if descriptor.get("path") != filename or sha256_file(target) != descriptor.get(
            "file_sha256"
        ):
            raise RankingError(f"CCDE encoding cache {name} file hash changed")
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != descriptor.get("shape") or value.dtype.str != descriptor.get(
            "dtype"
        ):
            raise RankingError(f"CCDE encoding cache {name} geometry changed")
        if numeric_sha256(value) != descriptor.get("numeric_sha256"):
            raise RankingError(f"CCDE encoding cache {name} numeric content changed")
        arrays[name] = value
    rows = len(arrays["primary_image_codes_16"])
    for bits in BITS:
        detail_count = min(CCDE_DETAIL_CAP, bits)
        for modality in ("image", "text"):
            primary_code = arrays[f"primary_{modality}_codes_{bits}"]
            detail_code = arrays[f"detail_{modality}_codes_{bits}"]
            if primary_code.shape != (rows, bits) or primary_code.dtype != np.int8:
                raise RankingError("cached primary binary-code geometry differs")
            if detail_code.shape != (rows, detail_count) or detail_code.dtype != np.int8:
                raise RankingError("cached detail binary-code geometry differs")
            if not np.all(np.isin(primary_code, (-1, 1))) or not np.all(
                np.isin(detail_code, (-1, 1))
            ):
                raise RankingError("cached CCDE codes are not bipolar")
    return CCDEEncodingCache(
        root=root,
        primary_image_codes={bits: arrays[f"primary_image_codes_{bits}"] for bits in BITS},
        primary_text_codes={bits: arrays[f"primary_text_codes_{bits}"] for bits in BITS},
        detail_image_codes={bits: arrays[f"detail_image_codes_{bits}"] for bits in BITS},
        detail_text_codes={bits: arrays[f"detail_text_codes_{bits}"] for bits in BITS},
        manifest=manifest,
    )


def ensure_ccde_encoding_cache(
    rank: LabelFreeRankInputs,
    primary: LoadedCheckpoint,
    detail: LoadedCheckpoint,
    bit_artifact: DetailBitArtifact,
    frozen: Mapping[str, Any],
    output_root: Path,
) -> CCDEEncodingCache:
    """Encode canonical rows with both frozen models without loading labels."""

    _validate_model_pair(primary, detail, bit_artifact)
    binding = _encoding_binding(rank, primary, detail, bit_artifact, frozen)
    root = output_root / "encoding_cache"
    if (root / "manifest.json").exists():
        return _open_encoding_cache(root, binding)
    root.mkdir(parents=True, exist_ok=True)
    primary_device = next(primary.model.parameters()).device
    detail_device = next(detail.model.parameters()).device
    arrays: dict[str, np.ndarray] = {}
    for modality, features in (("image", rank.image), ("text", rank.text)):
        primary_encoded = encode_clip512(
            primary.model,
            features,
            modality=modality,
            device=primary_device,
            batch_size=primary.model.config.inference_batch_size,
        )
        detail_encoded = encode_clip512(
            detail.model,
            features,
            modality=modality,
            device=detail_device,
            batch_size=detail.model.config.inference_batch_size,
        )
        for bits in BITS:
            primary_name = f"primary_{modality}_codes_{bits}"
            detail_name = f"detail_{modality}_codes_{bits}"
            primary_value = np.ascontiguousarray(
                primary_encoded.binary_codes[bits], dtype=np.int8
            )
            selected = np.asarray(bit_artifact.selected[bits], dtype=np.int64)
            detail_value = np.ascontiguousarray(
                detail_encoded.binary_codes[bits][:, selected], dtype=np.int8
            )
            atomic_save_npy(root / _cache_array_names()[primary_name], primary_value)
            atomic_save_npy(root / _cache_array_names()[detail_name], detail_value)
            arrays[primary_name] = primary_value
            arrays[detail_name] = detail_value
        del primary_encoded, detail_encoded
    descriptors = {
        name: array_descriptor(root / _cache_array_names()[name]) for name in arrays
    }
    manifest_body = {
        "schema": CCDE_CACHE_SCHEMA,
        "status": "COMPLETE",
        "binding": binding,
        "arrays": descriptors,
    }
    atomic_write_json(
        root / "manifest.json",
        {**manifest_body, "manifest_sha256": sha256_json(manifest_body)},
    )
    return _open_encoding_cache(root, binding)


def _direction_arrays(
    cache: CCDEEncodingCache,
    *,
    direction: str,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if direction == "i2t":
        return (
            cache.primary_image_codes[bits],
            cache.primary_text_codes[bits],
            cache.detail_image_codes[bits],
            cache.detail_text_codes[bits],
        )
    if direction == "t2i":
        return (
            cache.primary_text_codes[bits],
            cache.primary_image_codes[bits],
            cache.detail_text_codes[bits],
            cache.detail_image_codes[bits],
        )
    raise ValueError(f"unsupported direction {direction}")


def _freeze_cell(
    root: Path,
    *,
    rank: LabelFreeRankInputs,
    cache: CCDEEncodingCache,
    primary: LoadedCheckpoint,
    detail: LoadedCheckpoint,
    bit_artifact: DetailBitArtifact,
    frozen: Mapping[str, Any],
    config: CCDERankFreezeConfig,
    bits: int,
    direction: str,
) -> Path:
    cell_root = root / "cells" / direction / f"bits-{bits}" / CCDE_RANK_MODE
    cell_root.mkdir(parents=True, exist_ok=True)
    primary_query, primary_database, detail_query, detail_database = _direction_arrays(
        cache, direction=direction, bits=bits
    )
    q_idx = np.asarray(rank.query_idx, dtype=np.int64)
    d_idx = np.asarray(rank.database_idx, dtype=np.int64)
    if np.any(d_idx[1:] <= d_idx[:-1]):
        raise RankingError("database canonical indices must be strictly increasing")
    detail_count = min(CCDE_DETAIL_CAP, bits)
    binding_body = {
        "schema": CELL_SCHEMA,
        "bits": bits,
        "detail_bits": detail_count,
        "direction": direction,
        "rank_mode": CCDE_RANK_MODE,
        "query_rows": len(q_idx),
        "database_rows": len(d_idx),
        "source_seal_sha256": rank.source_seal_sha256,
        "checkpoint_sha256": primary.checkpoint_sha256,
        "primary_checkpoint_sha256": primary.checkpoint_sha256,
        "detail_checkpoint_sha256": detail.checkpoint_sha256,
        "detail_bit_artifact_sha256": bit_artifact.manifest[
            "detail_bit_artifact_sha256"
        ],
        "architecture_freeze_sha256": frozen["freeze_sha256"],
        "encoding_binding_sha256": cache.manifest["binding"][
            "encoding_binding_sha256"
        ],
        "query_idx_numeric_sha256": numeric_sha256(q_idx),
        "database_idx_numeric_sha256": numeric_sha256(d_idx),
        "query_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[q_idx]),
        "database_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[d_idx]),
        "labels_loaded_during_freeze": False,
        "ranking_rule": "primary_distance*(detail_bits+1)+detail_distance",
        "primary_shell_order_is_invariant": True,
        "formal_gate_or_fallback_used": False,
        "tie_semantics": "exact-composite-evidence-groups; canonical-row-order-storage-only",
    }
    binding = {**binding_body, "binding_sha256": sha256_json(binding_body)}
    binding_path = cell_root / "binding.json"
    if binding_path.exists():
        if load_json(binding_path) != binding:
            raise RankingError("CCDE rank cell is bound to another configuration")
    else:
        atomic_write_json(binding_path, binding)
    orders, groups = _open_or_create_rank_arrays(cell_root, (len(q_idx), len(d_idx)))
    start, chain = _verified_resume_position(
        cell_root, orders, groups, binding["binding_sha256"]
    )
    selected_primary_database = np.ascontiguousarray(primary_database[d_idx], dtype=np.int8)
    selected_detail_database = np.ascontiguousarray(detail_database[d_idx], dtype=np.int8)
    receipts_root = cell_root / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    for chunk_start in range(start, len(q_idx), config.query_chunk_size):
        chunk_end = min(len(q_idx), chunk_start + config.query_chunk_size)
        primary_group_counts: list[int] = []
        composite_group_counts: list[int] = []
        split_shell_counts: list[int] = []
        for q_position in range(chunk_start, chunk_end):
            canonical_q = int(q_idx[q_position])
            composite, primary_radius, _ = lexicographic_distance(
                primary_query[canonical_q],
                selected_primary_database,
                detail_query[canonical_q],
                selected_detail_database,
            )
            order = np.argsort(composite, kind="stable")
            ordered_primary = primary_radius[order]
            if np.any(ordered_primary[1:] < ordered_primary[:-1]):
                raise AssertionError("CCDE moved an item across a primary Hamming shell")
            ordered_composite = composite[order]
            group, _ = _groups_from_keys(ordered_composite)
            if np.unique(order).size != len(d_idx) or int(order.max(initial=0)) >= len(d_idx):
                raise AssertionError("CCDE rank worker did not return a database permutation")
            orders[q_position] = order.astype(np.uint32, copy=False)
            groups[q_position] = group
            primary_unique = np.unique(primary_radius).size
            composite_unique = np.unique(composite).size
            primary_group_counts.append(int(primary_unique))
            composite_group_counts.append(int(composite_unique))
            split_shell_counts.append(int(composite_unique - primary_unique))
        orders.flush()
        groups.flush()
        receipt_body = {
            "schema": CELL_SCHEMA,
            "binding_sha256": binding["binding_sha256"],
            "start": chunk_start,
            "end": chunk_end,
            "orders_numeric_sha256": numeric_sha256(orders[chunk_start:chunk_end]),
            "groups_numeric_sha256": numeric_sha256(groups[chunk_start:chunk_end]),
            "mean_primary_shell_count": float(np.mean(primary_group_counts)),
            "mean_composite_group_count": float(np.mean(composite_group_counts)),
            "mean_additional_groups_from_detail": float(np.mean(split_shell_counts)),
            "primary_shell_invariance_checked": True,
        }
        chain = sha256_json({"previous_chain_sha256": chain, "receipt": receipt_body})
        atomic_write_json(
            receipts_root / f"chunk-{chunk_start:06d}-{chunk_end:06d}.json",
            {**receipt_body, "chain_sha256": chain},
        )
    complete_start, complete_chain = _verified_resume_position(
        cell_root, orders, groups, binding["binding_sha256"]
    )
    if complete_start != len(q_idx):
        raise RankingError("CCDE rank cell receipts do not cover every query")
    contract_body = {
        "schema": CELL_SCHEMA,
        "status": "rank_state_frozen",
        "labels_loaded_during_freeze": False,
        "source_seal_sha256": rank.source_seal_sha256,
        "binding": binding,
        "orders": {
            "path": "orders.npy",
            "dtype": orders.dtype.str,
            "shape": list(orders.shape),
            "size": (cell_root / "orders.npy").stat().st_size,
        },
        "groups": {
            "path": "groups.npy",
            "dtype": groups.dtype.str,
            "shape": list(groups.shape),
            "size": (cell_root / "groups.npy").stat().st_size,
        },
        "receipt_count": len(list(receipts_root.glob("chunk-*.json"))),
        "final_receipt_chain_sha256": complete_chain,
        "primary_shell_invariance_checked_for_every_query": True,
    }
    atomic_write_json(
        cell_root / "rank_contract.json",
        {**contract_body, "rank_contract_sha256": sha256_json(contract_body)},
    )
    return cell_root / "rank_contract.json"


def freeze_ccde_ranks(
    runtime_root: Path,
    primary_checkpoint_path: Path,
    detail_checkpoint_path: Path,
    detail_bit_artifact_root: Path,
    architecture_freeze_path: Path,
    output_parent: Path,
    *,
    config: CCDERankFreezeConfig = CCDERankFreezeConfig(),
    device: str | torch.device = "auto",
    _test_allow_synthetic: bool = False,
) -> Path:
    """Freeze complete CCDE ranks; no label-bearing loader is reachable here."""

    frozen = freeze_binding(architecture_freeze_path)
    rank = load_label_free_rank_inputs(
        runtime_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    try:
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
        try:
            _validate_model_pair(primary, detail, bit_artifact)
            rank_config = asdict(config)
            config_sha = sha256_json(rank_config)
            root_name = sha256_json(
                {
                    "primary": primary.checkpoint_sha256,
                    "detail": detail.checkpoint_sha256,
                    "detail_bits": bit_artifact.manifest["detail_bit_artifact_sha256"],
                    "freeze": frozen["freeze_sha256"],
                    "config": config_sha,
                }
            )
            output = reject_unsafe_output_path(Path(output_parent), field="CCDE rank output")
            root = output / f"ccde-rank-{root_name[:16]}"
            root.mkdir(parents=True, exist_ok=True)
            cache = ensure_ccde_encoding_cache(
                rank, primary, detail, bit_artifact, frozen, root
            )
            try:
                contracts = []
                for bits in config.bits:
                    for direction in config.directions:
                        contract_path = _freeze_cell(
                            root,
                            rank=rank,
                            cache=cache,
                            primary=primary,
                            detail=detail,
                            bit_artifact=bit_artifact,
                            frozen=frozen,
                            config=config,
                            bits=bits,
                            direction=direction,
                        )
                        contracts.append(
                            {
                                "path": contract_path.relative_to(root).as_posix(),
                                "size": contract_path.stat().st_size,
                                "sha256": sha256_file(contract_path),
                            }
                        )
                manifest_body = {
                    "schema": RANK_SCHEMA,
                    "status": "rank_state_frozen",
                    "labels_loaded_during_freeze": False,
                    "dataset": primary.metadata["binding"]["dataset"],
                    "source_seal_sha256": rank.source_seal_sha256,
                    "checkpoint_sha256": primary.checkpoint_sha256,
                    "checkpoint_run_binding_sha256": primary.metadata["binding"][
                        "run_binding_sha256"
                    ],
                    "primary_checkpoint_sha256": primary.checkpoint_sha256,
                    "detail_checkpoint_sha256": detail.checkpoint_sha256,
                    "detail_checkpoint_run_binding_sha256": detail.metadata["binding"][
                        "run_binding_sha256"
                    ],
                    "detail_bit_artifact_sha256": bit_artifact.manifest[
                        "detail_bit_artifact_sha256"
                    ],
                    "architecture_freeze": frozen,
                    "rank_config": rank_config,
                    "rank_config_sha256": config_sha,
                    "rank_mode": CCDE_RANK_MODE,
                    "primary_shell_order_is_invariant": True,
                    "formal_gate_or_fallback_used": False,
                    "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
                    "query_idx_numeric_sha256": numeric_sha256(rank.query_idx),
                    "database_idx_numeric_sha256": numeric_sha256(rank.database_idx),
                    "query_row_ids_numeric_sha256": numeric_sha256(
                        rank.row_ids[rank.query_idx]
                    ),
                    "database_row_ids_numeric_sha256": numeric_sha256(
                        rank.row_ids[rank.database_idx]
                    ),
                    "encoding_manifest_sha256": sha256_file(
                        cache.root / "manifest.json"
                    ),
                    "cells": sorted(contracts, key=lambda value: value["path"]),
                    "code_inventory": production_code_inventory(),
                }
                atomic_write_json(
                    root / "rank_manifest.json",
                    {
                        **manifest_body,
                        "rank_manifest_sha256": sha256_json(manifest_body),
                    },
                )
                return root
            finally:
                cache.close()
        finally:
            bit_artifact.close()
    finally:
        rank.close()


__all__ = [
    "CCDE_CACHE_SCHEMA",
    "CCDE_RANK_MODE",
    "CCDEEncodingCache",
    "CCDERankFreezeConfig",
    "ensure_ccde_encoding_cache",
    "freeze_ccde_ranks",
    "lexicographic_distance",
]
