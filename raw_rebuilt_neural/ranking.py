"""Label-free encoding and resumable rank freezing for i2t/t2i retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from raw_rebuilt_runtime import LabelFreeRankInputs, load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import atomic_write_json, load_json, numeric_sha256, sha256_file, sha256_json
from rz_csd_clip512 import (
    BITS,
    decode_rz_local,
    encode_clip512,
    hamming_radius,
    reference_z_tables,
    semantic_relation_heads,
)

from .integrity import array_descriptor, atomic_save_npy, production_code_inventory, reject_unsafe_output_path
from .training import LoadedCheckpoint, TrainingError, load_trained_checkpoint


RANK_SCHEMA = "raw_rebuilt_neural_rank_freeze_v1"
CACHE_SCHEMA = "raw_rebuilt_neural_encoding_cache_v1"
CELL_SCHEMA = "raw_rebuilt_neural_rank_cell_v1"
DIRECTIONS = ("i2t", "t2i")
RANK_MODES = ("hamming", "rz_csd_local")


class RankingError(RuntimeError):
    """Raised when a label-free rank artifact cannot be proven complete."""


@dataclass(frozen=True)
class RankFreezeConfig:
    bits: tuple[int, ...] = BITS
    directions: tuple[str, ...] = DIRECTIONS
    modes: tuple[str, ...] = ("hamming",)
    query_chunk_size: int = 4
    semantic_window: int = 128
    max_active_candidates: int = 2048

    def __post_init__(self) -> None:
        if not self.bits or len(set(self.bits)) != len(self.bits) or any(value not in BITS for value in self.bits):
            raise ValueError(f"bits must be a unique nonempty subset of {BITS}")
        if not self.directions or len(set(self.directions)) != len(self.directions) or any(value not in DIRECTIONS for value in self.directions):
            raise ValueError(f"directions must be a unique nonempty subset of {DIRECTIONS}")
        if not self.modes or len(set(self.modes)) != len(self.modes) or any(value not in RANK_MODES for value in self.modes):
            raise ValueError(f"modes must be a unique nonempty subset of {RANK_MODES}")
        if self.query_chunk_size < 1 or self.semantic_window < 1:
            raise ValueError("rank chunk/window sizes must be positive")
        if self.max_active_candidates < self.semantic_window or self.max_active_candidates >= 65535:
            raise ValueError("active cap must be in [semantic_window,65534]")


@dataclass(frozen=True)
class EncodingCache:
    root: Path
    image_codes: Mapping[int, np.ndarray]
    text_codes: Mapping[int, np.ndarray]
    image_posterior: np.ndarray
    text_posterior: np.ndarray
    manifest: Mapping[str, Any]

    def close(self) -> None:
        values = [*self.image_codes.values(), *self.text_codes.values(), self.image_posterior, self.text_posterior]
        for value in values:
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _cache_array_names() -> dict[str, str]:
    names = {f"image_codes_{bits}": f"image_codes_{bits}.npy" for bits in BITS}
    names.update({f"text_codes_{bits}": f"text_codes_{bits}.npy" for bits in BITS})
    names.update({"image_posterior": "image_posterior.npy", "text_posterior": "text_posterior.npy"})
    return names


def _encoding_binding(rank: LabelFreeRankInputs, checkpoint: LoadedCheckpoint) -> dict[str, Any]:
    binding = checkpoint.metadata["binding"]
    body = {
        "schema": CACHE_SCHEMA,
        "status": "COMPLETE",
        "dataset": binding["dataset"],
        "label_dim": int(binding["label_dim"]),
        "source_seal_sha256": rank.source_seal_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_run_binding_sha256": binding["run_binding_sha256"],
        "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
        "train_idx_numeric_sha256": numeric_sha256(rank.train_idx),
        "query_idx_numeric_sha256": numeric_sha256(rank.query_idx),
        "database_idx_numeric_sha256": numeric_sha256(rank.database_idx),
        "labels_loaded_during_freeze": False,
        "code_inventory": production_code_inventory(),
    }
    return {**body, "encoding_binding_sha256": sha256_json(body)}


def _open_encoding_cache(root: Path, expected: Mapping[str, Any]) -> EncodingCache:
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("status") != "COMPLETE":
        raise RankingError("encoding cache manifest is incomplete")
    manifest_body = {
        key: manifest[key] for key in manifest if key != "manifest_sha256"
    }
    if sha256_json(manifest_body) != manifest.get("manifest_sha256"):
        raise RankingError("encoding cache manifest hash changed")
    if manifest.get("binding") != dict(expected):
        raise RankingError("encoding cache is bound to another checkpoint/runtime")
    files = manifest.get("arrays")
    names = _cache_array_names()
    if not isinstance(files, dict) or set(files) != set(names):
        raise RankingError("encoding cache array inventory differs")
    arrays: dict[str, np.ndarray] = {}
    for name, filename in names.items():
        target = root / filename
        descriptor = files[name]
        if descriptor.get("path") != filename or sha256_file(target) != descriptor.get("file_sha256"):
            raise RankingError(f"encoding cache {name} file hash changed")
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != descriptor.get("shape") or value.dtype.str != descriptor.get("dtype"):
            raise RankingError(f"encoding cache {name} geometry changed")
        if numeric_sha256(value) != descriptor.get("numeric_sha256"):
            raise RankingError(f"encoding cache {name} numeric content changed")
        arrays[name] = value
    rows = len(arrays["image_codes_16"])
    for bits in BITS:
        for modality in ("image", "text"):
            code = arrays[f"{modality}_codes_{bits}"]
            if code.shape != (rows, bits) or code.dtype != np.int8 or not np.all(np.isin(code, (-1, 1))):
                raise RankingError("cached binary codes are not bipolar")
    image_posterior = arrays["image_posterior"]
    text_posterior = arrays["text_posterior"]
    if image_posterior.shape != text_posterior.shape or image_posterior.ndim != 3 or image_posterior.shape[1] != rows:
        raise RankingError("cached posterior geometry differs")
    if image_posterior.shape[0] < 3 or image_posterior.shape[2] != int(expected["label_dim"]):
        raise RankingError("cached posterior head/label geometry differs")
    if not np.isfinite(image_posterior).all() or not np.isfinite(text_posterior).all():
        raise RankingError("cached posterior contains non-finite values")
    return EncodingCache(
        root=root,
        image_codes={bits: arrays[f"image_codes_{bits}"] for bits in BITS},
        text_codes={bits: arrays[f"text_codes_{bits}"] for bits in BITS},
        image_posterior=image_posterior,
        text_posterior=text_posterior,
        manifest=manifest,
    )


def ensure_encoding_cache(
    rank: LabelFreeRankInputs,
    checkpoint: LoadedCheckpoint,
    output_root: Path,
) -> EncodingCache:
    """Encode every canonical row without accepting any labels."""

    binding = _encoding_binding(rank, checkpoint)
    root = output_root / "encoding_cache"
    if (root / "manifest.json").exists():
        return _open_encoding_cache(root, binding)
    root.mkdir(parents=True, exist_ok=True)
    device = next(checkpoint.model.parameters()).device
    arrays: dict[str, np.ndarray] = {}
    for modality, features in (("image", rank.image), ("text", rank.text)):
        encoded = encode_clip512(
            checkpoint.model,
            features,
            modality=modality,
            device=device,
            batch_size=checkpoint.model.config.inference_batch_size,
        )
        for bits in BITS:
            name = f"{modality}_codes_{bits}"
            value = np.ascontiguousarray(encoded.binary_codes[bits], dtype=np.int8)
            atomic_save_npy(root / f"{name}.npy", value)
            arrays[name] = value
        posterior_name = f"{modality}_posterior"
        posterior = np.ascontiguousarray(encoded.posterior_heads, dtype=np.float32)
        atomic_save_npy(root / f"{posterior_name}.npy", posterior)
        arrays[posterior_name] = posterior
        del encoded
    descriptors = {
        name: array_descriptor(root / _cache_array_names()[name]) for name in arrays
    }
    manifest_body = {
        "schema": CACHE_SCHEMA,
        "status": "COMPLETE",
        "binding": binding,
        "arrays": descriptors,
    }
    atomic_write_json(root / "manifest.json", {**manifest_body, "manifest_sha256": sha256_json(manifest_body)})
    return _open_encoding_cache(root, binding)


def _direction_arrays(
    cache: EncodingCache,
    rank: LabelFreeRankInputs,
    *,
    direction: str,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if direction == "i2t":
        return (
            cache.image_codes[bits],
            cache.text_codes[bits],
            cache.image_posterior,
            cache.text_posterior,
            "text",
        )
    if direction == "t2i":
        return (
            cache.text_codes[bits],
            cache.image_codes[bits],
            cache.text_posterior,
            cache.image_posterior,
            "image",
        )
    raise ValueError(f"unsupported direction {direction}")


def _groups_from_keys(keys: np.ndarray, *, start: int = 0) -> tuple[np.ndarray, int]:
    value = np.asarray(keys)
    if value.ndim != 1 or value.size == 0:
        raise RankingError("rank group evidence must be a nonempty vector")
    result = np.empty(value.size, dtype=np.uint16)
    group = int(start)
    result[0] = group
    for index in range(1, len(value)):
        if value[index] != value[index - 1]:
            group += 1
        if group >= 65535:
            raise RankingError("rank group count exceeds uint16 capacity")
        result[index] = group
    return result, group


def _rank_one_query(
    *,
    query_code: np.ndarray,
    database_codes: np.ndarray,
    database_canonical_ids: np.ndarray,
    mode: str,
    query_posterior: np.ndarray,
    database_posterior: np.ndarray,
    bank_image_codes: np.ndarray,
    bank_text_codes: np.ndarray,
    target_modality: str,
    semantic_window: int,
    max_active_candidates: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    radius = hamming_radius(query_code, database_codes).astype(np.uint16, copy=False)
    # indD is strictly increasing; stable sorting therefore uses canonical row
    # identity only as a storage order inside exact evidence ties.
    base_order = np.argsort(radius, kind="stable")
    if mode == "hamming":
        groups, _ = _groups_from_keys(radius[base_order])
        return base_order.astype(np.uint32), groups, 0
    if mode != "rz_csd_local":
        raise RankingError(f"unsupported rank mode {mode}")
    window = min(semantic_window, len(base_order))
    end = window
    boundary = radius[base_order[end - 1]]
    while end < len(base_order) and radius[base_order[end]] == boundary:
        end += 1
    if end > max_active_candidates:
        raise RankingError(
            f"tie-closed raw Hamming shell has {end} candidates, cap={max_active_candidates}"
        )
    active_positions = base_order[:end]
    tail_positions = base_order[end:]
    tables = reference_z_tables(query_code, bank_image_codes, bank_text_codes)
    table = tables.text if target_modality == "text" else tables.image
    active_scores = table[radius[active_positions]]
    relevance, jaccard = semantic_relation_heads(
        query_posterior,
        database_posterior[:, database_canonical_ids[active_positions], :],
    )
    decoded = decode_rz_local(
        active_scores,
        database_canonical_ids[active_positions].astype(np.int64, copy=False),
        relevance,
        jaccard,
        window_size=window,
        rz_lower=active_scores,
        rz_upper=active_scores,
        max_active_candidates=max_active_candidates,
        max_active_fraction=1.0,
    )
    refined = active_positions[decoded.order]
    order = np.concatenate((refined, tail_positions)).astype(np.uint32, copy=False)
    active_keys = decoded.rank_group_keys[decoded.order]
    active_groups, final_group = _groups_from_keys(active_keys)
    if len(tail_positions):
        tail_groups, _ = _groups_from_keys(radius[tail_positions], start=final_group + 1)
        groups = np.concatenate((active_groups, tail_groups))
    else:
        groups = active_groups
    if np.any(groups[1:] < groups[:-1]):
        raise AssertionError("rank group IDs are not nondecreasing")
    return order, groups.astype(np.uint16, copy=False), end


def _open_or_create_rank_arrays(cell_root: Path, shape: tuple[int, int]) -> tuple[np.memmap, np.memmap]:
    order_path = cell_root / "orders.npy"
    group_path = cell_root / "groups.npy"
    if order_path.exists() != group_path.exists():
        raise RankingError("rank order/group files are only partially present")
    if not order_path.exists():
        orders = np.lib.format.open_memmap(order_path, mode="w+", dtype=np.uint32, shape=shape)
        groups = np.lib.format.open_memmap(group_path, mode="w+", dtype=np.uint16, shape=shape)
        orders.flush()
        groups.flush()
        return orders, groups
    orders = np.load(order_path, mmap_mode="r+", allow_pickle=False)
    groups = np.load(group_path, mmap_mode="r+", allow_pickle=False)
    if orders.shape != shape or orders.dtype != np.uint32 or groups.shape != shape or groups.dtype != np.uint16:
        raise RankingError("rank order/group array geometry changed")
    return orders, groups


def _verified_resume_position(cell_root: Path, orders: np.ndarray, groups: np.ndarray, binding_sha: str) -> tuple[int, str]:
    receipts = sorted((cell_root / "receipts").glob("chunk-*.json")) if (cell_root / "receipts").exists() else []
    expected_start = 0
    chain = "0" * 64
    for path in receipts:
        receipt = load_json(path)
        if receipt.get("binding_sha256") != binding_sha or int(receipt.get("start", -1)) != expected_start:
            raise RankingError("rank chunk receipts are noncontiguous or rebound")
        end = int(receipt.get("end", -1))
        if end <= expected_start or end > orders.shape[0]:
            raise RankingError("rank chunk receipt range is invalid")
        if numeric_sha256(orders[expected_start:end]) != receipt.get("orders_numeric_sha256"):
            raise RankingError("committed rank order chunk changed")
        if numeric_sha256(groups[expected_start:end]) != receipt.get("groups_numeric_sha256"):
            raise RankingError("committed rank group chunk changed")
        body = {key: receipt[key] for key in receipt if key != "chain_sha256"}
        expected_chain = sha256_json({"previous_chain_sha256": chain, "receipt": body})
        if expected_chain != receipt.get("chain_sha256"):
            raise RankingError("rank receipt chain changed")
        chain = expected_chain
        expected_start = end
    return expected_start, chain


def _freeze_cell(
    root: Path,
    *,
    rank: LabelFreeRankInputs,
    cache: EncodingCache,
    checkpoint: LoadedCheckpoint,
    config: RankFreezeConfig,
    bits: int,
    direction: str,
    mode: str,
) -> Path:
    cell_root = root / "cells" / direction / f"bits-{bits}" / mode
    cell_root.mkdir(parents=True, exist_ok=True)
    query_codes, database_codes, query_posterior, database_posterior, target_modality = _direction_arrays(
        cache, rank, direction=direction, bits=bits
    )
    q_idx = np.asarray(rank.query_idx, dtype=np.int64)
    d_idx = np.asarray(rank.database_idx, dtype=np.int64)
    t_idx = np.asarray(rank.train_idx, dtype=np.int64)
    if np.any(d_idx[1:] <= d_idx[:-1]):
        raise RankingError("database canonical indices must be strictly increasing")
    binding_body = {
        "schema": CELL_SCHEMA,
        "bits": bits,
        "direction": direction,
        "rank_mode": mode,
        "query_rows": len(q_idx),
        "database_rows": len(d_idx),
        "source_seal_sha256": rank.source_seal_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "encoding_binding_sha256": cache.manifest["binding"]["encoding_binding_sha256"],
        "query_idx_numeric_sha256": numeric_sha256(q_idx),
        "database_idx_numeric_sha256": numeric_sha256(d_idx),
        "query_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[q_idx]),
        "database_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[d_idx]),
        "semantic_window": config.semantic_window,
        "max_active_candidates": config.max_active_candidates,
        "labels_loaded_during_freeze": False,
        "tie_semantics": "exact-evidence-groups; canonical-row-order-storage-only",
    }
    binding = {**binding_body, "binding_sha256": sha256_json(binding_body)}
    binding_path = cell_root / "binding.json"
    if binding_path.exists():
        if load_json(binding_path) != binding:
            raise RankingError("rank cell directory is bound to another configuration")
    else:
        atomic_write_json(binding_path, binding)
    orders, groups = _open_or_create_rank_arrays(cell_root, (len(q_idx), len(d_idx)))
    start, chain = _verified_resume_position(cell_root, orders, groups, binding["binding_sha256"])
    bank_image = cache.image_codes[bits][t_idx]
    bank_text = cache.text_codes[bits][t_idx]
    # Materialize the target code bank once per cell.  Repeating advanced
    # indexing for every query would copy the full database thousands of times.
    selected_database_codes = np.ascontiguousarray(
        database_codes[d_idx], dtype=np.int8
    )
    receipts_root = cell_root / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    for chunk_start in range(start, len(q_idx), config.query_chunk_size):
        chunk_end = min(len(q_idx), chunk_start + config.query_chunk_size)
        active_sizes = []
        for q_position in range(chunk_start, chunk_end):
            canonical_q = int(q_idx[q_position])
            order, group, active_size = _rank_one_query(
                query_code=query_codes[canonical_q],
                database_codes=selected_database_codes,
                database_canonical_ids=d_idx,
                mode=mode,
                query_posterior=query_posterior[:, canonical_q : canonical_q + 1, :],
                database_posterior=database_posterior,
                bank_image_codes=bank_image,
                bank_text_codes=bank_text,
                target_modality=target_modality,
                semantic_window=config.semantic_window,
                max_active_candidates=config.max_active_candidates,
            )
            if np.unique(order).size != len(d_idx) or int(order.max(initial=0)) >= len(d_idx):
                raise AssertionError("rank worker did not return a database permutation")
            orders[q_position] = order
            groups[q_position] = group
            active_sizes.append(active_size)
        orders.flush()
        groups.flush()
        receipt_body = {
            "schema": CELL_SCHEMA,
            "binding_sha256": binding["binding_sha256"],
            "start": chunk_start,
            "end": chunk_end,
            "orders_numeric_sha256": numeric_sha256(orders[chunk_start:chunk_end]),
            "groups_numeric_sha256": numeric_sha256(groups[chunk_start:chunk_end]),
            "maximum_active_size": int(max(active_sizes, default=0)),
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
        raise RankingError("rank cell receipts do not cover every query")
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
    }
    atomic_write_json(
        cell_root / "rank_contract.json",
        {**contract_body, "rank_contract_sha256": sha256_json(contract_body)},
    )
    return cell_root / "rank_contract.json"


def freeze_ranks(
    runtime_root: Path,
    checkpoint_path: Path,
    output_parent: Path,
    *,
    config: RankFreezeConfig = RankFreezeConfig(),
    device: str | torch.device = "auto",
    _test_allow_synthetic: bool = False,
) -> Path:
    """Freeze complete label-free ranks; this function never loads labels."""

    rank = load_label_free_rank_inputs(
        runtime_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    checkpoint = load_trained_checkpoint(
        checkpoint_path,
        device=device,
        expected_source_seal_sha256=rank.source_seal_sha256,
    )
    rank_config = asdict(config)
    config_sha = sha256_json(rank_config)
    output = reject_unsafe_output_path(Path(output_parent), field="rank output")
    root = output / f"rank-{checkpoint.checkpoint_sha256[:12]}-{config_sha[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    cache = ensure_encoding_cache(rank, checkpoint, root)
    contracts = []
    for bits in config.bits:
        for direction in config.directions:
            for mode in config.modes:
                contract_path = _freeze_cell(
                    root,
                    rank=rank,
                    cache=cache,
                    checkpoint=checkpoint,
                    config=config,
                    bits=bits,
                    direction=direction,
                    mode=mode,
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
        "dataset": checkpoint.metadata["binding"]["dataset"],
        "source_seal_sha256": rank.source_seal_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_run_binding_sha256": checkpoint.metadata["binding"]["run_binding_sha256"],
        "rank_config": rank_config,
        "rank_config_sha256": config_sha,
        "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
        "query_idx_numeric_sha256": numeric_sha256(rank.query_idx),
        "database_idx_numeric_sha256": numeric_sha256(rank.database_idx),
        "query_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.query_idx]),
        "database_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.database_idx]),
        "encoding_manifest_sha256": sha256_file(cache.root / "manifest.json"),
        "cells": sorted(contracts, key=lambda value: value["path"]),
        "code_inventory": production_code_inventory(),
    }
    manifest = {**manifest_body, "rank_manifest_sha256": sha256_json(manifest_body)}
    atomic_write_json(root / "rank_manifest.json", manifest)
    cache.close()
    rank.close()
    return root


def verify_rank_cell(cell_root: Path, contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Replay chunk receipts before metric code consumes a rank cell."""

    root = Path(cell_root).resolve(strict=True)
    body = {key: contract[key] for key in contract if key != "rank_contract_sha256"}
    if sha256_json(body) != contract.get("rank_contract_sha256"):
        raise RankingError("rank cell contract hash changed")
    if contract.get("status") != "rank_state_frozen" or contract.get("labels_loaded_during_freeze") is not False:
        raise RankingError("rank cell was not frozen label-free")
    orders = np.load(root / contract["orders"]["path"], mmap_mode="r", allow_pickle=False)
    groups = np.load(root / contract["groups"]["path"], mmap_mode="r", allow_pickle=False)
    if list(orders.shape) != contract["orders"]["shape"] or orders.dtype.str != contract["orders"]["dtype"]:
        raise RankingError("rank order geometry changed")
    if list(groups.shape) != contract["groups"]["shape"] or groups.dtype.str != contract["groups"]["dtype"]:
        raise RankingError("rank group geometry changed")
    position, chain = _verified_resume_position(
        root, orders, groups, contract["binding"]["binding_sha256"]
    )
    if position != orders.shape[0] or chain != contract["final_receipt_chain_sha256"]:
        raise RankingError("rank receipt coverage differs from frozen contract")
    return orders, groups


__all__ = [
    "DIRECTIONS",
    "RANK_MODES",
    "RankFreezeConfig",
    "RankingError",
    "freeze_ranks",
    "verify_rank_cell",
]
