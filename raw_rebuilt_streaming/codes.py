"""Label-free, receipt-verified freezing of packed neural hash codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from .integrity import (
    StreamingIntegrityError,
    atomic_write_json,
    load_json,
    numeric_sha256,
    production_code_inventory,
    reject_unsafe_output_path,
    require_disjoint_paths,
    require_dataset_label_geometry,
    require_hashed_json,
    require_no_link_components,
    sha256_file,
    sha256_json,
)


CODE_STATE_SCHEMA = "raw_rebuilt_streaming_packed_code_state_v1"
ENCODING_RECEIPT_SCHEMA = "raw_rebuilt_streaming_encoding_receipt_v1"
MODALITIES = ("image", "text")
SCOPES = ("query", "database")
BITS = (16, 32, 64)


@dataclass(frozen=True)
class CodeFreezeConfig:
    bits: tuple[int, ...] = BITS
    feature_chunk_size: int = 8192

    def __post_init__(self) -> None:
        if tuple(self.bits) != tuple(BITS):
            raise ValueError(f"packed code state must contain exactly {BITS}")
        if type(self.feature_chunk_size) is not int or self.feature_chunk_size < 1:
            raise ValueError("feature_chunk_size must be a positive integer")


@dataclass(frozen=True)
class CodeState:
    root: Path
    arrays: Mapping[tuple[str, str, int], np.ndarray]
    manifest: Mapping[str, Any]

    @property
    def dataset(self) -> str:
        return str(self.manifest["dataset"])

    @property
    def rows(self) -> int:
        return int(self.manifest["rows"])

    @property
    def available_bits(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.manifest["available_bits"])

    def close(self) -> None:
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _array_relative(scope: str, modality: str, bits: int) -> str:
    return f"codes/{scope}_{modality}_bits{bits}_packed.npy"


def safe_encode_feature_chunk(
    model: Any,
    features: np.ndarray,
    *,
    modality: str,
    device: Any,
    batch_size: int,
    encoder: Callable[..., Any] | None = None,
) -> Any:
    """Encode an explicit writable C-order copy of a read-only memmap slice."""

    copied = np.array(features, dtype=np.float32, order="C", copy=True)
    if not copied.flags.c_contiguous or not copied.flags.writeable:
        raise AssertionError("feature safety copy is not writable C-contiguous memory")
    if encoder is None:
        from rz_csd_clip512 import encode_clip512

        encoder = encode_clip512
    return encoder(
        model,
        copied,
        modality=modality,
        device=device,
        batch_size=batch_size,
    )


def pack_bipolar_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    value = np.asarray(codes)
    if value.shape != (len(value), bits) or value.ndim != 2:
        raise StreamingIntegrityError(f"{bits}-bit encoder output geometry changed")
    if not np.all(np.isin(value, (-1, 1))):
        raise StreamingIntegrityError("encoder output is not bipolar {-1,+1}")
    packed = np.packbits(value > 0, axis=1, bitorder="little")
    expected_width = bits // 8
    if packed.shape != (len(value), expected_width) or packed.dtype != np.uint8:
        raise AssertionError("packed binary code geometry is invalid")
    return np.ascontiguousarray(packed)


def _runtime_binding(rank: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(metadata.get("dataset", ""))
    label_dim = int(metadata.get("label_dim", -1))
    require_dataset_label_geometry(dataset, label_dim)
    body = {
        "dataset": dataset,
        "rows": int(len(rank.row_ids)),
        "label_dim": label_dim,
        "source_seal_sha256": rank.source_seal_sha256,
        "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
        "query_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.query_idx]),
        "database_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.database_idx]),
        "indQ_numeric_sha256": numeric_sha256(rank.query_idx),
        "indT_numeric_sha256": numeric_sha256(rank.train_idx),
        "indD_numeric_sha256": numeric_sha256(rank.database_idx),
        "query_rows": int(len(rank.query_idx)),
        "train_rows": int(len(rank.train_idx)),
        "database_rows": int(len(rank.database_idx)),
    }
    return {**body, "runtime_identity_sha256": sha256_json(body)}


def _encoding_binding(
    runtime: Mapping[str, Any],
    checkpoint: Any,
    config: CodeFreezeConfig,
) -> dict[str, Any]:
    checkpoint_binding = checkpoint.metadata["binding"]
    if checkpoint_binding.get("dataset") != runtime["dataset"]:
        raise StreamingIntegrityError("checkpoint and runtime datasets differ")
    if int(checkpoint_binding.get("label_dim", -1)) != int(runtime["label_dim"]):
        raise StreamingIntegrityError("checkpoint and runtime label dimensions differ")
    body = {
        "schema": CODE_STATE_SCHEMA,
        "producer_type": "neural_v1_checkpoint",
        "runtime": dict(runtime),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_run_binding_sha256": checkpoint_binding["run_binding_sha256"],
        "checkpoint_v1_code_inventory_sha256": checkpoint_binding["code_inventory"][
            "code_inventory_sha256"
        ],
        "config": {
            "bits": list(config.bits),
            "feature_chunk_size": config.feature_chunk_size,
        },
        "streaming_code_inventory": production_code_inventory(),
        "labels_loaded_during_encoding": False,
    }
    return {**body, "encoding_binding_sha256": sha256_json(body)}


def _open_arrays_for_resume(
    root: Path,
    scope_rows: Mapping[str, int],
    *,
    available_bits: tuple[int, ...] = BITS,
) -> dict[tuple[str, str, int], np.memmap]:
    arrays: dict[tuple[str, str, int], np.memmap] = {}
    code_root = root / "codes"
    if code_root.exists() and (code_root.is_symlink() or not code_root.is_dir()):
        raise StreamingIntegrityError("partial packed code directory is unsafe")
    paths = [
        root / _array_relative(scope, modality, bits)
        for scope in SCOPES
        for modality in MODALITIES
        for bits in available_bits
    ]
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise StreamingIntegrityError("packed code state has only a subset of its arrays")
    for scope in SCOPES:
        for modality in MODALITIES:
            for bits in available_bits:
                path = root / _array_relative(scope, modality, bits)
                path.parent.mkdir(parents=True, exist_ok=True)
                shape = (int(scope_rows[scope]), bits // 8)
                if path.exists():
                    if path.is_symlink() or not path.is_file():
                        raise StreamingIntegrityError(
                            "partial packed code must be a regular non-symlink file"
                        )
                    value = np.load(path, mmap_mode="r+", allow_pickle=False)
                    if value.shape != shape or value.dtype != np.uint8:
                        raise StreamingIntegrityError("partial packed code array geometry changed")
                else:
                    value = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
                    value.flush()
                arrays[(scope, modality, bits)] = value
    return arrays


def _receipt_paths(root: Path, scope: str, modality: str) -> list[Path]:
    receipt_root = root / "receipts"
    if not receipt_root.exists():
        return []
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise StreamingIntegrityError("encoding receipt directory is unsafe")
    return sorted(receipt_root.glob(f"{scope}-{modality}-chunk-*.json"))


def _resume_position(
    root: Path,
    scope: str,
    modality: str,
    arrays: Mapping[tuple[str, str, int], np.ndarray],
    binding_sha256: str,
    rows: int,
    *,
    available_bits: tuple[int, ...] = BITS,
) -> tuple[int, str]:
    start = 0
    chain = "0" * 64
    for path in _receipt_paths(root, scope, modality):
        if path.is_symlink() or not path.is_file():
            raise StreamingIntegrityError(
                "encoding receipt must be a regular non-symlink file"
            )
        if path.resolve(strict=True).parent != (root / "receipts").resolve(strict=True):
            raise StreamingIntegrityError("encoding receipt escapes its state")
        receipt = load_json(path)
        receipt_body = {
            key: receipt[key]
            for key in receipt
            if key not in {"receipt_sha256", "chain_sha256"}
        }
        if (
            receipt.get("schema") != ENCODING_RECEIPT_SCHEMA
            or receipt.get("status") != "COMMITTED"
            or receipt.get("receipt_sha256") != sha256_json(receipt_body)
        ):
            raise StreamingIntegrityError("encoding receipt schema/hash changed")
        expected_keys = {
            "schema",
            "status",
            "encoding_binding_sha256",
            "scope",
            "modality",
            "start",
            "end",
            "packed_numeric_sha256",
            "previous_chain_sha256",
            "receipt_sha256",
            "chain_sha256",
        }
        if set(receipt) != expected_keys:
            raise StreamingIntegrityError("encoding receipt contains unbound fields")
        if receipt.get("encoding_binding_sha256") != binding_sha256:
            raise StreamingIntegrityError("encoding receipt was rebound")
        if (
            receipt.get("scope") != scope
            or receipt.get("modality") != modality
            or int(receipt.get("start", -1)) != start
        ):
            raise StreamingIntegrityError("encoding receipts are noncontiguous")
        end = int(receipt.get("end", -1))
        if end <= start or end > rows:
            raise StreamingIntegrityError("encoding receipt range is invalid")
        observed = {
            str(bits): numeric_sha256(arrays[(scope, modality, bits)][start:end])
            for bits in available_bits
        }
        if receipt.get("packed_numeric_sha256") != observed:
            raise StreamingIntegrityError("committed packed code chunk changed")
        if receipt.get("previous_chain_sha256") != chain:
            raise StreamingIntegrityError("encoding receipt chain predecessor changed")
        expected_chain = sha256_json(
            {
                "previous_chain_sha256": chain,
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
        if receipt.get("chain_sha256") != expected_chain:
            raise StreamingIntegrityError("encoding receipt chain changed")
        chain = expected_chain
        start = end
    return start, chain


def _write_encoding_receipt(
    root: Path,
    *,
    scope: str,
    modality: str,
    start: int,
    end: int,
    arrays: Mapping[tuple[str, str, int], np.ndarray],
    binding_sha256: str,
    previous_chain: str,
    available_bits: tuple[int, ...] = BITS,
) -> str:
    body = {
        "schema": ENCODING_RECEIPT_SCHEMA,
        "status": "COMMITTED",
        "encoding_binding_sha256": binding_sha256,
        "scope": scope,
        "modality": modality,
        "start": start,
        "end": end,
        "packed_numeric_sha256": {
            str(bits): numeric_sha256(arrays[(scope, modality, bits)][start:end])
            for bits in available_bits
        },
        "previous_chain_sha256": previous_chain,
    }
    receipt_sha = sha256_json(body)
    chain = sha256_json(
        {"previous_chain_sha256": previous_chain, "receipt_sha256": receipt_sha}
    )
    receipt = {**body, "receipt_sha256": receipt_sha, "chain_sha256": chain}
    path = root / "receipts" / f"{scope}-{modality}-chunk-{start:09d}-{end:09d}.json"
    atomic_write_json(path, receipt)
    return chain


def freeze_code_state(
    runtime_root: Path,
    checkpoint_path: Path,
    output_parent: Path,
    *,
    config: CodeFreezeConfig = CodeFreezeConfig(),
    device: Any = "auto",
    max_new_chunks: int | None = None,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> Path:
    """Freeze only packed 16/32/64-bit codes from a receipt-verified v1 model."""

    if max_new_chunks is not None and max_new_chunks < 0:
        raise ValueError("max_new_chunks must be nonnegative")
    require_no_link_components(runtime_root, field="label-free runtime")
    require_no_link_components(checkpoint_path, field="neural checkpoint")
    if process_data_root is not None:
        require_no_link_components(process_data_root, field="ProcessData source")
    from raw_rebuilt_neural.training import load_trained_checkpoint
    from raw_rebuilt_runtime.loader import load_label_free_rank_inputs

    rank = load_label_free_rank_inputs(
        runtime_root,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    try:
        metadata = load_json(Path(runtime_root).expanduser().resolve(strict=True) / "runtime_manifest.json")
        runtime = _runtime_binding(rank, metadata)
        checkpoint = load_trained_checkpoint(
            checkpoint_path,
            device=device,
            expected_source_seal_sha256=rank.source_seal_sha256,
            require_current_code=True,
        )
        binding = _encoding_binding(runtime, checkpoint, config)
        output = reject_unsafe_output_path(Path(output_parent), field="packed code output")
        root = reject_unsafe_output_path(
            output / f"code-state-{binding['encoding_binding_sha256'][:16]}",
            field="packed code state",
        )
        forbidden = {
            "runtime": Path(runtime_root),
            "checkpoint": Path(checkpoint_path),
        }
        if process_data_root is not None:
            forbidden["ProcessData"] = Path(process_data_root)
        require_disjoint_paths(root, forbidden, field="packed code state")
        if (root / "manifest.json").exists():
            state = open_code_state(root)
            try:
                if state.manifest.get("binding") != binding:
                    raise StreamingIntegrityError("completed code state was rebound")
            finally:
                state.close()
            return root
        root.mkdir(parents=True, exist_ok=True)
        scope_indices = {
            "query": np.asarray(rank.query_idx, dtype=np.int64),
            "database": np.asarray(rank.database_idx, dtype=np.int64),
        }
        scope_rows = {scope: int(len(indices)) for scope, indices in scope_indices.items()}
        arrays = _open_arrays_for_resume(
            root, scope_rows, available_bits=tuple(config.bits)
        )
        produced = 0
        try:
            model_device = next(checkpoint.model.parameters()).device
            inference_batch = int(checkpoint.model.config.inference_batch_size)
            for scope in SCOPES:
                canonical_indices = scope_indices[scope]
                for modality, features in (("image", rank.image), ("text", rank.text)):
                    start, chain = _resume_position(
                        root,
                        scope,
                        modality,
                        arrays,
                        binding["encoding_binding_sha256"],
                        scope_rows[scope],
                        available_bits=tuple(config.bits),
                    )
                    while start < scope_rows[scope]:
                        if max_new_chunks is not None and produced >= max_new_chunks:
                            return root
                        end = min(start + config.feature_chunk_size, scope_rows[scope])
                        take = canonical_indices[start:end]
                        encoded = safe_encode_feature_chunk(
                            checkpoint.model,
                            features[take],
                            modality=modality,
                            device=model_device,
                            batch_size=inference_batch,
                        )
                        for bits in config.bits:
                            arrays[(scope, modality, bits)][start:end] = pack_bipolar_codes(
                                encoded.binary_codes[bits], bits
                            )
                            arrays[(scope, modality, bits)].flush()
                        chain = _write_encoding_receipt(
                            root,
                            scope=scope,
                            modality=modality,
                            start=start,
                            end=end,
                            arrays=arrays,
                            binding_sha256=binding["encoding_binding_sha256"],
                            previous_chain=chain,
                            available_bits=tuple(config.bits),
                        )
                        produced += 1
                        start = end
            descriptors: dict[str, Any] = {}
            for scope in SCOPES:
                for modality in MODALITIES:
                    for bits in config.bits:
                        path = root / _array_relative(scope, modality, bits)
                        value = arrays[(scope, modality, bits)]
                        descriptors[f"{scope}_{modality}_{bits}"] = {
                            "path": _array_relative(scope, modality, bits),
                            "dtype": value.dtype.str,
                            "shape": list(value.shape),
                            "size": path.stat().st_size,
                            "file_sha256": sha256_file(path),
                            "numeric_sha256": numeric_sha256(value),
                        }
            semantic_receipts = sorted(
                path
                for scope in SCOPES
                for modality in MODALITIES
                for path in _receipt_paths(root, scope, modality)
            )
            actual_receipts = sorted((root / "receipts").glob("*.json"))
            if actual_receipts != semantic_receipts:
                raise StreamingIntegrityError("encoding receipt directory has unbound JSON files")
            if any(path.is_symlink() or not path.is_file() for path in semantic_receipts):
                raise StreamingIntegrityError("encoding receipt inventory contains a symlink")
            receipt_descriptors = [
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in semantic_receipts
            ]
            manifest_body = {
                "schema": CODE_STATE_SCHEMA,
                "status": "code_state_frozen",
                "dataset": runtime["dataset"],
                "rows": runtime["rows"],
                "label_dim": runtime["label_dim"],
                "source_seal_sha256": runtime["source_seal_sha256"],
                "runtime_identity": runtime,
                "binding": binding,
                "available_bits": list(config.bits),
                "arrays": descriptors,
                "receipts": receipt_descriptors,
                "stored_state": "packed_query_database_binary_codes_only",
                "labels_loaded_during_encoding": False,
            }
            atomic_write_json(
                root / "manifest.json",
                {**manifest_body, "manifest_sha256": sha256_json(manifest_body)},
            )
        finally:
            for value in arrays.values():
                mmap = getattr(value, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        verified = open_code_state(root)
        verified.close()
        return root
    finally:
        rank.close()


def open_code_state(root: Path, *, require_current_code: bool = True) -> CodeState:
    declared = require_no_link_components(root, field="packed code state")
    if declared.is_symlink() or not declared.is_dir():
        raise StreamingIntegrityError("code state must be a regular directory")
    path = declared.resolve(strict=True)
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StreamingIntegrityError("code manifest must be a regular non-symlink file")
    manifest = load_json(manifest_path)
    require_hashed_json(
        manifest,
        hash_field="manifest_sha256",
        schema=CODE_STATE_SCHEMA,
        status="code_state_frozen",
        field="packed code manifest",
    )
    if manifest.get("labels_loaded_during_encoding") is not False:
        raise StreamingIntegrityError("code state crossed the metric-label boundary")
    expected_manifest_keys = {
        "schema",
        "status",
        "dataset",
        "rows",
        "label_dim",
        "source_seal_sha256",
        "runtime_identity",
        "binding",
        "available_bits",
        "arrays",
        "receipts",
        "stored_state",
        "labels_loaded_during_encoding",
        "manifest_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise StreamingIntegrityError("packed code manifest contains unbound fields")
    if manifest.get("stored_state") != "packed_query_database_binary_codes_only":
        raise StreamingIntegrityError("code state contains a forbidden storage class")
    dataset = str(manifest.get("dataset", ""))
    label_dim = int(manifest.get("label_dim", -1))
    require_dataset_label_geometry(dataset, label_dim)
    binding = manifest.get("binding")
    if not isinstance(binding, Mapping):
        raise StreamingIntegrityError("code state has no immutable binding")
    binding_body = {key: binding[key] for key in binding if key != "encoding_binding_sha256"}
    if binding.get("encoding_binding_sha256") != sha256_json(binding_body):
        raise StreamingIntegrityError("encoding binding hash changed")
    producer = binding.get("producer_type")
    common_binding_keys = {
        "schema",
        "producer_type",
        "runtime",
        "config",
        "streaming_code_inventory",
        "labels_loaded_during_encoding",
        "encoding_binding_sha256",
    }
    producer_keys = {
        "neural_v1_checkpoint": {
            "checkpoint_sha256",
            "checkpoint_run_binding_sha256",
            "checkpoint_v1_code_inventory_sha256",
        },
        "baseline_v1_code_artifact": {
            "baseline_method",
            "baseline_bits",
            "baseline_seed",
            "baseline_artifact_manifest_sha256",
            "baseline_rank_contract_sha256",
            "baseline_checkpoint_sha256",
            "baseline_run_contract_sha256",
            "baseline_code_inventory_sha256",
            "baseline_split_binding_sha256",
            "baseline_image_codes_numeric_sha256",
            "baseline_text_codes_numeric_sha256",
        },
    }
    if producer not in producer_keys or set(binding) != common_binding_keys | producer_keys[producer]:
        raise StreamingIntegrityError("code-state producer binding contains unbound fields")
    if binding.get("schema") != CODE_STATE_SCHEMA or binding.get(
        "labels_loaded_during_encoding"
    ) is not False:
        raise StreamingIntegrityError("code-state producer binding schema changed")
    config = binding.get("config")
    expected_config_keys = (
        {"bits", "feature_chunk_size"}
        if producer == "neural_v1_checkpoint"
        else {"bits", "import_chunk_rows"}
    )
    if not isinstance(config, Mapping) or set(config) != expected_config_keys:
        raise StreamingIntegrityError("code-state producer config contains unbound fields")
    for key, value in binding.items():
        if key.endswith("sha256"):
            if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise StreamingIntegrityError(f"code-state producer {key} is invalid")
    if producer == "neural_v1_checkpoint":
        if type(config.get("feature_chunk_size")) is not int or int(
            config["feature_chunk_size"]
        ) < 1:
            raise StreamingIntegrityError("neural encoding chunk size is invalid")
    else:
        if (
            not isinstance(binding.get("baseline_method"), str)
            or type(binding.get("baseline_bits")) is not int
            or type(binding.get("baseline_seed")) is not int
            or type(config.get("import_chunk_rows")) is not int
            or int(config["import_chunk_rows"]) < 1
        ):
            raise StreamingIntegrityError("baseline import producer metadata is invalid")
    frozen_inventory = binding.get("streaming_code_inventory")
    if not isinstance(frozen_inventory, Mapping):
        raise StreamingIntegrityError("streaming code inventory is missing")
    if require_current_code:
        current = production_code_inventory()
        if current != frozen_inventory:
            raise StreamingIntegrityError("current streaming code differs from code-state code")
    available_raw = manifest.get("available_bits")
    if not isinstance(available_raw, list):
        raise StreamingIntegrityError("code state available bit inventory is missing")
    available_bits = tuple(int(value) for value in available_raw)
    if (
        not available_bits
        or tuple(sorted(set(available_bits))) != available_bits
        or any(value not in BITS for value in available_bits)
    ):
        raise StreamingIntegrityError("code state available bit inventory is invalid")
    config_bits = tuple(int(value) for value in config.get("bits", ()))
    if config_bits != available_bits:
        raise StreamingIntegrityError("code state bit inventory differs from its binding")
    runtime = manifest.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        raise StreamingIntegrityError("code state runtime identity is missing")
    runtime_keys = {
        "dataset",
        "rows",
        "label_dim",
        "source_seal_sha256",
        "row_ids_numeric_sha256",
        "query_row_ids_numeric_sha256",
        "database_row_ids_numeric_sha256",
        "indQ_numeric_sha256",
        "indT_numeric_sha256",
        "indD_numeric_sha256",
        "query_rows",
        "train_rows",
        "database_rows",
        "runtime_identity_sha256",
    }
    if set(runtime) != runtime_keys:
        raise StreamingIntegrityError("runtime identity contains unbound fields")
    runtime_body = {
        key: runtime[key] for key in runtime if key != "runtime_identity_sha256"
    }
    if runtime.get("runtime_identity_sha256") != sha256_json(runtime_body):
        raise StreamingIntegrityError("runtime identity hash changed")
    if (
        runtime.get("dataset") != dataset
        or int(runtime.get("rows", -1)) != int(manifest.get("rows", -1))
        or int(runtime.get("label_dim", -1)) != label_dim
        or runtime.get("source_seal_sha256") != manifest.get("source_seal_sha256")
        or binding.get("runtime") != runtime
    ):
        raise StreamingIntegrityError("runtime identity differs from code-state binding")
    for key in (
        "source_seal_sha256",
        "row_ids_numeric_sha256",
        "query_row_ids_numeric_sha256",
        "database_row_ids_numeric_sha256",
        "indQ_numeric_sha256",
        "indT_numeric_sha256",
        "indD_numeric_sha256",
    ):
        value = runtime.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise StreamingIntegrityError(f"runtime identity {key} is invalid")
    rows = int(manifest.get("rows", -1))
    query_rows = int(runtime.get("query_rows", -1))
    train_rows = int(runtime.get("train_rows", -1))
    database_rows = int(runtime.get("database_rows", -1))
    if (
        rows < 1
        or query_rows < 1
        or database_rows < 1
        or train_rows < 1
        or query_rows + database_rows != rows
        or train_rows > database_rows
    ):
        raise StreamingIntegrityError("runtime identity split counts are invalid")
    arrays_meta = manifest.get("arrays")
    expected_names = {
        f"{scope}_{modality}_{bits}"
        for scope in SCOPES
        for modality in MODALITIES
        for bits in available_bits
    }
    if not isinstance(arrays_meta, Mapping) or set(arrays_meta) != expected_names:
        raise StreamingIntegrityError("packed code array inventory differs")
    arrays: dict[tuple[str, str, int], np.ndarray] = {}
    scope_rows = {
        "query": query_rows,
        "database": database_rows,
    }
    for scope in SCOPES:
        for modality in MODALITIES:
            for bits in available_bits:
                name = f"{scope}_{modality}_{bits}"
                descriptor = arrays_meta[name]
                if not isinstance(descriptor, Mapping) or set(descriptor) != {
                    "path",
                    "dtype",
                    "shape",
                    "size",
                    "file_sha256",
                    "numeric_sha256",
                }:
                    raise StreamingIntegrityError(
                        "packed code descriptor contains unbound fields"
                    )
                relative = _array_relative(scope, modality, bits)
                if descriptor.get("path") != relative:
                    raise StreamingIntegrityError("packed code path changed")
                target = path / relative
                if target.is_symlink() or not target.is_file():
                    raise StreamingIntegrityError("packed code must be a regular non-symlink file")
                if target.resolve(strict=True).parent != (path / "codes").resolve(strict=True):
                    raise StreamingIntegrityError("packed code escapes its self-contained directory")
                if target.stat().st_size != descriptor.get("size") or sha256_file(target) != descriptor.get(
                    "file_sha256"
                ):
                    raise StreamingIntegrityError("packed code bytes changed")
                mapped = np.load(target, mmap_mode="r", allow_pickle=False)
                expected_shape = (scope_rows[scope], bits // 8)
                if (
                    mapped.shape != expected_shape
                    or mapped.dtype != np.uint8
                    or descriptor.get("shape") != list(expected_shape)
                    or descriptor.get("dtype") != np.dtype(np.uint8).str
                ):
                    raise StreamingIntegrityError("packed code geometry changed")
                if numeric_sha256(mapped) != descriptor.get("numeric_sha256"):
                    raise StreamingIntegrityError("packed code numeric content changed")
                snapshot = np.array(mapped, dtype=np.uint8, order="C", copy=True)
                mmap = getattr(mapped, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                if numeric_sha256(snapshot) != descriptor.get("numeric_sha256"):
                    raise StreamingIntegrityError("packed code changed during snapshot")
                if (
                    target.stat().st_size != descriptor.get("size")
                    or sha256_file(target) != descriptor.get("file_sha256")
                ):
                    raise StreamingIntegrityError("packed code bytes changed during snapshot")
                arrays[(scope, modality, bits)] = snapshot
    receipts = manifest.get("receipts")
    if not isinstance(receipts, list):
        raise StreamingIntegrityError("code state has no receipt inventory")
    declared = []
    for descriptor in receipts:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "size",
            "sha256",
        }:
            raise StreamingIntegrityError(
                "encoding receipt descriptor contains unbound fields"
            )
        target = path / str(descriptor.get("path", ""))
        if target.is_symlink() or not target.is_file():
            raise StreamingIntegrityError("encoding receipt must be a regular non-symlink file")
        if target.resolve(strict=True).parent != (path / "receipts").resolve(strict=True):
            raise StreamingIntegrityError("encoding receipt escapes its self-contained directory")
        if target.stat().st_size != descriptor.get("size") or sha256_file(target) != descriptor.get("sha256"):
            raise StreamingIntegrityError("encoding receipt bytes changed")
        declared.append(target.relative_to(path).as_posix())
    semantic_receipts = sorted(
        item.relative_to(path).as_posix()
        for scope in SCOPES
        for modality in MODALITIES
        for item in _receipt_paths(path, scope, modality)
    )
    if sorted(declared) != semantic_receipts:
        raise StreamingIntegrityError("receipt inventory is not exactly the four verified chains")
    for scope in SCOPES:
        for modality in MODALITIES:
            committed, _chain = _resume_position(
                path,
                scope,
                modality,
                arrays,
                str(binding["encoding_binding_sha256"]),
                scope_rows[scope],
                available_bits=available_bits,
            )
            if committed != scope_rows[scope]:
                raise StreamingIntegrityError(
                    f"packed {scope}/{modality} receipts do not cover every split row"
                )
    actual = sorted(item.relative_to(path).as_posix() for item in path.rglob("*"))
    expected = sorted(
        [
            "codes",
            "manifest.json",
            "receipts",
            *declared,
            *(
                _array_relative(scope, modality, bits)
                for scope in SCOPES
                for modality in MODALITIES
                for bits in available_bits
            ),
        ]
    )
    if actual != expected:
        raise StreamingIntegrityError("code state has missing or unbound extra files")
    return CodeState(root=path, arrays=arrays, manifest=manifest)


__all__ = [
    "CODE_STATE_SCHEMA",
    "CodeFreezeConfig",
    "CodeState",
    "BITS",
    "freeze_code_state",
    "open_code_state",
    "pack_bipolar_codes",
    "safe_encode_feature_chunk",
]
