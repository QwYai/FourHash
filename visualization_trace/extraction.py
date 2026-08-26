from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .adapters import DatasetAdapter
from .core import (
    TraceContractError,
    TraceRow,
    atomic_write_bytes,
    atomic_write_json,
    ensure_output_safe,
    file_inventory,
    sha256_bytes,
    sha256_file,
    sha256_json,
    stable_json_bytes,
)


FEATURE_SEMANTICS: Dict[str, Any] = {
    "schema": "kbs-clip-feature-semantics-v1",
    "image_embedding": {
        "source": "raw encoder image output",
        "l2_normalized_by_extractor": False,
    },
    "per_text_embedding": {
        "source": "raw encoder text output",
        "l2_normalized_by_extractor": False,
    },
    "text_row_aggregation": {
        "operator": "arithmetic_mean",
        "accumulator_dtype": "float64",
        "input_l2_normalized": False,
        "output_l2_normalized": False,
        "stored_dtype": "float32",
    },
    "stored_image_embedding": {
        "l2_normalized": False,
        "dtype": "float32",
    },
    "consumer_rule": (
        "Stored vectors are unnormalized. A downstream runner must apply one "
        "explicit, identical normalization policy to every compared method; "
        "the trace extractor performs no implicit L2 normalization."
    ),
}


_RUNTIME_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")


def canonicalize_runtime_repr(value: str) -> str:
    """Remove process-local addresses from an otherwise structural repr."""

    return _RUNTIME_ADDRESS.sub("", str(value))


class FeatureEncoder(Protocol):
    model_id: str

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def contract(self) -> Mapping[str, Any]:
        ...


class OpenAIClipViTB32Encoder:
    """Batched OpenAI CLIP ViT-B/32 encoder with lazy heavyweight imports."""

    model_id = "openai-clip:ViT-B/32"

    def __init__(
        self,
        device: str = "cuda",
        download_root: Optional[os.PathLike[str]] = None,
    ) -> None:
        # PyTorch requires this environment setting before the first CUDA
        # workspace is created when deterministic CUDA GEMMs are requested.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise TraceContractError(
                "OpenAI CLIP extraction requires torch, Pillow, and the `clip` package"
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise TraceContractError(f"requested CLIP device is unavailable: {device}")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        self.torch = torch
        self.clip = clip
        self.Image = Image
        self.device = device
        self.download_root = (
            Path(download_root).expanduser().resolve()
            if download_root is not None
            else (Path.home() / ".cache" / "clip").resolve()
        )
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=device, download_root=str(self.download_root)
        )
        self.model.eval()
        model_urls = getattr(clip, "_MODELS", None)
        if not model_urls:
            # Official OpenAI CLIP releases keep the registry in ``clip.clip``
            # while exposing only load/tokenize at package level.
            model_urls = getattr(getattr(clip, "clip", None), "_MODELS", {})
        self.checkpoint_url = model_urls.get("ViT-B/32")
        if not self.checkpoint_url:
            raise TraceContractError("installed clip package does not expose ViT-B/32 URL")
        self.checkpoint_path = self.download_root / Path(self.checkpoint_url).name
        if not self.checkpoint_path.is_file():
            raise TraceContractError(
                f"CLIP checkpoint was not retained at {self.checkpoint_path}"
            )

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, 512), dtype=np.float32)
        tensors = []
        for raw in image_paths:
            path = Path(raw)
            if not path.is_file():
                raise TraceContractError(f"missing image during CLIP extraction: {path}")
            with self.Image.open(path) as image:
                tensors.append(self.preprocess(image.convert("RGB")))
        batch = self.torch.stack(tensors, dim=0).to(self.device)
        with self.torch.no_grad():
            encoded = self.model.encode_image(batch)
        return encoded.detach().float().cpu().numpy().astype(np.float32, copy=False)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        tokens = self.clip.tokenize([str(value) for value in texts], truncate=True).to(
            self.device
        )
        with self.torch.no_grad():
            encoded = self.model.encode_text(tokens)
        return encoded.detach().float().cpu().numpy().astype(np.float32, copy=False)

    def contract(self) -> Mapping[str, Any]:
        module_path = Path(self.clip.__file__).resolve()
        try:
            import PIL
        except ImportError as exc:
            raise TraceContractError("Pillow disappeared after encoder setup") from exc
        package_root = Path(__file__).resolve().parent
        code_inventory = []
        for path in sorted(package_root.glob("*.py"), key=lambda value: value.name):
            code_inventory.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        try:
            clip_distribution_version = importlib_metadata.version("clip")
        except importlib_metadata.PackageNotFoundError:
            clip_distribution_version = None
        preprocess_repr = canonicalize_runtime_repr(repr(self.preprocess))
        torch_build_config = canonicalize_runtime_repr(self.torch.__config__.show())
        execution_environment: Dict[str, Any] = {
            "model_parameter_dtype": str(next(self.model.parameters()).dtype),
            "deterministic_algorithms": bool(
                self.torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": bool(
                self.torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "float32_matmul_precision": self.torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_cuda_build_version": self.torch.version.cuda,
            "cudnn_version": self.torch.backends.cudnn.version(),
            "cudnn_benchmark": bool(self.torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(self.torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": bool(self.torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(
                self.torch.backends.cuda.matmul.allow_tf32
            ),
            "torch_build_config": torch_build_config,
            "torch_build_config_sha256": sha256_bytes(
                torch_build_config.encode("utf-8")
            ),
        }
        if self.device.startswith("cuda"):
            requested = self.torch.device(self.device)
            device_index = (
                self.torch.cuda.current_device()
                if requested.index is None
                else requested.index
            )
            properties = self.torch.cuda.get_device_properties(device_index)
            try:
                nvidia_smi = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,driver_version,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                nvidia_smi = None
            execution_environment["cuda_device"] = {
                "index": int(device_index),
                "name": properties.name,
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
                "nvidia_smi_inventory": nvidia_smi,
            }
        return {
            "model_id": self.model_id,
            "device": self.device,
            "torch_version": self.torch.__version__,
            "pillow_version": PIL.__version__,
            "numpy_version": np.__version__,
            "clip_module_path": str(module_path),
            "clip_module_sha256": sha256_file(module_path),
            "clip_module_version": getattr(self.clip, "__version__", None),
            "clip_distribution_version": clip_distribution_version,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_bytes": self.checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "checkpoint_source_url": self.checkpoint_url,
            "preprocess": preprocess_repr,
            "preprocess_repr_sha256": sha256_bytes(preprocess_repr.encode("utf-8")),
            "preprocess_repr_canonicalization": "strip-process-address-v1",
            "text_tokenizer": "openai_clip.tokenize(truncate=True)",
            "execution_environment": execution_environment,
            "output_dtype": "float32",
            "normalization": {
                "encode_image_output_l2_normalized": False,
                "encode_text_output_l2_normalized": False,
                "extractor_applies_l2_normalization": False,
            },
            "trace_code_inventory": code_inventory,
            "trace_code_sha256": sha256_json(code_inventory),
        }


@dataclass(frozen=True)
class ExtractionConfig:
    output_root: Path
    run_name: str = "clip_vit_b32_v1"
    batch_size: int = 64
    text_batch_size: int = 512
    shard_rows: int = 1024
    resume: bool = True
    hash_images: bool = True
    hash_source_artifacts: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0 or self.text_batch_size <= 0 or self.shard_rows <= 0:
            raise TraceContractError("batch sizes and shard_rows must be positive")
        if not self.run_name or any(value in self.run_name for value in ("/", "\\")):
            raise TraceContractError("run_name must be one safe path component")
        if not self.hash_images or not self.hash_source_artifacts:
            raise TraceContractError(
                "production trace contracts require raw image and source SHA-256 hashing"
            )


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _feature_digest(row: np.ndarray) -> str:
    return sha256_bytes(np.asarray(row, dtype=np.float32).tobytes(order="C"))


def _merkle_root(hex_leaves: Sequence[str]) -> str:
    if not hex_leaves:
        return sha256_bytes(b"")
    level = [bytes.fromhex(value) for value in hex_leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            bytes.fromhex(sha256_bytes(level[index] + level[index + 1]))
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _dependency_inventory(
    adapter: DatasetAdapter,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    leaves = []
    total_bytes = 0
    for raw in adapter.dependency_paths():
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise TraceContractError(f"missing raw dependency: {path}")
        descriptor = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        lookup[str(path)] = descriptor
        total_bytes += int(descriptor["bytes"])
        leaves.append(sha256_json(descriptor))
    summary = {
        "algorithm": "sorted-path-sha256-merkle-v1",
        "files": len(lookup),
        "total_bytes": total_bytes,
        "merkle_root_sha256": _merkle_root(leaves),
    }
    summary["summary_sha256"] = sha256_json(summary)
    return summary, lookup


def _encode_text_rows(
    encoder: FeatureEncoder,
    rows: Sequence[TraceRow],
    text_batch_size: int,
) -> np.ndarray:
    flattened: List[str] = []
    offsets = [0]
    for row in rows:
        flattened.extend(row.encoded_texts)
        offsets.append(len(flattened))
    encoded_parts = []
    for start in range(0, len(flattened), text_batch_size):
        stop = min(len(flattened), start + text_batch_size)
        part = np.asarray(encoder.encode_texts(flattened[start:stop]), dtype=np.float32)
        if part.ndim != 2 or part.shape[0] != stop - start:
            raise TraceContractError("encoder returned an invalid text feature batch")
        encoded_parts.append(part)
    if not encoded_parts:
        raise TraceContractError("trace rows unexpectedly contained no encoded text")
    encoded = np.concatenate(encoded_parts, axis=0)
    output = []
    for row_index in range(len(rows)):
        begin, end = offsets[row_index], offsets[row_index + 1]
        output.append(encoded[begin:end].astype(np.float64).mean(axis=0))
    return np.asarray(output, dtype=np.float32)


def _encode_rows(
    encoder: FeatureEncoder,
    rows: Sequence[TraceRow],
    image_batch_size: int,
    text_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image_parts = []
    paths = [row.image_path for row in rows]
    for start in range(0, len(paths), image_batch_size):
        stop = min(len(paths), start + image_batch_size)
        part = np.asarray(encoder.encode_images(paths[start:stop]), dtype=np.float32)
        if part.ndim != 2 or part.shape[0] != stop - start:
            raise TraceContractError("encoder returned an invalid image feature batch")
        image_parts.append(part)
    image = np.concatenate(image_parts, axis=0)
    text = _encode_text_rows(encoder, rows, text_batch_size)
    if image.shape != text.shape:
        raise TraceContractError(
            f"image/text feature shapes disagree: {image.shape} versus {text.shape}"
        )
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise TraceContractError("encoder produced NaN or infinity")
    return image, text


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_receipts(run_dir: Path, contract_sha256: str) -> Tuple[int, str, int]:
    receipts = sorted((run_dir / "receipts").glob("part-*.json"))
    expected_start = 0
    previous_chain = "0" * 64
    for expected_part, path in enumerate(receipts):
        receipt = _load_json(path)
        if int(receipt.get("part", -1)) != expected_part:
            raise TraceContractError(f"non-contiguous shard receipt sequence at {path}")
        if receipt.get("contract_sha256") != contract_sha256:
            raise TraceContractError(f"receipt contract mismatch: {path}")
        if int(receipt.get("start", -1)) != expected_start:
            raise TraceContractError(f"receipt row gap or overlap: {path}")
        if receipt.get("previous_chain_sha256") != previous_chain:
            raise TraceContractError(f"receipt chain predecessor mismatch: {path}")
        npz_path = run_dir / str(receipt["npz_path"])
        manifest_path = run_dir / str(receipt["manifest_path"])
        if sha256_file(npz_path) != receipt.get("npz_sha256"):
            raise TraceContractError(f"NPZ content poison detected: {npz_path}")
        if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
            raise TraceContractError(f"manifest content poison detected: {manifest_path}")
        receipt_body = dict(receipt)
        observed_chain = receipt_body.pop("chain_sha256", None)
        expected_chain = sha256_json(receipt_body)
        if observed_chain != expected_chain:
            raise TraceContractError(f"receipt chain digest mismatch: {path}")
        with np.load(npz_path, allow_pickle=False) as values:
            row_indices = np.asarray(values["row_index"]).reshape(-1)
            row_ids = np.asarray(values["row_ids"]).reshape(-1)
            image_features = np.asarray(values["image_features"], dtype=np.float32)
            text_features = np.asarray(values["text_features"], dtype=np.float32)
            labels = np.asarray(values["labels"], dtype=np.uint8)
            if row_indices.size != int(receipt["rows"]):
                raise TraceContractError(f"NPZ row count mismatch: {npz_path}")
            expected = np.arange(
                int(receipt["start"]), int(receipt["stop"]), dtype=np.int64
            )
            if not np.array_equal(row_indices.astype(np.int64), expected):
                raise TraceContractError(f"NPZ row sequence mismatch: {npz_path}")
            if row_ids.dtype.kind != "S" or row_ids.dtype.itemsize != 64:
                raise TraceContractError(f"NPZ row_ids must be fixed S64: {npz_path}")
            if not (
                image_features.ndim == text_features.ndim == labels.ndim == 2
                and image_features.shape == text_features.shape
                and image_features.shape[0] == labels.shape[0] == row_indices.size
            ):
                raise TraceContractError(f"NPZ aligned-array shape mismatch: {npz_path}")
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TraceContractError(f"invalid manifest JSONL: {manifest_path}") from exc
        if len(records) != row_indices.size:
            raise TraceContractError(f"manifest row count mismatch: {manifest_path}")
        for offset, record in enumerate(records):
            expected_row = int(row_indices[offset])
            if int(record.get("row_index", -1)) != expected_row:
                raise TraceContractError(
                    f"manifest/NPZ row mismatch at {manifest_path}:{offset + 1}"
                )
            expected_row_id = row_ids[offset].decode("ascii")
            if record.get("canonical_row_id") != expected_row_id:
                raise TraceContractError(
                    f"manifest/NPZ canonical row ID mismatch at row {expected_row}"
                )
            observed_row_contract = record.get("row_contract_sha256")
            base_record = dict(record)
            base_record.pop("row_contract_sha256", None)
            base_record.pop("image_feature_sha256", None)
            base_record.pop("text_feature_sha256", None)
            base_record.pop("feature_binding_sha256", None)
            if sha256_json(base_record) != observed_row_contract:
                raise TraceContractError(
                    f"manifest row contract mismatch at row {expected_row}"
                )
            image_digest = _feature_digest(image_features[offset])
            text_digest = _feature_digest(text_features[offset])
            if image_digest != record.get("image_feature_sha256"):
                raise TraceContractError(f"image feature row digest mismatch at {expected_row}")
            if text_digest != record.get("text_feature_sha256"):
                raise TraceContractError(f"text feature row digest mismatch at {expected_row}")
            if list(labels[offset].astype(int)) != record.get("label_hot"):
                raise TraceContractError(f"label row mismatch at {expected_row}")
            binding = sha256_json(
                {
                    "row_contract_sha256": observed_row_contract,
                    "image_feature_sha256": image_digest,
                    "text_feature_sha256": text_digest,
                }
            )
            if binding != record.get("feature_binding_sha256"):
                raise TraceContractError(f"feature binding mismatch at {expected_row}")
        expected_start = int(receipt["stop"])
        previous_chain = expected_chain
    return expected_start, previous_chain, len(receipts)


def _contract(
    adapter: DatasetAdapter,
    encoder: FeatureEncoder,
    config: ExtractionConfig,
    dependency_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    source_inventory: Any
    if config.hash_source_artifacts:
        source_inventory = list(file_inventory(adapter.source_artifacts))
    else:
        source_inventory = [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": None}
            for path in adapter.source_artifacts
        ]
    return {
        "schema_version": 1,
        "adapter": adapter.contract(),
        "encoder": dict(encoder.contract()),
        "source_inventory": source_inventory,
        "raw_dependency_inventory": dict(dependency_summary),
        "feature_semantics": FEATURE_SEMANTICS,
        "extraction": {
            "run_name": config.run_name,
            "batch_size": config.batch_size,
            "text_batch_size": config.text_batch_size,
            "shard_rows": config.shard_rows,
            "hash_images": config.hash_images,
            "text_aggregation": (
                "unnormalized per-text encoder rows -> float64 arithmetic mean -> "
                "unnormalized float32 row"
            ),
            "image_aggregation": "unnormalized encoder image row -> float32 row",
            "l2_normalization_applied": False,
            "npz_dtype": "float32",
            "python": sys.version,
            "platform": platform.platform(),
        },
    }


def preflight_adapter(adapter: DatasetAdapter, limit: int = 12) -> Dict[str, Any]:
    """Validate every raw canonical row without loading or running CLIP."""

    if limit < 0:
        raise TraceContractError("preflight limit must be non-negative")
    if adapter.rows == 0:
        raise TraceContractError("adapter has no canonical rows")
    if limit:
        sample_indices = set(
            int(value)
            for value in np.linspace(
                0, adapter.rows - 1, min(limit, adapter.rows), dtype=np.int64
            )
        )
    else:
        sample_indices = set()
    label_dimension: Optional[int] = None
    minimum_texts: Optional[int] = None
    maximum_texts = 0
    fallback_rows = 0
    split_counts = {"indQ": 0, "indT": 0, "indD": 0}
    samples = []
    seen_ids = set()
    seen_rows = 0
    for expected_row, row in enumerate(adapter.iter_rows()):
        if row.row_index != expected_row:
            raise TraceContractError(
                f"preflight non-contiguous row: expected={expected_row}, got={row.row_index}"
            )
        if row.source_id in seen_ids:
            raise TraceContractError(f"preflight duplicate source ID: {row.source_id}")
        seen_ids.add(row.source_id)
        if not hasattr(adapter.split, "source_ids"):
            raise TraceContractError("preflight split does not expose canonical identities")
        expected_source_id = str(adapter.split.source_ids[expected_row])
        if row.source_id != expected_source_id:
            raise TraceContractError(
                "preflight adapter/split identity mismatch at row "
                f"{expected_row}: row={row.source_id!r}, split={expected_source_id!r}"
            )
        if dict(row.split) != adapter.split.flags(expected_row):
            raise TraceContractError(
                f"preflight split flag mismatch at row {expected_row}"
            )
        if not Path(row.image_path).is_file():
            raise TraceContractError(f"preflight missing image: {row.image_path}")
        current_dimension = len(row.label_hot)
        if label_dimension is None:
            label_dimension = current_dimension
        elif label_dimension != current_dimension:
            raise TraceContractError(
                f"preflight label dimension drift at row {row.row_index}"
            )
        text_count = len(row.encoded_texts)
        minimum_texts = text_count if minimum_texts is None else min(minimum_texts, text_count)
        maximum_texts = max(maximum_texts, text_count)
        if bool(row.raw_text.get("baseline_fallback", False)):
            fallback_rows += 1
        for key in split_counts:
            split_counts[key] += int(bool(row.split[key]))
        if row.row_index in sample_indices:
            samples.append(row.record(hash_image=False))
        seen_rows += 1
    if seen_rows != adapter.rows:
        raise TraceContractError(
            f"preflight row count mismatch: iterated={seen_rows}, declared={adapter.rows}"
        )
    split_summary = adapter.split.summary()
    expected_split_counts = {
        "indQ": int(split_summary["indQ_rows"]),
        "indT": int(split_summary["indT_rows"]),
        "indD": int(split_summary["indD_rows"]),
    }
    if split_counts != expected_split_counts:
        raise TraceContractError(
            f"preflight split counts differ: {split_counts} != {expected_split_counts}"
        )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "adapter": adapter.contract(),
        "canonical_rows": seen_rows,
        "unique_source_ids": len(seen_ids),
        "label_dimension": label_dimension,
        "encoded_text_items_min": minimum_texts,
        "encoded_text_items_max": maximum_texts,
        "baseline_fallback_rows": fallback_rows,
        "split_counts": split_counts,
        "sample_limit": limit,
        "samples": samples,
        "clip_loaded": False,
        "process_data_used_for_identity": False,
    }
    summary["preflight_sha256"] = sha256_json(summary)
    return summary


def extract_trace_bundle(
    adapter: DatasetAdapter,
    encoder: FeatureEncoder,
    config: ExtractionConfig,
) -> Dict[str, Any]:
    config.validate()
    safe_root = ensure_output_safe(
        config.output_root,
        [adapter.data_root / "OralData", adapter.data_root / "ProcessData"],
    )
    run_dir = safe_root / adapter.dataset / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "shards").mkdir(exist_ok=True)
    (run_dir / "manifests").mkdir(exist_ok=True)
    (run_dir / "receipts").mkdir(exist_ok=True)

    dependency_summary, dependency_lookup = _dependency_inventory(adapter)
    contract = _contract(adapter, encoder, config, dependency_summary)
    contract_sha256 = sha256_json(contract)
    contract_record = dict(contract)
    contract_record["contract_sha256"] = contract_sha256
    contract_path = run_dir / "contract.json"
    if contract_path.exists():
        existing = _load_json(contract_path)
        if existing != contract_record:
            raise TraceContractError(
                "resume contract differs from the sealed extraction contract"
            )
    else:
        atomic_write_json(contract_path, contract_record)

    if adapter.split is None or not hasattr(adapter.split, "arrays"):
        raise TraceContractError("adapter did not expose a canonical zero-based split")
    split_arrays = adapter.split.arrays()
    split_path = run_dir / "canonical_split.npz"
    split_record = {
        "schema_version": 1,
        "summary": adapter.split.summary(),
    }
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as existing_split:
            for key, expected in split_arrays.items():
                if key not in existing_split or not np.array_equal(
                    np.asarray(existing_split[key]), expected
                ):
                    raise TraceContractError(
                        f"canonical split content differs for {key}"
                    )
    else:
        _atomic_save_npz(split_path, split_arrays)
    split_record["npz_sha256"] = sha256_file(split_path)
    split_record["split_contract_sha256"] = sha256_json(split_record)
    split_json_path = run_dir / "canonical_split.json"
    if split_json_path.exists():
        if _load_json(split_json_path) != split_record:
            raise TraceContractError("canonical split JSON differs on resume")
    else:
        atomic_write_json(split_json_path, split_record)

    next_row, previous_chain, next_part = _validate_receipts(
        run_dir, contract_sha256
    )
    complete_path = run_dir / "complete.json"
    if complete_path.exists():
        complete = _load_json(complete_path)
        complete_body = dict(complete)
        complete_digest = complete_body.pop("complete_sha256", None)
        if (
            sha256_json(complete_body) != complete_digest
            or
            complete.get("contract_sha256") != contract_sha256
            or int(complete.get("rows", -1)) != adapter.rows
            or int(complete.get("shards", -1)) != next_part
            or complete.get("final_chain_sha256") != previous_chain
            or next_row != adapter.rows
        ):
            raise TraceContractError("complete marker does not match verified shard state")
        return complete
    if next_row and not config.resume:
        raise TraceContractError("existing shards found but resume is disabled")
    if next_row > adapter.rows:
        raise TraceContractError("existing receipts exceed adapter row count")

    buffered: List[TraceRow] = []
    for row in adapter.iter_rows(start=next_row):
        expected = next_row + len(buffered)
        if row.row_index != expected:
            raise TraceContractError(
                f"adapter yielded non-contiguous row {row.row_index}, expected {expected}"
            )
        expected_row_id = split_arrays["row_ids"][row.row_index].decode("ascii")
        if row.canonical_row_id() != expected_row_id:
            raise TraceContractError(
                "adapter row identity differs from canonical split at row "
                f"{row.row_index}"
            )
        buffered.append(row)
        if len(buffered) < config.shard_rows:
            continue
        previous_chain = _emit_shard(
            run_dir,
            contract_sha256,
            next_part,
            previous_chain,
            buffered,
            encoder,
            config,
            dependency_lookup,
        )
        next_row += len(buffered)
        next_part += 1
        buffered = []
    if buffered:
        previous_chain = _emit_shard(
            run_dir,
            contract_sha256,
            next_part,
            previous_chain,
            buffered,
            encoder,
            config,
            dependency_lookup,
        )
        next_row += len(buffered)
        next_part += 1
    if next_row != adapter.rows:
        raise TraceContractError(
            f"adapter ended early: emitted={next_row}, expected={adapter.rows}"
        )
    complete = {
        "schema_version": 1,
        "dataset": adapter.dataset,
        "rows": adapter.rows,
        "shards": next_part,
        "contract_sha256": contract_sha256,
        "final_chain_sha256": previous_chain,
        "run_dir": str(run_dir.resolve()),
    }
    complete["complete_sha256"] = sha256_json(complete)
    atomic_write_json(complete_path, complete)
    verification = verify_trace_bundle(run_dir)
    if (
        int(verification["rows"]) != adapter.rows
        or int(verification["shards"]) != next_part
    ):
        raise TraceContractError("post-write full bundle verification failed")
    return complete


def _emit_shard(
    run_dir: Path,
    contract_sha256: str,
    part: int,
    previous_chain: str,
    rows: Sequence[TraceRow],
    encoder: FeatureEncoder,
    config: ExtractionConfig,
    dependency_inventory: Mapping[str, Mapping[str, Any]],
) -> str:
    start = rows[0].row_index
    stop = rows[-1].row_index + 1
    if [row.row_index for row in rows] != list(range(start, stop)):
        raise TraceContractError("shard rows are not contiguous")
    image, text = _encode_rows(
        encoder, rows, config.batch_size, config.text_batch_size
    )
    labels = np.asarray([row.label_hot for row in rows], dtype=np.uint8)
    if labels.ndim != 2 or labels.shape[0] != len(rows):
        raise TraceContractError("invalid label batch")
    row_indices = np.asarray([row.row_index for row in rows], dtype=np.int64)
    row_ids = np.asarray([row.canonical_row_id() for row in rows], dtype="S64")

    records = []
    for offset, row in enumerate(rows):
        record = row.record(
            hash_image=True, dependency_inventory=dependency_inventory
        )
        record["image_feature_sha256"] = _feature_digest(image[offset])
        record["text_feature_sha256"] = _feature_digest(text[offset])
        record["feature_binding_sha256"] = sha256_json(
            {
                "row_contract_sha256": record["row_contract_sha256"],
                "image_feature_sha256": record["image_feature_sha256"],
                "text_feature_sha256": record["text_feature_sha256"],
            }
        )
        records.append(record)

    base = f"part-{part:06d}"
    npz_relative = Path("shards") / f"{base}.npz"
    manifest_relative = Path("manifests") / f"{base}.jsonl"
    npz_path = run_dir / npz_relative
    manifest_path = run_dir / manifest_relative
    _atomic_save_npz(
        npz_path,
        {
            "row_index": row_indices,
            "row_ids": row_ids,
            "image_features": image.astype(np.float32, copy=False),
            "text_features": text.astype(np.float32, copy=False),
            "labels": labels,
        },
    )
    manifest_payload = b"".join(
        stable_json_bytes(record) + b"\n" for record in records
    )
    atomic_write_bytes(manifest_path, manifest_payload)
    receipt = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "part": part,
        "start": start,
        "stop": stop,
        "rows": len(rows),
        "npz_path": npz_relative.as_posix(),
        "manifest_path": manifest_relative.as_posix(),
        "npz_sha256": sha256_file(npz_path),
        "manifest_sha256": sha256_file(manifest_path),
        "previous_chain_sha256": previous_chain,
    }
    receipt["chain_sha256"] = sha256_json(receipt)
    atomic_write_json(run_dir / "receipts" / f"{base}.json", receipt)
    return str(receipt["chain_sha256"])


def load_trace_bundle(
    run_dir: os.PathLike[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    verification = verify_trace_bundle(root)
    complete = verification["complete"]
    contract = verification["contract"]
    contract_sha256 = str(contract["contract_sha256"])
    _, _, _ = _validate_receipts(root, contract_sha256)
    rows = int(complete["rows"])
    image_parts = []
    text_parts = []
    label_parts = []
    records: List[Dict[str, Any]] = []
    for receipt_path in sorted((root / "receipts").glob("part-*.json")):
        receipt = _load_json(receipt_path)
        with np.load(root / receipt["npz_path"], allow_pickle=False) as values:
            image_parts.append(np.asarray(values["image_features"], dtype=np.float32))
            text_parts.append(np.asarray(values["text_features"], dtype=np.float32))
            label_parts.append(np.asarray(values["labels"], dtype=np.uint8))
        with (root / receipt["manifest_path"]).open(
            "r", encoding="utf-8"
        ) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    if len(records) != rows:
        raise TraceContractError("bundle manifest row count mismatch")
    return (
        np.concatenate(image_parts, axis=0),
        np.concatenate(text_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        records,
        complete,
    )


def verify_trace_bundle(run_dir: os.PathLike[str]) -> Dict[str, Any]:
    """Fully verify contracts, split, receipts, manifests, and NPZ row hashes."""

    root = Path(run_dir).expanduser().resolve()
    contract = _load_json(root / "contract.json")
    contract_sha256 = str(contract.get("contract_sha256", ""))
    body = dict(contract)
    body.pop("contract_sha256", None)
    if sha256_json(body) != contract_sha256:
        raise TraceContractError("bundle contract digest mismatch")
    split_record = _load_json(root / "canonical_split.json")
    split_body = dict(split_record)
    split_digest = split_body.pop("split_contract_sha256", None)
    if sha256_json(split_body) != split_digest:
        raise TraceContractError("canonical split contract digest mismatch")
    split_path = root / "canonical_split.npz"
    if sha256_file(split_path) != split_record.get("npz_sha256"):
        raise TraceContractError("canonical split NPZ digest mismatch")
    if split_record.get("summary") != contract.get("adapter", {}).get("split"):
        raise TraceContractError("canonical split summary differs from adapter contract")
    rows, chain, parts = _validate_receipts(root, contract_sha256)
    with np.load(split_path, allow_pickle=False) as split_values:
        split_row_ids = np.asarray(split_values["row_ids"]).reshape(-1)
    shard_row_ids = []
    for receipt_path in sorted((root / "receipts").glob("part-*.json")):
        receipt = _load_json(receipt_path)
        with np.load(root / receipt["npz_path"], allow_pickle=False) as values:
            shard_row_ids.append(np.asarray(values["row_ids"]).reshape(-1))
    concatenated_row_ids = (
        np.concatenate(shard_row_ids)
        if shard_row_ids
        else np.empty((0,), dtype="S64")
    )
    if not np.array_equal(split_row_ids, concatenated_row_ids):
        raise TraceContractError(
            "canonical split row_ids differ from concatenated shard row_ids"
        )
    complete = _load_json(root / "complete.json")
    complete_body = dict(complete)
    complete_digest = complete_body.pop("complete_sha256", None)
    if (
        sha256_json(complete_body) != complete_digest
        or int(complete.get("rows", -1)) != rows
        or int(complete.get("shards", -1)) != parts
        or complete.get("final_chain_sha256") != chain
    ):
        raise TraceContractError("bundle complete marker mismatch")
    return {
        "verified": True,
        "run_dir": str(root),
        "rows": rows,
        "shards": parts,
        "contract": contract,
        "split": split_record,
        "complete": complete,
    }
