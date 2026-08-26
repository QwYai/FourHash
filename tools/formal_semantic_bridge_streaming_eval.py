"""Two-phase complete-gallery evaluation for the ShellGuard semantic bridge.

``freeze`` may read the fit-only artifact labels and label-free formal
features, but it cannot reach formal query/database labels.  It calibrates one
posterior threshold on ``indT``, encodes a fixed 16-bit one-bit MinHash detail
code, copies the already frozen primary codes, and seals every array.  Only
``evaluate`` reopens query/database labels after the plan and all arrays verify.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_training import load_detail_checkpoint
from raw_rebuilt_neural.fit_artifact import open_fit_artifact
from raw_rebuilt_neural.integrity import (
    array_descriptor,
    atomic_save_npy,
    production_code_inventory as neural_code_inventory,
    reject_unsafe_output_path,
)
from raw_rebuilt_neural.semantic_bridge import (
    SemanticBridgeConfig,
    build_one_bit_minhash_map,
    calibrate_training_threshold,
    encode_mean_posterior,
    encode_semantic_bridge,
    semantic_bridge_composite_distance,
)
from raw_rebuilt_runtime import load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    load_json,
    numeric_sha256,
    sha256_file,
    sha256_json,
)
from raw_rebuilt_runtime.metric_loader import load_frozen_metric_labels
from raw_rebuilt_streaming.integrity import (
    production_code_inventory as streaming_code_inventory,
)
from raw_rebuilt_streaming.metrics import (
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
    mean_query_metrics,
)
from tools.formal_ccde_streaming_eval import (
    _PackedDistanceBackend,
    _query_truth,
    _runtime_identity,
    _runtime_manifest,
)


BITS = (16, 32, 64)
DIRECTIONS = ("i2t", "t2i")
PLAN_SCHEMA = "shellguard_semantic_bridge_plan_v1"
CACHE_SCHEMA = "shellguard_semantic_bridge_cache_v1"
PARTIAL_SCHEMA = "shellguard_semantic_bridge_partial_v1"
RESULT_SCHEMA = "shellguard_semantic_bridge_metric_result_v1"
EVALUATION_SCHEMA = "shellguard_semantic_bridge_evaluation_v1"


class FormalSemanticBridgeError(RuntimeError):
    """Raised when a semantic-bridge formal artifact fails closed."""


@dataclass(frozen=True)
class FormalSemanticBridgeConfig:
    bits: tuple[int, ...] = BITS
    directions: tuple[str, ...] = DIRECTIONS
    cutoffs: tuple[int, ...] = (50, 100, 1000)
    query_chunk_size: int = 32
    inference_batch_size: int = 512
    bridge: SemanticBridgeConfig = SemanticBridgeConfig()

    def __post_init__(self) -> None:
        if not self.bits or any(bits not in BITS for bits in self.bits):
            raise ValueError("bits must be a nonempty subset of 16/32/64")
        if not self.directions or any(value not in DIRECTIONS for value in self.directions):
            raise ValueError("directions must be a nonempty subset of i2t/t2i")
        if not self.cutoffs or any(value < 1 for value in self.cutoffs):
            raise ValueError("cutoffs must be positive")
        if self.query_chunk_size < 1 or self.inference_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.bridge.detail_bits != 16:
            raise ValueError("the frozen semantic bridge uses exactly 16 bits")


def _implementation_inventory() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(strict=True),
        (PROJECT_ROOT / "tools" / "formal_ccde_streaming_eval.py").resolve(strict=True),
        (PROJECT_ROOT / "raw_rebuilt_neural" / "semantic_bridge.py").resolve(strict=True),
        (PROJECT_ROOT / "raw_rebuilt_runtime" / "metric_loader.py").resolve(strict=True),
        (PROJECT_ROOT / "raw_rebuilt_streaming" / "metrics.py").resolve(strict=True),
    )
    files = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    body = {"schema": "shellguard_semantic_bridge_code_inventory_v1", "files": files}
    return {**body, "code_inventory_sha256": sha256_json(body)}


def _close_memmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _verify_array(path: Path, descriptor: Mapping[str, Any]) -> np.ndarray:
    if descriptor.get("path") != path.name:
        raise FormalSemanticBridgeError(f"array path changed: {path.name}")
    if not path.is_file() or path.stat().st_size != int(descriptor.get("size", -1)):
        raise FormalSemanticBridgeError(f"array size changed: {path.name}")
    if sha256_file(path) != descriptor.get("file_sha256"):
        raise FormalSemanticBridgeError(f"array bytes changed: {path.name}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != descriptor.get("shape") or value.dtype.str != descriptor.get(
        "dtype"
    ):
        _close_memmap(value)
        raise FormalSemanticBridgeError(f"array geometry changed: {path.name}")
    if numeric_sha256(value) != descriptor.get("numeric_sha256"):
        _close_memmap(value)
        raise FormalSemanticBridgeError(f"array numeric content changed: {path.name}")
    return value


def _load_base_primary_codes(
    base_plan_root: Path,
    identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    root = Path(base_plan_root).expanduser().resolve(strict=True)
    plan_path = root / "evaluation_plan.json"
    plan = load_json(plan_path)
    body = {key: plan[key] for key in plan if key != "rank_plan_sha256"}
    if plan.get("status") != "rank_state_frozen" or plan.get(
        "labels_loaded_during_freeze"
    ) is not False:
        raise FormalSemanticBridgeError("base plan did not freeze before label opening")
    if sha256_json(body) != plan.get("rank_plan_sha256"):
        raise FormalSemanticBridgeError("base plan content hash changed")
    if plan.get("runtime_identity") != dict(identity):
        raise FormalSemanticBridgeError("base plan runtime identity differs")
    cache_record = plan.get("encoding_cache", {})
    cache_root = root / str(cache_record.get("path", ""))
    manifest_path = cache_root / "manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != int(cache_record.get("manifest_size", -1))
        or sha256_file(manifest_path) != cache_record.get("manifest_sha256")
    ):
        raise FormalSemanticBridgeError("base primary-code cache manifest changed")
    manifest = load_json(manifest_path)
    manifest_body = {key: manifest[key] for key in manifest if key != "manifest_sha256"}
    if sha256_json(manifest_body) != manifest.get("manifest_sha256"):
        raise FormalSemanticBridgeError("base cache manifest content changed")
    descriptors = manifest.get("arrays")
    if not isinstance(descriptors, dict):
        raise FormalSemanticBridgeError("base cache array inventory is missing")
    arrays: dict[str, np.ndarray] = {}
    for modality in ("image", "text"):
        for bits in BITS:
            name = f"primary_{modality}_codes_{bits}"
            descriptor = descriptors.get(name)
            if not isinstance(descriptor, dict):
                raise FormalSemanticBridgeError(f"base cache lacks {name}")
            target = cache_root / str(descriptor.get("path", ""))
            value = _verify_array(target, descriptor)
            if value.shape != (int(identity["rows"]), bits) or value.dtype != np.int8:
                _close_memmap(value)
                raise FormalSemanticBridgeError(f"base primary geometry differs: {name}")
            if not np.all(np.isin(value, (-1, 1))):
                _close_memmap(value)
                raise FormalSemanticBridgeError(f"base primary code is not bipolar: {name}")
            arrays[name] = value
    receipt = {
        "path": str(plan_path),
        "size": plan_path.stat().st_size,
        "file_sha256": sha256_file(plan_path),
        "rank_plan_sha256": plan["rank_plan_sha256"],
        "cache_manifest_file_sha256": sha256_file(manifest_path),
        "cache_manifest_sha256": manifest["manifest_sha256"],
    }
    return plan, arrays, receipt


def _cache_names() -> tuple[str, ...]:
    primary = tuple(
        f"primary_{modality}_codes_{bits}"
        for modality in ("image", "text")
        for bits in BITS
    )
    return primary + (
        "semantic_image_codes_16",
        "semantic_text_codes_16",
        "minhash_ranks",
        "minhash_colors",
    )


def _verify_cache(root: Path, plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    cache_root = root / str(plan["cache"]["path"])
    manifest_path = cache_root / "manifest.json"
    if (
        manifest_path.stat().st_size != int(plan["cache"]["manifest_size"])
        or sha256_file(manifest_path) != plan["cache"]["manifest_file_sha256"]
    ):
        raise FormalSemanticBridgeError("semantic-bridge cache manifest bytes changed")
    manifest = load_json(manifest_path)
    body = {key: manifest[key] for key in manifest if key != "manifest_sha256"}
    if (
        manifest.get("schema") != CACHE_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or sha256_json(body) != manifest.get("manifest_sha256")
        or manifest.get("manifest_sha256") != plan["cache"]["manifest_sha256"]
        or manifest.get("plan_binding_sha256") != plan["binding"][
            "plan_binding_sha256"
        ]
    ):
        raise FormalSemanticBridgeError("semantic-bridge cache manifest changed")
    descriptors = manifest.get("arrays")
    if not isinstance(descriptors, dict) or set(descriptors) != set(_cache_names()):
        raise FormalSemanticBridgeError("semantic-bridge array inventory changed")
    return {
        name: _verify_array(cache_root / str(descriptors[name]["path"]), descriptors[name])
        for name in _cache_names()
    }


def freeze_formal_state(
    runtime_root: Path,
    base_plan_root: Path,
    detail_checkpoint_path: Path,
    architecture_freeze_path: Path,
    fit_artifact_root: Path,
    output_parent: Path,
    *,
    config: FormalSemanticBridgeConfig = FormalSemanticBridgeConfig(),
    device: str | torch.device = "auto",
    _test_allow_synthetic: bool = False,
) -> Path:
    """Freeze primary and neural semantic codes without formal labels."""

    rank = load_label_free_rank_inputs(
        runtime_root, _test_allow_synthetic=_test_allow_synthetic
    )
    fit = open_fit_artifact(
        fit_artifact_root, _test_allow_synthetic=_test_allow_synthetic
    )
    base_arrays: dict[str, np.ndarray] = {}
    try:
        metadata = _runtime_manifest(runtime_root)
        identity = _runtime_identity(rank, metadata)
        if fit.dataset != identity["dataset"] or fit.source_seal_sha256 != identity[
            "source_seal_sha256"
        ]:
            raise FormalSemanticBridgeError("fit artifact and formal runtime differ")
        if fit.label_dim != identity["label_dim"]:
            raise FormalSemanticBridgeError("fit and formal label dimensions differ")
        _base_plan, base_arrays, base_receipt = _load_base_primary_codes(
            base_plan_root, identity
        )
        resolved = torch.device(
            "cuda" if str(device) == "auto" and torch.cuda.is_available() else device
        )
        detail = load_detail_checkpoint(
            detail_checkpoint_path,
            architecture_freeze_path,
            device=resolved,
            expected_source_seal_sha256=rank.source_seal_sha256,
            require_current_code=False,
        )
        fit_image_posterior = encode_mean_posterior(
            detail.model,
            fit.image,
            modality="image",
            device=resolved,
            batch_size=config.inference_batch_size,
        )
        fit_text_posterior = encode_mean_posterior(
            detail.model,
            fit.text,
            modality="text",
            device=resolved,
            batch_size=config.inference_batch_size,
        )
        threshold, calibration = calibrate_training_threshold(
            fit_image_posterior,
            fit_text_posterior,
            np.asarray(fit.labels, dtype=np.uint8),
            config.bridge.threshold_candidates,
        )
        del fit_image_posterior, fit_text_posterior
        config_value = asdict(config)
        binding_body = {
            "schema": PLAN_SCHEMA,
            "dataset": identity["dataset"],
            "source_seal_sha256": identity["source_seal_sha256"],
            "runtime_identity": identity,
            "base_primary_plan": base_receipt,
            "detail_checkpoint_sha256": detail.checkpoint_sha256,
            "detail_checkpoint_run_binding_sha256": detail.metadata["binding"][
                "run_binding_sha256"
            ],
            "detail_checkpoint_loaded_with_strict_state_dict": True,
            "checkpoint_historical_code_receipt_retained": True,
            "fit_artifact_sha256": fit.fit_artifact_sha256,
            "fit_rows": len(fit.labels),
            "selected_threshold": threshold,
            "threshold_calibration": list(calibration),
            "config": config_value,
            "config_sha256": sha256_json(config_value),
            "architecture_freeze_file_sha256": sha256_file(
                Path(architecture_freeze_path).expanduser().resolve(strict=True)
            ),
            "neural_code_inventory": neural_code_inventory(),
            "streaming_code_inventory": streaming_code_inventory(),
            "implementation_inventory": _implementation_inventory(),
            "formal_query_or_database_labels_opened": False,
        }
        binding = {**binding_body, "plan_binding_sha256": sha256_json(binding_body)}
        output = reject_unsafe_output_path(
            Path(output_parent), field="semantic-bridge formal plan output"
        )
        root = output / f"semantic-bridge-plan-{binding['plan_binding_sha256'][:16]}"
        cache_root = root / "cache"
        plan_path = root / "evaluation_plan.json"
        if plan_path.exists():
            return _verify_plan(root)[1]
        root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        saved: dict[str, np.ndarray] = {}
        for name, value in base_arrays.items():
            target_value = np.ascontiguousarray(value, dtype=np.int8)
            atomic_save_npy(cache_root / f"{name}.npy", target_value)
            saved[name] = target_value
        mapping = build_one_bit_minhash_map(
            identity["label_dim"],
            bits=config.bridge.detail_bits,
            seed=config.bridge.minhash_seed,
        )
        atomic_save_npy(cache_root / "minhash_ranks.npy", mapping.ranks)
        atomic_save_npy(cache_root / "minhash_colors.npy", mapping.colors)
        saved["minhash_ranks"] = mapping.ranks
        saved["minhash_colors"] = mapping.colors
        for modality, features in (("image", rank.image), ("text", rank.text)):
            posterior = encode_mean_posterior(
                detail.model,
                features,
                modality=modality,
                device=resolved,
                batch_size=config.inference_batch_size,
            )
            code = encode_semantic_bridge(
                posterior, threshold=threshold, mapping=mapping
            )
            del posterior
            name = f"semantic_{modality}_codes_16"
            atomic_save_npy(cache_root / f"{name}.npy", code)
            saved[name] = code
        descriptors = {
            name: array_descriptor(cache_root / f"{name}.npy", value)
            for name, value in saved.items()
        }
        cache_body = {
            "schema": CACHE_SCHEMA,
            "status": "COMPLETE",
            "plan_binding_sha256": binding["plan_binding_sha256"],
            "rows": identity["rows"],
            "label_dim": identity["label_dim"],
            "detail_bits": config.bridge.detail_bits,
            "selected_threshold": threshold,
            "posterior_cache_retained": False,
            "formal_query_or_database_labels_opened": False,
            "arrays": descriptors,
        }
        cache_manifest = {
            **cache_body,
            "manifest_sha256": sha256_json(cache_body),
        }
        cache_manifest_path = cache_root / "manifest.json"
        atomic_write_json(cache_manifest_path, cache_manifest)
        plan_body = {
            "schema": PLAN_SCHEMA,
            "status": "rank_state_frozen",
            "dataset": identity["dataset"],
            "source_seal_sha256": identity["source_seal_sha256"],
            "runtime_identity": identity,
            "binding": binding,
            "cache": {
                "path": "cache",
                "manifest_size": cache_manifest_path.stat().st_size,
                "manifest_file_sha256": sha256_file(cache_manifest_path),
                "manifest_sha256": cache_manifest["manifest_sha256"],
            },
            "methods": ["primary_hamming", "shellguard_semantic_bridge"],
            "ranking_rule": "17*primary_hamming + fixed16_neural_minhash_hamming",
            "detail_bits_by_primary_width": {str(bits): 16 for bits in BITS},
            "primary_shell_order_is_invariant": True,
            "posterior_cache_retained": False,
            "formal_gate_or_fallback_used": False,
            "labels_loaded_during_freeze": False,
            "metric_label_boundary": "formal labels open only after plan and every binary array verify",
        }
        plan = {**plan_body, "rank_plan_sha256": sha256_json(plan_body)}
        atomic_write_json(plan_path, plan)
        _verify_plan(root)
        return root
    finally:
        for value in base_arrays.values():
            _close_memmap(value)
        fit.close()
        rank.close()


def _verify_plan(plan_root: Path) -> tuple[Mapping[str, Any], Path]:
    root = Path(plan_root).expanduser().resolve(strict=True)
    plan = load_json(root / "evaluation_plan.json")
    body = {key: plan[key] for key in plan if key != "rank_plan_sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "rank_state_frozen"
        or sha256_json(body) != plan.get("rank_plan_sha256")
        or plan.get("labels_loaded_during_freeze") is not False
        or plan.get("primary_shell_order_is_invariant") is not True
        or plan.get("formal_gate_or_fallback_used") is not False
    ):
        raise FormalSemanticBridgeError("semantic-bridge plan contract changed")
    binding = plan.get("binding", {})
    binding_body = {
        key: binding[key] for key in binding if key != "plan_binding_sha256"
    }
    if sha256_json(binding_body) != binding.get("plan_binding_sha256"):
        raise FormalSemanticBridgeError("semantic-bridge plan binding changed")
    if binding.get("implementation_inventory") != _implementation_inventory():
        raise FormalSemanticBridgeError("semantic-bridge implementation changed")
    if binding.get("neural_code_inventory") != neural_code_inventory():
        raise FormalSemanticBridgeError("current neural code differs from the plan")
    if binding.get("streaming_code_inventory") != streaming_code_inventory():
        raise FormalSemanticBridgeError("current streaming code differs from the plan")
    arrays = _verify_cache(root, plan)
    for value in arrays.values():
        _close_memmap(value)
    return plan, root


def _open_backend_cache(
    plan_root: Path,
    plan: Mapping[str, Any],
) -> tuple[Any, list[np.ndarray]]:
    arrays = _verify_cache(plan_root, plan)
    rows = int(plan["runtime_identity"]["rows"])
    primary_image = {}
    primary_text = {}
    for bits in BITS:
        primary_image[bits] = arrays[f"primary_image_codes_{bits}"]
        primary_text[bits] = arrays[f"primary_text_codes_{bits}"]
    semantic_image = arrays["semantic_image_codes_16"]
    semantic_text = arrays["semantic_text_codes_16"]
    if semantic_image.shape != (rows, 16) or semantic_text.shape != (rows, 16):
        raise FormalSemanticBridgeError("semantic bridge code geometry changed")
    cache = SimpleNamespace(
        primary_image_codes=primary_image,
        primary_text_codes=primary_text,
        detail_image_codes={bits: semantic_image for bits in BITS},
        detail_text_codes={bits: semantic_text for bits in BITS},
    )
    return cache, list(arrays.values())


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
    start = 0
    chain = "0" * 64
    primary_records: list[dict[str, Any]] = []
    bridge_records: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for path in paths:
        receipt = load_json(path)
        body = {
            key: receipt[key]
            for key in receipt
            if key not in {"receipt_sha256", "chain_sha256"}
        }
        receipt_sha = sha256_json(body)
        next_chain = sha256_json(
            {"previous_chain_sha256": chain, "receipt_sha256": receipt_sha}
        )
        end = int(receipt.get("end", -1))
        primary = receipt.get("primary_records")
        bridge = receipt.get("bridge_records")
        if (
            receipt.get("schema") != PARTIAL_SCHEMA
            or receipt.get("status") != "COMMITTED"
            or receipt.get("rank_plan_sha256") != plan["rank_plan_sha256"]
            or receipt.get("direction") != direction
            or int(receipt.get("bits", -1)) != bits
            or int(receipt.get("start", -1)) != start
            or end <= start
            or receipt.get("receipt_sha256") != receipt_sha
            or receipt.get("chain_sha256") != next_chain
            or not isinstance(primary, list)
            or not isinstance(bridge, list)
        ):
            raise FormalSemanticBridgeError("semantic-bridge partial chain changed")
        expected = list(range(start, end))
        if [row.get("query_position") for row in primary] != expected or [
            row.get("query_position") for row in bridge
        ] != expected:
            raise FormalSemanticBridgeError("partial query coverage changed")
        primary_records.extend(primary)
        bridge_records.extend(bridge)
        descriptors.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "receipt_sha256": receipt_sha,
            }
        )
        start = end
        chain = next_chain
    return start, chain, primary_records, bridge_records, descriptors


def evaluate_formal_state(
    runtime_root: Path,
    plan_root: Path,
    output_parent: Path,
    *,
    distance_device: str = "cuda",
    max_new_chunks: int | None = None,
    _test_allow_synthetic: bool = False,
) -> Path:
    """Verify frozen bytes, then score all complete-gallery queries."""

    if max_new_chunks is not None and max_new_chunks < 0:
        raise ValueError("max_new_chunks must be nonnegative")
    plan, root = _verify_plan(plan_root)
    config_value = dict(plan["binding"]["config"])
    bridge_value = dict(config_value["bridge"])
    bridge_value["threshold_candidates"] = tuple(
        bridge_value["threshold_candidates"]
    )
    config_value["bridge"] = SemanticBridgeConfig(**bridge_value)
    config = FormalSemanticBridgeConfig(**config_value)
    rank = load_label_free_rank_inputs(
        runtime_root, _test_allow_synthetic=_test_allow_synthetic
    )
    arrays: list[np.ndarray] = []
    try:
        identity = _runtime_identity(rank, _runtime_manifest(runtime_root))
        if identity != plan["runtime_identity"]:
            raise FormalSemanticBridgeError("runtime identity differs from frozen plan")
        query_idx = np.asarray(rank.query_idx, dtype=np.int64).copy()
        database_idx = np.asarray(rank.database_idx, dtype=np.int64).copy()
        cache, arrays = _open_backend_cache(root, plan)
        backend = _PackedDistanceBackend(
            cache, query_idx, database_idx, device=distance_device
        )
    finally:
        rank.close()
        for value in arrays:
            _close_memmap(value)
    labels = load_frozen_metric_labels(
        runtime_root,
        rank_contract=plan,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    if labels.source_seal_sha256 != plan["source_seal_sha256"]:
        raise FormalSemanticBridgeError("metric labels differ from frozen source")
    if numeric_sha256(labels.query_row_ids) != plan["runtime_identity"][
        "query_row_ids_numeric_sha256"
    ] or numeric_sha256(labels.database_row_ids) != plan["runtime_identity"][
        "database_row_ids_numeric_sha256"
    ]:
        raise FormalSemanticBridgeError("metric row identities changed")
    output = reject_unsafe_output_path(
        Path(output_parent), field="semantic-bridge formal metric output"
    )
    output_root = output / f"metrics-{plan['rank_plan_sha256'][:16]}"
    output_root.mkdir(parents=True, exist_ok=True)
    prefixes = build_metric_prefixes(len(labels.database), config.cutoffs)
    database_label_counts = np.asarray(
        labels.database.sum(axis=1, dtype=np.uint16), dtype=np.uint16
    )
    headline = 50 if 50 in config.cutoffs else config.cutoffs[0]
    produced = 0
    completed = []
    for direction in config.directions:
        for bits in config.bits:
            start, chain, primary_records, bridge_records, partials = _resume_cell(
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
                semantic_distance = backend.distances(
                    "detail", direction, bits, chunk_start, chunk_end
                )
                composite = semantic_bridge_composite_distance(
                    primary_distance, semantic_distance, detail_bits=16
                )
                primary_chunk = []
                bridge_chunk = []
                for offset, query_position in enumerate(range(chunk_start, chunk_end)):
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
                    bridge_record = expected_tie_metrics_from_distances(
                        relevance,
                        composite[offset],
                        bits=bits,
                        distance_levels=17 * (bits + 1),
                        graded_gains=gains,
                        cutoffs=config.cutoffs,
                        prefixes=prefixes,
                        ideal_jaccard_dcg=ideal,
                    )
                    row_id = bytes(labels.query_row_ids[query_position]).decode("ascii")
                    primary_record.update(
                        {"query_position": query_position, "query_row_id": row_id}
                    )
                    bridge_record.update(
                        {"query_position": query_position, "query_row_id": row_id}
                    )
                    primary_chunk.append(primary_record)
                    bridge_chunk.append(bridge_record)
                receipt_body = {
                    "schema": PARTIAL_SCHEMA,
                    "status": "COMMITTED",
                    "rank_plan_sha256": plan["rank_plan_sha256"],
                    "direction": direction,
                    "bits": bits,
                    "detail_bits": 16,
                    "start": chunk_start,
                    "end": chunk_end,
                    "primary_distances_numeric_sha256": numeric_sha256(primary_distance),
                    "semantic_distances_numeric_sha256": numeric_sha256(semantic_distance),
                    "composite_distances_numeric_sha256": numeric_sha256(composite),
                    "primary_shell_invariance_checked": True,
                    "formal_gate_or_fallback_used": False,
                    "primary_records": primary_chunk,
                    "bridge_records": bridge_chunk,
                    "previous_chain_sha256": chain,
                }
                receipt_sha = sha256_json(receipt_body)
                chain = sha256_json(
                    {"previous_chain_sha256": chain, "receipt_sha256": receipt_sha}
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
                primary_records.extend(primary_chunk)
                bridge_records.extend(bridge_chunk)
                partials.append(
                    {
                        "path": target.relative_to(output_root).as_posix(),
                        "size": target.stat().st_size,
                        "sha256": sha256_file(target),
                        "receipt_sha256": receipt_sha,
                    }
                )
                produced += 1
            if len(primary_records) != len(labels.query) or len(bridge_records) != len(
                labels.query
            ):
                raise FormalSemanticBridgeError("completed cell lacks formal queries")
            summaries = {
                "primary_hamming": mean_query_metrics(primary_records),
                "shellguard_semantic_bridge": mean_query_metrics(bridge_records),
            }
            delta = {
                key: float(summaries["shellguard_semantic_bridge"][key])
                - float(summaries["primary_hamming"][key])
                for key, value in summaries["primary_hamming"].items()
                if isinstance(value, float)
            }
            result_body = {
                "schema": RESULT_SCHEMA,
                "status": "COMPLETE",
                "dataset": plan["dataset"],
                "source_seal_sha256": plan["source_seal_sha256"],
                "rank_plan_sha256": plan["rank_plan_sha256"],
                "direction": direction,
                "bits": bits,
                "detail_bits": 16,
                "selected_threshold": plan["binding"]["selected_threshold"],
                "ranking_rule": plan["ranking_rule"],
                "primary_shell_order_is_invariant": True,
                "summaries": summaries,
                "shellguard_minus_primary": delta,
                "per_query_receipts": partials,
                "final_receipt_chain_sha256": chain,
                "metric_labels_opened_after_verified_frozen_codes": True,
            }
            result = {
                **result_body,
                "metric_result_sha256": sha256_json(result_body),
            }
            target = output_root / "results" / _cell_id(direction, bits) / "metrics.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target, result)
            completed.append(
                {
                    "path": target.relative_to(output_root).as_posix(),
                    "size": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "metric_result_sha256": result["metric_result_sha256"],
                    "direction": direction,
                    "bits": bits,
                    "map_delta": delta["map_expected_ties"],
                    "binary_ndcg_headline_delta": delta[
                        f"binary_ndcg_at_{headline}_expected_ties"
                    ],
                    "j_ndcg_headline_delta": delta[
                        f"j_ndcg_at_{headline}_expected_ties"
                    ],
                }
            )
    primary_deltas = [
        value
        for row in completed
        for value in (row["map_delta"], row["binary_ndcg_headline_delta"])
    ]
    graded_deltas = [row["j_ndcg_headline_delta"] for row in completed]
    complete_body = {
        "schema": EVALUATION_SCHEMA,
        "status": "COMPLETE",
        "dataset": plan["dataset"],
        "source_seal_sha256": plan["source_seal_sha256"],
        "rank_plan_sha256": plan["rank_plan_sha256"],
        "selected_threshold": plan["binding"]["selected_threshold"],
        "results": completed,
        "headline_cutoff": headline,
        "primary_cells": len(primary_deltas),
        "graded_cells": len(graded_deltas),
        "negative_primary_cells": sum(value < 0.0 for value in primary_deltas),
        "nonpositive_graded_cells": sum(value <= 0.0 for value in graded_deltas),
        "minimum_primary_delta": min(primary_deltas),
        "mean_primary_delta": float(np.mean(primary_deltas, dtype=np.float64)),
        "minimum_graded_delta": min(graded_deltas),
        "mean_graded_delta": float(np.mean(graded_deltas, dtype=np.float64)),
        "formal_gate_or_fallback_used": False,
        "primary_shell_order_is_invariant": True,
        "storage_bounded_complete_gallery_evaluation": True,
    }
    atomic_write_json(
        output_root / "evaluation_complete.json",
        {**complete_body, "complete_sha256": sha256_json(complete_body)},
    )
    return output_root


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a nonempty integer list")
    return result


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a nonempty string list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--runtime", type=Path, required=True)
    freeze.add_argument("--base-plan", type=Path, required=True)
    freeze.add_argument("--detail-checkpoint", type=Path, required=True)
    freeze.add_argument("--architecture-freeze", type=Path, required=True)
    freeze.add_argument("--fit-artifact", type=Path, required=True)
    freeze.add_argument("--output-parent", type=Path, required=True)
    freeze.add_argument("--bits", type=_csv_ints, default=BITS)
    freeze.add_argument("--directions", type=_csv_strings, default=DIRECTIONS)
    freeze.add_argument("--cutoffs", type=_csv_ints, default=(50, 100, 1000))
    freeze.add_argument("--query-chunk-size", type=int, default=32)
    freeze.add_argument("--inference-batch-size", type=int, default=512)
    freeze.add_argument("--device", default="auto")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--runtime", type=Path, required=True)
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--output-parent", type=Path, required=True)
    evaluate.add_argument("--distance-device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--max-new-chunks", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        config = FormalSemanticBridgeConfig(
            bits=args.bits,
            directions=args.directions,
            cutoffs=args.cutoffs,
            query_chunk_size=args.query_chunk_size,
            inference_batch_size=args.inference_batch_size,
        )
        output = freeze_formal_state(
            args.runtime,
            args.base_plan,
            args.detail_checkpoint,
            args.architecture_freeze,
            args.fit_artifact,
            args.output_parent,
            config=config,
            device=args.device,
        )
    else:
        output = evaluate_formal_state(
            args.runtime,
            args.plan,
            args.output_parent,
            distance_device=args.distance_device,
            max_new_chunks=args.max_new_chunks,
        )
    print(json.dumps({"status": "OK", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
