"""Fit-only coordinate selection for the frozen 16-bit CCDE detail code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from raw_rebuilt_runtime.contract import load_json, numeric_sha256, sha256_file, sha256_json
from rz_csd_clip512 import BITS, encode_clip512

from .ccde_contract import CCDE_DETAIL_CAP, freeze_binding
from .ccde_training import load_detail_checkpoint
from .fit_artifact import open_fit_artifact
from .integrity import array_descriptor, atomic_save_npy, atomic_write_json, production_code_inventory, reject_unsafe_output_path


DETAIL_BITS_SCHEMA = "raw_rebuilt_ccde_fit_only_detail_bits_v1"
DETAIL_BIT_STATISTICS = ("agreement", "balance", "label_separation", "score")


class DetailBitError(RuntimeError):
    """Raised when fit-only detail-coordinate provenance cannot be verified."""


@dataclass(frozen=True)
class DetailBitArtifact:
    root: Path
    dataset: str
    source_seal_sha256: str
    detail_checkpoint_sha256: str
    selected: Mapping[int, np.ndarray]
    order: Mapping[int, np.ndarray]
    manifest: Mapping[str, Any]

    def close(self) -> None:
        for value in (*self.selected.values(), *self.order.values()):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def rank_detail_bits(
    image_code: np.ndarray,
    text_code: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Frozen fit-only score used to order detail coordinates."""

    image = np.where(np.asarray(image_code) >= 0.0, 1.0, -1.0)
    text = np.where(np.asarray(text_code) >= 0.0, 1.0, -1.0)
    target = np.asarray(labels, dtype=np.float64)
    if image.ndim != 2 or text.shape != image.shape:
        raise ValueError("fit image/text codes must be aligned matrices")
    if target.ndim != 2 or target.shape[0] != image.shape[0]:
        raise ValueError("fit labels must align with the codes")
    if not np.all((target == 0.0) | (target == 1.0)):
        raise ValueError("fit labels must be binary")
    agreement = (image == text).mean(axis=0, dtype=np.float64)
    balance = 1.0 - np.abs(np.concatenate((image, text), axis=0).mean(axis=0))
    consensus = 0.5 * (image + text)
    positive_count = target.sum(axis=0)
    negative_count = len(target) - positive_count
    if np.any(positive_count <= 0.0) or np.any(negative_count <= 0.0):
        raise ValueError("every fit label must have positive and negative examples")
    positive_mean = (target.T @ consensus) / positive_count[:, None]
    negative_mean = ((1.0 - target).T @ consensus) / negative_count[:, None]
    prevalence = positive_count / len(target)
    separation = np.mean(
        prevalence[:, None]
        * (1.0 - prevalence[:, None])
        * np.square(positive_mean - negative_mean),
        axis=0,
    )
    score = agreement * balance * separation
    order = np.argsort(-score, kind="stable")
    return order.astype(np.int64), {
        "agreement": agreement,
        "balance": balance,
        "label_separation": separation,
        "score": score,
    }


def _array_filename(name: str) -> str:
    return f"{name}.npy"


def _declared_descriptor(name: str, value: np.ndarray) -> dict[str, Any]:
    return {
        "path": _array_filename(name),
        "dtype": np.dtype(value.dtype).str,
        "shape": list(value.shape),
        "numeric_sha256": numeric_sha256(value),
    }


def select_detail_bits_from_fit(
    fit_root: Path,
    detail_checkpoint_path: Path,
    architecture_freeze_path: Path,
    output_parent: Path,
    *,
    device: str | torch.device = "auto",
    _test_allow_synthetic: bool = False,
) -> Path:
    """Encode full indT and persist the frozen per-width coordinate order."""

    frozen = freeze_binding(architecture_freeze_path)
    fit = open_fit_artifact(fit_root, _test_allow_synthetic=_test_allow_synthetic)
    try:
        loaded = load_detail_checkpoint(
            detail_checkpoint_path,
            architecture_freeze_path,
            device=device,
            expected_source_seal_sha256=fit.source_seal_sha256,
        )
        if loaded.metadata["binding"]["fit_artifact_sha256"] != fit.fit_artifact_sha256:
            raise DetailBitError("detail checkpoint was not trained from this full-indT artifact")
        model_device = next(loaded.model.parameters()).device
        encoded_image = encode_clip512(
            loaded.model,
            fit.image,
            modality="image",
            device=model_device,
            batch_size=loaded.model.config.inference_batch_size,
        )
        encoded_text = encode_clip512(
            loaded.model,
            fit.text,
            modality="text",
            device=model_device,
            batch_size=loaded.model.config.inference_batch_size,
        )
        arrays: dict[str, np.ndarray] = {}
        for bits in BITS:
            order, statistics = rank_detail_bits(
                encoded_image.binary_codes[bits],
                encoded_text.binary_codes[bits],
                fit.labels,
            )
            detail_count = min(CCDE_DETAIL_CAP, bits)
            arrays[f"order_{bits}"] = np.ascontiguousarray(order, dtype=np.int64)
            arrays[f"selected_{bits}"] = np.ascontiguousarray(
                order[:detail_count], dtype=np.int64
            )
            for name in DETAIL_BIT_STATISTICS:
                arrays[f"{name}_{bits}"] = np.ascontiguousarray(
                    statistics[name], dtype=np.float64
                )
        del encoded_image, encoded_text
        declared = {
            name: _declared_descriptor(name, value) for name, value in arrays.items()
        }
        body = {
            "schema": DETAIL_BITS_SCHEMA,
            "status": "COMPLETE",
            "dataset": fit.dataset,
            "source_seal_sha256": fit.source_seal_sha256,
            "fit_artifact_sha256": fit.fit_artifact_sha256,
            "fit_split_indT_numeric_sha256": fit.manifest["split_indT_numeric_sha256"],
            "fit_rows": len(fit.image),
            "detail_checkpoint_sha256": loaded.checkpoint_sha256,
            "detail_checkpoint_run_binding_sha256": loaded.metadata["binding"][
                "run_binding_sha256"
            ],
            "architecture_freeze": frozen,
            "global_detail_cap": CCDE_DETAIL_CAP,
            "detail_bits_by_primary_width": {
                str(bits): min(CCDE_DETAIL_CAP, bits) for bits in BITS
            },
            "selection_rule": (
                "stable descending product of paired sign agreement, global bit "
                "balance, and prevalence-weighted multi-label mean separation"
            ),
            "labels_consumed": "full_indT_only",
            "formal_query_or_database_features_opened": False,
            "formal_query_or_database_labels_opened": False,
            "arrays": declared,
            "code_inventory": production_code_inventory(),
        }
        artifact_sha = sha256_json(body)
        manifest_without_inventory = {
            **body,
            "detail_bit_artifact_sha256": artifact_sha,
        }
        output = reject_unsafe_output_path(
            Path(output_parent), field="CCDE detail-bit output"
        )
        output.mkdir(parents=True, exist_ok=True)
        root = output / f"detail-bits-{artifact_sha[:16]}"
        if root.exists():
            opened = open_detail_bit_artifact(
                root,
                architecture_freeze_path,
                expected_source_seal_sha256=fit.source_seal_sha256,
                expected_checkpoint_sha256=loaded.checkpoint_sha256,
            )
            opened.close()
            return root
        root.mkdir(parents=False, exist_ok=False)
        for name, value in arrays.items():
            atomic_save_npy(root / _array_filename(name), value)
        file_inventory = {
            name: array_descriptor(root / _array_filename(name)) for name in arrays
        }
        atomic_write_json(
            root / "manifest.json",
            {**manifest_without_inventory, "file_inventory": file_inventory},
        )
        opened = open_detail_bit_artifact(
            root,
            architecture_freeze_path,
            expected_source_seal_sha256=fit.source_seal_sha256,
            expected_checkpoint_sha256=loaded.checkpoint_sha256,
        )
        opened.close()
        return root
    finally:
        fit.close()


def open_detail_bit_artifact(
    root: Path,
    architecture_freeze_path: Path,
    *,
    expected_source_seal_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    require_current_code: bool = True,
) -> DetailBitArtifact:
    """Verify every byte and semantic invariant of a fit-only bit artifact."""

    path = reject_unsafe_output_path(Path(root), field="CCDE detail-bit artifact").resolve(
        strict=True
    )
    manifest = load_json(path / "manifest.json")
    required = {
        "schema",
        "status",
        "dataset",
        "source_seal_sha256",
        "fit_artifact_sha256",
        "fit_split_indT_numeric_sha256",
        "fit_rows",
        "detail_checkpoint_sha256",
        "detail_checkpoint_run_binding_sha256",
        "architecture_freeze",
        "global_detail_cap",
        "detail_bits_by_primary_width",
        "selection_rule",
        "labels_consumed",
        "formal_query_or_database_features_opened",
        "formal_query_or_database_labels_opened",
        "arrays",
        "code_inventory",
        "detail_bit_artifact_sha256",
        "file_inventory",
    }
    if set(manifest) != required:
        raise DetailBitError("detail-bit manifest keys differ")
    if manifest.get("schema") != DETAIL_BITS_SCHEMA or manifest.get("status") != "COMPLETE":
        raise DetailBitError("detail-bit manifest schema/status differs")
    if manifest.get("architecture_freeze") != freeze_binding(architecture_freeze_path):
        raise DetailBitError("detail-bit artifact is bound to another architecture freeze")
    if manifest.get("global_detail_cap") != CCDE_DETAIL_CAP:
        raise DetailBitError("detail-bit cap differs from the freeze")
    if manifest.get("labels_consumed") != "full_indT_only":
        raise DetailBitError("detail-bit labels crossed the fit-only boundary")
    if manifest.get("formal_query_or_database_features_opened") is not False:
        raise DetailBitError("detail-bit selection opened formal query/database features")
    if manifest.get("formal_query_or_database_labels_opened") is not False:
        raise DetailBitError("detail-bit selection opened formal query/database labels")
    if expected_source_seal_sha256 is not None and manifest.get(
        "source_seal_sha256"
    ) != expected_source_seal_sha256:
        raise DetailBitError("detail-bit artifact belongs to another source seal")
    if expected_checkpoint_sha256 is not None and manifest.get(
        "detail_checkpoint_sha256"
    ) != expected_checkpoint_sha256:
        raise DetailBitError("detail-bit artifact belongs to another detail checkpoint")
    if require_current_code:
        current = production_code_inventory()["code_inventory_sha256"]
        if manifest.get("code_inventory", {}).get("code_inventory_sha256") != current:
            raise DetailBitError("current neural/runtime code differs from detail-bit code")
    arrays_meta = manifest.get("arrays")
    file_meta = manifest.get("file_inventory")
    expected_names = {
        *(f"order_{bits}" for bits in BITS),
        *(f"selected_{bits}" for bits in BITS),
        *(
            f"{statistic}_{bits}"
            for bits in BITS
            for statistic in DETAIL_BIT_STATISTICS
        ),
    }
    if not isinstance(arrays_meta, dict) or set(arrays_meta) != expected_names:
        raise DetailBitError("detail-bit declared array inventory differs")
    if not isinstance(file_meta, dict) or set(file_meta) != expected_names:
        raise DetailBitError("detail-bit file inventory differs")
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(expected_names):
        target = path / _array_filename(name)
        declared = arrays_meta[name]
        inventory = file_meta[name]
        if declared.get("path") != target.name or inventory.get("path") != target.name:
            raise DetailBitError(f"detail-bit {name} path changed")
        if sha256_file(target) != inventory.get("file_sha256"):
            raise DetailBitError(f"detail-bit {name} file hash changed")
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        if value.dtype.str != declared.get("dtype") or list(value.shape) != declared.get(
            "shape"
        ):
            raise DetailBitError(f"detail-bit {name} geometry changed")
        observed_numeric = numeric_sha256(value)
        if observed_numeric != declared.get("numeric_sha256") or observed_numeric != inventory.get(
            "numeric_sha256"
        ):
            raise DetailBitError(f"detail-bit {name} numeric content changed")
        arrays[name] = value
    selected: dict[int, np.ndarray] = {}
    order: dict[int, np.ndarray] = {}
    for bits in BITS:
        order_value = arrays[f"order_{bits}"]
        selected_value = arrays[f"selected_{bits}"]
        detail_count = min(CCDE_DETAIL_CAP, bits)
        if (
            order_value.dtype != np.int64
            or order_value.shape != (bits,)
            or not np.array_equal(np.sort(order_value), np.arange(bits, dtype=np.int64))
        ):
            raise DetailBitError(f"detail-bit order_{bits} is not a coordinate permutation")
        if (
            selected_value.dtype != np.int64
            or selected_value.shape != (detail_count,)
            or not np.array_equal(selected_value, order_value[:detail_count])
        ):
            raise DetailBitError(f"selected_{bits} is not the frozen order prefix")
        for statistic in DETAIL_BIT_STATISTICS:
            value = arrays[f"{statistic}_{bits}"]
            if value.dtype != np.float64 or value.shape != (bits,) or not np.isfinite(value).all():
                raise DetailBitError(f"detail-bit {statistic}_{bits} is invalid")
        selected[bits] = selected_value
        order[bits] = order_value
    content_body = {
        key: manifest[key]
        for key in required - {"detail_bit_artifact_sha256", "file_inventory"}
    }
    artifact_sha = sha256_json(content_body)
    if artifact_sha != manifest.get("detail_bit_artifact_sha256"):
        raise DetailBitError("detail-bit artifact content hash changed")
    if path.name != f"detail-bits-{artifact_sha[:16]}":
        raise DetailBitError("detail-bit artifact directory is not content addressed")
    return DetailBitArtifact(
        root=path,
        dataset=str(manifest["dataset"]),
        source_seal_sha256=str(manifest["source_seal_sha256"]),
        detail_checkpoint_sha256=str(manifest["detail_checkpoint_sha256"]),
        selected=selected,
        order=order,
        manifest=manifest,
    )


__all__ = [
    "DETAIL_BITS_SCHEMA",
    "DetailBitArtifact",
    "DetailBitError",
    "open_detail_bit_artifact",
    "rank_detail_bits",
    "select_detail_bits_from_fit",
]
