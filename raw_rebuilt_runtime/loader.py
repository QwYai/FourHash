"""Runner-facing, label-boundary-aware access to a verified runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import ARRAY_SPECS, FEATURE_DIM, RuntimeBridgeError
from .materialize import verify_label_free_runtime_directory, verify_runtime_directory


@dataclass(frozen=True)
class RuntimeDataset:
    root: Path
    dataset: str
    image: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    indQ: np.ndarray
    indT: np.ndarray
    indD: np.ndarray
    manifest: Mapping[str, Any]

    def close(self) -> None:
        for value in (
            self.image,
            self.text,
            self.labels,
            self.row_ids,
            self.indQ,
            self.indT,
            self.indD,
        ):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


@dataclass(frozen=True)
class LabelFreeRankInputs:
    image: np.ndarray
    text: np.ndarray
    train_idx: np.ndarray
    query_idx: np.ndarray
    database_idx: np.ndarray
    row_ids: np.ndarray
    source_seal_sha256: str
    labels_loaded_during_freeze: bool = False

    def close(self) -> None:
        for value in (
            self.image,
            self.text,
            self.train_idx,
            self.query_idx,
            self.database_idx,
            self.row_ids,
        ):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


@dataclass(frozen=True)
class TrainingInputs:
    image: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    identity_ids: np.ndarray
    row_ids: np.ndarray
    source_seal_sha256: str


@dataclass(frozen=True)
class MetricLabels:
    query: np.ndarray
    database: np.ndarray
    query_row_ids: np.ndarray
    database_row_ids: np.ndarray
    source_seal_sha256: str


def open_runtime_dataset(
    runtime_root: Path,
    *,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> RuntimeDataset:
    """Verify the runtime and its source, then expose read-only NPY memmaps."""

    root = Path(runtime_root).expanduser().resolve(strict=True)
    manifest = verify_runtime_directory(
        root,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    arrays = {
        name: np.load(root / relative, mmap_mode="r", allow_pickle=False)
        for name, (relative, _dtype) in ARRAY_SPECS.items()
    }
    return RuntimeDataset(
        root=root,
        dataset=str(manifest["dataset"]),
        image=arrays["image"],
        text=arrays["text"],
        labels=arrays["labels"],
        row_ids=arrays["row_ids"],
        indQ=arrays["indQ"],
        indT=arrays["indT"],
        indD=arrays["indD"],
        manifest=manifest,
    )


def load_label_free_rank_inputs(
    runtime_root: Path,
    *,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> LabelFreeRankInputs:
    """Return rank inputs without opening a label-bearing return boundary."""

    root = Path(runtime_root).expanduser().resolve(strict=True)
    manifest = verify_label_free_runtime_directory(
        root,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    arrays = {
        name: np.load(root / ARRAY_SPECS[name][0], mmap_mode="r", allow_pickle=False)
        for name in ("image", "text", "row_ids", "indQ", "indT", "indD")
    }
    return LabelFreeRankInputs(
        image=arrays["image"],
        text=arrays["text"],
        train_idx=arrays["indT"],
        query_idx=arrays["indQ"],
        database_idx=arrays["indD"],
        row_ids=arrays["row_ids"],
        source_seal_sha256=str(manifest["source_seal_sha256"]),
    )


def load_indt_training_inputs(
    runtime_root: Path,
    *,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> TrainingInputs:
    """Materialize only the frozen ``indT`` subset for model fitting."""

    dataset = open_runtime_dataset(
        runtime_root,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    take = np.asarray(dataset.indT, dtype=np.int64)
    image = np.ascontiguousarray(dataset.image[take], dtype=np.float32)
    text = np.ascontiguousarray(dataset.text[take], dtype=np.float32)
    labels = np.ascontiguousarray(dataset.labels[take], dtype=np.uint8)
    identity_ids = take.copy()
    row_ids = np.asarray(dataset.row_ids[take], dtype="S64")
    if image.ndim != 2 or image.shape[1] != FEATURE_DIM or text.shape != image.shape:
        raise RuntimeBridgeError("indT feature geometry changed after verification")
    if labels.shape[0] != image.shape[0] or row_ids.shape != (image.shape[0],):
        raise RuntimeBridgeError("indT labels/row IDs do not align with features")
    return TrainingInputs(
        image=image,
        text=text,
        labels=labels,
        identity_ids=identity_ids,
        row_ids=row_ids,
        source_seal_sha256=str(dataset.manifest["source_seal_sha256"]),
    )


def load_metric_labels(
    runtime_root: Path,
    *,
    rank_contract: Mapping[str, Any],
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> MetricLabels:
    """Open query/database labels only after a label-free rank freeze."""

    if rank_contract.get("status") != "rank_state_frozen" or rank_contract.get(
        "labels_loaded_during_freeze"
    ) is not False:
        raise RuntimeBridgeError(
            "metric labels require status=rank_state_frozen and labels_loaded_during_freeze=false"
        )
    dataset = open_runtime_dataset(
        runtime_root,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    seal = str(dataset.manifest["source_seal_sha256"])
    declared_seal = rank_contract.get("source_seal_sha256")
    if declared_seal is not None and declared_seal != seal:
        raise RuntimeBridgeError("rank contract was frozen against another raw-rebuilt source")
    query = np.asarray(dataset.indQ, dtype=np.int64)
    database = np.asarray(dataset.indD, dtype=np.int64)
    return MetricLabels(
        query=np.ascontiguousarray(dataset.labels[query], dtype=np.uint8),
        database=np.ascontiguousarray(dataset.labels[database], dtype=np.uint8),
        query_row_ids=np.asarray(dataset.row_ids[query], dtype="S64"),
        database_row_ids=np.asarray(dataset.row_ids[database], dtype="S64"),
        source_seal_sha256=seal,
    )
