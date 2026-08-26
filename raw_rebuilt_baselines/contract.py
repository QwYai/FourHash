"""Fail-closed array boundaries for raw-rebuilt fixed-feature baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from raw_rebuilt_neural.fit_artifact import FIT_SCHEMA, FitArtifact, open_fit_artifact
from raw_rebuilt_runtime import LabelFreeRankInputs
from raw_rebuilt_runtime.contract import FEATURE_DIM, numeric_sha256, sha256_json


SUPPORTED_DATASETS = ("mirflickr", "nuswide", "mscoco")
LABEL_DIMS = {"mirflickr": 24, "nuswide": 21, "mscoco": 80}
SUPPORTED_BITS = (16, 32, 64)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SPLIT_BINDING_SCHEMA = "raw_rebuilt_baseline_split_binding_v1"


class BaselineBoundaryError(RuntimeError):
    """Raised when provenance or the train/rank temporal boundary is invalid."""


def reject_legacy_path(path: Path, *, field: str) -> Path:
    """Reject every legacy MAT/ProcessData/ids.mat ingress or output path."""

    candidate = Path(path).expanduser().resolve(strict=False)
    folded = [part.casefold() for part in candidate.parts]
    if candidate.suffix.casefold() == ".mat" or candidate.name.casefold() == "ids.mat":
        raise BaselineBoundaryError(f"{field} may not be a legacy MAT/ids.mat artifact")
    if "processdata" in folded or "oraldata" in folded:
        raise BaselineBoundaryError(
            f"{field} must be outside protected OralData and ProcessData trees"
        )
    return candidate


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if HEX64.fullmatch(text) is None:
        raise BaselineBoundaryError(f"{field} must be one lowercase SHA-256 digest")
    return text


def _require_row_ids(value: np.ndarray, *, field: str) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 1 or rows.dtype != np.dtype("S64"):
        raise BaselineBoundaryError(f"{field} must be a one-dimensional S64 array")
    if np.unique(rows).size != rows.size:
        raise BaselineBoundaryError(f"{field} contains duplicate canonical row IDs")
    for raw in rows:
        try:
            text = bytes(raw).decode("ascii")
        except UnicodeDecodeError as error:
            raise BaselineBoundaryError(f"{field} contains a non-ASCII row ID") from error
        if HEX64.fullmatch(text) is None:
            raise BaselineBoundaryError(
                f"{field} row IDs must be lowercase SHA-256 strings"
            )
    return rows


def _require_split(value: np.ndarray, *, field: str, rows: int) -> np.ndarray:
    indices = np.asarray(value)
    if indices.ndim != 1 or indices.dtype != np.dtype("int64"):
        raise BaselineBoundaryError(f"{field} must be a one-dimensional int64 array")
    if indices.size and (int(indices[0]) < 0 or int(indices[-1]) >= rows):
        raise BaselineBoundaryError(f"{field} contains an out-of-range row")
    if indices.size and (
        np.unique(indices).size != indices.size or np.any(indices[1:] <= indices[:-1])
    ):
        raise BaselineBoundaryError(f"{field} must be strictly increasing and unique")
    return indices


def _require_dataset(dataset: str, *, label_dim: int) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise BaselineBoundaryError(
            f"dataset must be one of {SUPPORTED_DATASETS}, observed {dataset!r}"
        )
    expected = LABEL_DIMS[dataset]
    if label_dim != expected:
        if dataset == "nuswide":
            raise BaselineBoundaryError(
                "NUS-WIDE baselines are restricted to the 21-label TC21 task; "
                "legacy 81-label inputs are forbidden"
            )
        raise BaselineBoundaryError(
            f"{dataset} labels must have width {expected}, observed {label_dim}"
        )


def open_verified_fit(value: FitArtifact | Path) -> tuple[FitArtifact, bool]:
    """Accept only the shared content-addressed ``indT`` fit interface.

    A path is always re-opened through ``open_fit_artifact``.  An object must
    be the exact shared ``FitArtifact`` class; naked arrays and runtime roots
    are deliberately not accepted by baseline trainers.
    """

    if isinstance(value, FitArtifact):
        artifact = value
        owned = False
    elif isinstance(value, (str, Path)):
        path = reject_legacy_path(Path(value), field="fit artifact")
        if path.suffix:
            raise BaselineBoundaryError("fit artifact input must be a directory")
        artifact = open_fit_artifact(path)
        owned = True
    else:
        raise TypeError(
            "fit input must be raw_rebuilt_neural.FitArtifact or its directory"
        )
    validate_fit_artifact(artifact)
    return artifact, owned


def validate_fit_artifact(fit: FitArtifact) -> None:
    if not isinstance(fit, FitArtifact):
        raise TypeError("expected raw_rebuilt_neural.FitArtifact")
    _require_sha256(fit.source_seal_sha256, field="fit source seal")
    _require_sha256(fit.fit_artifact_sha256, field="fit artifact seal")
    _require_dataset(str(fit.dataset), label_dim=int(fit.label_dim))
    image = np.asarray(fit.image)
    text = np.asarray(fit.text)
    labels = np.asarray(fit.labels)
    if image.ndim != 2 or image.dtype != np.float32 or image.shape[1] != FEATURE_DIM:
        raise BaselineBoundaryError("fit image features must be float32 [T,512]")
    if text.shape != image.shape or text.dtype != np.float32:
        raise BaselineBoundaryError("fit text features must match float32 [T,512]")
    if labels.shape != (image.shape[0], fit.label_dim) or labels.dtype != np.uint8:
        raise BaselineBoundaryError("fit labels must align with indT and dataset classes")
    if not np.all(np.isin(labels, (0, 1))) or np.any(labels.sum(axis=1) == 0):
        raise BaselineBoundaryError("fit labels must be nonempty binary multi-hot rows")
    _require_row_ids(np.asarray(fit.row_ids), field="fit row_ids")
    canonical = np.asarray(fit.canonical_indices)
    if canonical.shape != (image.shape[0],) or canonical.dtype != np.int64:
        raise BaselineBoundaryError("fit canonical_indices must be aligned int64 indT")
    if np.any(canonical[1:] <= canonical[:-1]):
        raise BaselineBoundaryError("fit canonical_indices must be strictly increasing")
    manifest = fit.manifest
    if not isinstance(manifest, Mapping) or manifest.get("schema") != FIT_SCHEMA:
        raise BaselineBoundaryError("fit object lacks a verified fit-artifact manifest")
    declared = manifest.get("split_indT_numeric_sha256")
    if declared != numeric_sha256(canonical):
        raise BaselineBoundaryError("fit indT split hash differs from its manifest")


@dataclass(frozen=True)
class LabelFreeEncodingInputs:
    """Full features and identity/split state with no label-bearing field."""

    dataset: str
    image: np.ndarray
    text: np.ndarray
    row_ids: np.ndarray
    train_idx: np.ndarray
    query_idx: np.ndarray
    database_idx: np.ndarray
    source_seal_sha256: str
    labels_loaded_during_freeze: bool = False


def label_free_inputs_from_runtime(
    dataset: str, value: LabelFreeRankInputs
) -> LabelFreeEncodingInputs:
    """Narrow the runtime's dedicated label-free loader to the baseline API."""

    if not isinstance(value, LabelFreeRankInputs):
        raise TypeError("encoding input must originate from LabelFreeRankInputs")
    result = LabelFreeEncodingInputs(
        dataset=str(dataset),
        image=value.image,
        text=value.text,
        row_ids=value.row_ids,
        train_idx=value.train_idx,
        query_idx=value.query_idx,
        database_idx=value.database_idx,
        source_seal_sha256=value.source_seal_sha256,
        labels_loaded_during_freeze=value.labels_loaded_during_freeze,
    )
    validate_label_free_inputs(result)
    return result


def validate_label_free_inputs(value: LabelFreeEncodingInputs) -> None:
    if not isinstance(value, LabelFreeEncodingInputs):
        raise TypeError("expected LabelFreeEncodingInputs")
    if value.labels_loaded_during_freeze is not False:
        raise BaselineBoundaryError("encoding is allowed only before metric labels open")
    _require_sha256(value.source_seal_sha256, field="rank source seal")
    # The dataset profile is checked without opening or accepting any labels.
    if value.dataset not in SUPPORTED_DATASETS:
        raise BaselineBoundaryError(f"unsupported rank dataset {value.dataset!r}")
    image = np.asarray(value.image)
    text = np.asarray(value.text)
    if image.ndim != 2 or image.dtype != np.float32 or image.shape[1] != FEATURE_DIM:
        raise BaselineBoundaryError("rank image features must be float32 [N,512]")
    if text.shape != image.shape or text.dtype != np.float32:
        raise BaselineBoundaryError("rank text features must match float32 [N,512]")
    rows = int(image.shape[0])
    row_ids = _require_row_ids(np.asarray(value.row_ids), field="rank row_ids")
    if row_ids.shape != (rows,):
        raise BaselineBoundaryError("rank row_ids do not align with full features")
    train = _require_split(value.train_idx, field="train_idx", rows=rows)
    query = _require_split(value.query_idx, field="query_idx", rows=rows)
    database = _require_split(value.database_idx, field="database_idx", rows=rows)
    if np.intersect1d(query, database, assume_unique=True).size:
        raise BaselineBoundaryError("query and database splits overlap")
    if not np.array_equal(
        np.sort(np.concatenate((query, database))), np.arange(rows, dtype=np.int64)
    ):
        raise BaselineBoundaryError("query union database must cover every canonical row")
    if np.setdiff1d(train, database, assume_unique=True).size:
        raise BaselineBoundaryError("indT must be a subset of the database")


@dataclass(frozen=True)
class DatasetBinding:
    schema: str
    dataset: str
    label_dim: int
    rows: int
    train_rows: int
    source_seal_sha256: str
    fit_artifact_sha256: str
    full_row_ids_numeric_sha256: str
    train_row_ids_numeric_sha256: str
    train_idx_numeric_sha256: str
    query_idx_numeric_sha256: str
    database_idx_numeric_sha256: str
    split_binding_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset_binding(
    fit: FitArtifact, rank: LabelFreeEncodingInputs
) -> DatasetBinding:
    """Bind one verified ``indT`` fit slice to one full label-free split."""

    validate_fit_artifact(fit)
    validate_label_free_inputs(rank)
    if fit.dataset != rank.dataset:
        raise BaselineBoundaryError("fit and rank datasets differ")
    if fit.source_seal_sha256 != rank.source_seal_sha256:
        raise BaselineBoundaryError("fit and rank source seals differ")
    train_idx = np.asarray(rank.train_idx)
    canonical = np.asarray(fit.canonical_indices)
    if not np.array_equal(train_idx, canonical):
        raise BaselineBoundaryError("fit canonical indices are not the frozen rank indT")
    rank_train_rows = np.asarray(rank.row_ids)[train_idx]
    if not np.array_equal(np.asarray(fit.row_ids), rank_train_rows):
        raise BaselineBoundaryError("fit row IDs are not rank row IDs at indT")
    component = {
        "schema": SPLIT_BINDING_SCHEMA,
        "dataset": fit.dataset,
        "source_seal_sha256": fit.source_seal_sha256,
        "fit_artifact_sha256": fit.fit_artifact_sha256,
        "full_row_ids_numeric_sha256": numeric_sha256(np.asarray(rank.row_ids)),
        "train_row_ids_numeric_sha256": numeric_sha256(rank_train_rows),
        "train_idx_numeric_sha256": numeric_sha256(train_idx),
        "query_idx_numeric_sha256": numeric_sha256(np.asarray(rank.query_idx)),
        "database_idx_numeric_sha256": numeric_sha256(np.asarray(rank.database_idx)),
    }
    declared_indt = fit.manifest.get("split_indT_numeric_sha256")
    if declared_indt != component["train_idx_numeric_sha256"]:
        raise BaselineBoundaryError("fit manifest and full rank indT hashes differ")
    split_sha = sha256_json(component)
    return DatasetBinding(
        schema=SPLIT_BINDING_SCHEMA,
        dataset=fit.dataset,
        label_dim=int(fit.label_dim),
        rows=int(np.asarray(rank.image).shape[0]),
        train_rows=int(np.asarray(fit.image).shape[0]),
        source_seal_sha256=fit.source_seal_sha256,
        fit_artifact_sha256=fit.fit_artifact_sha256,
        full_row_ids_numeric_sha256=component["full_row_ids_numeric_sha256"],
        train_row_ids_numeric_sha256=component["train_row_ids_numeric_sha256"],
        train_idx_numeric_sha256=component["train_idx_numeric_sha256"],
        query_idx_numeric_sha256=component["query_idx_numeric_sha256"],
        database_idx_numeric_sha256=component["database_idx_numeric_sha256"],
        split_binding_sha256=split_sha,
    )


def binding_from_label_free_inputs(
    rank: LabelFreeEncodingInputs,
    *,
    fit_artifact_sha256: str,
    label_dim: int,
) -> DatasetBinding:
    """Recompute the observable binding at encoding time without fit labels."""

    validate_label_free_inputs(rank)
    _require_dataset(rank.dataset, label_dim=label_dim)
    fit_sha = _require_sha256(fit_artifact_sha256, field="fit artifact seal")
    train_idx = np.asarray(rank.train_idx)
    train_rows = np.asarray(rank.row_ids)[train_idx]
    component = {
        "schema": SPLIT_BINDING_SCHEMA,
        "dataset": rank.dataset,
        "source_seal_sha256": rank.source_seal_sha256,
        "fit_artifact_sha256": fit_sha,
        "full_row_ids_numeric_sha256": numeric_sha256(np.asarray(rank.row_ids)),
        "train_row_ids_numeric_sha256": numeric_sha256(train_rows),
        "train_idx_numeric_sha256": numeric_sha256(train_idx),
        "query_idx_numeric_sha256": numeric_sha256(np.asarray(rank.query_idx)),
        "database_idx_numeric_sha256": numeric_sha256(np.asarray(rank.database_idx)),
    }
    return DatasetBinding(
        schema=SPLIT_BINDING_SCHEMA,
        dataset=rank.dataset,
        label_dim=int(label_dim),
        rows=int(np.asarray(rank.image).shape[0]),
        train_rows=int(train_idx.size),
        source_seal_sha256=rank.source_seal_sha256,
        fit_artifact_sha256=fit_sha,
        full_row_ids_numeric_sha256=component["full_row_ids_numeric_sha256"],
        train_row_ids_numeric_sha256=component["train_row_ids_numeric_sha256"],
        train_idx_numeric_sha256=component["train_idx_numeric_sha256"],
        query_idx_numeric_sha256=component["query_idx_numeric_sha256"],
        database_idx_numeric_sha256=component["database_idx_numeric_sha256"],
        split_binding_sha256=sha256_json(component),
    )


__all__ = [
    "BaselineBoundaryError",
    "DatasetBinding",
    "LABEL_DIMS",
    "LabelFreeEncodingInputs",
    "SUPPORTED_BITS",
    "SUPPORTED_DATASETS",
    "binding_from_label_free_inputs",
    "build_dataset_binding",
    "label_free_inputs_from_runtime",
    "open_verified_fit",
    "reject_legacy_path",
    "validate_fit_artifact",
    "validate_label_free_inputs",
]
