"""Label-free full-dataset encoding and rank-state freezing."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from raw_rebuilt_neural.integrity import array_descriptor, atomic_save_npy
from raw_rebuilt_runtime.contract import atomic_write_json, numeric_sha256, sha256_json

from .adapters import enable_strict_determinism, encode_core
from .checkpoint import BaselineCheckpoint, load_checkpoint
from .contract import (
    BaselineBoundaryError,
    LabelFreeEncodingInputs,
    binding_from_label_free_inputs,
    reject_legacy_path,
    validate_label_free_inputs,
)


CODE_ARTIFACT_SCHEMA = "raw_rebuilt_fixed_feature_baseline_codes_v1"


@dataclass(frozen=True)
class EncodedCodes:
    image_codes: np.ndarray
    text_codes: np.ndarray
    row_ids: np.ndarray
    train_idx: np.ndarray
    query_idx: np.ndarray
    database_idx: np.ndarray
    rank_contract: Mapping[str, Any]


def _require_finite(features: np.ndarray, *, field: str, block_rows: int = 4096) -> None:
    for start in range(0, int(features.shape[0]), block_rows):
        if not np.all(np.isfinite(features[start : start + block_rows])):
            raise BaselineBoundaryError(f"{field} contains non-finite values")


def _verified_checkpoint(
    value: BaselineCheckpoint | Path,
) -> BaselineCheckpoint:
    # Re-open object inputs as well so a long-lived object cannot bypass a
    # later source/code receipt check.
    if isinstance(value, BaselineCheckpoint):
        return load_checkpoint(value.root)
    if isinstance(value, (str, Path)):
        return load_checkpoint(Path(value))
    raise TypeError("checkpoint must be BaselineCheckpoint or its directory")


def encode_label_free(
    checkpoint: BaselineCheckpoint | Path,
    inputs: LabelFreeEncodingInputs,
    *,
    batch_size: int = 1024,
    device: str | None = None,
) -> EncodedCodes:
    """Encode all rows without accepting a label array or label-bearing object."""

    validate_label_free_inputs(inputs)
    opened = _verified_checkpoint(checkpoint)
    observed = binding_from_label_free_inputs(
        inputs,
        fit_artifact_sha256=opened.fit_artifact_sha256,
        label_dim=opened.dataset_binding.label_dim,
    )
    if observed.to_dict() != opened.dataset_binding.to_dict():
        raise BaselineBoundaryError(
            "label-free encoding input does not match the checkpoint row/split seal"
        )
    _require_finite(np.asarray(inputs.image), field="rank image features")
    _require_finite(np.asarray(inputs.text), field="rank text features")
    enable_strict_determinism(opened.seed)
    requested = device if device is not None else str(opened.core_config["device"])
    image_codes, text_codes = encode_core(
        opened.method,
        opened.image_model,
        opened.text_model,
        np.asarray(inputs.image),
        np.asarray(inputs.text),
        batch_size=batch_size,
        device=requested,
    )
    expected_shape = (opened.dataset_binding.rows, opened.bits)
    if image_codes.shape != expected_shape or text_codes.shape != expected_shape:
        raise RuntimeError("baseline encoder returned unexpected code geometry")
    if not np.all(np.isin(image_codes, (-1, 1))) or not np.all(
        np.isin(text_codes, (-1, 1))
    ):
        raise RuntimeError("baseline encoder returned non-bipolar codes")
    rank_body = {
        "schema": CODE_ARTIFACT_SCHEMA,
        "status": "rank_state_frozen",
        "labels_loaded_during_freeze": False,
        "method": opened.method,
        "bits": opened.bits,
        "seed": opened.seed,
        "source_seal_sha256": opened.source_seal_sha256,
        "fit_artifact_sha256": opened.fit_artifact_sha256,
        "checkpoint_sha256": opened.checkpoint_sha256,
        "run_contract_sha256": opened.run_contract_sha256,
        "full_row_ids_numeric_sha256": observed.full_row_ids_numeric_sha256,
        "split_binding_sha256": observed.split_binding_sha256,
        "train_idx_numeric_sha256": observed.train_idx_numeric_sha256,
        "query_idx_numeric_sha256": observed.query_idx_numeric_sha256,
        "database_idx_numeric_sha256": observed.database_idx_numeric_sha256,
        "image_codes_numeric_sha256": numeric_sha256(image_codes),
        "text_codes_numeric_sha256": numeric_sha256(text_codes),
    }
    rank_contract = {**rank_body, "rank_contract_sha256": sha256_json(rank_body)}
    return EncodedCodes(
        image_codes=image_codes,
        text_codes=text_codes,
        row_ids=np.asarray(inputs.row_ids),
        train_idx=np.asarray(inputs.train_idx),
        query_idx=np.asarray(inputs.query_idx),
        database_idx=np.asarray(inputs.database_idx),
        rank_contract=rank_contract,
    )


def write_code_artifact(codes: EncodedCodes, output_parent: Path) -> Path:
    """Persist only codes/identity/splits; labels remain behind the metric gate."""

    if not isinstance(codes, EncodedCodes):
        raise TypeError("codes must be EncodedCodes")
    contract = dict(codes.rank_contract)
    if contract.get("status") != "rank_state_frozen" or contract.get(
        "labels_loaded_during_freeze"
    ) is not False:
        raise BaselineBoundaryError("only a label-free frozen rank state can be saved")
    output = reject_legacy_path(Path(output_parent), field="code output")
    if output.exists() and not output.is_dir():
        raise BaselineBoundaryError("code output parent must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    rank_sha = str(contract.get("rank_contract_sha256"))
    if rank_sha != sha256_json(
        {key: value for key, value in contract.items() if key != "rank_contract_sha256"}
    ):
        raise BaselineBoundaryError("rank contract seal differs")
    name = (
        f"{contract['method']}-b{contract['bits']}-s{contract['seed']}-"
        f"codes-{rank_sha[:16]}"
    )
    target = output / name
    arrays = {
        "image_codes": np.asarray(codes.image_codes, dtype=np.int8),
        "text_codes": np.asarray(codes.text_codes, dtype=np.int8),
        "row_ids": np.asarray(codes.row_ids, dtype="S64"),
        "indT": np.asarray(codes.train_idx, dtype=np.int64),
        "indQ": np.asarray(codes.query_idx, dtype=np.int64),
        "indD": np.asarray(codes.database_idx, dtype=np.int64),
    }
    filenames = {name: name + ".npy" for name in arrays}
    if target.exists():
        manifest = target / "manifest.json"
        if not manifest.is_file():
            raise BaselineBoundaryError("existing code artifact is incomplete")
        return target
    pending = output / ("." + name + f".pending-{os.getpid()}")
    if pending.exists():
        raise BaselineBoundaryError("code artifact output collision")
    pending.mkdir(parents=False, exist_ok=False)
    for key, value in arrays.items():
        atomic_save_npy(pending / filenames[key], value)
    descriptors = {
        key: array_descriptor(pending / filenames[key]) for key in arrays
    }
    atomic_write_json(
        pending / "manifest.json",
        {
            "schema": CODE_ARTIFACT_SCHEMA,
            "status": "rank_state_frozen",
            "labels_loaded_during_freeze": False,
            "rank_contract": contract,
            "arrays": descriptors,
        },
    )
    os.replace(pending, target)
    return target


__all__ = ["EncodedCodes", "encode_label_free", "write_code_artifact"]
