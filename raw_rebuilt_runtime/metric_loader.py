"""Bounded-memory post-freeze label loader for streaming evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import ARRAY_SPECS, RuntimeBridgeError
from .loader import MetricLabels
from .materialize import verify_metric_runtime_directory


def _close_memmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def load_frozen_metric_labels(
    runtime_root: Path,
    *,
    rank_contract: Mapping[str, Any],
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> MetricLabels:
    """Open only Q/D labels after a plan has frozen the admitted source seal.

    The baseline training loader remains byte-for-byte part of each checkpoint
    receipt.  This scoring-only loader is instead inventoried with the
    streaming evaluator and uses the bounded frozen-source verifier.
    """

    if rank_contract.get("status") != "rank_state_frozen" or rank_contract.get(
        "labels_loaded_during_freeze"
    ) is not False:
        raise RuntimeBridgeError(
            "metric labels require status=rank_state_frozen and "
            "labels_loaded_during_freeze=false"
        )
    root = Path(runtime_root).expanduser().resolve(strict=True)
    manifest = verify_metric_runtime_directory(
        root,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    seal = str(manifest["source_seal_sha256"])
    declared_seal = rank_contract.get("source_seal_sha256")
    if declared_seal is not None and declared_seal != seal:
        raise RuntimeBridgeError(
            "rank contract was frozen against another raw-rebuilt source"
        )

    arrays = {
        name: np.load(
            root / ARRAY_SPECS[name][0], mmap_mode="r", allow_pickle=False
        )
        for name in ("labels", "row_ids", "indQ", "indD")
    }
    try:
        query = np.asarray(arrays["indQ"], dtype=np.int64)
        database = np.asarray(arrays["indD"], dtype=np.int64)
        result = MetricLabels(
            query=np.ascontiguousarray(arrays["labels"][query], dtype=np.uint8),
            database=np.ascontiguousarray(
                arrays["labels"][database], dtype=np.uint8
            ),
            query_row_ids=np.asarray(arrays["row_ids"][query], dtype="S64"),
            database_row_ids=np.asarray(
                arrays["row_ids"][database], dtype="S64"
            ),
            source_seal_sha256=seal,
        )
    finally:
        for value in arrays.values():
            _close_memmap(value)
    return result


__all__ = ["load_frozen_metric_labels"]
