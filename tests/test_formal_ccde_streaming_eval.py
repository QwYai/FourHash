from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from raw_rebuilt_neural.ccde_detail_bits import select_detail_bits_from_fit
from raw_rebuilt_neural.ccde_training import train_detail_from_fit_artifact
from raw_rebuilt_neural.tests.test_neural_pipeline import (
    _fake_rank_inputs,
    _prepare_synthetic_fit,
    _small_config,
)
from raw_rebuilt_neural.training import train_from_fit_artifact
from raw_rebuilt_runtime.contract import load_json
from raw_rebuilt_runtime.loader import MetricLabels
from tools.formal_ccde_streaming_eval import (
    FormalCCDEConfig,
    evaluate_formal_state,
    freeze_formal_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE = (
    Path(os.environ["SHELLGUARD_CCDE_FREEZE"])
    if "SHELLGUARD_CCDE_FREEZE" in os.environ
    else PROJECT_ROOT
    / "output"
    / "audits"
    / "rz_csd_collision_detail_freeze_indt_20260824"
    / "freeze.json"
)


def test_storage_bounded_formal_state_is_label_free_then_exact(tmp_path: Path) -> None:
    if not FREEZE.is_file():
        pytest.skip("local registered CCDE freeze artifact is not present")
    fit = _prepare_synthetic_fit(tmp_path)
    config = _small_config(epochs=1, seed=37)
    primary_run = train_from_fit_artifact(
        fit,
        tmp_path / "primary",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    detail_run = train_detail_from_fit_artifact(
        fit,
        FREEZE,
        tmp_path / "detail",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    primary_checkpoint = primary_run / load_json(primary_run / "latest.json")["checkpoint"]
    detail_checkpoint = detail_run / load_json(detail_run / "latest.json")["checkpoint"]
    detail_bits = select_detail_bits_from_fit(
        fit,
        detail_checkpoint,
        FREEZE,
        tmp_path / "detail-bits",
        device="cpu",
        _test_allow_synthetic=True,
    )
    rank = _fake_rank_inputs()
    metadata = {
        "status": "COMPLETE",
        "dataset": "synthetic",
        "label_dim": 3,
        "source_seal_sha256": rank.source_seal_sha256,
    }
    with mock.patch(
        "tools.formal_ccde_streaming_eval.load_label_free_rank_inputs",
        return_value=rank,
    ), mock.patch(
        "tools.formal_ccde_streaming_eval._runtime_manifest", return_value=metadata
    ), mock.patch(
        "tools.formal_ccde_streaming_eval.load_metric_labels"
    ) as labels_not_opened:
        plan_root = freeze_formal_state(
            tmp_path / "runtime",
            primary_checkpoint,
            detail_checkpoint,
            detail_bits,
            FREEZE,
            tmp_path / "plan",
            config=FormalCCDEConfig(
                bits=(16,),
                directions=("i2t", "t2i"),
                cutoffs=(2,),
                query_chunk_size=1,
            ),
            device="cpu",
            _test_allow_synthetic=True,
        )
    labels_not_opened.assert_not_called()
    plan = load_json(plan_root / "evaluation_plan.json")
    assert plan["labels_loaded_during_freeze"] is False
    assert plan["distance_artifact_storage"].startswith("none")

    metric = MetricLabels(
        query=np.asarray([[1, 0, 1]], dtype=np.uint8),
        database=np.asarray(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.uint8
        ),
        query_row_ids=rank.row_ids[rank.query_idx],
        database_row_ids=rank.row_ids[rank.database_idx],
        source_seal_sha256=rank.source_seal_sha256,
    )
    with mock.patch(
        "tools.formal_ccde_streaming_eval.load_label_free_rank_inputs",
        return_value=rank,
    ), mock.patch(
        "tools.formal_ccde_streaming_eval._runtime_manifest", return_value=metadata
    ), mock.patch(
        "tools.formal_ccde_streaming_eval.load_metric_labels", return_value=metric
    ) as labels_opened:
        evaluation = evaluate_formal_state(
            tmp_path / "runtime",
            plan_root,
            primary_checkpoint,
            detail_checkpoint,
            detail_bits,
            FREEZE,
            tmp_path / "metrics",
            distance_device="cpu",
            _test_allow_synthetic=True,
        )
    labels_opened.assert_called_once()
    complete = load_json(evaluation / "evaluation_complete.json")
    assert complete["status"] == "COMPLETE"
    assert complete["primary_cells"] == 4
    assert complete["graded_cells"] == 2
    assert len(complete["results"]) == 2
    assert complete["formal_gate_or_fallback_used"] is False
