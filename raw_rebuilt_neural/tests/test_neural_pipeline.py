from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from raw_rebuilt_runtime.loader import LabelFreeRankInputs, MetricLabels
from raw_rebuilt_runtime.contract import load_json, numeric_sha256

from raw_rebuilt_neural.fit_artifact import (
    FitArtifactError,
    identity_ids_from_row_ids,
    open_fit_artifact,
    prepare_fit_artifact,
)
from raw_rebuilt_neural.metrics import evaluate_frozen_ranks, expected_tie_metrics
from raw_rebuilt_neural.ranking import RankFreezeConfig, RankingError, freeze_ranks
from raw_rebuilt_neural.training import (
    FROZEN_CODE_BCE_WEIGHT,
    FROZEN_FINE_TUNE_EPOCHS,
    FROZEN_FINE_TUNE_LEARNING_RATE,
    FROZEN_GRADED_WEIGHT,
    FROZEN_POSTERIOR_JACCARD_WEIGHT,
    FROZEN_WARMUP_EPOCHS,
    NeuralTrainConfig,
    TrainingError,
    load_trained_checkpoint,
    train_from_fit_artifact,
)


SOURCE_SEAL = "a" * 64


def _row_ids(count: int) -> np.ndarray:
    import hashlib

    return np.asarray(
        [hashlib.sha256(f"row-{index}".encode("ascii")).hexdigest() for index in range(count)],
        dtype="S64",
    )


def _prepare_synthetic_fit(parent: Path) -> Path:
    rng = np.random.default_rng(41)
    row_ids = _row_ids(2)
    training = SimpleNamespace(
        image=rng.normal(size=(2, 512)).astype(np.float32),
        text=rng.normal(size=(2, 512)).astype(np.float32),
        # Every synthetic class has both a positive and a negative fit row so
        # the fit-only CCDE separation score is defined in end-to-end tests.
        labels=np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        identity_ids=np.asarray([1, 3], dtype=np.int64),
        row_ids=row_ids,
        source_seal_sha256=SOURCE_SEAL,
    )
    metadata = {
        "dataset": "synthetic",
        "label_dim": 3,
        "source_seal_sha256": SOURCE_SEAL,
        "runtime_manifest_sha256": "b" * 64,
    }
    with mock.patch(
        "raw_rebuilt_neural.fit_artifact._runtime_metadata", return_value=metadata
    ), mock.patch(
        "raw_rebuilt_neural.fit_artifact.load_indt_training_inputs",
        return_value=training,
    ):
        return prepare_fit_artifact(
            parent / "unused-runtime",
            parent / "fit-output",
            _test_allow_synthetic=True,
        )


def _small_config(*, epochs: int = 2, seed: int = 17) -> NeuralTrainConfig:
    return NeuralTrainConfig(
        seed=seed,
        epochs=epochs,
        warmup_epochs=max(0, epochs - 1),
        auxiliary_ramp_epochs=1,
        batch_size=2,
        hidden_dim=8,
        feedforward_dim=16,
        residual_layers=1,
        posterior_hidden_dim=4,
        posterior_heads=3,
        dropout=0.0,
        inference_batch_size=2,
        checkpoint_every=1,
    )


def _train_small(fit: Path, output: Path, *, epochs: int = 2) -> Path:
    run = train_from_fit_artifact(
        fit,
        output,
        config=_small_config(epochs=epochs),
        device="cpu",
        _test_allow_synthetic=True,
    )
    latest = load_json(run / "latest.json")
    return run / latest["checkpoint"]


def _fake_rank_inputs() -> LabelFreeRankInputs:
    rng = np.random.default_rng(7)
    image = rng.normal(size=(4, 512)).astype(np.float32)
    text = rng.normal(size=(4, 512)).astype(np.float32)
    return LabelFreeRankInputs(
        image=image,
        text=text,
        train_idx=np.asarray([1, 2], dtype=np.int64),
        query_idx=np.asarray([0], dtype=np.int64),
        database_idx=np.asarray([1, 2, 3], dtype=np.int64),
        row_ids=_row_ids(4),
        source_seal_sha256=SOURCE_SEAL,
    )


def test_fit_artifact_is_indt_only_and_content_addressed(tmp_path: Path) -> None:
    fit_root = _prepare_synthetic_fit(tmp_path)
    fit = open_fit_artifact(fit_root, _test_allow_synthetic=True)
    assert fit.image.shape == (2, 512)
    assert fit.text.shape == (2, 512)
    assert fit.labels.shape == (2, 3)
    assert fit.identity_ids.dtype == np.uint64
    assert np.unique(fit.identity_ids).size == 2
    assert fit_root.name == f"fit-{fit.fit_artifact_sha256[:16]}"
    assert fit.manifest["split_indT_numeric_sha256"] == numeric_sha256(
        fit.canonical_indices
    )
    assert set(path.name for path in fit_root.glob("*.npy")) == {
        "image.npy",
        "text.npy",
        "labels.npy",
        "row_ids.npy",
        "identity_ids.npy",
        "canonical_indices.npy",
    }
    fit.close()
    assert "runtime_root" not in inspect.signature(train_from_fit_artifact).parameters


def test_identity_digest_is_row_bound_and_nus_tc21_is_enforced(tmp_path: Path) -> None:
    rows = _row_ids(3)
    first = identity_ids_from_row_ids(rows)
    second = identity_ids_from_row_ids(rows.copy())
    assert np.array_equal(first, second)
    changed = rows.copy()
    changed[0] = _row_ids(4)[3]
    assert first[0] != identity_ids_from_row_ids(changed)[0]
    from raw_rebuilt_neural.fit_artifact import _require_dataset_geometry

    with pytest.raises(FitArtifactError, match="TC21"):
        _require_dataset_geometry("nuswide", 21_000, 81)


def test_epoch_resume_is_bit_exact(tmp_path: Path) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    config = _small_config(epochs=3, seed=23)
    full_run = train_from_fit_artifact(
        fit,
        tmp_path / "full",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    interrupted_run = train_from_fit_artifact(
        fit,
        tmp_path / "resumed",
        config=config,
        device="cpu",
        max_epochs_this_call=1,
        _test_allow_synthetic=True,
    )
    assert not (interrupted_run / "training_complete.json").exists()
    resumed_run = train_from_fit_artifact(
        fit,
        tmp_path / "resumed",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    full_latest = load_json(full_run / "latest.json")
    resumed_latest = load_json(resumed_run / "latest.json")
    full = torch.load(full_run / full_latest["checkpoint"], map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_run / resumed_latest["checkpoint"], map_location="cpu", weights_only=False)
    assert full["history"] == resumed["history"]
    for key, value in full["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][key]), key
    for key, value in full["auxiliary_decoder_state_dict"].items():
        assert torch.equal(value, resumed["auxiliary_decoder_state_dict"][key]), key


def test_frozen_curriculum_matches_indt_selection_record() -> None:
    config = NeuralTrainConfig()
    assert config.epochs == FROZEN_WARMUP_EPOCHS + FROZEN_FINE_TUNE_EPOCHS
    assert config.warmup_epochs == FROZEN_WARMUP_EPOCHS
    assert config.auxiliary_ramp_epochs == FROZEN_FINE_TUNE_EPOCHS
    assert config.fine_tune_learning_rate == FROZEN_FINE_TUNE_LEARNING_RATE
    assert config.code_bce_weight == FROZEN_CODE_BCE_WEIGHT
    assert config.graded_weight == FROZEN_GRADED_WEIGHT
    assert config.posterior_jaccard_weight == FROZEN_POSTERIOR_JACCARD_WEIGHT


def test_training_checkpoint_records_auxiliary_curriculum(tmp_path: Path) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    checkpoint = _train_small(fit, tmp_path / "train", epochs=2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "auxiliary_decoder_state_dict" in state
    assert state["history"][0]["auxiliary_scale"] == 0.0
    assert state["history"][1]["auxiliary_scale"] == 1.0
    assert state["history"][1]["learning_rate"] == FROZEN_FINE_TUNE_LEARNING_RATE
    assert state["history"][1]["code_bce"] > 0.0
    assert state["history"][1]["graded"] >= 0.0


def test_checkpoint_rejects_another_source_seal(tmp_path: Path) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    checkpoint = _train_small(fit, tmp_path / "train", epochs=1)
    with pytest.raises(TrainingError, match="another raw-rebuilt source"):
        load_trained_checkpoint(
            checkpoint,
            device="cpu",
            expected_source_seal_sha256="c" * 64,
        )


def test_expected_tie_ap_does_not_credit_storage_order() -> None:
    tied = np.asarray([0, 0], dtype=np.uint16)
    first = expected_tie_metrics(np.asarray([1, 0]), tied, cutoffs=(1, 2))
    second = expected_tie_metrics(np.asarray([0, 1]), tied, cutoffs=(1, 2))
    assert first["map_expected_ties"] == pytest.approx(0.75)
    assert second["map_expected_ties"] == pytest.approx(0.75)
    assert first["precision_at_1_expected_ties"] == pytest.approx(0.5)
    assert first["map_canonical_storage"] != second["map_canonical_storage"]


def test_rank_freeze_and_metric_are_separate_label_boundaries(tmp_path: Path) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    checkpoint = _train_small(fit, tmp_path / "train", epochs=1)
    rank_inputs = _fake_rank_inputs()
    with mock.patch(
        "raw_rebuilt_neural.ranking.load_label_free_rank_inputs",
        return_value=rank_inputs,
    ):
        rank_root = freeze_ranks(
            tmp_path / "unopened-runtime",
            checkpoint,
            tmp_path / "rank-output",
            config=RankFreezeConfig(
                bits=(16,),
                directions=("i2t", "t2i"),
                modes=("hamming",),
                query_chunk_size=1,
                semantic_window=1,
                max_active_candidates=3,
            ),
            device="cpu",
            _test_allow_synthetic=True,
        )
    manifest = load_json(rank_root / "rank_manifest.json")
    assert manifest["status"] == "rank_state_frozen"
    assert manifest["labels_loaded_during_freeze"] is False
    assert len(manifest["cells"]) == 2
    metric = MetricLabels(
        query=np.asarray([[1, 0, 0]], dtype=np.uint8),
        database=np.asarray(
            [[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.uint8
        ),
        query_row_ids=rank_inputs.row_ids[rank_inputs.query_idx],
        database_row_ids=rank_inputs.row_ids[rank_inputs.database_idx],
        source_seal_sha256=SOURCE_SEAL,
    )
    with mock.patch(
        "raw_rebuilt_neural.metrics.load_metric_labels", return_value=metric
    ) as opened:
        evaluation = evaluate_frozen_ranks(
            tmp_path / "metric-runtime",
            rank_root,
            tmp_path / "metric-output",
            cutoffs=(1, 3),
            _test_allow_synthetic=True,
        )
    opened.assert_called_once()
    completion = load_json(evaluation / "evaluation_complete.json")
    assert completion["status"] == "COMPLETE"
    assert len(completion["results"]) == 2
    assert all(0.0 <= result["map_expected_ties"] <= 1.0 for result in completion["results"])


def test_rank_receipt_poison_is_rejected_before_metrics(tmp_path: Path) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    checkpoint = _train_small(fit, tmp_path / "train", epochs=1)
    rank_inputs = _fake_rank_inputs()
    with mock.patch(
        "raw_rebuilt_neural.ranking.load_label_free_rank_inputs",
        return_value=rank_inputs,
    ):
        rank_root = freeze_ranks(
            tmp_path / "runtime",
            checkpoint,
            tmp_path / "ranks",
            config=RankFreezeConfig(
                bits=(16,),
                directions=("i2t",),
                modes=("hamming",),
                query_chunk_size=1,
                semantic_window=1,
                max_active_candidates=3,
            ),
            device="cpu",
            _test_allow_synthetic=True,
        )
    order_path = next(rank_root.glob("cells/i2t/bits-16/hamming/orders.npy"))
    order = np.load(order_path, mmap_mode="r+", allow_pickle=False)
    order[0, 0], order[0, 1] = order[0, 1], order[0, 0]
    order.flush()
    order._mmap.close()
    metric = MetricLabels(
        query=np.asarray([[1, 0, 0]], dtype=np.uint8),
        database=np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.uint8),
        query_row_ids=rank_inputs.row_ids[rank_inputs.query_idx],
        database_row_ids=rank_inputs.row_ids[rank_inputs.database_idx],
        source_seal_sha256=SOURCE_SEAL,
    )
    with mock.patch(
        "raw_rebuilt_neural.metrics.load_metric_labels", return_value=metric
    ) as metric_open:
        with pytest.raises(RankingError, match="committed rank order chunk changed"):
            evaluate_frozen_ranks(
                tmp_path / "runtime",
                rank_root,
                tmp_path / "metrics",
                _test_allow_synthetic=True,
            )
    metric_open.assert_not_called()


def test_protected_output_path_is_rejected(tmp_path: Path) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        train_from_fit_artifact(
            fit,
            tmp_path / "ProcessData" / "run",
            config=_small_config(epochs=1),
            device="cpu",
            _test_allow_synthetic=True,
        )
