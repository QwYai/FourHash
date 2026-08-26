from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from raw_rebuilt_runtime.contract import load_json
from raw_rebuilt_runtime.loader import MetricLabels

from raw_rebuilt_neural.ccde_contract import (
    CCDEContractError,
    CCDE_DETAIL_CAP,
    CCDE_DETAIL_VARIANT_NAME,
    CCDE_FREEZE_CONTENT_SHA256,
    load_ccde_freeze,
)
from raw_rebuilt_neural.ccde_training import (
    load_detail_checkpoint,
    train_detail_from_fit_artifact,
)
from raw_rebuilt_neural.ccde_detail_bits import rank_detail_bits, select_detail_bits_from_fit
from raw_rebuilt_neural.ccde_ranking import (
    CCDERankFreezeConfig,
    freeze_ccde_ranks,
    lexicographic_distance,
)
from raw_rebuilt_neural.metrics import evaluate_frozen_ranks, expected_graded_ndcg
from raw_rebuilt_neural.training import train_from_fit_artifact
from raw_rebuilt_neural.tests.test_neural_pipeline import (
    _fake_rank_inputs,
    _prepare_synthetic_fit,
    _small_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE = (
    Path(os.environ["SHELLGUARD_CCDE_FREEZE"])
    if "SHELLGUARD_CCDE_FREEZE" in os.environ
    else PROJECT_ROOT
    / "output"
    / "audits"
    / "rz_csd_collision_detail_freeze_indt_20260824"
    / "freeze.json"
)


@pytest.fixture
def registered_freeze() -> Path:
    if not FREEZE.is_file():
        pytest.skip("local registered CCDE freeze artifact is not present")
    return FREEZE


def test_registered_ccde_freeze_is_exact(registered_freeze: Path) -> None:
    freeze, file_sha = load_ccde_freeze(registered_freeze)
    assert len(file_sha) == 64
    assert freeze["freeze_sha256"] == CCDE_FREEZE_CONTENT_SHA256
    assert freeze["frozen_architecture"]["global_detail_cap"] == CCDE_DETAIL_CAP
    assert freeze["frozen_architecture"]["development_gate_or_fallback_used_at_formal_inference"] is False


def test_modified_freeze_is_rejected_before_training(
    tmp_path: Path, registered_freeze: Path
) -> None:
    changed = json.loads(registered_freeze.read_text(encoding="utf-8"))
    changed["frozen_architecture"]["global_detail_cap"] = 32
    target = tmp_path / "changed-freeze.json"
    target.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(CCDEContractError, match="freeze file bytes differ"):
        load_ccde_freeze(target)


def test_fit_only_coordinate_score_is_stable_and_label_dependent() -> None:
    image = torch.tensor(
        [[1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1]],
        dtype=torch.int8,
    ).numpy()
    text = image.copy()
    text[1, 2] *= -1
    labels = torch.tensor(
        [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.uint8
    ).numpy()
    first, statistics = rank_detail_bits(image, text, labels)
    second, _ = rank_detail_bits(image.copy(), text.copy(), labels.copy())
    assert first.tolist() == second.tolist()
    assert sorted(first.tolist()) == [0, 1, 2]
    assert set(statistics) == {"agreement", "balance", "label_separation", "score"}


def test_ccde_distance_never_crosses_primary_shells() -> None:
    primary_query = torch.tensor([1, 1, 1], dtype=torch.int8).numpy()
    primary_database = torch.tensor(
        [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]],
        dtype=torch.int8,
    ).numpy()
    detail_query = torch.tensor([1, 1], dtype=torch.int8).numpy()
    detail_database = torch.tensor(
        [[-1, -1], [-1, -1], [1, 1], [1, 1]], dtype=torch.int8
    ).numpy()
    composite, primary, detail = lexicographic_distance(
        primary_query, primary_database, detail_query, detail_database
    )
    order = composite.argsort(kind="stable")
    assert primary[order].tolist() == [0, 1, 1, 2]
    # The expert changes order only inside the two-item primary radius-1 shell.
    assert order.tolist() == [0, 2, 1, 3]
    assert detail.tolist() == [2, 2, 0, 0]


def test_expected_jaccard_ndcg_does_not_credit_storage_order_inside_ties() -> None:
    groups = torch.tensor([0, 0, 1], dtype=torch.int64).numpy()
    first = expected_graded_ndcg(
        torch.tensor([1.0, 0.0, 0.5]).numpy(), groups, cutoffs=(2,)
    )
    second = expected_graded_ndcg(
        torch.tensor([0.0, 1.0, 0.5]).numpy(), groups, cutoffs=(2,)
    )
    assert first == second


def test_ccde_detail_training_resume_and_loader_are_bound(
    tmp_path: Path, registered_freeze: Path
) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    config = _small_config(epochs=2, seed=29)
    full_root = train_detail_from_fit_artifact(
        fit,
        registered_freeze,
        tmp_path / "full-detail",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    resumed_root = train_detail_from_fit_artifact(
        fit,
        registered_freeze,
        tmp_path / "resumed-detail",
        config=config,
        device="cpu",
        max_epochs_this_call=1,
        _test_allow_synthetic=True,
    )
    assert not (resumed_root / "training_complete.json").exists()
    train_detail_from_fit_artifact(
        fit,
        registered_freeze,
        tmp_path / "resumed-detail",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    full_latest = load_json(full_root / "latest.json")
    resumed_latest = load_json(resumed_root / "latest.json")
    full = torch.load(full_root / full_latest["checkpoint"], map_location="cpu", weights_only=False)
    resumed = torch.load(
        resumed_root / resumed_latest["checkpoint"], map_location="cpu", weights_only=False
    )
    assert full["history"] == resumed["history"]
    for key, value in full["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][key]), key
    binding = full["binding"]
    assert binding["hash_head_variant"]["name"] == CCDE_DETAIL_VARIANT_NAME
    assert binding["architecture_freeze"]["freeze_sha256"] == CCDE_FREEZE_CONTENT_SHA256
    assert binding["training_information_boundary"]["formal_query_or_database_labels_opened"] is False
    loaded = load_detail_checkpoint(
        full_root / full_latest["checkpoint"],
        registered_freeze,
        device="cpu",
        expected_source_seal_sha256=binding["source_seal_sha256"],
    )
    assert loaded.model.hash_head_variant.name == CCDE_DETAIL_VARIANT_NAME
    assert inspect.signature(train_detail_from_fit_artifact).parameters.get("runtime_root") is None


def test_complete_ccde_process_keeps_labels_after_rank_freeze(
    tmp_path: Path, registered_freeze: Path
) -> None:
    fit = _prepare_synthetic_fit(tmp_path)
    config = _small_config(epochs=1, seed=31)
    primary_run = train_from_fit_artifact(
        fit,
        tmp_path / "primary",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    detail_run = train_detail_from_fit_artifact(
        fit,
        registered_freeze,
        tmp_path / "detail",
        config=config,
        device="cpu",
        _test_allow_synthetic=True,
    )
    primary_latest = load_json(primary_run / "latest.json")
    detail_latest = load_json(detail_run / "latest.json")
    primary_checkpoint = primary_run / primary_latest["checkpoint"]
    detail_checkpoint = detail_run / detail_latest["checkpoint"]
    detail_bits = select_detail_bits_from_fit(
        fit,
        detail_checkpoint,
        registered_freeze,
        tmp_path / "detail-bits",
        device="cpu",
        _test_allow_synthetic=True,
    )
    rank_inputs = _fake_rank_inputs()
    with mock.patch(
        "raw_rebuilt_neural.ccde_ranking.load_label_free_rank_inputs",
        return_value=rank_inputs,
    ):
        rank_root = freeze_ccde_ranks(
            tmp_path / "unopened-runtime",
            primary_checkpoint,
            detail_checkpoint,
            detail_bits,
            registered_freeze,
            tmp_path / "ranks",
            config=CCDERankFreezeConfig(
                bits=(16,), directions=("i2t", "t2i"), query_chunk_size=1
            ),
            device="cpu",
            _test_allow_synthetic=True,
        )
    rank_manifest = load_json(rank_root / "rank_manifest.json")
    assert rank_manifest["labels_loaded_during_freeze"] is False
    assert rank_manifest["primary_shell_order_is_invariant"] is True
    assert rank_manifest["formal_gate_or_fallback_used"] is False
    metric = MetricLabels(
        query=np.asarray([[1, 0, 1]], dtype=np.uint8),
        database=np.asarray(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.uint8
        ),
        query_row_ids=rank_inputs.row_ids[rank_inputs.query_idx],
        database_row_ids=rank_inputs.row_ids[rank_inputs.database_idx],
        source_seal_sha256=rank_inputs.source_seal_sha256,
    )
    with mock.patch(
        "raw_rebuilt_neural.metrics.load_metric_labels", return_value=metric
    ) as labels_opened:
        evaluation_root = evaluate_frozen_ranks(
            tmp_path / "metric-runtime",
            rank_root,
            tmp_path / "metrics",
            cutoffs=(2,),
            _test_allow_synthetic=True,
        )
    labels_opened.assert_called_once()
    completion = load_json(evaluation_root / "evaluation_complete.json")
    assert len(completion["results"]) == 2
    metric_path = evaluation_root / completion["results"][0]["path"]
    result = load_json(metric_path)
    assert "jndcg_at_2_expected_ties" in result["summary"]
