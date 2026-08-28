from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from raw_rebuilt_neural.fit_artifact import FIT_SCHEMA, FitArtifact
from raw_rebuilt_runtime.contract import numeric_sha256

from raw_rebuilt_baselines import (
    BaselineBoundaryError,
    BaselineRunConfig,
    LabelFreeEncodingInputs,
    build_dataset_binding,
    encode_label_free,
    load_checkpoint,
    train_baseline,
    write_code_artifact,
)
from raw_rebuilt_baselines.adapters import (
    enable_strict_determinism,
    make_core_config,
    owned_float32_input,
)
from encoders import raneh_feature


def _row_ids(count: int) -> np.ndarray:
    return np.asarray(
        [hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(count)],
        dtype="S64",
    )


def _fixture(seed: int = 7) -> tuple[FitArtifact, LabelFreeEncodingInputs]:
    generator = np.random.default_rng(seed)
    rows = 12
    train = np.arange(8, dtype=np.int64)
    query = np.asarray([8, 10], dtype=np.int64)
    database = np.asarray([*range(8), 9, 11], dtype=np.int64)
    image = generator.normal(size=(rows, 512)).astype(np.float32)
    text = (0.8 * image + 0.2 * generator.normal(size=image.shape)).astype(np.float32)
    labels = np.zeros((train.size, 24), dtype=np.uint8)
    # Every row and class is positive at least once, matching fit admission.
    for label in range(24):
        labels[label % train.size, label] = 1
    row_ids = _row_ids(rows)
    source_seal = hashlib.sha256(b"source-seal").hexdigest()
    fit_seal = hashlib.sha256(b"fit-seal").hexdigest()
    fit = FitArtifact(
        root=Path("verified-fit-object"),
        dataset="mirflickr",
        source_seal_sha256=source_seal,
        fit_artifact_sha256=fit_seal,
        label_dim=24,
        image=np.ascontiguousarray(image[train]),
        text=np.ascontiguousarray(text[train]),
        labels=labels,
        row_ids=np.ascontiguousarray(row_ids[train]),
        identity_ids=np.arange(train.size, dtype=np.uint64),
        canonical_indices=train,
        manifest={
            "schema": FIT_SCHEMA,
            "split_indT_numeric_sha256": numeric_sha256(train),
        },
    )
    rank = LabelFreeEncodingInputs(
        dataset="mirflickr",
        image=image,
        text=text,
        row_ids=row_ids,
        train_idx=train,
        query_idx=query,
        database_idx=database,
        source_seal_sha256=source_seal,
    )
    return fit, rank


def test_owned_float32_input_copies_read_only_contiguous_storage() -> None:
    source = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
    source.setflags(write=False)
    owned = owned_float32_input(source, field="fixture")
    assert owned.flags.owndata
    assert owned.flags.writeable
    assert owned.flags.c_contiguous
    assert not np.shares_memory(source, owned)
    owned[0, 0] = -1.0
    assert source[0, 0] == 0.0


def _fast_dcmh(seed: int = 20260822) -> BaselineRunConfig:
    return BaselineRunConfig(
        method="dcmh-f-seminit",
        bits=16,
        seed=seed,
        device="cpu",
        overrides={
            "epochs": 1,
            "batch_size": 4,
            "hidden_dim": 8,
            "lr": 0.01,
            "min_lr": 0.01,
            "gamma": 1.0,
            "eta": 0.01,
            "warmup_epochs": 1,
            "warmup_lr": 0.01,
        },
    )


def test_label_free_api_has_no_label_ingress() -> None:
    field_names = {item.name for item in fields(LabelFreeEncodingInputs)}
    assert "labels" not in field_names
    assert "labels" not in inspect.signature(encode_label_free).parameters
    fit, rank = _fixture()
    with pytest.raises(TypeError):
        LabelFreeEncodingInputs(**{**rank.__dict__, "labels": np.zeros((12, 24))})
    # The only label-bearing object is the indT-only fit artifact.
    assert fit.labels.shape[0] == fit.canonical_indices.size


def test_legacy_paths_and_naked_arrays_are_rejected(tmp_path: Path) -> None:
    _fit, rank = _fixture()
    config = _fast_dcmh()
    with pytest.raises(BaselineBoundaryError, match="MAT|ids.mat"):
        train_baseline(
            tmp_path / "ProcessData" / "ids.mat", rank, config, tmp_path / "runs"
        )
    with pytest.raises(TypeError, match="FitArtifact"):
        train_baseline(np.zeros((8, 512), dtype=np.float32), rank, config, tmp_path)
    with pytest.raises(BaselineBoundaryError, match="ProcessData"):
        fit, _rank = _fixture()
        train_baseline(fit, rank, config, tmp_path / "ProcessData" / "runs")


def test_nus_tc21_and_clip512_are_fail_closed() -> None:
    fit, rank = _fixture()
    nus81 = replace(
        fit,
        dataset="nuswide",
        label_dim=81,
        labels=np.ones((fit.image.shape[0], 81), dtype=np.uint8),
    )
    with pytest.raises(BaselineBoundaryError, match="TC21|21-label"):
        build_dataset_binding(nus81, replace(rank, dataset="nuswide"))
    broken_width = replace(fit, image=fit.image[:, :511], text=fit.text[:, :511])
    with pytest.raises(BaselineBoundaryError, match="512"):
        build_dataset_binding(broken_width, rank)


def test_bits_methods_and_override_names_are_strict() -> None:
    with pytest.raises(ValueError, match="bits"):
        BaselineRunConfig("ucch-f", 128, 1).validate()
    with pytest.raises(ValueError, match="method"):
        BaselineRunConfig("legacy-dcmh", 16, 1).validate()
    with pytest.raises(ValueError, match="unknown"):
        make_core_config(
            BaselineRunConfig("ucch-f", 16, 1, overrides={"secret_knob": 2})
        )
    with pytest.raises(ValueError, match="semantic"):
        make_core_config(
            BaselineRunConfig(
                "dcmh-f-seminit", 16, 1, overrides={"initialization": "random"}
            )
        )
    with pytest.raises(ValueError, match="dataset"):
        make_core_config(BaselineRunConfig("raneh-f", 16, 1))


def test_raneh_uses_audited_dataset_specific_author_settings() -> None:
    mir = make_core_config(
        BaselineRunConfig("raneh-f", 16, 20260822), dataset="mirflickr"
    )
    nus = make_core_config(
        BaselineRunConfig("raneh-f", 32, 20260822), dataset="nuswide"
    )
    coco = make_core_config(
        BaselineRunConfig("raneh-f", 64, 20260822), dataset="mscoco"
    )
    assert (mir.batch_size, mir.affinity_prune_k, mir.affinity_a1, mir.affinity_a2) == (
        1024,
        4700,
        0.4,
        0.7,
    )
    assert (nus.lr_image, nus.lr_text, nus.lambda_hash_similarity) == (
        0.0004,
        0.00175,
        1.0,
    )
    assert (coco.lr_joint, coco.lambda_hash_similarity, coco.affinity_a1) == (
        0.00175,
        10.0,
        0.3,
    )
    assert len(raneh_feature.OFFICIAL_SOURCE_SHA256) == 5
    assert len(raneh_feature.RECOVERED_MIRROR_SOURCE_SHA256) == 3


def test_binding_detects_train_or_query_identity_change() -> None:
    fit, rank = _fixture()
    build_dataset_binding(fit, rank)
    reordered_ids = rank.row_ids.copy()
    reordered_ids[[0, 1]] = reordered_ids[[1, 0]]
    with pytest.raises(BaselineBoundaryError, match="fit row IDs"):
        build_dataset_binding(fit, replace(rank, row_ids=reordered_ids))
    changed_query = np.asarray([9, 10], dtype=np.int64)
    changed_database = np.asarray([*range(9), 11], dtype=np.int64)
    changed = replace(rank, query_idx=changed_query, database_idx=changed_database)
    binding_a = build_dataset_binding(fit, rank)
    binding_b = build_dataset_binding(fit, changed)
    assert binding_a.split_binding_sha256 != binding_b.split_binding_sha256


def test_checkpoint_receipt_and_label_free_encoding_end_to_end(
    tmp_path: Path,
) -> None:
    fit, rank = _fixture()
    checkpoint_dir = train_baseline(
        fit, rank, _fast_dcmh(), tmp_path / "runs", verbose=False
    )
    checkpoint = load_checkpoint(checkpoint_dir)
    binding = build_dataset_binding(fit, rank)
    assert checkpoint.dataset_binding == binding
    receipt = json.loads((checkpoint_dir / "code_receipt.json").read_text("utf-8"))
    for key in (
        "source_seal_sha256",
        "fit_artifact_sha256",
        "full_row_ids_numeric_sha256",
        "train_row_ids_numeric_sha256",
        "split_binding_sha256",
        "train_idx_numeric_sha256",
        "query_idx_numeric_sha256",
        "database_idx_numeric_sha256",
    ):
        assert receipt[key] == getattr(binding, key)
    codes = encode_label_free(checkpoint_dir, rank, batch_size=4, device="cpu")
    assert codes.image_codes.shape == (12, 16)
    assert codes.text_codes.shape == (12, 16)
    assert set(np.unique(codes.image_codes)).issubset({-1, 1})
    assert codes.rank_contract["status"] == "rank_state_frozen"
    assert codes.rank_contract["labels_loaded_during_freeze"] is False
    code_dir = write_code_artifact(codes, tmp_path / "codes")
    code_manifest = json.loads((code_dir / "manifest.json").read_text("utf-8"))
    assert code_manifest["labels_loaded_during_freeze"] is False
    assert "labels" not in code_manifest["arrays"]


def test_seeded_cpu_runs_encode_identically(tmp_path: Path) -> None:
    fit, rank = _fixture(seed=33)
    config = _fast_dcmh(seed=20260823)
    first = train_baseline(fit, rank, config, tmp_path / "first", verbose=False)
    second = train_baseline(fit, rank, config, tmp_path / "second", verbose=False)
    codes_a = encode_label_free(first, rank, batch_size=4, device="cpu")
    codes_b = encode_label_free(second, rank, batch_size=4, device="cpu")
    np.testing.assert_array_equal(codes_a.image_codes, codes_b.image_codes)
    np.testing.assert_array_equal(codes_a.text_codes, codes_b.text_codes)
    assert codes_a.rank_contract["image_codes_numeric_sha256"] == codes_b.rank_contract[
        "image_codes_numeric_sha256"
    ]
    state = enable_strict_determinism(config.seed)
    assert state["torch_deterministic_algorithms"] is True
    assert state["cudnn_benchmark"] is False
    assert state["cuda_matmul_allow_tf32"] is False


def test_checkpoint_rejects_changed_qd_split_before_encoding(tmp_path: Path) -> None:
    fit, rank = _fixture()
    checkpoint = train_baseline(
        fit, rank, _fast_dcmh(), tmp_path / "run", verbose=False
    )
    changed = replace(
        rank,
        query_idx=np.asarray([9, 10], dtype=np.int64),
        database_idx=np.asarray([*range(9), 11], dtype=np.int64),
    )
    with pytest.raises(BaselineBoundaryError, match="row/split seal"):
        encode_label_free(checkpoint, changed, batch_size=4, device="cpu")


@pytest.mark.parametrize(
    ("method", "overrides"),
    [
        (
            "ucch-f",
            {
                "epochs": 1,
                "batch_size": 4,
                "image_layers": 1,
                "text_layers": 1,
                "hidden_width": 8,
                "negatives": 4,
                "memory_warmup_epochs": 0,
            },
        ),
        (
            "cirh-f",
            {
                "epochs": 1,
                "batch_size": 4,
                "graph_k": 2,
                "image_hidden_dim": 8,
            },
        ),
        (
            "raneh-f",
            {
                "epochs": 1,
                "batch_size": 4,
                "affinity_prune_k": 2,
                "train_limit": 6,
                "kan_hidden_dim": 8,
                "kan_num_grids": 2,
                "image_hidden_dim": 8,
                "text_hidden_dim": 8,
            },
        ),
    ],
)
def test_unsupervised_wrappers_call_real_train_and_encode_cores(
    tmp_path: Path, method: str, overrides: dict[str, object]
) -> None:
    fit, rank = _fixture(seed=51)
    checkpoint = train_baseline(
        fit,
        rank,
        BaselineRunConfig(
            method=method,
            bits=16,
            seed=20260824,
            device="cpu",
            overrides=overrides,
        ),
        tmp_path / method,
        verbose=False,
    )
    opened = load_checkpoint(checkpoint)
    codes = encode_label_free(opened, rank, batch_size=4, device="cpu")
    assert codes.image_codes.shape == (12, 16)
    assert codes.text_codes.shape == (12, 16)
