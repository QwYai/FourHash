from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from raw_rebuilt_runtime.contract import atomic_write_json, numeric_sha256, sha256_file, sha256_json
from raw_rebuilt_runtime.loader import LabelFreeRankInputs, MetricLabels

from raw_rebuilt_streaming.codes import (
    BITS,
    CODE_STATE_SCHEMA,
    _array_relative,
    _write_encoding_receipt,
    open_code_state,
    pack_bipolar_codes,
    safe_encode_feature_chunk,
)
from raw_rebuilt_streaming.baseline_import import import_baseline_code_artifact
from raw_rebuilt_streaming.baseline_import import _checkpoint_snapshot
from raw_rebuilt_streaming.integrity import (
    StreamingIntegrityError,
    array_descriptor,
    production_code_inventory,
    require_dataset_label_geometry,
)
from raw_rebuilt_streaming.metric_worker import consume_metric_bundles
from raw_rebuilt_streaming.metrics import (
    MetricPrefixes,
    _StableEstimate,
    _canonical_unit_interval,
    _precompute_jaccard_idcg,
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
)
from raw_rebuilt_streaming.metrics import mean_query_metrics
from raw_rebuilt_streaming.orchestrator import (
    PROJECT_ROOT,
    _start_worker,
    _verify_evaluation_after_workers,
)
from raw_rebuilt_streaming.plan import StreamingPlanConfig, freeze_rank_plan, open_rank_plan
from raw_rebuilt_streaming.protocol import ack_path, bundle_path
from raw_rebuilt_streaming.rank_worker import produce_rank_bundles
from raw_rebuilt_streaming.rank_worker import hamming_distance_chunk


SOURCE_SEAL = "a" * 64


def test_worker_process_runs_from_project_root() -> None:
    with mock.patch("raw_rebuilt_streaming.orchestrator.subprocess.Popen") as popen:
        _start_worker(["python", "-m", "raw_rebuilt_streaming"])
    popen.assert_called_once_with(
        ["python", "-m", "raw_rebuilt_streaming"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )


def _row_ids(count: int) -> np.ndarray:
    import hashlib

    return np.asarray(
        [hashlib.sha256(f"stream-row-{index}".encode("ascii")).hexdigest() for index in range(count)],
        dtype="S64",
    )


def _rank_inputs() -> LabelFreeRankInputs:
    rows = 6
    return LabelFreeRankInputs(
        image=np.zeros((rows, 512), dtype=np.float32),
        text=np.zeros((rows, 512), dtype=np.float32),
        train_idx=np.asarray([2, 3], dtype=np.int64),
        query_idx=np.asarray([0, 1], dtype=np.int64),
        database_idx=np.asarray([2, 3, 4, 5], dtype=np.int64),
        row_ids=_row_ids(rows),
        source_seal_sha256=SOURCE_SEAL,
    )


def _metric_labels(rank: LabelFreeRankInputs) -> MetricLabels:
    query = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    database = np.asarray(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8
    )
    return MetricLabels(
        query=query,
        database=database,
        query_row_ids=rank.row_ids[rank.query_idx],
        database_row_ids=rank.row_ids[rank.database_idx],
        source_seal_sha256=SOURCE_SEAL,
    )


def test_metrics_accept_frozen_ccde_composite_distance_levels() -> None:
    relevance = np.asarray([True, False, True], dtype=bool)
    gains = np.asarray([1.0, 0.0, 0.5], dtype=np.float64)
    composite = np.asarray([0, 17, 288], dtype=np.uint16)
    result = expected_tie_metrics_from_distances(
        relevance,
        composite,
        bits=16,
        distance_levels=(16 + 1) * (16 + 1),
        graded_gains=gains,
        cutoffs=(3,),
    )
    assert result["average_precision_expected_ties"] == pytest.approx(5.0 / 6.0)
    assert 0.0 <= result["j_ndcg_at_3_expected_ties"] <= 1.0
    with pytest.raises(ValueError, match="outside"):
        expected_tie_metrics_from_distances(
            relevance,
            composite,
            bits=16,
            graded_gains=gains,
            cutoffs=(3,),
        )


def _make_code_state(parent: Path, rank: LabelFreeRankInputs) -> Path:
    root = parent / "code-state"
    (root / "codes").mkdir(parents=True)
    rng = np.random.default_rng(47)
    arrays = {}
    scope_indices = {"query": rank.query_idx, "database": rank.database_idx}
    for scope, indices in scope_indices.items():
        for modality in ("image", "text"):
            for bits in BITS:
                bipolar = np.where(rng.integers(0, 2, size=(len(indices), bits)), 1, -1).astype(
                    np.int8
                )
                packed = pack_bipolar_codes(bipolar, bits)
                path = root / _array_relative(scope, modality, bits)
                with path.open("wb") as handle:
                    np.save(handle, packed, allow_pickle=False)
                arrays[(scope, modality, bits)] = np.load(path, mmap_mode="r+", allow_pickle=False)
    runtime_body = {
        "dataset": "synthetic",
        "rows": len(rank.row_ids),
        "label_dim": 3,
        "source_seal_sha256": SOURCE_SEAL,
        "row_ids_numeric_sha256": numeric_sha256(rank.row_ids),
        "query_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.query_idx]),
        "database_row_ids_numeric_sha256": numeric_sha256(rank.row_ids[rank.database_idx]),
        "indQ_numeric_sha256": numeric_sha256(rank.query_idx),
        "indT_numeric_sha256": numeric_sha256(rank.train_idx),
        "indD_numeric_sha256": numeric_sha256(rank.database_idx),
        "query_rows": len(rank.query_idx),
        "train_rows": len(rank.train_idx),
        "database_rows": len(rank.database_idx),
    }
    runtime = {**runtime_body, "runtime_identity_sha256": sha256_json(runtime_body)}
    binding_body = {
        "schema": CODE_STATE_SCHEMA,
        "producer_type": "neural_v1_checkpoint",
        "runtime": runtime,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_run_binding_sha256": "c" * 64,
        "checkpoint_v1_code_inventory_sha256": "d" * 64,
        "config": {"bits": list(BITS), "feature_chunk_size": len(rank.row_ids)},
        "streaming_code_inventory": production_code_inventory(),
        "labels_loaded_during_encoding": False,
    }
    binding = {**binding_body, "encoding_binding_sha256": sha256_json(binding_body)}
    for scope, indices in scope_indices.items():
        for modality in ("image", "text"):
            _write_encoding_receipt(
                root,
                scope=scope,
                modality=modality,
                start=0,
                end=len(indices),
                arrays=arrays,
                binding_sha256=binding["encoding_binding_sha256"],
                previous_chain="0" * 64,
            )
    descriptors = {}
    for scope in ("query", "database"):
        for modality in ("image", "text"):
            for bits in BITS:
                value = arrays[(scope, modality, bits)]
                value.flush()
                path = root / _array_relative(scope, modality, bits)
                descriptors[f"{scope}_{modality}_{bits}"] = {
                    "path": _array_relative(scope, modality, bits),
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "size": path.stat().st_size,
                    "file_sha256": sha256_file(path),
                    "numeric_sha256": numeric_sha256(value),
                }
                value._mmap.close()
    receipts = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((root / "receipts").glob("*.json"))
    ]
    body = {
        "schema": CODE_STATE_SCHEMA,
        "status": "code_state_frozen",
        "dataset": "synthetic",
        "rows": len(rank.row_ids),
        "label_dim": 3,
        "source_seal_sha256": SOURCE_SEAL,
        "runtime_identity": runtime,
        "binding": binding,
        "available_bits": list(BITS),
        "arrays": descriptors,
        "receipts": receipts,
        "stored_state": "packed_query_database_binary_codes_only",
        "labels_loaded_during_encoding": False,
    }
    atomic_write_json(root / "manifest.json", {**body, "manifest_sha256": sha256_json(body)})
    return root


def _runtime_stub(parent: Path) -> Path:
    root = parent / "runtime"
    root.mkdir()
    (root / "runtime_manifest.json").write_text(
        json.dumps({"dataset": "synthetic", "label_dim": 3}), encoding="utf-8"
    )
    return root


def _plan(parent: Path, state: Path) -> Path:
    return freeze_rank_plan(
        state,
        parent / "plans",
        config=StreamingPlanConfig(
            bits=(16,),
            directions=("i2t", "t2i"),
            query_chunk_size=1,
            cutoffs=(1, 2, 4),
        ),
    )


def _ordered_metrics(relevance: np.ndarray, gains: np.ndarray, order: tuple[int, ...], cutoffs):
    rel = relevance[list(order)]
    gain = gains[list(order)]
    cumulative = np.cumsum(rel)
    positions = np.flatnonzero(rel)
    ap = float(np.mean(cumulative[positions] / (positions + 1))) if len(positions) else 0.0
    result = {"average_precision_expected_ties": ap}
    discount = 1.0 / np.log2(np.arange(1, len(rel) + 1) + 1.0)
    total = int(rel.sum())
    for cutoff in cutoffs:
        effective = min(cutoff, len(rel))
        take_rel = int(rel[:effective].sum())
        binary_idcg = float(discount[: min(total, effective)].sum())
        ideal_gain = np.sort(gains)[::-1][:effective]
        j_idcg = float(np.dot(ideal_gain, discount[:effective]))
        result[f"precision_at_{cutoff}_expected_ties"] = take_rel / effective
        result[f"recall_at_{cutoff}_expected_ties"] = take_rel / total if total else 0.0
        result[f"binary_ndcg_at_{cutoff}_expected_ties"] = (
            float(np.dot(rel[:effective], discount[:effective])) / binary_idcg if binary_idcg else 0.0
        )
        result[f"j_ndcg_at_{cutoff}_expected_ties"] = (
            float(np.dot(gain[:effective], discount[:effective])) / j_idcg if j_idcg else 0.0
        )
    return result


def test_safe_encoding_makes_writable_contiguous_copy() -> None:
    original = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
    original.setflags(write=False)

    def fake_encoder(model, value, *, modality, device, batch_size):
        assert value.flags.writeable
        assert value.flags.c_contiguous
        assert not np.shares_memory(value, original)
        return SimpleNamespace(
            binary_codes={bits: np.ones((len(value), bits), dtype=np.int8) for bits in BITS}
        )

    encoded = safe_encode_feature_chunk(
        None,
        original,
        modality="image",
        device=torch.device("cpu"),
        batch_size=2,
        encoder=fake_encoder,
    )
    assert set(encoded.binary_codes) == set(BITS)
    assert not original.flags.writeable


def test_expected_ties_matches_exhaustive_slow_reference() -> None:
    distances = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    gains = np.asarray([1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    relevance = gains > 0
    cutoffs = (1, 2, 4)
    exact = expected_tie_metrics_from_distances(
        relevance, distances, bits=16, graded_gains=gains, cutoffs=cutoffs
    )
    references = []
    for first, second in itertools.product(itertools.permutations((0, 1)), itertools.permutations((2, 3))):
        references.append(_ordered_metrics(relevance, gains, first + second, cutoffs))
    for key in references[0]:
        assert exact[key] == pytest.approx(np.mean([record[key] for record in references]))


def test_metric_boundary_canonicalizes_roundoff_but_rejects_real_overflow() -> None:
    # These shell boundaries reproduce the production MIR failure exactly:
    # summing prefix differences uses a different binary64 addition order than
    # the IDCG prefix and lands one ULP above one before canonicalization.
    distances = np.repeat(
        np.arange(3, dtype=np.uint8), np.asarray([4, 38, 8], dtype=np.int64)
    )
    relevance = np.ones(50, dtype=bool)
    gains = np.ones(50, dtype=np.float64)
    prefixes = build_metric_prefixes(50, (50,))
    raw_dcg = sum(
        float(prefixes.discount[end] - prefixes.discount[start])
        for start, end in ((0, 4), (4, 42), (42, 50))
    )
    raw_ratio = raw_dcg / float(prefixes.discount[50])
    one_ulp_above_one = float(np.nextafter(1.0, np.inf))
    assert raw_ratio in (1.0, one_ulp_above_one)

    # CPython/NumPy builds may sum this MIR construction to exactly one.
    # Exercise the overflow branch directly so every platform still proves
    # that a certified single-ULP overshoot is canonicalized, not rejected.
    assert (
        _canonical_unit_interval(
            one_ulp_above_one,
            field="forced-single-ulp-roundoff",
            stable_value=lambda: _StableEstimate(
                value=Decimal(1),
                absolute_error=Decimal(0),
                binary64_error=Decimal.from_float(one_ulp_above_one) - Decimal(1),
            ),
        )
        == 1.0
    )

    exact = expected_tie_metrics_from_distances(
        relevance,
        distances,
        bits=16,
        graded_gains=gains,
        cutoffs=(50,),
        prefixes=prefixes,
    )
    slow = _ordered_metrics(relevance, gains, tuple(range(50)), (50,))
    for key in (
        "average_precision_expected_ties",
        "precision_at_50_expected_ties",
        "recall_at_50_expected_ties",
        "binary_ndcg_at_50_expected_ties",
        "j_ndcg_at_50_expected_ties",
    ):
        assert exact[key] == 1.0
        assert abs(float(exact[key]) - float(slow[key])) <= 8.0 * np.finfo(np.float64).eps

    with pytest.raises(
        FloatingPointError,
        match="precomputed Jaccard IDCG differs|stable recomputation",
    ):
        expected_tie_metrics_from_distances(
            relevance,
            distances,
            bits=16,
            graded_gains=gains,
            cutoffs=(50,),
            prefixes=prefixes,
            ideal_jaccard_dcg={50: float(prefixes.discount[50]) / 2.0},
        )


def test_nus_scale_stable_j_ndcg_fallback_with_production_idcg() -> None:
    database_rows = 60_000
    shell_sizes = np.full(65, database_rows // 65, dtype=np.int64)
    shell_sizes[: database_rows - int(shell_sizes.sum())] += 1
    distances = np.repeat(np.arange(65, dtype=np.uint8), shell_sizes)
    shell_gains = np.concatenate(
        (
            np.arange(21, 0, -1, dtype=np.float64) / 21.0,
            np.full(44, 1.0 / 21.0, dtype=np.float64),
        )
    )
    gains = np.repeat(shell_gains, shell_sizes)
    relevance = gains > 0.0
    prefixes = build_metric_prefixes(database_rows, (database_rows,))
    direct_discounts = tuple(
        1.0 / math.log2(rank + 2.0) for rank in range(database_rows)
    )
    ideal_gains = np.sort(gains)[::-1]
    ideal = {
        database_rows: math.fsum(
            float(gain) * discount
            for gain, discount in zip(ideal_gains, direct_discounts)
        )
    }

    # This is the real production fast path: bincount gain accumulation and
    # prefix subtraction exceed one by many ULPs at this scale.
    gain_sums = np.bincount(
        distances.astype(np.int64), weights=gains, minlength=65
    )
    start = 0
    raw_dcg = 0.0
    for size_raw, gain_sum in zip(shell_sizes, gain_sums):
        size = int(size_raw)
        raw_dcg += (float(gain_sum) / size) * float(
            prefixes.discount[start + size] - prefixes.discount[start]
        )
        start += size
    raw_ratio = raw_dcg / ideal[database_rows]
    assert raw_ratio > 1.0 + 8.0 * np.finfo(np.float64).eps

    exact = expected_tie_metrics_from_distances(
        relevance,
        distances,
        bits=64,
        graded_gains=gains,
        cutoffs=(database_rows,),
        prefixes=prefixes,
        ideal_jaccard_dcg=ideal,
    )
    assert exact[f"j_ndcg_at_{database_rows}_expected_ties"] == 1.0


def test_stable_fallback_covers_ap_and_recall_boundary_roundoff() -> None:
    all_relevant = np.ones(12, dtype=bool)
    independent_shells = np.arange(12, dtype=np.uint8)
    ap = expected_tie_metrics_from_distances(
        all_relevant,
        independent_shells,
        bits=16,
        graded_gains=np.ones(12, dtype=np.float64),
        cutoffs=(12,),
    )
    assert ap["average_precision_expected_ties"] == 1.0

    relevance = np.zeros(25, dtype=bool)
    relevance[:7] = True
    recall = expected_tie_metrics_from_distances(
        relevance,
        np.zeros(25, dtype=np.uint8),
        bits=16,
        graded_gains=relevance.astype(np.float64),
        cutoffs=(25,),
    )
    assert recall["recall_at_25_expected_ties"] == 1.0


def test_stable_boundary_rejects_negative_and_nonfinite() -> None:
    with pytest.raises(FloatingPointError, match="stable recomputation"):
        _canonical_unit_interval(
            -0.01,
            field="negative-test-score",
            stable_value=lambda: Decimal("-0.01"),
        )
    with pytest.raises(FloatingPointError, match="not finite"):
        _canonical_unit_interval(
            float("nan"),
            field="nan-test-score",
            stable_value=lambda: Decimal(0),
        )
    for fast_score in (1.0 + 1.0e-12, 2.0):
        with pytest.raises(FloatingPointError, match="fast/stable difference"):
            _canonical_unit_interval(
                fast_score,
                field="gross-fast-error",
                stable_value=lambda: _StableEstimate(
                    value=Decimal(1),
                    absolute_error=Decimal("1e-70"),
                    binary64_error=Decimal("1e-14"),
                ),
            )


def test_wrong_large_idcg_and_malformed_fast_prefix_are_rejected() -> None:
    with pytest.raises(FloatingPointError, match="differs from stable direct IDCG"):
        expected_tie_metrics_from_distances(
            np.asarray([True]),
            np.asarray([0], dtype=np.uint8),
            bits=16,
            graded_gains=np.asarray([1.0]),
            cutoffs=(1,),
            ideal_jaccard_dcg={1: 2.0},
        )

    relevance = np.ones(12, dtype=bool)
    prefixes = build_metric_prefixes(12, (12,))
    attacked = replace(
        prefixes,
        harmonic=np.asarray(prefixes.harmonic) * 2.0,
    )
    with pytest.raises(ValueError, match="prefix certificate"):
        expected_tie_metrics_from_distances(
            relevance,
            np.arange(12, dtype=np.uint8),
            bits=16,
            graded_gains=np.ones(12, dtype=np.float64),
            cutoffs=(12,),
            prefixes=attacked,
        )


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test_metric_prefix_certificate_rejects_poison_and_is_immutable(
    scale: float,
) -> None:
    prefix = build_metric_prefixes(12, (12,))
    assert not prefix.harmonic.flags.writeable
    assert not prefix.discount.flags.writeable
    with pytest.raises(ValueError):
        prefix.harmonic.setflags(write=True)
    with pytest.raises(ValueError):
        prefix.discount.setflags(write=True)

    poisoned_harmonic = np.asarray(prefix.harmonic) * scale
    rebound = replace(prefix, harmonic=poisoned_harmonic)
    with pytest.raises(ValueError, match="prefix certificate"):
        expected_tie_metrics_from_distances(
            np.ones(12, dtype=bool),
            np.arange(12, dtype=np.uint8),
            bits=16,
            graded_gains=np.ones(12, dtype=np.float64),
            cutoffs=(12,),
            prefixes=rebound,
        )
    forged_identity = replace(
        prefix,
        harmonic=np.frombuffer(
            poisoned_harmonic.astype(np.float64).tobytes(), dtype=np.float64
        ),
    )
    forged_identity = replace(
        forged_identity,
        _harmonic_identity=id(forged_identity.harmonic),
    )
    with pytest.raises(ValueError, match="prefix certificate"):
        expected_tie_metrics_from_distances(
            np.ones(12, dtype=bool),
            np.arange(12, dtype=np.uint8),
            bits=16,
            graded_gains=np.ones(12, dtype=np.float64),
            cutoffs=(12,),
            prefixes=forged_identity,
        )

    manual = MetricPrefixes(
        database_rows=12,
        max_cutoff=12,
        harmonic=np.asarray(prefix.harmonic),
        discount=np.asarray(prefix.discount),
        _harmonic_identity=id(prefix.harmonic),
        _discount_identity=id(prefix.discount),
        _token=object(),
    )
    with pytest.raises(ValueError, match="prefix certificate"):
        expected_tie_metrics_from_distances(
            np.ones(12, dtype=bool),
            np.arange(12, dtype=np.uint8),
            bits=16,
            graded_gains=np.ones(12, dtype=np.float64),
            cutoffs=(12,),
            prefixes=manual,
        )


@pytest.mark.parametrize("field,scale", [("harmonic", 0.5), ("discount", 0.0)])
def test_metric_prefix_certificate_rejects_original_dict_rebinding(
    field: str,
    scale: float,
) -> None:
    prefix = build_metric_prefixes(12, (12,))
    poisoned = np.frombuffer(
        (np.asarray(getattr(prefix, field)) * scale).astype(np.float64).tobytes(),
        dtype=np.float64,
    )
    prefix.__dict__[field] = poisoned
    prefix.__dict__[f"_{field}_identity"] = id(poisoned)
    with pytest.raises(ValueError, match="prefix certificate"):
        expected_tie_metrics_from_distances(
            np.ones(12, dtype=bool),
            np.arange(12, dtype=np.uint8),
            bits=16,
            graded_gains=np.ones(12, dtype=np.float64),
            cutoffs=(12,),
            prefixes=prefix,
        )


def test_metric_prefix_certificate_rejects_original_dict_geometry_rebinding() -> None:
    prefix = build_metric_prefixes(12, (12,))
    prefix.__dict__["database_rows"] = 13
    with pytest.raises(ValueError, match="prefix certificate"):
        expected_tie_metrics_from_distances(
            np.ones(12, dtype=bool),
            np.arange(12, dtype=np.uint8),
            bits=16,
            graded_gains=np.ones(12, dtype=np.float64),
            cutoffs=(12,),
            prefixes=prefix,
        )


def test_verified_jaccard_gains_are_immutable_and_identity_bound() -> None:
    certificate = _precompute_jaccard_idcg(
        np.asarray([1.0, 0.5, 0.25], dtype=np.float64), (3,)
    )
    assert not certificate.gains.flags.writeable
    with pytest.raises(ValueError):
        certificate.gains.setflags(write=True)
    with pytest.raises(ValueError, match="binding differs"):
        expected_tie_metrics_from_distances(
            certificate.gains > 0.0,
            np.asarray([0, 1, 2], dtype=np.uint8),
            bits=16,
            graded_gains=certificate.gains.copy(),
            cutoffs=(3,),
            ideal_jaccard_dcg=certificate,
        )
    forged = replace(certificate, values=((3, 2.0),))
    with pytest.raises(ValueError, match="binding differs"):
        expected_tie_metrics_from_distances(
            certificate.gains > 0.0,
            np.asarray([0, 1, 2], dtype=np.uint8),
            bits=16,
            graded_gains=certificate.gains,
            cutoffs=(3,),
            ideal_jaccard_dcg=forged,
        )


def test_jaccard_idcg_reuses_one_maximum_prefix_without_changing_values() -> None:
    gains = np.random.default_rng(20260824).random(4096, dtype=np.float64)
    cutoffs = (50, 100, 1000)
    certificate = _precompute_jaccard_idcg(gains, cutoffs)
    ordered = np.sort(gains)[::-1]
    expected = {
        cutoff: math.fsum(
            float(gain) / math.log2(rank + 2.0)
            for rank, gain in enumerate(ordered[:cutoff])
        )
        for cutoff in cutoffs
    }
    assert certificate.as_dict() == expected


@pytest.mark.parametrize("poison", ["gains", "values"])
def test_verified_jaccard_certificate_rejects_original_dict_rebinding(
    poison: str,
) -> None:
    certificate = _precompute_jaccard_idcg(
        np.asarray([1.0, 0.5, 0.25], dtype=np.float64), (3,)
    )
    original_gains = certificate.gains
    if poison == "gains":
        certificate.__dict__["gains"] = np.frombuffer(
            (original_gains * 0.5).astype(np.float64).tobytes(), dtype=np.float64
        )
    elif poison == "values":
        certificate.__dict__["values"] = ((3, 2.0),)
    with pytest.raises(ValueError, match="binding differs"):
        expected_tie_metrics_from_distances(
            original_gains > 0.0,
            np.asarray([0, 1, 2], dtype=np.uint8),
            bits=16,
            graded_gains=original_gains,
            cutoffs=(3,),
            ideal_jaccard_dcg=certificate,
        )


def test_verified_jaccard_certificate_ignores_instance_method_shadow() -> None:
    certificate = _precompute_jaccard_idcg(
        np.asarray([1.0, 0.5, 0.25], dtype=np.float64), (3,)
    )
    certificate.__dict__["as_dict"] = lambda: {3: 2.0}
    result = expected_tie_metrics_from_distances(
        certificate.gains > 0.0,
        np.asarray([0, 1, 2], dtype=np.uint8),
        bits=16,
        graded_gains=certificate.gains,
        cutoffs=(3,),
        ideal_jaccard_dcg=certificate,
    )
    assert result["j_ndcg_at_3_expected_ties"] == pytest.approx(1.0)


def test_stable_fallback_does_not_depend_on_extended_numpy_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "longdouble", np.float64)
    distances = np.repeat(
        np.arange(3, dtype=np.uint8), np.asarray([4, 38, 8], dtype=np.int64)
    )
    result = expected_tie_metrics_from_distances(
        np.ones(50, dtype=bool),
        distances,
        bits=16,
        graded_gains=np.ones(50, dtype=np.float64),
        cutoffs=(50,),
    )
    assert result["binary_ndcg_at_50_expected_ties"] == 1.0
    assert result["j_ndcg_at_50_expected_ties"] == 1.0


def test_cutoff_inside_tie_and_graded_gain_are_not_storage_order_credit() -> None:
    distance = np.asarray([3, 3], dtype=np.uint8)
    gains = np.asarray([1.0, 0.5])
    first = expected_tie_metrics_from_distances(
        np.asarray([True, True]), distance, bits=16, graded_gains=gains, cutoffs=(1, 2)
    )
    second = expected_tie_metrics_from_distances(
        np.asarray([True, True]), distance[::-1], bits=16, graded_gains=gains[::-1], cutoffs=(1, 2)
    )
    assert first == second
    assert first["j_ndcg_at_1_expected_ties"] == pytest.approx(0.75)
    assert first["precision_at_1_expected_ties"] == 1.0


def test_expected_tie_metrics_randomized_property_against_all_tie_orders() -> None:
    generator = np.random.default_rng(20260823)
    cutoffs = (1, 3, 8)
    for _case in range(24):
        rows = int(generator.integers(1, 7))
        distances = generator.integers(0, 4, size=rows, dtype=np.uint8)
        gains = generator.choice(
            np.asarray([0.0, 0.25, 0.5, 1.0]), size=rows, replace=True
        )
        relevance = gains > 0
        exact = expected_tie_metrics_from_distances(
            relevance,
            distances,
            bits=16,
            graded_gains=gains,
            cutoffs=cutoffs,
        )
        shells = [
            tuple(int(index) for index in np.flatnonzero(distances == radius))
            for radius in sorted(int(value) for value in np.unique(distances))
        ]
        shell_orders = [tuple(itertools.permutations(shell)) for shell in shells]
        references = []
        for choices in itertools.product(*shell_orders):
            order = tuple(index for shell in choices for index in shell)
            references.append(_ordered_metrics(relevance, gains, order, cutoffs))
        for key in references[0]:
            assert exact[key] == pytest.approx(
                np.mean([record[key] for record in references]), abs=1.0e-12
            )


def test_streaming_resume_ack_delete_and_complete(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    labels = _metric_labels(rank)
    spool = tmp_path / "spool"
    output = tmp_path / "metrics"
    with mock.patch(
        "raw_rebuilt_streaming.metric_worker.load_metric_labels", return_value=labels
    ):
        rank_result = None
        for _ in range(8):
            rank_result = produce_rank_bundles(state, plan, spool, max_new_bundles=1)
            if rank_result["status"] == "COMPLETE":
                break
        assert rank_result is not None and rank_result["status"] == "COMPLETE"
        result = None
        for _ in range(8):
            result = consume_metric_bundles(
                runtime,
                state,
                plan,
                spool,
                output,
                max_new_acks=1,
                _test_allow_synthetic=True,
            )
            if result["status"] == "COMPLETE":
                break
    assert result is not None and result["status"] == "COMPLETE"
    frozen = open_rank_plan(plan, state)
    chunks = list(frozen.chunks())
    assert all(ack_path(spool, chunk).exists() for chunk in chunks)
    assert all(not bundle_path(spool, chunk).exists() for chunk in chunks)
    completion = json.loads((Path(result["evaluation_root"]) / "evaluation_complete.json").read_text())
    assert completion["status"] == "COMPLETE"
    assert len(completion["results"]) == 2
    for chunk in chunks:
        ack = json.loads(ack_path(spool, chunk).read_text("utf-8"))
        assert set(ack) == {
            "schema",
            "status",
            "binding",
            "bundle_manifest_sha256",
            "distances_file_sha256",
            "distances_numeric_sha256",
            "opaque_metric_commitment_sha256",
            "previous_ack_chain_sha256",
            "metric_payload_exposed_to_rank_worker",
            "ack_sha256",
            "ack_chain_sha256",
        }
        assert not any("precision" in key or "map" in key or "ndcg" in key for key in ack)


def test_metric_cannot_open_labels_before_full_rank_evidence_seal(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    spool = tmp_path / "spool"
    result = produce_rank_bundles(state, plan, spool, max_new_bundles=1)
    assert result["status"] == "IN_PROGRESS"
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="evidence seal"):
            consume_metric_bundles(
                runtime,
                state,
                plan,
                spool,
                tmp_path / "metrics",
                _test_allow_synthetic=True,
            )
    opened.assert_not_called()


def test_resealed_distance_dtype_poison_is_rejected_before_labels(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    spool = tmp_path / "spool"
    produce_rank_bundles(state, plan, spool)
    seal_path = spool / "rank_evidence_complete.json"
    seal = json.loads(seal_path.read_text("utf-8"))
    seal["distance_dtype"] = "float32"
    body = {
        key: value
        for key, value in seal.items()
        if key != "rank_evidence_seal_sha256"
    }
    seal["rank_evidence_seal_sha256"] = sha256_json(body)
    atomic_write_json(seal_path, seal)
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="binding changed"):
            consume_metric_bundles(
                runtime,
                state,
                plan,
                spool,
                tmp_path / "metrics",
                _test_allow_synthetic=True,
            )
    opened.assert_not_called()


def test_rank_frontier_hole_fails_instead_of_deadlocking(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    plan = _plan(tmp_path, state)
    spool = tmp_path / "spool"
    result = produce_rank_bundles(state, plan, spool, max_new_bundles=3)
    assert result["status"] == "IN_PROGRESS"
    chunks = list(open_rank_plan(plan, state).chunks())
    shutil.rmtree(bundle_path(spool, chunks[0]))
    with pytest.raises(StreamingIntegrityError, match="frontier has a hole"):
        produce_rank_bundles(state, plan, spool)


def test_metric_private_commit_then_ack_crash_window_resumes(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    labels = _metric_labels(rank)
    spool = tmp_path / "spool"
    output = tmp_path / "metrics"
    produce_rank_bundles(state, plan, spool)
    first = next(open_rank_plan(plan, state).chunks())
    with mock.patch(
        "raw_rebuilt_streaming.metric_worker.load_metric_labels", return_value=labels
    ), mock.patch(
        "raw_rebuilt_streaming.metric_worker.write_ack",
        side_effect=RuntimeError("simulated crash after private commit"),
    ):
        with pytest.raises(RuntimeError, match="simulated crash"):
            consume_metric_bundles(
                runtime, state, plan, spool, output, _test_allow_synthetic=True
            )
    private = (
        output
        / f"metrics-{open_rank_plan(plan, state).plan_sha256[:16]}"
        / "private_partials"
        / first.cell_id
        / f"{first.name}.json"
    )
    assert private.is_file()
    assert bundle_path(spool, first).is_dir()
    assert not ack_path(spool, first).exists()

    def labels_after_recovery(*args, **kwargs):
        assert ack_path(spool, first).is_file()
        assert not bundle_path(spool, first).exists()
        return labels

    with mock.patch(
        "raw_rebuilt_streaming.metric_worker.load_metric_labels",
        side_effect=labels_after_recovery,
    ):
        result = consume_metric_bundles(
            runtime, state, plan, spool, output, _test_allow_synthetic=True
        )
    assert result["status"] == "COMPLETE"


def test_private_metric_score_poison_is_rejected_before_label_reopen(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    labels = _metric_labels(rank)
    spool = tmp_path / "spool"
    output = tmp_path / "metrics"
    produce_rank_bundles(state, plan, spool)
    with mock.patch(
        "raw_rebuilt_streaming.metric_worker.load_metric_labels", return_value=labels
    ):
        result = consume_metric_bundles(
            runtime,
            state,
            plan,
            spool,
            output,
            max_new_acks=1,
            _test_allow_synthetic=True,
        )
    assert result["status"] == "IN_PROGRESS"
    first = next(open_rank_plan(plan, state).chunks())
    private = (
        output
        / f"metrics-{open_rank_plan(plan, state).plan_sha256[:16]}"
        / "private_partials"
        / first.cell_id
        / f"{first.name}.json"
    )
    receipt = json.loads(private.read_text("utf-8"))
    receipt["per_query"][0]["average_precision_expected_ties"] = 2.0
    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"partial_sha256", "private_chain_sha256"}
    }
    receipt["partial_sha256"] = sha256_json(body)
    receipt["private_chain_sha256"] = sha256_json(
        {
            "previous_private_chain_sha256": receipt[
                "previous_private_chain_sha256"
            ],
            "partial_sha256": receipt["partial_sha256"],
        }
    )
    atomic_write_json(private, receipt)
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match=r"outside \[0,1\]"):
            consume_metric_bundles(
                runtime, state, plan, spool, output, _test_allow_synthetic=True
            )
    opened.assert_not_called()


def test_distance_poison_is_rejected_before_metric_labels_open(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    spool = tmp_path / "spool"
    produce_rank_bundles(state, plan, spool)
    chunk = next(open_rank_plan(plan, state).chunks())
    distance_path = bundle_path(spool, chunk) / "distances.npy"
    value = np.load(distance_path, mmap_mode="r+", allow_pickle=False)
    value[0, 0] ^= np.uint8(1)
    value.flush()
    value._mmap.close()
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="bytes changed"):
            consume_metric_bundles(
                runtime, state, plan, spool, tmp_path / "metrics", _test_allow_synthetic=True
            )
    opened.assert_not_called()


def test_bundle_replay_is_rejected_before_metric_labels_open(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    spool = tmp_path / "spool"
    produce_rank_bundles(state, plan, spool)
    chunks = list(open_rank_plan(plan, state).chunks())
    replay = bundle_path(spool, chunks[1])
    shutil.rmtree(replay)
    shutil.copytree(bundle_path(spool, chunks[0]), replay)
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="replayed or rebound"):
            consume_metric_bundles(
                runtime, state, plan, spool, tmp_path / "metrics", _test_allow_synthetic=True
            )
    opened.assert_not_called()


def test_code_receipt_and_nus_tc21_fail_closed(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state_root = _make_code_state(tmp_path, rank)
    state = open_code_state(state_root)
    state.close()
    path = state_root / _array_relative("query", "image", 16)
    value = np.load(path, mmap_mode="r+", allow_pickle=False)
    value[0, 0] ^= np.uint8(1)
    value.flush()
    value._mmap.close()
    with pytest.raises(StreamingIntegrityError, match="bytes changed"):
        open_code_state(state_root)
    with pytest.raises(StreamingIntegrityError, match="TC21"):
        require_dataset_label_geometry("nuswide", 81)


def test_extra_receipt_and_output_path_overlap_fail_closed(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state_root = _make_code_state(tmp_path, rank)
    (state_root / "receipts" / "label-payload.json").write_text(
        '{"labels":[1]}', encoding="utf-8"
    )
    with pytest.raises(StreamingIntegrityError, match="receipt|unbound extra"):
        open_code_state(state_root)
    (state_root / "receipts" / "label-payload.json").unlink()
    with pytest.raises(StreamingIntegrityError, match="overlaps"):
        freeze_rank_plan(
            state_root,
            state_root / "nested-plan-output",
            config=StreamingPlanConfig(bits=(16,), directions=("i2t",)),
        )
    plan = _plan(tmp_path, state_root)
    with pytest.raises(StreamingIntegrityError, match="overlaps"):
        produce_rank_bundles(state_root, plan, plan / "nested-spool")


def test_metric_output_nested_in_spool_is_rejected_before_labels(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    spool = tmp_path / "spool"
    produce_rank_bundles(state, plan, spool)
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="overlaps"):
            consume_metric_bundles(
                runtime,
                state,
                plan,
                spool,
                spool,
                _test_allow_synthetic=True,
            )
    opened.assert_not_called()


def test_unregistered_private_or_ack_file_is_rejected_before_labels(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan = _plan(tmp_path, state)
    frozen = open_rank_plan(plan, state)
    spool = tmp_path / "spool"
    output = tmp_path / "metrics"
    produce_rank_bundles(state, plan, spool)
    metric_root = output / f"metrics-{frozen.plan_sha256[:16]}"
    (metric_root / "private_partials").mkdir(parents=True)
    leak = metric_root / "private_partials" / "unbound-label-leak.bin"
    leak.write_bytes(b"forbidden")
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="unregistered"):
            consume_metric_bundles(
                runtime, state, plan, spool, output, _test_allow_synthetic=True
            )
    opened.assert_not_called()
    leak.unlink()
    (spool / "acks").mkdir(exist_ok=True)
    (spool / "acks" / "unbound.bin").write_bytes(b"forbidden")
    with mock.patch("raw_rebuilt_streaming.metric_worker.load_metric_labels") as opened:
        with pytest.raises(StreamingIntegrityError, match="inventory|outside"):
            consume_metric_bundles(
                runtime, state, plan, spool, output, _test_allow_synthetic=True
            )
    opened.assert_not_called()


def test_orchestrator_deep_verifier_rejects_resealed_metric_tamper(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    runtime = _runtime_stub(tmp_path)
    plan_root = _plan(tmp_path, state)
    frozen = open_rank_plan(plan_root, state)
    labels = _metric_labels(rank)
    spool = tmp_path / "spool"
    output = tmp_path / "metrics"
    produce_rank_bundles(state, plan_root, spool)
    with mock.patch(
        "raw_rebuilt_streaming.metric_worker.load_metric_labels", return_value=labels
    ):
        result = consume_metric_bundles(
            runtime, state, plan_root, spool, output, _test_allow_synthetic=True
        )
    evaluation = Path(result["evaluation_root"])
    _verify_evaluation_after_workers(evaluation, spool, frozen.manifest)
    post_exit_leak = evaluation / "post-exit-unbound-labels.bin"
    post_exit_leak.write_bytes(b"forbidden")
    with pytest.raises(StreamingIntegrityError, match="entry inventory"):
        _verify_evaluation_after_workers(evaluation, spool, frozen.manifest)
    post_exit_leak.unlink()
    completion_path = evaluation / "evaluation_complete.json"
    completion = json.loads(completion_path.read_text("utf-8"))
    descriptor = completion["results"][0]
    metric_path = evaluation / descriptor["path"]
    metric = json.loads(metric_path.read_text("utf-8"))
    current = float(metric["per_query"][0]["average_precision_expected_ties"])
    metric["per_query"][0]["average_precision_expected_ties"] = (
        current + 0.125 if current <= 0.875 else current - 0.125
    )
    metric["summary"] = mean_query_metrics(metric["per_query"])
    metric_body = {
        key: value for key, value in metric.items() if key != "metric_result_sha256"
    }
    metric["metric_result_sha256"] = sha256_json(metric_body)
    atomic_write_json(metric_path, metric)
    descriptor["size"] = metric_path.stat().st_size
    descriptor["sha256"] = sha256_file(metric_path)
    descriptor["map_expected_ties"] = metric["summary"]["map_expected_ties"]
    descriptor["metric_result_sha256"] = metric["metric_result_sha256"]
    completion_body = {
        key: value for key, value in completion.items() if key != "complete_sha256"
    }
    completion["complete_sha256"] = sha256_json(completion_body)
    atomic_write_json(completion_path, completion)
    with pytest.raises(StreamingIntegrityError, match="anchored|receipt"):
        _verify_evaluation_after_workers(evaluation, spool, frozen.manifest)


def test_declared_ancestor_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state = _make_code_state(tmp_path, rank)
    linked = tmp_path / "linked-parent"
    try:
        os.symlink(tmp_path, linked, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is unavailable on this host")
    with pytest.raises(StreamingIntegrityError, match="symlink|reparse"):
        open_code_state(linked / state.name)


def test_rank_worker_source_has_no_metric_label_loader() -> None:
    import raw_rebuilt_streaming.rank_worker as worker

    assert "load_metric_labels" not in inspect.getsource(worker)


def test_rank_worker_subprocess_import_graph_and_cli_are_label_free() -> None:
    script = (
        "import json,sys; import raw_rebuilt_streaming.rank_worker; "
        "print(json.dumps(sorted(k for k in sys.modules if "
        "k.startswith(('raw_rebuilt_runtime','raw_rebuilt_neural',"
        "'raw_rebuilt_baselines','raw_rebuilt_streaming.metric_worker','torch')))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
    help_result = subprocess.run(
        [sys.executable, "-m", "raw_rebuilt_streaming", "rank-worker", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--runtime" not in help_result.stdout
    assert "--process-data-root" not in help_result.stdout
    assert "--rank-device {cpu,cuda}" in help_result.stdout


def test_cpu_packed_hamming_matches_unpacked_reference(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state_root = _make_code_state(tmp_path, rank)
    plan_root = _plan(tmp_path, state_root)
    state = open_code_state(state_root)
    try:
        chunk = next(open_rank_plan(plan_root, state_root).chunks())
        observed = hamming_distance_chunk(state, chunk, rank_device="cpu")
        query = np.unpackbits(
            state.arrays[("query", "image", 16)][chunk.start : chunk.end],
            axis=1,
            count=16,
            bitorder="little",
        )
        database = np.unpackbits(
            state.arrays[("database", "text", 16)],
            axis=1,
            count=16,
            bitorder="little",
        )
        reference = np.count_nonzero(query[:, None, :] != database[None, :, :], axis=2)
        np.testing.assert_array_equal(observed, reference.astype(np.uint8))
    finally:
        state.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_and_cpu_hamming_artifacts_are_byte_identical(tmp_path: Path) -> None:
    rank = _rank_inputs()
    state_root = _make_code_state(tmp_path, rank)
    plan_root = _plan(tmp_path, state_root)
    state = open_code_state(state_root)
    try:
        for chunk in open_rank_plan(plan_root, state_root).chunks():
            cpu = hamming_distance_chunk(state, chunk, rank_device="cpu")
            cuda = hamming_distance_chunk(state, chunk, rank_device="cuda")
            np.testing.assert_array_equal(cuda, cpu)
    finally:
        state.close()


def test_sealed_baseline_artifact_imports_into_common_evaluator(tmp_path: Path) -> None:
    from raw_rebuilt_baselines import (
        encode_label_free,
        train_baseline,
        write_code_artifact,
    )
    from raw_rebuilt_baselines.tests.test_baselines import _fast_dcmh, _fixture

    fit, label_free = _fixture(seed=404)
    checkpoint = train_baseline(
        fit,
        label_free,
        _fast_dcmh(seed=20260825),
        tmp_path / "baseline-runs",
        verbose=False,
    )
    encoded = encode_label_free(checkpoint, label_free, batch_size=4, device="cpu")
    artifact = write_code_artifact(encoded, tmp_path / "baseline-codes")
    state_root = import_baseline_code_artifact(
        artifact, checkpoint, tmp_path / "streaming-states"
    )
    state = open_code_state(state_root)
    try:
        assert state.available_bits == (16,)
        assert state.manifest["binding"]["producer_type"] == "baseline_v1_code_artifact"
        expected_qi = pack_bipolar_codes(encoded.image_codes[label_free.query_idx], 16)
        expected_dt = pack_bipolar_codes(encoded.text_codes[label_free.database_idx], 16)
        np.testing.assert_array_equal(state.arrays[("query", "image", 16)], expected_qi)
        np.testing.assert_array_equal(state.arrays[("database", "text", 16)], expected_dt)
    finally:
        state.close()
    plan = freeze_rank_plan(
        state_root,
        tmp_path / "baseline-plans",
        config=StreamingPlanConfig(bits=(16,), directions=("i2t",), query_chunk_size=1),
    )
    assert open_rank_plan(plan, state_root).manifest["cells"][0]["bits"] == 16
    with pytest.raises(StreamingIntegrityError, match="unavailable"):
        freeze_rank_plan(
            state_root,
            tmp_path / "bad-baseline-plans",
            config=StreamingPlanConfig(bits=(32,), directions=("i2t",)),
        )

    unbound_checkpoint_file = checkpoint / "unbound-labels.bin"
    unbound_checkpoint_file.write_bytes(b"forbidden")
    with pytest.raises(StreamingIntegrityError, match="checkpoint file inventory"):
        import_baseline_code_artifact(
            artifact, checkpoint, tmp_path / "extra-file-streaming-states"
        )
    unbound_checkpoint_file.unlink()

    poisoned = np.load(artifact / "image_codes.npy", allow_pickle=False)
    poisoned[0, 0] = 0
    with (artifact / "image_codes.npy").open("wb") as handle:
        np.save(handle, poisoned, allow_pickle=False)
    manifest = json.loads((artifact / "manifest.json").read_text("utf-8"))
    manifest["arrays"]["image_codes"] = array_descriptor(
        artifact / "image_codes.npy"
    )
    manifest["rank_contract"]["image_codes_numeric_sha256"] = numeric_sha256(poisoned)
    rank_body = {
        key: value
        for key, value in manifest["rank_contract"].items()
        if key != "rank_contract_sha256"
    }
    manifest["rank_contract"]["rank_contract_sha256"] = sha256_json(rank_body)
    atomic_write_json(artifact / "manifest.json", manifest)
    with mock.patch("raw_rebuilt_baselines.checkpoint.load_checkpoint") as checkpoint_open:
        with pytest.raises(StreamingIntegrityError, match="bipolar"):
            import_baseline_code_artifact(
                artifact, checkpoint, tmp_path / "poisoned-streaming-states"
            )
    checkpoint_open.assert_not_called()


def test_baseline_import_rejects_naked_arrays() -> None:
    with pytest.raises(TypeError):
        import_baseline_code_artifact(
            np.ones((2, 16), dtype=np.int8),  # type: ignore[arg-type]
            Path("checkpoint"),
            Path("output"),
        )


def test_baseline_checkpoint_child_symlink_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")
    (checkpoint / "code_receipt.json").write_text("{}", encoding="utf-8")
    original = Path.is_symlink

    def poisoned_child(path: Path) -> bool:
        return path.name == "checkpoint.pt" or original(path)

    with mock.patch.object(Path, "is_symlink", poisoned_child):
        with pytest.raises(StreamingIntegrityError, match="non-symlinks"):
            _checkpoint_snapshot(checkpoint)
