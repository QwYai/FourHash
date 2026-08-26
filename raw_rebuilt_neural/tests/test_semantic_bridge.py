from __future__ import annotations

import numpy as np
import pytest

from raw_rebuilt_neural.semantic_bridge import (
    SemanticBridgeConfig,
    build_one_bit_minhash_map,
    calibrate_training_threshold,
    encode_semantic_bridge,
    expected_minhash_mismatch,
    posterior_to_active_set,
    semantic_bridge_composite_distance,
)


def test_threshold_fallback_is_nonempty_and_deterministic() -> None:
    posterior = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.1, 0.2]], dtype=np.float32
    )
    active = posterior_to_active_set(posterior, 0.5)
    assert active.tolist() == [[False, False, True], [True, False, False]]


def test_training_threshold_uses_both_modalities() -> None:
    labels = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.uint8)
    image = np.asarray([[0.8, 0.2], [0.1, 0.7], [0.6, 0.4]], dtype=np.float32)
    text = np.asarray([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45]], dtype=np.float32)
    threshold, rows = calibrate_training_threshold(
        image, text, labels, (0.3, 0.5, 0.7)
    )
    assert threshold == 0.7
    assert len(rows) == 3
    assert rows[1]["mean_train_jaccard"] == pytest.approx(1.0)


def test_one_bit_minhash_code_is_bipolar_and_repeatable() -> None:
    posterior = np.asarray(
        [[0.8, 0.7, 0.1], [0.1, 0.9, 0.8], [0.2, 0.1, 0.7]],
        dtype=np.float32,
    )
    mapping = build_one_bit_minhash_map(3, bits=16, seed=7)
    first = encode_semantic_bridge(posterior, threshold=0.5, mapping=mapping)
    second = encode_semantic_bridge(posterior, threshold=0.5, mapping=mapping)
    assert first.shape == (3, 16)
    assert first.dtype == np.int8
    assert np.array_equal(first, second)
    assert np.all(np.isin(first, (-1, 1)))


def test_identical_predicted_sets_receive_identical_codes() -> None:
    posterior = np.asarray(
        [[0.8, 0.7, 0.1], [0.9, 0.6, 0.2]], dtype=np.float32
    )
    mapping = build_one_bit_minhash_map(3, bits=16, seed=11)
    code = encode_semantic_bridge(posterior, threshold=0.5, mapping=mapping)
    assert np.array_equal(code[0], code[1])


def test_mixed_radix_distance_preserves_primary_shells() -> None:
    primary = np.asarray([[0, 0, 1, 2], [3, 4, 4, 5]], dtype=np.uint8)
    detail = np.asarray([[16, 0, 0, 16], [15, 16, 0, 1]], dtype=np.uint8)
    composite = semantic_bridge_composite_distance(
        primary, detail, detail_bits=16
    )
    assert np.array_equal(composite // 17, primary)
    assert composite[0, 0] < composite[0, 2]
    assert composite[1, 1] < composite[1, 3]


def test_expected_one_bit_minhash_mismatch_matches_endpoints() -> None:
    assert expected_minhash_mismatch(1.0, 16) == 0.0
    assert expected_minhash_mismatch(0.0, 16) == 8.0
    assert expected_minhash_mismatch(0.5, 16) == 4.0


def test_semantic_bridge_config_rejects_unsorted_thresholds() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        SemanticBridgeConfig(threshold_candidates=(0.5, 0.3))
