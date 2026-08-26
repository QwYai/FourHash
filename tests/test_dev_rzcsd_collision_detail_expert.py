import numpy as np

from tools.dev_rzcsd_collision_detail_expert import (
    _assemble_by_primary_and_graded_gate,
    _lexicographic_distance,
)


def test_lexicographic_distance_never_crosses_primary_shells() -> None:
    primary = np.asarray([[0, 1, 1, 2]], dtype=np.uint16)
    secondary = np.asarray([[16, 16, 0, 0]], dtype=np.uint16)
    combined = _lexicographic_distance(primary, secondary, bits=16)
    order = np.argsort(combined[0], kind="stable")
    assert order.tolist() == [0, 2, 1, 3]


def test_lexicographic_distance_supports_a_shorter_secondary_code() -> None:
    primary = np.asarray([[0, 1, 1, 2]], dtype=np.uint16)
    secondary = np.asarray([[8, 8, 0, 0]], dtype=np.uint16)
    combined = _lexicographic_distance(
        primary,
        secondary,
        bits=64,
        secondary_bits=8,
    )
    assert np.argsort(combined[0], kind="stable").tolist() == [0, 2, 1, 3]


def _cell(map_value: float, ndcg: float, jndcg: float, marker: float) -> dict:
    return {
        "map_expected_ties": map_value,
        "ndcg_at_50_expected_ties": ndcg,
        "jndcg_at_50_expected_ties": jndcg,
        "precision_at_50_expected_ties": marker,
    }


def test_gate_requires_primary_no_harm_and_positive_graded_gain() -> None:
    control = {
        str(bits): {
            direction: _cell(0.50, 0.60, 0.40, float(bits))
            for direction in ("i2t", "t2i")
        }
        for bits in (16, 32, 64)
    }
    candidate = {
        str(bits): {
            direction: _cell(0.51, 0.61, 0.45, -float(bits))
            for direction in ("i2t", "t2i")
        }
        for bits in (16, 32, 64)
    }
    candidate["32"]["i2t"] = _cell(0.52, 0.59, 0.50, -32.0)
    candidate["64"]["t2i"] = _cell(0.52, 0.62, 0.39, -64.0)

    assembled, routes = _assemble_by_primary_and_graded_gate(control, candidate)

    assert routes["16"]["i2t"]["gate_passed"] is True
    assert assembled["16"]["i2t"]["precision_at_50_expected_ties"] == -16.0
    assert routes["32"]["i2t"]["gate_passed"] is False
    assert routes["64"]["t2i"]["gate_passed"] is False
