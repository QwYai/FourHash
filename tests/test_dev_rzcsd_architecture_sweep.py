from __future__ import annotations

import numpy as np

from tools.dev_rzcsd_architecture_sweep import _delta_report, _graded_ndcg_at_50


def test_graded_ndcg_integrates_exact_hamming_ties() -> None:
    distances = np.asarray([[0, 0, 1]], dtype=np.uint16)
    query = np.asarray([[1, 0]], dtype=np.uint8)
    database = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    first = _graded_ndcg_at_50(distances, query, database)
    swapped = _graded_ndcg_at_50(distances, query, database[[1, 0, 2]])
    assert first == swapped
    assert 0.0 < first < 1.0


def test_architecture_gate_detects_any_primary_regression() -> None:
    baseline = {}
    candidate = {}
    for bits in (16, 32, 64):
        baseline[str(bits)] = {}
        candidate[str(bits)] = {}
        for direction in ("i2t", "t2i"):
            baseline[str(bits)][direction] = {
                "map_expected_ties": 0.5,
                "ndcg_at_50_expected_ties": 0.6,
                "jndcg_at_50_expected_ties": 0.4,
            }
            candidate[str(bits)][direction] = {
                "map_expected_ties": 0.51,
                "ndcg_at_50_expected_ties": 0.61,
                "jndcg_at_50_expected_ties": 0.45,
            }
    candidate["32"]["t2i"]["ndcg_at_50_expected_ties"] = 0.59
    report = _delta_report(candidate, baseline)
    assert report["all_twelve_nonnegative"] is False
    assert report["negative_primary_cells"] == 1
    assert report["mean_graded_jndcg_at_50_delta"] > 0.0

