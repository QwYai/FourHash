import numpy as np

from tools.dev_rzcsd_detail_budget_sweep import (
    _candidate_budgets,
    _rank_detail_bits,
)


def test_candidate_budgets_are_predeclared_and_capped() -> None:
    assert _candidate_budgets(16) == (4, 8, 16)
    assert _candidate_budgets(32) == (4, 8, 16, 32)
    assert _candidate_budgets(64) == (4, 8, 16, 32, 64)


def test_fit_only_ranking_prioritizes_reliable_balanced_label_bit() -> None:
    labels = np.asarray([[0], [0], [1], [1]], dtype=np.uint8)
    # Bit 0 is perfectly paired, balanced, and label separating.  Bit 1 is
    # constant; bit 2 disagrees across modalities.
    image = np.asarray(
        [[-1, 1, -1], [-1, 1, -1], [1, 1, 1], [1, 1, 1]],
        dtype=np.float32,
    )
    text = np.asarray(
        [[-1, 1, 1], [-1, 1, 1], [1, 1, -1], [1, 1, -1]],
        dtype=np.float32,
    )
    order, components = _rank_detail_bits(image, text, labels)
    assert order[0] == 0
    assert components["score"][0] > components["score"][1]
    assert components["score"][0] > components["score"][2]
