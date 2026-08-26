from __future__ import annotations

import pytest

from tools.analyze_ccde_query_deltas import _distribution, _quantile


def test_query_delta_distribution_uses_deterministic_linear_quantiles() -> None:
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    summary = _distribution(values)
    assert summary["mean"] == 0.0
    assert summary["median"] == 0.0
    assert summary["q25"] == -1.0
    assert summary["q75"] == 1.0
    assert summary["positive_fraction"] == pytest.approx(0.4)
    assert summary["nonnegative_fraction"] == pytest.approx(0.6)
    assert _quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)
