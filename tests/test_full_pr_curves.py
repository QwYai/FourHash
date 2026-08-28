from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raw_rebuilt_runtime.contract import sha256_json
from tools.evaluate_full_pr_curves import (
    BASELINES,
    DRIVER_SCHEMA,
    PLAN_SCHEMA,
    interpolated_pr_sum,
    select_baseline_code_states,
    verified_frozen_ccde_plan,
)


def test_interpolated_pr_uses_complete_tie_blocks() -> None:
    total = np.asarray([[2, 2, 1], [1, 2, 2]], dtype=np.int64)
    relevant = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=np.int64)
    grid = np.asarray([0.0, 0.5, 1.0])
    summed, count = interpolated_pr_sum(total, relevant, grid)
    assert count == 2
    np.testing.assert_allclose(summed / count, [0.75, 0.75, 0.45])


def test_interpolated_pr_skips_queries_without_relevant_items() -> None:
    total = np.asarray([[2, 1], [1, 1]], dtype=np.int64)
    relevant = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
    summed, count = interpolated_pr_sum(total, relevant, np.asarray([0.0, 1.0]))
    assert count == 1
    np.testing.assert_allclose(summed, [1.0, 1.0])


def test_selects_every_registered_baseline(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    lines = []
    for method in BASELINES:
        state = tmp_path / method
        state.mkdir()
        body = {
            "schema": DRIVER_SCHEMA,
            "event": "cell_complete",
            "cell": f"mirflickr_{method}_b64_s20260822",
            "dataset": "mirflickr",
            "method": method,
            "bits": 64,
            "seed": 20260822,
            "code_state": str(state.resolve()),
            "utc": "2026-08-28T00:00:00+00:00",
        }
        lines.append(json.dumps({**body, "event_sha256": sha256_json(body)}))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = select_baseline_code_states(
        [log], dataset="mirflickr", bits=64, seed=20260822
    )
    assert tuple(selected) == BASELINES


def test_frozen_plan_verification_does_not_rebind_current_source(tmp_path: Path) -> None:
    binding_body = {
        "neural_code_inventory": {"code_inventory_sha256": "1" * 64},
        "streaming_code_inventory": {"code_inventory_sha256": "2" * 64},
        "implementation_inventory": {"code_inventory_sha256": "3" * 64},
    }
    binding = {
        **binding_body,
        "plan_binding_sha256": sha256_json(binding_body),
    }
    body = {
        "schema": PLAN_SCHEMA,
        "status": "rank_state_frozen",
        "labels_loaded_during_freeze": False,
        "formal_gate_or_fallback_used": False,
        "primary_shell_order_is_invariant": True,
        "binding": binding,
        "runtime_identity": {"source_seal_sha256": "4" * 64},
    }
    plan = {**body, "rank_plan_sha256": sha256_json(body)}
    (tmp_path / "evaluation_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    assert verified_frozen_ccde_plan(tmp_path)["rank_plan_sha256"] == plan[
        "rank_plan_sha256"
    ]
