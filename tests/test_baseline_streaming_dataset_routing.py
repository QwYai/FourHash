from __future__ import annotations

import json
from pathlib import Path

import pytest

from raw_rebuilt_streaming.integrity import sha256_json
from tools import aggregate_baseline_streaming_eval as aggregate
from tools import run_baseline_streaming_eval_sweep as sweep


def test_cell_pattern_preserves_dataset_identity() -> None:
    for dataset in sweep.DATASETS:
        match = sweep.CELL_PATTERN.fullmatch(
            f"{dataset}_dcmh-f-seminit_b32_s20260822"
        )
        assert match is not None
        assert match.groups() == (
            dataset,
            "dcmh-f-seminit",
            "32",
            "20260822",
        )
    assert sweep.CELL_PATTERN.fullmatch("nuswide_unknown_b32_s20260822") is None


def test_registered_training_events_are_filtered_by_dataset(tmp_path: Path) -> None:
    events = tmp_path / "sweep_events.jsonl"
    rows = []
    for method in sweep.METHODS:
        for bits in sweep.BITS:
            checkpoint = tmp_path / f"checkpoint-{method}-{bits}"
            codes = tmp_path / f"codes-{method}-{bits}"
            checkpoint.mkdir()
            codes.mkdir()
            rows.append(
                {
                    "event": "cell_complete",
                    "cell": f"nuswide_{method}_b{bits}_s20260822",
                    "checkpoint": str(checkpoint.resolve()),
                    "codes": str(codes.resolve()),
                }
            )
    events.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    found = sweep._registered_inputs(
        events, dataset="nuswide", seeds=(20260822,)
    )
    assert len(found) == 9
    assert {item["dataset"] for item in found.values()} == {"nuswide"}
    with pytest.raises(sweep.BaselineSweepError, match="missing cells"):
        sweep._registered_inputs(events, dataset="mscoco", seeds=(20260822,))


def test_aggregate_completion_grid_is_dataset_scoped(tmp_path: Path) -> None:
    events = tmp_path / "evaluation_events.jsonl"
    rows = []
    for method in sweep.METHODS:
        for bits in sweep.BITS:
            body = {
                "schema": sweep.DRIVER_SCHEMA,
                "event": "cell_complete",
                "cell": f"nuswide_{method}_b{bits}_s20260822",
            }
            rows.append({**body, "event_sha256": sha256_json(body)})
    events.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    found = aggregate._completion_events(
        events, dataset="nuswide", seeds=(20260822,)
    )
    assert len(found) == 9
    with pytest.raises(aggregate.BaselineAggregateError, match="incomplete"):
        aggregate._completion_events(
            events, dataset="mirflickr", seeds=(20260822,)
        )
