import json

import pytest

from raw_rebuilt_runtime.contract import sha256_json
from tools.dev_rzcsd_direction_expert_transfer import (
    EXPERT,
    _assemble_by_no_harm_gate,
    _load_transfer_control,
)


def _metrics(map_value: float, ndcg_value: float, marker: float) -> dict:
    return {
        "map_expected_ties": map_value,
        "ndcg_at_50_expected_ties": ndcg_value,
        "precision_at_50_expected_ties": marker,
        "jndcg_at_50_expected_ties": marker,
    }


def test_no_harm_gate_routes_whole_direction_only_when_both_metrics_hold() -> None:
    control = {
        str(bits): {
            direction: _metrics(0.50, 0.60, float(bits))
            for direction in ("i2t", "t2i")
        }
        for bits in (16, 32, 64)
    }
    expert = {
        str(bits): {
            direction: _metrics(0.51, 0.61, -float(bits))
            for direction in ("i2t", "t2i")
        }
        for bits in (16, 32, 64)
    }
    expert["32"]["i2t"] = _metrics(0.52, 0.59, -32.0)

    assembled, routes = _assemble_by_no_harm_gate(control, expert)

    assert routes["16"]["t2i"]["selected"] == EXPERT.name
    assert assembled["16"]["t2i"]["precision_at_50_expected_ties"] == -16.0
    assert routes["32"]["i2t"]["selected"] == "compact_unanchored_control"
    assert assembled["32"]["i2t"]["precision_at_50_expected_ties"] == 32.0


def _control_body() -> dict:
    return {
        "schema": "raw_rebuilt_rzcsd_frozen_route_transfer_candidate_indt_v1",
        "status": "DEVELOPMENT_TRANSFER_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": "mirflickr",
        "fit_artifact_sha256": "fit",
        "source_seal_sha256": "source",
        "seed": 11,
        "formal_query_or_database_labels_opened": False,
        "anchor_spec": {
            "name": "compact_unanchored_control",
            "clip_pca": False,
            "semantic_bridge": False,
        },
        "evaluation": {},
    }


def test_load_transfer_control_checks_hash_and_identity(tmp_path) -> None:
    body = _control_body()
    path = tmp_path / "compact.json"
    path.write_text(
        json.dumps({**body, "result_sha256": sha256_json(body)}),
        encoding="utf-8",
    )
    loaded = _load_transfer_control(
        path,
        manifest={
            "dataset": "mirflickr",
            "fit_artifact_sha256": "fit",
            "source_seal_sha256": "source",
        },
        seed=11,
    )
    assert loaded["anchor_spec"]["name"] == "compact_unanchored_control"


def test_load_transfer_control_rejects_tampering(tmp_path) -> None:
    body = _control_body()
    path = tmp_path / "compact.json"
    path.write_text(
        json.dumps({**body, "result_sha256": sha256_json(body)}),
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset"] = "mscoco"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _load_transfer_control(
            path,
            manifest={
                "dataset": "mirflickr",
                "fit_artifact_sha256": "fit",
                "source_seal_sha256": "source",
            },
            seed=11,
        )
