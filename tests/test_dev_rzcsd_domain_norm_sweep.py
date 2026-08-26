import json
from dataclasses import asdict

import pytest

from raw_rebuilt_neural.hash_head_variants import DOMAIN_NORM_VARIANTS
from raw_rebuilt_runtime.contract import sha256_json
from tools.dev_rzcsd_domain_norm_sweep import _load_control


def _control() -> dict:
    body = {
        "schema": "raw_rebuilt_rzcsd_hash_head_candidate_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": "nuswide",
        "fit_artifact_sha256": "fit",
        "source_seal_sha256": "source",
        "seed": 11,
        "formal_query_or_database_labels_opened": False,
        "variant": asdict(DOMAIN_NORM_VARIANTS[0]),
        "evaluation": {},
    }
    return {**body, "result_sha256": sha256_json(body)}


def test_load_control_verifies_identity_and_hash(tmp_path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(_control()), encoding="utf-8")
    loaded = _load_control(
        path,
        manifest={
            "dataset": "nuswide",
            "fit_artifact_sha256": "fit",
            "source_seal_sha256": "source",
        },
        seed=11,
    )
    assert loaded["variant"]["name"] == "compact_linear_control"


def test_load_control_rejects_tampering(tmp_path) -> None:
    control = _control()
    control["dataset"] = "mirflickr"
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _load_control(
            path,
            manifest={
                "dataset": "nuswide",
                "fit_artifact_sha256": "fit",
                "source_seal_sha256": "source",
            },
            seed=11,
        )
