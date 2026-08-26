from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import run_raw_rebuilt_baseline_sweep as sweep


def test_registered_grid_and_final_json_parser() -> None:
    sweep._require_registered(
        ("mirflickr",), ("ucch-f",), (16,), (20260822,)
    )
    with pytest.raises(sweep.SweepError, match="unregistered"):
        sweep._require_registered(
            ("legacy-mat",), ("ucch-f",), (16,), (20260822,)
        )
    assert sweep._last_json_object("progress\n{\"output\": \"/sealed\"}\n") == {
        "output": "/sealed"
    }


def test_subprocess_runs_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"output": "/sealed"}\n',
        )

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    result = sweep._run(["python", "-m", "raw_rebuilt_baselines"], tmp_path / "run.log")

    assert observed["cwd"] == sweep.PROJECT_ROOT
    assert result == {"output": "/sealed"}


def test_driver_verifies_checkpoint_and_label_free_codes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "status": "FINAL_EPOCH_FROZEN",
                "method": "ucch-f",
                "bits": 16,
                "seed": 20260822,
                "checkpoint_selection": (
                    "fixed final epoch; query/database labels inaccessible"
                ),
                "dataset_binding": {"dataset": "mirflickr"},
            }
        ),
        encoding="utf-8",
    )
    sweep._verify_checkpoint(
        checkpoint,
        dataset="mirflickr",
        method="ucch-f",
        bits=16,
        seed=20260822,
    )

    codes = tmp_path / "codes"
    codes.mkdir()
    (codes / "manifest.json").write_text(
        json.dumps(
            {
                "status": "rank_state_frozen",
                "labels_loaded_during_freeze": False,
                "rank_contract": {
                    "method": "ucch-f",
                    "bits": 16,
                    "seed": 20260822,
                    "source_seal_sha256": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    sweep._verify_codes(
        codes,
        dataset="mirflickr",
        method="ucch-f",
        bits=16,
        seed=20260822,
    )
    value = json.loads((codes / "manifest.json").read_text(encoding="utf-8"))
    value["labels_loaded_during_freeze"] = True
    (codes / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sweep.SweepError, match="label-free"):
        sweep._verify_codes(
            codes,
            dataset="mirflickr",
            method="ucch-f",
            bits=16,
            seed=20260822,
        )
