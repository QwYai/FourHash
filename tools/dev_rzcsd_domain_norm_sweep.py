"""Follow-up indT-only sweep for modality-specific hash normalization.

The experiment is motivated by, but does not overwrite, the completed v1
hash-head registry: shared BatchNorm improved every NUS-WIDE text-to-image
cell while regressing image-to-text.  These two candidates retain a shared
hash projection but isolate running statistics by modality.  The immutable v1
compact result is supplied as the control, avoiding an unnecessary retrain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.hash_head_variants import DOMAIN_NORM_VARIANTS
from raw_rebuilt_neural.training import FROZEN_SELECTION_RESULT_SHA256
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from tools.dev_rzcsd_architecture_sweep import _delta_report, _selection_key
from tools.dev_rzcsd_hash_head_sweep import _train_variant
from tools.dev_semantic_codebook_pilot import _load_fit, _split


def _load_control(path: Path, *, manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    control = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(control, dict):
        raise ValueError("control result must be a JSON object")
    claimed = control.get("result_sha256")
    body = {key: value for key, value in control.items() if key != "result_sha256"}
    if claimed != sha256_json(body):
        raise RuntimeError("control result hash mismatch")
    expected = {
        "schema": "raw_rebuilt_rzcsd_hash_head_candidate_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "seed": seed,
        "formal_query_or_database_labels_opened": False,
    }
    for field, value in expected.items():
        if control.get(field) != value:
            raise RuntimeError(f"control {field} mismatch")
    if control.get("variant") != asdict(DOMAIN_NORM_VARIANTS[0]):
        raise RuntimeError("control is not the exact compact linear variant")
    return control


def run(
    fit_root: Path,
    control_result: Path,
    output_dir: Path,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    fit, query, database = _split(identity_ids)
    control = _load_control(control_result, manifest=manifest, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [control]
    baseline = control["evaluation"]
    for variant in DOMAIN_NORM_VARIANTS[1:]:
        record, state = _train_variant(
            variant,
            image,
            text,
            labels_u8,
            identity_ids,
            fit,
            query,
            database,
            seed=seed,
            device=device,
        )
        record["delta_report"] = _delta_report(record["evaluation"], baseline)
        body = {
            "schema": "raw_rebuilt_rzcsd_domain_norm_candidate_indt_v1",
            "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
            "dataset": manifest["dataset"],
            "source_seal_sha256": manifest["source_seal_sha256"],
            "fit_artifact_sha256": manifest["fit_artifact_sha256"],
            "formal_query_or_database_labels_opened": False,
            "labels_consumed": "indT_internal_fit_and_development_only",
            "seed": seed,
            **record,
        }
        result = {**body, "result_sha256": sha256_json(body)}
        atomic_write_json(output_dir / f"{variant.name}.json", result)
        checkpoint = output_dir / f"{variant.name}.pt"
        torch.save(
            {
                "schema": body["schema"],
                "result_sha256": result["result_sha256"],
                "model_config": record["model_config"],
                "hash_head_variant": record["variant"],
                **state,
            },
            checkpoint,
        )
        records.append(result)
        print(
            json.dumps(
                {
                    "stage": "domain_norm_complete",
                    "variant": variant.name,
                    "delta_report": record["delta_report"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected_index = max(
        range(len(records)), key=lambda index: _selection_key(records[index])
    )
    selected = records[selected_index]
    selected_name = str(selected["variant"]["name"])
    selected_checkpoint = None
    selected_checkpoint_sha256 = None
    if selected_index > 0:
        selected_path = output_dir / f"{selected_name}.pt"
        selected_checkpoint = selected_path.name
        selected_checkpoint_sha256 = sha256_file(selected_path)
    selection = {
        "schema": "raw_rebuilt_rzcsd_domain_norm_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "frozen_training_selection_result_sha256": FROZEN_SELECTION_RESULT_SHA256,
        "parent_control_result": str(control_result.name),
        "parent_control_result_sha256": control["result_sha256"],
        "candidate_registry": [asdict(variant) for variant in DOMAIN_NORM_VARIANTS],
        "selection_rule": (
            "require all 12 mAP/NDCG50 deltas >=0 versus exact compact control; "
            "then maximize mean primary delta, mean graded JNDCG50 delta, "
            "minimum primary delta, and prefer fewer inference parameters"
        ),
        "records": records,
        "selected_variant": selected_name,
        "selected_candidate_result_sha256": selected["result_sha256"],
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
    }
    result = {**selection, "result_sha256": sha256_json(selection)}
    atomic_write_json(output_dir / "sweep.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(
        args.fit.resolve(strict=True),
        args.control_result.resolve(strict=True),
        args.output_dir.resolve(),
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_variant": result["selected_variant"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
