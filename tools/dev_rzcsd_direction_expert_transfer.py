"""IndT-only transfer test for a direction-conditioned hash expert.

The NUS-WIDE hash-head diagnosis found a repeatable direction asymmetry:
modality-specific BatchNorm improved the text-to-image cells but damaged the
image-to-text cells.  This development-only command asks whether a transparent
no-harm router transfers to another dataset.  It reuses a hash-verified exact
compact control from the earlier transfer run, trains only the predeclared
independent-modality normalization candidate, and never accepts formal query or
database artifacts.

For each bit width and retrieval direction, the expert is enabled only when
both internal-development mAP and NDCG@50 are no worse than the control and at
least one is strictly better.  Otherwise the exact control is retained.  The
result records every rejected cell and the extra-model deployment cost; it is
not a formal paper claim or a frozen production choice.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.hash_head_variants import DOMAIN_NORM_VARIANTS
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from tools.dev_rzcsd_architecture_sweep import _delta_report
from tools.dev_rzcsd_frozen_route_transfer import _transfer_split
from tools.dev_rzcsd_hash_head_sweep import _train_variant
from tools.dev_semantic_codebook_pilot import _load_fit


PRIMARY_METRICS = ("map_expected_ties", "ndcg_at_50_expected_ties")
EXPERT = next(
    variant
    for variant in DOMAIN_NORM_VARIANTS
    if variant.name == "compact_modality_batchnorm_independent"
)


def _load_transfer_control(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    control = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(control, dict):
        raise ValueError("control result must be a JSON object")
    claimed = control.get("result_sha256")
    body = {key: value for key, value in control.items() if key != "result_sha256"}
    if claimed != sha256_json(body):
        raise RuntimeError("control result hash mismatch")
    expected = {
        "schema": "raw_rebuilt_rzcsd_frozen_route_transfer_candidate_indt_v1",
        "status": "DEVELOPMENT_TRANSFER_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "seed": seed,
        "formal_query_or_database_labels_opened": False,
    }
    for field, value in expected.items():
        if control.get(field) != value:
            raise RuntimeError(f"control {field} mismatch")
    anchor = control.get("anchor_spec")
    if not isinstance(anchor, dict) or anchor.get("name") != "compact_unanchored_control":
        raise RuntimeError("control is not the exact compact unanchored model")
    if any(
        bool(anchor.get(field))
        for field in ("clip_pca", "semantic_bridge")
    ):
        raise RuntimeError("control unexpectedly enables an anchor")
    return control


def _assemble_by_no_harm_gate(
    control: Mapping[str, Any],
    expert: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assembled: dict[str, Any] = {}
    routes: dict[str, Any] = {}
    for bits in (16, 32, 64):
        key = str(bits)
        assembled[key] = {}
        routes[key] = {}
        for direction in ("i2t", "t2i"):
            deltas = {
                metric: float(expert[key][direction][metric])
                - float(control[key][direction][metric])
                for metric in PRIMARY_METRICS
            }
            use_expert = all(value >= 0.0 for value in deltas.values()) and any(
                value > 0.0 for value in deltas.values()
            )
            selected = EXPERT.name if use_expert else "compact_unanchored_control"
            source = expert if use_expert else control
            assembled[key][direction] = dict(source[key][direction])
            routes[key][direction] = {
                "selected": selected,
                "expert_primary_deltas": deltas,
                "gate_passed": use_expert,
            }
    return assembled, routes


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
    fit, query, database = _transfer_split(identity_ids)
    control = _load_transfer_control(control_result, manifest=manifest, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    record, state = _train_variant(
        EXPERT,
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
    control_evaluation = control["evaluation"]
    record["delta_report"] = _delta_report(record["evaluation"], control_evaluation)
    candidate_body = {
        "schema": "raw_rebuilt_rzcsd_direction_expert_candidate_indt_v1",
        "status": "DEVELOPMENT_TRANSFER_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "labels_consumed": "indT_internal_fit_and_development_only",
        "seed": seed,
        **record,
    }
    candidate = {
        **candidate_body,
        "result_sha256": sha256_json(candidate_body),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / f"{EXPERT.name}.json"
    atomic_write_json(candidate_path, candidate)
    checkpoint = output_dir / f"{EXPERT.name}.pt"
    torch.save(
        {
            "schema": candidate_body["schema"],
            "result_sha256": candidate["result_sha256"],
            "model_config": record["model_config"],
            "hash_head_variant": asdict(EXPERT),
            **state,
        },
        checkpoint,
    )

    assembled, routes = _assemble_by_no_harm_gate(
        control_evaluation,
        record["evaluation"],
    )
    assembled_delta = _delta_report(assembled, control_evaluation)
    routed_cells = sum(
        int(routes[str(bits)][direction]["gate_passed"])
        for bits in (16, 32, 64)
        for direction in ("i2t", "t2i")
    )
    body = {
        "schema": "raw_rebuilt_rzcsd_direction_expert_transfer_indt_v1",
        "status": "DEVELOPMENT_TRANSFER_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "labels_consumed": "indT_internal_fit_and_development_only",
        "seed": seed,
        "control_result": control_result.name,
        "control_result_sha256": control["result_sha256"],
        "candidate_result": candidate_path.name,
        "candidate_result_sha256": candidate["result_sha256"],
        "candidate_checkpoint": checkpoint.name,
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "candidate_variant": asdict(EXPERT),
        "selection_rule": (
            "per bit width and retrieval direction, enable the sole expert only "
            "when both indT-development mAP and NDCG@50 deltas are nonnegative "
            "and at least one is positive; otherwise retain the exact control"
        ),
        "routes": routes,
        "routed_cells": routed_cells,
        "assembled_evaluation": assembled,
        "assembled_delta_report": assembled_delta,
        "deployment_accounting": {
            "model_checkpoints": 2,
            "database_code_tables_per_supported_direction": 1,
            "database_code_tables_when_both_directions_are_served": 2,
            "direction_known_before_encoding": True,
        },
        "configuration_frozen_for_formal_evaluation": False,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    atomic_write_json(output_dir / "transfer.json", result)
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
                "dataset": result["dataset"],
                "routed_cells": result["routed_cells"],
                "delta_report": result["assembled_delta_report"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
