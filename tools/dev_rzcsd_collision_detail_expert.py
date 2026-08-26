"""IndT-only collision-detail evaluation for two trained RZ-CSD heads.

The exact compact model supplies the primary Hamming radius.  A separately
trained modality-normalized model supplies only a secondary Hamming radius
inside each primary collision shell.  Therefore the expert can never move an
item across compact-code radii; it only replaces random expected ordering
inside a tied shell.  This isolates whether the expert's large graded-semantic
signal can be retained without sacrificing the compact model's global
neighbourhood.

All checkpoints and result records are hash checked.  The command opens only a
sealed indT fit artifact and makes the same deterministic internal split used
by the parent development experiments.  It cannot accept formal query or
database inputs.
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

from raw_rebuilt_neural.hash_head_variants import (
    DOMAIN_NORM_VARIANTS,
    HashHeadRZCSD512,
)
from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    numeric_sha256,
    sha256_file,
    sha256_json,
)
from rz_csd_clip512 import BITS, RZCSD512, RZCSD512Config
from tools.dev_rzcsd_architecture_sweep import _delta_report, _graded_ndcg_at_50
from tools.dev_rzcsd_frozen_route_transfer import _transfer_split
from tools.dev_rzcsd_longer_training_pilot import _encode
from tools.dev_semantic_codebook_pilot import _expected_metrics, _hamming, _load_fit


EXPERT = next(
    variant
    for variant in DOMAIN_NORM_VARIANTS
    if variant.name == "compact_modality_batchnorm_independent"
)


def _load_result(path: Path, *, dataset: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    claimed = result.get("result_sha256")
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    if claimed != sha256_json(body):
        raise RuntimeError("result hash mismatch")
    if result.get("dataset") != dataset:
        raise RuntimeError("result dataset mismatch")
    if result.get("formal_query_or_database_labels_opened") is not False:
        raise RuntimeError("result does not carry the indT-only marker")
    return result


def _load_model(
    checkpoint_path: Path,
    result: Mapping[str, Any],
    *,
    label_dim: int,
    role: str,
    device: torch.device,
) -> tuple[RZCSD512, dict[str, Any]]:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if state.get("result_sha256") != result.get("result_sha256"):
        raise RuntimeError(f"{role} checkpoint/result binding mismatch")
    if state.get("model_config") != result.get("model_config"):
        raise RuntimeError(f"{role} model configuration mismatch")
    config = RZCSD512Config(**dict(result["model_config"]))
    if role == "control":
        anchor = result.get("anchor_spec")
        variant = result.get("variant")
        is_unanchored = isinstance(anchor, dict) and anchor.get("name") == (
            "compact_unanchored_control"
        )
        is_linear = isinstance(variant, dict) and variant.get("name") == (
            "compact_linear_control"
        )
        if not (is_unanchored or is_linear):
            raise RuntimeError("control result is not an exact compact model")
        model: RZCSD512 = RZCSD512(label_dim=label_dim, config=config)
    elif role == "expert":
        if result.get("variant") != asdict(EXPERT):
            raise RuntimeError("expert result has another hash-head variant")
        if state.get("hash_head_variant") != asdict(EXPERT):
            raise RuntimeError("expert checkpoint variant mismatch")
        model = HashHeadRZCSD512(
            label_dim=label_dim,
            config=config,
            variant=EXPERT,
        )
    else:
        raise ValueError("role must be control or expert")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, {
        "path": checkpoint_path.name,
        "sha256": checkpoint_sha256,
        "result_sha256": result["result_sha256"],
    }


def _lexicographic_distance(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    bits: int,
    secondary_bits: int | None = None,
) -> np.ndarray:
    primary_array = np.asarray(primary)
    secondary_array = np.asarray(secondary)
    if primary_array.shape != secondary_array.shape or primary_array.ndim != 2:
        raise ValueError("primary and secondary distances must be matching matrices")
    if type(bits) is not int or bits < 1:
        raise ValueError("bits must be a positive integer")
    detail_bits = bits if secondary_bits is None else secondary_bits
    if type(detail_bits) is not int or not 1 <= detail_bits <= bits:
        raise ValueError("secondary_bits must be an integer in [1,bits]")
    if (
        np.any(primary_array < 0)
        or np.any(primary_array > bits)
        or np.any(secondary_array < 0)
        or np.any(secondary_array > detail_bits)
    ):
        raise ValueError("Hamming distances lie outside the declared bit width")
    # A full secondary range is smaller than one primary step, so sorting the
    # scalar is exactly lexicographic in (primary, secondary).
    return (
        primary_array.astype(np.uint32) * np.uint32(detail_bits + 1)
        + secondary_array.astype(np.uint32)
    )


def _metrics(
    distance: np.ndarray,
    query_labels: np.ndarray,
    database_labels: np.ndarray,
) -> dict[str, float]:
    result = _expected_metrics(distance, query_labels, database_labels)
    result["jndcg_at_50_expected_ties"] = _graded_ndcg_at_50(
        distance,
        query_labels,
        database_labels,
    )
    return result


def _verify_parent_evaluation(
    recomputed: Mapping[str, Any],
    recorded: Mapping[str, Any],
    *,
    role: str,
) -> None:
    for bits in BITS:
        for direction in ("i2t", "t2i"):
            for metric, value in recomputed[str(bits)][direction].items():
                expected = float(recorded[str(bits)][direction][metric])
                if not np.isclose(float(value), expected, rtol=0.0, atol=1.0e-12):
                    raise RuntimeError(
                        f"{role} recomputed {bits}/{direction}/{metric} differs"
                    )


def _assemble_by_primary_and_graded_gate(
    control: Mapping[str, Any],
    collision_detail: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assembled: dict[str, Any] = {}
    routes: dict[str, Any] = {}
    for bits in BITS:
        key = str(bits)
        assembled[key] = {}
        routes[key] = {}
        for direction in ("i2t", "t2i"):
            primary_deltas = {
                metric: float(collision_detail[key][direction][metric])
                - float(control[key][direction][metric])
                for metric in ("map_expected_ties", "ndcg_at_50_expected_ties")
            }
            graded_delta = float(
                collision_detail[key][direction]["jndcg_at_50_expected_ties"]
            ) - float(control[key][direction]["jndcg_at_50_expected_ties"])
            passed = all(value >= 0.0 for value in primary_deltas.values()) and (
                graded_delta > 0.0
            )
            source = collision_detail if passed else control
            assembled[key][direction] = dict(source[key][direction])
            routes[key][direction] = {
                "selected": "collision_detail_expert" if passed else "compact_control",
                "primary_deltas": primary_deltas,
                "graded_jndcg_at_50_delta": graded_delta,
                "gate_passed": passed,
            }
    return assembled, routes


def run(
    fit_root: Path,
    control_result_path: Path,
    control_checkpoint_path: Path,
    expert_result_path: Path,
    expert_checkpoint_path: Path,
    output_path: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    fit, query, database = _transfer_split(identity_ids)
    del fit  # The trained checkpoints are fixed; only Q/D inside indT are encoded.
    dataset = str(manifest["dataset"])
    control_result = _load_result(control_result_path, dataset=dataset)
    expert_result = _load_result(expert_result_path, dataset=dataset)
    for result, role in ((control_result, "control"), (expert_result, "expert")):
        if result.get("fit_artifact_sha256") != manifest["fit_artifact_sha256"]:
            raise RuntimeError(f"{role} fit artifact mismatch")
        if result.get("source_seal_sha256") != manifest["source_seal_sha256"]:
            raise RuntimeError(f"{role} source seal mismatch")
        if int(result.get("seed", -1)) != 20260822:
            raise RuntimeError(f"{role} seed mismatch")

    control_model, control_checkpoint = _load_model(
        control_checkpoint_path,
        control_result,
        label_dim=labels_u8.shape[1],
        role="control",
        device=device,
    )
    expert_model, expert_checkpoint = _load_model(
        expert_checkpoint_path,
        expert_result,
        label_dim=labels_u8.shape[1],
        role="expert",
        device=device,
    )

    def encode_four(model: RZCSD512) -> dict[str, dict[int, np.ndarray]]:
        return {
            "query_image": _encode(model, image[query], "image", device),
            "query_text": _encode(model, text[query], "text", device),
            "database_image": _encode(model, image[database], "image", device),
            "database_text": _encode(model, text[database], "text", device),
        }

    control_codes = encode_four(control_model)
    expert_codes = encode_four(expert_model)
    del control_model, expert_model
    query_labels = labels_u8[query]
    database_labels = labels_u8[database]
    control_evaluation: dict[str, Any] = {}
    expert_evaluation: dict[str, Any] = {}
    collision_detail: dict[str, Any] = {}
    for bits in BITS:
        primary_i2t = _hamming(
            control_codes["query_image"][bits],
            control_codes["database_text"][bits],
        )
        primary_t2i = _hamming(
            control_codes["query_text"][bits],
            control_codes["database_image"][bits],
        )
        secondary_i2t = _hamming(
            expert_codes["query_image"][bits],
            expert_codes["database_text"][bits],
        )
        secondary_t2i = _hamming(
            expert_codes["query_text"][bits],
            expert_codes["database_image"][bits],
        )
        control_evaluation[str(bits)] = {
            "i2t": _metrics(primary_i2t, query_labels, database_labels),
            "t2i": _metrics(primary_t2i, query_labels, database_labels),
        }
        expert_evaluation[str(bits)] = {
            "i2t": _metrics(secondary_i2t, query_labels, database_labels),
            "t2i": _metrics(secondary_t2i, query_labels, database_labels),
        }
        collision_detail[str(bits)] = {
            "i2t": _metrics(
                _lexicographic_distance(primary_i2t, secondary_i2t, bits=bits),
                query_labels,
                database_labels,
            ),
            "t2i": _metrics(
                _lexicographic_distance(primary_t2i, secondary_t2i, bits=bits),
                query_labels,
                database_labels,
            ),
        }

    _verify_parent_evaluation(
        control_evaluation,
        control_result["evaluation"],
        role="control",
    )
    _verify_parent_evaluation(
        expert_evaluation,
        expert_result["evaluation"],
        role="expert",
    )
    assembled, routes = _assemble_by_primary_and_graded_gate(
        control_evaluation,
        collision_detail,
    )
    routed_cells = sum(
        int(routes[str(bits)][direction]["gate_passed"])
        for bits in BITS
        for direction in ("i2t", "t2i")
    )
    body = {
        "schema": "raw_rebuilt_rzcsd_collision_detail_expert_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": dataset,
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "labels_consumed": "indT_internal_development_only",
        "seed": 20260822,
        "split": {"query": len(query), "database": len(database)},
        "split_hashes": {
            "query_identity_sha256": numeric_sha256(identity_ids[query]),
            "database_identity_sha256": numeric_sha256(identity_ids[database]),
        },
        "control_result": control_result_path.name,
        "control_checkpoint": control_checkpoint,
        "expert_result": expert_result_path.name,
        "expert_checkpoint": expert_checkpoint,
        "ranking_rule": (
            "lexicographic compact Hamming radius then expert Hamming radius; "
            "the expert cannot move an item across compact collision shells"
        ),
        "selection_rule": (
            "enable per bit/direction only if indT-development mAP and binary "
            "NDCG@50 are both nonnegative versus compact and graded JNDCG@50 "
            "is strictly positive; otherwise exact compact fallback"
        ),
        "control_evaluation": control_evaluation,
        "expert_evaluation": expert_evaluation,
        "raw_collision_detail_evaluation": collision_detail,
        "raw_collision_detail_delta_report": _delta_report(
            collision_detail,
            control_evaluation,
        ),
        "routes": routes,
        "routed_cells": routed_cells,
        "assembled_evaluation": assembled,
        "assembled_delta_report": _delta_report(assembled, control_evaluation),
        "deployment_accounting": {
            "model_checkpoints": 2,
            "hash_tables_per_modality_and_width": 2,
            "primary_shell_order_preserved": True,
        },
        "configuration_frozen_for_formal_evaluation": False,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    atomic_write_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-result", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(
        args.fit.resolve(strict=True),
        args.control_result.resolve(strict=True),
        args.control_checkpoint.resolve(strict=True),
        args.expert_result.resolve(strict=True),
        args.expert_checkpoint.resolve(strict=True),
        args.output.resolve(),
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "routed_cells": result["routed_cells"],
                "raw_delta_report": result["raw_collision_detail_delta_report"],
                "assembled_delta_report": result["assembled_delta_report"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
