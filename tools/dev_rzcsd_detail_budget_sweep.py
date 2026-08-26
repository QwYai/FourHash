"""Train-only bit-budget sweep for the RZ-CSD collision detail expert.

The full detail expert already improves all internal-development cells on the
three datasets.  This follow-up reduces its database-code overhead.  Expert
bits are ranked using only the fit partition by the product of paired
cross-modal agreement, global bit balance, and multi-label separation.  A
predeclared 4/8/16/32/full grid is then evaluated on the disjoint indT
development query/database split.  No formal artifact is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import atomic_write_json, numeric_sha256, sha256_json
from rz_csd_clip512 import BITS, RZCSD512
from tools.dev_rzcsd_collision_detail_expert import (
    _lexicographic_distance,
    _load_model,
    _load_result,
    _metrics,
    _verify_parent_evaluation,
)
from tools.dev_rzcsd_frozen_route_transfer import _transfer_split
from tools.dev_rzcsd_longer_training_pilot import _encode
from tools.dev_semantic_codebook_pilot import _hamming, _load_fit


DETAIL_BUDGETS = (4, 8, 16, 32, 64)


def _rank_detail_bits(
    image_code: np.ndarray,
    text_code: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image = np.where(np.asarray(image_code) >= 0.0, 1.0, -1.0)
    text = np.where(np.asarray(text_code) >= 0.0, 1.0, -1.0)
    target = np.asarray(labels, dtype=np.float64)
    if image.ndim != 2 or text.shape != image.shape:
        raise ValueError("fit image/text codes must be aligned matrices")
    if target.ndim != 2 or target.shape[0] != image.shape[0]:
        raise ValueError("fit labels must align with the codes")
    if not np.all((target == 0.0) | (target == 1.0)):
        raise ValueError("fit labels must be binary")
    agreement = (image == text).mean(axis=0, dtype=np.float64)
    balance = 1.0 - np.abs(np.concatenate((image, text), axis=0).mean(axis=0))
    consensus = 0.5 * (image + text)
    positive_count = target.sum(axis=0)
    negative_count = len(target) - positive_count
    if np.any(positive_count <= 0.0) or np.any(negative_count <= 0.0):
        raise ValueError("every fit label must have positive and negative examples")
    positive_mean = (target.T @ consensus) / positive_count[:, None]
    negative_mean = ((1.0 - target).T @ consensus) / negative_count[:, None]
    prevalence = positive_count / len(target)
    separation = np.mean(
        prevalence[:, None]
        * (1.0 - prevalence[:, None])
        * np.square(positive_mean - negative_mean),
        axis=0,
    )
    score = agreement * balance * separation
    order = np.argsort(-score, kind="stable")
    return order.astype(np.int64), {
        "agreement": agreement,
        "balance": balance,
        "label_separation": separation,
        "score": score,
    }


def _candidate_budgets(bits: int) -> tuple[int, ...]:
    return tuple(value for value in DETAIL_BUDGETS if value <= bits)


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
    dataset = str(manifest["dataset"])
    control_result = _load_result(control_result_path, dataset=dataset)
    expert_result = _load_result(expert_result_path, dataset=dataset)
    for result, role in ((control_result, "control"), (expert_result, "expert")):
        if result.get("fit_artifact_sha256") != manifest["fit_artifact_sha256"]:
            raise RuntimeError(f"{role} fit artifact mismatch")
        if result.get("source_seal_sha256") != manifest["source_seal_sha256"]:
            raise RuntimeError(f"{role} source seal mismatch")

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

    def encode_development(model: RZCSD512) -> dict[str, dict[int, np.ndarray]]:
        return {
            "query_image": _encode(model, image[query], "image", device),
            "query_text": _encode(model, text[query], "text", device),
            "database_image": _encode(model, image[database], "image", device),
            "database_text": _encode(model, text[database], "text", device),
        }

    control_codes = encode_development(control_model)
    expert_codes = encode_development(expert_model)
    fit_expert = {
        "image": _encode(expert_model, image[fit], "image", device),
        "text": _encode(expert_model, text[fit], "text", device),
    }
    del control_model, expert_model
    query_labels = labels_u8[query]
    database_labels = labels_u8[database]
    control_evaluation: dict[str, Any] = {}
    expert_evaluation: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    bit_rankings: dict[str, Any] = {}
    for bits in BITS:
        key = str(bits)
        primary = {
            "i2t": _hamming(
                control_codes["query_image"][bits],
                control_codes["database_text"][bits],
            ),
            "t2i": _hamming(
                control_codes["query_text"][bits],
                control_codes["database_image"][bits],
            ),
        }
        full_expert = {
            "i2t": _hamming(
                expert_codes["query_image"][bits],
                expert_codes["database_text"][bits],
            ),
            "t2i": _hamming(
                expert_codes["query_text"][bits],
                expert_codes["database_image"][bits],
            ),
        }
        control_evaluation[key] = {
            direction: _metrics(distance, query_labels, database_labels)
            for direction, distance in primary.items()
        }
        expert_evaluation[key] = {
            direction: _metrics(distance, query_labels, database_labels)
            for direction, distance in full_expert.items()
        }
        order, components = _rank_detail_bits(
            fit_expert["image"][bits],
            fit_expert["text"][bits],
            labels_u8[fit],
        )
        bit_rankings[key] = {
            "order": order.tolist(),
            "order_numeric_sha256": numeric_sha256(order),
            **{
                name: [float(value) for value in values]
                for name, values in components.items()
            },
        }
        candidates[key] = {}
        for budget in _candidate_budgets(bits):
            selected = order[:budget]
            secondary = {
                "i2t": _hamming(
                    expert_codes["query_image"][bits][:, selected],
                    expert_codes["database_text"][bits][:, selected],
                ),
                "t2i": _hamming(
                    expert_codes["query_text"][bits][:, selected],
                    expert_codes["database_image"][bits][:, selected],
                ),
            }
            evaluation = {
                direction: _metrics(
                    _lexicographic_distance(
                        primary[direction],
                        secondary[direction],
                        bits=bits,
                        secondary_bits=budget,
                    ),
                    query_labels,
                    database_labels,
                )
                for direction in ("i2t", "t2i")
            }
            candidates[key][str(budget)] = {
                "selected_bit_indices": selected.tolist(),
                "evaluation": evaluation,
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
    # Add per-width/direction deltas after all parent evaluations are verified.
    for bits in BITS:
        key = str(bits)
        for record in candidates[key].values():
            record["deltas"] = {
                direction: {
                    metric: float(record["evaluation"][direction][metric])
                    - float(control_evaluation[key][direction][metric])
                    for metric in (
                        "map_expected_ties",
                        "ndcg_at_50_expected_ties",
                        "jndcg_at_50_expected_ties",
                    )
                }
                for direction in ("i2t", "t2i")
            }

    body = {
        "schema": "raw_rebuilt_rzcsd_detail_budget_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": dataset,
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "labels_consumed": "indT_fit_for_bit_ranking_and_disjoint_development_for_budget_comparison",
        "seed": 20260822,
        "candidate_budget_grid": list(DETAIL_BUDGETS),
        "bit_ranking_rule": (
            "descending fit-only product of paired sign agreement, global bit "
            "balance, and prevalence-weighted multi-label mean separation"
        ),
        "ranking_rule": (
            "lexicographic compact Hamming radius then selected expert-bit Hamming radius"
        ),
        "split": {
            "fit": len(fit),
            "query": len(query),
            "database": len(database),
        },
        "split_hashes": {
            "fit_identity_sha256": numeric_sha256(identity_ids[fit]),
            "query_identity_sha256": numeric_sha256(identity_ids[query]),
            "database_identity_sha256": numeric_sha256(identity_ids[database]),
        },
        "control_result_sha256": control_result["result_sha256"],
        "control_checkpoint": control_checkpoint,
        "expert_result_sha256": expert_result["result_sha256"],
        "expert_checkpoint": expert_checkpoint,
        "bit_rankings": bit_rankings,
        "control_evaluation": control_evaluation,
        "candidates": candidates,
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
                "result_sha256": result["result_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
