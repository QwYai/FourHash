"""Fair indT-only sweep of semantic-code and graded-overlap loss weights.

Every branch restores the same epoch-40 model, decoder, optimizer, and RNG
state.  Eligibility is fixed before execution: mAP and NDCG@50 must each be
non-decreasing for all three widths and both retrieval directions on the
deterministic indT development split.  Formal query/database artifacts are not
accepted, and the output is not a paper claim.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import numeric_sha256, sha256_json
from rz_csd_clip512 import FROZEN_CONFIG, RZCSD512, configure_training_label_prior
from tools.dev_graded_semantic_hash_pilot import HashSemanticDecoders, _evaluate
from tools.dev_posterior_weight_sweep import _parse_weights, _train_epoch, _weight_token
from tools.dev_rzcsd_longer_training_pilot import _seed_everything
from tools.dev_semantic_codebook_pilot import _load_fit, _split


def _delta_report(evaluation: dict, baseline: dict) -> dict[str, object]:
    cells = []
    values = []
    map_values = []
    ndcg_values = []
    for bits in (16, 32, 64):
        for direction in ("i2t", "t2i"):
            for metric, family in (
                ("map_expected_ties", map_values),
                ("ndcg_at_50_expected_ties", ndcg_values),
            ):
                delta = float(evaluation[str(bits)][direction][metric]) - float(
                    baseline[str(bits)][direction][metric]
                )
                cells.append(
                    {
                        "bits": bits,
                        "direction": direction,
                        "metric": metric,
                        "delta": delta,
                    }
                )
                values.append(delta)
                family.append(delta)
    array = np.asarray(values, dtype=np.float64)
    return {
        "all_twelve_nonnegative": bool(np.all(array >= 0.0)),
        "negative_cells": int(np.sum(array < 0.0)),
        "minimum_delta": float(array.min()),
        "mean_delta": float(array.mean()),
        "mean_map_delta": float(np.mean(np.asarray(map_values))),
        "mean_ndcg_at_50_delta": float(np.mean(np.asarray(ndcg_values))),
        "cells": cells,
    }


def _selection_key(branch: dict[str, object]) -> tuple[float, ...]:
    delta = branch["delta_report"]
    eligible = bool(delta["all_twelve_nonnegative"])
    # The first component enforces the predeclared all-metric gate.  Within
    # the same eligibility class, prefer aggregate gain, then the worst cell,
    # then the lower total auxiliary weight.
    return (
        1.0 if eligible else 0.0,
        float(delta["mean_delta"]),
        float(delta["minimum_delta"]),
        -float(branch["code_bce_weight"]) - float(branch["graded_weight"]),
    )


def run(
    fit_root: Path,
    output_dir: Path,
    *,
    code_weights: tuple[float, ...],
    graded_weights: tuple[float, ...],
    warmup_epochs: int,
    fine_tune_epochs: int,
    seed: int,
    fine_tune_learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    _seed_everything(seed)
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    labels = labels_u8.astype(np.float32)
    fit, query, database = _split(identity_ids)
    total_epochs = warmup_epochs + fine_tune_epochs
    config = replace(FROZEN_CONFIG, seed=seed, epochs=total_epochs)
    model = RZCSD512(label_dim=labels.shape[1], config=config).to(device)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    decoders = HashSemanticDecoders(label_dim=labels.shape[1]).to(device)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    positive_weight = configure_training_label_prior(model, labels_u8[fit]).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoders.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_history = []
    for epoch in range(warmup_epochs):
        record = _train_epoch(
            model,
            decoders,
            optimizer,
            image,
            text,
            labels,
            identity_ids,
            fit,
            positive_weight,
            epoch=epoch,
            seed=seed,
            device=device,
            code_bce_weight=0.0,
            graded_weight=0.0,
            posterior_weight=0.0,
            auxiliary_scale=0.0,
        )
        warmup_history.append(record)
        print(json.dumps({"stage": "warmup", **record}, sort_keys=True), flush=True)
    baseline = _evaluate(model, image, text, labels_u8, query, database, device)
    base_state = {
        "model": copy.deepcopy(model.state_dict()),
        "decoders": copy.deepcopy(decoders.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else None,
    }
    branches = []
    best_branch: dict[str, object] | None = None
    best_state: dict[str, object] | None = None
    for code_weight in code_weights:
        for graded_weight in graded_weights:
            branch_model = RZCSD512(label_dim=labels.shape[1], config=config).to(device)
            branch_decoders = HashSemanticDecoders(label_dim=labels.shape[1]).to(device)
            branch_model.load_state_dict(base_state["model"], strict=True)
            branch_decoders.load_state_dict(base_state["decoders"], strict=True)
            branch_optimizer = torch.optim.AdamW(
                list(branch_model.parameters()) + list(branch_decoders.parameters()),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            branch_optimizer.load_state_dict(base_state["optimizer"])
            for group in branch_optimizer.param_groups:
                group["lr"] = fine_tune_learning_rate
            torch.random.set_rng_state(base_state["cpu_rng"])
            if base_state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(base_state["cuda_rng"])
            history = []
            for offset in range(fine_tune_epochs):
                epoch = warmup_epochs + offset
                record = _train_epoch(
                    branch_model,
                    branch_decoders,
                    branch_optimizer,
                    image,
                    text,
                    labels,
                    identity_ids,
                    fit,
                    positive_weight,
                    epoch=epoch,
                    seed=seed,
                    device=device,
                    code_bce_weight=code_weight,
                    graded_weight=graded_weight,
                    posterior_weight=0.0,
                    auxiliary_scale=min(1.0, (offset + 1) / fine_tune_epochs),
                )
                history.append(record)
            evaluation = _evaluate(
                branch_model, image, text, labels_u8, query, database, device
            )
            branch = {
                "code_bce_weight": code_weight,
                "graded_weight": graded_weight,
                "posterior_jaccard_weight": 0.0,
                "history": history,
                "evaluation": evaluation,
                "delta_report": _delta_report(evaluation, baseline),
            }
            branches.append(branch)
            if best_branch is None or _selection_key(branch) > _selection_key(best_branch):
                best_branch = branch
                best_state = {
                    "model": copy.deepcopy(branch_model.state_dict()),
                    "decoders": copy.deepcopy(branch_decoders.state_dict()),
                }
            print(
                json.dumps(
                    {
                        "stage": "branch_complete",
                        "code_bce_weight": code_weight,
                        "graded_weight": graded_weight,
                        "delta_report": branch["delta_report"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if best_branch is None or best_state is None:
        raise AssertionError("weight grid unexpectedly produced no branches")
    eligible_count = sum(
        bool(branch["delta_report"]["all_twelve_nonnegative"]) for branch in branches
    )
    selection_status = (
        "ALL_METRIC_CONFIGURATION_FOUND" if eligible_count else "NO_ALL_METRIC_CONFIGURATION"
    )
    method = {
        "code_bce_weights": list(code_weights),
        "graded_weights": list(graded_weights),
        "posterior_jaccard_weight": 0.0,
        "warmup_epochs": warmup_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_learning_rate": fine_tune_learning_rate,
        "shared_model_decoder_optimizer_and_rng_branch_state": True,
        "selection_rule": (
            "require all 12 bit-direction mAP/NDCG50 deltas >=0; then maximize "
            "mean delta, minimum delta, and prefer lower total auxiliary weight"
        ),
    }
    body: dict[str, object] = {
        "schema": "raw_rebuilt_code_grade_weight_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "selection_status": selection_status,
        "eligible_branches": eligible_count,
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "labels_consumed": "indT_internal_fit_and_development_only",
        "split": {"fit": len(fit), "query": len(query), "database": len(database)},
        "split_hashes": {
            "fit_identity_sha256": numeric_sha256(identity_ids[fit]),
            "query_identity_sha256": numeric_sha256(identity_ids[query]),
            "database_identity_sha256": numeric_sha256(identity_ids[database]),
        },
        "config": asdict(config),
        "method": method,
        "warmup_history": warmup_history,
        "baseline_evaluation": baseline,
        "branches": branches,
        "selected_branch": best_branch,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_name = (
        f"selected_code_{_weight_token(float(best_branch['code_bce_weight']))}"
        f"_grade_{_weight_token(float(best_branch['graded_weight']))}.pt"
    )
    torch.save(
        {
            "schema": result["schema"],
            "status": result["status"],
            "selection_status": selection_status,
            "result_sha256": result["result_sha256"],
            "config": asdict(config),
            "method": {
                "code_bce_weight": best_branch["code_bce_weight"],
                "graded_weight": best_branch["graded_weight"],
                "posterior_jaccard_weight": 0.0,
                "warmup_epochs": warmup_epochs,
                "fine_tune_epochs": fine_tune_epochs,
                "fine_tune_learning_rate": fine_tune_learning_rate,
            },
            "model_state_dict": best_state["model"],
            "decoder_state_dict": best_state["decoders"],
        },
        output_dir / checkpoint_name,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--code-bce-weights", type=_parse_weights, default=(0.0, 0.02, 0.035, 0.05, 0.065, 0.08)
    )
    parser.add_argument(
        "--graded-weights", type=_parse_weights, default=(0.0, 0.04, 0.07, 0.10, 0.13, 0.16)
    )
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.warmup_epochs < 1 or args.fine_tune_epochs < 1:
        raise ValueError("warm-up and fine-tune epochs must be positive")
    if args.fine_tune_learning_rate <= 0.0:
        raise ValueError("fine-tune learning rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = run(
        args.fit.resolve(strict=True),
        args.output_dir.resolve(),
        code_weights=args.code_bce_weights,
        graded_weights=args.graded_weights,
        warmup_epochs=args.warmup_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        seed=args.seed,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        device=device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selection_status": result["selection_status"],
                "eligible_branches": result["eligible_branches"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
