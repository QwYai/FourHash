"""Predeclared indT-only capacity sweep for the RZ-CSD neural backbone.

The formal query/database runtime is intentionally not an argument.  Every
candidate uses the already frozen 40+5 curriculum and differs only in neural
capacity.  Selection first requires all 12 mAP/NDCG@50 cells to be no worse
than the compact control, then maximizes their mean gain.  Graded JNDCG@50 is
recorded and used only as a secondary tie-breaker.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.auxiliary import HashSemanticDecoders
from raw_rebuilt_neural.training import (
    FROZEN_CODE_BCE_WEIGHT,
    FROZEN_FINE_TUNE_EPOCHS,
    FROZEN_FINE_TUNE_LEARNING_RATE,
    FROZEN_GRADED_WEIGHT,
    FROZEN_POSTERIOR_JACCARD_WEIGHT,
    FROZEN_SELECTION_RESULT_SHA256,
    FROZEN_WARMUP_EPOCHS,
)
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from rz_csd_clip512 import BITS, FROZEN_CONFIG, RZCSD512, configure_training_label_prior, parameter_count
from tools.dev_posterior_weight_sweep import _train_epoch
from tools.dev_rzcsd_longer_training_pilot import _encode, _seed_everything
from tools.dev_semantic_codebook_pilot import _expected_metrics, _hamming, _load_fit, _split


ARCHITECTURES: tuple[dict[str, int | str], ...] = (
    {
        "name": "compact_h256_f512_l2_p128_e5",
        "hidden_dim": 256,
        "feedforward_dim": 512,
        "residual_layers": 2,
        "posterior_hidden_dim": 128,
        "posterior_heads": 5,
    },
    {
        "name": "deep_h256_f768_l4_p192_e5",
        "hidden_dim": 256,
        "feedforward_dim": 768,
        "residual_layers": 4,
        "posterior_hidden_dim": 192,
        "posterior_heads": 5,
    },
    {
        "name": "wide_h384_f768_l3_p192_e5",
        "hidden_dim": 384,
        "feedforward_dim": 768,
        "residual_layers": 3,
        "posterior_hidden_dim": 192,
        "posterior_heads": 5,
    },
    {
        "name": "wide_ensemble_h384_f768_l3_p256_e7",
        "hidden_dim": 384,
        "feedforward_dim": 768,
        "residual_layers": 3,
        "posterior_hidden_dim": 256,
        "posterior_heads": 7,
    },
    {
        "name": "large_ensemble_h512_f1024_l3_p256_e7",
        "hidden_dim": 512,
        "feedforward_dim": 1024,
        "residual_layers": 3,
        "posterior_hidden_dim": 256,
        "posterior_heads": 7,
    },
)


def _graded_ndcg_at_50(
    distances: np.ndarray,
    query_labels: np.ndarray,
    database_labels: np.ndarray,
) -> float:
    cutoff = min(50, distances.shape[1])
    discount = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    values = []
    database = database_labels.astype(np.float64, copy=False)
    database_count = database.sum(axis=1)
    for radius, query in zip(distances, query_labels):
        query_float = query.astype(np.float64, copy=False)
        intersection = database @ query_float
        union = database_count + query_float.sum() - intersection
        gain = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = np.argsort(radius, kind="stable")
        ordered_radius = radius[order]
        ordered_gain = gain[order]
        changes = np.r_[True, ordered_radius[1:] != ordered_radius[:-1]]
        starts = np.flatnonzero(changes)
        ends = np.r_[starts[1:], len(order)]
        sizes = ends - starts
        block_mean = np.add.reduceat(ordered_gain, starts) / sizes
        take = np.clip(cutoff - starts, 0, sizes)
        discounted = np.r_[0.0, np.cumsum(discount)]
        active = take > 0
        expected = float(
            np.sum(
                block_mean[active]
                * (
                    discounted[starts[active] + take[active]]
                    - discounted[starts[active]]
                )
            )
        )
        ideal = np.sort(gain)[::-1][:cutoff]
        ideal_dcg = float(np.sum(ideal * discount))
        values.append(expected / ideal_dcg if ideal_dcg else 0.0)
    return float(np.mean(values))


@torch.no_grad()
def _evaluate(
    model: RZCSD512,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    database: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    query_image = _encode(model, image[query], "image", device)
    query_text = _encode(model, text[query], "text", device)
    database_image = _encode(model, image[database], "image", device)
    database_text = _encode(model, text[database], "text", device)
    result: dict[str, Any] = {}
    for bits in BITS:
        i2t_distance = _hamming(query_image[bits], database_text[bits])
        t2i_distance = _hamming(query_text[bits], database_image[bits])
        i2t = _expected_metrics(i2t_distance, labels[query], labels[database])
        t2i = _expected_metrics(t2i_distance, labels[query], labels[database])
        i2t["jndcg_at_50_expected_ties"] = _graded_ndcg_at_50(
            i2t_distance, labels[query], labels[database]
        )
        t2i["jndcg_at_50_expected_ties"] = _graded_ndcg_at_50(
            t2i_distance, labels[query], labels[database]
        )
        result[str(bits)] = {"i2t": i2t, "t2i": t2i}
    return result


def _delta_report(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    primary = []
    graded = []
    cells = []
    for bits in BITS:
        for direction in ("i2t", "t2i"):
            for metric in ("map_expected_ties", "ndcg_at_50_expected_ties"):
                delta = float(candidate[str(bits)][direction][metric]) - float(
                    baseline[str(bits)][direction][metric]
                )
                primary.append(delta)
                cells.append(
                    {
                        "bits": bits,
                        "direction": direction,
                        "metric": metric,
                        "delta": delta,
                    }
                )
            graded.append(
                float(candidate[str(bits)][direction]["jndcg_at_50_expected_ties"])
                - float(baseline[str(bits)][direction]["jndcg_at_50_expected_ties"])
            )
    primary_array = np.asarray(primary, dtype=np.float64)
    graded_array = np.asarray(graded, dtype=np.float64)
    return {
        "all_twelve_nonnegative": bool(np.all(primary_array >= 0.0)),
        "negative_primary_cells": int(np.sum(primary_array < 0.0)),
        "minimum_primary_delta": float(primary_array.min()),
        "mean_primary_delta": float(primary_array.mean()),
        "mean_graded_jndcg_at_50_delta": float(graded_array.mean()),
        "minimum_graded_jndcg_at_50_delta": float(graded_array.min()),
        "cells": cells,
    }


def _selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    delta = record["delta_report"]
    return (
        1.0 if delta["all_twelve_nonnegative"] else 0.0,
        float(delta["mean_primary_delta"]),
        float(delta["mean_graded_jndcg_at_50_delta"]),
        float(delta["minimum_primary_delta"]),
        -float(record["inference_parameter_count"]),
    )


def _train_candidate(
    architecture: dict[str, int | str],
    image: np.ndarray,
    text: np.ndarray,
    labels_u8: np.ndarray,
    identity_ids: np.ndarray,
    fit: np.ndarray,
    query: np.ndarray,
    database: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    total_epochs = FROZEN_WARMUP_EPOCHS + FROZEN_FINE_TUNE_EPOCHS
    config = replace(
        FROZEN_CONFIG,
        seed=seed,
        epochs=total_epochs,
        hidden_dim=int(architecture["hidden_dim"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        residual_layers=int(architecture["residual_layers"]),
        posterior_hidden_dim=int(architecture["posterior_hidden_dim"]),
        posterior_heads=int(architecture["posterior_heads"]),
    )
    model = RZCSD512(label_dim=labels_u8.shape[1], config=config).to(device)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    decoders = HashSemanticDecoders(label_dim=labels_u8.shape[1]).to(device)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    positive_weight = configure_training_label_prior(model, labels_u8[fit]).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoders.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    labels = labels_u8.astype(np.float32)
    history = []
    for epoch in range(total_epochs):
        if epoch == FROZEN_WARMUP_EPOCHS:
            for group in optimizer.param_groups:
                group["lr"] = FROZEN_FINE_TUNE_LEARNING_RATE
        scale = (
            0.0
            if epoch < FROZEN_WARMUP_EPOCHS
            else min(
                1.0,
                (epoch - FROZEN_WARMUP_EPOCHS + 1) / FROZEN_FINE_TUNE_EPOCHS,
            )
        )
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
            code_bce_weight=FROZEN_CODE_BCE_WEIGHT,
            graded_weight=FROZEN_GRADED_WEIGHT,
            posterior_weight=FROZEN_POSTERIOR_JACCARD_WEIGHT,
            auxiliary_scale=scale,
        )
        history.append(record)
        print(
            json.dumps(
                {
                    "stage": "architecture_epoch",
                    "architecture": architecture["name"],
                    **record,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    evaluation = _evaluate(model, image, text, labels_u8, query, database, device)
    record = {
        "architecture": dict(architecture),
        "model_config": asdict(config),
        "inference_parameter_count": parameter_count(model),
        "training_only_auxiliary_parameter_count": parameter_count(decoders),
        "history": history,
        "evaluation": evaluation,
    }
    state = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "auxiliary_decoder_state_dict": copy.deepcopy(decoders.state_dict()),
    }
    return record, state


def run(fit_root: Path, output_dir: Path, *, seed: int, device: torch.device) -> dict[str, Any]:
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    fit, query, database = _split(identity_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    baseline = None
    for architecture in ARCHITECTURES:
        record, state = _train_candidate(
            architecture,
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
        if baseline is None:
            baseline = record["evaluation"]
        record["delta_report"] = _delta_report(record["evaluation"], baseline)
        body = {
            "schema": "raw_rebuilt_rzcsd_architecture_candidate_indt_v1",
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
        name = str(architecture["name"])
        atomic_write_json(output_dir / f"{name}.json", result)
        torch.save(
            {
                "schema": body["schema"],
                "result_sha256": result["result_sha256"],
                "model_config": record["model_config"],
                **state,
            },
            output_dir / f"{name}.pt",
        )
        records.append(result)
        print(
            json.dumps(
                {
                    "stage": "architecture_complete",
                    "architecture": name,
                    "delta_report": record["delta_report"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if baseline is None:
        raise AssertionError("architecture registry is empty")
    selected_index = max(range(len(records)), key=lambda index: _selection_key(records[index]))
    selected = records[selected_index]
    selected_name = str(selected["architecture"]["name"])
    selected_path = output_dir / f"{selected_name}.pt"
    selection = {
        "schema": "raw_rebuilt_rzcsd_architecture_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "frozen_training_selection_result_sha256": FROZEN_SELECTION_RESULT_SHA256,
        "candidate_registry": list(ARCHITECTURES),
        "selection_rule": (
            "require all 12 mAP/NDCG50 deltas >=0 versus compact control; then "
            "maximize mean primary delta, mean graded JNDCG50 delta, minimum "
            "primary delta, and prefer fewer inference parameters"
        ),
        "records": records,
        "selected_architecture": selected_name,
        "selected_candidate_result_sha256": selected["result_sha256"],
        "selected_checkpoint": selected_path.name,
        "selected_checkpoint_sha256": sha256_file(selected_path),
    }
    result = {**selection, "result_sha256": sha256_json(selection)}
    atomic_write_json(output_dir / "sweep.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(
        args.fit.resolve(strict=True),
        args.output_dir.resolve(),
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_architecture": result["selected_architecture"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
