"""Predeclared indT-only hash-anchor sweep after capacity/routing diagnostics.

The exact compact model is the control.  Candidate models retain that encoder
and add a fixed train-feature PCA residual, a calibrated posterior-to-code
bridge, or both.  The formal query/database runtime is not an argument.
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
from raw_rebuilt_neural.hash_anchors import (
    HASH_ANCHOR_SPECS,
    AnchoredHashRZCSD512,
    HashAnchorSpec,
    fit_clip_pca_anchor,
    hash_anchor_binding_record,
)
from raw_rebuilt_neural.training import (
    FROZEN_CODE_BCE_WEIGHT,
    FROZEN_FINE_TUNE_EPOCHS,
    FROZEN_FINE_TUNE_LEARNING_RATE,
    FROZEN_GRADED_WEIGHT,
    FROZEN_POSTERIOR_JACCARD_WEIGHT,
    FROZEN_SELECTION_RESULT_SHA256,
    FROZEN_WARMUP_EPOCHS,
)
from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    numeric_sha256,
    sha256_file,
    sha256_json,
)
from rz_csd_clip512 import FROZEN_CONFIG, RZCSD512, configure_training_label_prior, parameter_count
from tools.dev_posterior_weight_sweep import _train_epoch
from tools.dev_rzcsd_architecture_sweep import _delta_report, _evaluate, _selection_key
from tools.dev_rzcsd_longer_training_pilot import _seed_everything
from tools.dev_semantic_codebook_pilot import _load_fit, _split


CONTROL = {
    "name": "compact_unanchored_control",
    "clip_pca": False,
    "semantic_bridge": False,
    "clip_gate_initial": 0.0,
    "semantic_gate_initial": 0.0,
}


def _build_model(
    label_dim: int,
    config: Any,
    spec: HashAnchorSpec | None,
    pca_state: dict[str, np.ndarray],
) -> RZCSD512:
    if spec is None:
        return RZCSD512(label_dim=label_dim, config=config)
    model = AnchoredHashRZCSD512(
        label_dim=label_dim, config=config, anchor_spec=spec
    )
    if spec.clip_pca:
        model.bind_clip_pca_anchor(pca_state)
    return model


def _train_candidate(
    spec: HashAnchorSpec | None,
    pca_state: dict[str, np.ndarray],
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
    config = replace(FROZEN_CONFIG, seed=seed, epochs=total_epochs)
    model = _build_model(labels_u8.shape[1], config, spec, pca_state).to(device)
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
    name = CONTROL["name"] if spec is None else spec.name
    for epoch in range(total_epochs):
        if epoch == FROZEN_WARMUP_EPOCHS:
            for group in optimizer.param_groups:
                group["lr"] = FROZEN_FINE_TUNE_LEARNING_RATE
        auxiliary_scale = (
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
            auxiliary_scale=auxiliary_scale,
        )
        history.append(record)
        print(
            json.dumps(
                {"stage": "hash_anchor_epoch", "candidate": name, **record},
                sort_keys=True,
            ),
            flush=True,
        )
    evaluation = _evaluate(model, image, text, labels_u8, query, database, device)
    record = {
        "anchor_spec": CONTROL if spec is None else asdict(spec),
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
    pca_state = fit_clip_pca_anchor(image[fit], text[fit])
    pca_record = {
        **hash_anchor_binding_record(pca_state),
        "center_numeric_sha256": numeric_sha256(pca_state["center"]),
        "projection_numeric_sha256": numeric_sha256(pca_state["projection"]),
        "scale_numeric_sha256": numeric_sha256(pca_state["scale"]),
        "eigenvalues_numeric_sha256": numeric_sha256(pca_state["eigenvalues"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    baseline = None
    candidates: tuple[HashAnchorSpec | None, ...] = (None, *HASH_ANCHOR_SPECS)
    for spec in candidates:
        record, state = _train_candidate(
            spec,
            pca_state,
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
            "schema": "raw_rebuilt_rzcsd_hash_anchor_candidate_indt_v1",
            "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
            "dataset": manifest["dataset"],
            "source_seal_sha256": manifest["source_seal_sha256"],
            "fit_artifact_sha256": manifest["fit_artifact_sha256"],
            "formal_query_or_database_labels_opened": False,
            "labels_consumed": "indT_internal_fit_and_development_only",
            "pca_binding": pca_record,
            "seed": seed,
            **record,
        }
        result = {**body, "result_sha256": sha256_json(body)}
        name = str(record["anchor_spec"]["name"])
        atomic_write_json(output_dir / f"{name}.json", result)
        checkpoint = output_dir / f"{name}.pt"
        torch.save(
            {
                "schema": body["schema"],
                "result_sha256": result["result_sha256"],
                "model_config": record["model_config"],
                "anchor_spec": record["anchor_spec"],
                "pca_binding": pca_record,
                **state,
            },
            checkpoint,
        )
        records.append(result)
        print(
            json.dumps(
                {
                    "stage": "hash_anchor_complete",
                    "candidate": name,
                    "delta_report": record["delta_report"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if baseline is None:
        raise AssertionError("hash-anchor registry is empty")
    selected_index = max(range(len(records)), key=lambda index: _selection_key(records[index]))
    selected = records[selected_index]
    selected_name = str(selected["anchor_spec"]["name"])
    selected_path = output_dir / f"{selected_name}.pt"
    selection = {
        "schema": "raw_rebuilt_rzcsd_hash_anchor_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "frozen_training_selection_result_sha256": FROZEN_SELECTION_RESULT_SHA256,
        "pca_binding": pca_record,
        "candidate_registry": [CONTROL, *[asdict(spec) for spec in HASH_ANCHOR_SPECS]],
        "selection_rule": (
            "require all 12 mAP/NDCG50 deltas >=0 versus exact compact control; "
            "then maximize mean primary delta, mean graded JNDCG50 delta, "
            "minimum primary delta, and prefer fewer inference parameters"
        ),
        "records": records,
        "selected_candidate": selected_name,
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
                "selected_candidate": result["selected_candidate"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

