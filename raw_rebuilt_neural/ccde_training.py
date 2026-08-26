"""Deterministic full-indT training for the frozen CCDE detail encoder."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from raw_rebuilt_runtime.contract import atomic_write_json, load_json, sha256_json
from rz_csd_clip512 import (
    RZCSD512Config,
    compute_training_objective,
    configure_training_label_prior,
    parameter_count,
    seed_everything,
)

from .auxiliary import HashSemanticDecoders, compute_auxiliary_training_objective
from .ccde_contract import CCDE_DETAIL_VARIANT_NAME, freeze_binding
from .fit_artifact import FitArtifact, open_fit_artifact
from .hash_head_variants import DOMAIN_NORM_VARIANTS, HashHeadRZCSD512, HashHeadVariantSpec
from .integrity import production_code_inventory, reject_unsafe_output_path
from .training import (
    CHECKPOINT_SCHEMA,
    LoadedCheckpoint,
    NeuralTrainConfig,
    TrainingError,
    _environment_record,
    _epoch_seed,
    _load_resume,
    _receipt_for_checkpoint,
    _resolved_device,
    _save_checkpoint,
)


CCDE_DETAIL_TRAIN_SCHEMA = "raw_rebuilt_ccde_detail_training_v1"


def _detail_variant() -> HashHeadVariantSpec:
    matches = tuple(
        value for value in DOMAIN_NORM_VARIANTS if value.name == CCDE_DETAIL_VARIANT_NAME
    )
    if len(matches) != 1:
        raise TrainingError("the frozen CCDE detail variant registry changed")
    variant = matches[0]
    if variant.kind != "modality_batchnorm_independent":
        raise TrainingError("the frozen CCDE detail hash-head kind changed")
    return variant


def _detail_run_binding(
    fit: FitArtifact,
    config: NeuralTrainConfig,
    device: torch.device,
    architecture_freeze_path: Path,
) -> dict[str, Any]:
    train_config = asdict(config)
    model_config = asdict(config.model_config())
    variant = asdict(_detail_variant())
    frozen = freeze_binding(architecture_freeze_path)
    body = {
        "schema": CCDE_DETAIL_TRAIN_SCHEMA,
        "role": "collision_conditioned_secondary_detail_encoder",
        "dataset": fit.dataset,
        "label_dim": fit.label_dim,
        "source_seal_sha256": fit.source_seal_sha256,
        "fit_artifact_sha256": fit.fit_artifact_sha256,
        "fit_split_indT_numeric_sha256": fit.manifest["split_indT_numeric_sha256"],
        "train_config": train_config,
        "train_config_sha256": sha256_json(train_config),
        "model_config": model_config,
        "model_config_sha256": sha256_json(model_config),
        "hash_head_variant": variant,
        "hash_head_variant_sha256": sha256_json(variant),
        "architecture_freeze": frozen,
        "training_information_boundary": {
            "features": "self_extracted_clip512",
            "labels": "full_indT_only",
            "formal_query_or_database_features_opened": False,
            "formal_query_or_database_labels_opened": False,
            "same_schedule_for_every_frozen_dataset": True,
        },
        "code_inventory": production_code_inventory(),
        "environment": _environment_record(device),
    }
    return {**body, "run_binding_sha256": sha256_json(body)}


def _instantiate_detail_model(
    fit: FitArtifact,
    config: NeuralTrainConfig,
    device: torch.device,
) -> HashHeadRZCSD512:
    return HashHeadRZCSD512(
        label_dim=fit.label_dim,
        config=config.model_config(),
        variant=_detail_variant(),
    ).to(device)


def train_detail_from_fit_artifact(
    fit_root: Path,
    architecture_freeze_path: Path,
    output_parent: Path,
    *,
    config: NeuralTrainConfig = NeuralTrainConfig(),
    device: str | torch.device = "auto",
    max_epochs_this_call: int | None = None,
    _test_allow_synthetic: bool = False,
) -> Path:
    """Train or resume the frozen secondary encoder from full indT only."""

    if max_epochs_this_call is not None and max_epochs_this_call < 0:
        raise ValueError("max_epochs_this_call must be nonnegative")
    # Validate before creating an output directory or opening the fit artifact.
    freeze_binding(architecture_freeze_path)
    fit = open_fit_artifact(fit_root, _test_allow_synthetic=_test_allow_synthetic)
    try:
        resolved = _resolved_device(device)
        output = reject_unsafe_output_path(Path(output_parent), field="CCDE training output")
        run_root = output / f"detail-seed-{config.seed}"
        run_root.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed)
        model = _instantiate_detail_model(fit, config, resolved)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if resolved.type == "cuda" else None
        auxiliary_decoders = HashSemanticDecoders(label_dim=fit.label_dim).to(resolved)
        # Preserve the frozen backbone dropout stream exactly as in base training.
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        positive_weight = configure_training_label_prior(
            model, np.array(fit.labels, dtype=np.uint8, copy=True)
        ).to(resolved)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(auxiliary_decoders.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        binding = _detail_run_binding(fit, config, resolved, architecture_freeze_path)
        binding_path = run_root / "run_binding.json"
        if binding_path.exists():
            if load_json(binding_path) != binding:
                raise TrainingError("CCDE training output directory is bound to another run")
        else:
            atomic_write_json(binding_path, binding)
        start_epoch, history = _load_resume(
            run_root,
            binding=binding,
            model=model,
            auxiliary_decoders=auxiliary_decoders,
            optimizer=optimizer,
            device=resolved,
        )
        stop_epoch = config.epochs
        if max_epochs_this_call is not None:
            stop_epoch = min(stop_epoch, start_epoch + max_epochs_this_call)
        rows = len(fit.image)
        for epoch in range(start_epoch, stop_epoch):
            if epoch == config.warmup_epochs:
                for group in optimizer.param_groups:
                    group["lr"] = float(config.fine_tune_learning_rate)
            epoch_seed = _epoch_seed(config.seed, epoch)
            torch.manual_seed(epoch_seed)
            if resolved.type == "cuda":
                torch.cuda.manual_seed_all(epoch_seed)
            permutation = np.random.default_rng(epoch_seed).permutation(rows)
            model.train()
            auxiliary_decoders.train()
            totals: dict[str, float] = {}
            examples = 0
            for start in range(0, rows, config.batch_size):
                take = permutation[start : start + config.batch_size]
                image = torch.from_numpy(
                    np.array(fit.image[take], dtype=np.float32, copy=True)
                ).to(resolved)
                text = torch.from_numpy(
                    np.array(fit.text[take], dtype=np.float32, copy=True)
                ).to(resolved)
                labels = torch.from_numpy(
                    np.array(fit.labels[take], dtype=np.float32, copy=True)
                ).to(resolved)
                identity_ids = np.array(fit.identity_ids[take], dtype=np.uint64, copy=True)
                optimizer.zero_grad(set_to_none=True)
                base = compute_training_objective(
                    model,
                    image,
                    text,
                    labels,
                    identity_ids,
                    positive_weight,
                )
                if epoch < config.warmup_epochs:
                    auxiliary_scale = 0.0
                    code_bce = base["total"].new_zeros(())
                    graded = base["total"].new_zeros(())
                    posterior_jaccard = base["total"].new_zeros(())
                    augmented_total = base["total"]
                else:
                    auxiliary_scale = min(
                        1.0,
                        (epoch - config.warmup_epochs + 1)
                        / float(config.auxiliary_ramp_epochs),
                    )
                    auxiliary = compute_auxiliary_training_objective(
                        model,
                        auxiliary_decoders,
                        image,
                        text,
                        labels,
                        positive_weight,
                    )
                    code_bce = auxiliary["code_bce"]
                    graded = auxiliary["graded"]
                    posterior_jaccard = auxiliary["posterior_jaccard"]
                    augmented_total = base["total"] + auxiliary_scale * (
                        config.code_bce_weight * code_bce
                        + config.graded_weight * graded
                        + config.posterior_jaccard_weight * posterior_jaccard
                    )
                augmented_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(auxiliary_decoders.parameters()),
                    config.gradient_clip_norm,
                )
                optimizer.step()
                objective = {
                    **base,
                    "code_bce": code_bce,
                    "graded": graded,
                    "posterior_jaccard": posterior_jaccard,
                    "augmented_total": augmented_total,
                }
                count = len(take)
                examples += count
                for name, value in objective.items():
                    scalar = float(value.detach().cpu().item())
                    if not math.isfinite(scalar):
                        raise TrainingError(f"non-finite {name} loss at epoch {epoch}")
                    totals[name] = totals.get(name, 0.0) + scalar * count
            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "examples": examples,
                "auxiliary_scale": auxiliary_scale,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            record.update({name: value / examples for name, value in totals.items()})
            history.append(record)
            if (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.epochs:
                _save_checkpoint(
                    run_root,
                    epoch=epoch,
                    model=model,
                    auxiliary_decoders=auxiliary_decoders,
                    optimizer=optimizer,
                    history=history,
                    binding=binding,
                )
        if stop_epoch == config.epochs:
            latest = load_json(run_root / "latest.json")
            complete_body = {
                "schema": CCDE_DETAIL_TRAIN_SCHEMA,
                "status": "COMPLETE",
                "role": "collision_conditioned_secondary_detail_encoder",
                "epochs": config.epochs,
                "seed": config.seed,
                "dataset": fit.dataset,
                "source_seal_sha256": fit.source_seal_sha256,
                "fit_artifact_sha256": fit.fit_artifact_sha256,
                "run_binding_sha256": binding["run_binding_sha256"],
                "architecture_freeze_sha256": binding["architecture_freeze"]["freeze_sha256"],
                "hash_head_variant": binding["hash_head_variant"],
                "final_checkpoint": latest["checkpoint"],
                "final_checkpoint_sha256": latest["checkpoint_sha256"],
                "parameter_count": parameter_count(model),
                "inference_parameter_count": parameter_count(model),
                "training_only_auxiliary_parameter_count": sum(
                    value.numel() for value in auxiliary_decoders.parameters()
                ),
                "serving_uses_auxiliary_decoders": False,
                "serving_role": "secondary_distance_inside_equal_primary_hamming_shell_only",
                "curriculum": {
                    "warmup_epochs": config.warmup_epochs,
                    "fine_tune_epochs": config.epochs - config.warmup_epochs,
                    "fine_tune_learning_rate": config.fine_tune_learning_rate,
                    "code_bce_weight": config.code_bce_weight,
                    "graded_weight": config.graded_weight,
                    "posterior_jaccard_weight": config.posterior_jaccard_weight,
                },
            }
            atomic_write_json(
                run_root / "training_complete.json",
                {**complete_body, "complete_sha256": sha256_json(complete_body)},
            )
        return run_root
    finally:
        fit.close()


def load_detail_checkpoint(
    checkpoint_path: Path,
    architecture_freeze_path: Path,
    *,
    device: str | torch.device = "auto",
    expected_source_seal_sha256: str | None = None,
    require_current_code: bool = True,
) -> LoadedCheckpoint:
    """Receipt-verify and load the one frozen CCDE detail architecture."""

    frozen = freeze_binding(architecture_freeze_path)
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    receipt = _receipt_for_checkpoint(checkpoint)
    resolved = _resolved_device(device)
    state = torch.load(checkpoint, map_location=resolved, weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise TrainingError("CCDE detail checkpoint payload schema differs")
    binding = state.get("binding")
    if not isinstance(binding, dict) or binding.get("schema") != CCDE_DETAIL_TRAIN_SCHEMA:
        raise TrainingError("checkpoint is not a CCDE detail training artifact")
    body = {key: binding[key] for key in binding if key != "run_binding_sha256"}
    if sha256_json(body) != binding.get("run_binding_sha256"):
        raise TrainingError("CCDE detail checkpoint run binding changed")
    if receipt.get("run_binding_sha256") != binding["run_binding_sha256"]:
        raise TrainingError("CCDE detail checkpoint and receipt bindings differ")
    if binding.get("architecture_freeze") != frozen:
        raise TrainingError("CCDE detail checkpoint is bound to another architecture freeze")
    expected_variant = asdict(_detail_variant())
    if binding.get("hash_head_variant") != expected_variant:
        raise TrainingError("CCDE detail checkpoint hash-head variant changed")
    if binding.get("hash_head_variant_sha256") != sha256_json(expected_variant):
        raise TrainingError("CCDE detail checkpoint variant hash changed")
    if (
        expected_source_seal_sha256 is not None
        and binding.get("source_seal_sha256") != expected_source_seal_sha256
    ):
        raise TrainingError("CCDE detail checkpoint was trained on another raw-rebuilt source")
    if require_current_code:
        current = production_code_inventory()["code_inventory_sha256"]
        if binding["code_inventory"]["code_inventory_sha256"] != current:
            raise TrainingError("current neural/runtime code differs from CCDE detail checkpoint code")
    model_config = RZCSD512Config(**binding["model_config"])
    label_dim = int(binding["label_dim"])
    if binding.get("dataset") == "nuswide" and label_dim != 21:
        raise TrainingError("NUS-WIDE CCDE detail checkpoint must use TC21 labels")
    model = HashHeadRZCSD512(
        label_dim=label_dim,
        config=model_config,
        variant=_detail_variant(),
    ).to(resolved)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if not bool(model.posterior_prior_is_bound.item()):
        raise TrainingError("CCDE detail checkpoint lacks a train-bound posterior prior")
    model.eval()
    metadata: Mapping[str, Any] = {
        "epoch": int(state["epoch"]),
        "binding": binding,
        "receipt": receipt,
        "device": str(resolved),
    }
    return LoadedCheckpoint(
        model=model,
        metadata=metadata,
        checkpoint_sha256=receipt["checkpoint_sha256"],
    )


__all__ = [
    "CCDE_DETAIL_TRAIN_SCHEMA",
    "load_detail_checkpoint",
    "train_detail_from_fit_artifact",
]
