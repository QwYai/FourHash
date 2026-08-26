"""Deterministic RZ-CSD-512 training from a sealed indT-only artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from raw_rebuilt_runtime.contract import atomic_write_json, load_json, sha256_file, sha256_json
from rz_csd_clip512 import (
    FROZEN_CONFIG,
    RZCSD512,
    RZCSD512Config,
    compute_training_objective,
    configure_training_label_prior,
    parameter_count,
    seed_everything,
)

from .auxiliary import HashSemanticDecoders, compute_auxiliary_training_objective
from .fit_artifact import FitArtifact, open_fit_artifact
from .integrity import production_code_inventory, reject_unsafe_output_path


TRAIN_SCHEMA = "raw_rebuilt_neural_training_v2"
CHECKPOINT_SCHEMA = "raw_rebuilt_neural_checkpoint_v2"
DEFAULT_SEEDS = (20_260_822, 20_260_823, 20_260_824)
FROZEN_TRAINING_PROTOCOL = "RZ_CSD_CLIP512_CURRICULUM_V1"
FROZEN_WARMUP_EPOCHS = 40
FROZEN_FINE_TUNE_EPOCHS = 5
FROZEN_FINE_TUNE_LEARNING_RATE = 5.0e-5
FROZEN_CODE_BCE_WEIGHT = 0.035
FROZEN_GRADED_WEIGHT = 0.07
FROZEN_POSTERIOR_JACCARD_WEIGHT = 0.0
FROZEN_SELECTION_RESULT_SHA256 = (
    "4b600407bba82f0cea0f73664a30e6ab031e372bf8f129bcfca42ceb39929478"
)
FROZEN_SELECTION_CHECKPOINT_SHA256 = (
    "01218c8d7936ea386cfa08b49fe8d26014be1e028886b3072c2c0dd9ae1a0033"
)


class TrainingError(RuntimeError):
    """Raised when a checkpoint or deterministic run binding differs."""


@dataclass(frozen=True)
class NeuralTrainConfig:
    training_protocol: str = FROZEN_TRAINING_PROTOCOL
    seed: int = DEFAULT_SEEDS[0]
    epochs: int = FROZEN_WARMUP_EPOCHS + FROZEN_FINE_TUNE_EPOCHS
    warmup_epochs: int = FROZEN_WARMUP_EPOCHS
    auxiliary_ramp_epochs: int = FROZEN_FINE_TUNE_EPOCHS
    batch_size: int = FROZEN_CONFIG.batch_size
    learning_rate: float = FROZEN_CONFIG.learning_rate
    fine_tune_learning_rate: float = FROZEN_FINE_TUNE_LEARNING_RATE
    weight_decay: float = FROZEN_CONFIG.weight_decay
    code_bce_weight: float = FROZEN_CODE_BCE_WEIGHT
    graded_weight: float = FROZEN_GRADED_WEIGHT
    posterior_jaccard_weight: float = FROZEN_POSTERIOR_JACCARD_WEIGHT
    gradient_clip_norm: float = 5.0
    hidden_dim: int = FROZEN_CONFIG.hidden_dim
    feedforward_dim: int = FROZEN_CONFIG.feedforward_dim
    residual_layers: int = FROZEN_CONFIG.residual_layers
    posterior_hidden_dim: int = FROZEN_CONFIG.posterior_hidden_dim
    posterior_heads: int = FROZEN_CONFIG.posterior_heads
    dropout: float = FROZEN_CONFIG.dropout
    inference_batch_size: int = FROZEN_CONFIG.inference_batch_size
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.training_protocol != FROZEN_TRAINING_PROTOCOL:
            raise ValueError("training_protocol differs from the frozen curriculum")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be positive")
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("warmup_epochs must lie in [0, epochs]")
        if type(self.auxiliary_ramp_epochs) is not int or self.auxiliary_ramp_epochs < 1:
            raise ValueError("auxiliary_ramp_epochs must be positive")
        if type(self.batch_size) is not int or self.batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.fine_tune_learning_rate) or self.fine_tune_learning_rate <= 0:
            raise ValueError("fine_tune_learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        auxiliary_weights = (
            self.code_bce_weight,
            self.graded_weight,
            self.posterior_jaccard_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in auxiliary_weights):
            raise ValueError("auxiliary loss weights must be finite and nonnegative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")

    def model_config(self) -> RZCSD512Config:
        return replace(
            FROZEN_CONFIG,
            seed=self.seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
            feedforward_dim=self.feedforward_dim,
            residual_layers=self.residual_layers,
            posterior_hidden_dim=self.posterior_hidden_dim,
            posterior_heads=self.posterior_heads,
            dropout=self.dropout,
            inference_batch_size=self.inference_batch_size,
        )


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: RZCSD512
    metadata: Mapping[str, Any]
    checkpoint_sha256: str


def _resolved_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable")
    return device


def _epoch_seed(seed: int, epoch: int) -> int:
    payload = f"raw-rebuilt-neural-epoch-v1:{seed}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device_type": device.type,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": properties.name,
                "gpu_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return record


def _run_binding(
    fit: FitArtifact,
    config: NeuralTrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    train_config = asdict(config)
    model_config = asdict(config.model_config())
    code = production_code_inventory()
    body = {
        "schema": TRAIN_SCHEMA,
        "dataset": fit.dataset,
        "label_dim": fit.label_dim,
        "source_seal_sha256": fit.source_seal_sha256,
        "fit_artifact_sha256": fit.fit_artifact_sha256,
        "fit_split_indT_numeric_sha256": fit.manifest["split_indT_numeric_sha256"],
        "train_config": train_config,
        "train_config_sha256": sha256_json(train_config),
        "model_config": model_config,
        "model_config_sha256": sha256_json(model_config),
        "development_selection": {
            "dataset": "nuswide_indT_internal_only",
            "formal_query_or_database_labels_opened": False,
            "selection_result_sha256": FROZEN_SELECTION_RESULT_SHA256,
            "selected_checkpoint_sha256": FROZEN_SELECTION_CHECKPOINT_SHA256,
            "application": "same_schedule_for_mirflickr_nuswide_mscoco",
        },
        "code_inventory": code,
        "environment": _environment_record(device),
    }
    return {**body, "run_binding_sha256": sha256_json(body)}


def _checkpoint_paths(run_root: Path, epoch: int) -> tuple[Path, Path]:
    checkpoint = run_root / "checkpoints" / f"epoch-{epoch:04d}.pt"
    return checkpoint, checkpoint.with_suffix(".receipt.json")


def _save_checkpoint(
    run_root: Path,
    *,
    epoch: int,
    model: RZCSD512,
    auxiliary_decoders: HashSemanticDecoders,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float | int]],
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    checkpoint, receipt_path = _checkpoint_paths(run_root, epoch)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + ".pending")
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "binding": dict(binding),
        "model_state_dict": model.state_dict(),
        "auxiliary_decoder_state_dict": auxiliary_decoders.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, checkpoint)
    receipt_body = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "epoch": int(epoch),
        "checkpoint": checkpoint.name,
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "run_binding_sha256": binding["run_binding_sha256"],
        "source_seal_sha256": binding["source_seal_sha256"],
        "fit_artifact_sha256": binding["fit_artifact_sha256"],
        "code_inventory_sha256": binding["code_inventory"]["code_inventory_sha256"],
    }
    receipt = {**receipt_body, "receipt_sha256": sha256_json(receipt_body)}
    atomic_write_json(receipt_path, receipt)
    latest_body = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "checkpoint": checkpoint.relative_to(run_root).as_posix(),
        "receipt": receipt_path.relative_to(run_root).as_posix(),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "run_binding_sha256": binding["run_binding_sha256"],
    }
    atomic_write_json(
        run_root / "latest.json",
        {**latest_body, "latest_sha256": sha256_json(latest_body)},
    )
    return checkpoint, receipt


def _load_resume(
    run_root: Path,
    *,
    binding: Mapping[str, Any],
    model: RZCSD512,
    auxiliary_decoders: HashSemanticDecoders,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, list[dict[str, float | int]]]:
    latest_path = run_root / "latest.json"
    if not latest_path.exists():
        return 0, []
    latest = load_json(latest_path)
    latest_body = {key: latest[key] for key in latest if key != "latest_sha256"}
    if sha256_json(latest_body) != latest.get("latest_sha256"):
        raise TrainingError("latest checkpoint pointer changed")
    if latest.get("run_binding_sha256") != binding["run_binding_sha256"]:
        raise TrainingError("resume checkpoint is bound to another run")
    checkpoint = (run_root / str(latest["checkpoint"])).resolve(strict=True)
    receipt = load_json((run_root / str(latest["receipt"])).resolve(strict=True))
    receipt_body = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if sha256_json(receipt_body) != receipt.get("receipt_sha256"):
        raise TrainingError("checkpoint receipt changed")
    observed = sha256_file(checkpoint)
    if observed != receipt.get("checkpoint_sha256") or observed != latest.get("checkpoint_sha256"):
        raise TrainingError("resume checkpoint bytes changed")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("binding") != dict(binding):
        raise TrainingError("resume checkpoint payload binding changed")
    epoch = int(state.get("epoch", -1))
    if epoch != int(latest["epoch"]):
        raise TrainingError("resume epoch differs between checkpoint and pointer")
    model.load_state_dict(state["model_state_dict"], strict=True)
    auxiliary_decoders.load_state_dict(
        state["auxiliary_decoder_state_dict"], strict=True
    )
    optimizer.load_state_dict(state["optimizer_state_dict"])
    history = state.get("history")
    if not isinstance(history, list) or len(history) != epoch + 1:
        raise TrainingError("checkpoint history is incomplete")
    return epoch + 1, history


def train_from_fit_artifact(
    fit_root: Path,
    output_parent: Path,
    *,
    config: NeuralTrainConfig = NeuralTrainConfig(),
    device: str | torch.device = "auto",
    max_epochs_this_call: int | None = None,
    _test_allow_synthetic: bool = False,
) -> Path:
    """Train or resume one deterministic seed using only an indT artifact."""

    if max_epochs_this_call is not None and max_epochs_this_call < 0:
        raise ValueError("max_epochs_this_call must be nonnegative")
    fit = open_fit_artifact(fit_root, _test_allow_synthetic=_test_allow_synthetic)
    resolved = _resolved_device(device)
    output = reject_unsafe_output_path(Path(output_parent), field="training output")
    run_root = output / f"seed-{config.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    model_config = config.model_config()
    model = RZCSD512(label_dim=fit.label_dim, config=model_config).to(resolved)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if resolved.type == "cuda" else None
    auxiliary_decoders = HashSemanticDecoders(label_dim=fit.label_dim).to(resolved)
    # Decoder initialization must not perturb the backbone dropout stream used
    # by the 40-epoch warm-up selected in the indT-only development sweep.
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    # The fit artifact is opened read-only.  Copy the small label matrix before
    # handing it to torch so no non-writable NumPy view can escape.
    positive_weight = configure_training_label_prior(
        model, np.array(fit.labels, dtype=np.uint8, copy=True)
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(auxiliary_decoders.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    binding = _run_binding(fit, config, resolved)
    binding_path = run_root / "run_binding.json"
    if binding_path.exists():
        existing = load_json(binding_path)
        if existing != binding:
            raise TrainingError("training output directory is bound to another run")
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
            image = torch.from_numpy(np.array(fit.image[take], dtype=np.float32, copy=True)).to(resolved)
            text = torch.from_numpy(np.array(fit.text[take], dtype=np.float32, copy=True)).to(resolved)
            labels = torch.from_numpy(np.array(fit.labels[take], dtype=np.float32, copy=True)).to(resolved)
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
            "auxiliary_scale": 0.0
            if epoch < config.warmup_epochs
            else min(
                1.0,
                (epoch - config.warmup_epochs + 1)
                / float(config.auxiliary_ramp_epochs),
            ),
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
            "schema": TRAIN_SCHEMA,
            "status": "COMPLETE",
            "epochs": config.epochs,
            "seed": config.seed,
            "dataset": fit.dataset,
            "source_seal_sha256": fit.source_seal_sha256,
            "fit_artifact_sha256": fit.fit_artifact_sha256,
            "run_binding_sha256": binding["run_binding_sha256"],
            "final_checkpoint": latest["checkpoint"],
            "final_checkpoint_sha256": latest["checkpoint_sha256"],
            "parameter_count": parameter_count(model),
            "inference_parameter_count": parameter_count(model),
            "training_only_auxiliary_parameter_count": sum(
                value.numel() for value in auxiliary_decoders.parameters()
            ),
            "serving_uses_auxiliary_decoders": False,
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
    fit.close()
    return run_root


def _receipt_for_checkpoint(checkpoint: Path) -> dict[str, Any]:
    receipt_path = checkpoint.with_suffix(".receipt.json")
    receipt = load_json(receipt_path)
    body = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if sha256_json(body) != receipt.get("receipt_sha256"):
        raise TrainingError("checkpoint receipt hash changed")
    if sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
        raise TrainingError("checkpoint file hash changed")
    return receipt


def load_trained_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "auto",
    expected_source_seal_sha256: str | None = None,
    require_current_code: bool = True,
) -> LoadedCheckpoint:
    """Load a receipt-verified model for label-free encoding."""

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    receipt = _receipt_for_checkpoint(checkpoint)
    resolved = _resolved_device(device)
    state = torch.load(checkpoint, map_location=resolved, weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise TrainingError("checkpoint payload schema differs")
    binding = state.get("binding")
    if not isinstance(binding, dict) or sha256_json(
        {key: binding[key] for key in binding if key != "run_binding_sha256"}
    ) != binding.get("run_binding_sha256"):
        raise TrainingError("checkpoint run binding changed")
    if receipt.get("run_binding_sha256") != binding["run_binding_sha256"]:
        raise TrainingError("checkpoint and receipt bindings differ")
    if expected_source_seal_sha256 is not None and binding.get("source_seal_sha256") != expected_source_seal_sha256:
        raise TrainingError("checkpoint was trained on another raw-rebuilt source")
    if require_current_code:
        current = production_code_inventory()["code_inventory_sha256"]
        if binding["code_inventory"]["code_inventory_sha256"] != current:
            raise TrainingError("current neural/runtime code differs from checkpoint code")
    model_config = RZCSD512Config(**binding["model_config"])
    label_dim = int(binding["label_dim"])
    if binding.get("dataset") == "nuswide" and label_dim != 21:
        raise TrainingError("NUS-WIDE checkpoint must use TC21 labels")
    model = RZCSD512(label_dim=label_dim, config=model_config).to(resolved)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if not bool(model.posterior_prior_is_bound.item()):
        raise TrainingError("checkpoint does not contain a train-bound posterior prior")
    model.eval()
    metadata = {
        "epoch": int(state["epoch"]),
        "binding": binding,
        "receipt": receipt,
        "device": str(resolved),
    }
    return LoadedCheckpoint(model=model, metadata=metadata, checkpoint_sha256=receipt["checkpoint_sha256"])


__all__ = [
    "DEFAULT_SEEDS",
    "FROZEN_CODE_BCE_WEIGHT",
    "FROZEN_FINE_TUNE_EPOCHS",
    "FROZEN_FINE_TUNE_LEARNING_RATE",
    "FROZEN_GRADED_WEIGHT",
    "FROZEN_POSTERIOR_JACCARD_WEIGHT",
    "FROZEN_TRAINING_PROTOCOL",
    "FROZEN_WARMUP_EPOCHS",
    "LoadedCheckpoint",
    "NeuralTrainConfig",
    "TrainingError",
    "load_trained_checkpoint",
    "train_from_fit_artifact",
]
