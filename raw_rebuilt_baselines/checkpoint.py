"""Content-addressed, provenance-bound baseline checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from raw_rebuilt_runtime.contract import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
    sha256_json,
)

from .adapters import (
    BaselineRunConfig,
    METHOD_CLAIMS,
    make_core_config,
    reconstruct_models,
    train_core,
)
from .contract import (
    DatasetBinding,
    FitArtifact,
    LabelFreeEncodingInputs,
    build_dataset_binding,
    open_verified_fit,
    reject_legacy_path,
)


CHECKPOINT_SCHEMA = "raw_rebuilt_fixed_feature_baseline_checkpoint_v1"
RECEIPT_SCHEMA = "raw_rebuilt_fixed_feature_baseline_code_receipt_v1"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "checkpoint.pt"
RECEIPT_NAME = "code_receipt.json"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class BaselineCheckpointError(RuntimeError):
    """Raised when checkpoint code, state, or data provenance differs."""


def production_code_inventory() -> dict[str, Any]:
    """Hash wrappers, admitted fit boundary, runtime hashes, and reused cores."""

    paths: list[Path] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "tests" not in path.parts and "__pycache__" not in path.parts:
            paths.append(path)
    paths.extend(
        [
            PROJECT_ROOT / "encoders" / "ucch_feature.py",
            PROJECT_ROOT / "encoders" / "dcmh_feature.py",
            PROJECT_ROOT / "encoders" / "cirh_feature.py",
            PROJECT_ROOT / "raw_rebuilt_neural" / "fit_artifact.py",
            PROJECT_ROOT / "raw_rebuilt_neural" / "integrity.py",
            PROJECT_ROOT / "raw_rebuilt_runtime" / "contract.py",
            PROJECT_ROOT / "raw_rebuilt_runtime" / "loader.py",
        ]
    )
    files = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        resolved = path.resolve(strict=True)
        files.append(
            {
                "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    body = {"schema": "raw_rebuilt_baseline_code_inventory_v1", "files": files}
    return {**body, "code_inventory_sha256": sha256_json(body)}


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass
class BaselineCheckpoint:
    root: Path
    method: str
    bits: int
    seed: int
    source_seal_sha256: str
    fit_artifact_sha256: str
    checkpoint_sha256: str
    run_contract_sha256: str
    dataset_binding: DatasetBinding
    core_config: Mapping[str, Any]
    image_model: nn.Module
    text_model: nn.Module
    manifest: Mapping[str, Any]


def _public_config(config: BaselineRunConfig) -> dict[str, Any]:
    return {
        "method": config.method,
        "bits": int(config.bits),
        "seed": int(config.seed),
        "device": config.device,
        "overrides": dict(config.overrides),
    }


def train_baseline(
    fit_input: FitArtifact | Path,
    rank_input: LabelFreeEncodingInputs,
    config: BaselineRunConfig,
    output_parent: Path,
    *,
    verbose: bool = True,
) -> Path:
    """Fit from indT only and serialize a code/data-bound final checkpoint.

    ``rank_input`` contributes only row/split hashes.  Its full features are
    not passed to any optimizer, and its type contains no labels.
    """

    config.validate()
    output = reject_legacy_path(Path(output_parent), field="baseline output")
    if output.exists() and not output.is_dir():
        raise BaselineCheckpointError("baseline output parent must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    fit, owned = open_verified_fit(fit_input)
    try:
        binding = build_dataset_binding(fit, rank_input)
        core_config = asdict(make_core_config(config))
        code_inventory = production_code_inventory()
        run_body = {
            "schema": CHECKPOINT_SCHEMA,
            "dataset_binding": binding.to_dict(),
            "public_config": _public_config(config),
            "core_config": core_config,
            "code_inventory_sha256": code_inventory["code_inventory_sha256"],
            "claim_scope": METHOD_CLAIMS[config.method],
        }
        run_sha = sha256_json(run_body)
        run_name = f"{config.method}-b{config.bits}-s{config.seed}-{run_sha[:16]}"
        target = output / run_name
        if target.exists():
            opened = load_checkpoint(target)
            if opened.run_contract_sha256 != run_sha:
                raise BaselineCheckpointError(
                    "existing content-addressed baseline directory differs"
                )
            return target

        trained = train_core(fit, config, verbose=verbose)
        if trained.core_config != core_config:
            raise BaselineCheckpointError("preflight and trained core configs differ")
        if production_code_inventory() != code_inventory:
            raise BaselineCheckpointError("wrapper/core code changed during training")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "method": config.method,
            "bits": int(config.bits),
            "seed": int(config.seed),
            "run_contract_sha256": run_sha,
            "dataset_binding": binding.to_dict(),
            "public_config": _public_config(config),
            "core_config": trained.core_config,
            "state_dicts": trained.state_dicts,
            "history": trained.history,
            "training_summary": trained.training_summary,
            "deterministic_runtime": trained.deterministic_runtime,
        }
        pending = output / ("." + run_name + f".pending-{os.getpid()}")
        if pending.exists() or target.exists():
            raise BaselineCheckpointError("baseline output collision")
        pending.mkdir(parents=False, exist_ok=False)
        checkpoint_path = pending / CHECKPOINT_NAME
        _atomic_torch_save(checkpoint_path, payload)
        checkpoint_sha = sha256_file(checkpoint_path)
        receipt_body = {
            "schema": RECEIPT_SCHEMA,
            "run_contract_sha256": run_sha,
            "checkpoint_sha256": checkpoint_sha,
            "source_seal_sha256": binding.source_seal_sha256,
            "fit_artifact_sha256": binding.fit_artifact_sha256,
            "full_row_ids_numeric_sha256": binding.full_row_ids_numeric_sha256,
            "train_row_ids_numeric_sha256": binding.train_row_ids_numeric_sha256,
            "split_binding_sha256": binding.split_binding_sha256,
            "train_idx_numeric_sha256": binding.train_idx_numeric_sha256,
            "query_idx_numeric_sha256": binding.query_idx_numeric_sha256,
            "database_idx_numeric_sha256": binding.database_idx_numeric_sha256,
            "code_inventory": code_inventory,
        }
        atomic_write_json(pending / RECEIPT_NAME, receipt_body)
        receipt_sha = sha256_file(pending / RECEIPT_NAME)
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "FINAL_EPOCH_FROZEN",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": config.method,
            "bits": int(config.bits),
            "seed": int(config.seed),
            "claim_scope": METHOD_CLAIMS[config.method],
            "run_contract_sha256": run_sha,
            "checkpoint_path": CHECKPOINT_NAME,
            "checkpoint_sha256": checkpoint_sha,
            "code_receipt_path": RECEIPT_NAME,
            "code_receipt_sha256": receipt_sha,
            "dataset_binding": binding.to_dict(),
            "core_config": trained.core_config,
            "history_sha256": sha256_json(trained.history),
            "checkpoint_selection": "fixed final epoch; query/database labels inaccessible",
        }
        atomic_write_json(pending / MANIFEST_NAME, manifest)
        os.replace(pending, target)
        load_checkpoint(target)
        return target
    finally:
        if owned:
            fit.close()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, field: str
) -> None:
    if set(value) != expected:
        raise BaselineCheckpointError(
            f"{field} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def load_checkpoint(root: Path) -> BaselineCheckpoint:
    """Verify all files, current code, and embedded provenance before use."""

    path = reject_legacy_path(Path(root), field="baseline checkpoint").resolve(
        strict=True
    )
    if not path.is_dir():
        raise BaselineCheckpointError("baseline checkpoint must be a directory")
    manifest = load_json(path / MANIFEST_NAME)
    expected_manifest = {
        "schema",
        "status",
        "created_utc",
        "method",
        "bits",
        "seed",
        "claim_scope",
        "run_contract_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "code_receipt_path",
        "code_receipt_sha256",
        "dataset_binding",
        "core_config",
        "history_sha256",
        "checkpoint_selection",
    }
    _require_exact_keys(manifest, expected_manifest, field="checkpoint manifest")
    if (
        manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("status") != "FINAL_EPOCH_FROZEN"
        or manifest.get("checkpoint_path") != CHECKPOINT_NAME
        or manifest.get("code_receipt_path") != RECEIPT_NAME
    ):
        raise BaselineCheckpointError("checkpoint manifest schema/status differs")
    checkpoint_path = path / CHECKPOINT_NAME
    receipt_path = path / RECEIPT_NAME
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise BaselineCheckpointError("checkpoint bytes changed")
    if sha256_file(receipt_path) != manifest["code_receipt_sha256"]:
        raise BaselineCheckpointError("code receipt bytes changed")
    receipt = load_json(receipt_path)
    expected_receipt = {
        "schema",
        "run_contract_sha256",
        "checkpoint_sha256",
        "source_seal_sha256",
        "fit_artifact_sha256",
        "full_row_ids_numeric_sha256",
        "train_row_ids_numeric_sha256",
        "split_binding_sha256",
        "train_idx_numeric_sha256",
        "query_idx_numeric_sha256",
        "database_idx_numeric_sha256",
        "code_inventory",
    }
    _require_exact_keys(receipt, expected_receipt, field="code receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise BaselineCheckpointError("code receipt schema differs")
    if receipt["code_inventory"] != production_code_inventory():
        raise BaselineCheckpointError("current wrapper/core code differs from receipt")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise BaselineCheckpointError("checkpoint payload is not a mapping")
    expected_payload = {
        "schema",
        "method",
        "bits",
        "seed",
        "run_contract_sha256",
        "dataset_binding",
        "public_config",
        "core_config",
        "state_dicts",
        "history",
        "training_summary",
        "deterministic_runtime",
    }
    _require_exact_keys(payload, expected_payload, field="checkpoint payload")
    for key in ("schema", "method", "bits", "seed", "run_contract_sha256"):
        if payload[key] != manifest[key]:
            raise BaselineCheckpointError(f"checkpoint {key} differs from manifest")
    if payload["dataset_binding"] != manifest["dataset_binding"]:
        raise BaselineCheckpointError("checkpoint data binding differs from manifest")
    if payload["core_config"] != manifest["core_config"]:
        raise BaselineCheckpointError("checkpoint core config differs from manifest")
    if sha256_json(payload["history"]) != manifest["history_sha256"]:
        raise BaselineCheckpointError("checkpoint history differs from manifest")
    if receipt["checkpoint_sha256"] != checkpoint_sha:
        raise BaselineCheckpointError("receipt checkpoint hash differs")
    if receipt["run_contract_sha256"] != manifest["run_contract_sha256"]:
        raise BaselineCheckpointError("receipt run contract differs")
    binding = DatasetBinding(**dict(manifest["dataset_binding"]))
    for field in (
        "source_seal_sha256",
        "fit_artifact_sha256",
        "full_row_ids_numeric_sha256",
        "train_row_ids_numeric_sha256",
        "split_binding_sha256",
        "train_idx_numeric_sha256",
        "query_idx_numeric_sha256",
        "database_idx_numeric_sha256",
    ):
        if receipt[field] != getattr(binding, field):
            raise BaselineCheckpointError(f"receipt {field} differs from data binding")
    image_model, text_model = reconstruct_models(
        str(manifest["method"]), payload["core_config"], payload["state_dicts"]
    )
    return BaselineCheckpoint(
        root=path,
        method=str(manifest["method"]),
        bits=int(manifest["bits"]),
        seed=int(manifest["seed"]),
        source_seal_sha256=binding.source_seal_sha256,
        fit_artifact_sha256=binding.fit_artifact_sha256,
        checkpoint_sha256=checkpoint_sha,
        run_contract_sha256=str(manifest["run_contract_sha256"]),
        dataset_binding=binding,
        core_config=payload["core_config"],
        image_model=image_model,
        text_model=text_model,
        manifest=manifest,
    )


__all__ = [
    "BaselineCheckpoint",
    "BaselineCheckpointError",
    "load_checkpoint",
    "production_code_inventory",
    "train_baseline",
]
