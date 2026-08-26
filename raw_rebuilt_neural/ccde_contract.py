"""Immutable contract for the frozen collision-conditioned detail expert.

The architecture and 16-bit budget were selected on registered internal
``indT`` development splits.  Formal workers must bind to the exact freeze
artifact below before they train, select detail coordinates, or rank any
query/database row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from raw_rebuilt_runtime.contract import load_json, sha256_file, sha256_json


CCDE_FREEZE_SCHEMA = "rz_csd_collision_detail_architecture_freeze_v2"
CCDE_FREEZE_STATUS = "FROZEN_BEFORE_CCDE_FULL_INDT_TRAINING_AND_FORMAL_EVALUATION"
CCDE_FREEZE_CONTENT_SHA256 = (
    "bea16eff755ff3a3748cb85bddd6342e555630f10196329ac100e3c20e98a65b"
)
CCDE_FREEZE_FILE_SHA256 = (
    "23782cf50f458f0f144628be3dbce5e7265b4e19e114890dc7e5f0357fb673b8"
)
CCDE_DETAIL_CAP = 16
CCDE_DETAIL_VARIANT_NAME = "compact_modality_batchnorm_independent"


class CCDEContractError(RuntimeError):
    """Raised when a formal CCDE stage is not bound to the frozen rule."""


def load_ccde_freeze(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load and fully verify the immutable v2 CCDE architecture freeze."""

    target = Path(path).expanduser().resolve(strict=True)
    observed_file_sha = sha256_file(target)
    if observed_file_sha != CCDE_FREEZE_FILE_SHA256:
        raise CCDEContractError("CCDE freeze file bytes differ from the registered freeze")
    freeze = load_json(target)
    body = {key: freeze[key] for key in freeze if key != "freeze_sha256"}
    observed_content_sha = sha256_json(body)
    if observed_content_sha != freeze.get("freeze_sha256"):
        raise CCDEContractError("CCDE freeze content hash changed")
    if observed_content_sha != CCDE_FREEZE_CONTENT_SHA256:
        raise CCDEContractError("CCDE freeze is not the registered v2 decision")
    if freeze.get("schema") != CCDE_FREEZE_SCHEMA:
        raise CCDEContractError("CCDE freeze schema differs")
    if freeze.get("status") != CCDE_FREEZE_STATUS:
        raise CCDEContractError("CCDE architecture was not frozen before formal evaluation")

    architecture = freeze.get("frozen_architecture")
    required_architecture = {
        "name": "RZ-CSD Collision-Conditioned Detail Expert (CCDE)",
        "global_detail_cap": CCDE_DETAIL_CAP,
        "detail_bits_by_primary_width": {"16": 16, "32": 16, "64": 16},
        "detail_encoder": (
            "compact RZ-CSD with shared projections and modality-specific "
            "BatchNorm running statistics and affine parameters"
        ),
        "primary_encoder": "exact compact RZ-CSD linear hash heads",
        "primary_shell_order_is_invariant": True,
        "development_gate_or_fallback_used_at_formal_inference": False,
        "ranking": (
            "lexicographic (primary Hamming, selected detail Hamming), implemented "
            "as primary_distance*(detail_bits+1)+detail_distance"
        ),
        "detail_bit_order": (
            "fit-only descending product of paired sign agreement, global bit "
            "balance, and prevalence-weighted label separation"
        ),
        "training_schedule": "frozen 40-epoch warmup plus 5-epoch auxiliary curriculum",
    }
    if not isinstance(architecture, dict):
        raise CCDEContractError("CCDE frozen architecture is missing")
    for key, expected in required_architecture.items():
        if architecture.get(key) != expected:
            raise CCDEContractError(f"CCDE frozen architecture field changed: {key}")

    formal = freeze.get("formal_application_contract")
    if not isinstance(formal, dict):
        raise CCDEContractError("CCDE formal application contract is missing")
    required_flags = (
        "expert_bit_order_may_use_only_that_datasets_full_indT_labels",
        "formal_failures_must_be_reported_without_fallback",
        "formal_results_may_not_change_the_frozen_rule",
        "no_formal_query_or_database_retuning",
        "same_architecture_and_budget_rule_for_every_dataset",
    )
    if any(formal.get(flag) is not True for flag in required_flags):
        raise CCDEContractError("CCDE formal application contract was weakened")
    if formal.get("datasets") != ["mirflickr", "nuswide", "mscoco"]:
        raise CCDEContractError("CCDE formal dataset list changed")
    return freeze, observed_file_sha


def freeze_binding(path: Path) -> dict[str, Any]:
    """Return the immutable subset embedded in every downstream artifact."""

    freeze, file_sha = load_ccde_freeze(path)
    return {
        "schema": freeze["schema"],
        "status": freeze["status"],
        "freeze_sha256": freeze["freeze_sha256"],
        "freeze_file_sha256": file_sha,
        "frozen_architecture": freeze["frozen_architecture"],
        "formal_application_contract": freeze["formal_application_contract"],
        "selection_boundary": freeze["selection_boundary"],
    }


__all__ = [
    "CCDEContractError",
    "CCDE_DETAIL_CAP",
    "CCDE_DETAIL_VARIANT_NAME",
    "CCDE_FREEZE_CONTENT_SHA256",
    "CCDE_FREEZE_FILE_SHA256",
    "CCDE_FREEZE_SCHEMA",
    "CCDE_FREEZE_STATUS",
    "freeze_binding",
    "load_ccde_freeze",
]
