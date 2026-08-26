"""Immutable streaming rank plans frozen before metric labels may be opened."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .codes import BITS, CodeState, open_code_state
from .integrity import (
    StreamingIntegrityError,
    atomic_write_json,
    load_json,
    production_code_inventory,
    reject_unsafe_output_path,
    require_disjoint_paths,
    require_hashed_json,
    require_no_link_components,
    sha256_json,
)


PLAN_SCHEMA = "raw_rebuilt_streaming_rank_plan_v1"
DIRECTIONS = ("i2t", "t2i")


@dataclass(frozen=True)
class StreamingPlanConfig:
    bits: tuple[int, ...] = BITS
    directions: tuple[str, ...] = DIRECTIONS
    query_chunk_size: int = 8
    cutoffs: tuple[int, ...] = (50, 100, 1000)

    def __post_init__(self) -> None:
        bits = tuple(int(value) for value in self.bits)
        directions = tuple(str(value) for value in self.directions)
        cutoffs = tuple(int(value) for value in self.cutoffs)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "cutoffs", cutoffs)
        if not bits or len(set(bits)) != len(bits) or any(value not in BITS for value in bits):
            raise ValueError(f"bits must be a unique nonempty subset of {BITS}")
        if not directions or len(set(directions)) != len(directions) or any(
            value not in DIRECTIONS for value in directions
        ):
            raise ValueError(f"directions must be a unique nonempty subset of {DIRECTIONS}")
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        normalized = tuple(sorted(set(cutoffs)))
        if normalized != cutoffs or not normalized or normalized[0] < 1:
            raise ValueError("cutoffs must be sorted, unique, and positive")


@dataclass(frozen=True)
class ChunkSpec:
    cell_id: str
    direction: str
    bits: int
    ordinal: int
    start: int
    end: int
    database_rows: int

    @property
    def name(self) -> str:
        return f"chunk-{self.start:09d}-{self.end:09d}"


@dataclass(frozen=True)
class FrozenRankPlan:
    root: Path
    manifest: Mapping[str, Any]

    @property
    def plan_sha256(self) -> str:
        return str(self.manifest["rank_plan_sha256"])

    def chunks(self) -> Iterator[ChunkSpec]:
        cells = self.manifest["cells"]
        maximum = max(len(cell["chunks"]) for cell in cells)
        # Interleave cells by query chunk. The metric-only process can compute
        # one Q-chunk's label truth once and reuse it across all six cells.
        for ordinal in range(maximum):
            for cell in cells:
                if ordinal >= len(cell["chunks"]):
                    continue
                chunk = cell["chunks"][ordinal]
                yield ChunkSpec(
                    cell_id=str(cell["cell_id"]),
                    direction=str(cell["direction"]),
                    bits=int(cell["bits"]),
                    ordinal=int(chunk["ordinal"]),
                    start=int(chunk["start"]),
                    end=int(chunk["end"]),
                    database_rows=int(cell["database_rows"]),
                )


def _cell_id(direction: str, bits: int) -> str:
    return f"{direction}-bits-{bits}"


def _cells(state: CodeState, config: StreamingPlanConfig) -> list[dict[str, Any]]:
    unavailable = sorted(set(config.bits) - set(state.available_bits))
    if unavailable:
        raise StreamingIntegrityError(
            f"rank plan requests unavailable packed code widths: {unavailable}"
        )
    query_rows = int(state.manifest["runtime_identity"]["query_rows"])
    database_rows = int(state.manifest["runtime_identity"]["database_rows"])
    result = []
    for direction in config.directions:
        for bits in config.bits:
            chunks = []
            ordinal = 0
            for start in range(0, query_rows, config.query_chunk_size):
                end = min(start + config.query_chunk_size, query_rows)
                chunks.append({"ordinal": ordinal, "start": start, "end": end})
                ordinal += 1
            result.append(
                {
                    "cell_id": _cell_id(direction, bits),
                    "direction": direction,
                    "bits": bits,
                    "query_rows": query_rows,
                    "database_rows": database_rows,
                    "distance_dtype": "uint8",
                    "database_order": "frozen runtime indD order",
                    "chunks": chunks,
                }
            )
    return result


def freeze_rank_plan(
    code_state_root: Path,
    output_parent: Path,
    *,
    config: StreamingPlanConfig = StreamingPlanConfig(),
) -> Path:
    """Freeze all cells/chunks without opening a label-bearing runtime boundary."""

    state = open_code_state(code_state_root)
    try:
        binding_body = {
            "schema": PLAN_SCHEMA,
            "code_state_manifest_sha256": state.manifest["manifest_sha256"],
            "encoding_binding_sha256": state.manifest["binding"]["encoding_binding_sha256"],
            "source_seal_sha256": state.manifest["source_seal_sha256"],
            "runtime_identity_sha256": state.manifest["runtime_identity"]["runtime_identity_sha256"],
            "config": {
                "bits": list(config.bits),
                "directions": list(config.directions),
                "query_chunk_size": config.query_chunk_size,
                "cutoffs": list(config.cutoffs),
            },
            "streaming_code_inventory": production_code_inventory(),
            "labels_loaded_during_freeze": False,
        }
        binding = {**binding_body, "plan_binding_sha256": sha256_json(binding_body)}
        cells = _cells(state, config)
        plan_body = {
            "schema": PLAN_SCHEMA,
            "status": "rank_state_frozen",
            "dataset": state.manifest["dataset"],
            "label_dim": state.manifest["label_dim"],
            "source_seal_sha256": state.manifest["source_seal_sha256"],
            "runtime_identity": state.manifest["runtime_identity"],
            "binding": binding,
            "cells": cells,
            "ranking_evidence": "canonical-database-order uint8 Hamming distances",
            "tie_policy": "uniform expected credit within equal-distance shells",
            "metric_label_boundary": "labels may open only after this complete plan verifies",
            "labels_loaded_during_freeze": False,
        }
        plan = {**plan_body, "rank_plan_sha256": sha256_json(plan_body)}
        output = reject_unsafe_output_path(Path(output_parent), field="rank plan output")
        root = reject_unsafe_output_path(
            output / f"plan-{plan['rank_plan_sha256'][:16]}",
            field="rank plan",
        )
        require_disjoint_paths(
            root, {"packed code state": Path(code_state_root)}, field="rank plan"
        )
        root.mkdir(parents=True, exist_ok=True)
        target = root / "rank_plan.json"
        if target.exists():
            existing = load_json(target)
            if existing != plan:
                raise StreamingIntegrityError("rank plan output was rebound")
        else:
            atomic_write_json(target, plan)
        return root
    finally:
        state.close()


def open_rank_plan(plan_root: Path, code_state_root: Path) -> FrozenRankPlan:
    """Return a verified handle; callers must not manufacture the metric gate dict."""

    declared = require_no_link_components(plan_root, field="rank plan")
    if declared.is_symlink() or not declared.is_dir():
        raise StreamingIntegrityError("rank plan must be a regular directory")
    root = declared.resolve(strict=True)
    plan_path = root / "rank_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise StreamingIntegrityError("rank plan must be a regular non-symlink file")
    plan = load_json(plan_path)
    require_hashed_json(
        plan,
        hash_field="rank_plan_sha256",
        schema=PLAN_SCHEMA,
        status="rank_state_frozen",
        field="streaming rank plan",
    )
    expected_plan_keys = {
        "schema",
        "status",
        "dataset",
        "label_dim",
        "source_seal_sha256",
        "runtime_identity",
        "binding",
        "cells",
        "ranking_evidence",
        "tie_policy",
        "metric_label_boundary",
        "labels_loaded_during_freeze",
        "rank_plan_sha256",
    }
    if set(plan) != expected_plan_keys:
        raise StreamingIntegrityError("rank plan contains unbound fields")
    if plan.get("labels_loaded_during_freeze") is not False:
        raise StreamingIntegrityError("rank plan crossed the label boundary")
    if (
        plan.get("ranking_evidence")
        != "canonical-database-order uint8 Hamming distances"
        or plan.get("tie_policy")
        != "uniform expected credit within equal-distance shells"
        or plan.get("metric_label_boundary")
        != "labels may open only after this complete plan verifies"
    ):
        raise StreamingIntegrityError("rank plan fixed evaluation policy changed")
    if sorted(item.relative_to(root).as_posix() for item in root.rglob("*")) != [
        "rank_plan.json"
    ]:
        raise StreamingIntegrityError("rank plan directory has unbound files")
    state = open_code_state(code_state_root)
    try:
        binding = plan.get("binding")
        if not isinstance(binding, Mapping):
            raise StreamingIntegrityError("rank plan has no binding")
        expected_binding_keys = {
            "schema",
            "code_state_manifest_sha256",
            "encoding_binding_sha256",
            "source_seal_sha256",
            "runtime_identity_sha256",
            "config",
            "streaming_code_inventory",
            "labels_loaded_during_freeze",
            "plan_binding_sha256",
        }
        if set(binding) != expected_binding_keys:
            raise StreamingIntegrityError("rank plan binding contains unbound fields")
        binding_body = {key: binding[key] for key in binding if key != "plan_binding_sha256"}
        if binding.get("plan_binding_sha256") != sha256_json(binding_body):
            raise StreamingIntegrityError("rank plan binding hash changed")
        if binding.get("code_state_manifest_sha256") != state.manifest["manifest_sha256"]:
            raise StreamingIntegrityError("rank plan is bound to another code state")
        if binding.get("encoding_binding_sha256") != state.manifest["binding"]["encoding_binding_sha256"]:
            raise StreamingIntegrityError("rank plan encoding binding changed")
        if (
            binding.get("schema") != PLAN_SCHEMA
            or binding.get("labels_loaded_during_freeze") is not False
            or binding.get("source_seal_sha256") != state.manifest["source_seal_sha256"]
            or binding.get("runtime_identity_sha256")
            != state.manifest["runtime_identity"]["runtime_identity_sha256"]
            or plan.get("dataset") != state.manifest["dataset"]
            or plan.get("label_dim") != state.manifest["label_dim"]
            or plan.get("source_seal_sha256") != state.manifest["source_seal_sha256"]
        ):
            raise StreamingIntegrityError("rank plan data/label-free binding changed")
        current = production_code_inventory()
        if binding.get("streaming_code_inventory") != current:
            raise StreamingIntegrityError("current streaming code differs from rank-plan code")
        if plan.get("runtime_identity") != state.manifest["runtime_identity"]:
            raise StreamingIntegrityError("rank plan runtime identity changed")
        config = binding.get("config")
        if not isinstance(config, Mapping) or set(config) != {
            "bits",
            "directions",
            "query_chunk_size",
            "cutoffs",
        }:
            raise StreamingIntegrityError("rank plan config is missing")
        expected = _cells(state, StreamingPlanConfig(**config))
        if plan.get("cells") != expected:
            raise StreamingIntegrityError("rank plan cell/chunk geometry changed")
        for cell in expected:
            if cell["distance_dtype"] != "uint8" or int(cell["bits"]) > 255:
                raise StreamingIntegrityError("rank plan distance representation is unsafe")
    finally:
        state.close()
    return FrozenRankPlan(root=root, manifest=plan)


__all__ = [
    "DIRECTIONS",
    "PLAN_SCHEMA",
    "ChunkSpec",
    "FrozenRankPlan",
    "StreamingPlanConfig",
    "freeze_rank_plan",
    "open_rank_plan",
]
