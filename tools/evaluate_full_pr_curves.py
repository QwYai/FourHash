#!/usr/bin/env python3
"""Compute complete-gallery, tie-safe precision--recall curves.

The evaluator compares the registered 64-bit fixed-feature baselines with the
frozen primary and ShellGuard codes on one sealed raw-rebuilt dataset. Curves
are macro-averaged over queries at 101 recall levels. A point is formed only
after a complete integer-distance tie block, so no random within-tie order is
introduced. The ``render`` command combines three receipts into the paper
figure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_ranking import _open_encoding_cache
from raw_rebuilt_runtime import load_label_free_rank_inputs, load_metric_labels
from raw_rebuilt_runtime.contract import atomic_write_json, numeric_sha256, sha256_file, sha256_json
from raw_rebuilt_streaming.codes import open_code_state
from tools.formal_ccde_streaming_eval import PLAN_SCHEMA


RESULT_SCHEMA = "shellguard_complete_gallery_pr_v1"
FIGURE_SCHEMA = "shellguard_complete_gallery_pr_figure_v1"
DRIVER_SCHEMA = "raw_rebuilt_baseline_streaming_driver_event_v1"
DATASETS = ("mirflickr", "nuswide", "mscoco")
DATASET_LABELS = {
    "mirflickr": "MIRFlickr-25K",
    "nuswide": "NUS-WIDE-TC21",
    "mscoco": "MS COCO",
}
DIRECTIONS = ("i2t", "t2i")
BASELINES = ("ucch-f", "dcmh-f-seminit", "cirh-f", "raneh-f")
CELL_PATTERN = re.compile(
    r"^(mirflickr|nuswide|mscoco)_(ucch-f|dcmh-f-seminit|cirh-f|raneh-f)_b(16|32|64)_s(\d+)$"
)
METHODS = ("primary", *BASELINES, "shellguard")
METHOD_LABELS = {
    "primary": "Primary",
    "ucch-f": "UCCH-F",
    "dcmh-f-seminit": "DCMH-F-SemInit",
    "cirh-f": "CIRH-F",
    "raneh-f": "RANEH-F",
    "shellguard": "ShellGuard",
}


class PREvaluationError(RuntimeError):
    """Frozen PR evidence is missing, inconsistent, or rebound."""


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PREvaluationError(f"expected one JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise PREvaluationError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, Mapping):
                raise PREvaluationError(f"event is not an object at {path}:{line_number}")
            result.append(value)
    return result


def verified_frozen_ccde_plan(plan_root: Path) -> Mapping[str, Any]:
    """Verify an immutable CCDE plan without rebinding it to today's source.

    Historical plans retain and hash their own neural, streaming, and formal
    implementation inventories.  A later baseline addition must not make the
    frozen codes unreadable, so this check validates the sealed inventories
    rather than requiring byte equality with the current checkout.
    """

    root = Path(plan_root).expanduser()
    if root.is_symlink() or not root.resolve(strict=True).is_dir():
        raise PREvaluationError("CCDE plan root must be a regular directory")
    root = root.resolve(strict=True)
    plan_path = root / "evaluation_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise PREvaluationError("CCDE evaluation plan is missing or linked")
    plan = _load_json(plan_path)
    body = {key: plan[key] for key in plan if key != "rank_plan_sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "rank_state_frozen"
        or plan.get("rank_plan_sha256") != sha256_json(body)
    ):
        raise PREvaluationError("CCDE evaluation plan hash changed")
    if (
        plan.get("labels_loaded_during_freeze") is not False
        or plan.get("formal_gate_or_fallback_used") is not False
        or plan.get("primary_shell_order_is_invariant") is not True
    ):
        raise PREvaluationError("CCDE frozen-rank invariants changed")
    binding = plan.get("binding")
    if not isinstance(binding, Mapping):
        raise PREvaluationError("CCDE plan binding is missing")
    binding_body = {
        key: binding[key] for key in binding if key != "plan_binding_sha256"
    }
    if binding.get("plan_binding_sha256") != sha256_json(binding_body):
        raise PREvaluationError("CCDE plan binding hash changed")
    for field in (
        "neural_code_inventory",
        "streaming_code_inventory",
        "implementation_inventory",
    ):
        inventory = binding.get(field)
        if not isinstance(inventory, Mapping) or not any(
            str(key).endswith("sha256") for key in inventory
        ):
            raise PREvaluationError(f"CCDE frozen {field} is missing")
    if not isinstance(plan.get("runtime_identity"), Mapping):
        raise PREvaluationError("CCDE runtime identity is missing")
    return plan


def _validated_event(event: Mapping[str, Any]) -> None:
    body = {key: event[key] for key in event if key != "event_sha256"}
    if (
        event.get("schema") != DRIVER_SCHEMA
        or event.get("event_sha256") != sha256_json(body)
    ):
        raise PREvaluationError("baseline evaluation event hash changed")


def select_baseline_code_states(
    event_logs: Sequence[Path],
    *,
    dataset: str,
    bits: int,
    seed: int,
) -> Mapping[str, Path]:
    """Select the last verified completed event for every registered baseline."""

    selected: dict[str, Path] = {}
    for event_log in event_logs:
        path = Path(event_log).expanduser().resolve(strict=True)
        for event in _load_jsonl(path):
            if event.get("event") != "cell_complete":
                continue
            _validated_event(event)
            match = CELL_PATTERN.fullmatch(str(event.get("cell", "")))
            if match is None:
                continue
            event_dataset = str(event.get("dataset", match.group(1)))
            method = str(event.get("method", match.group(2)))
            event_bits = int(event.get("bits", match.group(3)))
            event_seed = int(event.get("seed", match.group(4)))
            if (
                event_dataset != match.group(1)
                or method != match.group(2)
                or event_bits != int(match.group(3))
                or event_seed != int(match.group(4))
            ):
                raise PREvaluationError("baseline event identity fields disagree")
            if (
                event_dataset != dataset
                or event_bits != bits
                or event_seed != seed
                or method not in BASELINES
            ):
                continue
            state = Path(str(event.get("code_state", "")))
            if not state.is_absolute():
                raise PREvaluationError(f"{method} code-state path is not absolute")
            selected[method] = state.resolve(strict=True)
    missing = sorted(set(BASELINES) - set(selected))
    if missing:
        raise PREvaluationError(f"baseline PR inputs are missing: {missing}")
    return selected


def interpolated_pr_sum(
    total_histogram: np.ndarray,
    relevant_histogram: np.ndarray,
    recall_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return summed all-point interpolated precision and valid-query count.

    Histogram bins are ascending integer distances. Each cumulative point is
    therefore the end of a complete distance tie block.
    """

    total = np.asarray(total_histogram, dtype=np.int64)
    relevant = np.asarray(relevant_histogram, dtype=np.int64)
    grid = np.asarray(recall_grid, dtype=np.float64)
    if total.ndim != 2 or relevant.shape != total.shape:
        raise ValueError("histograms must be aligned two-dimensional arrays")
    if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("recall grid must be one-dimensional from zero to one")
    if np.any(total < 0) or np.any(relevant < 0) or np.any(relevant > total):
        raise ValueError("histogram counts are invalid")

    retrieved = np.cumsum(total, axis=1, dtype=np.int64)
    true_positive = np.cumsum(relevant, axis=1, dtype=np.int64)
    positives = relevant.sum(axis=1, dtype=np.int64)
    valid = positives > 0
    curve_sum = np.zeros(len(grid), dtype=np.float64)
    for row in np.flatnonzero(valid):
        precision = np.divide(
            true_positive[row],
            retrieved[row],
            out=np.zeros(total.shape[1], dtype=np.float64),
            where=retrieved[row] != 0,
        )
        recall = true_positive[row].astype(np.float64) / float(positives[row])
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, grid, side="left")
        present = positions < len(envelope)
        sampled = np.zeros(len(grid), dtype=np.float64)
        sampled[present] = envelope[positions[present]]
        curve_sum += sampled
    return curve_sum, int(valid.sum())


def _unpack_bipolar(value: np.ndarray, bits: int) -> np.ndarray:
    packed = np.asarray(value, dtype=np.uint8)
    if packed.ndim != 2 or packed.shape[1] * 8 < bits:
        raise PREvaluationError("packed code geometry is invalid")
    binary = np.unpackbits(packed, axis=1, count=bits, bitorder="little")
    return np.ascontiguousarray(binary.astype(np.int8) * 2 - 1)


@dataclass
class MethodCodes:
    query_image: torch.Tensor
    query_text: torch.Tensor
    database_image: torch.Tensor
    database_text: torch.Tensor
    detail_query_image: torch.Tensor | None = None
    detail_query_text: torch.Tensor | None = None
    detail_database_image: torch.Tensor | None = None
    detail_database_text: torch.Tensor | None = None

    def score(self, direction: str, start: int, end: int, *, detail_scale: int) -> torch.Tensor:
        if direction == "i2t":
            query = self.query_image[start:end]
            database = self.database_text
            detail_query = self.detail_query_image
            detail_database = self.detail_database_text
        elif direction == "t2i":
            query = self.query_text[start:end]
            database = self.database_image
            detail_query = self.detail_query_text
            detail_database = self.detail_database_image
        else:
            raise ValueError(f"unknown direction: {direction}")
        bits = int(query.shape[1])
        primary = torch.round((bits - query @ database.T) * 0.5).to(torch.int64)
        if detail_query is None or detail_database is None:
            return primary
        width = int(detail_query.shape[1])
        detail = torch.round(
            (width - detail_query[start:end] @ detail_database.T) * 0.5
        ).to(torch.int64)
        return primary * detail_scale + detail


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)


def _baseline_method_codes(state: Any, bits: int, device: torch.device) -> MethodCodes:
    return MethodCodes(
        query_image=_tensor(_unpack_bipolar(state.arrays[("query", "image", bits)], bits), device),
        query_text=_tensor(_unpack_bipolar(state.arrays[("query", "text", bits)], bits), device),
        database_image=_tensor(
            _unpack_bipolar(state.arrays[("database", "image", bits)], bits), device
        ),
        database_text=_tensor(
            _unpack_bipolar(state.arrays[("database", "text", bits)], bits), device
        ),
    )


def _ccde_method_codes(
    cache: Any,
    query_idx: np.ndarray,
    database_idx: np.ndarray,
    bits: int,
    device: torch.device,
) -> tuple[MethodCodes, MethodCodes, int]:
    primary = MethodCodes(
        query_image=_tensor(cache.primary_image_codes[bits][query_idx], device),
        query_text=_tensor(cache.primary_text_codes[bits][query_idx], device),
        database_image=_tensor(cache.primary_image_codes[bits][database_idx], device),
        database_text=_tensor(cache.primary_text_codes[bits][database_idx], device),
    )
    detail_width = int(cache.detail_image_codes[bits].shape[1])
    shellguard = MethodCodes(
        query_image=primary.query_image,
        query_text=primary.query_text,
        database_image=primary.database_image,
        database_text=primary.database_text,
        detail_query_image=_tensor(cache.detail_image_codes[bits][query_idx], device),
        detail_query_text=_tensor(cache.detail_text_codes[bits][query_idx], device),
        detail_database_image=_tensor(cache.detail_image_codes[bits][database_idx], device),
        detail_database_text=_tensor(cache.detail_text_codes[bits][database_idx], device),
    )
    return primary, shellguard, detail_width + 1


def _runtime_contract(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": "rank_state_frozen",
        "labels_loaded_during_freeze": False,
        "source_seal_sha256": identity["source_seal_sha256"],
    }


def _verify_runtime_alignment(
    identity: Mapping[str, Any],
    *,
    query_row_ids: np.ndarray,
    database_row_ids: np.ndarray,
) -> None:
    if numeric_sha256(query_row_ids) != identity.get("query_row_ids_numeric_sha256"):
        raise PREvaluationError("query row identity differs from frozen codes")
    if numeric_sha256(database_row_ids) != identity.get("database_row_ids_numeric_sha256"):
        raise PREvaluationError("database row identity differs from frozen codes")


def evaluate_dataset(
    *,
    dataset: str,
    runtime_root: Path,
    baseline_event_logs: Sequence[Path],
    ccde_plan_root: Path,
    output_root: Path,
    bits: int = 64,
    seed: int = 20260822,
    query_chunk_size: int = 64,
    device_name: str = "cuda",
) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if bits != 64:
        raise ValueError("paper PR protocol is frozen at 64 bits")
    if query_chunk_size < 1:
        raise ValueError("query chunk size must be positive")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise PREvaluationError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    plan_root = Path(ccde_plan_root).expanduser().resolve(strict=True)
    plan = verified_frozen_ccde_plan(plan_root)
    if plan.get("dataset") != dataset:
        raise PREvaluationError("CCDE plan belongs to another dataset")
    identity = plan["runtime_identity"]
    events = [Path(value).expanduser().resolve(strict=True) for value in baseline_event_logs]
    selected = select_baseline_code_states(events, dataset=dataset, bits=bits, seed=seed)

    rank = load_label_free_rank_inputs(runtime_root)
    labels = load_metric_labels(runtime_root, rank_contract=_runtime_contract(identity))
    states: dict[str, Any] = {}
    cache = None
    try:
        if rank.source_seal_sha256 != identity.get("source_seal_sha256"):
            raise PREvaluationError("runtime and CCDE source seals differ")
        if numeric_sha256(rank.query_idx) != identity.get("indQ_numeric_sha256"):
            raise PREvaluationError("runtime query split differs from CCDE plan")
        if numeric_sha256(rank.database_idx) != identity.get("indD_numeric_sha256"):
            raise PREvaluationError("runtime database split differs from CCDE plan")
        _verify_runtime_alignment(
            identity,
            query_row_ids=labels.query_row_ids,
            database_row_ids=labels.database_row_ids,
        )

        method_codes: dict[str, MethodCodes] = {}
        provenance: dict[str, Any] = {}
        for method, state_path in selected.items():
            state = open_code_state(state_path, require_current_code=False)
            states[method] = state
            binding = state.manifest["binding"]
            if (
                state.dataset != dataset
                or bits not in state.available_bits
                or binding.get("baseline_method") != method
                or int(binding.get("baseline_bits", -1)) != bits
                or int(binding.get("baseline_seed", -1)) != seed
                or state.manifest.get("runtime_identity") != identity
            ):
                raise PREvaluationError(f"{method} code state differs from the PR contract")
            _verify_runtime_alignment(
                state.manifest["runtime_identity"],
                query_row_ids=labels.query_row_ids,
                database_row_ids=labels.database_row_ids,
            )
            method_codes[method] = _baseline_method_codes(state, bits, device)
            manifest_path = state.root / "manifest.json"
            provenance[method] = {
                "code_state": str(state.root),
                "manifest_sha256": sha256_file(manifest_path),
                "encoding_binding_sha256": binding["encoding_binding_sha256"],
            }

        cache_relative = Path(str(plan["encoding_cache"]["path"]))
        cache_root = (plan_root / cache_relative).resolve(strict=True)
        cache_manifest_path = cache_root / "manifest.json"
        if sha256_file(cache_manifest_path) != plan["encoding_cache"]["manifest_sha256"]:
            raise PREvaluationError("CCDE encoding-cache manifest differs from its plan")
        cache_manifest = _load_json(cache_manifest_path)
        cache = _open_encoding_cache(cache_root, cache_manifest["binding"])
        primary, shellguard, detail_scale = _ccde_method_codes(
            cache,
            np.asarray(rank.query_idx, dtype=np.int64),
            np.asarray(rank.database_idx, dtype=np.int64),
            bits,
            device,
        )
        method_codes["primary"] = primary
        method_codes["shellguard"] = shellguard
        for method in ("primary", "shellguard"):
            provenance[method] = {
                "ccde_plan": str(plan_root),
                "evaluation_plan_file_sha256": sha256_file(
                    plan_root / "evaluation_plan.json"
                ),
                "rank_plan_sha256": plan["rank_plan_sha256"],
                "encoding_cache_manifest_sha256": sha256_file(cache_manifest_path),
                "frozen_neural_code_inventory_sha256": plan["binding"][
                    "neural_code_inventory"
                ]["code_inventory_sha256"],
                "frozen_streaming_code_inventory_sha256": plan["binding"][
                    "streaming_code_inventory"
                ]["code_inventory_sha256"],
            }

        if set(method_codes) != set(METHODS):
            raise AssertionError("method inventory changed unexpectedly")
        recall_grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
        sums = {
            (method, direction): np.zeros_like(recall_grid)
            for method in METHODS
            for direction in DIRECTIONS
        }
        counts = {(method, direction): 0 for method in METHODS for direction in DIRECTIONS}
        query_labels = _tensor(labels.query, device)
        database_labels = _tensor(labels.database, device)
        query_rows = int(len(labels.query))

        with torch.inference_mode():
            for direction in DIRECTIONS:
                for start in range(0, query_rows, query_chunk_size):
                    end = min(start + query_chunk_size, query_rows)
                    relevant = query_labels[start:end] @ database_labels.T > 0
                    for method in METHODS:
                        score = method_codes[method].score(
                            direction, start, end, detail_scale=detail_scale
                        )
                        bins = bits * detail_scale + detail_scale if method == "shellguard" else bits + 1
                        if int(score.min()) < 0 or int(score.max()) >= bins:
                            raise PREvaluationError(f"{method} produced an invalid distance")
                        offsets = torch.arange(
                            end - start, dtype=torch.int64, device=device
                        ).unsqueeze(1) * bins
                        flat_index = (score + offsets).reshape(-1)
                        total_hist = torch.bincount(
                            flat_index, minlength=(end - start) * bins
                        ).reshape(end - start, bins)
                        relevant_hist = torch.bincount(
                            flat_index[relevant.reshape(-1)],
                            minlength=(end - start) * bins,
                        ).reshape(end - start, bins)
                        curve_sum, valid = interpolated_pr_sum(
                            total_hist.cpu().numpy(),
                            relevant_hist.cpu().numpy(),
                            recall_grid,
                        )
                        sums[(method, direction)] += curve_sum
                        counts[(method, direction)] += valid

        curves: dict[str, Any] = {}
        for direction in DIRECTIONS:
            curves[direction] = {}
            for method in METHODS:
                count = counts[(method, direction)]
                if count < 1:
                    raise PREvaluationError(f"{method}/{direction} has no valid queries")
                precision = sums[(method, direction)] / float(count)
                curves[direction][method] = {
                    "label": METHOD_LABELS[method],
                    "valid_queries": count,
                    "precision": [float(value) for value in precision],
                    "aupr": float(np.trapezoid(precision, recall_grid)),
                }

        event_descriptors = [
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in events
        ]
        body = {
            "schema": RESULT_SCHEMA,
            "status": "COMPLETE",
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "bits": bits,
            "seed": seed,
            "directions": list(DIRECTIONS),
            "methods": list(METHODS),
            "method_labels": METHOD_LABELS,
            "query_rows": query_rows,
            "database_rows": int(len(labels.database)),
            "source_seal_sha256": labels.source_seal_sha256,
            "recall_grid": [float(value) for value in recall_grid],
            "curves": curves,
            "protocol": {
                "relevance": "at least one shared dataset label",
                "gallery": "complete registered database split",
                "distance": "integer Hamming; ShellGuard uses primary*(detail_bits+1)+detail",
                "ties": "curve points occur only after complete equal-distance blocks",
                "interpolation": "all-point precision envelope on 101 recall levels",
                "averaging": "macro average across queries with at least one relevant database item",
                "model_selection": "registered seed; no PR-based selection",
            },
            "baseline_event_logs": event_descriptors,
            "provenance": provenance,
        }
        result = {**body, "result_sha256": sha256_json(body)}
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        result_path = root / f"{dataset}_pr64_seed{seed}.json"
        atomic_write_json(result_path, result)
        csv_path = root / f"{dataset}_pr64_seed{seed}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "direction", "recall", *METHODS])
            for direction in DIRECTIONS:
                for index, recall in enumerate(recall_grid):
                    writer.writerow(
                        [
                            dataset,
                            direction,
                            f"{recall:.2f}",
                            *[
                                f"{curves[direction][method]['precision'][index]:.8f}"
                                for method in METHODS
                            ],
                        ]
                    )
        return result_path
    finally:
        if cache is not None:
            cache.close()
        for state in states.values():
            state.close()
        rank.close()


def _verified_result(path: Path) -> Mapping[str, Any]:
    value = _load_json(path)
    body = {key: value[key] for key in value if key != "result_sha256"}
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("status") != "COMPLETE"
        or value.get("result_sha256") != sha256_json(body)
    ):
        raise PREvaluationError(f"PR result hash changed: {path}")
    if value.get("methods") != list(METHODS):
        raise PREvaluationError(f"PR method inventory changed: {path}")
    return value


def render_figure(result_paths: Sequence[Path], output_stem: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    loaded = [_verified_result(Path(path)) for path in result_paths]
    values = {value["dataset"]: value for value in loaded}
    if set(values) != set(DATASETS):
        raise PREvaluationError(f"render requires exactly {DATASETS}")

    # Keep every method on the same visual grammar: a solid line plus a simple
    # geometric marker.  This avoids assigning an undocumented meaning to dash
    # patterns while remaining legible in grayscale.
    styles = {
        "primary": dict(color="#4d4d4d", marker="o"),
        "ucch-f": dict(color="#377eb8", marker="s"),
        "dcmh-f-seminit": dict(color="#984ea3", marker="^"),
        "cirh-f": dict(color="#e68613", marker="v"),
        "raneh-f": dict(color="#1b9e77", marker="D"),
        "shellguard": dict(color="#d62728", marker="p"),
    }
    # Pair the two retrieval directions horizontally, as is customary for PR
    # figures, and devote one row to each dataset.
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 5.45), sharex=True, sharey=True)
    handles = []
    labels = []
    for row, dataset in enumerate(DATASETS):
        result = values[dataset]
        recall = np.asarray(result["recall_grid"], dtype=np.float64)
        for column, direction in enumerate(DIRECTIONS):
            axis = axes[row, column]
            direction_label = (
                "Image $\\to$ Text" if direction == "i2t" else "Text $\\to$ Image"
            )
            for method in METHODS:
                precision = np.asarray(
                    result["curves"][direction][method]["precision"], dtype=np.float64
                )
                (line,) = axis.plot(
                    recall,
                    precision,
                    linestyle="-",
                    linewidth=1.15 if method != "shellguard" else 1.65,
                    markersize=2.8 if method != "shellguard" else 3.2,
                    markevery=(4, 12),
                    markerfacecolor="white" if method != "shellguard" else styles[method]["color"],
                    markeredgewidth=0.75,
                    solid_capstyle="round",
                    zorder=3 if method == "shellguard" else 2,
                    label=METHOD_LABELS[method],
                    **styles[method],
                )
                if row == 0 and column == 0:
                    handles.append(line)
                    labels.append(METHOD_LABELS[method])
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.30, 1.005)
            axis.set_xticks(np.linspace(0, 1, 6))
            axis.set_yticks(np.arange(0.4, 1.01, 0.2))
            axis.grid(axis="y", color="#dedede", linewidth=0.35, alpha=0.55)
            axis.tick_params(
                labelsize=7,
                direction="in",
                top=True,
                right=True,
                length=2.5,
                width=0.55,
            )
            for spine in axis.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.55)
            axis.set_title(
                f"({chr(97 + row * 2 + column)}) {result['dataset_label']}: "
                f"{direction_label}",
                fontsize=8,
                pad=2.5,
            )
            if row == len(DATASETS) - 1:
                axis.set_xlabel("Recall", fontsize=8)
            if column == 0:
                axis.set_ylabel("Precision", fontsize=8)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#777777",
        fontsize=7.0,
        handlelength=1.9,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.3,
        bbox_to_anchor=(0.5, 0.998),
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.075,
        top=0.915,
        wspace=0.12,
        hspace=0.30,
    )
    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    manifest_body = {
        "schema": FIGURE_SCHEMA,
        "status": "COMPLETE",
        "inputs": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
            for path in result_paths
        ],
        "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
        "layout": "three dataset rows by two retrieval-direction columns",
        "methods": list(METHODS),
    }
    atomic_write_json(
        stem.with_name(stem.name + "_manifest.json"),
        {**manifest_body, "manifest_sha256": sha256_json(manifest_body)},
    )
    return pdf_path, png_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one dataset")
    evaluate.add_argument("--dataset", choices=DATASETS, required=True)
    evaluate.add_argument("--runtime", type=Path, required=True)
    evaluate.add_argument("--baseline-events", type=Path, action="append", required=True)
    evaluate.add_argument("--ccde-plan", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--bits", type=int, default=64)
    evaluate.add_argument("--seed", type=int, default=20260822)
    evaluate.add_argument("--query-chunk-size", type=int, default=64)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    render = subparsers.add_parser("render", help="render the three-dataset figure")
    render.add_argument("--result", type=Path, action="append", required=True)
    render.add_argument("--output-stem", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evaluate":
        path = evaluate_dataset(
            dataset=args.dataset,
            runtime_root=args.runtime,
            baseline_event_logs=args.baseline_events,
            ccde_plan_root=args.ccde_plan,
            output_root=args.output_root,
            bits=args.bits,
            seed=args.seed,
            query_chunk_size=args.query_chunk_size,
            device_name=args.device,
        )
        print(json.dumps({"result": str(path)}, sort_keys=True))
        return 0
    if len(args.result) != len(DATASETS):
        raise PREvaluationError("render requires three --result arguments")
    pdf_path, png_path = render_figure(args.result, args.output_stem)
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
