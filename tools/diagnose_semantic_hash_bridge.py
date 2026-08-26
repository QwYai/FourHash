"""Explore a train-calibrated neural-posterior-to-binary semantic bridge.

The diagnostic loads one receipt-verified CCDE detail network, calibrates a
single posterior threshold on ``indT`` only, and converts label posteriors to
binary semantic codes with deterministic one-bit MinHash and SimHash maps.
It then opens formal labels to measure a deterministic query prefix.  Because
the architecture was motivated after prior formal results were observed, the
output is exploratory and cannot be cited as a fresh formal test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.ccde_training import load_detail_checkpoint
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from raw_rebuilt_streaming.metrics import (
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
    mean_query_metrics,
)


BITS = (16, 32, 64)
DIRECTIONS = ("i2t", "t2i")
METRIC_KEYS = (
    "map_expected_ties",
    "binary_ndcg_at_50_expected_ties",
    "j_ndcg_at_50_expected_ties",
)


def _indices(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1:
        raise ValueError(f"{path.name} must be a vector")
    if value.dtype == np.bool_ or (
        len(value) == rows and np.all(np.isin(value, (0, 1)))
    ):
        result = np.flatnonzero(value)
    elif value.dtype.kind in "iu":
        result = value.astype(np.int64, copy=False)
    else:
        raise ValueError(f"{path.name} is not an index vector")
    if len(result) == 0 or np.any(result < 0) or np.any(result >= rows):
        raise ValueError(f"{path.name} contains invalid indices")
    if np.unique(result).size != len(result):
        raise ValueError(f"{path.name} contains duplicate indices")
    return np.ascontiguousarray(result, dtype=np.int64)


def _parse_ints(raw: str, *, allow_zero: bool) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    lower = 0 if allow_zero else 1
    if not result or result[0] < lower:
        raise ValueError(f"values must be integers >= {lower}")
    return result


def _parse_floats(raw: str) -> tuple[float, ...]:
    result = tuple(sorted({float(item.strip()) for item in raw.split(",") if item.strip()}))
    if not result or not all(0.0 < value < 1.0 for value in result):
        raise ValueError("thresholds must lie strictly inside (0,1)")
    return result


def _parse_strings(raw: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not result:
        raise ValueError("families must be nonempty")
    return result


def _bipolar(path: Path, rows: int, columns: int) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (rows, columns) or value.dtype != np.int8:
        raise ValueError(f"invalid bipolar code geometry in {path.name}")
    if not np.all(np.isin(value, (-1, 1))):
        raise ValueError(f"{path.name} is not bipolar")
    return value


def _encode_posteriors(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    modality: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = len(features)
    label_dim = int(model.label_dim)
    result = np.empty((rows, label_dim), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            end = min(rows, start + batch_size)
            batch = torch.from_numpy(
                np.array(features[start:end], dtype=np.float32, copy=True, order="C")
            ).to(device)
            posterior = model(batch, modality).posterior_heads.mean(dim=1)
            result[start:end] = posterior.cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("neural posteriors are not finite probabilities")
    return result


def _hard_sets(posterior: np.ndarray, threshold: float) -> np.ndarray:
    active = np.asarray(posterior >= float(threshold), dtype=bool)
    missing = np.flatnonzero(~active.any(axis=1))
    if missing.size:
        active[missing, np.argmax(posterior[missing], axis=1)] = True
    return active


def _mean_set_jaccard(active: np.ndarray, labels: np.ndarray) -> float:
    truth = labels.astype(bool, copy=False)
    intersection = np.logical_and(active, truth).sum(axis=1, dtype=np.int64)
    union = np.logical_or(active, truth).sum(axis=1, dtype=np.int64)
    return float(np.mean(intersection / np.maximum(union, 1), dtype=np.float64))


def _calibrate_threshold(
    image_posterior: np.ndarray,
    text_posterior: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    thresholds: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    train_labels = np.ascontiguousarray(labels[train_idx], dtype=np.uint8)
    rows = []
    for threshold in thresholds:
        image_score = _mean_set_jaccard(
            _hard_sets(image_posterior[train_idx], threshold), train_labels
        )
        text_score = _mean_set_jaccard(
            _hard_sets(text_posterior[train_idx], threshold), train_labels
        )
        rows.append(
            {
                "threshold": float(threshold),
                "image_train_jaccard": image_score,
                "text_train_jaccard": text_score,
                "mean_train_jaccard": 0.5 * (image_score + text_score),
            }
        )
    winner = max(rows, key=lambda row: (row["mean_train_jaccard"], row["threshold"]))
    return float(winner["threshold"]), rows


def _one_bit_minhash(active: np.ndarray, bits: int, seed: int) -> np.ndarray:
    rows, labels = active.shape
    rng = np.random.default_rng(seed + 1009 * bits)
    result = np.empty((rows, bits), dtype=np.int8)
    inactive_rank = labels + 1
    for bit in range(bits):
        permutation = rng.permutation(labels)
        rank = np.empty(labels, dtype=np.int16)
        rank[permutation] = np.arange(labels, dtype=np.int16)
        color = rng.integers(0, 2, size=labels, dtype=np.int8)
        chosen = np.argmin(np.where(active, rank[None, :], inactive_rank), axis=1)
        result[:, bit] = np.where(color[chosen] == 1, 1, -1)
    return result


def _simhash(
    posterior: np.ndarray,
    prevalence: np.ndarray,
    bits: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 2029 * bits)
    projection = rng.standard_normal((posterior.shape[1], bits), dtype=np.float32)
    centered = posterior - prevalence[None, :]
    value = centered @ projection
    return np.where(value >= 0.0, 1, -1).astype(np.int8)


def _rank_keys_descending(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    order = np.argsort(-value, kind="stable")
    ordered = value[order]
    group = np.zeros(len(value), dtype=np.int32)
    if len(value) > 1:
        group[1:] = np.cumsum(ordered[1:] != ordered[:-1], dtype=np.int32)
    result = np.empty(len(value), dtype=np.int32)
    result[order] = group
    return result


def _distance(query: torch.Tensor, database: torch.Tensor, bits: int) -> torch.Tensor:
    return torch.round((bits - query @ database.T) * 0.5).to(torch.int16)


def diagnose(
    cache_root: Path,
    runtime_root: Path,
    checkpoint_path: Path,
    architecture_freeze_path: Path,
    output_path: Path,
    *,
    query_limit: int,
    query_batch_size: int,
    inference_batch_size: int,
    multipliers: tuple[int, ...],
    thresholds: tuple[float, ...],
    families: tuple[str, ...],
    seed: int,
    device: str,
) -> dict[str, Any]:
    cache = cache_root.expanduser().resolve(strict=True)
    runtime = runtime_root.expanduser().resolve(strict=True)
    arrays = runtime / "arrays"
    labels = np.load(arrays / "labels.npy", mmap_mode="r", allow_pickle=False)
    image_features = np.load(
        arrays / "image_features_clip512.npy", mmap_mode="r", allow_pickle=False
    )
    text_features = np.load(
        arrays / "text_features_clip512.npy", mmap_mode="r", allow_pickle=False
    )
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("sealed labels must be a uint8 matrix")
    rows, label_dim = labels.shape
    if image_features.shape != (rows, 512) or text_features.shape != (rows, 512):
        raise ValueError("feature geometry differs from the label rows")
    train_idx = _indices(arrays / "indT.npy", rows)
    query_idx = _indices(arrays / "indQ.npy", rows)
    database_idx = _indices(arrays / "indD.npy", rows)
    if query_limit < 1 or query_batch_size < 1 or inference_batch_size < 1:
        raise ValueError("batch sizes and query limit must be positive")
    query_idx = query_idx[: min(query_limit, len(query_idx))]
    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    loaded = load_detail_checkpoint(
        checkpoint_path,
        architecture_freeze_path,
        device=resolved,
        require_current_code=False,
    )
    if int(loaded.model.label_dim) != label_dim:
        raise ValueError("checkpoint label dimension differs from sealed labels")
    image_posterior = _encode_posteriors(
        loaded.model,
        image_features,
        modality="image",
        device=resolved,
        batch_size=inference_batch_size,
    )
    text_posterior = _encode_posteriors(
        loaded.model,
        text_features,
        modality="text",
        device=resolved,
        batch_size=inference_batch_size,
    )
    threshold, calibration = _calibrate_threshold(
        image_posterior, text_posterior, labels, train_idx, thresholds
    )
    hard = {
        "image": _hard_sets(image_posterior, threshold),
        "text": _hard_sets(text_posterior, threshold),
    }
    posterior = {"image": image_posterior, "text": text_posterior}
    prevalence = np.asarray(labels[train_idx].mean(axis=0), dtype=np.float32)
    semantic_codes: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "one_bit_minhash": {},
        "one_bit_minhash_fixed16": {},
        "posterior_simhash": {},
    }
    fixed_minhash = {
        modality: _one_bit_minhash(hard[modality], 16, seed)
        for modality in ("image", "text")
    }
    semantic_widths: dict[str, dict[int, int]] = {
        "one_bit_minhash": {},
        "one_bit_minhash_fixed16": {},
        "posterior_simhash": {},
    }
    for bits in BITS:
        semantic_codes["one_bit_minhash"][bits] = {
            modality: _one_bit_minhash(hard[modality], bits, seed)
            for modality in ("image", "text")
        }
        semantic_widths["one_bit_minhash"][bits] = bits
        semantic_codes["one_bit_minhash_fixed16"][bits] = fixed_minhash
        semantic_widths["one_bit_minhash_fixed16"][bits] = 16
        semantic_codes["posterior_simhash"][bits] = {
            modality: _simhash(posterior[modality], prevalence, bits, seed)
            for modality in ("image", "text")
        }
        semantic_widths["posterior_simhash"][bits] = bits
    unknown_families = sorted(set(families).difference(semantic_codes))
    if unknown_families:
        raise ValueError(f"unknown semantic-code families: {unknown_families}")
    semantic_codes = {family: semantic_codes[family] for family in families}
    semantic_widths = {family: semantic_widths[family] for family in families}

    query_labels_all = torch.from_numpy(
        np.ascontiguousarray(labels[query_idx], dtype=np.float32)
    ).to(resolved)
    database_labels = torch.from_numpy(
        np.ascontiguousarray(labels[database_idx], dtype=np.float32)
    ).to(resolved)
    database_cardinality = database_labels.sum(dim=1)
    prefixes = build_metric_prefixes(len(database_idx), (50, 100, 1000))
    records: dict[tuple[int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    posterior_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for bits in BITS:
        for direction in DIRECTIONS:
            query_modality, database_modality = (
                ("image", "text") if direction == "i2t" else ("text", "image")
            )
            primary_query_all = _bipolar(
                cache / f"primary_{query_modality}_codes_{bits}.npy", rows, bits
            )
            primary_database_all = _bipolar(
                cache / f"primary_{database_modality}_codes_{bits}.npy", rows, bits
            )
            primary_database = torch.from_numpy(
                np.ascontiguousarray(primary_database_all[database_idx], dtype=np.float32)
            ).to(resolved)
            family_database = {
                family: torch.from_numpy(
                    np.ascontiguousarray(codes[bits][database_modality][database_idx], dtype=np.float32)
                ).to(resolved)
                for family, codes in semantic_codes.items()
            }
            posterior_database = torch.from_numpy(
                np.ascontiguousarray(posterior[database_modality][database_idx], dtype=np.float32)
            ).to(resolved)
            hard_database = torch.from_numpy(
                np.ascontiguousarray(hard[database_modality][database_idx], dtype=np.float32)
            ).to(resolved)
            hard_database_cardinality = hard_database.sum(dim=1)
            for start in range(0, len(query_idx), query_batch_size):
                end = min(len(query_idx), start + query_batch_size)
                canonical = query_idx[start:end]
                primary_query = torch.from_numpy(
                    np.ascontiguousarray(primary_query_all[canonical], dtype=np.float32)
                ).to(resolved)
                primary_distance = _distance(primary_query, primary_database, bits)
                family_distances = {}
                for family, codes in semantic_codes.items():
                    semantic_width = semantic_widths[family][bits]
                    semantic_query = torch.from_numpy(
                        np.ascontiguousarray(codes[bits][query_modality][canonical], dtype=np.float32)
                    ).to(resolved)
                    family_distances[family] = _distance(
                        semantic_query, family_database[family], semantic_width
                    )
                posterior_query = torch.from_numpy(
                    np.ascontiguousarray(posterior[query_modality][canonical], dtype=np.float32)
                ).to(resolved)
                soft_intersection = posterior_query @ posterior_database.T
                soft_union = (
                    posterior_query.sum(dim=1)[:, None]
                    + posterior_database.sum(dim=1)[None, :]
                    - soft_intersection
                )
                soft_jaccard = soft_intersection / soft_union.clamp_min(1.0e-7)
                hard_query = torch.from_numpy(
                    np.ascontiguousarray(hard[query_modality][canonical], dtype=np.float32)
                ).to(resolved)
                hard_intersection = hard_query @ hard_database.T
                hard_union = (
                    hard_query.sum(dim=1)[:, None]
                    + hard_database_cardinality[None, :]
                    - hard_intersection
                )
                hard_jaccard = hard_intersection / hard_union.clamp_min(1.0)
                query_labels = query_labels_all[start:end]
                true_intersection = query_labels @ database_labels.T
                true_union = (
                    query_labels.sum(dim=1)[:, None]
                    + database_cardinality[None, :]
                    - true_intersection
                )
                true_gain = true_intersection / true_union.clamp_min(1.0)

                primary_np = primary_distance.cpu().numpy().astype(np.int32, copy=False)
                family_np = {
                    family: value.cpu().numpy().astype(np.int32, copy=False)
                    for family, value in family_distances.items()
                }
                soft_np = soft_jaccard.cpu().numpy()
                hard_np = hard_jaccard.cpu().numpy()
                gain_np = true_gain.cpu().numpy().astype(np.float64, copy=False)
                for offset in range(end - start):
                    row_gain = np.ascontiguousarray(gain_np[offset], dtype=np.float64)
                    relevance = row_gain > 0.0
                    for family, distance_matrix in family_np.items():
                        semantic_width = semantic_widths[family][bits]
                        for multiplier in multipliers:
                            key = multiplier * primary_np[offset] + distance_matrix[offset]
                            metric = expected_tie_metrics_from_distances(
                                relevance,
                                key,
                                bits=bits,
                                graded_gains=row_gain,
                                cutoffs=(50, 100, 1000),
                                prefixes=prefixes,
                                distance_levels=(
                                    multiplier * bits + semantic_width + 1
                                ),
                            )
                            records[(bits, direction, family, multiplier)].append(metric)
                    if bits == 64:
                        for family, score in (
                            ("soft_posterior_jaccard", soft_np[offset]),
                            ("hard_posterior_jaccard", hard_np[offset]),
                        ):
                            rank_key = _rank_keys_descending(score)
                            metric = expected_tie_metrics_from_distances(
                                relevance,
                                rank_key,
                                bits=bits,
                                graded_gains=row_gain,
                                cutoffs=(50, 100, 1000),
                                prefixes=prefixes,
                                distance_levels=max(len(database_idx), bits + 1),
                            )
                            posterior_records[(direction, family)].append(metric)

    summaries = []
    for (bits, direction, family, multiplier), cell_records in sorted(records.items()):
        summaries.append(
            {
                "bits": bits,
                "direction": direction,
                "semantic_code": family,
                "semantic_bits": semantic_widths[family][bits],
                "primary_multiplier": multiplier,
                "primary_shell_invariant": (
                    multiplier > semantic_widths[family][bits]
                ),
                "metrics": mean_query_metrics(cell_records),
            }
        )
    posterior_summaries = [
        {
            "direction": direction,
            "ranker": family,
            "metrics": mean_query_metrics(cell_records),
        }
        for (direction, family), cell_records in sorted(posterior_records.items())
    ]
    best = []
    for bits in BITS:
        for direction in DIRECTIONS:
            candidates = [
                row for row in summaries if row["bits"] == bits and row["direction"] == direction
            ]
            for metric in METRIC_KEYS:
                winner = max(candidates, key=lambda row: float(row["metrics"][metric]))
                best.append(
                    {
                        "bits": bits,
                        "direction": direction,
                        "metric": metric,
                        "semantic_code": winner["semantic_code"],
                        "primary_multiplier": winner["primary_multiplier"],
                        "value": float(winner["metrics"][metric]),
                    }
                )

    body: dict[str, Any] = {
        "schema": "ccde_semantic_hash_bridge_postformal_diagnostic_v1",
        "status": "EXPLORATORY_LABEL_OPEN_NOT_FORMAL",
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "runtime_root": str(runtime),
        "checkpoint_path": str(checkpoint_path.expanduser().resolve(strict=True)),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "architecture_freeze_path": str(
            architecture_freeze_path.expanduser().resolve(strict=True)
        ),
        "architecture_freeze_file_sha256": sha256_file(
            architecture_freeze_path.expanduser().resolve(strict=True)
        ),
        "query_prefix_rows": len(query_idx),
        "database_rows": len(database_idx),
        "train_rows_used_for_threshold_only": len(train_idx),
        "threshold_candidates": list(thresholds),
        "selected_threshold": threshold,
        "threshold_calibration": calibration,
        "multipliers": list(multipliers),
        "semantic_code_families": list(families),
        "device": str(resolved),
        "summaries": summaries,
        "posterior_ranker_diagnostic_ceiling": posterior_summaries,
        "best_by_cell_metric": best,
        "scientific_boundary": (
            "The posterior threshold is fit on indT only, but the bridge architecture was "
            "created after earlier formal metrics were known. This output is diagnostic, not "
            "a fresh label-isolated formal test."
        ),
    }
    result = {**body, "diagnostic_sha256": sha256_json(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--multipliers", default="0,1,2,4,8,17,33,65")
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument(
        "--families",
        default="one_bit_minhash,one_bit_minhash_fixed16,posterior_simhash",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = diagnose(
        args.cache_root,
        args.runtime_root,
        args.checkpoint,
        args.architecture_freeze,
        args.output,
        query_limit=args.query_limit,
        query_batch_size=args.query_batch_size,
        inference_batch_size=args.inference_batch_size,
        multipliers=_parse_ints(args.multipliers, allow_zero=True),
        thresholds=_parse_floats(args.thresholds),
        families=_parse_strings(args.families),
        seed=args.seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": result["selected_threshold"],
                "query_prefix_rows": result["query_prefix_rows"],
                "diagnostic_sha256": result["diagnostic_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
