"""Train-only pilot for prototype-aligned cross-modal binary codes.

This script is deliberately a development diagnostic rather than a paper
result.  It opens one sealed ``raw_rebuilt_neural_fit_artifact_v1`` (indT
only), makes a deterministic disjoint fit/query/database split inside indT,
and never accepts the formal runtime or Q/D labels.

The pilot asks one narrow question: does directly regressing both CLIP512
modalities to the same multi-label error-correcting codebook improve exact
Hamming retrieval enough to justify a neural-v2 implementation?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import numeric_sha256


BITS = (16, 32, 64)
RIDGE = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)


def _load_fit(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "raw_rebuilt_neural_fit_artifact_v1":
        raise ValueError("pilot accepts only a sealed raw-rebuilt indT fit artifact")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("fit artifact is incomplete")
    arrays = {}
    for name in ("image", "text", "labels", "identity_ids"):
        descriptor = manifest["arrays"][name]
        value = np.load(root / descriptor["path"], allow_pickle=False)
        if list(value.shape) != descriptor["shape"] or value.dtype.str != descriptor["dtype"]:
            raise ValueError(f"{name} geometry differs from its sealed manifest")
        if numeric_sha256(value) != descriptor["numeric_sha256"]:
            raise ValueError(f"{name} numeric content differs from its sealed manifest")
        arrays[name] = value
    labels = np.asarray(arrays["labels"], dtype=np.uint8)
    if not np.all((labels == 0) | (labels == 1)) or not np.all(labels.sum(axis=1) > 0):
        raise ValueError("labels must be nonempty binary multi-label rows")
    return (
        np.asarray(arrays["image"], dtype=np.float64),
        np.asarray(arrays["text"], dtype=np.float64),
        labels,
        np.asarray(arrays["identity_ids"], dtype=np.uint64),
        manifest,
    )


def _split(identity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets = np.empty(len(identity_ids), dtype=np.uint64)
    for index, identity in enumerate(identity_ids.tolist()):
        payload = f"semantic-codebook-dev-v1:{identity}".encode("ascii")
        buckets[index] = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    order = np.argsort(buckets, kind="stable")
    query_count = min(1_500, max(500, len(order) // 12))
    database_count = min(4_500, max(1_500, len(order) // 4))
    query = np.sort(order[:query_count])
    database = np.sort(order[query_count : query_count + database_count])
    fit = np.sort(order[query_count + database_count :])
    if len(fit) < 5_000:
        raise ValueError("internal fit split is too small")
    if np.intersect1d(query, database).size or np.intersect1d(query, fit).size:
        raise AssertionError("internal indT partitions overlap")
    return fit, query, database


def _prototype_codebook(label_dim: int, bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + bits * 10_007)
    candidates = rng.choice(np.array([-1.0, 1.0]), size=(4096, bits))
    candidates -= candidates.mean(axis=1, keepdims=True)
    candidates = np.where(candidates >= 0.0, 1.0, -1.0)
    chosen = [int(np.argmax(np.abs(candidates).sum(axis=1)))]
    while len(chosen) < label_dim:
        selected = candidates[np.asarray(chosen)]
        agreement = candidates @ selected.T
        worst_abs_correlation = np.max(np.abs(agreement), axis=1)
        worst_abs_correlation[np.asarray(chosen)] = np.inf
        balance = np.abs(candidates.sum(axis=0)[None, :] + selected.sum(axis=0)).mean(axis=1)
        score = worst_abs_correlation + 0.05 * balance
        chosen.append(int(np.argmin(score)))
    codebook = candidates[np.asarray(chosen)].astype(np.float64)
    # Deterministic per-column sign makes the aggregate prototype balanced.
    column_sign = np.where(codebook.sum(axis=0) > 0.0, -1.0, 1.0)
    return codebook * column_sign[None, :]


def _semantic_targets(labels: np.ndarray, codebook: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    inverse = 1.0 / np.sqrt(np.clip(prevalence, 1.0e-6, None))
    weighted = labels.astype(np.float64) * inverse[None, :]
    target = weighted @ codebook
    norm = np.maximum(1.0, np.abs(target).max(axis=1, keepdims=True))
    return target / norm


def _ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    mean_x = features.mean(axis=0)
    mean_y = targets.mean(axis=0)
    x = features - mean_x
    y = targets - mean_y
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += float(ridge)
    weight = np.linalg.solve(gram, x.T @ y)
    bias = mean_y - mean_x @ weight
    return weight, bias


def _pack(code: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(code > 0, dtype=np.uint8), axis=1, bitorder="little")


_POPCOUNT = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def _hamming(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    packed_q, packed_d = _pack(query), _pack(database)
    distances = np.empty((len(query), len(database)), dtype=np.uint16)
    for start in range(0, len(query), 128):
        block = np.bitwise_xor(packed_q[start : start + 128, None, :], packed_d[None, :, :])
        distances[start : start + len(block)] = _POPCOUNT[block].sum(axis=2, dtype=np.uint16)
    return distances


def _expected_metrics(distances: np.ndarray, query_labels: np.ndarray, database_labels: np.ndarray) -> dict[str, float]:
    maps, precisions, ndcgs = [], [], []
    if len(distances) != len(query_labels):
        raise ValueError("distance/query-label row count differs")
    for radius, qlabel in zip(distances, query_labels):
        relevance = (database_labels @ qlabel.astype(np.uint8)) > 0
        order = np.argsort(radius, kind="stable")
        ordered_radius = radius[order]
        ordered_rel = relevance[order].astype(np.float64)
        changes = np.r_[True, ordered_radius[1:] != ordered_radius[:-1]]
        starts = np.flatnonzero(changes)
        ends = np.r_[starts[1:], len(order)]
        sizes = ends - starts
        relevant = np.add.reduceat(ordered_rel, starts)
        probability = relevant / sizes
        previous = np.r_[0.0, np.cumsum(relevant)[:-1]]
        ranks = np.arange(1, len(order) + 1, dtype=np.float64)
        harmonic = np.r_[0.0, np.cumsum(1.0 / ranks)]
        harmonic_span = harmonic[ends] - harmonic[starts]
        within = np.zeros_like(relevant)
        multi = sizes > 1
        within[multi] = (relevant[multi] - 1.0) / (sizes[multi] - 1.0)
        ap_num = np.sum(
            probability
            * (
                (previous + 1.0) * harmonic_span
                + within * (sizes - (starts + 1.0) * harmonic_span)
            )
        )
        total_rel = int(relevance.sum())
        maps.append(float(ap_num / total_rel) if total_rel else 0.0)
        cutoff = min(50, len(order))
        take = np.clip(cutoff - starts, 0, sizes)
        rel_at = float(np.sum(take * probability))
        precisions.append(rel_at / cutoff)
        discount = 1.0 / np.log2(np.arange(2, len(order) + 2, dtype=np.float64))
        discounted = np.r_[0.0, np.cumsum(discount)]
        dcg = float(np.sum(probability * (discounted[starts + take] - discounted[starts])))
        ideal = float(discount[: min(total_rel, cutoff)].sum()) if total_rel else 0.0
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        "map_expected_ties": float(np.mean(maps)),
        "precision_at_50_expected_ties": float(np.mean(precisions)),
        "ndcg_at_50_expected_ties": float(np.mean(ndcgs)),
    }


def _threshold(train_image: np.ndarray, train_text: np.ndarray) -> np.ndarray:
    return np.median(np.concatenate((train_image, train_text), axis=0), axis=0)


def run(root: Path, ridge_values: Iterable[float], seed: int) -> dict:
    image, text, labels, identity_ids, manifest = _load_fit(root)
    fit, query, database = _split(identity_ids)
    prevalence = labels[fit].mean(axis=0)
    records = []
    for bits in BITS:
        codebook = _prototype_codebook(labels.shape[1], bits, seed)
        target = _semantic_targets(labels[fit], codebook, prevalence)
        for ridge in ridge_values:
            image_weight, image_bias = _ridge_fit(image[fit], target, ridge)
            text_weight, text_bias = _ridge_fit(text[fit], target, ridge)
            fit_image = image[fit] @ image_weight + image_bias
            fit_text = text[fit] @ text_weight + text_bias
            threshold = _threshold(fit_image, fit_text)
            query_image = image[query] @ image_weight + image_bias - threshold
            query_text = text[query] @ text_weight + text_bias - threshold
            database_image = image[database] @ image_weight + image_bias - threshold
            database_text = text[database] @ text_weight + text_bias - threshold
            i2t = _expected_metrics(_hamming(query_image, database_text), labels[query], labels[database])
            t2i = _expected_metrics(_hamming(query_text, database_image), labels[query], labels[database])
            records.append(
                {
                    "bits": bits,
                    "ridge": float(ridge),
                    "i2t": i2t,
                    "t2i": t2i,
                    "mean_map": 0.5 * (i2t["map_expected_ties"] + t2i["map_expected_ties"]),
                    "codebook_sha256": numeric_sha256(codebook),
                }
            )
    best = {}
    for bits in BITS:
        options = [record for record in records if record["bits"] == bits]
        best[str(bits)] = max(options, key=lambda item: (item["mean_map"], -item["ridge"]))
    return {
        "schema": "raw_rebuilt_semantic_codebook_indt_dev_pilot_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "labels_consumed": "indT_internal_fit_and_development_only",
        "formal_query_or_database_labels_opened": False,
        "split": {"fit": len(fit), "query": len(query), "database": len(database)},
        "split_hashes": {
            "fit_identity_sha256": numeric_sha256(identity_ids[fit]),
            "query_identity_sha256": numeric_sha256(identity_ids[query]),
            "database_identity_sha256": numeric_sha256(identity_ids[database]),
        },
        "seed": seed,
        "records": records,
        "best_by_bits": best,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=lambda value: tuple(float(x) for x in value.split(",")), default=RIDGE)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    result = run(args.fit.resolve(strict=True), args.ridge, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "best_by_bits": result["best_by_bits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
