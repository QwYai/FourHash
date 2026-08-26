"""Train-calibrated neural-posterior semantic codes for ShellGuard.

The bridge converts a neural multi-label posterior into a nonempty predicted
label set and then into a fixed-width one-bit MinHash code.  The database keeps
only the resulting binary code; posterior vectors are transient.  A mixed-radix
distance uses the semantic code only inside an equal primary Hamming shell.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch


SEMANTIC_BRIDGE_SCHEMA = "shellguard_semantic_bridge_v1"


@dataclass(frozen=True)
class SemanticBridgeConfig:
    """One dataset-independent bridge rule.

    The threshold value is selected per dataset from ``threshold_candidates``
    using training identities only.  The code budget and random-map seed are
    shared across datasets and primary widths.
    """

    detail_bits: int = 16
    threshold_candidates: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    )
    minhash_seed: int = 20260822

    def __post_init__(self) -> None:
        if type(self.detail_bits) is not int or self.detail_bits < 1:
            raise ValueError("detail_bits must be a positive integer")
        if type(self.minhash_seed) is not int or self.minhash_seed < 0:
            raise ValueError("minhash_seed must be a nonnegative integer")
        if not self.threshold_candidates:
            raise ValueError("threshold_candidates must be nonempty")
        if tuple(sorted(set(self.threshold_candidates))) != self.threshold_candidates:
            raise ValueError("threshold_candidates must be unique and increasing")
        if not all(
            math.isfinite(value) and 0.0 < value < 1.0
            for value in self.threshold_candidates
        ):
            raise ValueError("threshold candidates must lie inside (0,1)")


@dataclass(frozen=True)
class OneBitMinHashMap:
    """Deterministic ranks and Rademacher colors for one-bit MinHash."""

    ranks: np.ndarray
    colors: np.ndarray
    seed: int

    @property
    def bits(self) -> int:
        return int(self.ranks.shape[0])

    @property
    def label_dim(self) -> int:
        return int(self.ranks.shape[1])


def _posterior_matrix(value: np.ndarray, *, field: str) -> np.ndarray:
    posterior = np.asarray(value)
    if posterior.ndim != 2 or posterior.shape[0] < 1 or posterior.shape[1] < 2:
        raise ValueError(f"{field} must be a nonempty [rows,labels] matrix")
    if posterior.dtype.kind != "f":
        raise TypeError(f"{field} must use a floating dtype")
    if not np.isfinite(posterior).all() or np.any(posterior < 0.0) or np.any(
        posterior > 1.0
    ):
        raise ValueError(f"{field} must contain finite probabilities in [0,1]")
    return posterior


def posterior_to_active_set(posterior: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold posteriors with a deterministic top-one nonempty fallback."""

    value = _posterior_matrix(posterior, field="posterior")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie inside (0,1)")
    active = np.asarray(value >= float(threshold), dtype=bool)
    missing = np.flatnonzero(~active.any(axis=1))
    if missing.size:
        active[missing, np.argmax(value[missing], axis=1)] = True
    if not bool(active.any(axis=1).all()):
        raise AssertionError("posterior active-set fallback failed")
    return np.ascontiguousarray(active, dtype=bool)


def mean_label_set_jaccard(active: np.ndarray, labels: np.ndarray) -> float:
    """Mean row-wise Jaccard between predicted and binary training label sets."""

    predicted = np.asarray(active)
    truth = np.asarray(labels)
    if predicted.ndim != 2 or predicted.dtype != np.bool_:
        raise TypeError("active must be a boolean matrix")
    if truth.shape != predicted.shape or truth.dtype.kind not in "biu":
        raise ValueError("labels must be a binary matrix aligned with active")
    if not np.all(np.isin(truth, (0, 1))) or np.any(truth.sum(axis=1) == 0):
        raise ValueError("labels must contain nonempty binary rows")
    intersection = np.logical_and(predicted, truth.astype(bool, copy=False)).sum(
        axis=1, dtype=np.int64
    )
    union = np.logical_or(predicted, truth.astype(bool, copy=False)).sum(
        axis=1, dtype=np.int64
    )
    return float(np.mean(intersection / np.maximum(union, 1), dtype=np.float64))


def calibrate_training_threshold(
    image_posterior: np.ndarray,
    text_posterior: np.ndarray,
    train_labels: np.ndarray,
    candidates: Sequence[float],
) -> tuple[float, tuple[dict[str, float], ...]]:
    """Select one threshold from training identities only.

    The objective is the mean of image and text label-set Jaccard.  Exact ties
    choose the larger threshold, yielding the sparser predicted knowledge set.
    """

    image = _posterior_matrix(image_posterior, field="image_posterior")
    text = _posterior_matrix(text_posterior, field="text_posterior")
    labels = np.asarray(train_labels)
    if image.shape != text.shape or labels.shape != image.shape:
        raise ValueError("posterior and training-label matrices must align")
    values = tuple(float(value) for value in candidates)
    if not values or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in values
    ):
        raise ValueError("candidate thresholds must lie inside (0,1)")
    if tuple(sorted(set(values))) != values:
        raise ValueError("candidate thresholds must be unique and increasing")
    rows = []
    for threshold in values:
        image_score = mean_label_set_jaccard(
            posterior_to_active_set(image, threshold), labels
        )
        text_score = mean_label_set_jaccard(
            posterior_to_active_set(text, threshold), labels
        )
        rows.append(
            {
                "threshold": threshold,
                "image_train_jaccard": image_score,
                "text_train_jaccard": text_score,
                "mean_train_jaccard": 0.5 * (image_score + text_score),
            }
        )
    winner = max(rows, key=lambda row: (row["mean_train_jaccard"], row["threshold"]))
    return float(winner["threshold"]), tuple(rows)


def build_one_bit_minhash_map(
    label_dim: int,
    *,
    bits: int,
    seed: int,
) -> OneBitMinHashMap:
    """Build independent label permutations and Rademacher colors."""

    if type(label_dim) is not int or label_dim < 2:
        raise ValueError("label_dim must be an integer greater than one")
    if type(bits) is not int or bits < 1:
        raise ValueError("bits must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed + 1009 * bits)
    ranks = np.empty((bits, label_dim), dtype=np.int16)
    colors = np.empty((bits, label_dim), dtype=np.int8)
    for bit in range(bits):
        permutation = rng.permutation(label_dim)
        ranks[bit, permutation] = np.arange(label_dim, dtype=np.int16)
        colors[bit] = np.where(
            rng.integers(0, 2, size=label_dim, dtype=np.int8) == 1,
            1,
            -1,
        )
    ranks.setflags(write=False)
    colors.setflags(write=False)
    return OneBitMinHashMap(ranks=ranks, colors=colors, seed=seed)


def encode_active_set_minhash(
    active: np.ndarray,
    mapping: OneBitMinHashMap,
) -> np.ndarray:
    """Encode nonempty label sets as bipolar one-bit MinHash coordinates."""

    value = np.asarray(active)
    if value.ndim != 2 or value.dtype != np.bool_:
        raise TypeError("active must be a boolean matrix")
    if value.shape[1] != mapping.label_dim or not bool(value.any(axis=1).all()):
        raise ValueError("active geometry or nonempty-set contract differs")
    result = np.empty((len(value), mapping.bits), dtype=np.int8)
    inactive_rank = mapping.label_dim + 1
    for bit in range(mapping.bits):
        chosen = np.argmin(
            np.where(value, mapping.ranks[bit][None, :], inactive_rank), axis=1
        )
        result[:, bit] = mapping.colors[bit, chosen]
    if not np.all(np.isin(result, (-1, 1))):
        raise AssertionError("one-bit MinHash did not produce bipolar codes")
    return np.ascontiguousarray(result, dtype=np.int8)


def encode_semantic_bridge(
    posterior: np.ndarray,
    *,
    threshold: float,
    mapping: OneBitMinHashMap,
) -> np.ndarray:
    """Convert neural label posteriors directly to transient-free binary state."""

    return encode_active_set_minhash(
        posterior_to_active_set(posterior, threshold), mapping
    )


def encode_mean_posterior(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    modality: str,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode the ensemble-mean posterior without retaining hidden states."""

    value = np.asarray(features)
    if value.ndim != 2 or value.shape[1] != 512 or value.dtype.kind != "f":
        raise ValueError("features must be a floating [rows,512] matrix")
    if not np.isfinite(value).all():
        raise ValueError("features must be finite")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if modality not in {"image", "text"}:
        raise ValueError("modality must be image or text")
    if not hasattr(model, "label_dim"):
        raise TypeError("model must expose label_dim")
    resolved = torch.device(device)
    result = np.empty((len(value), int(model.label_dim)), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(value), batch_size):
            end = min(len(value), start + batch_size)
            batch = torch.from_numpy(
                np.array(value[start:end], dtype=np.float32, copy=True, order="C")
            ).to(resolved)
            output = model(batch, modality)
            posterior = output.posterior_heads.mean(dim=1)
            result[start:end] = posterior.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
    _posterior_matrix(result, field="encoded_posterior")
    return result


def semantic_bridge_composite_distance(
    primary_distance: np.ndarray,
    semantic_distance: np.ndarray,
    *,
    detail_bits: int,
) -> np.ndarray:
    """Mixed-radix distance that cannot move candidates across primary shells."""

    primary = np.asarray(primary_distance)
    semantic = np.asarray(semantic_distance)
    if primary.shape != semantic.shape or primary.dtype.kind not in "iu" or semantic.dtype.kind not in "iu":
        raise ValueError("primary and semantic distances must be aligned integers")
    if type(detail_bits) is not int or detail_bits < 1:
        raise ValueError("detail_bits must be a positive integer")
    if np.any(primary < 0) or np.any(semantic < 0) or np.any(semantic > detail_bits):
        raise ValueError("Hamming distances lie outside their registered bounds")
    radix = np.uint32(detail_bits + 1)
    composite = primary.astype(np.uint32) * radix + semantic.astype(np.uint32)
    if not np.array_equal(composite // radix, primary.astype(np.uint32)):
        raise AssertionError("semantic bridge changed a primary Hamming shell")
    return composite


def expected_minhash_mismatch(jaccard: float, bits: int) -> float:
    """Expected Hamming distance for one-bit MinHash under label-set Jaccard."""

    if not math.isfinite(jaccard) or not 0.0 <= jaccard <= 1.0:
        raise ValueError("jaccard must lie in [0,1]")
    if type(bits) is not int or bits < 1:
        raise ValueError("bits must be a positive integer")
    return 0.5 * bits * (1.0 - jaccard)


__all__ = [
    "SEMANTIC_BRIDGE_SCHEMA",
    "OneBitMinHashMap",
    "SemanticBridgeConfig",
    "build_one_bit_minhash_map",
    "calibrate_training_threshold",
    "encode_active_set_minhash",
    "encode_mean_posterior",
    "encode_semantic_bridge",
    "expected_minhash_mismatch",
    "mean_label_set_jaccard",
    "posterior_to_active_set",
    "semantic_bridge_composite_distance",
]
