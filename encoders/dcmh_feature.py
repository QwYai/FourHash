#!/usr/bin/env python3
"""DCMH-F: DCMH's supervised objective on fixed 512-D features.

This is a controlled-feature adaptation of Deep Cross-Modal Hashing (DCMH),
not an official reproduction of the original raw-image system.

Paper:
    Qing-Yuan Jiang and Wu-Jun Li, "Deep Cross-Modal Hashing," CVPR 2017.
    DOI: 10.1109/CVPR.2017.348
    https://openaccess.thecvf.com/content_cvpr_2017/html/
        Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.html

The retained row-major objective is

    sum_ij [softplus(theta_ij) - S_ij theta_ij]
      + gamma (||B-F||_F^2 + ||B-G||_F^2)
      + eta   (||sum_i F_i||_2^2 + ||sum_i G_i||_2^2),

where theta_ij = 0.5 F_i^T G_j, S_ij is label overlap, and the
alternating discrete update is B = sign(F + G).  Out-of-sample image and text
codes are generated independently as sign(f(x)) and sign(g(y)).

DCMH-F deviations are deliberately explicit:
  * fixed CLIP-style 512-D features replace the original image CNN and text
    network;
  * each modality uses a small MLP (or a linear head when --hidden-dim 0);
  * the project's fixed indT/indQ/indD split is used without resampling;
  * modern PyTorch and numerically stable softplus replace the 2017 stack.
  * the evidence-producing default is DCMH-F-SemInit: both branches are
    warm-started from shared codes derived from train labels only.  The old
    random-buffer variant is retained solely as a failure ablation.

Only train features and train labels enter optimization.  Query/database
labels are merely exported after both networks have been frozen.

The alternating loss keeps the public DCMH implementation's single common
batch_size*n_train normalizer.  SemInit fixes the degenerate starting point;
it does not silently reweight gamma/eta.  A mandatory post-training gate
rejects artifacts whose NLL remains near log(2) or whose Hamming codes collapse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.nn import functional as torch_f


SUPPORTED_BITS = (16, 32, 64, 128)
SUPPORTED_INITIALIZATIONS = ("semantic", "random")
FORMAT_VERSION = 1
ENCODER_NAME = "DCMH-F"
PAPER_URL = (
    "https://openaccess.thecvf.com/content_cvpr_2017/html/"
    "Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.html"
)
OFFICIAL_REPOSITORY = "https://github.com/jiangqy/DCMH-CVPR2017"
PYTORCH_REFERENCE = "https://github.com/WendellGul/DCMH"


DATASETS: Mapping[str, Mapping[str, str]] = {
    "mirflickr": {
        "root": "MIRFLICKR",
        "image": "image_features_clip512.mat",
        "text": "text_features_clip512.mat",
        "labels": "labels_clip512.mat",
        "index": "index_5000.mat",
    },
    "mscoco": {
        "root": "MSCOCO",
        "image": "image_features_clip512.mat",
        "text": "text_features_clip512.mat",
        "labels": "labels_clip512.mat",
        "index": "index_10500.mat",
    },
    "nuswide": {
        "root": "NUSWIDE",
        "image": "image_features_clip512.mat",
        "text": "text_features_clip512.mat",
        "labels": "labels_clip512.mat",
        "index": "index_21000.mat",
    },
    "cifar10": {
        "root": "CIFAR10",
        "image": "image_features_clip512.mat",
        "text": "text_features_clip512.mat",
        "labels": "labels_clip512.mat",
        "index": "index.mat",
    },
}

DATASET_ALIASES = {
    "coco": "mscoco",
    "ms-coco": "mscoco",
    "flickr": "mirflickr",
    "flickr25k": "mirflickr",
    "mirflickr25k": "mirflickr",
    "nus-wide": "nuswide",
    "nus": "nuswide",
    "cifar": "cifar10",
}


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    image: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    query_idx: np.ndarray
    database_idx: np.ndarray
    original_index_base: int
    split_layout: str
    paths: Dict[str, str]


@dataclass(frozen=True)
class TrainConfig:
    bits: int = 64
    epochs: int = 20
    batch_size: int = 128
    hidden_dim: int = 256
    lr: float = 10.0 ** -1.5
    min_lr: float = 1e-6
    gamma: float = 1.0
    eta: float = 1.0
    seed: int = 20260805
    device: str = "auto"
    l2_normalize: bool = True
    initialization: str = "semantic"
    warmup_epochs: int = 20
    warmup_lr: float = 3e-3

    def validate(self) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(
                "bits must be one of {}, got {}".format(SUPPORTED_BITS, self.bits)
            )
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.hidden_dim < 0:
            raise ValueError("hidden_dim must be non-negative")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.min_lr) or self.min_lr < 0:
            raise ValueError("min_lr must be finite and non-negative")
        if self.min_lr > self.lr:
            raise ValueError("min_lr must not exceed lr")
        if not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and non-negative")
        if self.initialization not in SUPPORTED_INITIALIZATIONS:
            raise ValueError(
                "initialization must be one of {}, got {!r}".format(
                    SUPPORTED_INITIALIZATIONS, self.initialization
                )
            )
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.initialization == "semantic" and self.warmup_epochs < 1:
            raise ValueError("semantic initialization requires warmup_epochs >= 1")
        if not math.isfinite(self.warmup_lr) or self.warmup_lr <= 0:
            raise ValueError("warmup_lr must be finite and positive")


@dataclass
class TrainingResult:
    image_model: "FeatureHashNet"
    text_model: "FeatureHashNet"
    history: List[Dict[str, float]]
    device: str
    image_parameter_delta: float
    text_parameter_delta: float
    initialization_metadata: Dict[str, object]


class FeatureHashNet(nn.Module):
    """Small feature branch with an unbounded linear hash output.

    DCMH optimizes continuous F/G and applies sign only for discrete codes, so
    no sigmoid or tanh is placed after the hash head.
    """

    def __init__(
        self,
        input_dim: int,
        bits: int,
        hidden_dim: int,
        l2_normalize: bool,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bits = int(bits)
        self.hidden_dim = int(hidden_dim)
        self.l2_normalize = bool(l2_normalize)

        if hidden_dim > 0:
            self.feature_layer = nn.Linear(input_dim, hidden_dim)
            self.hash_head = nn.Linear(hidden_dim, bits)
            nn.init.kaiming_uniform_(self.feature_layer.weight, nonlinearity="relu")
            nn.init.zeros_(self.feature_layer.bias)
        else:
            self.feature_layer = None
            self.hash_head = nn.Linear(input_dim, bits)

        # The public PyTorch DCMH implementation initializes the final hash
        # layer with small Gaussian weights.  Retain that behavior here.
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.hash_head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "Expected [batch, {}] features, got {}".format(
                    self.input_dim, tuple(features.shape)
                )
            )
        x = features
        if self.l2_normalize:
            x = torch_f.normalize(x, p=2, dim=1, eps=1e-8)
        if self.feature_layer is not None:
            x = torch_f.relu(self.feature_layer(x))
        return self.hash_head(x)


def canonical_dataset_name(value: str) -> str:
    name = value.strip().lower()
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASETS:
        valid = sorted(set(DATASETS).union(DATASET_ALIASES))
        raise argparse.ArgumentTypeError(
            "Unknown dataset {!r}; choose one of {}".format(value, valid)
        )
    return name


def first_payload(path: Path) -> np.ndarray:
    """Load a non-v7.3 MAT file containing exactly one user payload."""

    try:
        data = sio.loadmat(str(path))
    except NotImplementedError as exc:
        raise ValueError(
            "{} is MATLAB v7.3/HDF5; convert it to a regular MAT file before "
            "running DCMH-F".format(path)
        ) from exc
    keys = [key for key in data if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError("Expected one payload in {}, found {}".format(path, keys))
    return np.asarray(data[keys[0]])


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _split_sha256(
    train_idx: np.ndarray, query_idx: np.ndarray, database_idx: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("indT", train_idx),
        ("indQ", query_idx),
        ("indD", database_idx),
    ):
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _convert_labels(raw_labels: np.ndarray, n_rows: int) -> np.ndarray:
    labels = np.asarray(raw_labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if labels.ndim != 2 or labels.shape[0] != n_rows:
        raise ValueError(
            "Labels must have shape [N, C] or padded [N, K], got {} for N={}".format(
                labels.shape, n_rows
            )
        )
    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels contain non-finite values")

    unique = np.unique(labels)
    if np.all(np.isin(unique, np.array([0, 1]))):
        multi_hot = labels.astype(np.uint8, copy=False)
    else:
        # Prepared MIRFlickr/MSCOCO labels are padded one-based class-id lists
        # with zero as padding.  This is the same conversion used by the pilot.
        if not np.all(np.equal(labels, np.floor(labels))):
            raise ValueError("Non-binary labels must be integer class ids")
        label_ids = labels.astype(np.int64)
        if label_ids.min() < 0:
            raise ValueError("Negative class ids are unsupported")
        max_label = int(label_ids.max())
        if max_label < 1:
            raise ValueError("Padded class-id labels contain no positive id")
        multi_hot = np.zeros((n_rows, max_label), dtype=np.uint8)
        row_ids = np.repeat(np.arange(n_rows), label_ids.shape[1])
        flat_ids = label_ids.reshape(-1)
        valid = flat_ids > 0
        multi_hot[row_ids[valid], flat_ids[valid] - 1] = 1

    if np.any(multi_hot.sum(axis=1) == 0):
        raise ValueError("Every item must have at least one positive label")
    return np.ascontiguousarray(multi_hot)


def _load_and_validate_indices(
    path: Path, n_rows: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    split = sio.loadmat(str(path))
    missing = [key for key in ("indT", "indQ", "indD") if key not in split]
    if missing:
        raise ValueError("Missing split arrays {} in {}".format(missing, path))
    raw = {
        key: np.asarray(split[key]).reshape(-1).astype(np.int64)
        for key in ("indT", "indQ", "indD")
    }
    union = np.concatenate(list(raw.values()))
    if np.any(union == 0):
        index_base = 0
    elif np.any(union == n_rows):
        index_base = 1
    else:
        raise ValueError(
            "Ambiguous split index base: neither 0 nor dataset size occurs"
        )

    converted = {key: values - index_base for key, values in raw.items()}
    for key, values in converted.items():
        if values.size == 0:
            raise ValueError("Empty split array {}".format(key))
        if values.min() < 0 or values.max() >= n_rows:
            raise ValueError("Out-of-range indices in {}".format(key))
        if np.unique(values).size != values.size:
            raise ValueError("Duplicate indices in {}".format(key))

    train_idx = np.ascontiguousarray(converted["indT"], dtype=np.int64)
    query_idx = np.ascontiguousarray(converted["indQ"], dtype=np.int64)
    database_idx = np.ascontiguousarray(converted["indD"], dtype=np.int64)
    if np.intersect1d(query_idx, database_idx).size:
        raise ValueError("Protocol violation: query and database overlap")
    if np.intersect1d(train_idx, query_idx).size:
        raise ValueError("Protocol violation: train and query overlap")
    train_in_database = bool(np.all(np.isin(train_idx, database_idx)))
    if train_in_database:
        if np.union1d(query_idx, database_idx).size != n_rows:
            raise ValueError("Query and database must cover all items")
        split_layout = "train_subset_database"
    else:
        # The prepared CIFAR negative-control split keeps train, query, and
        # database disjoint.  Partial train/database overlap is ambiguous and
        # rejected; a fully disjoint train set is accepted if all three sets
        # cover the dataset.
        if np.intersect1d(train_idx, database_idx).size:
            raise ValueError("Partial train/database overlap is unsupported")
        covered = np.union1d(np.union1d(query_idx, database_idx), train_idx)
        if covered.size != n_rows:
            raise ValueError("Train, query, and database must cover all items")
        split_layout = "train_disjoint_database"
    return train_idx, query_idx, database_idx, index_base, split_layout


def load_prepared_dataset(data_root: Path, name: str) -> PreparedDataset:
    canonical_name = canonical_dataset_name(name)
    spec = DATASETS[canonical_name]
    root = data_root / spec["root"]
    paths = {
        key: root / spec[key]
        for key in ("image", "text", "labels", "index")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared files: {}".format(missing))

    image = np.ascontiguousarray(first_payload(paths["image"]), dtype=np.float32)
    text = np.ascontiguousarray(first_payload(paths["text"]), dtype=np.float32)
    if image.ndim != 2 or text.ndim != 2 or image.shape != text.shape:
        raise ValueError(
            "Image/text features must have identical [N,D] shapes, got {} and {}".format(
                image.shape, text.shape
            )
        )
    if image.shape[1] != 512:
        raise ValueError(
            "DCMH-F protocol expects fixed 512-D features, got D={}".format(
                image.shape[1]
            )
        )
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(text)):
        raise ValueError("Features contain non-finite values")

    labels = _convert_labels(first_payload(paths["labels"]), image.shape[0])
    (
        train_idx,
        query_idx,
        database_idx,
        index_base,
        split_layout,
    ) = _load_and_validate_indices(paths["index"], image.shape[0])
    return PreparedDataset(
        name=canonical_name,
        image=image,
        text=text,
        labels=labels,
        train_idx=train_idx,
        query_idx=query_idx,
        database_idx=database_idx,
        original_index_base=index_base,
        split_layout=split_layout,
        paths={key: str(path.resolve()) for key, path in paths.items()},
    )


def stable_sign(values: torch.Tensor) -> torch.Tensor:
    """DCMH sign convention: zero maps to +1."""

    return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


def _resolve_device(requested: str) -> torch.device:
    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def _parameter_snapshot(model: nn.Module) -> List[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def _parameter_delta(
    before: Sequence[torch.Tensor], model: nn.Module
) -> float:
    squared = 0.0
    for original, current in zip(before, model.parameters()):
        difference = current.detach().cpu() - original
        squared += float(torch.sum(difference * difference).item())
    return math.sqrt(squared)


def _gradient_norm(model: nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach() ** 2).item())
    return math.sqrt(squared)


def _semantic_binary_targets(
    train_labels: np.ndarray, bits: int, seed: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build deterministic shared training codes from train labels only.

    DCMH's official objective already consumes the same label-overlap graph.
    This target does not add supervision, but using it to initialize both
    branches *is* an optimization deviation and is recorded as SemInit.
    Centering normalized label profiles makes disjoint classes point in
    opposing directions before random-hyperplane binarization.
    """

    labels = np.ascontiguousarray(train_labels, dtype=np.float32)
    profiles = labels / np.maximum(labels.sum(axis=1, keepdims=True), 1.0)
    centered = profiles - profiles.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) <= 1e-8:
        raise ValueError("Semantic initialization requires non-constant labels")
    rng = np.random.default_rng(seed + 1701)
    projection = rng.standard_normal((labels.shape[1], bits)).astype(np.float32)
    scores = centered @ projection
    targets = np.where(scores >= 0, 1.0, -1.0).astype(np.float32)

    # A random projection of a non-constant label matrix is almost surely
    # non-constant, but fail loudly rather than warm-start a collapsed bit.
    collapsed = np.flatnonzero(np.abs(targets.mean(axis=0)) >= 1.0)
    if collapsed.size:
        raise RuntimeError(
            "Semantic initialization produced collapsed target bits: {}".format(
                collapsed.tolist()
            )
        )
    metadata: Dict[str, object] = {
        "method": "centered normalized train labels + seeded Gaussian projection + sign",
        "uses": "train labels only",
        "seed": int(seed + 1701),
        "target_sha256": _array_sha256(targets),
        "mean_abs_target_bit_mean": float(np.mean(np.abs(targets.mean(axis=0)))),
        "max_abs_target_bit_mean": float(np.max(np.abs(targets.mean(axis=0)))),
        "unique_target_rows": int(np.unique(targets, axis=0).shape[0]),
    }
    return np.ascontiguousarray(targets), metadata


def _warmup_branch(
    model: FeatureHashNet,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, object]:
    """Fit one feature branch to the shared semantic code before DCMH SGD."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n_train = int(features.shape[0])
    history: List[float] = []
    gradient_history: List[float] = []
    model.train()
    for _ in range(epochs):
        loss_sum = 0.0
        gradient_sum = 0.0
        seen = 0
        for cpu_indices in torch.randperm(n_train, generator=generator).split(batch_size):
            indices = cpu_indices.to(features.device)
            output = model(features.index_select(0, indices))
            loss = torch_f.mse_loss(output, targets.index_select(0, indices))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = _gradient_norm(model)
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite semantic-warmup gradient")
            optimizer.step()
            count = int(indices.numel())
            seen += count
            loss_sum += float(loss.detach().item()) * count
            gradient_sum += gradient_norm * count
        history.append(loss_sum / seen)
        gradient_history.append(gradient_sum / seen)
    return {
        "loss_history": history,
        "gradient_norm_history": gradient_history,
        "first_mse": float(history[0]),
        "last_mse": float(history[-1]),
    }


@torch.no_grad()
def _forward_all_tensor(
    model: FeatureHashNet, features: torch.Tensor, batch_size: int
) -> torch.Tensor:
    model.eval()
    output = torch.empty(
        features.shape[0], model.bits, dtype=features.dtype, device=features.device
    )
    for start in range(0, int(features.shape[0]), batch_size):
        stop = min(start + batch_size, int(features.shape[0]))
        output[start:stop] = model(features[start:stop])
    return output


def _dcmh_minibatch_loss(
    current: torch.Tensor,
    opposite_buffer: torch.Tensor,
    own_buffer: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_labels: torch.Tensor,
    all_train_labels: torch.Tensor,
    binary_codes: torch.Tensor,
    gamma: float,
    eta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One DCMH alternating subproblem, normalized by batch_size * n.

    The common normalizer does not change the relative terms or optimum; it
    keeps modern autograd updates numerically manageable.  Similarities are
    constructed on demand from train labels, avoiding an n-by-n stored matrix.
    """

    similarity = (batch_labels @ all_train_labels.t() > 0).to(current.dtype)
    theta = 0.5 * (current @ opposite_buffer.t())
    pairwise = torch.sum(torch_f.softplus(theta) - similarity * theta)
    quantization = torch.sum(
        (binary_codes.index_select(0, batch_indices) - current) ** 2
    )

    # F/G values outside the current mini-batch are stale buffers, exactly as
    # in the alternating stochastic DCMH implementation.  The current batch
    # stays differentiable in the global bit-balance surrogate.
    detached_other_sum = (
        own_buffer.sum(dim=0)
        - own_buffer.index_select(0, batch_indices).sum(dim=0)
    )
    total_bit_sum = current.sum(dim=0) + detached_other_sum
    balance = torch.sum(total_bit_sum ** 2)

    normalizer = float(current.shape[0] * opposite_buffer.shape[0])
    parts = {
        "pairwise": pairwise / normalizer,
        "quantization": quantization / normalizer,
        "balance": balance / normalizer,
        "positive_pair_rate": similarity.mean(),
        "theta_mean": theta.mean(),
        "theta_rms": torch.sqrt(torch.mean(theta ** 2)),
        "output_rms": torch.sqrt(torch.mean(current ** 2)),
    }
    total = parts["pairwise"] + gamma * parts["quantization"] + eta * parts["balance"]
    return total, parts


def _run_branch_phase(
    model: FeatureHashNet,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    train_labels: torch.Tensor,
    own_buffer: torch.Tensor,
    opposite_buffer: torch.Tensor,
    binary_codes: torch.Tensor,
    permutation: torch.Tensor,
    batch_size: int,
    gamma: float,
    eta: float,
) -> Dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "pairwise": 0.0,
        "quantization": 0.0,
        "balance": 0.0,
        "positive_pair_rate": 0.0,
        "theta_mean": 0.0,
        "theta_rms": 0.0,
        "output_rms": 0.0,
        "gradient_norm": 0.0,
    }
    n_train = int(features.shape[0])
    seen = 0

    for start in range(0, n_train, batch_size):
        cpu_indices = permutation[start : start + batch_size]
        batch_indices = cpu_indices.to(features.device)
        batch_features = features.index_select(0, batch_indices)
        batch_labels = train_labels.index_select(0, batch_indices)

        optimizer.zero_grad(set_to_none=True)
        current = model(batch_features)
        if not torch.isfinite(current).all():
            raise FloatingPointError("Non-finite branch output")

        # Public DCMH PyTorch code refreshes the sampled rows before computing
        # the balance surrogate; the buffer write is detached from autograd.
        with torch.no_grad():
            own_buffer.index_copy_(0, batch_indices, current.detach())

        loss, parts = _dcmh_minibatch_loss(
            current=current,
            opposite_buffer=opposite_buffer,
            own_buffer=own_buffer,
            batch_indices=batch_indices,
            batch_labels=batch_labels,
            all_train_labels=train_labels,
            binary_codes=binary_codes,
            gamma=gamma,
            eta=eta,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DCMH-F loss")
        loss.backward()
        grad_norm = _gradient_norm(model)
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()

        count = int(batch_indices.numel())
        seen += count
        totals["loss"] += float(loss.detach().item()) * count
        for key, value in parts.items():
            totals[key] += float(value.detach().item()) * count
        totals["gradient_norm"] += grad_norm * count

    if seen != n_train:
        raise AssertionError("Branch phase did not visit every training row")
    return {key: value / seen for key, value in totals.items()}


def train_dcmh_f(
    train_image: np.ndarray,
    train_text: np.ndarray,
    train_labels: np.ndarray,
    config: TrainConfig,
    verbose: bool = True,
) -> TrainingResult:
    """Train DCMH-F using only the arrays passed to this function.

    Keeping the API train-only is intentional: it makes query/database label
    leakage structurally impossible inside optimization.
    """

    config.validate()
    image_np = np.ascontiguousarray(train_image, dtype=np.float32)
    text_np = np.ascontiguousarray(train_text, dtype=np.float32)
    labels_np = np.ascontiguousarray(train_labels, dtype=np.float32)
    if image_np.ndim != 2 or text_np.shape != image_np.shape:
        raise ValueError("Train image/text arrays must have the same [N,D] shape")
    if image_np.shape[0] < 2:
        raise ValueError("At least two training items are required")
    if labels_np.ndim != 2 or labels_np.shape[0] != image_np.shape[0]:
        raise ValueError("Train labels must have shape [N,C]")
    if not np.all(np.isin(np.unique(labels_np), np.array([0.0, 1.0]))):
        raise ValueError("Train labels must be binary multi-hot")
    if np.any(labels_np.sum(axis=1) == 0):
        raise ValueError("Every training row must have a positive label")
    if not np.all(np.isfinite(image_np)) or not np.all(np.isfinite(text_np)):
        raise ValueError("Train features contain non-finite values")

    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    input_dim = int(image_np.shape[1])
    image_model = FeatureHashNet(
        input_dim, config.bits, config.hidden_dim, config.l2_normalize
    ).to(device)
    text_model = FeatureHashNet(
        input_dim, config.bits, config.hidden_dim, config.l2_normalize
    ).to(device)
    image_before = _parameter_snapshot(image_model)
    text_before = _parameter_snapshot(text_model)

    image_tensor = torch.from_numpy(image_np).to(device)
    text_tensor = torch.from_numpy(text_np).to(device)
    label_tensor = torch.from_numpy(labels_np).to(device)
    n_train = int(image_tensor.shape[0])

    initialization_metadata: Dict[str, object]
    if config.initialization == "semantic":
        semantic_targets_np, target_metadata = _semantic_binary_targets(
            labels_np, config.bits, config.seed
        )
        semantic_targets = torch.from_numpy(semantic_targets_np).to(device)
        image_warmup = _warmup_branch(
            image_model,
            image_tensor,
            semantic_targets,
            epochs=config.warmup_epochs,
            batch_size=config.batch_size,
            lr=config.warmup_lr,
            seed=config.seed + 1801,
        )
        text_warmup = _warmup_branch(
            text_model,
            text_tensor,
            semantic_targets,
            epochs=config.warmup_epochs,
            batch_size=config.batch_size,
            lr=config.warmup_lr,
            seed=config.seed + 1901,
        )
        f_buffer = _forward_all_tensor(image_model, image_tensor, config.batch_size)
        g_buffer = _forward_all_tensor(text_model, text_tensor, config.batch_size)
        initialization_metadata = {
            "name": "semantic_warm_start",
            "reporting_name": "DCMH-F-SemInit",
            "official_dcmh": False,
            "reason": (
                "random buffers plus small fixed-feature heads stalled at "
                "pairwise NLL ~= log(2); train-label SemInit supplies a "
                "non-collapsed alternating starting point"
            ),
            "target": target_metadata,
            "image_warmup": image_warmup,
            "text_warmup": text_warmup,
        }
    else:
        # Kept only as a documented ablation matching the widely used public
        # PyTorch fork.  It is expected to fail the quality gate for small
        # fixed-feature heads at short training budgets.
        buffer_generator = torch.Generator(device="cpu")
        buffer_generator.manual_seed(config.seed + 101)
        f_buffer = torch.randn(
            n_train, config.bits, generator=buffer_generator, dtype=torch.float32
        ).to(device)
        g_buffer = torch.randn(
            n_train, config.bits, generator=buffer_generator, dtype=torch.float32
        ).to(device)
        initialization_metadata = {
            "name": "random_buffers",
            "reporting_name": "DCMH-F-RandomInit-ablation",
            "official_dcmh": False,
            "warning": (
                "The DCMH paper does not prescribe these buffers; this follows "
                "a public PyTorch fork and is not accepted as evidence unless "
                "it independently passes the quality gate."
            ),
        }

    # The DCMH alternating phase itself retains the original summed objective
    # with one common batch*n normalizer and SGD.  SemInit changes the starting
    # point, not the relative gamma/eta weighting of that objective.
    image_optimizer = torch.optim.SGD(image_model.parameters(), lr=config.lr)
    text_optimizer = torch.optim.SGD(text_model.parameters(), lr=config.lr)
    binary_codes = stable_sign(f_buffer + g_buffer)
    initialization_metadata["initial_f_rms"] = float(
        torch.sqrt(torch.mean(f_buffer * f_buffer)).item()
    )
    initialization_metadata["initial_g_rms"] = float(
        torch.sqrt(torch.mean(g_buffer * g_buffer)).item()
    )
    initialization_metadata["initial_buffer_code_agreement"] = float(
        (stable_sign(f_buffer) == stable_sign(g_buffer)).float().mean().item()
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(config.seed + 211)

    history: List[Dict[str, float]] = []
    for epoch in range(config.epochs):
        fraction = 0.0 if config.epochs == 1 else epoch / float(config.epochs - 1)
        lr = config.lr + fraction * (config.min_lr - config.lr)
        for optimizer in (image_optimizer, text_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = lr

        previous_binary = binary_codes.clone()
        image_stats = _run_branch_phase(
            model=image_model,
            optimizer=image_optimizer,
            features=image_tensor,
            train_labels=label_tensor,
            own_buffer=f_buffer,
            opposite_buffer=g_buffer,
            binary_codes=binary_codes,
            permutation=torch.randperm(n_train, generator=shuffle_generator),
            batch_size=config.batch_size,
            gamma=config.gamma,
            eta=config.eta,
        )
        text_stats = _run_branch_phase(
            model=text_model,
            optimizer=text_optimizer,
            features=text_tensor,
            train_labels=label_tensor,
            own_buffer=g_buffer,
            opposite_buffer=f_buffer,
            binary_codes=binary_codes,
            permutation=torch.randperm(n_train, generator=shuffle_generator),
            batch_size=config.batch_size,
            gamma=config.gamma,
            eta=config.eta,
        )
        with torch.no_grad():
            binary_codes = stable_sign(f_buffer + g_buffer)
            flip_rate = float((binary_codes != previous_binary).float().mean().item())
            buffer_agreement = float(
                (stable_sign(f_buffer) == stable_sign(g_buffer)).float().mean().item()
            )

        row: Dict[str, float] = {
            "epoch": float(epoch + 1),
            "lr": float(lr),
            "b_flip_rate": flip_rate,
            "buffer_code_agreement": buffer_agreement,
        }
        for prefix, stats in (("image", image_stats), ("text", text_stats)):
            for key, value in stats.items():
                row["{}_{}".format(prefix, key)] = float(value)
        history.append(row)
        if verbose:
            print(json.dumps(row, sort_keys=True))

    return TrainingResult(
        image_model=image_model,
        text_model=text_model,
        history=history,
        device=str(device),
        image_parameter_delta=_parameter_delta(image_before, image_model),
        text_parameter_delta=_parameter_delta(text_before, text_model),
        initialization_metadata=initialization_metadata,
    )


@torch.no_grad()
def encode_all(
    model: FeatureHashNet,
    features: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    target_device = torch.device(device)
    values = np.ascontiguousarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != model.input_dim:
        raise ValueError("Unexpected export feature shape {}".format(values.shape))
    codes = np.empty((values.shape[0], model.bits), dtype=np.int8)
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = torch.from_numpy(values[start:stop]).to(target_device)
        output = model(batch)
        signed = stable_sign(output).to(torch.int8).cpu().numpy()
        codes[start:stop] = signed
    if not np.all(np.isin(np.unique(codes), np.array([-1, 1], dtype=np.int8))):
        raise AssertionError("Exported codes are not bipolar")
    return codes


def _code_statistics(codes: np.ndarray) -> Dict[str, float]:
    floating = codes.astype(np.float32, copy=False)
    bit_means = floating.mean(axis=0)
    return {
        "mean_abs_bit_mean": float(np.mean(np.abs(bit_means))),
        "max_abs_bit_mean": float(np.max(np.abs(bit_means))),
        "plus_one_fraction": float(np.mean(floating > 0)),
    }


@torch.no_grad()
def training_semantic_diagnostics(
    image_model: FeatureHashNet,
    text_model: FeatureHashNet,
    train_image: np.ndarray,
    train_text: np.ndarray,
    train_labels: np.ndarray,
    batch_size: int,
    device: str,
) -> Dict[str, object]:
    """Evaluate every train cross-modal pair after alternating optimization."""

    target_device = torch.device(device)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(train_image, dtype=np.float32)
    ).to(target_device)
    text_tensor = torch.from_numpy(
        np.ascontiguousarray(train_text, dtype=np.float32)
    ).to(target_device)
    label_tensor = torch.from_numpy(
        np.ascontiguousarray(train_labels, dtype=np.float32)
    ).to(target_device)
    f_values = _forward_all_tensor(image_model, image_tensor, batch_size)
    g_values = _forward_all_tensor(text_model, text_tensor, batch_size)
    f_codes = stable_sign(f_values)
    g_codes = stable_sign(g_values)

    n_train = int(f_values.shape[0])
    pair_count = n_train * n_train
    nll_sum = 0.0
    positive_count = 0
    negative_count = 0
    positive_theta_sum = 0.0
    negative_theta_sum = 0.0
    positive_hamming_sum = 0.0
    negative_hamming_sum = 0.0
    for start in range(0, n_train, batch_size):
        stop = min(start + batch_size, n_train)
        similarity = (label_tensor[start:stop] @ label_tensor.t() > 0)
        theta = 0.5 * (f_values[start:stop] @ g_values.t())
        nll = torch_f.softplus(theta) - similarity.to(theta.dtype) * theta
        hamming = 0.5 * (
            image_model.bits - f_codes[start:stop] @ g_codes.t()
        )
        nll_sum += float(nll.sum().item())
        batch_positive = int(similarity.sum().item())
        batch_negative = int(similarity.numel() - batch_positive)
        positive_count += batch_positive
        negative_count += batch_negative
        if batch_positive:
            positive_theta_sum += float(theta[similarity].sum().item())
            positive_hamming_sum += float(hamming[similarity].sum().item())
        if batch_negative:
            negative_mask = ~similarity
            negative_theta_sum += float(theta[negative_mask].sum().item())
            negative_hamming_sum += float(hamming[negative_mask].sum().item())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Quality diagnostics require positive and negative train pairs")
    image_codes_np = f_codes.to(torch.int8).cpu().numpy()
    text_codes_np = g_codes.to(torch.int8).cpu().numpy()
    return {
        "scope": "all n_train^2 ordered image-text pairs",
        "pair_count": int(pair_count),
        "positive_pair_rate": float(positive_count / pair_count),
        "pairwise_nll": float(nll_sum / pair_count),
        "log_two": float(math.log(2.0)),
        "mean_positive_theta": float(positive_theta_sum / positive_count),
        "mean_negative_theta": float(negative_theta_sum / negative_count),
        "mean_positive_hamming": float(positive_hamming_sum / positive_count),
        "mean_negative_hamming": float(negative_hamming_sum / negative_count),
        "hamming_gap_negative_minus_positive": float(
            negative_hamming_sum / negative_count
            - positive_hamming_sum / positive_count
        ),
        "paired_code_agreement": float(np.mean(image_codes_np == text_codes_np)),
        "unique_image_code_rows": int(np.unique(image_codes_np, axis=0).shape[0]),
        "unique_text_code_rows": int(np.unique(text_codes_np, axis=0).shape[0]),
        "image_code_statistics": _code_statistics(image_codes_np),
        "text_code_statistics": _code_statistics(text_codes_np),
        "image_continuous_rms": float(torch.sqrt(torch.mean(f_values ** 2)).item()),
        "text_continuous_rms": float(torch.sqrt(torch.mean(g_values ** 2)).item()),
    }


def dcmh_f_quality_gate(
    diagnostics: Mapping[str, object], bits: int
) -> Dict[str, object]:
    """Reject random/collapsed artifacts before they can be exported as evidence."""

    nll_limit = math.log(2.0) - 0.05
    minimum_hamming_gap = max(1.0, 0.03 * bits)
    checks = {
        "pairwise_nll_below_log2_by_0.05": bool(
            float(diagnostics["pairwise_nll"]) <= nll_limit
        ),
        "positive_pairs_closer_by_required_margin": bool(
            float(diagnostics["hamming_gap_negative_minus_positive"])
            >= minimum_hamming_gap
        ),
        "image_codes_nonconstant": bool(
            int(diagnostics["unique_image_code_rows"]) > 1
        ),
        "text_codes_nonconstant": bool(
            int(diagnostics["unique_text_code_rows"]) > 1
        ),
        "image_bits_not_fully_collapsed": bool(
            float(diagnostics["image_code_statistics"]["mean_abs_bit_mean"]) < 0.95
        ),
        "text_bits_not_fully_collapsed": bool(
            float(diagnostics["text_code_statistics"]["mean_abs_bit_mean"]) < 0.95
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "maximum_pairwise_nll": float(nll_limit),
            "minimum_hamming_gap": float(minimum_hamming_gap),
        },
        "policy": (
            "failed artifacts are not exported by default and must not be used "
            "as encoder evidence"
        ),
    }


def _retrieval_metrics(
    query_codes: np.ndarray,
    retrieval_codes: np.ndarray,
    query_labels: np.ndarray,
    retrieval_labels: np.ndarray,
) -> Dict[str, float]:
    """Exact full-database Hamming mAP in bounded query chunks."""

    if query_codes.ndim != 2 or retrieval_codes.ndim != 2:
        raise ValueError("Retrieval codes must be matrices")
    if query_codes.shape[1] != retrieval_codes.shape[1]:
        raise ValueError("Query/retrieval bit dimensions differ")
    if query_labels.shape[0] != query_codes.shape[0]:
        raise ValueError("Query code/label row mismatch")
    if retrieval_labels.shape[0] != retrieval_codes.shape[0]:
        raise ValueError("Retrieval code/label row mismatch")

    bits = int(query_codes.shape[1])
    retrieval_rows = int(retrieval_codes.shape[0])
    if retrieval_rows < 2:
        raise ValueError("At least two retrieval rows are required")
    harmonic_n = float(np.sum(1.0 / np.arange(1, retrieval_rows + 1)))
    average_precisions: List[float] = []
    random_expected_aps: List[float] = []
    positive_distance_sum = 0.0
    negative_distance_sum = 0.0
    positive_count = 0
    negative_count = 0
    retrieval_i16 = retrieval_codes.astype(np.int16, copy=False)
    for start in range(0, query_codes.shape[0], 64):
        stop = min(start + 64, query_codes.shape[0])
        relevance = query_labels[start:stop] @ retrieval_labels.T > 0
        distances = 0.5 * (
            bits
            - query_codes[start:stop].astype(np.int16, copy=False)
            @ retrieval_i16.T
        )
        positive_distance_sum += float(distances[relevance].sum())
        negative_distance_sum += float(distances[~relevance].sum())
        positive_count += int(relevance.sum())
        negative_count += int(relevance.size - relevance.sum())
        for row in range(stop - start):
            order = np.argsort(distances[row], kind="stable")
            ranked = relevance[row, order]
            relevant_count = int(ranked.sum())
            if relevant_count == 0:
                continue
            precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
            average_precisions.append(
                float(np.sum(precision * ranked) / relevant_count)
            )
            # Exact expectation of AP under a uniformly random permutation
            # with R relevant items among N retrieval items.
            expected_random_ap = (
                harmonic_n
                + (relevant_count - 1)
                / (retrieval_rows - 1)
                * (retrieval_rows - harmonic_n)
            ) / retrieval_rows
            random_expected_aps.append(float(expected_random_ap))
    if not average_precisions:
        raise ValueError("No query has a relevant retrieval item")
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Retrieval evaluation needs positive and negative pairs")
    return {
        "map": float(np.mean(average_precisions)),
        "random_ranking_expected_map": float(np.mean(random_expected_aps)),
        "map_gain_over_random": float(
            np.mean(average_precisions) - np.mean(random_expected_aps)
        ),
        "evaluated_queries": int(len(average_precisions)),
        "mean_positive_hamming": float(positive_distance_sum / positive_count),
        "mean_negative_hamming": float(negative_distance_sum / negative_count),
    }


def heldout_retrieval_quality_gate(
    i2t: Mapping[str, float], t2i: Mapping[str, float]
) -> Dict[str, object]:
    minimum_map_gain = 0.05
    checks = {
        "i2t_map_beats_random_by_0.05": bool(
            float(i2t["map_gain_over_random"]) >= minimum_map_gain
        ),
        "t2i_map_beats_random_by_0.05": bool(
            float(t2i["map_gain_over_random"]) >= minimum_map_gain
        ),
        "i2t_positive_pairs_are_closer": bool(
            float(i2t["mean_positive_hamming"])
            < float(i2t["mean_negative_hamming"])
        ),
        "t2i_positive_pairs_are_closer": bool(
            float(t2i["mean_positive_hamming"])
            < float(t2i["mean_negative_hamming"])
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {"minimum_map_gain_over_random": minimum_map_gain},
        "role": (
            "post-freeze held-out acceptance only; never used for training or "
            "hyperparameter selection"
        ),
    }


def _random_retrieval_baseline(
    query_rows: int,
    retrieval_rows: int,
    bits: int,
    query_labels: np.ndarray,
    retrieval_labels: np.ndarray,
    seed: int,
    repetitions: int = 20,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    i2t_values: List[float] = []
    t2i_values: List[float] = []
    for _ in range(repetitions):
        query_image = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(query_rows, bits)
        )
        query_text = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(query_rows, bits)
        )
        retrieval_image = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(retrieval_rows, bits)
        )
        retrieval_text = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(retrieval_rows, bits)
        )
        i2t_values.append(
            _retrieval_metrics(
                query_image, retrieval_text, query_labels, retrieval_labels
            )["map"]
        )
        t2i_values.append(
            _retrieval_metrics(
                query_text, retrieval_image, query_labels, retrieval_labels
            )["map"]
        )
    return {
        "i2t_map_mean": float(np.mean(i2t_values)),
        "t2i_map_mean": float(np.mean(t2i_values)),
        "repetitions": int(repetitions),
    }


def export_npz(
    output: Path,
    image_codes: np.ndarray,
    text_codes: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    database_idx: np.ndarray,
    metadata: Dict[str, object],
    overwrite: bool,
) -> None:
    if output.suffix.lower() != ".npz":
        raise ValueError("Output path must end in .npz")
    if output.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace it".format(output)
        )
    n_rows = int(image_codes.shape[0])
    if image_codes.shape != text_codes.shape:
        raise ValueError("Image and text code shapes differ")
    if labels.shape[0] != n_rows:
        raise ValueError("Code/label row mismatch")
    for name, values in (("image", image_codes), ("text", text_codes)):
        if values.dtype != np.int8:
            raise ValueError("{} codes must be int8".format(name))
        if not np.all(np.isin(np.unique(values), np.array([-1, 1], dtype=np.int8))):
            raise ValueError("{} codes must contain only -1/+1".format(name))

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        image_codes=np.ascontiguousarray(image_codes, dtype=np.int8),
        text_codes=np.ascontiguousarray(text_codes, dtype=np.int8),
        labels=np.ascontiguousarray(labels, dtype=np.uint8),
        train_idx=np.ascontiguousarray(train_idx, dtype=np.int64),
        query_idx=np.ascontiguousarray(query_idx, dtype=np.int64),
        database_idx=np.ascontiguousarray(database_idx, dtype=np.int64),
        metadata_json=np.asarray(metadata_json),
    )
    os.replace(str(temporary), str(output))


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_metadata(
    dataset: PreparedDataset,
    config: TrainConfig,
    result: TrainingResult,
    image_codes: np.ndarray,
    text_codes: np.ndarray,
    train_diagnostics: Mapping[str, object],
    quality_gate: Mapping[str, object],
    heldout_retrieval: Mapping[str, object],
    heldout_quality_gate: Mapping[str, object],
) -> Dict[str, object]:
    train_labels = dataset.labels[dataset.train_idx]
    return {
        "format_version": FORMAT_VERSION,
        "encoder": ENCODER_NAME,
        "reporting_name": result.initialization_metadata["reporting_name"],
        "claim_scope": (
            "controlled-feature adaptation; not an official DCMH reproduction"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "title": "Deep Cross-Modal Hashing",
            "authors": "Qing-Yuan Jiang and Wu-Jun Li",
            "venue": "CVPR 2017",
            "doi": "10.1109/CVPR.2017.348",
            "url": PAPER_URL,
            "official_repository": OFFICIAL_REPOSITORY,
            "pytorch_reference": PYTORCH_REFERENCE,
        },
        "retained_dcmh_components": [
            "cross-modal Bernoulli negative log-likelihood",
            "shared discrete training code B",
            "quantization penalties ||B-F||^2 and ||B-G||^2",
            "bit-balance penalties",
            "alternating image/text optimization",
            "B = sign(F + G)",
            "independent out-of-sample sign(f(x)) and sign(g(y))",
        ],
        "dcmh_f_deviations": [
            "fixed 512-D features replace the original raw-image CNN and text network",
            "two small MLP/linear feature heads are used",
            "project-provided fixed split replaces the paper's sampling protocol",
            "modern PyTorch autograd and stable softplus are used",
            "training epochs are user-controlled and not the official 500-epoch default",
            (
                "the default DCMH-F-SemInit variant fits both branches to shared "
                "train-label-derived codes before the unchanged DCMH alternating phase"
            ),
        ],
        "objective": {
            "theta": "0.5 * F @ G.T",
            "similarity": "S_ij = 1 iff train labels i,j overlap",
            "pairwise": "sum(softplus(theta) - S * theta)",
            "quantization": "gamma*(||B-F||_F^2 + ||B-G||_F^2)",
            "balance": "eta*(||sum_rows(F)||_2^2 + ||sum_rows(G)||_2^2)",
            "discrete_update": "B = sign(F + G), with sign(0)=+1",
            "minibatch_scaling": (
                "all three summed terms share one batch_size*n_train normalizer; "
                "gamma/eta relative weighting matches the public DCMH implementation"
            ),
        },
        "leakage_contract": {
            "optimizer_inputs": "image[train_idx], text[train_idx], labels[train_idx] only",
            "pair_construction": "train-label overlap only",
            "query_features": "encoded only after networks are frozen",
            "query_database_labels": "exported for later metrics; never passed to train_dcmh_f",
            "hyperparameter_selection": "fixed before official query evaluation",
        },
        "dataset": {
            "name": dataset.name,
            "rows": int(dataset.image.shape[0]),
            "feature_dim": int(dataset.image.shape[1]),
            "label_dim": int(dataset.labels.shape[1]),
            "train_rows": int(dataset.train_idx.size),
            "query_rows": int(dataset.query_idx.size),
            "database_rows": int(dataset.database_idx.size),
            "original_index_base": int(dataset.original_index_base),
            "split_layout": dataset.split_layout,
            "paths": dataset.paths,
            "split_sha256": _split_sha256(
                dataset.train_idx, dataset.query_idx, dataset.database_idx
            ),
            "train_labels_sha256": _array_sha256(train_labels),
        },
        "training": {
            **asdict(config),
            "resolved_device": result.device,
            "image_parameter_delta_l2": result.image_parameter_delta,
            "text_parameter_delta_l2": result.text_parameter_delta,
            "initialization": result.initialization_metadata,
            "history": result.history,
            "train_semantic_diagnostics": train_diagnostics,
            "quality_gate": quality_gate,
            "heldout_retrieval": heldout_retrieval,
            "heldout_quality_gate": heldout_quality_gate,
            "overall_usable": bool(
                quality_gate["passed"] and heldout_quality_gate["passed"]
            ),
        },
        "architecture": {
            "branches": "independent image and text weights",
            "input_dim": int(dataset.image.shape[1]),
            "hidden_dim": int(config.hidden_dim),
            "hash_bits": int(config.bits),
            "hash_activation": "none during training; sign at export",
            "input_l2_normalize_per_item": bool(config.l2_normalize),
        },
        "export": {
            "image_codes_shape": list(image_codes.shape),
            "text_codes_shape": list(text_codes.shape),
            "dtype": "int8",
            "domain": [-1, 1],
            "image_statistics": _code_statistics(image_codes),
            "text_statistics": _code_statistics(text_codes),
            "paired_code_agreement": float(np.mean(image_codes == text_codes)),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "source_sha256": _source_sha256(),
        },
    }


def run_train(args: argparse.Namespace) -> None:
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace it".format(args.output)
        )
    if args.export_batch_size < 1:
        raise ValueError("--export-batch-size must be positive")
    config = TrainConfig(
        bits=args.bits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        min_lr=args.min_lr,
        gamma=args.gamma,
        eta=args.eta,
        seed=args.seed,
        device=args.device,
        l2_normalize=not args.no_l2_normalize,
        initialization=args.initialization,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
    )
    dataset = load_prepared_dataset(args.data_root, args.dataset)
    result = train_dcmh_f(
        train_image=dataset.image[dataset.train_idx],
        train_text=dataset.text[dataset.train_idx],
        train_labels=dataset.labels[dataset.train_idx],
        config=config,
        verbose=not args.quiet,
    )
    train_diagnostics = training_semantic_diagnostics(
        result.image_model,
        result.text_model,
        dataset.image[dataset.train_idx],
        dataset.text[dataset.train_idx],
        dataset.labels[dataset.train_idx],
        batch_size=args.batch_size,
        device=result.device,
    )
    quality_gate = dcmh_f_quality_gate(train_diagnostics, config.bits)
    print(
        json.dumps(
            {
                "train_semantic_diagnostics": train_diagnostics,
                "quality_gate": quality_gate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not quality_gate["passed"] and not args.allow_failed_quality_gate:
        raise RuntimeError(
            "DCMH-F quality gate failed; no NPZ was exported. This run must not "
            "be used as encoder evidence. Use --allow-failed-quality-gate only "
            "to preserve an explicitly failed diagnostic artifact."
        )
    image_codes = encode_all(
        result.image_model,
        dataset.image,
        batch_size=args.export_batch_size,
        device=result.device,
    )
    text_codes = encode_all(
        result.text_model,
        dataset.text,
        batch_size=args.export_batch_size,
        device=result.device,
    )
    heldout_retrieval: Dict[str, object] = {
        "protocol": "fixed query_idx against full fixed database_idx after freeze",
        "i2t": _retrieval_metrics(
            image_codes[dataset.query_idx],
            text_codes[dataset.database_idx],
            dataset.labels[dataset.query_idx],
            dataset.labels[dataset.database_idx],
        ),
        "t2i": _retrieval_metrics(
            text_codes[dataset.query_idx],
            image_codes[dataset.database_idx],
            dataset.labels[dataset.query_idx],
            dataset.labels[dataset.database_idx],
        ),
    }
    heldout_quality_gate = heldout_retrieval_quality_gate(
        heldout_retrieval["i2t"], heldout_retrieval["t2i"]
    )
    print(
        json.dumps(
            {
                "heldout_retrieval": heldout_retrieval,
                "heldout_quality_gate": heldout_quality_gate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not heldout_quality_gate["passed"] and not args.allow_failed_quality_gate:
        raise RuntimeError(
            "DCMH-F held-out retrieval quality gate failed; no NPZ was "
            "exported and this run is a no-go."
        )
    metadata = build_metadata(
        dataset,
        config,
        result,
        image_codes,
        text_codes,
        train_diagnostics,
        quality_gate,
        heldout_retrieval,
        heldout_quality_gate,
    )
    export_npz(
        output=args.output,
        image_codes=image_codes,
        text_codes=text_codes,
        labels=dataset.labels,
        train_idx=dataset.train_idx,
        query_idx=dataset.query_idx,
        database_idx=dataset.database_idx,
        metadata=metadata,
        overwrite=args.overwrite,
    )
    summary = {
        "encoder": result.initialization_metadata["reporting_name"],
        "claim_scope": metadata["claim_scope"],
        "dataset": dataset.name,
        "bits": config.bits,
        "rows": int(dataset.image.shape[0]),
        "train_rows": int(dataset.train_idx.size),
        "query_rows": int(dataset.query_idx.size),
        "database_rows": int(dataset.database_idx.size),
        "image_parameter_delta_l2": result.image_parameter_delta,
        "text_parameter_delta_l2": result.text_parameter_delta,
        "pairwise_nll": train_diagnostics["pairwise_nll"],
        "hamming_gap_negative_minus_positive": train_diagnostics[
            "hamming_gap_negative_minus_positive"
        ],
        "quality_gate_passed": quality_gate["passed"],
        "heldout_i2t_map": heldout_retrieval["i2t"]["map"],
        "heldout_t2i_map": heldout_retrieval["t2i"]["map"],
        "heldout_i2t_random_expected_map": heldout_retrieval["i2t"][
            "random_ranking_expected_map"
        ],
        "heldout_t2i_random_expected_map": heldout_retrieval["t2i"][
            "random_ranking_expected_map"
        ],
        "heldout_quality_gate_passed": heldout_quality_gate["passed"],
        "overall_usable": bool(
            quality_gate["passed"] and heldout_quality_gate["passed"]
        ),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _synthetic_dataset(seed: int) -> PreparedDataset:
    rng = np.random.default_rng(seed)
    n_rows, n_classes, latent_dim, feature_dim = 96, 6, 24, 512
    labels = np.zeros((n_rows, n_classes), dtype=np.uint8)
    primary = np.arange(n_rows) % n_classes
    labels[np.arange(n_rows), primary] = 1
    secondary_mask = rng.random(n_rows) < 0.25
    secondary = (primary + rng.integers(1, n_classes, size=n_rows)) % n_classes
    labels[np.flatnonzero(secondary_mask), secondary[secondary_mask]] = 1
    class_latent = rng.normal(size=(n_classes, latent_dim)).astype(np.float32)
    latent = labels.astype(np.float32) @ class_latent
    latent += 0.10 * rng.normal(size=latent.shape).astype(np.float32)
    image_projection = rng.normal(size=(latent_dim, feature_dim)).astype(np.float32)
    text_projection = rng.normal(size=(latent_dim, feature_dim)).astype(np.float32)
    image = latent @ image_projection
    text = latent @ text_projection
    image += 0.05 * rng.normal(size=image.shape).astype(np.float32)
    text += 0.05 * rng.normal(size=text.shape).astype(np.float32)

    query_idx = np.arange(0, 16, dtype=np.int64)
    database_idx = np.arange(16, n_rows, dtype=np.int64)
    train_idx = np.arange(16, 80, dtype=np.int64)
    return PreparedDataset(
        name="synthetic_smoke",
        image=np.ascontiguousarray(image, dtype=np.float32),
        text=np.ascontiguousarray(text, dtype=np.float32),
        labels=labels,
        train_idx=train_idx,
        query_idx=query_idx,
        database_idx=database_idx,
        original_index_base=0,
        split_layout="train_subset_database",
        paths={"source": "deterministic in-memory synthetic data"},
    )


def run_smoke(args: argparse.Namespace) -> None:
    dataset = _synthetic_dataset(args.seed)
    reports: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="dcmh_f_smoke_") as temporary_dir:
        for bits in SUPPORTED_BITS:
            config = TrainConfig(
                bits=bits,
                epochs=args.epochs,
                batch_size=16,
                hidden_dim=32,
                lr=0.03,
                min_lr=0.01,
                gamma=1.0,
                eta=1.0,
                seed=args.seed + bits,
                device=args.device,
                l2_normalize=True,
                initialization="semantic",
                warmup_epochs=args.warmup_epochs,
                warmup_lr=3e-3,
            )
            result = train_dcmh_f(
                dataset.image[dataset.train_idx],
                dataset.text[dataset.train_idx],
                dataset.labels[dataset.train_idx],
                config,
                verbose=False,
            )
            if result.image_parameter_delta <= 0 or result.text_parameter_delta <= 0:
                raise AssertionError("Smoke test found an unchanged branch")
            if not all(
                row["image_gradient_norm"] > 0 and row["text_gradient_norm"] > 0
                for row in result.history
            ):
                raise AssertionError("Smoke test found a zero branch gradient")

            train_diagnostics = training_semantic_diagnostics(
                result.image_model,
                result.text_model,
                dataset.image[dataset.train_idx],
                dataset.text[dataset.train_idx],
                dataset.labels[dataset.train_idx],
                batch_size=16,
                device=result.device,
            )
            quality_gate = dcmh_f_quality_gate(train_diagnostics, bits)
            if not quality_gate["passed"]:
                raise AssertionError(
                    "Synthetic semantic quality gate failed at {} bits: {}".format(
                        bits, quality_gate
                    )
                )

            image_codes = encode_all(
                result.image_model, dataset.image, batch_size=32, device=result.device
            )
            text_codes = encode_all(
                result.text_model, dataset.text, batch_size=32, device=result.device
            )
            i2t = _retrieval_metrics(
                image_codes[dataset.query_idx],
                text_codes[dataset.database_idx],
                dataset.labels[dataset.query_idx],
                dataset.labels[dataset.database_idx],
            )
            t2i = _retrieval_metrics(
                text_codes[dataset.query_idx],
                image_codes[dataset.database_idx],
                dataset.labels[dataset.query_idx],
                dataset.labels[dataset.database_idx],
            )
            random_baseline = _random_retrieval_baseline(
                query_rows=dataset.query_idx.size,
                retrieval_rows=dataset.database_idx.size,
                bits=bits,
                query_labels=dataset.labels[dataset.query_idx],
                retrieval_labels=dataset.labels[dataset.database_idx],
                seed=args.seed + 5000 + bits,
            )
            map_margin = 0.10
            if i2t["map"] < random_baseline["i2t_map_mean"] + map_margin:
                raise AssertionError("Synthetic I2T mAP did not beat random")
            if t2i["map"] < random_baseline["t2i_map_mean"] + map_margin:
                raise AssertionError("Synthetic T2I mAP did not beat random")
            if i2t["mean_positive_hamming"] >= i2t["mean_negative_hamming"]:
                raise AssertionError("Synthetic I2T positive pairs are not closer")
            if t2i["mean_positive_hamming"] >= t2i["mean_negative_hamming"]:
                raise AssertionError("Synthetic T2I positive pairs are not closer")
            heldout_retrieval: Dict[str, object] = {
                "protocol": "synthetic held-out query/database after freeze",
                "i2t": i2t,
                "t2i": t2i,
            }
            heldout_quality_gate = heldout_retrieval_quality_gate(i2t, t2i)
            if not heldout_quality_gate["passed"]:
                raise AssertionError("Synthetic held-out quality gate failed")

            output = Path(temporary_dir) / "dcmh_f_seminit_{}bit.npz".format(bits)
            metadata = build_metadata(
                dataset,
                config,
                result,
                image_codes,
                text_codes,
                train_diagnostics,
                quality_gate,
                heldout_retrieval,
                heldout_quality_gate,
            )
            export_npz(
                output,
                image_codes,
                text_codes,
                dataset.labels,
                dataset.train_idx,
                dataset.query_idx,
                dataset.database_idx,
                metadata,
                overwrite=False,
            )
            with np.load(str(output), allow_pickle=False) as payload:
                expected_keys = {
                    "image_codes",
                    "text_codes",
                    "labels",
                    "train_idx",
                    "query_idx",
                    "database_idx",
                    "metadata_json",
                }
                if set(payload.files) != expected_keys:
                    raise AssertionError("Unexpected NPZ keys {}".format(payload.files))
                if payload["image_codes"].shape != (dataset.image.shape[0], bits):
                    raise AssertionError("Bad image-code shape")
                if payload["text_codes"].shape != (dataset.text.shape[0], bits):
                    raise AssertionError("Bad text-code shape")
                if payload["image_codes"].dtype != np.int8:
                    raise AssertionError("Bad image-code dtype")
                loaded_metadata = json.loads(str(payload["metadata_json"].item()))
                if loaded_metadata["encoder"] != ENCODER_NAME:
                    raise AssertionError("Bad encoder metadata")

            reports.append(
                {
                    "bits": bits,
                    "image_shape": list(image_codes.shape),
                    "text_shape": list(text_codes.shape),
                    "image_parameter_delta_l2": result.image_parameter_delta,
                    "text_parameter_delta_l2": result.text_parameter_delta,
                    "last_image_gradient_norm": result.history[-1][
                        "image_gradient_norm"
                    ],
                    "last_text_gradient_norm": result.history[-1][
                        "text_gradient_norm"
                    ],
                    "pairwise_nll": train_diagnostics["pairwise_nll"],
                    "log_two": train_diagnostics["log_two"],
                    "train_positive_hamming": train_diagnostics[
                        "mean_positive_hamming"
                    ],
                    "train_negative_hamming": train_diagnostics[
                        "mean_negative_hamming"
                    ],
                    "i2t": i2t,
                    "t2i": t2i,
                    "random_baseline": random_baseline,
                    "quality_gate_passed": quality_gate["passed"],
                    "heldout_quality_gate_passed": heldout_quality_gate["passed"],
                    "npz_round_trip": True,
                }
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "encoder": ENCODER_NAME,
                "tested_bits": list(SUPPORTED_BITS),
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/export DCMH-F, a controlled fixed-feature adaptation of "
            "DCMH CVPR 2017 (not an official reproduction)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train on a prepared fixed split")
    train.add_argument("--dataset", type=canonical_dataset_name, required=True)
    train.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data/ProcessData"),
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--bits", type=int, choices=SUPPORTED_BITS, default=64)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--export-batch-size", type=int, default=4096)
    train.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="one ReLU hidden layer; use 0 for a linear hash head",
    )
    train.add_argument("--lr", type=float, default=10.0 ** -1.5)
    train.add_argument("--min-lr", type=float, default=1e-6)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--eta", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=20260805)
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--initialization",
        choices=SUPPORTED_INITIALIZATIONS,
        default="semantic",
        help=(
            "semantic is the evidence-producing DCMH-F-SemInit variant; "
            "random is a failure-prone audit ablation"
        ),
    )
    train.add_argument("--warmup-epochs", type=int, default=20)
    train.add_argument("--warmup-lr", type=float, default=3e-3)
    train.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="disable deterministic per-item L2 normalization",
    )
    train.add_argument("--overwrite", action="store_true")
    train.add_argument(
        "--allow-failed-quality-gate",
        action="store_true",
        help="export a failed artifact for diagnosis only; never use it as evidence",
    )
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=run_train)

    smoke = subparsers.add_parser(
        "smoke", help="synthetic gradient/shape/NPZ test for every supported bit"
    )
    smoke.add_argument("--epochs", type=int, default=10)
    smoke.add_argument("--warmup-epochs", type=int, default=15)
    smoke.add_argument("--seed", type=int, default=20260805)
    smoke.add_argument("--device", default="cpu")
    smoke.set_defaults(func=run_smoke)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
