#!/usr/bin/env python3
"""CIRH-F: CIRH's two-stage objective on the project's fixed 512-D features.

This is a controlled-feature adaptation of Correlation-Identity
Reconstruction Hashing (CIRH), not an official numerical reproduction.

Paper:
    Lei Zhu, Xize Wu, Jingjing Li, Zheng Zhang, Weili Guan, Heng Tao Shen,
    "Work Together: Correlation-Identity Reconstruction Hashing for
    Unsupervised Cross-Modal Retrieval," IEEE TKDE 35(9), 2023.
    DOI: 10.1109/TKDE.2022.3218656

Official source audited here:
    https://github.com/XizeWu/CIRH
    commit 0f6439d3cad0240ea4a9924ff168aa2cef3b3b1e

CIRH-F deliberately retains the official implementation's central mechanics:

* the full train-only collaborated similarity graph S;
* neighbor mixing with the third/fourth strongest within-batch affinities;
* the joint graph/reconstruction network and its identity-reconstruction,
  correlation-reconstruction, and discretization losses;
* the second-stage independent image/text hash functions, trained against the
  joint binary codes and correlation-identity consistency losses; and
* independent sign(image_net(x)) and sign(text_net(t)) out-of-sample codes.

Necessary controlled adaptations are explicit and auditable:

* fixed 512-D image/text features and the fixed indT/indQ/indD split replace
  CIRH's released feature files and split;
* device-agnostic modern PyTorch replaces hard-coded CUDA calls;
* the second-stage S_batch is indexed by the actual shuffled item IDs.  The
  official code slices S by batch *positions*, although its features and B are
  in shuffled record_index order; that silent misalignment is corrected;
* sign(0) maps to +1 so every exported code is genuinely binary;
* the official query-mAP checkpoint selection is removed.  Training has a
  fixed epoch count and never receives query/database features or labels;
* the final fixed checkpoint is encoded once, then held-out labels are opened
  only for a post-freeze quality audit.  They never select epochs, parameters,
  thresholds, or variants.

The default hyperparameters and modality-specific hash-network widths match
``main_mir.py`` and ``models.py`` at the audited commit.  ``-F`` must remain in
the reporting name because changing the input representation changes the
experimental protocol.
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
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.nn import functional as F


FORMAT_VERSION = 1
ENCODER_NAME = "CIRH-F"
SUPPORTED_BITS = (16, 32, 64, 128)
PAPER_URL = "https://doi.org/10.1109/TKDE.2022.3218656"
OFFICIAL_REPOSITORY = "https://github.com/XizeWu/CIRH"
OFFICIAL_COMMIT = "0f6439d3cad0240ea4a9924ff168aa2cef3b3b1e"

# SHA-256 values independently read from the fixed official checkout.
OFFICIAL_SOURCE_SHA256: Mapping[str, str] = {
    "README.md": "56663f3bd4376277339597936083dafa4ef8e3aae07f5d55658d5ee9cf094b3b",
    "load_data.py": "6191f16dbf53d5e1267c12214c0d719863af63a5addaccac84f8c186e74b5a4e",
    "main_mir.py": "a4e9594a2cadaae83e772128721ae58a7b079152c2a641aa21752ebff64616f5",
    "models.py": "333213135cb59b274b9694c6b5dfb43aed990c9fa8ff0448d860202590e9c28c",
    "my_opt.py": "1d00660e2ec69963ea757dcdbe8fdbe0af0da01f1d38e98ace74c95f2a0f3f0f",
    "utils.py": "a2787b20e44e0f58ae6af1e908a1ab2691efab1a8dbbd982afbcfab3d88ac339",
}

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
}

DATASET_ALIASES = {
    "coco": "mscoco",
    "ms-coco": "mscoco",
    "flickr": "mirflickr",
    "flickr25k": "mirflickr",
    "mirflickr25k": "mirflickr",
    "nus": "nuswide",
    "nus-wide": "nuswide",
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
    """Defaults reproduce the audited MIR configuration where applicable."""

    bits: int = 64
    epochs: int = 60
    batch_size: int = 512
    lr_joint: float = 1e-3
    lr_image: float = 1e-4
    lr_text: float = 1e-4
    lambda_identity: float = 10.0
    lambda_correlation: float = 1.0
    beta: float = 0.01
    graph_k: int = 3000
    graph_image_weight: float = 0.6
    graph_direct_weight: float = 0.6
    image_hidden_dim: int = 4096
    seed: int = 20260805
    device: str = "auto"

    def validate(self, n_train: Optional[int] = None) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise ValueError("bits must be one of {}, got {}".format(
                SUPPORTED_BITS, self.bits
            ))
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 4:
            raise ValueError("batch_size must be at least four for CIRH neighbors")
        for name, value in (
            ("lr_joint", self.lr_joint),
            ("lr_image", self.lr_image),
            ("lr_text", self.lr_text),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(name))
        for name, value in (
            ("lambda_identity", self.lambda_identity),
            ("lambda_correlation", self.lambda_correlation),
            ("beta", self.beta),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if self.graph_k < 1:
            raise ValueError("graph_k must be positive")
        for name, value in (
            ("graph_image_weight", self.graph_image_weight),
            ("graph_direct_weight", self.graph_direct_weight),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("{} must lie in [0,1]".format(name))
        if self.image_hidden_dim < 1:
            raise ValueError("image_hidden_dim must be positive")
        if n_train is not None:
            if n_train < 4:
                raise ValueError("CIRH-F needs at least four training pairs")
            if self.graph_k >= n_train:
                raise ValueError(
                    "graph_k={} must be smaller than n_train={}".format(
                        self.graph_k, n_train
                    )
                )


@dataclass
class TrainingResult:
    image_model: "ImageHashNet"
    text_model: "TextHashNet"
    joint_model: "CIRHJointNet"
    history: List[Dict[str, float]]
    device: str
    runtime_seconds: float
    image_parameter_delta: float
    text_parameter_delta: float
    joint_parameter_delta: float
    graph_diagnostics: Dict[str, object]


def canonical_dataset_name(value: str) -> str:
    name = DATASET_ALIASES.get(value.strip().lower(), value.strip().lower())
    if name not in DATASETS:
        valid = sorted(set(DATASETS).union(DATASET_ALIASES))
        raise argparse.ArgumentTypeError(
            "Unknown dataset {!r}; choose one of {}".format(value, valid)
        )
    return name


def first_payload(path: Path) -> np.ndarray:
    try:
        data = sio.loadmat(str(path))
    except NotImplementedError as exc:
        raise ValueError(
            "{} is MATLAB v7.3/HDF5; convert it before CIRH-F".format(path)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
            "Labels must have shape [N,C] or padded [N,K], got {} for N={}".format(
                labels.shape, n_rows
            )
        )
    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels contain non-finite values")
    unique = np.unique(labels)
    if np.all(np.isin(unique, np.array([0, 1]))):
        result = labels.astype(np.uint8, copy=False)
    else:
        if not np.all(np.equal(labels, np.floor(labels))):
            raise ValueError("Non-binary labels must be integer class IDs")
        ids = labels.astype(np.int64)
        if ids.min() < 0 or ids.max() < 1:
            raise ValueError("Padded class IDs must use zero padding and positive IDs")
        result = np.zeros((n_rows, int(ids.max())), dtype=np.uint8)
        rows = np.repeat(np.arange(n_rows), ids.shape[1])
        flattened = ids.reshape(-1)
        valid = flattened > 0
        result[rows[valid], flattened[valid] - 1] = 1
    if np.any(result.sum(axis=1) == 0):
        raise ValueError("Every item must have at least one positive label")
    return np.ascontiguousarray(result)


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
        raise ValueError("Ambiguous split index base")
    converted = {key: value - index_base for key, value in raw.items()}
    for key, values in converted.items():
        if values.size == 0:
            raise ValueError("Empty split array {}".format(key))
        if values.min() < 0 or values.max() >= n_rows:
            raise ValueError("Out-of-range indices in {}".format(key))
        if np.unique(values).size != values.size:
            raise ValueError("Duplicate indices in {}".format(key))
    train = np.ascontiguousarray(converted["indT"], dtype=np.int64)
    query = np.ascontiguousarray(converted["indQ"], dtype=np.int64)
    database = np.ascontiguousarray(converted["indD"], dtype=np.int64)
    if np.intersect1d(train, query).size:
        raise ValueError("Protocol violation: train/query overlap")
    if np.intersect1d(query, database).size:
        raise ValueError("Protocol violation: query/database overlap")
    train_in_database = bool(np.all(np.isin(train, database)))
    if train_in_database:
        if np.union1d(query, database).size != n_rows:
            raise ValueError("Query and database must cover all rows")
        layout = "train_subset_database"
    else:
        if np.intersect1d(train, database).size:
            raise ValueError("Partial train/database overlap is unsupported")
        covered = np.union1d(np.union1d(train, query), database)
        if covered.size != n_rows:
            raise ValueError("Train/query/database must cover all rows")
        layout = "train_disjoint_database"
    return train, query, database, index_base, layout


def load_prepared_dataset(data_root: Path, name: str) -> PreparedDataset:
    canonical = canonical_dataset_name(name)
    spec = DATASETS[canonical]
    root = data_root / spec["root"]
    paths = {key: root / spec[key] for key in ("image", "text", "labels", "index")}
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
        raise ValueError("CIRH-F protocol requires fixed 512-D features")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(text)):
        raise ValueError("Features contain non-finite values")
    labels = _convert_labels(first_payload(paths["labels"]), image.shape[0])
    train, query, database, base, layout = _load_and_validate_indices(
        paths["index"], image.shape[0]
    )
    return PreparedDataset(
        name=canonical,
        image=image,
        text=text,
        labels=labels,
        train_idx=train,
        query_idx=query,
        database_idx=database,
        original_index_base=base,
        split_layout=layout,
        paths={key: str(path.resolve()) for key, path in paths.items()},
    )


def stable_sign(values: torch.Tensor) -> torch.Tensor:
    """Map zero to +1, unlike torch.sign, which would export ternary codes."""

    return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


def _resolve_device(requested: str) -> torch.device:
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class CIRHJointNet(nn.Module):
    """The official MyNet graph/reconstruction network, device-agnostic."""

    def __init__(self, bits: int, image_dim: int, text_dim: int) -> None:
        super().__init__()
        self.bits = int(bits)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True)
        )
        self.image_graph = nn.Linear(512, 512)
        self.image_graph_bn = nn.BatchNorm1d(512)
        self.text_graph = nn.Linear(512, 512)
        self.text_graph_bn = nn.BatchNorm1d(512)
        self.joint_graph = nn.Linear(512, 512)
        self.joint_graph_bn = nn.BatchNorm1d(512)
        self.joint_hash = nn.Linear(512, bits)
        self.image_hash = nn.Linear(1024, bits)
        self.text_hash = nn.Linear(1024, bits)
        self.combine_hash = nn.Linear(3 * bits, bits)
        self.image_hash_bn = nn.BatchNorm1d(bits)
        self.text_hash_bn = nn.BatchNorm1d(bits)
        self.joint_hash_bn = nn.BatchNorm1d(bits)
        self.combine_hash_bn = nn.BatchNorm1d(bits)
        self.image_decoder = nn.Sequential(
            nn.Linear(bits, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, image_dim),
            nn.BatchNorm1d(image_dim),
        )
        self.text_decoder = nn.Sequential(
            nn.Linear(bits, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, text_dim),
            nn.BatchNorm1d(text_dim),
        )

    def forward(
        self, image: torch.Tensor, text: torch.Tensor, affinity: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        batch = image.shape[0]
        if image.ndim != 2 or text.ndim != 2 or image.shape[0] != text.shape[0]:
            raise ValueError("Expected paired image/text matrices")
        if affinity.shape != (batch, batch):
            raise ValueError("Affinity must be square with the batch size")
        vi = F.normalize(self.image_encoder(image), dim=1, eps=1e-8)
        vt = F.normalize(self.text_encoder(text), dim=1, eps=1e-8)
        vgi = F.relu(self.image_graph_bn(affinity @ self.image_graph(vi)))
        vgt = F.relu(self.text_graph_bn(affinity @ self.text_graph(vt)))

        vc = torch.cat((vi, vt), dim=0)
        identity = torch.eye(batch, dtype=affinity.dtype, device=affinity.device)
        cma = torch.cat(
            (
                torch.cat((affinity, identity), dim=1),
                torch.cat((identity, affinity), dim=1),
            ),
            dim=0,
        )
        vj_both = self.joint_graph_bn(cma @ self.joint_graph(vc))
        vj = F.relu(vj_both[:batch] + vj_both[batch:])
        hj = self.joint_hash_bn(self.joint_hash(vj))
        hi = self.image_hash_bn(self.image_hash(torch.cat((vgi, vj), dim=1)))
        ht = self.text_hash_bn(self.text_hash(torch.cat((vj, vgt), dim=1)))
        h = torch.tanh(
            self.combine_hash_bn(self.combine_hash(torch.cat((hi, hj, ht), dim=1)))
        )
        binary = stable_sign(h)
        decoded_image = self.image_decoder(h + torch.tanh(hi))
        decoded_text = self.text_decoder(h + torch.tanh(ht))
        return hi, ht, h, binary, decoded_image, decoded_text


class ImageHashNet(nn.Module):
    """Official Img_Net with its 4096-wide default hidden layer."""

    def __init__(self, bits: int, input_dim: int, hidden_dim: int = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.hash_head = nn.Linear(hidden_dim, bits)
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=1.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hash_head(self.dropout(F.relu(self.fc1(values)))))


class TextHashNet(nn.Module):
    """Official Txt_Net; its hidden width equals the text input dimension."""

    def __init__(self, bits: int, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.3)
        self.hash_head = nn.Linear(input_dim, bits)
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=1.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hash_head(self.dropout(F.relu(self.fc1(values)))))


@torch.no_grad()
def build_collaborated_similarity(
    image: torch.Tensor,
    text: torch.Tensor,
    graph_k: int,
    image_weight: float,
    direct_weight: float,
) -> torch.Tensor:
    """Build official CIRH S using only paired training features.

    ``topk(largest=False)`` is equivalent to the official full argsort followed
    by zeroing the K smallest entries, while avoiding an unnecessary N x N
    int64 permutation matrix.  Boundary ties can choose different equal-valued
    entries, which is recorded as an implementation modernization.
    """

    if image.ndim != 2 or text.ndim != 2 or image.shape[0] != text.shape[0]:
        raise ValueError("Expected paired feature matrices")
    n_train = image.shape[0]
    if not 1 <= graph_k < n_train:
        raise ValueError("graph_k must be in [1,n_train)")
    image_n = F.normalize(image, dim=1, eps=1e-8)
    text_n = F.normalize(text, dim=1, eps=1e-8)
    direct = image_weight * (image_n @ image_n.T)
    direct.add_((1.0 - image_weight) * (text_n @ text_n.T))
    smallest = torch.topk(
        direct, k=graph_k, dim=1, largest=False, sorted=False
    ).indices
    direct.scatter_(1, smallest, 0.0)
    transformed = 2.0 * torch.sigmoid(direct) - 1.0
    transformed.add_(torch.eye(n_train, dtype=direct.dtype, device=direct.device))
    transformed = 0.5 * (transformed + transformed.T)
    return direct_weight * direct + (1.0 - direct_weight) * transformed


def select_official_neighbors(affinity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CIRH's fourth- and third-largest affinity indices per row."""

    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square")
    if affinity.shape[0] < 4:
        raise ValueError("at least four samples are required")
    strongest = torch.topk(affinity, k=4, dim=1, largest=True, sorted=True).indices
    return strongest[:, 3], strongest[:, 2]


def select_square(matrix: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    """Select S[item_ids][:, item_ids], preserving true shuffled item identity."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if item_ids.ndim != 1 or item_ids.dtype != torch.long:
        raise ValueError("item_ids must be a one-dimensional torch.long tensor")
    return matrix.index_select(0, item_ids).index_select(1, item_ids)


def joint_objective(
    hi: torch.Tensor,
    ht: torch.Tensor,
    h: torch.Tensor,
    binary: torch.Tensor,
    decoded_image: torch.Tensor,
    decoded_text: torch.Tensor,
    target_image: torch.Tensor,
    target_text: torch.Tensor,
    affinity: torch.Tensor,
    lambda_identity: float,
    lambda_correlation: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    identity = lambda_identity * (
        F.mse_loss(decoded_image, target_image)
        + F.mse_loss(decoded_text, target_text)
    )
    correlation = lambda_correlation * (
        F.mse_loss(F.normalize(hi, dim=1, eps=1e-8) @ F.normalize(hi, dim=1, eps=1e-8).T, affinity)
        + F.mse_loss(F.normalize(ht, dim=1, eps=1e-8) @ F.normalize(ht, dim=1, eps=1e-8).T, affinity)
        + F.mse_loss(F.normalize(h, dim=1, eps=1e-8) @ F.normalize(h, dim=1, eps=1e-8).T, affinity)
    )
    discretization = F.mse_loss(h, binary.detach())
    total = identity + correlation + discretization
    return total, {
        "identity": identity,
        "correlation": correlation,
        "discretization": discretization,
    }


def hash_function_objective(
    image_hash: torch.Tensor,
    text_hash: torch.Tensor,
    target_binary: torch.Tensor,
    affinity: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    identity = (
        F.mse_loss(image_hash, target_binary)
        + F.mse_loss(text_hash, target_binary)
        + F.mse_loss(image_hash, text_hash)
    )
    image_n = F.normalize(image_hash, dim=1, eps=1e-8)
    text_n = F.normalize(text_hash, dim=1, eps=1e-8)
    correlation = beta * (
        F.mse_loss(image_n @ text_n.T, affinity)
        + F.mse_loss(image_n @ image_n.T, affinity)
        + F.mse_loss(text_n @ text_n.T, affinity)
    )
    return identity + correlation, {"identity": identity, "correlation": correlation}


def _parameter_snapshot(model: nn.Module) -> List[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def _parameter_delta(before: Sequence[torch.Tensor], model: nn.Module) -> float:
    squared = 0.0
    for original, current in zip(before, model.parameters()):
        difference = current.detach().cpu() - original
        squared += float(torch.sum(difference * difference).item())
    return math.sqrt(squared)


def _batch_indices(permutation: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    chunks = list(torch.split(permutation, batch_size))
    # BatchNorm cannot train on a singleton.  Merge it into the preceding batch
    # instead of silently dropping a training item.
    if len(chunks) > 1 and chunks[-1].numel() == 1:
        chunks[-2] = torch.cat((chunks[-2], chunks[-1]))
        chunks.pop()
    return chunks


def _matrix_bytes(n_train: int) -> Dict[str, int]:
    return {
        "one_float32_n_by_n": int(4 * n_train * n_train),
        "one_int64_n_by_k_at_default_k": int(8 * n_train * min(3000, n_train - 1)),
    }


def train_cirh_f(
    train_image: np.ndarray,
    train_text: np.ndarray,
    config: TrainConfig,
    verbose: bool = True,
) -> TrainingResult:
    """Train CIRH-F without accepting any labels or held-out arrays."""

    image_np = np.ascontiguousarray(train_image, dtype=np.float32)
    text_np = np.ascontiguousarray(train_text, dtype=np.float32)
    if image_np.ndim != 2 or text_np.ndim != 2 or image_np.shape[0] != text_np.shape[0]:
        raise ValueError("Expected paired [N,D] train features")
    if not np.all(np.isfinite(image_np)) or not np.all(np.isfinite(text_np)):
        raise ValueError("Train features contain non-finite values")
    n_train = int(image_np.shape[0])
    config.validate(n_train)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    image = torch.from_numpy(image_np).to(device)
    text = torch.from_numpy(text_np).to(device)

    start_time = time.perf_counter()
    similarity = build_collaborated_similarity(
        image,
        text,
        graph_k=config.graph_k,
        image_weight=config.graph_image_weight,
        direct_weight=config.graph_direct_weight,
    )
    if not torch.isfinite(similarity).all():
        raise RuntimeError("Collaborated similarity contains non-finite values")
    graph_diagnostics: Dict[str, object] = {
        "shape": list(similarity.shape),
        "minimum": float(similarity.min().item()),
        "maximum": float(similarity.max().item()),
        "mean": float(similarity.mean().item()),
        "rms": float(torch.sqrt(torch.mean(similarity ** 2)).item()),
        "symmetry_max_abs_error": float(torch.max(torch.abs(similarity - similarity.T)).item()),
        "memory_estimate": _matrix_bytes(n_train),
        "construction_inputs": "train image/text features only",
    }

    joint = CIRHJointNet(config.bits, image_np.shape[1], text_np.shape[1]).to(device)
    image_model = ImageHashNet(
        config.bits, image_np.shape[1], hidden_dim=config.image_hidden_dim
    ).to(device)
    text_model = TextHashNet(config.bits, text_np.shape[1]).to(device)
    joint_before = _parameter_snapshot(joint)
    image_before = _parameter_snapshot(image_model)
    text_before = _parameter_snapshot(text_model)
    joint_optimizer = torch.optim.Adam(joint.parameters(), lr=config.lr_joint)
    image_optimizer = torch.optim.Adam(image_model.parameters(), lr=config.lr_image)
    text_optimizer = torch.optim.Adam(text_model.parameters(), lr=config.lr_text)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 991)
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        permutation_cpu = torch.randperm(n_train, generator=generator)
        chunks = _batch_indices(permutation_cpu, config.batch_size)
        epoch_binary = torch.empty((n_train, config.bits), device=device)
        joint.train()
        accum = {
            "joint_total": 0.0,
            "joint_identity": 0.0,
            "joint_correlation": 0.0,
            "joint_discretization": 0.0,
        }
        seen = 0
        for ids_cpu in chunks:
            ids = ids_cpu.to(device)
            batch_image = image.index_select(0, ids)
            batch_text = text.index_select(0, ids)
            batch_similarity = select_square(similarity, ids)
            neighbor_fourth, neighbor_third = select_official_neighbors(batch_similarity)
            mixed_image = (
                0.7 * batch_image
                + 0.2 * batch_image.index_select(0, neighbor_third)
                + 0.1 * batch_image.index_select(0, neighbor_fourth)
            )
            mixed_text = (
                0.7 * batch_text
                + 0.2 * batch_text.index_select(0, neighbor_third)
                + 0.1 * batch_text.index_select(0, neighbor_fourth)
            )
            outputs = joint(mixed_image, mixed_text, batch_similarity)
            loss, pieces = joint_objective(
                *outputs,
                mixed_image,
                mixed_text,
                batch_similarity,
                config.lambda_identity,
                config.lambda_correlation,
            )
            joint_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            joint_optimizer.step()
            epoch_binary.index_copy_(0, ids, outputs[3].detach())
            weight = int(ids.numel())
            seen += weight
            accum["joint_total"] += float(loss.detach().item()) * weight
            for name in ("identity", "correlation", "discretization"):
                accum["joint_" + name] += float(pieces[name].detach().item()) * weight

        image_model.train()
        text_model.train()
        hash_total = hash_identity = hash_correlation = 0.0
        hash_seen = 0
        # Keep the same epoch permutation as the official two-stage cycle, but
        # index S by the *actual item IDs* to fix the official misalignment.
        for ids_cpu in chunks:
            ids = ids_cpu.to(device)
            batch_image = F.normalize(image.index_select(0, ids), dim=1, eps=1e-8)
            batch_text = F.normalize(text.index_select(0, ids), dim=1, eps=1e-8)
            batch_binary = epoch_binary.index_select(0, ids)
            batch_similarity = select_square(similarity, ids)
            image_hash = image_model(batch_image)
            text_hash = text_model(batch_text)
            loss, pieces = hash_function_objective(
                image_hash, text_hash, batch_binary, batch_similarity, config.beta
            )
            image_optimizer.zero_grad(set_to_none=True)
            text_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            image_optimizer.step()
            text_optimizer.step()
            weight = int(ids.numel())
            hash_seen += weight
            hash_total += float(loss.detach().item()) * weight
            hash_identity += float(pieces["identity"].detach().item()) * weight
            hash_correlation += float(pieces["correlation"].detach().item()) * weight

        record = {
            "epoch": float(epoch + 1),
            **{name: value / seen for name, value in accum.items()},
            "hash_total": hash_total / hash_seen,
            "hash_identity": hash_identity / hash_seen,
            "hash_correlation": hash_correlation / hash_seen,
            "joint_binary_plus_fraction": float((epoch_binary > 0).float().mean().item()),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise RuntimeError("Non-finite training history at epoch {}".format(epoch + 1))
        history.append(record)
        if verbose:
            print(json.dumps(record, sort_keys=True), flush=True)

    runtime = time.perf_counter() - start_time
    return TrainingResult(
        image_model=image_model,
        text_model=text_model,
        joint_model=joint,
        history=history,
        device=str(device),
        runtime_seconds=float(runtime),
        image_parameter_delta=_parameter_delta(image_before, image_model),
        text_parameter_delta=_parameter_delta(text_before, text_model),
        joint_parameter_delta=_parameter_delta(joint_before, joint),
        graph_diagnostics=graph_diagnostics,
    )


@torch.no_grad()
def encode_all(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = np.ascontiguousarray(features, dtype=np.float32)
    resolved = torch.device(device)
    model.eval()
    outputs: List[np.ndarray] = []
    for start in range(0, values.shape[0], batch_size):
        batch = torch.from_numpy(values[start:start + batch_size]).to(resolved)
        batch = F.normalize(batch, dim=1, eps=1e-8)
        outputs.append(stable_sign(model(batch)).to(torch.int8).cpu().numpy())
    result = np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.int8)
    if not np.all(np.isin(np.unique(result), np.array([-1, 1], dtype=np.int8))):
        raise RuntimeError("Non-binary output")
    return result


def _code_statistics(codes: np.ndarray) -> Dict[str, float]:
    means = codes.astype(np.float32).mean(axis=0)
    return {
        "plus_one_fraction": float(np.mean(codes > 0)),
        "mean_abs_bit_mean": float(np.mean(np.abs(means))),
        "max_abs_bit_mean": float(np.max(np.abs(means))),
        "unique_rows": int(np.unique(codes, axis=0).shape[0]),
    }


def structural_quality_gate(
    result: TrainingResult,
    train_image_codes: np.ndarray,
    train_text_codes: np.ndarray,
) -> Dict[str, object]:
    image_stats = _code_statistics(train_image_codes)
    text_stats = _code_statistics(train_text_codes)
    paired_hamming = 0.5 * np.mean(
        train_image_codes.shape[1]
        - np.sum(
            train_image_codes.astype(np.int16) * train_text_codes.astype(np.int16),
            axis=1,
        )
    )
    checks = {
        "history_finite": bool(
            all(math.isfinite(value) for row in result.history for value in row.values())
        ),
        "joint_parameters_updated": bool(result.joint_parameter_delta > 1e-8),
        "image_parameters_updated": bool(result.image_parameter_delta > 1e-8),
        "text_parameters_updated": bool(result.text_parameter_delta > 1e-8),
        "image_codes_nonconstant": bool(image_stats["unique_rows"] > 1),
        "text_codes_nonconstant": bool(text_stats["unique_rows"] > 1),
        "image_bits_not_globally_collapsed": bool(image_stats["mean_abs_bit_mean"] < 0.95),
        "text_bits_not_globally_collapsed": bool(text_stats["mean_abs_bit_mean"] < 0.95),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "image_code_statistics": image_stats,
        "text_code_statistics": text_stats,
        "paired_image_text_mean_hamming": float(paired_hamming),
        "policy": "label-free structural gate before held-out labels are opened",
    }


def _retrieval_metrics(
    query_codes: np.ndarray,
    retrieval_codes: np.ndarray,
    query_labels: np.ndarray,
    retrieval_labels: np.ndarray,
) -> Dict[str, float]:
    bits = int(query_codes.shape[1])
    n_database = int(retrieval_codes.shape[0])
    harmonic = float(np.sum(1.0 / np.arange(1, n_database + 1)))
    aps: List[float] = []
    random_aps: List[float] = []
    positive_sum = negative_sum = 0.0
    positive_count = negative_count = 0
    database_i16 = retrieval_codes.astype(np.int16, copy=False)
    for start in range(0, query_codes.shape[0], 64):
        stop = min(start + 64, query_codes.shape[0])
        relevance = query_labels[start:stop] @ retrieval_labels.T > 0
        distances = 0.5 * (
            bits
            - query_codes[start:stop].astype(np.int16, copy=False) @ database_i16.T
        )
        positive_sum += float(distances[relevance].sum())
        negative_sum += float(distances[~relevance].sum())
        positive_count += int(relevance.sum())
        negative_count += int(relevance.size - relevance.sum())
        for row in range(stop - start):
            order = np.argsort(distances[row], kind="stable")
            ranked = relevance[row, order]
            relevant_count = int(ranked.sum())
            if relevant_count == 0:
                continue
            precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
            aps.append(float(np.sum(precision * ranked) / relevant_count))
            random_aps.append(float(
                (
                    harmonic
                    + (relevant_count - 1) / (n_database - 1) * (n_database - harmonic)
                ) / n_database
            ))
    if not aps or positive_count == 0 or negative_count == 0:
        raise ValueError("Retrieval audit requires queries with positive and negative pairs")
    return {
        "map": float(np.mean(aps)),
        "random_ranking_expected_map": float(np.mean(random_aps)),
        "map_gain_over_random": float(np.mean(aps) - np.mean(random_aps)),
        "evaluated_queries": int(len(aps)),
        "mean_positive_hamming": float(positive_sum / positive_count),
        "mean_negative_hamming": float(negative_sum / negative_count),
        "tie_policy": "stable item order; asset gate only, not paper primary metric",
    }


def heldout_quality_gate(
    i2t: Mapping[str, float], t2i: Mapping[str, float]
) -> Dict[str, object]:
    minimum_gain = 0.05
    checks = {
        "i2t_map_beats_random_by_0.05": bool(i2t["map_gain_over_random"] >= minimum_gain),
        "t2i_map_beats_random_by_0.05": bool(t2i["map_gain_over_random"] >= minimum_gain),
        "i2t_positive_pairs_are_closer": bool(i2t["mean_positive_hamming"] < i2t["mean_negative_hamming"]),
        "t2i_positive_pairs_are_closer": bool(t2i["mean_positive_hamming"] < t2i["mean_negative_hamming"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {"minimum_map_gain_over_random": minimum_gain},
        "role": (
            "single post-freeze acceptance audit; held-out labels never select "
            "an epoch, checkpoint, hyperparameter, threshold, or method variant"
        ),
    }


def export_npz(
    output: Path,
    image_codes: np.ndarray,
    text_codes: np.ndarray,
    dataset: PreparedDataset,
    metadata: Mapping[str, object],
    overwrite: bool,
) -> None:
    if output.suffix.lower() != ".npz":
        raise ValueError("Output must end in .npz")
    if output.exists() and not overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(output))
    if image_codes.shape != text_codes.shape or image_codes.shape[0] != dataset.labels.shape[0]:
        raise ValueError("Code shape mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        image_codes=np.ascontiguousarray(image_codes, dtype=np.int8),
        text_codes=np.ascontiguousarray(text_codes, dtype=np.int8),
        labels=np.ascontiguousarray(dataset.labels, dtype=np.uint8),
        train_idx=np.ascontiguousarray(dataset.train_idx, dtype=np.int64),
        query_idx=np.ascontiguousarray(dataset.query_idx, dtype=np.int64),
        database_idx=np.ascontiguousarray(dataset.database_idx, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    os.replace(str(temporary), str(output))


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _build_metadata(
    dataset: PreparedDataset,
    config: TrainConfig,
    result: TrainingResult,
    image_codes: np.ndarray,
    text_codes: np.ndarray,
    structural_gate: Mapping[str, object],
    heldout: Mapping[str, object],
    heldout_gate: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "encoder": ENCODER_NAME,
        "reporting_name": ENCODER_NAME,
        "claim_scope": "controlled fixed-feature adaptation; not an official CIRH reproduction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "title": "Work Together: Correlation-Identity Reconstruction Hashing for Unsupervised Cross-Modal Retrieval",
            "authors": "Lei Zhu; Xize Wu; Jingjing Li; Zheng Zhang; Weili Guan; Heng Tao Shen",
            "venue": "IEEE TKDE 35(9), 2023, 8838-8851",
            "doi": "10.1109/TKDE.2022.3218656",
            "url": PAPER_URL,
        },
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "file_sha256": dict(OFFICIAL_SOURCE_SHA256),
        },
        "retained_components": [
            "train-only collaborated similarity graph",
            "official 0.7 self + 0.2 third-neighbor + 0.1 fourth-neighbor mixing",
            "joint homogeneous/heterogeneous graph aggregation",
            "identity reconstruction of both modality features",
            "correlation reconstruction for HI, HT, and fused H",
            "H-to-sign(H) discretization",
            "second-stage correlation-identity-consistent independent hash functions",
        ],
        "controlled_adaptations": [
            "project fixed 512-D features and fixed split replace released CIRH inputs",
            "modern device-agnostic PyTorch replaces hard-coded CUDA/Variable calls",
            "K-smallest zeroing uses equivalent topk rather than storing a full argsort",
            "second-stage similarity blocks use actual shuffled item IDs (official indexing bug fixed)",
            "sign(0)=+1 guarantees binary output",
            "fixed final epoch replaces held-out query-mAP checkpoint selection",
        ],
        "leakage_contract": {
            "optimizer_function_signature": "train_cirh_f(train_image, train_text, config); no labels accepted",
            "optimizer_inputs": "image[train_idx] and text[train_idx] only",
            "query_database_features": "encoded only after all optimizers and epoch selection are finished",
            "query_database_labels": "opened once for a post-freeze audit and export",
            "checkpoint_selection": "none; the fixed final epoch is used",
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
            "split_sha256": _split_sha256(dataset.train_idx, dataset.query_idx, dataset.database_idx),
            "train_image_sha256": _array_sha256(dataset.image[dataset.train_idx]),
            "train_text_sha256": _array_sha256(dataset.text[dataset.train_idx]),
        },
        "training": {
            **asdict(config),
            "resolved_device": result.device,
            "runtime_seconds": result.runtime_seconds,
            "joint_parameter_delta_l2": result.joint_parameter_delta,
            "image_parameter_delta_l2": result.image_parameter_delta,
            "text_parameter_delta_l2": result.text_parameter_delta,
            "graph_diagnostics": result.graph_diagnostics,
            "history": result.history,
            "structural_quality_gate": structural_gate,
            "heldout_retrieval": heldout,
            "heldout_quality_gate": heldout_gate,
            "overall_usable": bool(structural_gate["passed"] and heldout_gate["passed"]),
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
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("--output and --manifest must be different files")
    for name, path in (("output", args.output), ("manifest", args.manifest)):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                "{} {} exists; pass --overwrite before starting training".format(
                    name, path
                )
            )
    config = TrainConfig(
        bits=args.bits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_joint=args.lr_joint,
        lr_image=args.lr_image,
        lr_text=args.lr_text,
        lambda_identity=args.lambda_identity,
        lambda_correlation=args.lambda_correlation,
        beta=args.beta,
        graph_k=args.graph_k,
        graph_image_weight=args.graph_image_weight,
        graph_direct_weight=args.graph_direct_weight,
        image_hidden_dim=args.image_hidden_dim,
        seed=args.seed,
        device=args.device,
    )
    dataset = load_prepared_dataset(args.data_root, args.dataset)
    result = train_cirh_f(
        dataset.image[dataset.train_idx],
        dataset.text[dataset.train_idx],
        config,
        verbose=not args.quiet,
    )
    train_image_codes = encode_all(
        result.image_model, dataset.image[dataset.train_idx], args.export_batch_size, result.device
    )
    train_text_codes = encode_all(
        result.text_model, dataset.text[dataset.train_idx], args.export_batch_size, result.device
    )
    structural_gate = structural_quality_gate(result, train_image_codes, train_text_codes)
    print(json.dumps({"structural_quality_gate": structural_gate}, indent=2, sort_keys=True))
    if not structural_gate["passed"] and not args.allow_failed_quality_gate:
        raise RuntimeError("CIRH-F structural gate failed; no artifact exported")

    # This is the first point at which any non-train row is passed to a model.
    # Optimizers are gone and the fixed final checkpoint cannot change.
    image_codes = encode_all(result.image_model, dataset.image, args.export_batch_size, result.device)
    text_codes = encode_all(result.text_model, dataset.text, args.export_batch_size, result.device)
    heldout = {
        "protocol": "fixed query_idx against fixed database_idx after final-checkpoint freeze",
        "i2t": _retrieval_metrics(
            image_codes[dataset.query_idx], text_codes[dataset.database_idx],
            dataset.labels[dataset.query_idx], dataset.labels[dataset.database_idx],
        ),
        "t2i": _retrieval_metrics(
            text_codes[dataset.query_idx], image_codes[dataset.database_idx],
            dataset.labels[dataset.query_idx], dataset.labels[dataset.database_idx],
        ),
    }
    heldout_gate = heldout_quality_gate(heldout["i2t"], heldout["t2i"])
    print(json.dumps({"heldout_retrieval": heldout, "heldout_quality_gate": heldout_gate}, indent=2, sort_keys=True))
    if not heldout_gate["passed"] and not args.allow_failed_quality_gate:
        raise RuntimeError("CIRH-F held-out gate failed; no artifact exported")

    metadata = _build_metadata(
        dataset, config, result, image_codes, text_codes, structural_gate, heldout, heldout_gate
    )
    export_npz(args.output, image_codes, text_codes, dataset, metadata, args.overwrite)
    manifest = {
        "status": "USABLE" if metadata["training"]["overall_usable"] else "FAILED_QUALITY_GATE",
        "encoder": ENCODER_NAME,
        "claim_scope": metadata["claim_scope"],
        "dataset": dataset.name,
        "bits": config.bits,
        "seed": config.seed,
        "split_sha256": metadata["dataset"]["split_sha256"],
        "rows": int(dataset.image.shape[0]),
        "train_rows": int(dataset.train_idx.size),
        "query_rows": int(dataset.query_idx.size),
        "database_rows": int(dataset.database_idx.size),
        "fixed_final_epoch": config.epochs,
        "test_labels_used_for_selection": False,
        "i2t_map": heldout["i2t"]["map"],
        "t2i_map": heldout["t2i"]["map"],
        "structural_gate_passed": structural_gate["passed"],
        "heldout_gate_passed": heldout_gate["passed"],
        "npz": str(args.output.resolve()),
        "npz_bytes": args.output.stat().st_size,
        "npz_sha256": _file_sha256(args.output),
        "source_sha256": _source_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists() and not args.overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(args.manifest))
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _synthetic_data(seed: int, rows: int = 32, dim: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((rows, 6)).astype(np.float32)
    image = latent @ rng.standard_normal((6, dim)).astype(np.float32)
    text = latent @ rng.standard_normal((6, dim)).astype(np.float32)
    image += 0.05 * rng.standard_normal(image.shape).astype(np.float32)
    text += 0.05 * rng.standard_normal(text.shape).astype(np.float32)
    return image, text


def run_smoke(args: argparse.Namespace) -> None:
    image, text = _synthetic_data(args.seed)
    config = TrainConfig(
        bits=16,
        epochs=args.epochs,
        batch_size=8,
        graph_k=8,
        image_hidden_dim=32,
        seed=args.seed,
        device=args.device,
    )
    result = train_cirh_f(image, text, config, verbose=not args.quiet)
    image_codes = encode_all(result.image_model, image, 16, result.device)
    text_codes = encode_all(result.text_model, text, 16, result.device)
    gate = structural_quality_gate(result, image_codes, text_codes)
    report = {
        "status": "PASS" if gate["passed"] else "FAIL",
        "shape": list(image_codes.shape),
        "domain": sorted(np.unique(np.concatenate((image_codes.ravel(), text_codes.ravel()))).tolist()),
        "structural_quality_gate": gate,
        "runtime_seconds": result.runtime_seconds,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise RuntimeError("Synthetic CIRH-F smoke failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/export CIRH-F controlled fixed-feature codes"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fixed-split CIRH-F training and export")
    train.add_argument("--dataset", type=canonical_dataset_name, required=True)
    train.add_argument("--data-root", type=Path, default=Path("Data/ProcessData"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--bits", type=int, choices=SUPPORTED_BITS, default=64)
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--export-batch-size", type=int, default=4096)
    train.add_argument("--lr-joint", type=float, default=1e-3)
    train.add_argument("--lr-image", type=float, default=1e-4)
    train.add_argument("--lr-text", type=float, default=1e-4)
    train.add_argument("--lambda-identity", type=float, default=10.0)
    train.add_argument("--lambda-correlation", type=float, default=1.0)
    train.add_argument("--beta", type=float, default=0.01)
    train.add_argument("--graph-k", type=int, default=3000)
    train.add_argument("--graph-image-weight", type=float, default=0.6)
    train.add_argument("--graph-direct-weight", type=float, default=0.6)
    train.add_argument("--image-hidden-dim", type=int, default=4096)
    train.add_argument("--seed", type=int, default=20260805)
    train.add_argument("--device", default="auto")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--allow-failed-quality-gate", action="store_true")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=run_train)
    smoke = sub.add_parser("smoke", help="small end-to-end gradient/code smoke")
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=20260805)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--quiet", action="store_true")
    smoke.set_defaults(func=run_smoke)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
