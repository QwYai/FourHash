#!/usr/bin/env python3
"""RANEH-F: the 2025 KBS FastKAN hash learner on sealed CLIP-512 pairs.

Paper:
    Yunfei Chen, Renwei Xia, Zhan Yang, and Jun Long,
    "Radial Adaptive Node Embedding Hashing for cross-modal retrieval,"
    Knowledge-Based Systems 319 (2025), 113522.
    DOI: 10.1016/j.knosys.2025.113522

Author source audited here:
    https://github.com/YunfeiChenMY/RANEH
    commit cfab17f5d14a8e45c3405d52273efcfdaaed75db

The author repository contains the FastKAN/model/entry scripts but omits the
imported loader, optimizer, and metric modules.  A public complete mirror was
therefore used only to recover those missing files:
    https://github.com/Dreamyxxw/RANEH
    commit 0b8d4c0932a240a184e2c0dedb15e2742a60a8dd

Every file common to the author repository and mirror is byte-identical.  The
hashes below make that provenance check repeatable.  This module reimplements
the recovered array-level training path without copying its absolute paths,
query-label checkpoint selection, or top-50 evaluator.

Retained method mechanics:

* CLIP image/text vectors are the direct network inputs;
* first- and second-order modality similarities form the semantic affinity;
* FastKAN radial-basis/spline layers build the joint reconstruction network;
* the official within-batch neighbor mixture is retained;
* independent image/text hash functions learn the joint binary target; and
* all dataset-specific learning rates, loss weights, affinity cutoffs, batch
  sizes, and the 60-epoch schedule come from the three author entry scripts.

Controlled-protocol changes:

* sealed OpenAI CLIP ViT-B/32 vectors and the common split replace the released
  pickles; the original 5,000-pair CLIP training quota is retained by a stable,
  label-free row-ID subset when the common indT pool is larger;
* shuffled affinity blocks are indexed by their real item IDs;
* sign(0) maps to +1, so exported codes are binary rather than ternary;
* the final epoch is frozen without query/database labels; and
* the shared full-gallery expected-tie evaluator runs only after code freeze.

The reporting suffix ``-F`` is mandatory: this is a controlled shared-input
adaptation, not a claim of reproducing the paper's published absolute scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ENCODER_NAME = "RANEH-F"
PAPER_URL = "https://doi.org/10.1016/j.knosys.2025.113522"
OFFICIAL_REPOSITORY = "https://github.com/YunfeiChenMY/RANEH"
OFFICIAL_COMMIT = "cfab17f5d14a8e45c3405d52273efcfdaaed75db"
MIRROR_REPOSITORY = "https://github.com/Dreamyxxw/RANEH"
MIRROR_COMMIT = "0b8d4c0932a240a184e2c0dedb15e2742a60a8dd"
SUPPORTED_BITS = (16, 32, 64)

OFFICIAL_SOURCE_SHA256: Mapping[str, str] = {
    "fastkan.py": "13430496fef56f578d540dbe0d18498e546f8b5b77f24a012f2ae03e82fc88cd",
    "models.py": "9e89ad4d3627099645d7c04a08f8eb6f4b52ac4e9d48d2319400536486e6b609",
    "main_mir.py": "e00e2076454e655d7e0e719a3000fe5aa8311b3935be6686a8fe544dba887b7e",
    "main_nus.py": "9f16dd0d2340cc4a791111036056f0b2658230b11f6a353a2f617b30ce9ac81c",
    "main_coco.py": "21cbd1e213c3ad608b9f3151353087a45fc1c0289d39356974f7d8f224471225",
}
RECOVERED_MIRROR_SOURCE_SHA256: Mapping[str, str] = {
    "load_data.py": "3eb7fc4be13aa6187566009b788756ac0a70ace63c36d498c94b589c7aedbcf6",
    "my_opt.py": "38e95438d14af4ffce1e2fd2a1097f3cb629629d3679061e1ab3eaa507b326f7",
    "utils.py": "6d9eae0dc75cd0970bba84158b56b201770df25f65e2950a945944ede194a285",
}


@dataclass(frozen=True)
class RANEHConfig:
    dataset: str
    bits: int
    seed: int
    device: str = "auto"
    epochs: int = 60
    batch_size: int = 512
    lr_image: float = 4.0e-4
    lr_text: float = 1.75e-3
    lr_joint: float = 1.0e-3
    lambda_reconstruction: float = 100.0
    lambda_similarity: float = 10.0
    lambda_hash_similarity: float = 1.0
    affinity_prune_k: int = 4900
    affinity_a1: float = 0.6
    affinity_a2: float = 0.5
    train_limit: int = 5000
    kan_hidden_dim: int = 512
    kan_num_grids: int = 8
    image_hidden_dim: int = 4096
    text_hidden_dim: int = 512

    def validate(self, n_train: int | None = None) -> None:
        if self.dataset not in {"mirflickr", "nuswide", "mscoco"}:
            raise ValueError("dataset must be mirflickr, nuswide, or mscoco")
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 4:
            raise ValueError("batch_size must be at least four")
        if self.train_limit < 4:
            raise ValueError("train_limit must be at least four")
        if self.affinity_prune_k < 1:
            raise ValueError("affinity_prune_k must be positive")
        for name, value in (
            ("lr_image", self.lr_image),
            ("lr_text", self.lr_text),
            ("lr_joint", self.lr_joint),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("lambda_reconstruction", self.lambda_reconstruction),
            ("lambda_similarity", self.lambda_similarity),
            ("lambda_hash_similarity", self.lambda_hash_similarity),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("affinity_a1", self.affinity_a1),
            ("affinity_a2", self.affinity_a2),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.kan_hidden_dim < 2 or self.kan_num_grids < 2:
            raise ValueError("FastKAN dimensions must be at least two")
        if self.image_hidden_dim < 1 or self.text_hidden_dim < 1:
            raise ValueError("hash-network hidden dimensions must be positive")
        if n_train is not None:
            if n_train < 4:
                raise ValueError("RANEH-F requires at least four training pairs")
            if self.affinity_prune_k >= n_train:
                raise ValueError(
                    "affinity_prune_k must be smaller than the effective training set"
                )


DATASET_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "mirflickr": {
        "batch_size": 1024,
        "lr_image": 0.00069,
        "lr_text": 0.00125,
        "lr_joint": 0.00119,
        "lambda_reconstruction": 100.0,
        "lambda_similarity": 1.0,
        "lambda_hash_similarity": 1.0,
        "affinity_prune_k": 4700,
        "affinity_a1": 0.4,
        "affinity_a2": 0.7,
    },
    "nuswide": {
        "batch_size": 512,
        "lr_image": 0.0004,
        "lr_text": 0.00175,
        "lr_joint": 0.001,
        "lambda_reconstruction": 100.0,
        "lambda_similarity": 10.0,
        "lambda_hash_similarity": 1.0,
        "affinity_prune_k": 4900,
        "affinity_a1": 0.6,
        "affinity_a2": 0.5,
    },
    "mscoco": {
        "batch_size": 512,
        "lr_image": 0.00137,
        "lr_text": 0.00132,
        "lr_joint": 0.00175,
        "lambda_reconstruction": 100.0,
        "lambda_similarity": 10.0,
        "lambda_hash_similarity": 10.0,
        "affinity_prune_k": 4900,
        "affinity_a1": 0.3,
        "affinity_a2": 0.6,
    },
}


def config_for_dataset(
    dataset: str,
    *,
    bits: int,
    seed: int,
    device: str,
    overrides: Mapping[str, Any] | None = None,
) -> RANEHConfig:
    """Build the exact author-script profile for one controlled dataset."""

    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"unsupported RANEH-F dataset {dataset!r}")
    values: dict[str, Any] = dict(DATASET_DEFAULTS[dataset])
    values.update(dict(overrides or {}))
    config = RANEHConfig(
        dataset=dataset,
        bits=bits,
        seed=seed,
        device=device,
        **values,
    )
    config.validate()
    return config


@dataclass
class TrainingResult:
    image_model: "ImageHashNet"
    text_model: "TextHashNet"
    joint_model: "RANEHJointNet"
    history: list[dict[str, float]]
    device: str
    runtime_seconds: float
    image_parameter_delta: float
    text_parameter_delta: float
    joint_parameter_delta: float
    affinity_diagnostics: dict[str, Any]


def stable_sign(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1):
        self.init_scale = float(init_scale)
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float | None = None,
    ) -> None:
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid, persistent=True)
        self.denominator = float(
            denominator or (grid_max - grid_min) / (num_grids - 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((values[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    """Device-agnostic equivalent of the author's FastKAN layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 2 and use_layernorm:
            raise ValueError("layer-normalized FastKAN input must have dimension >=2")
        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm else None
        self.rbf = RadialBasisFunction(num_grids=num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim)
        self.use_base_update = bool(use_base_update)
        self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.layernorm(values) if self.layernorm is not None else values
        basis = self.rbf(normalized)
        output = self.spline_linear(basis.flatten(start_dim=-2))
        if self.base_linear is not None:
            output = output + self.base_linear(F.silu(values))
        return output


class RANEHJointNet(nn.Module):
    def __init__(
        self,
        bits: int,
        image_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        num_grids: int = 8,
    ) -> None:
        super().__init__()
        layer = lambda left, right: FastKANLayer(
            left, right, num_grids=num_grids
        )
        self.image_encoder = layer(image_dim, hidden_dim)
        self.text_encoder = layer(text_dim, hidden_dim)
        self.image_graph = layer(hidden_dim, hidden_dim)
        self.text_graph = layer(hidden_dim, hidden_dim)
        self.joint_graph = layer(hidden_dim, hidden_dim)
        self.joint_hash = layer(hidden_dim, bits)
        self.image_hash = layer(2 * hidden_dim, bits)
        self.text_hash = layer(2 * hidden_dim, bits)
        self.combined_hash = layer(3 * bits, bits)
        self.image_hash_bn = nn.BatchNorm1d(bits)
        self.text_hash_bn = nn.BatchNorm1d(bits)
        self.joint_hash_bn = nn.BatchNorm1d(bits)
        self.combined_hash_bn = nn.BatchNorm1d(bits)
        self.image_decoder = nn.Sequential(
            layer(bits, hidden_dim),
            nn.Linear(hidden_dim, image_dim),
            nn.BatchNorm1d(image_dim),
        )
        self.text_decoder = nn.Sequential(
            layer(bits, hidden_dim),
            nn.Linear(hidden_dim, text_dim),
            nn.BatchNorm1d(text_dim),
        )

    def forward(
        self, image: torch.Tensor, text: torch.Tensor, affinity: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        batch = image.shape[0]
        vi = F.normalize(self.image_encoder(image), dim=1, eps=1e-8)
        vt = F.normalize(self.text_encoder(text), dim=1, eps=1e-8)
        vgi = affinity @ self.image_graph(vi)
        vgt = affinity @ self.text_graph(vt)
        identity = torch.eye(batch, dtype=affinity.dtype, device=affinity.device)
        cma = torch.cat(
            (
                torch.cat((affinity, identity), dim=1),
                torch.cat((identity, affinity), dim=1),
            ),
            dim=0,
        )
        joint_both = cma @ self.joint_graph(torch.cat((vi, vt), dim=0))
        joint = F.relu(joint_both[:batch] + joint_both[batch:])
        hj = self.joint_hash_bn(self.joint_hash(joint))
        hi = self.image_hash_bn(self.image_hash(torch.cat((vgi, joint), dim=1)))
        ht = self.text_hash_bn(self.text_hash(torch.cat((joint, vgt), dim=1)))
        h = torch.tanh(
            self.combined_hash_bn(
                self.combined_hash(torch.cat((hi, hj, ht), dim=1))
            )
        )
        binary = stable_sign(h)
        decoded_image = self.image_decoder(h + torch.tanh(hi))
        decoded_text = self.text_decoder(h + torch.tanh(ht))
        return hi, ht, h, binary, decoded_image, decoded_text


class ImageHashNet(nn.Module):
    def __init__(self, bits: int, input_dim: int = 512, hidden_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.hash_head = nn.Linear(hidden_dim, bits)
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=1.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hash_head(self.dropout(F.relu(self.fc1(values)))))


class TextHashNet(nn.Module):
    def __init__(self, bits: int, input_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.hash_head = nn.Linear(hidden_dim, bits)
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=1.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hash_head(self.dropout(F.relu(self.fc1(values)))))


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [value.detach().cpu().clone() for value in model.parameters()]


def _parameter_delta(before: Sequence[torch.Tensor], model: nn.Module) -> float:
    squared = 0.0
    for original, current in zip(before, model.parameters()):
        difference = current.detach().cpu() - original
        squared += float(torch.sum(difference * difference).item())
    return math.sqrt(squared)


def _batch_indices(
    n_train: int, batch_size: int, generator: torch.Generator
) -> list[torch.Tensor]:
    chunks = list(torch.split(torch.randperm(n_train, generator=generator), batch_size))
    if len(chunks) > 1 and chunks[-1].numel() < 4:
        chunks[-2] = torch.cat((chunks[-2], chunks[-1]))
        chunks.pop()
    return chunks


def select_square(matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return matrix.index_select(0, indices).index_select(1, indices)


def select_official_neighbors(
    affinity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if affinity.shape[0] < 4:
        raise ValueError("RANEH-F neighbor mixing needs at least four samples")
    ordered = torch.sort(affinity, dim=1).indices
    # Author code takes the first two columns of ``[:, -4:-1]``.
    return ordered[:, -4], ordered[:, -3]


@torch.no_grad()
def build_semantic_affinity(
    image: torch.Tensor,
    text: torch.Tensor,
    *,
    prune_k: int,
    a1: float,
    a2: float,
) -> torch.Tensor:
    """Construct the author's direct plus second-order semantic affinity."""

    if image.shape != text.shape or image.ndim != 2:
        raise ValueError("RANEH-F expects paired image/text feature matrices")
    n_train = image.shape[0]
    if not 1 <= prune_k < n_train:
        raise ValueError("prune_k must be in [1,n_train)")
    image_n = F.normalize(image, dim=1, eps=1e-8)
    text_n = F.normalize(text, dim=1, eps=1e-8)
    image_similarity = image_n @ image_n.T
    text_similarity = text_n @ text_n.T
    direct = 0.5 * (image_similarity + text_similarity)

    image_rows = F.normalize(image_similarity, dim=1, eps=1e-8)
    text_rows = F.normalize(text_similarity, dim=1, eps=1e-8)
    second = a2 * (
        a1 * (image_rows @ image_rows.T)
        + (1.0 - a1) * (text_rows @ text_rows.T)
    )
    second.add_((1.0 - a2) * (image_rows @ text_rows.T))
    affinity = 0.5 * direct + 0.5 * second
    # ``topk`` avoids materializing the author's full N x N int64 argsort.
    # Only equal-valued boundary identities can differ; the modernization is
    # recorded in the checkpoint summary.
    smallest = torch.topk(
        affinity, k=prune_k, dim=1, largest=False, sorted=False
    ).indices
    affinity.scatter_(1, smallest, 0.0)
    affinity = 2.0 * torch.sigmoid(affinity) - 1.0
    affinity.add_(torch.eye(n_train, dtype=affinity.dtype, device=affinity.device))
    return 0.5 * (affinity + affinity.T)


def _joint_objective(
    outputs: tuple[torch.Tensor, ...],
    target_image: torch.Tensor,
    target_text: torch.Tensor,
    affinity: torch.Tensor,
    config: RANEHConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hi, ht, h, binary, decoded_image, decoded_text = outputs
    reconstruction = config.lambda_reconstruction * (
        F.mse_loss(decoded_image, target_image)
        + F.mse_loss(decoded_text, target_text)
    )
    hi_n = F.normalize(hi, dim=1, eps=1e-8)
    ht_n = F.normalize(ht, dim=1, eps=1e-8)
    h_n = F.normalize(h, dim=1, eps=1e-8)
    similarity = config.lambda_similarity * (
        F.mse_loss(hi_n @ hi_n.T, affinity)
        + F.mse_loss(ht_n @ ht_n.T, affinity)
        + F.mse_loss(h_n @ h_n.T, affinity)
    )
    quantization = F.mse_loss(h, binary.detach())
    return reconstruction + similarity + quantization, {
        "reconstruction": reconstruction,
        "similarity": similarity,
        "quantization": quantization,
    }


def _hash_objective(
    image_hash: torch.Tensor,
    text_hash: torch.Tensor,
    binary: torch.Tensor,
    affinity: torch.Tensor,
    config: RANEHConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    quantized = (
        F.mse_loss(image_hash, binary)
        + F.mse_loss(text_hash, binary)
        + F.mse_loss(image_hash, text_hash)
    )
    image_n = F.normalize(image_hash, dim=1, eps=1e-8)
    text_n = F.normalize(text_hash, dim=1, eps=1e-8)
    similarity = config.lambda_hash_similarity * (
        F.mse_loss(image_n @ text_n.T, affinity)
        + F.mse_loss(image_n @ image_n.T, affinity)
        + F.mse_loss(text_n @ text_n.T, affinity)
    )
    return quantized + similarity, {"quantized": quantized, "similarity": similarity}


def train_raneh_f(
    train_image: np.ndarray,
    train_text: np.ndarray,
    config: RANEHConfig,
    *,
    verbose: bool = True,
) -> TrainingResult:
    """Train RANEH-F from paired features only; no label argument exists."""

    image_np = np.ascontiguousarray(train_image, dtype=np.float32)
    text_np = np.ascontiguousarray(train_text, dtype=np.float32)
    if image_np.shape != text_np.shape or image_np.ndim != 2:
        raise ValueError("expected paired [N,D] train features")
    if image_np.shape[1] != 512:
        raise ValueError("RANEH-F requires 512-D CLIP features")
    if not np.all(np.isfinite(image_np)) or not np.all(np.isfinite(text_np)):
        raise ValueError("train features contain non-finite values")
    n_train = int(image_np.shape[0])
    config.validate(n_train)
    device = _resolve_device(config.device)
    image = torch.from_numpy(image_np).to(device)
    text = torch.from_numpy(text_np).to(device)

    started = time.perf_counter()
    affinity = build_semantic_affinity(
        image,
        text,
        prune_k=config.affinity_prune_k,
        a1=config.affinity_a1,
        a2=config.affinity_a2,
    )
    if not torch.isfinite(affinity).all():
        raise RuntimeError("RANEH-F affinity contains non-finite values")

    joint = RANEHJointNet(
        config.bits,
        512,
        512,
        hidden_dim=config.kan_hidden_dim,
        num_grids=config.kan_num_grids,
    ).to(device)
    image_model = ImageHashNet(
        config.bits, 512, hidden_dim=config.image_hidden_dim
    ).to(device)
    text_model = TextHashNet(
        config.bits, 512, hidden_dim=config.text_hidden_dim
    ).to(device)
    joint_before = _parameter_snapshot(joint)
    image_before = _parameter_snapshot(image_model)
    text_before = _parameter_snapshot(text_model)
    joint_optimizer = torch.optim.Adam(joint.parameters(), lr=config.lr_joint)
    image_optimizer = torch.optim.Adam(image_model.parameters(), lr=config.lr_image)
    text_optimizer = torch.optim.Adam(text_model.parameters(), lr=config.lr_text)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1709)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        chunks = _batch_indices(n_train, config.batch_size, generator)
        epoch_binary = torch.empty((n_train, config.bits), device=device)
        joint.train()
        totals = {
            "joint_total": 0.0,
            "joint_reconstruction": 0.0,
            "joint_similarity": 0.0,
            "joint_quantization": 0.0,
        }
        seen = 0
        for ids_cpu in chunks:
            ids = ids_cpu.to(device)
            batch_image = image.index_select(0, ids)
            batch_text = text.index_select(0, ids)
            batch_affinity = select_square(affinity, ids)
            neighbor_one, neighbor_two = select_official_neighbors(batch_affinity)
            mixed_image = (
                0.7 * batch_image
                + 0.2 * batch_image.index_select(0, neighbor_two)
                + 0.1 * batch_image.index_select(0, neighbor_one)
            )
            mixed_text = (
                0.7 * batch_text
                + 0.2 * batch_text.index_select(0, neighbor_two)
                + 0.1 * batch_text.index_select(0, neighbor_one)
            )
            outputs = joint(mixed_image, mixed_text, batch_affinity)
            loss, pieces = _joint_objective(
                outputs, mixed_image, mixed_text, batch_affinity, config
            )
            joint_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            joint_optimizer.step()
            epoch_binary.index_copy_(0, ids, outputs[3].detach())
            weight = int(ids.numel())
            seen += weight
            totals["joint_total"] += float(loss.detach().item()) * weight
            for key in ("reconstruction", "similarity", "quantization"):
                totals[f"joint_{key}"] += float(pieces[key].detach().item()) * weight

        image_model.train()
        text_model.train()
        hash_total = hash_quantized = hash_similarity = 0.0
        hash_seen = 0
        for ids_cpu in chunks:
            ids = ids_cpu.to(device)
            batch_image = F.normalize(image.index_select(0, ids), dim=1, eps=1e-8)
            batch_text = F.normalize(text.index_select(0, ids), dim=1, eps=1e-8)
            batch_binary = epoch_binary.index_select(0, ids)
            batch_affinity = select_square(affinity, ids)
            image_hash = image_model(batch_image)
            text_hash = text_model(batch_text)
            loss, pieces = _hash_objective(
                image_hash, text_hash, batch_binary, batch_affinity, config
            )
            image_optimizer.zero_grad(set_to_none=True)
            text_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            image_optimizer.step()
            text_optimizer.step()
            weight = int(ids.numel())
            hash_seen += weight
            hash_total += float(loss.detach().item()) * weight
            hash_quantized += float(pieces["quantized"].detach().item()) * weight
            hash_similarity += float(pieces["similarity"].detach().item()) * weight

        record = {
            "epoch": float(epoch + 1),
            **{key: value / seen for key, value in totals.items()},
            "hash_total": hash_total / hash_seen,
            "hash_quantized": hash_quantized / hash_seen,
            "hash_similarity": hash_similarity / hash_seen,
            "binary_plus_fraction": float((epoch_binary > 0).float().mean().item()),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise RuntimeError(f"non-finite RANEH-F history at epoch {epoch + 1}")
        history.append(record)
        if verbose:
            print(json.dumps(record, sort_keys=True), flush=True)

    diagnostics = {
        "shape": [n_train, n_train],
        "minimum": float(affinity.min().item()),
        "maximum": float(affinity.max().item()),
        "mean": float(affinity.mean().item()),
        "symmetry_max_abs_error": float(
            torch.max(torch.abs(affinity - affinity.T)).item()
        ),
        "one_float32_square_bytes": int(4 * n_train * n_train),
        "construction_inputs": "effective indT image/text features only",
        "author_formula_retained": True,
        "argsort_modernization": "topk smallest; equal-value boundary identity may differ",
    }
    runtime = time.perf_counter() - started
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
        affinity_diagnostics=diagnostics,
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
    model.to(resolved).eval()
    outputs: list[np.ndarray] = []
    for start in range(0, values.shape[0], batch_size):
        batch = torch.from_numpy(values[start : start + batch_size]).to(resolved)
        batch = F.normalize(batch, dim=1, eps=1e-8)
        outputs.append(stable_sign(model(batch)).to(torch.int8).cpu().numpy())
    result = np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.int8)
    if not np.all(np.isin(result, (-1, 1))):
        raise RuntimeError("RANEH-F exported non-binary codes")
    return result


__all__ = [
    "DATASET_DEFAULTS",
    "ENCODER_NAME",
    "ImageHashNet",
    "OFFICIAL_COMMIT",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_SOURCE_SHA256",
    "PAPER_URL",
    "RANEHConfig",
    "RANEHJointNet",
    "RECOVERED_MIRROR_SOURCE_SHA256",
    "SUPPORTED_BITS",
    "TextHashNet",
    "TrainingResult",
    "build_semantic_affinity",
    "config_for_dataset",
    "encode_all",
    "select_official_neighbors",
    "stable_sign",
    "train_raneh_f",
]
