#!/usr/bin/env python3
"""UCCH-F: a leakage-controlled fixed-feature adaptation of UCCH.

This module retains the official UCCH feature-mode architecture and its two
training terms:

* contrastive hashing with a momentum memory bank (``L_c``); and
* cross-modal ranking learning (``L_r``).

It is a controlled adaptation, not an official reproduction.  The project's
fixed 512-D features and fixed ``indT/indQ/indD`` split replace UCCH's bundled
features and split.  Only paired image/text rows in ``indT`` are available to
the optimizer.  Labels are deliberately absent from :func:`train_ucch_f`.
The final epoch is frozen by rule; held-out labels are opened exactly once
after the final checkpoint has been serialized, solely for an acceptance
diagnostic and never for checkpoint or hyperparameter selection.

Official source audited for this port:
    https://github.com/penghu-cs/UCCH
    commit 0c20e62b99875cd2ec9d7a496eae80b3ab8ba61b

Paper:
    Peng Hu et al., "Unsupervised Contrastive Cross-modal Hashing," TPAMI 2023.
    DOI: 10.1109/TPAMI.2022.3177356
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_f

# Reuse the already-audited common fixed-feature loader/export schema.  The
# fallback makes both ``python encoders/ucch_feature.py`` and test imports work.
try:
    from dcmh_feature import (
        PreparedDataset,
        _array_sha256,
        _code_statistics,
        _retrieval_metrics,
        _split_sha256,
        export_npz,
        heldout_retrieval_quality_gate,
        load_prepared_dataset,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .dcmh_feature import (  # type: ignore
        PreparedDataset,
        _array_sha256,
        _code_statistics,
        _retrieval_metrics,
        _split_sha256,
        export_npz,
        heldout_retrieval_quality_gate,
        load_prepared_dataset,
    )


FORMAT_VERSION = 1
ENCODER_NAME = "UCCH-F"
REPORTING_NAME = "UCCH-F"
SUPPORTED_BITS = (16, 32, 64, 128)
OFFICIAL_REPOSITORY = "https://github.com/penghu-cs/UCCH"
OFFICIAL_COMMIT = "0c20e62b99875cd2ec9d7a496eae80b3ab8ba61b"
PAPER_URL = "https://doi.org/10.1109/TPAMI.2022.3177356"


@dataclass(frozen=True)
class UCCHConfig:
    """Predeclared UCCH-F training configuration.

    Defaults follow the official README's MIRFLICKR feature-mode command and
    the corresponding parser defaults.  In particular, image/text branch
    depths are 3/2, hidden width is 8192, ``alpha=0.7``, ``margin=0.2``,
    ``shift=0.1``, and training lasts 20 fixed epochs.
    """

    bits: int = 64
    epochs: int = 20
    batch_size: int = 256
    image_layers: int = 3
    text_layers: int = 2
    hidden_width: int = 8192
    lr: float = 1e-4
    weight_decay: float = 1e-6
    alpha: float = 0.7
    margin: float = 0.2
    shift: float = 0.1
    negatives: int = 4096
    temperature: float = 0.9
    memory_momentum: float = 0.4
    memory_warmup_epochs: int = 1
    seed: int = 20260805
    device: str = "auto"

    def validate(self, n_train: Optional[int] = None) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise ValueError("bits must be one of {}".format(SUPPORTED_BITS))
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if n_train is not None and self.batch_size > n_train:
            raise ValueError("batch_size exceeds the number of training rows")
        if self.image_layers < 1 or self.text_layers < 1:
            raise ValueError("branch depths must be positive")
        if self.hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        for name, value in (
            ("lr", self.lr),
            ("temperature", self.temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(name))
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        if not math.isfinite(self.shift) or self.shift < 0:
            raise ValueError("shift must be finite and non-negative")
        if self.negatives < 1:
            raise ValueError("negatives must be positive")
        if not 0.0 <= self.memory_momentum < 1.0:
            raise ValueError("memory_momentum must be in [0,1)")
        if self.memory_warmup_epochs < 0:
            raise ValueError("memory_warmup_epochs must be non-negative")


@dataclass
class UCCHTrainingResult:
    image_model: "UCCHFeatureNet"
    text_model: "UCCHFeatureNet"
    memory_bank: "MomentumHashMemory"
    history: List[Dict[str, float]]
    device: str
    image_parameter_delta: float
    text_parameter_delta: float
    memory_delta: float


class UCCHFeatureNet(nn.Module):
    """Official feature-mode MLP: Linear/ReLU stack, tanh, L2 normalize."""

    def __init__(
        self,
        input_dim: int,
        bits: int,
        layers: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        if input_dim < 1 or bits < 1 or layers < 1 or hidden_width < 1:
            raise ValueError("network dimensions must be positive")
        self.input_dim = int(input_dim)
        self.bits = int(bits)
        self.layers = int(layers)
        self.hidden_width = int(hidden_width)

        if layers == 1:
            modules: List[nn.Module] = [nn.Linear(input_dim, bits)]
        else:
            modules = [nn.Linear(input_dim, hidden_width), nn.ReLU(inplace=True)]
            for _ in range(layers - 2):
                modules.extend(
                    [nn.Linear(hidden_width, hidden_width), nn.ReLU(inplace=True)]
                )
            modules.append(nn.Linear(hidden_width, bits))
        self.fc = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "Expected [batch, {}], got {}".format(
                    self.input_dim, tuple(features.shape)
                )
            )
        output = torch.tanh(self.fc(features))
        return torch_f.normalize(output, p=2, dim=1, eps=1e-12)


class MomentumHashMemory(nn.Module):
    """UCCH's discrete memory bank with distribution-equivalent sampling.

    The official ``AliasMethod`` samples from an all-ones unigram vector.  A
    seeded uniform integer sampler therefore has exactly the same categorical
    distribution while avoiding the obsolete CUDA-only sampler implementation.
    """

    def __init__(
        self,
        bits: int,
        n_items: int,
        negatives: int,
        temperature: float,
        momentum: float,
    ) -> None:
        super().__init__()
        if bits < 1 or n_items < 2 or negatives < 1:
            raise ValueError("invalid memory dimensions")
        self.bits = int(bits)
        self.n_items = int(n_items)
        self.negatives = int(negatives)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        stdv = 1.0 / math.sqrt(bits / 3.0)
        random_values = torch.randn(n_items, bits).mul_(2.0 * stdv).add_(-stdv)
        memory = torch_f.normalize(torch.sign(random_values), p=2, dim=1)
        self.register_buffer("memory", memory)

    def _warmup_weights(
        self, image_output: torch.Tensor, text_output: torch.Tensor
    ) -> torch.Tensor:
        batch_size = int(image_output.shape[0])
        shared = 0.5 * (image_output + text_output)
        rows = torch.arange(batch_size, device=shared.device)
        candidates = []
        for row in range(batch_size):
            other = rows[rows != row]
            candidates.append(torch.cat((rows[row : row + 1], other)))
        candidate_idx = torch.stack(candidates, dim=0)
        return shared[candidate_idx]

    def forward(
        self,
        image_output: torch.Tensor,
        text_output: torch.Tensor,
        item_index: torch.Tensor,
        use_memory: bool,
        sampling_generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if image_output.shape != text_output.shape:
            raise ValueError("image/text output shapes differ")
        if image_output.ndim != 2 or image_output.shape[1] != self.bits:
            raise ValueError("unexpected hash-output shape")
        batch_size = int(image_output.shape[0])
        if item_index.shape != (batch_size,):
            raise ValueError("item_index must have shape [batch]")
        if item_index.min() < 0 or item_index.max() >= self.n_items:
            raise ValueError("item_index is outside the memory bank")

        if use_memory:
            # Sample on CPU so a given seed defines one stream on CPU and CUDA.
            sampled = torch.randint(
                self.n_items,
                (batch_size, self.negatives + 1),
                generator=sampling_generator,
                device="cpu",
            ).to(item_index.device)
            sampled[:, 0] = item_index
            weights = self.memory.index_select(0, sampled.reshape(-1)).detach()
            weights = weights.reshape(batch_size, self.negatives + 1, self.bits)
            update_momentum = self.momentum
        else:
            weights = self._warmup_weights(image_output, text_output)
            update_momentum = 0.0

        # This is the central discrete operation in the official implementation.
        weights = torch.sign(weights)
        scale = self.temperature * math.sqrt(self.bits)
        image_logits = torch.bmm(
            weights, image_output.reshape(batch_size, self.bits, 1)
        ).squeeze(2) / scale
        text_logits = torch.bmm(
            weights, text_output.reshape(batch_size, self.bits, 1)
        ).squeeze(2) / scale

        with torch.no_grad():
            shared = torch_f.normalize(
                0.5 * (image_output + text_output), p=2, dim=1, eps=1e-12
            )
            previous = self.memory.index_select(0, item_index)
            updated = previous * update_momentum + shared * (1.0 - update_momentum)
            updated = torch_f.normalize(updated, p=2, dim=1, eps=1e-12)
            self.memory.index_copy_(0, item_index, updated)

        return image_logits, text_logits, int(weights.shape[1])


class CrossModalRankingLoss(nn.Module):
    """Numerically stable form of UCCH's official ``ContrastiveLoss``."""

    def __init__(self, margin: float, shift: float) -> None:
        super().__init__()
        self.margin = float(margin)
        self.shift = float(shift)

    def forward(
        self, image_output: torch.Tensor, text_output: torch.Tensor
    ) -> torch.Tensor:
        if image_output.shape != text_output.shape:
            raise ValueError("image/text output shapes differ")
        scores = image_output @ text_output.t()
        diagonal = torch.diagonal(scores)
        row_diagonal = diagonal.reshape(-1, 1)
        col_diagonal = diagonal.reshape(1, -1)
        row_mask = (scores >= row_diagonal - self.margin).detach()
        col_mask = (scores >= col_diagonal - self.margin).detach()
        row_scores = torch.where(row_mask, scores, scores - self.shift)
        col_scores = torch.where(col_mask, scores, scores - self.shift)
        row_loss = -diagonal + torch.logsumexp(row_scores, dim=1) + self.margin
        col_loss = -diagonal + torch.logsumexp(col_scores, dim=0) + self.margin
        return row_loss.mean() + col_loss.mean()


def nce_softmax_loss(logits: torch.Tensor) -> torch.Tensor:
    """Stable equivalent of ``-log(softmax(logits)[:, 0]).mean()``."""

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must be [batch, at least two candidates]")
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return torch_f.cross_entropy(logits, target)


def stable_sign(values: torch.Tensor) -> torch.Tensor:
    """Export convention: the measure-zero exact-zero case maps to +1."""

    return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


def _resolve_device(requested: str) -> torch.device:
    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _parameter_snapshot(model: nn.Module) -> List[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def _parameter_delta(before: Sequence[torch.Tensor], model: nn.Module) -> float:
    squared = 0.0
    for original, current in zip(before, model.parameters()):
        difference = current.detach().cpu() - original
        squared += float(torch.sum(difference * difference).item())
    return math.sqrt(squared)


def _gradient_norm(parameters: Sequence[torch.Tensor]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach() ** 2).item())
    return math.sqrt(squared)


def train_ucch_f(
    train_image: np.ndarray,
    train_text: np.ndarray,
    config: UCCHConfig,
    verbose: bool = True,
) -> UCCHTrainingResult:
    """Train UCCH-F from paired features; labels cannot enter this function."""

    config.validate()
    image = np.ascontiguousarray(train_image, dtype=np.float32)
    text = np.ascontiguousarray(train_text, dtype=np.float32)
    if image.ndim != 2 or text.ndim != 2 or image.shape != text.shape:
        raise ValueError("paired image/text features must share one [N,D] shape")
    if image.shape[0] < 2 or image.shape[1] != 512:
        raise ValueError("UCCH-F expects at least two paired 512-D rows")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(text)):
        raise ValueError("features contain non-finite values")
    n_train = int(image.shape[0])
    config.validate(n_train=n_train)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)

    image_model = UCCHFeatureNet(
        512, config.bits, config.image_layers, config.hidden_width
    ).to(device)
    text_model = UCCHFeatureNet(
        512, config.bits, config.text_layers, config.hidden_width
    ).to(device)
    memory_bank = MomentumHashMemory(
        bits=config.bits,
        n_items=n_train,
        negatives=config.negatives,
        temperature=config.temperature,
        momentum=config.memory_momentum,
    ).to(device)
    image_before = _parameter_snapshot(image_model)
    text_before = _parameter_snapshot(text_model)
    memory_before = memory_bank.memory.detach().cpu().clone()

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=config.lr, weight_decay=config.weight_decay
    )
    ranking_loss = CrossModalRankingLoss(config.margin, config.shift)
    image_tensor = torch.from_numpy(image).to(device)
    text_tensor = torch.from_numpy(text).to(device)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(config.seed + 101)
    sampling_generator = torch.Generator(device="cpu").manual_seed(config.seed + 211)
    history: List[Dict[str, float]] = []

    # Official feature mode uses drop_last=True.  Keep that rule explicitly.
    batches_per_epoch = n_train // config.batch_size
    if batches_per_epoch < 1:
        raise ValueError("no complete training batch")
    for epoch in range(config.epochs):
        image_model.train()
        text_model.train()
        permutation = torch.randperm(n_train, generator=shuffle_generator)
        sums = {
            "loss": 0.0,
            "contrastive": 0.0,
            "ranking": 0.0,
            "image_gradient_norm": 0.0,
            "text_gradient_norm": 0.0,
        }
        candidate_count = 0
        for batch_number in range(batches_per_epoch):
            start = batch_number * config.batch_size
            index_cpu = permutation[start : start + config.batch_size]
            index = index_cpu.to(device)
            image_output = image_model(image_tensor.index_select(0, index))
            text_output = text_model(text_tensor.index_select(0, index))
            image_logits, text_logits, candidate_count = memory_bank(
                image_output,
                text_output,
                index,
                use_memory=epoch >= config.memory_warmup_epochs,
                sampling_generator=sampling_generator,
            )
            contrastive = nce_softmax_loss(image_logits) + nce_softmax_loss(
                text_logits
            )
            ranking = ranking_loss(image_output, text_output)
            loss = config.alpha * contrastive + (1.0 - config.alpha) * ranking
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite UCCH-F loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            image_grad = _gradient_norm(list(image_model.parameters()))
            text_grad = _gradient_norm(list(text_model.parameters()))
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()

            sums["loss"] += float(loss.detach().item())
            sums["contrastive"] += float(contrastive.detach().item())
            sums["ranking"] += float(ranking.detach().item())
            sums["image_gradient_norm"] += image_grad
            sums["text_gradient_norm"] += text_grad

        row: Dict[str, float] = {
            "epoch": float(epoch + 1),
            "lr": float(config.lr),
            "memory_mode": float(epoch >= config.memory_warmup_epochs),
            "candidate_count": float(candidate_count),
            "complete_batches": float(batches_per_epoch),
            "dropped_rows": float(n_train - batches_per_epoch * config.batch_size),
        }
        for key, value in sums.items():
            row[key] = float(value / batches_per_epoch)
        history.append(row)
        if verbose:
            print(json.dumps(row, sort_keys=True))

    memory_delta = float(torch.linalg.vector_norm(memory_bank.memory.cpu() - memory_before))
    return UCCHTrainingResult(
        image_model=image_model,
        text_model=text_model,
        memory_bank=memory_bank,
        history=history,
        device=str(device),
        image_parameter_delta=_parameter_delta(image_before, image_model),
        text_parameter_delta=_parameter_delta(text_before, text_model),
        memory_delta=memory_delta,
    )


@torch.no_grad()
def encode_all(
    model: UCCHFeatureNet,
    features: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    values = np.ascontiguousarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != model.input_dim:
        raise ValueError("unexpected export feature shape")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    target_device = torch.device(device)
    codes = np.empty((values.shape[0], model.bits), dtype=np.int8)
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        output = model(torch.from_numpy(values[start:stop]).to(target_device))
        codes[start:stop] = stable_sign(output).to(torch.int8).cpu().numpy()
    if not np.all(np.isin(np.unique(codes), [-1, 1])):
        raise AssertionError("exported codes are not bipolar")
    return codes


def pairing_diagnostics(
    image_codes: np.ndarray,
    text_codes: np.ndarray,
    result: UCCHTrainingResult,
) -> Dict[str, object]:
    """Label-free train diagnostics based only on known instance pairing."""

    if image_codes.shape != text_codes.shape or image_codes.ndim != 2:
        raise ValueError("train code shapes differ")
    n_rows, bits = image_codes.shape
    if n_rows < 3:
        raise ValueError("pairing diagnostics need at least three rows")
    image_i16 = image_codes.astype(np.int16, copy=False)
    text_i16 = text_codes.astype(np.int16, copy=False)
    matched = 0.5 * (bits - np.sum(image_i16 * text_i16, axis=1))
    mismatched_sum = 0.0
    mismatched_count = 0
    for shift in range(1, min(33, n_rows)):
        rolled = np.roll(text_i16, shift=shift, axis=0)
        distances = 0.5 * (bits - np.sum(image_i16 * rolled, axis=1))
        mismatched_sum += float(distances.sum())
        mismatched_count += int(distances.size)
    mismatched_mean = mismatched_sum / mismatched_count
    losses = np.asarray([row["loss"] for row in result.history], dtype=np.float64)
    image_gradients = np.asarray(
        [row["image_gradient_norm"] for row in result.history], dtype=np.float64
    )
    text_gradients = np.asarray(
        [row["text_gradient_norm"] for row in result.history], dtype=np.float64
    )
    return {
        "scope": "fixed indT rows; paired item IDs only; no semantic labels",
        "rows": int(n_rows),
        "bits": int(bits),
        "mean_matched_cross_modal_hamming": float(matched.mean()),
        "mean_mismatched_cross_modal_hamming": float(mismatched_mean),
        "paired_gap_mismatched_minus_matched": float(
            mismatched_mean - matched.mean()
        ),
        "paired_code_agreement": float(np.mean(image_codes == text_codes)),
        "unique_image_code_rows": int(np.unique(image_codes, axis=0).shape[0]),
        "unique_text_code_rows": int(np.unique(text_codes, axis=0).shape[0]),
        "image_code_statistics": _code_statistics(image_codes),
        "text_code_statistics": _code_statistics(text_codes),
        "history_all_finite": bool(np.all(np.isfinite(losses))),
        "minimum_image_gradient_norm": float(image_gradients.min()),
        "minimum_text_gradient_norm": float(text_gradients.min()),
        "image_parameter_delta_l2": float(result.image_parameter_delta),
        "text_parameter_delta_l2": float(result.text_parameter_delta),
        "memory_delta_l2": float(result.memory_delta),
    }


def ucch_f_quality_gate(diagnostics: Mapping[str, object]) -> Dict[str, object]:
    bits = int(diagnostics["bits"])
    minimum_gap = max(1.0, 0.03 * bits)
    checks = {
        "all_epoch_losses_finite": bool(diagnostics["history_all_finite"]),
        "both_branches_receive_gradients": bool(
            float(diagnostics["minimum_image_gradient_norm"]) > 0.0
            and float(diagnostics["minimum_text_gradient_norm"]) > 0.0
        ),
        "both_branches_changed": bool(
            float(diagnostics["image_parameter_delta_l2"]) > 0.0
            and float(diagnostics["text_parameter_delta_l2"]) > 0.0
        ),
        "memory_bank_changed": bool(float(diagnostics["memory_delta_l2"]) > 0.0),
        "paired_items_are_closer": bool(
            float(diagnostics["paired_gap_mismatched_minus_matched"])
            >= minimum_gap
        ),
        "image_codes_nonconstant": bool(
            int(diagnostics["unique_image_code_rows"]) > 32
        ),
        "text_codes_nonconstant": bool(
            int(diagnostics["unique_text_code_rows"]) > 32
        ),
        "image_bits_not_fully_collapsed": bool(
            float(diagnostics["image_code_statistics"]["mean_abs_bit_mean"])
            < 0.95
        ),
        "text_bits_not_fully_collapsed": bool(
            float(diagnostics["text_code_statistics"]["mean_abs_bit_mean"])
            < 0.95
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_paired_hamming_gap": float(minimum_gap),
            "minimum_unique_code_rows_exclusive": 32,
            "maximum_mean_abs_bit_mean_exclusive": 0.95,
        },
        "policy": (
            "label-free train gate runs before held-out labels are opened; "
            "failure forbids use as encoder evidence"
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    return _file_sha256(Path(__file__))


def save_final_checkpoint(
    path: Path,
    result: UCCHTrainingResult,
    config: UCCHConfig,
    dataset: PreparedDataset,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("checkpoint exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "encoder": ENCODER_NAME,
        "official_commit": OFFICIAL_COMMIT,
        "selection_rule": "fixed final epoch; no validation/test checkpoint selection",
        "config": asdict(config),
        "dataset_name": dataset.name,
        "train_idx": dataset.train_idx,
        "train_image_sha256": _array_sha256(dataset.image[dataset.train_idx]),
        "train_text_sha256": _array_sha256(dataset.text[dataset.train_idx]),
        "image_model_state_dict": result.image_model.state_dict(),
        "text_model_state_dict": result.text_model.state_dict(),
        "memory_bank_state_dict": result.memory_bank.state_dict(),
        "history": result.history,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def build_metadata(
    dataset: PreparedDataset,
    config: UCCHConfig,
    result: UCCHTrainingResult,
    image_codes: np.ndarray,
    text_codes: np.ndarray,
    train_diagnostics: Mapping[str, object],
    train_gate: Mapping[str, object],
    heldout_retrieval: Mapping[str, object],
    heldout_gate: Mapping[str, object],
    checkpoint: Path,
) -> Dict[str, object]:
    loader_source = Path(__file__).with_name("dcmh_feature.py")
    return {
        "format_version": FORMAT_VERSION,
        "encoder": ENCODER_NAME,
        "reporting_name": REPORTING_NAME,
        "claim_scope": (
            "controlled fixed-feature adaptation; not an official UCCH reproduction"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "title": "Unsupervised Contrastive Cross-modal Hashing",
            "venue": "IEEE TPAMI 2023",
            "doi": "10.1109/TPAMI.2022.3177356",
            "url": PAPER_URL,
            "official_repository": OFFICIAL_REPOSITORY,
            "audited_commit": OFFICIAL_COMMIT,
        },
        "retained_ucch_components": [
            "feature-mode independent image and text MLP branches",
            "tanh followed by per-item L2 normalization",
            "sign-discretized contrastive candidate memory",
            "momentum update of a shared cross-modal memory bank",
            "one in-batch warm-up epoch before memory-bank negatives",
            "bidirectional contrastive hashing loss L_c",
            "cross-modal ranking loss L_r with margin and shift",
            "alpha*L_c + (1-alpha)*L_r",
            "Adam, official weight decay, gradient clipping at norm one",
        ],
        "ucch_f_differences": [
            "project fixed 512-D CLIP-style image/text features replace UCCH bundled VGG/tag features",
            "project indT/indQ/indD common split replaces UCCH contiguous train/test partition",
            "optimizer sees indT paired features only; semantic labels are absent",
            "fixed final epoch replaces official retrieval-label best-checkpoint selection",
            "seeded torch uniform sampling replaces AliasMethod over an all-ones unigram distribution",
            "stable cross-entropy/logsumexp replace explicit softmax-log/exp-log expressions",
            "zero maps to +1 only when exporting the measure-zero exact-zero code",
            "single-device deterministic modern PyTorch replaces hard-coded CUDA/DataParallel assumptions",
        ],
        "objective": {
            "contrastive_hashing": (
                "bidirectional CE against positive candidate 0 using sign(memory) "
                "and denominator T*sqrt(bits)"
            ),
            "memory_update": (
                "normalize(momentum*old + (1-momentum)*normalize((image+text)/2))"
            ),
            "ranking": (
                "official bidirectional CRL over the paired minibatch score matrix"
            ),
            "total": "alpha*L_c + (1-alpha)*L_r",
        },
        "leakage_contract": {
            "optimizer_function_signature": "train_ucch_f(train_image, train_text, config)",
            "optimizer_inputs": "image[indT] and text[indT] only; no labels",
            "checkpoint_selection": "fixed final epoch serialized before held-out label evaluation",
            "query_database_features": "encoded only after optimizer and final checkpoint are frozen",
            "query_database_labels": (
                "opened once after freeze for post-hoc acceptance metrics only"
            ),
            "hyperparameter_selection": (
                "official MIR feature-mode defaults predeclared before held-out evaluation"
            ),
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
            "train_image_sha256": _array_sha256(dataset.image[dataset.train_idx]),
            "train_text_sha256": _array_sha256(dataset.text[dataset.train_idx]),
            "labels_sha256": _array_sha256(dataset.labels),
        },
        "training": {
            **asdict(config),
            "resolved_device": result.device,
            "selection_rule": "fixed final epoch",
            "history": result.history,
            "image_parameter_delta_l2": result.image_parameter_delta,
            "text_parameter_delta_l2": result.text_parameter_delta,
            "memory_delta_l2": result.memory_delta,
            "label_free_train_diagnostics": train_diagnostics,
            "quality_gate": train_gate,
            "label_free_train_quality_gate": train_gate,
            "heldout_retrieval": heldout_retrieval,
            "heldout_quality_gate": heldout_gate,
            "overall_usable": bool(train_gate["passed"] and heldout_gate["passed"]),
        },
        "architecture": {
            "input_dim": 512,
            "bits": config.bits,
            "hash_bits": config.bits,
            "image_layers": config.image_layers,
            "text_layers": config.text_layers,
            "hidden_width": config.hidden_width,
            "hash_output": "tanh -> L2 normalize -> sign at export",
            "branches": "independent weights",
        },
        "export": {
            "image_codes_shape": list(image_codes.shape),
            "text_codes_shape": list(text_codes.shape),
            "dtype": "int8",
            "domain": [-1, 1],
            "image_statistics": _code_statistics(image_codes),
            "text_statistics": _code_statistics(text_codes),
            "paired_code_agreement": float(np.mean(image_codes == text_codes)),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _file_sha256(checkpoint),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and torch.cuda.device_count()
                else None
            ),
            "source_sha256": _source_sha256(),
            "common_loader_source_sha256": (
                _file_sha256(loader_source) if loader_source.is_file() else None
            ),
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def run_train(args: argparse.Namespace) -> None:
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    checkpoint = args.checkpoint or args.output.with_suffix(".final.pt")
    manifest = args.manifest or args.output.parent / "RUN_MANIFEST.json"
    for path, label in (
        (args.output, "output"),
        (checkpoint, "checkpoint"),
        (manifest, "manifest"),
    ):
        if path.exists() and not args.overwrite:
            raise FileExistsError("{} exists: {}".format(label, path))

    config = UCCHConfig(
        bits=args.bits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_layers=args.image_layers,
        text_layers=args.text_layers,
        hidden_width=args.hidden_width,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        margin=args.margin,
        shift=args.shift,
        negatives=args.negatives,
        temperature=args.temperature,
        memory_momentum=args.memory_momentum,
        memory_warmup_epochs=args.memory_warmup_epochs,
        seed=args.seed,
        device=args.device,
    )
    dataset = load_prepared_dataset(args.data_root, args.dataset)
    config.validate(n_train=int(dataset.train_idx.size))
    result = train_ucch_f(
        dataset.image[dataset.train_idx],
        dataset.text[dataset.train_idx],
        config,
        verbose=not args.quiet,
    )
    train_image_codes = encode_all(
        result.image_model,
        dataset.image[dataset.train_idx],
        batch_size=args.export_batch_size,
        device=result.device,
    )
    train_text_codes = encode_all(
        result.text_model,
        dataset.text[dataset.train_idx],
        batch_size=args.export_batch_size,
        device=result.device,
    )
    train_diagnostics = pairing_diagnostics(
        train_image_codes, train_text_codes, result
    )
    train_gate = ucch_f_quality_gate(train_diagnostics)
    print(
        json.dumps(
            {"label_free_train_diagnostics": train_diagnostics, "train_gate": train_gate},
            indent=2,
            sort_keys=True,
        )
    )
    if not train_gate["passed"] and not args.allow_failed_quality_gate:
        raise RuntimeError(
            "UCCH-F label-free train gate failed; checkpoint/codes were not exported"
        )

    # Freeze and serialize the final epoch before opening any held-out label.
    save_final_checkpoint(checkpoint, result, config, dataset, args.overwrite)
    checkpoint_sha_before_heldout = _file_sha256(checkpoint)
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

    # Held-out labels become visible only below this line.
    heldout_retrieval: Dict[str, object] = {
        "protocol": (
            "fixed query_idx against database_idx once after final checkpoint freeze"
        ),
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
        "checkpoint_sha256_before_heldout": checkpoint_sha_before_heldout,
    }
    heldout_gate = heldout_retrieval_quality_gate(
        heldout_retrieval["i2t"], heldout_retrieval["t2i"]
    )
    print(
        json.dumps(
            {"heldout_retrieval": heldout_retrieval, "heldout_gate": heldout_gate},
            indent=2,
            sort_keys=True,
        )
    )
    if not heldout_gate["passed"] and not args.allow_failed_quality_gate:
        raise RuntimeError(
            "UCCH-F held-out acceptance gate failed; frozen checkpoint is retained "
            "as a diagnostic but no code NPZ was exported"
        )

    metadata = build_metadata(
        dataset,
        config,
        result,
        image_codes,
        text_codes,
        train_diagnostics,
        train_gate,
        heldout_retrieval,
        heldout_gate,
        checkpoint,
    )
    export_npz(
        args.output,
        image_codes,
        text_codes,
        dataset.labels,
        dataset.train_idx,
        dataset.query_idx,
        dataset.database_idx,
        metadata,
        overwrite=args.overwrite,
    )
    manifest_payload: Dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "USABLE"
            if train_gate["passed"] and heldout_gate["passed"]
            else "FAILED_DIAGNOSTIC_ONLY"
        ),
        "encoder": ENCODER_NAME,
        "official_commit": OFFICIAL_COMMIT,
        "command": [sys.executable] + sys.argv,
        "selection_rule": "fixed final epoch; checkpoint saved before held-out labels",
        "split_sha256": metadata["dataset"]["split_sha256"],
        "source_sha256": _source_sha256(),
        "files": {
            args.output.name: {
                "bytes": args.output.stat().st_size,
                "sha256": _file_sha256(args.output),
            },
            checkpoint.name: {
                "bytes": checkpoint.stat().st_size,
                "sha256": _file_sha256(checkpoint),
            },
        },
        "metadata": metadata,
        "npz_arrays": {},
        "summary": {
            "dataset": dataset.name,
            "bits": config.bits,
            "train_rows": int(dataset.train_idx.size),
            "query_rows": int(dataset.query_idx.size),
            "database_rows": int(dataset.database_idx.size),
            "train_gate_passed": bool(train_gate["passed"]),
            "heldout_gate_passed": bool(heldout_gate["passed"]),
            "paired_hamming_gap": train_diagnostics[
                "paired_gap_mismatched_minus_matched"
            ],
            "i2t": heldout_retrieval["i2t"],
            "t2i": heldout_retrieval["t2i"],
        },
    }
    with np.load(str(args.output), allow_pickle=False) as archive:
        manifest_payload["npz_arrays"] = {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
                "sha256": _array_sha256(np.asarray(archive[name])),
            }
            for name in archive.files
        }
    _write_json_atomic(manifest, manifest_payload)
    print(json.dumps(manifest_payload["summary"], indent=2, sort_keys=True))


def _synthetic_paired_features(seed: int, n_rows: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_rows, 32)).astype(np.float32)
    image_projection = rng.normal(size=(32, 512)).astype(np.float32)
    text_projection = rng.normal(size=(32, 512)).astype(np.float32)
    image = latent @ image_projection
    text = latent @ text_projection
    image += 0.03 * rng.normal(size=image.shape).astype(np.float32)
    text += 0.03 * rng.normal(size=text.shape).astype(np.float32)
    return np.ascontiguousarray(image), np.ascontiguousarray(text)


def run_smoke(args: argparse.Namespace) -> None:
    image, text = _synthetic_paired_features(args.seed)
    config = UCCHConfig(
        bits=16,
        epochs=args.epochs,
        batch_size=32,
        image_layers=2,
        text_layers=2,
        hidden_width=64,
        lr=5e-4,
        weight_decay=1e-6,
        alpha=0.7,
        margin=0.2,
        shift=0.1,
        negatives=63,
        temperature=0.9,
        memory_momentum=0.4,
        memory_warmup_epochs=1,
        seed=args.seed,
        device=args.device,
    )
    result = train_ucch_f(image, text, config, verbose=False)
    image_codes = encode_all(result.image_model, image, 64, result.device)
    text_codes = encode_all(result.text_model, text, 64, result.device)
    diagnostics = pairing_diagnostics(image_codes, text_codes, result)
    if result.image_parameter_delta <= 0 or result.text_parameter_delta <= 0:
        raise AssertionError("a feature branch did not update")
    if result.memory_delta <= 0:
        raise AssertionError("memory bank did not update")
    if image_codes.shape != (128, 16) or text_codes.shape != (128, 16):
        raise AssertionError("unexpected smoke code shape")
    if diagnostics["unique_image_code_rows"] <= 1:
        raise AssertionError("image codes collapsed")
    if diagnostics["unique_text_code_rows"] <= 1:
        raise AssertionError("text codes collapsed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "encoder": ENCODER_NAME,
                "config": asdict(config),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/export UCCH-F, a controlled fixed-feature adaptation of UCCH"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train on a fixed prepared split")
    train.add_argument("--dataset", choices=("mirflickr", "mscoco", "nuswide"), required=True)
    train.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data/ProcessData"),
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--manifest", type=Path)
    train.add_argument("--bits", type=int, choices=SUPPORTED_BITS, default=64)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--export-batch-size", type=int, default=4096)
    train.add_argument("--image-layers", type=int, default=3)
    train.add_argument("--text-layers", type=int, default=2)
    train.add_argument("--hidden-width", type=int, default=8192)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-6)
    train.add_argument("--alpha", type=float, default=0.7)
    train.add_argument("--margin", type=float, default=0.2)
    train.add_argument("--shift", type=float, default=0.1)
    train.add_argument("--negatives", type=int, default=4096)
    train.add_argument("--temperature", type=float, default=0.9)
    train.add_argument("--memory-momentum", type=float, default=0.4)
    train.add_argument("--memory-warmup-epochs", type=int, default=1)
    train.add_argument("--seed", type=int, default=20260805)
    train.add_argument("--device", default="auto")
    train.add_argument("--quiet", action="store_true")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument(
        "--allow-failed-quality-gate",
        action="store_true",
        help="export an explicitly failed diagnostic artifact; never cite it as evidence",
    )
    train.set_defaults(func=run_train)

    smoke = subparsers.add_parser("smoke", help="small paired-feature gradient smoke")
    smoke.add_argument("--epochs", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=20260805)
    smoke.add_argument("--device", default="cpu")
    smoke.set_defaults(func=run_smoke)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
