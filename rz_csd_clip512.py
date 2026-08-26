"""RZ-CSD-512: paired CLIP512 hashing and local semantic decoding.

This module is an isolated KBS method candidate.  It learns multi-width binary
codes and multi-label probability predictions from paired 512-D image/text
features and *training* labels.  Every public inference and ranking function is
label-free.  The global mixed-index coordinate is supplied by reference-Z (RZ);
the learned decoder can only reorder a tie-closed, uncertainty-connected head
region and returns the inactive RZ tail verbatim.

No official metric is computed here.  The implementation is intentionally
small enough to support same-input linear/MLP controls and causal ablations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import random
from typing import Any, Literal, Mapping, Sequence

import numpy as np


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

import torch
from torch import nn
from torch.nn import functional as F


BITS = (16, 32, 64)
FEATURE_DIM = 512
MODALITIES = ("image", "text")


@dataclass(frozen=True)
class RZCSD512Config:
    """One architecture/loss schedule shared across datasets.

    ``label_dim`` is deliberately not part of the configuration: only the last
    layer changes from MIRFLICKR (24) to NUS-WIDE-TC21 (21) and MS COCO (80).
    """

    protocol_version: str = "RZ_CSD_CLIP512_V1"
    feature_dim: int = FEATURE_DIM
    hidden_dim: int = 256
    feedforward_dim: int = 512
    residual_layers: int = 2
    posterior_hidden_dim: int = 128
    posterior_heads: int = 5
    dropout: float = 0.10
    ranking_logit_scale: float = 8.0
    bce_weight: float = 1.0
    alignment_weight: float = 0.20
    ranking_weight: float = 0.25
    quantization_weight: float = 0.08
    balance_weight: float = 0.01
    decorrelation_weight: float = 0.005
    subbag_keep_probability: float = 0.80
    minimum_head_pairwise_mad: float = 1.0e-4
    minimum_head_cell_std: float = 5.0e-5
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 40
    batch_size: int = 256
    inference_batch_size: int = 256
    active_window: int = 128
    max_active_candidates: int = 2048
    max_active_fraction: float = 0.05
    seed: int = 20260821

    def __post_init__(self) -> None:
        if self.feature_dim != FEATURE_DIM:
            raise ValueError("RZ-CSD-512 requires 512-D CLIP features")
        if self.hidden_dim <= 0 or self.feedforward_dim < self.hidden_dim:
            raise ValueError("hidden/feedforward dimensions are invalid")
        if self.residual_layers < 1 or self.posterior_hidden_dim < 1:
            raise ValueError("the shared trunk and posterior heads must be nonempty")
        if self.posterior_heads < 3:
            raise ValueError("at least three deterministic heads are required")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.ranking_logit_scale <= 0.0:
            raise ValueError("ranking_logit_scale must be positive")
        if not 0.0 < self.subbag_keep_probability <= 1.0:
            raise ValueError("subbag_keep_probability must lie in (0,1]")
        if self.minimum_head_pairwise_mad <= 0.0 or self.minimum_head_cell_std <= 0.0:
            raise ValueError("head-diversity thresholds must be positive")
        if (
            self.epochs < 1
            or self.batch_size < 2
            or self.inference_batch_size < 1
            or self.active_window < 1
            or self.max_active_candidates < self.active_window
        ):
            raise ValueError("training/window sizes are invalid")
        if not 0.0 < self.max_active_fraction <= 1.0:
            raise ValueError("max_active_fraction must lie in (0,1]")
        weights = (
            self.bce_weight,
            self.alignment_weight,
            self.ranking_weight,
            self.quantization_weight,
            self.balance_weight,
            self.decorrelation_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("loss weights must be finite and nonnegative")


FROZEN_CONFIG = RZCSD512Config()


def seed_everything(seed: int) -> None:
    """Configure reproducible training and deterministic inference."""

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG}, "
            f"got {configured!r}"
        )
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _validate_modality(modality: str) -> str:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    return modality


def _validate_feature_tensor(features: torch.Tensor) -> None:
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError("features must have shape [batch,512]")
    if not features.dtype.is_floating_point:
        raise TypeError("features must use a floating dtype")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("features must be finite")


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.ffn(self.norm(value))


class ModalityAdapter(nn.Module):
    """Small modality-specific front end before the shared semantic trunk."""

    def __init__(self, config: RZCSD512Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass
class RZCSDTensorOutput:
    embedding: torch.Tensor
    continuous_codes: Mapping[int, torch.Tensor]
    binary_codes: Mapping[int, torch.Tensor]
    posterior_logits: torch.Tensor
    posterior_heads: torch.Tensor


class RZCSD512(nn.Module):
    """Paired CLIP512 encoder with multi-width hashes and ensemble heads."""

    def __init__(
        self,
        label_dim: int,
        config: RZCSD512Config = FROZEN_CONFIG,
    ) -> None:
        super().__init__()
        if type(label_dim) is not int or label_dim <= 1:
            raise ValueError("label_dim must be an integer greater than one")
        self.label_dim = int(label_dim)
        self.config = config
        self.adapters = nn.ModuleDict(
            {modality: ModalityAdapter(config) for modality in MODALITIES}
        )
        self.shared_trunk = nn.Sequential(
            *[
                ResidualMLPBlock(
                    config.hidden_dim,
                    config.feedforward_dim,
                    config.dropout,
                )
                for _ in range(config.residual_layers)
            ],
            nn.LayerNorm(config.hidden_dim),
        )
        self.hash_heads = nn.ModuleDict(
            {
                str(bits): nn.Linear(config.hidden_dim, bits)
                for bits in BITS
            }
        )
        self.semantic_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.posterior_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.posterior_hidden_dim, self.label_dim),
                )
                for _ in range(config.posterior_heads)
            ]
        )
        # Positive-weighted BCE shifts the optimum by log(pos_weight).  The
        # fixed offset is stored with the model and removed at inference.
        self.register_buffer(
            "posterior_logit_offset",
            torch.zeros(self.label_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_prior_is_bound",
            torch.tensor(False, dtype=torch.bool),
        )

    def forward(self, features: torch.Tensor, modality: str) -> RZCSDTensorOutput:
        _validate_feature_tensor(features)
        modality = _validate_modality(modality)
        adapted = self.adapters[modality](features)
        representation = self.shared_trunk(adapted)
        embedding = F.normalize(representation, dim=1, eps=1e-8)
        continuous = {
            bits: torch.tanh(self.hash_heads[str(bits)](representation))
            for bits in BITS
        }
        binary = {
            bits: torch.where(
                value >= 0.0,
                torch.ones_like(value, dtype=torch.int8),
                -torch.ones_like(value, dtype=torch.int8),
            )
            for bits, value in continuous.items()
        }
        logits = torch.stack(
            [head(representation) for head in self.semantic_heads], dim=1
        )
        posterior = torch.sigmoid(logits - self.posterior_logit_offset[None, None])
        return RZCSDTensorOutput(
            embedding=embedding,
            continuous_codes=continuous,
            binary_codes=binary,
            posterior_logits=logits,
            posterior_heads=posterior,
        )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def deterministic_head_subbag_mask(
    identity_ids: np.ndarray,
    *,
    heads: int,
    keep_probability: float,
    seed: int,
) -> np.ndarray:
    """Counter-based Bernoulli subbagging independent of row/chunk order.

    Each training identity occurs at most once in a head's subset.  This is
    subbagging, not bootstrap resampling with replacement.
    """

    ids = np.asarray(identity_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("identity_ids must be a nonnegative integer vector")
    if type(heads) is not int or heads < 2:
        raise ValueError("heads must be an integer of at least two")
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("keep_probability must lie in (0,1]")
    identity = ids.astype(np.uint64, copy=False).reshape(-1, 1)
    head = np.arange(heads, dtype=np.uint64).reshape(1, -1)
    with np.errstate(over="ignore"):
        value = identity ^ (head * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    uniform = (value >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    mask = uniform < float(keep_probability)
    missing = np.flatnonzero(~mask.any(axis=1))
    if missing.size:
        fallback = (ids[missing].astype(np.int64) + int(seed)) % heads
        mask[missing, fallback] = True
    return mask


def configure_training_label_prior(
    model: RZCSD512,
    train_labels: np.ndarray | torch.Tensor,
    *,
    maximum_positive_weight: float = 20.0,
) -> torch.Tensor:
    """Set the BCE weight and its posterior logit correction from train only."""

    labels = torch.as_tensor(train_labels, dtype=torch.float32)
    if labels.ndim != 2 or labels.shape[1] != model.label_dim or labels.shape[0] < 2:
        raise ValueError("train_labels must have shape [train,label_dim]")
    if not bool(((labels == 0.0) | (labels == 1.0)).all().item()):
        raise ValueError("train_labels must be binary multi-hot rows")
    if bool((labels.sum(dim=1) == 0.0).any().item()):
        raise ValueError("train_labels contain an empty-label identity")
    if not math.isfinite(maximum_positive_weight) or maximum_positive_weight < 1.0:
        raise ValueError("maximum_positive_weight must be finite and at least one")
    positive = labels.sum(dim=0)
    zero_positive = torch.nonzero(positive == 0.0, as_tuple=False).flatten().tolist()
    if zero_positive:
        raise ValueError(
            "every posterior class needs a positive indT example; "
            f"zero-positive class indices={zero_positive}"
        )
    rows = float(labels.shape[0])
    weight = ((rows - positive) / positive.clamp_min(1.0)).clamp(
        min=1.0, max=float(maximum_positive_weight)
    )
    proposed_offset = torch.log(weight).to(model.posterior_logit_offset.device)
    if bool(model.posterior_prior_is_bound.item()):
        if not torch.allclose(
            proposed_offset,
            model.posterior_logit_offset,
            rtol=1.0e-7,
            atol=1.0e-7,
        ):
            raise RuntimeError(
                "posterior prior is already bound to a different indT label set"
            )
    else:
        with torch.no_grad():
            model.posterior_logit_offset.copy_(proposed_offset)
            model.posterior_prior_is_bound.fill_(True)
    return weight


def supervised_cross_modal_ranking_loss(
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    """Balanced all-pair cross-modal logistic ranking on train labels."""

    if image_embedding.shape != text_embedding.shape or image_embedding.ndim != 2:
        raise ValueError("image/text embeddings must be matching matrices")
    if labels.ndim != 2 or labels.shape[0] != image_embedding.shape[0]:
        raise ValueError("labels must align with the embedding batch")
    if logit_scale <= 0.0:
        raise ValueError("logit_scale must be positive")
    relevance = (labels @ labels.T > 0.0)
    similarity = float(logit_scale) * (image_embedding @ text_embedding.T)
    positive = F.softplus(-similarity)[relevance]
    negative = F.softplus(similarity)[~relevance]
    parts = []
    if positive.numel():
        parts.append(positive.mean())
    if negative.numel():
        parts.append(negative.mean())
    if not parts:
        return similarity.sum() * 0.0
    return torch.stack(parts).mean()


def _decorrelation_penalty(code: torch.Tensor) -> torch.Tensor:
    if code.ndim != 2:
        raise ValueError("code must be a matrix")
    normalized = code - code.mean(dim=0, keepdim=True)
    gram = normalized.T @ normalized / max(1, code.shape[0])
    diagonal = torch.diag_embed(torch.diagonal(gram))
    return (gram - diagonal).square().mean()


def compute_training_objective(
    model: RZCSD512,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    train_labels: torch.Tensor,
    identity_ids: np.ndarray,
    positive_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the complete train-only objective for one aligned mini-batch."""

    _validate_feature_tensor(image_features)
    _validate_feature_tensor(text_features)
    if image_features.shape != text_features.shape:
        raise ValueError("paired image/text feature batches must align")
    labels = train_labels.to(device=image_features.device, dtype=torch.float32)
    if labels.shape != (image_features.shape[0], model.label_dim):
        raise ValueError("train_labels do not align with this batch/model")
    if not bool(((labels == 0.0) | (labels == 1.0)).all().item()):
        raise ValueError("train_labels must be binary")
    ids = np.asarray(identity_ids)
    if ids.shape != (image_features.shape[0],):
        raise ValueError("identity_ids must contain one id per paired row")
    pos_weight = positive_weight.to(device=image_features.device, dtype=torch.float32)
    if pos_weight.shape != (model.label_dim,) or bool((pos_weight < 1.0).any().item()):
        raise ValueError("positive_weight must contain one value >=1 per label")
    if not bool(model.posterior_prior_is_bound.item()):
        raise RuntimeError("configure_training_label_prior must bind indT before training")
    if not torch.allclose(
        torch.log(pos_weight),
        model.posterior_logit_offset.to(image_features.device),
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise RuntimeError("positive_weight does not match the model's bound logit offset")

    image = model(image_features, "image")
    text = model(text_features, "text")
    target = labels[:, None, :].expand_as(image.posterior_logits)
    subbag = deterministic_head_subbag_mask(
        ids,
        heads=model.config.posterior_heads,
        keep_probability=model.config.subbag_keep_probability,
        seed=model.config.seed,
    )
    subbag_t = torch.from_numpy(subbag.astype(np.float32)).to(image_features.device)

    def supervised(logits: torch.Tensor) -> torch.Tensor:
        element = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="none"
        ).mean(dim=2)
        return (element * subbag_t).sum() / subbag_t.sum().clamp_min(1.0)

    bce = 0.5 * (supervised(image.posterior_logits) + supervised(text.posterior_logits))
    embedding_alignment = (1.0 - (image.embedding * text.embedding).sum(dim=1)).mean()
    code_alignment = torch.stack(
        [
            (image.continuous_codes[bits] - text.continuous_codes[bits])
            .square()
            .mean()
            for bits in BITS
        ]
    ).mean()
    alignment = 0.5 * (embedding_alignment + code_alignment)
    embedding_ranking = supervised_cross_modal_ranking_loss(
        image.embedding,
        text.embedding,
        labels,
        logit_scale=model.config.ranking_logit_scale,
    )
    code_ranking = torch.stack(
        [
            supervised_cross_modal_ranking_loss(
                F.normalize(image.continuous_codes[bits], dim=1, eps=1.0e-8),
                F.normalize(text.continuous_codes[bits], dim=1, eps=1.0e-8),
                labels,
                logit_scale=model.config.ranking_logit_scale,
            )
            for bits in BITS
        ]
    ).mean()
    ranking = 0.5 * (embedding_ranking + code_ranking)
    all_codes = [
        output.continuous_codes[bits]
        for output in (image, text)
        for bits in BITS
    ]
    quantization = torch.stack(
        [(code.abs() - 1.0).square().mean() for code in all_codes]
    ).mean()
    balance = torch.stack([code.mean(dim=0).square().mean() for code in all_codes]).mean()
    decorrelation = torch.stack([_decorrelation_penalty(code) for code in all_codes]).mean()
    config = model.config
    total = (
        config.bce_weight * bce
        + config.alignment_weight * alignment
        + config.ranking_weight * ranking
        + config.quantization_weight * quantization
        + config.balance_weight * balance
        + config.decorrelation_weight * decorrelation
    )
    return {
        "total": total,
        "bce": bce,
        "alignment": alignment,
        "ranking": ranking,
        "quantization": quantization,
        "balance": balance,
        "decorrelation": decorrelation,
    }


@dataclass(frozen=True)
class EncodedClip512:
    embedding: np.ndarray
    continuous_codes: Mapping[int, np.ndarray]
    binary_codes: Mapping[int, np.ndarray]
    posterior_heads: np.ndarray


@dataclass(frozen=True)
class HeadDiversityReport:
    heads: int
    items: int
    labels: int
    minimum_pairwise_mad: float
    mean_pairwise_mad: float
    mean_cell_std: float
    required_pairwise_mad: float
    required_cell_std: float
    passed: bool


def posterior_head_diversity_report(
    posterior_heads: np.ndarray,
    *,
    minimum_pairwise_mad: float = 1.0e-4,
    minimum_cell_std: float = 5.0e-5,
) -> HeadDiversityReport:
    """Measure deterministic head separation after training, without labels."""

    posterior = np.asarray(posterior_heads, dtype=np.float64)
    if posterior.ndim != 3 or posterior.shape[0] < 3 or posterior.shape[1] < 2:
        raise ValueError("posterior_heads must have shape [head>=3,item>=2,label]")
    if not np.isfinite(posterior).all() or np.any((posterior < 0.0) | (posterior > 1.0)):
        raise ValueError("posterior_heads must be finite probabilities")
    if minimum_pairwise_mad <= 0.0 or minimum_cell_std <= 0.0:
        raise ValueError("diversity thresholds must be positive")
    pairwise = np.asarray(
        [
            np.mean(np.abs(posterior[left] - posterior[right]), dtype=np.float64)
            for left in range(posterior.shape[0])
            for right in range(left + 1, posterior.shape[0])
        ],
        dtype=np.float64,
    )
    cell_std = float(np.std(posterior, axis=0, dtype=np.float64).mean())
    minimum = float(pairwise.min())
    mean = float(pairwise.mean())
    passed = bool(
        minimum >= minimum_pairwise_mad and cell_std >= minimum_cell_std
    )
    return HeadDiversityReport(
        heads=int(posterior.shape[0]),
        items=int(posterior.shape[1]),
        labels=int(posterior.shape[2]),
        minimum_pairwise_mad=minimum,
        mean_pairwise_mad=mean,
        mean_cell_std=cell_std,
        required_pairwise_mad=float(minimum_pairwise_mad),
        required_cell_std=float(minimum_cell_std),
        passed=passed,
    )


def require_posterior_head_diversity(
    posterior_heads: np.ndarray,
    *,
    minimum_pairwise_mad: float = 1.0e-4,
    minimum_cell_std: float = 5.0e-5,
) -> HeadDiversityReport:
    report = posterior_head_diversity_report(
        posterior_heads,
        minimum_pairwise_mad=minimum_pairwise_mad,
        minimum_cell_std=minimum_cell_std,
    )
    if not report.passed:
        raise RuntimeError(
            "deterministic posterior heads collapsed: "
            f"min_pairwise_mad={report.minimum_pairwise_mad:.6g}, "
            f"mean_cell_std={report.mean_cell_std:.6g}"
        )
    return report


def validate_indt_training_inputs(
    image_features: np.ndarray,
    text_features: np.ndarray,
    train_labels: np.ndarray,
    identity_ids: np.ndarray,
    *,
    expected_rows: int,
    label_dim: int,
) -> dict[str, Any]:
    """Fail-closed model boundary for already-sliced prepared ``indT`` rows."""

    image = np.asarray(image_features)
    text = np.asarray(text_features)
    labels = np.asarray(train_labels)
    ids = np.asarray(identity_ids)
    if image.shape != (expected_rows, FEATURE_DIM) or text.shape != image.shape:
        raise ValueError("indT image/text features have the wrong aligned shape")
    if not np.issubdtype(image.dtype, np.floating) or not np.issubdtype(
        text.dtype, np.floating
    ):
        raise TypeError("indT features must be floating point")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("indT features must be finite")
    if labels.shape != (expected_rows, label_dim) or not np.all(
        np.isin(labels, (0, 1))
    ):
        raise ValueError("indT labels must be an aligned binary multi-hot matrix")
    if np.any(labels.sum(axis=1) == 0) or np.any(labels.sum(axis=0) == 0):
        raise ValueError("indT labels contain an empty identity or zero-positive class")
    if (
        ids.shape != (expected_rows,)
        or ids.dtype.kind not in "iu"
        or np.unique(ids).size != expected_rows
    ):
        raise ValueError("indT identity_ids must be a unique integer vector")
    return {
        "scope": "prepared_indT_only",
        "rows": int(expected_rows),
        "feature_dim": FEATURE_DIM,
        "label_dim": int(label_dim),
        "unique_identity_ids": int(np.unique(ids).size),
        "query_or_database_labels_opened": False,
    }


@torch.no_grad()
def encode_clip512(
    model: RZCSD512,
    features: np.ndarray | torch.Tensor,
    *,
    modality: str,
    device: torch.device,
    batch_size: int = 4096,
) -> EncodedClip512:
    """Byte-stable unique-feature inference; no labels are accepted.

    ``batch_size`` remains a compatibility/memory-hint argument.  Numerical
    execution always uses the frozen canonical microbatch size and pads its
    last chunk, so caller chunking cannot change GEMM shapes.  Exact duplicate
    feature rows are evaluated once and scattered back.
    """

    modality = _validate_modality(modality)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not bool(model.posterior_prior_is_bound.item()):
        raise RuntimeError(
            "encode_clip512 requires a checkpoint whose indT posterior prior is bound"
        )
    if isinstance(features, torch.Tensor):
        value = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
    else:
        # Verified artifacts are intentionally opened as read-only memmaps.
        # ``ascontiguousarray`` may preserve that read-only backing store, which
        # PyTorch rejects because a tensor could otherwise write through it.
        value = torch.from_numpy(
            np.array(features, dtype=np.float32, order="C", copy=True)
        )
    _validate_feature_tensor(value)
    if len(value) == 0:
        raise ValueError("at least one feature row is required")
    unique_value, inverse = np.unique(
        value.numpy(), axis=0, return_inverse=True
    )
    canonical_batch = int(model.config.inference_batch_size)
    model.eval()
    embeddings = []
    continuous: dict[int, list[np.ndarray]] = {bits: [] for bits in BITS}
    binary: dict[int, list[np.ndarray]] = {bits: [] for bits in BITS}
    posterior = []
    pad_row = unique_value[0]
    for start in range(0, len(unique_value), canonical_batch):
        real = unique_value[start : start + canonical_batch]
        real_rows = len(real)
        padded = np.empty((canonical_batch, FEATURE_DIM), dtype=np.float32)
        padded[:real_rows] = real
        if real_rows < canonical_batch:
            padded[real_rows:] = pad_row
        batch = torch.from_numpy(padded).to(device)
        output = model(batch, modality)
        embeddings.append(output.embedding[:real_rows].cpu().numpy())
        posterior.append(output.posterior_heads[:real_rows].cpu().numpy())
        for bits in BITS:
            continuous[bits].append(
                output.continuous_codes[bits][:real_rows].cpu().numpy()
            )
            binary[bits].append(output.binary_codes[bits][:real_rows].cpu().numpy())
    unique_embedding = np.concatenate(embeddings, axis=0)
    unique_continuous = {
        bits: np.concatenate(parts, axis=0) for bits, parts in continuous.items()
    }
    unique_binary = {
        bits: np.concatenate(parts, axis=0) for bits, parts in binary.items()
    }
    unique_posterior = np.concatenate(posterior, axis=0)
    # Model output is [item,head,label]; the cache contract is
    # [head,item,label] so a query and a candidate cache share the same axis.
    posterior_array = np.transpose(unique_posterior[inverse], (1, 0, 2))
    return EncodedClip512(
        embedding=unique_embedding[inverse].astype(np.float32, copy=False),
        continuous_codes={
            bits: unique_continuous[bits][inverse].astype(np.float32, copy=False)
            for bits in BITS
        },
        binary_codes={
            bits: unique_binary[bits][inverse].astype(np.int8, copy=False)
            for bits in BITS
        },
        posterior_heads=posterior_array.astype(np.float32, copy=False),
    )


def _validate_bipolar_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    value = np.asarray(codes)
    if bits not in BITS or value.ndim != 2 or value.shape[1] != bits:
        raise ValueError(f"codes must have shape [rows,{bits}]")
    if not np.all(np.isin(value, (-1, 1))):
        raise ValueError("binary codes must be bipolar {-1,+1}")
    return value.astype(np.int8, copy=False)


def hamming_radius(query_code: np.ndarray, candidate_codes: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate_codes)
    if candidate.ndim != 2:
        raise ValueError("candidate_codes must be a matrix")
    query = np.asarray(query_code).reshape(-1)
    if query.shape != (candidate.shape[1],):
        raise ValueError("query and candidate code widths differ")
    if not np.all(np.isin(query, (-1, 1))) or not np.all(np.isin(candidate, (-1, 1))):
        raise ValueError("Hamming inputs must be bipolar")
    return np.count_nonzero(candidate != query[None], axis=1).astype(np.int16)


@dataclass(frozen=True)
class ReferenceZTables:
    image: np.ndarray
    text: np.ndarray
    image_mean: float
    image_std: float
    text_mean: float
    text_std: float
    zero_variance_fallback: bool


def reference_z_tables(
    query_code: np.ndarray,
    bank_image_codes: np.ndarray,
    bank_text_codes: np.ndarray,
) -> ReferenceZTables:
    """Build per-query, label-free RZ lookup tables from a paired train bank."""

    image = np.asarray(bank_image_codes)
    text = np.asarray(bank_text_codes)
    if image.shape != text.shape or image.ndim != 2 or image.shape[0] < 2:
        raise ValueError("paired reference-bank code matrices must align")
    bits = int(image.shape[1])
    _validate_bipolar_codes(image, bits)
    _validate_bipolar_codes(text, bits)
    image_radius = hamming_radius(query_code, image).astype(np.float64)
    text_radius = hamming_radius(query_code, text).astype(np.float64)
    image_mean, text_mean = float(image_radius.mean()), float(text_radius.mean())
    image_std, text_std = float(image_radius.std()), float(text_radius.std())
    radius = np.arange(bits + 1, dtype=np.float64)
    fallback = image_std == 0.0 or text_std == 0.0
    if fallback:
        image_table = text_table = -radius / float(bits)
    else:
        image_table = -(radius - image_mean) / image_std
        text_table = -(radius - text_mean) / text_std
    if not np.all(np.diff(image_table) < 0.0) or not np.all(np.diff(text_table) < 0.0):
        raise AssertionError("RZ radius lookup must be strictly decreasing")
    return ReferenceZTables(
        image=image_table,
        text=text_table,
        image_mean=image_mean,
        image_std=image_std,
        text_mean=text_mean,
        text_std=text_std,
        zero_variance_fallback=fallback,
    )


def rz_mixed_gallery_scores(
    query_code: np.ndarray,
    gallery_image_codes: np.ndarray,
    gallery_text_codes: np.ndarray,
    text_mask: np.ndarray,
    *,
    bank_image_codes: np.ndarray,
    bank_text_codes: np.ndarray,
) -> np.ndarray:
    """Place two Hamming indexes in one per-query reference-Z coordinate."""

    image = np.asarray(gallery_image_codes)
    text = np.asarray(gallery_text_codes)
    if image.shape != text.shape or image.ndim != 2:
        raise ValueError("gallery image/text codes must be aligned matrices")
    mask = np.asarray(text_mask, dtype=bool)
    if mask.shape != (image.shape[0],):
        raise ValueError("text_mask must contain one flag per gallery identity")
    tables = reference_z_tables(query_code, bank_image_codes, bank_text_codes)
    image_radius = hamming_radius(query_code, image)
    text_radius = hamming_radius(query_code, text)
    score = np.where(mask, tables.text[text_radius], tables.image[image_radius])
    return score.astype(np.float64, copy=False)


def semantic_relation_heads(
    query_posterior_heads: np.ndarray,
    candidate_posterior_heads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """AP relevance probability and graded soft-Jaccard for each fixed head."""

    query = np.asarray(query_posterior_heads, dtype=np.float64)
    candidate = np.asarray(candidate_posterior_heads, dtype=np.float64)
    if query.ndim == 2:
        query = query[:, None, :]
    if query.ndim != 3 or candidate.ndim != 3 or query.shape[1] != 1:
        raise ValueError("posterior caches must be [head,1,label]/[head,item,label]")
    if query.shape[0] != candidate.shape[0] or query.shape[2] != candidate.shape[2]:
        raise ValueError("query/candidate posterior dimensions differ")
    if not np.isfinite(query).all() or not np.isfinite(candidate).all():
        raise ValueError("posteriors must be finite")
    if np.any((query < 0.0) | (query > 1.0)) or np.any(
        (candidate < 0.0) | (candidate > 1.0)
    ):
        raise ValueError("posteriors must lie in [0,1]")
    shared = query * candidate
    relevance = -np.expm1(
        np.log1p(-np.clip(shared, 0.0, 1.0 - 1.0e-7)).sum(axis=2)
    )
    union = query + candidate - shared
    jaccard = shared.sum(axis=2) / np.maximum(union.sum(axis=2), 1.0e-8)
    return relevance.astype(np.float64), jaccard.astype(np.float64)


@dataclass(frozen=True)
class RelationSummary:
    relevance_mean: np.ndarray
    relevance_lower: np.ndarray
    relevance_upper: np.ndarray
    jaccard_mean: np.ndarray
    jaccard_lower: np.ndarray
    jaccard_upper: np.ndarray


def summarize_relation_heads(
    relevance_heads: np.ndarray,
    jaccard_heads: np.ndarray,
) -> RelationSummary:
    relevance = np.asarray(relevance_heads, dtype=np.float64)
    jaccard = np.asarray(jaccard_heads, dtype=np.float64)
    if relevance.shape != jaccard.shape or relevance.ndim != 2 or relevance.shape[0] < 2:
        raise ValueError("relation samples must be matching [head,item] matrices")
    if not np.isfinite(relevance).all() or not np.isfinite(jaccard).all():
        raise ValueError("relation samples must be finite")
    return RelationSummary(
        relevance_mean=relevance.mean(axis=0),
        relevance_lower=relevance.min(axis=0),
        relevance_upper=relevance.max(axis=0),
        jaccard_mean=jaccard.mean(axis=0),
        jaccard_lower=jaccard.min(axis=0),
        jaccard_upper=jaccard.max(axis=0),
    )


def _validate_rz_intervals(
    scores: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if (lower is None) != (upper is None):
        raise ValueError("RZ lower/upper intervals must be supplied together")
    if lower is None:
        return scores.copy(), scores.copy()
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != scores.shape or hi.shape != scores.shape:
        raise ValueError("RZ intervals must match scores")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError("RZ intervals are invalid")
    if np.any(scores < lo) or np.any(scores > hi):
        raise ValueError("each RZ score must lie inside its interval")
    return lo, hi


def _rz_order(scores: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((candidate_ids, -scores)).astype(np.int64, copy=False)


def tie_closed_uncertainty_prefix(
    rz_scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    window_size: int,
    rz_lower: np.ndarray | None = None,
    rz_upper: np.ndarray | None = None,
    max_active_candidates: int = FROZEN_CONFIG.max_active_candidates,
    max_active_fraction: float = FROZEN_CONFIG.max_active_fraction,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded RZ prefix closed under boundary ties/overlap.

    The quadratic local partial-order stage is never allowed to degrade into a
    full-gallery reranker.  Tie/interval closure fails closed when it exceeds
    either the registered absolute cap or gallery-fraction cap.
    """

    score = np.asarray(rz_scores, dtype=np.float64)
    ids = np.asarray(candidate_ids)
    if score.ndim != 1 or score.size == 0 or not np.isfinite(score).all():
        raise ValueError("rz_scores must be a nonempty finite vector")
    if ids.shape != score.shape or ids.dtype.kind not in "iu" or np.unique(ids).size != ids.size:
        raise ValueError("candidate_ids must be a unique integer vector")
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if type(max_active_candidates) is not int or max_active_candidates < window_size:
        raise ValueError("max_active_candidates must be an integer >= window_size")
    if not 0.0 < max_active_fraction <= 1.0:
        raise ValueError("max_active_fraction must lie in (0,1]")
    active_limit = max(
        min(window_size, len(score)),
        min(
            max_active_candidates,
            int(math.ceil(max_active_fraction * len(score))),
        ),
    )
    lower, upper = _validate_rz_intervals(score, rz_lower, rz_upper)
    order = _rz_order(score, ids)
    end = min(window_size, len(score))
    threshold = score[order[end - 1]]
    while end < len(score) and score[order[end]] == threshold:
        end += 1
    if end > active_limit:
        raise RuntimeError(
            f"tie-closed active set {end} exceeds registered limit {active_limit}"
        )
    # Include the complete interval-connected boundary neighborhood.  Because
    # an active region must remain a prefix, every intervening row is included.
    ordered_upper = upper[order]
    suffix_max_upper = np.maximum.accumulate(ordered_upper[::-1])[::-1]
    while end < len(score):
        boundary_lower = float(lower[order[:end]].min())
        if (
            active_limit < len(score)
            and suffix_max_upper[active_limit] >= boundary_lower
        ):
            raise RuntimeError(
                "uncertainty closure reaches beyond the registered local cap"
            )
        eligible = np.flatnonzero(
            ordered_upper[end:active_limit] >= boundary_lower
        )
        if eligible.size == 0:
            break
        new_end = end + int(eligible[-1]) + 1
        threshold = score[order[new_end - 1]]
        while new_end < len(score) and score[order[new_end]] == threshold:
            new_end += 1
        if new_end == end:
            break
        if new_end > active_limit:
            raise RuntimeError(
                "uncertainty closure would expand local decoding to "
                f"{new_end} candidates (limit={active_limit})"
            )
        end = new_end
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:end]] = True
    return mask, order


def canonicalize_relation_evidence(
    relevance_heads: np.ndarray,
    jaccard_heads: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    """Canonical joint head distributions, invariant to head permutation."""

    relevance = np.asarray(relevance_heads, dtype=np.float64)
    jaccard = np.asarray(jaccard_heads, dtype=np.float64)
    summarize_relation_heads(relevance, jaccard)
    canonical: list[tuple[float, ...]] = []
    for item in range(relevance.shape[1]):
        pairs = np.column_stack((relevance[:, item], jaccard[:, item]))
        head_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        canonical.append(tuple(float(value) for value in pairs[head_order].reshape(-1)))
    return tuple(canonical)


def _semantic_priority(
    item: int,
    *,
    relation: RelationSummary,
    canonical: tuple[tuple[float, ...], ...],
    use_uncertainty: bool,
    use_graded: bool,
) -> tuple[float, ...]:
    relevance_spread = float(
        relation.relevance_upper[item] - relation.relevance_lower[item]
    )
    jaccard_spread = float(
        relation.jaccard_upper[item] - relation.jaccard_lower[item]
    )
    if use_uncertainty and use_graded:
        prefix = (
            float(relation.jaccard_mean[item]),
            float(relation.jaccard_lower[item]),
            -jaccard_spread,
            float(relation.relevance_mean[item]),
            float(relation.relevance_lower[item]),
            -relevance_spread,
        )
    elif use_uncertainty:
        prefix = (
            float(relation.relevance_mean[item]),
            float(relation.relevance_lower[item]),
            -relevance_spread,
        )
    elif use_graded:
        prefix = (
            float(relation.relevance_mean[item]),
            float(relation.jaccard_mean[item]),
        )
    else:
        prefix = (float(relation.relevance_mean[item]),)
    canonical_suffix = () if not use_uncertainty else (
        canonical[item] if use_graded else canonical[item][0::2]
    )
    return prefix + canonical_suffix


def _stable_hierarchical_groups(
    active: np.ndarray,
    *,
    ids: np.ndarray,
    rz_lower: np.ndarray,
    rz_upper: np.ndarray,
    relation: RelationSummary,
    canonical: tuple[tuple[float, ...], ...],
    use_uncertainty: bool,
    use_graded: bool,
) -> tuple[list[np.ndarray], "PartialOrderDiagnostics"]:
    """Stable acyclic RZ-first, AP-second, graded-third partial order.

    Only strict non-overlapping RZ intervals become graph edges.  They form an
    interval order and are therefore acyclic.  Kahn frontiers are then filtered
    by strict AP-interval dominance; semantic evidence never creates a graph
    edge and consequently cannot introduce a cycle or override RZ precedence.
    Equal canonical semantic evidence is removed as one expected-tie group.

    Both dominance matrices are built once.  RZ indegrees and the number of
    AP dominators in the live Kahn frontier are then updated incrementally as
    rows enter or leave that frontier.  Consequently a candidate pair is
    never re-tested inside the loop: time and boolean workspace are O(A^2)
    for active-set size A (subject to the explicit active-set cap).
    """

    size = len(active)
    active_lower = rz_lower[active]
    active_upper = rz_upper[active]
    rz_dominance = active_lower[:, None] > active_upper[None, :]
    np.fill_diagonal(rz_dominance, False)
    rz_indegree = rz_dominance.sum(axis=0, dtype=np.int64)
    rz_edge_count = int(np.count_nonzero(rz_dominance))

    if use_uncertainty:
        relevance_lower = relation.relevance_lower[active]
        relevance_upper = relation.relevance_upper[active]
        ap_dominance = relevance_lower[:, None] > relevance_upper[None, :]
        np.fill_diagonal(ap_dominance, False)
        ap_pair_tests = size * size
        ap_edge_count = int(np.count_nonzero(ap_dominance))
    else:
        ap_dominance = np.zeros((size, size), dtype=bool)
        ap_pair_tests = 0
        ap_edge_count = 0

    remaining = np.ones(size, dtype=bool)
    available = rz_indegree == 0
    ap_incoming = ap_dominance[available].sum(axis=0, dtype=np.int64)
    groups: list[np.ndarray] = []
    removed = 0
    frontier_priority_visits = 0
    dominance_update_cells = int(np.count_nonzero(available)) * size
    while bool(available.any()):
        frontier = np.flatnonzero(available)
        if use_uncertainty and frontier.size > 1:
            ap_maxima = frontier[ap_incoming[frontier] == 0]
            if ap_maxima.size == 0:
                raise AssertionError("strict AP interval order has no maximal element")
            frontier = ap_maxima
        frontier_priority_visits += int(frontier.size)
        priorities = {
            int(position): _semantic_priority(
                int(active[position]),
                relation=relation,
                canonical=canonical,
                use_uncertainty=use_uncertainty,
                use_graded=use_graded,
            )
            for position in frontier
        }
        best = max(priorities.values())
        chosen_positions = [
            position for position in frontier if priorities[position] == best
        ]
        # IDs affect display order only; all chosen rows receive one rank key.
        chosen_positions.sort(key=lambda position: int(ids[int(active[position])]))
        group = np.asarray(
            [int(active[position]) for position in chosen_positions], dtype=np.int64
        )
        groups.append(group)
        chosen = np.asarray(chosen_positions, dtype=np.int64)
        chosen_count = int(chosen.size)
        # Remove all selected sources from both incremental dominance systems.
        ap_incoming -= ap_dominance[chosen].sum(axis=0, dtype=np.int64)
        rz_indegree -= rz_dominance[chosen].sum(axis=0, dtype=np.int64)
        dominance_update_cells += 2 * chosen_count * size
        available[chosen] = False
        remaining[chosen] = False
        removed += chosen_count

        newly_available = remaining & ~available & (rz_indegree == 0)
        new_count = int(np.count_nonzero(newly_available))
        if new_count:
            ap_incoming += ap_dominance[newly_available].sum(
                axis=0, dtype=np.int64
            )
            dominance_update_cells += new_count * size
            available[newly_available] = True
    if removed != size:
        raise AssertionError("RZ interval graph unexpectedly contains a cycle")
    diagnostics = PartialOrderDiagnostics(
        active_size=int(size),
        rz_hard_edges=rz_edge_count,
        potential_ap_hard_edges=ap_edge_count,
        rz_pair_tests=size * size,
        ap_pair_tests=ap_pair_tests,
        frontier_priority_visits=int(frontier_priority_visits),
        dominance_update_cells=int(dominance_update_cells),
    )
    return groups, diagnostics


@dataclass(frozen=True)
class PartialOrderDiagnostics:
    """Deterministic work counters for the capped local partial-order pass."""

    active_size: int
    rz_hard_edges: int
    potential_ap_hard_edges: int
    rz_pair_tests: int
    ap_pair_tests: int
    frontier_priority_visits: int
    dominance_update_cells: int

    @property
    def operation_count(self) -> int:
        return int(
            self.rz_pair_tests
            + self.ap_pair_tests
            + self.frontier_priority_visits
            + self.dominance_update_cells
        )


@dataclass(frozen=True)
class LocalDecodeResult:
    order: np.ndarray
    rz_order: np.ndarray
    rank_group_keys: np.ndarray
    active_mask: np.ndarray
    active_size: int
    changed_pairs: int
    rz_hard_edges: int
    partial_order_diagnostics: PartialOrderDiagnostics
    use_uncertainty: bool
    use_graded: bool


def _compile_rank_group_keys(
    active_groups: Sequence[np.ndarray],
    tail: np.ndarray,
    *,
    rz_scores: np.ndarray,
) -> np.ndarray:
    """Compile active semantic groups then exact inactive RZ score blocks."""

    total = sum(len(group) for group in active_groups) + len(tail)
    keys = np.empty(total, dtype=np.int64)
    group_index = 0
    for group in active_groups:
        keys[group] = total - group_index
        group_index += 1
    previous_score: float | None = None
    for raw_item in tail:
        item = int(raw_item)
        score = float(rz_scores[item])
        if previous_score is not None and score != previous_score:
            group_index += 1
        keys[item] = total - group_index
        previous_score = score
    return keys


def decode_rz_local(
    rz_scores: np.ndarray,
    candidate_ids: np.ndarray,
    relevance_heads: np.ndarray,
    jaccard_heads: np.ndarray,
    *,
    window_size: int,
    rz_lower: np.ndarray | None = None,
    rz_upper: np.ndarray | None = None,
    use_uncertainty: bool = True,
    use_graded: bool = True,
    max_active_candidates: int = FROZEN_CONFIG.max_active_candidates,
    max_active_fraction: float = FROZEN_CONFIG.max_active_fraction,
) -> LocalDecodeResult:
    """Refine only an RZ tie/uncertainty head; labels are never accepted."""

    score = np.asarray(rz_scores, dtype=np.float64)
    ids = np.asarray(candidate_ids)
    relation = summarize_relation_heads(relevance_heads, jaccard_heads)
    if relation.relevance_mean.shape != score.shape:
        raise ValueError("semantic relation rows must match RZ candidates")
    active_mask, base_order = tie_closed_uncertainty_prefix(
        score,
        ids,
        window_size=window_size,
        rz_lower=rz_lower,
        rz_upper=rz_upper,
        max_active_candidates=max_active_candidates,
        max_active_fraction=max_active_fraction,
    )
    lower, upper = _validate_rz_intervals(score, rz_lower, rz_upper)
    active = base_order[active_mask[base_order]]
    canonical = canonicalize_relation_evidence(relevance_heads, jaccard_heads)
    active_groups, partial_order_diagnostics = _stable_hierarchical_groups(
        active,
        ids=ids,
        rz_lower=lower,
        rz_upper=upper,
        relation=relation,
        canonical=canonical,
        use_uncertainty=bool(use_uncertainty),
        use_graded=bool(use_graded),
    )
    refined = (
        np.concatenate(active_groups) if active_groups else np.empty(0, np.int64)
    )
    tail = base_order[~active_mask[base_order]]
    order = np.concatenate((refined, tail)).astype(np.int64, copy=False)
    if np.unique(order).size != len(score):
        raise AssertionError("local decoder did not return a complete permutation")
    if not np.array_equal(order[len(refined) :], tail):
        raise AssertionError("inactive RZ tail changed")
    rank_group_keys = _compile_rank_group_keys(
        active_groups,
        tail,
        rz_scores=score,
    )
    base_position = np.empty(len(score), dtype=np.int64)
    new_position = np.empty(len(score), dtype=np.int64)
    base_position[base_order] = np.arange(len(score))
    new_position[order] = np.arange(len(score))
    active_rows = np.flatnonzero(active_mask)
    changed = 0
    for left in range(len(active_rows)):
        i = int(active_rows[left])
        for right in range(left + 1, len(active_rows)):
            j = int(active_rows[right])
            changed += int(
                (base_position[i] - base_position[j])
                * (new_position[i] - new_position[j])
                < 0
            )
    return LocalDecodeResult(
        order=order,
        rz_order=base_order,
        rank_group_keys=rank_group_keys,
        active_mask=active_mask,
        active_size=int(active_mask.sum()),
        changed_pairs=int(changed),
        rz_hard_edges=int(partial_order_diagnostics.rz_hard_edges),
        partial_order_diagnostics=partial_order_diagnostics,
        use_uncertainty=bool(use_uncertainty),
        use_graded=bool(use_graded),
    )


def raw_clip_cosine(
    query_feature: np.ndarray,
    candidate_features: np.ndarray,
) -> np.ndarray:
    """Label-free same-input raw CLIP baseline."""

    query = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_features, dtype=np.float32)
    if query.shape != (FEATURE_DIM,) or candidate.ndim != 2 or candidate.shape[1] != FEATURE_DIM:
        raise ValueError("raw CLIP inputs must have 512 columns")
    if not np.isfinite(query).all() or not np.isfinite(candidate).all():
        raise ValueError("raw CLIP inputs must be finite")
    query = query / max(float(np.linalg.norm(query)), 1.0e-8)
    denominator = np.maximum(np.linalg.norm(candidate, axis=1), 1.0e-8)
    return ((candidate @ query) / denominator).astype(np.float32)


class SameInputPosteriorControl(nn.Module):
    """Linear or shallow-MLP posterior baseline on the identical CLIP512 input."""

    def __init__(
        self,
        label_dim: int,
        *,
        kind: Literal["linear", "mlp"],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if label_dim <= 1 or hidden_dim <= 0 or kind not in ("linear", "mlp"):
            raise ValueError("invalid same-input control configuration")
        self.kind = kind
        self.label_dim = int(label_dim)
        if kind == "linear":
            self.network = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, label_dim))
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(FEATURE_DIM),
                nn.Linear(FEATURE_DIM, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, label_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(features)
        return self.network(features)


def build_capacity_matched_mlp_control(
    full_model: RZCSD512,
) -> tuple[SameInputPosteriorControl, dict[str, Any]]:
    """Choose the shallow-control width nearest to the full parameter budget."""

    target = parameter_count(full_model)
    label_dim = full_model.label_dim
    # LayerNorm(512) + Linear(512,h) + LayerNorm(h) + Linear(h,L).
    fixed = 2 * FEATURE_DIM + label_dim
    per_hidden = FEATURE_DIM + 1 + 2 + label_dim
    estimate = max(1.0, (target - fixed) / float(per_hidden))
    centers = {max(1, int(math.floor(estimate))), max(1, int(math.ceil(estimate)))}
    candidates = sorted(
        {
            max(1, center + delta)
            for center in centers
            for delta in range(-2, 3)
        }
    )
    hidden_dim = min(
        candidates,
        key=lambda hidden: (
            abs((fixed + per_hidden * hidden) - target),
            hidden,
        ),
    )
    control = SameInputPosteriorControl(
        label_dim, kind="mlp", hidden_dim=hidden_dim
    )
    actual = parameter_count(control)
    relative_gap = abs(actual - target) / float(target)
    if relative_gap > 0.005:
        raise AssertionError(
            f"capacity-matched control gap {relative_gap:.3%} exceeds 0.5%"
        )
    return control, {
        "target_full_parameters": int(target),
        "control_parameters": int(actual),
        "hidden_dim": int(hidden_dim),
        "absolute_gap": int(abs(actual - target)),
        "relative_gap": float(relative_gap),
        "within_half_percent": True,
    }


def ablation_registry() -> dict[str, dict[str, Any]]:
    """Executable switches/controls required by the causal experiment table."""

    return {
        "raw_clip": {"adapter": "raw_clip_cosine", "uses_labels_at_fit": False},
        "linear": {"adapter": "SameInputPosteriorControl(kind='linear')", "uses_labels_at_fit": True},
        "capacity_mlp": {"adapter": "build_capacity_matched_mlp_control", "uses_labels_at_fit": True},
        "raw_hamming": {"global_score": "negative raw Hamming radius", "local_decoder": False},
        "rz": {"global_score": "reference-Z", "local_decoder": False},
        "rz_relevance": {"global_score": "reference-Z", "use_graded": False, "use_uncertainty": True},
        "no_rz": {"global_score": "negative raw Hamming radius", "decoder": "unchanged"},
        "no_graded": {"decode_rz_local.use_graded": False},
        "no_uncertainty": {"decode_rz_local.use_uncertainty": False},
        "full": {"global_score": "reference-Z", "use_uncertainty": True, "use_graded": True},
    }


def architecture_report(label_dim: int, config: RZCSD512Config = FROZEN_CONFIG) -> dict[str, Any]:
    model = RZCSD512(label_dim, config)
    _control, capacity_match = build_capacity_matched_mlp_control(model)
    return {
        "protocol": config.protocol_version,
        "feature_input": "paired image_features_clip512/text_features_clip512",
        "fit_label_input": "prepared indT rows only",
        "inference_label_input": "none",
        "bits": list(BITS),
        "label_dim": int(label_dim),
        "parameter_count": parameter_count(model),
        "deterministic_posterior_heads": config.posterior_heads,
        "head_training": "identity-keyed Bernoulli subbagging",
        "head_diversity_gate": {
            "minimum_pairwise_mad": config.minimum_head_pairwise_mad,
            "minimum_cell_std": config.minimum_head_cell_std,
        },
        "capacity_matched_mlp": capacity_match,
        "local_numeric_contract": "float64",
        "config": asdict(config),
        "ablations": ablation_registry(),
    }


__all__ = [
    "BITS",
    "FEATURE_DIM",
    "FROZEN_CONFIG",
    "EncodedClip512",
    "HeadDiversityReport",
    "LocalDecodeResult",
    "PartialOrderDiagnostics",
    "RZCSD512",
    "RZCSD512Config",
    "ReferenceZTables",
    "RelationSummary",
    "SameInputPosteriorControl",
    "ablation_registry",
    "architecture_report",
    "build_capacity_matched_mlp_control",
    "canonicalize_relation_evidence",
    "compute_training_objective",
    "configure_training_label_prior",
    "decode_rz_local",
    "deterministic_head_subbag_mask",
    "encode_clip512",
    "hamming_radius",
    "parameter_count",
    "posterior_head_diversity_report",
    "raw_clip_cosine",
    "reference_z_tables",
    "require_posterior_head_diversity",
    "rz_mixed_gallery_scores",
    "seed_everything",
    "semantic_relation_heads",
    "summarize_relation_heads",
    "supervised_cross_modal_ranking_loss",
    "tie_closed_uncertainty_prefix",
    "validate_indt_training_inputs",
]
