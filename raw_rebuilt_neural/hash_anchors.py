"""Train-only geometric and semantic anchor branches for RZ-CSD hash heads.

The compact encoder currently generalizes better than wider/deeper variants.
These branches therefore leave its adapters and residual trunk untouched and
add only conservative shared residuals to the hash logits:

* a fixed PCA projection fitted to paired CLIP features from ``indT`` only;
* a learned low-dimensional bridge from the calibrated posterior logits; or
* both branches together.

The PCA state is a buffer, not a fitted query/database transform.  All public
inference remains label-free after training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rz_csd_clip512 import (
    BITS,
    FEATURE_DIM,
    RZCSD512,
    RZCSD512Config,
    RZCSDTensorOutput,
    _validate_feature_tensor,
    _validate_modality,
)


@dataclass(frozen=True)
class HashAnchorSpec:
    name: str
    clip_pca: bool
    semantic_bridge: bool
    clip_gate_initial: float = 0.10
    semantic_gate_initial: float = 0.10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("anchor name must be nonempty")
        if not self.clip_pca and not self.semantic_bridge:
            raise ValueError("an anchored model needs at least one active branch")
        for value, label in (
            (self.clip_gate_initial, "clip_gate_initial"),
            (self.semantic_gate_initial, "semantic_gate_initial"),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{label} must lie in (0,1)")


HASH_ANCHOR_SPECS: tuple[HashAnchorSpec, ...] = (
    HashAnchorSpec(
        name="clip_pca_anchor_g010",
        clip_pca=True,
        semantic_bridge=False,
    ),
    HashAnchorSpec(
        name="posterior_semantic_anchor_g010",
        clip_pca=False,
        semantic_bridge=True,
    ),
    HashAnchorSpec(
        name="dual_clip_semantic_anchor_g010_g010",
        clip_pca=True,
        semantic_bridge=True,
    ),
    HashAnchorSpec(
        name="dual_clip_semantic_anchor_g020_g010",
        clip_pca=True,
        semantic_bridge=True,
        clip_gate_initial=0.20,
    ),
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def fit_clip_pca_anchor(
    image_features: np.ndarray,
    text_features: np.ndarray,
    *,
    components: int = max(BITS),
) -> dict[str, np.ndarray]:
    """Fit a deterministic shared PCA coordinate using aligned indT features."""

    image = np.asarray(image_features, dtype=np.float64)
    text = np.asarray(text_features, dtype=np.float64)
    if (
        image.ndim != 2
        or image.shape[1] != FEATURE_DIM
        or text.shape != image.shape
        or image.shape[0] < 2
    ):
        raise ValueError("image/text indT features must be aligned [n,512] matrices")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("PCA anchor features must be finite")
    if type(components) is not int or not 1 <= components <= FEATURE_DIM:
        raise ValueError("components must be an integer in [1,512]")

    def normalize(value: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        if np.any(norm <= 0.0):
            raise ValueError("PCA anchor contains a zero-norm feature")
        return value / norm

    image_unit = normalize(image)
    text_unit = normalize(text)
    rows = image.shape[0] + text.shape[0]
    center = (image_unit.sum(axis=0) + text_unit.sum(axis=0)) / rows
    image_centered = image_unit - center
    text_centered = text_unit - center
    covariance = (
        image_centered.T @ image_centered + text_centered.T @ text_centered
    ) / max(rows - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1][:components]
    projection = eigenvectors[:, order]
    # Eigenvector signs are arbitrary.  Canonicalize each direction by making
    # its largest-magnitude coordinate nonnegative.
    pivot = np.argmax(np.abs(projection), axis=0)
    signs = np.sign(projection[pivot, np.arange(components)])
    signs[signs == 0.0] = 1.0
    projection = projection * signs
    image_projected = image_centered @ projection
    text_projected = text_centered @ projection
    second_moment = (
        np.square(image_projected).sum(axis=0)
        + np.square(text_projected).sum(axis=0)
    ) / max(rows - 1, 1)
    scale = np.sqrt(second_moment).clip(min=1.0e-6)
    return {
        "center": np.asarray(center, dtype=np.float32),
        "projection": np.asarray(projection, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "eigenvalues": np.asarray(eigenvalues[order], dtype=np.float64),
    }


class AnchoredHashRZCSD512(RZCSD512):
    """Compact RZ-CSD with conservative shared residual hash coordinates."""

    def __init__(
        self,
        label_dim: int,
        config: RZCSD512Config,
        anchor_spec: HashAnchorSpec,
    ) -> None:
        super().__init__(label_dim=label_dim, config=config)
        self.anchor_spec = anchor_spec
        self.register_buffer("clip_anchor_center", torch.zeros(FEATURE_DIM))
        self.register_buffer(
            "clip_anchor_projection", torch.zeros(FEATURE_DIM, max(BITS))
        )
        self.register_buffer("clip_anchor_scale", torch.ones(max(BITS)))
        self.register_buffer("clip_anchor_is_bound", torch.tensor(False, dtype=torch.bool))
        if anchor_spec.clip_pca:
            self.clip_gate_logits = nn.ParameterDict(
                {
                    str(bits): nn.Parameter(
                        torch.tensor(_logit(anchor_spec.clip_gate_initial))
                    )
                    for bits in BITS
                }
            )
        else:
            self.clip_gate_logits = nn.ParameterDict()
        if anchor_spec.semantic_bridge:
            self.semantic_norm = nn.LayerNorm(label_dim)
            self.semantic_hash_bridges = nn.ModuleDict(
                {
                    str(bits): nn.Linear(label_dim, bits, bias=False)
                    for bits in BITS
                }
            )
            self.semantic_gate_logits = nn.ParameterDict(
                {
                    str(bits): nn.Parameter(
                        torch.tensor(_logit(anchor_spec.semantic_gate_initial))
                    )
                    for bits in BITS
                }
            )
        else:
            self.semantic_norm = None
            self.semantic_hash_bridges = nn.ModuleDict()
            self.semantic_gate_logits = nn.ParameterDict()

    def bind_clip_pca_anchor(self, state: dict[str, np.ndarray]) -> None:
        if not self.anchor_spec.clip_pca:
            raise RuntimeError("this variant does not use the CLIP PCA anchor")
        center = torch.as_tensor(state["center"], dtype=torch.float32)
        projection = torch.as_tensor(state["projection"], dtype=torch.float32)
        scale = torch.as_tensor(state["scale"], dtype=torch.float32)
        if center.shape != (FEATURE_DIM,):
            raise ValueError("PCA center has the wrong shape")
        if projection.shape != (FEATURE_DIM, max(BITS)):
            raise ValueError("PCA projection has the wrong shape")
        if scale.shape != (max(BITS),) or bool((scale <= 0.0).any().item()):
            raise ValueError("PCA scale has the wrong shape or values")
        if not all(bool(torch.isfinite(value).all().item()) for value in (center, projection, scale)):
            raise ValueError("PCA state must be finite")
        with torch.no_grad():
            self.clip_anchor_center.copy_(center.to(self.clip_anchor_center.device))
            self.clip_anchor_projection.copy_(
                projection.to(self.clip_anchor_projection.device)
            )
            self.clip_anchor_scale.copy_(scale.to(self.clip_anchor_scale.device))
            self.clip_anchor_is_bound.fill_(True)

    def _clip_anchor(self, features: torch.Tensor, bits: int) -> torch.Tensor:
        if not bool(self.clip_anchor_is_bound.item()):
            raise RuntimeError("bind_clip_pca_anchor must be called before inference")
        unit = F.normalize(features, dim=1, eps=1.0e-8)
        centered = unit - self.clip_anchor_center[None]
        projected = centered @ self.clip_anchor_projection[:, :bits]
        return projected / self.clip_anchor_scale[None, :bits]

    def forward(self, features: torch.Tensor, modality: str) -> RZCSDTensorOutput:
        _validate_feature_tensor(features)
        modality = _validate_modality(modality)
        adapted = self.adapters[modality](features)
        representation = self.shared_trunk(adapted)
        embedding = F.normalize(representation, dim=1, eps=1.0e-8)
        logits = torch.stack(
            [head(representation) for head in self.semantic_heads], dim=1
        )
        posterior = torch.sigmoid(
            logits - self.posterior_logit_offset[None, None]
        )
        semantic_signal = None
        if self.anchor_spec.semantic_bridge:
            calibrated_logits = (
                logits.mean(dim=1) - self.posterior_logit_offset[None]
            )
            if self.semantic_norm is None:
                raise AssertionError("semantic bridge norm is missing")
            semantic_signal = self.semantic_norm(calibrated_logits)
        continuous: dict[int, torch.Tensor] = {}
        for bits in BITS:
            code_logits = self.hash_heads[str(bits)](representation)
            if self.anchor_spec.clip_pca:
                clip_gate = torch.sigmoid(self.clip_gate_logits[str(bits)])
                code_logits = code_logits + clip_gate * self._clip_anchor(features, bits)
            if semantic_signal is not None:
                semantic_gate = torch.sigmoid(self.semantic_gate_logits[str(bits)])
                code_logits = code_logits + semantic_gate * self.semantic_hash_bridges[
                    str(bits)
                ](semantic_signal)
            continuous[bits] = torch.tanh(code_logits)
        binary = {
            bits: torch.where(
                value >= 0.0,
                torch.ones_like(value, dtype=torch.int8),
                -torch.ones_like(value, dtype=torch.int8),
            )
            for bits, value in continuous.items()
        }
        return RZCSDTensorOutput(
            embedding=embedding,
            continuous_codes=continuous,
            binary_codes=binary,
            posterior_logits=logits,
            posterior_heads=posterior,
        )


def hash_anchor_binding_record(state: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "rows_are_indt_only": True,
        "feature_normalization": "row_l2",
        "center_shape": list(np.asarray(state["center"]).shape),
        "projection_shape": list(np.asarray(state["projection"]).shape),
        "scale_shape": list(np.asarray(state["scale"]).shape),
        "leading_eigenvalues": [
            float(value) for value in np.asarray(state["eigenvalues"])[:8]
        ],
    }

