"""Training-only semantic objectives for the frozen RZ-CSD curriculum.

The auxiliary decoders consume only continuous hash codes and indT labels.
They are discarded at inference: the deployed encoder and its binary codes do
not gain an extra serving dependency.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from rz_csd_clip512 import BITS, RZCSD512


class HashSemanticDecoders(nn.Module):
    """Small width-specific heads that make every hash code label-decodable."""

    def __init__(self, label_dim: int) -> None:
        super().__init__()
        if type(label_dim) is not int or label_dim < 1:
            raise ValueError("label_dim must be a positive integer")
        self.heads = nn.ModuleDict(
            {
                str(bits): nn.Sequential(
                    nn.LayerNorm(bits),
                    nn.Linear(bits, label_dim),
                )
                for bits in BITS
            }
        )

    def forward(self, codes: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if set(codes) != set(BITS):
            raise ValueError(f"codes must contain exactly the registered widths {BITS}")
        return {bits: self.heads[str(bits)](codes[bits]) for bits in BITS}


def _balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=positive_weight,
    )


def _straight_through_binary(code: torch.Tensor) -> torch.Tensor:
    binary = torch.where(code >= 0.0, torch.ones_like(code), -torch.ones_like(code))
    return code + (binary - code).detach()


def _balanced_graded_pair_loss(
    image_code: torch.Tensor,
    text_code: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Regress cross-modal code similarity to label-set Jaccard overlap."""

    intersection = labels @ labels.T
    cardinality = labels.sum(dim=1)
    union = cardinality[:, None] + cardinality[None, :] - intersection
    target = intersection / union.clamp_min(1.0)
    prediction = 0.5 * (
        F.normalize(image_code, dim=1, eps=1.0e-8)
        @ F.normalize(text_code, dim=1, eps=1.0e-8).T
        + 1.0
    )
    element = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.10)
    positive = intersection > 0.0
    negative = ~positive
    parts = []
    if bool(positive.any().item()):
        parts.append(element[positive].mean())
    if bool(negative.any().item()):
        parts.append(element[negative].mean())
    return torch.stack(parts).mean()


def _posterior_soft_jaccard_loss(
    posterior_heads: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    target = labels[:, None, :].expand_as(posterior_heads)
    intersection = (posterior_heads * target).sum(dim=2)
    union = (posterior_heads + target - posterior_heads * target).sum(dim=2)
    return (1.0 - intersection / union.clamp_min(1.0e-6)).mean()


def compute_auxiliary_training_objective(
    model: RZCSD512,
    decoders: HashSemanticDecoders,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the frozen training-only code semantics on a paired batch."""

    image = model(image_features, "image")
    text = model(text_features, "text")
    image_ste = {
        bits: _straight_through_binary(image.continuous_codes[bits]) for bits in BITS
    }
    text_ste = {
        bits: _straight_through_binary(text.continuous_codes[bits]) for bits in BITS
    }
    image_logits = decoders(image_ste)
    text_logits = decoders(text_ste)
    code_bce = torch.stack(
        [
            0.5
            * (
                _balanced_bce(image_logits[bits], labels, positive_weight)
                + _balanced_bce(text_logits[bits], labels, positive_weight)
            )
            for bits in BITS
        ]
    ).mean()
    graded = torch.stack(
        [
            _balanced_graded_pair_loss(image_ste[bits], text_ste[bits], labels)
            for bits in BITS
        ]
    ).mean()
    posterior_jaccard = 0.5 * (
        _posterior_soft_jaccard_loss(image.posterior_heads, labels)
        + _posterior_soft_jaccard_loss(text.posterior_heads, labels)
    )
    return {
        "code_bce": code_bce,
        "graded": graded,
        "posterior_jaccard": posterior_jaccard,
    }


__all__ = ["HashSemanticDecoders", "compute_auxiliary_training_objective"]
