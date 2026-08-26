"""Compact hash-head variants for indT-only architecture diagnosis.

The encoder, posterior ensemble, losses, and training schedule stay fixed.
Only the final hash projection changes so that normalization, a very small
modality correction, and cross-width parameter sharing can be tested without
confounding them with a larger backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from rz_csd_clip512 import (
    BITS,
    MODALITIES,
    RZCSD512,
    RZCSD512Config,
    RZCSDTensorOutput,
    _validate_feature_tensor,
    _validate_modality,
)


@dataclass(frozen=True)
class HashHeadVariantSpec:
    """One isolated compact hash-head change."""

    name: str
    kind: str
    residual_bottleneck: int = 32
    residual_gate_initial: float = 0.02

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variant name must be nonempty")
        if self.kind not in {
            "linear",
            "batchnorm",
            "layernorm",
            "modality_batchnorm_shared_affine",
            "modality_batchnorm_independent",
            "modality_residual",
            "nested_prefix",
        }:
            raise ValueError("unsupported hash-head kind")
        if self.residual_bottleneck < 1:
            raise ValueError("residual_bottleneck must be positive")
        if not math.isfinite(self.residual_gate_initial) or not (
            0.0 < self.residual_gate_initial < 1.0
        ):
            raise ValueError("residual_gate_initial must lie in (0,1)")


HASH_HEAD_VARIANTS: tuple[HashHeadVariantSpec, ...] = (
    HashHeadVariantSpec(name="compact_linear_control", kind="linear"),
    HashHeadVariantSpec(name="compact_batchnorm_hash", kind="batchnorm"),
    HashHeadVariantSpec(name="compact_layernorm_hash", kind="layernorm"),
    HashHeadVariantSpec(
        name="compact_modality_residual_g002",
        kind="modality_residual",
        residual_gate_initial=0.02,
    ),
    HashHeadVariantSpec(
        name="compact_modality_residual_g005",
        kind="modality_residual",
        residual_gate_initial=0.05,
    ),
    HashHeadVariantSpec(name="compact_nested_prefix_hash", kind="nested_prefix"),
)


# Follow-up registry motivated by the direction-asymmetric result of the
# shared-BatchNorm candidate.  It is deliberately separate from the completed
# v1 registry so that the first sweep remains immutable and auditable.
DOMAIN_NORM_VARIANTS: tuple[HashHeadVariantSpec, ...] = (
    HASH_HEAD_VARIANTS[0],
    HashHeadVariantSpec(
        name="compact_modality_batchnorm_shared_affine",
        kind="modality_batchnorm_shared_affine",
    ),
    HashHeadVariantSpec(
        name="compact_modality_batchnorm_independent",
        kind="modality_batchnorm_independent",
    ),
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class _NormalizedHashHead(nn.Module):
    def __init__(self, direct: nn.Linear, *, normalization: str) -> None:
        super().__init__()
        self.direct = direct
        if normalization == "batchnorm":
            self.normalization = nn.BatchNorm1d(direct.out_features)
        elif normalization == "layernorm":
            self.normalization = nn.LayerNorm(direct.out_features)
        else:
            raise ValueError("normalization must be batchnorm or layernorm")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.direct(value))


class _ModalityResidualHashHead(nn.Module):
    """Shared linear coordinate plus a conservatively gated low-rank residual."""

    def __init__(
        self,
        direct: nn.Linear,
        *,
        bottleneck: int,
        gate_initial: float,
    ) -> None:
        super().__init__()
        self.direct = direct
        self.residuals = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(direct.in_features),
                    nn.Linear(direct.in_features, bottleneck, bias=False),
                    nn.SiLU(),
                    nn.Linear(bottleneck, direct.out_features, bias=False),
                )
                for modality in MODALITIES
            }
        )
        self.gate_logits = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.tensor(_logit(gate_initial)))
                for modality in MODALITIES
            }
        )

    def forward(self, value: torch.Tensor, modality: str) -> torch.Tensor:
        modality = _validate_modality(modality)
        gate = torch.sigmoid(self.gate_logits[modality])
        return self.direct(value) + gate * self.residuals[modality](value)


class _ModalityBatchNormHashHead(nn.Module):
    """One shared projection with modality-specific running statistics."""

    def __init__(self, direct: nn.Linear, *, shared_affine: bool) -> None:
        super().__init__()
        self.direct = direct
        self.shared_affine = bool(shared_affine)
        self.normalizations = nn.ModuleDict(
            {
                modality: nn.BatchNorm1d(
                    direct.out_features,
                    affine=not self.shared_affine,
                )
                for modality in MODALITIES
            }
        )
        if self.shared_affine:
            self.affine_weight = nn.Parameter(torch.ones(direct.out_features))
            self.affine_bias = nn.Parameter(torch.zeros(direct.out_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def forward(self, value: torch.Tensor, modality: str) -> torch.Tensor:
        modality = _validate_modality(modality)
        normalized = self.normalizations[modality](self.direct(value))
        if self.shared_affine:
            if self.affine_weight is None or self.affine_bias is None:
                raise AssertionError("shared affine parameters are missing")
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized


class HashHeadRZCSD512(RZCSD512):
    """Compact RZ-CSD with one predeclared hash-head replacement."""

    def __init__(
        self,
        label_dim: int,
        config: RZCSD512Config,
        variant: HashHeadVariantSpec,
    ) -> None:
        super().__init__(label_dim=label_dim, config=config)
        self.hash_head_variant = variant
        if variant.kind in {"batchnorm", "layernorm"}:
            self.hash_heads = nn.ModuleDict(
                {
                    str(bits): _NormalizedHashHead(
                        self.hash_heads[str(bits)], normalization=variant.kind
                    )
                    for bits in BITS
                }
            )
        elif variant.kind in {
            "modality_batchnorm_shared_affine",
            "modality_batchnorm_independent",
        }:
            self.hash_heads = nn.ModuleDict(
                {
                    str(bits): _ModalityBatchNormHashHead(
                        self.hash_heads[str(bits)],
                        shared_affine=(
                            variant.kind == "modality_batchnorm_shared_affine"
                        ),
                    )
                    for bits in BITS
                }
            )
        elif variant.kind == "modality_residual":
            self.hash_heads = nn.ModuleDict(
                {
                    str(bits): _ModalityResidualHashHead(
                        self.hash_heads[str(bits)],
                        bottleneck=variant.residual_bottleneck,
                        gate_initial=variant.residual_gate_initial,
                    )
                    for bits in BITS
                }
            )
        elif variant.kind == "nested_prefix":
            # Keep the already initialized 64-bit projection so upstream and
            # retained weights have the same seeded initialization as control.
            self.nested_hash_head = self.hash_heads[str(max(BITS))]
            self.hash_heads = nn.ModuleDict()

    def _code_logits(
        self,
        representation: torch.Tensor,
        modality: str,
    ) -> dict[int, torch.Tensor]:
        kind = self.hash_head_variant.kind
        if kind == "nested_prefix":
            full = self.nested_hash_head(representation)
            return {bits: full[:, :bits] for bits in BITS}
        if kind in {
            "modality_residual",
            "modality_batchnorm_shared_affine",
            "modality_batchnorm_independent",
        }:
            return {
                bits: self.hash_heads[str(bits)](representation, modality)
                for bits in BITS
            }
        return {
            bits: self.hash_heads[str(bits)](representation)
            for bits in BITS
        }

    def forward(self, features: torch.Tensor, modality: str) -> RZCSDTensorOutput:
        if self.hash_head_variant.kind == "linear":
            return super().forward(features, modality)
        _validate_feature_tensor(features)
        modality = _validate_modality(modality)
        adapted = self.adapters[modality](features)
        representation = self.shared_trunk(adapted)
        embedding = F.normalize(representation, dim=1, eps=1.0e-8)
        continuous = {
            bits: torch.tanh(value)
            for bits, value in self._code_logits(representation, modality).items()
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
        posterior = torch.sigmoid(
            logits - self.posterior_logit_offset[None, None]
        )
        return RZCSDTensorOutput(
            embedding=embedding,
            continuous_codes=continuous,
            binary_codes=binary,
            posterior_logits=logits,
            posterior_heads=posterior,
        )


__all__ = [
    "DOMAIN_NORM_VARIANTS",
    "HASH_HEAD_VARIANTS",
    "HashHeadRZCSD512",
    "HashHeadVariantSpec",
]
