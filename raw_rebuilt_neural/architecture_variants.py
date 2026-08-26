"""Development-only neural routing variants for the RZ-CSD encoder.

The variants keep the frozen loss, curriculum, input contract, posterior
ensemble, and inference outputs unchanged.  They isolate two architectural
questions on the sealed training/development split:

1. Does a shared CLIP-space anchor preserve the common image/text coordinate
   better than two independent adapters?
2. Can a small gated modality residual and a SwiGLU residual trunk recover
   modality-specific detail without destroying that common coordinate?

No query/database labels are accepted or used here.
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
class RoutingVariantSpec:
    """One predeclared routing ablation with compact backbone dimensions."""

    name: str
    adapter: str
    trunk: str
    hash_head: str = "linear"
    modality_bottleneck: int = 64
    modality_gate_initial: float = 0.10
    layer_scale_initial: float = 0.10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variant name must be nonempty")
        if self.adapter not in {"separate", "shared_anchor", "shared_gated"}:
            raise ValueError("unsupported adapter variant")
        if self.trunk not in {"gelu", "swiglu_layerscale"}:
            raise ValueError("unsupported trunk variant")
        if self.hash_head not in {"linear", "gated_residual"}:
            raise ValueError("unsupported hash-head variant")
        if self.modality_bottleneck < 1:
            raise ValueError("modality_bottleneck must be positive")
        for value, label in (
            (self.modality_gate_initial, "modality_gate_initial"),
            (self.layer_scale_initial, "layer_scale_initial"),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{label} must lie in (0,1)")


ROUTING_VARIANTS: tuple[RoutingVariantSpec, ...] = (
    RoutingVariantSpec(
        name="separate_gelu_linear_control",
        adapter="separate",
        trunk="gelu",
    ),
    RoutingVariantSpec(
        name="shared_anchor_gelu_linear",
        adapter="shared_anchor",
        trunk="gelu",
    ),
    RoutingVariantSpec(
        name="shared_gated_gelu_linear",
        adapter="shared_gated",
        trunk="gelu",
    ),
    RoutingVariantSpec(
        name="shared_gated_swiglu_linear",
        adapter="shared_gated",
        trunk="swiglu_layerscale",
    ),
    RoutingVariantSpec(
        name="shared_gated_swiglu_residual_hash",
        adapter="shared_gated",
        trunk="swiglu_layerscale",
        hash_head="gated_residual",
    ),
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class SharedSemanticAnchor(nn.Module):
    """Common projection plus optional low-rank gated modality correction."""

    def __init__(
        self,
        config: RZCSD512Config,
        *,
        gated_residual: bool,
        bottleneck: int,
        gate_initial: float,
    ) -> None:
        super().__init__()
        self.gated_residual = bool(gated_residual)
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.anchor = nn.Linear(config.feature_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        if self.gated_residual:
            self.residuals = nn.ModuleDict(
                {
                    modality: nn.Sequential(
                        nn.Linear(config.feature_dim, bottleneck, bias=False),
                        nn.SiLU(),
                        nn.Linear(bottleneck, config.hidden_dim, bias=False),
                    )
                    for modality in MODALITIES
                }
            )
            initial = torch.full(
                (config.hidden_dim,), _logit(gate_initial), dtype=torch.float32
            )
            self.gate_logits = nn.ParameterDict(
                {
                    modality: nn.Parameter(initial.clone())
                    for modality in MODALITIES
                }
            )
        else:
            self.residuals = nn.ModuleDict()
            self.gate_logits = nn.ParameterDict()

    def forward(self, value: torch.Tensor, modality: str) -> torch.Tensor:
        modality = _validate_modality(modality)
        normalized = self.input_norm(value)
        anchor = F.gelu(self.anchor(normalized))
        if not self.gated_residual:
            return self.output_norm(anchor)
        residual = self.dropout(self.residuals[modality](normalized))
        gate = torch.sigmoid(self.gate_logits[modality])
        return self.output_norm(anchor + gate * residual)


class SwiGLULayerScaleBlock(nn.Module):
    """Pre-normalized SwiGLU residual block with a stable learned layer scale."""

    def __init__(
        self,
        dim: int,
        feedforward_dim: int,
        dropout: float,
        layer_scale_initial: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 2 * feedforward_dim)
        self.down = nn.Linear(feedforward_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(
            torch.full((dim,), float(layer_scale_initial), dtype=torch.float32)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.up(self.norm(value)).chunk(2, dim=-1)
        update = self.down(F.silu(gate) * content)
        return value + self.layer_scale * self.dropout(update)


class GatedResidualHashHead(nn.Module):
    """Linear hash projection plus a conservatively gated nonlinear residual."""

    def __init__(self, hidden_dim: int, bits: int, dropout: float) -> None:
        super().__init__()
        bottleneck = max(bits * 2, hidden_dim // 2)
        self.direct = nn.Linear(hidden_dim, bits)
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, bits),
        )
        self.gate_logit = nn.Parameter(torch.tensor(_logit(0.10)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.direct(value) + torch.sigmoid(self.gate_logit) * self.residual(value)


class RoutingRZCSD512(RZCSD512):
    """RZ-CSD encoder with a predeclared adapter/trunk/hash routing variant."""

    def __init__(
        self,
        label_dim: int,
        config: RZCSD512Config,
        variant: RoutingVariantSpec,
    ) -> None:
        super().__init__(label_dim=label_dim, config=config)
        self.routing_variant = variant
        if variant.adapter != "separate":
            self.semantic_anchor = SharedSemanticAnchor(
                config,
                gated_residual=variant.adapter == "shared_gated",
                bottleneck=variant.modality_bottleneck,
                gate_initial=variant.modality_gate_initial,
            )
        else:
            self.semantic_anchor = None
        if variant.trunk == "swiglu_layerscale":
            self.shared_trunk = nn.Sequential(
                *[
                    SwiGLULayerScaleBlock(
                        config.hidden_dim,
                        config.feedforward_dim,
                        config.dropout,
                        variant.layer_scale_initial,
                    )
                    for _ in range(config.residual_layers)
                ],
                nn.LayerNorm(config.hidden_dim),
            )
        if variant.hash_head == "gated_residual":
            self.hash_heads = nn.ModuleDict(
                {
                    str(bits): GatedResidualHashHead(
                        config.hidden_dim, bits, config.dropout
                    )
                    for bits in BITS
                }
            )

    def forward(self, features: torch.Tensor, modality: str) -> RZCSDTensorOutput:
        _validate_feature_tensor(features)
        modality = _validate_modality(modality)
        if self.semantic_anchor is None:
            adapted = self.adapters[modality](features)
        else:
            adapted = self.semantic_anchor(features, modality)
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

