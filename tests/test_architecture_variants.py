from dataclasses import replace

import pytest
import torch

from raw_rebuilt_neural.architecture_variants import (
    ROUTING_VARIANTS,
    RoutingRZCSD512,
    RoutingVariantSpec,
)
from rz_csd_clip512 import BITS, FROZEN_CONFIG, RZCSD512


def test_routing_variants_preserve_public_tensor_contract() -> None:
    config = replace(
        FROZEN_CONFIG,
        hidden_dim=32,
        feedforward_dim=64,
        residual_layers=1,
        posterior_hidden_dim=16,
        posterior_heads=3,
        dropout=0.0,
    )
    features = torch.randn(5, 512)
    for variant in ROUTING_VARIANTS:
        model = RoutingRZCSD512(label_dim=7, config=config, variant=variant)
        output = model(features, "image")
        assert output.embedding.shape == (5, 32)
        assert output.posterior_logits.shape == (5, 3, 7)
        assert output.posterior_heads.shape == (5, 3, 7)
        assert set(output.continuous_codes) == set(BITS)
        for bits in BITS:
            assert output.continuous_codes[bits].shape == (5, bits)
            assert output.binary_codes[bits].dtype == torch.int8


def test_control_variant_is_exact_base_model_for_same_state() -> None:
    config = replace(FROZEN_CONFIG, dropout=0.0)
    variant = ROUTING_VARIANTS[0]
    assert variant.adapter == "separate"
    assert variant.trunk == "gelu"
    assert variant.hash_head == "linear"
    torch.manual_seed(19)
    base = RZCSD512(label_dim=4, config=config)
    torch.manual_seed(19)
    control = RoutingRZCSD512(label_dim=4, config=config, variant=variant)
    control.load_state_dict(base.state_dict(), strict=False)
    value = torch.randn(3, 512)
    base.eval()
    control.eval()
    expected = base(value, "text")
    actual = control(value, "text")
    assert torch.equal(expected.embedding, actual.embedding)
    for bits in BITS:
        assert torch.equal(expected.continuous_codes[bits], actual.continuous_codes[bits])


def test_variant_validation_rejects_unregistered_choices() -> None:
    with pytest.raises(ValueError, match="unsupported adapter"):
        RoutingVariantSpec(name="bad", adapter="unknown", trunk="gelu")

