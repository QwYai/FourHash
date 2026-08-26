from dataclasses import replace

import pytest
import torch

from raw_rebuilt_neural.hash_head_variants import (
    DOMAIN_NORM_VARIANTS,
    HASH_HEAD_VARIANTS,
    HashHeadRZCSD512,
    HashHeadVariantSpec,
)
from rz_csd_clip512 import BITS, FROZEN_CONFIG, RZCSD512


def test_hash_head_registry_is_unique_and_control_first() -> None:
    names = [variant.name for variant in HASH_HEAD_VARIANTS]
    assert len(names) == len(set(names)) == 6
    assert HASH_HEAD_VARIANTS[0].kind == "linear"
    domain_names = [variant.name for variant in DOMAIN_NORM_VARIANTS]
    assert len(domain_names) == len(set(domain_names)) == 3
    assert DOMAIN_NORM_VARIANTS[0] is HASH_HEAD_VARIANTS[0]


def test_linear_control_is_exact_base_model() -> None:
    config = replace(FROZEN_CONFIG, dropout=0.0, seed=13)
    torch.manual_seed(7)
    base = RZCSD512(label_dim=5, config=config)
    torch.manual_seed(7)
    control = HashHeadRZCSD512(5, config, HASH_HEAD_VARIANTS[0])
    base.eval()
    control.eval()
    value = torch.randn(4, 512)
    for modality in ("image", "text"):
        left = base(value, modality)
        right = control(value, modality)
        assert torch.equal(left.embedding, right.embedding)
        assert torch.equal(left.posterior_logits, right.posterior_logits)
        for bits in BITS:
            assert torch.equal(left.continuous_codes[bits], right.continuous_codes[bits])
            assert torch.equal(left.binary_codes[bits], right.binary_codes[bits])


@pytest.mark.parametrize(
    "variant",
    (*HASH_HEAD_VARIANTS[1:], *DOMAIN_NORM_VARIANTS[1:]),
)
def test_candidate_shapes_and_binary_contract(variant: HashHeadVariantSpec) -> None:
    config = replace(FROZEN_CONFIG, dropout=0.0)
    model = HashHeadRZCSD512(label_dim=6, config=config, variant=variant).eval()
    value = torch.randn(5, 512)
    for modality in ("image", "text"):
        output = model(value, modality)
        assert output.embedding.shape == (5, config.hidden_dim)
        assert output.posterior_heads.shape == (5, config.posterior_heads, 6)
        for bits in BITS:
            assert output.continuous_codes[bits].shape == (5, bits)
            assert output.binary_codes[bits].shape == (5, bits)
            assert set(output.binary_codes[bits].unique().tolist()) <= {-1, 1}


def test_nested_prefix_codes_are_exact_prefixes() -> None:
    variant = next(item for item in HASH_HEAD_VARIANTS if item.kind == "nested_prefix")
    model = HashHeadRZCSD512(4, FROZEN_CONFIG, variant).eval()
    output = model(torch.randn(3, 512), "image")
    assert torch.equal(output.continuous_codes[16], output.continuous_codes[64][:, :16])
    assert torch.equal(output.continuous_codes[32], output.continuous_codes[64][:, :32])


def test_domain_norm_uses_distinct_running_statistics() -> None:
    variant = DOMAIN_NORM_VARIANTS[1]
    model = HashHeadRZCSD512(4, FROZEN_CONFIG, variant)
    head = model.hash_heads["16"]
    assert head.normalizations["image"] is not head.normalizations["text"]
    model.train()
    model(torch.full((8, 512), 2.0), "image")
    model(torch.full((8, 512), -2.0), "text")
    image_mean = head.normalizations["image"].running_mean
    text_mean = head.normalizations["text"].running_mean
    assert not torch.equal(image_mean, text_mean)


def test_invalid_hash_head_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        HashHeadVariantSpec(name="bad", kind="attention")
