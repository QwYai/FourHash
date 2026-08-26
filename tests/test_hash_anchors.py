from dataclasses import replace

import numpy as np
import pytest
import torch

from raw_rebuilt_neural.hash_anchors import (
    HASH_ANCHOR_SPECS,
    AnchoredHashRZCSD512,
    fit_clip_pca_anchor,
)
from rz_csd_clip512 import BITS, FROZEN_CONFIG


def _features(rows: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    image = rng.normal(size=(rows, 512)).astype(np.float32)
    text = (image + 0.1 * rng.normal(size=image.shape)).astype(np.float32)
    return image, text


def test_clip_pca_anchor_is_deterministic_and_canonical() -> None:
    image, text = _features()
    first = fit_clip_pca_anchor(image, text)
    second = fit_clip_pca_anchor(image, text)
    for key in ("center", "projection", "scale", "eigenvalues"):
        assert np.array_equal(first[key], second[key])
    assert first["projection"].shape == (512, 64)
    assert np.all(first["scale"] > 0.0)


def test_all_anchor_variants_preserve_hash_contract() -> None:
    image, text = _features()
    state = fit_clip_pca_anchor(image, text)
    config = replace(
        FROZEN_CONFIG,
        hidden_dim=32,
        feedforward_dim=64,
        residual_layers=1,
        posterior_hidden_dim=16,
        posterior_heads=3,
        dropout=0.0,
    )
    value = torch.from_numpy(image[:4])
    for spec in HASH_ANCHOR_SPECS:
        model = AnchoredHashRZCSD512(label_dim=5, config=config, anchor_spec=spec)
        if spec.clip_pca:
            model.bind_clip_pca_anchor(state)
        output = model(value, "image")
        assert output.posterior_logits.shape == (4, 3, 5)
        for bits in BITS:
            assert output.continuous_codes[bits].shape == (4, bits)
            assert output.binary_codes[bits].dtype == torch.int8


def test_clip_variant_fails_closed_before_anchor_binding() -> None:
    image, _ = _features()
    model = AnchoredHashRZCSD512(
        label_dim=5, config=FROZEN_CONFIG, anchor_spec=HASH_ANCHOR_SPECS[0]
    )
    with pytest.raises(RuntimeError, match="bind_clip_pca_anchor"):
        model(torch.from_numpy(image[:2]), "text")

