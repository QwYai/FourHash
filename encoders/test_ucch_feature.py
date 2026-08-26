"""Focused audit tests for the leakage-controlled UCCH-F adaptation."""

from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ENCODER_DIR = Path(__file__).resolve().parent
if str(ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(ENCODER_DIR))

from ucch_feature import (  # noqa: E402
    CrossModalRankingLoss,
    MomentumHashMemory,
    UCCHConfig,
    UCCHFeatureNet,
    _synthetic_paired_features,
    encode_all,
    nce_softmax_loss,
    pairing_diagnostics,
    train_ucch_f,
    ucch_f_quality_gate,
)


class ObjectiveEquivalenceTest(unittest.TestCase):
    def test_nce_is_the_stable_official_expression(self) -> None:
        generator = torch.Generator().manual_seed(31)
        logits = torch.randn(9, 17, generator=generator)
        official = -torch.log(torch.softmax(logits, dim=1)[:, 0]).mean()
        stable = nce_softmax_loss(logits)
        self.assertTrue(torch.allclose(stable, official, atol=1e-6, rtol=1e-6))

    def test_ranking_is_the_stable_official_expression(self) -> None:
        generator = torch.Generator().manual_seed(43)
        image = torch.nn.functional.normalize(
            torch.randn(11, 16, generator=generator), dim=1
        )
        text = torch.nn.functional.normalize(
            torch.randn(11, 16, generator=generator), dim=1
        )
        margin, shift = 0.2, 0.1
        scores = image @ text.t()
        diagonal = scores.diag().reshape(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        mask_s = (scores >= d1 - margin).float().detach()
        cost_s = scores * mask_s + (1.0 - mask_s) * (scores - shift)
        mask_i = (scores >= d2 - margin).float().detach()
        cost_i = scores * mask_i + (1.0 - mask_i) * (scores - shift)
        official = (
            -cost_s.diag() + cost_s.exp().sum(1).log() + margin
        ).mean() + (
            -cost_i.diag() + cost_i.exp().sum(0).log() + margin
        ).mean()
        stable = CrossModalRankingLoss(margin, shift)(image, text)
        self.assertTrue(torch.allclose(stable, official, atol=1e-6, rtol=1e-6))


class ArchitectureAndLeakageTest(unittest.TestCase):
    def test_official_feature_mode_depths(self) -> None:
        image = UCCHFeatureNet(512, 64, layers=3, hidden_width=128)
        text = UCCHFeatureNet(512, 64, layers=2, hidden_width=128)
        self.assertEqual(sum(isinstance(x, torch.nn.Linear) for x in image.fc), 3)
        self.assertEqual(sum(isinstance(x, torch.nn.Linear) for x in text.fc), 2)
        output = image(torch.randn(7, 512))
        self.assertEqual(tuple(output.shape), (7, 64))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(7))

    def test_optimizer_api_has_no_label_argument(self) -> None:
        parameters = inspect.signature(train_ucch_f).parameters
        self.assertEqual(
            list(parameters), ["train_image", "train_text", "config", "verbose"]
        )
        self.assertFalse(any("label" in name.lower() for name in parameters))

    def test_memory_positive_is_candidate_zero_and_updates(self) -> None:
        torch.manual_seed(5)
        memory = MomentumHashMemory(8, 20, 7, 0.9, 0.4)
        before = memory.memory.clone()
        image = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        text = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        index = torch.tensor([2, 4, 6, 8])
        generator = torch.Generator().manual_seed(9)
        image_logits, text_logits, candidates = memory(
            image, text, index, use_memory=True, sampling_generator=generator
        )
        self.assertEqual(tuple(image_logits.shape), (4, 8))
        self.assertEqual(tuple(text_logits.shape), (4, 8))
        self.assertEqual(candidates, 8)
        self.assertGreater(float(torch.linalg.vector_norm(memory.memory - before)), 0.0)


class MinimalTrainingTest(unittest.TestCase):
    def test_two_epoch_training_changes_both_branches_and_exports_codes(self) -> None:
        image, text = _synthetic_paired_features(20260805, n_rows=96)
        config = UCCHConfig(
            bits=16,
            epochs=2,
            batch_size=32,
            image_layers=2,
            text_layers=2,
            hidden_width=48,
            lr=5e-4,
            negatives=17,
            seed=20260805,
            device="cpu",
        )
        result = train_ucch_f(image, text, config, verbose=False)
        self.assertGreater(result.image_parameter_delta, 0.0)
        self.assertGreater(result.text_parameter_delta, 0.0)
        self.assertGreater(result.memory_delta, 0.0)
        self.assertEqual(int(result.history[0]["candidate_count"]), 32)
        self.assertEqual(int(result.history[1]["candidate_count"]), 18)
        image_codes = encode_all(result.image_model, image, 64, result.device)
        text_codes = encode_all(result.text_model, text, 64, result.device)
        self.assertEqual(image_codes.dtype, np.int8)
        self.assertEqual(text_codes.dtype, np.int8)
        self.assertTrue(set(np.unique(image_codes)).issubset({-1, 1}))
        diagnostics = pairing_diagnostics(image_codes, text_codes, result)
        self.assertTrue(math.isfinite(diagnostics["paired_gap_mismatched_minus_matched"]))
        self.assertTrue(ucch_f_quality_gate(diagnostics)["passed"])


if __name__ == "__main__":
    unittest.main()
