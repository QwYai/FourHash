import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import rz_csd_clip512 as csd


def _config(dropout: float = 0.0) -> csd.RZCSD512Config:
    return csd.RZCSD512Config(
        hidden_dim=32,
        feedforward_dim=64,
        residual_layers=1,
        posterior_hidden_dim=16,
        posterior_heads=3,
        dropout=dropout,
        epochs=1,
        batch_size=4,
        inference_batch_size=4,
        active_window=3,
        max_active_candidates=16,
        max_active_fraction=1.0,
    )


def _model(label_dim: int = 6, dropout: float = 0.0) -> csd.RZCSD512:
    torch.manual_seed(4)
    return csd.RZCSD512(label_dim, _config(dropout))


def _bind_prior(model: csd.RZCSD512) -> csd.RZCSD512:
    csd.configure_training_label_prior(
        model, np.eye(model.label_dim, dtype=np.uint8)
    )
    return model


def _features(rows: int = 8) -> np.ndarray:
    return np.random.default_rng(7).normal(size=(rows, 512)).astype(np.float32)


def _decode(*args, **kwargs):
    kwargs.setdefault("max_active_candidates", 256)
    kwargs.setdefault("max_active_fraction", 1.0)
    return csd.decode_rz_local(*args, **kwargs)


def test_all_inference_and_ranking_signatures_are_label_free():
    functions = (
        csd.RZCSD512.forward,
        csd.encode_clip512,
        csd.reference_z_tables,
        csd.rz_mixed_gallery_scores,
        csd.semantic_relation_heads,
        csd.decode_rz_local,
        csd.raw_clip_cosine,
    )
    for function in functions:
        assert all("label" not in name for name in inspect.signature(function).parameters)


def test_model_emits_all_hash_widths_and_deterministic_posterior_heads():
    model = _bind_prior(_model()).eval()
    features = torch.from_numpy(_features(5))
    with torch.no_grad():
        first = model(features, "image")
        second = model(features, "image")
    assert first.embedding.shape == (5, 32)
    assert first.posterior_heads.shape == (5, 3, 6)
    assert torch.equal(first.posterior_heads, second.posterior_heads)
    for bits in csd.BITS:
        assert first.continuous_codes[bits].shape == (5, bits)
        assert first.binary_codes[bits].shape == (5, bits)
        assert set(first.binary_codes[bits].unique().tolist()) <= {-1, 1}


def test_eval_encoding_is_permutation_equivariant_and_duplicate_safe():
    model = _bind_prior(_model()).eval()
    features = _features(7)
    features[5] = features[2]
    permutation = np.asarray([5, 0, 6, 2, 1, 4, 3])
    direct = csd.encode_clip512(
        model, features, modality="text", device=torch.device("cpu"), batch_size=3
    )
    shuffled = csd.encode_clip512(
        model,
        features[permutation],
        modality="text",
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert np.array_equal(shuffled.binary_codes[64], direct.binary_codes[64][permutation])
    assert np.array_equal(shuffled.posterior_heads, direct.posterior_heads[:, permutation])
    assert np.array_equal(direct.binary_codes[16][2], direct.binary_codes[16][5])
    assert np.array_equal(direct.posterior_heads[:, 2], direct.posterior_heads[:, 5])


def test_encoding_is_invariant_to_chunk_size():
    model = _bind_prior(_model()).eval()
    features = _features(9)
    one = csd.encode_clip512(
        model, features, modality="image", device=torch.device("cpu"), batch_size=1
    )
    many = csd.encode_clip512(
        model, features, modality="image", device=torch.device("cpu"), batch_size=9
    )
    assert np.array_equal(one.binary_codes[32], many.binary_codes[32])
    # Unique/scatter plus one frozen padded microbatch makes even the float32
    # posterior byte-exact under a different caller chunk request.
    assert np.array_equal(one.continuous_codes[64], many.continuous_codes[64])
    assert np.array_equal(one.posterior_heads, many.posterior_heads)


def test_complete_training_objective_is_finite_and_differentiable():
    model = _model(dropout=0.1).train()
    image = torch.from_numpy(_features(6))
    text = torch.from_numpy(np.roll(_features(6), 1, axis=1).copy())
    labels = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    positive_weight = csd.configure_training_label_prior(model, labels)
    losses = csd.compute_training_objective(
        model,
        image,
        text,
        labels,
        np.arange(6, dtype=np.int64),
        positive_weight,
    )
    assert set(losses) == {
        "total",
        "bce",
        "alignment",
        "ranking",
        "quantization",
        "balance",
        "decorrelation",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_prior_weight_offset_is_removed_at_inference():
    model = _model(label_dim=3).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.posterior_logit_offset.fill_(np.log(4.0))
        posterior = model(torch.from_numpy(_features(2)), "image").posterior_heads
    assert torch.allclose(posterior, torch.full_like(posterior, 0.2), atol=1e-6)


def test_relation_heads_are_bounded_and_semantically_ordered():
    query = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]], dtype=np.float32)
    candidates = np.asarray(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.85, 0.15], [0.15, 0.85]],
        ],
        dtype=np.float32,
    )
    relevance, jaccard = csd.semantic_relation_heads(query, candidates)
    assert relevance.shape == jaccard.shape == (3, 2)
    assert np.all((0.0 <= relevance) & (relevance <= 1.0))
    assert np.all((0.0 <= jaccard) & (jaccard <= 1.0))
    assert np.all(relevance[:, 0] > relevance[:, 1])
    assert np.all(jaccard[:, 0] > jaccard[:, 1])


def test_reference_z_scores_are_gallery_permutation_equivariant():
    rng = np.random.default_rng(10)
    query = rng.choice((-1, 1), size=16).astype(np.int8)
    gallery_image = rng.choice((-1, 1), size=(13, 16)).astype(np.int8)
    gallery_text = rng.choice((-1, 1), size=(13, 16)).astype(np.int8)
    bank_image = rng.choice((-1, 1), size=(23, 16)).astype(np.int8)
    bank_text = rng.choice((-1, 1), size=(23, 16)).astype(np.int8)
    mask = np.arange(13) % 3 == 0
    permutation = rng.permutation(13)
    direct = csd.rz_mixed_gallery_scores(
        query,
        gallery_image,
        gallery_text,
        mask,
        bank_image_codes=bank_image,
        bank_text_codes=bank_text,
    )
    shuffled = csd.rz_mixed_gallery_scores(
        query,
        gallery_image[permutation],
        gallery_text[permutation],
        mask[permutation],
        bank_image_codes=bank_image,
        bank_text_codes=bank_text,
    )
    assert np.array_equal(shuffled, direct[permutation])


def test_active_prefix_closes_exact_boundary_ties_and_interval_neighbors():
    scores = np.asarray([5, 4, 4, 4, 3, 2], dtype=np.float32)
    ids = np.arange(6, dtype=np.int64)
    active, order = csd.tie_closed_uncertainty_prefix(
        scores, ids, window_size=2, max_active_fraction=1.0
    )
    assert np.array_equal(np.flatnonzero(active), [0, 1, 2, 3])
    assert np.array_equal(order[:4], [0, 1, 2, 3])

    scores = np.asarray([5, 4, 3, 2], dtype=np.float32)
    lower = np.asarray([5, 3.5, 2.9, 2.0], dtype=np.float32)
    upper = np.asarray([5, 4.5, 3.6, 2.0], dtype=np.float32)
    active, _ = csd.tie_closed_uncertainty_prefix(
        scores,
        np.arange(4),
        window_size=2,
        rz_lower=lower,
        rz_upper=upper,
        max_active_fraction=1.0,
    )
    assert np.array_equal(np.flatnonzero(active), [0, 1, 2])


def test_ap_precedence_cannot_be_reversed_by_graded_score_and_tail_is_exact():
    scores = np.asarray([3, 3, 3, 2, 1], dtype=np.float32)
    ids = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    relevance = np.asarray(
        [
            [0.90, 0.65, 0.35, 0.2, 0.1],
            [0.85, 0.60, 0.30, 0.2, 0.1],
            [0.95, 0.70, 0.40, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    # Candidate 2 has the largest graded score but a relevance interval wholly
    # below candidate 0, so it cannot pass candidate 0.
    jaccard = np.asarray([[0.1, 0.5, 0.99, 0.2, 0.1]] * 3, dtype=np.float32)
    result = _decode(
        scores,
        ids,
        relevance,
        jaccard,
        window_size=2,
        use_uncertainty=True,
        use_graded=True,
    )
    ordered_ids = ids[result.order].tolist()
    assert ordered_ids.index(10) < ordered_ids.index(12)
    assert result.active_size == 3
    assert np.array_equal(result.order[3:], result.rz_order[3:])


def test_local_decoder_is_permutation_invariant_by_immutable_candidate_id():
    rng = np.random.default_rng(13)
    scores = np.asarray([4, 4, 4, 3, 2, 1], dtype=np.float32)
    ids = np.asarray([101, 205, 309, 410, 511, 612], dtype=np.int64)
    relevance = rng.uniform(0.1, 0.9, size=(3, 6)).astype(np.float32)
    jaccard = rng.uniform(0.1, 0.9, size=(3, 6)).astype(np.float32)
    direct = _decode(
        scores, ids, relevance, jaccard, window_size=2
    )
    permutation = np.asarray([4, 0, 5, 2, 1, 3])
    shuffled = _decode(
        scores[permutation],
        ids[permutation],
        relevance[:, permutation],
        jaccard[:, permutation],
        window_size=2,
    )
    assert np.array_equal(ids[direct.order], ids[permutation][shuffled.order])


def test_rank_group_keys_keep_identical_active_predictions_and_inactive_rz_ties():
    scores = np.asarray([4, 4, 3, 2, 2], dtype=np.float32)
    ids = np.asarray([50, 10, 40, 20, 30], dtype=np.int64)
    relevance = np.asarray([[0.8, 0.8, 0.4, 0.2, 0.1]] * 3, dtype=np.float32)
    jaccard = np.asarray([[0.7, 0.7, 0.4, 0.2, 0.1]] * 3, dtype=np.float32)
    result = _decode(
        scores, ids, relevance, jaccard, window_size=1
    )
    assert result.active_size == 2
    assert result.rank_group_keys[0] == result.rank_group_keys[1]
    assert result.rank_group_keys[3] == result.rank_group_keys[4]
    assert result.rank_group_keys[2] > result.rank_group_keys[3]


def test_ablation_registry_and_same_input_controls_are_executable():
    assert set(csd.ablation_registry()) == {
        "raw_clip",
        "linear",
        "capacity_mlp",
        "raw_hamming",
        "rz",
        "rz_relevance",
        "no_rz",
        "no_graded",
        "no_uncertainty",
        "full",
    }
    features = torch.from_numpy(_features(3))
    assert csd.SameInputPosteriorControl(6, kind="linear")(features).shape == (3, 6)
    assert csd.SameInputPosteriorControl(6, kind="mlp", hidden_dim=12)(features).shape == (3, 6)
    cosine = csd.raw_clip_cosine(features[0].numpy(), features.numpy())
    assert cosine.shape == (3,)
    assert cosine[0] == pytest.approx(1.0, abs=1e-5)


def test_frozen_json_uses_prepared_clip_route_label_dimensions():
    payload = json.loads(
        (Path(__file__).parent / "rz_csd_clip512_config.json").read_text(encoding="utf-8")
    )
    geometry = payload["dataset_geometry"]
    assert geometry["mirflickr"]["label_dim"] == 24
    assert geometry["nuswide"]["label_dim"] == 81
    assert geometry["mscoco"]["label_dim"] == 80
    assert geometry["nuswide"]["train_rows"] == 21000


def test_last_chunk_duplicate_and_permutation_are_byte_exact():
    model = _bind_prior(_model()).eval()
    features = _features(10)
    features[9] = features[0]  # duplicate scatters from a padded final unique chunk
    direct = csd.encode_clip512(
        model, features, modality="image", device=torch.device("cpu"), batch_size=2
    )
    permutation = np.asarray([9, 3, 5, 1, 8, 0, 6, 2, 7, 4])
    shuffled = csd.encode_clip512(
        model,
        features[permutation],
        modality="image",
        device=torch.device("cpu"),
        batch_size=9,
    )
    assert direct.posterior_heads.tobytes() == direct.posterior_heads.copy().tobytes()
    assert np.array_equal(direct.posterior_heads[:, 0], direct.posterior_heads[:, 9])
    assert np.array_equal(
        shuffled.posterior_heads, direct.posterior_heads[:, permutation]
    )
    assert np.array_equal(
        shuffled.continuous_codes[64], direct.continuous_codes[64][permutation]
    )


def test_zero_positive_class_and_prior_offset_mismatch_fail_closed():
    model = _model(label_dim=3)
    zero_class = np.asarray(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.uint8
    )
    with pytest.raises(ValueError, match="zero-positive"):
        csd.configure_training_label_prior(model, zero_class)

    labels = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32
    )
    weight = csd.configure_training_label_prior(model, labels)
    assert bool(model.posterior_prior_is_bound.item())
    wrong = weight.clone()
    wrong[0] += 0.5
    feature = torch.from_numpy(_features(4))
    with pytest.raises(RuntimeError, match="bound logit offset"):
        csd.compute_training_objective(
            model,
            feature,
            feature.clone(),
            labels,
            np.arange(4, dtype=np.int64),
            wrong,
        )

    with pytest.raises(RuntimeError, match="posterior prior is bound"):
        csd.encode_clip512(
            _model(label_dim=3),
            _features(2),
            modality="image",
            device=torch.device("cpu"),
        )


def test_deterministic_subbag_is_identity_keyed_not_row_ordered():
    ids = np.asarray([20, 3, 77, 11, 9], dtype=np.int64)
    direct = csd.deterministic_head_subbag_mask(
        ids, heads=5, keep_probability=0.6, seed=19
    )
    permutation = np.asarray([3, 0, 4, 2, 1])
    shuffled = csd.deterministic_head_subbag_mask(
        ids[permutation], heads=5, keep_probability=0.6, seed=19
    )
    assert np.array_equal(shuffled, direct[permutation])
    assert direct.any(axis=1).all()


def test_post_training_head_diversity_gate_rejects_collapse():
    collapsed = np.full((5, 8, 4), 0.5, dtype=np.float32)
    report = csd.posterior_head_diversity_report(collapsed)
    assert not report.passed
    with pytest.raises(RuntimeError, match="heads collapsed"):
        csd.require_posterior_head_diversity(collapsed)

    base = np.linspace(0.1, 0.7, 32, dtype=np.float64).reshape(8, 4)
    diverse = np.stack([base + 0.02 * head for head in range(5)])
    passed = csd.require_posterior_head_diversity(
        diverse, minimum_pairwise_mad=0.01, minimum_cell_std=0.005
    )
    assert passed.passed


@pytest.mark.parametrize("label_dim", [24, 81, 80])
def test_capacity_matched_mlp_is_within_half_percent(label_dim):
    full = csd.RZCSD512(label_dim)
    control, report = csd.build_capacity_matched_mlp_control(full)
    assert control.label_dim == label_dim
    assert report["relative_gap"] <= 0.005
    assert report["control_parameters"] == csd.parameter_count(control)


def test_canonical_semantic_evidence_and_ranking_ignore_head_order():
    rng = np.random.default_rng(31)
    scores = np.ones(7, dtype=np.float64)
    ids = np.asarray([70, 10, 50, 20, 60, 30, 40], dtype=np.int64)
    relevance = rng.uniform(0.05, 0.95, size=(5, 7))
    jaccard = rng.uniform(0.05, 0.95, size=(5, 7))
    head_permutation = np.asarray([3, 0, 4, 1, 2])
    assert csd.canonicalize_relation_evidence(relevance, jaccard) == (
        csd.canonicalize_relation_evidence(
            relevance[head_permutation], jaccard[head_permutation]
        )
    )
    direct = _decode(scores, ids, relevance, jaccard, window_size=3)
    permuted_heads = _decode(
        scores,
        ids,
        relevance[head_permutation],
        jaccard[head_permutation],
        window_size=3,
    )
    assert np.array_equal(direct.rank_group_keys, permuted_heads.rank_group_keys)
    assert np.array_equal(direct.order, permuted_heads.order)


def test_no_graded_ablation_is_exactly_independent_of_jaccard_values():
    rng = np.random.default_rng(32)
    scores = np.ones(6, dtype=np.float64)
    ids = np.arange(100, 106, dtype=np.int64)
    relevance = rng.uniform(0.05, 0.95, size=(5, 6))
    first_jaccard = rng.uniform(0.01, 0.99, size=(5, 6))
    second_jaccard = 1.0 - first_jaccard
    first = _decode(
        scores,
        ids,
        relevance,
        first_jaccard,
        window_size=3,
        use_graded=False,
    )
    second = _decode(
        scores,
        ids,
        relevance,
        second_jaccard,
        window_size=3,
        use_graded=False,
    )
    assert np.array_equal(first.rank_group_keys, second.rank_group_keys)
    assert np.array_equal(first.order, second.order)


def test_no_uncertainty_ablation_depends_only_on_head_means():
    rng = np.random.default_rng(33)
    scores = np.ones(6, dtype=np.float64)
    ids = np.arange(200, 206, dtype=np.int64)
    relevance = rng.uniform(0.05, 0.95, size=(5, 6))
    jaccard = rng.uniform(0.05, 0.95, size=(5, 6))
    collapsed_relevance = np.broadcast_to(relevance.mean(axis=0), relevance.shape)
    collapsed_jaccard = np.broadcast_to(jaccard.mean(axis=0), jaccard.shape)
    first = _decode(
        scores,
        ids,
        relevance,
        jaccard,
        window_size=3,
        use_uncertainty=False,
    )
    second = _decode(
        scores,
        ids,
        collapsed_relevance,
        collapsed_jaccard,
        window_size=3,
        use_uncertainty=False,
    )
    assert np.array_equal(first.rank_group_keys, second.rank_group_keys)
    assert np.array_equal(first.order, second.order)


def test_hierarchical_poset_resolves_union_cycle_with_rz_priority():
    # Naively unioning these hard relations creates A->B (RZ), B->C (AP),
    # C->A (AP).  The hierarchical algorithm keeps RZ first, then applies AP
    # only inside the current acyclic Kahn frontier.
    scores = np.asarray([2.5, 0.5, 1.5], dtype=np.float64)  # A, B, C
    lower = np.asarray([2.0, 0.0, 0.5], dtype=np.float64)
    upper = np.asarray([3.0, 1.0, 2.5], dtype=np.float64)
    relevance = np.asarray([[0.1, 0.9, 0.5]] * 3, dtype=np.float64)
    jaccard = np.asarray([[0.1, 0.9, 0.5]] * 3, dtype=np.float64)
    result = _decode(
        scores,
        np.asarray([101, 102, 103]),
        relevance,
        jaccard,
        window_size=3,
        rz_lower=lower,
        rz_upper=upper,
    )
    assert result.rz_hard_edges == 1
    assert result.rank_group_keys[2] > result.rank_group_keys[0]
    assert result.rank_group_keys[0] > result.rank_group_keys[1]


def test_2000_random_interval_hard_edge_permutation_and_tie_cases():
    rng = np.random.default_rng(20260821)
    for iteration in range(2000):
        rows = int(rng.integers(3, 11))
        scores = rng.normal(size=rows)
        half_width = rng.uniform(0.0, 0.8, size=rows)
        lower = scores - half_width
        upper = scores + half_width
        ids = 10_000 + rng.permutation(rows).astype(np.int64)
        relevance = rng.uniform(0.01, 0.99, size=(5, rows))
        jaccard = rng.uniform(0.01, 0.99, size=(5, rows))
        if iteration % 2 == 0:
            scores[-1] = scores[0]
            lower[-1] = lower[0]
            upper[-1] = upper[0]
            relevance[:, -1] = relevance[:, 0]
            jaccard[:, -1] = jaccard[:, 0]
        result = _decode(
            scores,
            ids,
            relevance,
            jaccard,
            window_size=rows,
            rz_lower=lower,
            rz_upper=upper,
        )
        assert np.array_equal(np.sort(result.order), np.arange(rows))
        if iteration % 2 == 0:
            assert result.rank_group_keys[0] == result.rank_group_keys[-1]
        for left in range(rows):
            for right in range(rows):
                if lower[left] > upper[right]:
                    assert result.rank_group_keys[left] > result.rank_group_keys[right]
        candidate_permutation = rng.permutation(rows)
        head_permutation = rng.permutation(5)
        shuffled = _decode(
            scores[candidate_permutation],
            ids[candidate_permutation],
            relevance[head_permutation][:, candidate_permutation],
            jaccard[head_permutation][:, candidate_permutation],
            window_size=rows,
            rz_lower=lower[candidate_permutation],
            rz_upper=upper[candidate_permutation],
        )
        assert np.array_equal(
            shuffled.rank_group_keys,
            result.rank_group_keys[candidate_permutation],
        )
        assert np.array_equal(
            ids[candidate_permutation][shuffled.order], ids[result.order]
        )


def test_partial_order_operation_count_scales_quadratically_not_cubically():
    counts = []
    for rows in (128, 256, 512):
        # All rows enter one RZ frontier.  Strict AP intervals create singleton
        # layers, the worst case for the old repeated frontier scan.
        scores = np.ones(rows, dtype=np.float64)
        values = np.linspace(0.99, 0.01, rows, dtype=np.float64)
        relevance = np.broadcast_to(values, (5, rows)).copy()
        jaccard = np.broadcast_to(values[::-1], (5, rows)).copy()
        result = csd.decode_rz_local(
            scores,
            np.arange(rows, dtype=np.int64),
            relevance,
            jaccard,
            window_size=rows,
            max_active_candidates=rows,
            max_active_fraction=1.0,
        )
        diagnostics = result.partial_order_diagnostics
        assert diagnostics.rz_hard_edges == 0
        assert diagnostics.potential_ap_hard_edges == rows * (rows - 1) // 2
        assert diagnostics.operation_count <= 6 * rows * rows
        counts.append(diagnostics.operation_count)
    assert counts[1] / counts[0] < 4.05
    assert counts[2] / counts[1] < 4.05


def test_active_set_growth_gate_prevents_quadratic_full_gallery_fallback():
    scores = np.ones(30, dtype=np.float64)
    ids = np.arange(30, dtype=np.int64)
    with pytest.raises(RuntimeError, match="exceeds registered limit"):
        csd.tie_closed_uncertainty_prefix(
            scores,
            ids,
            window_size=2,
            max_active_candidates=5,
            max_active_fraction=1.0,
        )


def test_indt_contract_accepts_only_aligned_train_slice():
    image = _features(4)
    text = np.roll(image, 1, axis=1).copy()
    labels = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8
    )
    report = csd.validate_indt_training_inputs(
        image,
        text,
        labels,
        np.asarray([9, 2, 7, 4], dtype=np.int64),
        expected_rows=4,
        label_dim=3,
    )
    assert report["scope"] == "prepared_indT_only"
    assert report["query_or_database_labels_opened"] is False


def test_relation_and_rz_score_contract_is_float64():
    query = np.asarray([[0.8, 0.2]] * 3, dtype=np.float32)
    candidate = np.asarray([[[0.7, 0.3], [0.2, 0.8]]] * 3, dtype=np.float32)
    relevance, jaccard = csd.semantic_relation_heads(query, candidate)
    assert relevance.dtype == jaccard.dtype == np.float64

    rng = np.random.default_rng(44)
    query_code = rng.choice((-1, 1), size=16).astype(np.int8)
    bank_image = rng.choice((-1, 1), size=(20, 16)).astype(np.int8)
    bank_text = rng.choice((-1, 1), size=(20, 16)).astype(np.int8)
    gallery_image = rng.choice((-1, 1), size=(7, 16)).astype(np.int8)
    gallery_text = rng.choice((-1, 1), size=(7, 16)).astype(np.int8)
    score = csd.rz_mixed_gallery_scores(
        query_code,
        gallery_image,
        gallery_text,
        np.arange(7) % 2 == 0,
        bank_image_codes=bank_image,
        bank_text_codes=bank_text,
    )
    assert score.dtype == np.float64
