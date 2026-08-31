from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import math

import pytest
from safetensors.torch import save_file
import torch

from basisserve.core.c1_k_refine import GPUExactKeyPageStore
from basisserve.core.c1_k_reverse_shadow import (
    ReverseShadowConfig,
    build_c1_decoder_gram,
    build_c1_page_metadata,
    build_post_rope_k_landmarks,
    build_quest_minmax_landmarks,
    build_routing_page_geometry,
    c1_k_reverse_shadow_block_attention,
    c1_k_reverse_shadow_attention,
    reverse_shadow_quality_statistics,
)
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from evaluation.eval_qwen3_c1_k_reverse_shadow_oracle import evaluate
from evaluation.capture_qwen3_8b_c1_k_refine import _fit_pair_tensors


def _case(
    *,
    batch: int = 2,
    query_heads: int = 4,
    kv_heads: int = 2,
    sequence: int = 7,
    head_dim: int = 4,
    value_rank: int = 3,
    dtype: torch.dtype = torch.float64,
):
    generator = torch.Generator().manual_seed(20260826)
    query = torch.randn(
        batch, query_heads, 1, head_dim, generator=generator, dtype=dtype
    )
    exact_key = torch.randn(
        batch, kv_heads, sequence, head_dim, generator=generator, dtype=dtype
    )
    c1_value = torch.randn(
        batch, kv_heads, sequence, value_rank, generator=generator, dtype=dtype
    )
    return query, exact_key, c1_value


def _config(
    *,
    page_size: int = 3,
    budget: int = 3,
    recent: int = 0,
    landmarks: int = 1,
    selector: str = "mean_landmark",
    quest_support: str = "physical_shared",
    query_head_aggregation: str = "max_head",
) -> ReverseShadowConfig:
    return ReverseShadowConfig(
        page_size=page_size,
        exact_token_budget=budget,
        recent_exact_window=recent,
        landmarks_per_page=landmarks,
        selector=selector,
        landmark_dtype="float32",
        quest_support=quest_support,
        query_head_aggregation=query_head_aggregation,
    )


def _landmarks(
    exact_key: torch.Tensor,
    config: ReverseShadowConfig,
    mask: torch.Tensor | None = None,
):
    return build_post_rope_k_landmarks(
        exact_key,
        page_size=config.page_size,
        landmarks_per_page=config.landmarks_per_page,
        attention_mask=mask,
        landmark_dtype=config.landmark_dtype,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("page_size", 0, "page size"),
        ("exact_token_budget", -1, "budget"),
        ("recent_exact_window", -1, "recent"),
        ("landmarks_per_page", 0, "landmarks"),
        ("selector", "proxy", "selector"),
        ("landmark_dtype", "int8", "dtype"),
        ("quest_support", "group", "QUEST support"),
        ("query_head_aggregation", "sum_head", "query-head aggregation"),
    ],
)
def test_config_validation(field: str, value: object, message: str) -> None:
    values = dict(
        page_size=4,
        exact_token_budget=4,
        recent_exact_window=0,
        landmarks_per_page=1,
        selector="mean_landmark",
        landmark_dtype="float32",
        quest_support="physical_shared",
        query_head_aggregation="max_head",
    )
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ReverseShadowConfig(**values).validate(2)


def test_page_budget_is_page_granular() -> None:
    assert _config(page_size=4, budget=0).page_budget == 0
    assert _config(page_size=4, budget=1).page_budget == 1
    assert _config(page_size=4, budget=5).page_budget == 2


def test_adaptive_budget_geometry_is_page_granular() -> None:
    config = ReverseShadowConfig(
        page_size=64,
        exact_token_budget=1024,
        selector="kq_svd",
        adaptive_max_token_budget=2048,
        adaptive_tail_mass_ratio_threshold=0.25,
    )

    config.validate(128)

    assert config.page_budget == 16
    assert config.adaptive_max_page_budget == 32


@pytest.mark.parametrize(
    ("maximum", "threshold", "message"),
    [
        (2048, None, "set together"),
        (1024, 0.25, "must exceed"),
        (2048, 0.0, "must lie"),
    ],
)
def test_adaptive_budget_rejects_invalid_policy(
    maximum: int | None,
    threshold: float | None,
    message: str,
) -> None:
    config = ReverseShadowConfig(
        page_size=64,
        exact_token_budget=1024,
        selector="kq_svd",
        adaptive_max_token_budget=maximum,
        adaptive_tail_mass_ratio_threshold=threshold,
    )

    with pytest.raises(ValueError, match=message):
        config.validate(128)


def test_capture_can_omit_krefine_fit_pairs_for_heldout_replay() -> None:
    query, key, head = _fit_pair_tensors(
        [],
        [],
        [],
        head_dim=8,
        query_dtype=torch.bfloat16,
        key_dtype=torch.float16,
    )
    assert query.shape == (0, 8)
    assert key.shape == (0, 8)
    assert head.shape == (0,)
    assert query.dtype == torch.bfloat16
    assert key.dtype == torch.float16
    assert head.dtype == torch.long


def test_block_attention_matches_dense_causal_attention_at_full_budget() -> None:
    generator = torch.Generator().manual_seed(20260827)
    query = torch.randn(1, 4, 3, 4, generator=generator, dtype=torch.float64)
    exact_key = torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.float64)
    c1_value = torch.randn(1, 2, 5, 3, generator=generator, dtype=torch.float64)
    config = ReverseShadowConfig(
        page_size=2,
        exact_token_budget=5,
        selector="quest_minmax",
        landmark_dtype="float32",
    )
    observed, queries = c1_k_reverse_shadow_block_attention(
        query,
        exact_key,
        c1_value,
        config,
    )
    head_to_kv = torch.tensor([0, 0, 1, 1])
    expected = []
    prefix = exact_key.shape[2] - query.shape[2]
    for index in range(query.shape[2]):
        visible = prefix + index + 1
        key = exact_key[:, head_to_kv, :visible]
        value = c1_value[:, head_to_kv, :visible]
        scores = torch.einsum(
            "bhd,bhsd->bhs", query[:, :, index], key
        ) / math.sqrt(query.shape[-1])
        probability = torch.softmax(scores, dim=-1)
        expected.append(
            torch.einsum("bhs,bhsv->bhv", probability, value).unsqueeze(2)
        )
    torch.testing.assert_close(
        observed,
        torch.cat(expected, dim=2),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert len(queries) == 3
    assert all(result.statistics["selected_token_fraction"] == 1 for result in queries)


def test_block_attention_rejects_ambiguous_rank_three_mask() -> None:
    query, exact_key, c1_value = _case(batch=1, sequence=5)
    block_query = query.expand(-1, -1, 2, -1).contiguous()
    with pytest.raises(ValueError, match="rank 2 or rank 4"):
        c1_k_reverse_shadow_block_attention(
            block_query,
            exact_key,
            c1_value,
            _config(page_size=2, budget=2, selector="quest_minmax"),
            torch.ones(1, 2, 5, dtype=torch.bool),
        )


def test_vectorized_quest_metadata_and_attention_match_page_reference() -> None:
    query, exact_key, c1_value = _case(batch=1, sequence=7)
    mask = torch.tensor([[True, True, False, True, True, True, True]])
    config = _config(
        page_size=3,
        budget=3,
        selector="quest_minmax",
    )
    reference_landmarks = build_post_rope_k_landmarks(
        exact_key,
        page_size=config.page_size,
        landmarks_per_page=config.landmarks_per_page,
        attention_mask=mask,
        landmark_dtype=config.landmark_dtype,
    )
    vectorized_landmarks = build_quest_minmax_landmarks(
        exact_key,
        page_size=config.page_size,
        landmarks_per_page=config.landmarks_per_page,
        attention_mask=mask,
        landmark_dtype=config.landmark_dtype,
    )
    torch.testing.assert_close(
        vectorized_landmarks.page_mins, reference_landmarks.page_mins
    )
    torch.testing.assert_close(
        vectorized_landmarks.page_maxes, reference_landmarks.page_maxes
    )
    assert torch.equal(
        vectorized_landmarks.page_bounds_valid,
        reference_landmarks.page_bounds_valid,
    )
    reference = c1_k_reverse_shadow_attention(
        query,
        reference_landmarks,
        c1_value,
        config,
        exact_key,
        mask,
    )
    vectorized = c1_k_reverse_shadow_attention(
        query,
        vectorized_landmarks,
        c1_value,
        config,
        exact_key,
        mask,
        vectorized_reference=True,
    )
    assert torch.equal(vectorized.selected_page_mask, reference.selected_page_mask)
    torch.testing.assert_close(
        vectorized.output,
        reference.output,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_subchunk_mean_landmarks_handle_mask_and_short_final_page() -> None:
    key = torch.tensor([[[[1.0], [2.0], [3.0], [4.0], [5.0]]]])
    mask = torch.tensor([[True, False, True, True, True]])
    landmarks = build_post_rope_k_landmarks(
        key,
        page_size=4,
        landmarks_per_page=2,
        attention_mask=mask,
        landmark_dtype="float32",
    )
    assert landmarks.values.shape == (1, 1, 2, 2, 1)
    torch.testing.assert_close(
        landmarks.values[0, 0, :, :, 0],
        torch.tensor([[1.0, 3.5], [5.0, 0.0]]),
    )
    torch.testing.assert_close(
        landmarks.radii[0, 0], torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    )
    assert landmarks.valid[0, 0].tolist() == [[True, True], [True, False]]
    torch.testing.assert_close(
        landmarks.page_mins[0, 0, :, 0], torch.tensor([1.0, 5.0])
    )
    torch.testing.assert_close(
        landmarks.page_maxes[0, 0, :, 0], torch.tensor([4.0, 5.0])
    )
    assert landmarks.page_bounds_valid[0, 0].tolist() == [True, True]


def test_landmark_construction_rejects_head_dependent_validity() -> None:
    key = torch.randn(1, 1, 4, 2)
    mask = torch.tensor([[[True, True, False, False], [True, False, True, False]]])
    with pytest.raises(ValueError, match="head-independent"):
        build_post_rope_k_landmarks(
            key,
            page_size=2,
            attention_mask=mask,
            landmark_dtype="float32",
        )


def test_teacher_exact_exposes_landmark_page_miss() -> None:
    query = torch.ones(1, 1, 1, 1)
    exact_key = torch.tensor([[[[10.0], [-10.0], [1.0], [1.0]]]])
    value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    mean_config = _config(page_size=2, budget=2)
    landmarks = _landmarks(exact_key, mean_config)
    mean = c1_k_reverse_shadow_attention(
        query, landmarks, value, mean_config, exact_key
    )
    teacher_config = _config(
        page_size=2, budget=2, selector="teacher_exact"
    )
    teacher = c1_k_reverse_shadow_attention(
        query, landmarks, value, teacher_config, exact_key
    )
    assert mean.selected_page_ids.tolist() == [[[1]]]
    assert teacher.selected_page_ids.tolist() == [[[0]]]


def test_centroid_radius_recovers_a_page_hidden_by_mean_cancellation() -> None:
    query = torch.ones(1, 1, 1, 1)
    exact_key = torch.tensor([[[[10.0], [-10.0], [1.0], [1.0]]]])
    value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    config = _config(
        page_size=2, budget=2, selector="centroid_radius"
    )
    landmarks = _landmarks(exact_key, config)
    result = c1_k_reverse_shadow_attention(
        query, landmarks, value, config, exact_key
    )
    assert result.selected_page_ids.tolist() == [[[0]]]
    assert landmarks.values[0, 0, 0, 0, 0] == 0
    assert landmarks.radii[0, 0, 0, 0] == 10


def test_quest_minmax_recovers_a_page_hidden_by_mean_cancellation() -> None:
    query = torch.ones(1, 1, 1, 1)
    exact_key = torch.tensor([[[[10.0], [-10.0], [1.0], [1.0]]]])
    value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    config = _config(page_size=2, budget=2, selector="quest_minmax")
    landmarks = _landmarks(exact_key, config)
    result = c1_k_reverse_shadow_attention(
        query, landmarks, value, config, exact_key
    )
    assert result.selected_page_ids.tolist() == [[[0]]]
    assert result.statistics["resident_selector_metadata_bytes"] == 18


def test_quest_minmax_is_an_upper_bound_for_every_key_in_each_page() -> None:
    generator = torch.Generator().manual_seed(31)
    exact_key = torch.randn(2, 2, 7, 5, generator=generator)
    query = torch.randn(2, 2, 5, generator=generator)
    metadata = build_post_rope_k_landmarks(
        exact_key,
        page_size=4,
        landmarks_per_page=2,
        landmark_dtype="float32",
    )
    for batch in range(2):
        for head in range(2):
            for page in range(2):
                start = page * 4
                stop = min(start + 4, 7)
                upper = torch.maximum(
                    query[batch, head] * metadata.page_mins[batch, head, page],
                    query[batch, head] * metadata.page_maxes[batch, head, page],
                ).sum()
                exact_scores = torch.einsum(
                    "d,sd->s",
                    query[batch, head],
                    exact_key[batch, head, start:stop],
                )
                assert float(upper + 1.0e-5) >= float(exact_scores.max())


def test_c1_page_metadata_matches_explicit_latent_and_decoded_norms() -> None:
    c1_value = torch.tensor(
        [[[[3.0, 4.0], [0.0, 2.0], [1.0, -1.0], [2.0, 1.0], [99.0, 99.0]]]]
    )
    decoder = torch.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]],
            [[0.5, 1.0, 0.0], [1.0, -1.0, 2.0]],
        ]
    )
    mask = torch.tensor([[True, True, True, True, False]])
    gram = build_c1_decoder_gram(decoder)
    metadata = build_c1_page_metadata(
        c1_value,
        decoder,
        page_size=2,
        attention_mask=mask,
        decoder_gram=gram,
    )
    torch.testing.assert_close(gram, decoder @ decoder.transpose(-1, -2))
    assert metadata.valid_token_count.tolist() == [[[2, 2, 0]]]
    assert metadata.latent_max_norm[0, 0].tolist() == pytest.approx(
        [5.0, math.sqrt(5.0), 0.0]
    )
    for head in range(2):
        explicit = torch.linalg.vector_norm(
            c1_value[0, 0, :4] @ decoder[head], dim=-1
        ).reshape(2, 2).amax(dim=-1)
        torch.testing.assert_close(
            metadata.decoded_output_max_norm[0, head, :2], explicit
        )
    assert metadata.decoded_output_max_norm[:, :, 2].count_nonzero() == 0


def test_quest_head_aggregation_is_controlled_independently() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    exact_key = torch.tensor(
        [[[[10.0, -10.0], [10.0, -10.0], [9.5, 9.5], [9.5, 9.5]]]]
    )
    value = torch.ones(1, 1, 4, 1)
    landmarks = build_post_rope_k_landmarks(
        exact_key, page_size=2, landmark_dtype="float32"
    )
    maximum = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(
            page_size=2,
            budget=2,
            selector="quest_k",
            query_head_aggregation="max_head",
        ),
        exact_key,
    )
    logsumexp = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(
            page_size=2,
            budget=2,
            selector="quest_k",
            query_head_aggregation="logsumexp_head",
        ),
        exact_key,
    )
    assert maximum.selected_page_ids.tolist() == [[[0]]]
    assert logsumexp.selected_page_ids.tolist() == [[[1]]]


def test_c1_norm_selectors_reduce_to_quest_k_for_constant_page_norms() -> None:
    query = torch.tensor([[[[1.0]], [[-0.5]]]])
    exact_key = torch.tensor([[[[2.0], [2.0], [1.0], [1.0]]]])
    value = torch.ones(1, 1, 4, 1)
    decoder = torch.ones(2, 1, 1)
    landmarks = build_post_rope_k_landmarks(
        exact_key, page_size=2, landmark_dtype="float32"
    )
    metadata = build_c1_page_metadata(value, decoder, page_size=2)
    masks = []
    for selector in ("quest_k", "quest_c1_latent", "quest_c1_output"):
        result = c1_k_reverse_shadow_attention(
            query,
            landmarks,
            value,
            _config(
                page_size=2,
                budget=2,
                selector=selector,
                query_head_aggregation="logsumexp_head",
            ),
            exact_key,
            c1_page_metadata=metadata,
        )
        masks.append(result.selected_page_mask)
    torch.testing.assert_close(masks[0], masks[1])
    torch.testing.assert_close(masks[0], masks[2])


def test_quest_c1_output_can_prioritize_a_lower_key_bound_page() -> None:
    query = torch.ones(1, 1, 1, 1)
    exact_key = torch.tensor([[[[2.0], [2.0], [1.8], [1.8]]]])
    value = torch.tensor([[[[0.1], [0.1], [10.0], [10.0]]]])
    decoder = torch.ones(1, 1, 1)
    landmarks = build_post_rope_k_landmarks(
        exact_key, page_size=2, landmark_dtype="float32"
    )
    metadata = build_c1_page_metadata(value, decoder, page_size=2)
    key_only = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(page_size=2, budget=2, selector="quest_k"),
        exact_key,
    )
    output_aware = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(page_size=2, budget=2, selector="quest_c1_output"),
        exact_key,
        c1_page_metadata=metadata,
    )
    assert key_only.selected_page_ids.tolist() == [[[0]]]
    assert output_aware.selected_page_ids.tolist() == [[[1]]]
    assert output_aware.statistics["resident_c1_output_norm_bytes"] > 0


def test_stored_centroid_and_fp32_radius_upper_bound_every_subchunk_key() -> None:
    generator = torch.Generator().manual_seed(29)
    exact_key = torch.randn(2, 2, 7, 5, generator=generator)
    query = torch.randn(2, 2, 5, generator=generator)
    landmarks = build_post_rope_k_landmarks(
        exact_key,
        page_size=4,
        landmarks_per_page=2,
        landmark_dtype="bfloat16",
    )
    for batch in range(2):
        for head in range(2):
            query_norm = torch.linalg.vector_norm(query[batch, head])
            for page in range(2):
                for landmark in range(2):
                    start = page * 4 + landmark * 2
                    stop = min(start + 2, 7)
                    if start >= stop:
                        continue
                    center = landmarks.values[batch, head, page, landmark].float()
                    radius = landmarks.radii[batch, head, page, landmark]
                    upper = torch.dot(query[batch, head], center) + query_norm * radius
                    exact_max = torch.einsum(
                        "d,sd->s", query[batch, head], exact_key[batch, head, start:stop]
                    ).max()
                    assert float(upper + 1.0e-5) >= float(exact_max)


def test_teacher_mass_optimizes_aggregate_exact_page_probability() -> None:
    query = torch.ones(1, 1, 1, 1)
    # Page 0 has the largest individual score. Page 1 has greater total exp mass.
    exact_key = torch.tensor([[[[10.0], [-10.0], [9.5], [9.5]]]])
    value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    landmarks = build_post_rope_k_landmarks(
        exact_key, page_size=2, landmark_dtype="float32"
    )
    maximum = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(page_size=2, budget=2, selector="teacher_exact"),
        exact_key,
    )
    mass = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(page_size=2, budget=2, selector="teacher_mass"),
        exact_key,
    )
    assert maximum.selected_page_ids.tolist() == [[[0]]]
    assert mass.selected_page_ids.tolist() == [[[1]]]


def test_teacher_influence_accounts_for_softmax_renormalization() -> None:
    query = torch.ones(1, 1, 1, 1)
    # Two tokens per page. Page 0 owns 80% attention mass but has zero decoded
    # Value; page 1 owns 20% mass and has decoded Value 10.
    exact_key = torch.tensor(
        [[[[math.log(4.0)], [math.log(4.0)], [0.0], [0.0]]]]
    )
    value = torch.tensor([[[[0.0], [0.0], [10.0], [10.0]]]])
    decoder = torch.ones(1, 1, 1)
    landmarks = build_post_rope_k_landmarks(
        exact_key, page_size=2, landmark_dtype="float32"
    )
    output_only = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(page_size=2, budget=2, selector="teacher_output"),
        exact_key,
        decoder=decoder,
    )
    influence = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        _config(page_size=2, budget=2, selector="teacher_influence"),
        exact_key,
        decoder=decoder,
    )
    assert output_only.selected_page_ids.tolist() == [[[1]]]
    assert influence.selected_page_ids.tolist() == [[[0]]]
    torch.testing.assert_close(output_only.page_scores, torch.tensor([[[0.0, 2.0]]]))
    torch.testing.assert_close(influence.page_scores, torch.tensor([[[8.0, 2.0]]]))


def test_teacher_decoded_selectors_require_exact_key_and_decoder() -> None:
    query, exact_key, value = _case(batch=1, query_heads=2, kv_heads=1)
    config = _config(selector="teacher_influence")
    landmarks = _landmarks(exact_key, config)
    with pytest.raises(ValueError, match="exact Key"):
        c1_k_reverse_shadow_attention(query, landmarks, value, config)
    with pytest.raises(ValueError, match="decoder"):
        c1_k_reverse_shadow_attention(
            query, landmarks, value, config, exact_key
        )


def test_gqa_selection_unions_consuming_query_heads_at_physical_page_level() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    exact_key = torch.tensor(
        [[[[10.0, 0.0], [9.0, 0.0], [0.0, 8.0], [0.0, 7.0]]]]
    )
    value = torch.randn(1, 1, 4, 1, generator=torch.Generator().manual_seed(2))
    config = _config(page_size=2, budget=2)
    result = c1_k_reverse_shadow_attention(
        query, _landmarks(exact_key, config), value, config, exact_key
    )
    assert result.selected_page_ids.tolist() == [[[0]]]
    assert result.statistics["selected_pages"] == 1


def test_paper_faithful_quest_selects_pages_per_query_head_and_fetches_union() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    exact_key = torch.tensor(
        [[[[10.0, 0.0], [9.0, 0.0], [0.0, 8.0], [0.0, 7.0]]]]
    )
    value = torch.tensor([[[[1.0], [3.0], [5.0], [7.0]]]])
    config = _config(
        page_size=2,
        budget=2,
        selector="quest_minmax",
        quest_support="per_query_head",
    )
    landmarks = _landmarks(exact_key, config)
    reference = c1_k_reverse_shadow_attention(
        query, landmarks, value, config, exact_key
    )
    vectorized = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        config,
        exact_key,
        vectorized_reference=True,
    )
    assert reference.selected_page_ids.tolist() == [[[0], [1]]]
    assert reference.statistics["selected_pages"] == 2
    assert reference.statistics["logical_selected_pages"] == 2
    assert reference.statistics["selected_tokens"] == 4
    assert reference.statistics["query_selected_tokens"] == 4
    torch.testing.assert_close(reference.output, vectorized.output)
    full_config = _config(
        page_size=2,
        budget=4,
        selector="quest_minmax",
        quest_support="per_query_head",
    )
    full = c1_k_reverse_shadow_attention(
        query,
        landmarks,
        value,
        full_config,
        exact_key,
        vectorized_reference=True,
    )
    quality = reverse_shadow_quality_statistics(
        full,
        vectorized,
        query=query,
        exact_key=exact_key,
        config=config,
    )
    assert all(math.isfinite(metric) for metric in quality.values())


def test_kq_svd_routing_uses_query_head_page_union_for_exact_attention() -> None:
    query = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float64
    )
    exact_key = torch.tensor(
        [[[[10.0, 0.0], [9.0, 0.0], [0.0, 8.0], [0.0, 7.0]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[1.0], [3.0], [5.0], [7.0]]]], dtype=torch.float64
    )
    projector = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    config = _config(page_size=2, budget=2, selector="kq_svd")
    sidecar = build_routing_sidecar(exact_key, projector)
    result = c1_k_reverse_shadow_attention(
        query,
        build_routing_page_geometry(
            exact_key, page_size=2, landmark_dtype="float32"
        ),
        value,
        config,
        exact_key,
        routing_sidecar=sidecar,
        routing_query_projector=projector,
        vectorized_reference=True,
    )
    assert result.selected_page_mask.tolist() == [[[True, True]]]
    assert result.statistics["selected_pages"] == 2
    assert result.statistics["resident_routing_sidecar_bytes"] == 64
    head_to_kv = torch.tensor([0, 0])
    scores = torch.einsum(
        "bhd,bhsd->bhs",
        query[:, :, 0],
        exact_key.index_select(1, head_to_kv),
    ) / math.sqrt(2)
    expected = torch.einsum(
        "bhs,bhsv->bhv",
        torch.softmax(scores, dim=-1),
        value.index_select(1, head_to_kv),
    )[:, :, None]
    torch.testing.assert_close(result.output, expected, rtol=1.0e-6, atol=1.0e-6)


def test_kq_svd_block_routing_keeps_exact_k_and_causal_visibility() -> None:
    generator = torch.Generator().manual_seed(20260829)
    query = torch.randn(1, 4, 3, 4, generator=generator, dtype=torch.float64)
    exact_key = torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(1, 2, 5, 3, generator=generator, dtype=torch.float64)
    projector = torch.eye(4, dtype=torch.float64).expand(2, -1, -1).clone()
    observed, results = c1_k_reverse_shadow_block_attention(
        query,
        exact_key,
        value,
        _config(page_size=2, budget=5, selector="kq_svd"),
        routing_key_projector=projector,
        routing_query_projector=projector,
    )
    cached_observed, cached_results = c1_k_reverse_shadow_block_attention(
        query,
        exact_key,
        value,
        _config(page_size=2, budget=5, selector="kq_svd"),
        routing_key_projector=projector,
        routing_query_projector=projector,
        routing_sidecar=build_routing_sidecar(exact_key, projector),
    )
    torch.testing.assert_close(cached_observed, observed)
    assert [
        result.selected_page_ids.tolist() for result in cached_results
    ] == [result.selected_page_ids.tolist() for result in results]
    expected = []
    head_to_kv = torch.tensor([0, 0, 1, 1])
    prefix = exact_key.shape[2] - query.shape[2]
    for index in range(query.shape[2]):
        visible = prefix + index + 1
        scores = torch.einsum(
            "bhd,bhsd->bhs",
            query[:, :, index],
            exact_key[:, head_to_kv, :visible],
        ) / math.sqrt(query.shape[-1])
        expected.append(
            torch.einsum(
                "bhs,bhsv->bhv",
                torch.softmax(scores, dim=-1),
                value[:, head_to_kv, :visible],
            )[:, :, None]
        )
    torch.testing.assert_close(
        observed,
        torch.cat(expected, dim=2),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert [result.statistics["physical_valid_tokens"] for result in results] == [
        6,
        8,
        10,
    ]


def test_paper_faithful_quest_reserves_last_page_inside_budget() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    exact_key = torch.tensor(
        [[[[10.0, 10.0], [9.0, 9.0], [0.0, 0.0], [0.0, 0.0]]]]
    )
    value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    config = _config(
        page_size=2,
        budget=2,
        recent=1,
        selector="quest_minmax",
        quest_support="per_query_head",
    )
    result = c1_k_reverse_shadow_attention(
        query, _landmarks(exact_key, config), value, config, exact_key
    )
    assert result.selected_page_ids.tolist() == [[[1], [1]]]
    assert result.statistics["selected_pages"] == 1


def test_recent_and_external_hot_pages_are_forced_even_past_budget() -> None:
    query, exact_key, value = _case(batch=1, sequence=9)
    config = _config(page_size=3, budget=3, recent=2)
    forced = torch.zeros(1, 2, 3, dtype=torch.bool)
    forced[:, :, 0] = True
    result = c1_k_reverse_shadow_attention(
        query,
        _landmarks(exact_key, config),
        value,
        config,
        exact_key,
        forced_page_mask=forced,
    )
    assert result.selected_page_ids.shape == (1, 2, 2)
    assert result.selected_page_ids.tolist() == [[[0, 2], [0, 2]]]


def test_masked_padding_pages_are_not_selected() -> None:
    query, exact_key, value = _case(sequence=7)
    mask = torch.tensor(
        [[True, True, True, True, False, False, False]] * 2
    )
    config = _config(page_size=3, budget=6)
    result = c1_k_reverse_shadow_attention(
        query,
        _landmarks(exact_key, config, mask),
        value,
        config,
        exact_key,
        mask,
    )
    assert not torch.any(result.selected_page_ids == 2)
    assert result.selected_token_count == 2 * 2 * 4


def test_all_pages_match_direct_exact_gqa_attention_with_c1_values() -> None:
    query, exact_key, value = _case(sequence=7, head_dim=4, value_rank=3)
    config = _config(page_size=3, budget=7, selector="teacher_exact")
    result = c1_k_reverse_shadow_attention(
        query, _landmarks(exact_key, config), value, config, exact_key
    )
    head_to_kv = torch.tensor([0, 0, 1, 1])
    scores = torch.einsum(
        "bhd,bhsd->bhs",
        query[:, :, 0],
        exact_key.index_select(1, head_to_kv),
    ) / math.sqrt(query.shape[-1])
    expected = torch.einsum(
        "bhs,bhsv->bhv",
        torch.softmax(scores.float(), dim=-1),
        value.index_select(1, head_to_kv).float(),
    )[:, :, None].to(value.dtype)
    torch.testing.assert_close(result.output, expected, rtol=1e-6, atol=1e-7)
    assert result.output.shape == (2, 4, 1, 3)
    assert result.statistics["selected_token_fraction"] == 1


def test_sparse_one_page_output_matches_manual_exact_attention() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    exact_key = torch.tensor(
        [[[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]]]
    )
    value = torch.tensor([[[[3.0], [7.0], [11.0], [13.0]]]])
    config = _config(page_size=2, budget=2, selector="teacher_exact")
    result = c1_k_reverse_shadow_attention(
        query, _landmarks(exact_key, config), value, config, exact_key
    )
    probability = torch.softmax(torch.tensor([1.0, 2.0]) / math.sqrt(2), dim=0)
    expected = probability[0] * 3.0 + probability[1] * 7.0
    torch.testing.assert_close(result.output[0, 0, 0, 0], expected)
    assert result.selected_page_ids.tolist() == [[[0]]]


class _SpyPageStore:
    def __init__(self, exact_key: torch.Tensor) -> None:
        self.delegate = GPUExactKeyPageStore(exact_key)
        self.calls = 0

    def get_pages(self, **kwargs) -> torch.Tensor:
        self.calls += 1
        return self.delegate.get_pages(**kwargs)


def test_exact_keys_are_fetched_once_through_page_store_protocol() -> None:
    query, exact_key, value = _case()
    config = _config(budget=6)
    store = _SpyPageStore(exact_key)
    result = c1_k_reverse_shadow_attention(
        query,
        _landmarks(exact_key, config),
        value,
        config,
        None,
        page_store=store,
    )
    assert store.calls == 1
    assert store.delegate.last_request_count == int(
        result.statistics["selected_pages"]
    )
    assert store.delegate.last_unique_request_count == store.delegate.last_request_count


@pytest.mark.parametrize("magnitude", [1.0e4, -1.0e4])
def test_online_sparse_softmax_stays_finite_for_large_scores(magnitude: float) -> None:
    query, exact_key, value = _case(dtype=torch.float32)
    query.fill_(magnitude)
    exact_key.mul_(100.0)
    config = _config(budget=3, selector="teacher_exact")
    result = c1_k_reverse_shadow_attention(
        query, _landmarks(exact_key, config), value, config, exact_key
    )
    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.running_lse).all()


def test_zero_support_is_rejected() -> None:
    query, exact_key, value = _case()
    config = _config(budget=0)
    with pytest.raises(ValueError, match="at least one selected"):
        c1_k_reverse_shadow_attention(
            query, _landmarks(exact_key, config), value, config, exact_key
        )


def test_quality_metrics_reach_exact_endpoint_and_sparse_kl_is_finite() -> None:
    query, exact_key, value = _case(sequence=6)
    full_config = _config(page_size=2, budget=6, selector="teacher_exact")
    landmarks = _landmarks(exact_key, full_config)
    full = c1_k_reverse_shadow_attention(
        query, landmarks, value, full_config, exact_key
    )
    full_metrics = reverse_shadow_quality_statistics(
        full,
        full,
        query=query,
        exact_key=exact_key,
        config=full_config,
        decoder=torch.randn(
            4, 3, 5, generator=torch.Generator().manual_seed(8), dtype=torch.float64
        ),
    )
    assert full_metrics["exact_attention_mass_selected"] == pytest.approx(1)
    assert full_metrics["attention_kl_sparse_to_exact"] == pytest.approx(0, abs=1e-7)
    assert full_metrics["c1_latent_relative_l2"] == 0
    assert full_metrics["c1_decoded_output_relative_l2"] == 0

    sparse_config = _config(page_size=2, budget=2)
    sparse = c1_k_reverse_shadow_attention(
        query, landmarks, value, sparse_config, exact_key
    )
    sparse_metrics = reverse_shadow_quality_statistics(
        full,
        sparse,
        query=query,
        exact_key=exact_key,
        config=sparse_config,
    )
    assert 0 < sparse_metrics["exact_attention_mass_selected"] < 1
    assert math.isfinite(sparse_metrics["attention_kl_sparse_to_exact"])
    assert math.isfinite(sparse_metrics["c1_latent_relative_l2"])


def test_tiny_capture_replay_emits_json_and_markdown(tmp_path) -> None:
    generator = torch.Generator().manual_seed(14)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    layer_path = capture_dir / "layer_000.safetensors"
    save_file(
        {
            "query": torch.randn(2, 4, 1, 3, generator=generator),
            "exact_key": torch.randn(2, 2, 5, 3, generator=generator),
            "c1_value": torch.randn(2, 2, 5, 2, generator=generator),
            "attention_mask": torch.ones(2, 5, dtype=torch.bool),
            "decoder": torch.randn(4, 2, 6, generator=generator),
        },
        str(layer_path),
    )
    layer_sha = hashlib.sha256(layer_path.read_bytes()).hexdigest()
    manifest_path = capture_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "basisserve.qwen3_8b.c1_k_refine_capture.v1",
                "artifacts": {"0": {"file": layer_path.name, "sha256": layer_sha}},
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "oracle.json"
    output_markdown = tmp_path / "oracle.md"
    evaluate(
        Namespace(
            capture=capture_dir,
            c1_export=None,
            layers="0",
            full_exact_layers="",
            example_start=0,
            examples=1,
            selectors="teacher_exact,mean_landmark,quest_minmax",
            landmarks_per_page="1,2",
            page_sizes="2",
            exact_token_budgets="2,5",
            recent_exact_windows="0",
            landmark_dtype="float32",
            attention_top_k=2,
            device="cpu",
            output_json=output_json,
            output_markdown=output_markdown,
        )
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["format"] == "basisserve.c1_k_reverse_shadow_oracle.v5"
    assert len(payload["records"]) == 8
    assert {row["example"] for row in payload["records"]} == {0}
    assert {row["policy"] for row in payload["records"]} == {
        "teacher_exact",
        "mean_landmark",
        "quest_minmax",
    }
    assert all(math.isfinite(row["c1_latent_relative_l2"]) for row in payload["records"])
    assert len(payload["schedule_aggregate"]) == 8
    assert "teacher_exact" in output_markdown.read_text(encoding="utf-8")

    full_json = tmp_path / "full_exact_schedule.json"
    full_markdown = tmp_path / "full_exact_schedule.md"
    evaluate(
        Namespace(
            capture=capture_dir,
            c1_export=None,
            layers="0",
            full_exact_layers="0",
            example_start=0,
            examples=2,
            selectors="quest_minmax",
            landmarks_per_page="1",
            page_sizes="2",
            exact_token_budgets="2,5",
            recent_exact_windows="0",
            landmark_dtype="float32",
            attention_top_k=2,
            device="cpu",
            output_json=full_json,
            output_markdown=full_markdown,
        )
    )
    full_payload = json.loads(full_json.read_text(encoding="utf-8"))
    assert len(full_payload["records"]) == 4
    assert {row["example"] for row in full_payload["records"]} == {0, 1}
    assert {row["exact_token_budget"] for row in full_payload["records"]} == {2, 5}
    assert all(row["layer_exact_token_budget"] == 5 for row in full_payload["records"])
    assert all(row["selector"] == "full_exact_resident" for row in full_payload["records"])
    assert all(row["cpu_exact_key_bytes_fetched"] == 0 for row in full_payload["records"])
    assert all(row["c1_decoded_output_relative_l2"] == 0 for row in full_payload["records"])
    assert all(group["observations"] == 2 for group in full_payload["schedule_aggregate"])
