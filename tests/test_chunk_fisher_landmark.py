import torch

from basisserve.core.chunk_fisher_landmark import (
    ChunkFisherDataset,
    ChunkFisherExample,
    PackedChunkFisherDataset,
    PackedChunkFisherWindow,
    adapt_token_residual_initialization,
    chunk_fisher_metrics,
    chunk_mean_logits,
    chunk_teacher_logits,
    compact_chunk_fisher_metrics,
    compact_packed_chunk_fisher_dataset,
    factor_logits,
    fisher_multiply,
    fit_chunk_fisher_landmarks,
    fit_compact_chunk_fisher_landmarks,
    fit_packed_chunk_fisher_landmarks,
    packed_chunk_fisher_metrics,
    packed_chunk_feature_fisher_grams,
    residual_chunk_features,
)


def test_fisher_multiply_matches_dense_softmax_fisher() -> None:
    generator = torch.Generator().manual_seed(101)
    logits = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    values = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    probability = logits.softmax(-1)
    expected = torch.stack(
        [
            (torch.diag(row) - torch.outer(row, row)) @ value
            for row, value in zip(probability, values, strict=True)
        ]
    )
    torch.testing.assert_close(fisher_multiply(probability, values), expected)


def test_chunk_teacher_and_mean_are_exact_for_identical_rows() -> None:
    generator = torch.Generator().manual_seed(103)
    queries = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    row = torch.randn(2, 1, 5, generator=generator, dtype=torch.float64)
    keys = row.expand(2, 24, 5).clone()
    teacher = chunk_teacher_logits(
        queries,
        keys,
        chunk_size=8,
        scaling=5**-0.5,
    )
    mean = chunk_mean_logits(
        queries,
        keys,
        chunk_size=8,
        scaling=5**-0.5,
    )
    torch.testing.assert_close(teacher, mean)


def test_flat_token_initialization_equals_mean_initialization() -> None:
    generator = torch.Generator().manual_seed(107)
    residual = torch.randn(2, 40, 6, generator=generator, dtype=torch.float64)
    token_encoder = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
    token_query = torch.randn(8, 6, 3, generator=generator, dtype=torch.float64)
    mean_encoder, mean_query = adapt_token_residual_initialization(
        token_encoder,
        token_query,
        chunk_size=8,
        feature="mean",
    )
    flat_encoder, flat_query = adapt_token_residual_initialization(
        token_encoder,
        token_query,
        chunk_size=8,
        feature="flat",
    )
    mean = residual_chunk_features(residual, chunk_size=8, feature="mean")
    flat = residual_chunk_features(residual, chunk_size=8, feature="flat")
    torch.testing.assert_close(mean @ mean_encoder, flat @ flat_encoder)
    torch.testing.assert_close(mean_query, flat_query)


def _synthetic_dataset(seed: int, observations: int) -> tuple[
    ChunkFisherDataset,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    groups = 2
    heads_per_group = 2
    heads = groups * heads_per_group
    head_dim = 5
    feature_dim = 7
    rank = 2
    mapping = torch.arange(heads) // heads_per_group
    target_encoder = torch.randn(
        groups,
        feature_dim,
        rank,
        generator=generator,
        dtype=torch.float64,
    )
    target_query = torch.randn(
        heads,
        head_dim,
        rank,
        generator=generator,
        dtype=torch.float64,
    )
    examples = []
    scaling = head_dim**-0.5
    for observation in range(observations):
        chunks = 11 + observation
        queries = torch.randn(
            heads,
            head_dim,
            generator=generator,
            dtype=torch.float64,
        )
        features = torch.randn(
            groups,
            chunks,
            feature_dim,
            generator=generator,
            dtype=torch.float64,
        )
        chunk_codes = torch.einsum("gci,gir->gcr", features, target_encoder)
        query_codes = torch.einsum("hd,hdr->hr", queries, target_query)
        residual = scaling * torch.einsum(
            "hr,hcr->hc",
            query_codes,
            chunk_codes.index_select(0, mapping),
        )
        base = 0.2 * torch.randn(
            heads,
            chunks,
            generator=generator,
            dtype=torch.float64,
        )
        examples.append(
            ChunkFisherExample(
                position=100 + observation,
                queries=queries,
                features_by_group=features,
                base_logits=base,
                teacher_logits=base + residual,
            )
        )
    dataset = ChunkFisherDataset(tuple(examples), mapping, scaling)
    return dataset, target_encoder, target_query


def test_direct_chunk_fisher_als_reduces_loss_and_preserves_gqa_shapes() -> None:
    train, target_encoder, target_query = _synthetic_dataset(109, 5)
    heldout, _, _ = _synthetic_dataset(113, 2)
    generator = torch.Generator().manual_seed(111)
    initial_encoder = target_encoder + 0.35 * torch.randn(
        target_encoder.shape,
        generator=generator,
        dtype=target_encoder.dtype,
    )
    initial_query = target_query + 0.35 * torch.randn(
        target_query.shape,
        generator=generator,
        dtype=target_query.dtype,
    )
    initial = chunk_fisher_metrics(train, initial_encoder, initial_query)
    result = fit_chunk_fisher_landmarks(
        train,
        heldout,
        initial_encoders=initial_encoder,
        initial_query_factors=initial_query,
        sweeps=3,
        relative_damping=1e-10,
        relative_tolerance=1e-10,
        max_iterations=200,
    )
    final = chunk_fisher_metrics(train, result.encoders, result.query_factors)
    assert result.encoders.shape == (2, 7, 2)
    assert result.query_factors.shape == (4, 5, 2)
    assert len(result.half_steps) == 7
    assert final["fisher_loss"] < initial["fisher_loss"] * 0.05
    assert all(
        right.train["fisher_loss"] <= left.train["fisher_loss"] + 1e-10
        for left, right in zip(result.half_steps, result.half_steps[1:])
    )


def test_factor_logits_uses_one_encoder_per_physical_group() -> None:
    dataset, encoder, query = _synthetic_dataset(127, 1)
    example = dataset.examples[0]
    actual = factor_logits(dataset, example, encoder, query)
    codes = example.features_by_group @ encoder
    queries = torch.einsum("hd,hdr->hr", example.queries, query)
    expected = dataset.scaling * torch.einsum(
        "hr,hcr->hc",
        queries,
        codes.index_select(0, dataset.head_to_group),
    )
    torch.testing.assert_close(actual, expected)


def _pack(dataset: ChunkFisherDataset) -> PackedChunkFisherDataset:
    windows = tuple(
        PackedChunkFisherWindow(
            positions=torch.tensor([example.position]),
            queries=example.queries.unsqueeze(0),
            features_by_group=example.features_by_group,
            base_logits=example.base_logits.unsqueeze(0),
            teacher_logits=example.teacher_logits.unsqueeze(0),
            candidate_counts=torch.tensor([example.features_by_group.shape[1]]),
        )
        for example in dataset.examples
    )
    return PackedChunkFisherDataset(windows, dataset.head_to_group, dataset.scaling)


def test_packed_window_fit_matches_unpacked_objective() -> None:
    train, target_encoder, target_query = _synthetic_dataset(131, 4)
    heldout, _, _ = _synthetic_dataset(137, 2)
    packed_train = _pack(train)
    packed_heldout = _pack(heldout)
    unpacked_metric = chunk_fisher_metrics(train, target_encoder, target_query)
    packed_metric = packed_chunk_fisher_metrics(
        packed_train,
        target_encoder,
        target_query,
    )
    for name in unpacked_metric:
        assert abs(unpacked_metric[name] - packed_metric[name]) < 1e-10
    generator = torch.Generator().manual_seed(139)
    initial_encoder = target_encoder + 0.2 * torch.randn(
        target_encoder.shape,
        generator=generator,
        dtype=target_encoder.dtype,
    )
    initial_query = target_query + 0.2 * torch.randn(
        target_query.shape,
        generator=generator,
        dtype=target_query.dtype,
    )
    initial = packed_chunk_fisher_metrics(
        packed_train,
        initial_encoder,
        initial_query,
    )
    result = fit_packed_chunk_fisher_landmarks(
        packed_train,
        packed_heldout,
        initial_encoders=initial_encoder,
        initial_query_factors=initial_query,
        sweeps=2,
        relative_damping=1e-10,
        relative_tolerance=1e-10,
        max_iterations=200,
    )
    final = packed_chunk_fisher_metrics(
        packed_train,
        result.encoders,
        result.query_factors,
    )
    assert final["fisher_loss"] < initial["fisher_loss"] * 0.05
    assert all(
        right.train["fisher_loss"] <= left.train["fisher_loss"] + 1e-9
        for left, right in zip(result.half_steps, result.half_steps[1:])
    )


def test_compact_statistics_exactly_match_explicit_chunk_objective() -> None:
    train, target_encoder, target_query = _synthetic_dataset(149, 4)
    heldout, _, _ = _synthetic_dataset(151, 2)
    packed_train = _pack(train)
    packed_heldout = _pack(heldout)
    compact_train = compact_packed_chunk_fisher_dataset(
        packed_train,
        statistic_batch_size=3,
    )
    compact_heldout = compact_packed_chunk_fisher_dataset(
        packed_heldout,
        statistic_batch_size=3,
    )
    explicit = packed_chunk_fisher_metrics(
        packed_train,
        target_encoder,
        target_query,
    )
    compact = compact_chunk_fisher_metrics(
        compact_train,
        target_encoder,
        target_query,
    )
    for name in ("fisher_loss", "fisher_nmse", "base_fisher_energy"):
        assert abs(explicit[name] - compact[name]) < 1e-10
    feature_grams = packed_chunk_feature_fisher_grams(packed_train)
    for group in range(compact_train.groups):
        heads = torch.nonzero(
            compact_train.head_to_group == group,
            as_tuple=False,
        ).flatten()
        expected = compact_train.fisher_grams_by_head.index_select(
            0,
            heads,
        ).sum(dim=(0, 1))
        torch.testing.assert_close(feature_grams[group], expected)

    generator = torch.Generator().manual_seed(153)
    initial_encoder = target_encoder + 0.2 * torch.randn(
        target_encoder.shape,
        generator=generator,
        dtype=target_encoder.dtype,
    )
    initial_query = target_query + 0.2 * torch.randn(
        target_query.shape,
        generator=generator,
        dtype=target_query.dtype,
    )
    initial = compact_chunk_fisher_metrics(
        compact_train,
        initial_encoder,
        initial_query,
    )
    result = fit_compact_chunk_fisher_landmarks(
        compact_train,
        compact_heldout,
        initial_encoders=initial_encoder,
        initial_query_factors=initial_query,
        sweeps=2,
        relative_damping=1e-10,
        relative_tolerance=1e-10,
        max_iterations=200,
    )
    final = compact_chunk_fisher_metrics(
        compact_train,
        result.encoders,
        result.query_factors,
    )
    assert final["fisher_loss"] < initial["fisher_loss"] * 0.05
    assert all(
        right.train["fisher_loss"] <= left.train["fisher_loss"] + 1e-5
        for left, right in zip(result.half_steps, result.half_steps[1:])
    )
