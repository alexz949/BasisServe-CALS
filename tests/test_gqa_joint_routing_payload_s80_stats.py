from __future__ import annotations

import torch

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (
    S80PayloadAccumulator,
    S80RoutingAccumulator,
    joint_token_features,
    routing_proxy_score_statistics,
)


def test_joint_features_and_payload_accumulator_match_direct_rows() -> None:
    torch.manual_seed(20260830)
    value = torch.randn(3, 7, 4, 3, dtype=torch.float64)
    key = torch.randn(3, 7, 4, 2, dtype=torch.float64)
    output = torch.randn(3, 7, 5, dtype=torch.float64)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[0, -2:] = False
    mask[2, -1] = False

    joint = joint_token_features(value, key)
    torch.testing.assert_close(joint[..., :3], value)
    torch.testing.assert_close(joint[..., 3:], key)

    accumulator = S80PayloadAccumulator(
        num_query_heads=4,
        value_dim=3,
        key_dim=2,
        accumulation_dtype=torch.float64,
    )
    accumulator.update(value, key, output, valid_mask=mask)
    statistics = accumulator.finalize()
    selected = joint.reshape(-1, 20)[mask.reshape(-1)]
    selected_output = output.reshape(-1, 5)[mask.reshape(-1)]
    torch.testing.assert_close(
        statistics.flat_covariance(),
        selected.mT @ selected / selected.shape[0],
    )
    assert statistics.row_count == int(mask.sum())
    assert (
        abs(
            statistics.dense_output_energy
            - float(selected_output.square().sum() / selected.shape[0])
        )
        < 1e-10
    )


def test_routing_shard_grams_and_target_energy_match_explicit_scores() -> None:
    torch.manual_seed(20260831)
    mapping = torch.tensor([0, 0, 1, 1])
    query = torch.randn(2, 4, 2, dtype=torch.float64)
    value = torch.randn(6, 2, 3, dtype=torch.float64)
    key = torch.randn(6, 2, 2, dtype=torch.float64)
    accumulator = S80RoutingAccumulator(
        head_to_kv_group=mapping,
        value_dim=3,
        key_dim=2,
    )
    accumulator.update_shard(
        query,
        value,
        key,
        metadata={"document": 4, "visible_prefix": 6},
    )
    statistics = accumulator.finalize()
    shard = statistics.shards[0]

    for head in range(4):
        torch.testing.assert_close(
            shard.query_grams[head],
            query[:, head].mT @ query[:, head],
        )
    joint = joint_token_features(value, key)
    for group in range(2):
        torch.testing.assert_close(
            shard.joint_grams[group],
            joint[:, group].mT @ joint[:, group],
        )
        torch.testing.assert_close(shard.joint_sums[group], joint[:, group].sum(dim=0))
    torch.testing.assert_close(shard.query_sums, query.permute(1, 0, 2).sum(dim=1))
    expected_energy = sum(
        (query[:, head] @ key[:, int(mapping[head])].mT).square().sum()
        for head in range(4)
    )
    assert abs(statistics.target_score_energy - float(expected_energy)) < 1e-10
    assert shard.metadata["visible_prefix"] == 6

    effective_maps = torch.randn(4, 2, 5, dtype=torch.float64)
    moments = routing_proxy_score_statistics(
        statistics,
        effective_maps,
        scaling=0.25,
    )
    explicit_by_head = []
    for head, group in enumerate(mapping.tolist()):
        explicit_by_head.append(
            (query[:, head] @ effective_maps[head] @ joint[:, group].mT).reshape(-1)
        )
    explicit = torch.cat(explicit_by_head)
    torch.testing.assert_close(
        moments["raw_mean_by_head"],
        torch.stack([item.mean() for item in explicit_by_head]),
    )
    torch.testing.assert_close(
        moments["raw_variance_by_head"],
        torch.stack([item.var(unbiased=False) for item in explicit_by_head]),
    )
    torch.testing.assert_close(moments["raw_mean"], explicit.mean())
    torch.testing.assert_close(moments["raw_variance"], explicit.var(unbiased=False))
    torch.testing.assert_close(moments["scaled_mean"], explicit.mean() * 0.25)
    torch.testing.assert_close(
        moments["scaled_variance"],
        explicit.var(unbiased=False) * 0.25**2,
    )
