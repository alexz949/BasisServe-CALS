"""Numerical check of TP head slicing with nonuniform C1 ranks."""

import torch
from basisserve.core.qwen3_8b_vllm_folded_c1 import folded_c1_projection_weights
from basisserve.core.qwen3_8b_vllm_folded_palu import reconstructed_palu_value_projection


def test_c1_tp2_matches_full_fold():
    torch.manual_seed(81)
    ranks = (2, 3, 4, 3, 5, 6, 7, 8)
    value = torch.randn(64, 20)
    encoders = torch.randn(8, 8, 8)
    decoders = torch.randn(64, 8, 20)
    full_v, full_o = folded_c1_projection_weights(value, encoders, decoders, ranks)
    for shard in range(2):
        local_ranks = ranks[shard * 4:shard * 4 + 4]
        maximum = max(local_ranks)
        local_v, local_o = folded_c1_projection_weights(
            value[shard * 32:shard * 32 + 32],
            encoders[shard * 4:shard * 4 + 4, :, :maximum],
            decoders[shard * 32:shard * 32 + 32, :maximum], local_ranks,
        )
        torch.testing.assert_close(local_v, full_v[shard * 32:shard * 32 + 32])
        torch.testing.assert_close(local_o, full_o[:, shard * 256:shard * 256 + 256])


def test_palu_tp2_preserves_all_group_reconstructions():
    torch.manual_seed(82)
    for group_size in (1, 2, 4):
        ranks = tuple(3 * group_size for _ in range(8 // group_size))
        writer = torch.randn(sum(ranks), 20)
        decoder = torch.randn(len(ranks), 8 * group_size, max(ranks))
        full = reconstructed_palu_value_projection(writer, decoder, ranks)
        halves = [full[shard * 32:shard * 32 + 32] for shard in range(2)]
        inputs = torch.randn(4, 20)
        torch.testing.assert_close(
            torch.cat([inputs @ half.T for half in halves], dim=-1), inputs @ full.T,
        )
