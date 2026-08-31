from __future__ import annotations

from pathlib import Path

import torch

from basisserve.core.qwen3_32b_tp4_decode import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    QUERY_HEADS_PER_KV_HEAD,
    QUERY_HEADS_PER_PROCESS,
    TP_SIZE,
    Qwen3_32BTP4C1FactorLayer,
    fold_ragged_local_c1_value_projection,
)
from basisserve.core.qwen3_32b_vllm_c1 import (
    decode_qwen3_32b_c1_coordinates,
    pack_qwen3_32b_tp4_uniform_c1_output_factors,
)


def _uniform_rank_one_factors() -> Qwen3_32BTP4C1FactorLayer:
    generator = torch.Generator().manual_seed(41)
    return Qwen3_32BTP4C1FactorLayer(
        layer_index=3,
        source_ranks=(1,) * NUM_KV_HEADS,
        encoders=torch.randn(
            NUM_KV_HEADS,
            HEAD_DIM,
            1,
            generator=generator,
        ),
        decoders=torch.randn(
            NUM_QUERY_HEADS,
            1,
            HIDDEN_SIZE,
            generator=generator,
        ),
        path=Path("layer_003.safetensors"),
        sha256="a" * 64,
    )


def test_folded_v64_projection_matches_dense_v_then_encoder() -> None:
    generator = torch.Generator().manual_seed(17)
    tokens = 3
    source_rank = 64
    hidden_states = torch.randn(
        tokens, HIDDEN_SIZE, generator=generator, dtype=torch.float64
    )
    dense_weight = torch.randn(
        2 * HEAD_DIM, HIDDEN_SIZE, generator=generator, dtype=torch.float64
    )
    local_encoders = torch.randn(
        2,
        HEAD_DIM,
        source_rank,
        generator=generator,
        dtype=torch.float64,
    )
    folded, folded_bias = fold_ragged_local_c1_value_projection(
        dense_weight,
        tuple(local_encoders.unbind(0)),
    )
    assert folded_bias is None
    observed = torch.nn.functional.linear(hidden_states, folded).view(
        tokens,
        2,
        source_rank,
    )
    dense = torch.nn.functional.linear(hidden_states, dense_weight).view(
        tokens,
        2,
        HEAD_DIM,
    )
    expected = torch.einsum("tsh,shr->tsr", dense, local_encoders)
    torch.testing.assert_close(observed, expected, rtol=1.0e-11, atol=1.0e-10)


def test_uniform_tp4_pack_selects_two_rank_owned_sources() -> None:
    factors = _uniform_rank_one_factors()
    packed = pack_qwen3_32b_tp4_uniform_c1_output_factors(
        factors,
        process_rank=2,
        manifest_sha256="b" * 64,
        device="cpu",
        dtype=torch.float32,
    )

    torch.testing.assert_close(packed.local_encoders, factors.encoders[4:6])
    assert packed.local_wire_width == QUERY_HEADS_PER_PROCESS
    assert packed.global_wire_width == NUM_QUERY_HEADS
    assert tuple(packed.decoder_weight.shape) == (HIDDEN_SIZE, NUM_QUERY_HEADS)
    assert packed.manifest_sha256 == "b" * 64
    assert packed.artifact_sha256 == "a" * 64


def test_tp4_allgather_decoder_matches_sourcewise_c1_sum() -> None:
    factors = _uniform_rank_one_factors()
    packed = pack_qwen3_32b_tp4_uniform_c1_output_factors(
        factors,
        process_rank=0,
        manifest_sha256="b" * 64,
        device="cpu",
        dtype=torch.float32,
    )
    tokens = 2
    attention = torch.randn(
        tokens, NUM_QUERY_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(53)
    )
    local_coordinates = []
    for process_rank in range(TP_SIZE):
        source_start = process_rank * 2
        head_start = process_rank * QUERY_HEADS_PER_PROCESS
        grouped = attention[
            :,
            head_start : head_start + QUERY_HEADS_PER_PROCESS,
        ].reshape(tokens, 2, QUERY_HEADS_PER_KV_HEAD, HEAD_DIM)
        local_coordinates.append(
            torch.einsum(
                "tshd,sdr->tshr",
                grouped,
                factors.encoders[source_start : source_start + 2],
            ).reshape(tokens, -1)
        )
    gathered = torch.cat(local_coordinates, dim=-1)
    observed = decode_qwen3_32b_c1_coordinates(
        gathered,
        packed.decoder_weight,
    )

    expected = torch.zeros(tokens, HIDDEN_SIZE)
    for source in range(NUM_KV_HEADS):
        head_start = source * QUERY_HEADS_PER_KV_HEAD
        head_stop = head_start + QUERY_HEADS_PER_KV_HEAD
        coordinates = attention[:, head_start:head_stop] @ factors.encoders[source]
        expected += coordinates.squeeze(-1) @ factors.decoders[
            head_start:head_stop,
            0,
        ]

    torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-4)
