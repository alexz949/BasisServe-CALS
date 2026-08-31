from __future__ import annotations

from pathlib import Path

import torch

from basisserve.core.deepseek_v2_lite_tp4_c1 import (
    HIDDEN_SIZE,
    LOCAL_ATTENTION_WIDTH,
    LOGICAL_SOURCES,
    SOURCE_WIDTH,
    SOURCES_PER_PROCESS,
    TP_SIZE,
    DeepseekV2LiteC1FactorLayer,
    decode_deepseek_v2_lite_c1_coordinates,
    encode_deepseek_v2_lite_local_sources,
    pack_deepseek_v2_lite_tp4_uniform_c1_output_factors,
)


def _rank_one_factors() -> DeepseekV2LiteC1FactorLayer:
    generator = torch.Generator().manual_seed(37)
    return DeepseekV2LiteC1FactorLayer(
        layer_index=4,
        source_rank=1,
        encoders=torch.randn(
            LOGICAL_SOURCES,
            SOURCE_WIDTH,
            1,
            generator=generator,
        ),
        decoders=torch.randn(
            LOGICAL_SOURCES,
            1,
            HIDDEN_SIZE,
            generator=generator,
        ),
        path=Path("layer_004.safetensors"),
        sha256="a" * 64,
    )


def test_tp4_pack_selects_two_logical_sources() -> None:
    factors = _rank_one_factors()
    packed = pack_deepseek_v2_lite_tp4_uniform_c1_output_factors(
        factors,
        process_rank=2,
        manifest_sha256="b" * 64,
        device="cpu",
        dtype=torch.float32,
    )

    torch.testing.assert_close(packed.local_encoders, factors.encoders[4:6])
    assert packed.local_wire_width == SOURCES_PER_PROCESS
    assert packed.global_wire_width == LOGICAL_SOURCES
    assert tuple(packed.decoder_weight.shape) == (HIDDEN_SIZE, LOGICAL_SOURCES)
    assert packed.manifest_sha256 == "b" * 64
    assert packed.artifact_sha256 == "a" * 64


def test_headwise_encoder_avoids_block_diagonal_cross_terms() -> None:
    generator = torch.Generator().manual_seed(41)
    tokens = 3
    source_rank = 7
    attention = torch.randn(
        tokens,
        LOCAL_ATTENTION_WIDTH,
        generator=generator,
    )
    encoders = torch.randn(
        SOURCES_PER_PROCESS,
        SOURCE_WIDTH,
        source_rank,
        generator=generator,
    )

    observed = encode_deepseek_v2_lite_local_sources(attention, encoders)
    expected = torch.cat(
        [
            attention[:, source * SOURCE_WIDTH : (source + 1) * SOURCE_WIDTH]
            @ encoders[source]
            for source in range(SOURCES_PER_PROCESS)
        ],
        dim=-1,
    )

    torch.testing.assert_close(observed, expected)


def test_tp4_allgather_decoder_matches_sourcewise_sum() -> None:
    factors = _rank_one_factors()
    tokens = 2
    attention = torch.randn(
        tokens,
        LOGICAL_SOURCES,
        SOURCE_WIDTH,
        generator=torch.Generator().manual_seed(53),
    )
    local_coordinates = []
    for process_rank in range(TP_SIZE):
        packed = pack_deepseek_v2_lite_tp4_uniform_c1_output_factors(
            factors,
            process_rank=process_rank,
            manifest_sha256="b" * 64,
            device="cpu",
            dtype=torch.float32,
        )
        source_start = process_rank * SOURCES_PER_PROCESS
        source_stop = source_start + SOURCES_PER_PROCESS
        local_coordinates.append(
            encode_deepseek_v2_lite_local_sources(
                attention[:, source_start:source_stop].reshape(
                    tokens,
                    LOCAL_ATTENTION_WIDTH,
                ),
                packed.local_encoders,
            )
        )
    gathered = torch.cat(local_coordinates, dim=-1)
    decoder = pack_deepseek_v2_lite_tp4_uniform_c1_output_factors(
        factors,
        process_rank=0,
        manifest_sha256="b" * 64,
        device="cpu",
        dtype=torch.float32,
    ).decoder_weight
    observed = decode_deepseek_v2_lite_c1_coordinates(gathered, decoder)

    expected = torch.zeros(tokens, HIDDEN_SIZE)
    for source in range(LOGICAL_SOURCES):
        coordinates = attention[:, source] @ factors.encoders[source]
        expected += coordinates @ factors.decoders[source]

    torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-4)
