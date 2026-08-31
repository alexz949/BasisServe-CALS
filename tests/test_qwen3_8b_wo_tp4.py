from __future__ import annotations

from pathlib import Path

import torch

from basisserve.core.qwen3_8b_wo_tp4 import (
    C1_LOCAL_RANK,
    LR_CAPACITY_RANK,
    LR_SHARED_RANK,
    LOCAL_INPUT_WIDTH,
    factors_from_payload,
)
from evaluation.benchmark_qwen3_8b_wo_tp4 import _communication


HIDDEN_SIZE = 4096
TP_SIZE = 4


def test_c1_phase1_factors_preserve_tp_source_and_decoder_order() -> None:
    encoders = torch.zeros(
        TP_SIZE,
        LOCAL_INPUT_WIDTH,
        C1_LOCAL_RANK,
        dtype=torch.bfloat16,
    )
    decoders = torch.zeros(
        TP_SIZE,
        C1_LOCAL_RANK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    for source in range(TP_SIZE):
        encoders[source].fill_(source + 1)
        decoders[source].fill_(10 * (source + 1))

    factors = factors_from_payload(
        {
            "c1_source_encoders": encoders,
            "c1_source_decoders": decoders,
        },
        layer_index=7,
        arm="wo_c1_ag",
        process_rank=2,
        artifact_path=Path("layer_007.safetensors"),
        artifact_sha256="abc",
    )

    assert factors.layer_index == 7
    assert factors.local_input_factor.shape == (LOCAL_INPUT_WIDTH, C1_LOCAL_RANK)
    assert torch.equal(factors.local_input_factor, encoders[2])
    assert factors.global_decoder.shape == (TP_SIZE * C1_LOCAL_RANK, HIDDEN_SIZE)
    for source in range(TP_SIZE):
        start = source * C1_LOCAL_RANK
        stop = start + C1_LOCAL_RANK
        assert torch.equal(factors.global_decoder[start:stop], decoders[source])


def test_lr_phase1_factors_select_contiguous_tp_input_shard() -> None:
    input_factor = torch.zeros(
        HIDDEN_SIZE,
        LR_SHARED_RANK,
        dtype=torch.bfloat16,
    )
    for source in range(TP_SIZE):
        start = source * LOCAL_INPUT_WIDTH
        input_factor[start : start + LOCAL_INPUT_WIDTH].fill_(source + 1)
    decoder = torch.zeros(
        LR_SHARED_RANK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )

    factors = factors_from_payload(
        {
            "lr_wire_input_factor": input_factor,
            "lr_wire_shared_decoder": decoder,
        },
        layer_index=11,
        arm="wo_lr_ar_wire",
        process_rank=3,
        artifact_path=Path("layer_011.safetensors"),
        artifact_sha256="def",
    )

    assert factors.local_input_factor.shape == (
        LOCAL_INPUT_WIDTH,
        LR_SHARED_RANK,
    )
    assert torch.equal(
        factors.local_input_factor,
        input_factor[3 * LOCAL_INPUT_WIDTH : 4 * LOCAL_INPUT_WIDTH],
    )
    assert torch.equal(factors.global_decoder, decoder)


def test_capacity_lr_phase1_uses_full_c1_decoder_row_capacity() -> None:
    input_factor = torch.zeros(
        HIDDEN_SIZE,
        LR_CAPACITY_RANK,
        dtype=torch.bfloat16,
    )
    decoder = torch.zeros(
        LR_CAPACITY_RANK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )

    factors = factors_from_payload(
        {
            "lr_capacity_input_factor": input_factor,
            "lr_capacity_shared_decoder": decoder,
        },
        layer_index=3,
        arm="wo_lr_ar_capacity",
        process_rank=1,
        artifact_path=Path("layer_003.safetensors"),
        artifact_sha256="capacity",
    )

    assert LR_CAPACITY_RANK == TP_SIZE * C1_LOCAL_RANK
    assert factors.local_input_factor.shape == (
        LOCAL_INPUT_WIDTH,
        LR_CAPACITY_RANK,
    )
    assert factors.global_decoder.shape == (LR_CAPACITY_RANK, HIDDEN_SIZE)


def test_local_decode_allreduce_reuses_joint_c1_factors() -> None:
    encoders = torch.randn(
        TP_SIZE,
        LOCAL_INPUT_WIDTH,
        C1_LOCAL_RANK,
        dtype=torch.bfloat16,
    )
    decoders = torch.randn(
        TP_SIZE,
        C1_LOCAL_RANK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )

    factors = factors_from_payload(
        {
            "c1_source_encoders": encoders,
            "c1_source_decoders": decoders,
        },
        layer_index=5,
        arm="wo_c1_local_ar",
        process_rank=2,
        artifact_path=Path("layer_005.safetensors"),
        artifact_sha256="local-ar",
    )

    assert torch.equal(factors.local_input_factor, encoders[2])
    assert torch.equal(
        factors.global_decoder,
        decoders.reshape(TP_SIZE * C1_LOCAL_RANK, HIDDEN_SIZE),
    )


def test_wo_runtime_ideal_ring_budgets_match() -> None:
    dense = _communication("dense", layers=36, dtype_bytes=2)
    c1 = _communication("wo_c1_ag", layers=36, dtype_bytes=2)
    c1_local = _communication("wo_c1_local_ar", layers=36, dtype_bytes=2)
    lr = _communication("wo_lr_ar_wire", layers=36, dtype_bytes=2)
    lr_capacity = _communication("wo_lr_ar_capacity", layers=36, dtype_bytes=2)

    assert dense["ideal_ring_bytes_per_rank_per_activation_row_per_layer"] == 12288
    assert c1["ideal_ring_bytes_per_rank_per_activation_row_per_layer"] == 3072
    assert lr["ideal_ring_bytes_per_rank_per_activation_row_per_layer"] == 3072
    assert c1["reduction_vs_dense"] == lr["reduction_vs_dense"] == 0.75
    assert c1_local["ideal_ring_bytes_per_rank_per_activation_row_per_layer"] == 12288
    assert c1_local["reduction_vs_dense"] == 0.0
    assert lr_capacity["ideal_ring_bytes_per_rank_per_activation_row_per_layer"] == 6144
    assert lr_capacity["reduction_vs_dense"] == 0.5
