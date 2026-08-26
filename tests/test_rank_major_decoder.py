from __future__ import annotations

import pytest
import torch

from basisserve.kernels.rank_major_decoder import (
    rank_major_decoder,
    reference_rank_major_decoder,
)


def test_reference_rank_major_layout() -> None:
    processes = 4
    rows = 3
    local_width = 2
    blocks = tuple(
        torch.full((rows, local_width), float(process))
        for process in range(processes)
    )
    rank_major = torch.cat(blocks, dim=0)
    decoder = torch.eye(processes * local_width)
    observed = reference_rank_major_decoder(
        rank_major,
        decoder,
        processes=processes,
    )
    expected = torch.cat(blocks, dim=1)
    torch.testing.assert_close(observed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("local_width", (384, 512, 640, 896))
def test_rank_major_decoder_matches_materialized_reference(
    dtype: torch.dtype,
    local_width: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(117)
    processes = 4
    rows = 129
    output_width = 256
    rank_major = torch.randn(
        processes * rows,
        local_width,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ) / local_width**0.5
    decoder = torch.randn(
        processes * local_width,
        output_width,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ) / (processes * local_width) ** 0.5
    expected = reference_rank_major_decoder(
        rank_major,
        decoder,
        processes=processes,
    )
    observed = rank_major_decoder(
        rank_major,
        decoder,
        processes=processes,
    )
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_large_rank_major_decoder_segmented_path() -> None:
    generator = torch.Generator(device="cuda").manual_seed(441)
    processes = 4
    rows = 32768
    local_width = 512
    output_width = 256
    rank_major = torch.randn(
        processes * rows,
        local_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / local_width**0.5
    decoder = torch.randn(
        processes * local_width,
        output_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / (processes * local_width) ** 0.5
    expected = reference_rank_major_decoder(
        rank_major,
        decoder,
        processes=processes,
    )
    observed = rank_major_decoder(
        rank_major,
        decoder,
        processes=processes,
    )
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
