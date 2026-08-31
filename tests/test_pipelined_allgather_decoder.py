from __future__ import annotations

import pytest
import torch

from basisserve.kernels.pipelined_allgather_decoder import (
    pack_uniform_decoder_waves,
)


@pytest.mark.parametrize("waves", (2, 4))
def test_packed_decoder_waves_preserve_uniform_partial_decode(waves: int) -> None:
    generator = torch.Generator().manual_seed(7)
    processes = 4
    rows = 5
    local_width = 16
    hidden = 11
    parts = tuple(
        torch.randn(rows, local_width, generator=generator)
        for _ in range(processes)
    )
    decoder = torch.randn(
        processes * local_width,
        hidden,
        generator=generator,
    )

    packed = pack_uniform_decoder_waves(
        decoder,
        processes=processes,
        local_width=local_width,
        waves=waves,
    )
    wave_width = local_width // waves
    observed = torch.zeros(rows, hidden)
    for wave_index, decoder_wave in enumerate(packed):
        start = wave_index * wave_width
        coordinates = torch.cat(
            [part[:, start : start + wave_width] for part in parts],
            dim=1,
        )
        observed.addmm_(coordinates, decoder_wave)

    expected = torch.cat(parts, dim=1) @ decoder
    torch.testing.assert_close(observed, expected, rtol=1.0e-5, atol=1.0e-5)


def test_packed_decoder_waves_reject_invalid_geometry() -> None:
    decoder = torch.randn(64, 9)
    with pytest.raises(ValueError, match="2 or 4"):
        pack_uniform_decoder_waves(
            decoder,
            processes=4,
            local_width=16,
            waves=3,
        )
    with pytest.raises(ValueError, match="reduction width"):
        pack_uniform_decoder_waves(
            decoder[:-1],
            processes=4,
            local_width=16,
            waves=2,
        )
