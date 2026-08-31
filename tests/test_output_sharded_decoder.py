from __future__ import annotations

import pytest
import torch

from basisserve.kernels.output_sharded_decoder import (
    pack_hidden_for_reduce_scatter,
    restore_hidden_from_rank_major_shards,
)


def test_hidden_reduce_scatter_layout_round_trip() -> None:
    value = torch.arange(3 * 8).reshape(3, 8)

    packed = pack_hidden_for_reduce_scatter(value, processes=4)

    assert packed.shape == (12, 2)
    assert torch.equal(packed[:3], value[:, :2])
    assert torch.equal(packed[3:6], value[:, 2:4])
    assert torch.equal(
        restore_hidden_from_rank_major_shards(packed, processes=4),
        value,
    )


@pytest.mark.parametrize("processes", [0, 1])
def test_hidden_layout_rejects_non_distributed_process_count(processes: int) -> None:
    value = torch.ones(2, 8)
    with pytest.raises(ValueError, match="at least two"):
        pack_hidden_for_reduce_scatter(value, processes=processes)
    with pytest.raises(ValueError, match="at least two"):
        restore_hidden_from_rank_major_shards(value, processes=processes)


def test_hidden_layout_rejects_indivisible_shapes() -> None:
    with pytest.raises(ValueError, match="hidden width"):
        pack_hidden_for_reduce_scatter(torch.ones(2, 7), processes=4)
    with pytest.raises(ValueError, match="row count"):
        restore_hidden_from_rank_major_shards(torch.ones(7, 2), processes=4)
