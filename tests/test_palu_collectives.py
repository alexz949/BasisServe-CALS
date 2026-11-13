"""CPU distributed checks of the same group collectives used by PaLU decode/prefill."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from basisserve.vllm.palu_collectives import (
    assemble_latent, exchange_outputs, gather_group, latent_group,
)


def _check_rank(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    tp = SimpleNamespace(ranks=list(range(world)), rank_in_group=rank)
    group, side = latent_group(tp)
    size = world // 2
    assert dist.get_world_size(group) == size
    assert side == rank % size
    assert dist.get_rank(group) == side
    assert latent_group(tp)[0] is group
    heads = 32 // world
    # Each subgroup has different data, so an accidental exchange across groups fails.
    generator = torch.Generator().manual_seed(17 + rank // size)
    for dtype in (torch.float32, torch.bfloat16):
        for batch, length, shard in ((1, 7, 3), (3, 13, 5)):
            latent = torch.randn(batch, length, size * shard, generator=generator).to(dtype)
            keys = torch.randn(size, batch, heads, length, 8, generator=generator)
            queries = torch.randn(size, batch, heads, 8, generator=generator)
            scores = torch.einsum("gbhtd,gbhd->gbht", keys, queries) / 8 ** .5
            lengths = torch.arange(batch) + length - batch + 1
            scores.masked_fill_(torch.arange(length)[None, None, None] >= lengths[None, :, None, None],
                                float("-inf"))
            weights = scores.softmax(-1).to(dtype)
            local_latent = latent[..., side * shard:(side + 1) * shard]

            # Prefill gathers only latent shards, in original encoder-column order.
            full_latent = assemble_latent(gather_group(local_latent, group))
            torch.testing.assert_close(full_latent, latent, rtol=0, atol=0)

            gathered = gather_group(weights[side], group)
            torch.testing.assert_close(gathered, weights, rtol=0, atol=0)
            all_weights = gathered.permute(1, 0, 2, 3).flatten(1, 2)
            partials = all_weights.float() @ local_latent.float()
            partials = partials.view(batch, size, heads, shard).permute(1, 0, 2, 3).to(dtype)
            result = exchange_outputs(partials, group)
            # Independent dense attention over the original, unsharded latent.
            expected = (weights[side].float() @ latent.float()).to(dtype)
            torch.testing.assert_close(result, expected, rtol=1e-5 if dtype == torch.float32 else 1e-2,
                                       atol=1e-6 if dtype == torch.float32 else 1e-2)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 8])
def test_palu_groups_match_unsharded_attention(world, tmp_path):
    # No CUDA initialization: these are short Gloo tests, not model/GPU jobs.
    mp.spawn(_check_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
