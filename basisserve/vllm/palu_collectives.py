"""Collectives within a PaLU head group, independent of vLLM and attention kernels."""

from functools import lru_cache

import torch
import torch.distributed as dist


@lru_cache(maxsize=None)
def _latent_groups(ranks, size):
    # All TP ranks create every subgroup in the same order, including nonmembers.
    return tuple(dist.new_group(list(ranks[start:start + size]))
                 for start in range(0, len(ranks), size))


def latent_group(tp_group, kv_heads=8, group_heads=4):
    """Return the process group and local index owning one contiguous KV-head group."""
    ranks = tuple(tp_group.ranks)
    local_heads = kv_heads // len(ranks)
    size = group_heads // local_heads
    return (_latent_groups(ranks, size)[tp_group.rank_in_group // size],
            tp_group.rank_in_group % size)


def gather_group(tensor, group):
    """AllGather equal-shaped tensors, with a leading axis in subgroup-rank order."""
    size = dist.get_world_size(group)
    gathered = torch.empty((size * tensor.shape[0], *tensor.shape[1:]),
                           device=tensor.device, dtype=tensor.dtype)
    dist.all_gather_into_tensor(gathered, tensor.contiguous(), group=group)
    return gathered.view(size, *tensor.shape)


def assemble_latent(shards):
    """[group, ..., shard] -> [..., full latent], preserving encoder column order."""
    return shards.movedim(0, -2).flatten(-2)


def exchange_outputs(partials, group):
    """Send partials[destination] to its query owner and concatenate received shards.

    Each rank computes [group, sequences, local query heads, shard] partials using
    its own latent shard. AllToAll returns the same shape indexed by source rank.
    Only the query owner receives its partials; no rank receives others' outputs.
    """
    send = partials.contiguous().flatten(0, 1)
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return assemble_latent(received.view_as(partials))
