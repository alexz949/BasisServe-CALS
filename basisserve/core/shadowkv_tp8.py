"""Owner-rotated global ShadowKV factor construction under tensor parallelism (TP8, TP4, or one GPU).

Rank r of W holds KV heads [r*8/W, (r+1)*8/W). The rank-160 SVD is still the single global one of upstream
get_svd: the layer's owner (layer % W) gathers every KV head's pre-RoPE Keys, decomposes [T, 8*128] in FP32, and
redistributes U (replicated) and each rank's own SV rows. Only data placement and communication depend on W.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist


KV_HEADS = 8
QUERY_HEADS = 32
HEAD_DIM = 128
SVD_RANK = 160
# FlashAttention-2 (vLLM build) reads each request's buffer as one paged-KV block; block sizes must be multiples of 16.
BLOCK_ALIGNMENT = 16


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def make_tp8_cache(cache_class, *, layers, batch, length, decode_steps, device):
    """The unchanged official CPU cache, instantiated for this rank's local KV heads."""
    world = world_size()
    kv_heads, query_heads = KV_HEADS // world, QUERY_HEADS // world
    config = SimpleNamespace(hidden_size=query_heads * HEAD_DIM, num_attention_heads=query_heads,
                             num_key_value_heads=kv_heads, num_hidden_layers=layers)
    cache = cache_class(config, batch_size=batch, max_length=length,
                        device=str(device), dtype=torch.bfloat16,
                        sparse_budget=2048, rank=SVD_RANK, chunk_size=8)
    chunks = length // cache.chunk_size - cache.local_chunk
    chunks -= chunks % 8
    local_tokens = length - chunks * cache.chunk_size
    capacity = local_tokens + cache.outlier_chunk * cache.chunk_size + cache.sparse_budget + decode_steps
    capacity = -(-capacity // BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
    shape = (layers, batch, kv_heads, capacity, HEAD_DIM)
    cache.k_cache_buffer = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    cache.v_cache_buffer = torch.zeros_like(cache.k_cache_buffer)
    # U is broadcast directly into its final replicated GPU placement.
    cache.U = torch.empty(layers, batch, length, SVD_RANK, device=device, dtype=torch.bfloat16)
    cache.SV = torch.empty(layers, batch, kv_heads, HEAD_DIM, SVD_RANK, device=device, dtype=torch.bfloat16)
    return cache


def _exchange(operations):
    for work in dist.batch_isend_irecv(operations) if operations else ():
        work.wait()


@torch.inference_mode()
def build_global_factors(local_key, layer_idx, shared_u, local_sv, phase=None):
    """Fill U[B,T,160] and this rank's SV[B,kv,128,160] using the official FP32 SVD.

    Only the rotating owner holds global K. Batch items are decomposed in
    sequence to bound its temporary memory. ``phase`` is a timing context
    factory called by every rank, including ranks waiting for the owner.
    """
    world = world_size()
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert local_key.ndim == 4 and local_key.shape[1] * world == KV_HEADS
    batch, kv_heads, length, width = local_key.shape
    assert width == HEAD_DIM and length >= SVD_RANK
    assert local_key.is_cuda and local_key.dtype == torch.bfloat16
    assert shared_u.shape == (batch, length, SVD_RANK)
    assert local_sv.shape == (batch, kv_heads, HEAD_DIM, SVD_RANK)
    assert shared_u.dtype == local_sv.dtype == local_key.dtype
    assert shared_u.device == local_sv.device == local_key.device
    assert shared_u.is_contiguous() and local_sv.is_contiguous()
    owner = layer_idx % world
    if phase is None:
        phase = lambda name, layer, request: nullcontext()

    for request in range(batch):
        with phase("k_gather", layer_idx, request):
            shard = local_key[request].contiguous()
            if rank == owner:
                gathered = torch.empty(world, kv_heads, length, HEAD_DIM,
                                       dtype=shard.dtype, device=shard.device)
                gathered[owner].copy_(shard)
                operations = [dist.P2POp(dist.irecv, gathered[peer], peer)
                              for peer in range(world) if peer != owner]
            else:
                operations = [dist.P2POp(dist.isend, shard, owner)]
            _exchange(operations)
            if rank == owner:
                # Token-major, then global KV head (rank-major, then local head), as upstream get_svd.
                global_key = gathered.permute(2, 0, 1, 3).reshape(1, length, -1)
                del gathered
            del shard, operations

        with phase("svd", layer_idx, request):
            if rank == owner:
                # Match upstream get_svd: torch.svd, FP32, then BF16 factors.
                u, singular, v = torch.svd(global_key.float())
                shared_u[request].copy_(u[0, :, :SVD_RANK])
                sv = torch.matmul(torch.diag_embed(singular[:, :SVD_RANK]),
                                  v.transpose(1, 2)[:, :SVD_RANK]).to(local_key.dtype)
                shards = sv.view(SVD_RANK, world, kv_heads, HEAD_DIM).permute(1, 2, 3, 0).contiguous()
                del global_key, u, singular, v, sv

        with phase("factor_redistribution", layer_idx, request):
            if world > 1:
                dist.broadcast(shared_u[request], src=owner)
            if rank == owner:
                local_sv[request].copy_(shards[owner])
                operations = [dist.P2POp(dist.isend, shards[peer], peer)
                              for peer in range(world) if peer != owner]
            else:
                operations = [dist.P2POp(dist.irecv, local_sv[request], owner)]
            _exchange(operations)
            del operations
            if rank == owner:
                del shards

    return {"owner": owner, "svd_calls": batch if rank == owner else 0,
            "gathered_requests": batch, "svd_rank": SVD_RANK, "world_size": world}
