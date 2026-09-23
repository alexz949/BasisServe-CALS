"""Owner-rotated global ShadowKV factor construction for TP8."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist


TP_SIZE = 8
HEAD_DIM = 128
SVD_RANK = 160


def make_tp8_cache(cache_class, *, layers, batch, length, decode_steps, device):
    """Use the unchanged official CPU cache with one KV head per TP rank."""
    config = SimpleNamespace(hidden_size=512, num_attention_heads=4,
                             num_key_value_heads=1, num_hidden_layers=layers)
    cache = cache_class(config, batch_size=batch, max_length=length,
                        device=str(device), dtype=torch.bfloat16,
                        sparse_budget=2048, rank=SVD_RANK, chunk_size=8)
    chunks = length // cache.chunk_size - cache.local_chunk
    chunks -= chunks % 8
    local_tokens = length - chunks * cache.chunk_size
    capacity = local_tokens + cache.outlier_chunk * cache.chunk_size + cache.sparse_budget + decode_steps
    shape = (layers, batch, 1, capacity, HEAD_DIM)
    cache.k_cache_buffer = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    cache.v_cache_buffer = torch.zeros_like(cache.k_cache_buffer)
    # U is broadcast directly into its final replicated GPU placement.
    cache.U = torch.empty(layers, batch, length, SVD_RANK, device=device, dtype=torch.bfloat16)
    cache.SV = torch.empty(layers, batch, 1, HEAD_DIM, SVD_RANK, device=device, dtype=torch.bfloat16)
    return cache


@torch.inference_mode()
def build_global_factors(local_key, layer_idx, shared_u, local_sv, phase=None):
    """Fill U[B,T,160] and SV[B,1,128,160] using the official FP32 SVD.

    Only the rotating owner holds global K. Batch items are decomposed in
    sequence to bound its temporary memory. ``phase`` is a timing context
    factory called by every rank, including ranks waiting for the owner.
    """
    rank = dist.get_rank()
    assert dist.get_world_size() == TP_SIZE
    assert local_key.ndim == 4 and local_key.shape[1] == 1
    batch, _, length, width = local_key.shape
    assert width == HEAD_DIM and length >= SVD_RANK
    assert local_key.is_cuda and local_key.dtype == torch.bfloat16
    assert shared_u.shape == (batch, length, SVD_RANK)
    assert local_sv.shape == (batch, 1, HEAD_DIM, SVD_RANK)
    assert shared_u.dtype == local_sv.dtype == local_key.dtype
    assert shared_u.device == local_sv.device == local_key.device
    assert shared_u.is_contiguous() and local_sv.is_contiguous()
    owner = layer_idx % TP_SIZE
    if phase is None:
        phase = lambda name, layer, request: nullcontext()

    for request in range(batch):
        with phase("k_gather", layer_idx, request):
            shard = local_key[request, 0].contiguous()
            if rank == owner:
                gathered = torch.empty(TP_SIZE, length, HEAD_DIM,
                                       dtype=shard.dtype, device=shard.device)
                gathered[owner].copy_(shard)
                operations = [dist.P2POp(dist.irecv, gathered[peer], peer)
                              for peer in range(TP_SIZE) if peer != owner]
            else:
                operations = [dist.P2POp(dist.isend, shard, owner)]
            for work in dist.batch_isend_irecv(operations):
                work.wait()
            if rank == owner:
                global_key = gathered.permute(1, 0, 2).reshape(1, length, -1)
                del gathered
            del shard, operations

        with phase("svd", layer_idx, request):
            if rank == owner:
                # Match upstream get_svd: torch.svd, FP32, then BF16 factors.
                u, singular, v = torch.svd(global_key.float())
                shared_u[request].copy_(u[0, :, :SVD_RANK])
                sv = torch.matmul(torch.diag_embed(singular[:, :SVD_RANK]),
                                  v.transpose(1, 2)[:, :SVD_RANK]).to(local_key.dtype)
                shards = sv.view(SVD_RANK, TP_SIZE, HEAD_DIM).permute(1, 2, 0).contiguous()
                del global_key, u, singular, v, sv

        with phase("factor_redistribution", layer_idx, request):
            dist.broadcast(shared_u[request], src=owner)
            if rank == owner:
                local_sv[request, 0].copy_(shards[owner])
                operations = [dist.P2POp(dist.isend, shards[peer], peer)
                              for peer in range(TP_SIZE) if peer != owner]
            else:
                operations = [dist.P2POp(dist.irecv, local_sv[request, 0], owner)]
            for work in dist.batch_isend_irecv(operations):
                work.wait()
            del operations
            if rank == owner:
                del shards

    return {"owner": owner, "svd_calls": batch if rank == owner else 0,
            "gathered_requests": batch, "svd_rank": SVD_RANK}
