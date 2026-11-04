"""Head-sharded tensor parallelism for the K-routing RULER evaluator (``eval_k_routing_ruler_p16.py``).

Every routing step is local to one physical KV head and its query-head group: the Base/Residual sidecar, the proxy
page scores, the GQA max over the group, the page selection and the exact sparse attention. Sharding whole KV groups
across ranks therefore leaves every routing decision unchanged. Transformers' Qwen3 plan shards q/k/v colwise (rank r
holds KV heads [r*hkv/W, (r+1)*hkv/W) and their query groups) and o_proj/MLP rowwise, and gathers the lm_head
output, so each rank sees identical logits. The evaluator replaces o_proj with the compressed-V decoder of the local
query heads; that partial sum is the only extra all-reduce.

Launch one replica per GPU group with ``torchrun --nproc-per-node W ... --tensor-parallel W``.
"""
from __future__ import annotations

import copy
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.distributed import DistributedConfig


def init_mesh(size: int):
    """Join the torchrun group of this replica and build its TP mesh."""
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    mesh = init_device_mesh('cuda', (size,), mesh_dim_names=('tp',))
    assert dist.get_world_size() == size
    return mesh


def rank_and_size() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def is_writer() -> bool:
    return rank_and_size()[0] == 0


def load_options(mesh) -> dict:
    return dict(distributed_config=DistributedConfig(tp_size=mesh.size()), device_mesh=mesh)


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, 'to_local') else value


def head_slices(query_heads: int, kv_heads: int) -> tuple[slice, slice]:
    """(KV-head slice, query-head slice) owned by this rank; contiguous, matching the colwise q/k/v shards."""
    rank, size = rank_and_size()
    assert kv_heads % size == 0 and query_heads % kv_heads == 0
    local_kv, local_query = kv_heads // size, query_heads // size
    return slice(rank * local_kv, (rank + 1) * local_kv), slice(rank * local_query, (rank + 1) * local_query)


def local_attention_config(config, query_heads: int, kv_heads: int):
    """Config copy whose head counts are this rank's, for modules that size themselves from the config."""
    local = copy.deepcopy(config)
    local.num_attention_heads = query_heads
    local.num_key_value_heads = kv_heads
    return local


def all_reduce_output(module, inputs, output):
    """Forward hook: sum the local query heads' decoder contributions over the TP group."""
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output


def broadcast_flag(value: bool) -> bool:
    """Rank 0's decision, so every rank skips or runs the same sample."""
    if not dist.is_initialized():
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return bool(box[0])


def assert_same_tokens(ids: list[int]) -> None:
    """The gathered lm_head gives identical logits on every rank; check the decoded ids agree."""
    if not dist.is_initialized():
        return
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, ids)
    assert all(other == gathered[0] for other in gathered[1:]), 'TP ranks decoded different tokens'


def merge_layer_statistics(stats: list[dict]) -> list[dict]:
    """Per-layer routing statistics over all ranks: numeric fields averaged (every rank holds the same number of KV
    groups, so this equals the full-head mean), other fields required to agree."""
    if not dist.is_initialized():
        return stats
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, stats)
    merged = []
    for layer in zip(*gathered, strict=True):
        assert all(set(entry) == set(layer[0]) for entry in layer)
        record = {}
        for key, value in layer[0].items():
            values = [entry[key] for entry in layer]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                record[key] = sum(values) / len(values)
            else:
                assert all(v == value for v in values), f'TP ranks disagree on {key}'
                record[key] = value
        merged.append(record)
    return merged


def peak_gib_by_rank() -> dict[str, float]:
    local = torch.cuda.max_memory_allocated() / 2 ** 30
    if not dist.is_initialized():
        return {'0': local}
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return {f'tp_rank_{i}': value for i, value in enumerate(gathered)}


def finish() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
