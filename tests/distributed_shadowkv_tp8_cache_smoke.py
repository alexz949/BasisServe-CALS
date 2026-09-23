"""Check local-head routing and CPU V offload against official full-head cache."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.shadowkv_tp8 import build_global_factors, make_tp8_cache
from evaluation import official_shadowkv_cpu


def gather_heads(value):
    values = [torch.empty_like(value) for _ in range(8)]
    dist.all_gather(values, value)
    return torch.cat(values, dim=1)


def selected_state(cache, query, key, value, cos_sin):
    cache.update_kv_cache(key, value, 0)
    positions = cache.get_retrieval_position_ids(0, query)
    current = torch.cuda.current_stream()
    with torch.cuda.stream(cache.copy_stream):
        cache.copy_stream.wait_stream(current)
        selected_value = cache.get_value_cache(0, positions)
    selected_key = cache.get_key_cache(0, positions, None, cos_sin)
    current.wait_stream(cache.copy_stream)
    # Canonicalize slot order only in the test; native cache hit order may differ.
    order = positions.argsort(-1)
    token_order = (order.unsqueeze(-1) * 8 + torch.arange(8, device=key.device)).flatten(-2)
    indices = token_order.unsqueeze(-1).expand(-1, -1, -1, 128)
    sparse_key = selected_key[:, :, cache.sparse_start:cache.sparse_end].gather(-2, indices)
    sparse_value = selected_value[:, :, cache.sparse_start:cache.sparse_end].gather(-2, indices)
    canonical_key = torch.cat((selected_key[:, :, :cache.sparse_start], sparse_key,
                               selected_key[:, :, cache.sparse_end:]), dim=2)
    canonical_value = torch.cat((selected_value[:, :, :cache.sparse_start], sparse_value,
                                 selected_value[:, :, cache.sparse_end:]), dim=2)
    return positions.sort(-1).values.clone(), canonical_key, canonical_value


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    assert dist.get_world_size() == 8
    official_shadowkv_cpu.ROOT = args.upstream.resolve()
    cache_class = official_shadowkv_cpu.load_cache_class()
    length, steps = 4096, 8
    angles = torch.outer(torch.arange(length + steps, device=device).float(),
                         1 / 500000 ** (torch.arange(0, 128, 2, device=device).float() / 128))
    doubled = torch.cat((angles, angles), -1)
    cos, sin = doubled.cos().to(torch.bfloat16)[None], doubled.sin().to(torch.bfloat16)[None]
    cos_sin = torch.cat((cos[0, :, :64], sin[0, :, :64]), -1).contiguous()
    cases = []
    for batch in (1, 2):
        torch.manual_seed(2700 + rank)
        pre_key = torch.randn(batch, 1, length, 128, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(pre_key)
        query = torch.randn(batch, 4, 1, 128, device=device, dtype=torch.bfloat16)
        cache = make_tp8_cache(cache_class, layers=1, batch=batch, length=length,
                               decode_steps=steps, device=device)
        build_global_factors(pre_key, 0, cache.U[0], cache.SV[0])
        all_pre_key = gather_heads(pre_key)
        all_value = gather_heads(value)
        all_query = gather_heads(query)
        if rank == 0:
            config = SimpleNamespace(hidden_size=4096, num_attention_heads=32,
                                     num_key_value_heads=8, num_hidden_layers=1)
            reference = cache_class(config, batch_size=batch, max_length=length,
                                    device=str(device), dtype=torch.bfloat16,
                                    sparse_budget=2048, rank=160, chunk_size=8)
            reference.get_svd(all_pre_key, 0)
            _, global_key = apply_rotary_pos_emb(all_pre_key, all_pre_key,
                                                cos[:, :length], sin[:, :length])
            global_query, _ = apply_rotary_pos_emb(all_query, all_query,
                                                  cos[:, length - 1:length], sin[:, length - 1:length])
            reference.prefill_kv_cache(all_value, 0, global_key, global_query)
            reference.H2D()
        _, key = apply_rotary_pos_emb(pre_key, pre_key, cos[:, :length], sin[:, :length])
        query, _ = apply_rotary_pos_emb(query, query, cos[:, length - 1:length], sin[:, length - 1:length])
        cache.prefill_kv_cache(value, 0, key, query)
        cache.H2D()
        assert cache.v_cache_cpu.is_pinned() and cache.v_cache_cpu.device.type == "cpu"
        positions = gather_heads(cache.position_ids[0])
        if rank == 0:
            assert torch.equal(positions, reference.position_ids[0])
        del pre_key, value, query, key, all_pre_key, all_value, all_query
        for step in range(steps):
            query = torch.randn(batch, 4, 1, 128, device=device, dtype=torch.bfloat16)
            key = torch.randn(batch, 1, 1, 128, device=device, dtype=torch.bfloat16)
            value = torch.randn_like(key)
            query, key = apply_rotary_pos_emb(query, key, cos[:, length + step:length + step + 1],
                                              sin[:, length + step:length + step + 1])
            all_query, all_key, all_value = gather_heads(query), gather_heads(key), gather_heads(value)
            positions, selected_key, selected_value = selected_state(cache, query, key, value, cos_sin)
            all_positions = gather_heads(positions)
            all_selected_key = gather_heads(selected_key)
            all_selected_value = gather_heads(selected_value)
            if rank == 0:
                expected_positions, expected_key, expected_value = selected_state(
                    reference, all_query, all_key, all_value, cos_sin)
                assert torch.equal(all_positions, expected_positions)
                torch.testing.assert_close(all_selected_value, expected_value, rtol=0, atol=0)
                relative_error = float((all_selected_key.float() - expected_key.float()).norm()
                                       / expected_key.float().norm())
                assert relative_error < 0.02
                print(json.dumps({"batch": batch, "step": step,
                                  "selected_key_relative_error": relative_error}), flush=True)
            dist.barrier()
        cases.append({"batch": batch, "length": length, "decode_steps": steps,
                      "prefill_local_tokens": cache.prefill_local, "sparse_budget": cache.sparse_budget,
                      "u_shape": list(cache.U.shape), "sv_shape": list(cache.SV.shape),
                      "cpu_value_bytes": cache.v_cache_cpu.numel() * cache.v_cache_cpu.element_size()})
        del cache
        if rank == 0:
            del reference, global_key, global_query
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"cache_rank{rank}.json").write_text(json.dumps({
        "status": "complete", "rank": rank, "cases": cases, "command": sys.argv,
        "scope": "Synthetic cache correctness; not a language-model quality or speed result",
    }, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
