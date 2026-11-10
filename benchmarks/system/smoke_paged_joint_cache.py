"""Check mapped K, shuffled pages, R16, FP32 bounds and graph replay on TP ranks."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from basisserve.kernels.paged_joint_cache import PagedJointCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(70 + rank)
    heads = 8 // dist.get_world_size()
    batch, capacity, prompt = 2, 256, 128
    pages = capacity // 32
    cache = PagedJointCache(torch.empty(batch * pages, heads, 32, 128, device="cuda", dtype=torch.bfloat16))
    table = torch.randperm(batch * pages, device="cuda").view(batch, pages).int()
    positions = torch.arange(prompt, device="cuda").repeat(batch)
    row = torch.arange(batch, device="cuda").repeat_interleave(prompt)
    slots = table[row, positions // 32].long() * 32 + positions % 32
    key = torch.randn(batch * prompt, heads, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(batch * prompt, heads, 96, device="cuda", dtype=torch.bfloat16)
    right = (torch.randn(heads, 16, 128, device="cuda") * 0.1).bfloat16()
    bias = (torch.randn(heads, 128, device="cuda") * 0.1).bfloat16()
    encoder = (torch.randn(heads, 128, 16, device="cuda") * 0.1).bfloat16()
    angles = torch.randn(capacity, 64, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    lengths = torch.full((batch,), prompt, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * prompt
    cache.append(key, value, positions, slots, right, bias, encoder, rope)
    cache.update_bounds(table, lengths, starts)
    dense, dense_table = cache.prefill_cache(table, lengths, prompt)
    torch.cuda.synchronize()
    assert torch.equal(dense_table, torch.arange(8, device="cuda").view(2, 4))
    restored = dense.view(batch, prompt // 32, heads, 32, 224).permute(0, 1, 3, 2, 4).reshape(batch * prompt, heads, 224)
    torch.testing.assert_close(restored[..., :128], key, rtol=0, atol=0)
    torch.testing.assert_close(restored[..., 128:], value, rtol=0, atol=0)
    prediction = (value[..., :16].transpose(0, 1) @ right + bias[:, None]).transpose(0, 1)
    cosine = rope[positions, :64].repeat(1, 2)[:, None]
    sine = rope[positions, 64:].repeat(1, 2)[:, None]
    rotated = prediction * cosine + torch.cat((-prediction[..., 64:], prediction[..., :64]), -1) * sine
    residual = ((key - rotated).transpose(0, 1) @ encoder).transpose(0, 1)
    actual = cache.gpu[slots // 32, :, slots % 32, 96:112]
    torch.testing.assert_close(actual, residual, rtol=0, atol=0.03125)
    bounds = cache.bounds_view()
    for b in range(batch):
        reference = key.view(batch, prompt, heads, 128)[b, :prompt - 64]
        for page in range(2):
            selected = reference[page * 32:(page + 1) * 32].float()
            torch.testing.assert_close(bounds[table[b, page], :, 0], selected.amin(0), rtol=0, atol=0)
            torch.testing.assert_close(bounds[table[b, page], :, 1], selected.amax(0), rtol=0, atol=0)
    next_key = torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
    next_value = torch.randn(batch, heads, 96, device="cuda", dtype=torch.bfloat16)
    next_positions = torch.full((batch,), prompt, device="cuda", dtype=torch.long)
    next_slots = table[:, prompt // 32].long() * 32
    lengths.fill_(prompt + 1)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)

    def append_step():
        cache.append(next_key, next_value, next_positions, next_slots, right, bias, encoder, rope)
        cache.update_bounds(table, lengths, starts)

    # Compile outside capture, then exercise genuinely changing sequence lengths.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        append_step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        append_step()
    key_history = [key.view(batch, prompt, heads, 128)]
    value_history = [value.view(batch, prompt, heads, 96)]
    for step in range(5):
        next_key.normal_()
        next_value.normal_()
        next_positions.fill_(prompt + step)
        next_slots.copy_(table[:, (prompt + step) // 32].long() * 32 + (prompt + step) % 32)
        lengths.fill_(prompt + step + 1)
        key_history.append(next_key.clone()[:, None])
        value_history.append(next_value.clone()[:, None])
        graph.replay()
    dense, _ = cache.prefill_cache(table, lengths, prompt + 5)
    restored = dense.view(batch, 5, heads, 32, 224).permute(0, 1, 3, 2, 4).reshape(batch, 160, heads, 224)[:, :prompt + 5]
    torch.testing.assert_close(restored[..., :128], torch.cat(key_history, 1), rtol=0, atol=0)
    torch.testing.assert_close(restored[..., 128:], torch.cat(value_history, 1), rtol=0, atol=0)
    # The new partial historical page contains positions 64..68, not the recent tail.
    bounds = cache.bounds_view()
    for b in range(batch):
        selected = key.view(batch, prompt, heads, 128)[b, 64:69].float()
        torch.testing.assert_close(bounds[table[b, 2], :, 0], selected.amin(0), rtol=0, atol=0)
        torch.testing.assert_close(bounds[table[b, 2], :, 1], selected.amax(0), rtol=0, atol=0)
    # Recycle the same physical pages for shorter, unrelated requests.
    recycled_length = 96
    positions = torch.arange(recycled_length, device="cuda").repeat(batch)
    row = torch.arange(batch, device="cuda").repeat_interleave(recycled_length)
    slots = table[row, positions // 32].long() * 32 + positions % 32
    fresh_key = torch.randn(batch * recycled_length, heads, 128, device="cuda", dtype=torch.bfloat16)
    fresh_value = torch.randn(batch * recycled_length, heads, 96, device="cuda", dtype=torch.bfloat16)
    lengths.fill_(recycled_length)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * recycled_length
    cache.append(fresh_key, fresh_value, positions, slots, right, bias, encoder, rope)
    cache.update_bounds(table, lengths, starts)
    dense, _ = cache.prefill_cache(table, lengths, recycled_length)
    restored = dense.view(batch, 3, heads, 32, 224).permute(0, 1, 3, 2, 4).reshape(batch * recycled_length, heads, 224)
    torch.testing.assert_close(restored[..., :128], fresh_key, rtol=0, atol=0)
    torch.testing.assert_close(restored[..., 128:], fresh_value, rtol=0, atol=0)
    bounds = cache.bounds_view()
    for b in range(batch):
        selected = fresh_key.view(batch, recycled_length, heads, 128)[b, :32].float()
        torch.testing.assert_close(bounds[table[b, 0], :, 0], selected.amin(0), rtol=0, atol=0)
        torch.testing.assert_close(bounds[table[b, 0], :, 1], selected.amax(0), rtol=0, atol=0)
    torch.cuda.synchronize()
    summary = dict(status="complete", rank=rank, local_kv_heads=heads,
                   graph_replays=5, max_residual_abs_error=(actual - residual).abs().max().item(),
                   exact_k_and_v_match=True, historical_bounds_match=True, recycled_pages_match=True,
                   gpu_cache_bytes=cache.gpu.numel() * cache.gpu.element_size(),
                   host_key_bytes=cache.host.numel() * cache.host.element_size())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
