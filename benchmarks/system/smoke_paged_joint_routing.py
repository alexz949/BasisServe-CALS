"""TP smoke of paged two-stage selection, slot reuse and exact selected attention."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from basisserve.kernels.paged_joint_cache import PagedJointCache
from basisserve.kernels.paged_joint_router import PagedJointRouter
from basisserve.kernels.paged_joint_slots import PagedJointSlots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(123 + rank)
    batch, n, capacity = 2, 20000, 20160
    heads = 8 // dist.get_world_size()
    pages = capacity // 32
    cache = PagedJointCache(torch.empty(batch * pages, heads, 32, 128, device="cuda", dtype=torch.bfloat16))
    router = PagedJointRouter(batch, heads, capacity, "cuda")
    slots = PagedJointSlots(batch, heads, capacity, "cuda")
    table = torch.randperm(batch * pages, device="cuda").view(batch, pages).int()
    positions = torch.arange(n, device="cuda").repeat(batch)
    rows = torch.arange(batch, device="cuda").repeat_interleave(n)
    mapping = table[rows, positions // 32].long() * 32 + positions % 32
    key = torch.randn(batch * n, heads, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(batch * n, heads, 96, device="cuda", dtype=torch.bfloat16)
    right = (torch.randn(heads, 16, 128, device="cuda") * .1).bfloat16()
    bias = (torch.randn(heads, 128, device="cuda") * .1).bfloat16()
    encoder = (torch.randn(heads, 128, 16, device="cuda") * .1).bfloat16()
    residual_query = (torch.randn(heads * 4, 128, 16, device="cuda") * .1).bfloat16()
    angles = torch.randn(capacity, 64, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    lengths = torch.full((batch,), n, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * n
    query = torch.randn(batch, heads * 4, 128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(batch, heads * 4, 96, device="cuda", dtype=torch.bfloat16)
    cache.append(key, value, positions, mapping, right, bias, encoder, rope)
    cache.update_bounds(table, lengths, starts)

    def step():
        support = router.select(query, cache, table, lengths, right, bias, residual_query, rope)
        return slots.attend(query, cache, table, lengths, support, output)

    print(f"rank={rank} compiling paged routing", flush=True)
    step()
    cold = slots.counts.clone()
    assert bool((cold[..., 0] == 0).all())
    assert bool((cold[..., 1] <= 2048).all())

    # Independent PyTorch coarse bounds and normalized top512 reference.
    reference_key = key.view(batch, n, heads, 128)
    bounds = reference_key[:, :n - 64].view(batch, (n - 64) // 32, 32, heads, 128).float()
    lo, hi = bounds.amin(2), bounds.amax(2)
    q = query.view(batch, heads, 4, 128).float()
    coarse = torch.maximum(q[:, :, :, None] * lo.permute(0, 2, 1, 3)[:, :, None],
                           q[:, :, :, None] * hi.permute(0, 2, 1, 3)[:, :, None]).sum(-1) * (128 ** -.5)
    coarse += torch.tensor(32., device="cuda").log()
    torch.testing.assert_close(router.coarse[..., :coarse.shape[-1]], coarse, rtol=1e-5, atol=1e-5)
    scores = torch.full_like(router.coarse, -float("inf"))
    scores[..., 1:coarse.shape[-1]] = coarse[..., 1:]
    normalized = scores - scores.logsumexp(-1, keepdim=True)
    group = normalized.amax(2)
    group[..., 0] = float("inf")
    group[..., coarse.shape[-1]:(n + 31) // 32] = float("inf")
    expected_candidates = group.argsort(dim=-1, descending=True, stable=True)[..., :512].sort(-1).values
    torch.testing.assert_close(router.candidates, expected_candidates.int(), rtol=0, atol=0)
    largest_fine_error = 0.
    for b in range(batch):
        for h in range(heads):
            ids = router.candidates[b, h].long()[:, None] * 32 + torch.arange(32, device="cuda")
            valid = ids < n - 64
            ids_safe = ids.clamp_max(n - 1)
            base = value.view(batch, n, heads, 96)[b, ids_safe, h, :16]
            prediction = base @ right[h] + bias[h]
            cs = rope[ids_safe, :64].repeat(1, 1, 2)
            sn = rope[ids_safe, 64:].repeat(1, 1, 2)
            rotated = prediction * cs + torch.cat((-prediction[..., 64:], prediction[..., :64]), -1) * sn
            qh = query[b, h * 4:(h + 1) * 4]
            qc = (qh[:, None].float() @ residual_query[h * 4:(h + 1) * 4].float()).squeeze(1).bfloat16()
            residual = cache.gpu[table[b, ids_safe // 32], h, ids_safe % 32, 96:112]
            fine = ((rotated @ qh.T + residual @ qc.T) * (128 ** -.5)).float()
            fine = fine.masked_fill(~valid[..., None], -float("inf")).logsumexp(1).T
            finite = fine.isfinite()
            error = (router.fine[b, h][finite] - fine[finite]).abs().max().item()
            largest_fine_error = max(largest_fine_error, error)
            torch.testing.assert_close(router.fine[b, h], fine, rtol=0, atol=.05)
            # Test selection exactly against its input, isolating BF16 score differences.
            normalized = router.fine[b, h, :, 1:].softmax(-1).amax(0)
            selected = torch.cat((torch.zeros(1, device="cuda", dtype=torch.long),
                                  normalized.argsort(descending=True, stable=True)[:61] + 1)).sort().values
            torch.testing.assert_close(router.selected_pages[b, h], router.candidates[b, h, selected], rtol=0, atol=0)

    def check_attention():
        for b in range(batch):
            for h in range(heads):
                ids = router.support[b, h]
                ids = ids[ids >= 0].long()
                assert len(ids) == len(ids.unique()) and len(ids) <= 2048
                assert bool(torch.isin(torch.arange(32, device="cuda"), ids).all())
                assert bool(torch.isin(torch.arange(n - 64, n, device="cuda"), ids).all())
                q = query[b, h * 4:(h + 1) * 4].float()
                k = key.view(batch, n, heads, 128)[b, ids, h].float()
                v = value.view(batch, n, heads, 96)[b, ids, h].float()
                expected = ((q @ k.T * (128 ** -.5)).softmax(-1) @ v).bfloat16()
                torch.testing.assert_close(output[b, h * 4:(h + 1) * 4], expected, rtol=.01, atol=.002)

    check_attention()
    step()
    assert bool((slots.counts[..., 1] == 0).all())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    for _ in range(4):
        query.normal_()
        graph.replay()
        check_attention()
    # New request reuses the physical pages and batch rows: no stale K-slot hits.
    key.normal_()
    value.normal_()
    cache.append(key, value, positions, mapping, right, bias, encoder, rope)
    cache.update_bounds(table, lengths, starts)
    graph.replay()
    assert bool((slots.counts[..., 0] == 0).all())
    check_attention()
    summary = dict(status="complete", rank=rank, local_kv_heads=heads,
                   coarse_selection_match=True, fine_selection_match=True,
                   fine_lse_max_abs_error=largest_fine_error, exact_attention_match=True,
                   cold_counts=cold.tolist(), repeated_support_misses=0,
                   request_recycle_hits=0, graph_replays=5)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"routing_rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
