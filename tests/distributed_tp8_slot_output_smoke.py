"""Check slot attention -> prepared TP8 AllGather -> output decoder."""

from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention


@torch.inference_mode()
def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    communicator = FeatureRaggedCommunicator.from_distributed(device=device)
    plan = StaticRaggedPlan.from_source_widths((384,) * world)
    generator = torch.Generator(device=device).manual_seed(91)
    decoder = torch.randn(3072, 4096, device=device, dtype=torch.bfloat16, generator=generator) / 3072**0.5

    for batch in (1, 2, 8):
        torch.manual_seed(42 + rank)
        q = torch.randn(batch, 4, 1, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, 1, 2048, 128, device=device, dtype=q.dtype)
        v = torch.randn(batch, 1, 4096, 96, device=device, dtype=q.dtype)
        ids = torch.stack([torch.randperm(4096, device=device)[:2048] for _ in range(batch)])[:, None]
        slots = torch.stack([torch.randperm(2048, device=device) for _ in range(batch)])[:, None]
        ids[:, :, :128] = -1
        rows = torch.arange(batch, device=device)[:, None]
        selected_k = k[:, 0][rows, slots[:, 0]].float()
        selected_v = v[:, 0][rows, ids[:, 0].clamp_min(0)].float()
        scores = q[:, :, 0].float() @ selected_k.transpose(1, 2) * 128**-0.5
        reference = (scores.masked_fill(ids[:, 0, None] < 0, -torch.inf).softmax(-1) @ selected_v)
        reference = reference.reshape(batch, 384).to(q.dtype).contiguous()
        peers = [torch.empty_like(reference) for _ in range(world)]
        dist.all_gather(peers, reference)
        expected_features = torch.cat(peers, dim=1)

        prepared = communicator.prepare_uniform(plan, tokens=batch, dtype=q.dtype, backend="uniform_nccl")
        out = prepared.local_feature_major_view_fast().T.view(batch, 4, 1, 96)
        workspace = (
            torch.empty(batch, 4, 16, 96, device=device),
            torch.empty(batch, 4, 16, device=device),
            out,
        )
        slot_indexed_attention(q, k, v, ids, slots, workspace, scale=128**-0.5)
        actual_features = prepared.gather_inplace_fast().T
        torch.testing.assert_close(actual_features, expected_features, atol=0.002, rtol=0.02)
        actual = actual_features @ decoder
        expected = expected_features @ decoder
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
        error = (actual - expected).abs().max().float()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(json.dumps({"status": "passed", "tp": world, "batch": batch, "decoder_max_abs_error": error.item()}), flush=True)
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
