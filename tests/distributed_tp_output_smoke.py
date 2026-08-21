"""Two-or-more-rank smoke test for distributed checkpoint factorization.

Run from the repository root:

    torchrun --standalone --nproc-per-node=2 tests/distributed_tp_output_smoke.py
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist

from basisserve.core import (
    FixedTopKAllGatherOutput,
    LowRankAllReduceOutput,
    PrivateAllGatherOutput,
)


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    hidden_size = 32
    if hidden_size % world_size != 0:
        raise ValueError("hidden_size must be divisible by world_size")

    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    full_weight = torch.randn(hidden_size, hidden_size, generator=generator, device=device)
    hidden = torch.randn(5, hidden_size, generator=generator, device=device)

    local_width = hidden_size // world_size
    start = global_rank * local_width
    local_weight = full_weight[:, start : start + local_width].contiguous()
    local_hidden = hidden[:, start : start + local_width].contiguous()

    module = LowRankAllReduceOutput.from_local_weight_distributed(
        local_weight,
        rank=hidden_size,
        factor_dtype=full_weight.dtype,
    )
    with torch.inference_mode():
        output = module(local_hidden)
    expected = hidden @ full_weight.transpose(0, 1)
    max_error = (output - expected).abs().max()
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)

    if global_rank == 0:
        print(f"distributed full-rank max error: {float(max_error):.6e}")
    if float(max_error) > 1e-4:
        raise AssertionError(f"distributed factorization mismatch: {float(max_error)}")

    private_rank = 3
    private_bases = []
    for rank_index in range(world_size):
        basis_generator = torch.Generator(device=device)
        basis_generator.manual_seed(3000 + rank_index)
        basis = torch.linalg.qr(
            torch.randn(
                hidden_size,
                private_rank,
                generator=basis_generator,
                device=device,
            )
        ).Q
        private_bases.append(basis)
    local_basis = private_bases[global_rank]
    local_input_factor = local_weight.transpose(0, 1) @ local_basis
    concatenated_basis = torch.cat(private_bases, dim=1)
    private_module = PrivateAllGatherOutput(
        local_input_factor,
        concatenated_basis,
    )
    with torch.inference_mode():
        private_output, gathered = private_module(
            local_hidden,
            return_gathered_latent=True,
        )
    expected_private = torch.zeros_like(private_output)
    for rank_index, basis in enumerate(private_bases):
        shard_start = rank_index * local_width
        shard_stop = shard_start + local_width
        shard_weight = full_weight[:, shard_start:shard_stop]
        projected_weight = basis @ basis.transpose(0, 1) @ shard_weight
        expected_private += hidden[:, shard_start:shard_stop] @ projected_weight.transpose(0, 1)
    private_max_error = (private_output - expected_private).abs().max()
    dist.all_reduce(private_max_error, op=dist.ReduceOp.MAX)
    if global_rank == 0:
        print(
            "distributed private-all-gather max error: "
            f"{float(private_max_error):.6e}"
        )
    if tuple(gathered.shape) != (5, world_size * private_rank):
        raise AssertionError(f"unexpected gathered latent shape: {tuple(gathered.shape)}")
    if float(private_max_error) > 1e-4:
        raise AssertionError(f"private all-gather mismatch: {float(private_max_error)}")

    topk_module = FixedTopKAllGatherOutput(full_weight, keep_ratio=0.5)
    with torch.inference_mode():
        topk_output, topk_gathered = topk_module(
            local_hidden,
            return_gathered_input=True,
        )
    expected_topk_input = torch.zeros_like(hidden)
    kept = local_width // 2
    for rank_index in range(world_size):
        shard_start = rank_index * local_width
        shard_stop = shard_start + local_width
        source = hidden[:, shard_start:shard_stop]
        indices = torch.topk(source.float().abs(), kept, dim=-1, sorted=False).indices
        expected_topk_input[:, shard_start:shard_stop].scatter_(
            1,
            indices,
            source.gather(1, indices),
        )
    expected_topk_output = expected_topk_input @ full_weight.transpose(0, 1)
    topk_max_error = (topk_output - expected_topk_output).abs().max()
    topk_input_max_error = (topk_gathered - expected_topk_input).abs().max()
    dist.all_reduce(topk_max_error, op=dist.ReduceOp.MAX)
    dist.all_reduce(topk_input_max_error, op=dist.ReduceOp.MAX)
    if global_rank == 0:
        print(
            "distributed fixed-TopK-all-gather max errors: "
            f"input={float(topk_input_max_error):.6e} "
            f"output={float(topk_max_error):.6e} "
            f"packet={topk_module.packet_bytes}B"
        )
    if float(topk_input_max_error) > 1e-6 or float(topk_max_error) > 1e-4:
        raise AssertionError("packed fixed-TopK AllGather mismatch")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
