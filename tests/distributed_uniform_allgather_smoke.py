#!/usr/bin/env python3
"""Long-running exactness test for prepared uniform AllGather backends.

Run with torchrun. Producer data changes every iteration so stale slots, missing
system-scope ordering, or incorrect forwarding phases cannot pass by reusing a
previously correct arena.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    FeatureRaggedCommunicator,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan  # noqa: E402


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "uint8": torch.uint8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("uniform_nccl", "uniform_ipc"),
        default="uniform_ipc",
    )
    parser.add_argument(
        "--ipc-algorithm",
        choices=("auto", "fanout", "fanout_warp", "recursive_doubling", "ring"),
        default="auto",
    )
    parser.add_argument(
        "--ipc-channels",
        type=int,
        choices=(0, 1, 2, 4, 8),
        default=0,
    )
    parser.add_argument("--local-width", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="uint8")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--check-every", type=int, default=1)
    return parser.parse_args()


def pattern(
    *,
    iteration: int,
    rank: int,
    local_width: int,
    tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(local_width * tokens, device=device, dtype=torch.int64)
    values = (index * 13 + rank * 47 + iteration * 29).remainder(127)
    return values.view(local_width, tokens).to(dtype).contiguous()


def main() -> None:
    args = parse_args()
    if min(args.local_width, args.tokens, args.iterations, args.check_every) <= 0:
        raise ValueError("all dimensions and iteration counts must be positive")

    if args.ipc_channels > 1 and args.ipc_algorithm not in ("fanout", "ring"):
        raise ValueError("multiple IPC channels require fanout or ring")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = _DTYPES[args.dtype]
    plan = StaticRaggedPlan.from_source_widths((args.local_width,) * world_size)

    communicator = FeatureRaggedCommunicator.from_distributed(device=device)
    if args.backend == "uniform_nccl":
        communicator.configure_direct_workspace(
            tokens=args.tokens,
            max_total_width=plan.total_width,
            dtype=dtype,
        )
    else:
        communicator.prepare_ipc(
            tokens=args.tokens,
            max_total_width=plan.total_width,
            dtype=dtype,
        )
    prepared = communicator.prepare_uniform(
        plan,
        tokens=args.tokens,
        dtype=dtype,
        backend=args.backend,
        ipc_algorithm=args.ipc_algorithm,
        ipc_channels=args.ipc_channels,
    )

    for iteration in range(args.iterations):
        local = pattern(
            iteration=iteration,
            rank=rank,
            local_width=args.local_width,
            tokens=args.tokens,
            dtype=dtype,
            device=device,
        )
        prepared.local_feature_major_view().copy_(local)
        arena = prepared.gather_inplace()
        if iteration % args.check_every == 0 or iteration + 1 == args.iterations:
            expected = torch.cat(
                [
                    pattern(
                        iteration=iteration,
                        rank=source,
                        local_width=args.local_width,
                        tokens=args.tokens,
                        dtype=dtype,
                        device=device,
                    )
                    for source in range(world_size)
                ],
                dim=0,
            )
            torch.testing.assert_close(arena, expected, rtol=0.0, atol=0.0)

    torch.cuda.synchronize(device)
    dist.barrier()
    if rank == 0:
        print(
            f"PASS backend={args.backend} algorithm={args.ipc_algorithm} "
            f"channels={args.ipc_channels} "
            f"tp={world_size} width={args.local_width} tokens={args.tokens} "
            f"dtype={args.dtype} iterations={args.iterations}",
            flush=True,
        )
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
