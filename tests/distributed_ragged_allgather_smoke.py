#!/usr/bin/env python3
"""NCCL correctness test for the compiled static-ragged AllGather kernel.

Run from the repository root with at least two GPUs, for example:

    torchrun --standalone --nproc-per-node=2 \
      tests/distributed_ragged_allgather_smoke.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from basisserve.kernels.ragged_allgather import (
    RaggedNcclCommunicator,
    StaticRaggedPlan,
    load_ragged_allgather_extension,
    pack_rank_major_for_test,
)


def _reference_gathers(
    local: torch.Tensor,
    plan: StaticRaggedPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    maximum_width = max(plan.source_widths)
    padded = torch.zeros(
        local.shape[0],
        maximum_width,
        dtype=local.dtype,
        device=local.device,
    )
    padded[:, : local.shape[1]].copy_(local)
    rank_major = torch.empty(
        world_size * local.shape[0],
        maximum_width,
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(rank_major, padded)
    by_rank = rank_major.reshape(world_size, local.shape[0], maximum_width)
    compact_rank_major = torch.cat(
        tuple(
            by_rank[source, :, :width].reshape(-1)
            for source, width in enumerate(plan.source_widths)
        )
    )
    token_major = torch.cat(
        tuple(by_rank[source, :, :width] for source, width in enumerate(plan.source_widths)),
        dim=1,
    )
    return compact_rank_major, token_major


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batches", default="1,4,32")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--algorithms", default="direct")
    parser.add_argument(
        "--workspace-modes",
        default="fresh",
        help="comma-separated fresh and/or registered",
    )
    parser.add_argument(
        "--source-widths",
        help="optional process-rank-ordered widths, for example 128,192,256",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = _dtype(args.dtype)
    widths = (
        tuple(int(value) for value in args.source_widths.split(","))
        if args.source_widths
        else tuple(3 + 2 * source for source in range(world_size))
    )
    if len(widths) != world_size:
        raise ValueError(
            f"received {len(widths)} source widths for world size {world_size}"
        )
    plan = StaticRaggedPlan.from_source_widths(widths)
    extension = load_ragged_allgather_extension()

    # Exercise the CUDA layout kernel independently from NCCL using truly
    # ragged, multi-token source blocks.
    pack_batch = 3
    blocks = tuple(
        torch.arange(
            pack_batch * width,
            device=device,
            dtype=torch.float32,
        ).reshape(pack_batch, width)
        + 1000 * source
        for source, width in enumerate(widths)
    )
    rank_major = torch.cat(tuple(block.flatten() for block in blocks))
    packed = pack_rank_major_for_test(rank_major, plan, batch=pack_batch)
    torch.testing.assert_close(packed, torch.cat(blocks, dim=1))

    communicator = RaggedNcclCommunicator.from_process_group(device=device)
    generator = torch.Generator(device=device).manual_seed(20260819)
    decoder = torch.randn(
        plan.total_width,
        args.hidden_size,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    bias = torch.randn(
        args.hidden_size,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    records: list[dict[str, object]] = []
    algorithms = tuple(value.strip() for value in args.algorithms.split(","))
    workspace_modes = tuple(
        value.strip() for value in args.workspace_modes.split(",")
    )
    if any(value not in ("fresh", "registered") for value in workspace_modes):
        raise ValueError("workspace modes must be fresh and/or registered")
    try:
        for batch in tuple(int(value) for value in args.batches.split(",")):
            local = (
                torch.arange(
                    batch * widths[rank],
                    dtype=torch.float32,
                    device=device,
                ).reshape(batch, widths[rank])
                + 10000 * rank
                + 100 * batch
            ).to(dtype)
            reference_rank_major, reference_gather = _reference_gathers(local, plan)
            reference_decode = reference_gather @ decoder + bias
            for algorithm in algorithms:
                for workspace_mode in workspace_modes:
                    registered = workspace_mode == "registered"
                    observed_rank_major = communicator.all_gather_rank_major(
                        local,
                        plan,
                        algorithm=algorithm,
                        registered=registered,
                    )
                    torch.testing.assert_close(
                        observed_rank_major,
                        reference_rank_major,
                        rtol=0,
                        atol=0,
                    )
                    observed_gather = communicator.all_gather(
                        local,
                        plan,
                        algorithm=algorithm,
                        registered=registered,
                    )
                    torch.testing.assert_close(
                        observed_gather,
                        reference_gather,
                        rtol=0,
                        atol=0,
                    )

                    observed_decode = communicator.all_gather_decode(
                        local,
                        decoder,
                        plan,
                        bias,
                        algorithm=algorithm,
                        registered=registered,
                    )
                    difference = (
                        observed_decode.float() - reference_decode.float()
                    ).abs()
                    relative = torch.linalg.vector_norm(
                        difference
                    ) / torch.linalg.vector_norm(reference_decode.float()).clamp_min(
                        1e-12
                    )
                    metrics = torch.stack((difference.max(), relative)).to(
                        torch.float64
                    )
                    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
                    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
                    if float(metrics[1]) > tolerance:
                        raise AssertionError(
                            "ragged decode relative error "
                            f"{float(metrics[1]):.6e} exceeds {tolerance}"
                        )
                    records.append(
                        {
                            "batch": batch,
                            "algorithm": algorithm,
                            "workspace_mode": workspace_mode,
                            "maximum_absolute_error": float(metrics[0]),
                            "maximum_relative_l2_error": float(metrics[1]),
                        }
                    )
    finally:
        dist.barrier()
        communicator.close()

    if rank == 0:
        payload = {
            "format": "basisserve.ragged_allgather_smoke.v1",
            "status": "passed",
            "world_size": world_size,
            "device": torch.cuda.get_device_name(device),
            "dtype": args.dtype,
            "source_widths": list(widths),
            "total_width": plan.total_width,
            "padded_total_width": plan.padded_total_width,
            "padding_overhead": plan.padding_overhead,
            "nccl_version": int(extension.nccl_version()),
            "algorithms": list(algorithms),
            "workspace_modes": list(workspace_modes),
            "records": records,
        }
        print(json.dumps(payload, indent=2), flush=True)
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
