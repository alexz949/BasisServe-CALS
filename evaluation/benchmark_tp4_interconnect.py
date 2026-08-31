#!/usr/bin/env python3
"""Measure the real four-GPU NCCL transport used by TP decode.

The benchmark reports latency curves for AllReduce and AllGather, plus a
pairwise NCCL send/receive matrix.  Payload accounting is explicit: AllReduce
uses the ring-equivalent bytes moved per rank, AllGather uses the bytes each
rank must send to its peers, and pairwise results report application payload
bytes.  This avoids calling a small-message latency an unrealistic PCIe
bandwidth number.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor  # noqa: E402


FORMAT = "basisserve.tp4_interconnect_profile.v1"


def _parse_bytes(text: str) -> int:
    selected = text.strip().lower().replace("ib", "b")
    scales = {
        "b": 1,
        "k": 1 << 10,
        "kb": 1 << 10,
        "m": 1 << 20,
        "mb": 1 << 20,
        "g": 1 << 30,
        "gb": 1 << 30,
    }
    suffix = ""
    while selected and selected[-1].isalpha():
        suffix = selected[-1] + suffix
        selected = selected[:-1]
    if suffix not in scales or not selected:
        raise ValueError(f"invalid byte size {text!r}")
    value = float(selected) * scales[suffix]
    if not math.isfinite(value) or value <= 0 or not value.is_integer():
        raise ValueError(f"byte size must be a positive integer, got {text!r}")
    return int(value)


def _parse_byte_csv(text: str) -> tuple[int, ...]:
    values = tuple(_parse_bytes(item) for item in text.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("byte sizes must be nonempty and unique")
    return values


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float]:
    selected = tuple(map(float, values))
    if not selected:
        raise ValueError("cannot summarize empty timings")
    return {
        "mean_ms": statistics.fmean(selected),
        "minimum_ms": min(selected),
        "p50_ms": statistics.median(selected),
        "p95_ms": _quantile(selected, 0.95),
        "maximum_ms": max(selected),
    }


def _critical_cuda_timings(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize(device)
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return local.cpu().tolist()


def _bandwidth_record(
    *,
    operation: str,
    payload_bytes: int,
    traffic_bytes_per_rank: float,
    timings_ms: Sequence[float],
) -> dict[str, Any]:
    timing = _summary(timings_ms)
    p50_seconds = timing["p50_ms"] / 1000.0
    return {
        "operation": operation,
        "payload_bytes": int(payload_bytes),
        "traffic_bytes_per_rank": float(traffic_bytes_per_rank),
        "timing": timing,
        "effective_traffic_gbps_at_p50": traffic_bytes_per_rank / p50_seconds / 1.0e9,
    }


def _collective_records(
    *,
    byte_sizes: Sequence[int],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    world_size = dist.get_world_size()
    records: list[dict[str, Any]] = []
    for payload_bytes in byte_sizes:
        if payload_bytes % 2:
            raise ValueError("BF16 collective payloads must contain a whole element")
        elements = payload_bytes // 2
        all_reduce_input = torch.zeros(elements, dtype=torch.bfloat16, device=device)
        timings = _critical_cuda_timings(
            lambda: dist.all_reduce(all_reduce_input),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        records.append(
            _bandwidth_record(
                operation="all_reduce",
                payload_bytes=payload_bytes,
                traffic_bytes_per_rank=(
                    2.0 * (world_size - 1) / world_size * payload_bytes
                ),
                timings_ms=timings,
            )
        )

        all_gather_input = torch.zeros(elements, dtype=torch.bfloat16, device=device)
        all_gather_output = torch.empty(
            world_size * elements,
            dtype=torch.bfloat16,
            device=device,
        )
        timings = _critical_cuda_timings(
            lambda: dist.all_gather_into_tensor(all_gather_output, all_gather_input),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        records.append(
            _bandwidth_record(
                operation="all_gather",
                payload_bytes=payload_bytes,
                traffic_bytes_per_rank=(world_size - 1) * payload_bytes,
                timings_ms=timings,
            )
        )
    return records


def _p2p_once(
    *,
    source: int,
    destination: int,
    send: Tensor,
    receive: Tensor,
    bidirectional: bool,
) -> None:
    rank = dist.get_rank()
    operations: list[dist.P2POp] = []
    if rank == source:
        operations.append(dist.P2POp(dist.isend, send, destination))
        if bidirectional:
            operations.append(dist.P2POp(dist.irecv, receive, destination))
    elif rank == destination:
        operations.append(dist.P2POp(dist.irecv, receive, source))
        if bidirectional:
            operations.append(dist.P2POp(dist.isend, send, source))
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()


def _p2p_records(
    *,
    byte_sizes: Sequence[int],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    world_size = dist.get_world_size()
    records: list[dict[str, Any]] = []
    for source in range(world_size):
        for destination in range(source + 1, world_size):
            for payload_bytes in byte_sizes:
                send = torch.zeros(payload_bytes, dtype=torch.uint8, device=device)
                receive = torch.empty_like(send)
                for bidirectional in (False, True):
                    timings = _critical_cuda_timings(
                        lambda source=source, destination=destination, send=send,
                        receive=receive, bidirectional=bidirectional: _p2p_once(
                            source=source,
                            destination=destination,
                            send=send,
                            receive=receive,
                            bidirectional=bidirectional,
                        ),
                        warmup=warmup,
                        iterations=iterations,
                        device=device,
                    )
                    traffic = payload_bytes * (2 if bidirectional else 1)
                    record = _bandwidth_record(
                        operation=(
                            "send_recv_bidirectional"
                            if bidirectional
                            else "send_recv_unidirectional"
                        ),
                        payload_bytes=payload_bytes,
                        traffic_bytes_per_rank=traffic,
                        timings_ms=timings,
                    )
                    record.update(
                        {
                            "source_rank": source,
                            "destination_rank": destination,
                            "aggregate_bidirectional": bidirectional,
                        }
                    )
                    records.append(record)
    return records


def _command_output(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(command), "error": str(error)}
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collective-bytes",
        default="8KB,64KB,512KB,4MB,16MB,64MB",
    )
    parser.add_argument("--p2p-bytes", default="1MB,16MB,64MB")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != 4:
        raise RuntimeError("interconnect benchmark requires exactly four ranks")

    collective_bytes = _parse_byte_csv(args.collective_bytes)
    p2p_bytes = _parse_byte_csv(args.p2p_bytes)
    collectives = _collective_records(
        byte_sizes=collective_bytes,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    p2p = _p2p_records(
        byte_sizes=p2p_bytes,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    if dist.get_rank() == 0:
        payload = {
            "format": FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "protocol": {
                "world_size": dist.get_world_size(),
                "dtype": "bfloat16 for collectives; uint8 payload for P2P",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "collective_payload_bytes": list(collective_bytes),
                "p2p_payload_bytes": list(p2p_bytes),
                "timing": "maximum GPU elapsed time across ranks per iteration",
            },
            "environment": {
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "nccl_environment": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith("NCCL_")
                },
                "nvidia_smi_topology": _command_output(("nvidia-smi", "topo", "-m")),
                "nvidia_smi_pcie": _command_output(
                    (
                        "nvidia-smi",
                        "--query-gpu=index,name,pci.bus_id,pcie.link.gen.current,"
                        "pcie.link.width.current",
                        "--format=csv,noheader",
                    )
                ),
            },
            "collectives": collectives,
            "p2p": p2p,
        }
        output = Path(args.output_json).expanduser().resolve()
        _atomic_json(output, payload)
        print(json.dumps({"event": "result_written", "path": str(output)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
