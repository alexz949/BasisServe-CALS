#!/usr/bin/env python3
"""Benchmark one full fixed-context Qwen3-8B Wo-only TP4 decode graph."""

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
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.core.qwen3_8b_tp4_decode import (  # noqa: E402
    HIDDEN_SIZE,
    TP_SIZE,
    file_sha256,
)
from basisserve.core.qwen3_8b_wo_tp4 import (  # noqa: E402
    ARMS,
    Qwen3TP4WOC1DecodeAttention,
    Qwen3TP4WOC1LocalAllReduceDecodeAttention,
    Qwen3TP4WOLRAllReduceDecodeAttention,
    close_qwen3_8b_wo_tp4,
    install_qwen3_8b_wo_tp4_attention,
)
from evaluation.benchmark_qwen3_8b_tp4_decode import (  # noqa: E402
    _configure_caches,
    _distributed_greedy,
    _global_max_float,
    _global_max_int,
)
from evaluation.benchmark_qwen3_8b_wo_tp4 import (  # noqa: E402
    _communication,
    _load_quality,
    _projection_correctness_gate,
)


FORMAT = "basisserve.qwen3_8b.wo_cuda_graph_fixed_decode.tp4.v2"
DEFAULT_CONFIGURATIONS = "1x512,8x512,32x512,64x512"


def parse_configurations(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse unique batch x fixed-context pairs."""

    parsed = []
    for item in raw.split(","):
        fields = item.strip().lower().split("x")
        if len(fields) != 2:
            raise ValueError(f"invalid batch x context pair {item!r}")
        pair = tuple(map(int, fields))
        if min(pair) <= 0:
            raise ValueError("batch and context must be positive")
        parsed.append(pair)
    result = tuple(parsed)
    if not result or len(result) != len(set(result)):
        raise ValueError("configurations must be nonempty and unique")
    return result


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize steady-state latency using the required paper protocol."""

    selected = tuple(map(float, values))
    if not selected or any(not math.isfinite(value) or value <= 0.0 for value in selected):
        raise ValueError("timings must be finite, positive, and nonempty")
    ordered = sorted(selected)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "mean_ms": statistics.fmean(selected),
        "median_ms": statistics.median(selected),
        "p95_ms": p95,
        "std_ms": statistics.pstdev(selected),
        "minimum_ms": min(selected),
        "maximum_ms": max(selected),
    }


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


def _git_commit() -> str | None:
    result = _command_output(("git", "rev-parse", "HEAD"))
    if result.get("returncode") != 0:
        return None
    return str(result["stdout"]).strip() or None


def _global_consensus_int(value: int, *, device: torch.device) -> int:
    minimum = torch.tensor(int(value), dtype=torch.int64, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum) != int(maximum):
        raise AssertionError("TP ranks produced different integer values")
    return int(maximum)


def _set_cache_lengths(modules: Sequence[torch.nn.Module], length: int) -> None:
    for module in modules:
        module.set_cache_length(length)


def _initialize_zero_prefix(
    modules: Sequence[torch.nn.Module],
    *,
    context_length: int,
) -> None:
    """Create identical fixed-shape cache contents outside the measured path."""

    for module in modules:
        if module.key_cache is None or module.value_cache is None:
            raise RuntimeError("static caches are not configured")
        module.key_cache.zero_()
        module.value_cache.zero_()
        module.set_cache_length(context_length)


def _distributed_token_gate(token: torch.Tensor) -> list[int]:
    batch = int(token.numel())
    gathered = torch.empty(
        TP_SIZE * batch,
        dtype=token.dtype,
        device=token.device,
    )
    dist.all_gather_into_tensor(gathered, token.contiguous())
    by_rank = gathered.view(TP_SIZE, batch)
    if not torch.equal(by_rank, by_rank[0].expand_as(by_rank)):
        raise AssertionError("greedy token differs across TP ranks")
    return by_rank[0, : min(batch, 8)].cpu().tolist()


def _assert_same_tokens(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    label: str,
) -> None:
    if torch.equal(expected, actual):
        return
    limit = min(int(expected.numel()), 8)
    expected_prefix = expected.reshape(-1)[:limit].cpu().tolist()
    actual_prefix = actual.reshape(-1)[:limit].cpu().tolist()
    mismatch_count = int(torch.count_nonzero(expected != actual).cpu())
    raise AssertionError(
        f"{label}: {mismatch_count}/{expected.numel()} tokens differ; "
        f"expected prefix={expected_prefix}, actual prefix={actual_prefix}"
    )


def graph_decode_step(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    position: int,
) -> torch.Tensor:
    """Run one decode step with an explicitly resolved no-mask mapping."""

    batch = int(token_ids.shape[0])
    position_ids = torch.full(
        (1, 1),
        int(position),
        dtype=torch.int64,
        device=token_ids.device,
    )
    output = model.model(
        input_ids=token_ids.reshape(batch, 1),
        position_ids=position_ids,
        attention_mask={"full_attention": None},
        use_cache=False,
    )
    hidden = output.last_hidden_state[:, -1, :]
    return _distributed_greedy(F.linear(hidden, model.lm_head.weight))


@torch.inference_mode()
def _benchmark_configuration(
    model: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    *,
    batch_size: int,
    context_length: int,
    prompt_token_id: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        cache_bytes = _configure_caches(
            modules,
            batch=batch_size,
            capacity=context_length + 1,
        )
        _initialize_zero_prefix(modules, context_length=context_length)
        static_token = torch.full(
            (batch_size,),
            int(prompt_token_id),
            dtype=torch.int64,
            device=device,
        )
    capture_stream.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    eager_reference: torch.Tensor | None = None
    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            _set_cache_lengths(modules, context_length)
            eager_reference = graph_decode_step(
                model,
                static_token,
                position=context_length,
            )
    capture_stream.synchronize()
    if eager_reference is None:
        raise AssertionError("warmup produced no eager output")
    eager_reference = eager_reference.detach().clone()

    eager_times = []
    for _ in range(repeats):
        _set_cache_lengths(modules, context_length)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            start.record()
            eager_output = graph_decode_step(
                model,
                static_token,
                position=context_length,
            )
            end.record()
        end.synchronize()
        eager_times.append(_global_max_float(start.elapsed_time(end), device=device))
    _assert_same_tokens(
        eager_reference,
        eager_output,
        label="fixed-input eager decode is not deterministic",
    )

    _set_cache_lengths(modules, context_length)
    dist.barrier()
    torch.cuda.synchronize(device)
    allocated_before_capture = torch.cuda.memory_allocated(device)
    reserved_before_capture = torch.cuda.memory_reserved(device)
    capture_started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = graph_decode_step(
            model,
            static_token,
            position=context_length,
        )
    capture_stream.synchronize()
    dist.barrier()
    capture_build_seconds = _global_max_float(
        time.perf_counter() - capture_started,
        device=device,
    )
    allocated_after_capture = torch.cuda.memory_allocated(device)
    reserved_after_capture = torch.cuda.memory_reserved(device)

    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            graph.replay()
    capture_stream.synchronize()
    _assert_same_tokens(
        eager_reference,
        graph_output,
        label="CUDA Graph warmup replay and eager tokens differ",
    )

    graph_times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
        graph_times.append(_global_max_float(start.elapsed_time(end), device=device))
    _assert_same_tokens(
        eager_reference,
        graph_output,
        label="CUDA Graph timed replay and eager tokens differ",
    )
    token_prefix = _distributed_token_gate(graph_output)

    eager = timing_summary(eager_times)
    cuda_graph = timing_summary(graph_times)
    cache_bytes = _global_consensus_int(cache_bytes, device=device)
    return {
        "batch_size": batch_size,
        "fixed_context_length": context_length,
        "scope": "one complete fixed-context decode step",
        "includes": (
            "36 transformer layers + TP collectives + sharded LM head + "
            "distributed greedy selection"
        ),
        "excludes": "prefill, cache initialization, and host scheduler",
        "warmup_runs_per_mode": warmup,
        "timed_runs_per_mode": repeats,
        "eager": eager,
        "cuda_graph": cuda_graph,
        "graph_speedup_over_eager": eager["mean_ms"] / cuda_graph["mean_ms"],
        "cuda_graph_latency_reduction_vs_eager": (
            1.0 - cuda_graph["mean_ms"] / eager["mean_ms"]
        ),
        "cuda_graph_tokens_per_second": (
            batch_size * 1000.0 / cuda_graph["mean_ms"]
        ),
        "capture_build_seconds_excluded_from_latency": capture_build_seconds,
        "graph_memory": {
            "allocated_before_capture_bytes_per_rank": _global_max_int(
                allocated_before_capture,
                device=device,
            ),
            "allocated_after_capture_bytes_per_rank": _global_max_int(
                allocated_after_capture,
                device=device,
            ),
            "reserved_before_capture_bytes_per_rank": _global_max_int(
                reserved_before_capture,
                device=device,
            ),
            "reserved_after_capture_bytes_per_rank": _global_max_int(
                reserved_after_capture,
                device=device,
            ),
        },
        "memory": {
            "static_kv_cache_bytes_per_rank": cache_bytes,
            "peak_allocated_bytes_per_rank": _global_max_int(
                torch.cuda.max_memory_allocated(device),
                device=device,
            ),
            "peak_reserved_bytes_per_rank": _global_max_int(
                torch.cuda.max_memory_reserved(device),
                device=device,
            ),
        },
        "correctness": {
            "eager_replay_token_match": True,
            "capture_build_output_excluded_from_correctness": True,
            "tp_rank_token_consensus": True,
            "token_ids_prefix": token_prefix,
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Qwen3-8B Wo-only TP4 CUDA Graph: {payload['arm']}",
        "",
        "One fixed-context decode step; lower latency is better.",
        "",
        "| Batch | Context | Eager mean ms | Graph mean ms | Graph median ms | Graph p95 ms | Graph std ms | Graph tokens/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["records"]:
        eager = record["eager"]
        graph = record["cuda_graph"]
        lines.append(
            f"| {record['batch_size']} | {record['fixed_context_length']} | "
            f"{eager['mean_ms']:.6f} | {graph['mean_ms']:.6f} | "
            f"{graph['median_ms']:.6f} | {graph['p95_ms']:.6f} | "
            f"{graph['std_ms']:.6f} | "
            f"{record['cuda_graph_tokens_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Projection correctness: `{payload['correctness_gate']['status']}`",
            f"- Warmup runs per mode: `{payload['protocol']['warmup']}`",
            f"- Timed runs per mode: `{payload['protocol']['repeats']}`",
            "- Prefix cache: deterministic zeros, initialized outside timing",
            "- Capture/build time: reported separately and excluded",
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--quality-results", type=Path, required=True)
    parser.add_argument("--configurations", default=DEFAULT_CONFIGURATIONS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    configurations = parse_configurations(args.configurations)
    if min(args.warmup, args.repeats, args.torch_num_threads) <= 0:
        raise ValueError("warmup, repeats, and thread count must be positive")
    if args.warmup < 10 or args.repeats < 50:
        raise ValueError("paper protocol requires warmup >=10 and repeats >=50")
    if max(context for _, context in configurations) + 1 > 32768:
        raise ValueError("fixed context exceeds Qwen3-8B positional capacity")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    model_path = Path(args.model).expanduser().resolve()
    phase1_dir = args.phase1_dir.expanduser().resolve()
    quality_path = args.quality_results.expanduser().resolve()
    quality = _load_quality(
        quality_path,
        model_path=model_path,
        phase1_dir=phase1_dir,
        arm=args.arm,
    )

    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    from transformers import AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
    ).eval()
    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError("CUDA Graph benchmark requires torchrun TP4")
    modules, phase1 = install_qwen3_8b_wo_tp4_attention(
        model,
        arm=args.arm,
        model_path=model_path,
        phase1_dir=phase1_dir,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(
        time.perf_counter() - load_started,
        device=device,
    )
    correctness = _projection_correctness_gate(
        modules[0],
        arm=args.arm,
        device=device,
    )

    records = []
    for batch_size, context_length in configurations:
        record = _benchmark_configuration(
            model,
            modules,
            batch_size=batch_size,
            context_length=context_length,
            prompt_token_id=args.prompt_token_id,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        records.append(record)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "configuration_complete",
                        "arm": args.arm,
                        "batch_size": batch_size,
                        "context_length": context_length,
                        "graph_mean_ms": record["cuda_graph"]["mean_ms"],
                        "graph_tokens_per_second": record[
                            "cuda_graph_tokens_per_second"
                        ],
                    }
                ),
                flush=True,
            )

    if dist.get_rank() == 0:
        quality_name = "wo_c1_ag" if args.arm == "wo_c1_local_ar" else args.arm
        selected_quality = quality["arms"][quality_name]
        factor_records = [
            {
                "layer": module.layer_idx,
                "path": module.factor_path,
                "sha256": module.factor_sha256,
            }
            for module in modules
            if isinstance(
                module,
                (
                    Qwen3TP4WOC1DecodeAttention,
                    Qwen3TP4WOC1LocalAllReduceDecodeAttention,
                    Qwen3TP4WOLRAllReduceDecodeAttention,
                ),
            )
        ]
        payload = {
            "format": FORMAT,
            "schema_version": 1,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join((sys.executable, *sys.argv)),
            "git_commit": _git_commit(),
            "arm": args.arm,
            "model": {
                "path": str(model_path),
                "config_sha256": file_sha256(model_path / "config.json"),
            },
            "phase1": {
                "path": str(phase1_dir / "results.json"),
                "sha256": file_sha256(phase1_dir / "results.json"),
                "run_signature": phase1["method"]["run_signature"],
            },
            "quality": {
                "path": str(quality_path),
                "sha256": file_sha256(quality_path),
                "source_arm": quality_name,
                "wikitext2_ppl": selected_quality["wikitext2"]["ppl"],
                "c4_validation_ppl": selected_quality["c4_validation"]["ppl"],
            },
            "protocol": {
                "scope": "Qwen3-8B Wo-only TP4",
                "tp_size": TP_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "dtype": "bfloat16",
                "dense_v_and_kv_cache": True,
                "configurations": [
                    {"batch_size": batch, "fixed_context_length": context}
                    for batch, context in configurations
                ],
                "warmup": args.warmup,
                "repeats": args.repeats,
                "cuda_graph": True,
                "capture_build_excluded": True,
                "prefix_cache_initialization": "deterministic zeros outside timing",
                "prompt_token_id": args.prompt_token_id,
            },
            "correctness_gate": correctness,
            "communication": _communication(
                args.arm,
                layers=len(modules),
                dtype_bytes=2,
            ),
            "factors": factor_records,
            "records": records,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "world_size": dist.get_world_size(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "model_load_seconds": load_seconds,
                "nvidia_smi_topology": _command_output(("nvidia-smi", "topo", "-m")),
                "nvidia_smi": _command_output(
                    (
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    )
                ),
            },
        }
        output_dir.mkdir(parents=True)
        _atomic_text(
            output_dir / "results.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    dist.barrier()
    close_qwen3_8b_wo_tp4(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
