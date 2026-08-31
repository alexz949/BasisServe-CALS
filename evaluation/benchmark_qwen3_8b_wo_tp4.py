#!/usr/bin/env python3
"""Benchmark actual Qwen3-8B TP4 Wo collectives end to end on four GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Sequence


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
    C1_LOCAL_RANK,
    LR_CAPACITY_RANK,
    LR_SHARED_RANK,
    Qwen3TP4WOC1DecodeAttention,
    Qwen3TP4WOC1LocalAllReduceDecodeAttention,
    Qwen3TP4WOLRAllReduceDecodeAttention,
    close_qwen3_8b_wo_tp4,
    install_qwen3_8b_wo_tp4_attention,
)
from evaluation.benchmark_qwen3_8b_tp4_decode import (  # noqa: E402
    _global_max_float,
    _global_max_int,
    _run_configuration as _run_decode_configuration,
)
from evaluation.benchmark_qwen3_8b_tp4_prefill import (  # noqa: E402
    _run_configuration as _run_prefill_configuration,
)


FORMAT = "basisserve.qwen3_8b.wo_tp4_runtime_benchmark.v1"
QUALITY_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_quality.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("decode", "prefill"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--quality-results", type=Path, required=True)
    parser.add_argument(
        "--configurations",
        required=True,
        help="comma-separated batch x length pairs, for example 1x128,32x1024",
    )
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _parse_configurations(raw: str) -> tuple[tuple[int, int], ...]:
    parsed = []
    for item in raw.split(","):
        fields = item.strip().lower().split("x")
        if len(fields) != 2:
            raise ValueError(f"invalid batch x length pair {item!r}")
        pair = tuple(map(int, fields))
        if min(pair) <= 0:
            raise ValueError("batch and length must be positive")
        parsed.append(pair)
    result = tuple(parsed)
    if not result or len(result) != len(set(result)):
        raise ValueError("configurations must be nonempty and unique")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_quality(
    path: Path,
    *,
    model_path: Path,
    phase1_dir: Path,
    arm: str,
) -> dict[str, Any]:
    quality = json.loads(path.read_text(encoding="utf-8"))
    if quality.get("format") != QUALITY_FORMAT or quality.get("status") != "complete":
        raise ValueError("whole-model quality result is incomplete or incompatible")
    if quality["model"]["config_sha256"] != file_sha256(model_path / "config.json"):
        raise ValueError("quality result belongs to another model")
    phase1_path = phase1_dir / "results.json"
    if quality["phase1"]["sha256"] != file_sha256(phase1_path):
        raise ValueError("quality result and runtime use different Phase-1 factors")
    quality_arm = "wo_c1_ag" if arm == "wo_c1_local_ar" else arm
    if quality_arm not in quality["arms"]:
        raise ValueError(f"quality result has no arm {quality_arm}")
    return quality


def _configuration_metadata(
    workload: str,
    configurations: Sequence[tuple[int, int]],
) -> list[dict[str, int]]:
    length_key = "decode_length" if workload == "decode" else "prompt_length"
    return [
        {"batch_size": batch, length_key: length}
        for batch, length in configurations
    ]


@torch.inference_mode()
def _projection_correctness_gate(
    module: torch.nn.Module,
    *,
    arm: str,
    device: torch.device,
) -> dict[str, Any]:
    """Compare the custom collective with a plain PyTorch distributed reference."""

    if arm == "dense":
        return {"status": "not_applicable_dense_reference"}
    generator = torch.Generator(device=device).manual_seed(20260828 + dist.get_rank())
    local_input = torch.randn(
        3,
        HIDDEN_SIZE // TP_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    if isinstance(module, Qwen3TP4WOC1DecodeAttention):
        projection = module.o_proj
        projection.communicator.configure_direct_workspace(
            tokens=3,
            max_total_width=projection.gathered_rank,
            dtype=local_input.dtype,
        )
        projection.prepare(3)
        actual = projection(local_input)
        local_latent = F.linear(local_input, projection.local_projection_weight)
        parts = [torch.empty_like(local_latent) for _ in range(TP_SIZE)]
        dist.all_gather(parts, local_latent)
        expected = torch.cat(parts, dim=1) @ projection.global_decoder
    elif isinstance(module, Qwen3TP4WOC1LocalAllReduceDecodeAttention):
        projection = module.o_proj
        actual = projection(local_input)
        local_latent = F.linear(local_input, projection.local_projection_weight)
        expected = F.linear(local_latent, projection.local_decoder_weight)
        dist.all_reduce(expected)
    elif isinstance(module, Qwen3TP4WOLRAllReduceDecodeAttention):
        projection = module.o_proj
        actual = projection(local_input)
        reduced = F.linear(local_input, projection.local_projection_weight)
        dist.all_reduce(reduced)
        expected = F.linear(reduced, projection.output_basis_weight)
    else:
        raise TypeError("compressed arm installed an unexpected attention module")
    difference = (actual.float() - expected.float()).abs()
    maximum_absolute_error = _global_max_float(
        float(difference.max()),
        device=device,
    )
    denominator = expected.float().abs().max().clamp_min(1.0e-8)
    maximum_relative_error = _global_max_float(
        float(difference.max() / denominator),
        device=device,
    )
    if maximum_relative_error > 5.0e-3:
        raise AssertionError(
            f"{arm} projection correctness gate failed: {maximum_relative_error}"
        )
    return {
        "status": "passed",
        "rows": 3,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "reference": (
            "torch.distributed all_gather + source-ordered decoder"
            if arm == "wo_c1_ag"
            else (
                "source-private encoder/decoder + hidden all_reduce"
                if arm == "wo_c1_local_ar"
                else "torch.distributed all_reduce + shared decoder"
            )
        ),
    }


def _communication(arm: str, *, layers: int, dtype_bytes: int) -> dict[str, Any]:
    dense_bytes = 2 * (TP_SIZE - 1) * HIDDEN_SIZE * dtype_bytes / TP_SIZE
    if arm == "dense":
        bytes_per_row = dense_bytes
        collective = "full-output NCCL AllReduce"
        wire_width = HIDDEN_SIZE
    elif arm == "wo_c1_ag":
        bytes_per_row = (TP_SIZE - 1) * C1_LOCAL_RANK * dtype_bytes
        collective = "packed feature-major NCCL AllGather"
        wire_width = C1_LOCAL_RANK
    elif arm == "wo_c1_local_ar":
        bytes_per_row = dense_bytes
        collective = "C1 local decoder followed by full-output NCCL AllReduce"
        wire_width = HIDDEN_SIZE
    else:
        selected_rank = (
            LR_SHARED_RANK
            if arm == "wo_lr_ar_wire"
            else LR_CAPACITY_RANK
        )
        bytes_per_row = (
            2 * (TP_SIZE - 1) * selected_rank * dtype_bytes / TP_SIZE
        )
        collective = "latent NCCL AllReduce"
        wire_width = selected_rank
    return {
        "collective": collective,
        "wire_width": wire_width,
        "dtype": "bfloat16",
        "dtype_bytes": dtype_bytes,
        "ideal_ring_bytes_per_rank_per_activation_row_per_layer": bytes_per_row,
        "ideal_ring_bytes_per_rank_per_activation_row_all_layers": (
            bytes_per_row * layers
        ),
        "reduction_vs_dense": 1.0 - bytes_per_row / dense_bytes,
    }


def main() -> None:
    args = parse_args()
    configurations = _parse_configurations(args.configurations)
    positive = (
        args.output_tokens,
        args.warmup,
        args.repeats,
        args.torch_num_threads,
    )
    if min(positive) <= 0:
        raise ValueError("output, warmup, repeat, and thread counts must be positive")
    if args.workload == "prefill" and args.output_tokens < 2:
        raise ValueError("prefill workload requires at least two output tokens")
    maximum_position = max(
        length + (args.output_tokens - 1 if args.workload == "prefill" else 1)
        for _, length in configurations
    )
    if maximum_position > 32768:
        raise ValueError("configuration exceeds Qwen3-8B positional capacity")

    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
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
        raise RuntimeError("benchmark must run under torchrun TP4")
    modules, phase1 = install_qwen3_8b_wo_tp4_attention(
        model,
        arm=args.arm,
        model_path=model_path,
        phase1_dir=phase1_dir,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(time.perf_counter() - load_started, device=device)
    correctness = _projection_correctness_gate(
        modules[0],
        arm=args.arm,
        device=device,
    )

    local_vocab = int(model.lm_head.weight.shape[0])
    vocab_size = int(model.config.vocab_size)
    if local_vocab * TP_SIZE != vocab_size:
        raise ValueError("LM head is not an equal TP4 vocabulary shard")
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) + sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
        if buffer is not None
    )
    model_bytes = _global_max_int(model_bytes, device=device)

    records = []
    for batch, length in configurations:
        if args.workload == "decode":
            record = _run_decode_configuration(
                model,
                modules,
                batch=batch,
                decode_length=length,
                prompt_token_id=args.prompt_token_id,
                warmup_tokens=args.warmup,
                device=device,
            )
            event = {
                "event": "configuration_complete",
                "workload": args.workload,
                "arm": args.arm,
                "batch": batch,
                "decode_length": length,
                "tokens_per_second": record["generated_tokens_per_second"],
                "mean_step_ms": record["mean_critical_step_ms_from_total"],
            }
        else:
            record = _run_prefill_configuration(
                model,
                modules,
                batch=batch,
                prompt_length=length,
                output_tokens=args.output_tokens,
                warmup_runs=args.warmup,
                repeat_runs=args.repeats,
                vocab_size=vocab_size,
                device=device,
            )
            event = {
                "event": "configuration_complete",
                "workload": args.workload,
                "arm": args.arm,
                "batch": batch,
                "prompt_length": length,
                "prefill_tokens_per_second": record["prefill_tokens_per_second"],
                "end_to_end_ms": record["end_to_end"]["mean_ms"],
            }
        records.append(record)
        if dist.get_rank() == 0:
            print(json.dumps(event), flush=True)

    if dist.get_rank() == 0:
        quality_name = "wo_c1_ag" if args.arm == "wo_c1_local_ar" else args.arm
        quality_arm = quality["arms"][quality_name]
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
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "workload": args.workload,
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
                "wikitext2_ppl": quality_arm["wikitext2"]["ppl"],
                "c4_validation_ppl": quality_arm["c4_validation"]["ppl"],
                "source_arm": quality_name,
            },
            "protocol": {
                "tp_size": TP_SIZE,
                "dtype": "bfloat16",
                "dense_qkv_and_kv_cache": True,
                "workload": args.workload,
                "configurations": _configuration_metadata(
                    args.workload,
                    configurations,
                ),
                "output_tokens": (
                    args.output_tokens if args.workload == "prefill" else None
                ),
                "warmup": args.warmup,
                "repeats": (
                    args.repeats if args.workload == "prefill" else None
                ),
                "timed_scope": (
                    "transformer + TP collectives + sharded LM head + "
                    "distributed greedy selection"
                ),
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
                "model_load_seconds": load_seconds,
                "model_and_factor_bytes_per_rank": model_bytes,
            },
        }
        _atomic_json(output_path, payload)
        print(json.dumps({"event": "result_written", "path": str(output_path)}), flush=True)
    dist.barrier()
    close_qwen3_8b_wo_tp4(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
