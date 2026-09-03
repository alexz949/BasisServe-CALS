#!/usr/bin/env python3
"""Run mapped-host exact-K attention through the full Qwen3-8B TP4 model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.core.qwen3_8b_tp4_decode import (  # noqa: E402
    TP_SIZE,
    close_qwen3_tp4_packed_communicator,
    configure_qwen3_tp4_caches,
)
from basisserve.core.qwen3_8b_tp4_k_offload import (  # noqa: E402
    Qwen3TP4C1KOffloadDecodeAttention,
    install_qwen3_tp4_k_offload_attention,
    tp4_k_offload_cache_bytes,
)
from basisserve.kernels.mapped_host_paged_attention import (  # noqa: E402
    prepare_mapped_host_paged_attention_extension,
)


FORMAT = "basisserve.qwen3_8b.tp4_mapped_host_exact_k.v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distributed_greedy(local_logits: Tensor) -> Tensor:
    world_size = dist.get_world_size()
    process_rank = dist.get_rank()
    batch, local_vocab = map(int, local_logits.shape)
    values, local_ids = torch.max(local_logits.float(), dim=-1)
    global_ids = local_ids.to(torch.int64) + process_rank * local_vocab
    gathered_values = torch.empty(
        world_size * batch,
        device=values.device,
        dtype=values.dtype,
    )
    gathered_ids = torch.empty(
        world_size * batch,
        device=values.device,
        dtype=torch.int64,
    )
    dist.all_gather_into_tensor(gathered_values, values.contiguous())
    dist.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
    candidates = gathered_values.view(world_size, batch)
    winning_rank = torch.argmax(candidates, dim=0, keepdim=True)
    return torch.gather(
        gathered_ids.view(world_size, batch),
        0,
        winning_rank,
    ).squeeze(0)


def _global_max_float(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _global_max_int(value: int, *, device: torch.device) -> int:
    tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _next_token(model: torch.nn.Module, hidden: Tensor) -> Tensor:
    local_logits = F.linear(hidden, model.lm_head.weight)
    return _distributed_greedy(local_logits)


def _prefill(
    model: torch.nn.Module,
    input_ids: Tensor,
) -> Tensor:
    positions = torch.arange(
        int(input_ids.shape[1]),
        dtype=torch.int64,
        device=input_ids.device,
    ).view(1, -1)
    output = model.model(
        input_ids=input_ids,
        position_ids=positions,
        use_cache=False,
    )
    return _next_token(model, output.last_hidden_state[:, -1])


def _decode_step(
    model: torch.nn.Module,
    token: Tensor,
    *,
    position: int,
) -> Tensor:
    positions = torch.full(
        (1, 1),
        int(position),
        dtype=torch.int64,
        device=token.device,
    )
    output = model.model(
        input_ids=token.view(-1, 1),
        position_ids=positions,
        use_cache=False,
    )
    return _next_token(model, output.last_hidden_state[:, -1])


def _summarize(values: Sequence[float]) -> dict[str, float]:
    selected = tuple(map(float, values))
    return {
        "mean": statistics.fmean(selected),
        "minimum": min(selected),
        "maximum": max(selected),
        "median": statistics.median(selected),
    }


def _module_offload_totals(
    modules: Sequence[Qwen3TP4C1KOffloadDecodeAttention],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for module in modules:
        for name, value in module.offload_statistics().items():
            totals[name] = totals.get(name, 0.0) + float(value)
    return totals


@torch.inference_mode()
def _run_once(
    model: torch.nn.Module,
    modules: Sequence[Qwen3TP4C1KOffloadDecodeAttention],
    *,
    batch_size: int,
    prompt_length: int,
    decode_length: int,
    prompt_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    for module in modules:
        module.reset_cache()
    prompt = torch.full(
        (batch_size, prompt_length),
        int(prompt_token_id),
        dtype=torch.int64,
        device=device,
    )
    dist.barrier()
    torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    token = _prefill(model, prompt)
    torch.cuda.synchronize(device)
    prefill_seconds = _global_max_float(
        time.perf_counter() - prefill_started,
        device=device,
    )

    dist.barrier()
    torch.cuda.synchronize(device)
    decode_started = time.perf_counter()
    for step in range(decode_length):
        token = _decode_step(
            model,
            token,
            position=prompt_length + step,
        )
    torch.cuda.synchronize(device)
    decode_seconds = _global_max_float(
        time.perf_counter() - decode_started,
        device=device,
    )
    gathered = torch.empty(
        TP_SIZE * batch_size,
        dtype=torch.int64,
        device=device,
    )
    dist.all_gather_into_tensor(gathered, token.contiguous())
    by_rank = gathered.view(TP_SIZE, batch_size)
    assert torch.equal(by_rank, by_rank[0].expand_as(by_rank))
    return {
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": batch_size * prompt_length / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": batch_size * decode_length / decode_seconds,
        "decode_step_ms": 1000.0 * decode_seconds / decode_length,
        "final_token_ids_prefix": by_rank[0, : min(batch_size, 8)].cpu().tolist(),
        "offload_per_rank": _module_offload_totals(modules),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--c1-factor-dir", type=Path, required=True)
    parser.add_argument("--routing-factor-dir", type=Path)
    parser.add_argument(
        "--router",
        choices=("c1_base16_r8", "quest", "shadowkv"),
        default="c1_base16_r8",
    )
    parser.add_argument(
        "--exact-key-storage",
        choices=("mapped_host", "gpu"),
        default="mapped_host",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--decode-length", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--physical-token-budget", type=int, default=4096)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument(
        "--c1-allgather-backend",
        choices=("feature_direct", "uniform_nccl"),
        default="uniform_nccl",
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    assert args.batch_size > 0
    assert args.prompt_length > 1
    assert args.decode_length > 0
    assert args.warmup_runs >= 0
    assert args.repeat_runs > 0
    assert args.page_size > 0
    assert args.physical_token_budget % args.page_size == 0
    assert args.prompt_length + args.decode_length <= 32768
    assert (args.routing_factor_dir is not None) == (
        args.router == "c1_base16_r8"
    )
    assert args.exact_key_storage == "mapped_host" or args.router == "c1_base16_r8"
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    prepare_mapped_host_paged_attention_extension()

    from transformers import AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model.expanduser().resolve(),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
    ).eval()
    assert dist.is_initialized() and dist.get_world_size() == TP_SIZE
    modules = install_qwen3_tp4_k_offload_attention(
        model,
        c1_factor_dir=args.c1_factor_dir,
        routing_factor_dir=args.routing_factor_dir,
        routing_mode=args.router,
        exact_key_storage=args.exact_key_storage,
        base_rank=args.base_rank,
        residual_rank=args.residual_rank,
        page_size=args.page_size,
        exact_token_budget=args.physical_token_budget,
        pinned_prefix_pages=args.pinned_prefix_pages,
        allgather_backend=args.c1_allgather_backend,
    )
    capacity = args.prompt_length + args.decode_length
    configure_qwen3_tp4_caches(
        modules,
        batch_size=args.batch_size,
        capacity=capacity,
        max_forward_tokens=args.prompt_length,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(
        time.perf_counter() - load_started,
        device=device,
    )
    cache_bytes = tp4_k_offload_cache_bytes(modules)
    cache_bytes = {
        name: _global_max_int(value, device=device)
        for name, value in cache_bytes.items()
    }

    for _ in range(args.warmup_runs):
        _run_once(
            model,
            modules,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            decode_length=args.decode_length,
            prompt_token_id=args.prompt_token_id,
            device=device,
        )
    records = [
        _run_once(
            model,
            modules,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            decode_length=args.decode_length,
            prompt_token_id=args.prompt_token_id,
            device=device,
        )
        for _ in range(args.repeat_runs)
    ]

    if dist.get_rank() == 0:
        payload = {
            "format": FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "protocol": {
                "model": str(args.model.expanduser().resolve()),
                "tp_size": TP_SIZE,
                "dtype": "bfloat16",
                "batch_size": args.batch_size,
                "prompt_length": args.prompt_length,
                "decode_length": args.decode_length,
                "page_size": args.page_size,
                "physical_exact_k_tokens_per_local_kv_head": (
                    args.physical_token_budget
                ),
                "pinned_prefix_pages": args.pinned_prefix_pages,
                "force_current_page": True,
                "routing": args.router,
                "routing_kernel": (
                    "fused_cuda_cached_base16_r8_page32_lse_with_decode_append"
                ),
                "page_selection_kernel": (
                    "fused_cuda_softmax_gqa_max_topk_sorted"
                ),
                "exact_k_storage": args.exact_key_storage,
                "staging": (
                    "GPU page IDs + fused mapped-host exact QK/softmax/V80; "
                    "no CPU page pack or explicit H2D staging"
                ),
                "resident_gpu_cache": (
                    "C1-V80 + cached Base16 + residual "
                    f"R{args.residual_rank} + layer-shared RoPE metadata"
                ),
                "collective": (
                    "unchanged full-batch BF16 C1 latent AllGather followed by "
                    "the complete decoder"
                ),
                "c1_allgather_backend": args.c1_allgather_backend,
                "warmup_runs": args.warmup_runs,
                "repeat_runs": args.repeat_runs,
                "prompt": "repeated-token synthetic full-model workload",
            },
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "model_load_and_cache_setup_seconds": load_seconds,
            },
            "artifacts": {
                "c1_factor_dir": str(args.c1_factor_dir.expanduser().resolve()),
                "routing_factor_dir": str(
                    args.routing_factor_dir.expanduser().resolve()
                ) if args.routing_factor_dir is not None else None,
                "c1_factor_sha256_by_layer": [
                    module.factor_sha256 for module in modules
                ],
                "routing_factor_sha256_by_layer": [
                    module.router_factor_sha256 for module in modules
                ],
            },
            "cache": cache_bytes,
            "summary": {
                "prefill_seconds": _summarize(
                    [record["prefill_seconds"] for record in records]
                ),
                "prefill_tokens_per_second": _summarize(
                    [record["prefill_tokens_per_second"] for record in records]
                ),
                "decode_seconds": _summarize(
                    [record["decode_seconds"] for record in records]
                ),
                "decode_tokens_per_second": _summarize(
                    [record["decode_tokens_per_second"] for record in records]
                ),
                "decode_step_ms": _summarize(
                    [record["decode_step_ms"] for record in records]
                ),
            },
            "records": records,
        }
        _atomic_json(args.output_json.expanduser().resolve(), payload)
        print(
            json.dumps(
                {
                    "event": "result_written",
                    "path": str(args.output_json.expanduser().resolve()),
                    "decode_tokens_per_second": payload["summary"][
                        "decode_tokens_per_second"
                    ]["mean"],
                }
            ),
            flush=True,
        )
    dist.barrier()
    close_qwen3_tp4_packed_communicator(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
