#!/usr/bin/env python3
"""Evaluate TP4 mapped-host exact-K routing on fixed hard RULER samples."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence


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
from evaluation.eval_qwen3_c1_quest_ruler import (  # noqa: E402
    _eos_ids,
    _load_dataset_manifest,
)
from evaluation.eval_qwen3_dense_ruler import _qwen3_config  # noqa: E402
from evaluation.ruler_v1 import (  # noqa: E402
    load_task_records,
    parse_tasks,
    ruler_prompt,
    sample_score,
)


FORMAT = "basisserve.qwen3_8b.tp4_mapped_host_ruler.v1"
DEFAULT_TASKS = (
    "niah_multikey_2,niah_multivalue,niah_single_2,niah_single_3,fwe"
)
DEFAULT_SAMPLE_SPEC = (
    "niah_multikey_2:7,niah_multivalue:3,niah_single_2:3,"
    "niah_single_3:3,fwe:2"
)
ROUTER_LABELS = {
    "c1_base16_r8": "C1 Base16+R8 Page32 group-max",
    "quest": "QUEST Min/Max Page32 group-max",
    "shadowkv": "ShadowKV-style mean-landmark Page32 group-max",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _distributed_greedy(local_logits: Tensor) -> Tensor:
    process_rank = dist.get_rank()
    batch, local_vocab = map(int, local_logits.shape)
    values, local_ids = torch.max(local_logits.float(), dim=-1)
    global_ids = local_ids.to(torch.int64) + process_rank * local_vocab
    gathered_values = torch.empty(
        TP_SIZE * batch,
        device=values.device,
        dtype=values.dtype,
    )
    gathered_ids = torch.empty(
        TP_SIZE * batch,
        device=values.device,
        dtype=torch.int64,
    )
    dist.all_gather_into_tensor(gathered_values, values.contiguous())
    dist.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
    winning_rank = torch.argmax(
        gathered_values.view(TP_SIZE, batch),
        dim=0,
        keepdim=True,
    )
    return torch.gather(
        gathered_ids.view(TP_SIZE, batch),
        0,
        winning_rank,
    ).squeeze(0)


def _next_token(model: torch.nn.Module, hidden: Tensor) -> Tensor:
    return _distributed_greedy(F.linear(hidden, model.lm_head.weight))


def _prefill(model: torch.nn.Module, input_ids: Tensor) -> Tensor:
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
        input_ids=token.view(1, 1),
        position_ids=positions,
        use_cache=False,
    )
    return _next_token(model, output.last_hidden_state[:, -1])


def _global_max_float(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _global_cache_bytes(
    modules: Sequence[Qwen3TP4C1KOffloadDecodeAttention],
    *,
    device: torch.device,
) -> dict[str, int]:
    local = tp4_k_offload_cache_bytes(modules)
    result: dict[str, int] = {}
    for name, value in local.items():
        maximum = torch.tensor(int(value), dtype=torch.int64, device=device)
        total = maximum.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        result[f"{name}_maximum"] = int(maximum.item())
        result[f"{name}_total"] = int(total.item())
    return result


def _global_offload_statistics(
    modules: Sequence[Qwen3TP4C1KOffloadDecodeAttention],
    *,
    device: torch.device,
) -> dict[str, float]:
    names = (
        "decode_calls",
        "selected_pages",
        "exact_key_bytes_direct_read",
        "exact_key_bytes_d2h",
    )
    local = {name: 0.0 for name in names}
    for module in modules:
        for name, value in module.offload_statistics().items():
            local[name] += float(value)
    values = torch.tensor(
        [local[name] for name in names],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: float(values[index].item()) for index, name in enumerate(names)}


def _sample_rows(
    data_dir: Path,
    *,
    tasks_raw: str,
    samples_per_task: int,
    sample_spec: str,
) -> tuple[tuple[Any, int, dict[str, Any]], ...]:
    tasks = parse_tasks(tasks_raw)
    requested = {
        (task.strip(), int(ordinal))
        for item in sample_spec.split(",")
        for task, ordinal in (item.rsplit(":", 1),)
    }
    rows: list[tuple[Any, int, dict[str, Any]]] = []
    for task in tasks:
        records = load_task_records(data_dir, task, samples_per_task)
        for ordinal, record in enumerate(records):
            if (task.name, ordinal) in requested:
                rows.append((task, ordinal, record))
    assert len(rows) == len(requested)
    assert {(task.name, ordinal) for task, ordinal, _ in rows} == requested
    return tuple(rows)


def _load_oracle(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT and payload["status"] == "complete"
    return {
        str(record["key"]): list(map(int, record["generated_token_ids"]))
        for record in payload["records"]
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    summary = payload["summary"]
    cache = payload["cache"]
    lines = [
        f"# Qwen3-8B TP4 {protocol['router_label']} RULER",
        "",
        (
            f"Exact K storage: `{protocol['exact_key_storage']}`; Page32; "
            f"strict physical B{protocol['physical_token_budget_per_kv_head']} "
            "per KV head."
        ),
        "",
        "| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for record in payload["records"]:
        lines.append(
            f"| {record['key']} | {100.0 * record['score']:.2f}% | "
            f"{record['prompt_tokens']} | {record['generated_tokens']} | "
            f"{record['prefill_tokens_per_second']:.2f} | "
            f"{record['decode_tokens_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Task-balanced hard-5 accuracy: `{100.0 * summary['task_balanced_accuracy']:.2f}%`.",
            f"Aggregate decode throughput: `{summary['decode_tokens_per_second']:.4f} tok/s`.",
            f"Aggregate end-to-end model throughput: `{summary['end_to_end_model_tokens_per_second']:.2f} tok/s`.",
            (
                "Requested physical exact-K reads: "
                f"`{summary['physical_exact_k_gib_read']:.3f} GiB`, "
                f"`{summary['physical_k_vector_tokens_fetched']:.0f}` K-vector tokens."
            ),
            (
                "Requested physical exact-K read per decode step: "
                f"`{summary['physical_exact_k_gib_per_decode_step']:.5f} GiB`."
            ),
            (
                "GPU-resident runtime cache: "
                f"`{cache['gpu_bytes_per_rank_maximum'] / 2**30:.3f} GiB/rank`; "
                "mapped-host exact-K cache: "
                f"`{cache['cpu_pinned_bytes_per_rank_maximum'] / 2**30:.3f} GiB/rank`."
            ),
        ]
    )
    if summary["quality_oracle_samples"]:
        lines.append(
            "GPU-oracle token-sequence agreement: "
            f"`{summary['quality_oracle_sequence_matches']}/"
            f"{summary['quality_oracle_samples']}`."
        )
    lines.extend(
        [
            "",
            "Physical fetch is the requested Page32 exact-K payload read by the CUDA kernel; it is not a PCIe hardware-counter measurement.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--c1-factor-dir", type=Path, required=True)
    parser.add_argument("--routing-factor-dir", type=Path)
    parser.add_argument(
        "--router",
        choices=tuple(ROUTER_LABELS),
        required=True,
    )
    parser.add_argument(
        "--exact-key-storage",
        choices=("mapped_host", "gpu"),
        default="mapped_host",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--sample-spec", default=DEFAULT_SAMPLE_SPEC)
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--physical-token-budget", type=int, default=4096)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--warmup-prompt-tokens", type=int, default=128)
    parser.add_argument("--quality-oracle-json", type=Path)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    assert args.sequence_length > 0
    assert args.samples_per_task > 0
    assert args.page_size == 32
    assert args.physical_token_budget % args.page_size == 0
    assert args.pinned_prefix_pages >= 0
    assert (args.routing_factor_dir is not None) == (
        args.router == "c1_base16_r8"
    )
    assert args.exact_key_storage == "mapped_host" or args.router == "c1_base16_r8"
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    prepare_mapped_host_paged_attention_extension()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.distributed import DistributedConfig

    model_path = args.model.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    effective_config, rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=args.yarn_factor,
    )
    tasks = parse_tasks(args.tasks)
    dataset_manifest, dataset_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    rows = _sample_rows(
        data_dir,
        tasks_raw=args.tasks,
        samples_per_task=args.samples_per_task,
        sample_spec=args.sample_spec,
    )
    oracle = _load_oracle(args.quality_oracle_json)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=effective_config,
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
        allgather_backend="uniform_nccl",
    )
    configure_qwen3_tp4_caches(
        modules,
        batch_size=1,
        capacity=args.sequence_length,
        max_forward_tokens=args.sequence_length,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(
        time.perf_counter() - load_started,
        device=device,
    )
    cache = _global_cache_bytes(modules, device=device)

    warmup = torch.full(
        (1, args.warmup_prompt_tokens),
        1,
        dtype=torch.int64,
        device=device,
    )
    for module in modules:
        module.reset_cache()
        module.configure_uniform_allgather(
            tokens=args.warmup_prompt_tokens,
            dtype=torch.bfloat16,
        )
    warmup_token = _prefill(model, warmup)
    _decode_step(model, warmup_token, position=args.warmup_prompt_tokens)
    torch.cuda.synchronize(device)
    del warmup, warmup_token

    records: list[dict[str, Any]] = []
    eos_ids = _eos_ids(tokenizer, model)
    for task, ordinal, source in rows:
        key = f"{task.name}:{ordinal}"
        input_ids = tokenizer(
            ruler_prompt(source),
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        prompt_tokens = int(input_ids.shape[1])
        assert prompt_tokens + task.tokens_to_generate <= args.sequence_length
        for module in modules:
            module.reset_cache()
            module.configure_uniform_allgather(
                tokens=prompt_tokens,
                dtype=torch.bfloat16,
            )

        dist.barrier()
        torch.cuda.synchronize(device)
        prefill_started = time.perf_counter()
        token = _prefill(model, input_ids)
        torch.cuda.synchronize(device)
        prefill_seconds = _global_max_float(
            time.perf_counter() - prefill_started,
            device=device,
        )
        generated_ids = [int(token.item())]

        dist.barrier()
        torch.cuda.synchronize(device)
        decode_started = time.perf_counter()
        while (
            len(generated_ids) < task.tokens_to_generate
            and generated_ids[-1] not in eos_ids
        ):
            token = _decode_step(
                model,
                token,
                position=prompt_tokens + len(generated_ids) - 1,
            )
            generated_ids.append(int(token.item()))
        torch.cuda.synchronize(device)
        decode_seconds = _global_max_float(
            time.perf_counter() - decode_started,
            device=device,
        )
        runtime = _global_offload_statistics(modules, device=device)
        prediction = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decode_steps = max(len(generated_ids) - 1, 0)
        oracle_ids = oracle.get(key)
        sequence_match = None if oracle_ids is None else oracle_ids == generated_ids
        record = {
            "key": key,
            "task": task.name,
            "sample_ordinal": ordinal,
            "source_index": source.get("index"),
            "declared_length": source.get("length"),
            "prompt_tokens": prompt_tokens,
            "references": list(source["outputs"]),
            "prediction": prediction,
            "generated_token_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "decode_steps": decode_steps,
            "stopped_on_eos": generated_ids[-1] in eos_ids,
            "score": sample_score(prediction, source["outputs"], task.match_type),
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": prompt_tokens / prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                decode_steps / decode_seconds if decode_steps else 0.0
            ),
            "end_to_end_seconds": prefill_seconds + decode_seconds,
            "quality_oracle_sequence_match": sequence_match,
            "runtime_global": runtime,
        }
        records.append(record)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        "router": args.router,
                        "storage": args.exact_key_storage,
                        "key": key,
                        "score": record["score"],
                        "prefill_seconds": prefill_seconds,
                        "decode_seconds": decode_seconds,
                        "decode_tokens_per_second": record[
                            "decode_tokens_per_second"
                        ],
                        "oracle_match": sequence_match,
                    }
                ),
                flush=True,
            )
        del input_ids, token

    total_prefill_tokens = sum(record["prompt_tokens"] for record in records)
    total_decode_steps = sum(record["decode_steps"] for record in records)
    total_generated_tokens = sum(record["generated_tokens"] for record in records)
    total_prefill_seconds = sum(record["prefill_seconds"] for record in records)
    total_decode_seconds = sum(record["decode_seconds"] for record in records)
    total_seconds = total_prefill_seconds + total_decode_seconds
    physical_bytes = sum(
        record["runtime_global"]["exact_key_bytes_direct_read"]
        for record in records
    )
    selected_pages = sum(
        record["runtime_global"]["selected_pages"] for record in records
    )
    oracle_rows = [
        record
        for record in records
        if record["quality_oracle_sequence_match"] is not None
    ]
    summary = {
        "task_balanced_accuracy": sum(record["score"] for record in records)
        / len(records),
        "prefill_tokens_per_second": total_prefill_tokens / total_prefill_seconds,
        "decode_tokens_per_second": (
            total_decode_steps / total_decode_seconds if total_decode_steps else 0.0
        ),
        "output_tokens_per_second_including_prefill": (
            total_generated_tokens / total_seconds
        ),
        "end_to_end_model_tokens_per_second": (
            (total_prefill_tokens + total_decode_steps) / total_seconds
        ),
        "prefill_seconds": total_prefill_seconds,
        "decode_seconds": total_decode_seconds,
        "decode_steps": total_decode_steps,
        "end_to_end_seconds": total_seconds,
        "selected_physical_pages": selected_pages,
        "physical_k_vector_tokens_fetched": selected_pages * args.page_size,
        "physical_exact_k_bytes_read": physical_bytes,
        "physical_exact_k_gib_read": physical_bytes / 2**30,
        "physical_exact_k_gib_per_decode_step": (
            physical_bytes / total_decode_steps / 2**30
            if total_decode_steps
            else 0.0
        ),
        "quality_oracle_samples": len(oracle_rows),
        "quality_oracle_sequence_matches": sum(
            record["quality_oracle_sequence_match"] is True
            for record in oracle_rows
        ),
    }

    if dist.get_rank() == 0:
        payload = {
            "format": FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "protocol": {
                "model": str(model_path),
                "tp_size": TP_SIZE,
                "dtype": "bfloat16",
                "sequence_length": args.sequence_length,
                "tasks": [task.name for task in tasks],
                "sample_spec": args.sample_spec,
                "router": args.router,
                "router_label": ROUTER_LABELS[args.router],
                "exact_key_storage": args.exact_key_storage,
                "page_size": args.page_size,
                "physical_token_budget_per_kv_head": args.physical_token_budget,
                "pinned_prefix_pages": args.pinned_prefix_pages,
                "force_current_page": True,
                "routing_aggregation": (
                    "normalize page mass per Query head, then max over four "
                    "heads in each physical GQA group"
                ),
                "routing_kernel": (
                    "fused CUDA cached-Base16->RoPE QK + R8 Page32 LSE; "
                    "fused decode append and layer-shared RoPE metadata"
                    if args.router == "c1_base16_r8"
                    else "PyTorch"
                ),
                "page_selection_kernel": (
                    "fused CUDA softmax + four-Q-head max + Top-K + ID sort"
                ),
                "value_cache": "C1-V80 BF16 GPU resident",
                "exact_attention": (
                    "shared CUDA Page32 exact-QK online-softmax V80 kernel"
                ),
                "c1_collective": "TP4 BF16 latent AllGather",
                "rope_scaling": rope_scaling,
                "first_generated_token_from_dense_c1_prefill": True,
                "shadowkv_scope": (
                    "Page32 post-RoPE mean-landmark selector only; excludes "
                    "ShadowKV chunk8 outlier/local caches and online SVD payload"
                    if args.router == "shadowkv"
                    else None
                ),
            },
            "environment": {
                "conda_environment": (
                    os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
                ),
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "model_load_and_cache_setup_seconds": load_seconds,
                "maximum_cuda_memory_bytes_per_rank": torch.cuda.max_memory_allocated(
                    device
                ),
            },
            "dataset": {
                "directory": str(data_dir),
                "manifest_sha256": dataset_hash,
                "ruler_revision": dataset_manifest["ruler"]["revision"],
            },
            "artifacts": {
                "c1_factor_dir": str(args.c1_factor_dir.expanduser().resolve()),
                "routing_factor_dir": (
                    None
                    if args.routing_factor_dir is None
                    else str(args.routing_factor_dir.expanduser().resolve())
                ),
                "quality_oracle_json": (
                    None
                    if args.quality_oracle_json is None
                    else str(args.quality_oracle_json.expanduser().resolve())
                ),
                "c1_factor_sha256_by_layer": [
                    module.factor_sha256 for module in modules
                ],
                "routing_factor_sha256_by_layer": [
                    module.router_factor_sha256 for module in modules
                ],
            },
            "cache": cache,
            "summary": summary,
            "records": records,
        }
        output_json = args.output_json.expanduser().resolve()
        _atomic_json(output_json, payload)
        _atomic_text(output_json.with_suffix(".md"), _markdown(payload))
        print(
            json.dumps(
                {
                    "event": "result_written",
                    "path": str(output_json),
                    "accuracy": summary["task_balanced_accuracy"],
                    "decode_tokens_per_second": summary[
                        "decode_tokens_per_second"
                    ],
                }
            ),
            flush=True,
        )
    dist.barrier()
    close_qwen3_tp4_packed_communicator(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
