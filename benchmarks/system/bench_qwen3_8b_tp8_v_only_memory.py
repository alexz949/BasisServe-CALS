"""TP8 decode-ready GPU memory for Qwen3 Basis V64 and STAR-KV V-only."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import pynvml
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.distributed import DistributedConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.core.qwen3_tp8_v_only import (
    NUM_LAYERS,
    SKIP_LAYERS,
    TP_SIZE,
    VALUE_RANK,
    StarQwen3ForCausalLM,
    install_v_only,
    local_weight,
    star_tp_plan,
)
from benchmarks.system.bench_llama31_8b_tp8_combined import _bind_cpu
from benchmarks.system.common import command
from benchmarks.system.numa_memory import bind_host_allocations


DEFAULT_MODEL = Path(
    "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/"
    "snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
)
DEFAULT_STAR = Path(
    "/workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/"
    "snapshots/0ef83dff27205b131c82df6d62636129e9dac7b9/"
    "ICLR-results/qwen3-8b/star-v50-adaptive/full"
)
DEFAULT_FACTORS = Path(
    "/workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/"
    "snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/"
    "ICLR-results/qwen3-8b/c1/factor-banks/R64-S6"
)


def memory_snapshot(nvml_handle) -> dict[str, int]:
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(nvml_handle)
    own = [item for item in processes if item.pid == os.getpid()]
    assert len(own) == 1
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nvml_process_bytes": int(own[0].usedGpuMemory),
    }


def choose(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def select_prompts(input_ids: torch.Tensor, batch: int, length: int, repeat: bool) -> torch.Tensor:
    source = input_ids[:, :length].long()
    assert source.ndim == 2 and source.shape[0] > 0 and source.shape[1] == length
    if repeat:
        return source.repeat((batch + source.shape[0] - 1) // source.shape[0], 1)[:batch]
    return source[:batch]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("basis_v64", "star_v_adaptive"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--cohort", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--reserve-decode-tokens", type=int, default=1)
    parser.add_argument("--repeat-prompts", action="store_true")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--star-checkpoint", type=Path, default=DEFAULT_STAR)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTORS)
    args = parser.parse_args()
    assert args.batch > 0 and args.length > 0 and 0 < args.chunk_size <= args.length
    assert args.output_tokens >= 2 and args.reserve_decode_tokens >= args.output_tokens - 1
    assert args.tokens.is_file() and args.prompt_manifest.is_file()
    assert args.model.is_dir() and args.star_checkpoint.is_dir() and args.factor_root.is_dir()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == TP_SIZE
    numa_node = _bind_cpu(local_rank)
    bind_host_allocations(numa_node)
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    assert torch.cuda.get_device_capability() == (8, 9)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_log = args.output_dir / f"rank{rank}.log"

    def emit(status: str, **fields) -> None:
        row = {"status": status, "rank": rank, "time_utc": datetime.now(timezone.utc).isoformat(), **fields}
        with rank_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    emit("starting", arm=args.arm, length=args.length, batch=args.batch, cohort=args.cohort,
         output_tokens=args.output_tokens, reserve_decode_tokens=args.reserve_decode_tokens)
    prompt_metadata = json.loads(args.prompt_manifest.read_text())
    assert prompt_metadata["status"] == "complete"
    assert prompt_metadata["prompt_tokens"] == args.length
    assert prompt_metadata["cohort"] == args.cohort
    source_prompts = load_file(str(args.tokens))["input_ids"]
    prompts = select_prompts(source_prompts, args.batch, args.length, args.repeat_prompts)
    assert tuple(prompts.shape) == (args.batch, args.length)
    assert all(torch.unique(row).numel() > 1 for row in prompts)

    emit("model_load_start")
    star_ranks = None
    if args.arm == "star_v_adaptive":
        metadata = json.loads((args.star_checkpoint / "result.json").read_text())
        star_ranks = metadata["budget"]["ranks"]
        assert len(star_ranks) == NUM_LAYERS
        assert tuple(metadata["protocol"]["skip_layers"]) == SKIP_LAYERS
        config = AutoConfig.from_pretrained(args.star_checkpoint, local_files_only=True)
        config.star_value_ranks = star_ranks
        state = torch.load(
            args.star_checkpoint / "fused.pt", map_location="cpu", mmap=True, weights_only=True
        )
        model, loading_info = StarQwen3ForCausalLM.from_pretrained(
            None,
            config=config,
            state_dict=state,
            dtype=torch.bfloat16,
            distributed_config=DistributedConfig(tp_size=TP_SIZE, tp_plan=star_tp_plan(star_ranks)),
            output_loading_info=True,
        )
        assert not loading_info["missing_keys"] and not loading_info["unexpected_keys"]
        model.eval()
        del state
        projection_shapes = {}
        for index, layer in enumerate(model.model.layers):
            value = layer.self_attn.v_proj
            if index in SKIP_LAYERS:
                shape = tuple(local_weight(value.weight).shape)
                assert shape == (128, 4096)
                projection_shapes[index] = {"dense_v": shape}
            else:
                low = tuple(local_weight(value.VS.weight).shape)
                up = tuple(local_weight(value.U.weight).shape)
                assert low == (star_ranks[index], 4096)
                assert up == (128, star_ranks[index])
                projection_shapes[index] = {"shared_vs": low, "local_u": up}
        emit("star_projection_shapes", examples={
            index: projection_shapes[index] for index in (0, 2, 31, 35)
        })
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            distributed_config=DistributedConfig(tp_size=TP_SIZE),
            local_files_only=True,
        ).eval()
    emit("model_loaded", memory=memory_snapshot(handle))

    def report_layer(index: int) -> None:
        emit("cache_allocate_layer", layer=index)

    layers = install_v_only(
        model,
        arm=args.arm,
        batch=args.batch,
        capacity=args.length + args.reserve_decode_tokens,
        rank=rank,
        factor_root=args.factor_root if args.arm == "basis_v64" else None,
        on_layer=report_layer,
    )
    state_bytes = {
        name: sum(layer.state_bytes()[name] for layer in layers)
        for name in layers[0].state_bytes()
    }
    assert all(layer.value_cache.shape[2] == args.length + args.reserve_decode_tokens for layer in layers)
    if args.arm == "basis_v64":
        assert all(layer.value_cache.shape[-1] == VALUE_RANK for layer in layers)
    else:
        expected_widths = [128 if index in SKIP_LAYERS else width
                           for index, width in enumerate(star_ranks)]
        assert [layer.value_cache.shape[-1] for layer in layers] == expected_widths
    torch.cuda.synchronize()
    dist.barrier()
    cache_allocated = memory_snapshot(handle)
    emit("cache_allocated", state_bytes=state_bytes, memory=cache_allocated)

    torch.cuda.reset_peak_memory_stats()
    last_logits = None
    for start in range(0, args.length, args.chunk_size):
        end = min(start + args.chunk_size, args.length)
        emit("prefill_chunk_start", start=start, end=end)
        position_ids = torch.arange(start, end, device="cuda", dtype=torch.long)[None]
        output = model(
            input_ids=prompts[:, start:end].to("cuda"),
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        last_logits = output.logits
        del output
        torch.cuda.synchronize()
        emit("prefill_chunk_complete", end=end)
    assert all(layer.length == args.length for layer in layers)
    assert last_logits is not None and bool(torch.isfinite(last_logits).all())
    token = choose(last_logits)
    del last_logits
    torch.cuda.synchronize()
    dist.barrier()
    prefill_memory = memory_snapshot(handle)
    emit("prefill_complete", memory=prefill_memory)
    ready_memory = memory_snapshot(handle)
    emit("decode_ready", memory=ready_memory)

    torch.cuda.reset_peak_memory_stats()
    generated_tokens = [token]
    emit("decode_start", steps=args.output_tokens - 1)
    for step in range(args.output_tokens - 1):
        output = model(
            input_ids=token,
            position_ids=torch.full((1, 1), args.length + step, device="cuda", dtype=torch.long),
            use_cache=False,
            logits_to_keep=1,
        )
        assert bool(torch.isfinite(output.logits).all())
        token = choose(output.logits)
        generated_tokens.append(token)
        del output
        if (step + 1) % 16 == 0 or step + 2 == args.output_tokens:
            emit("decode_step_complete", output_tokens=step + 2)
    assert all(layer.length == args.length + args.output_tokens - 1 for layer in layers)
    torch.cuda.synchronize()
    dist.barrier()
    decode_memory = memory_snapshot(handle)
    emit("decode_complete", memory=decode_memory)

    probe_positions = sorted({0, args.length // 4, args.length // 2,
                              3 * args.length // 4, args.length - 1})
    cache_probes = [
        {
            "layer": index,
            "key": layer.key_cache[0, 0, probe_positions, :16].float().cpu().tolist(),
            "value": layer.value_cache[0, 0, probe_positions, :16].float().cpu().tolist(),
        }
        for index, layer in enumerate(layers)
    ]

    result = {
        "schema": "basisserve.qwen3_8b.tp8_v_only_memory.v1",
        "status": "complete",
        "arm": args.arm,
        "rank": rank,
        "local_rank": local_rank,
        "tp": TP_SIZE,
        "dp": 1,
        "pp": 1,
        "prompt_tokens": args.length,
        "batch": args.batch,
        "cohort": args.cohort,
        "chunk_size": args.chunk_size,
        "output_tokens": args.output_tokens,
        "reserve_decode_tokens": args.reserve_decode_tokens,
        "prompt_source_rows": source_prompts.shape[0],
        "prompt_repeated": args.repeat_prompts and args.batch > source_prompts.shape[0],
        "dtype": "bfloat16",
        "dense_key": True,
        "full_attention": True,
        "key_offload": False,
        "value_offload": False,
        "star_value_ranks": star_ranks,
        "star_actual_global_retention": sum(star_ranks) / (NUM_LAYERS * 1024) if star_ranks else None,
        "state_bytes": state_bytes,
        "cache_probe_positions": probe_positions,
        "cache_probes": cache_probes,
        "memory": {
            "cache_allocated": cache_allocated,
            "prefill_complete": prefill_memory,
            "decode_ready": ready_memory,
            "decode_complete": decode_memory,
        },
        "generated_token_ids": [item.flatten().cpu().tolist() for item in generated_tokens],
        "model": str(args.model.resolve()),
        "star_checkpoint": str(args.star_checkpoint.resolve()) if star_ranks else None,
        "factor_root": str(args.factor_root.resolve()) if args.arm == "basis_v64" else None,
        "prompts": str(args.tokens.resolve()),
        "prompt_manifest": prompt_metadata,
        "metadata": {
            "git_commit": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
            "git_status": command(["git", "status", "--porcelain"])["stdout"],
            "command_line": sys.argv,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "transformers": importlib.metadata.version("transformers"),
            "flash_attn": importlib.metadata.version("flash-attn"),
            "gpu_name": torch.cuda.get_device_name(),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "numa_node": numa_node,
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    emit("complete", result=str(args.output_dir / f"rank{rank}.json"))
    dist.barrier()
    pynvml.nvmlShutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
