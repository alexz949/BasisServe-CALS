#!/usr/bin/env python3
"""Collect PaLU Fisher importance and allocate K- or V-only M-LRD ranks."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.distributed import DistributedConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as builder
from evaluation.reproduce_palu_paper_llama2_distributed import (
    official_fisher_uniform_rank_map,
)


FORMAT = "basisserve.gqa.palu_projection_fisher_stats.v1"
OFFICIAL_PALU_COMMIT = "bb22666e2ef96707e8dd21d93fc00146c2e0d615"


def _cuda_device_indices() -> tuple[int, ...]:
    indices = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError(f"expected a Llama/Qwen decoder, found {type(model).__name__}")
    return list(layers)


def _load_windows(
    path: Path,
    model_metadata: dict[str, Any],
    *,
    samples: int,
) -> tuple[Tensor, dict[str, Any]]:
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing windows artifact or manifest under {path.parent}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in (
        builder.WINDOWS_FORMAT,
        builder.PACKED_WINDOWS_FORMAT,
    ):
        raise ValueError("incompatible calibration-window format")
    if manifest["model"]["config_sha256"] != model_metadata["config_sha256"]:
        raise ValueError("calibration windows belong to another model config")
    if builder._sha256(path) != manifest["artifact"]["sha256"]:
        raise ValueError("calibration-window artifact hash changed")
    input_ids = load_file(str(path), device="cpu")["input_ids"].to(torch.long)
    if tuple(input_ids.shape) != (samples, builder.SEQUENCE_LENGTH):
        raise ValueError(
            f"expected C4 windows [{samples}, {builder.SEQUENCE_LENGTH}], "
            f"found {list(input_ids.shape)}"
        )
    return input_ids, manifest


def _input_device(model: nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    if embedding.weight.device.type != "cuda":
        raise RuntimeError(f"input embeddings are on {embedding.weight.device}, expected CUDA")
    return embedding.weight.device


def _configure_projection_gradients(
    model: nn.Module,
    *,
    target: str,
) -> list[tuple[str, nn.Linear]]:
    if target not in {"k", "v"}:
        raise ValueError(f"unsupported projection target: {target!r}")
    projection_name = f"{target}_proj"
    layers = _decoder_layers(model)
    selected: list[tuple[str, nn.Linear]] = []
    for layer_index, layer in enumerate(layers):
        module = getattr(layer.self_attn, projection_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"layer {layer_index} {projection_name} is not dense Linear"
            )
        if module.weight.device.type != "cuda":
            raise RuntimeError(
                f"layer {layer_index} {projection_name} is on {module.weight.device}; "
                "Fisher collection does not support CPU/disk offload"
            )
        selected.append(
            (f"model.layers.{layer_index}.self_attn.{projection_name}", module)
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, module in selected:
        module.weight.requires_grad_(True)
    return selected


def _allocate_exact_official(
    fisher_scalars: dict[str, float],
    *,
    retained_ratio: float = 0.75,
) -> tuple[dict[str, list[int]], int, int]:
    return official_fisher_uniform_rank_map(
        fisher_scalars,
        output_width=builder.NUM_KV_HEADS * builder.HEAD_DIM,
        num_heads=builder.NUM_KV_HEADS,
        head_group_size=1,
        retained_ratio=retained_ratio,
        block_size=32,
    )


def _selective_activation_offload(enabled: bool):
    """Offload saved sequence activations without copying 2-D TP weights."""
    if not enabled:
        return nullcontext()

    def pack(tensor: Tensor):
        should_offload = tensor.device.type == "cuda" and tensor.ndim >= 3
        if not should_offload:
            return False, tensor
        cpu_tensor = torch.empty(
            tensor.size(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            pin_memory=True,
        )
        cpu_tensor.copy_(tensor)
        return True, tensor.device, cpu_tensor

    def unpack(packed):
        if not packed[0]:
            return packed[1]
        _, device, cpu_tensor = packed
        return cpu_tensor.to(device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _chunked_official_palu_loss_and_backward(
    model: nn.Module,
    batch: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    """Compute PaLU's CE in chunks, then backprop one assembled hidden gradient."""
    # PaLU calls the causal LM with input=batch[:, :-1], labels=batch[:, 1:].
    # Hugging Face then shifts labels once more and ignores the final position,
    # so valid targets are batch[:, 2:] for hidden positions [0, seqlen - 3].
    base_outputs = model.model(input_ids=batch[:, :-1], use_cache=False)
    hidden_states = base_outputs.last_hidden_state[:, :-1, :]
    targets = batch[:, 2:]
    valid_tokens = targets.numel()
    hidden_gradient = torch.empty_like(hidden_states)
    loss_total: Tensor | None = None
    for start in range(0, targets.shape[1], chunk_size):
        stop = min(start + chunk_size, targets.shape[1])
        chunk_hidden = (
            hidden_states[:, start:stop, :].detach().requires_grad_(True)
        )
        logits = model.lm_head(chunk_hidden).float()
        chunk_loss = (
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets[:, start:stop].reshape(-1),
                reduction="sum",
            )
            / valid_tokens
        )
        (chunk_gradient,) = torch.autograd.grad(chunk_loss, chunk_hidden)
        hidden_gradient[:, start:stop, :].copy_(chunk_gradient)
        detached_loss = chunk_loss.detach()
        loss_total = detached_loss if loss_total is None else loss_total + detached_loss
    hidden_states.backward(hidden_gradient)
    if loss_total is None:
        raise RuntimeError("empty calibration target")
    return loss_total


def collect(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    builder.activate_model_profile(args.profile)
    builder.SEQUENCE_LENGTH = args.sequence_length
    torch.set_num_threads(args.torch_num_threads)
    tensor_parallel = args.tensor_parallel_size > 1
    local_rank = 0
    if tensor_parallel:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("tensor parallel Fisher must be launched with torchrun")
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        if dist.get_world_size() != args.tensor_parallel_size:
            raise ValueError("torchrun world size differs from --tensor-parallel-size")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
    else:
        rank = 0
    if args.activation_offload_ranks == "all":
        activation_offload = tensor_parallel
    elif args.activation_offload_ranks == "none":
        activation_offload = False
    else:
        requested_offload_ranks = {
            int(value) for value in args.activation_offload_ranks.split(",")
        }
        if any(value < 0 or value >= args.tensor_parallel_size for value in requested_offload_ranks):
            raise ValueError("activation offload rank is outside the TP world")
        activation_offload = tensor_parallel and local_rank in requested_offload_ranks
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    model_path = Path(args.model).expanduser().resolve()
    builder._validate_config(AutoConfig.from_pretrained(str(model_path), local_files_only=True))
    model_metadata = builder._model_metadata(model_path)
    windows_path = Path(args.windows).expanduser().resolve()
    windows, windows_manifest = _load_windows(
        windows_path,
        model_metadata,
        samples=args.samples,
    )

    dtype = torch.bfloat16
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    if tensor_parallel:
        if args.device_map != "none":
            raise ValueError("tensor parallelism is mutually exclusive with device_map")
        model_kwargs["distributed_config"] = DistributedConfig(
            tp_size=args.tensor_parallel_size
        )
    elif args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
        model_kwargs["max_memory"] = {
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in _cuda_device_indices()
        }
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    if args.device_map == "none" and not tensor_parallel:
        model.to(torch.device(args.device))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()
    projection_modules = _configure_projection_gradients(
        model,
        target=args.target,
    )
    accumulators = {}
    for name, module in projection_modules:
        weight = module.weight
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        # Keep the long-lived squared-gradient sums off GPU.  For Llama-70B,
        # TP=4 leaves little activation headroom on 48 GiB cards, while these
        # buffers alone occupy about 1.25 GiB per rank in fp32.
        accumulators[name] = torch.zeros_like(
            local_weight, dtype=torch.float32, device="cpu"
        )
    input_device = _input_device(model)
    losses: list[float] = []
    for sample_index in tqdm(
        range(windows.shape[0]),
        desc=f"PaLU {args.target.upper()} Fisher",
        disable=rank != 0,
    ):
        batch = windows[sample_index : sample_index + 1].to(input_device)
        # Match PaLU's published implementation exactly: it passes next-token
        # labels alongside a one-token-shorter input to the HF causal-LM loss.
        outputs = None
        with _selective_activation_offload(activation_offload):
            if tensor_parallel:
                loss = _chunked_official_palu_loss_and_backward(
                    model,
                    batch,
                    chunk_size=args.loss_chunk_size,
                )
            else:
                outputs = model(
                    input_ids=batch[:, :-1],
                    labels=batch[:, 1:],
                    use_cache=False,
                )
                loss = outputs.loss
        if not bool(torch.isfinite(loss).cpu()):
            raise FloatingPointError(f"non-finite Fisher loss at sample {sample_index}")
        losses.append(float(loss.detach().cpu()))
        if not tensor_parallel:
            loss.backward()
        for name, module in projection_modules:
            gradient = module.weight.grad
            if gradient is None:
                raise RuntimeError(f"missing Fisher gradient for {name}")
            if hasattr(gradient, "to_local"):
                gradient = gradient.to_local()
            gradient_float = gradient.detach().float().cpu()
            accumulators[name].addcmul_(gradient_float, gradient_float)
            del gradient_float
        model.zero_grad(set_to_none=True)
        del outputs, loss

    fisher_scalars: dict[str, float] = {}
    for name, _ in projection_modules:
        fisher = accumulators.pop(name).div_(float(windows.shape[0])).sqrt_()
        totals = torch.tensor(
            [float(fisher.sum()), float(fisher.numel())],
            dtype=torch.float64,
            # NCCL only accepts CUDA tensors; the Fisher matrix itself is kept
            # on CPU to preserve GPU memory during collection.
            device=input_device if tensor_parallel else fisher.device,
        )
        if tensor_parallel:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        scalar = float((totals[0] / totals[1]).cpu())
        if not math.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"invalid Fisher scalar for {name}: {scalar}")
        fisher_scalars[name] = scalar
        del fisher
    rank_map, rank_sum, total_rank = _allocate_exact_official(
        fisher_scalars,
        retained_ratio=args.retained_ratio,
    )
    layer_ranks = [
        rank_map[f"model.layers.{layer_index}.self_attn.{args.target}_proj"]
        for layer_index in range(builder.NUM_LAYERS)
    ]
    result = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_palu_commit": OFFICIAL_PALU_COMMIT,
        "model": model_metadata,
        "geometry": {
            "num_layers": builder.NUM_LAYERS,
            "num_query_heads": builder.NUM_QUERY_HEADS,
            "num_physical_kv_heads": builder.NUM_KV_HEADS,
            "head_dim": builder.HEAD_DIM,
            "projection_width": builder.NUM_KV_HEADS * builder.HEAD_DIM,
        },
        "fisher": {
            "target": f"{args.target}_proj_only",
            "dataset": "allenai/c4",
            "samples": args.samples,
            "sequence_length": builder.SEQUENCE_LENGTH,
            "loss_semantics": "official PaLU shifted-input/shifted-label HF causal-LM loss",
            "loss_chunk_size": args.loss_chunk_size,
            "aggregation": "sqrt(mean(per-sample gradient squared)), then matrix mean",
            "scalars": fisher_scalars,
            "mean_loss": sum(losses) / len(losses),
        },
        "allocation": {
            "method": f"official_palu_fisher_uniform_adapted_to_{args.target}_only",
            "rank_block_size": 32,
            "target": args.target,
            "requested_retained_ratio": args.retained_ratio,
            "requested_cache_compression_ratio": 1.0 - args.retained_ratio,
            "rank_map": rank_map,
            "layer_ranks": layer_ranks,
            "rank_sum": rank_sum,
            "total_rank": total_rank,
            "realized_retained_ratio": rank_sum / total_rank,
            "realized_cache_compression_ratio": 1.0 - rank_sum / total_rank,
        },
        "calibration_windows": {
            "path": str(windows_path),
            "sha256": builder._sha256(windows_path),
            "manifest_sha256": builder._sha256(windows_path.parent / "manifest.json"),
            "sampling": windows_manifest.get(
                "sampling",
                windows_manifest.get("packing"),
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in _cuda_device_indices()
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in _cuda_device_indices()
            },
        },
    }
    if rank == 0:
        output_dir.mkdir(parents=True)
        builder._atomic_json(output_dir / "fisher.json", result)
        print(
            f"[Result] realized_compression={1.0 - rank_sum / total_rank:.6f} "
            f"rank_sum={rank_sum}/{total_rank} output={output_dir / 'fisher.json'}",
            flush=True,
        )
    if tensor_parallel:
        dist.barrier()
        try:
            dist.destroy_process_group()
        except dist.DistBackendError as error:
            # All reductions and the rank-0 artifact write are complete here.
            # A memory-starved NCCL communicator can still fail during teardown;
            # do not turn a valid Fisher artifact into a failed batch pipeline.
            print(f"[Warning] NCCL process-group teardown failed: {error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(builder.MODEL_PROFILES), required=True)
    parser.add_argument("--target", choices=("k", "v"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=builder.CALIBRATION_SAMPLES)
    parser.add_argument("--sequence-length", type=int, default=builder.SEQUENCE_LENGTH)
    parser.add_argument("--retained-ratio", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("none", "auto", "balanced", "balanced_low_0"),
        default="none",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--activation-offload-ranks",
        default="all",
        help="TP local ranks to offload (comma-separated), or 'all'/'none'",
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--loss-chunk-size", type=int, default=1024)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.loss_chunk_size <= 0:
        parser.error("--loss-chunk-size must be positive")
    if not 0.0 < args.retained_ratio <= 1.0:
        parser.error("--retained-ratio must be in (0, 1]")
    return args


if __name__ == "__main__":
    collect(parse_args())
