#!/usr/bin/env python3
"""Calibrate static per-layer/per-TP-source E4M3 wire scales for Qwen3-8B C1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from basisserve.core.qwen3_8b_tp4_decode import (  # noqa: E402
    Qwen3TP4C1DecodeAttention,
    TP_SIZE,
    close_qwen3_tp4_packed_communicator,
    configure_qwen3_tp4_caches,
    install_qwen3_tp4_decode_attention,
)
from basisserve.kernels.fp8_wire import FP8_E4M3_MAX  # noqa: E402


FORMAT = "basisserve.qwen3_8b.tp4_static_fp8_wire_calibration.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _global_max_float(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _configure(
    modules: Sequence[Qwen3TP4C1DecodeAttention],
    *,
    batch_size: int,
    seqlen: int,
) -> None:
    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    configure_qwen3_tp4_caches(
        modules,
        batch_size=batch_size,
        capacity=seqlen,
        max_forward_tokens=seqlen,
    )


@torch.inference_mode()
def _calibrate(
    model: torch.nn.Module,
    modules: Sequence[Qwen3TP4C1DecodeAttention],
    windows: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    for module in modules:
        module.reset_wire_amax()
        module.set_wire_amax_observation(True)
    seqlen = int(windows.shape[1])
    position_ids = torch.arange(
        seqlen,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(0)
    configured_batch = 0
    started = time.perf_counter()
    processed = 0
    try:
        for start in range(0, len(windows), batch_size):
            input_ids = windows[start : start + batch_size].to(
                device=device,
                dtype=torch.int64,
            )
            current_batch = int(input_ids.shape[0])
            if current_batch != configured_batch:
                _configure(modules, batch_size=current_batch, seqlen=seqlen)
                configured_batch = current_batch
            else:
                for module in modules:
                    module.reset_cache()
            model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )
            processed += current_batch
            if dist.get_rank() == 0 and (
                processed == len(windows) or processed % max(1, 8 * batch_size) == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "calibration_progress",
                            "windows": processed,
                            "total_windows": len(windows),
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize(device)
    finally:
        for module in modules:
            module.set_wire_amax_observation(False)
    return _global_max_float(time.perf_counter() - started, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--window-offset", type=int, default=0)
    parser.add_argument("--num-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-scales", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    args = parser.parse_args()

    if min(args.num_windows, args.batch_size, args.torch_num_threads) <= 0:
        raise ValueError("window count, batch size, and thread count must be positive")
    if args.window_offset < 0:
        raise ValueError("window offset must be nonnegative")
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    window_path = Path(args.windows).expanduser().resolve()
    payload = load_file(str(window_path), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("calibration artifact must contain exactly input_ids")
    all_windows = payload["input_ids"]
    if all_windows.ndim != 2:
        raise ValueError("calibration input_ids must be a matrix")
    stop = args.window_offset + args.num_windows
    if stop > len(all_windows):
        raise ValueError(
            f"requested windows [{args.window_offset}:{stop}] exceed {len(all_windows)}"
        )
    windows = all_windows[args.window_offset:stop].contiguous()

    from transformers import AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    model_path = Path(args.model).expanduser().resolve()
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
    ).eval()
    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError(f"FP8 wire calibration requires TP{TP_SIZE}")
    installed = install_qwen3_tp4_decode_attention(
        model,
        factor_dir=factor_dir,
        c1_decode_attention_backend="cuda",
        c1_wire_dtype="bfloat16",
    )
    modules = tuple(
        module for module in installed if isinstance(module, Qwen3TP4C1DecodeAttention)
    )
    if len(modules) != len(model.model.layers):
        raise RuntimeError("FP8 wire calibration did not install every C1 layer")

    elapsed_seconds = _calibrate(
        model,
        modules,
        windows,
        batch_size=args.batch_size,
        device=device,
    )
    local_amax = torch.stack([module.wire_amax for module in modules]).contiguous()
    gathered = torch.empty(
        TP_SIZE * len(modules),
        dtype=torch.float32,
        device=device,
    )
    dist.all_gather_into_tensor(gathered, local_amax)
    wire_amax = gathered.view(TP_SIZE, len(modules)).transpose(0, 1).contiguous()
    wire_scales = (wire_amax / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )

    if dist.get_rank() == 0:
        output_scales = Path(args.output_scales).expanduser().resolve()
        output_json = Path(args.output_json).expanduser().resolve()
        _atomic_safetensors(
            output_scales,
            {
                "wire_amax": wire_amax.cpu(),
                "wire_scales": wire_scales.cpu(),
            },
        )
        result: dict[str, object] = {
            "format": FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "protocol": {
                "tp_size": TP_SIZE,
                "encoder_compute_dtype": "bfloat16",
                "decoder_compute_dtype": "bfloat16",
                "observed_tensor": "post-compressed-V-attention local coordinates",
                "wire_dtype": "float8_e4m3fn",
                "fp8_max": FP8_E4M3_MAX,
                "scale_rule": "per-layer/per-TP-source amax / fp8_max",
                "static_scale_metadata_transmitted_per_forward": False,
                "window_offset": args.window_offset,
                "num_windows": args.num_windows,
                "seqlen": int(windows.shape[1]),
                "batch_size": args.batch_size,
            },
            "inputs": {
                "model": str(model_path),
                "factor_dir": str(factor_dir),
                "windows": str(window_path),
                "windows_sha256": _sha256(window_path),
            },
            "artifact": {
                "path": str(output_scales),
                "sha256": _sha256(output_scales),
                "shape": list(wire_scales.shape),
                "keys": ["wire_amax", "wire_scales"],
            },
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "world_size": dist.get_world_size(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "elapsed_seconds": elapsed_seconds,
        }
        _atomic_json(output_json, result)
        print(
            json.dumps(
                {
                    "event": "calibration_complete",
                    "scales": str(output_scales),
                    "manifest": str(output_json),
                    "elapsed_seconds": elapsed_seconds,
                }
            ),
            flush=True,
        )
    dist.barrier()
    close_qwen3_tp4_packed_communicator(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
