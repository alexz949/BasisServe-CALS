#!/usr/bin/env python3
"""Compare equal-wire pure AllReduce and AllGather for attention Wo.

This is a single-GPU offline oracle.  It captures the exact input of selected
decoder attention ``o_proj`` modules on disjoint C4 window banks, fits output POD
bases using calibration rows only, and evaluates held-out output error.  TP is
simulated by contiguous row-parallel input shards; no distributed runtime or
checkpoint is modified.
"""

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
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.common_private_collective import (  # noqa: E402
    CommonPrivateFactors,
    collective_accounting,
    common_private_output_metrics,
    shared_only_factors,
)
from basisserve.analysis.mlp_topk_chebyshev import (  # noqa: E402
    polar_retract_columns,
)
from basisserve.sketching.coordinate_selection import (  # noqa: E402
    compute_uncentered_pod,
)
from evaluation.analyze_qwen35_common_private_collective import (  # noqa: E402
    _dense_local_consistency,
    _local_outputs,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.analyze_qwen35_mlp_topk_chebyshev import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _file_sha256,
)


FORMAT = "basisserve.qwen3.o_proj_collective_endpoints.v1"
ROW_FORMAT = "basisserve.qwen3.o_proj_collective_endpoints.row.v1"
MODEL_FORMATS = {
    "qwen3": (FORMAT, ROW_FORMAT),
    "llama": (
        "basisserve.llama.o_proj_collective_endpoints.v1",
        "basisserve.llama.o_proj_collective_endpoints.row.v1",
    ),
}
MODEL_LABELS = {"qwen3": "Qwen3", "llama": "LLaMA"}
DEFAULT_MODEL = Path(
    "/zpool-00/home/lz299/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
DEFAULT_TRAIN_WINDOWS = REPO_ROOT / (
    "results/cache/"
    "qwen3_8b_sequential_pair_exchange_selection_c4_s2048_n64_seed20260726_v1/"
    "windows.safetensors"
)
DEFAULT_VALIDATION_WINDOWS = REPO_ROOT / (
    "results/cache/"
    "qwen3_8b_sequential_pair_exchange_final_audit_c4_s2048_n64_seed20260728_v1/"
    "windows.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--expected-model-type", choices=tuple(MODEL_FORMATS), default="qwen3"
    )
    parser.add_argument("--train-windows", default=str(DEFAULT_TRAIN_WINDOWS))
    parser.add_argument(
        "--validation-windows", default=str(DEFAULT_VALIDATION_WINDOWS)
    )
    parser.add_argument(
        "--train-window-split", choices=("all", "fit", "heldout"), default="all"
    )
    parser.add_argument(
        "--validation-window-split",
        choices=("all", "fit", "heldout"),
        default="all",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,18,35")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--baseline-rank", type=int, default=1536)
    parser.add_argument("--positions-per-window", type=int, default=128)
    parser.add_argument("--position-seed", type=int, default=20260812)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pod-oversample", type=int, default=16)
    parser.add_argument("--pod-niter", type=int, default=4)
    parser.add_argument("--local-output-chunk-size", type=int, default=512)
    parser.add_argument("--metric-chunk-size", type=int, default=128)
    parser.add_argument("--active-tokens", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--model-device-map", choices=("single", "balanced"), default="single"
    )
    parser.add_argument("--gpu-max-memory", default="22GiB")
    parser.add_argument("--cpu-max-memory", default="256GiB")
    parser.add_argument("--max-windows-per-split", type=int)
    parser.add_argument("--capture-sequence-length", type=int)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(
        sorted(set(int(piece.strip()) for piece in raw.split(",") if piece.strip()))
    )


def _tensor_sha256(value: Tensor) -> str:
    work = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(work.numpy().tobytes()).hexdigest()


def _window_hashes(windows: Tensor) -> tuple[str, ...]:
    if windows.ndim != 2:
        raise ValueError("window tensor must be [windows, sequence_length]")
    return tuple(_tensor_sha256(row) for row in windows)


def _bound_window_bank(
    windows: Tensor,
    *,
    max_windows: int | None,
    sequence_length: int | None,
) -> tuple[Tensor, tuple[str, ...]]:
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("maximum window count must be positive")
        windows = windows[:max_windows]
    if sequence_length is not None:
        if sequence_length <= 0 or sequence_length > int(windows.shape[1]):
            raise ValueError("capture sequence length is out of range")
        windows = windows[:, :sequence_length]
    if not int(windows.shape[0]):
        raise ValueError("bounded window bank is empty")
    windows = windows.contiguous()
    return windows, _window_hashes(windows)


def _model_loading_configuration(
    *,
    strategy: str,
    fit_device: torch.device,
    cuda_device_count: int,
    gpu_max_memory: str,
    cpu_max_memory: str,
) -> tuple[str | dict[str, str], dict[int | str, str] | None]:
    if strategy == "single":
        return {"": str(fit_device)}, None
    if strategy != "balanced":
        raise ValueError(f"unknown model device-map strategy: {strategy}")
    if cuda_device_count <= 1 or not gpu_max_memory or not cpu_max_memory:
        raise ValueError("balanced model loading requires multiple CUDA devices")
    return "balanced", {
        **{index: gpu_max_memory for index in range(cuda_device_count)},
        "cpu": cpu_max_memory,
    }


def _load_window_bank(
    path: Path, *, split: str = "all"
) -> tuple[Tensor, dict[str, Any], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = str(manifest.get("tensor_key", "input_ids"))
    tensors = load_file(str(path), device="cpu")
    if key not in tensors:
        raise KeyError(f"window tensor {key!r} is absent from {path}")
    windows = tensors[key].to(dtype=torch.long).contiguous()
    if windows.ndim != 2 or int(windows.shape[0]) <= 0:
        raise ValueError(f"invalid window matrix in {path}")
    recorded_window_count = manifest.get("windows", manifest.get("window_count", -1))
    if int(recorded_window_count) != int(windows.shape[0]):
        raise ValueError(f"window count differs from manifest in {path}")
    if int(manifest.get("sequence_length", -1)) != int(windows.shape[1]):
        raise ValueError(f"sequence length differs from manifest in {path}")
    hashes = _window_hashes(windows)
    recorded = tuple(map(str, manifest.get("window_hashes", ())))
    if recorded and hashes != recorded:
        raise ValueError(f"window hashes differ from manifest in {path}")
    if split != "all":
        if "split_codes" not in tensors:
            raise ValueError(f"window split requested but split_codes is absent: {path}")
        split_code = {"fit": 0, "heldout": 1}[split]
        indices = torch.nonzero(
            tensors["split_codes"].to(torch.long) == split_code,
            as_tuple=False,
        ).flatten()
        if not int(indices.numel()):
            raise ValueError(f"window split is empty: {split}")
        expected_split_count = manifest.get("split_counts", {}).get(split)
        if expected_split_count is not None and int(expected_split_count) != int(
            indices.numel()
        ):
            raise ValueError(f"window split count differs from manifest: {split}")
        windows = windows.index_select(0, indices).contiguous()
        hashes = tuple(hashes[int(index)] for index in indices.tolist())
    return windows, manifest, hashes


def _sample_positions(
    *,
    windows: int,
    sequence_length: int,
    positions_per_window: int,
    seed: int,
) -> Tensor:
    if (
        windows <= 0
        or sequence_length <= 0
        or not 0 < positions_per_window <= sequence_length
    ):
        raise ValueError("invalid token-position sampling geometry")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = [
        torch.randperm(sequence_length, generator=generator)[:positions_per_window]
        .sort()
        .values
        for _ in range(windows)
    ]
    return torch.stack(rows).to(dtype=torch.long).contiguous()


def _validate_disjoint_windows(
    train_hashes: Sequence[str], validation_hashes: Sequence[str]
) -> None:
    overlap = set(train_hashes) & set(validation_hashes)
    if overlap:
        raise ValueError(f"train/validation window overlap: {len(overlap)} rows")


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    current = model
    for name in ("model", "language_model"):
        child = getattr(current, name, None)
        if child is not None:
            current = child
    layers = getattr(current, "layers", None)
    if layers is None:
        raise AttributeError("could not locate decoder layers")
    return layers


def _attention_geometry(config: Any) -> tuple[int, int, int]:
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    if num_attention_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError("checkpoint attention geometry is invalid")
    configured_head_dim = getattr(config, "head_dim", None)
    head_dim = (
        int(configured_head_dim)
        if configured_head_dim is not None
        else hidden_size // num_attention_heads
    )
    if head_dim <= 0:
        raise ValueError("checkpoint attention head dimension is invalid")
    return hidden_size, num_attention_heads, head_dim


class _OProjCapture:
    def __init__(
        self,
        *,
        modules: Mapping[int, nn.Module],
        rows: int,
        input_width: int,
    ) -> None:
        self.rows = {
            layer: torch.empty(rows, input_width, dtype=torch.bfloat16)
            for layer in modules
        }
        self.active_positions: Tensor | None = None
        self.active_start: int | None = None
        self.seen: set[int] = set()
        self.handles = [
            module.register_forward_pre_hook(self._hook(layer))
            for layer, module in modules.items()
        ]

    def _hook(self, layer: int):
        def capture(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if self.active_positions is None or self.active_start is None:
                raise RuntimeError("o_proj hook ran outside an active batch")
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError("o_proj pre-hook did not receive a tensor")
            activation = inputs[0]
            if activation.ndim != 3 or int(activation.shape[0]) != int(
                self.active_positions.shape[0]
            ):
                raise ValueError("unexpected o_proj activation shape")
            positions = self.active_positions.to(device=activation.device)
            batch_indices = torch.arange(
                int(activation.shape[0]), device=activation.device
            ).unsqueeze(1)
            selected = activation[batch_indices, positions]
            flat = selected.reshape(-1, int(activation.shape[-1]))
            stop = self.active_start + int(flat.shape[0])
            self.rows[layer][self.active_start : stop].copy_(
                flat.to(device="cpu", dtype=torch.bfloat16)
            )
            self.seen.add(layer)

        return capture

    def begin(self, positions: Tensor, start: int) -> None:
        if self.active_positions is not None:
            raise RuntimeError("capture batch is already active")
        self.active_positions = positions
        self.active_start = int(start)
        self.seen.clear()

    def finish(self) -> None:
        expected = set(self.rows)
        if self.seen != expected:
            raise RuntimeError(
                f"o_proj hooks incomplete: seen={sorted(self.seen)}, "
                f"expected={sorted(expected)}"
            )
        self.active_positions = None
        self.active_start = None
        self.seen.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def _collect_split(
    *,
    model: nn.Module,
    modules: Mapping[int, nn.Module],
    windows: Tensor,
    positions: Tensor,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    split: str,
    input_width: int,
) -> dict[int, dict[str, Any]]:
    if tuple(positions.shape)[:1] != tuple(windows.shape)[:1]:
        raise ValueError("window and sampled-position counts differ")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    per_window = int(positions.shape[1])
    capture = _OProjCapture(
        modules=modules,
        rows=int(windows.shape[0]) * per_window,
        input_width=input_width,
    )
    try:
        batches = (int(windows.shape[0]) + batch_size - 1) // batch_size
        for batch_index, start_window in enumerate(
            range(0, int(windows.shape[0]), batch_size), start=1
        ):
            stop_window = min(start_window + batch_size, int(windows.shape[0]))
            batch_positions = positions[start_window:stop_window]
            capture.begin(batch_positions, start_window * per_window)
            input_ids = windows[start_window:stop_window].to(device=device)
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            capture.finish()
            print(
                f"[OProjEndpoint:{split}] batch={batch_index}/{batches} "
                f"windows={stop_window}/{int(windows.shape[0])}",
                flush=True,
            )
    finally:
        capture.close()
    records: dict[int, dict[str, Any]] = {}
    for layer, activation in capture.rows.items():
        path = output_dir / f"layer_{layer:03d}_{split}.safetensors"
        save_file({"C": activation.contiguous()}, str(path))
        records[layer] = {
            "file": path.name,
            "sha256": _file_sha256(path),
            "rows": int(activation.shape[0]),
            "width": int(activation.shape[1]),
            "dtype": str(activation.dtype),
        }
    return records


@torch.no_grad()
def _dense_teacher(
    activation: Tensor,
    weight: Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> Tensor:
    if activation.ndim != 2 or weight.ndim != 2:
        raise ValueError("activation and weight must be matrices")
    result = torch.empty(
        int(activation.shape[0]), int(weight.shape[0]), dtype=torch.bfloat16
    )
    work_weight = weight.to(device=device, dtype=torch.bfloat16)
    for start in range(0, int(activation.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(activation.shape[0]))
        result[start:stop] = F.linear(
            activation[start:stop].to(device=device, dtype=torch.bfloat16),
            work_weight,
            None,
        ).cpu()
    return result


@torch.no_grad()
def _fit_uniform_private(
    local_outputs: Sequence[Tensor],
    *,
    private_rank: int,
    baseline_rank: int,
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[CommonPrivateFactors, list[dict[str, Any]]]:
    bases = []
    diagnostics = []
    output_width = int(local_outputs[0].shape[1])
    for shard, output in enumerate(local_outputs):
        pod = compute_uncentered_pod(
            output.to(device=device),
            private_rank,
            oversample=oversample,
            niter=niter,
            seed=seed + 104729 * shard,
        )
        basis, retraction = polar_retract_columns(pod.basis)
        bases.append(basis.cpu().contiguous())
        diagnostics.append(
            {"shard": shard, **pod.diagnostics(), "polar_retraction": retraction}
        )
    return (
        CommonPrivateFactors(
            shared_basis=torch.empty(output_width, 0, dtype=torch.float32),
            private_bases=tuple(bases),
            baseline_rank=baseline_rank,
            allocation="uniform",
        ),
        diagnostics,
    )


def _record(
    *,
    layer: int,
    method: str,
    factors: CommonPrivateFactors,
    input_shard_widths: Sequence[int],
    active_tokens: int,
    dtype_bytes: int,
    fit: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    row_format: str = ROW_FORMAT,
    block_type: str = "qwen3_attention_o_proj",
) -> dict[str, Any]:
    return {
        "format": row_format,
        "block_type": block_type,
        "layer": int(layer),
        "baseline_rank": int(factors.baseline_rank),
        "method": method,
        "collective_model": (
            "pure_allreduce" if method == "pure_allreduce" else "pure_allgather"
        ),
        "allocation": factors.allocation,
        "shared_rank": factors.shared_rank,
        "private_ranks": list(factors.private_ranks),
        "total_private_rank": factors.total_private_rank,
        "fit": {
            "split": factors.fit_split,
            "validation_used": factors.validation_used_for_fit,
            **dict(fit),
        },
        "accounting": collective_accounting(
            factors,
            input_shard_widths=input_shard_widths,
            active_tokens=active_tokens,
            dtype_bytes=dtype_bytes,
        ),
        "splits": {
            "train": dict(train_metrics),
            "validation": dict(validation_metrics),
        },
    }


def _summary_csv(records: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "layer",
        "method",
        "shared_rank",
        "private_ranks",
        "ideal_ring_units",
        "padded_ring_units",
        "decoder_parameters",
        "total_local_encoder_parameters",
        "validation_relative_mse",
        "validation_cosine_similarity",
        "validation_p95_token_error",
    )
    lines = [",".join(columns)]
    for row in records:
        accounting = row["accounting"]
        validation = row["splits"]["validation"]
        values = {
            **row,
            **accounting,
            "private_ranks": "|".join(map(str, row["private_ranks"])),
            "validation_relative_mse": validation["relative_mse"],
            "validation_cosine_similarity": validation["cosine_similarity"],
            "validation_p95_token_error": validation[
                "per_token_relative_squared_error"
            ]["p95"],
        }
        lines.append(
            ",".join(str(values.get(column, "")) for column in columns)
        )
    return "\n".join(lines) + "\n"


def _decisions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for layer in sorted({int(row["layer"]) for row in records}):
        rows = [row for row in records if int(row["layer"]) == layer]
        allreduce = next(row for row in rows if row["method"] == "pure_allreduce")
        allgather = next(row for row in rows if row["method"] == "pure_allgather_uniform")
        ar_error = float(allreduce["splits"]["validation"]["relative_mse"])
        ag_error = float(allgather["splits"]["validation"]["relative_mse"])
        decisions.append(
            {
                "layer": layer,
                "pure_allreduce_validation_relative_mse": ar_error,
                "pure_allgather_validation_relative_mse": ag_error,
                "allgather_improvement": ar_error / max(ag_error, 1.0e-300),
                "lower_error_method": (
                    "pure_allgather_uniform" if ag_error < ar_error else "pure_allreduce"
                ),
            }
        )
    return decisions


def _summary_markdown(
    decisions: Sequence[Mapping[str, Any]],
    *,
    tp_size: int,
    allreduce_rank: int,
    allgather_rank_per_tp: int,
    model_label: str = "Qwen3",
) -> str:
    lines = [
        f"# {model_label} Attention Wo: Pure AllReduce vs Pure AllGather",
        "",
        (
            f"Both endpoints use equal ideal TP{tp_size} ring traffic. Pure "
            f"AllReduce uses a shared rank-{allreduce_rank} output basis; pure "
            f"uniform AllGather uses {tp_size} independent rank-"
            f"{allgather_rank_per_tp} local-output bases. All factors use "
            "calibration rows only."
        ),
        "",
        "| Layer | AllReduce MSE | AllGather MSE | AR/AG | Lower error |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['layer']} | "
            f"{float(row['pure_allreduce_validation_relative_mse']):.6g} | "
            f"{float(row['pure_allgather_validation_relative_mse']):.6g} | "
            f"{float(row['allgather_improvement']):.3f}x | "
            f"{row['lower_error_method']} |"
        )
    lines.extend(
        [
            "",
            (
                "This is an offline quality oracle. It does not claim realized "
                "NCCL latency or end-to-end serving speed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    layers = _csv_ints(args.layers)
    if (
        not layers
        or args.tp <= 1
        or args.baseline_rank <= 0
        or args.positions_per_window <= 0
        or args.batch_size <= 0
        or args.pod_oversample < 0
        or args.pod_niter < 0
        or min(args.local_output_chunk_size, args.metric_chunk_size) <= 0
        or min(args.active_tokens, args.dtype_bytes) <= 0
        or (
            args.max_windows_per_split is not None
            and args.max_windows_per_split <= 0
        )
        or (
            args.capture_sequence_length is not None
            and args.capture_sequence_length <= 0
        )
    ):
        raise ValueError("invalid attention Wo endpoint-oracle configuration")
    private_total = 2 * args.baseline_rank
    if private_total % args.tp:
        raise ValueError("equal-wire private rank is not divisible by TP")
    private_rank = private_total // args.tp

    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    partial_dir.mkdir(parents=True)
    started = time.perf_counter()
    timestamp_started = datetime.now(timezone.utc).isoformat()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this experiment requires a CUDA device")
    cuda_index = device.index if device.index is not None else 0
    torch.cuda.set_device(cuda_index)
    torch.cuda.init()
    cuda_device_count = torch.cuda.device_count()
    for device_index in range(cuda_device_count):
        torch.cuda.reset_peak_memory_stats(device_index)

    train_path = Path(args.train_windows).expanduser().resolve()
    validation_path = Path(args.validation_windows).expanduser().resolve()
    train_windows, train_manifest, _ = _load_window_bank(
        train_path, split=args.train_window_split
    )
    validation_windows, validation_manifest, _ = _load_window_bank(
        validation_path, split=args.validation_window_split
    )
    if int(train_windows.shape[1]) != int(validation_windows.shape[1]):
        raise ValueError("train/validation sequence lengths differ")
    source_train_windows = int(train_windows.shape[0])
    source_validation_windows = int(validation_windows.shape[0])
    source_sequence_length = int(train_windows.shape[1])
    train_windows, train_hashes = _bound_window_bank(
        train_windows,
        max_windows=args.max_windows_per_split,
        sequence_length=args.capture_sequence_length,
    )
    validation_windows, validation_hashes = _bound_window_bank(
        validation_windows,
        max_windows=args.max_windows_per_split,
        sequence_length=args.capture_sequence_length,
    )
    effective_sequence_length = int(train_windows.shape[1])
    _validate_disjoint_windows(train_hashes, validation_hashes)
    train_positions = _sample_positions(
        windows=int(train_windows.shape[0]),
        sequence_length=int(train_windows.shape[1]),
        positions_per_window=args.positions_per_window,
        seed=args.position_seed,
    )
    validation_positions = _sample_positions(
        windows=int(validation_windows.shape[0]),
        sequence_length=int(validation_windows.shape[1]),
        positions_per_window=args.positions_per_window,
        seed=args.position_seed + 1,
    )
    save_file(
        {"train": train_positions, "validation": validation_positions},
        str(partial_dir / "sampled_positions.safetensors"),
    )

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    model_device_map, max_memory = _model_loading_configuration(
        strategy=args.model_device_map,
        fit_device=device,
        cuda_device_count=cuda_device_count,
        gpu_max_memory=args.gpu_max_memory,
        cpu_max_memory=args.cpu_max_memory,
    )
    print(
        f"[OProjEndpoint] loading model={model_path} "
        f"device_map={model_device_map} max_memory={max_memory}",
        flush=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=model_dtype,
        device_map=model_device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).eval()
    raw_placement = getattr(model, "hf_device_map", {})
    placement = {str(key): str(value) for key, value in raw_placement.items()}
    if args.model_device_map == "balanced" and any(
        str(value) in {"cpu", "disk"} for value in raw_placement.values()
    ):
        raise RuntimeError(f"balanced model loading offloaded parameters: {placement}")
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != "cuda":
        raise RuntimeError("model input embeddings were not placed on CUDA")
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != args.expected_model_type:
        raise ValueError(
            f"expected a {args.expected_model_type} checkpoint, found {model_type}"
        )
    artifact_format, row_format = MODEL_FORMATS[model_type]
    model_label = MODEL_LABELS[model_type]
    decoder_layers = _decoder_layers(model)
    model_num_hidden_layers = len(decoder_layers)
    if min(layers) < 0 or max(layers) >= len(decoder_layers):
        raise ValueError("requested decoder layer is out of range")
    hidden_size, num_attention_heads, head_dim = _attention_geometry(model.config)
    o_proj_input_width = num_attention_heads * head_dim
    if (
        head_dim <= 0
        or o_proj_input_width % args.tp
        or args.baseline_rank > hidden_size
    ):
        raise ValueError("attention geometry is incompatible with TP/rank")
    input_shard_width = o_proj_input_width // args.tp
    if private_rank > input_shard_width:
        raise ValueError("private AllGather rank exceeds each TP input shard")
    modules = {layer: decoder_layers[layer].self_attn.o_proj for layer in layers}
    weights = {}
    module_metadata = {}
    for layer, module in modules.items():
        if getattr(module, "bias", None) is not None:
            raise ValueError("o_proj bias is unsupported by this row-parallel oracle")
        if tuple(module.weight.shape) != (hidden_size, o_proj_input_width):
            raise ValueError("unexpected o_proj geometry")
        weights[layer] = module.weight.detach().to(device="cpu", dtype=torch.bfloat16)
        attention = decoder_layers[layer].self_attn
        module_metadata[layer] = {
            "attention_class": type(attention).__name__,
            "o_proj_class": type(module).__name__,
            "has_output_gate_attribute": any(
                hasattr(attention, name)
                for name in ("gate_proj", "gate_up_proj", "o_gate", "output_gate")
            ),
        }
    snapshot_records = {"train": {}, "validation": {}}
    snapshot_records["train"] = _collect_split(
        model=model,
        modules=modules,
        windows=train_windows,
        positions=train_positions,
        batch_size=args.batch_size,
        device=input_device,
        output_dir=partial_dir,
        split="train",
        input_width=o_proj_input_width,
    )
    snapshot_records["validation"] = _collect_split(
        model=model,
        modules=modules,
        windows=validation_windows,
        positions=validation_positions,
        batch_size=args.batch_size,
        device=input_device,
        output_dir=partial_dir,
        split="validation",
        input_width=o_proj_input_width,
    )
    collection_peak_cuda_by_device = {
        str(device_index): int(torch.cuda.max_memory_allocated(device_index))
        for device_index in range(cuda_device_count)
    }
    collection_peak_cuda = collection_peak_cuda_by_device[str(cuda_index)]
    del modules, decoder_layers, model, train_windows, validation_windows
    for device_index in range(cuda_device_count):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()

    all_records = []
    layer_metadata = {}
    peak_cuda = collection_peak_cuda
    input_shard_widths = (input_shard_width,) * args.tp
    for layer in layers:
        layer_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(cuda_index)
        print(f"[OProjEndpoint] layer={layer} phase=load_train", flush=True)
        train_activation = load_file(
            str(partial_dir / snapshot_records["train"][layer]["file"]),
            device="cpu",
        )["C"]
        weight = weights[layer]
        train_teacher = _dense_teacher(
            train_activation,
            weight,
            device=device,
            chunk_size=args.local_output_chunk_size,
        )
        train_local = _local_outputs(
            train_activation,
            weight,
            tp_size=args.tp,
            chunk_size=args.local_output_chunk_size,
            device=device,
        )
        train_consistency = _dense_local_consistency(
            train_teacher, train_local, chunk_size=args.metric_chunk_size
        )

        print(f"[OProjEndpoint] layer={layer} phase=fit_allreduce", flush=True)
        shared_pod = compute_uncentered_pod(
            train_teacher.to(device=device),
            args.baseline_rank,
            oversample=args.pod_oversample,
            niter=args.pod_niter,
            seed=args.position_seed + 7919 * layer,
        )
        shared_basis, shared_retraction = polar_retract_columns(shared_pod.basis)
        allreduce_factors = shared_only_factors(
            shared_basis.cpu(), baseline_rank=args.baseline_rank, tp_size=args.tp
        )
        print(f"[OProjEndpoint] layer={layer} phase=fit_allgather", flush=True)
        allgather_factors, private_diagnostics = _fit_uniform_private(
            train_local,
            private_rank=private_rank,
            baseline_rank=args.baseline_rank,
            device=device,
            oversample=args.pod_oversample,
            niter=args.pod_niter,
            seed=args.position_seed + 104729 * layer,
        )
        train_metrics = {
            "pure_allreduce": common_private_output_metrics(
                train_teacher,
                train_local,
                allreduce_factors,
                device=device,
                chunk_size=args.metric_chunk_size,
            ),
            "pure_allgather_uniform": common_private_output_metrics(
                train_teacher,
                train_local,
                allgather_factors,
                device=device,
                chunk_size=args.metric_chunk_size,
            ),
        }
        del train_activation, train_teacher, train_local
        torch.cuda.empty_cache()

        print(f"[OProjEndpoint] layer={layer} phase=load_validation", flush=True)
        validation_activation = load_file(
            str(partial_dir / snapshot_records["validation"][layer]["file"]),
            device="cpu",
        )["C"]
        validation_teacher = _dense_teacher(
            validation_activation,
            weight,
            device=device,
            chunk_size=args.local_output_chunk_size,
        )
        validation_local = _local_outputs(
            validation_activation,
            weight,
            tp_size=args.tp,
            chunk_size=args.local_output_chunk_size,
            device=device,
        )
        validation_consistency = _dense_local_consistency(
            validation_teacher,
            validation_local,
            chunk_size=args.metric_chunk_size,
        )
        validation_metrics = {
            "pure_allreduce": common_private_output_metrics(
                validation_teacher,
                validation_local,
                allreduce_factors,
                device=device,
                chunk_size=args.metric_chunk_size,
            ),
            "pure_allgather_uniform": common_private_output_metrics(
                validation_teacher,
                validation_local,
                allgather_factors,
                device=device,
                chunk_size=args.metric_chunk_size,
            ),
        }
        all_records.extend(
            (
                _record(
                    layer=layer,
                    method="pure_allreduce",
                    factors=allreduce_factors,
                    input_shard_widths=input_shard_widths,
                    active_tokens=args.active_tokens,
                    dtype_bytes=args.dtype_bytes,
                    fit={
                        "basis": "train_teacher_output_pod",
                        "pod": shared_pod.diagnostics(),
                        "polar_retraction": shared_retraction,
                    },
                    train_metrics=train_metrics["pure_allreduce"],
                    validation_metrics=validation_metrics["pure_allreduce"],
                    row_format=row_format,
                    block_type=f"{model_type}_attention_o_proj",
                ),
                _record(
                    layer=layer,
                    method="pure_allgather_uniform",
                    factors=allgather_factors,
                    input_shard_widths=input_shard_widths,
                    active_tokens=args.active_tokens,
                    dtype_bytes=args.dtype_bytes,
                    fit={
                        "basis": "independent_train_local_output_pod",
                        "shards": private_diagnostics,
                    },
                    train_metrics=train_metrics["pure_allgather_uniform"],
                    validation_metrics=validation_metrics[
                        "pure_allgather_uniform"
                    ],
                    row_format=row_format,
                    block_type=f"{model_type}_attention_o_proj",
                ),
            )
        )
        layer_peak = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_cuda = max(peak_cuda, layer_peak)
        layer_metadata[str(layer)] = {
            **module_metadata[layer],
            "elapsed_seconds": time.perf_counter() - layer_started,
            "peak_cuda_allocated_bytes": layer_peak,
            "train_local_dense_consistency": train_consistency,
            "validation_local_dense_consistency": validation_consistency,
        }
        del (
            validation_activation,
            validation_teacher,
            validation_local,
            allreduce_factors,
            allgather_factors,
            shared_pod,
            shared_basis,
        )
        torch.cuda.empty_cache()

    decisions = _decisions(all_records)
    _atomic_text(
        partial_dir / "results.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_records),
    )
    _atomic_text(partial_dir / "summary.csv", _summary_csv(all_records))
    _atomic_text(
        partial_dir / "summary.md",
        _summary_markdown(
            decisions,
            tp_size=args.tp,
            allreduce_rank=args.baseline_rank,
            allgather_rank_per_tp=private_rank,
            model_label=model_label,
        ),
    )
    elapsed = time.perf_counter() - started
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    manifest = {
        "format": artifact_format,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": timestamp_started,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "model_type": model_type,
            "config_sha256": _file_sha256(config_path),
            "safetensors_index_sha256": _file_sha256(index_path),
            "hidden_size": hidden_size,
            "o_proj_input_width": o_proj_input_width,
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": model_num_hidden_layers,
        },
        "layers": list(layers),
        "tp_size": args.tp,
        "baseline_rank": args.baseline_rank,
        "pure_allreduce_shared_rank": args.baseline_rank,
        "pure_allgather_private_rank_per_tp": private_rank,
        "pure_allgather_total_rank": private_total,
        "equal_ideal_ring_units": 2 * args.baseline_rank,
        "fit_split": "train",
        "validation_loaded_after_factor_fitting": True,
        "validation_used_for_fit": False,
        "sampling": {
            "positions_per_window": args.positions_per_window,
            "position_seed_train": args.position_seed,
            "position_seed_validation": args.position_seed + 1,
            "sampled_positions_file": "sampled_positions.safetensors",
            "sampled_positions_sha256": _file_sha256(
                partial_dir / "sampled_positions.safetensors"
            ),
            "train_rows": len(train_hashes) * args.positions_per_window,
            "validation_rows": len(validation_hashes)
            * args.positions_per_window,
            "sequence_length": effective_sequence_length,
            "source_sequence_length": source_sequence_length,
        },
        "window_banks": {
            "train": {
                "path": str(train_path),
                "sha256": _file_sha256(train_path),
                "manifest": str(train_path.parent / "manifest.json"),
                "manifest_sha256": _file_sha256(train_path.parent / "manifest.json"),
                "windows": len(train_hashes),
                "source_windows": source_train_windows,
                "selected_split": args.train_window_split,
                "seed": train_manifest.get("seed"),
            },
            "validation": {
                "path": str(validation_path),
                "sha256": _file_sha256(validation_path),
                "manifest": str(validation_path.parent / "manifest.json"),
                "manifest_sha256": _file_sha256(
                    validation_path.parent / "manifest.json"
                ),
                "windows": len(validation_hashes),
                "source_windows": source_validation_windows,
                "selected_split": args.validation_window_split,
                "seed": validation_manifest.get("seed"),
            },
            "overlap_windows": 0,
        },
        "snapshots": {
            split: {str(layer): record for layer, record in records.items()}
            for split, records in snapshot_records.items()
        },
        "attention_control": {
            "post_attention_output_gate": False,
            "purpose": "equal-wire pure AllReduce versus uniform AllGather",
            "modules": {str(key): value for key, value in module_metadata.items()},
        },
        "communication_model": {
            "allreduce_units": f"2*{args.baseline_rank}",
            "allgather_units": f"{args.tp}*{private_rank}",
            "standard_allgather_is_uniform_and_unpadded": True,
            "active_tokens": args.active_tokens,
            "dtype_bytes": args.dtype_bytes,
        },
        "decisions": decisions,
        "rows": len(all_records),
        "layer_metadata": layer_metadata,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "device": str(device),
            "model_device_map_strategy": args.model_device_map,
            "model_device_map": placement,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": torch.cuda.get_device_name(cuda_index),
            "torch_num_threads": torch.get_num_threads(),
            "collection_peak_cuda_allocated_bytes": collection_peak_cuda,
            "collection_peak_cuda_allocated_bytes_by_device": (
                collection_peak_cuda_by_device
            ),
            "peak_cuda_allocated_bytes": peak_cuda,
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[OProjEndpoint] complete output={output_dir} rows={len(all_records)} "
        f"seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
