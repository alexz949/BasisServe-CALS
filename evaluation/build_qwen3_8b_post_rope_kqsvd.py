#!/usr/bin/env python3
"""Fit post-RoPE rank-64 K-SVD and KQ-SVD factors for Qwen3-8B."""

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
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.kq_svd import (  # noqa: E402
    key_svd_projector,
    kq_svd_projectors,
    relative_score_frobenius_error,
)


FORMAT = "basisserve.qwen3_8b.post_rope_kqsvd.v1"
NUM_LAYERS = 36
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _validate_config(config: Any) -> None:
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
    )
    expected = (
        "qwen3",
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    if observed != expected:
        raise ValueError(f"expected Qwen3-8B geometry {expected}, found {observed}")


class PostRopeGramAccumulator:
    """Accumulate per-layer/per-KV-head K and grouped-Q sufficient statistics."""

    def __init__(self) -> None:
        shape = (NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, HEAD_DIM)
        self.grams = {
            split: {
                "key": torch.zeros(shape, dtype=torch.float64),
                "query": torch.zeros(shape, dtype=torch.float64),
            }
            for split in ("fit", "heldout")
        }
        self.rows = {
            split: {
                "key": torch.zeros(NUM_LAYERS, dtype=torch.int64),
                "query": torch.zeros(NUM_LAYERS, dtype=torch.int64),
            }
            for split in ("fit", "heldout")
        }
        self.active_split: str | None = None

    def hook(self, layer_index: int):
        def accumulate(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            split = self.active_split
            if split is None:
                return
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                if not args:
                    raise RuntimeError("Qwen attention hook did not receive hidden states")
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError("Qwen attention hook requires position embeddings")
            batch, sequence, _ = hidden_states.shape
            query = module.q_norm(
                module.q_proj(hidden_states).view(
                    batch,
                    sequence,
                    NUM_QUERY_HEADS,
                    HEAD_DIM,
                )
            ).transpose(1, 2)
            key = module.k_norm(
                module.k_proj(hidden_states).view(
                    batch,
                    sequence,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                )
            ).transpose(1, 2)
            query, key = apply_rotary_pos_emb(
                query,
                key,
                *position_embeddings,
            )
            grouped_query = (
                query.reshape(
                    batch,
                    NUM_KV_HEADS,
                    NUM_QUERY_HEADS // NUM_KV_HEADS,
                    sequence,
                    HEAD_DIM,
                )
                .permute(1, 0, 2, 3, 4)
                .reshape(NUM_KV_HEADS, -1, HEAD_DIM)
                .float()
            )
            grouped_key = (
                key.permute(1, 0, 2, 3)
                .reshape(NUM_KV_HEADS, -1, HEAD_DIM)
                .float()
            )
            query_gram = torch.bmm(grouped_query.mT, grouped_query)
            key_gram = torch.bmm(grouped_key.mT, grouped_key)
            self.grams[split]["query"][layer_index].add_(
                query_gram.double().cpu()
            )
            self.grams[split]["key"][layer_index].add_(key_gram.double().cpu())
            self.rows[split]["query"][layer_index] += grouped_query.shape[1]
            self.rows[split]["key"][layer_index] += grouped_key.shape[1]

        return accumulate


def _score_summary(
    key_gram: Tensor,
    query_gram: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
) -> dict[str, float]:
    ratios = relative_score_frobenius_error(
        key_gram,
        query_gram,
        key_projector,
        query_projector,
    )
    denominators = torch.diagonal(
        key_gram @ query_gram,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    return {
        "weighted_relative_squared_error": float(
            (ratios * denominators).sum() / denominators.sum()
        ),
        "mean_head_relative_squared_error": float(ratios.mean()),
        "median_head_relative_squared_error": float(ratios.median()),
        "maximum_head_relative_squared_error": float(ratios.max()),
    }


def _run_windows(
    model: nn.Module,
    windows: Tensor,
    *,
    split: str,
    batch_size: int,
    accumulator: PostRopeGramAccumulator,
) -> None:
    accumulator.active_split = split
    device = next(model.parameters()).device
    try:
        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            input_ids = windows[start:stop].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            output = model.model(input_ids=input_ids, use_cache=False)
            del output, input_ids
            print(
                f"[post-RoPE Gram] {split}: {stop}/{len(windows)} "
                f"batch_size={batch_size}",
                flush=True,
            )
    finally:
        accumulator.active_split = None


@torch.inference_mode()
def build(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("post-RoPE KQ-SVD calibration requires CUDA")
    positive = (
        args.fit_windows,
        args.heldout_windows,
        args.sequence_length,
        args.rank,
        args.batch_size,
        args.torch_num_threads,
        args.max_memory_per_gpu_gib,
    )
    if min(positive) <= 0:
        raise ValueError("calibration and compute arguments must be positive")
    if args.sequence_length != 2048:
        raise ValueError("the controlled KQ-SVD experiment requires seq2048")
    if args.fit_start < 0 or args.heldout_start < 0:
        raise ValueError("window starts must be nonnegative")
    fit_range = set(range(args.fit_start, args.fit_start + args.fit_windows))
    heldout_range = set(
        range(args.heldout_start, args.heldout_start + args.heldout_windows)
    )
    if fit_range & heldout_range:
        raise ValueError("fit and heldout C4 windows must be disjoint")

    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_config(
        AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    )
    manifest_path = windows_path.parent / "manifest.json"
    windows_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if windows_manifest["model"]["config_sha256"] != _sha256(
        model_path / "config.json"
    ):
        raise ValueError("C4 windows belong to another model config")
    if windows_manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("C4 windows hash does not match its manifest")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    if stored.ndim != 2 or stored.shape[1] != args.sequence_length:
        raise ValueError("C4 window tensor has incompatible geometry")
    stop = max(
        args.fit_start + args.fit_windows,
        args.heldout_start + args.heldout_windows,
    )
    if stop > len(stored):
        raise ValueError("requested calibration windows exceed the stored bank")
    fit_windows = stored[args.fit_start : args.fit_start + args.fit_windows]
    heldout_windows = stored[
        args.heldout_start : args.heldout_start + args.heldout_windows
    ]

    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    accumulator = PostRopeGramAccumulator()
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        handles.append(
            layer.self_attn.register_forward_pre_hook(
                accumulator.hook(layer_index),
                with_kwargs=True,
            )
        )
    try:
        _run_windows(
            model,
            fit_windows,
            split="fit",
            batch_size=args.batch_size,
            accumulator=accumulator,
        )
        _run_windows(
            model,
            heldout_windows,
            split="heldout",
            batch_size=args.batch_size,
            accumulator=accumulator,
        )
    finally:
        for handle in handles:
            handle.remove()
    del model
    torch.cuda.empty_cache()

    expected_key_rows = {
        "fit": args.fit_windows * args.sequence_length,
        "heldout": args.heldout_windows * args.sequence_length,
    }
    for split in ("fit", "heldout"):
        expected_k = expected_key_rows[split]
        expected_q = expected_k * (NUM_QUERY_HEADS // NUM_KV_HEADS)
        if not torch.all(accumulator.rows[split]["key"] == expected_k):
            raise RuntimeError(f"incomplete {split} Key rows")
        if not torch.all(accumulator.rows[split]["query"] == expected_q):
            raise RuntimeError(f"incomplete {split} grouped-Query rows")

    fit_key = accumulator.grams["fit"]["key"]
    fit_query = accumulator.grams["fit"]["query"]
    key_svd, key_spectrum = key_svd_projector(fit_key, args.rank)
    kq_key, kq_query, kq_spectrum = kq_svd_projectors(
        fit_key,
        fit_query,
        args.rank,
    )
    score_metrics = {}
    for split in ("fit", "heldout"):
        key_gram = accumulator.grams[split]["key"]
        query_gram = accumulator.grams[split]["query"]
        score_metrics[split] = {
            "key_svd": _score_summary(
                key_gram,
                query_gram,
                key_svd,
                key_svd,
            ),
            "kq_svd": _score_summary(
                key_gram,
                query_gram,
                kq_key,
                kq_query,
            ),
        }

    output_dir.mkdir(parents=True)
    statistics_path = output_dir / "statistics.safetensors"
    factors_path = output_dir / "factors.safetensors"
    _atomic_safetensors(
        statistics_path,
        {
            "fit_key_gram": fit_key.contiguous(),
            "fit_query_gram": fit_query.contiguous(),
            "heldout_key_gram": accumulator.grams["heldout"]["key"].contiguous(),
            "heldout_query_gram": accumulator.grams["heldout"]["query"].contiguous(),
        },
    )
    _atomic_safetensors(
        factors_path,
        {
            "key_svd_projector": key_svd.float().contiguous(),
            "key_svd_spectrum": key_spectrum.float().contiguous(),
            "kq_svd_key_projector": kq_key.float().contiguous(),
            "kq_svd_query_projector": kq_query.float().contiguous(),
            "kq_svd_spectrum": kq_spectrum.float().contiguous(),
        },
    )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "geometry": {
            "layers": NUM_LAYERS,
            "query_heads": NUM_QUERY_HEADS,
            "physical_kv_heads": NUM_KV_HEADS,
            "query_heads_per_kv_head": NUM_QUERY_HEADS // NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rank": args.rank,
        },
        "calibration": {
            "dataset": "C4 train full documents",
            "windows_path": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest": str(manifest_path),
            "windows_manifest_sha256": _sha256(manifest_path),
            "fit_start": args.fit_start,
            "fit_windows": args.fit_windows,
            "heldout_start": args.heldout_start,
            "heldout_windows": args.heldout_windows,
            "sequence_length": args.sequence_length,
            "fit_key_rows_per_layer_head": expected_key_rows["fit"],
            "fit_query_rows_per_layer_group": (
                expected_key_rows["fit"] * NUM_QUERY_HEADS // NUM_KV_HEADS
            ),
            "heldout_disjoint_from_fit": True,
        },
        "method": {
            "coordinate": "post-RoPE, after Qwen3 q_norm/k_norm",
            "gqa_query_aggregation": (
                "concatenate all four Query heads assigned to each physical KV head"
            ),
            "key_svd": "top eigenspace of post-RoPE Key Gram",
            "kq_svd": (
                "closed-form truncated SVD of R_K R_Q^T from undamped Cholesky Grams"
            ),
            "damping": 0.0,
            "factor_balancing": "per-column norm balancing; operator unchanged",
        },
        "score_frobenius_metrics": score_metrics,
        "artifacts": {
            "statistics": {
                "file": statistics_path.name,
                "sha256": _sha256(statistics_path),
                "dtype": "float64",
            },
            "factors": {
                "file": factors_path.name,
                "sha256": _sha256(factors_path),
                "dtype": "float32",
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / "result.json", payload)
    print(json.dumps(score_metrics, indent=2, sort_keys=True), flush=True)
    print(f"[post-RoPE KQ-SVD] wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-windows", type=int, default=128)
    parser.add_argument("--heldout-start", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
