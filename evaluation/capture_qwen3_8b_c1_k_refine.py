#!/usr/bin/env python3
"""Capture real Qwen3-8B post-RoPE Q/exact-K/C1-V layer replays."""

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
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.fit_qwen3_8b_c1_k_output_closure import (  # noqa: E402
    HIDDEN_SIZE,
    LayerFeatureFactory,
    _batches,
    _load_layer_c1_factors,
    _propagate_dense_layer,
    _validate_config,
)
from basisserve.core.c1_k_proxy_fit import sample_valid_causal_pairs  # noqa: E402


FORMAT = "basisserve.qwen3_8b.c1_k_refine_capture.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _fit_pair_tensors(
    query_chunks: list[torch.Tensor],
    key_chunks: list[torch.Tensor],
    head_chunks: list[torch.Tensor],
    *,
    head_dim: int,
    query_dtype: torch.dtype,
    key_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_chunks:
        return (
            torch.cat(query_chunks).contiguous(),
            torch.cat(key_chunks).contiguous(),
            torch.cat(head_chunks).contiguous(),
        )
    return (
        torch.empty(0, head_dim, dtype=query_dtype),
        torch.empty(0, head_dim, dtype=key_dtype),
        torch.empty(0, dtype=torch.long),
    )


@torch.inference_mode()
def capture(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real C1-KRefine capture requires CUDA")
    if min(
        args.windows,
        args.sequence_length,
        args.batch_size,
        args.queries_per_sequence,
        args.keys_per_query,
    ) <= 0:
        raise ValueError("capture sizes must be positive")
    if not 0 <= args.fit_pair_windows <= args.windows:
        raise ValueError("fit-pair windows must be in [0, windows]")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model_path = args.model_path.expanduser().resolve()
    windows_path = args.calibration_data.expanduser().resolve()
    c1_dir = args.c1_export.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    window_manifest_path = windows_path.parent / "manifest.json"
    window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    if window_manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("calibration window hash does not match its manifest")
    if window_manifest["model"]["config_sha256"] != _sha256(
        model_path / "config.json"
    ):
        raise ValueError("calibration windows belong to another model")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    if stored.ndim != 2 or int(stored.shape[1]) != args.sequence_length:
        raise ValueError("calibration windows have incompatible geometry")
    stop_window = args.window_start + args.windows
    if args.window_start < 0 or stop_window > len(stored):
        raise ValueError("requested capture windows exceed the stored bank")
    windows = stored[args.window_start:stop_window]
    del stored
    query_position = (
        args.sequence_length - 1 if args.query_position < 0 else args.query_position
    )
    if not 0 <= query_position < args.sequence_length:
        raise ValueError("query position is outside the captured sequence")
    visible_sequence = query_position + 1

    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_dir, model_path)
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()
    model.config.use_cache = False
    device = model.model.embed_tokens.weight.device
    position_ids = torch.arange(
        args.sequence_length, device=device, dtype=torch.long
    ).unsqueeze(0)
    hidden_bank = torch.empty(
        args.windows,
        args.sequence_length,
        HIDDEN_SIZE,
        dtype=dtype,
        device="cpu",
    )
    for completed, input_ids in _batches(windows, 0, len(windows), args.batch_size):
        embeddings = model.model.embed_tokens(
            input_ids.to(device=device, dtype=torch.long)
        )
        start = completed - len(input_ids)
        hidden_bank[start:completed].copy_(embeddings.cpu())
    del windows, embeddings
    position_embeddings = model.model.rotary_emb(
        hidden_bank[:1].to(device=device), position_ids
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for layer_index, layer in enumerate(model.model.layers):
        encoder, decoder, c1_artifact = _load_layer_c1_factors(
            c1_dir, c1_result, layer_index, value_rank
        )
        factory = LayerFeatureFactory(
            layer,
            value_encoder=encoder,
            decoder=decoder,
            value_rank=value_rank,
            position_embeddings=position_embeddings,
        )
        query_chunks = []
        key_chunks = []
        value_chunks = []
        pair_query_chunks = []
        pair_key_chunks = []
        pair_head_chunks = []
        for completed, hidden in _batches(
            hidden_bank, 0, len(hidden_bank), args.batch_size
        ):
            grouped_query, exact_key, c1_value, _ = factory(hidden)
            full_query = grouped_query.reshape(
                len(hidden),
                grouped_query.shape[1] * grouped_query.shape[2],
                args.sequence_length,
                grouped_query.shape[-1],
            )
            query_chunks.append(
                full_query[:, :, query_position : query_position + 1].cpu()
            )
            key_chunks.append(exact_key[:, :, :visible_sequence].cpu())
            value_chunks.append(c1_value[:, :, :visible_sequence].cpu())
            batch_start = completed - len(hidden)
            fit_count = max(
                min(args.fit_pair_windows - batch_start, len(hidden)),
                0,
            )
            if fit_count:
                pairs = sample_valid_causal_pairs(
                    full_query[:fit_count],
                    exact_key[:fit_count],
                    queries_per_sequence=args.queries_per_sequence,
                    keys_per_query=args.keys_per_query,
                    recent_pair_fraction=args.recent_pair_fraction,
                    top_score_pair_fraction=args.top_score_pair_fraction,
                    random_seed=(
                        args.random_seed + layer_index * 100_000 + completed
                    ),
                )
                pair_query_chunks.append(pairs.samples.query.cpu())
                pair_key_chunks.append(pairs.samples.key.cpu())
                pair_head_chunks.append(pairs.samples.query_head.cpu())
        filename = f"layer_{layer_index:03d}.safetensors"
        path = output_dir / filename
        fit_pair_query, fit_pair_key, fit_pair_query_head = _fit_pair_tensors(
            pair_query_chunks,
            pair_key_chunks,
            pair_head_chunks,
            head_dim=int(query_chunks[0].shape[-1]),
            query_dtype=query_chunks[0].dtype,
            key_dtype=key_chunks[0].dtype,
        )
        tensors = {
            "query": torch.cat(query_chunks).contiguous(),
            "exact_key": torch.cat(key_chunks).contiguous(),
            "c1_value": torch.cat(value_chunks).contiguous(),
            "fit_pair_query": fit_pair_query,
            "fit_pair_key": fit_pair_key,
            "fit_pair_query_head": fit_pair_query_head,
        }
        _atomic_safetensors(path, tensors)
        artifacts[str(layer_index)] = {
            "file": filename,
            "sha256": _sha256(path),
            "query_shape": list(tensors["query"].shape),
            "exact_key_shape": list(tensors["exact_key"].shape),
            "c1_value_shape": list(tensors["c1_value"].shape),
            "fit_pairs": len(tensors["fit_pair_query"]),
            "c1_factor_file": c1_artifact["file"],
            "c1_factor_sha256": c1_artifact["sha256"],
        }
        print(
            f"[C1-KRefine capture] layer={layer_index} "
            f"windows={args.windows} visible_sequence={visible_sequence}",
            flush=True,
        )
        _propagate_dense_layer(
            model,
            layer,
            hidden_bank,
            batch_size=args.batch_size,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            layer_index=layer_index,
        )
        del (
            factory,
            encoder,
            decoder,
            query_chunks,
            key_chunks,
            value_chunks,
            pair_query_chunks,
            pair_key_chunks,
            pair_head_chunks,
            tensors,
        )
        torch.cuda.empty_cache()

    c1_result_path = c1_dir / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "c1_export": {
            "path": str(c1_dir),
            "results_sha256": _sha256(c1_result_path),
            "value_rank": value_rank,
        },
        "calibration": {
            "windows_path": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "manifest_sha256": _sha256(window_manifest_path),
            "window_start": args.window_start,
            "windows": args.windows,
            "sequence_length": args.sequence_length,
            "query_position": query_position,
            "visible_sequence": visible_sequence,
            "pair_sampling": {
                "queries_per_sequence": args.queries_per_sequence,
                "keys_per_query": args.keys_per_query,
                "fit_pair_windows": args.fit_pair_windows,
                "recent_pair_fraction": args.recent_pair_fraction,
                "top_score_pair_fraction": args.top_score_pair_fraction,
                "random_seed": args.random_seed,
                "weight_mode": "uniform",
            },
        },
        "artifacts": artifacts,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(output_dir / "manifest.json", payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--fit-pair-windows", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--query-position", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--queries-per-sequence", type=int, default=16)
    parser.add_argument("--keys-per-query", type=int, default=64)
    parser.add_argument("--recent-pair-fraction", type=float, default=0.25)
    parser.add_argument("--top-score-pair-fraction", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    capture(_parser().parse_args())
