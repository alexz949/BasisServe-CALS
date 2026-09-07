#!/usr/bin/env python3
"""Fit repository-faithful pre-RoPE Loki Key PCA."""

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

from datasets import load_dataset
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


FORMAT = "basisserve.qwen3_8b.loki_key_pca.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


class KeyMomentCollector:
    """Accumulate centered-PCA sufficient statistics after Qwen Key RMSNorm."""

    def __init__(
        self,
        *,
        layers: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        self.rows = torch.zeros(layers, dtype=torch.int64)
        self.row_sum = torch.zeros(layers, kv_heads, head_dim, dtype=torch.float64)
        self.gram = torch.zeros(
            layers, kv_heads, head_dim, head_dim, dtype=torch.float64
        )

    def hook(self, layer: int):
        def accumulate(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            batch, tokens, kv_heads, head_dim = map(int, output.shape)
            grouped = (
                output.detach()
                .float()
                .permute(2, 0, 1, 3)
                .reshape(kv_heads, batch * tokens, head_dim)
                .contiguous()
            )
            self.rows[layer] += batch * tokens
            self.row_sum[layer] += grouped.sum(dim=1).double().cpu()
            self.gram[layer] += torch.bmm(grouped.mT, grouped).double().cpu()

        return accumulate


def _fit_pca(
    rows: torch.Tensor,
    row_sum: torch.Tensor,
    gram: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = (
        gram
        - torch.einsum("lgd,lge->lgde", row_sum, row_sum) / rows[:, None, None, None]
    )
    centered = (centered + centered.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(centered)
    spectrum = eigenvalues.flip(-1).clamp_min(0.0)
    projector = eigenvectors.flip(-1)[..., :rank].contiguous()
    total = torch.diagonal(centered, dim1=-2, dim2=-1).sum(dim=-1)
    retained = spectrum[..., :rank].sum(dim=-1) / total
    mean = row_sum / rows[:, None, None]
    return projector, mean, spectrum, retained


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=REPO_ROOT / "results/cache/huggingface/datasets",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        help="prepared input_ids safetensors; otherwise use WikiText validation",
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    started = time.perf_counter()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_index = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank_index)))
    assert torch.cuda.is_available()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    model_root = args.model.expanduser().resolve()
    dataset_cache = args.dataset_cache.expanduser().resolve()
    assert args.sequence_length > 1
    assert args.max_windows is None or args.max_windows > 0
    prepared_windows = None
    dataset_identity: dict[str, Any]
    source_identity: dict[str, Any]
    if args.windows is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_root,
            local_files_only=True,
            use_fast=True,
        )
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="validation",
            cache_dir=str(dataset_cache),
            download_mode="reuse_dataset_if_exists",
        )
        text = "\n\n".join(str(row["text"]) for row in dataset)
        token_stream = (
            tokenizer(
                text,
                return_tensors="pt",
            )
            .input_ids[0]
            .to(dtype=torch.int32)
        )
        complete_windows = int(token_stream.numel() // args.sequence_length)
        dataset_identity = {
            "repo": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "validation",
            "fingerprint": dataset._fingerprint,
            "cache_dir": str(dataset_cache),
        }
        source_identity = {
            "kind": "token_stream",
            "token_stream_sha256": _tensor_sha256(token_stream),
            "tokenized_total": int(token_stream.numel()),
        }
    else:
        windows_path = args.windows.expanduser().resolve()
        prepared_windows = load_file(str(windows_path), device="cpu")["input_ids"]
        assert prepared_windows.ndim == 2
        assert int(prepared_windows.shape[1]) == args.sequence_length
        prepared_windows = prepared_windows.to(dtype=torch.int32)
        complete_windows = int(prepared_windows.shape[0])
        manifest_path = windows_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["artifact"]["sha256"] == _sha256(windows_path)
        dataset_identity = manifest["dataset"]
        source_identity = {
            "kind": "prepared_windows",
            "windows_file": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest": str(manifest_path),
            "windows_manifest_sha256": _sha256(manifest_path),
            "windows_format": manifest["format"],
        }
    window_count = (
        complete_windows
        if args.max_windows is None
        else min(complete_windows, args.max_windows)
    )
    assert window_count > 0

    model = AutoModelForCausalLM.from_pretrained(
        model_root,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.eval()
    layers = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    query_heads = int(model.config.num_attention_heads)
    head_dim = int(
        getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // query_heads,
        )
    )
    assert 0 < args.rank <= head_dim
    collector = KeyMomentCollector(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    handles = [
        layer.self_attn.k_norm.register_forward_hook(collector.hook(index))
        for index, layer in enumerate(model.model.layers)
    ]
    local_indices = list(range(rank_index, window_count, world_size))
    for progress, window_index in enumerate(local_indices, start=1):
        if prepared_windows is None:
            start = window_index * args.sequence_length
            stop = start + args.sequence_length
            input_ids = token_stream[start:stop]
        else:
            input_ids = prepared_windows[window_index]
        model.model(
            input_ids=input_ids.to(device=device, dtype=torch.long).unsqueeze(0),
            use_cache=False,
            return_dict=False,
        )
        print(
            f"[Loki PCA] rank={rank_index} window={window_index} "
            f"progress={progress}/{len(local_indices)}",
            flush=True,
        )
    for handle in handles:
        handle.remove()
    del model
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.all_reduce(collector.rows)
        dist.all_reduce(collector.row_sum)
        dist.all_reduce(collector.gram)
    if rank_index == 0:
        projector, mean, spectrum, retained = _fit_pca(
            collector.rows,
            collector.row_sum,
            collector.gram,
            rank=args.rank,
        )
        output_root = args.output_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        factor_path = output_root / "factors.safetensors"
        _atomic_safetensors(
            factor_path,
            {
                "key_projector": projector.float(),
                "fit_mean": mean.float(),
                "fit_spectrum": spectrum.float(),
            },
        )
        fit_values = retained.flatten().tolist()
        result = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "model": {
                "path": str(model_root),
                "config_sha256": _sha256(model_root / "config.json"),
            },
            "geometry": {
                "layers": layers,
                "query_heads": query_heads,
                "physical_kv_heads": kv_heads,
                "query_heads_per_kv_head": query_heads // kv_heads,
                "head_dim": head_dim,
                "rank": args.rank,
            },
            "calibration": {
                "dataset": dataset_identity,
                **source_identity,
                "samples": window_count,
                "complete_samples": complete_windows,
                "sequence_length": args.sequence_length,
                "trailing_tokens_discarded": (
                    int(token_stream.numel() - window_count * args.sequence_length)
                    if prepared_windows is None
                    else 0
                ),
                "rows_per_kv_head": int(collector.rows[0]),
            },
            "method": {
                "name": "Loki Key PCA",
                "repository": "https://github.com/hpcgroup/loki",
                "repository_commit": "005913bc0b64c3b54d0d96871f4e51e799d7b17b",
                "fit_coordinate": "pre_rope_after_key_rmsnorm",
                "runtime_coordinate": "post_rope",
                "centering": "per-layer per-physical-KV-head",
                "runtime_mean": "omitted as in the Loki repository",
                "query_and_key_projector": "shared orthonormal PCA basis",
                "covariance_accumulation": "FP32 GPU BMM into FP64 CPU moments",
                "damping": 0.0,
            },
            "metrics": {
                "fit_explained_variance_mean": sum(fit_values) / len(fit_values),
                "fit_explained_variance_minimum": min(fit_values),
                "fit_explained_variance_maximum": max(fit_values),
            },
            "artifacts": {
                "factors": {
                    "file": factor_path.name,
                    "sha256": _sha256(factor_path),
                    "dtype": "float32",
                }
            },
            "elapsed_seconds": time.perf_counter() - started,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "world_size": world_size,
                "torch_num_threads": torch.get_num_threads(),
            },
        }
        _atomic_json(output_root / "result.json", result)
        print(
            f"[Loki PCA] result={output_root / 'result.json'} "
            f"variance={result['metrics']['fit_explained_variance_mean']:.6f}",
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
