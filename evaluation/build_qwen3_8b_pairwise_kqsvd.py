#!/usr/bin/env python3
"""Fit matched-storage independent and adjacent-layer pairwise Qwen3 KQ-SVD."""

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

from basisserve.core.kq_svd import kq_svd_projectors  # noqa: E402
from basisserve.core.pairwise_kq_svd import (  # noqa: E402
    block_diagonal_pair_factors,
    fit_pairwise_kq_svd,
    pairwise_score_squared_errors,
)


FORMAT = "basisserve.qwen3_8b.pairwise_kq_svd.v1"
NUM_LAYERS = 36
NUM_PAIRS = NUM_LAYERS // 2
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
QUERY_HEADS_PER_KV_HEAD = NUM_QUERY_HEADS // NUM_KV_HEADS
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


class PairwisePostRopeGramAccumulator:
    """Accumulate complete post-RoPE Q Grams and aligned cross-layer K Grams."""

    def __init__(self) -> None:
        self.pair_key_gram = torch.zeros(
            NUM_PAIRS,
            NUM_KV_HEADS,
            2 * HEAD_DIM,
            2 * HEAD_DIM,
            dtype=torch.float64,
        )
        self.query_gram = torch.zeros(
            NUM_LAYERS,
            NUM_KV_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float64,
        )
        self.key_rows = torch.zeros(NUM_PAIRS, dtype=torch.int64)
        self.query_rows = torch.zeros(NUM_LAYERS, dtype=torch.int64)
        self.active = False
        self.pending_even_key: Tensor | None = None
        self.pending_even_layer: int | None = None

    def hook(self, layer_index: int):
        def accumulate(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            if not self.active:
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
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            grouped_query = (
                query.reshape(
                    batch,
                    NUM_KV_HEADS,
                    QUERY_HEADS_PER_KV_HEAD,
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
            self.query_gram[layer_index].add_(query_gram.double().cpu())
            self.query_rows[layer_index] += int(grouped_query.shape[1])

            if layer_index % 2 == 0:
                if self.pending_even_key is not None:
                    raise RuntimeError("an earlier even-layer Key is still pending")
                self.pending_even_key = grouped_key.detach()
                self.pending_even_layer = layer_index
                return
            if self.pending_even_key is None or self.pending_even_layer != layer_index - 1:
                raise RuntimeError("odd layer did not follow its paired even layer")
            if self.pending_even_key.shape != grouped_key.shape:
                raise RuntimeError("paired layer Key activations have different geometry")
            pair_key = torch.cat((self.pending_even_key, grouped_key), dim=-1)
            pair_gram = torch.bmm(pair_key.mT, pair_key)
            pair_index = layer_index // 2
            self.pair_key_gram[pair_index].add_(pair_gram.double().cpu())
            self.key_rows[pair_index] += int(pair_key.shape[1])
            self.pending_even_key = None
            self.pending_even_layer = None

        return accumulate

    def finish_batch(self) -> None:
        if self.pending_even_key is not None or self.pending_even_layer is not None:
            raise RuntimeError("calibration forward ended with an incomplete layer pair")


def _run_windows(
    model: nn.Module,
    windows: Tensor,
    *,
    batch_size: int,
    accumulator: PairwisePostRopeGramAccumulator,
) -> None:
    accumulator.active = True
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
            accumulator.finish_batch()
            del output, input_ids
            print(
                f"[pairwise KQ-SVD Gram] calibration={stop}/{len(windows)} "
                f"batch_size={batch_size}",
                flush=True,
            )
    finally:
        accumulator.active = False


def _score_summary(error: Tensor, energy: Tensor) -> dict[str, float]:
    relative = error / energy.clamp_min(torch.finfo(energy.dtype).tiny)
    return {
        "weighted_relative_squared_error": float(error.sum() / energy.sum()),
        "mean_layer_group_relative_squared_error": float(relative.mean()),
        "median_layer_group_relative_squared_error": float(relative.median()),
        "maximum_layer_group_relative_squared_error": float(relative.max()),
    }


@torch.inference_mode()
def build(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("pairwise KQ-SVD calibration requires CUDA")
    positive = (
        args.fit_windows,
        args.sequence_length,
        args.independent_rank,
        args.pair_rank,
        args.batch_size,
        args.torch_num_threads,
        args.max_memory_per_gpu_gib,
    )
    if min(positive) <= 0:
        raise ValueError("calibration and compute arguments must be positive")
    if args.sequence_length not in (2048, 4096, 32768):
        raise ValueError(
            "the controlled KQ-SVD experiment requires seq2048, seq4096, or seq32768"
        )
    if args.fit_start < 0:
        raise ValueError("fit start must be nonnegative")
    if (args.independent_rank, args.pair_rank) != (64, 128):
        raise ValueError("the controlled comparison requires rank64+64 versus pair-rank128")
    if args.pair_rank != 2 * args.independent_rank:
        raise ValueError("matched storage requires pair rank = 2 * independent rank")
    if args.independent_rank > HEAD_DIM or args.pair_rank > 2 * HEAD_DIM:
        raise ValueError("requested KQ-SVD rank exceeds its source width")

    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_config(AutoConfig.from_pretrained(str(model_path), local_files_only=True))
    manifest_path = windows_path.parent / "manifest.json"
    windows_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if windows_manifest["model"]["config_sha256"] != _sha256(
        model_path / "config.json"
    ):
        raise ValueError("C4 windows belong to another model config")
    if windows_manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("C4 windows hash does not match its manifest")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    if stored.ndim != 2 or int(stored.shape[1]) != args.sequence_length:
        raise ValueError("C4 window tensor has incompatible geometry")
    fit_stop = args.fit_start + args.fit_windows
    if fit_stop > len(stored):
        raise ValueError("requested calibration windows exceed the stored bank")
    windows = stored[args.fit_start:fit_stop]

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
    accumulator = PairwisePostRopeGramAccumulator()
    handles = [
        layer.self_attn.register_forward_pre_hook(
            accumulator.hook(layer_index),
            with_kwargs=True,
        )
        for layer_index, layer in enumerate(model.model.layers)
    ]
    try:
        _run_windows(
            model,
            windows,
            batch_size=args.batch_size,
            accumulator=accumulator,
        )
    finally:
        for handle in handles:
            handle.remove()
    del model
    torch.cuda.empty_cache()

    expected_key_rows = args.fit_windows * args.sequence_length
    expected_query_rows = expected_key_rows * QUERY_HEADS_PER_KV_HEAD
    if not torch.all(accumulator.key_rows == expected_key_rows):
        raise RuntimeError("incomplete pair-Key rows")
    if not torch.all(accumulator.query_rows == expected_query_rows):
        raise RuntimeError("incomplete grouped-Query rows")

    independent_keys = []
    independent_group_queries = []
    pair_keys = []
    pair_group_queries = []
    independent_errors = []
    pair_errors = []
    energies = []
    full_rank_relative_maximum = 0.0
    per_pair = []
    for pair_index in range(NUM_PAIRS):
        pair_key_gram = accumulator.pair_key_gram[pair_index]
        query_gram = accumulator.query_gram[2 * pair_index : 2 * pair_index + 2]
        layer_keys = []
        layer_queries = []
        for layer_slot in (0, 1):
            start = layer_slot * HEAD_DIM
            stop = start + HEAD_DIM
            key_factor, query_factor, _ = kq_svd_projectors(
                pair_key_gram[:, start:stop, start:stop],
                query_gram[layer_slot],
                args.independent_rank,
            )
            layer_keys.append(key_factor)
            layer_queries.append(query_factor)
            independent_keys.append(key_factor)
            independent_group_queries.append(query_factor)
        block_key, block_query = block_diagonal_pair_factors(
            torch.stack(layer_keys),
            torch.stack(layer_queries),
        )
        independent_error, energy, _ = pairwise_score_squared_errors(
            pair_key_gram,
            query_gram,
            block_key,
            block_query,
        )
        pair_result = fit_pairwise_kq_svd(
            pair_key_gram,
            query_gram,
            args.pair_rank,
        )
        pair_error, pair_energy, _ = pairwise_score_squared_errors(
            pair_key_gram,
            query_gram,
            pair_result.key_projector,
            pair_result.query_projector,
        )
        torch.testing.assert_close(pair_energy, energy)
        tolerance = 1e-10 * energy.sum(dim=0)
        if torch.any(pair_error.sum(dim=0) > independent_error.sum(dim=0) + tolerance):
            raise RuntimeError(
                f"pair {pair_index} closed-form objective lost to block-diagonal baseline"
            )
        full_rank = fit_pairwise_kq_svd(pair_key_gram, query_gram, 2 * HEAD_DIM)
        _, _, full_rank_relative = pairwise_score_squared_errors(
            pair_key_gram,
            query_gram,
            full_rank.key_projector,
            full_rank.query_projector,
        )
        full_rank_relative_maximum = max(
            full_rank_relative_maximum,
            float(full_rank_relative.max()),
        )
        if float(full_rank_relative.max()) > 1e-12:
            raise RuntimeError(f"pair {pair_index} failed the full-rank reconstruction gate")

        pair_keys.append(pair_result.key_projector)
        pair_group_queries.extend(
            (pair_result.query_projector[0], pair_result.query_projector[1])
        )
        independent_errors.append(independent_error)
        pair_errors.append(pair_error)
        energies.append(energy)
        independent_weighted = float(independent_error.sum() / energy.sum())
        pair_weighted = float(pair_error.sum() / energy.sum())
        per_pair.append(
            {
                "pair_index": pair_index,
                "layers": [2 * pair_index, 2 * pair_index + 1],
                "independent_weighted_relative_squared_error": independent_weighted,
                "pairwise_weighted_relative_squared_error": pair_weighted,
                "pairwise_relative_error_reduction": (
                    1.0 - pair_weighted / independent_weighted
                ),
            }
        )
        print(
            f"[pairwise KQ-SVD fit] pair={pair_index:02d} "
            f"independent={independent_weighted:.8e} pairwise={pair_weighted:.8e}",
            flush=True,
        )

    independent_error = torch.stack(independent_errors)
    pair_error = torch.stack(pair_errors)
    energy = torch.stack(energies)
    independent_summary = _score_summary(independent_error, energy)
    pair_summary = _score_summary(pair_error, energy)
    independent_key = torch.stack(independent_keys).float().contiguous()
    independent_group_query = torch.stack(independent_group_queries)
    independent_query = independent_group_query.repeat_interleave(
        QUERY_HEADS_PER_KV_HEAD,
        dim=1,
    ).float().contiguous()
    pair_key = torch.stack(pair_keys).float().contiguous()
    pair_group_query = torch.stack(pair_group_queries)
    pair_query = pair_group_query.repeat_interleave(
        QUERY_HEADS_PER_KV_HEAD,
        dim=1,
    ).float().contiguous()

    output_dir.mkdir(parents=True)
    statistics_path = output_dir / "statistics.safetensors"
    factors_path = output_dir / "factors.safetensors"
    _atomic_safetensors(
        statistics_path,
        {
            "pair_key_gram": accumulator.pair_key_gram.contiguous(),
            "query_gram": accumulator.query_gram.contiguous(),
        },
    )
    factor_tensors = {
        "independent_key_projector": independent_key,
        "independent_query_projector": independent_query,
        "pair_key_projector": pair_key,
        "pair_query_projector": pair_query,
    }
    _atomic_safetensors(factors_path, factor_tensors)
    independent_weighted = independent_summary["weighted_relative_squared_error"]
    pair_weighted = pair_summary["weighted_relative_squared_error"]
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
            "adjacent_layer_pairs": NUM_PAIRS,
            "query_heads": NUM_QUERY_HEADS,
            "physical_kv_heads": NUM_KV_HEADS,
            "query_heads_per_kv_head": QUERY_HEADS_PER_KV_HEAD,
            "head_dim": HEAD_DIM,
            "independent_rank_per_layer": args.independent_rank,
            "pair_rank_per_layer_pair": args.pair_rank,
            "matched_key_scalars_per_pair_token": args.pair_rank,
            "logical_key_cache_ratio_vs_bf16_dense": args.pair_rank / (2 * HEAD_DIM),
            "logical_dense_v_total_kv_ratio_vs_bf16_dense": (
                args.pair_rank / (2 * HEAD_DIM) + 1.0
            ) / 2.0,
        },
        "calibration": {
            "dataset": (
                "C4 train packed document windows"
                if "packing" in windows_manifest
                else "C4 train full documents"
            ),
            "window_format": windows_manifest.get("format"),
            "packing": windows_manifest.get("packing"),
            "windows_path": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest": str(manifest_path),
            "windows_manifest_sha256": _sha256(manifest_path),
            "fit_start": args.fit_start,
            "fit_windows": args.fit_windows,
            "sequence_length": args.sequence_length,
            "position_ids": f"contiguous 0..{args.sequence_length - 1} per window",
            "key_rows_per_layer_head": expected_key_rows,
            "query_rows_per_layer_group": expected_query_rows,
            "query_row_sampling": "none; all post-RoPE Query rows from all four GQA heads",
            "heldout_windows": 0,
        },
        "method": {
            "coordinate": "post-RoPE, after Qwen3 q_norm/k_norm",
            "objective": (
                "unmasked pre-softmax score Frobenius error over the complete "
                "Cartesian product of calibration Key and grouped-Query rows"
            ),
            "pair_code": (
                "one token-aligned rank-128 code formed by summing the two "
                "adjacent layers' slices of a shared 256x128 Key encoder"
            ),
            "solve": "closed-form reduced-rank regression from undamped float64 Grams",
            "independent_control": (
                "closed-form rank-64 KQ-SVD per layer, embedded as an exactly "
                "equivalent block-diagonal rank-128 pair codec"
            ),
            "gqa_query_aggregation": (
                "concatenate all four Query heads assigned to each physical KV head"
            ),
            "causal_mask": False,
            "softmax": False,
            "value_aware": False,
            "damping": 0.0,
        },
        "score_frobenius_metrics": {
            "independent_rank64_plus_rank64": independent_summary,
            "pairwise_rank128": pair_summary,
            "pairwise_weighted_relative_error_reduction": (
                1.0 - pair_weighted / independent_weighted
            ),
            "full_rank_maximum_relative_squared_error": full_rank_relative_maximum,
            "per_pair": per_pair,
        },
        "artifacts": {
            "statistics": {
                "file": statistics_path.name,
                "sha256": _sha256(statistics_path),
                "dtype": "float64",
                "tensors": {
                    "pair_key_gram": list(accumulator.pair_key_gram.shape),
                    "query_gram": list(accumulator.query_gram.shape),
                },
            },
            "factors": {
                "file": factors_path.name,
                "sha256": _sha256(factors_path),
                "dtype": "float32",
                "tensors": {
                    name: list(tensor.shape) for name, tensor in factor_tensors.items()
                },
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
    print(json.dumps(payload["score_frobenius_metrics"], indent=2, sort_keys=True), flush=True)
    print(f"[pairwise KQ-SVD] wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--independent-rank", type=int, default=64)
    parser.add_argument("--pair-rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
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
