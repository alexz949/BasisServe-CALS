#!/usr/bin/env python3
"""Diagnose Qwen3-8B Dense versus two-sided-KL C1-R80 generation drift."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shlex
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.sampled_terminal_kl import (  # noqa: E402
    TeacherTerminalStatistics,
    terminal_kl_from_statistics,
)
from evaluation import eval_qwen3_8b_iclr_quality as quality  # noqa: E402


C4_FORMAT = "basisserve.qwen3_8b.c1_r80.c4_position_kl.v1"
GSM_FORMAT = "basisserve.qwen3_8b.c1_r80.gsm_dense_prefix_replay.v1"
WINDOWS_FORMAT = "basisserve.calibration.c4_document_windows.v1"
STRICT_FILTER = "strict-match"
STRICT_ANSWER = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUMBER = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
DECODE_STEP_RANGES = (
    (0, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512),
)


@dataclass(frozen=True)
class StreamingTokenMetrics:
    """Full-vocabulary next-token diagnostics without retaining full logits."""

    logsumexp: Tensor
    top_token: Tensor
    top_logit: Tensor
    top1_margin: Tensor
    target_logit: Tensor
    target_rank: Tensor
    target_margin: Tensor


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def position_ranges(
    sequence_length: int,
    *,
    width: int = 256,
) -> tuple[tuple[int, int], ...]:
    """Return half-open absolute-position bins through ``sequence_length``."""

    return tuple(
        (start, min(sequence_length, start + width))
        for start in range(0, sequence_length, width)
    )


def requested_position_ranges(
    sequence_length: int,
    *,
    width: int = 256,
    long_range_start: int = 2048,
) -> tuple[tuple[int, int], ...]:
    """Return the requested 256-token early bins and one 2048+ tail bin."""

    early_stop = min(sequence_length, long_range_start)
    ranges = list(position_ranges(early_stop, width=width))
    if sequence_length > long_range_start:
        ranges.append((long_range_start, sequence_length))
    return tuple(ranges)


def _range_label(bounds: tuple[int, int]) -> str:
    return f"{bounds[0]}-{bounds[1] - 1}"


def sample_positions_by_fine_bucket(
    *,
    sequence_length: int,
    positions_per_bucket: int,
    seed: int,
    window_index: int,
) -> Tensor:
    """Sample equal counts from each 256-position C4 prediction bucket."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(window_index))
    pieces = []
    for start, stop in position_ranges(sequence_length):
        valid_stop = min(stop, sequence_length - 1)
        count = min(positions_per_bucket, max(0, valid_stop - start))
        if count:
            pieces.append(
                torch.randperm(valid_stop - start, generator=generator)[:count]
                + start
            )
    return torch.cat(pieces).to(torch.long)


@torch.no_grad()
def streaming_token_metrics(
    hidden_states: Tensor,
    output_weight: Tensor,
    target_ids: Tensor,
    *,
    vocab_chunk_size: int,
) -> StreamingTokenMetrics:
    """Compute exact top-2, target rank/margin, and log-normalizer in chunks."""

    original_shape = hidden_states.shape[:-1]
    hidden_size = int(hidden_states.shape[-1])
    flat_hidden = hidden_states.reshape(-1, hidden_size).float()
    flat_targets = target_ids.reshape(-1).to(
        device=flat_hidden.device,
        dtype=torch.long,
    )
    target_weight = output_weight.index_select(0, flat_targets).float()
    target_logits = (flat_hidden * target_weight).sum(dim=-1)
    top_values = torch.full(
        (len(flat_hidden), 2),
        -torch.inf,
        device=flat_hidden.device,
        dtype=torch.float32,
    )
    top_indices = torch.full(
        (len(flat_hidden), 2),
        -1,
        device=flat_hidden.device,
        dtype=torch.long,
    )
    ranks = torch.ones(
        len(flat_hidden), device=flat_hidden.device, dtype=torch.long
    )
    logsumexp: Tensor | None = None
    for start in range(0, int(output_weight.shape[0]), vocab_chunk_size):
        stop = min(int(output_weight.shape[0]), start + vocab_chunk_size)
        weight = output_weight[start:stop].to(
            device=flat_hidden.device,
            dtype=torch.float32,
        )
        logits = F.linear(flat_hidden, weight)
        chunk_lse = torch.logsumexp(logits, dim=-1)
        logsumexp = (
            chunk_lse
            if logsumexp is None
            else torch.logaddexp(logsumexp, chunk_lse)
        )
        greater = logits > target_logits.unsqueeze(-1)
        target_in_chunk = (flat_targets >= start) & (flat_targets < stop)
        target_rows = torch.nonzero(target_in_chunk, as_tuple=False).flatten()
        if len(target_rows):
            greater[
                target_rows,
                flat_targets.index_select(0, target_rows) - start,
            ] = False
        ranks.add_(greater.sum(dim=-1))
        chunk_k = min(2, stop - start)
        chunk_values, chunk_indices = logits.topk(chunk_k, dim=-1)
        chunk_indices.add_(start)
        candidates = torch.cat((top_values, chunk_values), dim=-1)
        candidate_indices = torch.cat((top_indices, chunk_indices), dim=-1)
        top_values, selected = candidates.topk(2, dim=-1)
        top_indices = candidate_indices.gather(1, selected)
        del weight, logits, chunk_lse, greater, target_in_chunk, target_rows
        del chunk_values, chunk_indices, candidates
        del candidate_indices, selected
    assert logsumexp is not None
    top_token = top_indices[:, 0]
    best_other = torch.where(
        top_token == flat_targets,
        top_values[:, 1],
        top_values[:, 0],
    )
    return StreamingTokenMetrics(
        logsumexp=logsumexp.reshape(original_shape),
        top_token=top_token.reshape(original_shape),
        top_logit=top_values[:, 0].reshape(original_shape),
        top1_margin=(top_values[:, 0] - top_values[:, 1]).reshape(original_shape),
        target_logit=target_logits.reshape(original_shape),
        target_rank=ranks.reshape(original_shape),
        target_margin=(target_logits - best_other).reshape(original_shape),
    )


@torch.no_grad()
def teacher_statistics_from_logsumexp(
    teacher_hidden: Tensor,
    output_weight: Tensor,
    logsumexp: Tensor,
    *,
    vocab_chunk_size: int,
) -> TeacherTerminalStatistics:
    """Compute teacher KL moments while reusing a streamed log-normalizer."""

    original_shape = teacher_hidden.shape[:-1]
    hidden_size = int(teacher_hidden.shape[-1])
    flat_hidden = teacher_hidden.reshape(-1, hidden_size).float()
    flat_lse = logsumexp.reshape(-1).float()
    expected_weight = torch.zeros_like(flat_hidden, dtype=torch.float32)
    expected_log_probability = torch.zeros(
        len(flat_hidden), device=flat_hidden.device, dtype=torch.float32
    )
    for start in range(0, int(output_weight.shape[0]), vocab_chunk_size):
        stop = min(int(output_weight.shape[0]), start + vocab_chunk_size)
        weight = output_weight[start:stop].to(
            device=flat_hidden.device,
            dtype=torch.float32,
        )
        logits = F.linear(flat_hidden, weight)
        log_probabilities = logits - flat_lse.unsqueeze(-1)
        probabilities = log_probabilities.exp()
        expected_weight.add_(probabilities @ weight)
        expected_log_probability.add_(
            (probabilities * log_probabilities).sum(dim=-1)
        )
        del weight, logits, log_probabilities, probabilities
    return TeacherTerminalStatistics(
        logsumexp=logsumexp.float(),
        expected_output_weight=expected_weight.reshape(
            *original_shape, hidden_size
        ),
        expected_log_probability=expected_log_probability.reshape(original_shape),
    )


def _statistics_to(
    statistics: TeacherTerminalStatistics,
    device: torch.device,
) -> TeacherTerminalStatistics:
    return TeacherTerminalStatistics(
        logsumexp=statistics.logsumexp.to(device),
        expected_output_weight=statistics.expected_output_weight.to(device),
        expected_log_probability=statistics.expected_log_probability.to(device),
    )


@torch.inference_mode()
def paired_hidden_metrics(
    dense_model: torch.nn.Module,
    c1_model: torch.nn.Module,
    input_ids: Tensor,
    prediction_positions: Tensor,
    target_ids: Tensor,
    *,
    vocab_chunk_size: int,
) -> dict[str, Tensor]:
    """Run paired teacher-forced models and return exact token-level metrics."""

    dense_device = dense_model.get_input_embeddings().weight.device
    c1_device = c1_model.get_input_embeddings().weight.device
    dense_output = dense_model.model(
        input_ids=input_ids.unsqueeze(0).to(dense_device),
        use_cache=False,
    )
    dense_hidden = dense_output.last_hidden_state[0].index_select(
        0, prediction_positions.to(dense_device)
    )
    dense_metrics = streaming_token_metrics(
        dense_hidden,
        dense_model.lm_head.weight,
        target_ids.to(dense_device),
        vocab_chunk_size=vocab_chunk_size,
    )
    teacher_statistics = teacher_statistics_from_logsumexp(
        dense_hidden,
        dense_model.lm_head.weight,
        dense_metrics.logsumexp,
        vocab_chunk_size=vocab_chunk_size,
    )
    del dense_output, dense_hidden

    c1_output = c1_model.model(
        input_ids=input_ids.unsqueeze(0).to(c1_device),
        use_cache=False,
    )
    c1_hidden = c1_output.last_hidden_state[0].index_select(
        0, prediction_positions.to(c1_device)
    )
    c1_metrics = streaming_token_metrics(
        c1_hidden,
        c1_model.lm_head.weight,
        target_ids.to(c1_device),
        vocab_chunk_size=vocab_chunk_size,
    )
    student_statistics = _statistics_to(teacher_statistics, c1_device)
    kl = terminal_kl_from_statistics(
        student_statistics,
        c1_hidden,
        c1_metrics.logsumexp,
    )
    output = {
        "terminal_kl": kl.cpu(),
        "top1_agreement": (
            dense_metrics.top_token.cpu() == c1_metrics.top_token.cpu()
        ),
        "dense_top_token": dense_metrics.top_token.cpu(),
        "c1_top_token": c1_metrics.top_token.cpu(),
        "dense_target_rank": dense_metrics.target_rank.cpu(),
        "c1_target_rank": c1_metrics.target_rank.cpu(),
        "dense_top1_margin": dense_metrics.top1_margin.cpu(),
        "c1_top1_margin": c1_metrics.top1_margin.cpu(),
        "dense_target_margin": dense_metrics.target_margin.cpu(),
        "c1_target_margin": c1_metrics.target_margin.cpu(),
    }
    del c1_output, c1_hidden, teacher_statistics, student_statistics, kl
    return output


def _new_store() -> dict[str, list[float]]:
    return defaultdict(list)


def _append_metrics(
    store: dict[str, list[float]],
    metrics: Mapping[str, Tensor],
    indices: Tensor | None = None,
) -> None:
    for name, values in metrics.items():
        selected = values if indices is None else values.index_select(0, indices)
        if selected.dtype == torch.bool:
            store[name].extend(map(float, selected.to(torch.float32).tolist()))
        else:
            store[name].extend(map(float, selected.tolist()))


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(torch.tensor(values, dtype=torch.float64).quantile(probability))


def summarize_metrics(store: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Summarize one token subset with distribution tails retained."""

    kl = list(store.get("terminal_kl", ()))
    if not kl:
        return {"tokens": 0}
    output: dict[str, Any] = {
        "tokens": len(kl),
        "terminal_kl": {
            "mean": statistics.fmean(kl),
            "median": statistics.median(kl),
            "p90": _quantile(kl, 0.90),
            "p99": _quantile(kl, 0.99),
            "maximum": max(kl),
        },
        "top1_agreement": statistics.fmean(store["top1_agreement"]),
    }
    for model_name in ("dense", "c1"):
        ranks = list(store[f"{model_name}_target_rank"])
        output[model_name] = {
            "target_top1_rate": sum(rank == 1 for rank in ranks) / len(ranks),
            "target_rank_mean": statistics.fmean(ranks),
            "target_rank_median": statistics.median(ranks),
            "target_rank_p90": _quantile(ranks, 0.90),
            "top1_margin_mean": statistics.fmean(
                store[f"{model_name}_top1_margin"]
            ),
            "target_margin_mean": statistics.fmean(
                store[f"{model_name}_target_margin"]
            ),
        }
    return output


def _indices_in_range(values: Tensor, bounds: tuple[int, int]) -> Tensor:
    return torch.nonzero(
        (values >= bounds[0]) & (values < bounds[1]), as_tuple=False
    ).flatten()


def _summaries_by_ranges(
    store_metrics: Mapping[str, Tensor],
    positions: Tensor,
    ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    output = {}
    for bounds in ranges:
        indices = _indices_in_range(positions, bounds)
        selected = _new_store()
        _append_metrics(selected, store_metrics, indices)
        output[_range_label(bounds)] = summarize_metrics(selected)
    return output


def _load_models(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]] | None:
    valid = all(
        (
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(torch.cuda.device_count() == 2, "exactly two L40S GPUs must be visible"),
            _check(args.vocab_chunk_size > 1, "vocabulary chunk size must exceed one"),
        )
    )
    if not valid:
        return None
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not _check(all(name == "NVIDIA L40S" for name in gpu_names), f"unexpected GPUs: {gpu_names}"):
        return None
    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.c1_checkpoint_dir).expanduser().resolve()
    manifest_path = checkpoint_dir / "manifest.json"
    if not _check(manifest_path.is_file(), f"missing checkpoint manifest: {manifest_path}"):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_valid = all(
        (
            _check(manifest.get("format") == quality.CHECKPOINT_FORMAT, "checkpoint format mismatch"),
            _check(manifest.get("status") == "complete", "checkpoint is incomplete"),
            _check(manifest.get("run_id") == "Q3-8B-C1-R80", "diagnostic requires Q3-8B-C1-R80"),
            _check(manifest.get("compression", {}).get("method") == "c1-two-sided-kl", "diagnostic requires two-sided-KL C1"),
            _check(manifest.get("compression", {}).get("equivalent_rank_target") == 80, "diagnostic requires C1-R80"),
            _check(manifest.get("model", {}).get("config_sha256") == quality._sha256(model_path / "config.json"), "checkpoint model hash mismatch"),
        )
    )
    if not manifest_valid:
        return None
    torch.set_num_threads(args.torch_num_threads)
    for index in range(2):
        torch.cuda.reset_peak_memory_stats(index)
    load_options = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    dense_model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map={"": 0}, **load_options
    ).eval()
    c1_model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map={"": 1}, **load_options
    ).eval()
    dense_model.config.use_cache = False
    c1_model.config.use_cache = False
    if not _check(
        dense_model.lm_head.bias is None and c1_model.lm_head.bias is None,
        "streamed terminal KL requires bias-free output heads",
    ):
        return None
    installation = quality.install_c1_allocation(
        c1_model, checkpoint_dir, manifest
    )
    if installation is None:
        return None
    runtime = {
        "model": manifest["model"],
        "model_dtype": str(torch.bfloat16),
        "attention_implementation": "sdpa",
        "dense_device": 0,
        "c1_device": 1,
        "cuda_device_names": gpu_names,
        "c1_installation": installation,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "manifest_sha256": quality._sha256(manifest_path),
        },
    }
    return dense_model, c1_model, runtime


def _environment(gpu_names: Sequence[str]) -> dict[str, Any]:
    return {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "lm_eval": importlib.metadata.version("lm-eval"),
        "cuda_devices": list(gpu_names),
        "peak_cuda_allocated_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        },
        "torch_num_threads": torch.get_num_threads(),
    }


def _load_c4_windows(
    path: Path,
    *,
    model_path: Path,
) -> tuple[Tensor, dict[str, Any]] | None:
    resolved = path.expanduser().resolve()
    manifest_path = resolved.parent / "manifest.json"
    if not _check(resolved.is_file() and manifest_path.is_file(), "missing C4 window bank or manifest"):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sampling = manifest.get("sampling", {})
    artifact = manifest.get("artifact", {})
    valid = all(
        (
            _check(manifest.get("format") == WINDOWS_FORMAT, "C4 window format mismatch"),
            _check(quality._sha256(resolved) == artifact.get("sha256"), "C4 window hash mismatch"),
            _check(manifest.get("model", {}).get("config_sha256") == quality._sha256(model_path / "config.json"), "C4 window model hash mismatch"),
            _check(sampling.get("sequence_length") == 4096, "C4 diagnostic requires 4096-token windows"),
            _check(sampling.get("samples") == 640, "C4 diagnostic requires the 640-window bank"),
        )
    )
    if not valid:
        return None
    tensors = load_file(str(resolved), device="cpu")
    if not _check(set(tensors) == {"input_ids"}, "C4 bank must contain only input_ids"):
        return None
    input_ids = tensors["input_ids"].to(torch.long)
    if not _check(tuple(input_ids.shape) == (640, 4096), "C4 window tensor shape mismatch"):
        return None
    provenance = {
        "path": str(resolved),
        "sha256": quality._sha256(resolved),
        "manifest": str(manifest_path),
        "manifest_sha256": quality._sha256(manifest_path),
        "dataset": manifest.get("dataset"),
        "sampling": sampling,
    }
    return input_ids.contiguous(), provenance


@torch.inference_mode()
def run_c4_position_kl(args: argparse.Namespace) -> int:
    output_path = Path(args.output_json).expanduser().resolve()
    if not _check(not output_path.exists(), f"output already exists: {output_path}"):
        return 2
    loaded_models = _load_models(args)
    if loaded_models is None:
        return 2
    dense_model, c1_model, runtime = loaded_models
    model_path = Path(args.model).expanduser().resolve()
    loaded_windows = _load_c4_windows(Path(args.windows), model_path=model_path)
    if loaded_windows is None:
        return 2
    all_windows, window_provenance = loaded_windows
    stop_index = args.window_start + args.num_windows
    if not _check(
        args.window_start >= 0
        and args.num_windows > 0
        and stop_index <= len(all_windows),
        "invalid C4 window slice",
    ):
        return 2
    windows = all_windows[args.window_start:stop_index]
    aggregate = _new_store()
    by_fine_range = {bounds: _new_store() for bounds in position_ranges(4096)}
    by_requested_range = {
        bounds: _new_store() for bounds in requested_position_ranges(4096)
    }
    per_window = []
    started = time.perf_counter()
    for local_index, input_ids in enumerate(windows):
        source_index = args.window_start + local_index
        positions = sample_positions_by_fine_bucket(
            sequence_length=4096,
            positions_per_bucket=args.positions_per_256_bucket,
            seed=args.seed,
            window_index=source_index,
        )
        targets = input_ids.index_select(0, positions + 1)
        metrics = paired_hidden_metrics(
            dense_model,
            c1_model,
            input_ids,
            positions,
            targets,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _append_metrics(aggregate, metrics)
        for bounds, store in by_fine_range.items():
            _append_metrics(store, metrics, _indices_in_range(positions, bounds))
        for bounds, store in by_requested_range.items():
            _append_metrics(store, metrics, _indices_in_range(positions, bounds))
        window_store = _new_store()
        _append_metrics(window_store, metrics)
        per_window.append(
            {
                "source_window_index": source_index,
                "sampled_positions": len(positions),
                "mean_terminal_kl": statistics.fmean(
                    window_store["terminal_kl"]
                ),
                "top1_agreement": statistics.fmean(
                    window_store["top1_agreement"]
                ),
            }
        )
        completed = local_index + 1
        if completed % 8 == 0 or completed == len(windows):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(windows) - completed)
            print(
                f"[C4] windows={completed}/{len(windows)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        del metrics

    result = {
        "format": C4_FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": runtime["model"],
        "protocol": {
            "teacher": "Dense Qwen3-8B-Base",
            "student": "Q3-8B-C1-R80 two-sided-KL",
            "teacher_forcing": True,
            "full_vocabulary_exact_kl": True,
            "sequence_length": 4096,
            "source_window_indices": [args.window_start, stop_index - 1],
            "num_windows": len(windows),
            "sampling_seed": args.seed,
            "positions_per_256_bucket_per_window": args.positions_per_256_bucket,
            "sampled_positions_per_window": len(
                sample_positions_by_fine_bucket(
                    sequence_length=4096,
                    positions_per_bucket=args.positions_per_256_bucket,
                    seed=args.seed,
                    window_index=args.window_start,
                )
            ),
            "vocab_chunk_size": args.vocab_chunk_size,
            "fine_position_bins": [
                _range_label(bounds) for bounds in position_ranges(4096)
            ],
            "requested_position_bins": [
                _range_label(bounds)
                for bounds in requested_position_ranges(4096)
            ],
        },
        "windows": window_provenance,
        "metrics": {
            "all_sampled_positions": summarize_metrics(aggregate),
            "by_256_position_bin": {
                _range_label(bounds): summarize_metrics(store)
                for bounds, store in by_fine_range.items()
            },
            "by_requested_position_bin": {
                _range_label(bounds): summarize_metrics(store)
                for bounds, store in by_requested_range.items()
            },
            "per_window": per_window,
        },
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": _environment(runtime["cuda_device_names"]),
    }
    quality._write_json(output_path, result)
    print(f"[Result] wrote {output_path}", flush=True)
    return 0


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _response(sample: Mapping[str, Any]) -> str:
    value = _unwrap_singleton(sample.get("resps"))
    return str(value)


def _load_gsm_strict_samples(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]] | None:
    resolved = path.expanduser().resolve()
    if not _check(resolved.is_file(), f"missing GSM result: {resolved}"):
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    valid = all(
        (
            _check(payload.get("status") == "complete", f"incomplete GSM result: {resolved}"),
            _check(payload.get("task") == "gsm8k", f"not a GSM stage: {resolved}"),
            _check(payload.get("protocol", {}).get("log_samples") is True, f"GSM samples were not logged: {resolved}"),
        )
    )
    if not valid:
        return None
    samples: dict[int, dict[str, Any]] = {}
    for sample in payload.get("evaluation", {}).get("samples", {}).get("gsm8k", ()):
        if sample.get("filter") != STRICT_FILTER:
            continue
        doc_id = int(sample["doc_id"])
        if not _check(doc_id not in samples, f"duplicate strict sample {doc_id} in {resolved}"):
            return None
        samples[doc_id] = sample
    if not _check(len(samples) == 1319, f"expected 1319 strict GSM samples in {resolved}"):
        return None
    provenance = {
        "path": str(resolved),
        "sha256": quality._sha256(resolved),
        "run_id": payload.get("run_id"),
        "checkpoint": payload.get("checkpoint"),
        "protocol": payload.get("protocol"),
    }
    return samples, provenance


def _token_intersects(
    offset: tuple[int, int],
    spans: Sequence[tuple[int, int]],
) -> bool:
    return any(offset[0] < stop and offset[1] > start for start, stop in spans)


def response_event_masks(
    response: str,
    offsets: Sequence[tuple[int, int]],
) -> dict[str, Tensor]:
    """Mark numerical, answer-marker, and gold-answer tokens by char overlap."""

    numeric_spans = [match.span() for match in NUMBER.finditer(response)]
    marker_spans = [match.span() for match in re.finditer(r"####", response)]
    strict_match = STRICT_ANSWER.search(response)
    answer_spans = [] if strict_match is None else [strict_match.span(1)]
    return {
        "numeric_token": torch.tensor(
            [_token_intersects(offset, numeric_spans) for offset in offsets],
            dtype=torch.bool,
        ),
        "answer_marker_token": torch.tensor(
            [_token_intersects(offset, marker_spans) for offset in offsets],
            dtype=torch.bool,
        ),
        "gold_answer_token": torch.tensor(
            [_token_intersects(offset, answer_spans) for offset in offsets],
            dtype=torch.bool,
        ),
    }


def outcome_label(dense_correct: bool, c1_correct: bool) -> str:
    if dense_correct and c1_correct:
        return "both_correct"
    if dense_correct:
        return "dense_correct_c1_wrong"
    if c1_correct:
        return "dense_wrong_c1_correct"
    return "both_wrong"


def classify_divergence(
    *,
    response: str,
    offset: tuple[int, int] | None,
    token_index: int,
    terminated: bool,
) -> tuple[str, str]:
    """Assign a first divergence to planning, calculation, or control."""

    if terminated or offset is None:
        return "control", "termination_length"
    marker = re.search(r"####", response)
    answer = STRICT_ANSWER.search(response)
    numeric_spans = [match.span() for match in NUMBER.finditer(response)]
    if marker is not None and _token_intersects(offset, [marker.span()]):
        return "control", "answer_marker"
    if answer is not None and _token_intersects(offset, [answer.span(1)]):
        return "calculation", "final_answer_number"
    if _token_intersects(offset, numeric_spans):
        return "calculation", "arithmetic_or_numeric"
    if marker is not None and offset[0] >= marker.end():
        return "control", "post_answer"
    if token_index < 20:
        return "planning", "early_reasoning_0_19"
    return "planning", "reasoning_pre_answer"


def analyze_first_divergences(
    dense_samples: Mapping[int, Mapping[str, Any]],
    c1_samples: Mapping[int, Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Analyze the first token-level split in all paired free generations."""

    rows = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    steps: dict[str, list[int]] = defaultdict(list)
    for doc_id in sorted(dense_samples):
        dense_sample = dense_samples[doc_id]
        c1_sample = c1_samples[doc_id]
        dense_response = _response(dense_sample)
        c1_response = _response(c1_sample)
        dense_encoding = tokenizer(
            dense_response,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        c1_encoding = tokenizer(c1_response, add_special_tokens=False)
        dense_ids = list(map(int, dense_encoding["input_ids"]))
        c1_ids = list(map(int, c1_encoding["input_ids"]))
        shared = min(len(dense_ids), len(c1_ids))
        divergence = next(
            (index for index in range(shared) if dense_ids[index] != c1_ids[index]),
            None,
        )
        terminated = divergence is None and len(dense_ids) != len(c1_ids)
        if terminated:
            divergence = shared
        dense_correct = bool(dense_sample.get("exact_match"))
        c1_correct = bool(c1_sample.get("exact_match"))
        outcome = outcome_label(dense_correct, c1_correct)
        counters[outcome]["documents"] += 1
        if divergence is None:
            counters[outcome]["identical_generation"] += 1
            rows.append(
                {
                    "doc_id": doc_id,
                    "outcome": outcome,
                    "first_divergence_step": None,
                    "category": "identical",
                    "phase": "identical",
                }
            )
            continue
        offset = (
            tuple(map(int, dense_encoding["offset_mapping"][divergence]))
            if divergence < len(dense_ids)
            else None
        )
        category, phase = classify_divergence(
            response=dense_response,
            offset=offset,
            token_index=divergence,
            terminated=terminated,
        )
        counters[outcome][f"category:{category}"] += 1
        counters[outcome][f"phase:{phase}"] += 1
        steps[outcome].append(divergence)
        char_start = len(dense_response) if offset is None else offset[0]
        c1_prefix = tokenizer.decode(c1_ids[:divergence])
        rows.append(
            {
                "doc_id": doc_id,
                "outcome": outcome,
                "first_divergence_step": divergence,
                "category": category,
                "phase": phase,
                "dense_token_id": None if divergence >= len(dense_ids) else dense_ids[divergence],
                "c1_token_id": None if divergence >= len(c1_ids) else c1_ids[divergence],
                "dense_token": None if divergence >= len(dense_ids) else tokenizer.decode([dense_ids[divergence]]),
                "c1_token": None if divergence >= len(c1_ids) else tokenizer.decode([c1_ids[divergence]]),
                "dense_context": dense_response[max(0, char_start - 80):char_start + 120],
                "c1_context": c1_response[max(0, len(c1_prefix) - 80):len(c1_prefix) + 120],
            }
        )

    summaries = {}
    for outcome in (
        "both_correct",
        "dense_correct_c1_wrong",
        "dense_wrong_c1_correct",
        "both_wrong",
    ):
        counts = counters[outcome]
        values = steps[outcome]
        summaries[outcome] = {
            "documents": counts["documents"],
            "identical_generations": counts["identical_generation"],
            "first_divergence_step": {
                "count": len(values),
                "mean": None if not values else statistics.fmean(values),
                "median": None if not values else statistics.median(values),
                "p90": None if not values else _quantile(values, 0.90),
            },
            "category_counts": {
                category: counts[f"category:{category}"]
                for category in ("planning", "calculation", "control")
            },
            "phase_counts": {
                phase.removeprefix("phase:"): value
                for phase, value in counts.items()
                if phase.startswith("phase:")
            },
        }
    return {"by_outcome": summaries, "documents": rows}


def select_replay_documents(
    dense_samples: Mapping[int, Mapping[str, Any]],
    c1_samples: Mapping[int, Mapping[str, Any]],
    *,
    maximum: int,
    seed: int,
) -> list[int]:
    """Select all Dense-only failures first, then an equal-size control set."""

    groups: dict[str, list[int]] = defaultdict(list)
    for doc_id, sample in dense_samples.items():
        if bool(sample.get("exact_match")):
            label = outcome_label(True, bool(c1_samples[doc_id].get("exact_match")))
            groups[label].append(doc_id)
    selected = sorted(groups["dense_correct_c1_wrong"] + groups["both_correct"])
    if maximum <= 0 or maximum >= len(selected):
        return selected
    generator = random.Random(seed)
    dense_only = sorted(groups["dense_correct_c1_wrong"])
    controls = sorted(groups["both_correct"])
    dense_only_quota = min(len(dense_only), (maximum + 1) // 2)
    control_quota = min(len(controls), maximum - dense_only_quota)
    chosen = generator.sample(dense_only, dense_only_quota)
    chosen.extend(generator.sample(controls, control_quota))
    if len(chosen) < maximum:
        remaining = sorted(set(selected) - set(chosen))
        chosen.extend(generator.sample(remaining, maximum - len(chosen)))
    return sorted(chosen)


@torch.inference_mode()
def run_gsm_dense_prefix(args: argparse.Namespace) -> int:
    output_path = Path(args.output_json).expanduser().resolve()
    if not _check(not output_path.exists(), f"output already exists: {output_path}"):
        return 2
    dense_loaded = _load_gsm_strict_samples(Path(args.dense_gsm_json))
    c1_loaded = _load_gsm_strict_samples(Path(args.c1_gsm_json))
    if dense_loaded is None or c1_loaded is None:
        return 2
    dense_samples, dense_provenance = dense_loaded
    c1_samples, c1_provenance = c1_loaded
    if not _check(set(dense_samples) == set(c1_samples), "Dense/C1 GSM document IDs differ"):
        return 2
    if not _check(dense_provenance["run_id"] == "Q3-8B-Dense", "Dense GSM run ID mismatch"):
        return 2
    if not _check(c1_provenance["run_id"] == "Q3-8B-C1-R80", "C1 GSM run ID mismatch"):
        return 2
    model_path = Path(args.model).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    if not _check(tokenizer.is_fast, "GSM event masks require a fast tokenizer"):
        return 2
    divergences = analyze_first_divergences(dense_samples, c1_samples, tokenizer)
    replay_doc_ids = select_replay_documents(
        dense_samples,
        c1_samples,
        maximum=args.max_replay_documents,
        seed=args.seed,
    )
    loaded_models = _load_models(args)
    if loaded_models is None:
        return 2
    dense_model, c1_model, runtime = loaded_models

    aggregate = _new_store()
    by_outcome: dict[str, dict[str, list[float]]] = defaultdict(_new_store)
    by_decode_range = {bounds: _new_store() for bounds in DECODE_STEP_RANGES}
    by_position_range = {
        bounds: _new_store() for bounds in requested_position_ranges(args.max_length)
    }
    by_event: dict[str, dict[str, list[float]]] = {
        name: _new_store()
        for name in (
            "numeric_token",
            "answer_marker_token",
            "gold_answer_token",
            "terminal_eos_probe",
        )
    }
    per_document = []
    excluded = []
    started = time.perf_counter()
    for replay_index, doc_id in enumerate(replay_doc_ids):
        dense_sample = dense_samples[doc_id]
        c1_sample = c1_samples[doc_id]
        prompt = str(dense_sample["arguments"][0][0])
        response = _response(dense_sample)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_encoding = tokenizer(
            response,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        response_ids = list(map(int, response_encoding["input_ids"]))
        offsets = [tuple(map(int, pair)) for pair in response_encoding["offset_mapping"]]
        total_length = len(prompt_ids) + len(response_ids)
        if not prompt_ids or not response_ids or total_length > args.max_length:
            excluded.append(
                {
                    "doc_id": doc_id,
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(response_ids),
                    "reason": "empty token sequence or maximum length exceeded",
                }
            )
            continue
        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        response_positions = torch.arange(
            len(prompt_ids) - 1,
            total_length - 1,
            dtype=torch.long,
        )
        prediction_positions = torch.cat(
            (response_positions, torch.tensor([total_length - 1]))
        )
        targets = torch.tensor(
            response_ids + [int(tokenizer.eos_token_id)], dtype=torch.long
        )
        metrics = paired_hidden_metrics(
            dense_model,
            c1_model,
            input_ids,
            prediction_positions,
            targets,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        response_metrics = {name: values[:-1] for name, values in metrics.items()}
        terminal_metrics = {name: values[-1:] for name, values in metrics.items()}
        _append_metrics(aggregate, response_metrics)
        outcome = outcome_label(
            True,
            bool(c1_sample.get("exact_match")),
        )
        _append_metrics(by_outcome[outcome], response_metrics)
        decode_steps = torch.arange(len(response_ids), dtype=torch.long)
        for bounds, store in by_decode_range.items():
            _append_metrics(
                store,
                response_metrics,
                _indices_in_range(decode_steps, bounds),
            )
        for bounds, store in by_position_range.items():
            _append_metrics(
                store,
                response_metrics,
                _indices_in_range(response_positions, bounds),
            )
        masks = response_event_masks(response, offsets)
        for event_name, mask in masks.items():
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            _append_metrics(by_event[event_name], response_metrics, indices)
        _append_metrics(by_event["terminal_eos_probe"], terminal_metrics)

        document_store = _new_store()
        _append_metrics(document_store, response_metrics)
        per_document.append(
            {
                "doc_id": doc_id,
                "outcome": outcome,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
                "response_position_start": int(response_positions[0]),
                "response_position_stop": int(response_positions[-1]),
                "mean_terminal_kl": statistics.fmean(
                    document_store["terminal_kl"]
                ),
                "top1_agreement": statistics.fmean(
                    document_store["top1_agreement"]
                ),
                "dense_replay_target_top1_rate": sum(
                    rank == 1 for rank in document_store["dense_target_rank"]
                )
                / len(response_ids),
                "c1_dense_token_top1_rate": sum(
                    rank == 1 for rank in document_store["c1_target_rank"]
                )
                / len(response_ids),
                "terminal_eos_probe": summarize_metrics(
                    {name: list(map(float, value.tolist())) for name, value in terminal_metrics.items()}
                ),
            }
        )
        completed = replay_index + 1
        if completed % 25 == 0 or completed == len(replay_doc_ids):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(replay_doc_ids) - completed)
            print(
                f"[GSM replay] documents={completed}/{len(replay_doc_ids)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        del metrics, response_metrics, terminal_metrics

    completed_documents = len(per_document)
    if not _check(completed_documents > 0, "no GSM replay documents completed"):
        return 2
    result = {
        "format": GSM_FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": runtime["model"],
        "protocol": {
            "teacher": "Dense Qwen3-8B-Base",
            "student": "Q3-8B-C1-R80 two-sided-KL",
            "prefix": "dense-generated GSM8K continuation",
            "selection": "Dense strict-match correct",
            "bounded_selection": "Dense-correct/C1-wrong failures first, then an equal-size both-correct control",
            "teacher_forcing": True,
            "full_vocabulary_exact_kl": True,
            "strict_correctness_is_primary": True,
            "requested_replay_documents": args.max_replay_documents,
            "selected_replay_documents": len(replay_doc_ids),
            "completed_replay_documents": completed_documents,
            "selection_seed": args.seed,
            "max_length": args.max_length,
            "vocab_chunk_size": args.vocab_chunk_size,
            "decode_step_bins": [
                _range_label(bounds) for bounds in DECODE_STEP_RANGES
            ],
            "absolute_position_bins": [
                _range_label(bounds)
                for bounds in requested_position_ranges(args.max_length)
            ],
            "terminal_eos_is_a_probe": True,
        },
        "inputs": {
            "dense_gsm": dense_provenance,
            "c1_gsm": c1_provenance,
        },
        "first_divergence": divergences,
        "metrics": {
            "all_dense_correct_response_tokens": summarize_metrics(aggregate),
            "by_final_outcome": {
                label: summarize_metrics(store)
                for label, store in by_outcome.items()
            },
            "by_decode_step": {
                _range_label(bounds): summarize_metrics(store)
                for bounds, store in by_decode_range.items()
            },
            "by_absolute_position": {
                _range_label(bounds): summarize_metrics(store)
                for bounds, store in by_position_range.items()
            },
            "by_event": {
                name: summarize_metrics(store)
                for name, store in by_event.items()
            },
            "per_document": per_document,
            "excluded_documents": excluded,
        },
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": _environment(runtime["cuda_device_names"]),
    }
    quality._write_json(output_path, result)
    print(f"[Result] wrote {output_path}", flush=True)
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--c1-checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vocab-chunk-size", type=int, default=1024)
    parser.add_argument("--torch-num-threads", type=int, default=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="diagnostic", required=True)
    c4 = subparsers.add_parser("c4-position-kl")
    _add_common_arguments(c4)
    c4.add_argument("--windows", required=True)
    c4.add_argument("--window-start", type=int, default=512)
    c4.add_argument("--num-windows", type=int, default=128)
    c4.add_argument("--positions-per-256-bucket", type=int, default=32)
    c4.add_argument("--seed", type=int, default=20260903)

    gsm = subparsers.add_parser("gsm-dense-prefix")
    _add_common_arguments(gsm)
    gsm.add_argument("--dense-gsm-json", required=True)
    gsm.add_argument("--c1-gsm-json", required=True)
    gsm.add_argument("--max-replay-documents", type=int, default=0)
    gsm.add_argument("--max-length", type=int, default=4096)
    gsm.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    if args.diagnostic == "c4-position-kl":
        return run_c4_position_kl(args)
    return run_gsm_dense_prefix(args)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
