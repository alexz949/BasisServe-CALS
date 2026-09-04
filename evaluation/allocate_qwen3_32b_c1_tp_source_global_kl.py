#!/usr/bin/env python3
"""Allocate Qwen3-32B C1 ranks over eight physical TP Value sources.

The uniform rank-96 ALS checkpoint is the anchor.  For every layer and physical
KV head, the script substitutes ranks 64/80/112/128 one source at a time,
closes the complete layer decoder against the 256-document fit covariance, and
measures paired dense-teacher terminal KL on eight fresh C4 documents.  Exact
dynamic programming preserves the 64 x 8 x 96 physical-rank budget.  Mean and
one-standard-error schedules are confirmed on eight additional fresh documents
before WikiText-2 is touched.

Quality evaluation zero-pads every ragged source back into Qwen's native
128-dimensional Value slots.  This preserves the ragged function exactly while
retaining the standard sharded Hugging Face SDPA implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import (  # noqa: E402
    close_ragged_decoder_with_fixed_encoders,
    tensor_sha256,
)
from basisserve.core.global_rank_sensitivity import (  # noqa: E402
    logits_logsumexp,
    teacher_kl_sum,
)
from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    covariance_with_trace_damping,
    head_products,
    quadratic_from_target,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_allocation.v1"
FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
MODEL_LABEL = "Qwen3-32B"
MODEL_TYPE = "qwen3"
ATTENTION_TYPE = "gqa"
SNAPSHOT_FORMAT = "basisserve.attention_o_proj_covariances.v1"
WINDOWS_FORMAT = "basisserve.calibration.c4_document_windows.v1"
NUM_LAYERS = 64
NUM_QUERY_HEADS = 64
NUM_KV_HEADS = 8
HEADS_PER_SOURCE = 8
HEAD_DIM = 128
HIDDEN_SIZE = 5120
QUERY_WIDTH = NUM_QUERY_HEADS * HEAD_DIM
FRESH_WINDOW_START = 320
RANK_DEPENDENT_CONFIG_KEYS = {
    "cache_rank_per_head",
    "total_v_cache_rank",
    "total_kv_retained_ratio_with_dense_k",
    "v_retained_ratio",
}


def activate_model_profile(name: str) -> None:
    """Select an audited Qwen3 GQA geometry for shared C1 tooling."""

    global FORMAT, FACTOR_FORMAT, MODEL_LABEL, MODEL_TYPE, ATTENTION_TYPE
    global NUM_LAYERS, NUM_QUERY_HEADS, NUM_KV_HEADS, HEADS_PER_SOURCE
    global HEAD_DIM, HIDDEN_SIZE, QUERY_WIDTH
    if name == "qwen3_32b":
        FORMAT = "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_allocation.v1"
        FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
        MODEL_LABEL = "Qwen3-32B"
        MODEL_TYPE = "qwen3"
        ATTENTION_TYPE = "gqa"
        NUM_LAYERS = 64
        NUM_QUERY_HEADS = 64
        NUM_KV_HEADS = 8
        HEAD_DIM = 128
        HIDDEN_SIZE = 5120
    elif name == "qwen3_8b":
        FORMAT = "basisserve.qwen3_8b.gqa_c1.tp_source_global_kl_allocation.v1"
        FACTOR_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
        MODEL_LABEL = "Qwen3-8B-Base"
        MODEL_TYPE = "qwen3"
        ATTENTION_TYPE = "gqa"
        NUM_LAYERS = 36
        NUM_QUERY_HEADS = 32
        NUM_KV_HEADS = 8
        HEAD_DIM = 128
        HIDDEN_SIZE = 4096
    elif name == "llama31_8b":
        FORMAT = "basisserve.llama31_8b.gqa_c1.tp_source_global_kl_allocation.v1"
        FACTOR_FORMAT = "basisserve.llama31_8b.gqa_c1_joint.v1"
        MODEL_LABEL = "Llama-3.1-8B"
        MODEL_TYPE = "llama"
        ATTENTION_TYPE = "gqa"
        NUM_LAYERS = 32
        NUM_QUERY_HEADS = 32
        NUM_KV_HEADS = 8
        HEAD_DIM = 128
        HIDDEN_SIZE = 4096
    elif name == "llama31_70b":
        FORMAT = "basisserve.llama31_70b.gqa_c1.tp_source_global_kl_allocation.v1"
        FACTOR_FORMAT = "basisserve.llama31_70b.gqa_c1_joint.v1"
        MODEL_LABEL = "Llama-3.1-70B"
        MODEL_TYPE = "llama"
        ATTENTION_TYPE = "gqa"
        NUM_LAYERS = 80
        NUM_QUERY_HEADS = 64
        NUM_KV_HEADS = 8
        HEAD_DIM = 128
        HIDDEN_SIZE = 8192
    elif name == "llama2_7b":
        FORMAT = "basisserve.llama2_7b.mha_c1.tp_source_global_kl_allocation.v1"
        FACTOR_FORMAT = "basisserve.llama2_7b.mha_c1_v96_joint.v1"
        MODEL_LABEL = "Llama-2-7B"
        MODEL_TYPE = "llama"
        ATTENTION_TYPE = "mha"
        NUM_LAYERS = 32
        NUM_QUERY_HEADS = 32
        NUM_KV_HEADS = 32
        HEAD_DIM = 128
        HIDDEN_SIZE = 4096
    else:
        raise ValueError(f"unknown Qwen3 C1 model profile: {name}")
    HEADS_PER_SOURCE = NUM_QUERY_HEADS // NUM_KV_HEADS
    QUERY_WIDTH = NUM_QUERY_HEADS * HEAD_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-dir",
        action="append",
        required=True,
        metavar="RANK=PATH",
        help="Repeat for ranks 64, 80, 96, and 112; rank 128 is exact/dense.",
    )
    parser.add_argument("--anchor-rank", type=int, default=96)
    parser.add_argument("--candidate-ranks", default="64,80,96,112,128")
    parser.add_argument("--window-start", type=int, default=FRESH_WINDOW_START)
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--decoder-relative-jitter", type=float, default=0.0)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int)
    parser.add_argument("--eval-max-tokens", type=int)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[Qwen3 Global-KL] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen-style decoder model")
    return layers


def _parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if (
        not ranks
        or ranks != tuple(sorted(set(ranks)))
        or any(rank <= 0 or rank > HEAD_DIM or rank % 16 for rank in ranks)
    ):
        raise ValueError("candidate ranks must be distinct increasing multiples of 16")
    return ranks


def _parse_factor_dirs(specs: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in specs:
        rank_text, separator, path_text = raw.partition("=")
        if not separator or not rank_text.strip() or not path_text.strip():
            raise ValueError(f"factor directory must use RANK=PATH: {raw!r}")
        rank = int(rank_text)
        if rank <= 0 or rank in result:
            raise ValueError(f"invalid or duplicate factor rank: {rank}")
        result[rank] = Path(path_text).expanduser().resolve()
    return dict(sorted(result.items()))


def _rank_invariant_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RANK_DEPENDENT_CONFIG_KEYS
    }


def _load_factor_results(
    factor_dirs: Mapping[int, Path],
    *,
    model_config_sha256: str,
    snapshot_dir: Path,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for rank, directory in factor_dirs.items():
        result_path = directory / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("format") != FACTOR_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"incomplete or incompatible factor bank: {result_path}")
        if tuple(map(int, payload.get("layers", ()))) != tuple(range(NUM_LAYERS)):
            raise ValueError(f"factor bank does not cover all layers: {result_path}")
        config = payload.get("fit_config", {})
        if int(config.get("cache_rank_per_head", -1)) != rank:
            raise ValueError(f"factor rank disagrees with directory spec: {result_path}")
        if config.get("model_config_sha256") != model_config_sha256:
            raise ValueError(f"factor bank belongs to another model: {result_path}")
        if Path(str(config.get("snapshot_dir"))).expanduser().resolve() != snapshot_dir:
            raise ValueError(f"factor bank uses another covariance snapshot: {result_path}")
        invariant = _rank_invariant_fit_config(config)
        if reference is None:
            reference = invariant
        elif invariant != reference:
            raise ValueError("candidate factor banks differ beyond rank-dependent fields")
        results[rank] = payload
    return results


def _select_fresh_windows(
    path: Path,
    *,
    window_start: int,
    profile_windows: int,
    confirmation_windows: int,
    sequence_length: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    manifest_path = resolved.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WINDOWS_FORMAT:
        raise ValueError("incompatible C4 window manifest")
    if _sha256(resolved) != manifest["artifact"]["sha256"]:
        raise ValueError("C4 window artifact hash mismatch")
    payload = load_file(str(resolved), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("ordered C4 window bank must contain only input_ids")
    input_ids = payload["input_ids"].to(torch.long)
    needed = profile_windows + confirmation_windows
    stop = window_start + needed
    if window_start < 0 or needed <= 0 or stop > len(input_ids):
        raise ValueError("fresh Global-KL window range exceeds the C4 bank")
    if sequence_length <= 1 or sequence_length > input_ids.shape[1]:
        raise ValueError("invalid Global-KL sequence length")
    records = manifest.get("records", ())
    if len(records) != len(input_ids):
        raise ValueError("C4 manifest records do not match the window tensor")
    document_ids = [str(row["document_id"]) for row in records]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("C4 bank is not document-disjoint")
    selected = input_ids[window_start:stop, :sequence_length].contiguous()
    provenance = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "window_start": window_start,
        "window_stop_exclusive": stop,
        "profile_indices": list(range(window_start, window_start + profile_windows)),
        "confirmation_indices": list(range(window_start + profile_windows, stop)),
        "document_ids": document_ids[window_start:stop],
        "disjoint_from_als_fit_and_heldout_prefix": window_start >= FRESH_WINDOW_START,
        "stored_sequence_length": int(input_ids.shape[1]),
        "used_sequence_length": sequence_length,
    }
    return selected[:profile_windows], selected[profile_windows:], provenance


@dataclass(frozen=True)
class TeacherBatch:
    input_ids: Tensor
    logits: Tensor
    logsumexp: Tensor


def _input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _copy_batched_logits_to_cpu(logits: Tensor) -> Tensor:
    cpu_logits = torch.empty(logits.shape, dtype=logits.dtype, device="cpu")
    for index in range(logits.shape[0]):
        cpu_logits[index].copy_(logits[index])
    return cpu_logits


@torch.inference_mode()
def _capture_teacher(
    model: nn.Module,
    sequences: Tensor,
    *,
    batch_size: int,
    vocab_chunk_size: int,
    label: str,
) -> list[TeacherBatch]:
    input_device = _input_device(model)
    result = []
    for start in range(0, len(sequences), batch_size):
        input_ids = sequences[start : start + batch_size].to(input_device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].detach()
        teacher_logsumexp = logits_logsumexp(
            logits, vocab_chunk_size=vocab_chunk_size
        ).cpu()
        teacher_logits = _copy_batched_logits_to_cpu(logits)
        result.append(
            TeacherBatch(
                input_ids=input_ids.cpu(),
                logits=teacher_logits,
                logsumexp=teacher_logsumexp,
            )
        )
        _log(
            f"dense teacher {label}: {min(start + batch_size, len(sequences))}/"
            f"{len(sequences)} batch_size={batch_size}"
        )
        del input_ids, logits
    return result


def _paired(values: Sequence[float]) -> dict[str, Any]:
    checked = [float(value) for value in values]
    if not checked or not all(math.isfinite(value) for value in checked):
        raise ValueError("paired metric values must be finite and nonempty")
    mean = statistics.fmean(checked)
    standard_error = (
        statistics.stdev(checked) / math.sqrt(len(checked))
        if len(checked) > 1
        else 0.0
    )
    return {
        "values": checked,
        "mean": mean,
        "paired_standard_error": standard_error,
        "one_standard_error_ucb": mean + standard_error,
    }


@torch.inference_mode()
def _evaluate_teacher_metrics(
    model: nn.Module,
    teacher: Sequence[TeacherBatch],
    *,
    vocab_chunk_size: int,
) -> dict[str, Any]:
    input_device = _input_device(model)
    kl_values: list[float] = []
    nll_values: list[float] = []
    for batch in teacher:
        input_ids = batch.input_ids.to(input_device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
        for index in range(input_ids.shape[0]):
            kl_sum, tokens = teacher_kl_sum(
                logits[index : index + 1],
                batch.logits[index : index + 1],
                teacher_logsumexp=batch.logsumexp[index : index + 1],
                vocab_chunk_size=vocab_chunk_size,
            )
            labels = input_ids[index, 1:].to(logits.device)
            nll = F.cross_entropy(
                logits[index].float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="mean",
            )
            kl_values.append(kl_sum / tokens)
            nll_values.append(float(nll.item()))
        del input_ids, logits
    return {"terminal_kl": _paired(kl_values), "nll": _paired(nll_values)}


def _paired_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    result = {}
    for metric in ("terminal_kl", "nll"):
        left = candidate[metric]["values"]
        right = baseline[metric]["values"]
        if len(left) != len(right):
            raise ValueError("candidate and anchor windows are not paired")
        result[metric] = _paired(
            [a - b for a, b in zip(left, right, strict=True)]
        )
    return result


@dataclass(frozen=True)
class SnapshotLayer:
    fit_covariance: Tensor
    dense_o_weight: Tensor
    file: Path
    sha256: str


@dataclass(frozen=True)
class RankFactors:
    A: Tensor
    D: Tensor
    file: Path
    sha256: str


@dataclass(frozen=True)
class CachedFactors:
    A: Tensor
    D: Tensor
    ranks: tuple[int, ...]


def _load_snapshot_cache(
    snapshot_dir: Path,
    *,
    model_path: Path,
) -> tuple[tuple[SnapshotLayer, ...], dict[str, Any]]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("incompatible covariance snapshot")
    geometry = manifest.get("model", {})
    observed = (
        geometry.get("model_type"),
        geometry.get("attention_type"),
        int(geometry.get("num_hidden_layers", -1)),
        int(geometry.get("num_attention_heads", -1)),
        int(geometry.get("num_key_value_heads", -1)),
        int(geometry.get("head_dim", -1)),
        int(geometry.get("hidden_size", -1)),
    )
    expected = (
        MODEL_TYPE,
        ATTENTION_TYPE,
        NUM_LAYERS,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        HIDDEN_SIZE,
    )
    if observed != expected:
        raise ValueError(f"unexpected covariance geometry: {observed}")
    if geometry.get("config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("covariance snapshot belongs to another model config")
    if tuple(map(int, manifest.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError("covariance snapshot does not cover all decoder layers")
    cache = []
    for layer in range(NUM_LAYERS):
        record = manifest["artifacts"][str(layer)]
        path = snapshot_dir / record["file"]
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"covariance artifact hash mismatch at layer {layer}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            fit = handle.get_tensor("fit_covariance").contiguous()
            weight = handle.get_tensor("weight").contiguous()
        if tuple(fit.shape) != (QUERY_WIDTH, QUERY_WIDTH):
            raise ValueError(f"unexpected fit covariance at layer {layer}")
        if tuple(weight.shape) != (HIDDEN_SIZE, QUERY_WIDTH):
            raise ValueError(f"unexpected dense O weight at layer {layer}")
        cache.append(SnapshotLayer(fit, weight, path, digest))
        _log(f"loaded fit covariance {layer + 1}/{NUM_LAYERS}")
    return tuple(cache), manifest


def _load_factor_cache(
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, RankFactors], ...]:
    cache: list[dict[int, RankFactors]] = []
    for layer in range(NUM_LAYERS):
        by_rank = {}
        for rank, directory in factor_dirs.items():
            record = factor_results[rank]["artifacts"][str(layer)]
            path = directory / record["file"]
            digest = _sha256(path)
            if digest != record["sha256"]:
                raise ValueError(f"factor artifact hash mismatch layer={layer} rank={rank}")
            payload = load_file(str(path), device="cpu")
            if set(payload) != {
                "value_coordinate_encoders",
                "head_output_decoders",
            }:
                raise ValueError(f"unexpected factor tensors in {path}")
            A = payload["value_coordinate_encoders"].contiguous()
            D = payload["head_output_decoders"].contiguous()
            if tuple(A.shape) != (NUM_KV_HEADS, HEAD_DIM, rank):
                raise ValueError(f"unexpected encoder shape in {path}")
            if tuple(D.shape) != (NUM_QUERY_HEADS, rank, HIDDEN_SIZE):
                raise ValueError(f"unexpected decoder shape in {path}")
            by_rank[rank] = RankFactors(A, D, path, digest)
        cache.append(by_rank)
        _log(f"loaded factor banks {layer + 1}/{NUM_LAYERS}")
    return tuple(cache)


def _covariance_matrix_to_blocks(
    covariance: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    if tuple(covariance.shape) != (QUERY_WIDTH, QUERY_WIDTH):
        raise ValueError("covariance matrix has incompatible Qwen attention width")
    work = covariance.to(device=device, dtype=torch.float32)
    blocks = (
        work.reshape(NUM_QUERY_HEADS, HEAD_DIM, NUM_QUERY_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return 0.5 * (blocks + blocks.permute(1, 0, 3, 2))


def _objective(
    snapshot: SnapshotLayer,
    *,
    layer: int,
    device: torch.device,
    covariance_damping: float,
) -> tuple[Any, float]:
    covariance = _covariance_matrix_to_blocks(
        snapshot.fit_covariance,
        device=device,
    )
    covariance, absolute_damping = covariance_with_trace_damping(
        covariance,
        relative_damping=covariance_damping,
    )
    target = (
        snapshot.dense_o_weight.to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .reshape(NUM_QUERY_HEADS, HEAD_DIM, HIDDEN_SIZE)
        .contiguous()
    )
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name=f"qwen3_32b_tp_source_global_kl_layer_{layer:03d}",
        trace_normalize=False,
    )
    return objective, absolute_damping


def _assemble_layer_factors(
    bank: Mapping[int, RankFactors],
    *,
    source_ranks: Sequence[int],
    dense_o_weight: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    ranks = tuple(int(rank) for rank in source_ranks)
    if len(ranks) != NUM_KV_HEADS or any(not 0 < rank <= HEAD_DIM for rank in ranks):
        raise ValueError("source ranks do not match Qwen's eight physical KV heads")
    maximum_rank = max(ranks)
    A = torch.zeros(
        NUM_KV_HEADS,
        HEAD_DIM,
        maximum_rank,
        device=device,
        dtype=torch.float32,
    )
    D = torch.zeros(
        NUM_QUERY_HEADS,
        maximum_rank,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.float32,
    )
    dense_D = (
        dense_o_weight.to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .reshape(NUM_QUERY_HEADS, HEAD_DIM, HIDDEN_SIZE)
    )
    for source, rank in enumerate(ranks):
        heads = slice(
            source * HEADS_PER_SOURCE,
            (source + 1) * HEADS_PER_SOURCE,
        )
        if rank == HEAD_DIM:
            A[source, :, :rank].copy_(
                torch.eye(HEAD_DIM, device=device, dtype=torch.float32)
            )
            D[heads, :rank].copy_(dense_D[heads])
            continue
        if rank not in bank:
            raise KeyError(f"rank {rank} is absent from the fitted factor bank")
        A[source, :, :rank].copy_(
            bank[rank].A[source].to(device=device, dtype=torch.float32)
        )
        D[heads, :rank].copy_(
            bank[rank].D[heads].to(device=device, dtype=torch.float32)
        )
    return A, D


@torch.no_grad()
def _fold_ragged_to_padded_weights(
    *,
    dense_v_weight: Tensor,
    A: Tensor,
    D: Tensor,
    source_ranks: Sequence[int],
) -> tuple[Tensor, Tensor]:
    ranks = tuple(int(rank) for rank in source_ranks)
    maximum_rank = max(ranks)
    if A.ndim != 3 or D.ndim != 3 or dense_v_weight.ndim != 2:
        raise ValueError("ragged folding expects rank-two weights and rank-three factors")
    num_sources, head_dim, padded_rank = map(int, A.shape)
    num_heads, decoder_rank, hidden_size = map(int, D.shape)
    if len(ranks) != num_sources or num_heads % num_sources:
        raise ValueError("ragged ranks do not match the routed GQA geometry")
    heads_per_source = num_heads // num_sources
    if tuple(dense_v_weight.shape) != (num_sources * head_dim, hidden_size):
        raise ValueError("dense V weight has incompatible routed GQA geometry")
    if padded_rank != maximum_rank:
        raise ValueError("ragged encoder tensor has incompatible shape")
    if decoder_rank != maximum_rank:
        raise ValueError("ragged decoder tensor has incompatible shape")
    device = dense_v_weight.device
    dense_v = dense_v_weight.to(device=device, dtype=torch.float32)
    work_A = A.to(device=device, dtype=torch.float32)
    work_D = D.to(device=device, dtype=torch.float32)
    padded_v = torch.zeros_like(dense_v)
    padded_o = torch.zeros(
        hidden_size,
        num_heads * head_dim,
        device=device,
        dtype=torch.float32,
    )
    for source, rank in enumerate(ranks):
        dense_rows = slice(source * head_dim, (source + 1) * head_dim)
        target_rows = slice(source * head_dim, source * head_dim + rank)
        padded_v[target_rows].copy_(
            work_A[source, :, :rank].transpose(0, 1) @ dense_v[dense_rows]
        )
        for head in range(
            source * heads_per_source,
            (source + 1) * heads_per_source,
        ):
            target_columns = slice(head * head_dim, head * head_dim + rank)
            padded_o[:, target_columns].copy_(work_D[head, :rank].transpose(0, 1))
    return padded_v, padded_o


@torch.no_grad()
def _install_factors(
    layer: nn.Module,
    *,
    dense_v_weight: Tensor,
    factors: CachedFactors,
) -> None:
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    if (
        not isinstance(v_proj, nn.Linear)
        or not isinstance(o_proj, nn.Linear)
        or v_proj.bias is not None
        or o_proj.bias is not None
    ):
        raise TypeError("Global-KL requires bias-free dense Qwen V/O projections")
    padded_v, padded_o = _fold_ragged_to_padded_weights(
        dense_v_weight=dense_v_weight.to(v_proj.weight.device),
        A=factors.A,
        D=factors.D,
        source_ranks=factors.ranks,
    )
    v_proj.weight.copy_(padded_v.to(dtype=v_proj.weight.dtype))
    o_proj.weight.copy_(padded_o.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
    del padded_v, padded_o


def _quantized_cache(A: Tensor, D: Tensor, ranks: Sequence[int]) -> CachedFactors:
    return CachedFactors(
        A=A.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        D=D.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        ranks=tuple(map(int, ranks)),
    )


def _closed_factors(
    *,
    bank: Mapping[int, RankFactors],
    snapshot: SnapshotLayer,
    objective: Any,
    source_ranks: Sequence[int],
    anchor_rank: int,
    decoder_relative_jitter: float,
    device: torch.device,
) -> tuple[CachedFactors, dict[str, Any]]:
    ranks = tuple(map(int, source_ranks))
    initial_A, initial_D = _assemble_layer_factors(
        bank,
        source_ranks=ranks,
        dense_o_weight=snapshot.dense_o_weight,
        device=device,
    )
    if all(rank == HEAD_DIM for rank in ranks):
        cached = _quantized_cache(initial_A, initial_D, ranks)
        return cached, {"closure": "exact_identity_dense_o_endpoint"}
    if all(rank == anchor_rank for rank in ranks):
        cached = _quantized_cache(initial_A, initial_D, ranks)
        return cached, {"closure": "uniform_anchor_factor_bank"}
    mapping = torch.arange(NUM_QUERY_HEADS, device=device, dtype=torch.long) // (
        NUM_QUERY_HEADS // NUM_KV_HEADS
    )
    closure = close_ragged_decoder_with_fixed_encoders(
        objective=objective,
        initial_A=initial_A,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        group_ranks=ranks,
        relative_jitter=decoder_relative_jitter,
    )
    cached = _quantized_cache(closure.A_unique, closure.D_heads, ranks)
    deployed_A = cached.A.to(device=device, dtype=torch.float32)
    deployed_D = cached.D.to(device=device, dtype=torch.float32)
    deployed_product = head_products(deployed_A, deployed_D, mapping)
    target = (
        snapshot.dense_o_weight.to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .reshape(NUM_QUERY_HEADS, HEAD_DIM, HIDDEN_SIZE)
    )
    residual = target - deployed_product
    deployed_loss_unclamped = float(
        torch.einsum(
            "hio,hkij,kjo->",
            residual,
            objective.covariance,
            residual,
        )
    )
    deployed_loss = max(0.0, deployed_loss_unclamped)
    diagnostics = {
        "closure": "full_layer_closed_form_decoder_refit",
        "deployed_bfloat16_fit_relative_mse": float(
            deployed_loss / objective.constant
        ),
        "deployed_bfloat16_fit_quadratic_unclamped": deployed_loss_unclamped,
        "deployed_bfloat16_fit_evaluation": "direct residual quadratic R^T C R",
        "absolute_jitter": closure.decoder.absolute_jitters[0],
        "condition_estimate": closure.decoder.condition_estimates[0],
        "matrix_dimension": closure.decoder.matrix_dimensions[0],
        "relative_residual": closure.decoder.relative_residuals[0],
        "solve_wall_time_seconds": closure.decoder.wall_times_seconds[0],
        "encoder_sha256": closure.encoder_sha256_after_solve,
    }
    del deployed_A, deployed_D, deployed_product, target, residual
    return cached, diagnostics


def _allocate(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    cost_key: str,
) -> tuple[list[list[int]], float, list[dict[str, Any]]]:
    indexed = {
        (int(row["layer"]), int(row["source"]), int(row["candidate_rank"])): row
        for row in records
    }
    options = []
    coordinates = []
    for layer in range(NUM_LAYERS):
        for source in range(NUM_KV_HEADS):
            coordinate = []
            for rank in candidate_ranks:
                cost = (
                    0.0
                    if rank == anchor_rank
                    else float(
                        indexed[(layer, source, rank)]["terminal_kl_delta"][cost_key]
                    )
                )
                coordinate.append(
                    MetricRankOption(
                        option_id=f"layer_{layer:03d}.source_{source}.r{rank}.{cost_key}",
                        source_family="exact_terminal_kl",
                        rank=rank,
                        scalar_cost=cost,
                        is_anchor=rank == anchor_rank,
                    )
                )
            options.append(tuple(coordinate))
            coordinates.append((layer, source))
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=NUM_LAYERS * NUM_KV_HEADS * anchor_rank,
        anchor_rank=anchor_rank,
    )
    schedule = [[anchor_rank] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    contributions = []
    for (layer, source), option in zip(
        coordinates, allocation.selected_options, strict=True
    ):
        schedule[layer][source] = int(option.rank)
        contributions.append(
            {
                "layer": layer,
                "source": source,
                "rank": int(option.rank),
                "cost": float(option.scalar_cost),
            }
        )
    return schedule, float(allocation.total_cost), contributions


def _schedule_accounting(
    schedule: Sequence[Sequence[int]],
    *,
    anchor_rank: int,
) -> dict[str, Any]:
    selected = tuple(tuple(map(int, layer)) for layer in schedule)
    if len(selected) != NUM_LAYERS or any(
        len(layer) != NUM_KV_HEADS for layer in selected
    ):
        raise ValueError("schedule must contain 64 layers by eight TP sources")
    ideal_by_layer = [sum(layer) for layer in selected]
    padded_by_layer = [NUM_KV_HEADS * max(layer) for layer in selected]
    uniform_width = NUM_KV_HEADS * anchor_rank
    flat = [rank for layer in selected for rank in layer]
    return {
        "source_rank_sum": sum(flat),
        "source_rank_histogram": {
            str(rank): flat.count(rank) for rank in sorted(set(flat))
        },
        "anchor_rank": anchor_rank,
        "changed_sources_from_anchor": sum(rank != anchor_rank for rank in flat),
        "ideal_variable_allgather_width_by_layer": ideal_by_layer,
        "padded_rectangular_allgather_width_by_layer": padded_by_layer,
        "ideal_variable_allgather_total_width": sum(ideal_by_layer),
        "padded_rectangular_allgather_total_width": sum(padded_by_layer),
        "uniform_anchor_total_width": NUM_LAYERS * uniform_width,
        "dense_value_total_width": NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM,
        "ideal_dense_reduction": (
            NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM / sum(ideal_by_layer)
        ),
        "padded_overhead_vs_uniform_anchor": (
            sum(padded_by_layer) / (NUM_LAYERS * uniform_width) - 1.0
        ),
        "layers_with_padded_overhead": sum(
            width > uniform_width for width in padded_by_layer
        ),
    }


@torch.no_grad()
def _install_schedule(
    model: nn.Module,
    *,
    schedule: Sequence[Sequence[int]],
    dense_v_weights: Sequence[Tensor],
    factor_cache: Sequence[Mapping[int, RankFactors]],
    snapshot_cache: Sequence[SnapshotLayer],
    anchor_rank: int,
    covariance_damping: float,
    decoder_relative_jitter: float,
    keep_factors: bool,
) -> tuple[tuple[CachedFactors, ...] | None, list[dict[str, Any]]]:
    layers = _decoder_layers(model)
    kept = [] if keep_factors else None
    diagnostics = []
    for layer_index, ranks in enumerate(schedule):
        layer = layers[layer_index]
        device = layer.self_attn.v_proj.weight.device
        started = time.perf_counter()
        if all(int(rank) == anchor_rank for rank in ranks):
            A, D = _assemble_layer_factors(
                factor_cache[layer_index],
                source_ranks=ranks,
                dense_o_weight=snapshot_cache[layer_index].dense_o_weight,
                device=device,
            )
            factors = _quantized_cache(A, D, ranks)
            closure = {"closure": "uniform_anchor_factor_bank"}
            absolute_damping = None
            objective = None
            del A, D
        else:
            objective, absolute_damping = _objective(
                snapshot_cache[layer_index],
                layer=layer_index,
                device=device,
                covariance_damping=covariance_damping,
            )
            factors, closure = _closed_factors(
                bank=factor_cache[layer_index],
                snapshot=snapshot_cache[layer_index],
                objective=objective,
                source_ranks=ranks,
                anchor_rank=anchor_rank,
                decoder_relative_jitter=decoder_relative_jitter,
                device=device,
            )
        _install_factors(
            layer,
            dense_v_weight=dense_v_weights[layer_index],
            factors=factors,
        )
        diagnostics.append(
            {
                "layer": layer_index,
                "source_ranks": list(map(int, ranks)),
                "absolute_covariance_damping": absolute_damping,
                "elapsed_seconds": time.perf_counter() - started,
                **closure,
            }
        )
        if kept is not None:
            kept.append(factors)
        del objective, factors
        torch.cuda.empty_cache()
        _log(f"installed schedule layer {layer_index + 1}/{NUM_LAYERS}")
    return (None if kept is None else tuple(kept)), diagnostics


def _summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-32B C1 per-TP-source Global-KL allocation",
        "",
        "## Outcome",
        "",
        "| Schedule | Confirmation KL | WikiText-2 PPL | Changed sources | Padded overhead |",
        "|:---|---:|---:|---:|---:|",
    ]
    for label, row in result["schedules"].items():
        test = row.get("test")
        ppl = f"{test['ppl']:.9f}" if test is not None else "—"
        lines.append(
            f"| {label} | {row['confirmation']['terminal_kl']['mean']:.9g} | "
            f"{ppl} | {row['accounting']['changed_sources_from_anchor']} | "
            f"{100.0 * row['accounting']['padded_overhead_vs_uniform_anchor']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected: **{result['selection']['selected_candidate']}**.",
            "",
            "## Selected TP-source ranks",
            "",
            "| Layer | Physical KV-head ranks 0–7 |",
            "|---:|:---|",
        ]
    )
    for layer, ranks in enumerate(result["selection"]["selected_schedule"]):
        lines.append(f"| {layer} | {ranks} |")
    lines.extend(
        [
            "",
            "The DP constrains ideal variable-width TP=8 collective rank. "
            "Rectangular padded-AllGather width is diagnostic only.",
            "",
            "## Command",
            "",
            f"`{result['command']}`",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-32B Global-KL requires CUDA")
    started = time.perf_counter()

    positive = (
        args.profile_windows,
        args.confirmation_windows,
        args.sequence_length,
        args.batch_size,
        args.vocab_chunk_size,
        args.eval_seqlen,
        args.eval_batch_size,
        args.torch_num_threads,
        args.max_memory_per_gpu_gib,
    )
    if min(positive) <= 0:
        raise ValueError("sample and compute arguments must be positive")
    if args.covariance_damping < 0 or args.decoder_relative_jitter < 0:
        raise ValueError("damping and jitter must be non-negative")
    candidate_ranks = _parse_ranks(args.candidate_ranks)
    if candidate_ranks != (64, 80, 96, 112, 128) or args.anchor_rank != 96:
        raise ValueError("this experiment is fixed to ranks 64/80/96/112/128 at anchor 96")
    factor_dirs = _parse_factor_dirs(args.factor_dir)
    if set(factor_dirs) != set(candidate_ranks) - {HEAD_DIM}:
        raise ValueError("factor directories must provide exactly ranks 64/80/96/112")
    if args.window_start < FRESH_WINDOW_START:
        raise ValueError("Global-KL windows must begin after the 320 ALS documents")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    model_path = Path(args.model).expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    model_config_sha256 = _sha256(model_path / "config.json")
    factor_results = _load_factor_results(
        factor_dirs,
        model_config_sha256=model_config_sha256,
        snapshot_dir=snapshot_dir,
    )
    reference_fit_config = next(iter(factor_results.values()))["fit_config"]
    if float(reference_fit_config["covariance_damping"]) != args.covariance_damping:
        raise ValueError("Global-KL covariance damping must match the ALS factor banks")
    if reference_fit_config.get("work_dtype") != "float32":
        raise ValueError("Global-KL decoder closure requires the float32 ALS banks")
    if reference_fit_config.get("factor_dtype") != "bfloat16":
        raise ValueError("Global-KL deployment requires the bfloat16 ALS banks")
    profile_sequences, confirmation_sequences, windows_provenance = (
        _select_fresh_windows(
            args.windows,
            window_start=args.window_start,
            profile_windows=args.profile_windows,
            confirmation_windows=args.confirmation_windows,
            sequence_length=args.sequence_length,
        )
    )

    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    geometry = (
        str(model.config.model_type),
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        int(model.config.num_key_value_heads),
        int(
            getattr(model.config, "head_dim", 0)
            or model.config.hidden_size // model.config.num_attention_heads
        ),
        int(model.config.hidden_size),
    )
    expected_geometry = (
        MODEL_TYPE,
        NUM_LAYERS,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        HIDDEN_SIZE,
    )
    if geometry != expected_geometry:
        raise ValueError(f"unexpected Qwen3-32B geometry: {geometry}")
    placement = {
        parameter.device.type for parameter in model.parameters()
    }
    if placement - {"cuda"}:
        raise RuntimeError(f"Global-KL model contains offloaded parameters: {placement}")

    profile_teacher = _capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="profile",
    )
    confirmation_teacher = _capture_teacher(
        model,
        confirmation_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="confirmation",
    )
    snapshot_cache, snapshot_manifest = _load_snapshot_cache(
        snapshot_dir,
        model_path=model_path,
    )
    expected_manifest_sha = next(iter(factor_results.values()))["fit_config"][
        "snapshot_manifest_sha256"
    ]
    if _sha256(snapshot_dir / "manifest.json") != expected_manifest_sha:
        raise ValueError("factor banks and covariance manifest hashes disagree")
    factor_cache = _load_factor_cache(factor_dirs, factor_results)
    layers = _decoder_layers(model)
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone() for layer in layers
    )

    uniform_schedule = [[args.anchor_rank] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    _, anchor_install = _install_schedule(
        model,
        schedule=uniform_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=args.decoder_relative_jitter,
        keep_factors=False,
    )
    anchor_metrics = _evaluate_teacher_metrics(
        model,
        profile_teacher,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    _log(f"uniform anchor profile KL={anchor_metrics['terminal_kl']['mean']:.9g}")

    records = []
    damping_by_layer = []
    interventions = [
        (source, rank)
        for source in range(NUM_KV_HEADS)
        for rank in candidate_ranks
        if rank != args.anchor_rank
    ]
    for layer_index in range(NUM_LAYERS):
        layer = layers[layer_index]
        device = layer.self_attn.v_proj.weight.device
        anchor_v = layer.self_attn.v_proj.weight.detach().clone()
        anchor_o = layer.self_attn.o_proj.weight.detach().clone()
        objective, absolute_damping = _objective(
            snapshot_cache[layer_index],
            layer=layer_index,
            device=device,
            covariance_damping=args.covariance_damping,
        )
        damping_by_layer.append(absolute_damping)
        for intervention_index, (source, candidate_rank) in enumerate(interventions):
            source_ranks = [args.anchor_rank] * NUM_KV_HEADS
            source_ranks[source] = candidate_rank
            factors, closure = _closed_factors(
                bank=factor_cache[layer_index],
                snapshot=snapshot_cache[layer_index],
                objective=objective,
                source_ranks=source_ranks,
                anchor_rank=args.anchor_rank,
                decoder_relative_jitter=args.decoder_relative_jitter,
                device=device,
            )
            _install_factors(
                layer,
                dense_v_weight=dense_v_weights[layer_index],
                factors=factors,
            )
            metrics = _evaluate_teacher_metrics(
                model,
                profile_teacher,
                vocab_chunk_size=args.vocab_chunk_size,
            )
            layer.self_attn.v_proj.weight.copy_(anchor_v)
            layer.self_attn.o_proj.weight.copy_(anchor_o)
            delta = _paired_delta(metrics, anchor_metrics)
            records.append(
                {
                    "layer": layer_index,
                    "source": source,
                    "physical_kv_head": source,
                    "routed_query_heads": list(
                        range(
                            source * HEADS_PER_SOURCE,
                            (source + 1) * HEADS_PER_SOURCE,
                        )
                    ),
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": candidate_rank,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                    "ideal_width_delta": candidate_rank - args.anchor_rank,
                    "padded_width_delta": (
                        NUM_KV_HEADS * max(source_ranks)
                        - NUM_KV_HEADS * args.anchor_rank
                    ),
                    "decoder_solve": closure,
                }
            )
            _log(
                f"profile layer={layer_index} candidate="
                f"{intervention_index + 1}/{len(interventions)} source={source} "
                f"rank={candidate_rank} dKL={delta['terminal_kl']['mean']:.6g}"
            )
            del factors, metrics
            torch.cuda.empty_cache()
        del objective, anchor_v, anchor_o
        torch.cuda.empty_cache()
        _log(f"profile layer {layer_index + 1}/{NUM_LAYERS} complete")

    candidates = {}
    predicted = {}
    contributions = {}
    for label, cost_key in (
        ("mean_dp", "mean"),
        ("ucb_dp", "one_standard_error_ucb"),
    ):
        schedule, cost, rows = _allocate(
            records,
            candidate_ranks=candidate_ranks,
            anchor_rank=args.anchor_rank,
            cost_key=cost_key,
        )
        candidates[label] = schedule
        predicted[label] = cost
        contributions[label] = rows

    schedules = {"uniform_anchor": uniform_schedule, **candidates}
    confirmation = {}
    closure_diagnostics = {"uniform_anchor": anchor_install}
    for label, schedule in schedules.items():
        if label != "uniform_anchor":
            _, diagnostics = _install_schedule(
                model,
                schedule=schedule,
                dense_v_weights=dense_v_weights,
                factor_cache=factor_cache,
                snapshot_cache=snapshot_cache,
                anchor_rank=args.anchor_rank,
                covariance_damping=args.covariance_damping,
                decoder_relative_jitter=args.decoder_relative_jitter,
                keep_factors=False,
            )
            closure_diagnostics[label] = diagnostics
        confirmation[label] = _evaluate_teacher_metrics(
            model,
            confirmation_teacher,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {label}: "
            f"KL={confirmation[label]['terminal_kl']['mean']:.9g}"
        )

    selected_name = min(
        schedules,
        key=lambda label: (confirmation[label]["terminal_kl"]["mean"], label),
    )
    selected_schedule = schedules[selected_name]
    selected_factors, selected_install = _install_schedule(
        model,
        schedule=selected_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=args.decoder_relative_jitter,
        keep_factors=True,
    )
    assert selected_factors is not None
    closure_diagnostics[selected_name] = selected_install
    selected_test = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset="wikitext2",
        split="test",
        seqlen=args.eval_seqlen,
        batch_size=args.eval_batch_size,
        max_samples=args.eval_max_samples,
        max_tokens=args.eval_max_tokens,
    )
    test_metrics = {selected_name: selected_test}
    if selected_name != "uniform_anchor":
        _install_schedule(
            model,
            schedule=uniform_schedule,
            dense_v_weights=dense_v_weights,
            factor_cache=factor_cache,
            snapshot_cache=snapshot_cache,
            anchor_rank=args.anchor_rank,
            covariance_damping=args.covariance_damping,
            decoder_relative_jitter=args.decoder_relative_jitter,
            keep_factors=False,
        )
        test_metrics["uniform_anchor"] = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset="wikitext2",
            split="test",
            seqlen=args.eval_seqlen,
            batch_size=args.eval_batch_size,
            max_samples=args.eval_max_samples,
            max_tokens=args.eval_max_tokens,
        )
    _log(f"selected {selected_name}: PPL={selected_test['ppl']:.9f}")

    output_dir.mkdir(parents=True, exist_ok=False)
    selected_dir = output_dir / "selected_factors"
    selected_dir.mkdir()
    artifacts = {}
    for layer_index, factors in enumerate(selected_factors):
        path = selected_dir / f"layer_{layer_index:03d}.safetensors"
        save_file(
            {
                "value_coordinate_encoders": factors.A,
                "head_output_decoders": factors.D,
                "source_ranks": torch.tensor(factors.ranks, dtype=torch.int32),
            },
            str(path),
        )
        artifacts[str(layer_index)] = {
            "file": str(path.relative_to(output_dir)),
            "sha256": _sha256(path),
            "encoder_sha256": tensor_sha256(factors.A),
            "decoder_sha256": tensor_sha256(factors.D),
        }

    schedule_rows = {}
    for label, schedule in schedules.items():
        row = {
            "schedule": schedule,
            "accounting": _schedule_accounting(
                schedule,
                anchor_rank=args.anchor_rank,
            ),
            "confirmation": confirmation[label],
            "closure_diagnostics": closure_diagnostics[label],
        }
        if label in test_metrics:
            row["test"] = test_metrics[label]
        schedule_rows[label] = row
    result = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": str(model_path),
        "model_config_sha256": model_config_sha256,
        "geometry": {
            "layers": NUM_LAYERS,
            "query_heads": NUM_QUERY_HEADS,
            "physical_kv_heads": NUM_KV_HEADS,
            "tp_sources": NUM_KV_HEADS,
            "query_heads_per_source": HEADS_PER_SOURCE,
            "head_dim": HEAD_DIM,
        },
        "profile": {
            "dataset": "c4_train_fresh_documents",
            "windows": args.profile_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "dense_teacher": True,
            "uniform_anchor": anchor_metrics,
            "records": records,
            "absolute_covariance_damping_by_layer": damping_by_layer,
        },
        "confirmation": {
            "dataset": "c4_train_fresh_documents",
            "windows": args.confirmation_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "disjoint_from_profile": True,
            "disjoint_from_als_fit_and_heldout": True,
            "windows_provenance": windows_provenance,
        },
        "selection": {
            "constraint": "exact ideal variable-width TP-source rank budget",
            "target_source_rank_sum": NUM_LAYERS * NUM_KV_HEADS * args.anchor_rank,
            "candidate_ranks": list(candidate_ranks),
            "rank_128_endpoint": "exact identity Value encoder plus dense O initialization",
            "predicted_additive_costs": predicted,
            "contributions": contributions,
            "selected_candidate": selected_name,
            "selected_schedule": selected_schedule,
            "selected_accounting": _schedule_accounting(
                selected_schedule,
                anchor_rank=args.anchor_rank,
            ),
            "uniform_is_eligible": True,
            "selection_metric": "lowest disjoint-confirmation mean terminal KL",
        },
        "schedules": schedule_rows,
        "selected_artifacts": artifacts,
        "factor_sources": {
            str(rank): {
                "path": str(factor_dirs[rank]),
                "results_sha256": _sha256(factor_dirs[rank] / "results.json"),
            }
            for rank in factor_dirs
        },
        "snapshot": {
            "path": str(snapshot_dir),
            "manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
            "fit_windows": snapshot_manifest["calibration"]["fit_windows"],
            "fit_rows": snapshot_manifest["calibration"]["fit_rows"],
            "heldout_not_used_for_global_kl_selection": True,
        },
        "test_protocol": {
            "dataset": "wikitext2_test",
            "seqlen": args.eval_seqlen,
            "batch_size": args.eval_batch_size,
            "schedule_frozen_before_test": True,
        },
        "numerics": {
            "model_dtype": str(dtype),
            "deployed_factor_dtype": "torch.bfloat16",
            "decoder_closure": "float32",
            "terminal_kl_probability": "float32",
            "terminal_kl_accumulation": "float64",
            "quality_runtime": (
                "ragged source coordinates exactly zero-padded into native "
                "Qwen V/O slots for sharded Hugging Face SDPA"
            ),
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    _log(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
