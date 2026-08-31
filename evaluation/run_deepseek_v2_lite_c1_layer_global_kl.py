#!/usr/bin/env python3
"""Allocate DeepSeek-V2-Lite TP-source ranks by post-ALS layer Global-KL.

Every candidate ``(layer, rank)`` is an already fitted, decoder-closed ALS5
factor bank.  Profiling replaces all eight TP sources in one decoder layer,
measures terminal-logit KL against the dense model, and checkpoints each layer.
Exact dynamic programming then preserves the average source-rank-128 budget.
The MLA latent KV representation and KV cache are never modified.
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

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.global_rank_sensitivity import (  # noqa: E402
    logits_logsumexp,
    teacher_kl_sum,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from basisserve.core.tp_source_wo_c1 import (  # noqa: E402
    TPSourceWOLayout,
    fold_factors_to_dense_weight,
    validate_factors,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation.fit_deepseek_v2_lite_tp8_c1_joint import (  # noqa: E402
    FORMAT as FACTOR_FORMAT,
    HIDDEN_SIZE,
    MODEL_TYPE,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    SOURCE_WIDTH,
    TP_SIZE,
    VALUE_HEAD_DIM,
)
from evaluation.prepare_deepseek_v2_lite_c1_windows import (  # noqa: E402
    FORMAT as WINDOWS_FORMAT,
)


FORMAT = "basisserve.deepseek_v2_lite.tp8_source_wo_c1.layer_global_kl.v1"
PROFILE_FORMAT = (
    "basisserve.deepseek_v2_lite.tp8_source_wo_c1.layer_global_kl_profile.v1"
)
SNAPSHOT_FORMAT = "basisserve.attention_o_proj_covariances.v1"
FRESH_WINDOW_START = 320
RANK_DEPENDENT_CONFIG_KEYS = {"source_rank", "communication"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _cuda_device_indices() -> tuple[int, ...]:
    """Return CUDA ordinals that the runtime can actually access.

    On some Slurm nodes NVML reports every physical GPU to
    ``torch.cuda.device_count()``, while the CUDA runtime is constrained to the
    GPUs allocated to the job.  Probe the runtime before using an ordinal for
    model placement or memory statistics.
    """
    indices: list[int] = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)


def _reset_cuda_peak_memory_stats() -> None:
    for index in _cuda_device_indices():
        torch.cuda.reset_peak_memory_stats(index)


def _cuda_environment() -> dict[str, Any]:
    indices = _cuda_device_indices()
    return {
        "cuda_devices": [torch.cuda.get_device_name(index) for index in indices],
        "peak_cuda_allocated_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in indices
        },
    }


def _parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(sorted(set(int(piece) for piece in raw.split(",") if piece.strip())))
    if not ranks or min(ranks) <= 0 or max(ranks) > SOURCE_WIDTH:
        raise ValueError("candidate ranks must lie within the TP source width")
    return ranks


def _parse_factor_dirs(raw: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for item in raw:
        rank_text, separator, path_text = item.partition("=")
        if not separator or not rank_text or not path_text:
            raise ValueError(f"invalid rank factor directory: {item}")
        rank = int(rank_text)
        if rank in result:
            raise ValueError(f"duplicate factor directory for rank {rank}")
        result[rank] = Path(path_text).expanduser().resolve()
    return result


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    current = model
    for name in ("model", "language_model"):
        child = getattr(current, name, None)
        if child is not None:
            current = child
    layers = getattr(current, "layers", None)
    if layers is None:
        raise AttributeError("could not locate DeepSeek decoder layers")
    return layers


def _model_geometry(model: nn.Module) -> tuple[str, int, int, int, int]:
    config = model.config
    return (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.v_head_dim),
    )


def _load_model(args: argparse.Namespace, model_path: Path) -> nn.Module:
    cuda_indices = _cuda_device_indices()
    if not cuda_indices:
        raise RuntimeError("DeepSeek Global-KL requires CUDA")
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in cuda_indices
        },
    ).eval()
    model.config.use_cache = False
    expected = (
        MODEL_TYPE,
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        VALUE_HEAD_DIM,
    )
    if _model_geometry(model) != expected:
        raise ValueError(f"unexpected DeepSeek-V2-Lite geometry: {_model_geometry(model)}")
    placement = {parameter.device.type for parameter in model.parameters()}
    if placement - {"cuda"}:
        raise RuntimeError(f"Global-KL model contains offloaded parameters: {placement}")
    return model


def _normalized_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
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
    expected_snapshot_sha = _sha256(snapshot_dir / "manifest.json")
    results: dict[int, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for rank, directory in sorted(factor_dirs.items()):
        path = directory / "results.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FACTOR_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"incomplete DeepSeek factor bank: {path}")
        if tuple(map(int, payload.get("layers", ()))) != tuple(range(NUM_LAYERS)):
            raise ValueError(f"factor bank does not cover every layer: {path}")
        config = payload["fit_config"]
        if int(config.get("source_rank", -1)) != rank:
            raise ValueError(f"factor rank does not match its directory key: {path}")
        if config.get("model_config_sha256") != model_config_sha256:
            raise ValueError(f"factor bank belongs to another model: {path}")
        if config.get("snapshot_manifest_sha256") != expected_snapshot_sha:
            raise ValueError(f"factor bank belongs to another covariance snapshot: {path}")
        if config.get("compression_target") != "post_attention_tp_source_wo_allgather":
            raise ValueError("factor bank targets another communication boundary")
        if config.get("kv_cache_compression") != "none":
            raise ValueError("DeepSeek Global-KL must not alter MLA KV")
        for record in payload["records"]:
            if record["solver"]["method"] != "full_layer_joint_source_als":
                raise ValueError("candidate factors are not full-layer ALS factors")
            if record["selection"]["boundary"] not in {
                "decoder_only",
                "after_redecoder",
            }:
                raise ValueError("candidate factor is not decoder-closed")
        normalized = _normalized_fit_config(config)
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise ValueError("candidate factor banks differ beyond source rank")
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
        raise ValueError("incompatible DeepSeek C4 window manifest")
    if _sha256(resolved) != manifest["artifact"]["sha256"]:
        raise ValueError("DeepSeek C4 window artifact hash mismatch")
    tensors = load_file(str(resolved), device="cpu")
    if set(tensors) != {"input_ids"}:
        raise ValueError("ordered C4 window bank must contain only input_ids")
    input_ids = tensors["input_ids"].to(torch.long)
    needed = profile_windows + confirmation_windows
    stop = window_start + needed
    if (
        window_start < FRESH_WINDOW_START
        or min(profile_windows, confirmation_windows) <= 0
        or stop > len(input_ids)
        or sequence_length <= 1
        or sequence_length > int(input_ids.shape[1])
    ):
        raise ValueError("invalid or non-independent Global-KL window range")
    records = manifest.get("records", ())
    if len(records) != len(input_ids):
        raise ValueError("C4 manifest records do not match the window bank")
    document_ids = [str(record["document_id"]) for record in records]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Global-KL C4 bank is not document-disjoint")
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
        "disjoint_from_als_fit_and_heldout": True,
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


def _log(message: str, *, shard: int | None = None) -> None:
    prefix = "[DeepSeek layer Global-KL]"
    if shard is not None:
        prefix = f"[DeepSeek layer Global-KL shard {shard}]"
    print(f"{prefix} {message}", flush=True)


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
        result.append(
            TeacherBatch(
                input_ids=input_ids.cpu(),
                logits=logits.cpu(),
                logsumexp=logits_logsumexp(
                    logits, vocab_chunk_size=vocab_chunk_size
                ).cpu(),
            )
        )
        _log(
            f"dense teacher {label}: {min(start + batch_size, len(sequences))}/"
            f"{len(sequences)}"
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


class _FactorCache:
    def __init__(
        self,
        factor_dirs: Mapping[int, Path],
        factor_results: Mapping[int, Mapping[str, Any]],
    ) -> None:
        self.factor_dirs = dict(factor_dirs)
        self.factor_results = dict(factor_results)
        self.cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

    def get(self, rank: int, layer: int) -> tuple[Tensor, Tensor]:
        key = (int(rank), int(layer))
        if key not in self.cache:
            result = self.factor_results[rank]
            artifact = result["artifacts"][str(layer)]
            path = self.factor_dirs[rank] / artifact["file"]
            if _sha256(path) != artifact["sha256"]:
                raise ValueError(f"factor hash mismatch: rank={rank} layer={layer}")
            tensors = load_file(str(path), device="cpu")
            encoders = tensors["source_encoders"].contiguous()
            decoders = tensors["source_decoders"].contiguous()
            layout = TPSourceWOLayout(
                input_width=HIDDEN_SIZE,
                output_width=HIDDEN_SIZE,
                tp_size=TP_SIZE,
                source_rank=rank,
            )
            validate_factors(encoders, decoders, layout)
            self.cache[key] = (encoders, decoders)
        return self.cache[key]

    @torch.no_grad()
    def folded_weight(
        self,
        rank: int,
        layer: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        encoders, decoders = self.get(rank, layer)
        layout = TPSourceWOLayout(
            input_width=HIDDEN_SIZE,
            output_width=HIDDEN_SIZE,
            tp_size=TP_SIZE,
            source_rank=rank,
        )
        folded = fold_factors_to_dense_weight(
            encoders.to(device=device, dtype=torch.float32),
            decoders.to(device=device, dtype=torch.float32),
            layout,
        )
        return folded.to(dtype=dtype)


@torch.no_grad()
def _install_layer(
    layer: nn.Module,
    *,
    layer_index: int,
    rank: int,
    factor_cache: _FactorCache,
) -> None:
    o_proj = layer.self_attn.o_proj
    if getattr(o_proj, "bias", None) is not None or tuple(o_proj.weight.shape) != (
        HIDDEN_SIZE,
        HIDDEN_SIZE,
    ):
        raise TypeError(f"unsupported DeepSeek o_proj at layer {layer_index}")
    folded = factor_cache.folded_weight(
        rank,
        layer_index,
        device=o_proj.weight.device,
        dtype=o_proj.weight.dtype,
    )
    o_proj.weight.copy_(folded)
    del folded


@torch.no_grad()
def _install_schedule(
    model: nn.Module,
    schedule: Sequence[int],
    factor_cache: _FactorCache,
) -> None:
    if len(schedule) != NUM_LAYERS:
        raise ValueError("rank schedule must contain one rank per layer")
    for layer_index, (layer, rank) in enumerate(
        zip(_decoder_layers(model), schedule, strict=True)
    ):
        _install_layer(
            layer,
            layer_index=layer_index,
            rank=int(rank),
            factor_cache=factor_cache,
        )


def _shard_layers(shard_index: int, shard_count: int) -> tuple[int, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("profile shard index/count are invalid")
    layers = tuple(range(shard_index, NUM_LAYERS, shard_count))
    if not layers:
        raise ValueError("profile shard has no assigned layers")
    return layers


def _validate_shared(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    tuple[int, ...],
    dict[int, Path],
    dict[int, dict[str, Any]],
    Tensor,
    Tensor,
    dict[str, Any],
]:
    positive = (
        args.profile_windows,
        args.confirmation_windows,
        args.sequence_length,
        args.batch_size,
        args.vocab_chunk_size,
        args.torch_num_threads,
        args.max_memory_per_gpu_gib,
    )
    if min(positive) <= 0:
        raise ValueError("Global-KL sample and compute arguments must be positive")
    if args.batch_size < max(args.profile_windows, args.confirmation_windows):
        raise ValueError("batch size must cover each profile/confirmation split")
    candidate_ranks = _parse_ranks(args.candidate_ranks)
    if args.anchor_rank not in candidate_ranks:
        raise ValueError("anchor rank must be a candidate")
    if sum(candidate_ranks) <= 0:
        raise AssertionError("candidate rank validation failed")
    factor_dirs = _parse_factor_dirs(args.factor_dir)
    if set(factor_dirs) != set(candidate_ranks):
        raise ValueError("factor directories must provide every candidate rank")
    model_path = Path(args.model).expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    snapshot_manifest_path = snapshot_dir / "manifest.json"
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError(snapshot_manifest_path)
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    if snapshot_manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("incompatible covariance snapshot")
    if (
        snapshot_manifest["model"].get("model_type") != MODEL_TYPE
        or snapshot_manifest["model"].get("attention_type") != "mla"
        or int(snapshot_manifest["model"].get("num_hidden_layers", -1))
        != NUM_LAYERS
    ):
        raise ValueError("covariance snapshot is not DeepSeek-V2-Lite MLA")
    model_config_sha256 = _sha256(model_path / "config.json")
    if snapshot_manifest["model"]["config_sha256"] != model_config_sha256:
        raise ValueError("model and covariance snapshot differ")
    factor_results = _load_factor_results(
        factor_dirs,
        model_config_sha256=model_config_sha256,
        snapshot_dir=snapshot_dir,
    )
    fit_config = next(iter(factor_results.values()))["fit_config"]
    if (
        fit_config.get("work_dtype") != "float32"
        or fit_config.get("factor_dtype") != "bfloat16"
        or float(fit_config.get("covariance_damping", -1.0)) != 1.0e-5
        or int(fit_config.get("encoder_sweeps", -1)) != 5
        or int(fit_config.get("minimum_encoder_sweeps", -1)) != 5
        or float(fit_config.get("encoder_relative_damping", -1.0)) != 0.0
        or fit_config.get("selection_boundaries") != "decoder-closed"
    ):
        raise ValueError("Global-KL requires the fixed post-ALS5 factor protocol")
    profile, confirmation, provenance = _select_fresh_windows(
        args.windows,
        window_start=args.window_start,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
        sequence_length=args.sequence_length,
    )
    return (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        profile,
        confirmation,
        provenance,
    )


def _configuration(
    args: argparse.Namespace,
    *,
    model_path: Path,
    snapshot_dir: Path,
    candidate_ranks: Sequence[int],
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
    windows_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    fit_config = next(iter(factor_results.values()))["fit_config"]
    return {
        "model": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
        "factor_results_sha256": {
            str(rank): _sha256(directory / "results.json")
            for rank, directory in sorted(factor_dirs.items())
        },
        "factor_stage": {
            "encoder_initialization": fit_config["encoder_initialization"],
            "encoder_sweeps": int(fit_config["encoder_sweeps"]),
            "minimum_encoder_sweeps": int(fit_config["minimum_encoder_sweeps"]),
            "selection_boundaries": fit_config["selection_boundaries"],
            "decoder_objective": "full_layer_joint_source_als",
        },
        "allocation_unit": "decoder_layer",
        "intervention": "all_eight_tp_sources_at_one_rank",
        "anchor_rank": args.anchor_rank,
        "candidate_ranks": list(map(int, candidate_ranks)),
        "target_layer_rank_sum": NUM_LAYERS * args.anchor_rank,
        "profile_windows": args.profile_windows,
        "confirmation_windows": args.confirmation_windows,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "vocab_chunk_size": args.vocab_chunk_size,
        "model_dtype": args.model_dtype,
        "attn_implementation": args.attn_implementation,
        "windows_provenance": dict(windows_provenance),
        "kv_cache_compression": "none",
    }


def _profile_payload(
    *,
    status: str,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    assigned_layers: Sequence[int],
    completed_layers: Sequence[int],
    anchor_metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    started: float,
) -> dict[str, Any]:
    return {
        "format": PROFILE_FORMAT,
        "status": status,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile_shard_index": args.profile_shard_index,
        "profile_shard_count": args.profile_shard_count,
        "assigned_layers": list(map(int, assigned_layers)),
        "completed_layers": list(map(int, completed_layers)),
        "configuration": dict(configuration),
        "uniform_anchor": dict(anchor_metrics),
        "records": list(records),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            **_cuda_environment(),
        },
    }


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    _reset_cuda_peak_memory_stats()
    assigned_layers = _shard_layers(
        args.profile_shard_index, args.profile_shard_count
    )
    (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        profile_sequences,
        _,
        windows_provenance,
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    shard_path = profile_dir / f"shard_{args.profile_shard_index:02d}.json"
    records: list[dict[str, Any]] = []
    completed_layers: list[int] = []
    prior_anchor: dict[str, Any] | None = None
    if shard_path.is_file():
        prior = json.loads(shard_path.read_text(encoding="utf-8"))
        if (
            prior.get("format") != PROFILE_FORMAT
            or prior.get("configuration") != configuration
            or tuple(map(int, prior.get("assigned_layers", ())))
            != assigned_layers
        ):
            raise ValueError(f"incompatible profile checkpoint: {shard_path}")
        if prior.get("status") == "complete":
            _log(f"complete checkpoint already exists: {shard_path}")
            return
        records = [dict(record) for record in prior.get("records", ())]
        completed_layers = list(map(int, prior.get("completed_layers", ())))
        prior_anchor = dict(prior["uniform_anchor"])
    expected_records = len(completed_layers) * (len(candidate_ranks) - 1)
    if len(records) != expected_records:
        raise ValueError("profile checkpoint record count is incomplete")

    started = time.perf_counter()
    model = _load_model(args, model_path)
    teacher = _capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label=f"profile shard {args.profile_shard_index}",
    )
    factor_cache = _FactorCache(factor_dirs, factor_results)
    anchor_schedule = [args.anchor_rank] * NUM_LAYERS
    _install_schedule(model, anchor_schedule, factor_cache)
    anchor_metrics = _evaluate_teacher_metrics(
        model, teacher, vocab_chunk_size=args.vocab_chunk_size
    )
    if prior_anchor is not None:
        previous = prior_anchor["terminal_kl"]["values"]
        current = anchor_metrics["terminal_kl"]["values"]
        if len(previous) != len(current) or max(
            abs(float(left) - float(right))
            for left, right in zip(previous, current, strict=True)
        ) > 1.0e-7:
            raise RuntimeError("resumed profile anchor changed numerically")
    _log(
        f"anchor KL={anchor_metrics['terminal_kl']['mean']:.9g}",
        shard=args.profile_shard_index,
    )

    completed = set(completed_layers)
    intervention_ranks = [
        rank for rank in candidate_ranks if rank != args.anchor_rank
    ]
    layers = _decoder_layers(model)
    for layer_index in assigned_layers:
        if layer_index in completed:
            _log(f"resume skips layer {layer_index}", shard=args.profile_shard_index)
            continue
        layer = layers[layer_index]
        anchor_weight = layer.self_attn.o_proj.weight.detach().clone()
        layer_records = []
        for candidate_index, rank in enumerate(intervention_ranks):
            _install_layer(
                layer,
                layer_index=layer_index,
                rank=rank,
                factor_cache=factor_cache,
            )
            metrics = _evaluate_teacher_metrics(
                model, teacher, vocab_chunk_size=args.vocab_chunk_size
            )
            layer.self_attn.o_proj.weight.copy_(anchor_weight)
            delta = _paired_delta(metrics, anchor_metrics)
            layer_records.append(
                {
                    "layer": layer_index,
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": rank,
                    "tp_sources_changed": TP_SIZE,
                    "layer_rank_delta": rank - args.anchor_rank,
                    "collective_width": TP_SIZE * rank,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                }
            )
            _log(
                f"layer={layer_index} candidate={candidate_index + 1}/"
                f"{len(intervention_ranks)} rank={rank} "
                f"dKL={delta['terminal_kl']['mean']:.7g}",
                shard=args.profile_shard_index,
            )
            del metrics
            torch.cuda.empty_cache()
        records.extend(layer_records)
        completed_layers.append(layer_index)
        _atomic_json(
            shard_path,
            _profile_payload(
                status="running",
                args=args,
                configuration=configuration,
                assigned_layers=assigned_layers,
                completed_layers=completed_layers,
                anchor_metrics=anchor_metrics,
                records=records,
                started=started,
            ),
        )
        _log(
            f"checkpointed {len(completed_layers)}/{len(assigned_layers)} layers",
            shard=args.profile_shard_index,
        )
    _atomic_json(
        shard_path,
        _profile_payload(
            status="complete",
            args=args,
            configuration=configuration,
            assigned_layers=assigned_layers,
            completed_layers=completed_layers,
            anchor_metrics=anchor_metrics,
            records=records,
            started=started,
        ),
    )
    _log(f"completed {shard_path}", shard=args.profile_shard_index)


def _merge_profile_shards(
    profile_dir: Path,
    *,
    shard_count: int,
    configuration: Mapping[str, Any],
    candidate_ranks: Sequence[int],
) -> dict[str, Any]:
    shards = []
    for shard_index in range(shard_count):
        path = profile_dir / f"shard_{shard_index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_layers = _shard_layers(shard_index, shard_count)
        if (
            payload.get("format") != PROFILE_FORMAT
            or payload.get("status") != "complete"
            or int(payload.get("profile_shard_index", -1)) != shard_index
            or int(payload.get("profile_shard_count", -1)) != shard_count
            or payload.get("configuration") != configuration
            or tuple(map(int, payload.get("assigned_layers", ())))
            != expected_layers
            or tuple(sorted(map(int, payload.get("completed_layers", ()))))
            != expected_layers
        ):
            raise ValueError(f"incompatible or incomplete profile shard: {path}")
        shards.append((path, payload))
    records = sorted(
        [dict(record) for _, shard in shards for record in shard["records"]],
        key=lambda record: (record["layer"], record["candidate_rank"]),
    )
    expected_records = NUM_LAYERS * (len(candidate_ranks) - 1)
    keys = {
        (int(record["layer"]), int(record["candidate_rank"]))
        for record in records
    }
    if len(records) != expected_records or len(keys) != expected_records:
        raise ValueError("merged profile lacks a unique record for every intervention")
    return {
        "records": records,
        "uniform_anchor_by_shard": [
            shard["uniform_anchor"] for _, shard in shards
        ],
        "shards": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "assigned_layers": shard["assigned_layers"],
                "elapsed_seconds": shard["elapsed_seconds"],
            }
            for path, shard in shards
        ],
    }


def _allocate_layer_ranks(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    cost_key: str,
) -> tuple[list[int], float, list[dict[str, Any]]]:
    indexed = {
        (int(record["layer"]), int(record["candidate_rank"])): record
        for record in records
    }
    options = []
    for layer in range(NUM_LAYERS):
        coordinate = []
        for rank in candidate_ranks:
            cost = (
                0.0
                if rank == anchor_rank
                else float(indexed[(layer, rank)]["terminal_kl_delta"][cost_key])
            )
            coordinate.append(
                MetricRankOption(
                    option_id=f"layer_{layer:03d}.r{rank}.{cost_key}",
                    source_family="post_als_whole_layer_terminal_kl",
                    rank=rank,
                    scalar_cost=cost,
                    is_anchor=rank == anchor_rank,
                )
            )
        options.append(tuple(coordinate))
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=NUM_LAYERS * anchor_rank,
        anchor_rank=anchor_rank,
    )
    schedule = [int(option.rank) for option in allocation.selected_options]
    contributions = [
        {
            "layer": layer,
            "rank": int(option.rank),
            "cost": float(option.scalar_cost),
        }
        for layer, option in enumerate(allocation.selected_options)
    ]
    return schedule, float(allocation.total_cost), contributions


def _schedule_accounting(
    schedule: Sequence[int],
    *,
    anchor_rank: int,
) -> dict[str, Any]:
    ranks = tuple(map(int, schedule))
    if len(ranks) != NUM_LAYERS or min(ranks) <= 0 or max(ranks) > SOURCE_WIDTH:
        raise ValueError("invalid DeepSeek layer-rank schedule")
    layer_rank_sum = sum(ranks)
    if layer_rank_sum != NUM_LAYERS * anchor_rank:
        raise ValueError("schedule violates the exact average-rank budget")
    source_rank_sum = TP_SIZE * layer_rank_sum
    dense_source_rank_sum = NUM_LAYERS * TP_SIZE * SOURCE_WIDTH
    return {
        "layer_ranks": list(ranks),
        "layer_rank_sum": layer_rank_sum,
        "average_source_rank": layer_rank_sum / NUM_LAYERS,
        "layer_rank_histogram": {
            str(rank): ranks.count(rank) for rank in sorted(set(ranks))
        },
        "source_rank_sum": source_rank_sum,
        "dense_source_rank_sum": dense_source_rank_sum,
        "retained_ratio_vs_dense_allgather": (
            source_rank_sum / dense_source_rank_sum
        ),
        "reduction_vs_dense_allgather": 1.0
        - source_rank_sum / dense_source_rank_sum,
        "changed_layers_from_anchor": sum(rank != anchor_rank for rank in ranks),
        "collective_width_by_layer": [TP_SIZE * rank for rank in ranks],
        "ragged_sources_within_layer": False,
        "kv_cache_compression": "none",
    }


def _summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# DeepSeek-V2-Lite C1 per-layer Global-KL allocation",
        "",
        "- MLA latent KV and KV cache are unchanged.",
        "- Exact average source rank: `128`.",
        "- Attention-output reduction versus dense AllGather: `50%`.",
        "",
        "| Schedule | Confirmation KL | WikiText-2 PPL | Changed layers |",
        "|:---|---:|---:|---:|",
    ]
    for label, row in result["schedules"].items():
        ppl = row.get("test", {}).get("ppl", "")
        ppl_text = f"{float(ppl):.9g}" if ppl != "" else ""
        lines.append(
            f"| {label} | {row['confirmation']['terminal_kl']['mean']:.9g} | "
            f"{ppl_text} | {row['accounting']['changed_layers_from_anchor']} |"
        )
    lines.extend(
        [
            "",
            f"Selected: **{result['selection']['selected_candidate']}**.",
            "",
            "| Layer | Source rank | Gathered width |",
            "|---:|---:|---:|",
        ]
    )
    for layer, rank in enumerate(
        result["selection"]["selected_accounting"]["layer_ranks"]
    ):
        lines.append(f"| {layer} | {rank} | {TP_SIZE * rank} |")
    lines.extend(["", "## Command", "", f"`{result['command']}`", ""])
    return "\n".join(lines)


@torch.inference_mode()
def _finalize(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    _reset_cuda_peak_memory_stats()
    started = time.perf_counter()
    (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        _,
        confirmation_sequences,
        windows_provenance,
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    merged = _merge_profile_shards(
        profile_dir,
        shard_count=args.profile_shard_count,
        configuration=configuration,
        candidate_ranks=candidate_ranks,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    model = _load_model(args, model_path)
    teacher = _capture_teacher(
        model,
        confirmation_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="confirmation",
    )
    factor_cache = _FactorCache(factor_dirs, factor_results)
    uniform_schedule = [args.anchor_rank] * NUM_LAYERS
    candidates: dict[str, list[int]] = {}
    predicted: dict[str, float] = {}
    contributions: dict[str, list[dict[str, Any]]] = {}
    for label, cost_key in (
        ("mean_dp", "mean"),
        ("ucb_dp", "one_standard_error_ucb"),
    ):
        schedule, cost, rows = _allocate_layer_ranks(
            merged["records"],
            candidate_ranks=candidate_ranks,
            anchor_rank=args.anchor_rank,
            cost_key=cost_key,
        )
        candidates[label] = schedule
        predicted[label] = cost
        contributions[label] = rows
    schedules = {"uniform_anchor": uniform_schedule, **candidates}
    confirmation = {}
    for label, schedule in schedules.items():
        _install_schedule(model, schedule, factor_cache)
        confirmation[label] = _evaluate_teacher_metrics(
            model, teacher, vocab_chunk_size=args.vocab_chunk_size
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

    test_metrics = {}
    if not args.skip_wikitext:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        for label in dict.fromkeys((selected_name, "uniform_anchor")):
            _install_schedule(model, schedules[label], factor_cache)
            test_metrics[label] = _eval_ppl_fp32_loss(
                model,
                tokenizer,
                dataset="wikitext2",
                split="test",
                seqlen=args.eval_seqlen,
                batch_size=args.eval_batch_size,
                max_samples=None,
                max_tokens=None,
            )
            _log(f"WikiText {label}: PPL={test_metrics[label]['ppl']:.9f}")

    output_dir.mkdir(parents=True, exist_ok=False)
    selected_dir = output_dir / "selected_factors"
    selected_dir.mkdir()
    artifacts = {}
    for layer, rank in enumerate(selected_schedule):
        encoders, decoders = factor_cache.get(rank, layer)
        path = selected_dir / f"layer_{layer:03d}.safetensors"
        save_file(
            {
                "source_encoders": encoders.contiguous(),
                "source_decoders": decoders.contiguous(),
            },
            str(path),
        )
        artifacts[str(layer)] = {
            "file": str(path.relative_to(output_dir)),
            "sha256": _sha256(path),
            "source_rank": rank,
        }
    schedule_rows = {}
    for label, schedule in schedules.items():
        row = {
            "schedule": schedule,
            "accounting": _schedule_accounting(
                schedule, anchor_rank=args.anchor_rank
            ),
            "confirmation": confirmation[label],
        }
        if label in test_metrics:
            row["test"] = test_metrics[label]
        schedule_rows[label] = row
    selected_accounting = _schedule_accounting(
        selected_schedule, anchor_rank=args.anchor_rank
    )
    result = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "geometry": {
            "layers": NUM_LAYERS,
            "attention": "MLA",
            "attention_heads": NUM_ATTENTION_HEADS,
            "value_head_dim": VALUE_HEAD_DIM,
            "tp_sources": TP_SIZE,
            "source_width": SOURCE_WIDTH,
            "hidden_size": HIDDEN_SIZE,
        },
        "profile": {
            "execution": "two independent two-GPU model replicas",
            "intervention_unit": "whole decoder layer",
            "sources_changed_per_intervention": TP_SIZE,
            "factor_stage": configuration["factor_stage"],
            "profile_shard_count": args.profile_shard_count,
            "dataset": "c4_train_fresh_documents",
            "windows": args.profile_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_candidate": math.ceil(
                args.profile_windows / args.batch_size
            ),
            "uniform_anchor_by_shard": merged["uniform_anchor_by_shard"],
            "records": merged["records"],
            "shards": merged["shards"],
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
            "constraint": "exact per-layer source-rank budget",
            "target_layer_rank_sum": NUM_LAYERS * args.anchor_rank,
            "candidate_ranks": list(candidate_ranks),
            "predicted_additive_costs": predicted,
            "contributions": contributions,
            "selected_candidate": selected_name,
            "selected_schedule": selected_schedule,
            "selected_accounting": selected_accounting,
            "uniform_is_eligible": True,
            "selection_metric": "lowest independent-confirmation mean terminal KL",
        },
        "schedules": schedule_rows,
        "selected_artifacts": artifacts,
        "factor_sources": {
            str(rank): {
                "path": str(factor_dirs[rank]),
                "results_sha256": _sha256(factor_dirs[rank] / "results.json"),
            }
            for rank in candidate_ranks
        },
        "snapshot": {
            "path": str(snapshot_dir),
            "manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
        },
        "numerics": {
            "model_dtype": args.model_dtype,
            "deployed_factor_dtype": "bfloat16",
            "terminal_kl_probability": "float32",
            "terminal_kl_accumulation": "float64",
            "kv_cache_compression": "none",
            "collective": "private_allgather",
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            **_cuda_environment(),
        },
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    _log(f"selected {selected_name}; wrote {output_dir}")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-dir", action="append", required=True, metavar="RANK=PATH"
    )
    parser.add_argument("--anchor-rank", type=int, default=128)
    parser.add_argument("--candidate-ranks", default="64,96,128,160,192")
    parser.add_argument("--window-start", type=int, default=FRESH_WINDOW_START)
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    profile = commands.add_parser("profile")
    _add_shared_args(profile)
    profile.add_argument("--profile-shard-index", type=int, required=True)
    profile.add_argument("--profile-shard-count", type=int, default=2)
    finalize = commands.add_parser("finalize")
    _add_shared_args(finalize)
    finalize.add_argument("--profile-shard-count", type=int, default=2)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--eval-seqlen", type=int, default=2048)
    finalize.add_argument("--eval-batch-size", type=int, default=4)
    finalize.add_argument("--skip-wikitext", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "profile":
        _profile(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
