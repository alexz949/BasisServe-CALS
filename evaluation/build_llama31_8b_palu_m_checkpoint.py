#!/usr/bin/env python3
"""Build the activation-aware PaLU-M V-cache checkpoint for Llama-3.1-8B.

The workflow is intentionally split into three auditable stages:

1. ``prepare-windows`` samples 128 document-disjoint C4 windows of 2048 tokens.
2. ``prepare-whitening`` accumulates the per-layer V-input X^T X statistics with
   torchrun and writes their Cholesky factors.
3. ``build`` factorizes only ``v_proj`` into eight independent groups at the
   requested uniform rank and writes a compact factor checkpoint. The default
   remains rank 96 for the matched V25 protocol, and K remains dense.

The checkpoint contains factors, not a second copy of the Hugging Face model.
Its manifest pins the base weights, calibration windows, whitening artifact,
geometry, and numerical diagnostics needed by later PPL and lm-eval runners.

The Qwen3-8B entry point imports this implementation and activates its own
model profile before parsing commands. Keeping one implementation ensures the
two matched baselines use identical sampling and factorization semantics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    _CaptureExit,
    _FirstLayerCatcher,
    _all_reduce_sum,
    _model_layers,
    _stable_cholesky,
    distributed_whitening_cholesky,
)
from palu.model.modules.svd_linear import HeadwiseLowRankModule  # noqa: E402


WINDOWS_FORMAT = "basisserve.calibration.c4_document_windows.v1"
PACKED_WINDOWS_FORMAT = "basisserve.calibration.c4_packed_windows.v1"
MODEL_PROFILES = {
    "llama2_7b": {
        "label": "Llama-2-7B",
        "model_type": "llama",
        "repo": "meta-llama/Llama-2-7b-hf",
        "revision": "01c7f73d771dfac7d292323805ebc428287df4f9",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_query_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "format_slug": "llama2_7b",
    },
    "llama31_8b": {
        "label": "Llama-3.1-8B",
        "model_type": "llama",
        "repo": "meta-llama/Llama-3.1-8B",
        "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "llama31_8b",
    },
    "qwen3_8b": {
        "label": "Qwen3-8B-Base",
        "model_type": "qwen3",
        "repo": "Qwen/Qwen3-8B-Base",
        "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        "num_layers": 36,
        "hidden_size": 4096,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "qwen3_8b",
    },
    "qwen3_32b": {
        "label": "Qwen3-32B-Base",
        "model_type": "qwen3",
        "repo": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "num_layers": 64,
        "hidden_size": 5120,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "qwen3_32b",
    },
    "qwen35_9b": {
        "label": "Qwen3.5-9B",
        "model_type": "qwen3_5_text",
        "repo": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_query_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "format_slug": "qwen35_9b",
    },
    "llama31_70b": {
        "label": "Llama-3.1-70B",
        "model_type": "llama",
        "repo": "meta-llama/Llama-3.1-70B",
        "revision": "349b2ddb53ce8f2849a6c168a81980ab25258dac",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "format_slug": "llama31_70b",
    },
}
MODEL_LABEL = ""
MODEL_TYPE = ""
MODEL_REPO = ""
MODEL_REVISION = ""
NUM_LAYERS = 0
HIDDEN_SIZE = 0
NUM_QUERY_HEADS = 0
NUM_KV_HEADS = 0
HEAD_DIM = 0
WHITENING_FORMAT = ""
CHECKPOINT_FORMAT = ""
RANK_PER_KV_HEAD = 96
SEQUENCE_LENGTH = 2048
CALIBRATION_SAMPLES = 128


def activate_model_profile(name: str) -> None:
    profile = MODEL_PROFILES[name]
    global MODEL_LABEL, MODEL_TYPE, MODEL_REPO, MODEL_REVISION
    global NUM_LAYERS, HIDDEN_SIZE, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM
    global WHITENING_FORMAT, CHECKPOINT_FORMAT
    MODEL_LABEL = str(profile["label"])
    MODEL_TYPE = str(profile["model_type"])
    MODEL_REPO = str(profile["repo"])
    MODEL_REVISION = str(profile["revision"])
    NUM_LAYERS = int(profile["num_layers"])
    HIDDEN_SIZE = int(profile["hidden_size"])
    NUM_QUERY_HEADS = int(profile["num_query_heads"])
    NUM_KV_HEADS = int(profile["num_kv_heads"])
    HEAD_DIM = int(profile["head_dim"])
    slug = str(profile["format_slug"])
    WHITENING_FORMAT = f"basisserve.{slug}.palu_v_whitening.v1"
    CHECKPOINT_FORMAT = f"basisserve.{slug}.palu_m_v_only.v1"


activate_model_profile("llama31_8b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _hf_download_revision(model_path: Path) -> str | None:
    metadata = (
        model_path
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    if metadata.is_file():
        first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        return first_line or None
    # Standard huggingface_hub caches encode the immutable commit directly in
    # ``.../snapshots/<revision>`` and store model files as symlinks to blobs.
    if model_path.parent.name == "snapshots":
        revision = model_path.name.lower()
        if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision):
            return revision
    return None


def _model_metadata(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("model must contain config.json and model.safetensors.index.json")
    revision = _hf_download_revision(model_path)
    if revision != MODEL_REVISION:
        raise ValueError(
            f"expected pinned {MODEL_LABEL} revision {MODEL_REVISION}, found {revision!r}"
        )
    return {
        "path": str(model_path),
        "huggingface_repo": MODEL_REPO,
        "revision": revision,
        "config_sha256": _sha256(config_path),
        "safetensors_index_sha256": _sha256(index_path),
    }


def _validate_config(config: Any) -> None:
    text_config = getattr(config, "text_config", config)
    head_dim = int(
        getattr(text_config, "head_dim", 0)
        or text_config.hidden_size // text_config.num_attention_heads
    )
    observed = (
        str(text_config.model_type),
        int(text_config.num_hidden_layers),
        int(text_config.hidden_size),
        int(text_config.num_attention_heads),
        int(text_config.num_key_value_heads),
        head_dim,
    )
    expected = (
        MODEL_TYPE,
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    if observed != expected:
        raise ValueError(f"expected {MODEL_LABEL} GQA geometry {expected}, found {observed}")


def _document_id(row: Mapping[str, Any], stream_index: int) -> str:
    for key in ("id", "url", "timestamp"):
        value = row.get(key)
        if value:
            return f"{key}:{value}"
    digest = hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()[:24]
    return f"sha256:{digest}:stream:{stream_index}"


def _prepare_windows(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_config(AutoConfig.from_pretrained(str(model_path), local_files_only=True))
    model_metadata = _model_metadata(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )

    requested_revision = args.dataset_revision
    dataset_revision = HfApi().dataset_info(
        "allenai/c4", revision=requested_revision
    ).sha
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split=args.dataset_split,
        streaming=True,
        revision=dataset_revision,
        cache_dir=args.dataset_cache,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    generator = random.Random(args.seed)
    seen_ids: set[str] = set()
    windows: list[Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for stream_index, row in enumerate(dataset):
        document_id = _document_id(row, stream_index)
        if document_id in seen_ids:
            continue
        input_ids = tokenizer(
            str(row["text"]),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        if input_ids.numel() < args.sequence_length:
            continue
        token_start = generator.randint(0, input_ids.numel() - args.sequence_length)
        window = input_ids[token_start : token_start + args.sequence_length].to(torch.int32)
        windows.append(window.cpu().contiguous())
        records.append(
            {
                "sample_index": len(windows) - 1,
                "document_id": document_id,
                "stream_index": stream_index,
                "token_start": token_start,
                "document_token_count": int(input_ids.numel()),
                "input_ids_sha256": _tensor_sha256(window),
            }
        )
        seen_ids.add(document_id)
        if len(windows) == args.samples:
            break
    if len(windows) != args.samples:
        raise RuntimeError(f"C4 stream yielded only {len(windows)} usable unique documents")

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "windows.safetensors"
    stacked = torch.stack(windows)
    _atomic_safetensors(artifact_path, {"input_ids": stacked})
    manifest = {
        "format": WINDOWS_FORMAT,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
        },
        "dataset": {
            "repo": "allenai/c4",
            "config": "en",
            "split": args.dataset_split,
            "requested_revision": requested_revision,
            "resolved_revision": dataset_revision,
            "streaming": True,
            "shuffle_buffer": args.shuffle_buffer,
        },
        "sampling": {
            "seed": args.seed,
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "document_disjoint": len(seen_ids) == len(windows),
            "add_special_tokens": False,
            "within_document_start": "uniform integer",
        },
        "records": records,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensor": "input_ids",
            "dtype": str(stacked.dtype),
            "shape": list(stacked.shape),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"[C4] wrote {artifact_path} shape={tuple(stacked.shape)}", flush=True)


def _distributed_identity() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("prepare-whitening must be launched with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    return dist.get_rank(), local_rank, dist.get_world_size()


def _move_nested_tensors(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors(item, device) for key, item in value.items()
        }
    return value


@torch.no_grad()
def sequential_cpu_offload_whitening_cholesky(
    model: nn.Module,
    windows: list[dict[str, Tensor]],
    *,
    seqlen: int,
    device: torch.device,
) -> list[Tensor]:
    """Collect whitening statistics with only one decoder layer on GPU.

    The dense model remains on CPU. Hidden states stay on the selected GPU
    while decoder layers stream CPU -> GPU -> CPU in model order. This computes
    the same per-layer X^T X objective as the full-model single-GPU path.
    """

    layers = _model_layers(model)
    hidden_size = int(model.config.hidden_size)
    samples = len(windows)
    dtype = next(model.parameters()).dtype
    inputs = torch.empty(
        (samples, seqlen, hidden_size), dtype=dtype, device=device
    )
    outputs = torch.empty_like(inputs)
    captured_kwargs: dict[str, Any] = {}
    first_layer = layers[0]
    catcher = _FirstLayerCatcher(first_layer, inputs, captured_kwargs)
    layers[0] = catcher
    try:
        for batch in windows:
            try:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            except _CaptureExit:
                pass
    finally:
        layers[0] = first_layer
    if catcher.index != samples:
        raise RuntimeError(f"captured {catcher.index} inputs, expected {samples}")
    layer_kwargs = _move_nested_tensors(captured_kwargs, device)

    cholesky_factors: list[Tensor] = []
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        layer.to(device=device)
        raw_xtx = torch.zeros(
            (hidden_size, hidden_size), dtype=torch.float32, device=device
        )

        def accumulate_input(
            module: nn.Module, hook_inputs: tuple[Tensor, ...]
        ) -> None:
            del module
            hidden = hook_inputs[0].detach().reshape(-1, hidden_size).float()
            raw_xtx.addmm_(hidden.transpose(0, 1), hidden)

        handle = layer.self_attn.k_proj.register_forward_pre_hook(accumulate_input)
        try:
            for sample_index in range(samples):
                result = layer(
                    inputs[sample_index].unsqueeze(0),
                    **layer_kwargs,
                )
                hidden_states = result if torch.is_tensor(result) else result[0]
                outputs[sample_index].copy_(hidden_states[0])
        finally:
            handle.remove()
        _all_reduce_sum(raw_xtx)
        cholesky = _stable_cholesky(raw_xtx)
        cholesky_factors.append(cholesky.cpu())
        layer.to(device="cpu")
        del raw_xtx, cholesky
        inputs, outputs = outputs, inputs
        torch.cuda.empty_cache()
        print(
            f"[Sequential whitening] layer={layer_index}/{len(layers) - 1} "
            f"seconds={time.perf_counter() - layer_started:.2f}",
            flush=True,
        )
    return cholesky_factors


def _load_windows(
    path: Path,
    *,
    rank: int,
    world_size: int,
    samples: int,
) -> tuple[list[dict[str, Tensor]], dict[str, Any]]:
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing windows artifact or manifest under {path.parent}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in (WINDOWS_FORMAT, PACKED_WINDOWS_FORMAT):
        raise ValueError("incompatible calibration-window manifest")
    if _sha256(path) != manifest["artifact"]["sha256"]:
        raise ValueError("calibration-window artifact hash changed")
    input_ids = load_file(str(path), device="cpu")["input_ids"].to(torch.long)
    if tuple(input_ids.shape) != (samples, SEQUENCE_LENGTH):
        raise ValueError(
            f"expected calibration windows [{samples}, {SEQUENCE_LENGTH}], found {list(input_ids.shape)}"
        )
    if samples % world_size:
        raise ValueError("calibration samples must divide evenly across torchrun workers")
    local = input_ids[rank::world_size].contiguous()
    windows = [
        {
            "input_ids": row.unsqueeze(0),
            "attention_mask": torch.ones_like(row).unsqueeze(0),
        }
        for row in local
    ]
    return windows, manifest


@torch.no_grad()
def _prepare_whitening(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    rank, local_rank, world_size = _distributed_identity()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    exists = torch.tensor(int(output_dir.exists()), dtype=torch.int32, device=device)
    _all_reduce_sum(exists)
    if int(exists.item()):
        raise FileExistsError(output_dir)

    model_path = Path(args.model).expanduser().resolve()
    model_metadata = _model_metadata(model_path)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    windows_path = Path(args.windows).expanduser().resolve()
    windows, windows_manifest = _load_windows(
        windows_path,
        rank=rank,
        world_size=world_size,
        samples=args.samples,
    )
    if windows_manifest["model"]["config_sha256"] != model_metadata["config_sha256"]:
        raise ValueError("calibration windows were tokenized for another model")

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    _validate_config(model.config)
    model.config.use_cache = False
    if args.sequential_cpu_offload:
        factors = sequential_cpu_offload_whitening_cholesky(
            model,
            windows,
            seqlen=SEQUENCE_LENGTH,
            device=device,
        )
        execution_strategy = (
            f"{world_size}-rank data-parallel calibration with one CPU-offloaded "
            "decoder layer per GPU and all_reduce(sum(local X^T X))"
        )
    else:
        model.to(device)
        factors = distributed_whitening_cholesky(
            model,
            windows,
            seqlen=SEQUENCE_LENGTH,
            device=device,
        )
        execution_strategy = "data-parallel full-model replicas"
    if rank == 0:
        output_dir.mkdir(parents=True)
        artifact_path = output_dir / "whitening.safetensors"
        tensors = {
            f"layer_{layer_index:03d}": factor.contiguous()
            for layer_index, factor in enumerate(factors)
        }
        _atomic_safetensors(artifact_path, tensors)
        manifest = {
            "format": WHITENING_FORMAT,
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_metadata,
            "windows": {
                "path": str(windows_path),
                "sha256": _sha256(windows_path),
                "manifest_sha256": _sha256(windows_path.parent / "manifest.json"),
                "format": windows_manifest["format"],
            },
            "samples": args.samples,
            "sequence_length": SEQUENCE_LENGTH,
            "world_size": world_size,
            "execution_strategy": execution_strategy,
            "model_dtype": "torch.bfloat16",
            "accumulator_dtype": "torch.float32",
            "cholesky_dtype": "torch.float64_to_float32",
            "aggregation": "all_reduce(sum(local X^T X)) before Cholesky",
            "artifact": {
                "file": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "layers": NUM_LAYERS,
                "shape_per_layer": [HIDDEN_SIZE, HIDDEN_SIZE],
            },
            "elapsed_seconds": time.perf_counter() - started,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
        }
        _atomic_json(output_dir / "manifest.json", manifest)
        print(f"[Whitening] wrote {artifact_path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def _load_whitening(
    directory: Path, model_metadata: Mapping[str, Any]
) -> tuple[list[Tensor], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WHITENING_FORMAT:
        raise ValueError("incompatible whitening artifact")
    immutable_model_keys = (
        "huggingface_repo",
        "revision",
        "config_sha256",
        "safetensors_index_sha256",
    )
    if any(
        manifest.get("model", {}).get(key) != model_metadata.get(key)
        for key in immutable_model_keys
    ):
        raise ValueError("whitening artifact belongs to another base-model snapshot")
    artifact_path = directory / manifest["artifact"]["file"]
    if _sha256(artifact_path) != manifest["artifact"]["sha256"]:
        raise ValueError("whitening artifact hash changed")
    payload = load_file(str(artifact_path), device="cpu")
    factors = []
    for layer_index in range(NUM_LAYERS):
        factor = payload[f"layer_{layer_index:03d}"]
        if tuple(factor.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
            raise ValueError(f"invalid whitening shape at layer {layer_index}")
        factors.append(factor)
    return factors, manifest


def _load_indexed_tensor(model_path: Path, tensor_name: str) -> Tensor:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"].get(tensor_name)
    if shard_name is None:
        raise KeyError(f"tensor absent from model index: {tensor_name}")
    with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


@torch.no_grad()
def factorize_v_projection(
    weight: Tensor,
    cholesky: Tensor,
    *,
    ranks: Sequence[int],
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Return concatenated writers and stacked decoders using PaLU whitening."""

    if weight.ndim != 2 or weight.shape[1] != cholesky.shape[0]:
        raise ValueError("weight and whitening dimensions do not agree")
    if weight.shape[0] % len(ranks):
        raise ValueError("V output width must divide evenly across PaLU groups")
    linear = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=False,
        device="cpu",
        dtype=weight.dtype,
    )
    linear.weight.copy_(weight)
    linear.scaling_diag_matrix = cholesky
    replacement = HeadwiseLowRankModule.from_linear_whiten(
        linear,
        list(ranks),
        shared_basis_rank=0,
        latent_recovery_ranks=[0] * len(ranks),
        preserve_budget=False,
        factorization="svd",
        factorization_work_device="cpu",
        factorization_work_dtype="float64",
    )
    writer = replacement.VT.weight.detach().to(output_dtype).cpu().contiguous()
    decoder = torch.stack(
        [module.weight.detach() for module in replacement.U]
    ).to(output_dtype).cpu().contiguous()

    group_dim = int(weight.shape[0]) // len(ranks)
    offset = 0
    fro_error_sq = 0.0
    fro_reference_sq = 0.0
    weighted_error_sq = 0.0
    weighted_reference_sq = 0.0
    scale = cholesky.to(torch.float64)
    weight64 = weight.to(torch.float64)
    for group_index, rank in enumerate(ranks):
        dense = weight64[group_index * group_dim : (group_index + 1) * group_dim]
        reconstructed = (
            decoder[group_index].to(torch.float64)
            @ writer[offset : offset + rank].to(torch.float64)
        )
        residual = dense - reconstructed
        fro_error_sq += float(residual.square().sum())
        fro_reference_sq += float(dense.square().sum())
        weighted_error_sq += float((residual @ scale).square().sum())
        weighted_reference_sq += float((dense @ scale).square().sum())
        offset += rank
    diagnostics = {
        "relative_frobenius_error": (fro_error_sq / fro_reference_sq) ** 0.5,
        "relative_activation_weighted_error": (
            weighted_error_sq / weighted_reference_sq
        ) ** 0.5,
    }
    return writer, decoder, diagnostics


@torch.no_grad()
def _build(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    model_metadata = _model_metadata(model_path)
    whitening_dir = Path(args.whitening_dir).expanduser().resolve()
    whitening, whitening_manifest = _load_whitening(whitening_dir, model_metadata)

    rank_per_kv_head = int(args.rank_per_kv_head)
    if not 1 <= rank_per_kv_head <= HEAD_DIM:
        raise ValueError(
            f"rank per KV head must be in [1, {HEAD_DIM}], got {rank_per_kv_head}"
        )
    ranks = [rank_per_kv_head] * NUM_KV_HEADS
    payload: dict[str, Tensor] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_index in range(NUM_LAYERS):
        layer_started = time.perf_counter()
        tensor_name = f"model.layers.{layer_index}.self_attn.v_proj.weight"
        dense = _load_indexed_tensor(model_path, tensor_name)
        if tuple(dense.shape) != (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE):
            raise ValueError(f"unexpected v_proj shape at layer {layer_index}: {dense.shape}")
        writer, decoder, diagnostics = factorize_v_projection(
            dense,
            whitening[layer_index],
            ranks=ranks,
            output_dtype=torch.bfloat16,
        )
        payload[f"layers.{layer_index}.v_writer.weight"] = writer
        payload[f"layers.{layer_index}.v_decoder.weight"] = decoder
        record = {
            "layer": layer_index,
            "source_tensor": tensor_name,
            "source_dtype": str(dense.dtype),
            "writer_shape": list(writer.shape),
            "decoder_shape": list(decoder.shape),
            "factor_dtype": str(writer.dtype),
            **diagnostics,
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        records.append(record)
        print(
            f"[PaLU-M] layer={layer_index}/{NUM_LAYERS - 1} "
            f"weighted_error={diagnostics['relative_activation_weighted_error']:.6f} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "palu_m_v_factors.safetensors"
    _atomic_safetensors(artifact_path, payload)
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": {
            "target": "value_cache_only",
            "key_cache": "dense",
            "method": "PaLU M-LRD activation-aware whitened SVD",
            "allocation": "uniform",
            "num_query_heads": NUM_QUERY_HEADS,
            "num_physical_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "groups": NUM_KV_HEADS,
            "group_width": HEAD_DIM,
            "ranks": ranks,
            "rank_sum": sum(ranks),
            "retained_v_ratio": sum(ranks) / (NUM_KV_HEADS * HEAD_DIM),
            "v_cache_compression_ratio": 1.0 - sum(ranks) / (NUM_KV_HEADS * HEAD_DIM),
            "factorization_work_device": "cpu",
            "factorization_work_dtype": "torch.float64",
            "stored_factor_dtype": "torch.bfloat16",
        },
        "calibration": {
            "dataset": "allenai/c4",
            "samples": int(whitening_manifest["samples"]),
            "sequence_length": int(whitening_manifest["sequence_length"]),
            "whitening_manifest": str(whitening_dir / "manifest.json"),
            "whitening_manifest_sha256": _sha256(whitening_dir / "manifest.json"),
            "whitening_artifact_sha256": whitening_manifest["artifact"]["sha256"],
            "windows": whitening_manifest["windows"],
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensor_count": len(payload),
            "bytes": artifact_path.stat().st_size,
        },
        "layers": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"[PaLU-M] wrote checkpoint {artifact_path}", flush=True)


def _add_common_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Build the matched V-only PaLU-M checkpoint for {MODEL_LABEL}: "
            f"{NUM_KV_HEADS} physical KV groups x rank {RANK_PER_KV_HEAD}."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    windows = subparsers.add_parser("prepare-windows")
    _add_common_model(windows)
    windows.add_argument("--dataset-revision", default="main")
    windows.add_argument("--dataset-split", default="train")
    windows.add_argument("--dataset-cache", default=None)
    windows.add_argument("--samples", type=int, default=CALIBRATION_SAMPLES)
    windows.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    windows.add_argument("--seed", type=int, default=20260821)
    windows.add_argument("--shuffle-buffer", type=int, default=10_000)

    whitening = subparsers.add_parser("prepare-whitening")
    _add_common_model(whitening)
    whitening.add_argument("--windows", required=True)
    whitening.add_argument("--samples", type=int, default=CALIBRATION_SAMPLES)
    whitening.add_argument("--torch-num-threads", type=int, default=4)
    whitening.add_argument("--sequential-cpu-offload", action="store_true")

    build = subparsers.add_parser("build")
    _add_common_model(build)
    build.add_argument("--whitening-dir", required=True)
    build.add_argument(
        "--rank-per-kv-head",
        type=int,
        default=RANK_PER_KV_HEAD,
        help=(
            "uniform rank for each physical KV head "
            f"(default: {RANK_PER_KV_HEAD}; valid range: 1..{HEAD_DIM})"
        ),
    )
    build.add_argument("--torch-num-threads", type=int, default=16)

    args = parser.parse_args()
    if args.command == "prepare-windows":
        if args.samples <= 0:
            raise ValueError("calibration sample count must be positive")
        if args.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(f"this checkpoint protocol requires sequence length {SEQUENCE_LENGTH}")
        if args.shuffle_buffer < args.samples:
            raise ValueError("shuffle buffer must be at least the sample count")
    if args.command == "prepare-whitening" and args.samples <= 0:
        raise ValueError("calibration sample count must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare-windows":
        _prepare_windows(args)
    elif args.command == "prepare-whitening":
        _prepare_whitening(args)
    elif args.command == "build":
        _build(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
