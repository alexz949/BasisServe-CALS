#!/usr/bin/env python3
"""Prepare and evaluate the matched-budget Llama-2-7B MHA V25 study.

All compressed arms keep K dense and retain 75% of the V-cache coordinates:

* ``palu_m``: 32 independent rank-96 PaLU Value groups;
* ``palu_g4``: eight rank-384 four-head PaLU Value groups; and
* ``palu_j``: one rank-3072 joint PaLU Value group spanning all 32 heads;
* ``c1_joint``: 32 equal-rank Value writers jointly optimized with Wo decoders;
  and
* ``recalkv_global_ovc_reference``: one global rank-3072 Value factor that
  reconstructs dense V before stock MHA (oracle/reference only).

The C1 factors are installed in a padded Hugging Face reference path: the
first 96 coordinates of every 128-wide Value head contain the folded latent
writer and the remaining coordinates are zero.  The matching Wo block reads
only those 96 coordinates.  This executes the same floating-point operation
order as a rank-96 cache while retaining the stock no-cache attention kernel.
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

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.fit_llama2_mha_c1_joint import (  # noqa: E402
    FORMAT as C1_FORMAT,
)
from evaluation.fit_llama2_mha_recalkv_global_ovc import (  # noqa: E402
    FORMAT as RECALKV_FORMAT,
)
from evaluation.fit_llama2_mha_recalkv_paper_controlled_ovc import (  # noqa: E402
    FORMAT as RECALKV_CONTROLLED_FORMAT,
)
from evaluation.fit_llama2_mha_recalkv_glrd_ovc import (  # noqa: E402
    FORMAT as RECALKV_GLRD_FORMAT,
)
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    _all_reduce_sum,
    _model_layers,
    distributed_whitening_cholesky,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from palu.model.modules.svd_linear import HeadwiseLowRankModule  # noqa: E402


FORMAT = "basisserve.llama2_7b.mha_v25_matched_comparison.v1"
WHITENING_FORMAT = "basisserve.llama2_7b.mha_v25_c4_whitening.v1"
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 4096


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


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _validate_model_config(config: Any) -> None:
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(getattr(config, "head_dim", HEAD_DIM)),
    )
    expected = ("llama", NUM_LAYERS, HIDDEN_SIZE, NUM_HEADS, NUM_HEADS, HEAD_DIM)
    if observed != expected:
        raise ValueError(f"expected Llama-2-7B MHA, found {observed}")


def _distributed_identity() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("prepare-whitening must be launched with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    return dist.get_rank(), local_rank, dist.get_world_size()


def _load_fit_windows(
    path: Path,
    world_size: int,
    rank: int,
    *,
    fit_windows: int,
) -> tuple[list[dict[str, Tensor]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load_file(str(path), device="cpu")
    input_ids = payload["input_ids"].to(torch.long)
    split_codes = payload.get("split_codes")
    if split_codes is not None:
        indices = torch.nonzero(
            split_codes.to(torch.long) == 0, as_tuple=False
        ).flatten()
        input_ids = input_ids.index_select(0, indices)
    if input_ids.ndim != 2 or int(input_ids.shape[1]) != 2048:
        raise ValueError("whitening windows must have shape [N, 2048]")
    if fit_windows <= 0 or fit_windows > int(input_ids.shape[0]):
        raise ValueError("requested fit windows exceed the available fit split")
    input_ids = input_ids[:fit_windows].contiguous()
    if fit_windows % world_size:
        raise ValueError("fit windows must divide across torchrun workers")
    local = input_ids[rank::world_size].contiguous()
    windows = [
        {
            "input_ids": row.unsqueeze(0),
            "attention_mask": torch.ones_like(row).unsqueeze(0),
        }
        for row in local
    ]
    manifest_path = path.parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
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
    windows_path = Path(args.windows).expanduser().resolve()
    windows, windows_manifest = _load_fit_windows(
        windows_path,
        world_size,
        rank,
        fit_windows=args.fit_windows,
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    del tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    _validate_model_config(model.config)
    model.config.use_cache = False
    factors = distributed_whitening_cholesky(
        model,
        windows,
        seqlen=2048,
        device=device,
    )
    if rank == 0:
        output_dir.mkdir(parents=True)
        artifact_path = output_dir / "whitening.safetensors"
        tensors = {
            f"layer_{layer:03d}": factor.contiguous()
            for layer, factor in enumerate(factors)
        }
        _atomic_safetensors(artifact_path, tensors)
        result = {
            "format": WHITENING_FORMAT,
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(model_path),
            "model_config_sha256": _sha256(model_path / "config.json"),
            "windows": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_format": windows_manifest.get("format"),
            "fit_windows": args.fit_windows,
            "sequence_length": 2048,
            "world_size": world_size,
            "aggregation": "all_reduce(sum(local X^T X)) before FP64 Cholesky",
            "artifact": {
                "file": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "dtype": "torch.float32",
                "layers": NUM_LAYERS,
                "shape_per_layer": [HIDDEN_SIZE, HIDDEN_SIZE],
            },
            "elapsed_seconds": time.perf_counter() - started,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
            },
        }
        _atomic_json(output_dir / "manifest.json", result)
        print(f"[Whitening] wrote {artifact_path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def _load_whitening(directory: Path, model_path: Path) -> tuple[list[Tensor], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WHITENING_FORMAT:
        raise ValueError("incompatible whitening artifact")
    if manifest.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("whitening artifact belongs to another model")
    artifact = directory / manifest["artifact"]["file"]
    if _sha256(artifact) != manifest["artifact"]["sha256"]:
        raise ValueError("whitening artifact hash changed")
    payload = load_file(str(artifact), device="cpu")
    factors = []
    for layer in range(NUM_LAYERS):
        factor = payload[f"layer_{layer:03d}"]
        if tuple(factor.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
            raise ValueError(f"invalid whitening shape at layer {layer}")
        factors.append(factor)
    return factors, manifest


def _palu_geometry(arm: str) -> tuple[int, int]:
    if arm == "palu_m":
        return NUM_HEADS, 96
    if arm == "palu_g4":
        return NUM_HEADS // 4, 384
    if arm == "palu_j":
        return 1, 3072
    raise ValueError(f"unknown PaLU arm: {arm}")


@torch.no_grad()
def _install_palu(
    model: nn.Module,
    *,
    arm: str,
    whitening: Sequence[Tensor],
) -> list[dict[str, Any]]:
    group_count, rank_per_group = _palu_geometry(arm)
    records = []
    for layer_index, layer in enumerate(_model_layers(model)):
        started = time.perf_counter()
        dense = layer.self_attn.v_proj
        if not isinstance(dense, nn.Linear) or dense.bias is not None:
            raise TypeError(f"unsupported dense v_proj at layer {layer_index}")
        dense.scaling_diag_matrix = whitening[layer_index]
        replacement = HeadwiseLowRankModule.from_linear_whiten(
            dense,
            [rank_per_group] * group_count,
            shared_basis_rank=0,
            latent_recovery_ranks=[0] * group_count,
            preserve_budget=False,
            factorization="svd",
            factorization_work_device="cpu",
            factorization_work_dtype="float64",
        )
        layer.self_attn.v_proj = replacement
        records.append(
            {
                "layer": layer_index,
                "groups": group_count,
                "rank_per_group": rank_per_group,
                "rank_sum": group_count * rank_per_group,
                "work_device": replacement.factorization_work_device,
                "work_dtype": replacement.factorization_work_dtype,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del dense
        torch.cuda.empty_cache()
        print(
            f"[Install] arm={arm} layer={layer_index}/{NUM_LAYERS - 1} "
            f"seconds={records[-1]['elapsed_seconds']:.2f}",
            flush=True,
        )
    return records


@torch.no_grad()
def _fold_c1_weights(
    dense_v_weight: Tensor,
    dense_o_weight: Tensor,
    encoders: Tensor,
    decoders: Tensor,
    *,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
) -> tuple[Tensor, Tensor]:
    """Materialize a padded but operation-order-faithful folded C1 path."""

    hidden_size = num_heads * head_dim
    if tuple(dense_v_weight.shape) != (hidden_size, hidden_size):
        raise ValueError("dense V weight does not match the requested head geometry")
    if tuple(dense_o_weight.shape) != (hidden_size, hidden_size):
        raise ValueError("dense O weight does not match the requested head geometry")
    if encoders.ndim != 3 or tuple(encoders.shape[:2]) != (num_heads, head_dim):
        raise ValueError("C1 encoder geometry differs from Llama MHA")
    rank = int(encoders.shape[2])
    if tuple(decoders.shape) != (num_heads, rank, hidden_size):
        raise ValueError("C1 decoder geometry differs from encoders")
    device = dense_v_weight.device
    work_v = dense_v_weight.float()
    work_A = encoders.to(device=device, dtype=torch.float32)
    work_D = decoders.to(device=device, dtype=torch.float32)
    folded_v = torch.zeros_like(work_v)
    folded_o = torch.zeros_like(dense_o_weight, dtype=torch.float32)
    for head in range(num_heads):
        start = head * head_dim
        dense_v_head = work_v[start : start + head_dim]
        writer = work_A[head].transpose(0, 1) @ dense_v_head
        folded_v[start : start + rank].copy_(writer)
        folded_o[:, start : start + rank].copy_(work_D[head].transpose(0, 1))
    return folded_v.contiguous(), folded_o.contiguous()


@torch.no_grad()
def _install_c1(model: nn.Module, factor_dir: Path, model_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_path = factor_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    results = json.loads(result_path.read_text(encoding="utf-8"))
    if results.get("format") != C1_FORMAT or results.get("status") != "complete":
        raise ValueError("C1 factor result is incomplete or incompatible")
    config = results["fit_config"]
    if config.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("C1 factors belong to another model")
    cache_rank = int(config.get("cache_rank_per_head", -1))
    if not 0 < cache_rank <= HEAD_DIM:
        raise ValueError("C1 cache rank must lie within the MHA head width")
    records = []
    for layer_index, layer in enumerate(_model_layers(model)):
        started = time.perf_counter()
        artifact = results["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"C1 artifact hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {"value_coordinate_encoders", "head_output_decoders"}:
            raise ValueError(f"unexpected C1 tensors at layer {layer_index}")
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        if (
            not isinstance(v_proj, nn.Linear)
            or not isinstance(o_proj, nn.Linear)
            or v_proj.bias is not None
            or o_proj.bias is not None
        ):
            raise TypeError(f"unsupported Llama projections at layer {layer_index}")
        folded_v, folded_o = _fold_c1_weights(
            v_proj.weight.detach(),
            o_proj.weight.detach(),
            payload["value_coordinate_encoders"],
            payload["head_output_decoders"],
        )
        v_proj.weight.copy_(folded_v.to(dtype=v_proj.weight.dtype))
        o_proj.weight.copy_(folded_o.to(dtype=o_proj.weight.dtype))
        records.append(
            {
                "layer": layer_index,
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "encoder_shape": list(payload["value_coordinate_encoders"].shape),
                "decoder_shape": list(payload["head_output_decoders"].shape),
                "factor_dtype": str(payload["head_output_decoders"].dtype),
                "cache_rank_per_head": cache_rank,
                "runtime": "padded folded V writer plus Wo decoder",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, folded_v, folded_o
        torch.cuda.empty_cache()
        print(f"[Install] arm=c1_joint layer={layer_index}/{NUM_LAYERS - 1}", flush=True)
    return records, results


class _DenseReconstructionLinear(nn.Module):
    """Low-rank writer followed by an explicit dense-output reconstruction."""

    def __init__(self, encoder_weight: Tensor, reconstruction_weight: Tensor) -> None:
        super().__init__()
        if encoder_weight.ndim != 2 or reconstruction_weight.ndim != 2:
            raise ValueError("ReCalKV factors must be matrices")
        rank, in_features = map(int, encoder_weight.shape)
        out_features, decoder_rank = map(int, reconstruction_weight.shape)
        if rank != decoder_rank:
            raise ValueError("ReCalKV encoder and reconstruction ranks differ")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.encoder = nn.Linear(
            in_features,
            rank,
            bias=False,
            device=encoder_weight.device,
            dtype=encoder_weight.dtype,
        )
        self.reconstruction = nn.Linear(
            rank,
            out_features,
            bias=False,
            device=reconstruction_weight.device,
            dtype=reconstruction_weight.dtype,
        )
        with torch.no_grad():
            self.encoder.weight.copy_(encoder_weight)
            self.reconstruction.weight.copy_(reconstruction_weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.reconstruction(self.encoder(hidden_states))


def _replace_v_with_dense_reconstruction(
    attention: nn.Module,
    encoder_weight: Tensor,
    reconstruction_weight: Tensor,
) -> _DenseReconstructionLinear:
    dense = attention.v_proj
    if not isinstance(dense, nn.Linear) or dense.bias is not None:
        raise TypeError("ReCalKV reference requires a bias-free dense v_proj")
    expected_encoder = (int(encoder_weight.shape[0]), dense.in_features)
    expected_decoder = (dense.out_features, int(encoder_weight.shape[0]))
    if tuple(encoder_weight.shape) != expected_encoder:
        raise ValueError("ReCalKV encoder does not match v_proj input width")
    if tuple(reconstruction_weight.shape) != expected_decoder:
        raise ValueError("ReCalKV reconstruction does not match v_proj output width")
    device, dtype = dense.weight.device, dense.weight.dtype
    replacement = _DenseReconstructionLinear(
        encoder_weight.to(device=device, dtype=dtype),
        reconstruction_weight.to(device=device, dtype=dtype),
    )
    attention.v_proj = replacement
    return replacement


class _GroupedDenseReconstructionLinear(nn.Module):
    """Group writers followed by explicit dense Value reconstruction."""

    def __init__(self, encoder_weight: Tensor, reconstruction_weight: Tensor) -> None:
        super().__init__()
        if encoder_weight.ndim != 3 or reconstruction_weight.ndim != 3:
            raise ValueError("G-LRD OVC factors must be rank-three tensors")
        groups, rank, in_features = map(int, encoder_weight.shape)
        decoder_groups, group_dim, decoder_rank = map(
            int, reconstruction_weight.shape
        )
        if groups != decoder_groups or rank != decoder_rank:
            raise ValueError("G-LRD OVC encoder and reconstruction geometry differs")
        self.in_features = in_features
        self.out_features = groups * group_dim
        self.groups = groups
        self.group_dim = group_dim
        self.rank_per_group = rank
        self.encoder = nn.Linear(
            in_features,
            groups * rank,
            bias=False,
            device=encoder_weight.device,
            dtype=encoder_weight.dtype,
        )
        self.reconstruction_weight = nn.Parameter(
            reconstruction_weight.detach().clone(), requires_grad=False
        )
        with torch.no_grad():
            self.encoder.weight.copy_(
                encoder_weight.reshape(groups * rank, in_features)
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        latent = self.encoder(hidden_states).unflatten(
            -1, (self.groups, self.rank_per_group)
        )
        reconstructed = torch.einsum(
            "...gr,gdr->...gd", latent, self.reconstruction_weight
        )
        return reconstructed.flatten(-2)


def _replace_v_with_grouped_dense_reconstruction(
    attention: nn.Module,
    encoder_weight: Tensor,
    reconstruction_weight: Tensor,
) -> _GroupedDenseReconstructionLinear:
    dense = attention.v_proj
    if not isinstance(dense, nn.Linear) or dense.bias is not None:
        raise TypeError("G-LRD OVC reference requires a bias-free dense v_proj")
    groups, rank, in_features = map(int, encoder_weight.shape)
    expected_decoder = (groups, dense.out_features // groups, rank)
    if in_features != dense.in_features:
        raise ValueError("G-LRD OVC encoder does not match v_proj input width")
    if (
        dense.out_features % groups
        or tuple(reconstruction_weight.shape) != expected_decoder
    ):
        raise ValueError("G-LRD OVC reconstruction does not match v_proj output width")
    device, dtype = dense.weight.device, dense.weight.dtype
    replacement = _GroupedDenseReconstructionLinear(
        encoder_weight.to(device=device, dtype=dtype),
        reconstruction_weight.to(device=device, dtype=dtype),
    )
    attention.v_proj = replacement
    return replacement


def _validate_recalkv_results(
    results: Mapping[str, Any],
    *,
    model_config_sha256: str,
    expected_format: str = RECALKV_FORMAT,
) -> int:
    if results.get("format") != expected_format or results.get("status") != "complete":
        raise ValueError("ReCalKV factor result is incomplete or incompatible")
    config = results.get("fit_config", {})
    if config.get("model_config_sha256") != model_config_sha256:
        raise ValueError("ReCalKV factors belong to another model")
    rank = int(config.get("value_rank", -1))
    if not 0 < rank <= HIDDEN_SIZE:
        raise ValueError("ReCalKV global Value rank is invalid")
    if config.get("key_projection") != "dense and unchanged":
        raise ValueError("ReCalKV reference must keep K dense")
    if config.get("value_factorization_scope") != "one global/full-layer W_V matrix":
        raise ValueError("ReCalKV reference must use one global Value factor")
    if config.get("dense_reconstruction_before_mha") is not True:
        raise ValueError("ReCalKV reference must reconstruct dense V before MHA")
    return rank


def _validate_recalkv_glrd_results(
    results: Mapping[str, Any], *, model_config_sha256: str
) -> tuple[int, int, int]:
    if (
        results.get("format") != RECALKV_GLRD_FORMAT
        or results.get("status") != "complete"
    ):
        raise ValueError("ReCalKV G-LRD factor result is incomplete or incompatible")
    config = results.get("fit_config", {})
    if config.get("model_config_sha256") != model_config_sha256:
        raise ValueError("ReCalKV G-LRD factors belong to another model")
    groups = int(config.get("group_count", -1))
    group_size = int(config.get("group_size", -1))
    rank = int(config.get("rank_per_group", -1))
    if groups * group_size != NUM_HEADS or not 0 < rank <= group_size * HEAD_DIM:
        raise ValueError("ReCalKV G-LRD Value geometry is invalid")
    if config.get("key_projection") != "dense and unchanged":
        raise ValueError("ReCalKV G-LRD reference must keep K dense")
    if config.get("value_factorization_scope") != "contiguous G-LRD head groups":
        raise ValueError("ReCalKV G-LRD reference has an incompatible scope")
    if config.get("fusion_compatible") is not True:
        raise ValueError("ReCalKV G-LRD reference must be fusion-compatible")
    if (
        config.get("dense_reconstruction_before_mha_for_quality_evaluation")
        is not True
    ):
        raise ValueError("ReCalKV G-LRD quality reference must reconstruct dense V")
    return groups, group_size, rank


@torch.no_grad()
def _install_recalkv_reference(
    model: nn.Module,
    factor_dir: Path,
    model_path: Path,
    *,
    expected_format: str = RECALKV_FORMAT,
    arm: str = "recalkv_global_ovc_reference",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_path = factor_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    results = json.loads(result_path.read_text(encoding="utf-8"))
    rank = _validate_recalkv_results(
        results,
        model_config_sha256=_sha256(model_path / "config.json"),
        expected_format=expected_format,
    )
    if tuple(map(int, results.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError("ReCalKV reference must cover all Llama layers")
    records = []
    for layer_index, layer in enumerate(_model_layers(model)):
        started = time.perf_counter()
        artifact = results["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"ReCalKV artifact hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {"v_encoder_weight", "v_reconstruction_weight"}:
            raise ValueError(f"unexpected ReCalKV tensors at layer {layer_index}")
        encoder = payload["v_encoder_weight"]
        reconstruction = payload["v_reconstruction_weight"]
        if tuple(encoder.shape) != (rank, HIDDEN_SIZE):
            raise ValueError(f"invalid ReCalKV encoder at layer {layer_index}")
        if tuple(reconstruction.shape) != (HIDDEN_SIZE, rank):
            raise ValueError(f"invalid ReCalKV reconstruction at layer {layer_index}")
        key_projection = layer.self_attn.k_proj
        replacement = _replace_v_with_dense_reconstruction(
            layer.self_attn, encoder, reconstruction
        )
        if layer.self_attn.k_proj is not key_projection:
            raise RuntimeError("ReCalKV installation changed dense K")
        records.append(
            {
                "layer": layer_index,
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "encoder_shape": list(encoder.shape),
                "reconstruction_shape": list(reconstruction.shape),
                "factor_dtype": str(encoder.dtype),
                "value_rank": rank,
                "key_projection": "original dense module unchanged",
                "runtime": "rank writer then explicit dense V reconstruction before MHA",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, encoder, reconstruction, replacement
        torch.cuda.empty_cache()
        print(
            f"[Install] arm={arm} "
            f"layer={layer_index}/{NUM_LAYERS - 1}",
            flush=True,
        )
    return records, results


@torch.no_grad()
def _install_recalkv_glrd_reference(
    model: nn.Module, factor_dir: Path, model_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_path = factor_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    results = json.loads(result_path.read_text(encoding="utf-8"))
    groups, group_size, rank = _validate_recalkv_glrd_results(
        results, model_config_sha256=_sha256(model_path / "config.json")
    )
    if tuple(map(int, results.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError("ReCalKV G-LRD reference must cover all Llama layers")
    records = []
    for layer_index, layer in enumerate(_model_layers(model)):
        started = time.perf_counter()
        artifact = results["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(
                f"ReCalKV G-LRD artifact hash mismatch at layer {layer_index}"
            )
        payload = load_file(str(path), device="cpu")
        expected_names = {
            "v_group_encoder_weight",
            "v_group_reconstruction_weight",
        }
        if set(payload) != expected_names:
            raise ValueError(
                f"unexpected ReCalKV G-LRD tensors at layer {layer_index}"
            )
        encoder = payload["v_group_encoder_weight"]
        reconstruction = payload["v_group_reconstruction_weight"]
        expected_encoder = (groups, rank, HIDDEN_SIZE)
        expected_reconstruction = (groups, group_size * HEAD_DIM, rank)
        if tuple(encoder.shape) != expected_encoder:
            raise ValueError(f"invalid G-LRD encoder at layer {layer_index}")
        if tuple(reconstruction.shape) != expected_reconstruction:
            raise ValueError(f"invalid G-LRD reconstruction at layer {layer_index}")
        key_projection = layer.self_attn.k_proj
        replacement = _replace_v_with_grouped_dense_reconstruction(
            layer.self_attn, encoder, reconstruction
        )
        if layer.self_attn.k_proj is not key_projection:
            raise RuntimeError("ReCalKV G-LRD installation changed dense K")
        records.append(
            {
                "layer": layer_index,
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "encoder_shape": list(encoder.shape),
                "reconstruction_shape": list(reconstruction.shape),
                "factor_dtype": str(encoder.dtype),
                "group_count": groups,
                "group_size": group_size,
                "rank_per_group": rank,
                "key_projection": "original dense module unchanged",
                "runtime": (
                    "group writers then explicit dense V reconstruction before MHA; "
                    "quality-equivalent to per-head decoder fusion"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, encoder, reconstruction, replacement
        torch.cuda.empty_cache()
        print(
            f"[Install] arm=recalkv_glrd_ovc_reference "
            f"layer={layer_index}/{NUM_LAYERS - 1}",
            flush=True,
        )
    return records, results


def _cache_accounting(
    arm: str,
    *,
    c1_rank_per_head: int = 96,
    recalkv_value_rank: int = 3072,
    recalkv_glrd_value_rank: int = 3072,
) -> dict[str, Any]:
    if arm == "dense":
        v_rank = NUM_HEADS * HEAD_DIM
    elif arm == "c1_joint":
        if not 0 < c1_rank_per_head <= HEAD_DIM:
            raise ValueError("C1 cache rank must lie within the MHA head width")
        v_rank = NUM_HEADS * c1_rank_per_head
    elif arm in (
        "recalkv_global_ovc_reference",
        "recalkv_paper_controlled_ovc_reference",
    ):
        if not 0 < recalkv_value_rank <= HIDDEN_SIZE:
            raise ValueError("ReCalKV global Value rank is invalid")
        v_rank = recalkv_value_rank
    elif arm == "recalkv_glrd_ovc_reference":
        if not 0 < recalkv_glrd_value_rank <= HIDDEN_SIZE:
            raise ValueError("ReCalKV G-LRD total Value rank is invalid")
        v_rank = recalkv_glrd_value_rank
    else:
        v_rank = NUM_HEADS * 96
    k_rank = NUM_HEADS * HEAD_DIM
    return {
        "key_cache_rank_per_layer": k_rank,
        "value_cache_rank_per_layer": v_rank,
        "total_kv_rank_per_layer": k_rank + v_rank,
        "dense_total_kv_rank_per_layer": 2 * NUM_HEADS * HEAD_DIM,
        "value_retained_ratio": v_rank / (NUM_HEADS * HEAD_DIM),
        "total_kv_retained_ratio": (k_rank + v_rank) / (2 * NUM_HEADS * HEAD_DIM),
        "total_kv_reduction_fraction": 1.0
        - (k_rank + v_rank) / (2 * NUM_HEADS * HEAD_DIM),
        "tp_size": 8,
        "value_latent_elements_per_tp_rank": v_rank // 8,
    }


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("comparison evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.arm.startswith("palu_") and args.whitening_dir is None:
        raise ValueError("PaLU arms require --whitening-dir")
    if args.arm == "c1_joint" and args.factor_dir is None:
        raise ValueError("C1 arm requires --factor-dir")
    if args.arm in (
        "recalkv_global_ovc_reference",
        "recalkv_paper_controlled_ovc_reference",
    ) and args.factor_dir is None:
        raise ValueError("ReCalKV reference arm requires --factor-dir")
    if args.arm == "recalkv_glrd_ovc_reference" and args.factor_dir is None:
        raise ValueError("ReCalKV G-LRD reference arm requires --factor-dir")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    _validate_model_config(model.config)
    model.config.use_cache = False
    install_started = time.perf_counter()
    provenance: dict[str, Any] = {}
    c1_rank_per_head = 96
    recalkv_value_rank = 3072
    recalkv_glrd_value_rank = 3072
    if args.arm.startswith("palu_"):
        whitening, whitening_manifest = _load_whitening(
            Path(args.whitening_dir).expanduser().resolve(), model_path
        )
        install_records = _install_palu(
            model, arm=args.arm, whitening=whitening
        )
        provenance["whitening"] = whitening_manifest
        del whitening
    elif args.arm == "c1_joint":
        install_records, c1_results = _install_c1(
            model,
            Path(args.factor_dir).expanduser().resolve(),
            model_path,
        )
        provenance["c1_fit"] = {
            "path": str(Path(args.factor_dir).expanduser().resolve() / "results.json"),
            "sha256": _sha256(
                Path(args.factor_dir).expanduser().resolve() / "results.json"
            ),
            "aggregate": c1_results["aggregate"],
            "fit_config": c1_results["fit_config"],
        }
        c1_rank_per_head = int(c1_results["fit_config"]["cache_rank_per_head"])
    elif args.arm in (
        "recalkv_global_ovc_reference",
        "recalkv_paper_controlled_ovc_reference",
    ):
        factor_dir = Path(args.factor_dir).expanduser().resolve()
        controlled = args.arm == "recalkv_paper_controlled_ovc_reference"
        install_records, recalkv_results = _install_recalkv_reference(
            model,
            factor_dir,
            model_path,
            expected_format=(
                RECALKV_CONTROLLED_FORMAT if controlled else RECALKV_FORMAT
            ),
            arm=args.arm,
        )
        recalkv_value_rank = int(recalkv_results["fit_config"]["value_rank"])
        provenance["recalkv_reference"] = {
            "path": str(factor_dir / "results.json"),
            "sha256": _sha256(factor_dir / "results.json"),
            "label": recalkv_results.get("label"),
            "aggregate": recalkv_results["aggregate"],
            "fit_config": recalkv_results["fit_config"],
            "dense_reconstruction_oracle_reference": True,
            "deployable_fused_kernel_claim": False,
            "paper_controlled_protocol": controlled,
        }
    elif args.arm == "recalkv_glrd_ovc_reference":
        factor_dir = Path(args.factor_dir).expanduser().resolve()
        install_records, recalkv_glrd_results = _install_recalkv_glrd_reference(
            model, factor_dir, model_path
        )
        glrd_config = recalkv_glrd_results["fit_config"]
        recalkv_glrd_value_rank = int(glrd_config["total_value_rank"])
        provenance["recalkv_glrd_reference"] = {
            "path": str(factor_dir / "results.json"),
            "sha256": _sha256(factor_dir / "results.json"),
            "label": recalkv_glrd_results.get("label"),
            "aggregate": recalkv_glrd_results["aggregate"],
            "fit_config": glrd_config,
            "dense_reconstruction_quality_reference": True,
            "fusion_compatible": True,
            "deployable_fused_kernel_implemented": False,
        }
    else:
        install_records = []
    install_seconds = time.perf_counter() - install_started
    ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    result = {
        "format": FORMAT,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "model": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "model_dtype": "float16",
        "attention_implementation": "sdpa",
        "loss_accumulation_dtype": "float32",
        "compression": _cache_accounting(
            args.arm,
            c1_rank_per_head=c1_rank_per_head,
            recalkv_value_rank=recalkv_value_rank,
            recalkv_glrd_value_rank=recalkv_glrd_value_rank,
        ),
        "installation": {
            "seconds": install_seconds,
            "records": install_records,
        },
        "provenance": provenance,
        "ppl": ppl,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, result)
    print(f"[Result] arm={args.arm} ppl={ppl['ppl']:.9f}", flush=True)
    print(f"[Write] {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    prepare = subparsers.add_parser("prepare-whitening")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--windows", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--fit-windows", type=int, default=64)
    prepare.add_argument("--torch-num-threads", type=int, default=2)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument(
        "--arm",
        choices=(
            "dense",
            "palu_m",
            "palu_g4",
            "palu_j",
            "c1_joint",
            "recalkv_global_ovc_reference",
            "recalkv_paper_controlled_ovc_reference",
            "recalkv_glrd_ovc_reference",
        ),
        required=True,
    )
    evaluate.add_argument("--whitening-dir")
    evaluate.add_argument("--factor-dir")
    evaluate.add_argument("--output-json", required=True)
    evaluate.add_argument("--dataset", default="wikitext2")
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--seqlen", type=int, default=2048)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--max-tokens", type=int)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command_name == "prepare-whitening":
        _prepare_whitening(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
