#!/usr/bin/env python3
"""Evaluate WikiText-2 PPL for activation-aware AR/AG ``o_proj`` factors.

For quality isolation, each factorized collective map is collapsed into its
mathematically equivalent dense weight before evaluation. This avoids timing a
Python low-rank runtime while preserving the intended linear approximation.
Communication and HBM accounting come from the factor manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.o_proj_collective_ppl import (  # noqa: E402
    project_weight_private,
    project_weight_shared,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.fit_attention_o_proj_ppl_factors import (  # noqa: E402
    FORMAT as FACTOR_FORMAT,
    METHOD_KEYS,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import (  # noqa: E402
    _input_device,
    _iter_batches,
    _token_ids,
)


FORMAT = "basisserve.attention_o_proj_collective_ppl.v2"
JOINT_FACTOR_FORMAT = "basisserve.qwen3.ag_joint_decoder_refit.v2"
JOINT_METHOD_TO_VARIANT = {
    "joint_tp4": "tp4_cv_per_layer",
    "joint_tp8": "tp8_physical",
}
SCOPEWIRE_FACTOR_FORMAT = "basisserve.scopewire.mixed_frontier.v1"
SCOPEWIRE_METHOD = "scopewire"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--factor-dir")
    parser.add_argument(
        "--method",
        choices=(
            "dense",
            "ar",
            "tp_ag",
            "head_ag",
            *JOINT_METHOD_TO_VARIANT,
            SCOPEWIRE_METHOD,
        ),
        required=True,
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), required=True
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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


def _load_factor_manifest(
    factor_dir: Path, model_path: Path, method: str
) -> dict[str, Any]:
    manifest_path = factor_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FACTOR_FORMAT:
        raise ValueError("factor artifact has an incompatible format")
    if method not in manifest["fit_config"]["methods"]:
        raise ValueError(f"method {method} is absent from factor artifact")
    if _sha256(model_path / "config.json") != manifest["model"]["config_sha256"]:
        raise ValueError("model config differs from the factor artifact")
    return manifest


def _load_joint_factor_results(
    factor_dir: Path,
    model_path: Path,
    method: str,
) -> dict[str, Any]:
    results_path = factor_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("format") != JOINT_FACTOR_FORMAT:
        raise ValueError("joint factor artifact has an incompatible format")
    variant = JOINT_METHOD_TO_VARIANT[method]
    layers = tuple(map(int, results["config"]["layers"]))
    records = [row for row in results["records"] if row.get("variant") == variant]
    if layers != tuple(range(len(layers))) or {
        int(row["layer"]) for row in records
    } != set(layers):
        raise ValueError("joint factors do not cover contiguous decoder layers")
    source_layer = min(records, key=lambda row: int(row["layer"]))
    basis_artifact = Path(source_layer["sources"]["basis_snapshot"]["path"])
    basis_manifest_path = basis_artifact.parent / "manifest.json"
    if not basis_manifest_path.is_file():
        raise FileNotFoundError(basis_manifest_path)
    if (
        _sha256(basis_manifest_path)
        != results["config"]["basis_snapshot_manifest_sha256"]
    ):
        raise ValueError("joint factor basis-snapshot manifest hash mismatch")
    basis_manifest = json.loads(basis_manifest_path.read_text(encoding="utf-8"))
    if _sha256(model_path / "config.json") != basis_manifest["model"]["config_sha256"]:
        raise ValueError("model config differs from the joint factor artifact")
    results["selected_variant"] = variant
    results["selected_records"] = {
        str(row["layer"]): row for row in records
    }
    results["model"] = basis_manifest["model"]
    return results


def _load_scopewire_results(
    factor_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    results_path = factor_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("format") != SCOPEWIRE_FACTOR_FORMAT:
        raise ValueError("ScopeWire factor artifact has an incompatible format")
    if not bool(results.get("final_audit_deferred")):
        raise ValueError("ScopeWire schedule was not frozen before final audit")
    layers = tuple(map(int, results["layers"]))
    records = {int(row["layer"]): row for row in results["rows"]}
    if layers != tuple(range(len(layers))) or set(records) != set(layers):
        raise ValueError("ScopeWire factors do not cover contiguous decoder layers")
    fit_source = Path(
        results["selection_protocol"]["fit_source"]
    ).expanduser().resolve()
    snapshot_manifest_path = fit_source / "manifest.json"
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError(snapshot_manifest_path)
    snapshot_manifest = json.loads(
        snapshot_manifest_path.read_text(encoding="utf-8")
    )
    if _sha256(model_path / "config.json") != snapshot_manifest["model"][
        "config_sha256"
    ]:
        raise ValueError("model config differs from the ScopeWire factor artifact")
    results["selected_records"] = {str(layer): records[layer] for layer in layers}
    results["model"] = snapshot_manifest["model"]
    return results


@torch.no_grad()
def _eval_ppl_fp32_loss(
    model: nn.Module,
    tokenizer: Any,
    *,
    dataset: str,
    split: str | None,
    seqlen: int,
    batch_size: int,
    max_samples: int | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Evaluate chunked PPL while accumulating cross entropy in float32.

    Model execution keeps its requested BF16/FP16 dtype. Only logits entering
    cross entropy are promoted, preventing large reduced losses from being
    quantized at the model dtype's coarse magnitude-dependent resolution.
    """

    input_ids = _token_ids(tokenizer, dataset, split, max_tokens)
    nsamples_total = int(input_ids.numel() // seqlen)
    nsamples = (
        nsamples_total
        if max_samples is None
        else min(nsamples_total, int(max_samples))
    )
    if nsamples <= 0:
        raise ValueError(
            "no complete evaluation chunks; reduce --seqlen or increase --max-tokens"
        )
    input_device = _input_device(model)
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll_sum = 0.0
    token_count = 0
    print(
        f"[Eval] dataset={dataset} split={split or 'default'} seqlen={seqlen} "
        f"batch_size={batch_size} chunks={nsamples}/{nsamples_total} "
        f"input_device={input_device} loss_dtype=float32",
        flush=True,
    )
    for _, batch in tqdm(
        _iter_batches(input_ids, seqlen, batch_size, max_samples),
        total=math.ceil(nsamples / batch_size),
    ):
        batch = batch.to(input_device)
        logits = model(input_ids=batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = batch[:, 1:].contiguous().to(shift_logits.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nll_sum += float(loss.detach().cpu())
        token_count += int(shift_labels.numel())
    if use_cache is not None:
        model.config.use_cache = use_cache
    return {
        "dataset": dataset,
        "split": split,
        "seqlen": seqlen,
        "batch_size": batch_size,
        "chunks": nsamples,
        "tokens": token_count,
        "nll_sum": nll_sum,
        "loss_dtype": "float32",
        "ppl": math.exp(nll_sum / token_count),
    }


@torch.inference_mode()
def _install_projected_weights(
    model: nn.Module,
    *,
    factor_dir: Path,
    manifest: Mapping[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    decoder_layers = _decoder_layers(model)
    layers = tuple(map(int, manifest["layers"]))
    if layers != tuple(range(len(decoder_layers))):
        raise ValueError("PPL factors must cover every decoder layer")
    key = METHOD_KEYS[method]
    records = []
    for layer in layers:
        started = time.perf_counter()
        record = manifest["artifacts"][str(layer)]
        path = factor_dir / record["file"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"factor hash mismatch at layer {layer}")
        payload = load_file(str(path), device="cpu")
        if key not in payload:
            raise KeyError(f"{key} is absent at layer {layer}")
        module = decoder_layers[layer].self_attn.o_proj
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise TypeError(f"layer {layer} o_proj is unsupported")
        basis = payload[key]
        selected_rank = record.get("selected_rank_per_group")
        if selected_rank is not None:
            if method not in {"tp_ag", "head_ag"}:
                raise ValueError(
                    "selected_rank_per_group is valid only for private AG factors"
                )
            selected_rank = int(selected_rank)
            if not 0 < selected_rank <= int(basis.shape[-1]):
                raise ValueError(
                    f"invalid selected private rank at layer {layer}: "
                    f"{selected_rank} not in [1, {int(basis.shape[-1])}]"
                )
            basis = basis[..., :selected_rank].contiguous()
        if method == "ar":
            projected = project_weight_shared(module.weight.detach(), basis)
        else:
            projected = project_weight_private(module.weight.detach(), basis)
        module.weight.copy_(projected.to(dtype=module.weight.dtype))
        records.append(
            {
                "layer": layer,
                "factor_file": path.name,
                "factor_sha256": record["sha256"],
                "basis_shape": list(basis.shape),
                "basis_dtype": str(basis.dtype),
                "projected_weight_dtype": str(module.weight.dtype),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, basis, projected
        torch.cuda.empty_cache()
        print(
            f"[Install] method={method} layer={layer}/{layers[-1]}", flush=True
        )
    return records


@torch.no_grad()
def _project_weight_joint_private(
    private_encoders: torch.Tensor,
    joint_decoder: torch.Tensor,
    input_indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if private_encoders.ndim != 3 or joint_decoder.ndim != 2:
        raise ValueError("joint private factors have invalid ranks")
    groups, group_width, rank_per_group = map(int, private_encoders.shape)
    total_rank, output_width = map(int, joint_decoder.shape)
    input_width = groups * group_width
    if total_rank != groups * rank_per_group:
        raise ValueError("joint decoder rank differs from private encoders")
    indices = input_indices.to(device=device, dtype=torch.long).flatten()
    if int(indices.numel()) != input_width or not torch.equal(
        torch.sort(indices).values,
        torch.arange(input_width, device=device),
    ):
        raise ValueError("joint factor input indices are not a permutation")
    encoders = private_encoders.to(device=device, dtype=torch.float32)
    decoder = joint_decoder.to(device=device, dtype=torch.float32)
    ordered = torch.empty(
        output_width,
        input_width,
        device=device,
        dtype=torch.float32,
    )
    for group in range(groups):
        input_start = group * group_width
        input_stop = input_start + group_width
        rank_start = group * rank_per_group
        rank_stop = rank_start + rank_per_group
        ordered[:, input_start:input_stop] = (
            decoder[rank_start:rank_stop].transpose(0, 1)
            @ encoders[group].transpose(0, 1)
        )
    projected = torch.empty_like(ordered)
    projected[:, indices] = ordered
    return projected.contiguous()


@torch.no_grad()
def _project_weight_scopewire(
    payload: Mapping[str, torch.Tensor],
    layout: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not layout or "input_indices" not in payload:
        raise ValueError("ScopeWire artifact has no components or input indices")
    indices = payload["input_indices"].to(device=device, dtype=torch.long).flatten()
    input_width = int(indices.numel())
    if not torch.equal(
        torch.sort(indices).values,
        torch.arange(input_width, device=device),
    ):
        raise ValueError("ScopeWire input indices are not a permutation")
    supports = [tuple(map(int, row["support"])) for row in layout]
    source_count = max(max(support) for support in supports) + 1
    if input_width % source_count:
        raise ValueError("ScopeWire input width is not divisible by source count")
    source_width = input_width // source_count
    required_keys = {"input_indices"}
    output_width = None
    ordered = None
    for row, support in zip(layout, supports, strict=True):
        if not support or tuple(sorted(set(support))) != support:
            raise ValueError("ScopeWire component support is invalid")
        encoder_key = str(row["encoder_key"])
        decoder_key = str(row["decoder_key"])
        required_keys.update((encoder_key, decoder_key))
        if encoder_key not in payload or decoder_key not in payload:
            raise KeyError("ScopeWire component tensor is absent")
        encoder = payload[encoder_key].to(device=device, dtype=torch.float32)
        decoder = payload[decoder_key].to(device=device, dtype=torch.float32)
        rank = int(row["rank"])
        if (
            tuple(encoder.shape) != (len(support) * source_width, rank)
            or decoder.ndim != 2
            or int(decoder.shape[0]) != rank
        ):
            raise ValueError("ScopeWire component factor geometry is invalid")
        if output_width is None:
            output_width = int(decoder.shape[1])
            ordered = torch.zeros(
                input_width,
                output_width,
                device=device,
                dtype=torch.float32,
            )
        elif int(decoder.shape[1]) != output_width:
            raise ValueError("ScopeWire component output widths differ")
        assert ordered is not None
        for offset, source in enumerate(support):
            encoder_block = encoder[
                offset * source_width : (offset + 1) * source_width
            ]
            ordered[
                source * source_width : (source + 1) * source_width
            ].add_(encoder_block @ decoder)
    if set(payload) != required_keys or ordered is None or output_width is None:
        raise ValueError("ScopeWire artifact tensors differ from its layout")
    projected = torch.empty(
        output_width,
        input_width,
        device=device,
        dtype=torch.float32,
    )
    projected[:, indices] = ordered.transpose(0, 1)
    return projected.contiguous()


def _scopewire_layer_accounting(
    layout: Sequence[Mapping[str, Any]],
    *,
    hidden_size: int,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    supports = [tuple(map(int, row["support"])) for row in layout]
    if not supports:
        raise ValueError("ScopeWire accounting requires components")
    tp = max(max(support) for support in supports) + 1
    if hidden_size % tp:
        raise ValueError("hidden size is incompatible with ScopeWire TP size")
    source_width = hidden_size // tp
    per_source_encoder_parameters = [0] * tp
    rank_by_support_size: dict[int, int] = {}
    link_units = 0
    decoder_rows = 0
    for row, support in zip(layout, supports, strict=True):
        rank = int(row["rank"])
        support_size = len(support)
        if rank <= 0 or not support or tuple(sorted(set(support))) != support:
            raise ValueError("ScopeWire accounting component is invalid")
        decoder_rows += rank
        link_units += (tp + support_size - 2) * rank
        rank_by_support_size[support_size] = (
            rank_by_support_size.get(support_size, 0) + rank
        )
        for source in support:
            per_source_encoder_parameters[source] += source_width * rank
    return {
        "collective": "mixed_scope",
        "tp_size": tp,
        "dtype_bytes": dtype_bytes,
        "link_units": link_units,
        "ideal_ring_elements_per_rank": link_units / tp,
        "ideal_ring_bytes_per_rank": link_units * dtype_bytes // tp,
        "rank_by_support_size": {
            str(size): rank for size, rank in sorted(rank_by_support_size.items())
        },
        "decoder_rows": decoder_rows,
        "decoder_parameters_per_rank": decoder_rows * hidden_size,
        "decoder_hbm_bytes_per_rank": decoder_rows * hidden_size * dtype_bytes,
        "encoder_parameters_by_source": per_source_encoder_parameters,
        "maximum_encoder_hbm_bytes_per_rank": (
            max(per_source_encoder_parameters) * dtype_bytes
        ),
    }


def _joint_collective_accounting(
    record: Mapping[str, Any],
    *,
    hidden_size: int,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    tp = int(record["tp"])
    groups = int(record["groups"])
    rank_per_group = int(record["rank_per_group"])
    total_rank = int(record["total_rank"])
    if tp != groups or total_rank != groups * rank_per_group:
        raise ValueError("joint factor communication geometry is inconsistent")
    input_width_per_rank = hidden_size // tp
    encoder_parameters_per_rank = input_width_per_rank * rank_per_group
    decoder_parameters = total_rank * hidden_size
    return {
        "collective": "all_gather",
        "logical_groups": groups,
        "tp_size": tp,
        "latent_elements_per_rank": rank_per_group,
        "rank_per_group": rank_per_group,
        "total_rank": total_rank,
        "dtype_bytes": dtype_bytes,
        "ideal_ring_bytes_per_rank": (
            (tp - 1) * total_rank * dtype_bytes // tp
        ),
        "encoder_parameters_per_rank": encoder_parameters_per_rank,
        "encoder_hbm_bytes_per_rank": encoder_parameters_per_rank * dtype_bytes,
        "decoder_parameters_per_rank": decoder_parameters,
        "decoder_hbm_bytes_per_rank": decoder_parameters * dtype_bytes,
    }


@torch.inference_mode()
def _install_joint_projected_weights(
    model: nn.Module,
    *,
    factor_dir: Path,
    results: Mapping[str, Any],
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoder_layers = _decoder_layers(model)
    layers = tuple(map(int, results["config"]["layers"]))
    if layers != tuple(range(len(decoder_layers))):
        raise ValueError("joint PPL factors must cover every decoder layer")
    records_by_layer = results["selected_records"]
    installed = []
    accounting = None
    for layer in layers:
        started = time.perf_counter()
        record = records_by_layer[str(layer)]
        artifact = record["artifact"]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"joint factor hash mismatch at layer {layer}")
        payload = load_file(str(path), device="cpu")
        expected = {"private_encoders", "joint_decoder", "input_indices"}
        if set(payload) != expected:
            raise ValueError(f"joint factor tensors differ at layer {layer}")
        module = decoder_layers[layer].self_attn.o_proj
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise TypeError(f"layer {layer} o_proj is unsupported")
        projected = _project_weight_joint_private(
            payload["private_encoders"],
            payload["joint_decoder"],
            payload["input_indices"],
            device=module.weight.device,
        )
        if tuple(projected.shape) != tuple(module.weight.shape):
            raise ValueError(f"joint projected weight shape differs at layer {layer}")
        module.weight.copy_(projected.to(dtype=module.weight.dtype))
        layer_accounting = _joint_collective_accounting(
            record,
            hidden_size=int(module.weight.shape[0]),
        )
        if accounting is None:
            accounting = layer_accounting
        elif accounting != layer_accounting:
            raise ValueError("joint collective accounting differs across layers")
        installed.append(
            {
                "layer": layer,
                "variant": record["variant"],
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "private_encoder_shape": list(payload["private_encoders"].shape),
                "joint_decoder_shape": list(payload["joint_decoder"].shape),
                "factor_dtype": str(payload["joint_decoder"].dtype),
                "projected_weight_dtype": str(module.weight.dtype),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, projected
        torch.cuda.empty_cache()
        print(f"[Install] method={method} layer={layer}/{layers[-1]}", flush=True)
    if accounting is None:
        raise ValueError("joint factor installation produced no accounting")
    return installed, accounting


@torch.inference_mode()
def _install_scopewire_projected_weights(
    model: nn.Module,
    *,
    factor_dir: Path,
    results: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoder_layers = _decoder_layers(model)
    layers = tuple(map(int, results["layers"]))
    if layers != tuple(range(len(decoder_layers))):
        raise ValueError("ScopeWire PPL factors must cover every decoder layer")
    records_by_layer = results["selected_records"]
    installed = []
    accounting_by_layer = []
    for layer in layers:
        started = time.perf_counter()
        record = records_by_layer[str(layer)]
        artifact = record["artifact"]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"ScopeWire factor hash mismatch at layer {layer}")
        payload = load_file(str(path), device="cpu")
        module = decoder_layers[layer].self_attn.o_proj
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise TypeError(f"layer {layer} o_proj is unsupported")
        projected = _project_weight_scopewire(
            payload,
            artifact["layout"],
            device=module.weight.device,
        )
        if tuple(projected.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"ScopeWire projected weight shape differs at layer {layer}"
            )
        module.weight.copy_(projected.to(dtype=module.weight.dtype))
        layer_accounting = _scopewire_layer_accounting(
            artifact["layout"],
            hidden_size=int(module.weight.shape[0]),
        )
        if int(record["link_units"]) != int(layer_accounting["link_units"]):
            raise ValueError(f"ScopeWire link accounting differs at layer {layer}")
        accounting_by_layer.append({"layer": layer, **layer_accounting})
        installed.append(
            {
                "layer": layer,
                "selected": record["selected"],
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "factor_dtype": str(
                    payload[str(artifact["layout"][0]["decoder_key"])].dtype
                ),
                "component_count": len(artifact["layout"]),
                "component_layout": artifact["layout"],
                "decoder_rows": layer_accounting["decoder_rows"],
                "projected_weight_dtype": str(module.weight.dtype),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, projected
        torch.cuda.empty_cache()
        print(
            f"[Install] method={SCOPEWIRE_METHOD} layer={layer}/{layers[-1]} "
            f"selected={record['selected']}",
            flush=True,
        )
    if not accounting_by_layer:
        raise ValueError("ScopeWire factor installation produced no accounting")
    tp_sizes = {int(row["tp_size"]) for row in accounting_by_layer}
    link_bytes = {
        int(row["ideal_ring_bytes_per_rank"]) for row in accounting_by_layer
    }
    if len(tp_sizes) != 1 or len(link_bytes) != 1:
        raise ValueError("ScopeWire TP or link accounting differs across layers")
    total_decoder_rows = sum(
        int(row["decoder_rows"]) for row in accounting_by_layer
    )
    hidden_size = int(decoder_layers[0].self_attn.o_proj.weight.shape[0])
    accounting = {
        "collective": "mixed_scope",
        "tp_size": tp_sizes.pop(),
        "dtype_bytes": 2,
        "ideal_ring_bytes_per_rank_per_layer": link_bytes.pop(),
        "total_decoder_rows": total_decoder_rows,
        "total_decoder_parameters_per_rank": total_decoder_rows * hidden_size,
        "total_decoder_hbm_bytes_per_rank": total_decoder_rows * hidden_size * 2,
        "layers": accounting_by_layer,
    }
    return installed, accounting


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if min(args.seqlen, args.batch_size) <= 0:
        raise ValueError("PPL sequence and batch sizes must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PPL evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if (args.method == "dense") != (args.factor_dir is None):
        raise ValueError("dense requires no factor dir; compressed methods require one")
    factor_dir = (
        Path(args.factor_dir).expanduser().resolve()
        if args.factor_dir is not None
        else None
    )
    joint_results = (
        _load_joint_factor_results(factor_dir, model_path, args.method)
        if factor_dir is not None and args.method in JOINT_METHOD_TO_VARIANT
        else None
    )
    scopewire_results = (
        _load_scopewire_results(factor_dir, model_path)
        if factor_dir is not None and args.method == SCOPEWIRE_METHOD
        else None
    )
    factor_manifest = (
        _load_factor_manifest(factor_dir, model_path, args.method)
        if (
            factor_dir is not None
            and joint_results is None
            and scopewire_results is None
        )
        else None
    )

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    print(
        f"[PPL] loading model={model_path} method={args.method} device={device}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=model_dtype,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).eval()
    installed = []
    accounting = None
    factor_source = (
        scopewire_results
        if scopewire_results is not None
        else joint_results if joint_results is not None else factor_manifest
    )
    if factor_source is not None and factor_dir is not None:
        geometry = factor_source["model"]
        if (
            int(model.config.hidden_size) != int(geometry["hidden_size"])
            or int(model.config.num_hidden_layers)
            != int(geometry["num_hidden_layers"])
            or int(model.config.num_attention_heads)
            != int(geometry["num_attention_heads"])
            or int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
            != int(geometry["num_key_value_heads"])
        ):
            raise ValueError("loaded model geometry differs from factors")
        if scopewire_results is not None:
            installed, accounting = _install_scopewire_projected_weights(
                model,
                factor_dir=factor_dir,
                results=scopewire_results,
            )
        elif joint_results is not None:
            installed, accounting = _install_joint_projected_weights(
                model,
                factor_dir=factor_dir,
                results=joint_results,
                method=args.method,
            )
        else:
            assert factor_manifest is not None
            installed = _install_projected_weights(
                model,
                factor_dir=factor_dir,
                manifest=factor_manifest,
                method=args.method,
            )
            accounting = factor_manifest["fit_config"]["accounting"][args.method]

    result = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    elapsed = time.perf_counter() - started
    output = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "method": args.method,
        "quality_representation": (
            "dense_baseline"
            if args.method == "dense"
            else "dense_reconstruction_of_collective_linear_map"
        ),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "model_type": str(model.config.model_type),
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(
                getattr(
                    model.config,
                    "num_key_value_heads",
                    model.config.num_attention_heads,
                )
            ),
            "dtype": str(model_dtype),
        },
        "factor_manifest": (
            {
                "path": str(
                    factor_dir
                    / (
                        "results.json"
                        if joint_results is not None or scopewire_results is not None
                        else "manifest.json"
                    )
                ),
                "sha256": _sha256(
                    factor_dir
                    / (
                        "results.json"
                        if joint_results is not None or scopewire_results is not None
                        else "manifest.json"
                    )
                ),
                "activation_aware": True,
                "error_evaluation": (
                    joint_results is not None or scopewire_results is not None
                ),
                "joint_output_fitted": (
                    joint_results is not None or scopewire_results is not None
                ),
                "scope_selected": scopewire_results is not None,
            }
            if factor_dir is not None
            else None
        ),
        "collective_accounting": accounting,
        "installed_layers": installed,
        "ppl": result,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, output)
    print(
        f"[PPL] complete method={args.method} ppl={result['ppl']:.8f} "
        f"output={output_path} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
