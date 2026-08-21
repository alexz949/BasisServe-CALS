#!/usr/bin/env python3
"""Fit a controlled, code-faithful ReCalKV OVC reference on Llama-2-7B.

K remains dense, every Value matrix uses one global rank-3072 factor, and
evaluation explicitly reconstructs dense Value before stock MHA.  The update
follows the pinned ReCalKV code, including its FP16 initialization boundary,
batch-of-eight summation, explicit least squares, and pseudoinverse refit.

Unlike the earlier static-covariance reference, update activations propagate
through already-compressed upstream layers.  Initialization and update windows
are non-overlapping slices of the same auditable C4 window bank.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.fit_llama2_mha_recalkv_global_ovc import (  # noqa: E402
    HIDDEN_SIZE,
    NUM_LAYERS,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    _atomic_json,
    _load_whitening_layer,
    _model_metadata,
    _sha256,
    _whitening_metadata,
)
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    _CaptureExit,
    _FirstLayerCatcher,
    _model_layers,
)


FORMAT = "basisserve.llama2_7b.recalkv_paper_controlled_ovc_reference.v1"
LAYER_FORMAT = (
    "basisserve.llama2_7b.recalkv_paper_controlled_ovc_reference.layer.v1"
)


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _relative_explicit_error(
    inputs: Tensor,
    outputs: Tensor,
    decoder: Tensor,
    encoder: Tensor,
) -> float:
    rows = inputs.reshape(-1, int(inputs.shape[-1]))
    targets = outputs.reshape(-1, int(outputs.shape[-1])).float()
    prediction = (
        rows.float() @ encoder.float().transpose(0, 1)
    ) @ decoder.float().transpose(0, 1)
    denominator = torch.linalg.vector_norm(targets)
    if float(denominator) == 0.0:
        raise ValueError("zero OVC target norm")
    result = float(torch.linalg.vector_norm(targets - prediction) / denominator)
    del rows, targets, prediction, denominator
    return result


def _decoder_lstsq_with_equivalent_scaling(
    latent: Tensor,
    targets: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    """Run official CUDA LS, with an objective-equivalent scaling fallback."""

    solution = torch.linalg.lstsq(latent, targets).solution
    if bool(torch.isfinite(solution).all()):
        return solution, {
            "solver": "torch.linalg.lstsq",
            "column_scaling_fallback": False,
        }

    column_norms = torch.linalg.vector_norm(latent, dim=0)
    if not bool(torch.isfinite(column_norms).all()) or bool((column_norms == 0).any()):
        raise FloatingPointError("invalid latent column norm in OVC LS fallback")
    scaled_latent = latent / column_norms.unsqueeze(0)
    scaled_solution = torch.linalg.lstsq(scaled_latent, targets).solution
    solution = scaled_solution / column_norms.unsqueeze(1)
    scaling_diagnostics = {
        "column_norm_min": float(column_norms.min()),
        "column_norm_max": float(column_norms.max()),
        "column_norm_condition_proxy": float(
            column_norms.max() / column_norms.min()
        ),
    }
    if bool(torch.isfinite(solution).all()):
        return solution, {
            "solver": "torch.linalg.lstsq",
            "column_scaling_fallback": True,
            "gram_eigh_fallback": False,
            **scaling_diagnostics,
        }

    # CUDA ``lstsq`` only exposes the non-rank-revealing GELS driver.  At the
    # controlled global rank 3072 it can return non-finite values even after
    # column equilibration.  The eigendecomposition below computes the
    # Moore-Penrose least-squares solution to the same objective, with the same
    # default rcond convention documented for torch.linalg.lstsq/pinv.
    gram = scaled_latent.transpose(0, 1) @ scaled_latent
    gram = (gram + gram.transpose(0, 1)).mul_(0.5)
    cross = scaled_latent.transpose(0, 1) @ targets
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[-1]
    singular_rcond = max(latent.shape) * torch.finfo(latent.dtype).eps
    eigenvalue_threshold = maximum * singular_rcond**2
    retained = eigenvalues > eigenvalue_threshold
    retained_rank = int(retained.sum())
    if retained_rank == 0:
        raise FloatingPointError("rank-revealing OVC LS fallback retained no modes")
    basis = eigenvectors[:, retained]
    inverse_projected = (basis.transpose(0, 1) @ cross) / eigenvalues[
        retained
    ].unsqueeze(1)
    scaled_solution = basis @ inverse_projected
    solution = scaled_solution / column_norms.unsqueeze(1)
    if not bool(torch.isfinite(solution).all()):
        raise FloatingPointError("non-finite rank-revealing OVC decoder solution")
    return solution, {
        "solver": "column-scaled Gram eigendecomposition Moore-Penrose LS",
        "column_scaling_fallback": True,
        "gram_eigh_fallback": True,
        "retained_numerical_rank": retained_rank,
        "requested_rank": int(latent.shape[1]),
        "singular_value_rcond": singular_rcond,
        "gram_eigenvalue_threshold": float(eigenvalue_threshold),
        "maximum_gram_eigenvalue": float(maximum),
        "minimum_retained_gram_eigenvalue": float(eigenvalues[retained][0]),
        **scaling_diagnostics,
    }


@torch.no_grad()
def official_global_ovc_from_explicit_rows(
    weight: Tensor,
    scaling_diag_matrix: Tensor,
    inputs: Tensor,
    outputs: Tensor,
    *,
    rank: int,
    iterations: int,
    progress_prefix: str | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Transcribe ReCalKV ``_per_head_whiten_update...`` for one V group."""

    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    output_width, input_width = map(int, weight.shape)
    if tuple(scaling_diag_matrix.shape) != (input_width, input_width):
        raise ValueError("initialization whitening has incompatible geometry")
    if inputs.ndim != 3 or int(inputs.shape[-1]) != input_width:
        raise ValueError("OVC inputs have incompatible geometry")
    if outputs.ndim != 3 or tuple(outputs.shape[:2]) != tuple(inputs.shape[:2]):
        raise ValueError("OVC outputs have incompatible leading dimensions")
    if int(outputs.shape[-1]) != output_width:
        raise ValueError("OVC outputs have incompatible width")
    if not 0 < rank <= min(output_width, input_width):
        raise ValueError("rank exceeds the Value matrix geometry")
    if iterations < 0:
        raise ValueError("OVC iterations must be nonnegative")

    phase_timings: dict[str, float] = {}
    phase_started = time.perf_counter()

    def finish_phase(name: str) -> None:
        nonlocal phase_started
        if weight.is_cuda:
            torch.cuda.synchronize(weight.device)
        now = time.perf_counter()
        phase_timings[name] = now - phase_started
        phase_started = now
        if progress_prefix is not None:
            print(
                f"{progress_prefix} phase={name} seconds={phase_timings[name]:.3f}",
                flush=True,
            )

    for name, tensor in (
        ("weight", weight),
        ("initialization_whitening", scaling_diag_matrix),
        ("update_inputs", inputs),
        ("update_outputs", outputs),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"non-finite {name} in official OVC update")
    finish_phase("finite_input_checks")

    original_dtype = weight.dtype
    scaling = scaling_diag_matrix.to(device=weight.device, dtype=torch.float32)
    scaling_inverse = torch.linalg.inv(scaling)
    finish_phase("whitening_inverse")
    scaled_weight = weight.float() @ scaling
    left, singular_values, right_t = torch.linalg.svd(
        scaled_weight, full_matrices=False
    )
    finish_phase("whitened_svd")
    right = right_t @ scaling_inverse
    roots = singular_values[:rank].sqrt()
    # Official code casts both initial factors back to the source dtype here.
    decoder = (left[:, :rank] * roots.unsqueeze(0)).to(original_dtype)
    encoder = (roots.unsqueeze(1) * right[:rank]).to(original_dtype)

    diagnostics: dict[str, Any] = {
        "official_initial_factor_dtype": str(original_dtype),
        "initial_relative_rmse": _relative_explicit_error(
            inputs, outputs, decoder, encoder
        ),
        "updates": [],
        "truncated_singular_value_first": float(singular_values[0]),
        "truncated_singular_value_last": float(singular_values[rank - 1]),
        "phase_timings_seconds": phase_timings,
    }
    finish_phase("initial_error")
    flat_inputs = inputs.reshape(-1, input_width)
    flat_outputs = outputs.reshape(-1, output_width)
    for iteration in range(iterations):
        before = _relative_explicit_error(inputs, outputs, decoder, encoder)
        latent = flat_inputs.float() @ encoder.float().transpose(0, 1)
        finish_phase(f"iteration_{iteration + 1}_latent")
        decoder_t, decoder_solver = _decoder_lstsq_with_equivalent_scaling(
            latent, flat_outputs.float()
        )
        finish_phase(f"iteration_{iteration + 1}_decoder_lstsq")
        decoder = decoder_t.transpose(0, 1).contiguous()
        after_decoder = _relative_explicit_error(
            inputs, outputs, decoder, encoder
        )
        finish_phase(f"iteration_{iteration + 1}_decoder_error")
        # Exactly ((W.T @ pinv(L)).T) from the official source.
        encoder = (
            weight.float().transpose(0, 1) @ torch.linalg.pinv(decoder_t)
        ).transpose(0, 1).contiguous()
        if not bool(torch.isfinite(encoder).all()):
            raise FloatingPointError("non-finite encoder solution in official OVC update")
        finish_phase(f"iteration_{iteration + 1}_encoder_pinv")
        after_encoder = _relative_explicit_error(
            inputs, outputs, decoder, encoder
        )
        finish_phase(f"iteration_{iteration + 1}_encoder_error")
        diagnostics["updates"].append(
            {
                "iteration": iteration + 1,
                "relative_rmse_before": before,
                "relative_rmse_after_decoder": after_decoder,
                "relative_rmse_after_encoder": after_encoder,
                "decoder_solver": decoder_solver,
            }
        )
        del latent, decoder_t

    # ``from_linear_adasvd`` stores both final module weights as FP16.
    decoder = decoder.to(torch.float16).contiguous()
    encoder = encoder.to(torch.float16).contiguous()
    diagnostics["saved_factor_relative_rmse"] = _relative_explicit_error(
        inputs, outputs, decoder, encoder
    )
    finish_phase("saved_factor_error")
    diagnostics["saved_factor_dtype"] = "torch.float16"
    diagnostics["explicit_update_rows"] = int(flat_inputs.shape[0])
    return decoder, encoder, diagnostics


class _GlobalLowRankValue(nn.Module):
    def __init__(self, encoder: Tensor, decoder: Tensor) -> None:
        super().__init__()
        rank, input_width = map(int, encoder.shape)
        output_width, decoder_rank = map(int, decoder.shape)
        if decoder_rank != rank:
            raise ValueError("encoder and decoder ranks differ")
        self.in_features = input_width
        self.out_features = output_width
        self.rank = rank
        self.encoder = nn.Linear(
            input_width,
            rank,
            bias=False,
            device=encoder.device,
            dtype=encoder.dtype,
        )
        self.decoder = nn.Linear(
            rank,
            output_width,
            bias=False,
            device=decoder.device,
            dtype=decoder.dtype,
        )
        self.encoder.weight.copy_(encoder)
        self.decoder.weight.copy_(decoder)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.decoder(self.encoder(hidden_states))


class _OfficialBatchSumCapture:
    """Replicate ReCalKV BatchUpdater's sum over each forward batch."""

    def __init__(self) -> None:
        self.inputs: list[Tensor] = []
        self.outputs: list[Tensor] = []

    def __call__(
        self,
        module: nn.Module,
        hook_inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        del module
        hidden = hook_inputs[0].detach()
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        projected = output.detach()
        if projected.ndim == 2:
            projected = projected.unsqueeze(0)
        self.inputs.append(hidden.sum(dim=0))
        self.outputs.append(projected.sum(dim=0))


def _expand_batch(value: Any, batch_size: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim > 0 and int(value.shape[0]) == 1 and batch_size != 1:
            return value.expand(batch_size, *value.shape[1:])
        return value
    if isinstance(value, tuple):
        return tuple(_expand_batch(item, batch_size) for item in value)
    if isinstance(value, list):
        return [_expand_batch(item, batch_size) for item in value]
    return value


def _batch_kwargs(captured: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    return {key: _expand_batch(value, batch_size) for key, value in captured.items()}


def _move_capture_modules(model: nn.Module, device: torch.device) -> list[nn.Module]:
    core = model.model
    modules = [core.embed_tokens, core.norm]
    rotary = getattr(core, "rotary_emb", None)
    if isinstance(rotary, nn.Module):
        modules.append(rotary)
    unique: list[nn.Module] = []
    for module in modules:
        if all(module is not observed for observed in unique):
            module.to(device)
            unique.append(module)
    return unique


@torch.no_grad()
def _capture_first_layer_inputs(
    model: nn.Module,
    input_ids: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    samples, sequence_length = map(int, input_ids.shape)
    inputs = torch.empty(
        (samples, sequence_length, HIDDEN_SIZE),
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    outputs = torch.empty_like(inputs)
    captured_kwargs: dict[str, Any] = {}
    layers = _model_layers(model)
    first_layer = layers[0].to(device)
    catcher = _FirstLayerCatcher(first_layer, inputs, captured_kwargs)
    moved_modules = _move_capture_modules(model, device)
    layers[0] = catcher
    try:
        for sample in range(samples):
            row = input_ids[sample].unsqueeze(0).to(device)
            try:
                model(
                    input_ids=row,
                    attention_mask=torch.ones_like(row),
                    use_cache=False,
                )
            except _CaptureExit:
                pass
    finally:
        layers[0] = first_layer
        first_layer.cpu()
        for module in moved_modules:
            module.cpu()
    if catcher.index != samples:
        raise RuntimeError(f"captured {catcher.index} inputs, expected {samples}")
    torch.cuda.empty_cache()
    return inputs, outputs, captured_kwargs


def _window_payload(
    windows_path: Path,
    *,
    whitening: Mapping[str, Any],
    update_offset: int,
    update_windows: int,
    update_batch_size: int,
) -> tuple[Tensor, dict[str, Any]]:
    if not windows_path.is_file():
        raise FileNotFoundError(windows_path)
    observed_sha = _sha256(windows_path)
    if whitening.get("windows_sha256") != observed_sha:
        raise ValueError("update window bank differs from initialization whitening")
    payload = load_file(str(windows_path), device="cpu")
    input_ids = payload.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("window bank must contain matrix input_ids")
    if int(input_ids.shape[1]) != 2048:
        raise ValueError("controlled OVC requires 2048-token windows")
    initialization_windows = int(whitening.get("fit_windows") or 0)
    stop = update_offset + update_windows
    if update_offset < initialization_windows:
        raise ValueError("update windows overlap initialization windows")
    if update_windows <= 0 or stop > int(input_ids.shape[0]):
        raise ValueError("requested update window range is unavailable")
    if update_batch_size <= 0 or update_windows % update_batch_size:
        raise ValueError("update windows must divide into official-size batches")
    selected = input_ids[update_offset:stop].to(torch.long).contiguous()
    split_codes = payload.get("split_codes")
    split_counts: dict[str, int] = {}
    if split_codes is not None:
        selected_codes = split_codes[update_offset:stop].to(torch.long).tolist()
        split_counts = {
            str(code): count
            for code, count in sorted(Counter(selected_codes).items())
        }
    return selected, {
        "path": str(windows_path),
        "sha256": observed_sha,
        "available_windows": int(input_ids.shape[0]),
        "initialization_window_range": [0, initialization_windows],
        "update_window_range": [update_offset, stop],
        "overlap": 0,
        "update_windows": update_windows,
        "update_batch_size": update_batch_size,
        "update_batches": update_windows // update_batch_size,
        "split_code_counts": split_counts,
        "update_hook_reduction": "sum over each forward batch before explicit LS",
    }


def _fit_config(
    args: argparse.Namespace,
    *,
    model_metadata: Mapping[str, Any],
    whitening: Mapping[str, Any],
    windows: Mapping[str, Any],
) -> dict[str, Any]:
    if args.rank != 3072:
        raise ValueError("the controlled V75 experiment requires global rank 3072")
    if args.ovc_iterations != 1:
        raise ValueError("the paper-controlled experiment requires exactly one update")
    return {
        "method": "paper-controlled ReCalKV global OVC",
        "classification": "dense-reconstruction oracle/reference",
        "model": model_metadata["path"],
        "model_config_sha256": model_metadata["config_sha256"],
        "model_safetensors_index_sha256": model_metadata[
            "safetensors_index_sha256"
        ],
        "key_projection": "dense and unchanged",
        "value_factorization_scope": "one global/full-layer W_V matrix",
        "value_rank": args.rank,
        "dense_value_width": HIDDEN_SIZE,
        "value_retained_ratio": args.rank / HIDDEN_SIZE,
        "total_kv_retained_ratio_with_dense_k": (HIDDEN_SIZE + args.rank)
        / (2 * HIDDEN_SIZE),
        "dense_reconstruction_before_mha": True,
        "deployable_fused_kernel_claim": False,
        "ovc_iterations": args.ovc_iterations,
        "initialization_whitening": dict(whitening),
        "updating_windows": dict(windows),
        "progressive_upstream": (
            "each layer updates on activations propagated through already-compressed "
            "upstream Value projections"
        ),
        "official_numeric_sequence": [
            "FP32 activation-whitened SVD",
            "initial decoder and encoder cast to source FP16",
            "FP32 explicit torch.linalg.lstsq decoder update",
            "FP32 pseudoinverse encoder refit",
            "final decoder and encoder cast to FP16",
        ],
        "numerical_safety_fallback": (
            "if CUDA GELS is non-finite: column equilibration, then a "
            "rank-revealing Gram-eigh Moore-Penrose solution to the unchanged LS "
            "objective using the default singular-value rcond convention"
        ),
        "controlled_differences_from_full_paper_pipeline": {
            "rank": "fixed global V3072 instead of Fisher allocation",
            "key": "dense instead of HSR-compressed",
            "dataset": "independent C4 update slice instead of default Wikitext2",
            "update_samples": args.update_windows,
            "evaluation_runtime": "explicit dense V reconstruction before stock MHA",
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "source_function": (
                "palu/model/modules/svd_linear.py::"
                "_per_head_whiten_update_decomposition_from_weight"
            ),
            "orchestration_function": "palu/decomposition.py::compress_model_ours",
        },
    }


def _load_resume_layer(
    output_dir: Path,
    *,
    layer_index: int,
    fit_config: Mapping[str, Any],
) -> tuple[dict[str, Any], Tensor, Tensor] | None:
    record_path = output_dir / f"layer_{layer_index:03d}.json"
    artifact_path = output_dir / f"layer_{layer_index:03d}.safetensors"
    if not record_path.exists() and not artifact_path.exists():
        return None
    if not record_path.is_file() or not artifact_path.is_file():
        raise FileExistsError(f"partial resume output at layer {layer_index}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    observed_config = dict(record.get("fit_config") or {})
    expected_config = dict(fit_config)
    # This field documents a solver safety path added after layer 0 was already
    # fitted.  Layer 0 used a finite official GELS solution, so its factors are
    # independent of the newly added fallback.  All scientific configuration
    # fields remain exact resume requirements.
    observed_config.pop("numerical_safety_fallback", None)
    expected_config.pop("numerical_safety_fallback", None)
    if (
        record.get("format") != LAYER_FORMAT
        or int(record.get("layer", -1)) != layer_index
        or observed_config != expected_config
        or record.get("artifact", {}).get("sha256") != _sha256(artifact_path)
    ):
        raise ValueError(f"incompatible resume output at layer {layer_index}")
    factors = load_file(str(artifact_path), device="cpu")
    if set(factors) != {"v_encoder_weight", "v_reconstruction_weight"}:
        raise ValueError(f"unexpected resume tensors at layer {layer_index}")
    return record, factors["v_encoder_weight"], factors["v_reconstruction_weight"]


@torch.no_grad()
def _propagate_layer(
    layer: nn.Module,
    inputs: Tensor,
    outputs: Tensor,
    captured_kwargs: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    for sample in range(int(inputs.shape[0])):
        result = layer(
            inputs[sample].unsqueeze(0),
            **_batch_kwargs(captured_kwargs, 1),
        )
        outputs[sample].copy_(result[0][0])
    return outputs, inputs


@torch.no_grad()
def _fit(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-controlled OVC fitting requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(args.model).expanduser().resolve()
    whitening_dir = Path(args.whitening_dir).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_path = output_dir / "results.json"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    if args.resume and results_path.is_file():
        print(f"[Controlled OVC] complete result already exists: {results_path}")
        return

    model_metadata, _ = _model_metadata(model_path)
    whitening, whitening_path = _whitening_metadata(
        whitening_dir,
        model_config_sha256=model_metadata["config_sha256"],
    )
    update_input_ids, windows = _window_payload(
        windows_path,
        whitening=whitening,
        update_offset=args.update_offset,
        update_windows=args.update_windows,
        update_batch_size=args.update_batch_size,
    )
    fit_config = _fit_config(
        args,
        model_metadata=model_metadata,
        whitening=whitening,
        windows=windows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    layers = _model_layers(model)
    if len(layers) != NUM_LAYERS:
        raise ValueError("controlled OVC expects all 32 Llama layers")
    inputs, outputs, captured_kwargs = _capture_first_layer_inputs(
        model,
        update_input_ids,
        device=device,
    )
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        layer.to(device)
        dense_v = layer.self_attn.v_proj
        dense_k = layer.self_attn.k_proj
        if not isinstance(dense_v, nn.Linear) or dense_v.bias is not None:
            raise TypeError(f"unsupported v_proj at layer {layer_index}")
        resumed = (
            _load_resume_layer(
                output_dir,
                layer_index=layer_index,
                fit_config=fit_config,
            )
            if args.resume
            else None
        )
        if resumed is None:
            capture = _OfficialBatchSumCapture()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dense_capture_started = time.perf_counter()
            handle = dense_v.register_forward_hook(capture)
            try:
                for start in range(
                    0, int(inputs.shape[0]), args.update_batch_size
                ):
                    batch = inputs[start : start + args.update_batch_size]
                    layer(
                        batch,
                        **_batch_kwargs(captured_kwargs, int(batch.shape[0])),
                    )
            finally:
                handle.remove()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(
                f"[Controlled OVC] layer={layer_index} phase=dense_capture "
                f"seconds={time.perf_counter() - dense_capture_started:.3f}",
                flush=True,
            )
            expected_batches = args.update_windows // args.update_batch_size
            if len(capture.inputs) != expected_batches:
                raise RuntimeError(
                    f"layer {layer_index} captured {len(capture.inputs)} update "
                    f"batches, expected {expected_batches}"
                )
            update_inputs = torch.stack(capture.inputs)
            update_outputs = torch.stack(capture.outputs)
            initialization = _load_whitening_layer(
                whitening_path, layer_index
            ).to(device)
            decoder, encoder, diagnostics = official_global_ovc_from_explicit_rows(
                dense_v.weight.detach(),
                initialization,
                update_inputs,
                update_outputs,
                rank=args.rank,
                iterations=args.ovc_iterations,
                progress_prefix=f"[Controlled OVC] layer={layer_index}",
            )
            artifact_path = output_dir / f"layer_{layer_index:03d}.safetensors"
            cpu_encoder = encoder.cpu().contiguous()
            cpu_decoder = decoder.cpu().contiguous()
            _atomic_safetensors(
                artifact_path,
                {
                    "v_encoder_weight": cpu_encoder,
                    "v_reconstruction_weight": cpu_decoder,
                },
            )
            record = {
                "format": LAYER_FORMAT,
                "layer": layer_index,
                "fit_config": fit_config,
                "artifact": {
                    "file": artifact_path.name,
                    "sha256": _sha256(artifact_path),
                    "tensors": {
                        "v_encoder_weight": {
                            "shape": list(cpu_encoder.shape),
                            "dtype": str(cpu_encoder.dtype),
                        },
                        "v_reconstruction_weight": {
                            "shape": list(cpu_decoder.shape),
                            "dtype": str(cpu_decoder.dtype),
                        },
                    },
                },
                "diagnostics": diagnostics,
                "elapsed_seconds_before_propagation": time.perf_counter()
                - layer_started,
            }
            _atomic_json(output_dir / f"layer_{layer_index:03d}.json", record)
            del capture, update_inputs, update_outputs, initialization
        else:
            record, cpu_encoder, cpu_decoder = resumed
            encoder = cpu_encoder.to(device)
            decoder = cpu_decoder.to(device)
            print(f"[Controlled OVC] layer={layer_index} resume hit", flush=True)

        replacement = _GlobalLowRankValue(encoder, decoder)
        layer.self_attn.v_proj = replacement
        if layer.self_attn.k_proj is not dense_k:
            raise RuntimeError("controlled OVC changed the dense Key projection")
        inputs, outputs = _propagate_layer(
            layer,
            inputs,
            outputs,
            captured_kwargs,
        )
        layer.cpu()
        records.append(record)
        print(
            f"[Controlled OVC] layer={layer_index}/{NUM_LAYERS - 1} "
            f"rmse={record['diagnostics']['saved_factor_relative_rmse']:.8g} "
            f"seconds={time.perf_counter() - layer_started:.2f}",
            flush=True,
        )
        del dense_v, dense_k, encoder, decoder, cpu_encoder, cpu_decoder, replacement
        torch.cuda.empty_cache()

    errors = [
        float(record["diagnostics"]["saved_factor_relative_rmse"])
        for record in records
    ]
    result = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": (
            "paper-controlled ReCalKV global-OVC rank3072 "
            "dense-reconstruction oracle/reference"
        ),
        "layers": list(range(NUM_LAYERS)),
        "fit_config": fit_config,
        "artifacts": {
            str(record["layer"]): record["artifact"] for record in records
        },
        "records": records,
        "aggregate": {
            "mean_saved_factor_relative_rmse": sum(errors) / len(errors),
            "maximum_saved_factor_relative_rmse": max(errors),
        },
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
    _atomic_json(results_path, result)
    print(f"[Controlled OVC] wrote {results_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--whitening-dir", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=3072)
    parser.add_argument("--ovc-iterations", type=int, default=1)
    parser.add_argument("--update-offset", type=int, default=128)
    parser.add_argument("--update-windows", type=int, default=128)
    parser.add_argument("--update-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    _fit(parse_args())


if __name__ == "__main__":
    main()
