"""Collect PaLU Fisher scores and build Nemotron-H MLRD/GLRD4 V factors.

Only full-attention ``v_proj`` modules are compressed.  Mamba2 output
projections and every attention ``o_proj`` remain dense.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results/tools/nemotron_deps"))
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM

from evaluation.collect_gqa_palu_fisher import (
    OFFICIAL_PALU_COMMIT,
    _selective_activation_offload,
)
from evaluation.capture_attention_o_proj_covariances import _StreamingCovarianceCapture
from evaluation.nemotron_h_runtime import install_mamba_device_guards
from evaluation.v96kl_common import (
    code_hashes,
    configure,
    read_json,
    save_tensors,
    sha256,
    write_json,
)

FISHER_FORMAT = "basisserve.nemotron_h.palu_fisher.v1"
V_COVARIANCE_FORMAT = "basisserve.nemotron_h.palu_v_covariances.v1"
CHECKPOINT_FORMAT = "basisserve.nemotron_h.palu_v_only_fisher.v1"
METHODS = {"mlrd": 1, "glrd4": 4}
RETENTIONS = {64: 0.5, 96: 0.75}
FAST_MAMBA_IMPLEMENTATIONS = {
    "mamba2_chunk_scan": "mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined",
    "mamba2_selective_state_update": "mamba_ssm.ops.triton.selective_state_update.selective_state_update",
    "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface.causal_conv1d_update",
}


def activate_fast_mamba(expected=FAST_MAMBA_IMPLEMENTATIONS):
    import causal_conv1d
    import inspect
    import mamba_ssm
    from transformers.models.nemotron_h import modeling_nemotron_h as native

    del causal_conv1d, mamba_ssm
    implementations = {}
    for name in (
        "mamba2_chunk_scan",
        "mamba2_selective_state_update",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        implementation = inspect.getclosurevars(getattr(native, name)).nonlocals[
            "implementation"
        ]
        implementations[name] = implementation.__module__ + "." + implementation.__name__
    assert implementations == expected
    return implementations


def _load_protocol(audit_path, smoke_path, windows_path):
    audit = read_json(audit_path)
    smoke = read_json(smoke_path)
    windows_manifest = read_json(windows_path.with_name("manifest.json"))
    assert audit["status"] == smoke["status"] == windows_manifest["status"] == "complete"
    assert smoke["full_model_tested"] and smoke["audit_sha256"] == sha256(audit_path)
    assert windows_manifest["protocol"]["audit_sha256"] == sha256(audit_path)
    assert windows_manifest["sha256"] == sha256(windows_path)
    assert windows_manifest["shape"] == [336, 2048]
    full_layers = [row["layer"] for row in audit["layers"] if row["kind"] == "full_attention"]
    mamba_layers = [row["layer"] for row in audit["layers"] if row["kind"] == "linear_attention"]
    assert full_layers and mamba_layers
    return audit, smoke, windows_manifest, full_layers, mamba_layers


def _attention_modules(model, full_layers):
    modules = {}
    for layer_index in full_layers:
        layer = model.model.layers[layer_index]
        assert layer.block_type == "full_attention"
        projection = layer.mixer.v_proj
        assert isinstance(projection, torch.nn.Linear) and projection.bias is None
        modules[layer_index] = projection
    return modules


def _chunked_multidevice_palu_loss_and_backward(model, batch, chunk_size):
    """PaLU's official double-shift loss with labels on the lm-head device."""
    base_outputs = model.model(input_ids=batch[:, :-1], use_cache=False)
    hidden_states = base_outputs.last_hidden_state[:, :-1, :]
    targets = batch[:, 2:].to(hidden_states.device)
    valid_tokens = targets.numel()
    hidden_gradient = torch.empty_like(hidden_states)
    loss_total = None
    for start in range(0, targets.shape[1], chunk_size):
        stop = min(start + chunk_size, targets.shape[1])
        chunk_hidden = hidden_states[:, start:stop, :].detach().requires_grad_(True)
        logits = model.lm_head(chunk_hidden).float()
        chunk_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets[:, start:stop].reshape(-1),
            reduction="sum",
        ) / valid_tokens
        (chunk_gradient,) = torch.autograd.grad(chunk_loss, chunk_hidden)
        hidden_gradient[:, start:stop, :].copy_(chunk_gradient)
        detached_loss = chunk_loss.detach()
        loss_total = detached_loss if loss_total is None else loss_total + detached_loss
    assert loss_total is not None
    hidden_states.backward(hidden_gradient)
    return loss_total


def _write_fisher_progress(args, accumulators, losses, processed):
    statistics_path = args.output.with_name(args.output.stem + "_sums.safetensors")
    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_statistics = statistics_path.with_name(statistics_path.name + ".tmp")
    save_file(
        {
            f"layer_{layer:03d}": value.detach().cpu().contiguous()
            for layer, value in accumulators.items()
        },
        str(temporary_statistics),
    )
    os.replace(temporary_statistics, statistics_path)
    progress = {
        "status": "in_progress" if processed < args.samples else "statistics_complete",
        "format": FISHER_FORMAT,
        "processed_samples": processed,
        "requested_samples": args.samples,
        "sequence_length": args.sequence_length,
        "loss_chunk_size": args.loss_chunk_size,
        "losses": losses,
        "statistics_file": statistics_path.name,
        "statistics_sha256": sha256(statistics_path),
        "windows_sha256": sha256(args.windows),
        "source_sha256": sha256(__file__),
    }
    temporary_progress = progress_path.with_name(progress_path.name + ".tmp")
    temporary_progress.write_text(json.dumps(progress, indent=2) + "\n")
    os.replace(temporary_progress, progress_path)
    return statistics_path, progress_path


@torch.inference_mode()
def collect_covariances(args):
    configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpu_count
    audit, smoke, windows_manifest, full_layers, mamba_layers = _load_protocol(
        args.audit, args.full_smoke, args.windows
    )
    fast_implementations = activate_fast_mamba()
    assert 1 <= args.fit_windows <= 256 and 1 <= args.heldout_windows <= 64
    assert args.sequence_length == 2048 and not args.output.exists()
    model_path = Path(audit["model"])
    ids = list(range(args.fit_windows)) + list(
        range(256, 256 + args.heldout_windows)
    )
    tokens = load_file(str(args.windows))["input_ids"][ids, : args.sequence_length].long()
    assert tokens.shape == (args.fit_windows + args.heldout_windows, args.sequence_length)
    memory = {
        index: torch.cuda.get_device_properties(index).total_memory - args.reserve_gib * 2**30
        for index in range(args.expected_gpu_count)
    }
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory=memory,
    ).eval()
    guarded = install_mamba_device_guards(model)
    assert guarded == mamba_layers
    model.config.use_cache = False
    modules = _attention_modules(model, full_layers)
    hidden_size = int(model.config.hidden_size)
    assert all(module.in_features == hidden_size for module in modules.values())
    capture = _StreamingCovarianceCapture(modules, hidden_size, torch.float32)
    input_device = model.get_input_embeddings().weight.device
    for split, offsets in (
        ("fit", range(args.fit_windows)),
        (
            "heldout",
            range(args.fit_windows, args.fit_windows + args.heldout_windows),
        ),
    ):
        for offset in offsets:
            sample_started = time.monotonic()
            batch = tokens[offset : offset + 1].to(input_device)
            capture.begin(split, rows=args.sequence_length)
            output = model.model(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                use_cache=False,
            )
            assert torch.isfinite(output.last_hidden_state).all()
            capture.finish()
            del output
            print(
                json.dumps(
                    {
                        "split": split,
                        "window_id": ids[offset],
                        "seconds": time.monotonic() - sample_started,
                    }
                ),
                flush=True,
            )
        capture.normalize_and_offload(
            split, expected_rows=len(offsets) * args.sequence_length
        )
    capture.close()
    args.output.mkdir(parents=True)
    artifacts = {}
    for layer, projection in modules.items():
        tensors = {
            "fit_covariance": capture.sums["fit"].pop(layer),
            "heldout_covariance": capture.sums["heldout"].pop(layer),
            "weight": projection.weight.detach().cpu().contiguous(),
        }
        path = args.output / f"layer_{layer:03d}.safetensors"
        save_tensors(path, tensors)
        artifacts[str(layer)] = {
            "file": path.name,
            "sha256": sha256(path),
            "weight_shape": list(tensors["weight"].shape),
            "covariance_shape": list(tensors["fit_covariance"].shape),
        }
        print("PALU V COVARIANCE SAVED", layer, flush=True)
    result = {
        "status": "complete",
        "format": V_COVARIANCE_FORMAT,
        "model": {
            "path": str(model_path),
            "config_sha256": audit["config_sha256"],
            "revision": model_path.name,
        },
        "target": "full_attention_v_proj_input",
        "layers": full_layers,
        "artifacts": artifacts,
        "calibration": {
            "dataset": "allenai/c4 train",
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": args.sequence_length,
            "fit_window_ids": ids[: args.fit_windows],
            "heldout_window_ids": ids[args.fit_windows :],
            "windows_sha256": sha256(args.windows),
            "manifest_sha256": sha256(args.windows.with_name("manifest.json")),
            "window_protocol": windows_manifest["protocol"],
            "statistic": "normalized uncentered E[x^T x] in float32",
        },
        "architecture": {
            "full_attention_layers": full_layers,
            "mamba_layers": mamba_layers,
            "mamba_wo": "dense_unchanged",
        },
        "runtime": {
            "fast_implementations": fast_implementations,
            "guarded_mamba_layers": guarded,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(args.expected_gpu_count)
            ],
            "peak_cuda_allocated_bytes": {
                str(index): torch.cuda.max_memory_allocated(index)
                for index in range(args.expected_gpu_count)
            },
        },
        "command": shlex.join(sys.argv),
        "environment": {"conda_environment": os.environ.get("CONDA_DEFAULT_ENV")},
        "seconds": time.monotonic() - started,
        "source_sha256": sha256(__file__),
    }
    write_json(args.output / "manifest.json", result)
    print("PALU V COVARIANCE COMPLETE", args.output, flush=True)


def collect_fisher(args):
    configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpu_count
    audit, smoke, windows_manifest, full_layers, mamba_layers = _load_protocol(
        args.audit, args.full_smoke, args.windows
    )
    fast_implementations = activate_fast_mamba()
    assert 1 <= args.samples <= 256 and 1 <= args.sequence_length <= 2048
    assert not args.output.exists()
    model_path = Path(audit["model"])
    tokens = load_file(str(args.windows))["input_ids"][: args.samples, : args.sequence_length].long()
    assert tokens.shape == (args.samples, args.sequence_length)
    memory = {
        index: torch.cuda.get_device_properties(index).total_memory - args.reserve_gib * 2**30
        for index in range(args.expected_gpu_count)
    }
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory=memory,
    ).eval()
    guarded = install_mamba_device_guards(model)
    assert guarded == mamba_layers
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    modules = _attention_modules(model, full_layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for projection in modules.values():
        assert projection.weight.device.type == "cuda"
        projection.weight.requires_grad_(True)
    accumulators = {
        layer: torch.zeros_like(projection.weight, dtype=torch.float32)
        for layer, projection in modules.items()
    }
    statistics_path = args.output.with_name(args.output.stem + "_sums.safetensors")
    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    start_sample = 0
    losses = []
    if args.resume:
        progress = read_json(progress_path)
        assert progress["format"] == FISHER_FORMAT
        assert progress["requested_samples"] == args.samples
        assert progress["sequence_length"] == args.sequence_length
        assert progress["loss_chunk_size"] == args.loss_chunk_size
        assert progress["windows_sha256"] == sha256(args.windows)
        assert progress["source_sha256"] == sha256(__file__)
        assert progress["statistics_sha256"] == sha256(statistics_path)
        saved = load_file(str(statistics_path), device="cpu")
        assert set(saved) == {f"layer_{layer:03d}" for layer in full_layers}
        for layer in full_layers:
            accumulators[layer].copy_(saved[f"layer_{layer:03d}"].to(accumulators[layer].device))
        start_sample = int(progress["processed_samples"])
        losses = [float(value) for value in progress["losses"]]
        assert start_sample == len(losses) <= args.samples
    else:
        assert not statistics_path.exists() and not progress_path.exists()
    input_device = model.get_input_embeddings().weight.device
    for sample_index in range(start_sample, args.samples):
        sample_started = time.monotonic()
        model.zero_grad(set_to_none=True)
        batch = tokens[sample_index : sample_index + 1].to(input_device)
        context = _selective_activation_offload(True) if args.activation_offload else nullcontext()
        with context:
            loss = _chunked_multidevice_palu_loss_and_backward(
                model, batch, chunk_size=args.loss_chunk_size
            )
        assert torch.isfinite(loss)
        losses.append(float(loss))
        for layer, projection in modules.items():
            gradient = projection.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            accumulators[layer].addcmul_(gradient.float(), gradient.float())
        record = {
            "sample": sample_index,
            "loss": losses[-1],
            "seconds": time.monotonic() - sample_started,
        }
        print(json.dumps(record), flush=True)
        processed = sample_index + 1
        if processed % args.checkpoint_every == 0 or processed == args.samples:
            _write_fisher_progress(args, accumulators, losses, processed)
            print("PALU FISHER PROGRESS", processed, "/", args.samples, flush=True)
    scalars = {}
    for layer in full_layers:
        statistic = accumulators[layer].div(args.samples).sqrt_()
        value = float(statistic.mean())
        assert math.isfinite(value) and value > 0
        scalars[str(layer)] = value
    result = {
        "status": "complete",
        "format": FISHER_FORMAT,
        "model": {
            "path": str(model_path),
            "config_sha256": audit["config_sha256"],
            "revision": model_path.name,
        },
        "fisher": {
            "target": "full_attention_v_proj_only",
            "dataset": "allenai/c4",
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "window_ids": list(range(args.samples)),
            "loss_semantics": "official PaLU shifted-input/shifted-label HF causal-LM loss",
            "loss_chunk_size": args.loss_chunk_size,
            "aggregation": "sqrt(mean(per-window gradient squared)), then matrix mean",
            "scalars": scalars,
            "mean_loss": sum(losses) / len(losses),
            "official_palu_commit": OFFICIAL_PALU_COMMIT,
            "squared_gradient_sum": {
                "file": statistics_path.name,
                "sha256": sha256(statistics_path),
                "dtype": "torch.float32",
                "tensor_count": len(full_layers),
            },
        },
        "architecture": {
            "full_attention_layers": full_layers,
            "mamba_layers": mamba_layers,
            "mamba_wo": "dense_unchanged",
        },
        "calibration": {
            "windows": str(args.windows.resolve()),
            "windows_sha256": sha256(args.windows),
            "manifest_sha256": sha256(args.windows.with_name("manifest.json")),
            "protocol": windows_manifest["protocol"],
        },
        "runtime": {
            "activation_offload": args.activation_offload,
            "fast_implementations": fast_implementations,
            "guarded_mamba_layers": guarded,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(args.expected_gpu_count)
            ],
            "peak_cuda_allocated_bytes": {
                str(index): torch.cuda.max_memory_allocated(index)
                for index in range(args.expected_gpu_count)
            },
        },
        "command": shlex.join(sys.argv),
        "environment": {"conda_environment": os.environ.get("CONDA_DEFAULT_ENV")},
        "seconds": time.monotonic() - started,
        "source_sha256": sha256(__file__),
    }
    write_json(args.output, result)
    print("PALU FISHER COMPLETE", args.output, flush=True)


def _exact_fisher_rank_map(
    scalars,
    *,
    output_width,
    num_heads,
    head_group_size,
    retained_ratio,
    block_size,
):
    """Allocate PaLU Fisher-weighted ranks with an exact global block budget."""
    assert num_heads % head_group_size == 0
    group_count = num_heads // head_group_size
    assert output_width % group_count == 0
    group_width = output_width // group_count
    assert group_width % block_size == 0
    quantum = group_count * block_size
    dense_rank_sum = len(scalars) * output_width
    target_rank_sum = int(round(dense_rank_sum * retained_ratio))
    assert target_rank_sum % quantum == 0
    fisher_sum = sum(scalars.values())
    assert math.isfinite(fisher_sum) and fisher_sum > 0

    target_units = target_rank_sum // quantum
    max_units = group_width // block_size
    units = {name: 1 for name in scalars}
    assert len(units) <= target_units <= len(units) * max_units
    ideal_units = {
        name: target_units * fisher / fisher_sum
        for name, fisher in scalars.items()
    }
    order = {name: index for index, name in enumerate(scalars)}
    remaining = target_units - len(units)
    while remaining:
        candidates = [name for name in scalars if units[name] < max_units]
        assert candidates
        selected = max(
            candidates,
            key=lambda name: (
                ideal_units[name] - units[name],
                scalars[name],
                -order[name],
            ),
        )
        units[selected] += 1
        remaining -= 1

    rank_map = {
        name: [unit_count * block_size] * group_count
        for name, unit_count in units.items()
    }
    rank_sum = sum(sum(ranks) for ranks in rank_map.values())
    assert rank_sum == target_rank_sum
    return rank_map, rank_sum, dense_rank_sum


def allocate_schedules(scalars, *, num_kv_heads, head_dim):
    schedules = {}
    for method, head_group_size in METHODS.items():
        for mean_rank, retained_ratio in RETENTIONS.items():
            rank_map, rank_sum, dense_rank_sum = _exact_fisher_rank_map(
                scalars,
                output_width=num_kv_heads * head_dim,
                num_heads=num_kv_heads,
                head_group_size=head_group_size,
                retained_ratio=retained_ratio,
                block_size=32,
            )
            schedules[(method, mean_rank)] = {
                "rank_map": rank_map,
                "rank_sum": rank_sum,
                "dense_rank_sum": dense_rank_sum,
                "realized_retained_ratio": rank_sum / dense_rank_sum,
            }
    return schedules


def _group_eigendecomposition(weight, covariance):
    gram = weight @ covariance @ weight.mT
    values, vectors = torch.linalg.eigh(gram)
    signs = vectors.gather(0, vectors.abs().argmax(0)[None]).sign()
    return values.flip(0).clamp_min(1e-30), (vectors * signs).flip(-1)


def _balanced_group_factors(weight, values, vectors, rank):
    values = values[:rank]
    vectors = vectors[:, :rank]
    root_sigma = values.pow(0.25)
    decoder = vectors * root_sigma
    writer = (vectors.mT @ weight) / root_sigma[:, None]
    return writer, decoder, values


def _weighted_relative_error(weight, writer, decoder, covariance, ranks):
    group_width = weight.shape[0] // len(ranks)
    offset = 0
    error = weight.new_zeros(())
    reference = weight.new_zeros(())
    for group, rank in enumerate(ranks):
        rows = slice(group * group_width, (group + 1) * group_width)
        dense = weight[rows]
        reconstructed = decoder[group] @ writer[offset : offset + rank]
        residual = dense - reconstructed
        error += (residual @ covariance * residual).sum()
        reference += (dense @ covariance * dense).sum()
        offset += rank
    return float((error.clamp_min(0) / reference).sqrt())


@torch.inference_mode()
def build(args):
    configure()
    assert torch.cuda.is_available()
    audit, smoke, windows_manifest, full_layers, mamba_layers = _load_protocol(
        args.audit, args.full_smoke, args.windows
    )
    fisher = read_json(args.fisher)
    assert fisher["status"] == "complete" and fisher["format"] == FISHER_FORMAT
    assert fisher["model"]["config_sha256"] == audit["config_sha256"]
    assert fisher["fisher"]["samples"] == 256 and fisher["fisher"]["sequence_length"] == 2048
    assert fisher["calibration"]["windows_sha256"] == sha256(args.windows)
    assert not args.output.exists()
    model_path = Path(audit["model"])
    config = json.loads((model_path / "config.json").read_text())
    num_kv_heads = int(config["num_key_value_heads"])
    num_query_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("attention_head_dim") or config.get("head_dim")
        or config["hidden_size"] // num_query_heads)
    hidden_size = int(config["hidden_size"])
    assert num_kv_heads == 8 and head_dim == 128
    schedules = allocate_schedules(
        fisher["fisher"]["scalars"], num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    payloads = {key: {} for key in schedules}
    records = {key: [] for key in schedules}
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.monotonic()
    covariance_manifest = read_json(args.covariances / "manifest.json")
    assert covariance_manifest["status"] == "complete"
    assert covariance_manifest["format"] == V_COVARIANCE_FORMAT
    assert covariance_manifest["target"] == "full_attention_v_proj_input"
    assert covariance_manifest["model"]["config_sha256"] == audit["config_sha256"]
    assert covariance_manifest["layers"] == full_layers
    assert covariance_manifest["calibration"]["fit_windows"] == 256
    assert covariance_manifest["calibration"]["heldout_windows"] == 64
    assert covariance_manifest["calibration"]["sequence_length"] == 2048
    for layer_index in full_layers:
        layer_started = time.monotonic()
        artifact = covariance_manifest["artifacts"][str(layer_index)]
        path = args.covariances / artifact["file"]
        assert sha256(path) == artifact["sha256"]
        tensors = load_file(str(path), device="cpu")
        weight = tensors["weight"].to(device=device, dtype=torch.float64)
        fit_covariance = tensors["fit_covariance"].to(device=device, dtype=torch.float64)
        heldout_covariance = tensors["heldout_covariance"].to(device=device, dtype=torch.float64)
        assert weight.shape == (num_kv_heads * head_dim, hidden_size)
        for method, group_size in METHODS.items():
            group_count = num_kv_heads // group_size
            group_width = group_size * head_dim
            decompositions = []
            for group in range(group_count):
                rows = slice(group * group_width, (group + 1) * group_width)
                group_weight = weight[rows]
                values, vectors = _group_eigendecomposition(
                    group_weight, fit_covariance
                )
                decompositions.append((group_weight, values, vectors))
            for mean_rank in RETENTIONS:
                key = (method, mean_rank)
                ranks = schedules[key]["rank_map"][str(layer_index)]
                assert len(ranks) == group_count and len(set(ranks)) == 1
                writers, decoders = [], []
                for group, rank in enumerate(ranks):
                    writer, decoder, values = _balanced_group_factors(
                        *decompositions[group], rank
                    )
                    writers.append(writer)
                    decoders.append(decoder)
                writer_bf16 = torch.cat(writers).to(torch.bfloat16).cpu().contiguous()
                decoder_bf16 = torch.stack(decoders).to(torch.bfloat16).cpu().contiguous()
                payloads[key][f"layers.{layer_index}.v_writer.weight"] = writer_bf16
                payloads[key][f"layers.{layer_index}.v_decoder.weight"] = decoder_bf16
                fit_error = _weighted_relative_error(
                    weight,
                    writer_bf16.to(device=device, dtype=torch.float64),
                    decoder_bf16.to(device=device, dtype=torch.float64),
                    fit_covariance,
                    ranks,
                )
                heldout_error = _weighted_relative_error(
                    weight,
                    writer_bf16.to(device=device, dtype=torch.float64),
                    decoder_bf16.to(device=device, dtype=torch.float64),
                    heldout_covariance,
                    ranks,
                )
                assert math.isfinite(fit_error) and math.isfinite(heldout_error)
                records[key].append({
                    "layer": layer_index,
                    "ranks": ranks,
                    "writer_shape": list(writer_bf16.shape),
                    "decoder_shape": list(decoder_bf16.shape),
                    "quantized_fit_relative_error": fit_error,
                    "quantized_heldout_relative_error": heldout_error,
                })
        del tensors, weight, fit_covariance, heldout_covariance
        torch.cuda.empty_cache()
        print("PALU BUILD LAYER", layer_index, "seconds", time.monotonic() - layer_started, flush=True)
    args.output.mkdir(parents=True)
    for key, tensors in payloads.items():
        method, mean_rank = key
        directory = args.output / f"{method}_r{mean_rank}"
        directory.mkdir()
        artifact_path = directory / "palu_v_factors.safetensors"
        save_tensors(artifact_path, tensors)
        schedule = schedules[key]
        layer_ranks = [schedule["rank_map"][str(layer)] for layer in full_layers]
        manifest = {
            "status": "complete",
            "format": CHECKPOINT_FORMAT,
            "model": {
                "path": str(model_path),
                "config_sha256": audit["config_sha256"],
                "index_sha256": audit["index_sha256"],
                "revision": model_path.name,
            },
            "compression": {
                "target": "full_attention_value_cache_only",
                "method": "PaLU " + ("M-LRD" if method == "mlrd" else "G-LRD4"),
                "method_slug": method,
                "allocation": "palu_fisher_exact_block_budget",
                "rank_block_size": 32,
                "num_query_heads": num_query_heads,
                "num_physical_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "head_group_size": METHODS[method],
                "groups": num_kv_heads // METHODS[method],
                "group_width": METHODS[method] * head_dim,
                "equivalent_mean_rank_per_head": mean_rank,
                "requested_retained_v_ratio": RETENTIONS[mean_rank],
                "realized_retained_v_ratio": schedule["realized_retained_ratio"],
                "layer_ranks": layer_ranks,
                "full_attention_layers": full_layers,
                "attention_o_proj": "dense_unchanged",
                "mamba2_wo": "dense_unchanged",
                "stored_factor_dtype": "torch.bfloat16",
                "factorization": "PaLU activation-aware balanced SVD from W C W^T in float64",
            },
            "calibration": {
                "dataset": "allenai/c4 train",
                "fit_windows": 256,
                "heldout_windows": 64,
                "sequence_length": 2048,
                "windows_sha256": sha256(args.windows),
                "covariance_manifest_sha256": sha256(args.covariances / "manifest.json"),
                "fisher_result": str(args.fisher.resolve()),
                "fisher_sha256": sha256(args.fisher),
                "fisher_loss": fisher["fisher"]["loss_semantics"],
            },
            "artifact": {
                "file": artifact_path.name,
                "sha256": sha256(artifact_path),
                "tensor_count": len(tensors),
                "bytes": artifact_path.stat().st_size,
            },
            "layers": records[key],
            "architecture": {
                "full_attention_layers": full_layers,
                "mamba_layers": mamba_layers,
            },
            "provenance": {
                "native_audit_sha256": sha256(args.audit),
                "full_smoke_sha256": sha256(args.full_smoke),
                "official_palu_commit": OFFICIAL_PALU_COMMIT,
                "source_sha256": sha256(__file__),
                "source_files": code_hashes([
                    "evaluation/reproduce_palu_paper_llama2_distributed.py",
                    "evaluation/collect_gqa_palu_fisher.py",
                ]),
            },
            "command": shlex.join(sys.argv),
            "environment": {"conda_environment": os.environ.get("CONDA_DEFAULT_ENV")},
            "seconds": time.monotonic() - started,
        }
        write_json(directory / "manifest.json", manifest)
        print("PALU CHECKPOINT COMPLETE", directory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    covariance = subparsers.add_parser("covariance")
    covariance.add_argument("--audit", type=Path, required=True)
    covariance.add_argument("--full-smoke", type=Path, required=True)
    covariance.add_argument("--windows", type=Path, required=True)
    covariance.add_argument("--output", type=Path, required=True)
    covariance.add_argument("--fit-windows", type=int, default=256)
    covariance.add_argument("--heldout-windows", type=int, default=64)
    covariance.add_argument("--sequence-length", type=int, default=2048)
    covariance.add_argument("--expected-gpu-count", type=int, required=True)
    covariance.add_argument("--reserve-gib", type=int, default=8)
    fisher = subparsers.add_parser("fisher")
    fisher.add_argument("--audit", type=Path, required=True)
    fisher.add_argument("--full-smoke", type=Path, required=True)
    fisher.add_argument("--windows", type=Path, required=True)
    fisher.add_argument("--output", type=Path, required=True)
    fisher.add_argument("--samples", type=int, default=256)
    fisher.add_argument("--sequence-length", type=int, default=2048)
    fisher.add_argument("--loss-chunk-size", type=int, default=128)
    fisher.add_argument("--expected-gpu-count", type=int, required=True)
    fisher.add_argument("--reserve-gib", type=int, default=8)
    fisher.add_argument("--activation-offload", action="store_true")
    fisher.add_argument("--checkpoint-every", type=int, default=8)
    fisher.add_argument("--resume", action="store_true")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--audit", type=Path, required=True)
    build_parser.add_argument("--full-smoke", type=Path, required=True)
    build_parser.add_argument("--windows", type=Path, required=True)
    build_parser.add_argument("--fisher", type=Path, required=True)
    build_parser.add_argument("--covariances", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    stages = {
        "covariance": collect_covariances,
        "fisher": collect_fisher,
        "build": build,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
