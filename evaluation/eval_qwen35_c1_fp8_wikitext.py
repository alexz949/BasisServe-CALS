#!/usr/bin/env python3
"""Calibrate static C1 FP8 wire scales and evaluate WikiText-2 test PPL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_fp8_private_ag import (  # noqa: E402
    FP8_E4M3_MAX,
    Qwen35FP8PrivateAGRecord,
    Qwen35FP8PrivateAGRuntime,
)
from basisserve.core.qwen35_full_attention_private_ag_runtime import (  # noqa: E402
    load_qwen35_full_attention_private_ag_factors,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (  # noqa: E402
    load_qwen35_gdn_private_ag_factors,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation.eval_qwen35_wo_compression_wikitext import (  # noqa: E402
    EXPECTED_FULL,
    EXPECTED_GDN,
    _private_rank_schedule,
    _result,
    _sha256,
    _validate_private_factors,
)
from scripts.collect_qwen35_gdn_moments import _batches  # noqa: E402
from scripts.collect_qwen35_gdn_wo_activations import (  # noqa: E402
    _load_window_split,
)
from scripts.eval_qwen35_projected_gdn_nll import _dtype  # noqa: E402
from scripts.eval_svdllm_safetensors_ppl_accelerate import (  # noqa: E402
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
)


FORMAT = "basisserve.qwen35.c1_static_fp8_wikitext2_ppl.v1"
SCALE_FORMAT = "basisserve.qwen35.c1_static_fp8_scales.v1"
NUM_LAYERS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--uniform-gdn-factors", required=True)
    parser.add_argument("--uniform-full-factors", required=True)
    parser.add_argument("--ragged-gdn-factors", required=True)
    parser.add_argument("--ragged-full-factors", required=True)
    parser.add_argument("--output-scales", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-windows", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _factor_pair(
    gdn_path: Path,
    full_path: Path,
    *,
    model_path: Path,
    tp_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, int]]:
    gdn = load_qwen35_gdn_private_ag_factors(gdn_path)
    full = load_qwen35_full_attention_private_ag_factors(full_path)
    _validate_private_factors(
        gdn,
        model_path=model_path,
        tp_size=tp_size,
        expected_layers=EXPECTED_GDN,
    )
    _validate_private_factors(
        full,
        model_path=model_path,
        tp_size=tp_size,
        expected_layers=EXPECTED_FULL,
    )
    return gdn, full, _private_rank_schedule(gdn, full)


@torch.no_grad()
def _calibrate(
    model: torch.nn.Module,
    samples: torch.Tensor,
    *,
    gdn_factors: Mapping[str, Any],
    full_factors: Mapping[str, Any],
    batch_size: int,
    device: str,
    label: str,
) -> tuple[dict[int, dict[str, TensorOrScalar]], float]:
    started = time.perf_counter()
    runtime = Qwen35FP8PrivateAGRuntime(
        model,
        gdn_factors,
        full_factors,
        mode="observe",
    )
    processed = 0
    with runtime:
        for batch_index, input_ids in enumerate(_batches(samples, batch_size), start=1):
            input_ids = input_ids.to(device=device, dtype=torch.long)
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
            processed += int(input_ids.shape[0])
            print(
                f"[FP8 calibration] variant={label} batch={batch_index} "
                f"windows={processed}/{len(samples)}",
                flush=True,
            )
        state = runtime.scale_state()
    return state, time.perf_counter() - started


TensorOrScalar = torch.Tensor | int | str


def _scales(state: Mapping[int, Mapping[str, TensorOrScalar]]) -> dict[int, torch.Tensor]:
    result = {}
    for layer, record in state.items():
        value = record["source_scales"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("FP8 calibration scale must be a tensor")
        result[int(layer)] = value
    if tuple(sorted(result)) != tuple(range(NUM_LAYERS)):
        raise ValueError("FP8 calibration did not cover all decoder layers")
    return result


def _wire_metadata(
    records: tuple[Qwen35FP8PrivateAGRecord, ...],
    *,
    element_bytes: int,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda record: record.layer_index))
    if tuple(record.layer_index for record in ordered) != tuple(range(NUM_LAYERS)):
        raise ValueError("FP8 runtime did not cover all decoder layers")
    ranks = tuple(record.local_rank for record in ordered)
    widths = {record.local_width for record in ordered}
    tp_sizes = {record.tp_size for record in ordered}
    if len(widths) != 1 or len(tp_sizes) != 1:
        raise ValueError("FP8 runtime has inconsistent TP geometry")
    local_width = next(iter(widths))
    dense_element_bytes = 2
    fraction = (
        sum(ranks)
        * element_bytes
        / (len(ranks) * 2 * local_width * dense_element_bytes)
    )
    return {
        "collective": "source_private_allgather",
        "tp_size": next(iter(tp_sizes)),
        "local_width": local_width,
        "local_rank_schedule": list(ranks),
        "average_local_rank": sum(ranks) / len(ranks),
        "wire_dtype": "float8_e4m3fn" if element_bytes == 1 else "float16",
        "wire_element_bytes": element_bytes,
        "dense_allreduce_element_bytes": dense_element_bytes,
        "communication_fraction_of_dense_fp16_allreduce": fraction,
        "communication_reduction_fraction_vs_dense_fp16_allreduce": 1.0 - fraction,
        "static_scale_metadata_transmitted_per_token": False,
        "encoder_compute_dtype": "float16",
        "decoder_compute_dtype": "float16",
        "accumulation_dtype": "float16",
        "gdn_recurrent_state_compressed": False,
        "attention_cache_compressed": False,
        "single_gpu_tp_math_simulation": True,
    }


def _json_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    return value


def _profile_summary(
    profile: Mapping[int, Mapping[str, TensorOrScalar | float]],
) -> dict[str, Any]:
    elements = sum(int(record["wire_elements"]) for record in profile.values())
    clipped = sum(int(record["clipped_elements"]) for record in profile.values())
    max_ratio = 0.0
    for record in profile.values():
        observed = record["evaluation_source_amax"]
        scales = record["source_scales"]
        if not isinstance(observed, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise TypeError("FP8 quantization profile tensors are malformed")
        calibrated_amax = scales.float() * FP8_E4M3_MAX
        ratio = (observed.float() / calibrated_amax).max().item()
        max_ratio = max(max_ratio, float(ratio))
    return {
        "wire_elements": elements,
        "clipped_elements": clipped,
        "clipped_fraction": clipped / max(elements, 1),
        "maximum_evaluation_to_calibration_amax_ratio": max_ratio,
        "layers": _json_tree(profile),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    positive = (
        args.tp,
        args.calibration_windows,
        args.calibration_batch_size,
        args.seqlen,
        args.batch_size,
    )
    if min(positive) <= 0 or args.calibration_offset < 0:
        raise ValueError("evaluation sizes must be positive and offset nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max samples must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("max tokens must be positive")

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_scales = Path(args.output_scales).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    if output_scales.exists() or output_json.exists():
        raise FileExistsError("refusing to overwrite FP8 scale or PPL artifacts")
    factor_paths = {
        "uniform_gdn": Path(args.uniform_gdn_factors).expanduser().resolve(),
        "uniform_full": Path(args.uniform_full_factors).expanduser().resolve(),
        "ragged_gdn": Path(args.ragged_gdn_factors).expanduser().resolve(),
        "ragged_full": Path(args.ragged_full_factors).expanduser().resolve(),
    }
    uniform_gdn, uniform_full, uniform_schedule = _factor_pair(
        factor_paths["uniform_gdn"],
        factor_paths["uniform_full"],
        model_path=model_path,
        tp_size=args.tp,
    )
    ragged_gdn, ragged_full, ragged_schedule = _factor_pair(
        factor_paths["ragged_gdn"],
        factor_paths["ragged_full"],
        model_path=model_path,
        tp_size=args.tp,
    )
    if set(uniform_schedule.values()) != {192}:
        raise ValueError("uniform checkpoint must use local rank 192")
    if sum(ragged_schedule.values()) != NUM_LAYERS * 192:
        raise ValueError("ragged checkpoint must preserve average local rank 192")
    calibration_samples, calibration_records, windows_manifest = _load_window_split(
        windows_path,
        sample_offset=args.calibration_offset,
        num_samples=args.calibration_windows,
    )
    calibration_samples = calibration_samples.long()

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()

    calibration = {}
    for label, gdn_factors, full_factors in (
        ("uniform", uniform_gdn, uniform_full),
        ("ragged", ragged_gdn, ragged_full),
    ):
        state, elapsed = _calibrate(
            model,
            calibration_samples,
            gdn_factors=gdn_factors,
            full_factors=full_factors,
            batch_size=args.calibration_batch_size,
            device=args.device,
            label=label,
        )
        calibration[label] = {"layers": state, "elapsed_seconds": elapsed}
        _cleanup()

    eval_kwargs = {
        "dataset": "wikitext2",
        "split": "test",
        "seqlen": args.seqlen,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
    }
    dense = _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs)
    results = [
        _result(
            dense,
            variant="dense",
            dense=None,
            metadata={
                "attention_output_projection_compressed": False,
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
            },
        )
    ]
    for label, gdn_factors, full_factors in (
        ("uniform", uniform_gdn, uniform_full),
        ("ragged", ragged_gdn, ragged_full),
    ):
        fp16_runtime = Qwen35FP8PrivateAGRuntime(
            model,
            gdn_factors,
            full_factors,
            mode="passthrough",
        )
        with fp16_runtime:
            fp16_metrics = _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs)
            fp16_metadata = _wire_metadata(fp16_runtime.records, element_bytes=2)
        fp16_result = _result(
            fp16_metrics,
            variant=f"c1_{label}_fp16_wire",
            dense=dense,
            metadata=fp16_metadata,
        )
        results.append(fp16_result)
        del fp16_runtime
        _cleanup()

        fp8_runtime = Qwen35FP8PrivateAGRuntime(
            model,
            gdn_factors,
            full_factors,
            mode="quantize",
            scales=_scales(calibration[label]["layers"]),
        )
        with fp8_runtime:
            fp8_metrics = _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs)
            fp8_metadata = _wire_metadata(fp8_runtime.records, element_bytes=1)
            quantization_profile = fp8_runtime.quantization_profile()
        fp8_result = _result(
            fp8_metrics,
            variant=f"c1_{label}_static_fp8_e4m3_wire",
            dense=dense,
            metadata={
                **fp8_metadata,
                "fp8_quantization": _profile_summary(quantization_profile),
                "delta_mean_nll_vs_matching_fp16": (
                    float(fp8_metrics["nll_sum"]) - float(fp16_metrics["nll_sum"])
                )
                / int(fp8_metrics["tokens"]),
                "perplexity_ratio_vs_matching_fp16": (
                    float(fp8_metrics["ppl"]) / float(fp16_metrics["ppl"])
                ),
            },
        )
        results.append(fp8_result)
        del fp8_runtime, fp16_metrics, fp8_metrics
        _cleanup()

    scale_payload = {
        "format": SCALE_FORMAT,
        "schema_version": 1,
        "model_path": str(model_path),
        "tp_size": args.tp,
        "wire_dtype": "float8_e4m3fn",
        "fp8_max": FP8_E4M3_MAX,
        "scale_granularity": "static_per_layer_per_tp_source",
        "calibration": {
            "windows": str(
                windows_path / "windows.safetensors"
                if windows_path.is_dir()
                else windows_path
            ),
            "windows_sha256": windows_manifest["artifact"]["sha256"],
            "sample_offset": args.calibration_offset,
            "num_samples": args.calibration_windows,
            "sequence_length": int(calibration_samples.shape[1]),
            "records": list(calibration_records),
        },
        "factor_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in factor_paths.items()
        },
        "variants": calibration,
    }
    output_scales.parent.mkdir(parents=True, exist_ok=True)
    scale_temporary = output_scales.with_suffix(output_scales.suffix + ".tmp")
    torch.save(scale_payload, scale_temporary)
    scale_sha256 = _sha256(scale_temporary)

    device = torch.device(args.device)
    cuda_index = device.index if device.index is not None else 0
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "scale_artifact": {
            "path": str(output_scales),
            "sha256": scale_sha256,
            "format": SCALE_FORMAT,
        },
        "dataset": {
            "repo": WIKITEXT_REPO,
            "config": "wikitext-2-raw-v1",
            "revision": WIKITEXT_REVISION,
            "split": "test",
            "protocol": "concatenate_then_nonoverlapping_chunks",
            "seqlen": args.seqlen,
            "dropped_incomplete_tail": True,
        },
        "calibration": {
            "dataset": "C4 English train frozen document-disjoint fit split",
            "windows_sha256": windows_manifest["artifact"]["sha256"],
            "sample_offset": args.calibration_offset,
            "num_samples": args.calibration_windows,
            "sequence_length": int(calibration_samples.shape[1]),
            "scale_granularity": "static_per_layer_per_tp_source",
            "fp8_format": "E4M3FN",
            "fp8_max": FP8_E4M3_MAX,
        },
        "dtype": args.dtype,
        "tp_size": args.tp,
        "results": results,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(cuda_index),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(cuda_index)
            ),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(scale_temporary, output_scales)
    os.replace(json_temporary, output_json)
    print(f"[Saved scales] {output_scales}", flush=True)
    print(f"[Saved PPL] {output_json}", flush=True)


if __name__ == "__main__":
    main()
