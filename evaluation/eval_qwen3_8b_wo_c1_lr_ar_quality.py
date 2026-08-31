#!/usr/bin/env python3
"""Evaluate whole-model quality for the Qwen3-8B Wo-only Phase-1 arms.

The collective maps are folded into mathematically equivalent dense BF16
``o_proj`` weights.  This isolates approximation quality from custom runtime
kernel effects while leaving V and the KV cache dense for every arm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _eval_ppl_fp32_loss,
    _sha256,
)
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import (  # noqa: E402
    _evaluate_document_ppl,
    _load_windows,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_quality.v1"
PHASE1_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"
PHASE1_LAYER_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.layer.v2"
ARMS = ("dense", "wo_c1_ag", "wo_lr_ar_wire", "wo_lr_ar_capacity")
EXPECTED_FACTOR_KEYS = {
    "c1_source_encoders",
    "c1_source_decoders",
    "lr_wire_input_factor",
    "lr_wire_shared_decoder",
    "lr_wire_singular_values",
    "lr_capacity_input_factor",
    "lr_capacity_shared_decoder",
    "lr_capacity_singular_values",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--c4-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wikitext-batch-size", type=int, default=1)
    parser.add_argument("--c4-batch-size", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _validate_phase1(
    phase1_dir: Path,
    *,
    model_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    result_path = phase1_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = _load_json(result_path)
    if result.get("format") != PHASE1_FORMAT or result.get("status") != "complete":
        raise ValueError("Phase-1 factor bank is incomplete or incompatible")
    if result["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("model config differs from the Phase-1 factor bank")
    if result["method"]["scope"] != (
        "post-attention W_o only; dense V and dense KV cache"
    ):
        raise ValueError("Phase-1 factor scope is not the audited Wo-only setup")

    records: dict[int, dict[str, Any]] = {}
    for row in result["layers"]:
        layer = int(row["layer"])
        if layer in records:
            raise ValueError(f"duplicate Phase-1 layer {layer}")
        record_path = phase1_dir / f"layer_{layer:03d}.json"
        artifact_path = phase1_dir / row["artifact"]["file"]
        record = _load_json(record_path)
        if record.get("format") != PHASE1_LAYER_FORMAT:
            raise ValueError(f"incompatible Phase-1 record at layer {layer}")
        if record != row:
            raise ValueError(f"summary and layer record differ at layer {layer}")
        if record.get("status") != "passed":
            raise ValueError(f"Phase-1 layer {layer} did not pass its gates")
        if _sha256(artifact_path) != record["artifact"]["sha256"]:
            raise ValueError(f"Phase-1 artifact hash mismatch at layer {layer}")
        geometry = record["geometry"]
        if not bool(geometry["dense_v"]) or geometry["kv_cache_compression"] != "none":
            raise ValueError(f"layer {layer} is not a Wo-only factor artifact")
        records[layer] = record

    expected_layers = tuple(range(int(result["model"]["num_hidden_layers"])))
    if tuple(sorted(records)) != expected_layers:
        raise ValueError("Phase-1 factors do not cover every decoder layer")
    return result, records


def _validate_independent_c4(
    phase1: Mapping[str, Any],
    c4_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_path = Path(
        str(phase1["calibration"]["windows_file"])
    ).expanduser().resolve()
    calibration_manifest_path = calibration_path.parent / "manifest.json"
    if not calibration_manifest_path.is_file():
        raise FileNotFoundError(calibration_manifest_path)
    calibration = _load_json(calibration_manifest_path)
    if calibration["dataset"]["split"] != "train":
        raise ValueError("Phase-1 calibration windows are not from C4 train")
    if c4_provenance["dataset"]["split"] != "validation":
        raise ValueError("quality audit C4 windows are not from validation")
    fit_ids = {str(row["document_id"]) for row in calibration["records"]}
    audit_ids = set(map(str, c4_provenance["document_ids"]))
    overlap = fit_ids & audit_ids
    if overlap:
        raise ValueError(f"C4 fit/audit document overlap: {len(overlap)}")
    return {
        "phase1_windows": str(calibration_path),
        "phase1_windows_sha256": _sha256(calibration_path),
        "phase1_manifest": str(calibration_manifest_path),
        "phase1_manifest_sha256": _sha256(calibration_manifest_path),
        "phase1_split": calibration["dataset"]["split"],
        "phase1_documents": len(fit_ids),
        "audit_split": c4_provenance["dataset"]["split"],
        "audit_documents": len(audit_ids),
        "overlap_documents": 0,
    }


def _materialize_weight(
    payload: Mapping[str, Tensor],
    arm: str,
    *,
    device: torch.device,
) -> Tensor:
    """Return the FP32 dense weight represented by one Phase-1 arm."""

    if arm == "wo_c1_ag":
        encoders = payload["c1_source_encoders"].to(
            device=device, dtype=torch.float32
        )
        decoders = payload["c1_source_decoders"].to(
            device=device, dtype=torch.float32
        )
        if encoders.ndim != 3 or decoders.ndim != 3:
            raise ValueError("C1 factors must be rank-three tensors")
        sources, source_width, source_rank = map(int, encoders.shape)
        if tuple(decoders.shape[:2]) != (sources, source_rank):
            raise ValueError("C1 encoder/decoder geometry differs")
        output_width = int(decoders.shape[2])
        return (
            torch.bmm(encoders, decoders)
            .reshape(sources * source_width, output_width)
            .transpose(0, 1)
            .contiguous()
        )

    prefix = {
        "wo_lr_ar_wire": "lr_wire",
        "wo_lr_ar_capacity": "lr_capacity",
    }.get(arm)
    if prefix is None:
        raise ValueError(f"cannot materialize arm {arm}")
    input_factor = payload[f"{prefix}_input_factor"].to(
        device=device, dtype=torch.float32
    )
    decoder = payload[f"{prefix}_shared_decoder"].to(
        device=device, dtype=torch.float32
    )
    if input_factor.ndim != 2 or decoder.ndim != 2:
        raise ValueError("LR-AllReduce factors must be matrices")
    if int(input_factor.shape[1]) != int(decoder.shape[0]):
        raise ValueError("LR-AllReduce encoder/decoder ranks differ")
    return decoder.transpose(0, 1) @ input_factor.transpose(0, 1)


@torch.inference_mode()
def _install_arm(
    model: nn.Module,
    *,
    arm: str,
    phase1_dir: Path,
    records: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if arm == "dense":
        return []
    layers = _decoder_layers(model)
    if len(layers) != len(records):
        raise ValueError("loaded model and Phase-1 layer counts differ")
    installed = []
    for layer, module_wrapper in enumerate(layers):
        started = time.perf_counter()
        record = records[layer]
        artifact_path = phase1_dir / record["artifact"]["file"]
        payload = load_file(str(artifact_path), device="cpu")
        if set(payload) != EXPECTED_FACTOR_KEYS:
            raise ValueError(f"Phase-1 tensors differ at layer {layer}")
        module = module_wrapper.self_attn.o_proj
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise TypeError(f"unsupported o_proj module at layer {layer}")
        reconstructed = _materialize_weight(
            payload,
            arm,
            device=module.weight.device,
        )
        if tuple(reconstructed.shape) != tuple(module.weight.shape):
            raise ValueError(f"reconstructed weight shape differs at layer {layer}")
        if not bool(torch.isfinite(reconstructed).all()):
            raise FloatingPointError(f"non-finite reconstruction at layer {layer}")
        module.weight.copy_(reconstructed.to(dtype=module.weight.dtype))
        installed.append(
            {
                "layer": layer,
                "artifact": artifact_path.name,
                "artifact_sha256": record["artifact"]["sha256"],
                "weight_shape": list(reconstructed.shape),
                "factor_dtype": "bfloat16",
                "reconstruction_dtype": "float32",
                "installed_dtype": str(module.weight.dtype),
                "seconds": time.perf_counter() - started,
            }
        )
        del payload, reconstructed
        torch.cuda.empty_cache()
        print(f"[Wo quality] install arm={arm} layer={layer}/35", flush=True)
    return installed


def _accounting(
    arm: str,
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    first = records[0]
    geometry = first["geometry"]
    communication_key = {
        "dense": "dense_wo_allreduce",
        "wo_c1_ag": "c1_allgather",
        "wo_lr_ar_wire": "lr_ar_wire_matched",
        "wo_lr_ar_capacity": "lr_ar_capacity_matched",
    }[arm]
    bytes_per_row = float(first["communication"][communication_key])
    for layer, record in records.items():
        if float(record["communication"][communication_key]) != bytes_per_row:
            raise ValueError(f"communication accounting differs at layer {layer}")
    return {
        "scope": "post-attention W_o only",
        "collective": {
            "dense": "all_reduce",
            "wo_c1_ag": "private_all_gather",
            "wo_lr_ar_wire": "all_reduce",
            "wo_lr_ar_capacity": "all_reduce",
        }[arm],
        "ideal_ring_bytes_per_rank_per_activation_row": bytes_per_row,
        "tp_size": int(geometry["tp_size"]),
        "source_rank": (
            int(geometry["c1_source_ranks"][0]) if arm == "wo_c1_ag" else None
        ),
        "shared_rank": {
            "wo_lr_ar_wire": int(geometry["wire_matched_lr_ar_rank"]),
            "wo_lr_ar_capacity": int(geometry["capacity_matched_lr_ar_rank"]),
        }.get(arm),
        "dense_v": True,
        "kv_cache_compression": "none",
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    dense_wiki = float(payload["arms"]["dense"]["wikitext2"]["ppl"])
    dense_c4 = float(payload["arms"]["dense"]["c4_validation"]["ppl"])
    lines = [
        "# Qwen3-8B Wo-only C1 vs LR-AllReduce: whole-model quality",
        "",
        (
            "Every arm keeps V and the KV cache dense. Factorized collective "
            "maps are folded into equivalent BF16 `o_proj` weights, so these "
            "results isolate approximation quality from runtime kernels."
        ),
        "",
        "| Arm | WikiText-2 PPL | Δ vs dense | C4 validation PPL | Δ vs dense |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        wiki = float(payload["arms"][arm]["wikitext2"]["ppl"])
        c4 = float(payload["arms"][arm]["c4_validation"]["ppl"])
        lines.append(
            f"| `{arm}` | {wiki:.8f} | {(wiki / dense_wiki - 1) * 100:+.3f}% | "
            f"{c4:.8f} | {(c4 / dense_c4 - 1) * 100:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "Protocol:",
            "",
            "- WikiText-2 test, concatenated non-overlapping 2048-token chunks.",
            "- C4 validation, 128 document-disjoint 2048-token windows.",
            "- BF16 model execution, SDPA attention, FP32 cross-entropy accumulation.",
            "- C4 audit documents are disjoint from Phase-1 C4 train documents.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(
        args.wikitext_batch_size,
        args.c4_batch_size,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("batch sizes and thread count must be positive")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("whole-model quality evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = Path(args.model).expanduser().resolve()
    phase1_dir = args.phase1_dir.expanduser().resolve()
    c4_path = args.c4_windows.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    output_path = output_dir / "results.json"

    phase1, records = _validate_phase1(phase1_dir, model_path=model_path)
    c4_sequences, c4_provenance = _load_windows(c4_path, model_path=model_path)
    independence = _validate_independent_c4(phase1, c4_provenance)

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"[Wo quality] loading model={model_path} device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=model_dtype,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).eval()
    model.config.use_cache = False

    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        installed = _install_arm(
            model,
            arm=arm,
            phase1_dir=phase1_dir,
            records=records,
        )
        wikitext = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset="wikitext2",
            split="test",
            seqlen=2048,
            batch_size=args.wikitext_batch_size,
            max_samples=None,
            max_tokens=None,
        )
        c4 = _evaluate_document_ppl(
            model,
            c4_sequences,
            batch_size=args.c4_batch_size,
            label=arm,
        )
        arms[arm] = {
            "quality_representation": (
                "unaltered_dense_model"
                if arm == "dense"
                else "dense_reconstruction_of_collective_linear_map"
            ),
            "collective_accounting": _accounting(arm, records),
            "installation": installed,
            "wikitext2": wikitext,
            "c4_validation": c4,
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
        print(
            f"[Wo quality] arm={arm} wiki={wikitext['ppl']:.9f} "
            f"c4={c4['ppl']:.9f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    dense_wiki = float(arms["dense"]["wikitext2"]["ppl"])
    dense_c4 = float(arms["dense"]["c4_validation"]["ppl"])
    for row in arms.values():
        row["relative_to_dense"] = {
            "wikitext2_ppl_change": (
                float(row["wikitext2"]["ppl"]) / dense_wiki - 1.0
            ),
            "c4_validation_ppl_change": (
                float(row["c4_validation"]["ppl"]) / dense_c4 - 1.0
            ),
        }

    phase1_results_path = phase1_dir / "results.json"
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "dtype": str(model_dtype),
        },
        "phase1": {
            "path": str(phase1_results_path),
            "sha256": _sha256(phase1_results_path),
            "format": phase1["format"],
            "calibration": phase1["calibration"],
            "run_signature": phase1["method"]["run_signature"],
        },
        "c4_windows": c4_provenance,
        "selection_audit_independence": independence,
        "protocol": {
            "arms": list(ARMS),
            "scope": "post-attention W_o only; V and KV cache remain dense",
            "wikitext2": {
                "split": "test",
                "sequence_length": 2048,
                "batch_size": args.wikitext_batch_size,
                "cross_document_transitions": True,
            },
            "c4": {
                "split": "validation",
                "documents": 128,
                "sequence_length": 2048,
                "batch_size": args.c4_batch_size,
                "cross_document_transitions": False,
            },
            "model_dtype": args.model_dtype,
            "attn_implementation": args.attn_implementation,
            "loss_dtype": "float32",
        },
        "arms": arms,
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
    _atomic_json(output_path, payload)
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    print(f"[Wo quality] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
