#!/usr/bin/env python3
"""Evaluate one Qwen3-8B Wo-only uniform-rank factor bank on WikiText-2."""

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
from torch import nn
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
from evaluation.eval_qwen3_8b_wo_c1_lr_ar_quality import (  # noqa: E402
    EXPECTED_FACTOR_KEYS,
    _accounting,
    _materialize_weight,
    _validate_phase1,
)
from evaluation.fit_qwen3_8b_wo_c1_independent_local import (  # noqa: E402
    FORMAT as INDEPENDENT_LOCAL_FORMAT,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_uniform_rank_quality.v1"
ARMS = (
    "dense",
    "joint_c1",
    "independent_local_c1",
    "wire_matched_lr_allreduce",
    "capacity_matched_lr_allreduce",
)
PHASE_ARM = {
    "joint_c1": "wo_c1_ag",
    "wire_matched_lr_allreduce": "wo_lr_ar_wire",
    "capacity_matched_lr_allreduce": "wo_lr_ar_capacity",
}
LOCAL_FACTOR_KEYS = {"c1_source_encoders", "c1_source_decoders"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--independent-local-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _validate_independent_local(
    local_dir: Path,
    *,
    phase1_dir: Path,
    phase1: Mapping[str, Any],
    model_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    results_path = local_dir / "results.json"
    results = _load_json(results_path)
    if (
        results.get("format") != INDEPENDENT_LOCAL_FORMAT
        or results.get("status") != "complete"
    ):
        raise ValueError("independent-local factor bank is incomplete")
    phase1_results_path = phase1_dir / "results.json"
    if results["source"]["joint_phase1_results_sha256"] != _sha256(phase1_results_path):
        raise ValueError("independent-local bank does not reference Phase-1")
    if results["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("model config differs from independent-local factors")
    expected_rank = int(phase1["method"]["run_signature"]["source_rank"])
    if int(results["protocol"]["source_rank"]) != expected_rank:
        raise ValueError("independent-local and joint source ranks differ")
    records: dict[int, dict[str, Any]] = {}
    for row in results["layers"]:
        layer = int(row["layer"])
        if layer in records:
            raise ValueError(f"duplicate independent-local layer {layer}")
        artifact_path = local_dir / row["artifact"]["file"]
        if _sha256(artifact_path) != row["artifact"]["sha256"]:
            raise ValueError(f"independent-local hash mismatch at layer {layer}")
        records[layer] = row
    expected_layers = tuple(range(int(phase1["model"]["num_hidden_layers"])))
    if tuple(sorted(records)) != expected_layers:
        raise ValueError("independent-local factors do not cover every layer")
    return results, records


@torch.inference_mode()
def _install_arm(
    model: nn.Module,
    *,
    arm: str,
    phase1_dir: Path,
    phase1_records: Mapping[int, Mapping[str, Any]],
    local_dir: Path,
    local_records: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if arm == "dense":
        return []
    layers = _decoder_layers(model)
    if len(layers) != len(phase1_records) or len(layers) != len(local_records):
        raise ValueError("model and factor layer counts differ")
    installed = []
    for layer, module_wrapper in enumerate(layers):
        started = time.perf_counter()
        if arm == "independent_local_c1":
            record = local_records[layer]
            artifact_path = local_dir / record["artifact"]["file"]
            payload = load_file(str(artifact_path), device="cpu")
            if set(payload) != LOCAL_FACTOR_KEYS:
                raise ValueError(f"independent-local tensors differ at layer {layer}")
            materialization_arm = "wo_c1_ag"
        else:
            record = phase1_records[layer]
            artifact_path = phase1_dir / record["artifact"]["file"]
            payload = load_file(str(artifact_path), device="cpu")
            if set(payload) != EXPECTED_FACTOR_KEYS:
                raise ValueError(f"Phase-1 tensors differ at layer {layer}")
            materialization_arm = PHASE_ARM[arm]
        module = module_wrapper.self_attn.o_proj
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise TypeError(f"unsupported o_proj at layer {layer}")
        reconstructed = _materialize_weight(
            payload,
            materialization_arm,
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
                "artifact": str(artifact_path),
                "artifact_sha256": record["artifact"]["sha256"],
                "materialization_arm": materialization_arm,
                "installed_dtype": str(module.weight.dtype),
                "seconds": time.perf_counter() - started,
            }
        )
        del payload, reconstructed
        torch.cuda.empty_cache()
        print(f"[Uniform quality] install arm={arm} layer={layer}/35", flush=True)
    return installed


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    dense = float(payload["arms"]["dense"]["wikitext2"]["ppl"])
    lines = [
        f"# Qwen3-8B Wo-only uniform rank-{payload['protocol']['source_rank']} quality",
        "",
        (
            "Every compressed arm uses the same C4 calibration covariances and "
            "uniform rank. V and the KV cache remain dense. Factors are folded "
            "into equivalent BF16 `o_proj` weights for quality isolation."
        ),
        "",
        "| Arm | WikiText-2 PPL | Change vs dense | Heldout output MSE |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["arms"][arm]
        ppl = float(row["wikitext2"]["ppl"])
        heldout = row["mean_heldout_output_relative_mse"]
        heldout_text = "—" if heldout is None else f"{float(heldout):.8g}"
        lines.append(
            f"| `{arm}` | {ppl:.8f} | {(ppl / dense - 1.0) * 100:+.3f}% | "
            f"{heldout_text} |"
        )
    lines.extend(
        [
            "",
            "Protocol:",
            "",
            "- WikiText-2 test, concatenated non-overlapping 2048-token chunks.",
            "- BF16 model execution, SDPA attention, FP32 loss accumulation.",
            "- Joint and independent-local C1 use the same private-AllGather wire.",
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(args.seqlen, args.batch_size, args.torch_num_threads) <= 0:
        raise ValueError(
            "sequence length, batch size, and thread count must be positive"
        )
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("uniform-rank quality evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = args.model.expanduser().resolve()
    phase1_dir = args.phase1_dir.expanduser().resolve()
    local_dir = args.independent_local_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    phase1, phase1_records = _validate_phase1(phase1_dir, model_path=model_path)
    local, local_records = _validate_independent_local(
        local_dir,
        phase1_dir=phase1_dir,
        phase1=phase1,
        model_path=model_path,
    )
    output_dir.mkdir(parents=True)

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"[Uniform quality] loading model={model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=args.local_files_only, use_fast=True
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

    c1_mse = float(phase1["summary"]["mean_c1_heldout_relative_mse"])
    local_mse = float(
        local["aggregate"]["independent_local_bfloat16"][
            "mean_final_heldout_relative_mse"
        ]
    )
    wire_mse = float(phase1["summary"]["mean_wire_lr_ar_heldout_relative_mse"])
    capacity_mse = float(phase1["summary"]["mean_capacity_lr_ar_heldout_relative_mse"])
    heldout_by_arm = {
        "dense": None,
        "joint_c1": c1_mse,
        "independent_local_c1": local_mse,
        "wire_matched_lr_allreduce": wire_mse,
        "capacity_matched_lr_allreduce": capacity_mse,
    }
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        installed = _install_arm(
            model,
            arm=arm,
            phase1_dir=phase1_dir,
            phase1_records=phase1_records,
            local_dir=local_dir,
            local_records=local_records,
        )
        quality = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset=args.dataset,
            split=args.split,
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
        )
        accounting_arm = (
            "wo_c1_ag" if arm == "independent_local_c1" else PHASE_ARM.get(arm, arm)
        )
        arms[arm] = {
            "quality_representation": (
                "unaltered_dense_model"
                if arm == "dense"
                else "dense_reconstruction_of_collective_linear_map"
            ),
            "collective_accounting": _accounting(accounting_arm, phase1_records),
            "mean_heldout_output_relative_mse": heldout_by_arm[arm],
            "installation": installed,
            "wikitext2": quality,
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
        print(f"[Uniform quality] arm={arm} ppl={quality['ppl']:.9f}", flush=True)
        torch.cuda.empty_cache()

    dense_ppl = float(arms["dense"]["wikitext2"]["ppl"])
    for row in arms.values():
        row["relative_to_dense"] = {
            "wikitext2_ppl_change": (float(row["wikitext2"]["ppl"]) / dense_ppl - 1.0)
        }
    source_rank = int(phase1["method"]["run_signature"]["source_rank"])
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join((sys.executable, *sys.argv)),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "dtype": str(model_dtype),
        },
        "sources": {
            "phase1_results": str(phase1_dir / "results.json"),
            "phase1_results_sha256": _sha256(phase1_dir / "results.json"),
            "independent_local_results": str(local_dir / "results.json"),
            "independent_local_results_sha256": _sha256(local_dir / "results.json"),
        },
        "protocol": {
            "arms": list(ARMS),
            "source_rank": source_rank,
            "source_width": 1024,
            "retained_ratio": source_rank / 1024.0,
            "scope": "post-attention W_o only; V and KV cache remain dense",
            "dataset": args.dataset,
            "split": args.split,
            "sequence_length": args.seqlen,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
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
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / "results.json", payload)
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    print(f"[Uniform quality] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
