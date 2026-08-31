#!/usr/bin/env python3
"""Evaluate Experiment-J Wo-C1 covariance arms on WikiText-2."""

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
from evaluation.fit_qwen3_8b_wo_c1_cross_source_covariance import (  # noqa: E402
    FORMAT as FACTOR_FORMAT,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_cross_source_covariance_ppl.v1"
ARMS = ("dense", "full_covariance", "block_diagonal")
DECODER_KEYS = {
    "full_covariance": "full_source_decoders",
    "block_diagonal": "block_diagonal_source_decoders",
}
EXPECTED_FACTOR_KEYS = {
    "fixed_source_encoders",
    *DECODER_KEYS.values(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
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


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _validate_factor_bank(
    factor_dir: Path,
    *,
    model_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    results_path = factor_dir / "results.json"
    results = _load_json(results_path)
    if results.get("format") != FACTOR_FORMAT or results.get("status") != "complete":
        raise ValueError("Experiment-J factor bank is incomplete or incompatible")
    if results["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("model config differs from the Experiment-J factor bank")
    protocol = results["protocol"]
    if (
        int(protocol["tp_size"]) != 4
        or int(protocol["source_rank"]) != 512
        or protocol["quality_scope"]
        != "attention layer output; dense V and dense KV cache"
    ):
        raise ValueError("factor bank is not the audited TP4 Wo-only Experiment J")
    records: dict[int, dict[str, Any]] = {}
    for row in results["layers"]:
        layer = int(row["layer"])
        if layer in records:
            raise ValueError(f"duplicate Experiment-J layer {layer}")
        artifact_path = factor_dir / row["artifact"]["file"]
        if _sha256(artifact_path) != row["artifact"]["sha256"]:
            raise ValueError(f"factor hash mismatch at layer {layer}")
        records[layer] = row
    expected_layers = tuple(range(int(results["model"]["num_hidden_layers"])))
    if tuple(sorted(records)) != expected_layers:
        raise ValueError("Experiment-J factors do not cover every decoder layer")
    for source_key, hash_key in (
        ("joint_phase1_dir", "joint_phase1_results_sha256"),
        ("independent_local_dir", "independent_local_results_sha256"),
        ("covariance_dir", "covariance_manifest_sha256"),
    ):
        source_dir = Path(results["source"][source_key]).expanduser().resolve()
        filename = "manifest.json" if source_key == "covariance_dir" else "results.json"
        source_path = source_dir / filename
        if _sha256(source_path) != results["source"][hash_key]:
            raise ValueError(f"Experiment-J source changed: {source_path}")
    return results, records


def materialize_private_weight(
    payload: Mapping[str, Tensor],
    *,
    decoder_key: str,
    device: torch.device,
) -> Tensor:
    """Materialize one private-source linear map as a dense FP32 weight."""

    if set(payload) != EXPECTED_FACTOR_KEYS:
        raise ValueError("Experiment-J artifact tensors differ")
    if decoder_key not in DECODER_KEYS.values():
        raise ValueError(f"unsupported Experiment-J decoder key: {decoder_key}")
    encoders = payload["fixed_source_encoders"].to(device=device, dtype=torch.float32)
    decoders = payload[decoder_key].to(device=device, dtype=torch.float32)
    if encoders.ndim != 3 or decoders.ndim != 3:
        raise ValueError("private factors must be rank-three tensors")
    sources, source_width, source_rank = map(int, encoders.shape)
    if tuple(decoders.shape[:2]) != (sources, source_rank):
        raise ValueError("private encoder and decoder geometry differ")
    output_width = int(decoders.shape[2])
    return (
        torch.bmm(encoders, decoders)
        .reshape(sources * source_width, output_width)
        .transpose(0, 1)
        .contiguous()
    )


@torch.inference_mode()
def _install_arm(
    model: nn.Module,
    *,
    arm: str,
    factor_dir: Path,
    records: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if arm == "dense":
        return []
    decoder_key = DECODER_KEYS[arm]
    layers = _decoder_layers(model)
    if len(layers) != len(records):
        raise ValueError("loaded model and Experiment-J layer counts differ")
    installed = []
    for layer, module_wrapper in enumerate(layers):
        started = time.perf_counter()
        record = records[layer]
        artifact_path = factor_dir / record["artifact"]["file"]
        payload = load_file(str(artifact_path), device="cpu")
        module = module_wrapper.self_attn.o_proj
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise TypeError(f"unsupported o_proj module at layer {layer}")
        reconstructed = materialize_private_weight(
            payload,
            decoder_key=decoder_key,
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
                "encoder_key": "fixed_source_encoders",
                "decoder_key": decoder_key,
                "installed_dtype": str(module.weight.dtype),
                "seconds": time.perf_counter() - started,
            }
        )
        del payload, reconstructed
        torch.cuda.empty_cache()
        print(f"[J PPL] install arm={arm} layer={layer}/35", flush=True)
    return installed


def _accounting(arm: str) -> dict[str, Any]:
    if arm == "dense":
        return {
            "collective": "all_reduce",
            "ideal_ring_bytes_per_rank_per_activation_row": 12288,
            "tp_size": 4,
            "source_rank": None,
        }
    return {
        "collective": "private_all_gather",
        "ideal_ring_bytes_per_rank_per_activation_row": 3072,
        "tp_size": 4,
        "source_rank": 512,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    arms = payload["arms"]
    dense = float(arms["dense"]["wikitext2"]["ppl"])
    full = float(arms["full_covariance"]["wikitext2"]["ppl"])
    block = float(arms["block_diagonal"]["wikitext2"]["ppl"])
    lines = [
        "# Qwen3-8B Wo-C1 cross-source covariance WikiText-2 PPL",
        "",
        (
            "The two compressed arms use identical stored BF16 joint-s20 "
            "encoders and identical private-AllGather communication. Only the "
            "decoder covariance model differs. Factors are folded into equivalent "
            "dense BF16 `o_proj` weights, isolating quality from runtime kernels."
        ),
        "",
        "| Arm | WikiText-2 PPL | Change vs dense |",
        "|---|---:|---:|",
        f"| Dense | {dense:.8f} | +0.000% |",
        f"| Full covariance | {full:.8f} | {(full / dense - 1.0) * 100:+.3f}% |",
        f"| Block diagonal | {block:.8f} | {(block / dense - 1.0) * 100:+.3f}% |",
        "",
        (
            "Full covariance change relative to block diagonal: "
            f"`{(full / block - 1.0) * 100:+.3f}%` PPL."
        ),
        "",
        "Protocol:",
        "",
        (
            f"- `{payload['protocol']['dataset']}` "
            f"`{payload['protocol']['split']}`, concatenated non-overlapping "
            f"{payload['protocol']['sequence_length']}-token chunks."
        ),
        (
            f"- {payload['protocol']['model_dtype'].upper()} model execution, "
            f"{payload['protocol']['attn_implementation'].upper()} attention, "
            "FP32 cross-entropy accumulation."
        ),
        "- V and the KV cache remain dense for every arm.",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
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
        raise RuntimeError("whole-model PPL evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = args.model.expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    factor_results, records = _validate_factor_bank(
        factor_dir,
        model_path=model_path,
    )
    output_dir.mkdir(parents=True)
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"[J PPL] loading model={model_path} device={device}", flush=True)
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

    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        installed = _install_arm(model, arm=arm, factor_dir=factor_dir, records=records)
        wikitext = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset=args.dataset,
            split=args.split,
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
        )
        arms[arm] = {
            "quality_representation": (
                "unaltered_dense_model"
                if arm == "dense"
                else "dense_reconstruction_of_private_collective_linear_map"
            ),
            "collective_accounting": _accounting(arm),
            "installation": installed,
            "wikitext2": wikitext,
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
        print(f"[J PPL] arm={arm} ppl={wikitext['ppl']:.9f}", flush=True)
        torch.cuda.empty_cache()

    dense_ppl = float(arms["dense"]["wikitext2"]["ppl"])
    for row in arms.values():
        row["relative_to_dense"] = {
            "wikitext2_ppl_change": (float(row["wikitext2"]["ppl"]) / dense_ppl - 1.0)
        }
    factor_results_path = factor_dir / "results.json"
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
        "factors": {
            "path": str(factor_results_path),
            "sha256": _sha256(factor_results_path),
            "format": factor_results["format"],
            "protocol": factor_results["protocol"],
            "aggregate_attention_output_metrics": factor_results["aggregate"],
        },
        "protocol": {
            "arms": list(ARMS),
            "scope": "post-attention W_o only; V and KV cache remain dense",
            "dataset": args.dataset,
            "split": args.split,
            "sequence_length": args.seqlen,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
            "cross_document_transitions": True,
            "model_dtype": args.model_dtype,
            "attn_implementation": args.attn_implementation,
            "loss_dtype": "float32",
        },
        "arms": arms,
        "comparison": {
            "full_covariance_ppl_change_vs_block_diagonal": (
                float(arms["full_covariance"]["wikitext2"]["ppl"])
                / float(arms["block_diagonal"]["wikitext2"]["ppl"])
                - 1.0
            )
        },
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
    print(f"[J PPL] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
