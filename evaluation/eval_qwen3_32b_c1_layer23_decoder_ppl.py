#!/usr/bin/env python3
"""Evaluate layer-23 TSQR decoder overrides in a full Qwen3-32B C1 model.

The model is loaded once.  Dense WikiText-2 PPL is measured first, the stored
uniform C1 checkpoint is installed second, and QR(0) decoders reconstructed
from streaming-TSQR checkpoints then replace only layer 23.  The padded V/O
path is function-equivalent to C1 but intentionally does not benchmark the
compressed-cache runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.pairwise_qr import (  # noqa: E402
    solve_square_root_least_squares,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    _load_results,
    install_c1_factors,
)
from evaluation.run_qwen3_32b_c1_layer23_tsqr_scaling import (  # noqa: E402
    FORMAT as CAPTURE_FORMAT,
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER,
    NUM_QUERY_HEADS,
    QUERY_WIDTH,
    _dense_block_basis,
    _load_fixed_factors,
    _load_packed_r,
    _sha256,
    _validated_artifact_path,
)


FORMAT = "basisserve.qwen3_32b.c1_layer23_decoder_ppl.v1"
DEFAULT_MILESTONES = (32_768, 65_536)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=DEFAULT_MILESTONES,
    )
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-column-chunk-size", type=int, default=256)
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen-style decoder model")
    return layers


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


@torch.no_grad()
def decoder_to_padded_o_weight(
    decoder: Tensor,
    *,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Pad a headwise C1 decoder into the dense Hugging Face O projection."""

    if decoder.ndim != 3:
        raise ValueError("decoder must have shape [heads, rank, hidden_size]")
    heads, rank, hidden_size = map(int, decoder.shape)
    if heads <= 0 or rank <= 0 or rank > head_dim or hidden_size <= 0:
        raise ValueError("decoder has invalid C1 geometry")
    weight = torch.zeros(
        hidden_size,
        heads * head_dim,
        device=device,
        dtype=dtype,
    )
    source = decoder.to(device=device, dtype=dtype)
    for head in range(heads):
        start = head * head_dim
        weight[:, start : start + rank].copy_(source[head].T)
    return weight


@torch.no_grad()
def _solve_decoder_overrides(
    *,
    capture_dir: Path,
    factor_dir: Path,
    milestones: Sequence[int],
    device: torch.device,
    output_column_chunk_size: int,
) -> tuple[dict[int, Tensor], dict[int, dict[str, Any]], dict[str, Any]]:
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CAPTURE_FORMAT:
        raise ValueError("incompatible streaming TSQR capture")
    recorded = tuple(map(int, manifest["calibration"]["fit_milestones"]))
    selected = tuple(map(int, milestones))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("--milestones must contain distinct values")
    if any(rows not in recorded for rows in selected):
        raise ValueError("requested milestone is absent from the TSQR capture")

    fixed_a_cpu, _, factor_source = _load_fixed_factors(
        factor_dir,
        model_config_sha256=str(manifest["model"]["config_sha256"]),
    )
    weight_record = manifest["artifacts"]["weight"]
    weight_payload = load_file(
        str(_validated_artifact_path(capture_dir, weight_record)),
        device="cpu",
    )
    if set(weight_payload) != {"weight"}:
        raise ValueError("unexpected target-weight artifact tensors")
    weight = weight_payload["weight"]
    if tuple(weight.shape) != (HIDDEN_SIZE, QUERY_WIDTH):
        raise ValueError("target weight has incompatible shape")

    solve_dtype = torch.float64
    fixed_a = fixed_a_cpu.to(device=device, dtype=solve_dtype)
    target_weight = weight.to(device=device, dtype=solve_dtype).T.contiguous()
    basis = _dense_block_basis(fixed_a, device=device)
    decoders: dict[int, Tensor] = {}
    records: dict[int, dict[str, Any]] = {}
    fit_positions_per_window = int(
        manifest["calibration"]["fit_positions_per_window"]
    )
    for rows in selected:
        fit_record = manifest["artifacts"]["fit"][str(rows)]
        fit_r = _load_packed_r(
            _validated_artifact_path(capture_dir, fit_record),
            device=device,
            dtype=solve_dtype,
        )
        decoder_flat, diagnostics = solve_square_root_least_squares(
            left_factor=fit_r,
            basis=basis,
            target_weight=target_weight,
            absolute_product_damping=0.0,
            output_chunk_size=output_column_chunk_size,
        )
        rank = int(fixed_a.shape[2])
        decoder = (
            decoder_flat.reshape(NUM_QUERY_HEADS, rank, HIDDEN_SIZE)
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        if not bool(torch.isfinite(decoder).all()):
            raise FloatingPointError(f"non-finite BF16 decoder at {rows} rows")
        decoders[rows] = decoder
        records[rows] = {
            "fit_rows": rows,
            "fit_windows": rows // fit_positions_per_window,
            "decoder_shape": list(decoder.shape),
            "decoder_dtype": str(decoder.dtype),
            "decoder_sha256": _tensor_sha256(decoder),
            "solve": asdict(diagnostics),
            "fit_r": {
                "file": fit_record["file"],
                "sha256": fit_record["sha256"],
            },
        }
        del fit_r, decoder_flat
    source = {
        "capture_manifest": str(manifest_path),
        "capture_manifest_sha256": _sha256(manifest_path),
        "factor": factor_source,
        "solver": "pairwise_tsqr_square_root_qr_lambda_0",
        "solve_dtype": "float64",
        "installed_decoder_dtype": "bfloat16",
    }
    return decoders, records, source


def _arm_record(name: str, ppl: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "ppl": dict(ppl),
        "mean_nll": float(ppl["nll_sum"]) / int(ppl["tokens"]),
    }


def _add_comparisons(arms: list[dict[str, Any]]) -> None:
    by_name = {str(arm["name"]): arm for arm in arms}
    dense = float(by_name["dense"]["ppl"]["ppl"])
    ridge = float(by_name["c1_ridge_d1e5"]["ppl"]["ppl"])
    for arm in arms:
        value = float(arm["ppl"]["ppl"])
        arm["ppl_delta_vs_dense"] = value - dense
        arm["ppl_relative_change_vs_dense"] = value / dense - 1.0
        arm["ppl_delta_vs_ridge"] = value - ridge
        arm["ppl_relative_change_vs_ridge"] = value / ridge - 1.0


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-32B C1 layer-23 TSQR decoder PPL",
        "",
        "The model is loaded once. Dense PPL is measured before installing the "
        "uniform rank-96 C1 checkpoint; QR(0) arms then replace only the layer-23 "
        "decoder. The padded Hugging Face V/O path measures function quality, not "
        "compressed-cache performance.",
        "",
        "## Results",
        "",
        "| Arm | PPL | Mean NLL | ΔPPL vs dense | ΔPPL vs ridge |",
        "|:---|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        lines.append(
            f"| {arm['name']} | {arm['ppl']['ppl']:.9f} | "
            f"{arm['mean_nll']:.9f} | {arm['ppl_delta_vs_dense']:+.9f} | "
            f"{arm['ppl_delta_vs_ridge']:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            str(payload["command"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("layer-23 decoder PPL evaluation requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model_path = Path(args.model).expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    capture_dir = args.capture_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    for output in (output_json, output_markdown):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    milestones = tuple(map(int, args.milestones))

    started = time.perf_counter()
    print(f"[Solve] milestones={milestones} dtype=float64", flush=True)
    decoders, decoder_records, decoder_source = _solve_decoder_overrides(
        capture_dir=capture_dir,
        factor_dir=factor_dir,
        milestones=milestones,
        device=device,
        output_column_chunk_size=args.output_column_chunk_size,
    )
    solve_seconds = time.perf_counter() - started

    factor_result = _load_results(factor_dir, model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    print(f"[Load] model={model_path} device_map={args.device_map}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    eval_kwargs = {
        "dataset": args.dataset,
        "split": args.split,
        "seqlen": args.seqlen,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
    }
    arms: list[dict[str, Any]] = []
    print("[Arm] dense", flush=True)
    arms.append(
        _arm_record(
            "dense",
            _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs),
        )
    )

    print("[Install] uniform C1 ridge checkpoint", flush=True)
    installation = install_c1_factors(model, factor_dir, factor_result)
    print("[Arm] c1_ridge_d1e5", flush=True)
    arms.append(
        _arm_record(
            "c1_ridge_d1e5",
            _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs),
        )
    )

    layer = _decoder_layers(model)[LAYER]
    o_proj = layer.self_attn.o_proj
    if not isinstance(o_proj, nn.Linear) or o_proj.weight.device.type != "cuda":
        raise TypeError("layer-23 o_proj is not a CUDA linear projection")
    for rows in milestones:
        print(f"[Arm] c1_qr0_{rows}", flush=True)
        padded_o = decoder_to_padded_o_weight(
            decoders[rows],
            head_dim=HEAD_DIM,
            device=o_proj.weight.device,
            dtype=o_proj.weight.dtype,
        )
        o_proj.weight.copy_(padded_o)
        del padded_o
        arms.append(
            _arm_record(
                f"c1_qr0_{rows}",
                _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs),
            )
        )
    _add_comparisons(arms)

    result_path = factor_dir / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "configuration": {
            "layer": LAYER,
            "milestones": list(milestones),
            "dataset": args.dataset,
            "split": args.split,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
            "model_dtype": args.model_dtype,
            "loss_dtype": "float32",
            "attn_implementation": args.attn_implementation,
            "quality_runtime": "padded_huggingface_v_o",
        },
        "factor_result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "format": factor_result["format"],
        },
        "decoder_source": decoder_source,
        "decoder_overrides": {
            str(rows): decoder_records[rows] for rows in milestones
        },
        "arms": arms,
        "timing": {
            "solve_seconds": solve_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV")
            or Path(sys.prefix).name,
            "python_executable": sys.executable,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "installation": installation,
    }
    _atomic_json(output_json, payload)
    _atomic_text(output_markdown, _render_markdown(payload))
    for arm in arms:
        print(f"[Result] {arm['name']} ppl={arm['ppl']['ppl']:.9f}", flush=True)
    print(f"[Result] wrote {output_json}", flush=True)
    print(f"[Result] wrote {output_markdown}", flush=True)


if __name__ == "__main__":
    main()
