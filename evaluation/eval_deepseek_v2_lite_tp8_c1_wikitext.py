#!/usr/bin/env python3
"""Evaluate dense or TP8 post-attention C1 DeepSeek-V2-Lite PPL."""

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

from safetensors.torch import load_file
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.tp_source_wo_c1 import (  # noqa: E402
    TPSourceWOLayout,
    fold_factors_to_dense_weight,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation.fit_deepseek_v2_lite_tp8_c1_joint import (  # noqa: E402
    FORMAT as FACTOR_FORMAT,
    HIDDEN_SIZE,
    MODEL_TYPE,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    TP_SIZE,
    VALUE_HEAD_DIM,
)


FORMAT = "basisserve.deepseek_v2_lite.tp8_source_wo_c1_wikitext.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        raise AttributeError("could not locate DeepSeek decoder layers")
    return layers


def _validate_model(model: nn.Module) -> None:
    config = model.config
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(getattr(config, "v_head_dim", VALUE_HEAD_DIM)),
    )
    expected = (
        MODEL_TYPE,
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        VALUE_HEAD_DIM,
    )
    if observed != expected:
        raise ValueError(f"unexpected DeepSeek-V2-Lite geometry: {observed}")
    if len(_decoder_layers(model)) != NUM_LAYERS:
        raise ValueError("DeepSeek decoder layer count differs from config")


def _load_factor_result(factor_dir: Path, model_path: Path) -> dict[str, Any]:
    path = factor_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError("factor directory is not a completed DeepSeek C1 fit")
    if result["fit_config"]["model_config_sha256"] != _sha256(
        model_path / "config.json"
    ):
        raise ValueError("model config differs from fitted factors")
    if tuple(map(int, result["layers"])) != tuple(range(NUM_LAYERS)):
        raise ValueError("factor result does not cover every decoder layer")
    return result


@torch.no_grad()
def _install_factors(
    model: nn.Module,
    factor_dir: Path,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_rank = int(result["fit_config"]["source_rank"])
    layout = TPSourceWOLayout(
        input_width=NUM_ATTENTION_HEADS * VALUE_HEAD_DIM,
        output_width=HIDDEN_SIZE,
        tp_size=TP_SIZE,
        source_rank=source_rank,
        dtype_bytes=2,
    )
    records = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        artifact = result["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"factor hash mismatch at layer {layer_index}")
        factors = load_file(str(path), device="cpu")
        o_proj = layer.self_attn.o_proj
        if getattr(o_proj, "bias", None) is not None or tuple(o_proj.weight.shape) != (
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        ):
            raise TypeError(f"unsupported o_proj at layer {layer_index}")
        device = o_proj.weight.device
        encoders = factors["source_encoders"].to(device=device, dtype=torch.float32)
        decoders = factors["source_decoders"].to(device=device, dtype=torch.float32)
        folded = fold_factors_to_dense_weight(encoders, decoders, layout)
        if not bool(torch.isfinite(folded).all()):
            raise FloatingPointError(f"non-finite folded weight at layer {layer_index}")
        o_proj.weight.copy_(folded.to(dtype=o_proj.weight.dtype))
        records.append(
            {
                "layer": layer_index,
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "encoder_shape": list(encoders.shape),
                "decoder_shape": list(decoders.shape),
                "runtime": (
                    "factors folded into dense o_proj for quality-equivalent "
                    "evaluation; MLA and KV cache unchanged"
                ),
            }
        )
        del factors, encoders, decoders, folded
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir")
    parser.add_argument("--output-json", required=True)
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
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("balanced", "balanced_low_0", "auto"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DeepSeek PPL evaluation requires CUDA")
    torch.cuda.set_device(device)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factor_dir = (
        None
        if args.factor_dir is None
        else Path(args.factor_dir).expanduser().resolve()
    )
    factor_result = (
        None
        if factor_dir is None
        else _load_factor_result(factor_dir, model_path)
    )
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"DeepSeek model offloaded parameters: {placement}")
    _validate_model(model)
    install_started = time.perf_counter()
    installation = (
        []
        if factor_result is None or factor_dir is None
        else _install_factors(model, factor_dir, factor_result)
    )
    install_seconds = time.perf_counter() - install_started
    metrics = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    compression = None
    factor_record = None
    if factor_result is not None and factor_dir is not None:
        config = factor_result["fit_config"]
        compression = {
            "target": "post_attention_tp_source_wo_allgather",
            "kv_cache_compression": "none",
            "tp_size": TP_SIZE,
            "source_width": int(config["source_width"]),
            "source_rank": int(config["source_rank"]),
            "communication": config["communication"],
        }
        result_path = factor_dir / "results.json"
        factor_record = {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "format": factor_result["format"],
            "aggregate": factor_result["aggregate"],
        }
    payload = {
        "format": FORMAT,
        "status": "complete",
        "arm": "dense" if factor_result is None else "tp8_source_wo_c1_als",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "factor_result": factor_record,
        "compression": compression,
        "quality_reference_runtime": {
            "description": (
                "dense o_proj folding preserves the C1 linear function for PPL; "
                "it does not benchmark the compressed AllGather runtime"
            ),
            "installation": installation,
            "installation_seconds": install_seconds,
        },
        "ppl": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "device_map": placement,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, payload)
    print(
        f"[DeepSeek PPL] arm={payload['arm']} ppl={metrics['ppl']:.9f} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
