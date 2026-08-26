#!/usr/bin/env python3
"""Evaluate Qwen3.5 MLP pre-down-projection Top-K on WikiText-2.

This is a quality oracle for sparse MLP compute.  It applies token-wise Top-K
to the exact SwiGLU product immediately before every ``mlp.down_proj``.  The
selected activation is materialized as a dense zero-filled tensor and passed
to the original dense projection, so wall time is not a sparse-kernel result.
Standard row-parallel MLP output AllReduce traffic is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Mapping

from safetensors.torch import save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers  # noqa: E402
from basisserve.core.qwen35_postgate_topk import (  # noqa: E402
    Qwen35PostGateTopKRuntime,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.eval_qwen35_postgate_topk_crossdomain import (  # noqa: E402
    _file_sha256,
    _serializable_ppl,
    _wikitext_samples,
)
from scripts.eval_qwen35_projected_gdn_nll import _dtype, _evaluate  # noqa: E402


FORMAT = "basisserve.qwen35.mlp_predown_topk_wikitext.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wikitext-file", required=True)
    parser.add_argument("--keep-ratios", default="0.5,0.75")
    parser.add_argument(
        "--score",
        choices=("magnitude", "output_weighted"),
        default="magnitude",
    )
    parser.add_argument(
        "--selection-scope",
        choices=("global", "source_local"),
        default="source_local",
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--ppl-sequence-length", type=int, default=512)
    parser.add_argument("--ppl-max-tokens", type=int, default=8192)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ratios(raw: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(piece.strip()) for piece in raw.split(",") if piece.strip()})
    )
    if not values or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in values
    ):
        raise ValueError("keep ratios must be finite values in (0,1)")
    return values


def _ratio_tag(ratio: float) -> str:
    return f"{ratio * 100:.6g}".replace(".", "p")


def _variant_name(ratio: float, score: str, selection_scope: str) -> str:
    return f"mlp_topk_{_ratio_tag(ratio)}_{score}_{selection_scope}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    dense = payload["ppl"]["dense"]
    geometry = payload["geometry"]
    lines = [
        "# Qwen3.5 MLP pre-down Top-K WikiText-2 pilot",
        "",
        (
            "Top-K is applied to the exact SwiGLU product immediately before every "
            "MLP down projection. The oracle zero-fills and executes the original "
            "dense projection. Potential MAC reduction requires a sparse kernel; "
            "standard row-parallel output AllReduce traffic is unchanged."
        ),
        "",
        "| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained energy | Potential down MAC reduction | AllReduce reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| dense | {geometry['local_intermediate_width']} | "
            f"{dense['perplexity']:.7g} | 0 | 0 | 0 | 1 | 1 | 0% | 0% |"
        ),
    ]
    metadata = {row["name"]: row for row in payload["variants"]}
    for name, result in payload["ppl"]["candidates"].items():
        row = metadata[name]
        ppl_delta = result["perplexity"] / dense["perplexity"] - 1.0
        lines.append(
            f"| {name} | {row['kept_per_source']} | {result['perplexity']:.7g} | "
            f"{ppl_delta:+.3%} | {result['delta_mean_nll']:+.7g} | "
            f"{result['paired_window_standard_error']:.7g} | "
            f"{result['top1_agreement']:.6f} | "
            f"{row['mean_retained_input_energy']:.6f} | "
            f"{row['potential_down_projection_mac_reduction']:.1%} | 0% |"
        )
    lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    ratios = _ratios(args.keep_ratios)
    if args.tp_size <= 0 or args.torch_num_threads <= 0:
        raise ValueError("TP size and thread count must be positive")
    if (
        args.ppl_sequence_length <= 1
        or args.ppl_max_tokens < args.ppl_sequence_length
        or args.ppl_batch_size <= 0
    ):
        raise ValueError("invalid WikiText evaluation controls")

    model_path = Path(args.model_path).expanduser().resolve()
    wikitext_path = Path(args.wikitext_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if not wikitext_path.is_file():
        raise FileNotFoundError(wikitext_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite MLP Top-K result: {output_dir}")

    plan = {
        "model": str(model_path),
        "wikitext": str(wikitext_path),
        "variants": [
            "dense",
            *[_variant_name(ratio, args.score, args.selection_scope) for ratio in ratios],
        ],
        "tp_size": args.tp_size,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("Qwen3.5 MLP Top-K evaluation requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
    )
    samples, wikitext_metadata = _wikitext_samples(
        tokenizer,
        wikitext_path,
        sequence_length=args.ppl_sequence_length,
        maximum_tokens=args.ppl_max_tokens,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    ).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers = qwen35_decoder_layers(model)
    projections = [getattr(getattr(layer, "mlp", None), "down_proj", None) for layer in layers]
    if not projections or any(not isinstance(projection, torch.nn.Linear) for projection in projections):
        raise TypeError("every decoder layer must expose a linear MLP down projection")
    widths = {int(projection.in_features) for projection in projections}
    output_widths = {int(projection.out_features) for projection in projections}
    if len(widths) != 1 or len(output_widths) != 1:
        raise ValueError("MLP geometry is not uniform across decoder layers")
    intermediate_width = widths.pop()
    hidden_width = output_widths.pop()
    if intermediate_width % args.tp_size:
        raise ValueError("MLP intermediate width is not divisible by TP size")
    local_width = intermediate_width // args.tp_size

    dense_raw = _evaluate(
        model,
        samples,
        batch_size=args.ppl_batch_size,
        device=args.device,
        label="dense",
    )
    dense = _serializable_ppl(dense_raw, baseline=None)
    candidates: dict[str, Any] = {}
    variant_metadata: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    frequencies: dict[str, torch.Tensor] = {}
    for index, ratio in enumerate(ratios, start=1):
        name = _variant_name(ratio, args.score, args.selection_scope)
        print(f"[Variant] {index}/{len(ratios)} name={name}", flush=True)
        with Qwen35PostGateTopKRuntime(
            model,
            intervention="mlp",
            keep_ratio=ratio,
            selection_scope=args.selection_scope,
            score=args.score,
            tp_size=args.tp_size,
            profile=True,
        ) as runtime:
            if len(runtime.records) != len(layers):
                raise RuntimeError("MLP Top-K runtime did not cover every decoder layer")
            runtime.reset_profile()
            candidate_raw = _evaluate(
                model,
                samples,
                batch_size=args.ppl_batch_size,
                device=args.device,
                label=name,
            )
            candidates[name] = _serializable_ppl(candidate_raw, baseline=dense_raw)
            summaries, tensors = runtime.profile_snapshot()
            profiles.extend({"variant": name, **row} for row in summaries)
            frequencies.update(
                {f"wikitext2__{name}__{key}": value for key, value in tensors.items()}
            )
            kept_per_source = runtime.records[0].kept_per_source
            realized_ratio = runtime.records[0].realized_ratio
            if args.selection_scope == "source_local" and kept_per_source is None:
                raise AssertionError("source-local runtime did not report local K")
            variant_metadata.append(
                {
                    "name": name,
                    "keep_ratio": ratio,
                    "realized_ratio": realized_ratio,
                    "selection_scope": args.selection_scope,
                    "score": args.score,
                    "layer_count": len(runtime.records),
                    "kept_per_vector": runtime.records[0].kept_per_vector,
                    "kept_per_source": kept_per_source,
                    "mean_retained_input_energy": statistics.fmean(
                        float(row["retained_input_energy"]) for row in summaries
                    ),
                    "estimated_down_projection_mac_fraction": realized_ratio,
                    "potential_down_projection_mac_reduction": 1.0 - realized_ratio,
                    "standard_tp_allreduce_fraction": 1.0,
                    "standard_tp_allreduce_reduction": 0.0,
                }
            )
            del candidate_raw
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True)
    frequency_path = output_dir / "selection_frequency.safetensors"
    temporary_frequency = frequency_path.with_suffix(".safetensors.tmp")
    save_file(frequencies, str(temporary_frequency))
    os.replace(temporary_frequency, frequency_path)
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _file_sha256(model_path / "config.json"),
            "weights_updated": False,
        },
        "geometry": {
            "layers": len(layers),
            "hidden_width": hidden_width,
            "intermediate_width": intermediate_width,
            "tp_size": args.tp_size,
            "local_intermediate_width": local_width,
            "dense_down_projection_macs_per_rank_per_token": local_width * hidden_width,
            "standard_row_parallel_collective": "hidden_width_allreduce",
            "standard_allreduce_elements_per_token": hidden_width,
        },
        "protocol": {
            "intervention_point": "post_swiglu_pre_down_projection",
            "quality_oracle_dense_zero_fill": True,
            "sparse_kernel_executed": False,
            "collective_modified": False,
            "standard_tp_communication_reduction": 0.0,
            "selection_scope": args.selection_scope,
            "score": args.score,
            "score_definition": (
                "abs_post_swiglu_activation"
                if args.score == "magnitude"
                else "abs_post_swiglu_activation_times_fp32_down_weight_column_l2"
            ),
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "variants": variant_metadata,
        "profiles": profiles,
        "selection_frequency": {
            "path": str(frequency_path),
            "sha256": _file_sha256(frequency_path),
            "tensor_count": len(frequencies),
        },
        "ppl": {
            "dataset": "wikitext2",
            "source": wikitext_metadata,
            "dense": dense,
            "candidates": candidates,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(output_dir / "results.json", payload)
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    print(f"[Done] elapsed={elapsed:.2f}s output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
