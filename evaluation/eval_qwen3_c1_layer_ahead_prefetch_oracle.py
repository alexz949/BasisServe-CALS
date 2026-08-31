#!/usr/bin/env python3
"""Evaluate InfiniGen-style layer-ahead exact-K page prefetch for Qwen3-8B."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_reverse_shadow import (  # noqa: E402
    ReverseShadowConfig,
    build_post_rope_k_landmarks,
    c1_k_reverse_shadow_attention,
)
from basisserve.core.c1_v_k_index import relative_squared_error  # noqa: E402
from basisserve.core.layer_ahead_prefetch import (  # noqa: E402
    apply_headwise_linear,
    fit_headwise_linear,
    group_query_heads,
    prefetch_set_statistics,
    ungroup_query_heads,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.eval_qwen3_c1_k_refine_oracle import _sha256  # noqa: E402
from evaluation.fit_qwen3_8b_c1_k_output_closure import (  # noqa: E402
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    LayerFeatureFactory,
    _batches,
    _load_layer_c1_factors,
    _propagate_dense_layer,
    _validate_config,
)


FORMAT = "basisserve.qwen3_8b.c1_layer_ahead_prefetch_oracle.v2"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _mean_cosine(reference: Tensor, candidate: Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            reference.float(), candidate.float(), dim=-1
        ).mean()
    )


def _runtime_rotary(
    query: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Use the same GPU/dtype RoPE path as the source Qwen3 capture."""

    runtime_query = query.to(device=device, dtype=dtype)
    runtime_cos = cos.to(device=device, dtype=dtype)
    runtime_sin = sin.to(device=device, dtype=dtype)
    rotated, _ = apply_rotary_pos_emb(
        runtime_query,
        runtime_query,
        runtime_cos,
        runtime_sin,
    )
    return rotated


@torch.inference_mode()
def _pre_rope_query(layer: nn.Module, hidden_states: Tensor) -> Tensor:
    attention = layer.self_attn
    hidden_states = hidden_states.to(
        device=attention.q_proj.weight.device,
        dtype=attention.q_proj.weight.dtype,
        non_blocking=True,
    )
    attention_input = layer.input_layernorm(hidden_states)
    batch, sequence, _ = map(int, attention_input.shape)
    return attention.q_norm(
        attention.q_proj(attention_input).view(
            batch,
            sequence,
            NUM_QUERY_HEADS,
            HEAD_DIM,
        )
    ).transpose(1, 2)


@torch.inference_mode()
def _capture_predictors(
    *,
    model: nn.Module,
    hidden_bank: Tensor,
    c1_dir: Path,
    c1_result: dict[str, Any],
    value_rank: int,
    target_layers: list[int],
    batch_size: int,
    position_ids: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> dict[int, dict[str, Tensor]]:
    targets = set(target_layers)
    features: dict[int, dict[str, Tensor]] = {layer: {} for layer in target_layers}
    maximum_target = max(target_layers)
    for layer_index, layer in enumerate(model.model.layers):
        next_layer_index = layer_index + 1
        if next_layer_index in targets:
            encoder, decoder, _ = _load_layer_c1_factors(
                c1_dir, c1_result, layer_index, value_rank
            )
            factory = LayerFeatureFactory(
                layer,
                value_encoder=encoder,
                decoder=decoder,
                value_rank=value_rank,
                position_embeddings=position_embeddings,
            )
            rehearsal_chunks = []
            c1_chunks = []
            for _, hidden in _batches(hidden_bank, 0, len(hidden_bank), batch_size):
                rehearsal_chunks.append(
                    _pre_rope_query(model.model.layers[next_layer_index], hidden).cpu()
                )
                _, _, previous_c1, _ = factory(hidden)
                c1_chunks.append(previous_c1.cpu())
            features[next_layer_index]["input_rehearsal_pre_query"] = torch.cat(
                rehearsal_chunks
            ).contiguous()
            features[next_layer_index]["previous_c1_value"] = torch.cat(
                c1_chunks
            ).contiguous()
            del factory, encoder, decoder, rehearsal_chunks, c1_chunks

        if layer_index in targets:
            encoder, decoder, _ = _load_layer_c1_factors(
                c1_dir, c1_result, layer_index, value_rank
            )
            factory = LayerFeatureFactory(
                layer,
                value_encoder=encoder,
                decoder=decoder,
                value_rank=value_rank,
                position_embeddings=position_embeddings,
            )
            actual_chunks = []
            actual_post_chunks = []
            exact_key_chunks = []
            target_c1_chunks = []
            for _, hidden in _batches(hidden_bank, 0, len(hidden_bank), batch_size):
                actual_chunks.append(_pre_rope_query(layer, hidden).cpu())
                grouped_query, exact_key, target_c1, _ = factory(hidden)
                actual_post_chunks.append(
                    grouped_query.reshape(
                        len(hidden),
                        NUM_QUERY_HEADS,
                        grouped_query.shape[-2],
                        grouped_query.shape[-1],
                    ).cpu()
                )
                exact_key_chunks.append(exact_key.cpu())
                target_c1_chunks.append(target_c1.cpu())
            features[layer_index]["actual_pre_query"] = torch.cat(
                actual_chunks
            ).contiguous()
            features[layer_index]["actual_post_query"] = torch.cat(
                actual_post_chunks
            ).contiguous()
            features[layer_index]["exact_key"] = torch.cat(
                exact_key_chunks
            ).contiguous()
            features[layer_index]["target_c1_value"] = torch.cat(
                target_c1_chunks
            ).contiguous()
            print(
                f"[Layer-ahead capture] target_layer={layer_index} "
                f"windows={len(hidden_bank)}",
                flush=True,
            )
            del (
                factory,
                encoder,
                decoder,
                actual_chunks,
                actual_post_chunks,
                exact_key_chunks,
                target_c1_chunks,
            )
            if layer_index == maximum_target:
                break

        _propagate_dense_layer(
            model,
            layer,
            hidden_bank,
            batch_size=batch_size,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            layer_index=layer_index,
        )
        torch.cuda.empty_cache()
    for target, tensors in features.items():
        expected = {
            "actual_pre_query",
            "actual_post_query",
            "exact_key",
            "input_rehearsal_pre_query",
            "previous_c1_value",
            "target_c1_value",
        }
        if set(tensors) != expected:
            raise RuntimeError(f"target layer {target} capture is incomplete")
    return features


def _query_diagnostics(
    actual: Tensor,
    predictions: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    return {
        policy: {
            "relative_mse": relative_squared_error(actual, predicted),
            "mean_cosine": _mean_cosine(actual, predicted),
        }
        for policy, predicted in predictions.items()
    }


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["policy"]),
            int(record["page_size"]),
            int(record["actual_token_budget"]),
            float(record["overfetch_factor"]),
        )
        grouped[key].append(record)
    result = []
    metric_names = {
        "mean_prefetch_page_recall": "prefetch_page_recall",
        "mean_prefetch_page_precision": "prefetch_page_precision",
        "mean_late_pages_per_kv_head": "mean_late_pages_per_kv_head",
        "mean_wasted_pages_per_kv_head": "mean_wasted_pages_per_kv_head",
        "mean_fully_covered_kv_head_fraction": "fully_covered_kv_head_fraction",
    }
    for (policy, page_size, budget, overfetch), rows in sorted(grouped.items()):
        result.append(
            {
                "policy": policy,
                "page_size": page_size,
                "actual_token_budget": budget,
                "overfetch_factor": overfetch,
                "observations": len(rows),
                **{
                    output_name: statistics.fmean(
                        float(row[source_name]) for row in rows
                    )
                    for output_name, source_name in metric_names.items()
                },
            }
        )
    return result


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B layer-ahead K-page prefetch oracle",
        "",
        "The final selector is exact-K QUEST with the actual query. Predicted "
        "queries affect prefetch timing only, so misses require late fetches but "
        "do not change model output.",
        "",
        "## Heldout query prediction diagnostics",
        "",
        "| layer | policy | pre-RoPE rel-MSE | cosine |",
        "|---:|:---|---:|---:|",
    ]
    for layer, policies in sorted(payload["query_diagnostics"].items(), key=lambda x: int(x[0])):
        for policy, metrics in sorted(policies.items()):
            lines.append(
                f"| {layer} | {policy} | {metrics['relative_mse']:.6e} | "
                f"{metrics['mean_cosine']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Prefetch set quality (mean over layers and query positions)",
            "",
            "| policy | page | actual tokens | overfetch | recall | precision | "
            "late pages/KV head | wasted pages/KV head | full coverage |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["summary"]:
        lines.append(
            f"| {row['policy']} | {row['page_size']} | "
            f"{row['actual_token_budget']} | {row['overfetch_factor']:.2f} | "
            f"{row['mean_prefetch_page_recall']:.6f} | "
            f"{row['mean_prefetch_page_precision']:.6f} | "
            f"{row['mean_late_pages_per_kv_head']:.3f} | "
            f"{row['mean_wasted_pages_per_kv_head']:.3f} | "
            f"{row['mean_fully_covered_kv_head_fraction']:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("layer-ahead prefetch oracle requires CUDA")
    started = time.perf_counter()
    torch.cuda.set_device(0)
    capture_dir = args.capture.expanduser().resolve()
    capture_manifest_path = capture_dir / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    model_path = Path(capture_manifest["model"]["path"])
    c1_dir = args.c1_export.expanduser().resolve()
    c1_result_path = c1_dir / "results.json"
    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_dir, model_path)
    if capture_manifest["c1_export"]["results_sha256"] != _sha256(c1_result_path):
        raise ValueError("capture and replay use different C1 exports")
    target_layers = _parse_ints(args.target_layers)
    if not target_layers or min(target_layers) < 1:
        raise ValueError("target layers must be nonempty and exclude layer 0")
    page_sizes = _parse_ints(args.page_sizes)
    actual_budgets = _parse_ints(args.actual_token_budgets)
    query_positions = _parse_ints(args.query_positions)
    overfetch_factors = _parse_floats(args.overfetch_factors)
    if not page_sizes or not actual_budgets or not query_positions or not overfetch_factors:
        raise ValueError("oracle grids must be nonempty")
    if min(page_sizes + actual_budgets) <= 0 or min(overfetch_factors) < 1.0:
        raise ValueError("page/budget sizes must be positive and overfetch must be >= 1")

    windows_path = Path(capture_manifest["calibration"]["windows_path"])
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    window_start = int(capture_manifest["calibration"]["window_start"])
    windows = int(capture_manifest["calibration"]["windows"])
    input_ids = stored[window_start : window_start + windows]
    del stored
    sequence = int(capture_manifest["calibration"]["sequence_length"])
    if int(input_ids.shape[1]) != sequence or max(query_positions) >= sequence:
        raise ValueError("query positions are incompatible with captured windows")
    fit_slice = slice(args.fit_start, args.fit_start + args.fit_examples)
    heldout_stop = args.heldout_start + args.heldout_examples
    heldout_slice = slice(args.heldout_start, heldout_stop)
    if args.fit_start + args.fit_examples > args.heldout_start:
        raise ValueError("fit and heldout windows must be disjoint and ordered")
    if heldout_stop > windows:
        raise ValueError("requested fit/heldout windows exceed the capture")

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()
    model.config.use_cache = False
    device = model.model.embed_tokens.weight.device
    position_ids = torch.arange(sequence, device=device, dtype=torch.long).unsqueeze(0)
    hidden_bank = torch.empty(
        windows,
        sequence,
        HIDDEN_SIZE,
        dtype=dtype,
        device="cpu",
    )
    for completed, batch_ids in _batches(input_ids, 0, windows, args.batch_size):
        embeddings = model.model.embed_tokens(batch_ids.to(device=device))
        start = completed - len(batch_ids)
        hidden_bank[start:completed].copy_(embeddings.cpu())
    del input_ids, embeddings
    position_embeddings = model.model.rotary_emb(
        hidden_bank[:1].to(device=device), position_ids
    )
    cos = position_embeddings[0].cpu()
    sin = position_embeddings[1].cpu()
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    features = _capture_predictors(
        model=model,
        hidden_bank=hidden_bank,
        c1_dir=c1_dir,
        c1_result=c1_result,
        value_rank=value_rank,
        target_layers=target_layers,
        batch_size=args.batch_size,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    del model, hidden_bank, position_embeddings
    torch.cuda.empty_cache()

    records: list[dict[str, Any]] = []
    query_diagnostics: dict[str, dict[str, dict[str, float]]] = {}
    factor_tensors: dict[str, Tensor] = {}
    for layer in target_layers:
        actual_pre = features[layer]["actual_pre_query"]
        rehearsal_pre = features[layer]["input_rehearsal_pre_query"]
        previous_c1 = features[layer]["previous_c1_value"]
        actual_grouped = group_query_heads(actual_pre, kv_heads=NUM_KV_HEADS)
        rehearsal_grouped = group_query_heads(
            rehearsal_pre, kv_heads=NUM_KV_HEADS
        )
        c1_direct = fit_headwise_linear(
            previous_c1[fit_slice], actual_grouped[fit_slice]
        )
        c1_residual = fit_headwise_linear(
            previous_c1[fit_slice],
            (actual_grouped.float() - rehearsal_grouped.float())[fit_slice],
        )
        factor_tensors[f"layers.{layer}.c1_to_query"] = c1_direct.weight
        factor_tensors[f"layers.{layer}.c1_to_rehearsal_residual"] = (
            c1_residual.weight
        )
        heldout_c1 = previous_c1[heldout_slice]
        predictions_pre = {
            "input_rehearsal": rehearsal_pre[heldout_slice].float(),
            "previous_c1": ungroup_query_heads(
                apply_headwise_linear(heldout_c1, c1_direct),
                query_heads=NUM_QUERY_HEADS,
            ),
            "input_plus_c1_residual": rehearsal_pre[heldout_slice].float()
            + ungroup_query_heads(
                apply_headwise_linear(heldout_c1, c1_residual),
                query_heads=NUM_QUERY_HEADS,
            ),
        }
        heldout_actual_pre = actual_pre[heldout_slice].float()
        query_diagnostics[str(layer)] = _query_diagnostics(
            heldout_actual_pre, predictions_pre
        )
        actual_post = features[layer]["actual_post_query"][heldout_slice].to(
            device=device
        )
        predictions_post = {
            policy: _runtime_rotary(
                predicted,
                cos,
                sin,
                device=device,
                dtype=dtype,
            )
            for policy, predicted in predictions_pre.items()
        }

        exact_key = features[layer]["exact_key"][heldout_slice].to(device=device)
        c1_value = features[layer]["target_c1_value"][heldout_slice].to(
            device=device
        )
        mask = None
        for query_position in query_positions:
            prefix = query_position + 1
            prefix_key = exact_key[:, :, :prefix]
            prefix_value = c1_value[:, :, :prefix]
            prefix_mask = None if mask is None else mask[..., :prefix]
            actual_query = actual_post[:, :, query_position : query_position + 1]
            for page_size in page_sizes:
                landmarks = build_post_rope_k_landmarks(
                    prefix_key,
                    page_size=page_size,
                    attention_mask=prefix_mask,
                    landmark_dtype=args.landmark_dtype,
                )
                page_count = math.ceil(prefix / page_size)
                for actual_budget in actual_budgets:
                    actual_page_budget = min(
                        math.ceil(actual_budget / page_size), page_count
                    )
                    actual_config = ReverseShadowConfig(
                        page_size=page_size,
                        exact_token_budget=actual_page_budget * page_size,
                        selector="quest_minmax",
                        landmark_dtype=args.landmark_dtype,
                    )
                    actual_selection = c1_k_reverse_shadow_attention(
                        actual_query,
                        landmarks,
                        prefix_value,
                        actual_config,
                        prefix_key,
                        prefix_mask,
                        layer_idx=layer,
                    ).selected_page_mask
                    for policy, predicted_post in predictions_post.items():
                        predicted_query = predicted_post[
                            :, :, query_position : query_position + 1
                        ]
                        for overfetch in overfetch_factors:
                            prefetched_page_budget = min(
                                math.ceil(actual_page_budget * overfetch),
                                page_count,
                            )
                            predicted_config = ReverseShadowConfig(
                                page_size=page_size,
                                exact_token_budget=prefetched_page_budget * page_size,
                                selector="quest_minmax",
                                landmark_dtype=args.landmark_dtype,
                            )
                            prefetched = c1_k_reverse_shadow_attention(
                                predicted_query,
                                landmarks,
                                prefix_value,
                                predicted_config,
                                prefix_key,
                                prefix_mask,
                                layer_idx=layer,
                            ).selected_page_mask
                            statistics_row = prefetch_set_statistics(
                                actual_selection, prefetched
                            )
                            record = {
                                "layer": layer,
                                "query_position": query_position,
                                "policy": policy,
                                "page_size": page_size,
                                "actual_token_budget": actual_budget,
                                "actual_page_budget": actual_page_budget,
                                "overfetch_factor": overfetch,
                                "prefetched_page_budget": prefetched_page_budget,
                                **statistics_row,
                            }
                            records.append(record)
                            print(
                                f"[Layer-ahead] layer={layer} q={query_position} "
                                f"policy={policy} page={page_size} "
                                f"budget={actual_budget} overfetch={overfetch:g} "
                                f"recall={statistics_row['prefetch_page_recall']:.4f} "
                                f"late={statistics_row['mean_late_pages_per_kv_head']:.2f}",
                                flush=True,
                            )
        del exact_key, c1_value, actual_post, predictions_post
        torch.cuda.empty_cache()

    output_factors = args.output_factors.expanduser().resolve()
    _atomic_safetensors(output_factors, factor_tensors)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": {
            "final_selector": "actual-Q exact-K QUEST",
            "prefetch_only": True,
            "numerical_pairing": (
                "actual Q, exact K, and target C1-V captured in the same dense replay"
            ),
            "input_rehearsal": (
                "target-layer input norm and Q projection applied to the "
                "preceding block input"
            ),
            "c1_predictors": (
                "per-physical-head FP64 SVD least squares without ridge"
            ),
        },
        "capture": {
            "directory": str(capture_dir),
            "manifest_sha256": _sha256(capture_manifest_path),
            "fit_windows": [args.fit_start, args.fit_start + args.fit_examples],
            "heldout_windows": [args.heldout_start, heldout_stop],
        },
        "c1_export": {
            "directory": str(c1_dir),
            "results_sha256": _sha256(c1_result_path),
        },
        "factors": {
            "file": str(output_factors),
            "sha256": _sha256(output_factors),
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "query_diagnostics": query_diagnostics,
        "records": records,
        "summary": _summarize(records),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    _atomic_text(output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_markdown, _markdown(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--target-layers", default="1,17,35")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-examples", type=int, default=1)
    parser.add_argument("--heldout-start", type=int, default=1)
    parser.add_argument("--heldout-examples", type=int, default=1)
    parser.add_argument("--query-positions", default="1023,1535,2047")
    parser.add_argument("--page-sizes", default="16,32")
    parser.add_argument("--actual-token-budgets", default="256,512")
    parser.add_argument("--overfetch-factors", default="1,1.5,2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--landmark-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-factors", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
