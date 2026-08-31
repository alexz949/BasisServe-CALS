#!/usr/bin/env python3
"""Fit and evaluate head-specific Dense/C1-Value QK score proxies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.v_qk_score_probe import (  # noqa: E402
    CenteredScoreProbeFit,
    apply_qk_score_probe,
    fit_centered_score_probe,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.eval_qwen3_8b_dense_v_k_proxy import (  # noqa: E402
    _accumulate_score_metrics,
    _extract_layer_features,
    _finalize_score_sums,
    _merge_score_sums,
    _new_score_sums,
)
from evaluation.fit_qwen3_8b_c1_k_output_closure import (  # noqa: E402
    HEAD_DIM,
    HEADS_PER_GROUP,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    _batches,
    _load_layer_c1_factors,
    _propagate_dense_layer,
    _sha256,
    _validate_config,
)


FORMAT = "basisserve.qwen3_8b.v_qk_centered_score_proxy.v1"


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


@torch.inference_mode()
def _build_centered_statistics(
    query: Tensor,
    exact_key: Tensor,
    source: Tensor,
    *,
    query_positions: list[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Build exact causal sufficient statistics without storing score matrices."""

    if int(query.shape[0]) != int(exact_key.shape[0]) or int(query.shape[0]) != int(
        source.shape[0]
    ):
        raise ValueError("query, Key, and source must contain the same windows")
    query_rows = []
    gram_rows = []
    cross_rows = []
    total_scores = 0
    for example in range(int(query.shape[0])):
        example_query = query[example].to(device=device, dtype=torch.float32)
        example_key = exact_key[example].to(device=device, dtype=torch.float32)
        example_source = source[example].to(device=device, dtype=torch.float32)
        for position in query_positions:
            visible = position + 1
            query_row = example_query[:, position].reshape(
                NUM_KV_HEADS, HEADS_PER_GROUP, HEAD_DIM
            )
            centered_source = example_source[:, :visible]
            centered_source = centered_source - centered_source.mean(
                dim=1, keepdim=True
            )
            exact_score = torch.einsum(
                "kgd,kld->kgl",
                query_row,
                example_key[:, :visible],
            ) / math.sqrt(HEAD_DIM)
            exact_score = exact_score - exact_score.mean(dim=-1, keepdim=True)
            query_rows.append(query_row)
            gram_rows.append(
                torch.einsum("klr,kls->krs", centered_source, centered_source)
            )
            cross_rows.append(
                torch.einsum("klr,kgl->kgr", centered_source, exact_score)
            )
            total_scores += NUM_QUERY_HEADS * visible
    # Stack the shared row dimension after KV/group dimensions.
    query_stat = torch.stack(query_rows, dim=2)
    gram_stat = torch.stack(gram_rows, dim=1)
    cross_stat = torch.stack(cross_rows, dim=2)
    return query_stat, gram_stat, cross_stat, total_scores


@torch.inference_mode()
def _evaluate_proxy(
    query: Tensor,
    exact_key: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    query_positions: list[int],
    budgets: list[int],
    device: torch.device,
) -> dict[str, Any]:
    sums = _new_score_sums(budgets)
    kv_index = torch.arange(NUM_QUERY_HEADS, device=device) // HEADS_PER_GROUP
    weight = weight.to(device=device, dtype=torch.float32)
    for example in range(int(query.shape[0])):
        example_query = query[example].to(device=device, dtype=torch.float32)
        example_key = exact_key[example].to(device=device, dtype=torch.float32)
        example_source = source[example].to(device=device, dtype=torch.float32)
        for position in query_positions:
            visible = position + 1
            query_row = example_query[:, position]
            repeated_key = example_key.index_select(0, kv_index)[:, :visible]
            exact_score = torch.einsum(
                "hd,hld->hl", query_row, repeated_key
            ) / math.sqrt(HEAD_DIM)
            proxy_score = apply_qk_score_probe(
                query_row,
                example_source[:, :visible],
                weight,
                heads_per_group=HEADS_PER_GROUP,
            )
            _accumulate_score_metrics(
                sums,
                exact_scores=exact_score,
                proxy_scores=proxy_score,
                budgets=budgets,
            )
    return {"metric_sums": sums, "metrics": _finalize_score_sums(sums)}


def _fit_payload(fit: CenteredScoreProbeFit) -> dict[str, Any]:
    return {
        "iterations": fit.iterations,
        "maximum_relative_residual": fit.maximum_relative_residual,
        "relative_residual_history": list(fit.relative_residual_history),
        "damping_min": float(fit.damping.min()),
        "damping_mean": float(fit.damping.mean()),
        "damping_max": float(fit.damping.max()),
    }


def _aggregate(records: list[dict[str, Any]], budgets: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for split in ("fit", "heldout"):
            grouped[(str(record["source"]), split)].append(record[split])
    result = []
    for (source, split), rows in sorted(grouped.items()):
        raw_rows = []
        for row in rows:
            raw_rows.append(row["metric_sums"])
        result.append(
            {
                "source": source,
                "split": split,
                "layers": sorted({int(record["layer"]) for record in records}),
                "metrics": _finalize_score_sums(
                    _merge_score_sums(raw_rows, budgets)
                ),
            }
        )
    return result


def _table_rows(rows: list[dict[str, Any]], report_budget: int) -> list[str]:
    lines = []
    for row in rows:
        metric = row["metrics"]
        budget = metric["budgets"][str(report_budget)]
        lines.append(
            f"| {row['source']} | {row['split']} | "
            f"{metric['centered_score_relative_rmse']:.6f} | "
            f"{metric['mean_attention_kl_teacher_to_proxy']:.6f} | "
            f"{budget['mean_top_k_recall']:.6f} | "
            f"{budget['mean_teacher_mass_selected']:.6f} | "
            f"{budget['mean_oracle_top_k_teacher_mass']:.6f} |"
        )
    return lines


def _markdown(payload: dict[str, Any]) -> str:
    report_budget = max(
        int(item) for item in payload["configuration"]["budgets"].split(",")
    )
    lines = [
        "# Qwen3-8B Value-to-QK centered-score proxy",
        "",
        "Each query head has its own linear map `M_h`. The fit directly minimizes "
        "causal per-query centered QK score error; it does not fit or materialize "
        "a full approximate Key as its objective.",
        "",
        "## Aggregate",
        "",
        f"| source | split | centered-score rel-RMSE | KL(P||proxy) | R@{report_budget} | "
        f"mass@{report_budget} | oracle-mass@{report_budget} |",
        "|:---|:---|---:|---:|---:|---:|---:|",
        *_table_rows(payload["aggregate"], report_budget),
        "",
        "## Per layer held-out",
        "",
        f"| layer/source | split | centered-score rel-RMSE | KL(P||proxy) | R@{report_budget} | "
        f"mass@{report_budget} | oracle-mass@{report_budget} |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ]
    heldout_rows = []
    for record in payload["records"]:
        heldout_rows.append(
            {
                "source": f"L{record['layer']} {record['source']}",
                "split": "heldout",
                "metrics": record["heldout"]["metrics"],
            }
        )
    lines.extend(_table_rows(heldout_rows, report_budget))
    lines.extend(
        [
            "",
            "Exact KL and Top-k metrics are evaluation metrics. The fitted "
            "objective is centered score MSE solved by batched matrix-free CG.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Value-to-QK score probing requires CUDA")
    layers = _parse_ints(args.layers)
    budgets = _parse_ints(args.budgets)
    if not layers or not budgets or min(budgets) <= 0:
        raise ValueError("layers and positive budgets must be nonempty")
    if min(
        args.fit_windows,
        args.heldout_windows,
        args.sequence_length,
        args.fit_query_stride,
        args.heldout_query_stride,
        args.batch_size,
        args.torch_num_threads,
        args.cg_iterations,
    ) <= 0:
        raise ValueError("window, sequence, query, batch, thread, and CG sizes must be positive")
    fit_indices = set(range(args.fit_start, args.fit_start + args.fit_windows))
    heldout_indices = set(
        range(args.heldout_start, args.heldout_start + args.heldout_windows)
    )
    if fit_indices & heldout_indices:
        raise ValueError("fit and held-out windows must be disjoint")
    if min(args.fit_query_start, args.heldout_query_start) < max(budgets) - 1:
        raise ValueError("query starts must expose the maximum Top-k budget")

    started = time.perf_counter()
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model_path = args.model.expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    c1_dir = args.c1_export.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    if any(layer < 0 or layer >= int(config.num_hidden_layers) for layer in layers):
        raise ValueError("requested layer is outside the model")
    window_manifest_path = windows_path.parent / "manifest.json"
    window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    if window_manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("window bank hash mismatch")
    if window_manifest["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("window bank belongs to another model")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    if stored.ndim != 2 or int(stored.shape[1]) != args.sequence_length:
        raise ValueError("window bank geometry does not match the requested sequence")
    selected_indices = sorted(fit_indices) + sorted(heldout_indices)
    if max(selected_indices) >= len(stored):
        raise ValueError("requested windows exceed the stored bank")
    windows = stored.index_select(0, torch.tensor(selected_indices, dtype=torch.long))
    del stored

    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_dir, model_path)
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
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
    position_ids = torch.arange(
        args.sequence_length, device=device, dtype=torch.long
    ).unsqueeze(0)
    hidden_bank = torch.empty(
        len(windows),
        args.sequence_length,
        HIDDEN_SIZE,
        dtype=dtype,
        device="cpu",
    )
    for completed, input_ids in _batches(windows, 0, len(windows), args.batch_size):
        embeddings = model.model.embed_tokens(input_ids.to(device=device, dtype=torch.long))
        start = completed - len(input_ids)
        hidden_bank[start:completed].copy_(embeddings.to(device="cpu"))
    del windows, embeddings
    position_embeddings = model.model.rotary_emb(
        hidden_bank[:1].to(device=device), position_ids
    )

    fit_count = args.fit_windows
    fit_slice = slice(0, fit_count)
    heldout_slice = slice(fit_count, fit_count + args.heldout_windows)
    fit_query_positions = list(
        range(args.fit_query_start, args.sequence_length, args.fit_query_stride)
    )
    heldout_query_positions = list(
        range(
            args.heldout_query_start,
            args.sequence_length,
            args.heldout_query_stride,
        )
    )
    if fit_query_positions[-1] != args.sequence_length - 1:
        fit_query_positions.append(args.sequence_length - 1)
    if heldout_query_positions[-1] != args.sequence_length - 1:
        heldout_query_positions.append(args.sequence_length - 1)

    records = []
    factor_artifacts = {}
    requested_layers = set(layers)
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index in requested_layers:
            encoder, _, c1_artifact = _load_layer_c1_factors(
                c1_dir, c1_result, layer_index, value_rank
            )
            features = _extract_layer_features(
                layer,
                hidden_bank,
                value_encoder=encoder,
                position_embeddings=position_embeddings,
                batch_size=args.batch_size,
            )
            layer_factors = {}
            layer_records = []
            for source_name, source_tensor in (
                ("dense_v128", features["dense_value"]),
                (f"c1_v{value_rank}", features["c1_value"]),
            ):
                query_stat, gram_stat, cross_stat, fit_score_entries = (
                    _build_centered_statistics(
                        features["query"][fit_slice],
                        features["post_key"][fit_slice],
                        source_tensor[fit_slice],
                        query_positions=fit_query_positions,
                        device=device,
                    )
                )
                fit = fit_centered_score_probe(
                    query_stat,
                    gram_stat,
                    cross_stat,
                    relative_damping=args.relative_damping,
                    cg_iterations=args.cg_iterations,
                    cg_relative_tolerance=args.cg_relative_tolerance,
                )
                if not bool(torch.isfinite(fit.weight).all()):
                    raise FloatingPointError(
                        f"non-finite centered-score factors for {source_name} layer {layer_index}"
                    )
                layer_factors[f"{source_name}.weight"] = fit.weight.float().cpu().contiguous()
                layer_factors[f"{source_name}.damping"] = fit.damping.float().cpu().contiguous()
                fit_evaluation = _evaluate_proxy(
                    features["query"][fit_slice],
                    features["post_key"][fit_slice],
                    source_tensor[fit_slice],
                    fit.weight,
                    query_positions=fit_query_positions,
                    budgets=budgets,
                    device=device,
                )
                heldout_evaluation = _evaluate_proxy(
                    features["query"][heldout_slice],
                    features["post_key"][heldout_slice],
                    source_tensor[heldout_slice],
                    fit.weight,
                    query_positions=heldout_query_positions,
                    budgets=budgets,
                    device=device,
                )
                record = {
                    "layer": layer_index,
                    "source": source_name,
                    "source_rank": int(source_tensor.shape[-1]),
                    "fit_score_entries": fit_score_entries,
                    "fit_solver": _fit_payload(fit),
                    "fit": fit_evaluation,
                    "heldout": heldout_evaluation,
                }
                layer_records.append(record)
                print(
                    f"[V->QK centered] layer={layer_index} source={source_name} "
                    f"cg={fit.iterations} residual={fit.maximum_relative_residual:.3e} "
                    f"heldout_score={heldout_evaluation['metrics']['centered_score_relative_rmse']:.4f}",
                    flush=True,
                )
                del query_stat, gram_stat, cross_stat, fit
                torch.cuda.empty_cache()
            factor_path = output_dir / f"layer_{layer_index:03d}.safetensors"
            _atomic_safetensors(factor_path, layer_factors)
            factor_artifacts[str(layer_index)] = {
                "file": factor_path.name,
                "sha256": _sha256(factor_path),
                "c1_factor_file": c1_artifact["file"],
                "c1_factor_sha256": c1_artifact["sha256"],
            }
            records.extend(layer_records)
            del features, encoder, layer_factors, layer_records
            torch.cuda.empty_cache()

        _propagate_dense_layer(
            model,
            layer,
            hidden_bank,
            batch_size=args.batch_size,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            layer_index=layer_index,
        )

    c1_result_path = c1_dir / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": {
            "proxy": "head-specific (Q_h M_h) C_g.T / sqrt(head_dim)",
            "fit_objective": "exact valid-causal per-query centered score MSE",
            "solver": "batched Jacobi-preconditioned matrix-free conjugate gradient on sufficient statistics",
            "sources": ["dense_v128", f"c1_v{value_rank}"],
            "exact_kl": "evaluation only",
            "key_reconstruction": False,
        },
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "windows": {
            "path": str(windows_path),
            "sha256": _sha256(windows_path),
            "manifest_sha256": _sha256(window_manifest_path),
            "fit_indices": sorted(fit_indices),
            "heldout_indices": sorted(heldout_indices),
            "sequence_length": args.sequence_length,
            "fit_query_positions": fit_query_positions,
            "heldout_query_positions": heldout_query_positions,
        },
        "c1_export": {
            "path": str(c1_dir),
            "results_sha256": _sha256(c1_result_path),
            "value_rank": value_rank,
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "factor_artifacts": factor_artifacts,
        "records": records,
        "aggregate": _aggregate(records, budgets),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_text(output_dir / "result.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_dir / "summary.md", _markdown(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-windows", type=int, default=8)
    parser.add_argument("--heldout-start", type=int, default=8)
    parser.add_argument("--heldout-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--fit-query-start", type=int, default=511)
    parser.add_argument("--fit-query-stride", type=int, default=128)
    parser.add_argument("--heldout-query-start", type=int, default=511)
    parser.add_argument("--heldout-query-stride", type=int, default=256)
    parser.add_argument("--budgets", default="64,256,512")
    parser.add_argument("--relative-damping", type=float, default=1.0e-5)
    parser.add_argument("--cg-iterations", type=int, default=64)
    parser.add_argument("--cg-relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
