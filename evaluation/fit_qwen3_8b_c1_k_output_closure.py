#!/usr/bin/env python3
"""Fit output-aware post-RoPE K/Q factors with fixed Qwen3-8B C1-V factors."""

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
from typing import Any, Iterator

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_output_closure import (  # noqa: E402
    C1KOutputMetrics,
    c1k_jvp,
    c1k_teacher_and_student_output,
    c1k_vjp,
    canonicalize_c1k_factors,
    conjugate_gradient,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation import eval_qwen3_8b_post_rope_kqsvd_c1_wikitext as kq_evaluator  # noqa: E402


FORMAT = "basisserve.qwen3_8b.c1_k_output_closure.v1"
NUM_LAYERS = 36
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEADS_PER_GROUP = NUM_QUERY_HEADS // NUM_KV_HEADS
HEAD_DIM = 128
HIDDEN_SIZE = 4096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _validate_config(config: Any) -> None:
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
    )
    expected = (
        "qwen3",
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    if observed != expected:
        raise ValueError(f"expected Qwen3-8B geometry {expected}, found {observed}")


def _metric_dict(metric: C1KOutputMetrics) -> dict[str, float | int]:
    return {
        "relative_output_mse": metric.relative_output_mse,
        "relative_causal_score_error": metric.relative_causal_score_error,
        "output_squared_error": metric.output_squared_error,
        "target_output_energy": metric.target_output_energy,
        "causal_score_squared_error": metric.causal_score_squared_error,
        "causal_dense_score_energy": metric.causal_dense_score_energy,
        "tokens": metric.tokens,
    }


def _sum_metrics(metrics: list[C1KOutputMetrics]) -> C1KOutputMetrics:
    return C1KOutputMetrics(
        output_squared_error=sum(item.output_squared_error for item in metrics),
        target_output_energy=sum(item.target_output_energy for item in metrics),
        causal_score_squared_error=sum(
            item.causal_score_squared_error for item in metrics
        ),
        causal_dense_score_energy=sum(
            item.causal_dense_score_energy for item in metrics
        ),
        tokens=sum(item.tokens for item in metrics),
    )


class LayerFeatureFactory:
    """Recompute one layer's Q/K/C1-V features from a CPU hidden-state bank."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        value_encoder: Tensor,
        decoder: Tensor,
        value_rank: int,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> None:
        self.layer = layer
        self.attention = layer.self_attn
        self.position_embeddings = position_embeddings
        self.value_rank = int(value_rank)
        self.device = self.attention.q_proj.weight.device
        dense_v = self.attention.v_proj.weight.detach().float().reshape(
            NUM_KV_HEADS,
            HEAD_DIM,
            HIDDEN_SIZE,
        )
        self.compressed_v_weight = torch.bmm(
            value_encoder.to(device=self.device, dtype=torch.float32).mT,
            dense_v,
        ).reshape(NUM_KV_HEADS * self.value_rank, HIDDEN_SIZE).to(
            self.attention.v_proj.weight.dtype
        )
        self.decoder = decoder.to(device=self.device, dtype=torch.float32)

    def __call__(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        hidden_states = hidden_states.to(
            device=self.device,
            dtype=self.attention.q_proj.weight.dtype,
            non_blocking=True,
        )
        attention_input = self.layer.input_layernorm(hidden_states)
        batch, sequence, _ = attention_input.shape
        query = self.attention.q_norm(
            self.attention.q_proj(attention_input).view(
                batch,
                sequence,
                NUM_QUERY_HEADS,
                HEAD_DIM,
            )
        ).transpose(1, 2)
        key = self.attention.k_norm(
            self.attention.k_proj(attention_input).view(
                batch,
                sequence,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
        ).transpose(1, 2)
        query, key = apply_rotary_pos_emb(
            query,
            key,
            *self.position_embeddings,
        )
        grouped_query = query.reshape(
            batch,
            NUM_KV_HEADS,
            HEADS_PER_GROUP,
            sequence,
            HEAD_DIM,
        )
        value = F.linear(attention_input, self.compressed_v_weight).view(
            batch,
            sequence,
            NUM_KV_HEADS,
            self.value_rank,
        ).permute(0, 2, 1, 3)
        return grouped_query, key, value, self.decoder


def _batches(
    hidden_bank: Tensor,
    row_start: int,
    row_count: int,
    batch_size: int,
) -> Iterator[tuple[int, Tensor]]:
    for relative_start in range(0, row_count, batch_size):
        relative_stop = min(relative_start + batch_size, row_count)
        yield relative_stop, hidden_bank[
            row_start + relative_start : row_start + relative_stop
        ]


def _evaluate_objective(
    factory: LayerFeatureFactory,
    hidden_bank: Tensor,
    *,
    row_start: int,
    row_count: int,
    batch_size: int,
    key_projector: Tensor,
    query_projector: Tensor,
    query_chunk_size: int,
    label: str,
) -> C1KOutputMetrics:
    metrics = []
    for completed, hidden in _batches(hidden_bank, row_start, row_count, batch_size):
        query, key, value, decoder = factory(hidden)
        _, _, metric = c1k_teacher_and_student_output(
            query,
            key,
            value,
            decoder,
            key_projector,
            query_projector,
            query_chunk_size=query_chunk_size,
        )
        metrics.append(metric)
        if completed == row_count or completed % max(8 * batch_size, 1) == 0:
            print(f"[C1-K] {label}: {completed}/{row_count}", flush=True)
        del query, key, value
    return _sum_metrics(metrics)


def _right_hand_side(
    factory: LayerFeatureFactory,
    hidden_bank: Tensor,
    *,
    row_count: int,
    batch_size: int,
    key_projector: Tensor,
    query_projector: Tensor,
    query_chunk_size: int,
    block: str,
    layer_index: int,
    sweep: int,
) -> tuple[Tensor, C1KOutputMetrics]:
    right_hand_side = torch.zeros_like(
        query_projector if block == "query" else key_projector
    )
    metrics = []
    for completed, hidden in _batches(hidden_bank, 0, row_count, batch_size):
        query, key, value, decoder = factory(hidden)
        teacher, student, metric = c1k_teacher_and_student_output(
            query,
            key,
            value,
            decoder,
            key_projector,
            query_projector,
            query_chunk_size=query_chunk_size,
        )
        gradient = c1k_vjp(
            query,
            key,
            value,
            decoder,
            key_projector,
            query_projector,
            teacher - student,
            query_chunk_size=query_chunk_size,
        )
        right_hand_side.add_(gradient.query if block == "query" else gradient.key)
        metrics.append(metric)
        if completed == row_count or completed % max(8 * batch_size, 1) == 0:
            print(
                f"[C1-K] layer={layer_index} sweep={sweep} block={block} "
                f"rhs {completed}/{row_count}",
                flush=True,
            )
        del query, key, value, teacher, student, gradient
    return right_hand_side, _sum_metrics(metrics)


def _normal_operator(
    factory: LayerFeatureFactory,
    hidden_bank: Tensor,
    *,
    row_count: int,
    batch_size: int,
    key_projector: Tensor,
    query_projector: Tensor,
    query_chunk_size: int,
    block: str,
    damping: float,
    layer_index: int,
    sweep: int,
):
    call_index = 0

    def apply(direction: Tensor) -> Tensor:
        nonlocal call_index
        call_index += 1
        image = torch.zeros_like(direction)
        for completed, hidden in _batches(hidden_bank, 0, row_count, batch_size):
            query, key, value, decoder = factory(hidden)
            output_tangent = c1k_jvp(
                query,
                key,
                value,
                decoder,
                key_projector,
                query_projector,
                delta_key_projector=direction if block == "key" else None,
                delta_query_projector=direction if block == "query" else None,
                query_chunk_size=query_chunk_size,
            )
            gradient = c1k_vjp(
                query,
                key,
                value,
                decoder,
                key_projector,
                query_projector,
                output_tangent,
                query_chunk_size=query_chunk_size,
            )
            image.add_(gradient.query if block == "query" else gradient.key)
            if completed == row_count or completed % max(8 * batch_size, 1) == 0:
                print(
                    f"[C1-K] layer={layer_index} sweep={sweep} block={block} "
                    f"normal_call={call_index} {completed}/{row_count}",
                    flush=True,
                )
            del query, key, value, output_tangent, gradient
        if damping:
            image.add_(direction, alpha=damping)
        return image

    return apply


def _fit_block(
    factory: LayerFeatureFactory,
    hidden_bank: Tensor,
    *,
    fit_rows: int,
    batch_size: int,
    key_projector: Tensor,
    query_projector: Tensor,
    query_chunk_size: int,
    block: str,
    damping: float,
    cg_iterations: int,
    cg_relative_tolerance: float,
    maximum_backtracks: int,
    layer_index: int,
    sweep: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    right_hand_side, before = _right_hand_side(
        factory,
        hidden_bank,
        row_count=fit_rows,
        batch_size=batch_size,
        key_projector=key_projector,
        query_projector=query_projector,
        query_chunk_size=query_chunk_size,
        block=block,
        layer_index=layer_index,
        sweep=sweep,
    )
    operator = _normal_operator(
        factory,
        hidden_bank,
        row_count=fit_rows,
        batch_size=batch_size,
        key_projector=key_projector,
        query_projector=query_projector,
        query_chunk_size=query_chunk_size,
        block=block,
        damping=damping,
        layer_index=layer_index,
        sweep=sweep,
    )
    delta, cg = conjugate_gradient(
        operator,
        right_hand_side,
        max_iterations=cg_iterations,
        relative_tolerance=cg_relative_tolerance,
    )
    accepted_step = 0.0
    after = before
    next_key = key_projector
    next_query = query_projector
    for backtrack in range(maximum_backtracks + 1):
        step = 2.0**-backtrack
        candidate_key = key_projector + step * delta if block == "key" else key_projector
        candidate_query = (
            query_projector + step * delta if block == "query" else query_projector
        )
        candidate = _evaluate_objective(
            factory,
            hidden_bank,
            row_start=0,
            row_count=fit_rows,
            batch_size=batch_size,
            key_projector=candidate_key,
            query_projector=candidate_query,
            query_chunk_size=query_chunk_size,
            label=(
                f"layer={layer_index} sweep={sweep} block={block} "
                f"backtrack={backtrack}"
            ),
        )
        if candidate.output_squared_error < before.output_squared_error:
            accepted_step = step
            after = candidate
            next_key = candidate_key
            next_query = candidate_query
            break
    return next_key, next_query, {
        "block": block,
        "cg": cg,
        "accepted_step": accepted_step,
        "accepted": accepted_step > 0.0,
        "fit_before": _metric_dict(before),
        "fit_after": _metric_dict(after),
    }


def _load_layer_c1_factors(
    c1_dir: Path,
    c1_result: dict[str, Any],
    layer_index: int,
    value_rank: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    artifact = c1_result["artifacts"][str(layer_index)]
    path = c1_dir / artifact["file"]
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"C1 factor hash mismatch at layer {layer_index}")
    factors = load_file(str(path), device="cpu")
    if set(factors) != {"value_coordinate_encoders", "head_output_decoders"}:
        raise ValueError(f"unexpected C1 tensors at layer {layer_index}")
    encoder = factors["value_coordinate_encoders"]
    decoder = factors["head_output_decoders"].reshape(
        NUM_KV_HEADS,
        HEADS_PER_GROUP,
        value_rank,
        HIDDEN_SIZE,
    )
    if tuple(encoder.shape) != (NUM_KV_HEADS, HEAD_DIM, value_rank):
        raise ValueError(f"invalid C1 encoder geometry at layer {layer_index}")
    return encoder, decoder, artifact


def _propagate_dense_layer(
    model: nn.Module,
    layer: nn.Module,
    hidden_bank: Tensor,
    *,
    batch_size: int,
    position_ids: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    layer_index: int,
) -> None:
    device = layer.self_attn.q_proj.weight.device
    for completed, hidden in _batches(hidden_bank, 0, len(hidden_bank), batch_size):
        hidden = hidden.to(
            device=device,
            dtype=layer.self_attn.q_proj.weight.dtype,
            non_blocking=True,
        )
        causal_mask = create_causal_mask(
            config=model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        output = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        start = completed - len(hidden)
        hidden_bank[start:completed].copy_(output.to(device="cpu"))
        print(
            f"[C1-K] dense propagation layer={layer_index}: "
            f"{completed}/{len(hidden_bank)}",
            flush=True,
        )
        del hidden, output, causal_mask


@torch.inference_mode()
def fit(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("C1-K output closure fitting requires CUDA")
    positive = (
        args.fit_windows,
        args.heldout_windows,
        args.sequence_length,
        args.batch_size,
        args.query_chunk_size,
        args.gn_sweeps,
        args.cg_iterations,
        args.torch_num_threads,
    )
    if min(positive) <= 0:
        raise ValueError("calibration and solver sizes must be positive")
    if args.sequence_length != 2048:
        raise ValueError("controlled C1-K experiment requires seq2048")
    if not 0 < args.rank <= HEAD_DIM:
        raise ValueError(f"Key rank must be in [1, {HEAD_DIM}]")
    if args.damping < 0 or args.cg_relative_tolerance < 0:
        raise ValueError("damping and CG tolerance must be nonnegative")
    fit_indices = set(range(args.fit_start, args.fit_start + args.fit_windows))
    heldout_indices = set(
        range(args.heldout_start, args.heldout_start + args.heldout_windows)
    )
    if fit_indices & heldout_indices:
        raise ValueError("fit and heldout C4 windows must be disjoint")

    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    c1_dir = Path(args.c1_factor_dir).expanduser().resolve()
    kq_dir = Path(args.kq_factor_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_config(AutoConfig.from_pretrained(str(model_path), local_files_only=True))

    manifest_path = windows_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("C4 windows belong to another model config")
    if manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("C4 windows hash does not match its manifest")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    if stored.ndim != 2 or int(stored.shape[1]) != args.sequence_length:
        raise ValueError("C4 windows have incompatible geometry")
    selected_indices = list(range(args.fit_start, args.fit_start + args.fit_windows))
    selected_indices += list(
        range(args.heldout_start, args.heldout_start + args.heldout_windows)
    )
    if max(selected_indices) >= len(stored):
        raise ValueError("requested calibration windows exceed the stored bank")
    windows = stored.index_select(0, torch.tensor(selected_indices, dtype=torch.long))
    del stored

    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_dir, model_path)
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    if not 0 < value_rank <= HEAD_DIM:
        raise ValueError(f"C1 value rank must be in [1, {HEAD_DIM}]")
    kq_result, kq_factors = kq_evaluator._load_kq_result(kq_dir, model_path)
    initial_kq_rank = int(kq_result["geometry"]["rank"])
    if args.rank > initial_kq_rank:
        raise ValueError(
            f"requested Key rank {args.rank} exceeds KQ-SVD bank rank {initial_kq_rank}"
        )

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
        args.sequence_length,
        device=device,
        dtype=torch.long,
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
        print(f"[C1-K] embeddings: {completed}/{len(windows)}", flush=True)
    del windows, embeddings
    position_embeddings = model.model.rotary_emb(
        hidden_bank[:1].to(device=device),
        position_ids,
    )

    fitted_keys = []
    fitted_queries = []
    layer_records = []
    for layer_index, layer in enumerate(model.model.layers):
        encoder, decoder, c1_artifact = _load_layer_c1_factors(
            c1_dir,
            c1_result,
            layer_index,
            value_rank,
        )
        factory = LayerFeatureFactory(
            layer,
            value_encoder=encoder,
            decoder=decoder,
            value_rank=value_rank,
            position_embeddings=position_embeddings,
        )
        key_projector = kq_factors["kq_svd_key_projector"][
            layer_index, :, :, : args.rank
        ].to(
            device=device,
            dtype=torch.float32,
        )
        grouped_query = kq_factors["kq_svd_query_projector"][
            layer_index, :, :, : args.rank
        ].to(
            device=device,
            dtype=torch.float32,
        )
        query_projector = grouped_query[:, None].expand(
            NUM_KV_HEADS,
            HEADS_PER_GROUP,
            HEAD_DIM,
            args.rank,
        ).clone()
        key_projector, query_projector = canonicalize_c1k_factors(
            key_projector,
            query_projector,
        )
        initial_fit = _evaluate_objective(
            factory,
            hidden_bank,
            row_start=0,
            row_count=args.fit_windows,
            batch_size=args.batch_size,
            key_projector=key_projector,
            query_projector=query_projector,
            query_chunk_size=args.query_chunk_size,
            label=f"layer={layer_index} initial_fit",
        )
        initial_heldout = _evaluate_objective(
            factory,
            hidden_bank,
            row_start=args.fit_windows,
            row_count=args.heldout_windows,
            batch_size=args.batch_size,
            key_projector=key_projector,
            query_projector=query_projector,
            query_chunk_size=args.query_chunk_size,
            label=f"layer={layer_index} initial_heldout",
        )
        best_key = key_projector.clone()
        best_query = query_projector.clone()
        best_heldout = initial_heldout
        best_boundary = "initial_kq_svd"
        boundaries = [
            {
                "name": best_boundary,
                "fit": _metric_dict(initial_fit),
                "heldout": _metric_dict(initial_heldout),
            }
        ]
        block_records = []
        for sweep in range(1, args.gn_sweeps + 1):
            for block in ("query", "key"):
                key_projector, query_projector, block_record = _fit_block(
                    factory,
                    hidden_bank,
                    fit_rows=args.fit_windows,
                    batch_size=args.batch_size,
                    key_projector=key_projector,
                    query_projector=query_projector,
                    query_chunk_size=args.query_chunk_size,
                    block=block,
                    damping=args.damping,
                    cg_iterations=args.cg_iterations,
                    cg_relative_tolerance=args.cg_relative_tolerance,
                    maximum_backtracks=args.maximum_backtracks,
                    layer_index=layer_index,
                    sweep=sweep,
                )
                if block == "key" and block_record["accepted"]:
                    key_projector, query_projector = canonicalize_c1k_factors(
                        key_projector,
                        query_projector,
                    )
                heldout = _evaluate_objective(
                    factory,
                    hidden_bank,
                    row_start=args.fit_windows,
                    row_count=args.heldout_windows,
                    batch_size=args.batch_size,
                    key_projector=key_projector,
                    query_projector=query_projector,
                    query_chunk_size=args.query_chunk_size,
                    label=f"layer={layer_index} sweep={sweep} block={block} heldout",
                )
                name = f"sweep_{sweep}_{block}"
                boundary = {
                    "name": name,
                    "fit": block_record["fit_after"],
                    "heldout": _metric_dict(heldout),
                }
                boundaries.append(boundary)
                block_record["heldout"] = _metric_dict(heldout)
                block_records.append(block_record)
                if heldout.relative_output_mse < best_heldout.relative_output_mse:
                    best_key = key_projector.clone()
                    best_query = query_projector.clone()
                    best_heldout = heldout
                    best_boundary = name
        fitted_keys.append(best_key.cpu())
        fitted_queries.append(
            best_query.reshape(NUM_QUERY_HEADS, HEAD_DIM, args.rank).cpu()
        )
        layer_records.append(
            {
                "layer": layer_index,
                "selected_boundary": best_boundary,
                "selected_heldout": _metric_dict(best_heldout),
                "boundaries": boundaries,
                "blocks": block_records,
                "c1_factor_file": c1_artifact["file"],
                "c1_factor_sha256": c1_artifact["sha256"],
            }
        )
        print(
            f"[C1-K] layer={layer_index} selected={best_boundary} "
            f"heldout_mse={best_heldout.relative_output_mse:.9e}",
            flush=True,
        )
        _propagate_dense_layer(
            model,
            layer,
            hidden_bank,
            batch_size=args.batch_size,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            layer_index=layer_index,
        )
        del factory, encoder, decoder, best_key, best_query
        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True)
    factors_path = output_dir / "factors.safetensors"
    _atomic_safetensors(
        factors_path,
        {
            "key_projector": torch.stack(fitted_keys).float().contiguous(),
            "query_projector": torch.stack(fitted_queries).float().contiguous(),
        },
    )
    c1_result_path = c1_dir / "results.json"
    kq_result_path = kq_dir / "result.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "geometry": {
            "layers": NUM_LAYERS,
            "query_heads": NUM_QUERY_HEADS,
            "physical_kv_heads": NUM_KV_HEADS,
            "query_heads_per_kv_head": HEADS_PER_GROUP,
            "head_dim": HEAD_DIM,
            "key_rank": args.rank,
            "value_rank": value_rank,
            "retained_kv_ratio": (args.rank + value_rank) / (2 * HEAD_DIM),
        },
        "fixed_c1_value_factors": {
            "path": str(c1_result_path),
            "sha256": _sha256(c1_result_path),
            "format": c1_result["format"],
        },
        "initial_kq_factors": {
            "path": str(kq_result_path),
            "sha256": _sha256(kq_result_path),
            "format": kq_result["format"],
        },
        "calibration": {
            "dataset": "C4 train full documents",
            "windows_path": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest": str(manifest_path),
            "windows_manifest_sha256": _sha256(manifest_path),
            "fit_start": args.fit_start,
            "fit_windows": args.fit_windows,
            "heldout_start": args.heldout_start,
            "heldout_windows": args.heldout_windows,
            "sequence_length": args.sequence_length,
            "heldout_disjoint_from_fit": True,
        },
        "method": {
            "target": (
                f"Dense-K attention using fixed C1-V{value_rank} encoder and decoder"
            ),
            "objective": "full-layer causal attention-output MSE after softmax and C1 decoder",
            "initialization": (
                f"first {args.rank} ordered coordinates of post-RoPE "
                f"KQ-SVD rank{initial_kq_rank}"
            ),
            "query_geometry": f"one rank{args.rank} projector per Query head",
            "key_geometry": (
                f"one shared rank{args.rank} projector per physical KV group"
            ),
            "solver": "alternating Query/Key matrix-free Gauss-Newton CG",
            "gn_sweeps": args.gn_sweeps,
            "cg_iterations": args.cg_iterations,
            "cg_relative_tolerance": args.cg_relative_tolerance,
            "damping": args.damping,
            "maximum_backtracks": args.maximum_backtracks,
            "gauge": "per-group K QR with exact Q compensation",
            "attention_scaling": (
                f"1/sqrt({HEAD_DIM}); unchanged at rank{args.rank}"
            ),
            "layer_inputs": "dense sequential hidden states",
            "selection": "earliest minimum fresh-heldout output MSE per layer",
        },
        "layers": layer_records,
        "artifacts": {
            "factors": {
                "file": factors_path.name,
                "sha256": _sha256(factors_path),
                "dtype": "float32",
                "tensors": {
                    "key_projector": [NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, args.rank],
                    "query_projector": [
                        NUM_LAYERS,
                        NUM_QUERY_HEADS,
                        HEAD_DIM,
                        args.rank,
                    ],
                },
            }
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / "result.json", payload)
    print(f"[C1-K] wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--c1-factor-dir", required=True)
    parser.add_argument("--kq-factor-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-windows", type=int, default=128)
    parser.add_argument("--heldout-start", type=int, default=336)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--gn-sweeps", type=int, default=1)
    parser.add_argument("--cg-iterations", type=int, default=4)
    parser.add_argument("--cg-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--maximum-backtracks", type=int, default=5)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    return parser.parse_args()


if __name__ == "__main__":
    fit(parse_args())
