#!/usr/bin/env python3
"""Run Qwen3-32B TP-source Global-KL with independent profile replicas.

Two ``profile`` processes can each shard one BF16 model over two L40S GPUs and
score disjoint layer sets concurrently.  Every completed layer is checkpointed
atomically.  A later ``finalize`` process merges the profile shards, solves the
exact rank-budget DPs, and confirms schedules on disjoint C4 documents.
Candidate banks must share one decoder-closed activation-aware fitting
protocol.  They may be initial AASVD fits or post-ALS fits; the exact stage is
validated and recorded so Global-KL always describes the deployed candidates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import tensor_sha256  # noqa: E402
from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_allocation.v2"
PROFILE_FORMAT = "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_profile_shard.v1"
MODEL_LABEL = "Qwen3-32B"


def activate_model_profile(name: str) -> None:
    global FORMAT, PROFILE_FORMAT, MODEL_LABEL
    common.activate_model_profile(name)
    if name == "qwen3_32b":
        slug = "qwen3_32b"
        MODEL_LABEL = "Qwen3-32B"
        attention = "gqa"
    elif name == "qwen3_8b":
        slug = "qwen3_8b"
        MODEL_LABEL = "Qwen3-8B-Base"
        attention = "gqa"
    elif name == "llama31_8b":
        slug = "llama31_8b"
        MODEL_LABEL = "Llama-3.1-8B"
        attention = "gqa"
    elif name == "llama31_70b":
        slug = "llama31_70b"
        MODEL_LABEL = "Llama-3.1-70B"
        attention = "gqa"
    elif name == "llama2_7b":
        slug = "llama2_7b"
        MODEL_LABEL = "Llama-2-7B"
        attention = "mha"
    else:
        raise ValueError(f"unknown Qwen3 C1 model profile: {name}")
    FORMAT = f"basisserve.{slug}.{attention}_c1.tp_source_global_kl_allocation.v2"
    PROFILE_FORMAT = (
        f"basisserve.{slug}.{attention}_c1.tp_source_global_kl_profile_shard.v1"
    )


def _cuda_device_indices() -> tuple[int, ...]:
    indices = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)


def _select_fit_windows(path, *, window_start, profile_windows, confirmation_windows, sequence_length):
    resolved = path.expanduser().resolve()
    manifest_path = resolved.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete" and not manifest.get("test_only", False)
    assert common._sha256(resolved) == manifest["sha256"]
    payload = load_file(str(resolved), device="cpu")
    assert set(payload) == {"input_ids"}
    tokens = payload["input_ids"]
    stop = window_start + profile_windows + confirmation_windows
    indices = list(range(window_start, stop))
    assert window_start >= 0 and stop <= len(tokens)
    assert set(indices) <= set(manifest["fit_ids"])
    assert not set(indices).intersection(manifest["validation_ids"])
    assert sequence_length == tokens.shape[1]
    selected = tokens[window_start:stop].to(torch.long).contiguous()
    provenance = dict(path=str(resolved), sha256=manifest["sha256"],
        manifest=str(manifest_path), manifest_sha256=common._sha256(manifest_path),
        window_start=window_start, window_stop_exclusive=stop,
        profile_indices=indices[:profile_windows], confirmation_indices=indices[profile_windows:],
        stored_sequence_length=int(tokens.shape[1]), used_sequence_length=sequence_length,
        disjoint_from_als_fit_and_heldout_prefix=False, disjoint_from_diagnostic=True)
    return selected[:profile_windows], selected[profile_windows:], provenance


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-dir",
        action="append",
        required=True,
        metavar="RANK=PATH",
    )
    parser.add_argument("--anchor-rank", type=int, default=96)
    parser.add_argument("--candidate-ranks", default="64,80,96,112,128")
    parser.add_argument("--window-start", type=int, default=common.FRESH_WINDOW_START)
    parser.add_argument("--probe-source", choices=("fresh", "fit"), default="fresh")
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mlp-chunk-size", type=int, default=0)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=40)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--local-files-only", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    profile = subparsers.add_parser("profile")
    _add_shared_args(profile)
    profile.add_argument("--profile-shard-index", type=int, required=True)
    profile.add_argument("--profile-shard-count", type=int, default=2)

    finalize = subparsers.add_parser("finalize")
    _add_shared_args(finalize)
    finalize.add_argument("--profile-shard-count", type=int, default=2)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--eval-seqlen", type=int, default=2048)
    finalize.add_argument("--eval-batch-size", type=int, default=1)
    finalize.add_argument("--eval-max-samples", type=int)
    finalize.add_argument("--eval-max-tokens", type=int)
    finalize.add_argument(
        "--skip-wikitext",
        action="store_true",
        help="Freeze/export the schedule without evaluating its pre-ALS PPL",
    )
    return parser.parse_args()


def _log(message: str, *, shard: int | None = None) -> None:
    prefix = "[Qwen3 Global-KL sharded]"
    if shard is not None:
        prefix = f"[Qwen3 Global-KL shard {shard}]"
    print(f"{prefix} {message}", flush=True)


def _shard_layers(shard_index: int, shard_count: int) -> tuple[int, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("profile shard index/count are invalid")
    layers = tuple(range(shard_index, common.NUM_LAYERS, shard_count))
    if not layers:
        raise ValueError("profile shard has no assigned layers")
    return layers


def _validate_shared(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    tuple[int, ...],
    dict[int, Path],
    dict[int, dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    positive = (
        args.profile_windows,
        args.confirmation_windows,
        args.sequence_length,
        args.batch_size,
        args.vocab_chunk_size,
        args.torch_num_threads,
        args.max_memory_per_gpu_gib,
    )
    if min(positive) <= 0:
        raise ValueError("sample and compute arguments must be positive")
    if args.covariance_damping < 0:
        raise ValueError("damping must be non-negative")
    candidate_ranks = common._parse_ranks(args.candidate_ranks)
    if args.anchor_rank not in candidate_ranks:
        raise ValueError("anchor rank must be one of the candidate ranks")
    factor_dirs = common._parse_factor_dirs(args.factor_dir)
    if set(factor_dirs) != set(candidate_ranks) - {common.HEAD_DIM}:
        raise ValueError(
            "factor directories must provide every non-full-rank candidate"
        )
    model_path = Path(args.model).expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    model_config_sha256 = common._sha256(model_path / "config.json")
    factor_results = common._load_factor_results(
        factor_dirs,
        model_config_sha256=model_config_sha256,
        snapshot_dir=snapshot_dir,
    )
    reference = next(iter(factor_results.values()))["fit_config"]
    calibration_window_count = int(reference["fit_windows"]) + int(
        reference["validation_windows"]
    )
    assert args.probe_source == "fit" or args.window_start >= calibration_window_count, (
        "Global-KL windows overlap the ALS fit/validation prefix: "
        f"start={args.window_start}, required={calibration_window_count}"
    )
    if float(reference["covariance_damping"]) != args.covariance_damping:
        raise ValueError("Global-KL damping must match the ALS factor banks")
    if reference.get("work_dtype") != "float32":
        raise ValueError("Global-KL decoder closure requires float32 factor banks")
    if reference.get("factor_dtype") != "bfloat16":
        raise ValueError("Global-KL deployment requires bfloat16 factor banks")
    if reference.get("encoder_initialization") != "activation-weighted-svd":
        raise ValueError(
            "Global-KL requires activation-weighted SVD initialization"
        )
    encoder_sweeps = int(reference.get("encoder_sweeps", -1))
    assert encoder_sweeps == 6
    assert (
        reference.get("checkpoint_policy")
        == "fixed decoder-refitted endpoint after encoder sweep 6"
    )
    assert reference.get("decoder_objective") == "full_layer"
    forbidden = {
        "minimum_encoder_sweeps",
        "encoder_relative_tolerance",
        "encoder_patience",
        "decoder_relative_jitter",
        "encoder_relative_damping",
        "maximum_backtracks",
        "selection_boundaries",
        "selection",
    }
    assert not forbidden.intersection(reference)
    expected_manifest_sha = reference["snapshot_manifest_sha256"]
    if common._sha256(snapshot_dir / "manifest.json") != expected_manifest_sha:
        raise ValueError("factor banks and covariance manifest hashes disagree")
    selector = _select_fit_windows if args.probe_source == "fit" else common._select_fresh_windows
    if args.probe_source == "fit":
        manifest = json.loads((args.windows.parent / "manifest.json").read_text())
        assert manifest["model_config_sha256"] == model_config_sha256
        assert manifest["fit_ids"] == list(range(int(reference["fit_windows"])))
        assert args.window_start + args.profile_windows + args.confirmation_windows <= int(reference["fit_windows"])
    profile, confirmation, provenance = selector(
        args.windows,
        window_start=args.window_start,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
        sequence_length=args.sequence_length,
    )
    provenance = {
        **provenance,
        "probe_source": args.probe_source,
        "als_fit_and_heldout_window_count": calibration_window_count,
        "disjoint_from_als_fit_and_heldout_prefix": (
            args.window_start >= calibration_window_count
        ),
    }
    return (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        profile,
        confirmation,
        provenance,
    )


def _configuration(
    args: argparse.Namespace,
    *,
    model_path: Path,
    snapshot_dir: Path,
    candidate_ranks: Sequence[int],
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
    windows_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    fit_config = next(iter(factor_results.values()))["fit_config"]
    return {
        "model": str(model_path),
        "model_config_sha256": common._sha256(model_path / "config.json"),
        "windows_provenance": dict(windows_provenance),
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": common._sha256(snapshot_dir / "manifest.json"),
        "factor_results_sha256": {
            str(rank): common._sha256(directory / "results.json")
            for rank, directory in factor_dirs.items()
        },
        "factor_stage": {
            "encoder_initialization": fit_config["encoder_initialization"],
            "encoder_sweeps": int(fit_config["encoder_sweeps"]),
            "checkpoint_policy": fit_config["checkpoint_policy"],
            "decoder_objective": fit_config["decoder_objective"],
        },
        "anchor_rank": args.anchor_rank,
        "candidate_ranks": list(map(int, candidate_ranks)),
        "profile_windows": args.profile_windows,
        "confirmation_windows": args.confirmation_windows,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "mlp_chunk_size": args.mlp_chunk_size,
        "covariance_damping": args.covariance_damping,
        "vocab_chunk_size": args.vocab_chunk_size,
        "model_dtype": args.model_dtype,
        "attn_implementation": args.attn_implementation,
    }


def _load_model(args: argparse.Namespace, model_path: Path) -> nn.Module:
    cuda_indices = _cuda_device_indices()
    if not cuda_indices:
        raise RuntimeError("sharded Global-KL requires CUDA")
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in cuda_indices
        },
    ).eval()
    model.config.use_cache = False
    if args.mlp_chunk_size:
        from evaluation.chunked_prefill_mlp import ChunkedTokenwise
        assert args.mlp_chunk_size > 0
        for layer in model.model.layers:
            layer.mlp = ChunkedTokenwise(layer.mlp, args.mlp_chunk_size)
    geometry = (
        str(model.config.model_type),
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        int(model.config.num_key_value_heads),
        int(
            getattr(model.config, "head_dim", 0)
            or model.config.hidden_size // model.config.num_attention_heads
        ),
        int(model.config.hidden_size),
    )
    expected = (
        common.MODEL_TYPE,
        common.NUM_LAYERS,
        common.NUM_QUERY_HEADS,
        common.NUM_KV_HEADS,
        common.HEAD_DIM,
        common.HIDDEN_SIZE,
    )
    if geometry != expected:
        raise ValueError(f"unexpected {MODEL_LABEL} geometry: {geometry}")
    placement = {parameter.device.type for parameter in model.parameters()}
    if placement - {"cuda"}:
        raise RuntimeError(f"Global-KL model contains offloaded parameters: {placement}")
    return model


def _profile_payload(
    *,
    status: str,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    assigned_layers: Sequence[int],
    completed_layers: Sequence[int],
    anchor_metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    damping_by_layer: Mapping[int, float],
    started: float,
) -> dict[str, Any]:
    return {
        "format": PROFILE_FORMAT,
        "status": status,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile_shard_index": args.profile_shard_index,
        "profile_shard_count": args.profile_shard_count,
        "assigned_layers": list(map(int, assigned_layers)),
        "completed_layers": list(map(int, completed_layers)),
        "configuration": dict(configuration),
        "uniform_anchor": dict(anchor_metrics),
        "records": list(records),
        "absolute_covariance_damping_by_layer": {
            str(layer): float(value) for layer, value in damping_by_layer.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "visible_cuda_devices": len(_cuda_device_indices()),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in _cuda_device_indices()
            ],
        },
    }


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    assigned_layers = _shard_layers(
        args.profile_shard_index,
        args.profile_shard_count,
    )
    (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        profile_sequences,
        _,
        windows_provenance,
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    shard_path = profile_dir / f"shard_{args.profile_shard_index:02d}.json"
    records: list[dict[str, Any]] = []
    completed_layers: list[int] = []
    damping_by_layer: dict[int, float] = {}
    prior_anchor: dict[str, Any] | None = None
    if shard_path.is_file():
        prior = json.loads(shard_path.read_text(encoding="utf-8"))
        if prior.get("format") != PROFILE_FORMAT:
            raise ValueError(f"incompatible profile checkpoint: {shard_path}")
        if prior.get("configuration") != configuration:
            raise ValueError(f"profile checkpoint configuration changed: {shard_path}")
        if tuple(map(int, prior.get("assigned_layers", ()))) != assigned_layers:
            raise ValueError(f"profile checkpoint layer assignment changed: {shard_path}")
        if prior.get("status") == "complete":
            _log(f"completed checkpoint already exists: {shard_path}", shard=args.profile_shard_index)
            return
        records = [dict(row) for row in prior.get("records", ())]
        completed_layers = list(map(int, prior.get("completed_layers", ())))
        damping_by_layer = {
            int(layer): float(value)
            for layer, value in prior.get(
                "absolute_covariance_damping_by_layer", {}
            ).items()
        }
        prior_anchor = dict(prior["uniform_anchor"])
    if any(layer not in assigned_layers for layer in completed_layers):
        raise ValueError("profile checkpoint completed an unassigned layer")
    expected_prior_records = len(completed_layers) * common.NUM_KV_HEADS * (
        len(candidate_ranks) - 1
    )
    if len(records) != expected_prior_records:
        raise ValueError("profile checkpoint record count is incomplete")

    started = time.perf_counter()
    model = _load_model(args, model_path)
    profile_teacher = common._capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label=f"profile shard {args.profile_shard_index}",
    )
    snapshot_cache, _ = common._load_snapshot_cache(
        snapshot_dir,
        model_path=model_path,
    )
    factor_cache = common._load_factor_cache(factor_dirs, factor_results)
    layers = common._decoder_layers(model)
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone() for layer in layers
    )
    uniform_schedule = [
        [args.anchor_rank] * common.NUM_KV_HEADS
        for _ in range(common.NUM_LAYERS)
    ]
    common._install_schedule(
        model,
        schedule=uniform_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=0.0,
        keep_factors=False,
    )
    anchor_metrics = common._evaluate_teacher_metrics(
        model,
        profile_teacher,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    if prior_anchor is not None:
        previous = prior_anchor["terminal_kl"]["values"]
        current = anchor_metrics["terminal_kl"]["values"]
        if len(previous) != len(current) or max(
            abs(float(left) - float(right))
            for left, right in zip(previous, current, strict=True)
        ) > 1.0e-7:
            raise RuntimeError("resumed profile anchor changed numerically")
    _log(
        f"anchor KL={anchor_metrics['terminal_kl']['mean']:.9g} "
        f"batch_size={args.batch_size}",
        shard=args.profile_shard_index,
    )

    interventions = [
        (source, rank)
        for source in range(common.NUM_KV_HEADS)
        for rank in candidate_ranks
        if rank != args.anchor_rank
    ]
    completed = set(completed_layers)
    for layer_index in assigned_layers:
        if layer_index in completed:
            _log(f"resume skips layer {layer_index}", shard=args.profile_shard_index)
            continue
        layer = layers[layer_index]
        device = layer.self_attn.v_proj.weight.device
        anchor_v = layer.self_attn.v_proj.weight.detach().clone()
        anchor_o = layer.self_attn.o_proj.weight.detach().clone()
        objective, absolute_damping = common._objective(
            snapshot_cache[layer_index],
            layer=layer_index,
            device=device,
            covariance_damping=args.covariance_damping,
        )
        layer_records = []
        for intervention_index, (source, candidate_rank) in enumerate(interventions):
            source_ranks = [args.anchor_rank] * common.NUM_KV_HEADS
            source_ranks[source] = candidate_rank
            factors, closure = common._closed_factors(
                bank=factor_cache[layer_index],
                snapshot=snapshot_cache[layer_index],
                objective=objective,
                source_ranks=source_ranks,
                anchor_rank=args.anchor_rank,
                decoder_relative_jitter=0.0,
                device=device,
            )
            common._install_factors(
                layer,
                dense_v_weight=dense_v_weights[layer_index],
                factors=factors,
            )
            metrics = common._evaluate_teacher_metrics(
                model,
                profile_teacher,
                vocab_chunk_size=args.vocab_chunk_size,
            )
            layer.self_attn.v_proj.weight.copy_(anchor_v)
            layer.self_attn.o_proj.weight.copy_(anchor_o)
            delta = common._paired_delta(metrics, anchor_metrics)
            layer_records.append(
                {
                    "layer": layer_index,
                    "source": source,
                    "physical_kv_head": source,
                    "routed_query_heads": list(
                        range(
                            source * common.HEADS_PER_SOURCE,
                            (source + 1) * common.HEADS_PER_SOURCE,
                        )
                    ),
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": candidate_rank,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                    "ideal_width_delta": candidate_rank - args.anchor_rank,
                    "padded_width_delta": (
                        common.NUM_KV_HEADS * max(source_ranks)
                        - common.NUM_KV_HEADS * args.anchor_rank
                    ),
                    "decoder_solve": closure,
                }
            )
            _log(
                f"layer={layer_index} candidate={intervention_index + 1}/"
                f"{len(interventions)} source={source} rank={candidate_rank} "
                f"dKL={delta['terminal_kl']['mean']:.6g}",
                shard=args.profile_shard_index,
            )
            del factors, metrics
            torch.cuda.empty_cache()
        records.extend(layer_records)
        completed_layers.append(layer_index)
        damping_by_layer[layer_index] = absolute_damping
        common._atomic_json(
            shard_path,
            _profile_payload(
                status="running",
                args=args,
                configuration=configuration,
                assigned_layers=assigned_layers,
                completed_layers=completed_layers,
                anchor_metrics=anchor_metrics,
                records=records,
                damping_by_layer=damping_by_layer,
                started=started,
            ),
        )
        del objective, anchor_v, anchor_o, layer_records
        torch.cuda.empty_cache()
        _log(
            f"checkpointed {len(completed_layers)}/{len(assigned_layers)} layers",
            shard=args.profile_shard_index,
        )

    common._atomic_json(
        shard_path,
        _profile_payload(
            status="complete",
            args=args,
            configuration=configuration,
            assigned_layers=assigned_layers,
            completed_layers=completed_layers,
            anchor_metrics=anchor_metrics,
            records=records,
            damping_by_layer=damping_by_layer,
            started=started,
        ),
    )
    _log(f"completed {shard_path}", shard=args.profile_shard_index)


def _merge_profile_shards(
    profile_dir: Path,
    *,
    shard_count: int,
    configuration: Mapping[str, Any],
    candidate_ranks: Sequence[int],
) -> dict[str, Any]:
    shards = []
    for shard_index in range(shard_count):
        path = profile_dir / f"shard_{shard_index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != PROFILE_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"profile shard is incomplete: {path}")
        if int(payload.get("profile_shard_index", -1)) != shard_index:
            raise ValueError(f"profile shard index mismatch: {path}")
        if int(payload.get("profile_shard_count", -1)) != shard_count:
            raise ValueError(f"profile shard count mismatch: {path}")
        if payload.get("configuration") != configuration:
            raise ValueError(f"profile shard configuration changed: {path}")
        expected_layers = _shard_layers(shard_index, shard_count)
        if tuple(map(int, payload.get("assigned_layers", ()))) != expected_layers:
            raise ValueError(f"profile shard assignment mismatch: {path}")
        if tuple(sorted(map(int, payload.get("completed_layers", ())))) != expected_layers:
            raise ValueError(f"profile shard missed assigned layers: {path}")
        shards.append((path, payload))
    records = sorted(
        [dict(row) for _, shard in shards for row in shard["records"]],
        key=lambda row: (row["layer"], row["source"], row["candidate_rank"]),
    )
    expected_records = common.NUM_LAYERS * common.NUM_KV_HEADS * (
        len(candidate_ranks) - 1
    )
    if len(records) != expected_records:
        raise ValueError("merged profile does not contain every intervention")
    keys = {
        (int(row["layer"]), int(row["source"]), int(row["candidate_rank"]))
        for row in records
    }
    if len(keys) != expected_records:
        raise ValueError("merged profile contains duplicate interventions")
    damping = {}
    for _, shard in shards:
        for layer, value in shard["absolute_covariance_damping_by_layer"].items():
            layer_index = int(layer)
            if layer_index in damping:
                raise ValueError("profile shards duplicate covariance damping")
            damping[layer_index] = float(value)
    if set(damping) != set(range(common.NUM_LAYERS)):
        raise ValueError("profile shards lack per-layer covariance damping")
    return {
        "records": records,
        "absolute_covariance_damping_by_layer": [
            damping[layer] for layer in range(common.NUM_LAYERS)
        ],
        "uniform_anchor_by_shard": [
            shard["uniform_anchor"] for _, shard in shards
        ],
        "shards": [
            {
                "path": str(path),
                "sha256": common._sha256(path),
                "assigned_layers": shard["assigned_layers"],
                "elapsed_seconds": shard["elapsed_seconds"],
            }
            for path, shard in shards
        ],
    }


@torch.inference_mode()
def _finalize(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    started = time.perf_counter()
    (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        _,
        confirmation_sequences,
        windows_provenance,
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    merged = _merge_profile_shards(
        profile_dir,
        shard_count=args.profile_shard_count,
        configuration=configuration,
        candidate_ranks=candidate_ranks,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    tokenizer = (
        None
        if args.skip_wikitext
        else AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=args.local_files_only, use_fast=True
        )
    )
    model = _load_model(args, model_path)
    confirmation_teacher = common._capture_teacher(
        model,
        confirmation_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="confirmation",
    )
    snapshot_cache, snapshot_manifest = common._load_snapshot_cache(
        snapshot_dir,
        model_path=model_path,
    )
    factor_cache = common._load_factor_cache(factor_dirs, factor_results)
    layers = common._decoder_layers(model)
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone() for layer in layers
    )
    uniform_schedule = [
        [args.anchor_rank] * common.NUM_KV_HEADS
        for _ in range(common.NUM_LAYERS)
    ]
    _, anchor_install = common._install_schedule(
        model,
        schedule=uniform_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=0.0,
        keep_factors=False,
    )

    candidates = {}
    predicted = {}
    contributions = {}
    for label, cost_key in (
        ("mean_dp", "mean"),
        ("ucb_dp", "one_standard_error_ucb"),
    ):
        schedule, cost, rows = common._allocate(
            merged["records"],
            candidate_ranks=candidate_ranks,
            anchor_rank=args.anchor_rank,
            cost_key=cost_key,
        )
        candidates[label] = schedule
        predicted[label] = cost
        contributions[label] = rows

    schedules = {"uniform_anchor": uniform_schedule, **candidates}
    confirmation = {}
    closure_diagnostics = {"uniform_anchor": anchor_install}
    for label, schedule in schedules.items():
        if label != "uniform_anchor":
            _, diagnostics = common._install_schedule(
                model,
                schedule=schedule,
                dense_v_weights=dense_v_weights,
                factor_cache=factor_cache,
                snapshot_cache=snapshot_cache,
                anchor_rank=args.anchor_rank,
                covariance_damping=args.covariance_damping,
                decoder_relative_jitter=0.0,
                keep_factors=False,
            )
            closure_diagnostics[label] = diagnostics
        confirmation[label] = common._evaluate_teacher_metrics(
            model,
            confirmation_teacher,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {label}: "
            f"KL={confirmation[label]['terminal_kl']['mean']:.9g}"
        )

    selected_name = min(
        schedules,
        key=lambda label: (confirmation[label]["terminal_kl"]["mean"], label),
    )
    selected_schedule = schedules[selected_name]
    selected_factors, selected_install = common._install_schedule(
        model,
        schedule=selected_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=0.0,
        keep_factors=True,
    )
    assert selected_factors is not None
    closure_diagnostics[selected_name] = selected_install
    test_metrics = {}
    if not args.skip_wikitext:
        assert tokenizer is not None
        selected_test = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset="wikitext2",
            split="test",
            seqlen=args.eval_seqlen,
            batch_size=args.eval_batch_size,
            max_samples=args.eval_max_samples,
            max_tokens=args.eval_max_tokens,
        )
        test_metrics[selected_name] = selected_test
        if selected_name != "uniform_anchor":
            common._install_schedule(
                model,
                schedule=uniform_schedule,
                dense_v_weights=dense_v_weights,
                factor_cache=factor_cache,
                snapshot_cache=snapshot_cache,
                anchor_rank=args.anchor_rank,
                covariance_damping=args.covariance_damping,
                decoder_relative_jitter=0.0,
                keep_factors=False,
            )
            test_metrics["uniform_anchor"] = _eval_ppl_fp32_loss(
                model,
                tokenizer,
                dataset="wikitext2",
                split="test",
                seqlen=args.eval_seqlen,
                batch_size=args.eval_batch_size,
                max_samples=args.eval_max_samples,
                max_tokens=args.eval_max_tokens,
            )
        _log(f"selected {selected_name}: PPL={selected_test['ppl']:.9f}")
    else:
        _log(f"selected {selected_name}: pre-ALS WikiText skipped")

    output_dir.mkdir(parents=True, exist_ok=False)
    selected_dir = output_dir / "selected_factors"
    selected_dir.mkdir()
    artifacts = {}
    for layer_index, factors in enumerate(selected_factors):
        path = selected_dir / f"layer_{layer_index:03d}.safetensors"
        save_file(
            {
                "value_coordinate_encoders": factors.A,
                "head_output_decoders": factors.D,
                "source_ranks": torch.tensor(factors.ranks, dtype=torch.int32),
            },
            str(path),
        )
        artifacts[str(layer_index)] = {
            "file": str(path.relative_to(output_dir)),
            "sha256": common._sha256(path),
            "encoder_sha256": tensor_sha256(factors.A),
            "decoder_sha256": tensor_sha256(factors.D),
        }

    schedule_rows = {}
    for label, schedule in schedules.items():
        row = {
            "schedule": schedule,
            "accounting": common._schedule_accounting(
                schedule, anchor_rank=args.anchor_rank
            ),
            "confirmation": confirmation[label],
            "closure_diagnostics": closure_diagnostics[label],
        }
        if label in test_metrics:
            row["test"] = test_metrics[label]
        schedule_rows[label] = row
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    result = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": str(model_path),
        "model_config_sha256": common._sha256(model_path / "config.json"),
        "geometry": {
            "layers": common.NUM_LAYERS,
            "query_heads": common.NUM_QUERY_HEADS,
            "physical_kv_heads": common.NUM_KV_HEADS,
            "tp_sources": common.NUM_KV_HEADS,
            "query_heads_per_source": common.HEADS_PER_SOURCE,
            "head_dim": common.HEAD_DIM,
        },
        "profile": {
            "execution": "two independent 2-GPU model replicas",
            "factor_stage": configuration["factor_stage"],
            "profile_shard_count": args.profile_shard_count,
            "dataset": "c4_train_fresh_documents",
            "windows": args.profile_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_candidate": math.ceil(
                args.profile_windows / args.batch_size
            ),
            "uniform_anchor_by_shard": merged["uniform_anchor_by_shard"],
            "records": merged["records"],
            "absolute_covariance_damping_by_layer": merged[
                "absolute_covariance_damping_by_layer"
            ],
            "shards": merged["shards"],
        },
        "confirmation": {
            "dataset": "c4_train_fresh_documents",
            "windows": args.confirmation_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_schedule": math.ceil(
                args.confirmation_windows / args.batch_size
            ),
            "disjoint_from_profile": True,
            "disjoint_from_als_fit_and_heldout": windows_provenance["disjoint_from_als_fit_and_heldout_prefix"],
            "windows_provenance": windows_provenance,
        },
        "selection": {
            "constraint": "exact ideal variable-width TP-source rank budget",
            "target_source_rank_sum": (
                common.NUM_LAYERS * common.NUM_KV_HEADS * args.anchor_rank
            ),
            "candidate_ranks": list(candidate_ranks),
            "rank_128_endpoint": "exact identity Value encoder plus dense O initialization",
            "predicted_additive_costs": predicted,
            "contributions": contributions,
            "selected_candidate": selected_name,
            "selected_schedule": selected_schedule,
            "selected_accounting": common._schedule_accounting(
                selected_schedule, anchor_rank=args.anchor_rank
            ),
            "uniform_is_eligible": True,
            "selection_metric": "lowest disjoint-confirmation mean terminal KL",
        },
        "schedules": schedule_rows,
        "selected_artifacts": artifacts,
        "factor_sources": {
            str(rank): {
                "path": str(factor_dirs[rank]),
                "results_sha256": common._sha256(factor_dirs[rank] / "results.json"),
            }
            for rank in factor_dirs
        },
        "snapshot": {
            "path": str(snapshot_dir),
            "manifest_sha256": common._sha256(snapshot_dir / "manifest.json"),
            "fit_windows": snapshot_manifest["calibration"]["fit_windows"],
            "fit_rows": snapshot_manifest["calibration"]["fit_rows"],
            "heldout_not_used_for_global_kl_selection": True,
        },
        "test_protocol": {
            "dataset": "wikitext2_test",
            "seqlen": args.eval_seqlen,
            "batch_size": args.eval_batch_size,
            "schedule_frozen_before_test": True,
            "skipped_for_post_allocation_als": bool(args.skip_wikitext),
        },
        "numerics": {
            "model_dtype": str(dtype),
            "deployed_factor_dtype": "torch.bfloat16",
            "decoder_closure": "float32",
            "terminal_kl_probability": "float32",
            "terminal_kl_accumulation": "float64",
            "quality_runtime": (
                "ragged source coordinates exactly zero-padded into native "
                "Qwen V/O slots for sharded Hugging Face SDPA"
            ),
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in _cuda_device_indices()
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in _cuda_device_indices()
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    common._atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(common._summary(result), encoding="utf-8")
    _log(f"wrote {output_dir}")


def main() -> None:
    args = parse_args()
    if args.stage == "profile":
        _profile(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
