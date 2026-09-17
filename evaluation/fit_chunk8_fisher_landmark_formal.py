"""Formal packed-window direct Chunk8 Fisher fitting for Llama-3.1-8B-Instruct."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.chunk_fisher_landmark import (
    PackedChunkFisherDataset,
    PackedChunkFisherWindow,
    adapt_token_residual_initialization,
    chunk_mean_logits,
    chunk_teacher_logits,
    compact_chunk_fisher_metrics,
    compact_packed_chunk_fisher_dataset,
    fit_compact_chunk_fisher_landmarks,
    fit_packed_chunk_fisher_landmarks,
    packed_chunk_fisher_metrics,
    packed_factor_logits,
)
from evaluation.fit_k_routing_streaming import verified
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json


CHUNK_SIZE = 8
SINK_TOKENS = 32
RECENT_TOKENS = 64
RESIDUAL_RANK = 16
FORMAT = "basisserve.chunk8_fisher_landmark.packed.v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--capture-root", type=Path)
    result.add_argument("--layer", type=int, required=True)
    result.add_argument("--fit-count", type=int, default=64)
    result.add_argument("--heldout-count", type=int, default=16)
    result.add_argument("--fit-queries", type=int, default=64)
    result.add_argument("--heldout-queries", type=int, default=32)
    result.add_argument("--mean-sweeps", type=int, default=8)
    result.add_argument("--flat-sweeps", type=int, default=4)
    result.add_argument("--relative-damping", type=float, default=1e-5)
    result.add_argument("--cg-tolerance", type=float, default=1e-5)
    result.add_argument("--cg-iterations", type=int, default=50)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--pilot", action="store_true")
    return result


def selected_positions(values: list[int], count: int, smoke: bool) -> list[int]:
    assert 0 < count <= len(values)
    selected = values[-count:] if smoke else values[:count]
    result = [
        value
        for value in selected
        if value + 1 > SINK_TOKENS + RECENT_TOKENS
    ]
    candidate = 127
    while len(result) < count:
        if candidate not in selected:
            result.append(candidate)
        candidate += 64
    result.sort()
    assert result == sorted(result) and len(set(result)) == len(result)
    assert all((value + 1) % 64 == 0 for value in result)
    assert min(result) + 1 > SINK_TOKENS + RECENT_TOKENS
    return result


def packed_windows_from_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    base_post: torch.Tensor,
    *,
    positions: list[int],
    feature_types: tuple[str, ...] = ("mean", "flat"),
    output_device: torch.device | str = "cpu",
) -> dict[str, PackedChunkFisherWindow]:
    """Build mean and flat views sharing one set of direct Chunk8 targets."""

    assert query.ndim == key.ndim == base_post.ndim == 4
    assert query.shape[0] == key.shape[0] == base_post.shape[0] == 1
    heads = int(query.shape[1])
    groups = int(key.shape[1])
    head_dim = int(query.shape[-1])
    heads_per_group = heads // groups
    pinned_chunks = SINK_TOKENS // CHUNK_SIZE
    maximum_complete = (max(positions) + 1 - RECENT_TOKENS) // CHUNK_SIZE
    maximum_candidates = maximum_complete - pinned_chunks
    complete_tokens = maximum_complete * CHUNK_SIZE
    residual_chunks = (
        (key - base_post)[0, :, :complete_tokens]
        .reshape(groups, maximum_complete, CHUNK_SIZE, head_dim)
    )
    assert feature_types and set(feature_types).issubset({"mean", "flat"})
    device = torch.device(output_device)
    feature_builders = {
        "mean": lambda: residual_chunks[:, pinned_chunks:].float().mean(dim=-2),
        "flat": lambda: residual_chunks[:, pinned_chunks:].flatten(-2),
    }
    features = {
        feature: feature_builders[feature]().to(device)
        for feature in feature_types
    }
    query_rows = (
        query[0, :, positions].permute(1, 0, 2).contiguous().to(device)
    )
    base_logits = torch.zeros(
        len(positions),
        heads,
        maximum_candidates,
        device=device,
    )
    teacher_logits = torch.zeros_like(base_logits)
    candidate_counts = torch.empty(len(positions), dtype=torch.long, device=device)
    scale = head_dim**-0.5
    for index, position in enumerate(positions):
        complete = (int(position) + 1 - RECENT_TOKENS) // CHUNK_SIZE
        candidates = complete - pinned_chunks
        candidate_counts[index] = candidates
        stop = complete * CHUNK_SIZE
        grouped_query = query[0, :, position].float().reshape(
            groups,
            heads_per_group,
            head_dim,
        )
        teacher = chunk_teacher_logits(
            grouped_query,
            key[0, :, :stop].float(),
            chunk_size=CHUNK_SIZE,
            scaling=scale,
        )[..., pinned_chunks:]
        base = chunk_mean_logits(
            grouped_query,
            base_post[0, :, :stop].float(),
            chunk_size=CHUNK_SIZE,
            scaling=scale,
        )[..., pinned_chunks:]
        teacher_logits[index, :, :candidates].copy_(
            teacher.reshape(heads, candidates).to(device)
        )
        base_logits[index, :, :candidates].copy_(
            base.reshape(heads, candidates).to(device)
        )
    common = {
        "positions": torch.tensor(positions, dtype=torch.long, device=device),
        "queries": query_rows,
        "base_logits": base_logits,
        "teacher_logits": teacher_logits,
        "candidate_counts": candidate_counts,
    }
    result = {
        feature: PackedChunkFisherWindow(
            features_by_group=feature_rows,
            **common,
        )
        for feature, feature_rows in features.items()
    }
    mapping = torch.arange(heads) // heads_per_group
    for window in result.values():
        window.validate(mapping)
    return result


def packed_dataset_to(
    dataset: PackedChunkFisherDataset,
    device: torch.device,
) -> PackedChunkFisherDataset:
    windows = tuple(
        PackedChunkFisherWindow(
            positions=window.positions.to(device),
            queries=window.queries.to(device=device, dtype=torch.float32),
            features_by_group=window.features_by_group.to(
                device=device,
                dtype=torch.float32,
            ),
            base_logits=window.base_logits.to(device=device, dtype=torch.float32),
            teacher_logits=window.teacher_logits.to(
                device=device,
                dtype=torch.float32,
            ),
            candidate_counts=window.candidate_counts.to(device),
        )
        for window in dataset.windows
    )
    result = PackedChunkFisherDataset(
        windows,
        dataset.head_to_group.to(device),
        dataset.scaling,
    )
    result.validate()
    return result


def capture(
    args: argparse.Namespace,
    identity: dict,
    factors: dict[str, torch.Tensor],
):
    model = AutoModelForCausalLM.from_pretrained(
        identity["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval().cuda()
    assert model.config.model_type == "llama"
    assert model.config.num_attention_heads == 32
    assert model.config.num_key_value_heads == 8
    assert model.config.hidden_size // model.config.num_attention_heads == 128
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    windows = load_file(str(args.root / "calibration/windows.safetensors"))["input_ids"]
    assert windows.shape == (80, 65_536)
    moments_path = args.root / "moments" / f"layer_{args.layer:03d}.json"
    moments = read_json(moments_path)
    assert moments["status"] == "complete"
    assert moments["protocol"]["windows_sha256"] == sha256(
        args.root / "calibration/windows.safetensors"
    )
    selections = {
        "fit": selected_positions(
            moments["selections"]["fit"]["selected_positions"],
            args.fit_queries,
            args.smoke,
        ),
        "heldout": selected_positions(
            moments["selections"]["heldout"]["selected_positions"],
            args.heldout_queries,
            args.smoke,
        ),
    }
    split_windows = {
        "fit": list(range(args.fit_count)),
        "heldout": list(range(64, 64 + args.heldout_count)),
    }
    collected = {
        feature: {"fit": [], "heldout": []}
        for feature in ("mean", "flat")
    }
    active = {}

    def hook(attention, positional, kwargs):
        hidden = kwargs["hidden_states"]
        length = int(hidden.shape[1])
        query = attention.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
        key = attention.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
        value = attention.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        sidecar = build_conditional_routing_sidecar(
            value,
            key,
            base_left=factors["base_left_b16"],
            base_right=factors["base_right_b16"],
            base_bias=factors["base_bias_b16"],
            residual_encoder=factors["residual_encoder_b16_r16"],
            cos=cos,
            sin=sin,
        )
        built = packed_windows_from_attention(
            query,
            key,
            sidecar[..., :128],
            positions=selections[active["split"]],
        )
        for feature, window in built.items():
            collected[feature][active["split"]].append(window)
        active["captured"] = True

    handle = model.model.layers[args.layer].self_attn.register_forward_pre_hook(
        hook,
        with_kwargs=True,
    )
    for split, indices in split_windows.items():
        for index in indices:
            active.update(split=split, index=index, captured=False)
            output = model.model(windows[index : index + 1].long().cuda(), use_cache=False)
            assert active["captured"] and torch.isfinite(output.last_hidden_state).all()
            del output
            print(
                {
                    "stage": "capture",
                    "split": split,
                    "window": index,
                    "layer": args.layer,
                    "query_count": len(selections[split]),
                },
                flush=True,
            )
    handle.remove()
    del model, windows
    gc.collect()
    torch.cuda.empty_cache()
    mapping = torch.arange(32, dtype=torch.long) // 4
    datasets = {
        feature: {
            split: PackedChunkFisherDataset(
                tuple(collected[feature][split]),
                mapping,
                128**-0.5,
            )
            for split in ("fit", "heldout")
        }
        for feature in ("mean", "flat")
    }
    for variants in datasets.values():
        for dataset in variants.values():
            dataset.validate()
    return datasets, selections, split_windows, moments_path


def load_capture_bank(
    args: argparse.Namespace,
    factor_path: Path,
):
    record_path = args.capture_root / f"layer_{args.layer:03d}.json"
    record = read_json(record_path)
    protocol = record["protocol"]
    assert record["status"] == "complete"
    assert record["layer"] == args.layer
    assert protocol["scope"] == ("smoke" if args.smoke else "formal")
    assert protocol["fit_count"] == args.fit_count
    assert protocol["heldout_count"] == args.heldout_count
    assert protocol["fit_queries"] == args.fit_queries
    assert protocol["heldout_queries"] == args.heldout_queries
    assert protocol["root"] == str(args.root)
    assert record["factor_source_sha256"] == sha256(factor_path)
    windows = {"fit": [], "heldout": []}
    for artifact in record["windows"]:
        path = args.capture_root / artifact["file"]
        assert sha256(path) == artifact["sha256"]
        payload = load_file(str(path))
        flat = payload["flat_features_by_group"]
        groups, chunks, flat_dim = map(int, flat.shape)
        assert flat_dim == CHUNK_SIZE * 128
        mean = flat.reshape(groups, chunks, CHUNK_SIZE, 128).float().mean(dim=-2)
        common = {
            "positions": payload["positions"],
            "queries": payload["queries"],
            "base_logits": payload["base_logits"],
            "teacher_logits": payload["teacher_logits"],
            "candidate_counts": payload["candidate_counts"],
        }
        mapping = torch.arange(32, dtype=torch.long) // 4
        for feature, features in (("mean", mean), ("flat", flat)):
            window = PackedChunkFisherWindow(
                features_by_group=features,
                **common,
            )
            window.validate(mapping)
            windows[artifact["split"]].append((artifact["window"], feature, window))
    datasets = {}
    mapping = torch.arange(32, dtype=torch.long) // 4
    for feature in ("mean", "flat"):
        datasets[feature] = {}
        for split in ("fit", "heldout"):
            selected = sorted(
                (index, window)
                for index, current_feature, window in windows[split]
                if current_feature == feature
            )
            expected = (
                list(range(args.fit_count))
                if split == "fit"
                else list(range(64, 64 + args.heldout_count))
            )
            assert [index for index, _ in selected] == expected
            dataset = PackedChunkFisherDataset(
                tuple(window for _, window in selected),
                mapping,
                128**-0.5,
            )
            dataset.validate()
            datasets[feature][split] = dataset
    return (
        datasets,
        record["query_positions"],
        record["split_windows"],
        Path(record["moments_source"]),
        record_path,
    )


def initial_equivalence(
    datasets: dict[str, dict[str, PackedChunkFisherDataset]],
    factors: dict[str, torch.Tensor],
) -> float:
    adapted = {
        feature: adapt_token_residual_initialization(
            factors["residual_encoder_b16_r16"].cpu(),
            factors["residual_query_b16_r16"].cpu(),
            chunk_size=CHUNK_SIZE,
            feature=feature,
        )
        for feature in ("mean", "flat")
    }
    mean_dataset = datasets["mean"]["fit"]
    flat_dataset = datasets["flat"]["fit"]
    mean_score = packed_factor_logits(
        mean_dataset,
        mean_dataset.windows[0],
        *adapted["mean"],
    )
    flat_score = packed_factor_logits(
        flat_dataset,
        flat_dataset.windows[0],
        *adapted["flat"],
    )
    return float((mean_score - flat_score).abs().max())


def fit_feature(
    args: argparse.Namespace,
    datasets: dict[str, PackedChunkFisherDataset],
    factors: dict[str, torch.Tensor],
    feature: str,
):
    sweep_count = args.mean_sweeps if feature == "mean" else args.flat_sweeps
    device = torch.device("cuda")
    train = packed_dataset_to(datasets["fit"], device)
    heldout = packed_dataset_to(datasets["heldout"], device)
    initial_encoder, initial_query = adapt_token_residual_initialization(
        factors["residual_encoder_b16_r16"],
        factors["residual_query_b16_r16"],
        chunk_size=CHUNK_SIZE,
        feature=feature,
    )
    initial_encoder = initial_encoder.to(device=device, dtype=torch.float32)
    initial_query = initial_query.to(device=device, dtype=torch.float32)
    initial = {
        "fit": packed_chunk_fisher_metrics(train, initial_encoder, initial_query),
        "heldout": packed_chunk_fisher_metrics(
            heldout,
            initial_encoder,
            initial_query,
        ),
    }
    torch.cuda.synchronize()
    statistic_start = time.perf_counter()
    sufficient = feature == "mean"
    if sufficient:
        fit_statistics = compact_packed_chunk_fisher_dataset(train)
        heldout_statistics = compact_packed_chunk_fisher_dataset(heldout)
        compact_initial = {
            "fit": compact_chunk_fisher_metrics(
                fit_statistics,
                initial_encoder,
                initial_query,
            ),
            "heldout": compact_chunk_fisher_metrics(
                heldout_statistics,
                initial_encoder,
                initial_query,
            ),
        }
    else:
        fit_statistics = train
        heldout_statistics = heldout
        compact_initial = initial
    torch.cuda.synchronize()
    statistic_wall_seconds = time.perf_counter() - statistic_start
    fit_start = time.perf_counter()
    fitter = (
        fit_compact_chunk_fisher_landmarks
        if sufficient
        else fit_packed_chunk_fisher_landmarks
    )
    fitted = fitter(
        fit_statistics,
        heldout_statistics,
        initial_encoders=initial_encoder,
        initial_query_factors=initial_query,
        sweeps=sweep_count,
        relative_damping=args.relative_damping,
        relative_tolerance=args.cg_tolerance,
        max_iterations=args.cg_iterations,
    )
    fit_wall_seconds = time.perf_counter() - fit_start
    final = {
        "fit": packed_chunk_fisher_metrics(
            train,
            fitted.encoders,
            fitted.query_factors,
        ),
        "heldout": packed_chunk_fisher_metrics(
            heldout,
            fitted.encoders,
            fitted.query_factors,
        ),
    }
    tensors = {
        "chunk_encoder_r16": fitted.encoders.float().cpu().contiguous(),
        "chunk_query_r16": fitted.query_factors.float().cpu().contiguous(),
    }
    half_steps = [asdict(item) for item in fitted.half_steps]
    sweep_records = []
    for sweep in range(1, sweep_count + 1):
        query_step = next(
            item
            for item in half_steps
            if item["sweep"] == sweep and item["boundary"] == "query"
        )
        encoder_step = next(
            item
            for item in half_steps
            if item["sweep"] == sweep and item["boundary"] == "encoder"
        )
        sweep_records.append(
            {
                "sweep": sweep,
                "train_fisher_nmse": encoder_step["train"]["fisher_nmse"],
                "heldout_fisher_nmse": encoder_step["heldout"]["fisher_nmse"],
                "query_cg": {
                    key: query_step[key]
                    for key in (
                        "solver_count",
                        "converged_count",
                        "hit_iteration_limit_count",
                        "total_iterations",
                        "mean_iterations",
                        "maximum_iterations",
                    )
                },
                "encoder_cg": {
                    key: encoder_step[key]
                    for key in (
                        "solver_count",
                        "converged_count",
                        "hit_iteration_limit_count",
                        "total_iterations",
                        "mean_iterations",
                        "maximum_iterations",
                    )
                },
                "query_solve_wall_seconds": query_step["solve_wall_seconds"],
                "encoder_solve_wall_seconds": encoder_step["solve_wall_seconds"],
                "diagnostic_wall_seconds": query_step["diagnostic_wall_seconds"]
                + encoder_step["diagnostic_wall_seconds"],
                "wall_seconds": query_step["total_wall_seconds"]
                + encoder_step["total_wall_seconds"],
            }
        )
    diagnostics = {
        "initial": initial,
        "final": final,
        "sufficient_statistics": (
            "exact per-query 128x128 Fisher Gram and target cross"
            if sufficient
            else "exact matrix-free candidate scan with reusable feature-Fisher Kronecker preconditioner for 1024D Flat feature"
        ),
        "sufficient_statistics_wall_seconds": statistic_wall_seconds,
        "sufficient_statistics_initial_fisher_max_abs_difference": max(
            abs(initial[split]["fisher_loss"] - compact_initial[split]["fisher_loss"])
            for split in ("fit", "heldout")
        ),
        "half_steps": half_steps,
        "sweeps": sweep_records,
        "fit_wall_seconds_including_diagnostics_and_final_closure": fit_wall_seconds,
        "preconditioner_wall_seconds": fitted.preconditioner_wall_seconds,
        "final_query_maximum_iterations": max(
            item.iterations for item in fitted.final_query_diagnostics
        ),
        "final_query_maximum_relative_residual": max(
            item.relative_residual for item in fitted.final_query_diagnostics
        ),
    }
    del train, heldout, fit_statistics, heldout_statistics
    del fitted, initial_encoder, initial_query
    gc.collect()
    torch.cuda.empty_cache()
    return tensors, diagnostics


def result_markdown(result: dict) -> str:
    lines = [
        f"# Layer {result['protocol']['layer']} Direct Chunk8 Fisher Fit",
        "",
        "| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for feature in ("mean", "flat"):
        for split in ("fit", "heldout"):
            initial = result["features"][feature]["initial"][split]
            final = result["features"][feature]["final"][split]
            lines.append(
                f"| {feature} | {split} | {initial['fisher_nmse']:.6g} | "
                f"{final['fisher_nmse']:.6g} | {final['chunk_logit_rel_mse']:.6g} | "
                f"{final['exact_chunk_support_recall']:.4f} | "
                f"{final['routed_candidate_attention_mass']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    configure()
    assert 0 <= args.layer < 32
    assert args.mean_sweeps > 0 and args.flat_sweeps > 0
    assert args.cg_iterations > 0
    assert not (args.smoke and args.pilot)
    if args.smoke:
        assert args.fit_count == args.heldout_count == 1
        assert 1 <= args.fit_queries <= 64 and 1 <= args.heldout_queries <= 32
    else:
        assert args.fit_count == 64 and args.heldout_count == 16
        assert args.fit_queries == 64 and args.heldout_queries == 32
    identity = read_json(args.root / "manifests/v128.json")
    assert identity["value_mode"] == "dense original V and Wo"
    assert identity["layer_ranks"] == [128] * 32
    factor_path = args.root / "ours_b16r16" / f"layer_{args.layer:03d}.safetensors"
    factors, factor_record = verified(factor_path)
    assert factor_record["v_rank"] == 128
    factors = {name: value.cuda() for name, value in factors.items()}
    capture_record_path = None
    if args.capture_root is None:
        datasets, selections, split_windows, moments_path = capture(
            args,
            identity,
            factors,
        )
    else:
        (
            datasets,
            selections,
            split_windows,
            moments_path,
            capture_record_path,
        ) = load_capture_bank(args, factor_path)
    equivalence = initial_equivalence(datasets, factors)
    assert equivalence < 2e-5
    feature_results = {}
    artifacts = {}
    for feature in ("mean", "flat"):
        tensors, diagnostics = fit_feature(
            args,
            datasets[feature],
            factors,
            feature,
        )
        path = args.output / f"layer_{args.layer:03d}_{feature}.safetensors"
        save_tensors(path, tensors)
        artifacts[feature] = {
            "file": path.name,
            "sha256": sha256(path),
            "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        }
        feature_results[feature] = diagnostics
        print(
            {
                "stage": "fit complete",
                "feature": feature,
                "layer": args.layer,
                "initial": diagnostics["initial"],
                "final": diagnostics["final"],
            },
            flush=True,
        )
    protocol = {
        "format": FORMAT,
        "scope": (
            "packed smoke"
            if args.smoke
            else "three-layer convergence pilot"
            if args.pilot
            else "formal 64-fit/16-heldout fit"
        ),
        "model": identity["model"],
        "model_variant": "Llama-3.1-8B-Instruct",
        "layer": args.layer,
        "value_path": "Dense V128 identity",
        "base_rank": 16,
        "residual_rank": RESIDUAL_RANK,
        "chunk_size": CHUNK_SIZE,
        "sequence_length": 65_536,
        "fit_windows": split_windows["fit"],
        "heldout_windows": split_windows["heldout"],
        "query_positions": selections,
        "feature_types": ["mean", "flat"],
        "objective": "direct exact-teacher Chunk8 softmax-Fisher over routed historical candidates only",
        "candidate_mask": "exclude fixed sink32 and exact recent64",
        "als_sweeps": {
            "mean": args.mean_sweeps,
            "flat": args.flat_sweeps,
        },
        "final_query_closure": True,
        "relative_damping": args.relative_damping,
        "cg_tolerance": args.cg_tolerance,
        "cg_iterations": args.cg_iterations,
        "initialization": "old token B16R16 factors adapted as initialization only",
        "deployment": "one E_g per KV group and one U_h per query head; no token R16 state",
        "als_linear_algebra": "Mean uses exact per-query Fisher Gram/cross sufficient statistics; Flat keeps exact matrix-free normal-equation products and reuses an aggregate feature-Fisher Kronecker preconditioner",
        "factor_source": str(factor_path),
        "factor_source_sha256": sha256(factor_path),
        "moments_source": str(moments_path),
        "moments_source_sha256": sha256(moments_path),
        "identity_sha256": sha256(args.root / "manifests/v128.json"),
        "windows_sha256": sha256(args.root / "calibration/windows.safetensors"),
        "source_sha256": {
            name: sha256(Path(name))
            for name in (
                "basisserve/core/chunk_fisher_landmark.py",
                "evaluation/fit_chunk8_fisher_landmark_formal.py",
            )
        },
    }
    if capture_record_path is not None:
        protocol["capture_bank"] = str(args.capture_root)
        protocol["capture_record"] = str(capture_record_path)
        protocol["capture_record_sha256"] = sha256(capture_record_path)
    result = {
        "status": "complete",
        "protocol": protocol,
        "audits": {
            "mean_and_flat_initial_scores_max_abs_difference": equivalence,
            "query_positions_reuse_audited_query_gram_selection": True,
            "sink_chunks_excluded_from_fisher": True,
            "recent64_excluded_from_fisher": True,
            "base_is_frozen": True,
            "token_r16_is_initialization_only": True,
            "group_encoder_is_shared": True,
            "query_factor_is_head_specific": True,
            "configured_sweeps_plus_final_query_closure": True,
        },
        "features": feature_results,
        "artifacts": artifacts,
        "command": shlex.join(sys.argv),
        "python": sys.executable,
    }
    record_path = args.output / f"layer_{args.layer:03d}.json"
    write_json(record_path, result)
    markdown_path = args.output / f"layer_{args.layer:03d}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(result_markdown(result))
    print(result_markdown(result), flush=True)


if __name__ == "__main__":
    main()
