"""Smoke-test direct MeanResidual and FlatResidual Chunk8 Fisher ALS."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import math
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
)
from basisserve.core.chunk_fisher_landmark import (
    ChunkFisherDataset,
    ChunkFisherExample,
    adapt_token_residual_initialization,
    chunk_fisher_metrics,
    chunk_mean_logits,
    chunk_teacher_logits,
    factor_logits,
    fisher_multiply,
    fit_chunk_fisher_landmarks,
    residual_chunk_features,
)
from evaluation.fit_k_routing_streaming import verified
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json


CHUNK_SIZE = 8
SINK_TOKENS = 32
RECENT_TOKENS = 64
RESIDUAL_RANK = 16
FORMAT = "basisserve.chunk8_fisher_landmark.smoke.v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--layer", type=int, default=0)
    result.add_argument("--fit-count", type=int, default=1)
    result.add_argument("--heldout-count", type=int, default=1)
    result.add_argument("--fit-queries", type=int, default=1)
    result.add_argument("--heldout-queries", type=int, default=1)
    result.add_argument("--sweeps", type=int, default=4)
    result.add_argument("--relative-damping", type=float, default=1e-5)
    result.add_argument("--cg-tolerance", type=float, default=1e-5)
    result.add_argument("--cg-iterations", type=int, default=100)
    return result


def choose_positions(positions: list[int], count: int) -> list[int]:
    """Use the latest audited long-context Query-Gram positions for smoke."""

    assert 0 < count <= len(positions)
    selected = positions[-count:]
    assert min(selected) + 1 > SINK_TOKENS + RECENT_TOKENS
    return selected


def examples_from_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    base_post: torch.Tensor,
    *,
    positions: list[int],
    feature: str,
) -> tuple[ChunkFisherExample, ...]:
    """Create causal candidate-only examples without sink or recent chunks."""

    assert query.ndim == key.ndim == base_post.ndim == 4
    assert query.shape[0] == key.shape[0] == base_post.shape[0] == 1
    heads = int(query.shape[1])
    groups = int(key.shape[1])
    head_dim = int(query.shape[-1])
    heads_per_group = heads // groups
    mapping = torch.arange(heads, device=query.device) // heads_per_group
    assert key.shape == base_post.shape
    residual = key - base_post
    maximum_complete = (max(positions) + 1 - RECENT_TOKENS) // CHUNK_SIZE
    complete_tokens = maximum_complete * CHUNK_SIZE
    all_features = residual_chunk_features(
        residual[0, :, :complete_tokens].float(),
        chunk_size=CHUNK_SIZE,
        feature=feature,
    ).cpu()
    examples = []
    scale = head_dim**-0.5
    pinned_chunks = SINK_TOKENS // CHUNK_SIZE
    for position in positions:
        complete = (int(position) + 1 - RECENT_TOKENS) // CHUNK_SIZE
        assert complete > pinned_chunks
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
        candidate_count = complete - pinned_chunks
        examples.append(
            ChunkFisherExample(
                position=int(position),
                queries=query[0, :, position].float().cpu(),
                features_by_group=all_features[:, pinned_chunks:complete],
                base_logits=base.reshape(heads, candidate_count).cpu(),
                teacher_logits=teacher.reshape(heads, candidate_count).cpu(),
            )
        )
    result = tuple(examples)
    for example in result:
        example.validate(mapping.cpu())
    return result


def dataset_to(dataset: ChunkFisherDataset, device: torch.device) -> ChunkFisherDataset:
    examples = tuple(
        ChunkFisherExample(
            position=example.position,
            queries=example.queries.to(device=device, dtype=torch.float32),
            features_by_group=example.features_by_group.to(
                device=device,
                dtype=torch.float32,
            ),
            base_logits=example.base_logits.to(device=device, dtype=torch.float32),
            teacher_logits=example.teacher_logits.to(
                device=device,
                dtype=torch.float32,
            ),
        )
        for example in dataset.examples
    )
    result = ChunkFisherDataset(
        examples,
        dataset.head_to_group.to(device),
        dataset.scaling,
    )
    result.validate()
    return result


def capture(args: argparse.Namespace, identity: dict, factors: dict[str, torch.Tensor]):
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
    moments = read_json(args.root / "moments" / f"layer_{args.layer:03d}.json")
    assert moments["status"] == "complete"
    assert moments["protocol"]["windows_sha256"] == sha256(
        args.root / "calibration/windows.safetensors"
    )
    selections = {
        "fit": choose_positions(
            moments["selections"]["fit"]["selected_positions"],
            args.fit_queries,
        ),
        "heldout": choose_positions(
            moments["selections"]["heldout"]["selected_positions"],
            args.heldout_queries,
        ),
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
        base_post = sidecar[..., :128]
        for feature in ("mean", "flat"):
            collected[feature][active["split"]].extend(
                examples_from_attention(
                    query,
                    key,
                    base_post,
                    positions=selections[active["split"]],
                    feature=feature,
                )
            )
        direct_base = base_post.float().reshape(1, 8, length // CHUNK_SIZE, CHUNK_SIZE, 128).mean(-2)
        rebuilt_base = sidecar[..., :128].float().reshape(1, 8, length // CHUNK_SIZE, CHUNK_SIZE, 128).mean(-2)
        torch.testing.assert_close(direct_base, rebuilt_base)
        active["captured"] = True

    handle = model.model.layers[args.layer].self_attn.register_forward_pre_hook(
        hook,
        with_kwargs=True,
    )
    split_windows = {
        "fit": list(range(args.fit_count)),
        "heldout": list(range(64, 64 + args.heldout_count)),
    }
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
                    "positions": selections[split],
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
            split: ChunkFisherDataset(
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
    return datasets, selections, split_windows


def initial_equivalence(
    datasets: dict[str, dict[str, ChunkFisherDataset]],
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
    maximum = 0.0
    for split in ("fit", "heldout"):
        mean = datasets["mean"][split]
        flat = datasets["flat"][split]
        for mean_example, flat_example in zip(
            mean.examples,
            flat.examples,
            strict=True,
        ):
            mean_score = factor_logits(mean, mean_example, *adapted["mean"])
            flat_score = factor_logits(flat, flat_example, *adapted["flat"])
            maximum = max(maximum, float((mean_score - flat_score).abs().max()))
    return maximum


def fit_feature(
    args: argparse.Namespace,
    datasets: dict[str, ChunkFisherDataset],
    factors: dict[str, torch.Tensor],
    feature: str,
):
    device = torch.device("cuda")
    train = dataset_to(datasets["fit"], device)
    heldout = dataset_to(datasets["heldout"], device)
    initial_encoder, initial_query = adapt_token_residual_initialization(
        factors["residual_encoder_b16_r16"],
        factors["residual_query_b16_r16"],
        chunk_size=CHUNK_SIZE,
        feature=feature,
    )
    initial_encoder = initial_encoder.to(device=device, dtype=torch.float32)
    initial_query = initial_query.to(device=device, dtype=torch.float32)
    initial = {
        "fit": chunk_fisher_metrics(train, initial_encoder, initial_query),
        "heldout": chunk_fisher_metrics(heldout, initial_encoder, initial_query),
    }
    fitted = fit_chunk_fisher_landmarks(
        train,
        heldout,
        initial_encoders=initial_encoder,
        initial_query_factors=initial_query,
        sweeps=args.sweeps,
        relative_damping=args.relative_damping,
        relative_tolerance=args.cg_tolerance,
        max_iterations=args.cg_iterations,
    )
    final = {
        "fit": chunk_fisher_metrics(train, fitted.encoders, fitted.query_factors),
        "heldout": chunk_fisher_metrics(
            heldout,
            fitted.encoders,
            fitted.query_factors,
        ),
    }
    tensors = {
        "chunk_encoder_r16": fitted.encoders.float().cpu().contiguous(),
        "chunk_query_r16": fitted.query_factors.float().cpu().contiguous(),
    }
    diagnostics = {
        "initial": initial,
        "final": final,
        "half_steps": [asdict(item) for item in fitted.half_steps],
        "final_query_maximum_iterations": max(
            item.iterations for item in fitted.final_query_diagnostics
        ),
        "final_query_maximum_relative_residual": max(
            item.relative_residual for item in fitted.final_query_diagnostics
        ),
    }
    del train, heldout, fitted, initial_encoder, initial_query
    gc.collect()
    torch.cuda.empty_cache()
    return tensors, diagnostics


def fisher_audit() -> bool:
    generator = torch.Generator().manual_seed(20260917)
    logits = torch.randn(11, generator=generator, dtype=torch.float64)
    values = torch.randn(11, generator=generator, dtype=torch.float64)
    probability = logits.softmax(-1)
    dense = torch.diag(probability) - torch.outer(probability, probability)
    torch.testing.assert_close(fisher_multiply(probability, values), dense @ values)
    return True


def summary(result: dict) -> str:
    lines = [
        "# Llama-3.1-8B-Instruct Direct Chunk8 Fisher ALS Smoke",
        "",
        "This smoke uses Dense V128, frozen Base16, audited 64K calibration windows, four ALS sweeps, and a final query closure solve.",
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
            "The objective masks the four fixed sink chunks and excludes exact recent64 tokens. Saved deployment factors contain only the direct chunk encoder and per-query-head factor; the old token R16 is initialization-only.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    configure()
    assert 0 <= args.layer < 32
    assert args.fit_count == args.heldout_count == 1
    assert args.sweeps == 4
    assert args.cg_iterations > 0
    identity = read_json(args.root / "manifests/v128.json")
    assert identity["value_mode"] == "dense original V and Wo"
    assert identity["layer_ranks"] == [128] * 32
    factor_path = args.root / "ours_b16r16" / f"layer_{args.layer:03d}.safetensors"
    factors, factor_record = verified(factor_path)
    assert factor_record["v_rank"] == 128
    factors = {name: value.cuda() for name, value in factors.items()}
    datasets, selections, split_windows = capture(args, identity, factors)
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
                "stage": "fit",
                "feature": feature,
                "initial": diagnostics["initial"],
                "final": diagnostics["final"],
            },
            flush=True,
        )
    protocol = {
        "format": FORMAT,
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
        "als_sweeps": args.sweeps,
        "final_query_closure": True,
        "relative_damping": args.relative_damping,
        "cg_tolerance": args.cg_tolerance,
        "cg_iterations": args.cg_iterations,
        "initialization": "old token B16R16 factors adapted as initialization only",
        "deployment": "chunk encoder E_g shared by four query heads; U_h query-head-specific; no token R16 state",
        "factor_source": str(factor_path),
        "factor_source_sha256": sha256(factor_path),
        "identity_sha256": sha256(args.root / "manifests/v128.json"),
        "windows_sha256": sha256(args.root / "calibration/windows.safetensors"),
        "source_sha256": {
            name: sha256(Path(name))
            for name in (
                "basisserve/core/chunk_fisher_landmark.py",
                "evaluation/fit_chunk8_fisher_landmark.py",
            )
        },
    }
    result = {
        "status": "complete",
        "scope": "single-layer end-to-end smoke; not a formal 32-layer fit",
        "protocol": protocol,
        "audits": {
            "matrix_free_fisher_matches_dense": fisher_audit(),
            "mean_and_flat_old_r16_initial_scores_max_abs_difference": equivalence,
            "sink_chunks_excluded_from_fisher": True,
            "recent64_excluded_from_fisher": True,
            "base_is_frozen": True,
            "token_r16_is_not_saved_for_deployment": True,
            "group_encoder_is_shared": True,
            "query_factor_is_head_specific": True,
        },
        "features": feature_results,
        "artifacts": artifacts,
        "command": shlex.join(sys.argv),
        "python": sys.executable,
    }
    write_json(args.output / "result.json", result)
    text = summary(result)
    summary_path = args.output / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
