"""Capture exact compact Mean-128 Chunk8 Fisher statistics for many layers."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.chunk_fisher_landmark import (
    PackedChunkFisherDataset,
    PackedChunkFisherWindow,
    compact_packed_chunk_fisher_dataset,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    pack_symmetric_fisher_grams,
)
from evaluation.fit_chunk8_fisher_landmark_formal import (
    CHUNK_SIZE,
    packed_windows_from_attention,
    selected_positions,
)
from evaluation.fit_k_routing_streaming import verified
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json


FORMAT = "basisserve.chunk8_mean_fisher.compact.v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--layers", type=int, nargs="+", required=True)
    result.add_argument("--fit-count", type=int, default=64)
    result.add_argument("--heldout-count", type=int, default=16)
    result.add_argument("--fit-queries", type=int, default=64)
    result.add_argument("--heldout-queries", type=int, default=32)
    result.add_argument("--smoke", action="store_true")
    return result


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    configure()
    layers = sorted(args.layers)
    assert layers and layers == sorted(set(layers))
    assert all(0 <= layer < 32 for layer in layers)
    if args.smoke:
        assert args.fit_count == args.heldout_count == 1
        assert 1 <= args.fit_queries <= 64 and 1 <= args.heldout_queries <= 32
    else:
        assert args.fit_count == 64 and args.heldout_count == 16
        assert args.fit_queries == 64 and args.heldout_queries == 32

    identity_path = args.root / "manifests/v128.json"
    identity = read_json(identity_path)
    assert identity["value_mode"] == "dense original V and Wo"
    assert identity["layer_ranks"] == [128] * 32
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

    windows_path = args.root / "calibration/windows.safetensors"
    calibration = load_file(str(windows_path))["input_ids"]
    assert calibration.shape == (80, 65_536)
    mapping = torch.arange(32, device="cuda", dtype=torch.long) // 4
    state = {}
    for layer in layers:
        moments_path = args.root / "moments" / f"layer_{layer:03d}.json"
        moments = read_json(moments_path)
        factor_path = args.root / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        factors, factor_record = verified(factor_path)
        assert moments["status"] == "complete" and factor_record["v_rank"] == 128
        original = {
            "fit": moments["selections"]["fit"]["selected_positions"],
            "heldout": moments["selections"]["heldout"]["selected_positions"],
        }
        positions = {
            "fit": selected_positions(original["fit"], args.fit_queries, args.smoke),
            "heldout": selected_positions(
                original["heldout"], args.heldout_queries, args.smoke
            ),
        }
        selected_original = {
            split: values[-len(positions[split]) :] if args.smoke else values[: len(positions[split])]
            for split, values in original.items()
        }
        state[layer] = {
            "factors": {name: value.cuda() for name, value in factors.items()},
            "factor_path": factor_path,
            "moments_path": moments_path,
            "positions": positions,
            "position_replacements": {
                split: {
                    "removed": sorted(set(selected_original[split]) - set(positions[split])),
                    "added": sorted(set(positions[split]) - set(selected_original[split])),
                }
                for split in ("fit", "heldout")
            },
            "artifacts": [],
        }

    active = {"split": "", "window": -1, "captured": set()}
    handles = []
    for layer in layers:
        attention = model.model.layers[layer].self_attn

        def hook(attention, positional, kwargs, layer=layer):
            hidden = kwargs["hidden_states"]
            length = int(hidden.shape[1])
            query = attention.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
            key = attention.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            value = attention.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            factors = state[layer]["factors"]
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
            window = packed_windows_from_attention(
                query,
                key,
                sidecar[..., :128],
                positions=state[layer]["positions"][active["split"]],
                feature_types=("mean",),
                output_device=query.device,
            )["mean"]
            float_window = PackedChunkFisherWindow(
                positions=window.positions,
                queries=window.queries.float(),
                features_by_group=window.features_by_group.float(),
                base_logits=window.base_logits.float(),
                teacher_logits=window.teacher_logits.float(),
                candidate_counts=window.candidate_counts,
            )
            dataset = PackedChunkFisherDataset((float_window,), mapping, 128**-0.5)
            statistics = compact_packed_chunk_fisher_dataset(dataset)
            path = (
                args.output
                / f"layer_{layer:03d}"
                / f"window_{active['window']:03d}.safetensors"
            )
            save_tensors(
                path,
                {
                    "queries_by_head": statistics.queries_by_head.to(torch.bfloat16).cpu(),
                    "fisher_grams_packed_by_head": pack_symmetric_fisher_grams(
                        statistics.fisher_grams_by_head
                    ).cpu(),
                    "target_cross_by_head": statistics.target_cross_by_head.cpu(),
                    "target_fisher_energy": torch.tensor(
                        statistics.target_fisher_energy,
                        dtype=torch.float64,
                    ),
                },
            )
            state[layer]["artifacts"].append(
                {
                    "window": active["window"],
                    "split": active["split"],
                    "file": str(path.relative_to(args.output)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
            active["captured"].add(layer)

        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))

    split_windows = {
        "fit": list(range(args.fit_count)),
        "heldout": list(range(64, 64 + args.heldout_count)),
    }
    start = time.perf_counter()
    for split, indices in split_windows.items():
        for index in indices:
            active.update(split=split, window=index, captured=set())
            output = model.model(calibration[index : index + 1].long().cuda(), use_cache=False)
            assert active["captured"] == set(layers)
            assert torch.isfinite(output.last_hidden_state).all()
            del output
            print(
                {
                    "stage": "compact Mean capture",
                    "split": split,
                    "window": index,
                    "layers": layers,
                    "elapsed_seconds": time.perf_counter() - start,
                },
                flush=True,
            )
    for handle in handles:
        handle.remove()
    del model, calibration
    gc.collect()
    torch.cuda.empty_cache()

    for layer in layers:
        current = state[layer]
        assert [item["window"] for item in current["artifacts"]] == (
            split_windows["fit"] + split_windows["heldout"]
        )
        record = {
            "status": "complete",
            "layer": layer,
            "protocol": {
                "format": FORMAT,
                "scope": "smoke" if args.smoke else "formal",
                "root": str(args.root),
                "model": identity["model"],
                "model_variant": "Llama-3.1-8B-Instruct",
                "sequence_length": 65_536,
                "chunk_size": CHUNK_SIZE,
                "feature": "mean",
                "feature_dim": 128,
                "fit_count": args.fit_count,
                "heldout_count": args.heldout_count,
                "fit_queries": args.fit_queries,
                "heldout_queries": args.heldout_queries,
                "statistics": "exact per-query X.T J X, X.T J target, and target Fisher energy",
                "gram_storage": "FP32 symmetric upper triangle",
                "query_storage": "BF16 source values expanded to FP32 on load",
                "invalid_query_policy": "replace a selected position with no routed historical Chunk8 candidate by the nearest later unused 64-token query-grid position",
                "source_sha256": {
                    name: sha256(Path(name))
                    for name in (
                        "basisserve/core/chunk_fisher_landmark.py",
                        "evaluation/capture_chunk8_mean_fisher_statistics.py",
                        "evaluation/fit_chunk8_fisher_landmark_formal.py",
                    )
                },
            },
            "query_positions": current["positions"],
            "query_position_replacements": current["position_replacements"],
            "split_windows": split_windows,
            "moments_source": str(current["moments_path"]),
            "moments_source_sha256": sha256(current["moments_path"]),
            "factor_source": str(current["factor_path"]),
            "factor_source_sha256": sha256(current["factor_path"]),
            "identity_source": str(identity_path),
            "identity_sha256": sha256(identity_path),
            "windows_source": str(windows_path),
            "windows_sha256": sha256(windows_path),
            "windows": current["artifacts"],
            "total_bytes": sum(item["bytes"] for item in current["artifacts"]),
            "capture_wall_seconds": time.perf_counter() - start,
        }
        write_json(args.output / f"layer_{layer:03d}.json", record)
        print(
            {
                "stage": "compact Mean layer complete",
                "layer": layer,
                "windows": len(current["artifacts"]),
                "bytes": record["total_bytes"],
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
