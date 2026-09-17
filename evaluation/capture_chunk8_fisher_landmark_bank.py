"""Capture reusable all-layer direct Chunk8 Fisher fitting inputs."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.fit_chunk8_fisher_landmark_formal import (
    CHUNK_SIZE,
    packed_windows_from_attention,
    selected_positions,
)
from evaluation.fit_k_routing_streaming import verified
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json


FORMAT = "basisserve.chunk8_fisher_landmark.capture_bank.v1"


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

    calibration_path = args.root / "calibration/windows.safetensors"
    calibration = load_file(str(calibration_path))["input_ids"]
    assert calibration.shape == (80, 65_536)
    layer_state = {}
    for layer in layers:
        moments_path = args.root / "moments" / f"layer_{layer:03d}.json"
        moments = read_json(moments_path)
        assert moments["status"] == "complete"
        assert moments["protocol"]["windows_sha256"] == sha256(calibration_path)
        factor_path = args.root / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        factors, factor_record = verified(factor_path)
        assert factor_record["v_rank"] == 128
        original_positions = {
            "fit": moments["selections"]["fit"]["selected_positions"][: args.fit_queries],
            "heldout": moments["selections"]["heldout"]["selected_positions"][: args.heldout_queries],
        }
        positions = {
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
        if args.smoke:
            original_positions = {
                "fit": moments["selections"]["fit"]["selected_positions"][-args.fit_queries :],
                "heldout": moments["selections"]["heldout"]["selected_positions"][-args.heldout_queries :],
            }
        layer_state[layer] = {
            "factors": {name: value.cuda() for name, value in factors.items()},
            "factor_path": factor_path,
            "moments_path": moments_path,
            "positions": positions,
            "position_replacements": {
                split: {
                    "removed": sorted(set(original_positions[split]) - set(positions[split])),
                    "added": sorted(set(positions[split]) - set(original_positions[split])),
                }
                for split in ("fit", "heldout")
            },
            "windows": [],
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
            factors = layer_state[layer]["factors"]
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
                positions=layer_state[layer]["positions"][active["split"]],
            )["flat"]
            path = (
                args.output
                / f"layer_{layer:03d}"
                / f"window_{active['window']:03d}.safetensors"
            )
            save_tensors(
                path,
                {
                    "positions": window.positions.cpu().contiguous(),
                    "queries": window.queries.cpu().contiguous(),
                    "flat_features_by_group": window.features_by_group.to(
                        dtype=torch.bfloat16
                    ).cpu().contiguous(),
                    "base_logits": window.base_logits.cpu().contiguous(),
                    "teacher_logits": window.teacher_logits.cpu().contiguous(),
                    "candidate_counts": window.candidate_counts.cpu().contiguous(),
                },
            )
            layer_state[layer]["windows"].append(
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
                    "stage": "capture bank",
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
        state = layer_state[layer]
        assert [item["window"] for item in state["windows"]] == (
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
                "fit_count": args.fit_count,
                "heldout_count": args.heldout_count,
                "fit_queries": args.fit_queries,
                "heldout_queries": args.heldout_queries,
                "storage": "BF16 exact source Flat features; BF16 queries; FP32 base and teacher Chunk8 logits",
                "invalid_query_policy": "replace a selected position with no routed historical Chunk8 candidate by the nearest later unused 64-token query-grid position",
                "source_sha256": {
                    name: sha256(Path(name))
                    for name in (
                        "basisserve/core/chunk_fisher_landmark.py",
                        "evaluation/capture_chunk8_fisher_landmark_bank.py",
                        "evaluation/fit_chunk8_fisher_landmark_formal.py",
                    )
                },
            },
            "query_positions": state["positions"],
            "query_position_replacements": state["position_replacements"],
            "split_windows": split_windows,
            "moments_source": str(state["moments_path"]),
            "moments_source_sha256": sha256(state["moments_path"]),
            "factor_source": str(state["factor_path"]),
            "factor_source_sha256": sha256(state["factor_path"]),
            "identity_source": str(identity_path),
            "identity_sha256": sha256(identity_path),
            "windows_source": str(calibration_path),
            "windows_sha256": sha256(calibration_path),
            "windows": state["windows"],
            "total_bytes": sum(item["bytes"] for item in state["windows"]),
            "capture_wall_seconds": time.perf_counter() - start,
        }
        write_json(args.output / f"layer_{layer:03d}.json", record)
        print(
            {
                "stage": "layer bank complete",
                "layer": layer,
                "windows": len(state["windows"]),
                "bytes": record["total_bytes"],
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
