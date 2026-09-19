"""Capture clean, unweighted causal QK-score statistics for Section 4."""

import argparse
import gc
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.gqa_joint_routing_payload_s80_fisher import pack_symmetric_fisher_grams
from evaluation.fit_k_routing_streaming import layer_file, save_record, verified
from evaluation.v96kl_common import configure, read_json, sha256


SEQUENCE_LENGTH = 65536
FIT_IDS = tuple(range(64))
HELDOUT_IDS = tuple(range(64, 80))
FIT_QUERIES = 64
HELDOUT_QUERIES = 32
EXCLUDED_PREFIX_TOKENS = 32


def length_stratified_positions(length, count):
    """Return the integer midpoint of each equal-width length stratum."""
    return [((2 * index + 1) * length) // (2 * count) for index in range(count)]


def capture_protocol(root, identity, windows_manifest, *, smoke):
    fit_positions = length_stratified_positions(SEQUENCE_LENGTH, FIT_QUERIES)
    heldout_positions = length_stratified_positions(SEQUENCE_LENGTH, HELDOUT_QUERIES)
    return {
        "format": "basisserve.section4.score_only_statistics.v1",
        "identity_sha256": sha256(root / "manifests/v128.json"),
        "model_config_sha256": identity["model_config_sha256"],
        "windows_sha256": windows_manifest["sha256"],
        "sequence_length": SEQUENCE_LENGTH,
        "fit_ids": [0] if smoke else list(FIT_IDS),
        "heldout_ids": [64] if smoke else list(HELDOUT_IDS),
        "fit_queries": FIT_QUERIES,
        "heldout_queries": HELDOUT_QUERIES,
        "fit_positions": fit_positions,
        "heldout_positions": heldout_positions,
        "position_sampling": "fixed midpoint of equal-width length strata; shared by every layer and window",
        "base_rank": 16,
        "excluded_prefix_tokens": EXCLUDED_PREFIX_TOKENS,
        "value_mode": "dense original V and Wo",
        "residual": "exact post-RoPE K minus affine Base16 prediction after native token-position RoPE",
        "objective_statistics": "unweighted causal residual QK score squared error",
        "fisher_artifacts_read": [],
        "smoke": smoke,
        "source_sha256": {
            "evaluation/capture_llama_section4_score_only_statistics.py": sha256(Path(__file__)),
            "basisserve/core/c1_v_conditional_k_router.py": sha256(
                Path("basisserve/core/c1_v_conditional_k_router.py")
            ),
        },
    }


def prefix_grams(residual, positions):
    groups, _, dim = residual.shape
    running = torch.zeros(groups, dim, dim, device=residual.device, dtype=torch.float32)
    by_position = {}
    cursor = EXCLUDED_PREFIX_TOKENS
    for position in sorted(positions):
        stop = int(position) + 1
        assert stop > cursor
        rows = residual[:, cursor:stop].float()
        running.add_(torch.einsum("gtd,gte->gde", rows, rows))
        by_position[int(position)] = running.clone()
        cursor = stop
    return torch.stack([by_position[int(position)] for position in positions], dim=1)


@torch.inference_mode()
def run(args, *, smoke):
    root = args.root
    output = args.output / "smoke" if smoke else args.output
    identity = read_json(root / "manifests/v128.json")
    windows_manifest = read_json(root / "calibration/manifest.json")
    assert windows_manifest["sha256"] == sha256(root / "calibration/windows.safetensors")
    assert identity["status"] == "complete" and identity["layer_ranks"] == [128] * 32
    protocol = capture_protocol(root, identity, windows_manifest, smoke=smoke)
    layers = [0] if smoke else list(range(args.shard_index * 8, (args.shard_index + 1) * 8))
    bases = {}
    base_hashes = {}
    for layer in layers:
        path = layer_file(root, "base", layer)
        payload, base_meta = verified(path)
        assert base_meta["protocol"]["base_rank"] == 16
        assert base_meta["identity_sha256"] == protocol["identity_sha256"]
        bases[layer] = {name: payload[name].cuda() for name in ("left", "right", "bias")}
        base_hashes[layer] = base_meta["sha256"]

    model = AutoModelForCausalLM.from_pretrained(
        identity["model"], dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    windows = load_file(str(root / "calibration/windows.safetensors"))["input_ids"]
    assert tuple(windows.shape) == (80, SEQUENCE_LENGTH)
    active = {}
    handles = []
    for layer in layers:
        base = bases[layer]

        def hook(module, positional, kwargs, layer=layer, base=base):
            split = active["split"]
            index = active["index"]
            hidden = kwargs["hidden_states"]
            length = hidden.shape[1]
            query = module.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
            key = module.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            value = module.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            predicted = build_conditional_routing_sidecar(
                value,
                key,
                base_left=base["left"],
                base_right=base["right"],
                base_bias=base["bias"],
                residual_encoder=torch.empty(8, 128, 0, device=key.device, dtype=key.dtype),
                cos=cos,
                sin=sin,
            )
            residual = key[0] - predicted[0]
            selected = protocol[f"{split}_positions"]
            grams_by_group = prefix_grams(residual, selected)
            mapping = torch.arange(32, device=key.device) // 4
            grams = grams_by_group.index_select(0, mapping)
            selected_queries = query[0, :, selected].float()
            scale = 128 ** -0.5
            teacher_energy = 0.5 * scale**2 * float(
                torch.einsum("hqd,hqde,hqe->", selected_queries, grams, selected_queries)
            )
            path = output / "score" / f"l{layer:03d}" / f"w{index:03d}.safetensors"
            save_record(
                path,
                {
                    "queries": selected_queries.cpu(),
                    "packed": pack_symmetric_fisher_grams(grams).cpu(),
                    "teacher_energy": torch.tensor(teacher_energy, dtype=torch.float64),
                },
                {
                    "protocol": protocol,
                    "layer": layer,
                    "window_id": index,
                    "split": split,
                    "base_sha256": base_hashes[layer],
                    "storage": "symmetric upper triangle FP32",
                    "command": shlex.join(sys.argv),
                    "python": sys.executable,
                },
            )
            print({"layer": layer, "window": index, "split": split, "score_energy": teacher_energy}, flush=True)

        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))

    selected_windows = (("fit", [0]), ("heldout", [64])) if smoke else (
        ("fit", FIT_IDS),
        ("heldout", HELDOUT_IDS),
    )
    for split, indices in selected_windows:
        for index in indices:
            active.update(split=split, index=index)
            result = model.model(windows[index:index + 1].long().cuda(), use_cache=False)
            assert torch.isfinite(result.last_hidden_state).all()
            del result
    for handle in handles:
        handle.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "capture"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    configure()
    assert 0 <= args.shard_index < 4
    run(args, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
