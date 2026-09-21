"""Controlled Dense-V rank-16 pre-RoPE versus direct post-RoPE Base ablation."""

import argparse
import csv
import math
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from evaluation.fit_k_routing_streaming import encoder_for, layer_file, save_record, verified
from evaluation.llama_sink_recent_routing import page_support
from evaluation.streaming_k_statistics import base_from_moments, base_mse
from evaluation.v96kl_common import configure, read_json, sha256, write_json


METHODS = ("pre_rope_base", "post_rope_base")
MOMENT_FIELDS = ("count", "sum_v", "sum_k", "vv", "vk", "kk")
POSITION_BUCKETS = ((0, 8192), (8192, 32768), (32768, 65536))
ROUTING_METRICS = (
    "routed_page_recall",
    "page_kl",
    "attention_mass_recall",
    "nonsink_mass_recall",
)


def target_moments(payload, split, method):
    infix = "" if method == "pre_rope_base" else "post_"
    return {name: payload[f"{split}_{infix}{name}"] for name in MOMENT_FIELDS}


def centered_energy(moments):
    count = int(moments["count"])
    centered = moments["kk"] - torch.einsum(
        "gd,ge->gde", moments["sum_k"], moments["sum_k"]
    ) / count
    energy = float(centered.diagonal(dim1=-2, dim2=-1).sum())
    assert energy > 0
    return energy


def checkpoint_dir(section4, method, smoke):
    root = section4 / "pre_vs_post_rope" / "checkpoint"
    return root / "smoke" / method if smoke else root / method


def build_layer(args, layer, smoke):
    identity = read_json(args.root / "manifests/v128.json")
    assert identity["layer_ranks"] == [128] * 32
    payload, moment_meta = verified(layer_file(args.root, "moments", layer))
    assert moment_meta["protocol"]["smoke"] == smoke
    encoder = encoder_for(identity, layer)
    eye = torch.eye(128, dtype=encoder.dtype).expand(8, -1, -1)
    assert torch.equal(encoder, eye)
    for method in METHODS:
        splits = {
            split: target_moments(payload, split, method)
            for split in ("fit", "heldout")
        }
        fitted = base_from_moments(splits["fit"], encoder, rank=16)[16]
        metrics = {}
        full = base_from_moments(splits["fit"], encoder, rank=128)[128]
        for split, moments in splits.items():
            raw = base_mse(moments, encoder, fitted)
            full_raw = base_mse(moments, encoder, full)
            centered = centered_energy(moments)
            captured = centered - raw["squared_error"]
            predictable = centered - full_raw["squared_error"]
            assert predictable > 0
            metrics[split] = {
                **raw,
                "centered_key_energy": centered,
                "centered_relative_mse": raw["squared_error"] / centered,
                "centered_explained_fraction": captured / centered,
                "full_rank_predictable_energy": predictable,
                "predictable_energy_captured_fraction": captured / predictable,
            }
        tensors = {
            f"base_{name}_b16": torch.stack([getattr(base, name) for base in fitted]).float()
            for name in ("left", "right", "bias")
        }
        tensors["residual_encoder_b16_r0"] = torch.empty(8, 128, 0)
        tensors["residual_query_b16_r0"] = torch.empty(32, 128, 0)
        path = checkpoint_dir(args.section4, method, smoke) / f"layer_{layer:03d}.safetensors"
        save_record(
            path,
            tensors,
            {
                "layer": layer,
                "method": method,
                "base_rank": 16,
                "residual_rank": 0,
                "identity_sha256": sha256(args.root / "manifests/v128.json"),
                "moments_sha256": moment_meta["sha256"],
                "moments_protocol": moment_meta["protocol"],
                "target": "pre-RoPE K followed by native RoPE" if method == "pre_rope_base" else "direct post-RoPE K without position features",
                "metrics": metrics,
                "value_basis_audit": "Dense V128 identity; no gauge rotation required",
            },
        )
        print({"stage": "build", "method": method, "layer": layer, "heldout": metrics["heldout"]}, flush=True)


def predict(value, factors):
    left = factors["base_left_b16"].to(device=value.device, dtype=value.dtype)
    right = factors["base_right_b16"].to(device=value.device, dtype=value.dtype)
    bias = factors["base_bias_b16"].to(device=value.device, dtype=value.dtype)
    return torch.einsum("bhtv,hvr,hrd->bhtd", value, left, right) + bias[None, :, None, :]


def rotate(values, cos, sin):
    half = values.shape[-1] // 2
    rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return values * cos.to(values.dtype).unsqueeze(1) + rotated * sin.to(values.dtype).unsqueeze(1)


def bucket_name(position):
    for start, end in POSITION_BUCKETS:
        if start <= position < end:
            return f"{start}-{end}"
    assert False


def new_state():
    return {
        method: {
            "all": {"key_squared_error": 0.0, "key_energy": 0.0, "key_tokens": 0, **{f"{metric}_sum": 0.0 for metric in ROUTING_METRICS}, **{f"{metric}_count": 0 for metric in ROUTING_METRICS}},
            **{
                f"{start}-{end}": {"key_squared_error": 0.0, "key_energy": 0.0, "key_tokens": 0, **{f"{metric}_sum": 0.0 for metric in ROUTING_METRICS}, **{f"{metric}_count": 0 for metric in ROUTING_METRICS}}
                for start, end in POSITION_BUCKETS
            },
        }
        for method in METHODS
    }


def selected_pages(ids, valid, length):
    historical = length - 64
    page_count = math.ceil(length / 32)
    selected = torch.zeros(ids.shape[0], ids.shape[1], page_count, device=ids.device, dtype=torch.int32)
    valid = valid & (ids >= 32) & (ids < historical)
    selected.scatter_add_(-1, (ids // 32).clamp_max(page_count - 1), valid.int())
    return selected > 0


def page_kl(teacher_scores, proxy_scores, length):
    historical = length - 64
    assert historical > 32

    def log_pages(scores):
        scores = scores[..., :historical].reshape(scores.shape[0], -1, historical).float()
        pages = math.ceil(historical / 32)
        padding = pages * 32 - historical
        if padding:
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
        masses = torch.logsumexp(scores.reshape(*scores.shape[:-1], pages, 32), dim=-1)
        masses[..., 0] = -torch.inf
        return torch.log_softmax(masses, dim=-1)

    teacher_log = log_pages(teacher_scores)
    proxy_log = log_pages(proxy_scores)
    teacher = teacher_log.exp()
    value = torch.where(teacher > 0, teacher * (teacher_log - proxy_log), 0).sum(-1)
    assert torch.isfinite(value).all() and value.min() >= -1e-6
    return float(value.mean())


def update_key_metrics(state, exact, predictions):
    length = exact.shape[2]
    for method, prediction in predictions.items():
        for start, end in POSITION_BUCKETS:
            stop = min(end, length)
            if start >= stop:
                continue
            target = exact[:, :, start:stop].float()
            estimate = prediction[:, :, start:stop].float()
            squared_error = float((target - estimate).double().square().sum())
            energy = float(target.double().square().sum())
            token_count = target.shape[0] * target.shape[1] * target.shape[2]
            for name in ("all", f"{start}-{end}"):
                state[method][name]["key_squared_error"] += squared_error
                state[method][name]["key_energy"] += energy
                state[method][name]["key_tokens"] += token_count


def update_routing_metrics(state, exact_key, predictions, queries, positions):
    batch, kv_heads, _, dim = exact_key.shape
    head_groups = queries.shape[1] // kv_heads
    for offset, position in enumerate(positions):
        length = int(position) + 1
        assert length > 32
        query = queries[:, :, offset].float()
        exact_scores = (
            query.reshape(batch, kv_heads, head_groups, dim)
            @ exact_key[:, :, :length].float().transpose(-1, -2)
        ) * dim**-0.5
        probabilities = exact_scores.softmax(-1)
        nonsink_scores = exact_scores.clone()
        nonsink_scores[..., :32] = -torch.inf
        nonsink_probabilities = nonsink_scores.softmax(-1)
        exact_ids, exact_valid = page_support(exact_scores, budget=2048)
        reference = selected_pages(exact_ids, exact_valid, length)
        name = bucket_name(int(position))
        for method, prediction in predictions.items():
            proxy_scores = (
                query.reshape(batch, kv_heads, head_groups, dim)
                @ prediction[:, :, :length].float().transpose(-1, -2)
            ) * dim**-0.5
            ids, valid = page_support(proxy_scores, budget=2048)
            selected = selected_pages(ids, valid, length)
            denominator = reference.sum(-1)
            eligible = denominator > 0
            recall = ((selected & reference).sum(-1) / denominator.clamp_min(1))[eligible]
            gathered = ids.clamp_max(position)[:, :, None].expand(batch, kv_heads, head_groups, -1)
            mask = valid[:, :, None].expand_as(gathered)
            values = {
                "attention_mass_recall": float(probabilities.gather(-1, gathered).masked_fill(~mask, 0).sum(-1).mean()),
                "nonsink_mass_recall": float(nonsink_probabilities.gather(-1, gathered).masked_fill(~mask, 0).sum(-1).mean()),
            }
            if eligible.any():
                values["routed_page_recall"] = float(recall.mean())
            if length - 64 > 32:
                values["page_kl"] = page_kl(exact_scores, proxy_scores, length)
            for destination in ("all", name):
                for metric, value in values.items():
                    state[method][destination][f"{metric}_sum"] += value
                    state[method][destination][f"{metric}_count"] += 1


def merge_state(destination, source):
    for method in METHODS:
        for bucket in source[method]:
            for key, value in source[method][bucket].items():
                destination[method][bucket][key] += value


@torch.inference_mode()
def evaluate(args, smoke):
    identity_path = args.root / "manifests/v128.json"
    identity = read_json(identity_path)
    windows_path = args.root / "calibration/windows.safetensors"
    windows_meta = read_json(windows_path.with_name("manifest.json"))
    assert windows_meta["sha256"] == sha256(windows_path)
    layers = [0] if smoke else list(range(args.shard_index * 8, (args.shard_index + 1) * 8))
    factors = {layer: {} for layer in layers}
    positions = {}
    artifact_hashes = {}
    for layer in layers:
        moment_meta = read_json(layer_file(args.root, "moments", layer).with_suffix(".json"))
        assert moment_meta["protocol"]["smoke"] == smoke
        positions[layer] = moment_meta["selections"]["heldout"]["selected_positions"]
        assert len(positions[layer]) == 32
        artifact_hashes[str(layer)] = {}
        for method in METHODS:
            path = checkpoint_dir(args.section4, method, smoke) / f"layer_{layer:03d}.safetensors"
            payload, meta = verified(path)
            assert meta["method"] == method and meta["base_rank"] == 16 and meta["residual_rank"] == 0
            assert payload["base_left_b16"].shape == (8, 128, 16)
            assert payload["residual_encoder_b16_r0"].numel() == 0
            factors[layer][method] = payload
            artifact_hashes[str(layer)][method] = meta["sha256"]
    model = AutoModelForCausalLM.from_pretrained(
        identity["model"], dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    windows = load_file(str(windows_path))["input_ids"]
    sequence_length = 4096 if smoke else 65536
    selected_windows = [64] if smoke else list(range(64, 80))
    states = {layer: new_state() for layer in layers}
    rope_audits = {layer: {"max_abs_error": 0.0, "dtype": None, "rotary_dimension": 128} for layer in layers}
    active = {}
    handles = []
    for layer in layers:
        def hook(module, positional, kwargs, layer=layer):
            hidden = kwargs["hidden_states"]
            length = hidden.shape[1]
            query = module.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
            pre_key = module.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            value = module.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            post_query, exact_post = apply_rotary_pos_emb(query, pre_key, cos, sin)
            manual = rotate(pre_key, cos, sin)
            rope_error = float((manual - exact_post).abs().max())
            rope_audits[layer]["max_abs_error"] = max(rope_audits[layer]["max_abs_error"], rope_error)
            rope_audits[layer]["dtype"] = str(exact_post.dtype)
            pre_prediction = rotate(predict(value, factors[layer]["pre_rope_base"]), cos, sin)
            post_prediction = predict(value, factors[layer]["post_rope_base"])
            predictions = {"pre_rope_base": pre_prediction, "post_rope_base": post_prediction}
            update_key_metrics(states[layer], exact_post, predictions)
            update_routing_metrics(states[layer], exact_post, predictions, post_query[:, :, positions[layer]], positions[layer])
            print({"stage": "evaluate", "layer": layer, "window": active["window"], "smoke": smoke}, flush=True)

        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    for window in selected_windows:
        active["window"] = window
        result = model.model(windows[window:window + 1, :sequence_length].long().cuda(), use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all()
        del result
    for handle in handles:
        handle.remove()
    protocol = {
        "format": "basisserve.section4.pre_vs_post_rope_base.v1",
        "model": identity["model"],
        "model_config_sha256": identity["model_config_sha256"],
        "identity_sha256": sha256(identity_path),
        "windows_sha256": windows_meta["sha256"],
        "window_ids": selected_windows,
        "sequence_length": sequence_length,
        "base_rank": 16,
        "residual_rank": 0,
        "page_size": 32,
        "budget": 2048,
        "sink_tokens_inside_budget": 32,
        "recent_tokens_inside_budget": 64,
        "final_attention": "proxy selects support; metrics recomputed with exact post-RoPE QK",
        "post_rope_features": "Dense V128 only; no position, RoPE angle, or token index",
        "artifacts": artifact_hashes,
        "smoke": smoke,
        "source_sha256": {str(Path(__file__)): sha256(Path(__file__))},
    }
    output = args.section4 / "pre_vs_post_rope" / "diagnostics"
    path = output / ("smoke.json" if smoke else f"shard_{args.shard_index:02d}.json")
    write_json(path, {"status": "complete", "protocol": protocol, "states": states, "rope_audits": rope_audits, "command": shlex.join(sys.argv), "python": sys.executable})
    print({"output": str(path), "rope_audits": rope_audits}, flush=True)


def finalized(raw):
    result = {
        "post_rope_rel_mse": raw["key_squared_error"] / raw["key_energy"],
        "key_tokens": raw["key_tokens"],
    }
    for metric in ROUTING_METRICS:
        count = raw[f"{metric}_count"]
        result[metric] = raw[f"{metric}_sum"] / count if count else None
        result[f"{metric}_samples"] = count
    return result


def summarize(args):
    diagnostic = args.section4 / "pre_vs_post_rope" / "diagnostics"
    reports = [read_json(diagnostic / f"shard_{index:02d}.json") for index in range(4)]
    state = new_state()
    audits = {}
    for report in reports:
        assert report["status"] == "complete" and not report["protocol"]["smoke"]
        for layer, layer_state in report["states"].items():
            merge_state(state, layer_state)
            audits[layer] = report["rope_audits"][layer]
    assert set(audits) == {str(layer) for layer in range(32)}
    assert max(row["max_abs_error"] for row in audits.values()) == 0
    aggregate_rows = []
    position_rows = []
    layer_rows = []
    for method in METHODS:
        checkpoint = checkpoint_dir(args.section4, method, False)
        captured = predictable = 0.0
        for layer in range(32):
            meta = read_json((checkpoint / f"layer_{layer:03d}.safetensors").with_suffix(".json"))
            metric = meta["metrics"]["fit"]
            captured += metric["centered_key_energy"] - metric["squared_error"]
            predictable += metric["full_rank_predictable_energy"]
        overall = finalized(state[method]["all"])
        aggregate_rows.append({"method": method, "base_rank": 16, "predictable_energy": captured / predictable, **overall})
        for start, end in POSITION_BUCKETS:
            values = finalized(state[method][f"{start}-{end}"])
            position_rows.append({"method": method, "position_start": start, "position_end": end, "num_samples": values["routed_page_recall_samples"], **values})
        for layer in range(32):
            report = reports[layer // 8]
            values = finalized(report["states"][str(layer)][method]["all"])
            layer_rows.append({"method": method, "layer": layer, **values})
    repo_output = args.repo_output
    repo_output.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (repo_output / "pre_vs_post_rope_base.csv", aggregate_rows),
        (repo_output / "pre_vs_post_rope_base_position.csv", position_rows),
        (repo_output / "pre_vs_post_rope_base_layers.csv", layer_rows),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(repo_output / "pre_vs_post_rope_manifest.json", {"status": "complete", "protocol": reports[0]["protocol"], "rope_audits": audits, "aggregate": aggregate_rows, "position": position_rows})
    print(aggregate_rows, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("build-smoke", "build", "smoke", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument("--repo-output", type=Path, default=Path("results/section4_ablation/llama31_8b_instruct"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    configure()
    assert 0 <= args.layer < 32 and 0 <= args.shard_index < 4
    if args.stage == "build-smoke":
        build_layer(args, args.layer, True)
    elif args.stage == "build":
        build_layer(args, args.layer, False)
    elif args.stage == "smoke":
        evaluate(args, True)
    elif args.stage == "evaluate":
        evaluate(args, False)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
