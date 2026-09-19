"""Unified held-out routing diagnostics for the Dense-V Section 4 ablations."""

import argparse
import math
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.fit_k_routing_streaming import verified
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import configure, read_json, sha256, write_json


BASE_RANKS = (4, 8, 16, 24, 32, 48, 64, 80, 96)
METRICS = (
    "attention_mass_recall",
    "nonsink_mass_recall",
    "page_recall",
    "routed_page_recall",
    "page_kl_exact_to_proxy",
)
ERROR_FIELDS = ("output_squared_error", "output_energy", "wo_squared_error", "wo_energy")


def bank_paths(root, section4, matrix, *, smoke):
    if matrix == "components":
        return {
            "r16_only": root / "b0r16/ours_b0r16",
            "r20_only": root / "b0r20/ours_b0r20",
            "b4r16": root / "b4r16/ours_b4r16",
            "b16_only": section4 / "base_rank_sweep" / ("smoke" if smoke else "") / "b16r0",
            "b16r16": root / "ours_b16r16",
            "r32_only": root / "b0r32" / ("smoke" if smoke else "") / "ours_b0r32",
        }
    if matrix == "base_rank_sweep":
        base = section4 / "base_rank_sweep" / ("smoke" if smoke else "")
        return {f"b{rank}_only": base / f"b{rank}r0" for rank in BASE_RANKS}
    if matrix == "objectives":
        objective_root = section4 / ("smoke" if smoke else "") / "objectives"
        return {
            "residual_mse": objective_root / "residual_mse/ours_b16r16",
            "score_mse": objective_root / "score_mse/ours_b16r16",
            "page_fisher": root / "ours_b16r16",
        }
    if matrix == "score_only":
        clean_root = section4 / "score_only/checkpoint" / ("smoke" if smoke else "")
        return {
            "score_only": clean_root / "ours_b16r16",
            "score_mse": section4 / "objectives/score_mse/ours_b16r16",
            "page_fisher": root / "ours_b16r16",
        }
    assert matrix == "qgram_score_only"
    qgram_root = section4 / "qgram_score_only/checkpoint" / ("smoke" if smoke else "")
    fixed_root = section4 / "score_only/checkpoint" / ("smoke" if smoke else "")
    return {
        "qgram_score_only": qgram_root / "ours_b16r16",
        "score_only": fixed_root / "ours_b16r16",
        "score_mse": section4 / "objectives/score_mse/ours_b16r16",
        "page_fisher": root / "ours_b16r16",
    }


def factor_ranks(payload):
    base_keys = [key for key in payload if key.startswith("base_left_b")]
    residual_keys = [key for key in payload if key.startswith("residual_encoder_b")]
    assert len(base_keys) == len(residual_keys) == 1
    base_rank = int(base_keys[0].split("_b")[-1])
    residual_tag = residual_keys[0].split("_b")[-1]
    prefix, residual = residual_tag.split("_r")
    assert int(prefix) == base_rank
    return base_rank, int(residual)


def proxy_scores(query, key, value, cos, sin, factors):
    batch, heads, dim = query.shape
    groups = key.shape[1]
    base_rank, residual_rank = factor_ranks(factors)
    encoder = factors[f"residual_encoder_b{base_rank}_r{residual_rank}"].to(key.device)
    query_factor = factors[f"residual_query_b{base_rank}_r{residual_rank}"].to(query.device)
    if base_rank == 0:
        sidecar = torch.einsum("bhtd,hdr->bhtr", key, encoder.to(key.dtype))
        code = torch.einsum("bhd,hdr->bhr", query, query_factor.float())
    else:
        sidecar = build_conditional_routing_sidecar(
            value,
            key,
            base_left=factors[f"base_left_b{base_rank}"],
            base_right=factors[f"base_right_b{base_rank}"],
            base_bias=factors[f"base_bias_b{base_rank}"],
            residual_encoder=encoder,
            cos=cos,
            sin=sin,
        )
        residual_code = torch.einsum("bhd,hdr->bhr", query, query_factor.float())
        code = torch.cat((query, residual_code), dim=-1)
    grouped = code.reshape(batch, groups, heads // groups, -1)
    return (grouped @ sidecar.float().transpose(-1, -2)) * dim**-0.5


def selected_pages(ids, valid, *, length, historical_only):
    page_count = math.ceil(length / 32)
    selected = torch.zeros(ids.shape[0], ids.shape[1], page_count, device=ids.device, dtype=torch.int32)
    if historical_only:
        historical = length - 64
        valid = valid & (ids >= 32) & (ids < historical)
    selected.scatter_add_(-1, (ids // 32).clamp_max(page_count - 1), valid.int())
    return selected > 0


def page_kl(teacher_scores, proxy, *, length):
    historical = length - 64
    assert historical > 32

    def log_distribution(scores):
        scores = scores[..., :historical].reshape(scores.shape[0], -1, historical).float()
        pages = math.ceil(historical / 32)
        padding = pages * 32 - historical
        if padding:
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
        log_mass = torch.logsumexp(scores.reshape(*scores.shape[:-1], pages, 32), dim=-1)
        log_mass[..., 0] = -torch.inf
        return torch.log_softmax(log_mass, dim=-1)

    teacher_log = log_distribution(teacher_scores)
    proxy_log = log_distribution(proxy)
    teacher = teacher_log.exp()
    terms = torch.where(teacher > 0, teacher * (teacher_log - proxy_log), torch.zeros_like(teacher))
    value = terms.sum(-1)
    assert torch.isfinite(value).all() and value.min() >= -1e-6
    return float(value.mean())


@torch.inference_mode()
def window_metrics(value, key, queries, positions, cos, sin, banks, output_weight):
    batch, kv_heads, length, dim = key.shape
    heads = queries.shape[1]
    head_groups = heads // kv_heads
    assert batch == 1 and queries.shape == (1, heads, len(positions), dim)
    values = value.float()
    output_weight = output_weight.float()
    totals = {
        name: {**{metric: 0.0 for metric in METRICS}, **{field: 0.0 for field in ERROR_FIELDS}}
        for name in [*banks, "exact_k"]
    }
    page_kl_counts = {name: 0 for name in totals}
    recall_counts = {
        name: {metric: 0 for metric in ("page_recall", "routed_page_recall")}
        for name in totals
    }
    for offset, position in enumerate(positions):
        length_now = int(position) + 1
        query = queries[:, :, offset].float()
        exact_scores = (
            query.reshape(batch, kv_heads, head_groups, dim)
            @ key[:, :, :length_now].float().transpose(-1, -2)
        ) * dim**-0.5
        probabilities = exact_scores.softmax(-1)
        nonsink_logits = exact_scores.clone()
        nonsink_logits[..., :32] = -torch.inf
        nonsink_probabilities = nonsink_logits.softmax(-1)
        dense_output = probabilities @ values[:, :, :length_now]
        dense_wo = dense_output.reshape(batch, -1) @ output_weight.T
        supports = {"exact_k": page_support(exact_scores, budget=2048)}
        scores = {"exact_k": exact_scores}
        for name, factors in banks.items():
            scores[name] = proxy_scores(
                query,
                key[:, :, :length_now],
                value[:, :, :length_now],
                cos[:, :length_now],
                sin[:, :length_now],
                factors,
            )
            supports[name] = page_support(scores[name], budget=2048)
        exact_pages = selected_pages(*supports["exact_k"], length=length_now, historical_only=False)
        exact_routed = selected_pages(*supports["exact_k"], length=length_now, historical_only=True)
        for name, (ids, valid) in supports.items():
            assert (valid.sum(-1) <= 2048).all()
            gathered_ids = ids.clamp_max(position)[:, :, None].expand(batch, kv_heads, head_groups, -1)
            gathered_valid = valid[:, :, None].expand_as(gathered_ids)
            for metric, distribution in (
                ("attention_mass_recall", probabilities),
                ("nonsink_mass_recall", nonsink_probabilities),
            ):
                mass = distribution.gather(-1, gathered_ids).masked_fill(~gathered_valid, 0).sum(-1)
                assert torch.isfinite(mass).all() and mass.min() >= 0 and mass.max() <= 1.00001
                totals[name][metric] += float(mass.mean()) / len(positions)
            for metric, reference, historical_only in (
                ("page_recall", exact_pages, False),
                ("routed_page_recall", exact_routed, True),
            ):
                selected = selected_pages(ids, valid, length=length_now, historical_only=historical_only)
                denominator = reference.sum(-1)
                eligible = denominator > 0
                recall = ((selected & reference).sum(-1) / denominator.clamp_min(1))[eligible]
                if eligible.any():
                    totals[name][metric] += float(recall.mean())
                    recall_counts[name][metric] += 1
            if length_now - 64 > 32:
                totals[name]["page_kl_exact_to_proxy"] += (
                    0.0 if name == "exact_k" else page_kl(exact_scores, scores[name], length=length_now)
                )
                page_kl_counts[name] += 1
            selected_logits = exact_scores.gather(-1, gathered_ids).masked_fill(~gathered_valid, -torch.inf)
            selected_values = values[:, :, :length_now].gather(
                2, ids.clamp_max(position)[..., None].expand(batch, kv_heads, ids.shape[-1], dim)
            )
            sparse_output = selected_logits.softmax(-1) @ selected_values
            difference = sparse_output - dense_output
            wo_difference = difference.reshape(batch, -1) @ output_weight.T
            for field, tensor in (
                ("output_squared_error", difference),
                ("output_energy", dense_output),
                ("wo_squared_error", wo_difference),
                ("wo_energy", dense_wo),
            ):
                assert torch.isfinite(tensor).all()
                totals[name][field] += float(tensor.double().square().sum())
    for name in totals:
        assert page_kl_counts[name] > 0
        totals[name]["page_kl_exact_to_proxy"] /= page_kl_counts[name]
        for metric in ("page_recall", "routed_page_recall"):
            assert recall_counts[name][metric] > 0
            totals[name][metric] /= recall_counts[name][metric]
    return totals


def aggregate_layer(windows, arms):
    means = {
        arm: {metric: sum(window["metrics"][arm][metric] for window in windows) / len(windows) for metric in METRICS}
        for arm in arms
    }
    sums = {
        arm: {field: sum(window["metrics"][arm][field] for window in windows) for field in ERROR_FIELDS}
        for arm in arms
    }
    for arm in arms:
        means[arm]["attention_rel_mse"] = sums[arm]["output_squared_error"] / sums[arm]["output_energy"]
        means[arm]["post_wo_rel_mse"] = sums[arm]["wo_squared_error"] / sums[arm]["wo_energy"]
    return means, sums


@torch.inference_mode()
def evaluate(args, *, smoke):
    root = args.root
    banks_by_name = bank_paths(root, args.section4, args.matrix, smoke=smoke)
    output = args.section4 / "diagnostics" / args.matrix
    identity = read_json(root / "manifests/v128.json")
    windows_manifest = read_json(root / "calibration/manifest.json")
    assert identity["layer_ranks"] == [128] * 32
    assert windows_manifest["sha256"] == sha256(root / "calibration/windows.safetensors")
    layers = [0] if smoke else list(range(args.shard_index * 8, (args.shard_index + 1) * 8))
    positions = {}
    bank_hashes = {}
    banks = {layer: {} for layer in layers}
    for layer in (layers if smoke else range(32)):
        moment_meta = read_json(root / "moments" / f"layer_{layer:03d}.json")
        positions[layer] = moment_meta["selections"]["heldout"]["selected_positions"]
        assert len(positions[layer]) == 32
        bank_hashes[str(layer)] = {}
        for name, folder in banks_by_name.items():
            path = folder / f"layer_{layer:03d}.safetensors"
            meta = read_json(path.with_suffix(".json"))
            assert meta["status"] == "complete" and meta["sha256"] == sha256(path)
            assert meta["layer"] == layer and meta["identity_sha256"] == sha256(root / "manifests/v128.json")
            bank_hashes[str(layer)][name] = meta["sha256"]
            if layer in banks:
                payload = load_file(str(path))
                banks[layer][name] = {key: value.cuda() for key, value in payload.items()}
    protocol = {
        "format": "basisserve.section4.routing_diagnostics.v1",
        "matrix": args.matrix,
        "identity_sha256": sha256(root / "manifests/v128.json"),
        "windows_sha256": windows_manifest["sha256"],
        "sequence_length": 65536,
        "heldout_ids": [64] if smoke else list(range(64, 80)),
        "queries_per_window": 32,
        "positions": {str(layer): value for layer, value in positions.items()},
        "budget": 2048,
        "page_size": 32,
        "sink_tokens_inside_budget": 32,
        "recent_tokens_inside_budget": 64,
        "historical_pages": 62,
        "freely_routed_historical_pages": 61,
        "value_mode": "dense original V and Wo",
        "final_attention": "exact selected post-RoPE K and exact QK; proxy scores select support only",
        "page_kl": "KL(exact Page32 LSE distribution || proxy Page32 LSE distribution), per query head over routed historical pages excluding pinned sink32 and recent64",
        "bank_sha256": bank_hashes,
        "smoke": smoke,
        "source_sha256": {
            "evaluation/eval_llama_section4_diagnostics.py": sha256(Path(__file__)),
            "evaluation/llama_sink_recent_routing.py": sha256(Path("evaluation/llama_sink_recent_routing.py")),
            "basisserve/core/c1_v_conditional_k_router.py": sha256(
                Path("basisserve/core/c1_v_conditional_k_router.py")
            ),
        },
    }
    model = AutoModelForCausalLM.from_pretrained(
        identity["model"], dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    windows_tensor = load_file(str(root / "calibration/windows.safetensors"))["input_ids"]
    records = {layer: [] for layer in layers}
    active = {}
    handles = []
    for layer in layers:

        def hook(module, positional, kwargs, layer=layer):
            hidden = kwargs["hidden_states"]
            length = hidden.shape[1]
            query = module.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
            key = module.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            value = module.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            metrics = window_metrics(
                value,
                key,
                query[:, :, positions[layer]],
                positions[layer],
                cos,
                sin,
                banks[layer],
                module.o_proj.weight,
            )
            records[layer].append({"window": active["window"], "metrics": metrics})
            print({"layer": layer, "window": active["window"], "matrix": args.matrix}, flush=True)

        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    selected_windows = [64] if smoke else list(range(64, 80))
    for window in selected_windows:
        active["window"] = window
        result = model.model(windows_tensor[window:window + 1].long().cuda(), use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all()
        del result
    for handle in handles:
        handle.remove()
    arms = [*banks_by_name, "exact_k"]
    for layer in layers:
        means, sums = aggregate_layer(records[layer], arms)
        path = output / ("smoke.json" if smoke else f"layer_{layer:03d}.json")
        write_json(
            path,
            {
                "status": "complete",
                "layer": layer,
                "protocol": protocol,
                "means": means,
                "error_sums": sums,
                "windows": records[layer],
                "command": shlex.join(sys.argv),
                "python": sys.executable,
            },
        )


def summarize(args):
    output = args.section4 / "diagnostics" / args.matrix
    reports = [read_json(output / f"layer_{layer:03d}.json") for layer in range(32)]
    protocol = reports[0]["protocol"]
    assert all(report["status"] == "complete" and report["layer"] == layer for layer, report in enumerate(reports))
    arms = list(reports[0]["means"])
    means = {}
    for arm in arms:
        means[arm] = {
            metric: sum(report["means"][arm][metric] for report in reports) / 32
            for metric in (*METRICS, "attention_rel_mse", "post_wo_rel_mse")
        }
        means[arm]["pooled_attention_rel_mse"] = sum(
            report["error_sums"][arm]["output_squared_error"] for report in reports
        ) / sum(report["error_sums"][arm]["output_energy"] for report in reports)
        means[arm]["pooled_post_wo_rel_mse"] = sum(
            report["error_sums"][arm]["wo_squared_error"] for report in reports
        ) / sum(report["error_sums"][arm]["wo_energy"] for report in reports)
    write_json(
        output / "summary.json",
        {
            "status": "complete",
            "protocol": protocol,
            "means": means,
            "layers": [{"layer": report["layer"], "means": report["means"]} for report in reports],
        },
    )
    print(means, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        choices=("components", "base_rank_sweep", "objectives", "score_only", "qgram_score_only"),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    configure()
    assert 0 <= args.shard_index < 4
    if args.stage == "summarize":
        summarize(args)
    else:
        evaluate(args, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
