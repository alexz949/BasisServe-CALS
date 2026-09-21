#!/usr/bin/env python3
"""Numerical FP8 compatibility pilot for routing-aligned Llama V96."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from basisserve.core.fp8_value_latent import base_payload_hadamard, quantize_paged_e4m3
from basisserve.core.routing_basis import make_routing_basis
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import read_json, sha256, write_json


ARMS = ("bf16_no_h", "bf16_h", "fp8_no_h", "fp8_h")
ROUTING_METRICS = (
    "attention_mass_recall",
    "nonsink_mass_recall",
    "routed_page_recall",
    "page_kl_exact_to_proxy",
    "selected_page_overlap",
    "selected_page_jaccard",
)


def relative_sums(actual, reference):
    difference = actual.double() - reference.double()
    return float(difference.square().sum()), float(reference.double().square().sum())


def selected_pages(ids, valid, *, length, historical_only):
    page_count = math.ceil(length / 32)
    selected = torch.zeros(ids.shape[0], ids.shape[1], page_count, device=ids.device, dtype=torch.int32)
    if historical_only:
        historical = length - 64
        valid = valid & (ids >= 32) & (ids < historical)
    selected.scatter_add_(-1, (ids // 32).clamp_max(page_count - 1), valid.int())
    return selected > 0


def page_kl(teacher_scores, proxy_scores, *, length):
    historical = length - 64
    assert historical > 32

    def distribution(scores):
        scores = scores[..., :historical].reshape(scores.shape[0], -1, historical).float()
        pages = math.ceil(historical / 32)
        padding = pages * 32 - historical
        if padding:
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
        page_scores = torch.logsumexp(scores.reshape(*scores.shape[:-1], pages, 32), dim=-1)
        page_scores[..., 0] = -torch.inf
        return torch.log_softmax(page_scores, dim=-1)

    teacher_log = distribution(teacher_scores)
    proxy_log = distribution(proxy_scores)
    teacher = teacher_log.exp()
    terms = torch.where(teacher > 0, teacher * (teacher_log - proxy_log), torch.zeros_like(teacher))
    result = terms.sum(-1)
    assert torch.isfinite(result).all() and result.min() >= -1e-6
    return float(result.mean())


def predicted_key_and_scores(coordinates, query, exact_key, cos, sin, factors, base_right):
    dtype = coordinates.dtype
    predicted_pre = torch.einsum(
        "bhtv,hvd->bhtd",
        coordinates[..., :16],
        base_right.to(device=coordinates.device, dtype=dtype),
    )
    predicted_pre.add_(factors["base_bias_b16"].to(device=coordinates.device, dtype=dtype)[None, :, None])
    rotary_cos = cos.to(dtype=dtype).unsqueeze(1)
    rotary_sin = sin.to(dtype=dtype).unsqueeze(1)
    half = predicted_pre.shape[-1] // 2
    rotated_half = torch.cat((-predicted_pre[..., half:], predicted_pre[..., :half]), dim=-1)
    predicted_post = predicted_pre * rotary_cos + rotated_half * rotary_sin
    residual = exact_key - predicted_post
    residual_code = torch.einsum(
        "bhtd,hdr->bhtr",
        residual,
        factors["residual_encoder_b16_r16"].to(device=coordinates.device, dtype=dtype),
    )
    query_residual = torch.einsum(
        "bhd,hdr->bhr",
        query,
        factors["residual_query_b16_r16"].to(device=query.device, dtype=query.dtype),
    )
    query_code = torch.cat((query, query_residual), dim=-1).float()
    sidecar = torch.cat((predicted_post, residual_code), dim=-1).float()
    batch, query_heads, width = query_code.shape
    kv_heads = exact_key.shape[1]
    scores = (
        query_code.reshape(batch, kv_heads, query_heads // kv_heads, width)
        @ sidecar.transpose(-1, -2)
    ) * exact_key.shape[-1] ** -0.5
    return predicted_post, scores


def decoded_attention_output(probabilities, coordinates, decoder):
    latent = torch.einsum("bkgp,bkpd->bkgd", probabilities.float(), coordinates.float())
    latent = latent.reshape(latent.shape[0], -1, latent.shape[-1])
    return torch.einsum("bhd,hdm->bm", latent, decoder.float())


def selected_attention_output(exact_scores, support, coordinates, decoder):
    ids, valid = support
    batch, kv_heads, groups, _ = exact_scores.shape
    gather_scores = ids[:, :, None].expand(batch, kv_heads, groups, -1)
    gather_valid = valid[:, :, None].expand_as(gather_scores)
    selected_logits = exact_scores.gather(-1, gather_scores).masked_fill(~gather_valid, -torch.inf)
    weights = selected_logits.softmax(-1)
    selected_coordinates = coordinates.gather(
        2, ids[..., None].expand(batch, kv_heads, ids.shape[-1], coordinates.shape[-1])
    )
    return decoded_attention_output(weights, selected_coordinates, decoder)


def make_arm_coordinates(value, encoder, decoder, factors):
    device = value.device
    basis = make_routing_basis(factors["base_left_b16"].to(device))
    aligned_encoder = basis.encoder(encoder.double()).to(torch.bfloat16)
    aligned_decoder = basis.decoder(decoder.double()).to(torch.bfloat16)
    hadamard = base_payload_hadamard(device=device)
    per_head_hadamard = hadamard.expand(8, -1, -1)
    per_query_hadamard = hadamard.expand(32, -1, -1)
    rotated_encoder = torch.bmm(aligned_encoder.double(), per_head_hadamard).to(torch.bfloat16)
    rotated_decoder = torch.bmm(per_query_hadamard.mT, aligned_decoder.double()).to(torch.bfloat16)
    base_hadamard = hadamard[:16, :16]
    rotated_base_right = torch.bmm(
        base_hadamard.mT.expand(8, -1, -1), factors["base_right_b16"].to(device).double()
    ).to(torch.bfloat16)

    raw_coordinates = torch.einsum("bhtd,hdr->bhtr", value, encoder)
    aligned = torch.einsum("bhtd,hdr->bhtr", value, aligned_encoder)
    rotated = torch.einsum("bhtd,hdr->bhtr", value, rotated_encoder)

    def fp8_roundtrip(coordinates):
        base = quantize_paged_e4m3(coordinates[..., :16])
        payload = quantize_paged_e4m3(coordinates[..., 16:])
        return torch.cat((base.dequantize(), payload.dequantize()), dim=-1), {
            "code_bytes": base.codes.numel() + payload.codes.numel(),
            "scale_bytes": base.scales.numel() * base.scales.element_size()
            + payload.scales.numel() * payload.scales.element_size(),
            "base_scale_amax": float(base.scales.max()),
            "payload_scale_amax": float(payload.scales.max()),
        }

    fp8_aligned, no_h_storage = fp8_roundtrip(aligned)
    fp8_rotated, h_storage = fp8_roundtrip(rotated)
    arms = {
        "bf16_no_h": (aligned, aligned_decoder, factors["base_right_b16"].to(device).to(torch.bfloat16)),
        "bf16_h": (rotated, rotated_decoder, rotated_base_right),
        "fp8_no_h": (fp8_aligned, aligned_decoder, factors["base_right_b16"].to(device).to(torch.bfloat16)),
        "fp8_h": (fp8_rotated, rotated_decoder, rotated_base_right),
    }
    storage = {
        "bf16_no_h": {
            "code_bytes": aligned.numel() * aligned.element_size(),
            "scale_bytes": 0,
        },
        "bf16_h": {
            "code_bytes": rotated.numel() * rotated.element_size(),
            "scale_bytes": 0,
        },
        "fp8_no_h": no_h_storage,
        "fp8_h": h_storage,
    }
    return arms, storage, raw_coordinates, decoder, aligned, per_head_hadamard, basis


def empty_totals():
    pairs = (
        "latent",
        "predicted_key",
        "predicted_key_delta",
        "full_attention_output",
        "fixed_support_output",
        "routed_output",
    )
    return {
        arm: {
            **{f"{name}_{field}": 0.0 for name in pairs for field in ("squared_error", "energy")},
            **{name: 0.0 for name in ROUTING_METRICS},
            "queries": 0,
        }
        for arm in ARMS
    }


@torch.inference_mode()
def layer_metrics(module, hidden, cos, sin, *, layer, positions, value_factors, routing_factors):
    length = hidden.shape[1]
    query = module.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
    pre_key = module.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
    value = module.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, pre_key, cos, sin)
    encoder = value_factors["value_coordinate_encoders"].to(value.device)
    decoder = value_factors["head_output_decoders"].to(value.device)
    factors = {name: tensor.to(value.device) for name, tensor in routing_factors.items()}
    arms, storage, raw_coordinates, raw_decoder, aligned, hadamard, basis = make_arm_coordinates(
        value, encoder, decoder, factors
    )
    totals = empty_totals()
    baseline_coordinates, baseline_decoder, baseline_base_right = arms["bf16_no_h"]

    recovered = {
        "bf16_no_h": arms["bf16_no_h"][0].float(),
        "fp8_no_h": arms["fp8_no_h"][0].float(),
        "bf16_h": torch.einsum("bhtr,hrs->bhts", arms["bf16_h"][0].float(), hadamard.mT.float()),
        "fp8_h": torch.einsum("bhtr,hrs->bhts", arms["fp8_h"][0].float(), hadamard.mT.float()),
    }
    for arm in ARMS:
        error, energy = relative_sums(recovered[arm], aligned.float())
        totals[arm]["latent_squared_error"] += error
        totals[arm]["latent_energy"] += energy

    baseline_prediction, _ = predicted_key_and_scores(
        baseline_coordinates,
        query[:, :, positions[0]],
        key,
        cos,
        sin,
        factors,
        baseline_base_right,
    )
    predictions = {}
    for arm, (coordinates, _, base_right) in arms.items():
        prediction, _ = predicted_key_and_scores(
            coordinates, query[:, :, positions[0]], key, cos, sin, factors, base_right
        )
        predictions[arm] = prediction
        error, energy = relative_sums(prediction, key)
        totals[arm]["predicted_key_squared_error"] += error
        totals[arm]["predicted_key_energy"] += energy
        error, energy = relative_sums(prediction, baseline_prediction)
        totals[arm]["predicted_key_delta_squared_error"] += error
        totals[arm]["predicted_key_delta_energy"] += energy

    raw_dense_errors = []
    h_dense_errors = []
    h_prediction_errors = []
    h_support_equal = []
    for position in positions:
        length_now = position + 1
        current_query = query[:, :, position]
        exact_scores = (
            current_query.float().reshape(1, 8, 4, 128)
            @ key[:, :, :length_now].float().transpose(-1, -2)
        ) * 128**-0.5
        probabilities = exact_scores.softmax(-1)
        nonsink_scores = exact_scores.clone()
        nonsink_scores[..., :32] = -torch.inf
        nonsink_probabilities = nonsink_scores.softmax(-1)
        exact_support = page_support(exact_scores, budget=2048)
        scores = {}
        supports = {}
        for arm, (coordinates, _, base_right) in arms.items():
            _, scores[arm] = predicted_key_and_scores(
                coordinates[:, :, :length_now],
                current_query,
                key[:, :, :length_now],
                cos[:, :length_now],
                sin[:, :length_now],
                factors,
                base_right,
            )
            supports[arm] = page_support(scores[arm], budget=2048)
        baseline_support = supports["bf16_no_h"]
        exact_pages = selected_pages(*exact_support, length=length_now, historical_only=True)
        baseline_pages = selected_pages(*baseline_support, length=length_now, historical_only=True)

        baseline_full_output = decoded_attention_output(
            probabilities, baseline_coordinates[:, :, :length_now], baseline_decoder
        )
        baseline_fixed_output = selected_attention_output(
            exact_scores, baseline_support, baseline_coordinates[:, :, :length_now], baseline_decoder
        )
        raw_output = decoded_attention_output(probabilities, raw_coordinates[:, :, :length_now], raw_decoder)
        raw_dense_errors.append(relative_sums(baseline_full_output, raw_output))

        for arm, (coordinates, arm_decoder, _) in arms.items():
            support = supports[arm]
            ids, valid = support
            gather = ids[:, :, None].expand(1, 8, 4, -1)
            gather_valid = valid[:, :, None].expand_as(gather)
            for field, distribution in (
                ("attention_mass_recall", probabilities),
                ("nonsink_mass_recall", nonsink_probabilities),
            ):
                mass = distribution.gather(-1, gather).masked_fill(~gather_valid, 0).sum(-1)
                totals[arm][field] += float(mass.mean())
            current_pages = selected_pages(*support, length=length_now, historical_only=True)
            exact_denominator = exact_pages.sum(-1).clamp_min(1)
            totals[arm]["routed_page_recall"] += float(
                ((current_pages & exact_pages).sum(-1) / exact_denominator).mean()
            )
            baseline_denominator = baseline_pages.sum(-1).clamp_min(1)
            intersection = (current_pages & baseline_pages).sum(-1)
            union = (current_pages | baseline_pages).sum(-1).clamp_min(1)
            totals[arm]["selected_page_overlap"] += float((intersection / baseline_denominator).mean())
            totals[arm]["selected_page_jaccard"] += float((intersection / union).mean())
            totals[arm]["page_kl_exact_to_proxy"] += page_kl(
                exact_scores, scores[arm], length=length_now
            )

            full_output = decoded_attention_output(
                probabilities, coordinates[:, :, :length_now], arm_decoder
            )
            fixed_output = selected_attention_output(
                exact_scores, baseline_support, coordinates[:, :, :length_now], arm_decoder
            )
            routed_output = selected_attention_output(
                exact_scores, support, coordinates[:, :, :length_now], arm_decoder
            )
            for name, actual, reference in (
                ("full_attention_output", full_output, baseline_full_output),
                ("fixed_support_output", fixed_output, baseline_fixed_output),
                ("routed_output", routed_output, baseline_fixed_output),
            ):
                error, energy = relative_sums(actual, reference)
                totals[arm][f"{name}_squared_error"] += error
                totals[arm][f"{name}_energy"] += energy
            totals[arm]["queries"] += 1

        h_dense_errors.append(
            relative_sums(
                decoded_attention_output(
                    probabilities, arms["bf16_h"][0][:, :, :length_now], arms["bf16_h"][1]
                ),
                baseline_full_output,
            )
        )
        h_prediction_errors.append(relative_sums(predictions["bf16_h"][:, :, :length_now], baseline_prediction[:, :, :length_now]))
        h_support_equal.append(bool(torch.equal(supports["bf16_h"][0], baseline_support[0])))

    metrics = {}
    for arm, current in totals.items():
        row = {}
        for name in (
            "latent",
            "predicted_key",
            "predicted_key_delta",
            "full_attention_output",
            "fixed_support_output",
            "routed_output",
        ):
            row[f"{name}_rel_mse"] = current[f"{name}_squared_error"] / max(
                current[f"{name}_energy"], 1e-30
            )
        for name in ROUTING_METRICS:
            row[name] = current[name] / current["queries"]
        row["queries"] = current["queries"]
        row["storage"] = storage[arm]
        token_heads = value.shape[0] * value.shape[1] * value.shape[2]
        row["storage"]["bytes_per_token_per_kv_head"] = (
            row["storage"]["code_bytes"] + row["storage"]["scale_bytes"]
        ) / token_heads
        metrics[arm] = row

    def combined_relative(pairs):
        return sum(pair[0] for pair in pairs) / max(sum(pair[1] for pair in pairs), 1e-30)

    audit = {
        "routing_basis_max_condition": float(basis.condition.max()),
        "raw_to_aligned_full_attention_rel_mse": combined_relative(raw_dense_errors),
        "hadamard_full_attention_rel_mse": combined_relative(h_dense_errors),
        "hadamard_predicted_key_delta_rel_mse": combined_relative(h_prediction_errors),
        "hadamard_selected_token_indices_identical_fraction": sum(h_support_equal) / len(h_support_equal),
        "aligned_coordinate_amax": float(aligned.float().abs().max()),
        "rotated_coordinate_amax": float(arms["bf16_h"][0].float().abs().max()),
    }
    return {"layer": layer, "metrics": metrics, "gauge_audit": audit}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/k_routing_fit/llama31_8b_instruct_128k")
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert args.sequence_length > 2048 and args.sequence_length <= 131072
    assert args.queries > 0 and all(0 <= layer < 32 for layer in args.layers)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20260920)

    v_checkpoint = args.root / "checkpoints/L31-8B-Instruct-C1U-V96-32x128K"
    router_checkpoint = args.root / "checkpoints/L31-8B-Instruct-C1U-V96-B16R16-32x128K"
    identity = read_json(args.root / "manifests/v96.json")
    window_manifest = read_json(args.root / "calibration/manifest.json")
    assert identity["layer_ranks"] == [96] * 32
    assert window_manifest["sha256"] == sha256(args.root / "calibration/windows.safetensors")
    assert args.window in range(32, 48)
    value_factors = {
        layer: load_file(str(v_checkpoint / "layers" / f"layer_{layer:03d}.safetensors"))
        for layer in args.layers
    }
    routing_factors = {
        layer: load_file(str(router_checkpoint / "layers" / f"layer_{layer:03d}.safetensors"))
        for layer in args.layers
    }
    positions = torch.linspace(2048, args.sequence_length - 1, args.queries + 1, dtype=torch.long)[1:].tolist()
    assert len(set(positions)) == args.queries and positions[-1] == args.sequence_length - 1

    model = AutoModelForCausalLM.from_pretrained(
        identity["model"], dtype=torch.bfloat16, attn_implementation="flash_attention_2", local_files_only=True
    ).eval().cuda()
    records = []
    handles = []
    for layer in args.layers:

        def hook(module, positional, kwargs, layer=layer):
            record = layer_metrics(
                module,
                kwargs["hidden_states"],
                *kwargs["position_embeddings"],
                layer=layer,
                positions=positions,
                value_factors=value_factors[layer],
                routing_factors=routing_factors[layer],
            )
            records.append(record)
            print({"layer": layer, "status": "complete"}, flush=True)

        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    windows = load_file(str(args.root / "calibration/windows.safetensors"))["input_ids"]
    input_ids = windows[args.window : args.window + 1, : args.sequence_length].long().cuda()
    output = model.model(input_ids, use_cache=False)
    assert torch.isfinite(output.last_hidden_state).all()
    for handle in handles:
        handle.remove()
    assert [record["layer"] for record in records] == args.layers

    result = {
        "status": "complete",
        "format": "basisserve.llama31_instruct.v96_fp8_compatibility.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "python": sys.executable,
        "environment": {
            "conda": "basis",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "protocol": {
            "model": identity["model"],
            "model_config_sha256": identity["model_config_sha256"],
            "v_checkpoint": str(v_checkpoint.resolve()),
            "v_manifest_sha256": sha256(v_checkpoint / "manifest.json"),
            "router_checkpoint": str(router_checkpoint.resolve()),
            "router_manifest_sha256": sha256(router_checkpoint / "manifest.json"),
            "router_objective": "Page-Fisher",
            "window": args.window,
            "sequence_length": args.sequence_length,
            "positions": positions,
            "layers": args.layers,
            "page_size": 32,
            "budget": 2048,
            "sink_tokens_inside_budget": 32,
            "recent_tokens_inside_budget": 64,
            "quantization": {
                "format": "torch.float8_e4m3fn",
                "physical_fp8_storage": True,
                "scale_granularity": "separate Base16/Payload80, per page, per KV head",
                "scale_dtype": "torch.float32",
                "base_rank": 16,
                "payload_rank": 80,
                "residual_rank": 16,
                "residual_dtype": "torch.bfloat16",
                "exact_key_dtype": "torch.bfloat16",
            },
            "hadamard": "routing-aligned blockdiag(H16,H64,H16), normalized Sylvester",
            "residual_semantics": "BF16 residual code recomputed from each arm's dequantized Base prediction; factors are unchanged",
            "output_metrics": "BF16 BasisKV no-H reference; fixed-support isolates Value error and routed-output includes support changes",
        },
        "source_sha256": {
            "evaluation/eval_llama_v96_fp8_compatibility.py": sha256(Path(__file__)),
            "basisserve/core/fp8_value_latent.py": sha256(Path("basisserve/core/fp8_value_latent.py")),
            "basisserve/core/routing_basis.py": sha256(Path("basisserve/core/routing_basis.py")),
            "evaluation/llama_sink_recent_routing.py": sha256(Path("evaluation/llama_sink_recent_routing.py")),
        },
        "layers": records,
    }
    write_json(args.output, result)
    print({"output": str(args.output), "status": "complete"}, flush=True)


if __name__ == "__main__":
    main()
