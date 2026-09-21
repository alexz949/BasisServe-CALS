#!/usr/bin/env python3
"""RULER-64K with Page32 sparse exact-QK attention and an FP8 V96 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from types import MethodType

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
    conditional_routing_query_projector,
)
from basisserve.core.fp8_value_latent import SplitPagedE4M3Cache, base_payload_hadamard
from basisserve.core.routing_basis import make_routing_basis
from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention
from evaluation import eval_k_routing_ruler as common
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import configure, read_json, sha256, tensor_hash, write_json


METHODS = ("bf16", "bf16_full", "bf16_exact_sparse", "fp8_h")
TASKS = common.DEFAULT_TASK_NAMES
SAMPLES_PER_TASK = 8
GOOD_V96_IDENTITY = Path(
    "results/k_routing_fit/llama31_8b_instruct_128k/manifests/v96_hf_good.json"
)
GOOD_V96_ROUTER = Path(
    "/deac/csc/yangGrp/zhangal/.cache/huggingface/hub/"
    "models--alexz949--BasisServe-CALS/snapshots/"
    "6e72ddd726cc14b8a56d3d7233898f11d0e1fbc6/"
    "checkpoints/k_routing/llama31_8b_instruct_v96_b16r16_32x128k_als40_pcg100"
)
GOOD_V96_ROUTER_REVISION = "6e72ddd726cc14b8a56d3d7233898f11d0e1fbc6"


class FP8RoutingCache(common.RoutingCache):
    def __init__(self, config):
        super().__init__(config)
        self.fp8_values = {}


def load_inputs(args):
    identity_path = args.identity
    identity = read_json(identity_path)
    tokenizer = AutoTokenizer.from_pretrained(identity["model"], local_files_only=True)
    namespace = argparse.Namespace(
        identity=identity_path,
        data=args.data,
        bank=args.router / "layers",
        output=args.output,
        sequence_length=65536,
        rope="native",
        dense_v=False,
        native_audit=None,
        full_smoke=None,
        wo_bank=None,
        arm="full",
        stage="evaluate",
        official_lrqk=False,
        chat_template=True,
    )
    identity, manifest, rows, _, _, protocol = common.inputs(
        namespace, tokenizer, task_names=TASKS, samples_per_task=SAMPLES_PER_TASK
    )
    assert identity["layer_ranks"] == [96] * 32 and len(rows) == 88
    router_manifest_path = args.router / "manifest.json"
    router_manifest = read_json(router_manifest_path)
    router_identity_path = args.router / "value_identity.json"
    router_identity = read_json(router_identity_path)
    router_identity_sha256 = sha256(router_identity_path)
    assert router_manifest["status"] == "complete"
    assert router_manifest["format"] == "basisserve.v96_router_upload.v1"
    assert router_manifest["value_checkpoint"]["manifest_sha256"] == identity["manifest_sha256"]
    assert router_manifest["value_checkpoint"]["revision"] == identity["source_revision"]
    assert router_identity["manifest_sha256"] == identity["manifest_sha256"]
    assert router_identity["repository_revision"] == identity["source_revision"]
    files = {record["file"]: record for record in router_manifest["files"]}
    assert len(files) == 64
    bank, hashes = {}, {}
    for layer in range(32):
        tensor_name = f"layer_{layer:03d}.safetensors"
        record_name = f"layer_{layer:03d}.json"
        path = args.router / tensor_name
        record_path = args.router / record_name
        meta = read_json(record_path)
        assert files[tensor_name]["sha256"] == meta["sha256"] == sha256(path)
        assert files[record_name]["sha256"] == sha256(record_path)
        assert meta["layer"] == layer and meta["v_rank"] == 96
        assert meta["identity_sha256"] == router_identity_sha256
        assert meta["protocol"] == router_manifest["protocol"]
        assert meta["protocol"]["format"] == "basisserve.k_router.streaming.v2"
        assert meta["protocol"]["sequence_length"] == 131072
        assert meta["protocol"]["fit_ids"] == list(range(32))
        assert meta["protocol"]["diagnostic_ids"] == list(range(32, 48))
        bank[layer] = load_file(str(path))
        hashes[str(layer)] = files[tensor_name]["sha256"]
    protocol.update(
        experiment="V96 FP8 compatibility with sparse decode attention",
        method=args.method,
        v_checkpoint=str(Path(identity["checkpoint"]).resolve()),
        v_manifest_sha256=identity["manifest_sha256"],
        router_checkpoint=str(args.router.resolve()),
        router_repository_revision=args.router_revision,
        router_manifest_sha256=sha256(router_manifest_path),
        router_identity_sha256=router_identity_sha256,
        router_objective=("Page-Fisher" if args.method in ("bf16", "fp8_h") else None),
        value_mode="C1 V96; BF16 reference or physical E4M3 cache",
        routing={
            "bf16": "B16R16 Page32, B2048, sink32 and recent64 inside budget",
            "bf16_full": "full attention; no routing or token sparsity",
            "bf16_exact_sparse": "exact-QK Page32 oracle, B2048, sink32 and recent64 inside budget",
            "fp8_h": "B16R16 Page32, B2048, sink32 and recent64 inside budget",
        }[args.method],
        final_attention=(
            "full exact QK"
            if args.method == "bf16_full"
            else "sparse exact selected post-RoPE QK; selected V coordinates only"
        ),
        fp8=(
            {
                "format": "torch.float8_e4m3fn",
                "hadamard": "routing-aligned blockdiag(H16,H64,H16)",
                "scale": "separate Base16/Payload80 FP32 scale per Page32 and KV head",
                "append": "page scale fixed when page is created; later tokens clamp to that scale",
                "read": "only selected FP8 codes and their page scales are gathered/dequantized",
                "residual": "BF16 and recomputed against deployed dequantized Base16",
                "exact_key": "BF16",
                "systems_claim": False,
            }
            if args.method == "fp8_h"
            else None
        ),
    )
    protocol["source_sha256"]["evaluation/eval_llama_v96_fp8_sparse_ruler.py"] = sha256(Path(__file__))
    protocol["source_sha256"]["basisserve/core/fp8_value_latent.py"] = sha256(
        Path("basisserve/core/fp8_value_latent.py")
    )
    return tokenizer, identity, manifest, rows, bank, hashes, json.loads(json.dumps(protocol))


def transformed_factors(value_factors, router_factors, *, fp8):
    encoder = value_factors["value_coordinate_encoders"]
    decoder = value_factors["head_output_decoders"]
    if not fp8:
        return encoder, decoder, {
            "base_left": router_factors["base_left_b16"],
            "base_right": router_factors["base_right_b16"],
            "base_bias": router_factors["base_bias_b16"],
            "residual_encoder": router_factors["residual_encoder_b16_r16"],
            "residual_query": router_factors["residual_query_b16_r16"],
        }
    basis = make_routing_basis(router_factors["base_left_b16"])
    hadamard = base_payload_hadamard()
    encoder = torch.bmm(
        basis.encoder(encoder.double()), hadamard.expand(8, -1, -1)
    ).to(torch.bfloat16)
    decoder = torch.bmm(
        hadamard.mT.expand(32, -1, -1), basis.decoder(decoder.double())
    ).to(torch.bfloat16)
    base_right = torch.bmm(
        hadamard[:16, :16].mT.expand(8, -1, -1),
        router_factors["base_right_b16"].double(),
    ).float()
    return encoder, decoder, {
        "base_left": torch.eye(16).expand(8, -1, -1).contiguous(),
        "base_right": base_right,
        "base_bias": router_factors["base_bias_b16"],
        "residual_encoder": router_factors["residual_encoder_b16_r16"],
        "residual_query": router_factors["residual_query_b16_r16"],
    }


def sparse_fp8_attention(query, exact_key, selected_value, ids, valid, *, scale):
    batch, query_heads, query_length, head_dim = query.shape
    kv_heads, selected_tokens, value_rank = selected_value.shape[1:]
    assert query_length == 1 and query_heads % kv_heads == 0
    groups = query_heads // kv_heads
    safe_ids = ids.clamp_max(exact_key.shape[2] - 1)
    selected_key = exact_key.gather(
        2, safe_ids[..., None].expand(batch, kv_heads, selected_tokens, head_dim)
    )
    grouped_query = query[:, :, 0].reshape(batch, kv_heads, groups, head_dim)
    scores = torch.einsum("bkgd,bksd->bkgs", grouped_query, selected_key).mul_(scale)
    scores.masked_fill_(~valid[:, :, None], -torch.inf)
    probability = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    output = torch.einsum("bkgs,bksr->bkgr", probability, selected_value)
    return output.reshape(batch, query_heads, 1, value_rank)


@torch.inference_mode()
def fp8_routing_forward(
    self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs
):
    batch, length, _ = hidden_states.shape
    assert batch == 1
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    query = self.q_proj(hidden_states).view(batch, length, 32, 128).transpose(1, 2)
    pre_key = self.k_proj(hidden_states).view(batch, length, 8, 128).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch, length, 8, 96).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, pre_key, cos, sin)
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or length == 1
    store = past_key_values.fp8_values.setdefault(self.layer_idx, SplitPagedE4M3Cache())
    deployed_value = store.append(value)
    assert store.token_count == previous + length
    factors = self._routing_factors
    current = build_conditional_routing_sidecar(
        deployed_value[..., :16],
        key,
        base_left=factors["base_left"],
        base_right=factors["base_right"],
        base_bias=factors["base_bias"],
        residual_encoder=factors["residual_encoder"],
        cos=cos,
        sin=sin,
    )
    past_key_values.sidecars[self.layer_idx] = (
        torch.cat((past_key_values.sidecars[self.layer_idx], current), dim=2) if previous else current
    )
    key, dense_value = past_key_values.update(key, value, self.layer_idx)
    if previous == 0:
        output = compressed_v_prefill_attention(query, key, dense_value, scale=self.scaling)
    else:
        groups = self.num_attention_heads // self.num_key_value_heads
        query_code = torch.einsum(
            "bhqd,hdr->bhqr", query.float(), self._routing_projector.float()
        )[:, :, 0].reshape(batch, self.num_key_value_heads, groups, -1)
        scores = (
            query_code @ past_key_values.sidecars[self.layer_idx].float().transpose(-1, -2)
        ) * self.scaling
        ids, valid = page_support(scores, budget=2048)
        selected_value = store.gather(ids.clamp_max(store.token_count - 1), dtype=query.dtype)
        output = sparse_fp8_attention(
            query, key, selected_value, ids, valid, scale=self.scaling
        )
        past_key_values.statistics[self.layer_idx] = {
            "selected_tokens_mean": float(valid.sum(-1).float().mean()),
            "sink_tokens": 32,
            "recent_tokens": 64,
            "token_budget": 2048,
            "fp8_cache_bytes": store.storage_bytes,
            "sparse_selected_fp8_value_bytes": int(valid.sum()) * 96,
            "dense_value_read": False,
        }
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


@torch.inference_mode()
def install(model, identity, manifest, bank, *, method):
    fp8 = method == "fp8_h"
    arm = {
        "bf16": "ours",
        "bf16_full": "full",
        "bf16_exact_sparse": "exact_sparse",
        "fp8_h": "ours",
    }[method]
    checkpoint = Path(identity["checkpoint"])
    for layer, record in zip(model.model.layers, manifest["layers"], strict=True):
        module = layer.self_attn
        module.q_norm = torch.nn.Identity()
        module.k_norm = torch.nn.Identity()
        module.num_key_value_groups = 4
        module.sliding_window = None
        value_factors = load_file(str(checkpoint / record["file"]))
        assert {"value_coordinate_encoders", "head_output_decoders"}.issubset(value_factors)
        if "source_ranks" in value_factors:
            assert value_factors["source_ranks"].tolist() == record["ranks"]
        encoder, decoder, factors = transformed_factors(value_factors, bank[record["layer"]], fp8=fp8)
        device = module.v_proj.weight.device
        weight = torch.bmm(
            encoder.to(device).float().mT, module.v_proj.weight.float().reshape(8, 128, 4096)
        ).reshape(768, 4096)
        output = decoder.to(device).float().permute(2, 0, 1).reshape(4096, 3072)
        replacement = GQATiedVOQwen3Attention(
            module,
            v_proj_compressed_weight=weight,
            o_decoder_weight=output,
            attention_backend="triton",
            value_coordinate_encoder=encoder,
        )
        replacement._routing_arm = arm
        replacement._routing_base_rank = 16
        replacement._routing_factors = {name: tensor.to(device) for name, tensor in factors.items()}
        replacement._routing_projector = conditional_routing_query_projector(
            replacement._routing_factors["residual_query"]
        ).to(device)
        replacement.forward = MethodType(fp8_routing_forward if fp8 else common.routing_forward, replacement)
        layer.self_attn = replacement
    model.eval()


@torch.inference_mode()
def generate(model, tokenizer, row, cap):
    cache = FP8RoutingCache(model.config)
    device = model.get_input_embeddings().weight.device
    tokens = torch.tensor([row["input_ids"]], device=device)
    output = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
    assert torch.isfinite(output.logits).all()
    first = output.logits[0, -1].float().cpu()
    ids = [int(first.argmax())]
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    while len(ids) < cap and ids[-1] not in eos:
        mask = torch.ones(1, 1, 1, cache.get_seq_length() + 1, device=device, dtype=torch.bool)
        output = model(
            input_ids=torch.tensor([[ids[-1]]], device=device),
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            logits_to_keep=1,
        )
        assert torch.isfinite(output.logits).all()
        ids.append(int(output.logits[0, -1].argmax()))
    statistics = [cache.statistics.get(layer, {}) for layer in range(32)]
    return ids, first, statistics, ids[-1] in eos


@torch.inference_mode()
def run(args, *, smoke):
    tokenizer, identity, manifest, rows, bank, hashes, protocol = load_inputs(args)
    assert torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name()
    model = common.load_evaluation_model(
        identity, common.routing_config(identity, rope="native", sequence_length=65536)
    )
    install(model, identity, manifest, bank, method=args.method)
    selected = [rows[0]] if smoke else rows[args.shard_index :: args.num_shards]
    stage = "smoke" if smoke else "evaluate"
    for row in selected:
        path = args.output / args.method / stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            continue
        torch.manual_seed(0)
        started = time.monotonic()
        cap = min(args.smoke_max_tokens, row["maximum_tokens"]) if smoke else row["maximum_tokens"]
        ids, first, statistics, stopped = generate(model, tokenizer, row, cap)
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        saved = {
            "status": "complete",
            "sample": row,
            "protocol": protocol,
            "bank_sha256": hashes,
            "command": shlex.join(sys.argv),
            "python": sys.executable,
            "gpu": torch.cuda.get_device_name(),
            "result": {
                "ids": ids,
                "first_logits_sha256": tensor_hash(first),
                "prediction": prediction,
                "score": common.sample_score(prediction, row["answers"], row["match_type"]),
                "stopped": stopped,
                "routing": statistics,
                "seconds": time.monotonic() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        }
        write_json(path, saved)
        print("COMPLETE", args.method, row["index"], saved["result"]["score"], flush=True)
    if smoke:
        write_json(
            args.output / args.method / "smoke_audit.json",
            {"status": "complete", "method": args.method, "protocol": protocol, "sample": 0},
        )


def summarize(args):
    tokenizer, _, _, rows, _, hashes, protocol = load_inputs(args)
    results = []
    for row in rows:
        saved = read_json(args.output / args.method / "evaluate" / f"sample_{row['index']:03d}.json")
        assert saved["protocol"] == protocol and saved["bank_sha256"] == hashes
        result = saved["result"]
        assert common.sample_score(result["prediction"], row["answers"], row["match_type"]) == result["score"]
        results.append(result)
    tasks = {
        task: 100
        * sum(result["score"] for row, result in zip(rows, results, strict=True) if row["task"] == task)
        / SAMPLES_PER_TASK
        for task in TASKS
    }
    summary = {
        "status": "complete",
        "method": args.method,
        "samples": len(rows),
        "mean": 100 * sum(result["score"] for result in results) / len(results),
        "tasks": tasks,
        "protocol": protocol,
        "bank_sha256": hashes,
        "results": results,
    }
    write_json(args.output / args.method / "summary.json", summary)
    print(json.dumps({"method": args.method, "mean": summary["mean"], "tasks": tasks}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "evaluate", "summarize"))
    parser.add_argument("--identity", type=Path, default=GOOD_V96_IDENTITY)
    parser.add_argument(
        "--router",
        type=Path,
        default=GOOD_V96_ROUTER,
    )
    parser.add_argument("--router-revision", default=GOOD_V96_ROUTER_REVISION)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/zhangal/BasisServe-CALS-runs/llama31_8b_instruct_64k/ruler64k"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation/llama31_v96_hf_good_fp8_sparse_ruler64k"),
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--smoke-max-tokens", type=int, default=32)
    args = parser.parse_args()
    configure()
    assert 0 <= args.shard_index < args.num_shards
    if args.stage == "smoke":
        run(args, smoke=True)
    elif args.stage == "evaluate":
        gate = read_json(args.output / args.method / "smoke_audit.json")
        assert gate["status"] == "complete" and gate["method"] == args.method
        run(args, smoke=False)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
