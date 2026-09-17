#!/usr/bin/env python3
"""Paired Llama-3.1-8B-Instruct 64K RULER88 Chunk8 routing pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys
import time
from types import MethodType

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.core.chunk8_fisher_routing import (
    CHUNK_SIZE,
    RECENT_TOKENS,
    ChunkLandmarkState,
    chunk8_hard_budget_support,
    exact_chunk_logits,
    landmark_chunk_logits,
    predict_post_rope_base,
)
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_prefill_attention,
)
from basisserve.kernels.split_indexed_attention import split_indexed_attention
from evaluation import eval_k_routing_ruler as common
from evaluation.k_routing_config import routing_config
from evaluation.ruler_v1 import parse_tasks, ruler_prompt, sample_score
from evaluation.v96kl_common import configure, read_json, sha256, write_json


ARMS = ("exact_chunk8", "base16_chunk8", "mean_r16_chunk8")
TASK_NAMES = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "fwe",
    "qa_1",
    "qa_2",
)
REPORT_ARMS = (
    "full",
    "old_b16r16_page32",
    "shadowkv",
    "lrqk",
    *ARMS,
)


class Chunk8RoutingCache(DynamicCache):
    def __init__(self, config):
        super().__init__(config=config)
        self.landmarks: dict[int, ChunkLandmarkState] = {}
        self.statistics: dict[int, dict[str, int | str]] = {}


def _base_and_residual_update(
    module,
    value: torch.Tensor,
    exact_key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    factors = module._chunk_base_factors
    base = predict_post_rope_base(
        value,
        base_left=factors["base_left_b16"],
        base_right=factors["base_right_b16"],
        base_bias=factors["base_bias_b16"],
        cos=cos,
        sin=sin,
    )
    code = None
    if module._chunk_arm == "mean_r16_chunk8":
        code = torch.einsum(
            "bhtd,hdr->bhtr",
            exact_key - base,
            module._chunk_encoder.to(device=base.device, dtype=base.dtype),
        ).contiguous()
    return base, code


@torch.inference_mode()
def chunk8_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    batch, length, _ = hidden_states.shape
    assert batch == 1 and past_key_values is not None
    query = self.q_norm(
        self.q_proj(hidden_states).view(
            batch, length, self.num_attention_heads, self.head_dim
        )
    ).transpose(1, 2)
    pre_key = self.k_norm(
        self.k_proj(hidden_states).view(
            batch, length, self.num_key_value_heads, self.head_dim
        )
    ).transpose(1, 2)
    value = self.v_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.value_head_dim
    ).transpose(1, 2)
    assert self.value_head_dim == self.head_dim
    cos, sin = position_embeddings
    query, key_update = apply_rotary_pos_emb(query, pre_key, cos, sin)
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or length == 1
    assert attention_mask is None or length == 1

    if self._chunk_arm != "exact_chunk8":
        base_update, residual_update = _base_and_residual_update(
            self,
            value,
            key_update,
            cos,
            sin,
        )
        if previous == 0:
            past_key_values.landmarks[self.layer_idx] = (
                ChunkLandmarkState.from_tokens(base_update, residual_update)
            )
        else:
            past_key_values.landmarks[self.layer_idx].append(
                base_update,
                residual_update,
            )

    key, value = past_key_values.update(key_update, value, self.layer_idx)
    if previous == 0:
        output = compressed_v_prefill_attention(
            query,
            key,
            value,
            scale=self.scaling,
        )
    elif int(key.shape[2]) <= 2048:
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            enable_gqa=True,
            is_causal=False,
            dropout_p=0.0,
            scale=self.scaling,
        )
    else:
        if attention_mask is not None:
            assert attention_mask.shape[-2] == 1
            assert (
                bool(attention_mask.all())
                if attention_mask.dtype == torch.bool
                else bool((attention_mask == 0).all())
            )
        total_tokens = int(key.shape[2])
        complete_chunks = (total_tokens - RECENT_TOKENS) // CHUNK_SIZE
        groups = self.num_attention_heads // self.num_key_value_heads
        grouped_query = query[:, :, 0].float().reshape(
            batch,
            self.num_key_value_heads,
            groups,
            self.head_dim,
        )
        if self._chunk_arm == "exact_chunk8":
            logits = exact_chunk_logits(
                grouped_query,
                key,
                complete_chunks,
                scale=self.scaling,
            )
        else:
            state = past_key_values.landmarks[self.layer_idx]
            query_factor = None
            if self._chunk_arm == "mean_r16_chunk8":
                query_factor = self._chunk_query.reshape(
                    self.num_key_value_heads,
                    groups,
                    self.head_dim,
                    -1,
                )
            logits = landmark_chunk_logits(
                grouped_query,
                state,
                complete_chunks,
                scale=self.scaling,
                query_factor=query_factor,
            )
        selected, statistics = chunk8_hard_budget_support(
            logits,
            total_tokens,
            budget=2048,
        )
        selected = selected.repeat_interleave(groups, dim=1)
        output = split_indexed_attention(
            query,
            key,
            value,
            selected,
            scale=self.scaling,
        )
        past_key_values.statistics[self.layer_idx] = {
            "arm": self._chunk_arm,
            **statistics,
        }

    del query, pre_key
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


def _load_rows(args, tokenizer, identity) -> tuple[dict, list[dict]]:
    data = read_json(args.data / "manifest.json")
    assert data["status"] == "complete"
    assert data["protocol"]["samples_per_task"] == 8
    assert data["protocol"]["sequence_length"] == args.sequence_length
    assert data["tokenizer_config_sha256"] == sha256(
        Path(identity["model"]) / "tokenizer_config.json"
    )
    assert set(TASK_NAMES).issubset(data["protocol"]["tasks"])
    tasks = parse_tasks(",".join(TASK_NAMES))
    rows = []
    for task in tasks:
        path = args.data / task.name / "validation.jsonl"
        assert sha256(path) == data["artifacts"][task.name]["sha256"]
        sources = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(sources) == 8
        for ordinal, source in enumerate(sources):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": ruler_prompt(source)}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            assert len(ids) + task.tokens_to_generate <= args.sequence_length
            rows.append(
                {
                    "index": len(rows),
                    "task": task.name,
                    "ordinal": ordinal,
                    "input_ids": ids,
                    "answers": source["outputs"],
                    "match_type": task.match_type,
                    "maximum_tokens": task.tokens_to_generate,
                }
            )
    assert len(rows) == 88
    return data, rows


def _load_factors(args, identity, manifest):
    base_bank = {}
    mean_bank = {}
    base_hashes = {}
    mean_hashes = {}
    for record in manifest["layers"]:
        layer = int(record["layer"])
        base_path = args.base_bank / f"layer_{layer:03d}.safetensors"
        base_record = read_json(base_path.with_suffix(".json"))
        assert base_record["status"] == "complete"
        assert base_record["layer"] == layer and base_record["v_rank"] == 128
        assert base_record["protocol"]["format"] == "basisserve.k_router.streaming.v1"
        assert base_record["protocol"]["sequence_length"] == 65536
        assert base_record["protocol"]["fit_queries"] == 64
        assert base_record["protocol"]["diagnostic_queries"] == 32
        assert not base_record["protocol"]["smoke"]
        assert base_record["sweeps"] == 40 and base_record["pcg_iterations"] == 100
        assert base_record["identity_sha256"] == sha256(args.identity)
        assert base_record["sha256"] == sha256(base_path)
        base_payload = load_file(str(base_path))
        base_keys = {"base_left_b16", "base_right_b16", "base_bias_b16"}
        assert base_keys.issubset(base_payload)
        base_bank[layer] = {name: base_payload[name] for name in base_keys}
        base_hashes[str(layer)] = base_record["sha256"]

        mean_record_path = args.mean_bank / f"layer_{layer:03d}.json"
        mean_record = read_json(mean_record_path)
        protocol = mean_record["protocol"]
        assert mean_record["status"] == "complete"
        assert protocol["format"] == "basisserve.chunk8_mean_fisher.fit.v1"
        assert protocol["scope"] == "formal all-layer Mean-only fit"
        assert protocol["model"] == identity["model"] and protocol["layer"] == layer
        assert protocol["feature"] == "mean" and protocol["feature_dim"] == 128
        assert protocol["residual_rank"] == 16 and protocol["als_sweeps"] == 8
        assert protocol["final_query_closure"]
        assert protocol["cg_iterations"] == 50
        assert protocol["factor_source_sha256"] == base_record["sha256"]
        mean_path = args.mean_bank / mean_record["artifact"]["file"]
        assert mean_record["artifact"]["sha256"] == sha256(mean_path)
        mean_payload = load_file(str(mean_path))
        assert set(mean_payload) == {"chunk_encoder_r16", "chunk_query_r16"}
        assert mean_payload["chunk_encoder_r16"].shape == (8, 128, 16)
        assert mean_payload["chunk_query_r16"].shape == (32, 128, 16)
        mean_bank[layer] = mean_payload
        mean_hashes[str(layer)] = mean_record["artifact"]["sha256"]
    assert len(base_bank) == len(mean_bank) == 32
    return base_bank, mean_bank, base_hashes, mean_hashes


def inputs(args, tokenizer):
    identity = read_json(args.identity)
    assert identity["value_mode"] == "dense original V and Wo"
    assert identity["layer_ranks"] == [128] * 32
    assert sha256(Path(identity["model"]) / "config.json") == identity["model_config_sha256"]
    checkpoint = Path(identity["checkpoint"])
    manifest = read_json(checkpoint / "manifest.json")
    assert sha256(checkpoint / "manifest.json") == identity["manifest_sha256"]
    for record in manifest["layers"]:
        assert sha256(checkpoint / record["file"]) == record["sha256"]
    runtime = routing_config(identity, rope="native", sequence_length=args.sequence_length)
    assert runtime.model_type == "llama" and runtime.num_hidden_layers == 32
    data, rows = _load_rows(args, tokenizer, identity)
    base_bank, mean_bank, base_hashes, mean_hashes = _load_factors(
        args,
        identity,
        manifest,
    )

    baseline = read_json(args.baseline)
    assert baseline["status"] == "complete" and baseline["verified_predictions"] == 352
    baseline_protocol = baseline["protocol"]
    assert baseline_protocol["identity_sha256"] == sha256(args.identity)
    assert baseline_protocol["data_sha256"] == sha256(args.data / "manifest.json")
    assert baseline_protocol["sequence_length"] == args.sequence_length
    assert baseline_protocol["samples"] == 88
    assert baseline_protocol["value_mode"] == "dense original V and Wo"
    assert baseline_protocol["generation"] == "greedy, native EOS, official caps"
    assert baseline_protocol["input_template"] == (
        "tokenizer.apply_chat_template user message, add_generation_prompt=True"
    )
    assert set(baseline["results"]) == {"full", "lrqk", "shadowkv", "ours"}
    assert all(len(records) == 88 for records in baseline["results"].values())
    baseline_root = args.baseline.parent
    for row in rows:
        saved = read_json(
            baseline_root / "full" / "evaluate" / f"sample_{row['index']:03d}.json"
        )
        assert saved["sample"] == row
        assert saved["result"] == baseline["results"]["full"][row["index"]]

    source_names = (
        "evaluation/eval_llama_chunk8_fisher_ruler.py",
        "basisserve/core/chunk8_fisher_routing.py",
        "evaluation/eval_k_routing_ruler.py",
        "evaluation/k_routing_config.py",
        "evaluation/chunked_prefill_mlp.py",
        "basisserve/checkpoint/gqa_vo_qwen3.py",
        "basisserve/checkpoint/c1_attention_layers.py",
        "basisserve/kernels/compressed_v_decode_attention.py",
        "basisserve/kernels/split_indexed_attention.py",
        "basisserve/core/c1_conditional_page_attention.py",
    )
    protocol = {
        "format": "basisserve.llama31_8b.chunk8_fisher.ruler88.v1",
        "identity_sha256": sha256(args.identity),
        "data_sha256": sha256(args.data / "manifest.json"),
        "baseline_sha256": sha256(args.baseline),
        "sequence_length": args.sequence_length,
        "samples": 88,
        "tasks": list(TASK_NAMES),
        "dtype": "bfloat16",
        "generation": "greedy, native EOS, official caps",
        "input_template": "tokenizer.apply_chat_template user message, add_generation_prompt=True",
        "prefill": "full causal dense-V attention; Chunk8 routing begins at decode",
        "selected_attention": "exact post-RoPE K and dense V with original Wo",
        "arms": {
            "exact_chunk8": "exact token-QK Chunk8 log-sum-exp oracle",
            "base16_chunk8": "frozen Value-derived Base16 post-RoPE Chunk8 means",
            "mean_r16_chunk8": "Base16 plus direct Mean-Chunk Fisher R16",
        },
        "routing": {
            "chunk_size": 8,
            "logical_budget_tokens": 2048,
            "equivalent_chunk_slots": 256,
            "sink_tokens": 32,
            "recent_tokens": 64,
            "aligned_support": "4 sink + 244 routed historical + 8 recent chunks",
            "ragged_support": "4 sink + 243 routed historical + 9 exact-tail chunks; 2041--2047 actual tokens",
            "gqa": "normalize each query head then max within each physical KV group",
        },
        "mean_fit": {
            "calibration": "64 fit + 16 held-out C4 windows, 65536 tokens",
            "queries": "Q32 fit + Q16 held-out per window",
            "sweeps": 8,
            "pcg_iterations": 50,
            "relative_damping": 1e-5,
            "final_query_closure": True,
        },
        "base_bank_sha256": base_hashes,
        "mean_bank_sha256": mean_hashes,
        "source_sha256": {name: sha256(ROOT / name) for name in source_names},
    }
    return (
        identity,
        manifest,
        rows,
        base_bank,
        mean_bank,
        base_hashes,
        mean_hashes,
        baseline,
        json.loads(json.dumps(protocol, allow_nan=False)),
    )


def install(model, identity, manifest, arm, base_bank, mean_bank) -> None:
    common.install(
        model,
        Path(identity["checkpoint"]),
        manifest,
        "full",
        {},
        dense_v=True,
    )
    targets = c1_attention_layers(model)
    assert len(targets) == 32
    for layer, module in targets:
        module._chunk_arm = arm
        module._chunk_base_factors = None
        module._chunk_encoder = None
        module._chunk_query = None
        if arm != "exact_chunk8":
            device = module.q_proj.weight.device
            module._chunk_base_factors = {
                name: value.to(device)
                for name, value in base_bank[layer].items()
            }
            if arm == "mean_r16_chunk8":
                module._chunk_encoder = mean_bank[layer]["chunk_encoder_r16"].to(device)
                module._chunk_query = mean_bank[layer]["chunk_query_r16"].to(device)
        module.forward = MethodType(chunk8_forward, module)
    model.eval()


@torch.inference_mode()
def generate(model, tokenizer, row, arm, cap):
    torch.manual_seed(0)
    cache = Chunk8RoutingCache(config=model.config)
    input_device = model.get_input_embeddings().weight.device
    tokens = torch.tensor([row["input_ids"]], device=input_device)
    output = model(
        input_ids=tokens,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    assert torch.isfinite(output.logits).all()
    first = output.logits[0, -1].float().cpu()
    ids = [int(first.argmax())]
    del output, tokens
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    while len(ids) < cap and ids[-1] not in eos:
        mask = torch.ones(
            1,
            1,
            1,
            cache.get_seq_length() + 1,
            device=input_device,
            dtype=torch.bool,
        )
        output = model(
            input_ids=torch.tensor([[ids[-1]]], device=input_device),
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            logits_to_keep=1,
        )
        assert torch.isfinite(output.logits).all()
        ids.append(int(output.logits[0, -1].argmax()))
        del output
    statistics = []
    expected_tokens = len(row["input_ids"]) + len(ids) - 1
    for layer, module in c1_attention_layers(model):
        key = cache.layers[layer].keys
        value = cache.layers[layer].values
        assert key.shape == (
            1,
            module.num_key_value_heads,
            expected_tokens,
            module.head_dim,
        )
        assert value.shape == key.shape
        if arm != "exact_chunk8":
            state = cache.landmarks[layer]
            assert state.tokens == expected_tokens
            assert state.residual is not None if arm == "mean_r16_chunk8" else state.residual is None
        statistics.append(cache.statistics.get(layer, {}))
    return ids, first, statistics, ids[-1] in eos


def _verify_result(saved, row, protocol, tokenizer, baseline_first):
    assert saved["status"] == "complete"
    assert saved["protocol"] == protocol and saved["sample"] == row
    result = saved["result"]
    ids = result["ids"]
    assert 0 < len(ids) <= row["maximum_tokens"]
    assert ids[0] == baseline_first
    assert tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ) == result["prediction"]
    assert sample_score(
        result["prediction"], row["answers"], row["match_type"]
    ) == result["score"]


def audit_smoke(args, rows, protocol, tokenizer, baseline):
    verified = 0
    for arm in ARMS:
        for index in (0, 56):
            saved = read_json(args.output / arm / "smoke" / f"sample_{index:03d}.json")
            _verify_result(
                saved,
                rows[index],
                protocol,
                tokenizer,
                baseline["results"]["full"][index]["ids"][0],
            )
            assert len(saved["result"]["ids"]) > 1
            assert all(saved["result"]["routing"])
            verified += 1
    write_json(
        args.output / "smoke_audit.json",
        {
            "status": "complete",
            "protocol": protocol,
            "verified": verified,
            "dense_prefill_first_token_agreement": True,
            "sparse_decode_exercised": True,
        },
    )
    print("ALL CHUNK8 SMOKES VERIFIED", verified, flush=True)


def _paired(student, reference):
    differences = [
        float(candidate["score"] - target["score"])
        for candidate, target in zip(student, reference, strict=True)
    ]
    return {
        "mean_score_delta_points": 100 * sum(differences) / len(differences),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "identical_generations": sum(
            candidate["ids"] == target["ids"]
            for candidate, target in zip(student, reference, strict=True)
        ),
    }


def summarize(args, rows, protocol, tokenizer, baseline):
    results = {
        "full": baseline["results"]["full"],
        "old_b16r16_page32": baseline["results"]["ours"],
        "shadowkv": baseline["results"]["shadowkv"],
        "lrqk": baseline["results"]["lrqk"],
    }
    for arm in ARMS:
        records = []
        for row in rows:
            saved = read_json(
                args.output / arm / "evaluate" / f"sample_{row['index']:03d}.json"
            )
            _verify_result(
                saved,
                row,
                protocol,
                tokenizer,
                results["full"][row["index"]]["ids"][0],
            )
            records.append(saved["result"])
        results[arm] = records
    tasks = {
        task: {
            arm: 100
            * sum(
                result["score"]
                for result, row in zip(records, rows, strict=True)
                if row["task"] == task
            )
            / 8
            for arm, records in results.items()
        }
        for task in TASK_NAMES
    }
    means = {
        arm: 100 * sum(result["score"] for result in records) / 88
        for arm, records in results.items()
    }
    paired = {
        arm: _paired(results[arm], results["full"])
        for arm in (*ARMS, "old_b16r16_page32", "shadowkv", "lrqk")
    }
    write_json(
        args.output / "summary.json",
        {
            "status": "complete",
            "new_verified_predictions": len(ARMS) * 88,
            "reused_baseline_predictions": 4 * 88,
            "protocol": protocol,
            "means": means,
            "tasks": tasks,
            "paired_vs_full": paired,
            "results": results,
        },
    )
    lines = [
        "# Llama-3.1-8B-Instruct Chunk8 Fisher routing: RULER 64K pilot",
        "",
        "11 tasks × 8 prompts; 88 paired examples. Dense V128 and original Wo for every arm.",
        "",
        "| Task | " + " | ".join(REPORT_ARMS) + " |",
        "|---|" + "---:|" * len(REPORT_ARMS),
    ]
    for task, values in [*tasks.items(), ("Mean", means)]:
        lines.append(
            "| "
            + task
            + " | "
            + " | ".join(f"{values[arm]:.4f}" for arm in REPORT_ARMS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired comparison against Full-K",
            "",
            "| Arm | Delta (points) | Wins | Ties | Losses | Identical generations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in (*ARMS, "old_b16r16_page32", "shadowkv", "lrqk"):
        values = paired[arm]
        lines.append(
            f"| {arm} | {values['mean_score_delta_points']:.4f} | "
            f"{values['wins']} | {values['ties']} | {values['losses']} | "
            f"{values['identical_generations']} |"
        )
    text = "\n".join(lines) + "\n"
    path = args.output / "summary.md"
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
    print("VERIFIED", means, flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("smoke", "audit-smoke", "evaluate", "summarize"))
    result.add_argument("--arm", choices=ARMS, default="mean_r16_chunk8")
    result.add_argument("--identity", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--base-bank", type=Path, required=True)
    result.add_argument("--mean-bank", type=Path, required=True)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--sequence-length", type=int, default=65536)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--num-shards", type=int, default=4)
    return result


def main():
    args = parser().parse_args()
    configure()
    torch.manual_seed(0)
    identity = read_json(args.identity)
    tokenizer = AutoTokenizer.from_pretrained(
        identity["model"],
        local_files_only=True,
    )
    (
        identity,
        manifest,
        rows,
        base_bank,
        mean_bank,
        base_hashes,
        mean_hashes,
        baseline,
        protocol,
    ) = inputs(args, tokenizer)
    if args.stage == "summarize":
        summarize(args, rows, protocol, tokenizer, baseline)
        return
    if args.stage == "audit-smoke":
        audit_smoke(args, rows, protocol, tokenizer, baseline)
        return
    if args.stage == "evaluate":
        gate = read_json(args.output / "smoke_audit.json")
        assert gate["status"] == "complete" and gate["protocol"] == protocol
    assert 0 <= args.shard_index < args.num_shards
    runtime = routing_config(identity, rope="native", sequence_length=args.sequence_length)
    model = common.load_evaluation_model(identity, runtime)
    install(model, identity, manifest, args.arm, base_bank, mean_bank)
    selected = (
        [rows[0], rows[56]]
        if args.stage == "smoke"
        else rows[args.shard_index :: args.num_shards]
    )
    for row in selected:
        path = args.output / args.arm / args.stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = read_json(path)
            _verify_result(
                saved,
                row,
                protocol,
                tokenizer,
                baseline["results"]["full"][row["index"]]["ids"][0],
            )
            continue
        cap = min(4, row["maximum_tokens"]) if args.stage == "smoke" else row["maximum_tokens"]
        print(
            "START",
            args.arm,
            row["index"],
            row["task"],
            len(row["input_ids"]),
            flush=True,
        )
        started = time.monotonic()
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        ids, first, statistics, stopped = generate(
            model,
            tokenizer,
            row,
            args.arm,
            cap,
        )
        if args.stage == "smoke":
            again, other, _, _ = generate(model, tokenizer, row, args.arm, cap)
            assert ids == again
            torch.testing.assert_close(first, other, atol=0, rtol=0)
        assert ids[0] == baseline["results"]["full"][row["index"]]["ids"][0]
        prediction = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = {
            "ids": ids,
            "prediction": prediction,
            "score": sample_score(prediction, row["answers"], row["match_type"]),
            "stopped": stopped,
            "routing": statistics,
            "seconds": time.monotonic() - started,
            "peak_gib_by_device": {
                str(device): torch.cuda.max_memory_allocated(device) / 2**30
                for device in range(torch.cuda.device_count())
            },
        }
        saved = {
            "status": "complete",
            "sample": row,
            "result": result,
            "protocol": protocol,
            "base_bank_sha256": base_hashes,
            "mean_bank_sha256": mean_hashes,
            "command": shlex.join(sys.argv),
            "python": sys.executable,
            "gpu": torch.cuda.get_device_name(0),
            "device_map": {
                str(key): str(value)
                for key, value in getattr(model, "hf_device_map", {"": 0}).items()
            },
        }
        _verify_result(
            saved,
            row,
            protocol,
            tokenizer,
            baseline["results"]["full"][row["index"]]["ids"][0],
        )
        write_json(path, saved)
        print(
            "COMPLETE",
            args.arm,
            row["index"],
            result["score"],
            result["seconds"],
            flush=True,
        )


if __name__ == "__main__":
    main()
