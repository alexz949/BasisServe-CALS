#!/usr/bin/env python3
"""Evaluate Qwen3-32B V96 ShadowKV, LRQK, and Loki on frozen 128K RULER."""

import argparse
from collections import Counter
from dataclasses import asdict
import functools
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import sys
import time
from types import MethodType

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.checkpoint.c1_lrqk_qwen3 import install_c1_lrqk
from basisserve.checkpoint.c1_shadowkv_qwen3 import (
    install_c1_shadowkv,
)
from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_lrqk import LRQKConfig
from evaluation import eval_k_routing_ruler as runtime
from evaluation.deterministic_evaluation import (
    NUMERICAL_POLICY,
    configure_deterministic_evaluation,
)
from evaluation.qwen3_32b_v96_baseline_offload import (
    audit_selective_residency,
    cache_shapes,
    create_cache,
    install as install_offload,
    residency as selective_residency,
)
from evaluation.ruler_v1 import parse_tasks, ruler_prompt, sample_score
from evaluation.v96kl_common import read_json, save_tensors, sha256, write_json


ARMS = ("shadowkv", "lrqk", "loki")
TASKS = (
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
SEQUENCE_LENGTH = 131072
SAMPLES_PER_TASK = 100
GENERATION_SEED = 0
RULER_SEED = 42
PROMPT_MARGIN = 128
YARN_FACTOR = 4.0
SMOKE_IDS = (0, 7 * SAMPLES_PER_TASK)
LOKI_FORMAT = "basisserve.qwen3_32b.loki_key_pca_r32_128k.v1"
LOKI_FIT_COORDINATE = "centered post-RoPE K PCA after Qwen3 k_norm and YaRN RoPE"
LOKI_RUNTIME_COORDINATE = "post-RoPE Q/K projection without mean subtraction"
LOKI_TOPK = 856
LRQK = LRQKConfig(
    rank=32,
    topk=832,
    recent=64,
    prefill_iterations=2,
    decode_iterations=2,
    tolerance=0.01,
    seed=0,
)
SOURCES = (
    "evaluation/eval_qwen3_32b_v96_ruler.py",
    "evaluation/qwen3_32b_v96_baseline_offload.py",
    "evaluation/deterministic_evaluation.py",
    "evaluation/ruler_v1.py",
    "evaluation/v96kl_common.py",
    "evaluation/chunked_prefill_mlp.py",
    "evaluation/eval_k_routing_ruler.py",
    "basisserve/checkpoint/gqa_vo_qwen3.py",
    "basisserve/checkpoint/c1_lrqk_qwen3.py",
    "basisserve/checkpoint/c1_shadowkv_qwen3.py",
    "basisserve/core/c1_lrqk.py",
    "basisserve/core/c1_shadowkv.py",
    "basisserve/core/c1_loki_attention.py",
    "basisserve/kernels/compressed_v_decode_attention.py",
    "basisserve/kernels/indexed_sparse_decode_attention.py",
)


def effective_config(model_path):
    native = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert native.model_type == "qwen3"
    assert native.num_hidden_layers == 64
    assert native.num_attention_heads == 64
    assert native.num_key_value_heads == 8
    assert native.head_dim == 128
    assert native.max_position_embeddings == 40960
    maximum = int(round(native.max_position_embeddings * YARN_FACTOR))
    assert maximum >= SEQUENCE_LENGTH
    rope = {
        **dict(native.rope_parameters),
        "rope_type": "yarn",
        "factor": YARN_FACTOR,
        "original_max_position_embeddings": native.max_position_embeddings,
    }
    return AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        max_position_embeddings=maximum,
        rope_parameters=rope,
    )


def factor_identity(model_path, factors):
    path = factors / "results.json"
    result = read_json(path)
    fit = result["fit_config"]
    assert result["format"] == "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
    assert result["status"] == "complete"
    assert result["layers"] == list(range(64))
    assert set(map(int, result["artifacts"])) == set(range(64))
    assert fit["model_type"] == "qwen3"
    assert fit["hidden_size"] == 5120
    assert fit["num_query_heads"] == 64
    assert fit["num_physical_kv_heads"] == 8
    assert fit["head_dim"] == 128
    assert fit["num_hidden_layers"] == 64
    assert fit["cache_rank_per_head"] == 96
    model_config_sha = sha256(model_path / "config.json")
    if "model_config_sha256" in fit:
        assert fit["model_config_sha256"] == model_config_sha
    artifact_hashes = {}
    for layer in range(64):
        record = result["artifacts"][str(layer)]
        artifact = factors / record["file"]
        assert sha256(artifact) == record["sha256"]
        artifact_hashes[str(layer)] = record["sha256"]
    return {
        "results_sha256": sha256(path),
        "artifact_sha256": artifact_hashes,
        "fit_config": fit,
        "aggregate": result["aggregate"],
    }


def loki_identity(model_path, loki, expected_fit_windows=32):
    path = loki / "manifest.json"
    manifest = read_json(path)
    assert manifest["format"] == LOKI_FORMAT
    assert manifest["status"] == "complete"
    assert manifest["coordinate"] == LOKI_FIT_COORDINATE
    assert manifest["runtime"] == LOKI_RUNTIME_COORDINATE
    assert manifest["rank"] == 32
    assert manifest["fit_windows"] == expected_fit_windows
    assert manifest["fit_ids"] == list(range(expected_fit_windows))
    assert manifest["smoke"] == (expected_fit_windows < 32)
    assert manifest["sequence_length"] == SEQUENCE_LENGTH
    assert manifest["model_config_sha256"] == sha256(model_path / "config.json")
    assert [record["layer"] for record in manifest["layers"]] == list(range(64))
    hashes = {}
    for record in manifest["layers"]:
        path = loki / record["file"]
        assert sha256(path) == record["sha256"]
        hashes[str(record["layer"])] = record["sha256"]
    return manifest, {"manifest_sha256": sha256(loki / "manifest.json"), "layer_sha256": hashes}


def prepare(args, tokenizer):
    data = read_json(args.data / "manifest.json")
    protocol = data["protocol"]
    assert data["status"] == "complete"
    assert data["format"] == "basisserve.ruler_v1.qwen3_base_dataset.v1"
    assert protocol["model_template_type"] == "base"
    assert protocol["sequence_length"] == SEQUENCE_LENGTH
    assert protocol["samples_per_task"] == SAMPLES_PER_TASK
    assert protocol["random_seed"] == RULER_SEED
    assert protocol["prompt_margin"] == PROMPT_MARGIN
    assert protocol["tasks"] == list(TASKS)
    assert data["tokenizer_config_sha256"] == sha256(args.model / "tokenizer_config.json")
    rows, tensors = [], {}
    for task in parse_tasks(",".join(TASKS)):
        source_path = args.data / task.name / "validation.jsonl"
        assert sha256(source_path) == data["artifacts"][task.name]["sha256"]
        sources = [json.loads(line) for line in source_path.read_text().splitlines()]
        assert len(sources) == SAMPLES_PER_TASK
        for ordinal, source in enumerate(sources):
            ids = tokenizer(ruler_prompt(source), add_special_tokens=True)["input_ids"]
            assert len(ids) + task.tokens_to_generate <= SEQUENCE_LENGTH
            index = len(rows)
            tensor = torch.tensor(ids, dtype=torch.int32)
            tensors[str(index)] = tensor
            rows.append(
                {
                    "index": index,
                    "task": task.name,
                    "ordinal": ordinal,
                    "input_tokens": len(ids),
                    "input_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                    "answers": source["outputs"],
                    "maximum_tokens": task.tokens_to_generate,
                    "match_type": task.match_type,
                }
            )
    assert len(rows) == len(TASKS) * SAMPLES_PER_TASK
    save_tensors(args.output / "prompts.safetensors", tensors)
    write_json(
        args.output / "prompts.json",
        {
            "status": "complete",
            "rows": rows,
            "model_config_sha256": sha256(args.model / "config.json"),
            "tokenizer_config_sha256": sha256(args.model / "tokenizer_config.json"),
            "data_sha256": sha256(args.data / "manifest.json"),
            "tokens_sha256": sha256(args.output / "prompts.safetensors"),
            "input_template": "official RULER base completion; tokenizer special tokens enabled",
        },
    )
    print(
        {
            "status": "prepared",
            "prompts": len(rows),
            "minimum_tokens": min(row["input_tokens"] for row in rows),
            "maximum_tokens": max(row["input_tokens"] for row in rows),
        },
        flush=True,
    )


def inputs(args):
    data = read_json(args.data / "manifest.json")
    assert data["status"] == "complete"
    assert data["protocol"] == {
        "model_template_type": "base",
        "sequence_length": SEQUENCE_LENGTH,
        "samples_per_task": SAMPLES_PER_TASK,
        "random_seed": RULER_SEED,
        "prompt_margin": PROMPT_MARGIN,
        "tasks": list(TASKS),
    }
    prompts = read_json(args.output / "prompts.json")
    assert prompts["status"] == "complete"
    assert prompts["model_config_sha256"] == sha256(args.model / "config.json")
    assert prompts["tokenizer_config_sha256"] == sha256(args.model / "tokenizer_config.json")
    assert prompts["data_sha256"] == sha256(args.data / "manifest.json")
    assert prompts["tokens_sha256"] == sha256(args.output / "prompts.safetensors")
    rows = prompts["rows"]
    assert len(rows) == len(TASKS) * SAMPLES_PER_TASK
    assert Counter(row["task"] for row in rows) == dict.fromkeys(TASKS, SAMPLES_PER_TASK)
    assert [row["index"] for row in rows] == list(range(len(rows)))
    factors = factor_identity(args.model, args.factors)
    _, loki = loki_identity(args.model, args.loki)
    config = effective_config(args.model)
    spec = {
        "format": "basisserve.qwen3_32b_v96_ruler128k_baselines.v1",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "factors": factors,
        "loki": loki,
        "prompts_sha256": sha256(args.output / "prompts.json"),
        "data_sha256": sha256(args.data / "manifest.json"),
        "dtype": "bfloat16",
        "sequence_length": SEQUENCE_LENGTH,
        "samples": len(rows),
        "samples_per_task": SAMPLES_PER_TASK,
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "value_cache": "uniform C1 V96 with refitted output decoder",
        "rope": {
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": 40960,
            "effective_max_position_embeddings": config.max_position_embeddings,
        },
        "ruler_seed": RULER_SEED,
        "generation_seed": GENERATION_SEED,
        "generation": "greedy; native EOS; official task caps",
        "batch_size": 1,
        "parallelism": "four independent one-visible-GPU processes; one prompt per process at a time",
        "memory_policy": (
            "one full BF16 model per visible GPU; Loki/LRQK keep V96 on GPU and "
            "fetch selected exact-K rows from pinned CPU; ShadowKV keeps reconstructed "
            "K state on GPU and fetches selected V96 rows from pinned CPU; LRQK rank-32 "
            "codes use pinned CPU plus one shared GPU layer workspace"
        ),
        "shadowkv": {
            "rank": 160,
            "chunk": 8,
            "routed": 2048,
            "outlier_chunks": 48,
            "implementation": "repository equation adapter with resident ShadowKV compressed state",
        },
        "lrqk": {**asdict(LRQK), "implementation": "repository equation adapter"},
        "loki_config": {
            "pca_rank": 32,
            "historical_topk_per_query_head": LOKI_TOPK,
            "recent_tokens": 0,
            "fit_coordinate": LOKI_FIT_COORDINATE,
            "runtime_coordinate": LOKI_RUNTIME_COORDINATE,
        },
        "numerical_policy": NUMERICAL_POLICY,
        "source_sha256": {name: sha256(Path(name)) for name in SOURCES},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "triton", "safetensors")
        },
    }
    native_eos = read_json(args.model / "config.json")["eos_token_id"]
    spec["eos_ids"] = sorted(
        native_eos if isinstance(native_eos, list) else [native_eos]
    )
    return rows, load_file(str(args.output / "prompts.safetensors")), spec


def load_model(args, config):
    assert torch.cuda.device_count() == 1
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    ).eval()
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    records = install_qwen3_gqa_vo_als_export(
        model, args.factors, attention_backend="triton"
    )
    assert len(records) == 64
    assert all(record.value_head_dim == 96 for record in records)
    return model


def install_arm(model, arm, loki_root, model_path, loki_fit_windows=32):
    assert arm in ARMS
    routing_hashes = {}
    if arm == "shadowkv":
        install_c1_shadowkv(model)
    elif arm == "lrqk":
        install_c1_lrqk(model, LRQK)
    else:
        manifest, identity = loki_identity(
            model_path, loki_root, expected_fit_windows=loki_fit_windows
        )
        routing_hashes = identity
        for record, (layer_index, attention) in zip(
            manifest["layers"], c1_attention_layers(model), strict=True
        ):
            assert record["layer"] == layer_index
            payload = load_file(str(loki_root / record["file"]))
            assert set(payload) == {"projector", "mean", "spectrum"}
            projector = payload["projector"]
            assert projector.shape == (8, 128, 32)
            assert projector.dtype == torch.bfloat16
            attention._routing_arm = "loki"
            attention._loki_projector = projector.to(
                device=attention.q_proj.weight.device, dtype=torch.bfloat16
            )
            attention.forward = MethodType(runtime.routing_forward, attention)
        from basisserve.core import c1_loki_attention

        c1_loki_attention.c1_loki_recent_decode = functools.partial(
            c1_loki_attention.c1_loki_recent_decode,
            top_k=LOKI_TOPK,
            recent_tokens=0,
        )
    install_offload(model, arm)
    return routing_hashes


@torch.inference_mode()
def generate(model, tokenizer, row, input_ids, arm, cap):
    torch.manual_seed(GENERATION_SEED)
    cache = create_cache(
        model.config,
        arm,
        row["input_tokens"] + cap,
    )
    device = model.get_input_embeddings().weight.device
    tokens = input_ids.to(device=device, dtype=torch.long).unsqueeze(0)
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
            dtype=torch.bool,
            device=device,
        )
        output = model(
            input_ids=torch.tensor([[ids[-1]]], device=device),
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            logits_to_keep=1,
        )
        assert torch.isfinite(output.logits).all()
        ids.append(int(output.logits[0, -1].argmax()))
        del output
    statistics = []
    expected_length = row["input_tokens"] + len(ids) - 1
    for layer_index, attention in c1_attention_layers(model):
        key_shape, value_shape, stored_length = cache_shapes(
            cache, arm, layer_index
        )
        assert stored_length == expected_length
        if arm == "shadowkv":
            assert key_shape is None
        else:
            assert key_shape == (
                1,
                attention.num_key_value_heads,
                expected_length,
                attention.head_dim,
            )
        assert value_shape == (
            1,
            attention.num_key_value_heads,
            expected_length,
            attention.value_head_dim,
        )
        if layer_index in cache.statistics:
            statistics.append(cache.statistics[layer_index])
        elif arm == "lrqk":
            statistics.append(
                cache.lrqk_states[layer_index].statistics(
                    attention.num_key_value_heads
                )
            )
        elif arm == "shadowkv":
            statistics.append(cache.shadow_states[layer_index].statistics())
        else:
            statistics.append({})
    audit_selective_residency(cache, arm, len(model.model.layers))
    residency = selective_residency(arm)
    stopped = ids[-1] in eos
    del cache
    torch.cuda.empty_cache()
    return ids, first, statistics, stopped, residency


# Provenance-only protocol fields. They record where a run happened, not what it
# computed, so a resumed sample may differ here while staying algorithmically
# identical. This matches the comparison eval_qwen3_32b_v96_shadowkv_tp2.py uses
# for its resumed samples. Every other protocol field stays strictly equal.
PROVENANCE_FIELDS = ("model", "versions", "source_sha256")


def comparable_protocol(protocol):
    return {k: v for k, v in protocol.items() if k not in PROVENANCE_FIELDS}


def audit(saved, row, spec, arm, tokenizer, smoke=False):
    assert saved["status"] == "complete"
    assert comparable_protocol(saved["protocol"]) == comparable_protocol(spec)
    assert saved["sample"] == row
    assert saved["arm"] == arm
    result = saved["result"]
    ids = result["ids"]
    cap = min(4, row["maximum_tokens"]) if smoke else row["maximum_tokens"]
    eos = set(spec["eos_ids"])
    assert 0 < len(ids) <= cap
    assert not any(token in eos for token in ids[:-1])
    assert result["stopped"] == (ids[-1] in eos)
    assert result["stopped"] or len(ids) == cap
    assert saved["first_argmax"] == ids[0]
    assert tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ) == result["prediction"]
    assert sample_score(
        result["prediction"], row["answers"], row["match_type"]
    ) == result["score"]
    if arm == "shadowkv":
        assert result["cache_residency"]["exact_key"] == "not_stored"
        assert result["cache_residency"]["value"] == "pinned_cpu_selected_rows_only"
    else:
        assert result["cache_residency"]["exact_key"] == "pinned_cpu_selected_rows_only"
        assert result["cache_residency"]["value"] == "gpu"
    assert len(result["routing"]) == 64
    assert 0 < result["peak_gib"] < result["gpu_total_gib"]
    if len(ids) > 1 and arm == "lrqk":
        assert all(stat["decode_steps"] == len(ids) - 1 for stat in result["routing"])
        assert all(stat["selected_per_query_head"] <= 896 for stat in result["routing"])
    if len(ids) > 1 and arm == "shadowkv":
        assert all(stat["decode_steps"] == len(ids) - 1 for stat in result["routing"])
        assert all(stat["svd_rank"] == 160 for stat in result["routing"])
        assert all(stat["routed_tokens"] == 2048 for stat in result["routing"])
    if len(ids) > 1 and arm == "loki":
        assert all(stat["selected_per_query_head"] <= LOKI_TOPK for stat in result["routing"])
        assert all(stat["recent_tokens"] == 0 for stat in result["routing"])
    return result


def gate(spec, arm):
    return {"status": "complete", "protocol": spec, "arm": arm, "verified": 2}


def summarize(args, rows, spec, tokenizer):
    results = {}
    for arm in ARMS:
        assert read_json(args.output / arm / "smoke_audit.json") == gate(spec, arm)
        results[arm] = []
        for row in rows:
            path = args.output / arm / "evaluate" / f"sample_{row['index']:04d}.json"
            results[arm].append(audit(read_json(path), row, spec, arm, tokenizer))
    tasks = {
        task: {
            arm: 100
            * sum(
                result["score"]
                for result, row in zip(results[arm], rows, strict=True)
                if row["task"] == task
            )
            / SAMPLES_PER_TASK
            for arm in ARMS
        }
        for task in TASKS
    }
    means = {
        arm: sum(values[arm] for values in tasks.values()) / len(TASKS)
        for arm in ARMS
    }
    summary = {
        "status": "complete",
        "protocol": spec,
        "tasks": tasks,
        "means": means,
        "verified_predictions": len(rows) * len(ARMS),
    }
    write_json(args.output / "summary.json", summary)
    print({"tasks": tasks, "means": means}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "smoke", "audit-smoke", "evaluate", "summarize")
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--loki", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="shadowkv")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--indices", type=int, nargs="+", help="evaluate only these prompt indices"
    )
    args = parser.parse_args()
    configure_deterministic_evaluation()
    assert 0 <= args.shard < args.shards
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.stage == "prepare":
        prepare(args, tokenizer)
        return
    rows, tokens, spec = inputs(args)
    if args.stage == "audit-smoke":
        for index in SMOKE_IDS:
            row = rows[index]
            path = args.output / args.arm / "smoke" / f"sample_{index:04d}.json"
            result = audit(read_json(path), row, spec, args.arm, tokenizer, True)
            assert len(result["ids"]) > 1
        write_json(args.output / args.arm / "smoke_audit.json", gate(spec, args.arm))
        print("SMOKE AUDIT COMPLETE", args.arm, flush=True)
        return
    if args.stage == "summarize":
        summarize(args, rows, spec, tokenizer)
        return
    if args.stage == "evaluate":
        assert read_json(args.output / args.arm / "smoke_audit.json") == gate(
            spec, args.arm
        )
    config = effective_config(args.model)
    model = load_model(args, config)
    install_arm(model, args.arm, args.loki, args.model)
    if args.stage == "smoke":
        selected = [rows[index] for index in SMOKE_IDS]
    elif args.indices:
        selected = [rows[index] for index in args.indices]
    else:
        selected = rows[args.shard :: args.shards]
    directory = "smoke" if args.stage == "smoke" else "evaluate"
    for row in selected:
        path = args.output / args.arm / directory / f"sample_{row['index']:04d}.json"
        if path.exists():
            audit(
                read_json(path),
                row,
                spec,
                args.arm,
                tokenizer,
                args.stage == "smoke",
            )
            continue
        tensor = tokens[str(row["index"])]
        assert len(tensor) == row["input_tokens"]
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row["input_sha256"]
        cap = min(4, row["maximum_tokens"]) if args.stage == "smoke" else row["maximum_tokens"]
        torch.cuda.reset_peak_memory_stats()
        print("START", args.arm, row["index"], row["task"], len(tensor), flush=True)
        started = time.monotonic()
        ids, first, statistics, stopped, residency = generate(
            model, tokenizer, row, tensor, args.arm, cap
        )
        prediction = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = {
            "ids": ids,
            "prediction": prediction,
            "stopped": stopped,
            "score": sample_score(prediction, row["answers"], row["match_type"]),
            "routing": statistics,
            "seconds": time.monotonic() - started,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "cache_residency": residency,
            "cache_value_head_dim": 96,
        }
        saved = {
            "status": "complete",
            "protocol": spec,
            "sample": row,
            "arm": args.arm,
            "result": result,
            "first_argmax": int(first.argmax()),
            "command": shlex.join(sys.argv),
            "python": sys.executable,
            "gpu": torch.cuda.get_device_name(),
        }
        audit(saved, row, spec, args.arm, tokenizer, args.stage == "smoke")
        write_json(path, saved)
        print(
            "COMPLETE",
            args.arm,
            row["index"],
            result["score"],
            result["seconds"],
            result["peak_gib"],
            flush=True,
        )


if __name__ == "__main__":
    main()
