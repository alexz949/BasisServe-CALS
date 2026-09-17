#!/usr/bin/env python3
"""RULER88 with 2048 routed historical tokens plus an external recent64."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys
import time

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.core.c1_conditional_page_attention import _selected_pages
from evaluation import eval_k_routing_ruler as common
from evaluation import eval_llama_chunk8_fisher_ruler as strict
from evaluation.k_routing_config import routing_config
from evaluation.ruler_v1 import sample_score
from evaluation.v96kl_common import configure, read_json, sha256, write_json


ARMS = (
    "exact_chunk8_r64",
    "mean_r16_chunk8_r64",
    "old_b16r16_page32_r64",
)
REPORT_ARMS = (
    "full",
    "old_b16r16_hard2048",
    "old_b16r16_r64",
    "exact_chunk8_hard2048",
    "exact_chunk8_r64",
    "mean_r16_hard2048",
    "mean_r16_chunk8_r64",
)
HISTORICAL_BUDGET = 2048
RECENT_TOKENS = 64
TOTAL_CAP = HISTORICAL_BUDGET + RECENT_TOKENS


def _routed_plus_recent_support(
    scores: torch.Tensor,
    total_tokens: int,
    *,
    page_size: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    batch, groups, heads, pages = map(int, scores.shape)
    assert total_tokens > TOTAL_CAP
    assert HISTORICAL_BUDGET % page_size == 0
    assert RECENT_TOKENS % page_size == 0
    assert sink_tokens % page_size == 0
    recent_start = total_tokens - RECENT_TOKENS
    routed_stop = (recent_start // page_size) * page_size
    assert pages == routed_stop // page_size
    tail_tokens = total_tokens - routed_stop
    historical_pages = (TOTAL_CAP - tail_tokens) // page_size
    pinned_pages = sink_tokens // page_size
    proxy = scores.reshape(batch, groups * heads, 1, pages)
    selected, selected_valid = _selected_pages(
        proxy,
        torch.ones_like(proxy, dtype=torch.bool),
        kv_heads=groups,
        page_size=1,
        page_budget=historical_pages,
        pinned_prefix_pages=pinned_pages,
    )
    selected, order = selected.squeeze(-2).sort(dim=-1)
    selected_valid = selected_valid.squeeze(-2).gather(-1, order)
    assert selected_valid.all()
    offsets = torch.arange(page_size, device=scores.device)
    historical = (selected[..., None] * page_size + offsets).flatten(-2)
    tail = torch.arange(routed_stop, total_tokens, device=scores.device).expand(
        batch, groups, -1
    )
    ids = torch.cat((historical, tail), dim=-1)
    actual = int(ids.shape[-1])
    tail_pages = math.ceil(tail_tokens / page_size)
    assert actual <= TOTAL_CAP
    assert historical_pages + tail_pages == TOTAL_CAP // page_size
    assert int(ids.min()) == 0 and int(ids.max()) == total_tokens - 1
    assert torch.all(ids[..., 1:] > ids[..., :-1])
    return ids, {
        "historical_budget_tokens": HISTORICAL_BUDGET,
        "recent_outside_tokens": RECENT_TOKENS,
        "logical_support_cap_tokens": TOTAL_CAP,
        "actual_support_tokens": actual,
        "page_size": page_size,
        "equivalent_page_slots": TOTAL_CAP // page_size,
        "historical_pages": historical_pages,
        "pinned_sink_pages": pinned_pages,
        "routed_historical_pages": historical_pages - pinned_pages,
        "tail_pages": tail_pages,
        "exact_tail_tokens": tail_tokens,
        "unused_token_capacity": TOTAL_CAP - actual,
    }


def chunk8_routed_plus_recent_support(
    chunk_logits: torch.Tensor,
    total_tokens: int,
    *,
    budget: int = 2048,
) -> tuple[torch.Tensor, dict[str, int]]:
    assert budget == HISTORICAL_BUDGET
    return _routed_plus_recent_support(
        chunk_logits,
        total_tokens,
        page_size=8,
        sink_tokens=32,
    )


def page32_routed_plus_recent_support(
    token_scores: torch.Tensor,
    budget: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert budget == HISTORICAL_BUDGET
    batch, groups, heads, total_tokens = map(int, token_scores.shape)
    recent_start = total_tokens - RECENT_TOKENS
    routed_stop = (recent_start // 32) * 32
    page_scores = torch.logsumexp(
        token_scores[..., :routed_stop]
        .float()
        .reshape(batch, groups, heads, routed_stop // 32, 32),
        dim=-1,
    )
    ids, _ = _routed_plus_recent_support(
        page_scores,
        total_tokens,
        page_size=32,
        sink_tokens=32,
    )
    return ids, torch.ones_like(ids, dtype=torch.bool)


def inputs(args, tokenizer):
    (
        identity,
        manifest,
        rows,
        base_bank,
        mean_bank,
        base_hashes,
        mean_hashes,
        baseline,
        strict_protocol,
    ) = strict.inputs(args, tokenizer)
    strict_summary = read_json(args.strict_summary)
    assert strict_summary["status"] == "complete"
    assert strict_summary["new_verified_predictions"] == 264
    assert strict_summary["protocol"] == strict_protocol
    assert set(strict_summary["results"]).issuperset(
        {"exact_chunk8", "mean_r16_chunk8"}
    )
    old_bank = {}
    for record in manifest["layers"]:
        layer = int(record["layer"])
        path = args.base_bank / f"layer_{layer:03d}.safetensors"
        payload = load_file(str(path))
        assert set(payload) == {
            "base_left_b16",
            "base_right_b16",
            "base_bias_b16",
            "residual_encoder_b16_r16",
            "residual_query_b16_r16",
        }
        old_bank[layer] = payload
    source_names = (
        "evaluation/eval_llama_chunk8_recent64_ruler.py",
        "evaluation/eval_llama_chunk8_fisher_ruler.py",
        "basisserve/core/chunk8_fisher_routing.py",
        "evaluation/eval_k_routing_ruler.py",
        "evaluation/llama_sink_recent_routing.py",
        "basisserve/core/c1_conditional_page_attention.py",
        "basisserve/core/c1_v_conditional_k_router.py",
        "basisserve/kernels/split_indexed_attention.py",
    )
    protocol = {
        "format": "basisserve.llama31_8b.routed2048_plus_recent64.ruler88.v1",
        "identity_sha256": sha256(args.identity),
        "data_sha256": sha256(args.data / "manifest.json"),
        "baseline_sha256": sha256(args.baseline),
        "strict_chunk8_summary_sha256": sha256(args.strict_summary),
        "sequence_length": args.sequence_length,
        "samples": 88,
        "tasks": list(strict.TASK_NAMES),
        "dtype": "bfloat16",
        "generation": "greedy, native EOS, official caps",
        "input_template": "tokenizer.apply_chat_template user message, add_generation_prompt=True",
        "selected_attention": "exact post-RoPE K and dense V with original Wo",
        "budget": {
            "historical_tokens": HISTORICAL_BUDGET,
            "recent_tokens_outside": RECENT_TOKENS,
            "total_support_cap_tokens": TOTAL_CAP,
            "sink_tokens_inside_historical_budget": 32,
            "chunk8_aligned": "4 sink + 252 routed historical + 8 recent chunks",
            "page32_aligned": "1 sink + 63 routed historical + 2 recent pages",
            "ragged_policy": "retain the complete page crossing the recent boundary and remain at or below 2112 tokens",
        },
        "arms": {
            "exact_chunk8_r64": "exact token-QK Chunk8 log-sum-exp",
            "mean_r16_chunk8_r64": "Base16 plus direct Mean-Chunk Fisher R16",
            "old_b16r16_page32_r64": "token-level Base16/Residual16 Page32 log-mass",
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
        old_bank,
        base_hashes,
        mean_hashes,
        baseline,
        strict_summary,
        json.loads(json.dumps(protocol, allow_nan=False)),
    )


def install(model, identity, manifest, arm, base_bank, mean_bank, old_bank):
    if arm == "old_b16r16_page32_r64":
        from evaluation import llama_sink_recent_routing

        llama_sink_recent_routing.page_support = page32_routed_plus_recent_support
        common.install(
            model,
            Path(identity["checkpoint"]),
            manifest,
            "ours",
            old_bank,
            dense_v=True,
        )
        return
    strict.chunk8_hard_budget_support = chunk8_routed_plus_recent_support
    strict.install(
        model,
        identity,
        manifest,
        "exact_chunk8" if arm == "exact_chunk8_r64" else "mean_r16_chunk8",
        base_bank,
        mean_bank,
    )


@torch.inference_mode()
def generate(model, tokenizer, row, arm, cap):
    if arm == "old_b16r16_page32_r64":
        ids, first, statistics, stopped = common.generate(
            model,
            tokenizer,
            row,
            "ours",
            cap,
        )
        for record in statistics:
            if record:
                record["historical_budget_tokens"] = HISTORICAL_BUDGET
                record["recent_outside_tokens"] = RECENT_TOKENS
                record["logical_support_cap_tokens"] = TOTAL_CAP
                record["actual_support_tokens"] = int(record["selected_tokens_mean"])
                record["token_budget"] = TOTAL_CAP
        return ids, first, statistics, stopped
    mapped = "exact_chunk8" if arm == "exact_chunk8_r64" else "mean_r16_chunk8"
    return strict.generate(model, tokenizer, row, mapped, cap)


def _verify(saved, row, protocol, tokenizer, baseline_first):
    strict._verify_result(saved, row, protocol, tokenizer, baseline_first)
    assert all(saved["result"]["routing"])
    for record in saved["result"]["routing"]:
        assert record["historical_budget_tokens"] == HISTORICAL_BUDGET
        assert record["recent_outside_tokens"] == RECENT_TOKENS
        assert record["logical_support_cap_tokens"] == TOTAL_CAP
        assert record["actual_support_tokens"] <= TOTAL_CAP


def audit_smoke(args, rows, protocol, tokenizer, baseline):
    verified = 0
    for arm in ARMS:
        for index in (0, 56):
            saved = read_json(args.output / arm / "smoke" / f"sample_{index:03d}.json")
            _verify(
                saved,
                rows[index],
                protocol,
                tokenizer,
                baseline["results"]["full"][index]["ids"][0],
            )
            assert len(saved["result"]["ids"]) > 1
            verified += 1
    write_json(
        args.output / "smoke_audit.json",
        {
            "status": "complete",
            "protocol": protocol,
            "verified": verified,
            "dense_prefill_first_token_agreement": True,
            "sparse_decode_exercised": True,
            "support_never_exceeds_2112": True,
        },
    )
    print("ALL RECENT-OUTSIDE SMOKES VERIFIED", verified, flush=True)


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


def summarize(args, rows, protocol, tokenizer, baseline, strict_summary):
    results = {
        "full": baseline["results"]["full"],
        "old_b16r16_hard2048": baseline["results"]["ours"],
        "exact_chunk8_hard2048": strict_summary["results"]["exact_chunk8"],
        "mean_r16_hard2048": strict_summary["results"]["mean_r16_chunk8"],
    }
    for arm in ARMS:
        records = []
        for row in rows:
            saved = read_json(
                args.output / arm / "evaluate" / f"sample_{row['index']:03d}.json"
            )
            _verify(
                saved,
                row,
                protocol,
                tokenizer,
                results["full"][row["index"]]["ids"][0],
            )
            records.append(saved["result"])
        results[arm] = records
    results["old_b16r16_r64"] = results.pop("old_b16r16_page32_r64")
    means = {
        arm: 100 * sum(record["score"] for record in records) / 88
        for arm, records in results.items()
    }
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
        for task in strict.TASK_NAMES
    }
    comparisons = {
        "exact_r64_vs_exact_hard2048": _paired(
            results["exact_chunk8_r64"], results["exact_chunk8_hard2048"]
        ),
        "mean_r64_vs_mean_hard2048": _paired(
            results["mean_r16_chunk8_r64"], results["mean_r16_hard2048"]
        ),
        "old_r64_vs_old_hard2048": _paired(
            results["old_b16r16_r64"], results["old_b16r16_hard2048"]
        ),
        "mean_r64_vs_full": _paired(results["mean_r16_chunk8_r64"], results["full"]),
        "mean_r64_vs_exact_r64": _paired(
            results["mean_r16_chunk8_r64"], results["exact_chunk8_r64"]
        ),
    }
    write_json(
        args.output / "summary.json",
        {
            "status": "complete",
            "new_verified_predictions": len(ARMS) * 88,
            "protocol": protocol,
            "means": means,
            "tasks": tasks,
            "paired": comparisons,
            "results": results,
        },
    )
    lines = [
        "# Llama-3.1-8B-Instruct RULER 64K: B2048 routed + recent64",
        "",
        "11 tasks × 8 prompts; 88 paired examples. Dense V128 and original Wo.",
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
            "## Paired budget ablations",
            "",
            "| Comparison | Delta (points) | Wins | Ties | Losses | Identical generations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in comparisons.items():
        lines.append(
            f"| {name} | {values['mean_score_delta_points']:.4f} | "
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
    result.add_argument("--arm", choices=ARMS, default="mean_r16_chunk8_r64")
    result.add_argument("--identity", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--base-bank", type=Path, required=True)
    result.add_argument("--mean-bank", type=Path, required=True)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--strict-summary", type=Path, required=True)
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
    tokenizer = AutoTokenizer.from_pretrained(identity["model"], local_files_only=True)
    (
        identity,
        manifest,
        rows,
        base_bank,
        mean_bank,
        old_bank,
        base_hashes,
        mean_hashes,
        baseline,
        strict_summary,
        protocol,
    ) = inputs(args, tokenizer)
    if args.stage == "summarize":
        summarize(args, rows, protocol, tokenizer, baseline, strict_summary)
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
    install(model, identity, manifest, args.arm, base_bank, mean_bank, old_bank)
    selected = (
        [rows[0], rows[56]]
        if args.stage == "smoke"
        else rows[args.shard_index :: args.num_shards]
    )
    for row in selected:
        path = args.output / args.arm / args.stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            _verify(
                read_json(path),
                row,
                protocol,
                tokenizer,
                baseline["results"]["full"][row["index"]]["ids"][0],
            )
            continue
        cap = min(4, row["maximum_tokens"]) if args.stage == "smoke" else row["maximum_tokens"]
        print("START", args.arm, row["index"], row["task"], len(row["input_ids"]), flush=True)
        started = time.monotonic()
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        ids, first, statistics, stopped = generate(model, tokenizer, row, args.arm, cap)
        if args.stage == "smoke":
            again, other, _, _ = generate(model, tokenizer, row, args.arm, cap)
            assert ids == again
            torch.testing.assert_close(first, other, atol=0, rtol=0)
        baseline_first = baseline["results"]["full"][row["index"]]["ids"][0]
        assert ids[0] == baseline_first
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
        _verify(saved, row, protocol, tokenizer, baseline_first)
        write_json(path, saved)
        print("COMPLETE", args.arm, row["index"], result["score"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
