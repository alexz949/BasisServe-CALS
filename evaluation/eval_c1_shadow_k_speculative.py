#!/usr/bin/env python3
"""Evaluate the C1 shadow-Key speculative correctness/acceptance oracle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402 - direct-script path bootstrap above
import transformers  # noqa: E402 - direct-script path bootstrap above
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    install_qwen3_gqa_vo_als_export,
)
from basisserve.core.c1_shadow_kv import (  # noqa: E402
    C1ShadowKeyValueCache,
    ShadowKeyConfig,
)
from basisserve.core.c1_speculative_decode import (  # noqa: E402
    BlockCommitConfig,
    C1SpeculativeDecodeResult,
    c1_shadow_greedy_decode,
    exact_c1_greedy_decode,
)
from basisserve.core.c1_tp_decode import file_sha256  # noqa: E402


FORMAT = "basisserve.c1_shadow_k_speculative_oracle.v2"


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("draft lengths must be a non-empty positive integer list")
    if len(result) != len(set(result)):
        raise ValueError("draft lengths must be unique")
    return result


def _dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def _load_prompts(path: Path) -> tuple[str, ...]:
    prompts: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            prompt = payload["prompt"]
        else:
            raise ValueError(
                f"prompt record {line_number} must be a JSON string or an object with 'prompt'"
            )
        if not prompt:
            raise ValueError(f"prompt record {line_number} is empty")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("prompt file contains no prompts")
    return tuple(prompts)


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _first_sequence_divergence(
    reference: Sequence[int], observed: Sequence[int]
) -> int | None:
    for index, (reference_token, observed_token) in enumerate(
        zip(reference, observed, strict=False)
    ):
        if reference_token != observed_token:
            return index
    if len(reference) != len(observed):
        return min(len(reference), len(observed))
    return None


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_worktree_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout) if completed.returncode == 0 else None


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return float(torch.quantile(tensor, quantile, interpolation="linear"))


def _sum_storage(results: Sequence[C1SpeculativeDecodeResult], name: str) -> int:
    return sum(result.metrics.cache_storage[name] for result in results)


def _aggregate(
    results: Sequence[C1SpeculativeDecodeResult],
    exact_matches: Sequence[bool],
    *,
    draft_length: int,
) -> dict[str, Any]:
    rounds = [round_result for result in results for round_result in result.rounds]
    accepted = [item.accepted_length for item in rounds]
    shadow_accepted = [item.shadow_continuation_accepted for item in rounds]
    generated = sum(result.metrics.generated_target_tokens for result in results)
    verification_calls = sum(
        result.metrics.target_verification_calls for result in results
    )
    commit_sync_calls = sum(
        result.metrics.target_commit_sync_calls for result in results
    )
    accepted_replay_calls = sum(
        result.metrics.target_accepted_replay_calls for result in results
    )
    correction_calls = sum(
        result.metrics.target_correction_sync_calls for result in results
    )
    directly_committed = sum(
        result.metrics.directly_committed_target_tokens for result in results
    )
    sequentially_committed = sum(
        result.metrics.sequentially_committed_target_tokens for result in results
    )
    rollbacks = sum(result.metrics.rollback_count for result in results)
    rejected_suffix = sum(result.metrics.rejected_suffix_tokens for result in results)
    agreements = sum(result.metrics.verifier_top1_agreements for result in results)
    comparisons = sum(result.metrics.verifier_top1_comparisons for result in results)
    kl_sum = sum(result.metrics.shadow_logit_kl_sum for result in results)
    kl_count = sum(result.metrics.shadow_logit_kl_comparisons for result in results)
    block_sequential_agreements = sum(
        result.metrics.block_sequential_top1_agreements for result in results
    )
    block_sequential_comparisons = sum(
        result.metrics.block_sequential_top1_comparisons for result in results
    )
    block_sequential_kl_sum = sum(
        result.metrics.block_sequential_logit_kl_sum for result in results
    )
    block_sequential_kl_count = sum(
        result.metrics.block_sequential_logit_kl_comparisons for result in results
    )
    prompt_count = len(results)
    return {
        "draft_length": int(draft_length),
        "number_of_prompts": prompt_count,
        "number_of_generated_target_tokens": generated,
        "exact_target_sequence_match": bool(all(exact_matches)),
        "exact_target_sequence_matches": sum(bool(item) for item in exact_matches),
        "speculative_rounds": len(rounds),
        "mean_accepted_prefix_length": statistics.fmean(accepted) if accepted else 0.0,
        "median_accepted_prefix_length": statistics.median(accepted)
        if accepted
        else 0.0,
        "p10_accepted_prefix_length": _percentile(accepted, 0.10),
        "p50_accepted_prefix_length": _percentile(accepted, 0.50),
        "p90_accepted_prefix_length": _percentile(accepted, 0.90),
        "mean_shadow_continuation_accepted": (
            statistics.fmean(shadow_accepted) if shadow_accepted else 0.0
        ),
        "full_block_acceptance_rate": (
            sum(item.full_block_accepted for item in rounds) / len(rounds)
            if rounds
            else 0.0
        ),
        "draft_target_token_top1_agreement": agreements / comparisons
        if comparisons
        else 1.0,
        "draft_target_token_top1_comparisons": comparisons,
        "target_verification_calls": verification_calls,
        "target_commit_sync_calls": commit_sync_calls,
        "target_accepted_replay_calls": accepted_replay_calls,
        "target_correction_sync_calls": correction_calls,
        "directly_committed_target_tokens": directly_committed,
        "sequentially_committed_target_tokens": sequentially_committed,
        "direct_commit_token_fraction": directly_committed / generated
        if generated
        else 0.0,
        "total_exact_target_calls_after_prefill": verification_calls
        + commit_sync_calls,
        "target_calls_per_output_token": (
            (verification_calls + commit_sync_calls) / generated if generated else 0.0
        ),
        "rollback_count": rollbacks,
        "mean_rejected_suffix_length": rejected_suffix / rollbacks
        if rollbacks
        else 0.0,
        "mean_shadow_logit_kl": kl_sum / kl_count if kl_count else None,
        "block_sequential_top1_agreement": (
            block_sequential_agreements / block_sequential_comparisons
            if block_sequential_comparisons
            else None
        ),
        "block_sequential_top1_comparisons": block_sequential_comparisons,
        "mean_block_sequential_logit_kl": (
            block_sequential_kl_sum / block_sequential_kl_count
            if block_sequential_kl_count
            else None
        ),
        "draft_latency_seconds": sum(
            result.metrics.draft_seconds for result in results
        ),
        "verification_latency_seconds": sum(
            result.metrics.verification_seconds for result in results
        ),
        "commit_sync_latency_seconds": sum(
            result.metrics.commit_sync_seconds for result in results
        ),
        "correction_sync_latency_seconds": sum(
            result.metrics.correction_sync_seconds for result in results
        ),
        "mean_shadow_logical_bytes": _sum_storage(results, "shadow_logical_bytes")
        / prompt_count,
        "mean_shadow_physical_reference_bytes": _sum_storage(
            results, "shadow_physical_bytes"
        )
        / prompt_count,
        "mean_c1_value_cache_bytes": _sum_storage(results, "c1_value_bytes")
        / prompt_count,
        "mean_exact_key_bytes": _sum_storage(results, "exact_key_bytes") / prompt_count,
        "mean_recent_exact_key_bytes": _sum_storage(results, "recent_exact_key_bytes")
        / prompt_count,
        "storage_note": (
            "recent exact bytes are a subset/view of exact-Key storage in this GPU oracle; "
            "logical INT4 bytes include scales, physical reference bytes use int8 values"
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# C1 shadow-Key speculative verification oracle",
        "",
        "This is a correctness and acceptance oracle, not a production throughput or "
        "CPU-offload benchmark.",
        "",
        f"Model: `{metadata['model_path']}`; prompts: `{metadata['number_of_prompts']}`; "
        f"shadow bits: `{metadata['shadow_bits']}`; recent exact window: "
        f"`{metadata['recent_exact_window']}`.",
        "",
        "| Draft | Exact match | Mean accepted | Mean shadow accepted | Full block | "
        "Shadow top-1 | Block/seq top-1 | Exact calls/token | Corrections |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        block_sequential = row["block_sequential_top1_agreement"]
        block_sequential_text = (
            "n/a" if block_sequential is None else f"{block_sequential:.4f}"
        )
        lines.append(
            f"| {row['draft_length']} | {row['exact_target_sequence_match']} | "
            f"{row['mean_accepted_prefix_length']:.4f} | "
            f"{row['mean_shadow_continuation_accepted']:.4f} | "
            f"{row['full_block_acceptance_rate']:.4f} | "
            f"{row['draft_target_token_top1_agreement']:.4f} | "
            f"{block_sequential_text} | "
            f"{row['target_calls_per_output_token']:.6f} | "
            f"{row['target_correction_sync_calls']} |"
        )
    lines.extend(
        [
            "",
            "The first proposal in every round is seeded by exact-target logits. "
            "`Mean shadow accepted` removes that forced seed. The selected commit policy "
            f"is `{metadata['commit_policy']}`. Direct block commit uses pending exact "
            "target KV and only performs an additional one-token target forward for a "
            "correction. Exact sequence match is always measured against ordinary "
            "sequential C1 greedy decoding.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--c1-export", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--shadow-bits", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--shadow-group-size", type=int, default=32)
    parser.add_argument("--recent-exact-window", type=int, default=256)
    parser.add_argument("--draft-lengths", default="4,8,16,32")
    parser.add_argument(
        "--commit-policy",
        choices=("strict_replay", "direct_block"),
        default="strict_replay",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    c1_export = Path(args.c1_export).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    model_config_path = model_path / "config.json"
    export_manifest_path = c1_export / "results.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    if not export_manifest_path.is_file():
        raise FileNotFoundError(export_manifest_path)
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    recorded_model_hash = export_manifest.get("fit_config", {}).get(
        "model_config_sha256"
    )
    observed_model_hash = file_sha256(model_config_path)
    if recorded_model_hash != observed_model_hash:
        raise ValueError(
            "uniform C1 ALS factors belong to a different model config: "
            f"{recorded_model_hash!r} vs {observed_model_hash!r}"
        )
    prompts = _load_prompts(prompt_path)
    draft_lengths = _parse_int_csv(args.draft_lengths)
    config = ShadowKeyConfig(
        bits=args.shadow_bits,
        group_size=args.shadow_group_size,
        recent_exact_window=args.recent_exact_window,
    )
    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model C1 shadow-Key evaluation requires CUDA")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    replacements = install_qwen3_gqa_vo_als_export(model, c1_export)
    if not replacements:
        raise RuntimeError("C1 export did not replace any Qwen3 attention layers")
    eos_token_id = tokenizer.eos_token_id
    per_configuration: dict[int, list[C1SpeculativeDecodeResult]] = {
        draft_length: [] for draft_length in draft_lengths
    }
    exact_matches: dict[int, list[bool]] = {
        draft_length: [] for draft_length in draft_lengths
    }
    prompt_records: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded.input_ids.to(device)
        if int(input_ids.shape[0]) != 1:
            raise AssertionError("one prompt must tokenize to batch size one")
        exact = exact_c1_greedy_decode(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        exact_generated_token_ids = tuple(
            int(token) for token in exact[0, int(input_ids.shape[1]) :].tolist()
        )
        configurations: list[dict[str, Any]] = []
        for draft_length in draft_lengths:
            cache = C1ShadowKeyValueCache(
                num_layers=int(model.config.num_hidden_layers),
                config=config,
            )
            result = c1_shadow_greedy_decode(
                model,
                input_ids,
                cache=cache,
                max_new_tokens=args.max_new_tokens,
                draft_length=draft_length,
                block_commit=BlockCommitConfig(policy=args.commit_policy),
                eos_token_id=eos_token_id,
            )
            matches = bool(torch.equal(result.token_ids, exact))
            first_divergence = _first_sequence_divergence(
                exact_generated_token_ids,
                result.generated_token_ids,
            )
            per_configuration[draft_length].append(result)
            exact_matches[draft_length].append(matches)
            configurations.append(
                {
                    "draft_length": draft_length,
                    "exact_target_sequence_match": matches,
                    "first_generated_token_divergence": first_divergence,
                    "generated_token_ids": list(result.generated_token_ids),
                    "generated_tokens": result.metrics.generated_target_tokens,
                    "rounds": [asdict(item) for item in result.rounds],
                    "metrics": asdict(result.metrics),
                }
            )
        prompt_records.append(
            {
                "prompt_index": prompt_index,
                "prompt_sha256": _prompt_sha256(prompt),
                "prompt_tokens": int(input_ids.shape[1]),
                "exact_generated_tokens": int(exact.shape[1] - input_ids.shape[1]),
                "exact_generated_token_ids": list(exact_generated_token_ids),
                "configurations": configurations,
            }
        )

    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "metadata": {
            "cli_parameters": vars(args),
            "model_path": str(model_path),
            "model_config_sha256": observed_model_hash,
            "c1_export": str(c1_export),
            "c1_export_manifest_sha256": file_sha256(export_manifest_path),
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": file_sha256(prompt_path),
            "number_of_prompts": len(prompts),
            "git_commit": _git_commit(),
            "git_worktree_dirty": _git_worktree_dirty(),
            "oracle_source_sha256": {
                "c1_shadow_kv.py": file_sha256(
                    REPO_ROOT / "basisserve/core/c1_shadow_kv.py"
                ),
                "c1_speculative_decode.py": file_sha256(
                    REPO_ROOT / "basisserve/core/c1_speculative_decode.py"
                ),
                "eval_c1_shadow_k_speculative.py": file_sha256(Path(__file__)),
            },
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "shadow_bits": config.bits,
            "shadow_group_size": config.group_size,
            "recent_exact_window": config.recent_exact_window,
            "draft_lengths": list(draft_lengths),
            "max_new_tokens": args.max_new_tokens,
            "c1_replaced_layers": len(replacements),
            "semantic_target": "exact-Key deployed C1 greedy model",
            "commit_policy": args.commit_policy,
            "forced_target_seed_per_round": True,
            "draft_target_metrics_exclude_forced_seed": True,
            "rejection_policy": "exact correction-token sync and commit",
            "commit_semantics": (
                "one-token exact replay after diagnostic block verify"
                if args.commit_policy == "strict_replay"
                else "accepted pending exact block KV plus correction-only target forward"
            ),
        },
        "aggregate": [
            _aggregate(
                per_configuration[draft_length],
                exact_matches[draft_length],
                draft_length=draft_length,
            )
            for draft_length in draft_lengths
        ],
        "prompts": prompt_records,
        "limitations": [
            "single GPU, batch size one, greedy decoding only",
            "reference int8 storage is used for logical INT4 values",
            "exact Keys remain GPU-resident; no CPU offload is measured",
            "dequantization/materialization and timings are unfused oracle paths",
            "reference exact calls and timings are non-production",
            "uniform-rank ALS C1 safetensors runtime only",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
