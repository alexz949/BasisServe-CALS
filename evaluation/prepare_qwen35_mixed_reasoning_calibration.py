#!/usr/bin/env python3
"""Build target-model mixed reasoning calibration for Qwen3.5-9B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.prepare_qwen3_32b_reasoning_calibration import (
    DATASET,
    DATASET_REVISION,
    DOMAINS,
    STRATA,
    _candidate_start,
    _select_prompts,
    _stratum_targets,
)
from evaluation.qwen35_hybrid_common import atomic_save, sha256


FORMAT = "basisserve.qwen35_9b.mixed_reasoning_calibration.v1"
REASONING_ONLY_FORMAT = "basisserve.qwen35_9b.reasoning_only_calibration.v1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_prompts(path: Path, count_per_domain: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload["prompts"]
    selected = []
    for domain in DOMAINS:
        rows = [row for row in source if row["domain"] == domain]
        assert len(rows) >= count_per_domain
        selected.extend(rows[:count_per_domain])
    return selected


def rollout(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["status"] == "complete"
        print(f"[Resume] {result_path}", flush=True)
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    prompts_path = output / "prompts.json"
    if prompts_path.is_file():
        prompt_payload = json.loads(prompts_path.read_text(encoding="utf-8"))
        prompts = prompt_payload["prompts"]
        selection = prompt_payload["selection"]
    else:
        if args.prompt_source:
            source_path = Path(args.prompt_source).expanduser().resolve()
            prompts = _source_prompts(source_path, args.prompt_count_per_domain)
            selection = {
                "method": "frozen_decontaminated_prompt_subset",
                "source": str(source_path),
                "source_sha256": sha256(source_path),
            }
        else:
            prompts, counters = _select_prompts(
                tokenizer=tokenizer,
                count_per_domain=args.prompt_count_per_domain,
                maximum_prompt_tokens=args.maximum_prompt_tokens,
                seed=args.seed,
            )
            selection = {
                "method": "seeded_OpenThoughts3_selection",
                "counters": counters,
            }
        assert len(prompts) == len(DOMAINS) * args.prompt_count_per_domain
        _write_json(
            prompts_path,
            {
                "format": FORMAT + ".prompts",
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "selection": selection,
                "prompts": prompts,
            },
        )

    rollouts_path = output / "rollouts.jsonl"
    completed_rows = _load_jsonl(rollouts_path)
    completed = {row["id"] for row in completed_rows}
    pending = [row for row in prompts if row["id"] not in completed]
    if pending:
        from vllm import LLM, SamplingParams

        engine = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=args.max_model_length,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            disable_log_stats=False,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.maximum_generation_tokens,
            detokenize=False,
        )
        for start in range(0, len(pending), args.generation_batch_size):
            batch = pending[start : start + args.generation_batch_size]
            outputs = engine.generate(
                [row["rendered_prompt"] for row in batch],
                sampling_params=sampling,
                use_tqdm=False,
            )
            assert len(outputs) == len(batch)
            generated = []
            for prompt, output_row in zip(batch, outputs, strict=True):
                assert len(output_row.outputs) == 1
                completion = output_row.outputs[0]
                prompt_ids = list(map(int, output_row.prompt_token_ids))
                output_ids = list(map(int, completion.token_ids))
                assert prompt_ids and output_ids
                generated.append(
                    {
                        "id": prompt["id"],
                        "domain": prompt["domain"],
                        "source": prompt["source"],
                        "difficulty": prompt.get("difficulty"),
                        "prompt_sha256": prompt["prompt_sha256"],
                        "rendered_prompt_sha256": prompt["rendered_prompt_sha256"],
                        "prompt_token_ids": prompt_ids,
                        "output_token_ids": output_ids,
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(output_ids),
                        "total_tokens": len(prompt_ids) + len(output_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    }
                )
            _append_jsonl(rollouts_path, generated)
            print(
                f"[Rollout] completed={start + len(batch)}/{len(pending)} "
                f"existing={len(completed_rows)}",
                flush=True,
            )
        del engine

    rows = _load_jsonl(rollouts_path)
    assert len(rows) == len(prompts)
    assert {row["id"] for row in rows} == {row["id"] for row in prompts}
    model_path = Path(args.model).expanduser().resolve()
    result = {
        "format": FORMAT + ".rollouts",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "tokenizer_sha256": sha256(model_path / "tokenizer.json"),
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "prompt_selection": selection,
        "sampling": {
            "temperature": 0.0,
            "maximum_generation_tokens": args.maximum_generation_tokens,
            "maximum_model_length": args.max_model_length,
            "prompt_format": "raw rendered OpenThoughts problem plus domain suffix",
            "chat_template": False,
            "teacher_answers_used": False,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in DOMAINS
        },
        "lengths": {
            "minimum_total_tokens": min(row["total_tokens"] for row in rows),
            "maximum_total_tokens": max(row["total_tokens"] for row in rows),
            "eligible_for_2048": sum(row["total_tokens"] >= 2048 for row in rows),
        },
        "artifacts": {
            "prompts": {"file": prompts_path.name, "sha256": sha256(prompts_path)},
            "rollouts": {
                "file": rollouts_path.name,
                "sha256": sha256(rollouts_path),
            },
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _write_json(result_path, result)
    print(f"[Result] {result_path}", flush=True)
    return 0


def _select_reasoning_splits(
    rows: list[dict[str, Any]],
    *,
    sequence_length: int,
    counts: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected = {split: [] for split in counts}
    audit: dict[str, Any] = {"eligible": {}, "selected": {}}
    for domain in DOMAINS:
        eligible = [
            row
            for row in rows
            if row["domain"] == domain and row["total_tokens"] >= sequence_length
        ]
        eligible.sort(key=lambda row: _text_hash(f"{seed}:{row['id']}"))
        needed = sum(counts.values())
        assert len(eligible) >= needed, (domain, len(eligible), needed)
        cursor = 0
        audit["eligible"][domain] = len(eligible)
        audit["selected"][domain] = {}
        for split, count in counts.items():
            targets = _stratum_targets(count)
            audit["selected"][domain][split] = targets
            for stratum in STRATA:
                for row in eligible[cursor : cursor + targets[stratum]]:
                    tokens = row["prompt_token_ids"] + row["output_token_ids"]
                    maximum_start = len(tokens) - sequence_length
                    start = _candidate_start(
                        record_id=row["id"],
                        stratum=stratum,
                        maximum_start=maximum_start,
                        seed=seed,
                    )
                    token_ids = tokens[start : start + sequence_length]
                    assert len(token_ids) == sequence_length
                    selected[split].append(
                        {
                            "input_ids": token_ids,
                            "id": row["id"],
                            "domain": domain,
                            "source": row["source"],
                            "stratum": stratum,
                            "token_start": start,
                            "rollout_tokens": len(tokens),
                            "input_ids_sha256": hashlib.sha256(
                                torch.tensor(token_ids, dtype=torch.int32)
                                .numpy()
                                .tobytes()
                            ).hexdigest(),
                        }
                    )
                cursor += targets[stratum]
            assert len([row for row in selected[split] if row["domain"] == domain]) == count
    return selected, audit


def _c4_split(
    tensor: torch.Tensor,
    manifest: Mapping[str, Any],
    split: str,
    count: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    source = manifest["splits"][split]
    assert source["count"] >= count
    start = source["offset"]
    rows = tensor[start : start + count]
    records = [
        {"domain": "c4", "source_split": split, "source_index": start + index}
        for index in range(count)
    ]
    return rows, records


def _reasoning_tensor(
    rows: list[dict[str, Any]], sequence_length: int
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, sequence_length), dtype=torch.int32)
    return torch.tensor([row["input_ids"] for row in rows], dtype=torch.int32)


def _model_identity(model_path: Path, windows_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()))
    return {
        "format": FORMAT + ".model_identity",
        "status": "complete",
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "config_sha256": sha256(model_path / "config.json"),
        "tokenizer_sha256": sha256(model_path / "tokenizer.json"),
        "model_files_sha256": {
            name: sha256(model_path / name) for name in names
        },
        "windows_sha256": sha256(windows_path),
    }


def windows(args: argparse.Namespace) -> int:
    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    rollout_result_path = rollout_dir / "result.json"
    rollout_path = rollout_dir / "rollouts.jsonl"
    rollout_result = json.loads(rollout_result_path.read_text(encoding="utf-8"))
    assert rollout_result["status"] == "complete"
    assert rollout_result["artifacts"]["rollouts"]["sha256"] == sha256(rollout_path)
    assert rollout_result["model_config_sha256"] == sha256(
        Path(args.model).expanduser().resolve() / "config.json"
    )
    reasoning, selection_audit = _select_reasoning_splits(
        _load_jsonl(rollout_path),
        sequence_length=args.sequence_length,
        counts={
            "fit": args.fit_reasoning_per_domain,
            "profile": args.profile_reasoning_per_domain,
            "confirm": args.confirm_reasoning_per_domain,
        },
        seed=args.seed,
    )

    c4_path = Path(args.c4_windows).expanduser().resolve()
    c4_manifest_path = c4_path.parent / "manifest.json"
    c4_manifest = json.loads(c4_manifest_path.read_text(encoding="utf-8"))
    assert c4_manifest["artifact"]["sha256"] == sha256(c4_path)
    c4 = load_file(str(c4_path), device="cpu")["input_ids"].to(torch.int32)
    assert c4.shape[1] >= args.sequence_length
    c4 = c4[:, : args.sequence_length].contiguous()

    fit_c4, fit_records = _c4_split(c4, c4_manifest, "fit", args.fit_c4)
    heldout, heldout_records = _c4_split(
        c4, c4_manifest, "heldout", args.heldout_c4
    )
    profile_c4, profile_records = _c4_split(
        c4, c4_manifest, "profile", args.profile_c4
    )
    confirm_c4, confirm_records = _c4_split(
        c4, c4_manifest, "confirm", args.confirm_c4
    )
    payload = {
        "fit": torch.cat(
            (fit_c4, _reasoning_tensor(reasoning["fit"], args.sequence_length))
        ),
        "heldout": heldout,
        "profile": torch.cat(
            (
                profile_c4,
                _reasoning_tensor(reasoning["profile"], args.sequence_length),
            )
        ),
        "confirm": torch.cat(
            (
                confirm_c4,
                _reasoning_tensor(reasoning["confirm"], args.sequence_length),
            )
        ),
    }
    expected = {
        "fit": args.fit_c4 + 2 * args.fit_reasoning_per_domain,
        "heldout": args.heldout_c4,
        "profile": args.profile_c4 + 2 * args.profile_reasoning_per_domain,
        "confirm": args.confirm_c4 + 2 * args.confirm_reasoning_per_domain,
    }
    assert all(tuple(payload[name].shape) == (count, args.sequence_length) for name, count in expected.items())
    output = Path(args.output_dir).expanduser().resolve()
    windows_path = output / "windows.pt"
    atomic_save(windows_path, payload)
    records = {
        "fit": fit_records + [
            {key: value for key, value in row.items() if key != "input_ids"}
            for row in reasoning["fit"]
        ],
        "heldout": heldout_records,
        "profile": profile_records + [
            {key: value for key, value in row.items() if key != "input_ids"}
            for row in reasoning["profile"]
        ],
        "confirm": confirm_records + [
            {key: value for key, value in row.items() if key != "input_ids"}
            for row in reasoning["confirm"]
        ],
    }
    manifest = {
        "format": FORMAT + ".windows",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "sequence_length": args.sequence_length,
        "composition": {
            "fit": {"c4": args.fit_c4, "math": args.fit_reasoning_per_domain, "code": args.fit_reasoning_per_domain},
            "heldout": {"c4": args.heldout_c4},
            "profile": {"c4": args.profile_c4, "math": args.profile_reasoning_per_domain, "code": args.profile_reasoning_per_domain},
            "confirm": {"c4": args.confirm_c4, "math": args.confirm_reasoning_per_domain, "code": args.confirm_reasoning_per_domain},
        },
        "controls": {
            "target_model_self_rollout": True,
            "teacher_answers_used": False,
            "reasoning_split_prompt_disjoint": True,
            "heldout_is_c4_only": True,
            "profile_and_confirm_are_mixed": True,
            "reasoning_position_strata": list(STRATA),
        },
        "selection_audit": selection_audit,
        "counts": {name: list(tensor.shape) for name, tensor in payload.items()},
        "records": records,
        "sources": {
            "c4_manifest": str(c4_manifest_path),
            "c4_manifest_sha256": sha256(c4_manifest_path),
            "c4_windows": str(c4_path),
            "c4_windows_sha256": sha256(c4_path),
            "rollout_result": str(rollout_result_path),
            "rollout_result_sha256": sha256(rollout_result_path),
        },
        "artifact": {
            "file": windows_path.name,
            "sha256": sha256(windows_path),
            "bytes": windows_path.stat().st_size,
        },
    }
    atomic_save(output / "manifest.json", manifest)
    identity = _model_identity(Path(args.model).expanduser().resolve(), windows_path)
    atomic_save(output / "model_manifest.json", identity)
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}), flush=True)
    return 0


def _reasoning_only_items(
    rollout_rows: list[dict[str, Any]],
    mixed_manifest: Mapping[str, Any],
    *,
    sequence_length: int,
    windows_per_domain: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["id"]: row for row in rollout_rows}
    targets = _stratum_targets(windows_per_domain)
    selected: dict[str, dict[str, list[dict[str, Any]]]] = {
        domain: {stratum: [] for stratum in STRATA} for domain in DOMAINS
    }
    existing_pairs: set[tuple[str, str]] = set()
    for record in mixed_manifest["records"]["fit"]:
        domain = record["domain"]
        if domain not in DOMAINS:
            continue
        row = by_id[record["id"]]
        tokens = row["prompt_token_ids"] + row["output_token_ids"]
        start = record["token_start"]
        token_ids = tokens[start : start + sequence_length]
        assert len(token_ids) == sequence_length
        digest = hashlib.sha256(
            torch.tensor(token_ids, dtype=torch.int32).numpy().tobytes()
        ).hexdigest()
        assert digest == record["input_ids_sha256"]
        item = dict(record)
        item["input_ids"] = token_ids
        item["selection_origin"] = "nested_mixed_fit"
        selected[domain][record["stratum"]].append(item)
        existing_pairs.add((record["id"], record["stratum"]))

    extension_counts: dict[str, dict[str, int]] = {
        domain: {} for domain in DOMAINS
    }
    for domain in DOMAINS:
        domain_rows = [
            row
            for row in rollout_rows
            if row["domain"] == domain and row["total_tokens"] >= sequence_length
        ]
        for stratum in STRATA:
            current = selected[domain][stratum]
            needed = targets[stratum] - len(current)
            assert needed >= 0
            candidates = []
            for row in domain_rows:
                if (row["id"], stratum) in existing_pairs:
                    continue
                tokens = row["prompt_token_ids"] + row["output_token_ids"]
                maximum_start = len(tokens) - sequence_length
                start = _candidate_start(
                    record_id=row["id"],
                    stratum=stratum,
                    maximum_start=maximum_start,
                    seed=seed,
                )
                token_ids = tokens[start : start + sequence_length]
                candidates.append(
                    {
                        "id": row["id"],
                        "domain": domain,
                        "source": row["source"],
                        "stratum": stratum,
                        "token_start": start,
                        "rollout_tokens": len(tokens),
                        "input_ids": token_ids,
                        "input_ids_sha256": hashlib.sha256(
                            torch.tensor(token_ids, dtype=torch.int32).numpy().tobytes()
                        ).hexdigest(),
                        "selection_origin": "reasoning_only_extension",
                        "selection_key": _text_hash(
                            f"{seed}:reasoning-only:{row['id']}:{stratum}:{start}"
                        ),
                    }
                )
            candidates.sort(key=lambda item: item["selection_key"])
            assert len(candidates) >= needed
            current.extend(candidates[:needed])
            extension_counts[domain][stratum] = needed
            assert len(current) == targets[stratum]

    items = [
        item
        for domain in DOMAINS
        for stratum in STRATA
        for item in selected[domain][stratum]
    ]
    assert len(items) == len(DOMAINS) * windows_per_domain
    audit = {
        "target_per_domain": windows_per_domain,
        "target_strata_per_domain": targets,
        "nested_mixed_windows_per_domain": {
            domain: sum(
                item["selection_origin"] == "nested_mixed_fit"
                for stratum in STRATA
                for item in selected[domain][stratum]
            )
            for domain in DOMAINS
        },
        "extension_counts": extension_counts,
        "eligible_rollouts": {
            domain: sum(
                row["domain"] == domain and row["total_tokens"] >= sequence_length
                for row in rollout_rows
            )
            for domain in DOMAINS
        },
    }
    return items, audit


def reasoning_windows(args: argparse.Namespace) -> int:
    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    rollout_result_path = rollout_dir / "result.json"
    rollout_path = rollout_dir / "rollouts.jsonl"
    rollout_result = json.loads(rollout_result_path.read_text(encoding="utf-8"))
    assert rollout_result["status"] == "complete"
    assert rollout_result["artifacts"]["rollouts"]["sha256"] == sha256(rollout_path)

    mixed_dir = Path(args.mixed_data).expanduser().resolve()
    mixed_manifest_path = mixed_dir / "manifest.json"
    mixed_windows_path = mixed_dir / "windows.pt"
    mixed_manifest = json.loads(mixed_manifest_path.read_text(encoding="utf-8"))
    assert mixed_manifest["status"] == "complete"
    assert mixed_manifest["artifact"]["sha256"] == sha256(mixed_windows_path)
    assert mixed_manifest["composition"]["fit"] == {
        "c4": 128,
        "math": 64,
        "code": 64,
    }
    mixed_payload = torch.load(mixed_windows_path, weights_only=True, map_location="cpu")
    assert tuple(mixed_payload["heldout"].shape) == (
        args.heldout_c4,
        args.sequence_length,
    )

    items, selection_audit = _reasoning_only_items(
        _load_jsonl(rollout_path),
        mixed_manifest,
        sequence_length=args.sequence_length,
        windows_per_domain=args.fit_reasoning_per_domain,
        seed=args.seed,
    )
    fit = _reasoning_tensor(items, args.sequence_length)
    payload = {
        "fit": fit,
        "heldout": mixed_payload["heldout"].to(torch.int32).contiguous(),
    }
    assert tuple(fit.shape) == (
        len(DOMAINS) * args.fit_reasoning_per_domain,
        args.sequence_length,
    )
    output = Path(args.output_dir).expanduser().resolve()
    windows_path = output / "windows.pt"
    atomic_save(windows_path, payload)
    records = {
        "fit": [
            {key: value for key, value in item.items() if key not in {"input_ids", "selection_key"}}
            for item in items
        ],
        "heldout": mixed_manifest["records"]["heldout"],
    }
    manifest = {
        "format": REASONING_ONLY_FORMAT + ".windows",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "sequence_length": args.sequence_length,
        "composition": {
            "fit": {
                "math": args.fit_reasoning_per_domain,
                "code": args.fit_reasoning_per_domain,
            },
            "heldout": {"c4": args.heldout_c4},
        },
        "controls": {
            "target_model_self_rollout": True,
            "teacher_answers_used": False,
            "heldout_is_identical_to_mixed_c4_heldout": True,
            "mixed_reasoning_fit_is_exact_nested_subset": True,
            "rank_schedule_is_frozen_from_c4": True,
            "reasoning_position_strata": list(STRATA),
        },
        "selection_audit": selection_audit,
        "counts": {name: list(tensor.shape) for name, tensor in payload.items()},
        "records": records,
        "sources": {
            "rollout_result": str(rollout_result_path),
            "rollout_result_sha256": sha256(rollout_result_path),
            "mixed_manifest": str(mixed_manifest_path),
            "mixed_manifest_sha256": sha256(mixed_manifest_path),
            "mixed_windows": str(mixed_windows_path),
            "mixed_windows_sha256": sha256(mixed_windows_path),
        },
        "artifact": {
            "file": windows_path.name,
            "sha256": sha256(windows_path),
            "bytes": windows_path.stat().st_size,
        },
    }
    atomic_save(output / "manifest.json", manifest)
    identity = _model_identity(Path(args.model).expanduser().resolve(), windows_path)
    atomic_save(output / "model_manifest.json", identity)
    print(json.dumps({"output": str(output), "counts": manifest["counts"], "selection_audit": selection_audit}), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    generate = subparsers.add_parser("rollout")
    generate.add_argument("--model", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--prompt-source")
    generate.add_argument("--prompt-count-per-domain", type=int, default=120)
    generate.add_argument("--maximum-prompt-tokens", type=int, default=1536)
    generate.add_argument("--maximum-generation-tokens", type=int, default=6144)
    generate.add_argument("--max-model-length", type=int, default=8192)
    generate.add_argument("--tensor-parallel-size", type=int, default=1)
    generate.add_argument("--max-num-seqs", type=int, default=32)
    generate.add_argument("--max-num-batched-tokens", type=int, default=8192)
    generate.add_argument("--generation-batch-size", type=int, default=32)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    generate.add_argument("--seed", type=int, default=20260920)

    build = subparsers.add_parser("windows")
    build.add_argument("--model", required=True)
    build.add_argument("--rollout-dir", required=True)
    build.add_argument("--c4-windows", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--sequence-length", type=int, default=2048)
    build.add_argument("--fit-c4", type=int, default=128)
    build.add_argument("--fit-reasoning-per-domain", type=int, default=64)
    build.add_argument("--heldout-c4", type=int, default=64)
    build.add_argument("--profile-c4", type=int, default=64)
    build.add_argument("--profile-reasoning-per-domain", type=int, default=32)
    build.add_argument("--confirm-c4", type=int, default=8)
    build.add_argument("--confirm-reasoning-per-domain", type=int, default=4)
    build.add_argument("--seed", type=int, default=20260920)

    reasoning = subparsers.add_parser("reasoning-windows")
    reasoning.add_argument("--model", required=True)
    reasoning.add_argument("--rollout-dir", required=True)
    reasoning.add_argument("--mixed-data", required=True)
    reasoning.add_argument("--output-dir", required=True)
    reasoning.add_argument("--sequence-length", type=int, default=2048)
    reasoning.add_argument("--fit-reasoning-per-domain", type=int, default=128)
    reasoning.add_argument("--heldout-c4", type=int, default=64)
    reasoning.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "rollout":
        return rollout(args)
    if args.stage == "reasoning-windows":
        return reasoning_windows(args)
    return windows(args)


if __name__ == "__main__":
    sys.exit(main())
