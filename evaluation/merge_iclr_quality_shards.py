#!/usr/bin/env python3
"""Merge parallel two-GPU ICLR quality shards into one audited result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_qwen3_8b_iclr_quality as evaluator  # noqa: E402


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _load_json(path: Path) -> dict[str, Any] | None:
    if not _check(path.is_file(), f"missing quality artifact: {path}"):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_matches(
    payload: Mapping[str, Any],
    *,
    stage: str,
    run_id: str,
    checkpoint_sha256: str,
) -> bool:
    return _check(
        all(
            (
                payload.get("format") == evaluator.STAGE_FORMATS[stage],
                payload.get("status") == "complete",
                payload.get("run_id") == run_id,
                payload.get("checkpoint", {}).get("manifest_sha256")
                == checkpoint_sha256,
            )
        ),
        f"{stage} artifact provenance mismatch",
    )


def merge(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    evaluator.activate_quality_profile(args.profile)
    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    final_path = output_dir / "result.json"
    if not _check(not final_path.exists(), f"final result already exists: {final_path}"):
        return 2

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        return 2
    checkpoint_sha256 = evaluator._sha256(manifest_path)
    valid_manifest = all(
        (
            _check(
                manifest.get("format") == evaluator.CHECKPOINT_FORMAT,
                "checkpoint format mismatch",
            ),
            _check(manifest.get("status") == "complete", "checkpoint is incomplete"),
            _check(manifest.get("run_id") == args.run_id, "checkpoint run ID mismatch"),
            _check(
                manifest.get("model", {}).get("config_sha256")
                == evaluator._sha256(model_path / "config.json"),
                "checkpoint belongs to another model config",
            ),
        )
    )
    if not valid_manifest:
        return 2

    wiki_path = output_dir / "wikitext2.json"
    c4_path = output_dir / "c4.json"
    wiki = _load_json(wiki_path)
    c4 = _load_json(c4_path)
    if wiki is None or c4 is None:
        return 2
    if not all(
        (
            _stage_matches(
                wiki,
                stage="wikitext2",
                run_id=args.run_id,
                checkpoint_sha256=checkpoint_sha256,
            ),
            _stage_matches(
                c4,
                stage="c4",
                run_id=args.run_id,
                checkpoint_sha256=checkpoint_sha256,
            ),
        )
    ):
        return 2

    shard_paths = [
        output_dir / f"commonsense-shard-{index:02d}.json"
        for index in range(args.shard_count)
    ]
    shard_payloads: list[dict[str, Any]] = []
    task_rows: dict[str, dict[str, Any]] = {}
    evaluation_results: dict[str, Any] = {}
    gpu_names: list[str] = []
    for index, shard_path in enumerate(shard_paths):
        shard = _load_json(shard_path)
        if shard is None:
            return 2
        protocol = shard.get("protocol", {})
        valid_shard = all(
            (
                _stage_matches(
                    shard,
                    stage="commonsense",
                    run_id=args.run_id,
                    checkpoint_sha256=checkpoint_sha256,
                ),
                _check(
                    protocol.get("quality_shard_index") == index,
                    f"commonsense shard {index} index mismatch",
                ),
                _check(
                    protocol.get("quality_shard_count") == args.shard_count,
                    f"commonsense shard {index} count mismatch",
                ),
            )
        )
        if not valid_shard:
            return 2
        for row in shard.get("task_accuracy", ()):
            task = str(row.get("task"))
            if not _check(task not in task_rows, f"duplicate task across shards: {task}"):
                return 2
            task_rows[task] = dict(row)
        evaluation_results.update(shard.get("evaluation", {}).get("results", {}))
        gpu_names.extend(map(str, shard.get("environment", {}).get("cuda_devices", ())))
        shard_payloads.append(shard)

    if not _check(set(task_rows) == set(evaluator.TASKS), "merged task set mismatch"):
        return 2
    if not _check(
        gpu_names == ["NVIDIA L40S"] * (args.shard_count * args.gpus_per_shard),
        f"unexpected shard GPU inventory: {gpu_names}",
    ):
        return 2
    ordered_rows = [task_rows[task] for task in evaluator.TASKS]
    average_accuracy = sum(float(row["value"]) for row in ordered_rows) / len(
        ordered_rows
    )
    compression = manifest["compression"]
    artifact_sha256 = (
        None
        if compression["method"] == "dense"
        else str(manifest["artifact"]["sha256"])
    )
    shard_records = [
        {
            "file": path.name,
            "sha256": evaluator._sha256(path),
            "tasks": payload["protocol"]["tasks"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "runtime": payload["runtime"],
            "environment": payload["environment"],
        }
        for path, payload in zip(shard_paths, shard_payloads, strict=True)
    ]
    commonsense_path = output_dir / "commonsense.json"
    commonsense = {
        **evaluator._stage_base(
            stage="commonsense",
            run_id=args.run_id,
            checkpoint_dir=checkpoint_dir,
            checkpoint_manifest_sha256=checkpoint_sha256,
            artifact_sha256=artifact_sha256,
            compression=compression,
        ),
        "command": shlex.join(sys.argv),
        "model": manifest["model"],
        "runtime": {
            "evaluation_layout": "parallel_quality_shards",
            "quality_shard_count": args.shard_count,
            "gpus_per_shard": args.gpus_per_shard,
            "shards": shard_records,
        },
        "protocol": {
            "tasks": list(evaluator.TASKS),
            "num_fewshot": 0,
            "batch_size": args.lm_eval_batch_size,
            "max_length": 4096,
            "metric_selection": "acc_norm when present, otherwise acc",
            "parallel_quality_shards": args.shard_count,
            "gpus_per_shard": args.gpus_per_shard,
        },
        "task_accuracy": ordered_rows,
        "average_accuracy": average_accuracy,
        "evaluation": {
            "results": evaluation_results,
            "source_shards": [record["file"] for record in shard_records],
        },
        "elapsed_seconds": max(
            float(payload["elapsed_seconds"]) for payload in shard_payloads
        ),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "cuda_devices": gpu_names,
            "evaluation_layout": "parallel_quality_shards",
        },
    }
    evaluator._write_json(commonsense_path, commonsense)

    stage_paths = {
        "wikitext2": wiki_path,
        "c4": c4_path,
        "commonsense": commonsense_path,
    }
    result = {
        "format": evaluator.FORMAT,
        "status": "complete",
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": manifest["model"],
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "manifest_sha256": checkpoint_sha256,
            "artifact_sha256": artifact_sha256,
            "format": evaluator.CHECKPOINT_FORMAT,
        },
        "compression": compression,
        "metrics": {
            "wikitext2_ppl": wiki["metrics"]["ppl"],
            "c4_validation_128_ppl": c4["metrics"]["ppl"],
            "task_accuracy": ordered_rows,
            "average_accuracy": average_accuracy,
        },
        "stages": {
            stage: {"file": path.name, "sha256": evaluator._sha256(path)}
            for stage, path in stage_paths.items()
        },
        "elapsed_seconds_this_attempt": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "cuda_devices": gpu_names,
            "evaluation_layout": "parallel_quality_shards",
            "quality_shard_count": args.shard_count,
            "gpus_per_shard": args.gpus_per_shard,
            "source_slurm_jobs": [
                payload.get("environment", {}).get("slurm_job_id")
                for payload in shard_payloads
            ],
        },
    }
    evaluator._atomic_json(final_path, result)
    print(
        f"[Result] run_id={args.run_id} wiki_ppl={result['metrics']['wikitext2_ppl']:.9f} "
        f"c4_ppl={result['metrics']['c4_validation_128_ppl']:.9f} "
        f"average_accuracy={average_accuracy:.9f}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(evaluator.QUALITY_PROFILES), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--gpus-per-shard", type=int, default=2)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(merge(parse_args()))
