#!/usr/bin/env python3
"""Generate the official RULER-v1 base-model dataset for a local tokenizer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.ruler_v1 import parse_tasks, ruler_prompt  # noqa: E402


FORMAT = "basisserve.ruler_v1.qwen3_base_dataset.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ruler_revision(ruler_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(ruler_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _generate_task(
    *,
    python: Path,
    prepare_script: Path,
    output_dir: Path,
    tokenizer_path: Path,
    task_name: str,
    sequence_length: int,
    samples: int,
    seed: int,
) -> tuple[str, str]:
    command = [
        str(python),
        str(prepare_script),
        "--save_dir",
        str(output_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task_name,
        "--subset",
        "validation",
        "--tokenizer_path",
        str(tokenizer_path),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(sequence_length),
        "--num_samples",
        str(samples),
        "--random_seed",
        str(seed),
        "--model_template_type",
        "base",
    ]
    environment = dict(os.environ)
    environment["PATH"] = f"{python.parent}:{environment.get('PATH', '')}"
    result = subprocess.run(
        command,
        cwd=prepare_script.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    path = output_dir / task_name / "validation.jsonl"
    if not path.is_file():
        raise RuntimeError(
            f"official RULER generator returned without creating {path}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != samples:
        raise RuntimeError(
            f"official RULER generator wrote {len(lines)} rows for {task_name}, "
            f"expected {samples}"
        )
    return task_name, shlex.join(command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-margin", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    ruler_root = args.ruler_root.expanduser().resolve()
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tasks = parse_tasks(args.tasks)
    if min(args.max_seq_length, args.num_samples, args.workers) <= 0:
        raise ValueError("sequence length, samples, and workers must be positive")
    prepare_script = ruler_root / "scripts" / "data" / "prepare.py"
    if not prepare_script.is_file():
        raise FileNotFoundError(prepare_script)
    required_sources = (
        "PaulGrahamEssays.json",
        "english_words.json",
        "hotpotqa.json",
        "squad.json",
    )
    source_dir = prepare_script.parent / "synthetic" / "json"
    missing = [name for name in required_sources if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"official RULER source data is missing from {source_dir}: {missing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [
            executor.submit(
                _generate_task,
                python=Path(sys.executable).resolve(),
                prepare_script=prepare_script,
                output_dir=output_dir,
                tokenizer_path=tokenizer_path,
                task_name=task.name,
                sequence_length=args.max_seq_length-args.prompt_margin,
                samples=args.num_samples,
                seed=args.random_seed,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            task_name, command = future.result()
            commands[task_name] = command
            print(f"[RULER data] complete task={task_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    artifacts = {}
    for task in tasks:
        path = output_dir / task.name / "validation.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        prompt_lengths = [
            len(tokenizer(ruler_prompt(row), add_special_tokens=True)["input_ids"])
            for row in rows
        ]
        if any(
            length + task.tokens_to_generate > args.max_seq_length
            for length in prompt_lengths
        ):
            raise RuntimeError(
                f"generated task {task.name} exceeds its total RULER length budget"
            )
        artifacts[task.name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(rows),
            "minimum_prompt_tokens": min(prompt_lengths),
            "maximum_prompt_tokens": max(prompt_lengths),
            "tokens_to_generate": task.tokens_to_generate,
            "match_type": task.match_type,
        }

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "ruler": {
            "root": str(ruler_root),
            "revision": _ruler_revision(ruler_root),
            "prepare_script_sha256": _sha256(prepare_script),
        },
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_config_sha256": _sha256(tokenizer_path / "tokenizer_config.json"),
        "protocol": {
            "model_template_type": "base",
            "sequence_length": args.max_seq_length,
            "samples_per_task": args.num_samples,
            "random_seed": args.random_seed,
            "prompt_margin": args.prompt_margin,
            "tasks": [task.name for task in tasks],
        },
        "commands": commands,
        "artifacts": artifacts,
    }
    _atomic_json(output_dir / "manifest.json", payload)
    print(f"[RULER data] manifest={output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
